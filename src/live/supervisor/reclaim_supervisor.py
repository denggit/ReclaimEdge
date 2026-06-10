from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from src.live import time_utils as live_time_utils
from src.live.runtime_paths import RuntimePaths
from src.live.supervisor.child_process import ChildProcess, ChildProcessSnapshot, ChildProcessSpec
from src.live.supervisor.heartbeat_monitor import HeartbeatMonitor, HeartbeatMonitorConfig, HeartbeatStatus
from src.live.supervisor.restart_policy import RestartDecision, RestartPolicy, RestartPolicyConfig
from src.utils.log import get_logger

logger = get_logger(__name__)


def _default_project_root() -> Path:
    """Return the repository root directory.

    ``src/live/supervisor/reclaim_supervisor.py`` is three levels deep,
    so ``parents[3]`` resolves to the repo root.
    """
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SupervisorShutdownResult:
    child_name: str
    child_started: bool
    child_running_before_shutdown: bool
    child_pid: int | None
    child_returncode_before_shutdown: int | None
    terminate_attempted: bool
    terminate_error: str | None = None

    @property
    def ok(self) -> bool:
        return self.terminate_error is None


@dataclass(frozen=True)
class SupervisorHealthEvent:
    """Immutable record of a supervisor health observation.

    Captures child exit, heartbeat problems, and restart decisions for
    downstream consumers (logging, future outbox).  Never writes to disk —
    stays in memory.
    """

    event_type: str
    child_name: str
    ts_ms: int
    pid: int | None
    returncode: int | None
    heartbeat_status: str | None = None
    heartbeat_age_seconds: float | None = None
    heartbeat_error: str | None = None
    restart_reason: str | None = None
    restart_count_in_window: int | None = None
    restart_suppressed_reason: str | None = None


@dataclass(frozen=True)
class ReclaimSupervisorConfig:
    poll_interval_seconds: float = 5.0
    project_root: Path = field(default_factory=_default_project_root)
    child_name: str = "ETH-USDT-SWAP"
    worker_script: Path = Path("scripts/run_symbol_worker.py")
    child_terminate_timeout_seconds: float = 10.0
    child_kill_timeout_seconds: float = 5.0
    runtime_dir: Path = Path("runtime")
    heartbeat_check_enabled: bool = True
    heartbeat_check_interval_seconds: float = 5.0
    heartbeat_default_stale_after_seconds: float = 30.0
    stop_on_child_exit: bool = True
    stop_on_bad_heartbeat: bool = True
    heartbeat_startup_grace_seconds: float = 20.0
    restart_policy: RestartPolicyConfig = field(default_factory=RestartPolicyConfig)
    restart_on_child_exit: bool = True
    restart_on_bad_heartbeat: bool = True
    child_env: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.poll_interval_seconds <= 0:
            raise ValueError("supervisor poll_interval_seconds must be > 0")
        if not self.child_name.strip():
            raise ValueError("supervisor child_name must not be empty")
        if self.child_terminate_timeout_seconds <= 0:
            raise ValueError("child_terminate_timeout_seconds must be > 0")
        if self.child_kill_timeout_seconds <= 0:
            raise ValueError("child_kill_timeout_seconds must be > 0")
        if self.heartbeat_check_interval_seconds <= 0:
            raise ValueError("heartbeat_check_interval_seconds must be > 0")
        if self.heartbeat_default_stale_after_seconds <= 0:
            raise ValueError("heartbeat_default_stale_after_seconds must be > 0")
        if self.heartbeat_startup_grace_seconds <= 0:
            raise ValueError("heartbeat_startup_grace_seconds must be > 0")
        object.__setattr__(self, "project_root", Path(self.project_root))
        object.__setattr__(self, "worker_script", Path(self.worker_script))
        object.__setattr__(self, "runtime_dir", Path(self.runtime_dir))
        # -- child_env normalization ----------------------------------------
        if self.child_env is not None:
            normalized: dict[str, str] = {}
            for key, value in self.child_env.items():
                key_str = str(key)
                if not key_str.strip():
                    raise ValueError("child_env key must not be empty")
                normalized[key_str] = str(value)
            object.__setattr__(self, "child_env", normalized)


class ReclaimSupervisor:
    def __init__(
        self,
        *,
        config: ReclaimSupervisorConfig | None = None,
        event_pipeline: object | None = None,
    ) -> None:
        if event_pipeline is not None and not hasattr(event_pipeline, "process_once"):
            raise ValueError("event_pipeline must have 'process_once' attribute")
        self._config = config or ReclaimSupervisorConfig()
        self._event_pipeline: object | None = event_pipeline
        self._started_at_ms: int | None = None
        self._stop_requested = False
        self._child: ChildProcess | None = None
        self._shutdown_started = False
        self._shutdown_result: SupervisorShutdownResult | None = None
        self._heartbeat_monitor = HeartbeatMonitor(
            config=HeartbeatMonitorConfig(default_stale_after_seconds=self._config.heartbeat_default_stale_after_seconds)
        )
        self._last_heartbeat_check_monotonic = 0.0
        self._last_heartbeat_status: HeartbeatStatus | None = None
        self._health_events: list[SupervisorHealthEvent] = []
        self._child_started_monotonic: float | None = None
        self._restart_policy = RestartPolicy(self._config.restart_policy)

    @classmethod
    def from_env(cls) -> "ReclaimSupervisor":
        # D05 intentionally does not parse symbol lists or child configs from env.
        return cls()

    @property
    def config(self) -> ReclaimSupervisorConfig:
        return self._config

    @property
    def event_pipeline(self) -> object | None:
        return self._event_pipeline

    @property
    def started_at_ms(self) -> int | None:
        return self._started_at_ms

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    @property
    def child(self) -> ChildProcess | None:
        return self._child

    @property
    def child_snapshot(self) -> ChildProcessSnapshot | None:
        return self._child.snapshot() if self._child is not None else None

    @property
    def shutdown_started(self) -> bool:
        return self._shutdown_started

    @property
    def shutdown_result(self) -> SupervisorShutdownResult | None:
        return self._shutdown_result

    @property
    def last_heartbeat_status(self) -> HeartbeatStatus | None:
        return self._last_heartbeat_status

    @property
    def health_events(self) -> tuple[SupervisorHealthEvent, ...]:
        return tuple(self._health_events)

    @property
    def restart_policy(self) -> RestartPolicy:
        return self._restart_policy

    def request_stop(self) -> None:
        if not self._stop_requested:
            logger.info("RECLAIM_SUPERVISOR_STOP_REQUESTED")
            self._stop_requested = True

    def build_child_spec(self) -> ChildProcessSpec:
        script_path = self._config.project_root / self._config.worker_script
        return ChildProcessSpec(
            name=self._config.child_name,
            argv=(sys.executable, str(script_path)),
            cwd=self._config.project_root,
            env=self._config.child_env,
            terminate_timeout_seconds=self._config.child_terminate_timeout_seconds,
            kill_timeout_seconds=self._config.child_kill_timeout_seconds,
        )

    def create_child_process(self) -> ChildProcess:
        return ChildProcess(self.build_child_spec())

    # ------------------------------------------------------------------
    # runtime paths
    # ------------------------------------------------------------------

    def runtime_paths(self) -> RuntimePaths:
        runtime_dir = self._config.runtime_dir
        if not runtime_dir.is_absolute():
            runtime_dir = self._config.project_root / runtime_dir
        return RuntimePaths(runtime_dir=runtime_dir, inst_id=self._config.child_name)

    @property
    def heartbeat_path(self) -> Path:
        return self.runtime_paths().heartbeat_file

    # ------------------------------------------------------------------
    # health events
    # ------------------------------------------------------------------

    def _append_health_event(
        self,
        *,
        event_type: str,
        snapshot: ChildProcessSnapshot | None,
        heartbeat_status: HeartbeatStatus | None = None,
        restart_reason: str | None = None,
        restart_count_in_window: int | None = None,
        restart_suppressed_reason: str | None = None,
    ) -> SupervisorHealthEvent:
        event = SupervisorHealthEvent(
            event_type=event_type,
            child_name=self._config.child_name,
            ts_ms=live_time_utils.utc_ms(),
            pid=snapshot.pid if snapshot is not None else None,
            returncode=snapshot.returncode if snapshot is not None else None,
            heartbeat_status=heartbeat_status.status if heartbeat_status is not None else None,
            heartbeat_age_seconds=heartbeat_status.age_seconds if heartbeat_status is not None else None,
            heartbeat_error=heartbeat_status.error if heartbeat_status is not None else None,
            restart_reason=restart_reason,
            restart_count_in_window=restart_count_in_window,
            restart_suppressed_reason=restart_suppressed_reason,
        )
        self._health_events.append(event)
        return event

    # ------------------------------------------------------------------
    # child exit detection
    # ------------------------------------------------------------------

    async def check_child_exit_once(self) -> bool:
        """Check whether the current child has exited.

        Returns ``True`` when a restart or cooldown was handled (caller should
        skip heartbeat this iteration).  Returns ``False`` otherwise.
        """
        if self._child is None:
            return False
        snapshot = self._child.snapshot()
        if not (snapshot.started and not snapshot.running and snapshot.returncode is not None):
            return False

        event = self._append_health_event(event_type="CHILD_EXITED", snapshot=snapshot)
        logger.error(
            "RECLAIM_SUPERVISOR_CHILD_EXITED | child=%s pid=%s returncode=%s",
            snapshot.name,
            snapshot.pid,
            snapshot.returncode,
        )

        if self._config.restart_on_child_exit:
            restarted = await self._restart_child_after_exit_once(reason="child_exit")
            if restarted:
                return True
            # If not restarted due to cooldown, return True so caller skips
            # heartbeat this iteration (avoids double-processing).
            decision = self._restart_policy.evaluate(now_monotonic=time.monotonic())
            if decision.reason == "cooldown":
                return True
            # disabled / max exceeded / max_restarts_zero → fall through to stop
            if self._config.stop_on_child_exit:
                self.request_stop()
            return True

        if self._config.stop_on_child_exit:
            self.request_stop()
        return True

    # ------------------------------------------------------------------
    # restart helpers
    # ------------------------------------------------------------------

    async def _restart_child_after_exit_once(
        self,
        *,
        reason: str,
        now_monotonic: float | None = None,
    ) -> bool:
        """Evaluate restart policy and restart the child if allowed.

        Returns ``True`` if the child was restarted, ``False`` otherwise.
        """
        now = time.monotonic() if now_monotonic is None else now_monotonic
        decision = self._restart_policy.evaluate(now_monotonic=now)

        if not decision.allowed:
            # cooldown: do NOT request stop, just wait for next cycle
            if decision.reason == "cooldown":
                logger.debug(
                    "RECLAIM_SUPERVISOR_CHILD_RESTART_SUPPRESSED | child=%s reason=%s "
                    "restart_count_in_window=%s cooldown_next_allowed=%s",
                    self._config.child_name,
                    decision.reason,
                    decision.restart_count_in_window,
                    decision.next_allowed_monotonic,
                )
                return False

            # max exceeded / disabled / max_restarts_zero → suppressed
            self._append_health_event(
                event_type="CHILD_RESTART_SUPPRESSED",
                snapshot=None,
                restart_reason=decision.reason,
                restart_count_in_window=decision.restart_count_in_window,
                restart_suppressed_reason=decision.reason,
            )
            logger.error(
                "RECLAIM_SUPERVISOR_CHILD_RESTART_SUPPRESSED | child=%s reason=%s "
                "restart_count_in_window=%s",
                self._config.child_name,
                decision.reason,
                decision.restart_count_in_window,
            )
            return False

        # Allowed: request restart
        self._append_health_event(
            event_type="CHILD_RESTART_REQUESTED",
            snapshot=None,
            restart_reason=reason,
            restart_count_in_window=decision.restart_count_in_window,
        )

        child = self.create_child_process()
        await child.start()
        self._child = child
        self._child_started_monotonic = now
        self._last_heartbeat_status = None
        self._restart_policy.record_restart(now_monotonic=now)

        snapshot = child.snapshot()
        self._append_health_event(
            event_type="CHILD_RESTARTED",
            snapshot=snapshot,
            restart_reason=reason,
            restart_count_in_window=self._restart_policy.restart_count_in_window,
        )
        logger.warning(
            "RECLAIM_SUPERVISOR_CHILD_RESTARTED | child=%s pid=%s reason=%s "
            "restart_count_in_window=%s",
            snapshot.name,
            snapshot.pid,
            reason,
            self._restart_policy.restart_count_in_window,
        )
        return True

    async def _restart_child_after_bad_heartbeat_once(
        self,
        *,
        heartbeat_status: HeartbeatStatus,
        reason: str,
        now_monotonic: float | None = None,
    ) -> bool:
        """Evaluate restart policy, terminate the old child, and restart.

        Returns ``True`` if the child was restarted, ``False`` otherwise.

        **Guard**: a new child is started **only** when the old child is
        confirmed to no longer be running.  If terminate raises or the child
        is still running after the terminate attempt, the restart is
        suppressed and ``request_stop`` is called.
        """
        now = time.monotonic() if now_monotonic is None else now_monotonic
        decision = self._restart_policy.evaluate(now_monotonic=now)

        if not decision.allowed:
            # cooldown: do NOT request stop, return False for caller to handle
            if decision.reason == "cooldown":
                logger.debug(
                    "RECLAIM_SUPERVISOR_CHILD_RESTART_SUPPRESSED | child=%s reason=%s "
                    "restart_count_in_window=%s cooldown_next_allowed=%s",
                    self._config.child_name,
                    decision.reason,
                    decision.restart_count_in_window,
                    decision.next_allowed_monotonic,
                )
                return False

            self._append_health_event(
                event_type="CHILD_RESTART_SUPPRESSED",
                snapshot=None,
                heartbeat_status=heartbeat_status,
                restart_reason=decision.reason,
                restart_count_in_window=decision.restart_count_in_window,
                restart_suppressed_reason=decision.reason,
            )
            logger.error(
                "RECLAIM_SUPERVISOR_CHILD_RESTART_SUPPRESSED | child=%s reason=%s "
                "restart_count_in_window=%s",
                self._config.child_name,
                decision.reason,
                decision.restart_count_in_window,
            )
            return False

        # ── Allowed: terminate old child first ──────────────────────────
        self._append_health_event(
            event_type="CHILD_TERMINATE_FOR_RESTART_REQUESTED",
            snapshot=self._child.snapshot() if self._child is not None else None,
            heartbeat_status=heartbeat_status,
            restart_reason=reason,
            restart_count_in_window=decision.restart_count_in_window,
        )
        logger.warning(
            "RECLAIM_SUPERVISOR_CHILD_TERMINATE_FOR_RESTART_REQUESTED | child=%s reason=%s",
            self._config.child_name,
            reason,
        )

        if self._child is not None and self._child.snapshot().running:
            try:
                await self._child.terminate()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                terminate_error = f"{type(exc).__name__}: {exc}"
                logger.error(
                    "RECLAIM_SUPERVISOR_CHILD_TERMINATE_FOR_RESTART_FAILED | "
                    "name=%s pid=%s error=%s",
                    self._config.child_name,
                    self._child.pid,
                    terminate_error,
                )
                self._append_health_event(
                    event_type="CHILD_TERMINATE_FOR_RESTART_FAILED",
                    snapshot=self._child.snapshot(),
                    heartbeat_status=heartbeat_status,
                    restart_reason=reason,
                    restart_count_in_window=decision.restart_count_in_window,
                    restart_suppressed_reason="terminate_failed",
                )
                self.request_stop()
                return False

        # ── Verify old child is truly stopped ────────────────────────────
        if self._child is not None and self._child.snapshot().running:
            logger.error(
                "RECLAIM_SUPERVISOR_CHILD_TERMINATE_FOR_RESTART_STILL_RUNNING | "
                "child=%s pid=%s",
                self._config.child_name,
                self._child.pid,
            )
            self._append_health_event(
                event_type="CHILD_RESTART_SUPPRESSED",
                snapshot=self._child.snapshot(),
                heartbeat_status=heartbeat_status,
                restart_reason=reason,
                restart_count_in_window=decision.restart_count_in_window,
                restart_suppressed_reason="old_child_still_running",
            )
            self.request_stop()
            return False

        # ── Old child confirmed stopped — record and start new child ─────
        self._append_health_event(
            event_type="CHILD_TERMINATED_FOR_RESTART",
            snapshot=self._child.snapshot() if self._child is not None else None,
            heartbeat_status=heartbeat_status,
            restart_reason=reason,
            restart_count_in_window=decision.restart_count_in_window,
        )

        # Start new child
        child = self.create_child_process()
        await child.start()
        self._child = child
        self._child_started_monotonic = now
        self._last_heartbeat_status = None
        self._restart_policy.record_restart(now_monotonic=now)

        snapshot = child.snapshot()
        self._append_health_event(
            event_type="CHILD_RESTARTED",
            snapshot=snapshot,
            restart_reason=reason,
            restart_count_in_window=self._restart_policy.restart_count_in_window,
        )
        logger.warning(
            "RECLAIM_SUPERVISOR_CHILD_RESTARTED | child=%s pid=%s reason=%s "
            "restart_count_in_window=%s",
            snapshot.name,
            snapshot.pid,
            reason,
            self._restart_policy.restart_count_in_window,
        )
        return True

    # ------------------------------------------------------------------
    # heartbeat detection
    # ------------------------------------------------------------------

    def _heartbeat_in_startup_grace(self, now_monotonic: float | None = None) -> bool:
        if self._child_started_monotonic is None:
            return False
        now = time.monotonic() if now_monotonic is None else now_monotonic
        return now - self._child_started_monotonic < self._config.heartbeat_startup_grace_seconds

    async def check_heartbeat_once(self, *, now_monotonic: float | None = None) -> HeartbeatStatus | None:
        if not self._config.heartbeat_check_enabled:
            return None

        status = self._heartbeat_monitor.read_status(
            symbol=self._config.child_name,
            path=self.heartbeat_path,
        )
        self._last_heartbeat_status = status

        if status.fresh:
            return status

        if self._heartbeat_in_startup_grace(now_monotonic):
            logger.info(
                "RECLAIM_SUPERVISOR_HEARTBEAT_GRACE | child=%s heartbeat_status=%s path=%s",
                self._config.child_name,
                status.status,
                status.path,
            )
            return status

        event_type = (
            "HEARTBEAT_MISSING" if status.missing
            else "HEARTBEAT_STALE" if status.stale
            else "HEARTBEAT_INVALID"
        )
        snapshot = self._child.snapshot() if self._child is not None else None
        self._append_health_event(
            event_type=event_type,
            snapshot=snapshot,
            heartbeat_status=status,
        )
        logger.error(
            "RECLAIM_SUPERVISOR_HEARTBEAT_BAD | child=%s heartbeat_status=%s path=%s age_seconds=%s error=%s",
            self._config.child_name,
            status.status,
            status.path,
            status.age_seconds,
            status.error,
        )

        if self._config.restart_on_bad_heartbeat:
            restarted = await self._restart_child_after_bad_heartbeat_once(
                heartbeat_status=status,
                reason=status.status,
                now_monotonic=now_monotonic,
            )
            if restarted:
                return status
            # Check if cooldown suppressed
            decision = self._restart_policy.evaluate(
                now_monotonic=time.monotonic() if now_monotonic is None else now_monotonic
            )
            if decision.reason == "cooldown":
                return status
            # disabled / max exceeded → fall through to stop
            if self._config.stop_on_bad_heartbeat:
                self.request_stop()
            return status

        if self._config.stop_on_bad_heartbeat:
            self.request_stop()
        return status

    async def maybe_check_heartbeat(self, *, now_monotonic: float | None = None) -> HeartbeatStatus | None:
        if not self._config.heartbeat_check_enabled:
            return None
        now = time.monotonic() if now_monotonic is None else now_monotonic
        if self._last_heartbeat_check_monotonic > 0 and now - self._last_heartbeat_check_monotonic < self._config.heartbeat_check_interval_seconds:
            return self._last_heartbeat_status
        self._last_heartbeat_check_monotonic = now
        return await self.check_heartbeat_once(now_monotonic=now)

    # ------------------------------------------------------------------
    # event pipeline
    # ------------------------------------------------------------------

    async def process_child_events_once(self) -> object | None:
        """Call the injected event pipeline's ``process_once``, if configured.

        Returns the result object on success, or ``None`` when no pipeline
        is configured or when ``process_once`` raises an unexpected error.

        ``asyncio.CancelledError`` is **not** caught — it propagates so the
        supervisor's cancellation handling can clean up correctly.
        """
        if self._event_pipeline is None:
            return None

        try:
            return await self._event_pipeline.process_once()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "RECLAIM_SUPERVISOR_EVENT_PIPELINE_FAILED | child=%s error=%s",
                self._config.child_name,
                f"{type(exc).__name__}: {exc}",
            )
            self._append_health_event(
                event_type="EVENT_PIPELINE_FAILED",
                snapshot=self._child.snapshot() if self._child is not None else None,
                restart_suppressed_reason=f"{type(exc).__name__}: {exc}",
            )
            return None

    # ------------------------------------------------------------------
    # shutdown
    # ------------------------------------------------------------------

    async def shutdown(self) -> SupervisorShutdownResult:
        if self._shutdown_result is not None:
            return self._shutdown_result

        self._shutdown_started = True
        self._stop_requested = True

        child = self._child
        if child is None:
            result = SupervisorShutdownResult(
                child_name=self._config.child_name,
                child_started=False,
                child_running_before_shutdown=False,
                child_pid=None,
                child_returncode_before_shutdown=None,
                terminate_attempted=False,
                terminate_error=None,
            )
            self._shutdown_result = result
            return result

        snapshot = child.snapshot()
        terminate_attempted = snapshot.running
        terminate_error: str | None = None

        if snapshot.running:
            try:
                await child.terminate()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                terminate_error = f"{type(exc).__name__}: {exc}"
                logger.error(
                    "RECLAIM_SUPERVISOR_CHILD_TERMINATE_FAILED | name=%s pid=%s error=%s",
                    snapshot.name,
                    snapshot.pid,
                    terminate_error,
                )

        result = SupervisorShutdownResult(
            child_name=snapshot.name,
            child_started=snapshot.started,
            child_running_before_shutdown=snapshot.running,
            child_pid=snapshot.pid,
            child_returncode_before_shutdown=snapshot.returncode,
            terminate_attempted=terminate_attempted,
            terminate_error=terminate_error,
        )
        self._shutdown_result = result
        return result

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    async def run_forever(self) -> None:
        self._started_at_ms = live_time_utils.utc_ms()
        logger.info(
            "RECLAIM_SUPERVISOR_STARTED | mode=single_child child=%s poll_interval_seconds=%s",
            self._config.child_name,
            self._config.poll_interval_seconds,
        )
        self._child = self.create_child_process()
        try:
            snapshot = await self._child.start()
            self._child_started_monotonic = time.monotonic()
            logger.info(
                "RECLAIM_SUPERVISOR_CHILD_STARTED | name=%s pid=%s",
                snapshot.name,
                snapshot.pid,
            )
            while not self._stop_requested:
                await asyncio.sleep(self._config.poll_interval_seconds)

                # Process child event outbox before exit/heartbeat checks
                # so that events written just before child exit are not lost.
                await self.process_child_events_once()

                # Check child exit first; if restarted or cooldown-handled,
                # skip heartbeat this iteration to avoid double-processing.
                handled = await self.check_child_exit_once()
                if self._stop_requested:
                    break
                if handled:
                    self._last_heartbeat_check_monotonic = time.monotonic()
                    continue
                await self.maybe_check_heartbeat()
        except asyncio.CancelledError:
            logger.info("RECLAIM_SUPERVISOR_CANCELLED")
            raise
        finally:
            logger.info("RECLAIM_SUPERVISOR_STOPPING")
            result = await self.shutdown()
            logger.info(
                "RECLAIM_SUPERVISOR_STOPPED | child=%s terminate_attempted=%s terminate_error=%s",
                result.child_name,
                result.terminate_attempted,
                result.terminate_error,
            )
