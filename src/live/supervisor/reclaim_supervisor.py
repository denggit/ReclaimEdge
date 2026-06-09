from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.live import time_utils as live_time_utils
from src.live.runtime_paths import RuntimePaths
from src.live.supervisor.child_process import ChildProcess, ChildProcessSnapshot, ChildProcessSpec
from src.live.supervisor.heartbeat_monitor import HeartbeatMonitor, HeartbeatMonitorConfig, HeartbeatStatus
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

    Captures child exit and heartbeat problems for downstream consumers
    (logging, future outbox).  Never writes to disk — stays in memory.
    """

    event_type: str
    child_name: str
    ts_ms: int
    pid: int | None
    returncode: int | None
    heartbeat_status: str | None = None
    heartbeat_age_seconds: float | None = None
    heartbeat_error: str | None = None


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


class ReclaimSupervisor:
    def __init__(self, *, config: ReclaimSupervisorConfig | None = None) -> None:
        self._config = config or ReclaimSupervisorConfig()
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

    @classmethod
    def from_env(cls) -> "ReclaimSupervisor":
        # D05 intentionally does not parse symbol lists or child configs from env.
        return cls()

    @property
    def config(self) -> ReclaimSupervisorConfig:
        return self._config

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
            env=None,
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
        )
        self._health_events.append(event)
        return event

    # ------------------------------------------------------------------
    # child exit detection
    # ------------------------------------------------------------------

    def check_child_exit_once(self) -> SupervisorHealthEvent | None:
        if self._child is None:
            return None
        snapshot = self._child.snapshot()
        if snapshot.started and not snapshot.running and snapshot.returncode is not None:
            event = self._append_health_event(event_type="CHILD_EXITED", snapshot=snapshot)
            logger.error(
                "RECLAIM_SUPERVISOR_CHILD_EXITED | child=%s pid=%s returncode=%s stop_on_child_exit=%s",
                snapshot.name,
                snapshot.pid,
                snapshot.returncode,
                self._config.stop_on_child_exit,
            )
            if self._config.stop_on_child_exit:
                self.request_stop()
            return event
        return None

    # ------------------------------------------------------------------
    # heartbeat detection
    # ------------------------------------------------------------------

    def _heartbeat_in_startup_grace(self, now_monotonic: float | None = None) -> bool:
        if self._child_started_monotonic is None:
            return False
        now = time.monotonic() if now_monotonic is None else now_monotonic
        return now - self._child_started_monotonic < self._config.heartbeat_startup_grace_seconds

    def check_heartbeat_once(self, *, now_monotonic: float | None = None) -> HeartbeatStatus | None:
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
        event = self._append_health_event(
            event_type=event_type,
            snapshot=snapshot,
            heartbeat_status=status,
        )
        logger.error(
            "RECLAIM_SUPERVISOR_HEARTBEAT_BAD | child=%s heartbeat_status=%s path=%s age_seconds=%s error=%s stop_on_bad_heartbeat=%s",
            self._config.child_name,
            status.status,
            status.path,
            status.age_seconds,
            status.error,
            self._config.stop_on_bad_heartbeat,
        )
        if self._config.stop_on_bad_heartbeat:
            self.request_stop()
        return status

    def maybe_check_heartbeat(self, *, now_monotonic: float | None = None) -> HeartbeatStatus | None:
        if not self._config.heartbeat_check_enabled:
            return None
        now = time.monotonic() if now_monotonic is None else now_monotonic
        if self._last_heartbeat_check_monotonic > 0 and now - self._last_heartbeat_check_monotonic < self._config.heartbeat_check_interval_seconds:
            return self._last_heartbeat_status
        self._last_heartbeat_check_monotonic = now
        return self.check_heartbeat_once(now_monotonic=now)

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
                self.check_child_exit_once()
                if self._stop_requested:
                    break
                self.maybe_check_heartbeat()
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
