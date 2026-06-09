from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path

from src.live import time_utils as live_time_utils
from src.live.supervisor.child_process import ChildProcess, ChildProcessSnapshot, ChildProcessSpec
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
class ReclaimSupervisorConfig:
    poll_interval_seconds: float = 5.0
    project_root: Path = field(default_factory=_default_project_root)
    child_name: str = "ETH-USDT-SWAP"
    worker_script: Path = Path("scripts/run_symbol_worker.py")
    child_terminate_timeout_seconds: float = 10.0
    child_kill_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.poll_interval_seconds <= 0:
            raise ValueError("supervisor poll_interval_seconds must be > 0")
        if not self.child_name.strip():
            raise ValueError("supervisor child_name must not be empty")
        if self.child_terminate_timeout_seconds <= 0:
            raise ValueError("child_terminate_timeout_seconds must be > 0")
        if self.child_kill_timeout_seconds <= 0:
            raise ValueError("child_kill_timeout_seconds must be > 0")
        object.__setattr__(self, "project_root", Path(self.project_root))
        object.__setattr__(self, "worker_script", Path(self.worker_script))


class ReclaimSupervisor:
    def __init__(self, *, config: ReclaimSupervisorConfig | None = None) -> None:
        self._config = config or ReclaimSupervisorConfig()
        self._started_at_ms: int | None = None
        self._stop_requested = False
        self._child: ChildProcess | None = None
        self._shutdown_started = False
        self._shutdown_result: SupervisorShutdownResult | None = None

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
            logger.info(
                "RECLAIM_SUPERVISOR_CHILD_STARTED | name=%s pid=%s",
                snapshot.name,
                snapshot.pid,
            )
            while not self._stop_requested:
                await asyncio.sleep(self._config.poll_interval_seconds)
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
