from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.live import time_utils as live_time_utils
from src.live.runtime_paths import RuntimePaths
from src.utils.log import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class HeartbeatWriterConfig:
    """Configuration for :class:`HeartbeatWriter`.

    Defaults to ``enabled=False`` so heartbeat file writing is opt-in.
    """

    enabled: bool = False
    interval_seconds: float = 10.0
    stale_after_seconds: float = 30.0
    failure_log_interval_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0:
            raise ValueError("heartbeat interval_seconds must be > 0")
        if self.stale_after_seconds <= 0:
            raise ValueError("heartbeat stale_after_seconds must be > 0")
        if self.failure_log_interval_seconds <= 0:
            raise ValueError("heartbeat failure_log_interval_seconds must be > 0")


@dataclass(frozen=True)
class HeartbeatWriteResult:
    """Result of a single ``write_once`` call."""

    wrote: bool
    path: Path
    reason: str | None = None
    sequence: int = 0


class HeartbeatWriter:
    """Writes a low-frequency heartbeat JSON file for a supervisor process.

    The writer is **disabled by default**.  When disabled, ``write_once`` is
    a no-op — it does not create directories or write files.

    When enabled, ``write_once`` atomically writes a heartbeat JSON file to
    ``runtime_paths.heartbeat_file``.  The file contains the fields a future
    supervisor process needs to determine whether the symbol worker is
    running, stuck, stale, exited, or in need of a restart.

    C05 does **not** integrate this writer into the live runtime — the
    live entry script, the app runner, and the tick-path workers are all
    unchanged.
    """

    def __init__(
        self,
        *,
        runtime_paths: RuntimePaths,
        config: HeartbeatWriterConfig | None = None,
        clock_ms: Callable[[], int] = live_time_utils.utc_ms,
        pid_provider: Callable[[], int] = os.getpid,
    ) -> None:
        self._runtime_paths = runtime_paths
        self._config = config or HeartbeatWriterConfig()
        self._clock_ms = clock_ms
        self._pid_provider = pid_provider
        self._sequence = 0
        self._started_at_ms = self._clock_ms()
        self._consecutive_failures = 0
        self._last_error: str | None = None
        self._last_failure_log_monotonic = 0.0

    @property
    def config(self) -> HeartbeatWriterConfig:
        return self._config

    @property
    def path(self) -> Path:
        return self._runtime_paths.heartbeat_file

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._last_error = None

    def _record_failure(self, exc: Exception) -> None:
        self._consecutive_failures += 1
        self._last_error = f"{type(exc).__name__}: {exc}"

    def build_payload(self, *, status: str = "running") -> dict:
        """Build the heartbeat payload dict without writing to disk.

        The returned payload includes ``sequence`` set to the **next**
        sequence number, but ``write_once`` has not been called — the
        writer's internal ``_sequence`` counter is unchanged.
        """
        now_ms = self._clock_ms()
        return {
            "schema_version": 1,
            "inst_id": self._runtime_paths.symbol_slug,
            "symbol_slug": self._runtime_paths.symbol_slug,
            "pid": self._pid_provider(),
            "status": status,
            "sequence": self._sequence + 1,
            "started_at_ms": self._started_at_ms,
            "updated_at_ms": now_ms,
            "stale_after_seconds": self._config.stale_after_seconds,
        }

    def write_once(self, *, status: str = "running") -> HeartbeatWriteResult:
        """Write the heartbeat file once (synchronously).

        When the writer is **disabled** this is a true no-op: no directory
        is created and no file is written.

        When **enabled** the heartbeat JSON is written atomically via a
        temporary file + ``os.replace``.
        """
        if not self._config.enabled:
            return HeartbeatWriteResult(
                wrote=False,
                path=self.path,
                reason="disabled",
                sequence=self._sequence,
            )

        payload = self.build_payload(status=status)
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = path.with_name(f".{path.name}.{self._pid_provider()}.tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
        self._sequence = int(payload["sequence"])
        self._record_success()

        return HeartbeatWriteResult(
            wrote=True,
            path=path,
            reason=None,
            sequence=self._sequence,
        )

    async def run_until_cancelled(
        self,
        *,
        status_provider: Callable[[], str] | None = None,
    ) -> None:
        """Run the heartbeat writer loop until cancelled.

        When the writer is **disabled** this returns immediately without
        writing a file.

        When **enabled** this writes the heartbeat file at the configured
        interval until the coroutine is cancelled.  ``CancelledError`` is
        **not** swallowed — it propagates naturally so the caller can
        handle cancellation.

        Write failures are **degraded**: a single failed write does not
        kill the loop.  Consecutive failures are counted and logged at a
        throttled interval so the supervisor can still detect a stuck
        worker.
        """
        if not self._config.enabled:
            return

        provider = status_provider or (lambda: "running")
        while True:
            try:
                self.write_once(status=provider())
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._record_failure(exc)
                now = time.monotonic()
                if now - self._last_failure_log_monotonic >= self._config.failure_log_interval_seconds:
                    logger.warning(
                        "HEARTBEAT_WRITE_FAILED | path=%s consecutive_failures=%s error=%s",
                        self.path,
                        self._consecutive_failures,
                        self._last_error,
                    )
                    self._last_failure_log_monotonic = now
            await asyncio.sleep(self._config.interval_seconds)
