from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.live import time_utils as live_time_utils
from src.live.runtime_paths import RuntimePaths


@dataclass(frozen=True)
class HeartbeatWriterConfig:
    """Configuration for :class:`HeartbeatWriter`.

    Defaults to ``enabled=False`` so heartbeat file writing is opt-in.
    """

    enabled: bool = False
    interval_seconds: float = 10.0
    stale_after_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0:
            raise ValueError("heartbeat interval_seconds must be > 0")
        if self.stale_after_seconds <= 0:
            raise ValueError("heartbeat stale_after_seconds must be > 0")


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

    @property
    def config(self) -> HeartbeatWriterConfig:
        return self._config

    @property
    def path(self) -> Path:
        return self._runtime_paths.heartbeat_file

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

        C05 does **not** call this method from any live runtime path.
        """
        if not self._config.enabled:
            return

        provider = status_provider or (lambda: "running")
        while True:
            self.write_once(status=provider())
            await asyncio.sleep(self._config.interval_seconds)
