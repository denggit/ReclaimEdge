from __future__ import annotations

import asyncio
from dataclasses import dataclass

from src.live import time_utils as live_time_utils
from src.utils.log import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ReclaimSupervisorConfig:
    poll_interval_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.poll_interval_seconds <= 0:
            raise ValueError("supervisor poll_interval_seconds must be > 0")


class ReclaimSupervisor:
    def __init__(self, *, config: ReclaimSupervisorConfig | None = None) -> None:
        self._config = config or ReclaimSupervisorConfig()
        self._started_at_ms: int | None = None
        self._stop_requested = False

    @classmethod
    def from_env(cls) -> "ReclaimSupervisor":
        # D02 intentionally does not parse symbol lists or child configs.
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

    def request_stop(self) -> None:
        self._stop_requested = True

    async def run_forever(self) -> None:
        self._started_at_ms = live_time_utils.utc_ms()
        logger.info(
            "RECLAIM_SUPERVISOR_STARTED | mode=empty_shell poll_interval_seconds=%s",
            self._config.poll_interval_seconds,
        )
        try:
            while not self._stop_requested:
                await asyncio.sleep(self._config.poll_interval_seconds)
        except asyncio.CancelledError:
            logger.info("RECLAIM_SUPERVISOR_CANCELLED")
            raise
        finally:
            logger.info("RECLAIM_SUPERVISOR_STOPPED")
