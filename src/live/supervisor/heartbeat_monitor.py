from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from src.live import time_utils as live_time_utils


@dataclass(frozen=True)
class HeartbeatMonitorConfig:
    default_stale_after_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.default_stale_after_seconds <= 0:
            raise ValueError("default_stale_after_seconds must be > 0")


@dataclass(frozen=True)
class HeartbeatStatus:
    symbol: str
    path: Path
    status: str
    fresh: bool
    missing: bool
    stale: bool
    invalid: bool
    age_seconds: float | None
    sequence: int | None
    pid: int | None
    worker_status: str | None
    updated_at_ms: int | None
    stale_after_seconds: float
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.fresh and not self.missing and not self.stale and not self.invalid


class HeartbeatMonitor:
    def __init__(
        self,
        *,
        config: HeartbeatMonitorConfig | None = None,
        clock_ms=live_time_utils.utc_ms,
    ) -> None:
        self._config = config or HeartbeatMonitorConfig()
        self._clock_ms = clock_ms

    @property
    def config(self) -> HeartbeatMonitorConfig:
        return self._config

    def read_status(self, *, symbol: str, path: str | Path) -> HeartbeatStatus:
        path_obj = Path(path)
        if not path_obj.exists():
            return HeartbeatStatus(
                symbol=symbol,
                path=path_obj,
                status="missing",
                fresh=False,
                missing=True,
                stale=False,
                invalid=False,
                age_seconds=None,
                sequence=None,
                pid=None,
                worker_status=None,
                updated_at_ms=None,
                stale_after_seconds=self._config.default_stale_after_seconds,
                error="heartbeat file missing",
            )

        try:
            payload = json.loads(path_obj.read_text(encoding="utf-8"))
        except Exception as exc:
            return self._invalid(
                symbol=symbol,
                path=path_obj,
                error=f"{type(exc).__name__}: {exc}",
            )

        if not isinstance(payload, dict):
            return self._invalid(symbol=symbol, path=path_obj, error="heartbeat payload must be a JSON object")

        return self.evaluate_payload(symbol=symbol, path=path_obj, payload=payload)

    def evaluate_payload(
        self,
        *,
        symbol: str,
        path: str | Path,
        payload: Mapping[str, Any],
    ) -> HeartbeatStatus:
        path_obj = Path(path)
        stale_after_seconds = self._coerce_positive_float(
            payload.get("stale_after_seconds"),
            self._config.default_stale_after_seconds,
        )
        updated_at_ms = self._coerce_int(payload.get("updated_at_ms"))
        sequence = self._coerce_int(payload.get("sequence"))
        pid = self._coerce_int(payload.get("pid"))
        worker_status = self._coerce_str(payload.get("status"))

        if updated_at_ms is None:
            return self._invalid(
                symbol=symbol,
                path=path_obj,
                error="heartbeat updated_at_ms missing or invalid",
                stale_after_seconds=stale_after_seconds,
                sequence=sequence,
                pid=pid,
                worker_status=worker_status,
                updated_at_ms=None,
            )

        age_seconds = max((self._clock_ms() - updated_at_ms) / 1000.0, 0.0)
        stale = age_seconds > stale_after_seconds
        status = "stale" if stale else "fresh"
        return HeartbeatStatus(
            symbol=symbol,
            path=path_obj,
            status=status,
            fresh=not stale,
            missing=False,
            stale=stale,
            invalid=False,
            age_seconds=age_seconds,
            sequence=sequence,
            pid=pid,
            worker_status=worker_status,
            updated_at_ms=updated_at_ms,
            stale_after_seconds=stale_after_seconds,
            error=None,
        )

    def _invalid(
        self,
        *,
        symbol: str,
        path: Path,
        error: str,
        stale_after_seconds: float | None = None,
        sequence: int | None = None,
        pid: int | None = None,
        worker_status: str | None = None,
        updated_at_ms: int | None = None,
    ) -> HeartbeatStatus:
        return HeartbeatStatus(
            symbol=symbol,
            path=path,
            status="invalid",
            fresh=False,
            missing=False,
            stale=False,
            invalid=True,
            age_seconds=None,
            sequence=sequence,
            pid=pid,
            worker_status=worker_status,
            updated_at_ms=updated_at_ms,
            stale_after_seconds=stale_after_seconds or self._config.default_stale_after_seconds,
            error=error,
        )

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_str(value: Any) -> str | None:
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _coerce_positive_float(value: Any, default: float) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        return result if result > 0 else default
