from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Literal

TradeSide = Literal["buy", "sell", "unknown"]
CvdReversal = Literal["NONE", "CROSS_POSITIVE", "CROSS_NEGATIVE"]


@dataclass(frozen=True)
class CvdTrackerConfig:
    window_seconds: int = 60
    min_reversal_delta: float = 0.0


@dataclass(frozen=True)
class CvdSnapshot:
    ts_ms: int
    price: float
    side: TradeSide
    size: float
    signed_delta: float
    total_cvd: float
    window_cvd: float
    buy_volume: float
    sell_volume: float
    reversal: CvdReversal


class CvdTracker:
    """Realtime rolling CVD tracker.

    OKX public trades provide side and size. We treat buy-side trades as positive
    delta and sell-side trades as negative delta.

    The tracker keeps only a rolling time window in memory, so memory usage is
    bounded by recent tick volume rather than total runtime.
    """

    def __init__(self, config: CvdTrackerConfig):
        self.config = config
        self._events: Deque[tuple[int, float, float, float]] = deque()
        self._total_cvd: float = 0.0
        self._last_window_cvd: float | None = None

    def update(self, side: str, size: float, price: float, ts_ms: int) -> CvdSnapshot:
        normalized_side = self._normalize_side(side)
        signed_delta = self._signed_delta(normalized_side, size)
        self._total_cvd += signed_delta

        buy_volume = size if normalized_side == "buy" else 0.0
        sell_volume = size if normalized_side == "sell" else 0.0
        self._events.append((ts_ms, signed_delta, buy_volume, sell_volume))
        self._drop_old_events(ts_ms)

        window_cvd = sum(item[1] for item in self._events)
        window_buy_volume = sum(item[2] for item in self._events)
        window_sell_volume = sum(item[3] for item in self._events)
        reversal = self._detect_reversal(window_cvd)
        self._last_window_cvd = window_cvd

        return CvdSnapshot(
            ts_ms=ts_ms,
            price=price,
            side=normalized_side,
            size=size,
            signed_delta=signed_delta,
            total_cvd=self._total_cvd,
            window_cvd=window_cvd,
            buy_volume=window_buy_volume,
            sell_volume=window_sell_volume,
            reversal=reversal,
        )

    def latest(self) -> CvdSnapshot | None:
        if not self._events:
            return None
        latest_ts = self._events[-1][0]
        window_cvd = sum(item[1] for item in self._events)
        buy_volume = sum(item[2] for item in self._events)
        sell_volume = sum(item[3] for item in self._events)
        return CvdSnapshot(
            ts_ms=latest_ts,
            price=0.0,
            side="unknown",
            size=0.0,
            signed_delta=0.0,
            total_cvd=self._total_cvd,
            window_cvd=window_cvd,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            reversal="NONE",
        )

    def _drop_old_events(self, ts_ms: int) -> None:
        cutoff = ts_ms - self.config.window_seconds * 1000
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def _detect_reversal(self, window_cvd: float) -> CvdReversal:
        previous = self._last_window_cvd
        threshold = abs(self.config.min_reversal_delta)
        if previous is None:
            return "NONE"
        if previous <= 0 < window_cvd and abs(window_cvd) >= threshold:
            return "CROSS_POSITIVE"
        if previous >= 0 > window_cvd and abs(window_cvd) >= threshold:
            return "CROSS_NEGATIVE"
        return "NONE"

    @staticmethod
    def _normalize_side(side: str) -> TradeSide:
        text = side.strip().lower()
        if text == "buy":
            return "buy"
        if text == "sell":
            return "sell"
        return "unknown"

    @staticmethod
    def _signed_delta(side: TradeSide, size: float) -> float:
        if side == "buy":
            return abs(size)
        if side == "sell":
            return -abs(size)
        return 0.0
