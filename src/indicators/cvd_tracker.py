from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Literal

TradeSide = Literal["buy", "sell", "unknown"]


@dataclass(frozen=True)
class CvdTrackerConfig:
    fast_window_seconds: float = 5.0
    price_stall_seconds: float = 2.0
    price_stall_tolerance_pct: float = 0.0005


@dataclass(frozen=True)
class CvdSnapshot:
    ts_ms: int
    price: float
    side: TradeSide
    size: float
    signed_delta: float
    total_cvd: float
    fast_cvd: float
    previous_fast_cvd: float
    buy_volume: float
    sell_volume: float
    buy_ratio: float
    sell_ratio: float
    cross_positive: bool
    cross_negative: bool
    cvd_increasing: bool
    cvd_decreasing: bool
    no_new_low: bool
    no_new_high: bool
    window_low: float
    window_high: float


class CvdTracker:
    """Fast-window CVD tracker for BOLL reclaim strategy.

    The tracker only keeps a short rolling window in memory. It is designed for
    quick reclaim detection, not for long-session CVD analysis.
    """

    def __init__(self, config: CvdTrackerConfig):
        self.config = config
        self._events: Deque[tuple[int, float, float, float, float]] = deque()
        self._total_cvd: float = 0.0
        self._last_fast_cvd: float = 0.0

    def update(self, side: str, size: float, price: float, ts_ms: int) -> CvdSnapshot:
        normalized_side = self._normalize_side(side)
        signed_delta = self._signed_delta(normalized_side, size)
        self._total_cvd += signed_delta

        buy_volume = size if normalized_side == "buy" else 0.0
        sell_volume = size if normalized_side == "sell" else 0.0
        self._events.append((ts_ms, price, signed_delta, buy_volume, sell_volume))
        self._drop_old_events(ts_ms)

        previous_fast_cvd = self._last_fast_cvd
        fast_cvd = sum(item[2] for item in self._events)
        window_buy_volume = sum(item[3] for item in self._events)
        window_sell_volume = sum(item[4] for item in self._events)
        total_volume = window_buy_volume + window_sell_volume
        buy_ratio = window_buy_volume / total_volume if total_volume > 0 else 0.0
        sell_ratio = window_sell_volume / total_volume if total_volume > 0 else 0.0

        prices = [item[1] for item in self._events]
        window_low = min(prices) if prices else price
        window_high = max(prices) if prices else price
        no_new_low = self._is_no_new_low(ts_ms, price)
        no_new_high = self._is_no_new_high(ts_ms, price)

        cross_positive = previous_fast_cvd <= 0 < fast_cvd
        cross_negative = previous_fast_cvd >= 0 > fast_cvd
        cvd_increasing = fast_cvd > previous_fast_cvd
        cvd_decreasing = fast_cvd < previous_fast_cvd
        self._last_fast_cvd = fast_cvd

        return CvdSnapshot(
            ts_ms=ts_ms,
            price=price,
            side=normalized_side,
            size=size,
            signed_delta=signed_delta,
            total_cvd=self._total_cvd,
            fast_cvd=fast_cvd,
            previous_fast_cvd=previous_fast_cvd,
            buy_volume=window_buy_volume,
            sell_volume=window_sell_volume,
            buy_ratio=buy_ratio,
            sell_ratio=sell_ratio,
            cross_positive=cross_positive,
            cross_negative=cross_negative,
            cvd_increasing=cvd_increasing,
            cvd_decreasing=cvd_decreasing,
            no_new_low=no_new_low,
            no_new_high=no_new_high,
            window_low=window_low,
            window_high=window_high,
        )

    def _drop_old_events(self, ts_ms: int) -> None:
        cutoff = ts_ms - int(self.config.fast_window_seconds * 1000)
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def _is_no_new_low(self, ts_ms: int, current_price: float) -> bool:
        samples = self._stall_samples(ts_ms)
        if len(samples) < 2:
            return False
        min_ts, min_price = min(samples, key=lambda item: item[1])
        tolerance = abs(self.config.price_stall_tolerance_pct)
        low_is_old_enough = ts_ms - min_ts >= int(self.config.price_stall_seconds * 1000)
        recovered_from_low = current_price >= min_price * (1 + tolerance)
        return low_is_old_enough or recovered_from_low

    def _is_no_new_high(self, ts_ms: int, current_price: float) -> bool:
        samples = self._stall_samples(ts_ms)
        if len(samples) < 2:
            return False
        max_ts, max_price = max(samples, key=lambda item: item[1])
        tolerance = abs(self.config.price_stall_tolerance_pct)
        high_is_old_enough = ts_ms - max_ts >= int(self.config.price_stall_seconds * 1000)
        pulled_back_from_high = current_price <= max_price * (1 - tolerance)
        return high_is_old_enough or pulled_back_from_high

    def _stall_samples(self, ts_ms: int) -> list[tuple[int, float]]:
        cutoff = ts_ms - int(self.config.price_stall_seconds * 1000)
        return [(item[0], item[1]) for item in self._events if item[0] >= cutoff]

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
