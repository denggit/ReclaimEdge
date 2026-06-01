from __future__ import annotations

import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from statistics import mean
from typing import Deque, Literal

TradeSide = Literal["buy", "sell", "unknown"]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CvdTrackerConfig:
    fast_window_seconds: float = 5.0
    price_stall_seconds: float = 2.0
    price_stall_tolerance_pct: float = 0.0005
    burst_window_seconds: float = 3.0
    burst_baseline_seconds: float = 60.0
    burst_min_move_ratio: float = 2.5
    burst_min_volume_ratio: float = 2.0

    @classmethod
    def from_env(cls) -> "CvdTrackerConfig":
        return cls(
            fast_window_seconds=float(os.getenv("CVD_FAST_WINDOW_SECONDS", "5")),
            price_stall_seconds=float(os.getenv("PRICE_STALL_SECONDS", "2")),
            price_stall_tolerance_pct=float(os.getenv("PRICE_STALL_TOLERANCE_PCT", "0.0005")),
            burst_window_seconds=float(os.getenv("BURST_WINDOW_SECONDS", "3")),
            burst_baseline_seconds=float(os.getenv("BURST_BASELINE_SECONDS", "60")),
            burst_min_move_ratio=float(os.getenv("BURST_MIN_MOVE_RATIO", "2.5")),
            burst_min_volume_ratio=float(os.getenv("BURST_MIN_VOLUME_RATIO", "2.0")),
        )


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
    burst_net_move_pct: float
    burst_range_pct: float
    baseline_range_pct: float
    burst_move_ratio: float
    burst_volume: float
    baseline_volume: float
    burst_volume_ratio: float
    up_burst: bool
    down_burst: bool


class CvdTracker:
    """Fast-window CVD tracker for BOLL reclaim strategy.

    Burst detection is relative, not absolute:
    - recent short-window price range must be N times larger than previous
      baseline short-window ranges
    - recent volume intensity must be N times larger than baseline intensity

    This blocks slow grinding moves that crawl along the BOLL band.
    """

    def __init__(self, config: CvdTrackerConfig):
        self.config = config
        self._events: Deque[tuple[int, float, float, float, float, float]] = deque()
        self._total_cvd: float = 0.0
        self._last_fast_cvd: float = 0.0
        self._last_tick_ts_ms: int | None = None
        self._last_out_of_order_log_monotonic: float = 0.0

    def update(self, side: str, size: float, price: float, ts_ms: int) -> CvdSnapshot:
        normalized_side = self._normalize_side(side)
        if self._last_tick_ts_ms is not None and ts_ms < self._last_tick_ts_ms:
            now_monotonic = time.monotonic()
            if self._last_out_of_order_log_monotonic == 0.0 or now_monotonic - self._last_out_of_order_log_monotonic >= 5:
                logger.warning(
                    "CVD_TICK_OUT_OF_ORDER | last_ts_ms=%s current_ts_ms=%s price=%.4f side=%s size=%.8f",
                    self._last_tick_ts_ms,
                    ts_ms,
                    price,
                    normalized_side,
                    size,
                )
                self._last_out_of_order_log_monotonic = now_monotonic
        else:
            self._last_tick_ts_ms = ts_ms

        signed_delta = self._signed_delta(normalized_side, size)
        self._total_cvd += signed_delta

        buy_volume = size if normalized_side == "buy" else 0.0
        sell_volume = size if normalized_side == "sell" else 0.0
        total_trade_volume = buy_volume + sell_volume
        self._events.append((ts_ms, price, signed_delta, buy_volume, sell_volume, total_trade_volume))
        self._drop_old_events(ts_ms)

        fast_events = self._events_since(ts_ms, self.config.fast_window_seconds)
        previous_fast_cvd = self._last_fast_cvd
        fast_cvd = sum(item[2] for item in fast_events)
        window_buy_volume = sum(item[3] for item in fast_events)
        window_sell_volume = sum(item[4] for item in fast_events)
        total_volume = window_buy_volume + window_sell_volume
        buy_ratio = window_buy_volume / total_volume if total_volume > 0 else 0.0
        sell_ratio = window_sell_volume / total_volume if total_volume > 0 else 0.0

        prices = [item[1] for item in fast_events]
        window_low = min(prices) if prices else price
        window_high = max(prices) if prices else price
        no_new_low = self._is_no_new_low(ts_ms, price)
        no_new_high = self._is_no_new_high(ts_ms, price)

        cross_positive = previous_fast_cvd <= 0 < fast_cvd
        cross_negative = previous_fast_cvd >= 0 > fast_cvd
        cvd_increasing = fast_cvd > previous_fast_cvd
        cvd_decreasing = fast_cvd < previous_fast_cvd
        self._last_fast_cvd = fast_cvd

        burst = self._burst_stats(ts_ms, price)

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
            burst_net_move_pct=float(burst["net_move_pct"]),
            burst_range_pct=float(burst["range_pct"]),
            baseline_range_pct=float(burst["baseline_range_pct"]),
            burst_move_ratio=float(burst["move_ratio"]),
            burst_volume=float(burst["burst_volume"]),
            baseline_volume=float(burst["baseline_volume"]),
            burst_volume_ratio=float(burst["volume_ratio"]),
            up_burst=bool(burst["up_burst"]),
            down_burst=bool(burst["down_burst"]),
        )

    def _drop_old_events(self, ts_ms: int) -> None:
        keep_seconds = max(
            self.config.fast_window_seconds,
            self.config.price_stall_seconds,
            self.config.burst_window_seconds,
            self.config.burst_baseline_seconds,
        )
        cutoff = ts_ms - int(keep_seconds * 1000)
        self._events = deque(item for item in self._events if item[0] >= cutoff)

    def _events_since(self, ts_ms: int, seconds: float) -> list[tuple[int, float, float, float, float, float]]:
        cutoff = ts_ms - int(seconds * 1000)
        return sorted((item for item in self._events if cutoff <= item[0] <= ts_ms), key=lambda item: item[0])

    def _burst_stats(self, ts_ms: int, current_price: float) -> dict[str, float | bool]:
        burst_ms = int(self.config.burst_window_seconds * 1000)
        baseline_ms = int(self.config.burst_baseline_seconds * 1000)
        burst_cutoff = ts_ms - burst_ms
        baseline_cutoff = ts_ms - baseline_ms

        burst_events = sorted(
            (item for item in self._events if burst_cutoff <= item[0] <= ts_ms),
            key=lambda item: item[0],
        )
        baseline_events = sorted(
            (item for item in self._events if baseline_cutoff <= item[0] < burst_cutoff),
            key=lambda item: item[0],
        )

        empty = {
            "net_move_pct": 0.0,
            "range_pct": 0.0,
            "baseline_range_pct": 0.0,
            "move_ratio": 0.0,
            "burst_volume": 0.0,
            "baseline_volume": 0.0,
            "volume_ratio": 0.0,
            "up_burst": False,
            "down_burst": False,
        }
        if len(burst_events) < 2 or len(baseline_events) < 2 or current_price <= 0:
            return empty

        start_price = burst_events[0][1]
        net_move_pct = (current_price - start_price) / start_price if start_price > 0 else 0.0
        burst_range_pct = self._range_pct(burst_events)
        baseline_range_pct = self._baseline_avg_range_pct(ts_ms, baseline_events)
        if baseline_range_pct <= 0:
            return empty

        burst_volume = sum(item[5] for item in burst_events)
        baseline_volume = sum(item[5] for item in baseline_events)
        burst_vps = burst_volume / max(self.config.burst_window_seconds, 0.001)
        baseline_elapsed_seconds = (baseline_events[-1][0] - baseline_events[0][0]) / 1000
        if baseline_elapsed_seconds <= 0:
            return empty
        baseline_vps = baseline_volume / baseline_elapsed_seconds if baseline_volume > 0 else 0.0

        move_ratio = burst_range_pct / baseline_range_pct
        volume_ratio = burst_vps / baseline_vps if baseline_vps > 0 else 0.0
        enough_move = move_ratio >= self.config.burst_min_move_ratio
        enough_volume = volume_ratio >= self.config.burst_min_volume_ratio

        return {
            "net_move_pct": net_move_pct,
            "range_pct": burst_range_pct,
            "baseline_range_pct": baseline_range_pct,
            "move_ratio": move_ratio,
            "burst_volume": burst_volume,
            "baseline_volume": baseline_volume,
            "volume_ratio": volume_ratio,
            "up_burst": net_move_pct > 0 and enough_move and enough_volume,
            "down_burst": net_move_pct < 0 and enough_move and enough_volume,
        }

    def _baseline_avg_range_pct(self, ts_ms: int, baseline_events: list[tuple[int, float, float, float, float, float]]) -> float:
        window_ms = int(self.config.burst_window_seconds * 1000)
        sorted_events = sorted(baseline_events, key=lambda item: item[0])
        ranges: list[float] = []
        for end_event in sorted_events:
            end_ts = end_event[0]
            start_ts = end_ts - window_ms
            samples = [item for item in sorted_events if start_ts <= item[0] <= end_ts]
            if len(samples) >= 2:
                ranges.append(self._range_pct(samples))
        return mean(ranges) if ranges else 0.0

    @staticmethod
    def _range_pct(events: list[tuple[int, float, float, float, float, float]]) -> float:
        prices = [item[1] for item in events if item[1] > 0]
        if len(prices) < 2:
            return 0.0
        mid = sum(prices) / len(prices)
        return (max(prices) - min(prices)) / mid if mid > 0 else 0.0

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
        return [(item[0], item[1]) for item in sorted(self._events, key=lambda item: item[0]) if cutoff <= item[0] <= ts_ms]

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
