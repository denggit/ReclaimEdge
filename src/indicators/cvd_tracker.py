from __future__ import annotations

import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Literal

TradeSide = Literal["buy", "sell", "unknown"]
OutOfOrderPolicy = Literal["drop_for_realtime", "accept_slow_debug"]
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


@dataclass(frozen=True)
class Event:
    ts_ms: int
    price: float
    signed_delta: float
    buy_volume: float
    sell_volume: float
    volume: float


@dataclass(frozen=True)
class RangeSample:
    ts_ms: int
    range_pct: float


class CvdTracker:
    """Fast-window CVD tracker for BOLL reclaim strategy.

    CVD update is a live hot path. Do not add per-tick sorting,
    full-window scans, or nested loops.

    Burst detection is relative, not absolute:
    - recent short-window price range must be N times larger than previous
      baseline short-window ranges
    - recent volume intensity must be N times larger than baseline intensity

    This blocks slow grinding moves that crawl along the BOLL band.
    """

    def __init__(self, config: CvdTrackerConfig):
        self.config = config
        self._fast_window_ms = int(config.fast_window_seconds * 1000)
        self._stall_window_ms = int(config.price_stall_seconds * 1000)
        self._burst_window_ms = int(config.burst_window_seconds * 1000)
        self._baseline_window_ms = int(config.burst_baseline_seconds * 1000)

        self._fast_events: Deque[Event] = deque()
        self._fast_signed_sum: float = 0.0
        self._fast_buy_sum: float = 0.0
        self._fast_sell_sum: float = 0.0

        self._stall_events: Deque[Event] = deque()
        self._stall_min_price: Deque[tuple[int, float]] = deque()
        self._stall_max_price: Deque[tuple[int, float]] = deque()

        self._burst_events: Deque[Event] = deque()
        self._burst_volume_sum: float = 0.0
        self._burst_min_price: Deque[tuple[int, float]] = deque()
        self._burst_max_price: Deque[tuple[int, float]] = deque()

        self._recent_for_baseline: Deque[Event] = deque()
        self._baseline_events: Deque[Event] = deque()
        self._baseline_volume_sum: float = 0.0

        self._recent_range_samples: Deque[RangeSample] = deque()
        self._baseline_range_samples: Deque[RangeSample] = deque()
        self._baseline_range_sum: float = 0.0

        self._total_cvd: float = 0.0
        self._last_fast_cvd: float = 0.0
        self._last_tick_ts_ms: int | None = None
        self._last_snapshot: CvdSnapshot | None = None
        self._last_out_of_order_log_monotonic: float = 0.0
        self._last_slow_log_monotonic: float = 0.0
        self._out_of_order_policy = self._load_out_of_order_policy()
        self._slow_log_threshold_ms = float(os.getenv("CVD_UPDATE_SLOW_LOG_MS", "20"))

    def update(self, side: str, size: float, price: float, ts_ms: int) -> CvdSnapshot:
        started_monotonic = time.monotonic() if self._slow_log_threshold_ms > 0 else None
        normalized_side = self._normalize_side(side)
        signed_delta = self._signed_delta(normalized_side, size)

        if self._last_tick_ts_ms is not None and ts_ms < self._last_tick_ts_ms:
            self._log_out_of_order_tick(ts_ms, price, normalized_side, size)
            snapshot = self._snapshot_without_mutating_windows(ts_ms, price, normalized_side, size, signed_delta)
            self._log_slow_update_if_needed(started_monotonic)
            return snapshot

        self._last_tick_ts_ms = ts_ms
        self._total_cvd += signed_delta

        buy_volume = size if normalized_side == "buy" else 0.0
        sell_volume = size if normalized_side == "sell" else 0.0
        event = Event(
            ts_ms=ts_ms,
            price=price,
            signed_delta=signed_delta,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            volume=buy_volume + sell_volume,
        )

        previous_fast_cvd = self._last_fast_cvd
        self._append_fast_event(event, ts_ms)
        self._append_stall_event(event, ts_ms)
        self._append_burst_event(event, ts_ms)
        self._append_baseline_event(event, ts_ms)

        fast_cvd = self._fast_signed_sum
        total_volume = self._fast_buy_sum + self._fast_sell_sum
        buy_ratio = self._fast_buy_sum / total_volume if total_volume > 0 else 0.0
        sell_ratio = self._fast_sell_sum / total_volume if total_volume > 0 else 0.0

        window_low = self._stall_min_price[0][1] if self._stall_min_price else price
        window_high = self._stall_max_price[0][1] if self._stall_max_price else price
        no_new_low = self._is_no_new_low(ts_ms, price)
        no_new_high = self._is_no_new_high(ts_ms, price)

        burst_net_move_pct = self._burst_net_move_pct(price)
        burst_range_pct = self._current_burst_range_pct()
        self._append_range_sample(RangeSample(ts_ms, burst_range_pct), ts_ms)
        baseline_range_pct = self._baseline_range_pct()

        burst_move_ratio = burst_range_pct / baseline_range_pct if baseline_range_pct > 0 else 0.0
        burst_volume = self._burst_volume_sum
        baseline_volume = self._baseline_volume_sum
        burst_vps = burst_volume / max(self.config.burst_window_seconds, 0.001)
        baseline_vps = self._baseline_volume_per_second()
        burst_volume_ratio = burst_vps / baseline_vps if baseline_vps > 0 else 0.0
        enough_move = burst_move_ratio >= self.config.burst_min_move_ratio
        enough_volume = burst_volume_ratio >= self.config.burst_min_volume_ratio

        cross_positive = previous_fast_cvd <= 0 < fast_cvd
        cross_negative = previous_fast_cvd >= 0 > fast_cvd
        cvd_increasing = fast_cvd > previous_fast_cvd
        cvd_decreasing = fast_cvd < previous_fast_cvd
        self._last_fast_cvd = fast_cvd

        snapshot = CvdSnapshot(
            ts_ms=ts_ms,
            price=price,
            side=normalized_side,
            size=size,
            signed_delta=signed_delta,
            total_cvd=self._total_cvd,
            fast_cvd=fast_cvd,
            previous_fast_cvd=previous_fast_cvd,
            buy_volume=self._fast_buy_sum,
            sell_volume=self._fast_sell_sum,
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
            burst_net_move_pct=burst_net_move_pct,
            burst_range_pct=burst_range_pct,
            baseline_range_pct=baseline_range_pct,
            burst_move_ratio=burst_move_ratio,
            burst_volume=burst_volume,
            baseline_volume=baseline_volume,
            burst_volume_ratio=burst_volume_ratio,
            up_burst=burst_net_move_pct > 0 and enough_move and enough_volume,
            down_burst=burst_net_move_pct < 0 and enough_move and enough_volume,
        )
        self._last_snapshot = snapshot
        self._log_slow_update_if_needed(started_monotonic)
        return snapshot

    def _append_fast_event(self, event: Event, ts_ms: int) -> None:
        self._fast_events.append(event)
        self._fast_signed_sum += event.signed_delta
        self._fast_buy_sum += event.buy_volume
        self._fast_sell_sum += event.sell_volume
        cutoff = ts_ms - self._fast_window_ms
        while self._fast_events and self._fast_events[0].ts_ms < cutoff:
            old = self._fast_events.popleft()
            self._fast_signed_sum -= old.signed_delta
            self._fast_buy_sum -= old.buy_volume
            self._fast_sell_sum -= old.sell_volume

    def _append_stall_event(self, event: Event, ts_ms: int) -> None:
        self._stall_events.append(event)
        while self._stall_min_price and self._stall_min_price[-1][1] >= event.price:
            self._stall_min_price.pop()
        self._stall_min_price.append((event.ts_ms, event.price))
        while self._stall_max_price and self._stall_max_price[-1][1] <= event.price:
            self._stall_max_price.pop()
        self._stall_max_price.append((event.ts_ms, event.price))

        cutoff = ts_ms - self._stall_window_ms
        while self._stall_events and self._stall_events[0].ts_ms < cutoff:
            self._stall_events.popleft()
        self._drop_expired_monotonic_heads(self._stall_min_price, cutoff)
        self._drop_expired_monotonic_heads(self._stall_max_price, cutoff)

    def _append_burst_event(self, event: Event, ts_ms: int) -> None:
        self._burst_events.append(event)
        self._burst_volume_sum += event.volume
        while self._burst_min_price and self._burst_min_price[-1][1] >= event.price:
            self._burst_min_price.pop()
        self._burst_min_price.append((event.ts_ms, event.price))
        while self._burst_max_price and self._burst_max_price[-1][1] <= event.price:
            self._burst_max_price.pop()
        self._burst_max_price.append((event.ts_ms, event.price))

        cutoff = ts_ms - self._burst_window_ms
        while self._burst_events and self._burst_events[0].ts_ms < cutoff:
            old = self._burst_events.popleft()
            self._burst_volume_sum -= old.volume
        self._drop_expired_monotonic_heads(self._burst_min_price, cutoff)
        self._drop_expired_monotonic_heads(self._burst_max_price, cutoff)

    def _append_baseline_event(self, event: Event, ts_ms: int) -> None:
        self._recent_for_baseline.append(event)
        burst_cutoff = ts_ms - self._burst_window_ms
        while self._recent_for_baseline and self._recent_for_baseline[0].ts_ms <= burst_cutoff:
            baseline_event = self._recent_for_baseline.popleft()
            self._baseline_events.append(baseline_event)
            self._baseline_volume_sum += baseline_event.volume

        baseline_cutoff = ts_ms - self._baseline_window_ms
        while self._baseline_events and self._baseline_events[0].ts_ms < baseline_cutoff:
            old = self._baseline_events.popleft()
            self._baseline_volume_sum -= old.volume

    def _append_range_sample(self, sample: RangeSample, ts_ms: int) -> None:
        self._recent_range_samples.append(sample)
        burst_cutoff = ts_ms - self._burst_window_ms
        while self._recent_range_samples and self._recent_range_samples[0].ts_ms <= burst_cutoff:
            baseline_sample = self._recent_range_samples.popleft()
            self._baseline_range_samples.append(baseline_sample)
            self._baseline_range_sum += baseline_sample.range_pct

        baseline_cutoff = ts_ms - self._baseline_window_ms
        while self._baseline_range_samples and self._baseline_range_samples[0].ts_ms < baseline_cutoff:
            old = self._baseline_range_samples.popleft()
            self._baseline_range_sum -= old.range_pct

    @staticmethod
    def _drop_expired_monotonic_heads(items: Deque[tuple[int, float]], cutoff: int) -> None:
        while items and items[0][0] < cutoff:
            items.popleft()

    def _is_no_new_low(self, ts_ms: int, current_price: float) -> bool:
        if len(self._stall_events) < 2 or not self._stall_min_price:
            return False
        min_ts, min_price = self._stall_min_price[0]
        tolerance = abs(self.config.price_stall_tolerance_pct)
        low_is_old_enough = ts_ms - min_ts >= self._stall_window_ms
        recovered_from_low = current_price >= min_price * (1 + tolerance)
        return low_is_old_enough or recovered_from_low

    def _is_no_new_high(self, ts_ms: int, current_price: float) -> bool:
        if len(self._stall_events) < 2 or not self._stall_max_price:
            return False
        max_ts, max_price = self._stall_max_price[0]
        tolerance = abs(self.config.price_stall_tolerance_pct)
        high_is_old_enough = ts_ms - max_ts >= self._stall_window_ms
        pulled_back_from_high = current_price <= max_price * (1 - tolerance)
        return high_is_old_enough or pulled_back_from_high

    def _burst_net_move_pct(self, current_price: float) -> float:
        if len(self._burst_events) < 2 or current_price <= 0:
            return 0.0
        start_price = self._burst_events[0].price
        return (current_price - start_price) / start_price if start_price > 0 else 0.0

    def _current_burst_range_pct(self) -> float:
        if len(self._burst_events) < 2 or not self._burst_min_price or not self._burst_max_price:
            return 0.0
        min_price = self._burst_min_price[0][1]
        max_price = self._burst_max_price[0][1]
        mid_price = (min_price + max_price) / 2
        return (max_price - min_price) / mid_price if mid_price > 0 else 0.0

    def _baseline_range_pct(self) -> float:
        if not self._baseline_range_samples:
            return 0.0
        return self._baseline_range_sum / len(self._baseline_range_samples)

    def _baseline_volume_per_second(self) -> float:
        if len(self._baseline_events) < 2 or self._baseline_volume_sum <= 0:
            return 0.0
        elapsed_seconds = (self._baseline_events[-1].ts_ms - self._baseline_events[0].ts_ms) / 1000
        return self._baseline_volume_sum / elapsed_seconds if elapsed_seconds > 0 else 0.0

    def _snapshot_without_mutating_windows(
        self,
        ts_ms: int,
        price: float,
        side: TradeSide,
        size: float,
        signed_delta: float,
    ) -> CvdSnapshot:
        if self._last_snapshot is None:
            return CvdSnapshot(
                ts_ms=ts_ms,
                price=price,
                side=side,
                size=size,
                signed_delta=signed_delta,
                total_cvd=self._total_cvd,
                fast_cvd=self._fast_signed_sum,
                previous_fast_cvd=self._last_fast_cvd,
                buy_volume=self._fast_buy_sum,
                sell_volume=self._fast_sell_sum,
                buy_ratio=0.0,
                sell_ratio=0.0,
                cross_positive=False,
                cross_negative=False,
                cvd_increasing=False,
                cvd_decreasing=False,
                no_new_low=False,
                no_new_high=False,
                window_low=price,
                window_high=price,
                burst_net_move_pct=0.0,
                burst_range_pct=0.0,
                baseline_range_pct=0.0,
                burst_move_ratio=0.0,
                burst_volume=0.0,
                baseline_volume=0.0,
                burst_volume_ratio=0.0,
                up_burst=False,
                down_burst=False,
            )
        last = self._last_snapshot
        return CvdSnapshot(
            ts_ms=ts_ms,
            price=price,
            side=side,
            size=size,
            signed_delta=signed_delta,
            total_cvd=last.total_cvd,
            fast_cvd=last.fast_cvd,
            previous_fast_cvd=last.previous_fast_cvd,
            buy_volume=last.buy_volume,
            sell_volume=last.sell_volume,
            buy_ratio=last.buy_ratio,
            sell_ratio=last.sell_ratio,
            cross_positive=False,
            cross_negative=False,
            cvd_increasing=False,
            cvd_decreasing=False,
            no_new_low=last.no_new_low,
            no_new_high=last.no_new_high,
            window_low=last.window_low,
            window_high=last.window_high,
            burst_net_move_pct=last.burst_net_move_pct,
            burst_range_pct=last.burst_range_pct,
            baseline_range_pct=last.baseline_range_pct,
            burst_move_ratio=last.burst_move_ratio,
            burst_volume=last.burst_volume,
            baseline_volume=last.baseline_volume,
            burst_volume_ratio=last.burst_volume_ratio,
            up_burst=False,
            down_burst=False,
        )

    def _log_out_of_order_tick(self, ts_ms: int, price: float, side: TradeSide, size: float) -> None:
        now_monotonic = time.monotonic()
        if self._last_out_of_order_log_monotonic and now_monotonic - self._last_out_of_order_log_monotonic < 5:
            return
        logger.warning(
            "CVD_TICK_OUT_OF_ORDER | policy=%s last_ts_ms=%s current_ts_ms=%s price=%.4f side=%s size=%.8f",
            self._out_of_order_policy,
            self._last_tick_ts_ms,
            ts_ms,
            price,
            side,
            size,
        )
        self._last_out_of_order_log_monotonic = now_monotonic

    def _log_slow_update_if_needed(self, started_monotonic: float | None) -> None:
        if started_monotonic is None or self._slow_log_threshold_ms <= 0:
            return
        now_monotonic = time.monotonic()
        elapsed_ms = (now_monotonic - started_monotonic) * 1000
        if elapsed_ms < self._slow_log_threshold_ms:
            return
        if self._last_slow_log_monotonic and now_monotonic - self._last_slow_log_monotonic < 30:
            return
        logger.warning(
            "CVD_UPDATE_SLOW | elapsed_ms=%.3f events_fast=%s events_burst=%s events_baseline=%s range_samples=%s",
            elapsed_ms,
            len(self._fast_events),
            len(self._burst_events),
            len(self._baseline_events),
            len(self._baseline_range_samples),
        )
        self._last_slow_log_monotonic = now_monotonic

    @staticmethod
    def _load_out_of_order_policy() -> OutOfOrderPolicy:
        raw = os.getenv("CVD_OUT_OF_ORDER_POLICY", "drop_for_realtime").strip().lower()
        if raw == "accept_slow_debug":
            logger.warning("CVD_OUT_OF_ORDER_POLICY=accept_slow_debug is not supported in live O(1) update; using drop_for_realtime")
        return "drop_for_realtime"

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
