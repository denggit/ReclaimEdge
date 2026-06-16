"""Trend Breakout Metrics — episode-level statistics tracked from breakout anchor.

Maintains cumulative buy/sell volume, CVD extremes, expansion/occupancy
metrics, new-extreme count, and inside-reclaim detection for a single
trend breakout episode.  Pure logic — no exchange calls, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrendBreakoutMetrics:
    """Running episode statistics from a breakout anchor to the current tick."""

    # ── Expansion / occupancy booleans ────────────────────────────────
    range_expansion_passed: bool = False
    volume_expansion_passed: bool = False
    sustained_volume_passed: bool = False
    outside_occupancy_passed: bool = False

    # ── Extreme tracking ──────────────────────────────────────────────
    new_extreme_count: int = 0

    # ── Inside reclaim ────────────────────────────────────────────────
    inside_reclaim_seconds: float = 0.0
    price_reclaimed_inside: bool = False

    # ── Episode CVD / volume aggregates (cumulative from anchor) ──────
    episode_buy_volume: float = 0.0
    episode_sell_volume: float = 0.0
    episode_cvd_max: float = 0.0
    episode_cvd_min: float = 0.0

    # ── Internal anchor state ─────────────────────────────────────────
    _anchor_ts_ms: int = 0
    _anchor_cvd: float = 0.0
    _anchor_cumulative_buy_volume: float = 0.0
    _anchor_cumulative_sell_volume: float = 0.0
    _breakout_direction: str | None = None  # "UP" | "DOWN" | None
    _prev_price: float = 0.0
    _last_extreme_price: float = 0.0
    _price_outside: bool = False
    _outside_start_ts_ms: int = 0
    _total_outside_ms: int = 0
    _inside_start_ts_ms: int = 0
    _pre_breakout_range: float = 0.0
    _pre_breakout_volume: float = 0.0
    _episode_volume: float = 0.0


class TrendBreakoutMetricsTracker:
    """Tracks trend breakout episode metrics from anchor to current tick.

    The tracker is initialised with a breakout anchor (price, CVD, volumes)
    and then updated on each tick with fresh CVD and BOLL band data.

    All metric computation is self-contained — no external state reads.
    """

    def __init__(
        self,
        *,
        range_expansion_ratio_min: float = 3.0,
        volume_expansion_ratio_min: float = 3.0,
        outside_occupancy_min_ratio: float = 0.70,
        min_new_extreme_count: int = 2,
        max_inside_reclaim_seconds: int = 3,
        confirm_min_seconds: int = 60,
    ) -> None:
        self._range_expansion_ratio_min = range_expansion_ratio_min
        self._volume_expansion_ratio_min = volume_expansion_ratio_min
        self._outside_occupancy_min_ratio = outside_occupancy_min_ratio
        self._min_new_extreme_count = min_new_extreme_count
        self._max_inside_reclaim_seconds = max_inside_reclaim_seconds
        self._confirm_min_seconds = confirm_min_seconds

        self._m = TrendBreakoutMetrics()
        self._initialised: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def metrics(self) -> TrendBreakoutMetrics:
        return self._m

    @property
    def initialised(self) -> bool:
        return self._initialised

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def anchor(
        self,
        *,
        ts_ms: int,
        price: float,
        fast_cvd: float,
        cumulative_buy_volume: float,
        cumulative_sell_volume: float,
        direction: str,  # "UP" | "DOWN"
        boll_upper: float,
        boll_lower: float,
        pre_breakout_range: float = 0.0,
        pre_breakout_volume: float = 0.0,
    ) -> None:
        """Set (or reset) the breakout anchor.

        Call this when a new breakout direction is detected.
        """
        self._m = TrendBreakoutMetrics(
            _anchor_ts_ms=ts_ms,
            _anchor_cvd=fast_cvd,
            _anchor_cumulative_buy_volume=cumulative_buy_volume,
            _anchor_cumulative_sell_volume=cumulative_sell_volume,
            _breakout_direction=direction,
            _prev_price=price,
            _last_extreme_price=price,
            _price_outside=True,
            _outside_start_ts_ms=ts_ms,
            _total_outside_ms=0,
            _pre_breakout_range=pre_breakout_range,
            _pre_breakout_volume=pre_breakout_volume,
            episode_cvd_max=fast_cvd,
            episode_cvd_min=fast_cvd,
        )
        self._initialised = True

    def update(
        self,
        *,
        ts_ms: int,
        price: float,
        fast_cvd: float,
        cumulative_buy_volume: float,
        cumulative_sell_volume: float,
        boll_upper: float,
        boll_middle: float,
        boll_lower: float,
        band_range: float = 0.0,
        tick_volume: float = 0.0,
    ) -> TrendBreakoutMetrics:
        """Update metrics with one new tick and return the current snapshot.

        Returns:
            The updated :class:`TrendBreakoutMetrics` snapshot.
        """
        if not self._initialised:
            return self._m

        m = self._m
        direction = m._breakout_direction

        # ── Episode volume aggregates ──────────────────────────────────
        m.episode_buy_volume = max(0.0, cumulative_buy_volume - m._anchor_cumulative_buy_volume)
        m.episode_sell_volume = max(0.0, cumulative_sell_volume - m._anchor_cumulative_sell_volume)
        m.episode_cvd_max = max(m.episode_cvd_max, fast_cvd)
        m.episode_cvd_min = min(m.episode_cvd_min, fast_cvd)
        m._episode_volume += tick_volume

        # ── Outside / inside tracking ──────────────────────────────────
        currently_outside = self._is_price_outside(price, boll_upper, boll_lower)
        if currently_outside and not m._price_outside:
            # Price just went outside
            m._price_outside = True
            m._outside_start_ts_ms = ts_ms
            m.price_reclaimed_inside = False
        elif not currently_outside and m._price_outside:
            # Price just came back inside
            m._price_outside = False
            m._inside_start_ts_ms = ts_ms
            m._total_outside_ms += max(0, ts_ms - m._outside_start_ts_ms)
            m.price_reclaimed_inside = True
        elif not currently_outside and m.price_reclaimed_inside:
            # Still inside — update reclaim duration
            if m._inside_start_ts_ms > 0:
                m.inside_reclaim_seconds = (ts_ms - m._inside_start_ts_ms) / 1000.0

        # ── New extreme tracking ───────────────────────────────────────
        if direction == "UP" and price > m._last_extreme_price:
            m.new_extreme_count += 1
            m._last_extreme_price = price
        elif direction == "DOWN" and price < m._last_extreme_price:
            m.new_extreme_count += 1
            m._last_extreme_price = price

        # ── Range expansion ────────────────────────────────────────────
        elapsed_sec = max((ts_ms - m._anchor_ts_ms) / 1000.0, 0.001)
        if m._pre_breakout_range > 0 and band_range > 0:
            m.range_expansion_passed = (
                band_range / m._pre_breakout_range >= self._range_expansion_ratio_min
            )
        # else: no pre-breakout baseline → not passed (never auto-pass)

        # ── Volume expansion ───────────────────────────────────────────
        episode_volume_rate = m._episode_volume / elapsed_sec
        pre_breakout_volume_rate = (
            m._pre_breakout_volume / max(elapsed_sec, 1.0)
            if m._pre_breakout_volume > 0
            else 0.0
        )
        if pre_breakout_volume_rate > 0:
            m.volume_expansion_passed = (
                episode_volume_rate / pre_breakout_volume_rate >= self._volume_expansion_ratio_min
            )
        # else: no pre-breakout volume baseline → not passed (never auto-pass)

        # ── Sustained volume ───────────────────────────────────────────
        # Conservative rule: volume expansion must already be confirmed
        # AND the episode must have lasted at least confirm_min_seconds.
        m.sustained_volume_passed = (
            m.volume_expansion_passed
            and elapsed_sec >= self._confirm_min_seconds
        )

        # ── Outside occupancy ──────────────────────────────────────────
        total_elapsed = max(ts_ms - m._anchor_ts_ms, 1)
        total_outside = m._total_outside_ms
        if currently_outside:
            total_outside += max(0, ts_ms - m._outside_start_ts_ms)
        m.outside_occupancy_passed = (
            total_outside / total_elapsed >= self._outside_occupancy_min_ratio
        )

        m._prev_price = price
        return m

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_price_outside(price: float, boll_upper: float, boll_lower: float) -> bool:
        return price > boll_upper or price < boll_lower

    def snapshot(self) -> TrendBreakoutMetrics:
        """Return the current metrics snapshot without updating."""
        return self._m

    def reset(self) -> None:
        """Reset all tracking state."""
        self._m = TrendBreakoutMetrics()
        self._initialised = False
