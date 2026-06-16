"""Tests for Trend Breakout live blocking fixes.

Covers:
1. cvd.size → cvd.buy_volume + cvd.sell_volume
2. Range expansion uses cvd.burst_move_ratio
3. Volume expansion uses cvd.burst_volume_ratio
4. Baseline missing → expansion stays False
5. Sustained volume uses 10-second subwindows
6. Direction switch resets metrics tracker
7. Inside reclaim does NOT reset metrics tracker
8. UPDATE_TREND_SL "new first, old later" order
9. State pollution: intent generation does NOT modify strategy state
10. State updated ONLY on execution success
11. State unchanged on UPDATE_TREND_SL failure
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ======================================================================
# Shared helpers
# ======================================================================


@dataclass
class _FakeSize:
    margin_usdt: float = 10.0
    notional_usdt: float = 500.0
    eth_qty: float = 0.1
    layer_index: int = 1
    layer_multiplier: float = 1.0


# ======================================================================
# Helpers
# ======================================================================


def _cvd_snapshot(**overrides):
    """Build a real CvdSnapshot with sensible defaults."""
    from src.indicators.cvd_tracker import CvdSnapshot

    kwargs = dict(
        ts_ms=1000000,
        price=3000.0,
        side="buy",
        size=1.0,
        signed_delta=0.001,
        total_cvd=0.005,
        fast_cvd=0.002,
        previous_fast_cvd=0.001,
        buy_volume=60.0,
        sell_volume=40.0,
        buy_ratio=0.60,
        sell_ratio=0.40,
        cross_positive=False,
        cross_negative=False,
        cvd_increasing=True,
        cvd_decreasing=False,
        no_new_low=True,
        no_new_high=False,
        window_low=2900.0,
        window_high=3100.0,
        burst_net_move_pct=0.0,
        burst_range_pct=0.02,
        baseline_range_pct=0.005,
        burst_move_ratio=0.0,
        burst_volume=200.0,
        baseline_volume=100.0,
        burst_volume_ratio=2.0,
        up_burst=False,
        down_burst=False,
        cumulative_buy_volume=5000.0,
        cumulative_sell_volume=4500.0,
    )
    kwargs.update(overrides)
    return CvdSnapshot(**kwargs)


def _boll_snapshot(**overrides):
    from src.monitors.boll_band_breakout_monitor import BollSnapshot

    kwargs = dict(
        inst_id="ETH-USDT-SWAP",
        candle_ts_ms=1000000,
        close=3000.0,
        middle=3000.0,
        upper=3100.0,
        lower=2900.0,
        upper_distance_pct=100.0 / 3000.0,
        lower_distance_pct=100.0 / 3000.0,
        alert_switch_on=True,
        live_mode=True,
        tp_upper=3120.0,
        tp_middle=3020.0,
        tp_lower=2920.0,
        tp_window=15,
        high=3110.0,
        low=2890.0,
    )
    kwargs.update(overrides)
    return BollSnapshot(**kwargs)


def _make_config(**overrides):
    from src.strategies.boll_cvd_reclaim_strategy import BollCvdReclaimStrategyConfig

    kwargs = dict(
        trend_breakout_enabled=True,
        trend_middle_trailing_sl_enabled=True,
        trend_middle_sl_buffer_pct=0.001,
        trend_max_stop_distance_pct=0.02,
        trend_sl_update_interval_seconds=900,
        trend_confirm_min_seconds=60,
        entry_cvd_structure_mode="DIVERGENCE_OR_ABSORPTION",
    )
    kwargs.update(overrides)
    return BollCvdReclaimStrategyConfig(**kwargs)


def _make_sizer():
    from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig

    return SimplePositionSizer(SimplePositionSizerConfig(
        dry_run_equity_usdt=1000.0,
        trade_risk_pct=0.003,
        leverage=20.0,
    ))


def _make_strategy(config=None, **config_overrides):
    from src.strategies.boll_cvd_reclaim_strategy import BollCvdReclaimStrategy

    if config is None:
        config = _make_config(**config_overrides)
    return BollCvdReclaimStrategy(config, _make_sizer())


def _make_tracker(**kwargs):
    """Build TrendBreakoutMetricsTracker with defaults suitable for testing."""
    from src.strategies.trend_breakout_metrics import TrendBreakoutMetricsTracker

    defaults = dict(
        range_expansion_ratio_min=3.0,
        volume_expansion_ratio_min=3.0,
        outside_occupancy_min_ratio=0.70,
        min_new_extreme_count=2,
        max_inside_reclaim_seconds=3,
        confirm_min_seconds=60,
        volume_subwindow_seconds=10,
        volume_subwindow_pass_count=4,
        volume_subwindow_ratio_min=2.0,
        volume_persistence_window_seconds=60,
    )
    defaults.update(kwargs)
    return TrendBreakoutMetricsTracker(**defaults)


# ======================================================================
# 1. cvd.size removal tests
# ======================================================================


class TestNoCvdSizeInRouteRegime:
    """Verify cvd.size is never referenced in _route_regime."""

    def test_route_regime_source_has_no_cvd_dot_size(self):
        import inspect
        from src.strategies.boll_cvd_reclaim_strategy import BollCvdReclaimStrategy

        source = inspect.getsource(BollCvdReclaimStrategy._route_regime)
        assert "cvd.size" not in source, (
            "cvd.size must not appear in _route_regime — use cvd.buy_volume + cvd.sell_volume"
        )

    def test_tick_volume_uses_buy_plus_sell_volume(self):
        """tick_volume is computed from cvd.buy_volume + cvd.sell_volume."""
        import inspect
        from src.strategies.boll_cvd_reclaim_strategy import BollCvdReclaimStrategy

        source = inspect.getsource(BollCvdReclaimStrategy._route_regime)
        assert "cvd.buy_volume" in source, (
            "tick_volume must use cvd.buy_volume + cvd.sell_volume"
        )
        assert "cvd.sell_volume" in source, (
            "tick_volume must use cvd.buy_volume + cvd.sell_volume"
        )


class TestRouteRegimeDoesNotCrashOnRealCvdSnapshot:
    """_route_regime() must not crash with real CvdSnapshot instances."""

    def test_trend_enabled_route_regime_no_crash(self):
        """Trend enabled + real CvdSnapshot → no attribute error."""
        strategy = _make_strategy()
        boll = _boll_snapshot()
        cvd = _cvd_snapshot()

        # This must NOT raise AttributeError
        decision = strategy._route_regime(
            price=cvd.price,
            ts_ms=cvd.ts_ms,
            boll=boll,
            cvd=cvd,
            mr_long_allowed=False,
            mr_short_allowed=False,
        )
        # Even if decision is None (no breakout), the method must not crash
        assert decision is None or hasattr(decision, "decision_type")

    def test_trend_enabled_route_regime_handles_fresh_cvd(self):
        """Fresh CvdSnapshot with default values works without crash."""
        strategy = _make_strategy()
        boll = _boll_snapshot()
        cvd = _cvd_snapshot(
            buy_volume=0.0,
            sell_volume=0.0,
            fast_cvd=0.0,
            baseline_range_pct=0.0,
            baseline_volume=0.0,
        )

        # Must not crash even with zero volumes and zero baseline
        decision = strategy._route_regime(
            price=cvd.price,
            ts_ms=cvd.ts_ms,
            boll=boll,
            cvd=cvd,
            mr_long_allowed=False,
            mr_short_allowed=False,
        )
        assert decision is None or hasattr(decision, "decision_type")


# ======================================================================
# 2. Direction property on metrics tracker
# ======================================================================


class TestMetricsTrackerDirection:
    """TrendBreakoutMetricsTracker.direction returns current breakout direction."""

    def test_direction_returns_none_when_not_initialised(self):
        tracker = _make_tracker()
        assert tracker.direction is None

    def test_direction_returns_up_after_up_anchor(self):
        tracker = _make_tracker()
        tracker.anchor(
            ts_ms=1000000, price=3200.0, fast_cvd=0.01,
            cumulative_buy_volume=5000.0, cumulative_sell_volume=4000.0,
            direction="UP", boll_upper=3100.0, boll_lower=2900.0,
        )
        assert tracker.direction == "UP"

    def test_direction_returns_down_after_down_anchor(self):
        tracker = _make_tracker()
        tracker.anchor(
            ts_ms=1000000, price=2800.0, fast_cvd=0.01,
            cumulative_buy_volume=5000.0, cumulative_sell_volume=4000.0,
            direction="DOWN", boll_upper=3100.0, boll_lower=2900.0,
        )
        assert tracker.direction == "DOWN"

    def test_direction_resets_to_none_after_reset(self):
        tracker = _make_tracker()
        tracker.anchor(
            ts_ms=1000000, price=3200.0, fast_cvd=0.01,
            cumulative_buy_volume=5000.0, cumulative_sell_volume=4000.0,
            direction="UP", boll_upper=3100.0, boll_lower=2900.0,
        )
        tracker.reset()
        assert tracker.direction is None


# ======================================================================
# 3. Range expansion uses burst_move_ratio
# ======================================================================


class TestRangeExpansionUsesBurstMoveRatio:
    """Range expansion is now driven by cvd.burst_move_ratio, not band_range."""

    def test_range_expansion_passed_when_burst_move_ratio_sufficient(self):
        """burst_move_ratio >= 3.0 with baseline → range_expansion_passed=True."""
        tracker = _make_tracker(range_expansion_ratio_min=3.0)
        tracker.anchor(
            ts_ms=1000000, price=3200.0, fast_cvd=0.01,
            cumulative_buy_volume=5000.0, cumulative_sell_volume=4000.0,
            direction="UP", boll_upper=3100.0, boll_lower=2900.0,
        )

        m = tracker.update(
            ts_ms=1010000,
            price=3250.0,
            fast_cvd=0.02,
            cumulative_buy_volume=6000.0,
            cumulative_sell_volume=4500.0,
            boll_upper=3300.0,
            boll_middle=3100.0,
            boll_lower=2900.0,
            burst_move_ratio=4.0,  # ≥ 3.0
            baseline_range_pct=0.005,  # > 0
        )

        assert m.range_expansion_passed is True

    def test_range_expansion_not_passed_when_burst_move_ratio_insufficient(self):
        """burst_move_ratio < 3.0 → range_expansion_passed=False."""
        tracker = _make_tracker(range_expansion_ratio_min=3.0)
        tracker.anchor(
            ts_ms=1000000, price=3200.0, fast_cvd=0.01,
            cumulative_buy_volume=5000.0, cumulative_sell_volume=4000.0,
            direction="UP", boll_upper=3100.0, boll_lower=2900.0,
        )

        m = tracker.update(
            ts_ms=1010000,
            price=3150.0,
            fast_cvd=0.015,
            cumulative_buy_volume=5500.0,
            cumulative_sell_volume=4200.0,
            boll_upper=3300.0,
            boll_middle=3100.0,
            boll_lower=2900.0,
            burst_move_ratio=1.5,  # < 3.0
            baseline_range_pct=0.005,  # > 0
        )

        assert m.range_expansion_passed is False

    def test_range_expansion_false_when_baseline_range_missing(self):
        """baseline_range_pct=0 → range_expansion_passed=False regardless of burst."""
        tracker = _make_tracker(range_expansion_ratio_min=3.0)
        tracker.anchor(
            ts_ms=1000000, price=3200.0, fast_cvd=0.01,
            cumulative_buy_volume=5000.0, cumulative_sell_volume=4000.0,
            direction="UP", boll_upper=3100.0, boll_lower=2900.0,
        )

        m = tracker.update(
            ts_ms=1010000,
            price=3250.0,
            fast_cvd=0.02,
            cumulative_buy_volume=6000.0,
            cumulative_sell_volume=4500.0,
            boll_upper=3300.0,
            boll_middle=3100.0,
            boll_lower=2900.0,
            burst_move_ratio=5.0,  # well above threshold
            baseline_range_pct=0.0,  # no baseline → blocked
        )

        assert m.range_expansion_passed is False, (
            "range_expansion_passed must be False when baseline_range_pct=0"
        )

    def test_no_band_range_in_source(self):
        """Update method must NOT reference band_range in its source."""
        import inspect
        from src.strategies.trend_breakout_metrics import TrendBreakoutMetricsTracker

        source = inspect.getsource(TrendBreakoutMetricsTracker.update)
        # The old band_range / pre_breakout_range mixing must be removed
        assert "band_range / m._pre_breakout_range" not in source, (
            "band_range / pre_breakout_range mixing must be removed"
        )


# ======================================================================
# 4. Volume expansion uses burst_volume_ratio
# ======================================================================


class TestVolumeExpansionUsesBurstVolumeRatio:
    """Volume expansion is now driven by cvd.burst_volume_ratio."""

    def test_volume_expansion_passed_when_burst_volume_ratio_sufficient(self):
        """burst_volume_ratio >= 3.0 with baseline → volume_expansion_passed=True."""
        tracker = _make_tracker(volume_expansion_ratio_min=3.0)
        tracker.anchor(
            ts_ms=1000000, price=3200.0, fast_cvd=0.01,
            cumulative_buy_volume=5000.0, cumulative_sell_volume=4000.0,
            direction="UP", boll_upper=3100.0, boll_lower=2900.0,
        )

        m = tracker.update(
            ts_ms=1010000,
            price=3250.0,
            fast_cvd=0.02,
            cumulative_buy_volume=6000.0,
            cumulative_sell_volume=4500.0,
            boll_upper=3300.0,
            boll_middle=3100.0,
            boll_lower=2900.0,
            burst_volume_ratio=4.0,  # ≥ 3.0
            baseline_volume=100.0,  # > 0
        )

        assert m.volume_expansion_passed is True

    def test_volume_expansion_not_passed_when_burst_volume_ratio_insufficient(self):
        """burst_volume_ratio < 3.0 → volume_expansion_passed=False."""
        tracker = _make_tracker(volume_expansion_ratio_min=3.0)
        tracker.anchor(
            ts_ms=1000000, price=3200.0, fast_cvd=0.01,
            cumulative_buy_volume=5000.0, cumulative_sell_volume=4000.0,
            direction="UP", boll_upper=3100.0, boll_lower=2900.0,
        )

        m = tracker.update(
            ts_ms=1010000,
            price=3150.0,
            fast_cvd=0.015,
            cumulative_buy_volume=5500.0,
            cumulative_sell_volume=4200.0,
            boll_upper=3300.0,
            boll_middle=3100.0,
            boll_lower=2900.0,
            burst_volume_ratio=1.0,  # < 3.0
            baseline_volume=100.0,  # > 0
        )

        assert m.volume_expansion_passed is False

    def test_volume_expansion_false_when_baseline_volume_missing(self):
        """baseline_volume=0 → volume_expansion_passed=False regardless of burst."""
        tracker = _make_tracker(volume_expansion_ratio_min=3.0)
        tracker.anchor(
            ts_ms=1000000, price=3200.0, fast_cvd=0.01,
            cumulative_buy_volume=5000.0, cumulative_sell_volume=4000.0,
            direction="UP", boll_upper=3100.0, boll_lower=2900.0,
        )

        m = tracker.update(
            ts_ms=1010000,
            price=3250.0,
            fast_cvd=0.02,
            cumulative_buy_volume=6000.0,
            cumulative_sell_volume=4500.0,
            boll_upper=3300.0,
            boll_middle=3100.0,
            boll_lower=2900.0,
            burst_volume_ratio=5.0,  # well above threshold
            baseline_volume=0.0,  # no baseline → blocked
        )

        assert m.volume_expansion_passed is False, (
            "volume_expansion_passed must be False when baseline_volume=0"
        )


# ======================================================================
# 5. Sustained volume — 10-second subwindow logic
# ======================================================================


class TestSustainedVolumeSubwindows:
    """Sustained volume requires multiple 10-second subwindows passing threshold."""

    def test_sustained_volume_false_when_baseline_volume_rate_zero(self):
        """baseline_volume_rate=0 → sustained_volume_passed=False."""
        tracker = _make_tracker()
        tracker.anchor(
            ts_ms=1000000, price=3200.0, fast_cvd=0.01,
            cumulative_buy_volume=5000.0, cumulative_sell_volume=4000.0,
            direction="UP", boll_upper=3100.0, boll_lower=2900.0,
        )

        m = tracker.update(
            ts_ms=1010000, price=3250.0, fast_cvd=0.02,
            cumulative_buy_volume=6000.0, cumulative_sell_volume=4500.0,
            boll_upper=3300.0, boll_middle=3100.0, boll_lower=2900.0,
            baseline_volume_rate=0.0,  # zero rate → blocked
            tick_volume=25.0,
        )

        assert m.sustained_volume_passed is False

    def test_sustained_volume_passes_with_four_passing_windows(self):
        """4 subwindows with ratio ≥ 2.0 within 60s → sustained_volume_passed=True."""
        tracker = _make_tracker(
            volume_subwindow_seconds=10,
            volume_subwindow_pass_count=4,
            volume_subwindow_ratio_min=2.0,
            volume_persistence_window_seconds=60,
        )
        tracker.anchor(
            ts_ms=1000000, price=3200.0, fast_cvd=0.01,
            cumulative_buy_volume=5000.0, cumulative_sell_volume=4000.0,
            direction="UP", boll_upper=3100.0, boll_lower=2900.0,
        )

        # baseline_volume_rate=1.0 → baseline_window_volume = 1.0 * 10 = 10.0
        # tick_volume=25.0 → window_volume_ratio = 25.0 / 10.0 = 2.5 ≥ 2.0
        # Each update is 10s apart → creates separate 10s windows
        base_ts = 1000000
        for i in range(6):
            ts = base_ts + i * 10000  # each 10s later
            m = tracker.update(
                ts_ms=ts, price=3200.0 + i * 10, fast_cvd=0.02,
                cumulative_buy_volume=6000.0 + i * 100,
                cumulative_sell_volume=4500.0 + i * 50,
                boll_upper=3300.0, boll_middle=3100.0, boll_lower=2900.0,
                baseline_volume_rate=1.0,
                tick_volume=25.0,
            )

        # After 6 windows, 6 should pass → sustained_volume_passed=True
        assert m.sustained_volume_passed is True, (
            f"Expected sustained_volume_passed=True after 6 passing windows, "
            f"got {m.sustained_volume_passed}"
        )

    def test_sustained_volume_fails_with_insufficient_passing_windows(self):
        """Only 3 windows with ratio ≥ 2.0 within 60s → sustained_volume_passed=False."""
        tracker = _make_tracker(
            volume_subwindow_seconds=10,
            volume_subwindow_pass_count=4,
            volume_subwindow_ratio_min=2.0,
            volume_persistence_window_seconds=60,
        )
        tracker.anchor(
            ts_ms=1000000, price=3200.0, fast_cvd=0.01,
            cumulative_buy_volume=5000.0, cumulative_sell_volume=4000.0,
            direction="UP", boll_upper=3100.0, boll_lower=2900.0,
        )

        # baseline_volume_rate=1.0 → baseline_window_volume = 10.0
        # tick_volume=5.0 → window_volume_ratio = 5.0 / 10.0 = 0.5 < 2.0
        # None of these windows pass
        base_ts = 1000000
        for i in range(6):
            ts = base_ts + i * 10000
            m = tracker.update(
                ts_ms=ts, price=3200.0 + i * 2, fast_cvd=0.02,
                cumulative_buy_volume=6000.0 + i * 10,
                cumulative_sell_volume=4500.0 + i * 5,
                boll_upper=3300.0, boll_middle=3100.0, boll_lower=2900.0,
                baseline_volume_rate=1.0,
                tick_volume=5.0,  # low volume → ratio 0.5 < 2.0
            )

        assert m.sustained_volume_passed is False, (
            "sustained_volume_passed must be False when < 4 windows pass threshold"
        )

    def test_sustained_volume_respects_persistence_window(self):
        """Windows outside persistence_window_seconds are pruned."""
        tracker = _make_tracker(
            volume_subwindow_seconds=10,
            volume_subwindow_pass_count=4,
            volume_subwindow_ratio_min=2.0,
            volume_persistence_window_seconds=60,
        )
        tracker.anchor(
            ts_ms=1000000, price=3200.0, fast_cvd=0.01,
            cumulative_buy_volume=5000.0, cumulative_sell_volume=4000.0,
            direction="UP", boll_upper=3100.0, boll_lower=2900.0,
        )

        # First 3 windows: high volume (passing)
        # Then skip past 60s, then 3 more windows: low volume (not passing)
        # Old windows should be pruned → only 3 recent non-passing → False
        base_ts = 1000000
        # Windows at t=0s, 10s, 20s with high volume
        for i in range(3):
            ts = base_ts + i * 10000
            tracker.update(
                ts_ms=ts, price=3200.0 + i * 10, fast_cvd=0.02,
                cumulative_buy_volume=6000.0 + i * 100,
                cumulative_sell_volume=4500.0 + i * 50,
                boll_upper=3300.0, boll_middle=3100.0, boll_lower=2900.0,
                baseline_volume_rate=1.0,
                tick_volume=25.0,  # high volume → ratio 2.5 ≥ 2.0
            )

        # Jump forward 70s — old windows expire
        ts_jump = base_ts + 70000  # +70s
        for i in range(3):
            ts = ts_jump + i * 10000
            m = tracker.update(
                ts_ms=ts, price=3200.0 + i * 2, fast_cvd=0.02,
                cumulative_buy_volume=6000.0 + i * 10,
                cumulative_sell_volume=4500.0 + i * 5,
                boll_upper=3300.0, boll_middle=3100.0, boll_lower=2900.0,
                baseline_volume_rate=1.0,
                tick_volume=5.0,  # low volume
            )

        # Only 3 recent low-volume windows → sustained_volume_passed=False
        assert m.sustained_volume_passed is False, (
            "sustained_volume_passed must be False after old windows expire"
        )

    def test_volume_windows_cleared_on_reset(self):
        """reset() must clear accumulated volume windows."""
        tracker = _make_tracker(volume_subwindow_pass_count=1)
        tracker.anchor(
            ts_ms=1000000, price=3200.0, fast_cvd=0.01,
            cumulative_buy_volume=5000.0, cumulative_sell_volume=4000.0,
            direction="UP", boll_upper=3100.0, boll_lower=2900.0,
        )

        tracker.update(
            ts_ms=1010000, price=3250.0, fast_cvd=0.02,
            cumulative_buy_volume=6000.0, cumulative_sell_volume=4500.0,
            boll_upper=3300.0, boll_middle=3100.0, boll_lower=2900.0,
            baseline_volume_rate=1.0, tick_volume=25.0,
        )

        tracker.reset()
        assert tracker._volume_windows == [], (
            "Volume windows must be cleared on reset"
        )


# ======================================================================
# 6. Direction switch resets metrics tracker
# ======================================================================


class TestDirectionSwitchResetsMetrics:
    """When breakout direction switches, old metrics must be reset."""

    def test_up_to_down_switch_resets_and_reanchors(self):
        """UP episode → DOWN breakout: reset old UP metrics, anchor new DOWN."""
        tracker = _make_tracker()
        tracker.anchor(
            ts_ms=1000000, price=3200.0, fast_cvd=0.01,
            cumulative_buy_volume=5000.0, cumulative_sell_volume=4000.0,
            direction="UP", boll_upper=3100.0, boll_lower=2900.0,
        )
        assert tracker.direction == "UP"
        assert tracker.initialised

        # Simulate direction switch by reset + re-anchor
        tracker.reset()
        assert not tracker.initialised
        assert tracker.direction is None

        tracker.anchor(
            ts_ms=2000000, price=2800.0, fast_cvd=-0.01,
            cumulative_buy_volume=7000.0, cumulative_sell_volume=8000.0,
            direction="DOWN", boll_upper=3100.0, boll_lower=2900.0,
        )
        assert tracker.direction == "DOWN"

    def test_old_up_episode_volume_not_inherited_after_switch(self):
        """After UP→DOWN switch, old UP episode buy_volume is not inherited."""
        tracker = _make_tracker()
        tracker.anchor(
            ts_ms=1000000, price=3200.0, fast_cvd=0.01,
            cumulative_buy_volume=5000.0, cumulative_sell_volume=4000.0,
            direction="UP", boll_upper=3100.0, boll_lower=2900.0,
        )

        # Build up some UP episode volume
        tracker.update(
            ts_ms=1010000, price=3250.0, fast_cvd=0.02,
            cumulative_buy_volume=6000.0, cumulative_sell_volume=4500.0,
            boll_upper=3300.0, boll_middle=3100.0, boll_lower=2900.0,
            tick_volume=100.0,
        )
        assert tracker.snapshot().episode_buy_volume > 0

        # Reset and re-anchor for DOWN
        tracker.reset()
        tracker.anchor(
            ts_ms=2000000, price=2800.0, fast_cvd=-0.01,
            cumulative_buy_volume=7000.0, cumulative_sell_volume=8000.0,
            direction="DOWN", boll_upper=3100.0, boll_lower=2900.0,
        )

        m = tracker.snapshot()
        # Episode volumes should start fresh from new anchor
        assert m.episode_buy_volume == 0.0, (
            "Episode buy volume must be reset after direction switch"
        )
        assert m.episode_sell_volume == 0.0, (
            "Episode sell volume must be reset after direction switch"
        )
        # Extremes should be reset to anchor CVD
        assert m.episode_cvd_max == -0.01
        assert m.episode_cvd_min == -0.01

    def test_direction_switch_clears_new_extreme_count(self):
        """After direction switch, new_extreme_count starts from 0."""
        tracker = _make_tracker()
        tracker.anchor(
            ts_ms=1000000, price=3200.0, fast_cvd=0.01,
            cumulative_buy_volume=5000.0, cumulative_sell_volume=4000.0,
            direction="UP", boll_upper=3100.0, boll_lower=2900.0,
        )

        # Build up extremes
        tracker.update(
            ts_ms=1010000, price=3300.0, fast_cvd=0.03,
            cumulative_buy_volume=6000.0, cumulative_sell_volume=4500.0,
            boll_upper=3300.0, boll_middle=3100.0, boll_lower=2900.0,
        )
        tracker.update(
            ts_ms=1020000, price=3400.0, fast_cvd=0.04,
            cumulative_buy_volume=6500.0, cumulative_sell_volume=4600.0,
            boll_upper=3300.0, boll_middle=3100.0, boll_lower=2900.0,
        )
        assert tracker.snapshot().new_extreme_count > 0

        # Reset and re-anchor for DOWN
        tracker.reset()
        tracker.anchor(
            ts_ms=2000000, price=2800.0, fast_cvd=-0.01,
            cumulative_buy_volume=7000.0, cumulative_sell_volume=8000.0,
            direction="DOWN", boll_upper=3100.0, boll_lower=2900.0,
        )
        assert tracker.snapshot().new_extreme_count == 0, (
            "new_extreme_count must be 0 after direction switch"
        )

    def test_route_regime_direction_switch_logs_and_resets(self):
        """_route_regime must detect and reset on direction switch."""
        strategy = _make_strategy()
        boll = _boll_snapshot(upper=3100.0, lower=2900.0, middle=3000.0)

        # First tick: UP breakout
        cvd_up = _cvd_snapshot(price=3200.0, ts_ms=1000000)
        strategy._route_regime(
            price=3200.0, ts_ms=1000000, boll=boll, cvd=cvd_up,
            mr_long_allowed=False, mr_short_allowed=False,
        )
        # Tracker should be initialised with UP
        tracker = strategy._get_trend_metrics_tracker()
        assert tracker is not None
        assert tracker.direction == "UP"

        # Second tick: DOWN breakout (direction switch)
        cvd_down = _cvd_snapshot(price=2800.0, ts_ms=2000000)
        strategy._route_regime(
            price=2800.0, ts_ms=2000000, boll=boll, cvd=cvd_down,
            mr_long_allowed=False, mr_short_allowed=False,
        )
        # Tracker should have been reset and re-anchored to DOWN
        assert tracker.direction == "DOWN", (
            f"Expected direction=DOWN after switch, got {tracker.direction}"
        )


# ======================================================================
# 7. Inside reclaim does NOT reset metrics tracker
# ======================================================================


class TestInsideReclaimPreservesMetrics:
    """When price returns inside band, metrics tracker is NOT reset."""

    def test_inside_reclaim_does_not_reset_tracker(self):
        """UP breakout metrics initialised, price back inside → not reset."""
        tracker = _make_tracker()
        tracker.anchor(
            ts_ms=1000000, price=3200.0, fast_cvd=0.01,
            cumulative_buy_volume=5000.0, cumulative_sell_volume=4000.0,
            direction="UP", boll_upper=3100.0, boll_lower=2900.0,
        )

        # Price goes back inside band
        m = tracker.update(
            ts_ms=1010000,
            price=3050.0,  # inside band (2900 < 3050 < 3100)
            fast_cvd=0.015,
            cumulative_buy_volume=5500.0,
            cumulative_sell_volume=4200.0,
            boll_upper=3100.0,
            boll_middle=3000.0,
            boll_lower=2900.0,
        )

        assert tracker.initialised, "Tracker must remain initialised after inside reclaim"
        assert tracker.direction == "UP", "Direction must remain UP after inside reclaim"
        assert m.price_reclaimed_inside is True, (
            "price_reclaimed_inside should be True when price returns inside"
        )

    def test_inside_reclaim_seconds_updates(self):
        """After reclaim, inside_reclaim_seconds accumulates while inside."""
        tracker = _make_tracker()
        tracker.anchor(
            ts_ms=1000000, price=3200.0, fast_cvd=0.01,
            cumulative_buy_volume=5000.0, cumulative_sell_volume=4000.0,
            direction="UP", boll_upper=3100.0, boll_lower=2900.0,
        )

        # Price returns inside at t=1010s
        tracker.update(
            ts_ms=1010000, price=3050.0, fast_cvd=0.015,
            cumulative_buy_volume=5500.0, cumulative_sell_volume=4200.0,
            boll_upper=3100.0, boll_middle=3000.0, boll_lower=2900.0,
        )

        # Still inside at t=1050s → inside_reclaim_seconds should grow
        m = tracker.update(
            ts_ms=1050000, price=3060.0, fast_cvd=0.016,
            cumulative_buy_volume=5600.0, cumulative_sell_volume=4300.0,
            boll_upper=3100.0, boll_middle=3000.0, boll_lower=2900.0,
        )

        assert m.inside_reclaim_seconds > 0, (
            f"inside_reclaim_seconds should be > 0, got {m.inside_reclaim_seconds}"
        )

    def test_route_regime_preserves_metrics_on_inside_reclaim(self):
        """_route_regime must NOT reset tracker when price goes inside band."""
        strategy = _make_strategy()
        boll = _boll_snapshot(upper=3100.0, lower=2900.0, middle=3000.0)

        # First: UP breakout
        cvd_up = _cvd_snapshot(price=3200.0, ts_ms=1000000)
        strategy._route_regime(
            price=3200.0, ts_ms=1000000, boll=boll, cvd=cvd_up,
            mr_long_allowed=False, mr_short_allowed=False,
        )
        tracker = strategy._get_trend_metrics_tracker()
        assert tracker is not None
        assert tracker.direction == "UP"

        # Then: price returns inside band
        cvd_inside = _cvd_snapshot(price=3050.0, ts_ms=2000000)
        strategy._route_regime(
            price=3050.0, ts_ms=2000000, boll=boll, cvd=cvd_inside,
            mr_long_allowed=False, mr_short_allowed=False,
        )

        # Tracker must still be initialised and direction unchanged
        assert tracker.initialised, "Tracker must still be initialised after inside reclaim"
        assert tracker.direction == "UP", (
            f"Direction must remain UP after inside reclaim, got {tracker.direction}"
        )
        m = tracker.snapshot()
        assert m.price_reclaimed_inside is True, (
            "price_reclaimed_inside should be True"
        )


# ======================================================================
# 8. No hardcoded unconditional passes
# ======================================================================


class TestNoUnconditionalPass:
    """Verify no metrics are unconditionally set to True."""

    def test_no_unconditional_true_in_metrics(self):
        """Source-code scan: updates must not set any metric to True unconditionally."""
        import inspect
        from src.strategies.trend_breakout_metrics import TrendBreakoutMetricsTracker

        source = inspect.getsource(TrendBreakoutMetricsTracker.update)
        # sustained_volume_passed must not be set to True unconditionally
        assert "simplified" not in source, (
            "No unconditional 'simplified: real check needs sub-windows' True pass allowed"
        )
        # range expansion must not auto-pass on timeout alone
        assert "m.range_expansion_passed = True" not in source, (
            "range_expansion_passed must not be set unconditionally True"
        )
        # Must not contain old band_range division logic
        assert "band_range /" not in source, (
            "Old band_range division logic must be removed"
        )
        # Must use burst_move_ratio for range expansion
        assert "burst_move_ratio" in source, (
            "update() must reference burst_move_ratio for range expansion"
        )
        # Must use burst_volume_ratio for volume expansion
        assert "burst_volume_ratio" in source, (
            "update() must reference burst_volume_ratio for volume expansion"
        )


# ======================================================================
# 9. UPDATE_TREND_SL "new first, old later" order
# ======================================================================


class TestUpdateTrendSLNewFirstOldLater:
    """UPDATE_TREND_SL places new SL FIRST, only cancels old after success."""

    def _make_fake_trading_client(self):
        """Build a FakeTradingClient that returns controlled order IDs."""

        class FakeTC:
            def __init__(self):
                self.market_calls = []
                self.next_order_id = "entry-1"

            async def place_market_order(self, *, side, qty, reduce_only, client_order_id):
                self.market_calls.append({
                    "side": side, "qty": qty, "reduce_only": reduce_only,
                    "client_order_id": client_order_id,
                })
                from src.execution.trading_client_port import OrderResult
                return OrderResult(ok=True, order_id=self.next_order_id, client_order_id=None, raw={})

            async def fetch_balance(self):
                class FakeBalance:
                    total = 500.0
                return FakeBalance()

            async def configure_instrument(self):
                pass

            async def fetch_position(self):
                from src.execution.trading_client_port import PositionResult
                return PositionResult(
                    has_position=True,
                    side="LONG",
                    qty=Decimal("1"),
                    avg_entry_price=3000.0,
                    raw={"raw_pos": "1"},
                )

            async def place_algo_order(self, *, side, qty, stop_price, reduce_only):
                from src.execution.trading_client_port import AlgoOrderResult
                return AlgoOrderResult(ok=True, algo_id="new-sl-1", raw={})

            async def cancel_algo_order(self, *, algo_id):
                from src.execution.trading_client_port import CancelAlgoResult
                return CancelAlgoResult(ok=True, raw={}, message="cancelled")

            async def fetch_open_orders(self):
                return []

            async def fetch_open_algo_orders(self):
                return []

        return FakeTC()

    def _make_trend_sl_trader(self, trading_client):
        """Build a minimal Trader for testing _execute_update_trend_sl."""
        from src.execution.trader import Trader

        trader = object.__new__(Trader)
        trader.trading_client = trading_client
        trader.position_contracts = Decimal("1")
        trader.symbol = "ETH-USDT-SWAP"
        trader.td_mode = "isolated"
        trader.pos_side_mode = "net"
        trader.contract_multiplier = Decimal("0.1")
        trader.contract_precision = Decimal("0.01")
        trader.min_contracts = Decimal("0.01")
        trader.entry_protective_sl_order_id = None

        from src.execution.trader import PositionSnapshot
        trader.fetch_position_snapshot = AsyncMock(
            return_value=PositionSnapshot(
                side="LONG",
                contracts=Decimal("1"),
                avg_entry_price=3000.0,
                eth_qty=0.1,
                raw_pos=Decimal("1"),
            )
        )

        # Bind tp_sl_manager so cancel_protective_stop works
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager
        trader._tp_sl_manager = TpSlExecutionManager(trader, trading_client=trading_client)

        # Monkey-patch place_entry_protective_stop_with_retries on trader
        trader.place_entry_protective_stop_with_retries = AsyncMock(
            return_value=(True, "entry-sl-1", "protective_sl_placed")
        )

        return trader

    def _make_update_trend_sl_intent(self, sl_price=2990.0, side="LONG"):
        from src.strategies.boll_cvd_reclaim_strategy import TradeIntent
        return TradeIntent(
            intent_type="UPDATE_TREND_SL",
            side=side,
            price=3050.0,
            layer_index=1,
            tp_price=3500.0,
            reason="trend_trailing_sl_tightened",
            size=_FakeSize(eth_qty=0.1),
            fast_cvd=0.01,
            previous_fast_cvd=0.005,
            buy_ratio=0.6,
            sell_ratio=0.4,
            boll_upper=3500.0,
            boll_middle=3000.0,
            boll_lower=2500.0,
            ts_ms=2000000,
            avg_entry_price=3200.0,
            breakeven_price=3205.0,
            tp_mode="UPPER",
            entry_protective_sl_price=sl_price,
        )

    @pytest.mark.asyncio
    async def test_old_sl_not_cancelled_before_new_placed(self):
        """Old SL cancel must NOT be called before new SL placement."""
        fake = self._make_fake_trading_client()
        trader = self._make_trend_sl_trader(fake)
        trader.entry_protective_sl_order_id = "old-sl-1"

        # Track call order
        call_order = []

        async def track_place(*args, **kwargs):
            call_order.append("place_new")
            return (True, "new-sl-1", "ok")

        async def track_cancel(*args, **kwargs):
            call_order.append("cancel_old")
            return True

        # Replace the instance-level mocks with tracking versions
        trader.place_entry_protective_stop_with_retries = AsyncMock(side_effect=track_place)

        from unittest.mock import patch
        with patch.object(
            trader._tp_sl_manager, "cancel_protective_stop",
            side_effect=track_cancel,
        ):
            intent = self._make_update_trend_sl_intent()
            result = await trader._execute_update_trend_sl(intent)

        # place_new MUST come before cancel_old
        if "cancel_old" in call_order:
            place_idx = call_order.index("place_new")
            cancel_idx = call_order.index("cancel_old")
            assert place_idx < cancel_idx, (
                f"New SL must be placed before old SL cancelled. "
                f"Got: place_new at {place_idx}, cancel_old at {cancel_idx}"
            )

    @pytest.mark.asyncio
    async def test_new_sl_failure_does_not_cancel_old(self):
        """When new SL placement fails, old SL must NOT be cancelled."""
        fake = self._make_fake_trading_client()
        trader = self._make_trend_sl_trader(fake)
        trader.entry_protective_sl_order_id = "old-sl-1"

        cancel_called_with = []

        from unittest.mock import patch

        async def fail_place(*args, **kwargs):
            return (False, None, "place_failed")

        async def track_cancel(algo_id):
            cancel_called_with.append(algo_id)
            return True

        trader.place_entry_protective_stop_with_retries = AsyncMock(side_effect=fail_place)

        with patch.object(
            trader._tp_sl_manager, "cancel_protective_stop",
            side_effect=track_cancel,
        ):
            intent = self._make_update_trend_sl_intent()
            result = await trader._execute_update_trend_sl(intent)

        # Result must be failure
        assert result.ok is False
        # Old SL must NOT have been cancelled
        assert len(cancel_called_with) == 0, (
            "cancel_protective_stop must NOT be called when new SL fails"
        )
        # Old SL ID must still be tracked
        assert trader.entry_protective_sl_order_id == "old-sl-1", (
            "entry_protective_sl_order_id must remain old-sl-1 after failure"
        )

    @pytest.mark.asyncio
    async def test_new_sl_success_then_cancels_old(self):
        """When new SL succeeds, old SL is cancelled AFTER new succeeds."""
        fake = self._make_fake_trading_client()
        trader = self._make_trend_sl_trader(fake)
        trader.entry_protective_sl_order_id = "old-sl-1"

        cancel_called_with = []

        from unittest.mock import patch

        async def success_place(*args, **kwargs):
            return (True, "new-sl-1", "ok")

        async def track_cancel(algo_id):
            cancel_called_with.append(algo_id)
            return True

        trader.place_entry_protective_stop_with_retries = AsyncMock(side_effect=success_place)

        with patch.object(
            trader._tp_sl_manager, "cancel_protective_stop",
            side_effect=track_cancel,
        ):
            intent = self._make_update_trend_sl_intent()
            result = await trader._execute_update_trend_sl(intent)

        assert result.ok is True
        assert result.protective_sl_order_id == "new-sl-1"
        # Old SL should have been cancelled
        assert "old-sl-1" in cancel_called_with, (
            "Old SL must be cancelled after new SL succeeds"
        )
        # New SL ID now tracked
        assert trader.entry_protective_sl_order_id == "new-sl-1"

    @pytest.mark.asyncio
    async def test_no_old_sl_no_cancel(self):
        """When no old SL exists, no cancel is attempted."""
        fake = self._make_fake_trading_client()
        trader = self._make_trend_sl_trader(fake)
        trader.entry_protective_sl_order_id = None

        cancel_called = False

        from unittest.mock import patch

        async def success_place(*args, **kwargs):
            return (True, "new-sl-1", "ok")

        async def track_cancel(*args, **kwargs):
            nonlocal cancel_called
            cancel_called = True
            return True

        trader.place_entry_protective_stop_with_retries = AsyncMock(side_effect=success_place)

        with patch.object(
            trader._tp_sl_manager, "cancel_protective_stop",
            side_effect=track_cancel,
        ):
            intent = self._make_update_trend_sl_intent()
            result = await trader._execute_update_trend_sl(intent)

        assert result.ok is True
        assert not cancel_called, (
            "cancel_protective_stop must NOT be called when no old SL exists"
        )

    @pytest.mark.asyncio
    async def test_same_sl_id_no_cancel(self):
        """When new SL ID matches old SL ID, no cancel is needed."""
        fake = self._make_fake_trading_client()
        trader = self._make_trend_sl_trader(fake)
        trader.entry_protective_sl_order_id = "same-sl-1"

        cancel_called = False

        from unittest.mock import patch

        async def success_place(*args, **kwargs):
            return (True, "same-sl-1", "ok")

        async def track_cancel(*args, **kwargs):
            nonlocal cancel_called
            cancel_called = True
            return True

        trader.place_entry_protective_stop_with_retries = AsyncMock(side_effect=success_place)

        with patch.object(
            trader._tp_sl_manager, "cancel_protective_stop",
            side_effect=track_cancel,
        ):
            intent = self._make_update_trend_sl_intent()
            result = await trader._execute_update_trend_sl(intent)

        assert result.ok is True
        assert not cancel_called, (
            "cancel_protective_stop must NOT be called when old SL ID == new SL ID"
        )


# ======================================================================
# 10. State pollution: intent generation does NOT modify strategy state
# ======================================================================


class TestTrendSLIntentDoesNotPolluteState:
    """Intent generation for UPDATE_TREND_SL must NOT write to strategy state."""

    def test_intent_generation_does_not_update_trend_trailing_sl_price(self):
        """After emitting UPDATE_TREND_SL intent, state.trend_trailing_sl_price
        must remain the OLD value (state is not polluted)."""
        strategy = _make_strategy()
        strategy.state.side = "LONG"
        strategy.state.layers = 1
        strategy.state.entry_regime = "TREND_BREAKOUT"
        strategy.state.trend_trailing_sl_price = 2900.0  # OLD SL
        strategy.state.trend_last_sl_update_ts_ms = 0
        strategy.state.avg_entry_price = 3050.0
        strategy.state.tp_price = 3500.0
        strategy.state.last_tp_update_candle_ts_ms = 0
        strategy.state.entry_protective_sl_price = 2900.0

        old_sl = strategy.state.trend_trailing_sl_price
        old_entry_sl = strategy.state.entry_protective_sl_price
        old_last_sl_update = strategy.state.trend_last_sl_update_ts_ms

        boll = _boll_snapshot(middle=3000.0, candle_ts_ms=2000000)
        cvd = _cvd_snapshot()

        intents = strategy.on_tick(
            price=3100.0, ts_ms=2000000, boll=boll, cvd=cvd,
        )

        # Even if an UPDATE_TREND_SL intent was emitted, state must be unchanged
        assert strategy.state.trend_trailing_sl_price == old_sl, (
            "trend_trailing_sl_price must NOT change during intent generation"
        )
        assert strategy.state.entry_protective_sl_price == old_entry_sl, (
            "entry_protective_sl_price must NOT change during intent generation"
        )
        assert strategy.state.trend_last_sl_update_ts_ms == old_last_sl_update, (
            "trend_last_sl_update_ts_ms must NOT change during intent generation"
        )

    def test_intent_carries_new_sl_price(self):
        """The intent itself carries the new SL price via entry_protective_sl_price."""
        strategy = _make_strategy()
        strategy.state.side = "LONG"
        strategy.state.layers = 1
        strategy.state.entry_regime = "TREND_BREAKOUT"
        strategy.state.trend_trailing_sl_price = 2900.0
        strategy.state.trend_last_sl_update_ts_ms = 0
        strategy.state.avg_entry_price = 3050.0
        strategy.state.tp_price = 3500.0
        strategy.state.last_tp_update_candle_ts_ms = 0

        boll = _boll_snapshot(middle=3000.0, candle_ts_ms=2000000)
        cvd = _cvd_snapshot()

        intents = strategy.on_tick(
            price=3100.0, ts_ms=2000000, boll=boll, cvd=cvd,
        )

        trend_sl_intents = [i for i in intents if i.intent_type == "UPDATE_TREND_SL"]
        if trend_sl_intents:
            intent = trend_sl_intents[0]
            # The intent must carry the new SL price
            assert intent.entry_protective_sl_price is not None
            # The new SL should be ~ 3000 * (1 - 0.001) = 2997.0
            assert intent.entry_protective_sl_price > 2900.0, (
                f"Intent should carry tightened SL (new_sl > old_sl 2900.0), "
                f"got {intent.entry_protective_sl_price}"
            )
            assert intent.entry_protective_sl_price <= 3000.0, (
                f"SL should be <= middle 3000.0, got {intent.entry_protective_sl_price}"
            )

    def test_failed_execution_does_not_update_state(self):
        """Even if simulated intent fails, strategy state remains old values."""
        strategy = _make_strategy()
        strategy.state.side = "LONG"
        strategy.state.layers = 1
        strategy.state.entry_regime = "TREND_BREAKOUT"
        strategy.state.trend_trailing_sl_price = 2900.0
        strategy.state.trend_last_sl_update_ts_ms = 0
        strategy.state.entry_protective_sl_price = 2900.0
        strategy.state.entry_protective_sl_order_id = "old-sl-1"
        strategy.state.avg_entry_price = 3050.0
        strategy.state.tp_price = 3500.0
        strategy.state.last_tp_update_candle_ts_ms = 0

        old_sl = strategy.state.trend_trailing_sl_price
        old_order_id = strategy.state.entry_protective_sl_order_id
        old_entry_sl = strategy.state.entry_protective_sl_price

        # Emit intents (this does NOT pollute state per the fix)
        boll = _boll_snapshot(middle=3000.0, candle_ts_ms=2000000)
        cvd = _cvd_snapshot()
        strategy.on_tick(price=3100.0, ts_ms=2000000, boll=boll, cvd=cvd)

        # State must still be old values (execution hasn't happened yet)
        assert strategy.state.trend_trailing_sl_price == old_sl
        assert strategy.state.entry_protective_sl_order_id == old_order_id
        assert strategy.state.entry_protective_sl_price == old_entry_sl


# ======================================================================
# 11. Execution layer: state updated only on success
# ======================================================================


class TestApplyUpdateTrendSLResult:
    """_apply_update_trend_sl_result must update state fields on success.

    These tests verify that the execution command processor correctly
    updates trend state fields ONLY when UPDATE_TREND_SL succeeds.
    We test the method directly with a simple processor built with mocks.
    """

    def test_success_updates_trend_trailing_sl_price(self):
        """On success, trend_trailing_sl_price is updated from the intent."""
        from src.strategies.boll_cvd_reclaim_strategy import (
            BollCvdReclaimStrategy,
            BollCvdReclaimStrategyConfig,
            TradeIntent,
        )

        import asyncio
        import time
        from unittest.mock import MagicMock

        # Setup strategy
        sizer = _make_sizer()
        config = BollCvdReclaimStrategyConfig()
        strategy = BollCvdReclaimStrategy(config, sizer)
        strategy.state.side = "LONG"
        strategy.state.layers = 1
        strategy.state.entry_regime = "TREND_BREAKOUT"
        strategy.state.trend_trailing_sl_price = 2900.0  # old SL
        strategy.state.trend_last_sl_update_ts_ms = 0
        strategy.state.entry_protective_sl_price = 2900.0
        strategy.state.entry_protective_sl_order_id = "old-sl-1"
        strategy.state.last_tp_update_ts_ms = 0

        # Intent with new SL price
        intent = TradeIntent(
            intent_type="UPDATE_TREND_SL",
            side="LONG",
            price=3100.0,
            layer_index=1,
            tp_price=3500.0,
            reason="trend_trailing_sl_tightened",
            size=_FakeSize(eth_qty=0.1),
            fast_cvd=0.01,
            previous_fast_cvd=0.005,
            buy_ratio=0.6,
            sell_ratio=0.4,
            boll_upper=3500.0,
            boll_middle=3000.0,
            boll_lower=2500.0,
            ts_ms=2000000,
            avg_entry_price=3200.0,
            breakeven_price=3205.0,
            tp_mode="UPPER",
            entry_protective_sl_price=2990.0,
        )

        # Build processor with minimal mocks
        from src.live.workers.execution_command_processor import ExecutionCommandProcessor
        from src.live import runtime_types as rt

        state_lock = asyncio.Lock()
        exec_state = rt.ExecutionState(
            current_position_id="test-pos-1",
            cash_before_position=500.0,
        )
        now = time.monotonic()
        account_snap = rt.AccountSnapshot(
            position=None, cash=500.0, equity=500.0,
            updated_monotonic=now, updated_ts_ms=2000000,
        )

        journal = MagicMock()
        journal.append = MagicMock()
        journal.record_tp_update = MagicMock()

        email_sender = MagicMock()
        store = MagicMock()
        trader = MagicMock()
        trader.symbol = "ETH-USDT-SWAP"

        processor = ExecutionCommandProcessor(
            state_lock=state_lock,
            execution_state=exec_state,
            account_snapshot=account_snap,
            trader=trader,
            strategy=strategy,
            journal=journal,
            state_store=store,
            email_sender=email_sender,
        )

        # Build a minimal TradeCommand
        sn = copy.deepcopy(strategy.state)
        cmd = rt.TradeCommand(
            intent=intent,
            strategy_state_snapshot=sn,
            tick_ts_ms=intent.ts_ms,
            created_monotonic=now,
            account_snapshot_updated_ts_ms=2000000,
            reason="trend_trailing_sl_tightened",
        )

        # Simulate SUCCESS result
        class FakeResult:
            ok = True
            protective_sl_order_id = "new-sl-1"
            protective_sl_price = "2990.00"
            contracts = "1"

        result = FakeResult()

        # Apply result directly
        asyncio.run(processor._apply_update_trend_sl_result(cmd, result))

        # State must reflect the NEW values
        assert strategy.state.trend_trailing_sl_price == 2990.0, (
            f"trend_trailing_sl_price should be 2990.0 after success, "
            f"got {strategy.state.trend_trailing_sl_price}"
        )
        assert strategy.state.trend_last_sl_update_ts_ms == 2000000, (
            "trend_last_sl_update_ts_ms should be set to intent.ts_ms on success"
        )
        assert strategy.state.entry_protective_sl_price == 2990.0, (
            "entry_protective_sl_price should be updated from intent"
        )
        assert strategy.state.entry_protective_sl_order_id == "new-sl-1", (
            "entry_protective_sl_order_id should be updated from result"
        )
        assert strategy.state.last_tp_update_ts_ms == 2000000, (
            "last_tp_update_ts_ms should be set to intent.ts_ms on success"
        )


# ======================================================================
# 12. No hardcoded small risk parameters
# ======================================================================


class TestNoHardcodedRiskParams:
    """Verify no hardcoded small risk parameters in trend breakout code."""

    def test_no_hardcoded_trade_risk_pct(self):
        """TRADE_RISK_PCT must not have a hardcoded small value in strategy code."""
        import inspect
        from src.strategies.boll_cvd_reclaim_strategy import BollCvdReclaimStrategy

        source = inspect.getsource(BollCvdReclaimStrategy)
        lines = source.split("\n")
        for line in lines:
            # Must not contain hardcoded risk percentage assignments
            if "trade_risk_pct" in line and "=" in line and "config" not in line:
                assert "os.getenv" in line or "self.config" in line or "self.sizer" in line or "sizer" in line, (
                    f"Hardcoded risk param found: {line.strip()}"
                )

    def test_no_hardcoded_max_order_notional(self):
        """MAX_ORDER_NOTIONAL_USDT must not be hardcoded in strategy."""
        import inspect
        from src.strategies.boll_cvd_reclaim_strategy import BollCvdReclaimStrategy

        source = inspect.getsource(BollCvdReclaimStrategy)
        assert "max_order_notional" not in source.lower(), (
            "Strategy must not hardcode MAX_ORDER_NOTIONAL_USDT"
        )

    def test_no_dry_run_flag(self):
        """DRY_RUN must not be checked in strategy layer."""
        import inspect
        from src.strategies.boll_cvd_reclaim_strategy import BollCvdReclaimStrategy

        source = inspect.getsource(BollCvdReclaimStrategy)
        assert "dry_run" not in source.lower(), (
            "Strategy must not contain dry_run logic"
        )

    def test_no_shadow_mode_flag(self):
        """Shadow mode must not exist in strategy layer."""
        import inspect
        from src.strategies.boll_cvd_reclaim_strategy import BollCvdReclaimStrategy

        source = inspect.getsource(BollCvdReclaimStrategy)
        assert "shadow" not in source.lower(), (
            "Strategy must not contain shadow mode logic"
        )
