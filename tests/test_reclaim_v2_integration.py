"""Integration tests for Reclaim V2: anchored divergence + POC stop + arming flow."""
from __future__ import annotations

import pytest
from src.indicators.cvd_tracker import CvdSnapshot
from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
)


def _boll(middle=2000, upper=2100, lower=1900, alert_switch_on=True) -> BollSnapshot:
    return BollSnapshot(
        inst_id="ETH-USDT-SWAP", candle_ts_ms=1000, close=middle,
        middle=middle, upper=upper, lower=lower,
        upper_distance_pct=0.0, lower_distance_pct=0.001,
        alert_switch_on=alert_switch_on, live_mode=True,
    )


def _cvd(ts_ms=1000, price=1901, fast_cvd=0.0, previous_fast_cvd=0.0,
         buy_ratio=0.6, sell_ratio=0.4,
         buy_volume=60.0, sell_volume=40.0,
         cumulative_buy_volume=1_000_000.0, cumulative_sell_volume=1_500_000.0,
         cross_positive=True, cross_negative=False,
         cvd_increasing=True, cvd_decreasing=False,
         no_new_low=True, no_new_high=True) -> CvdSnapshot:
    return CvdSnapshot(
        ts_ms=ts_ms, price=price,
        side="buy" if buy_ratio >= sell_ratio else "sell",
        size=1.0, signed_delta=1.0, total_cvd=10.0,
        fast_cvd=fast_cvd, previous_fast_cvd=previous_fast_cvd,
        buy_volume=buy_volume, sell_volume=sell_volume,
        buy_ratio=buy_ratio, sell_ratio=sell_ratio,
        cross_positive=cross_positive, cross_negative=cross_negative,
        cvd_increasing=cvd_increasing, cvd_decreasing=cvd_decreasing,
        no_new_low=no_new_low, no_new_high=no_new_high,
        window_low=1897.0, window_high=1905.0,
        burst_net_move_pct=0.0, burst_range_pct=0.002, baseline_range_pct=0.001,
        burst_move_ratio=2.0, burst_volume=10.0, baseline_volume=5.0,
        burst_volume_ratio=2.0,
        up_burst=False, down_burst=False,
        cumulative_buy_volume=cumulative_buy_volume,
        cumulative_sell_volume=cumulative_sell_volume,
    )


def _strategy(**overrides) -> BollCvdReclaimStrategy:
    cfg_kwargs = dict(
        min_outside_pct=0.001,
        entry_min_reward_risk=0.0,
        entry_fee_slippage_buffer_pct=0.0,
        order_cooldown_seconds=0,
        entry_reclaim_v2_enabled=True,
        entry_reclaim_require_anchored_divergence=True,
        entry_sweep_profile_enabled=True,
        entry_sweep_profile_bucket_pct=0.0002,
        entry_poc_stop_enabled=True,
        entry_poc_stop_min_tail_pct=0.003,
        entry_poc_stop_buffer_pct=0.001,
        entry_extreme_stop_buffer_pct=0.001,
        entry_sl_buffer_pct=0.0005,
        entry_reclaim_confirm_seconds=0,
        entry_reclaim_inside_band=False,
        entry_cvd_divergence_enabled=False,
        entry_cvd_absorption_enabled=False,
        entry_max_stop_distance_pct=0,
    )
    cfg_kwargs.update(overrides)
    cfg = BollCvdReclaimStrategyConfig(**cfg_kwargs)
    sizer = SimplePositionSizer(SimplePositionSizerConfig(
        dry_run_equity_usdt=10_000, leverage=20, trade_risk_pct=0.01,
        fee_slippage_buffer_pct=0.001,
    ))
    return BollCvdReclaimStrategy(cfg, sizer)


# ── Test 1: first outside tick → LOWER_OUTSIDE_OBSERVED, NOT armed ──

def test_first_outside_only_observed_not_armed() -> None:
    """First tick outside lower band should be OBSERVED, not ARMED."""
    strat = _strategy()
    boll = _boll()
    price = boll.lower * 0.998  # 0.2% below lower
    cvd = _cvd(ts_ms=1000, price=price, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000)
    strat.on_tick(price, 1000, boll, cvd)

    assert strat.state.lower_outside_observed is True
    assert strat.state.lower_armed is False  # NOT armed on first outside
    assert strat._lower_orderflow.initialised is True
    assert strat._lower_orderflow.direction == "DOWN"


# ── Test 2: first valid extreme → NOT armed ──────────────────────────

def test_first_extreme_not_armed() -> None:
    """First valid lower extreme should NOT arm (needs divergence)."""
    strat = _strategy(min_outside_pct=0.001)
    boll = _boll()

    # Tick 1: outside observed
    p1 = boll.lower * 0.998
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))

    # Tick 2: lower — first extreme
    p2 = boll.lower * 0.995  # deeper
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000))

    assert strat.state.lower_first_extreme_price is not None
    assert strat.state.lower_anchored_divergence_confirmed is False
    assert strat.state.lower_armed is False  # still not armed


# ── Test 3: new low + CVD follows lower → NOT armed ──────────────────

def test_new_low_cvd_follows_not_armed() -> None:
    """Price makes new low but CVD also makes new low — no divergence."""
    strat = _strategy(min_outside_pct=0.001)
    boll = _boll()

    # Anchor
    p1 = boll.lower * 0.998
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))
    # First extreme
    p2 = boll.lower * 0.994
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000))
    assert strat.state.lower_first_extreme_price is not None

    # New extreme — CVD also makes new low (more sell volume)
    p3 = boll.lower * 0.990
    strat.on_tick(p3, 3000, boll, _cvd(ts_ms=3000, price=p3, cumulative_buy_volume=500_000, cumulative_sell_volume=1_800_000))

    assert strat.state.lower_anchored_divergence_confirmed is False
    assert strat.state.lower_armed is False


# ── Test 4: new low + CVD recovers → ARMED ───────────────────────────

def test_new_low_cvd_recovers_armed() -> None:
    """Price makes new low but CVD recovers → anchored divergence → ARMED."""
    strat = _strategy(min_outside_pct=0.001)
    boll = _boll()

    # Anchor
    p1 = boll.lower * 0.998
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))
    # First extreme: anchored_cvd = -1_000_000 (500k - 1.5M)
    p2 = boll.lower * 0.994
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))

    # New extreme with CVD recovery: anchored_cvd = 600k - 1.4M = -800k (improved from -1M)
    p3 = boll.lower * 0.990
    strat.on_tick(p3, 3000, boll, _cvd(ts_ms=3000, price=p3, cumulative_buy_volume=600_000, cumulative_sell_volume=1_400_000))

    assert strat.state.lower_anchored_divergence_confirmed is True
    assert strat.state.lower_armed is True
    assert strat._lower_orderflow.new_extreme_count >= 2


# ── Test 5: upper mirror — new high + CVD reverses → ARMED ───────────

def test_upper_new_high_cvd_reverses_armed() -> None:
    """Price makes new high but CVD reverses down → bearish divergence → ARMED."""
    strat = _strategy(min_outside_pct=0.001)
    boll = _boll()

    # Anchor
    p1 = boll.upper * 1.002
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
                                         buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                                         cvd_increasing=False, cvd_decreasing=True, no_new_high=True))

    # First extreme
    p2 = boll.upper * 1.006
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
                                         buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                                         cvd_increasing=False, cvd_decreasing=True, no_new_high=True))

    # New extreme with CVD reversal (less bullish)
    p3 = boll.upper * 1.010
    strat.on_tick(p3, 3000, boll, _cvd(ts_ms=3000, price=p3, cumulative_buy_volume=1_400_000, cumulative_sell_volume=600_000,
                                         buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                                         cvd_increasing=False, cvd_decreasing=True, no_new_high=True))

    assert strat.state.upper_anchored_divergence_confirmed is True
    assert strat.state.upper_armed is True


# ── Test 6: absorption alone does NOT arm ────────────────────────────

def test_absorption_alone_does_not_arm_v2() -> None:
    """When V2 is enabled, absorption confirmed must NOT allow entry."""
    strat = _strategy(entry_cvd_absorption_enabled=True)
    boll = _boll()

    # Simulate absorption being set (old path)
    strat.state.lower_cvd_absorption_confirmed = True
    assert strat._lower_cvd_structure_ok() is False  # V2 gate: needs anchored divergence
    assert strat.state.lower_anchored_divergence_confirmed is False


# ── Test 7: V2 disabled → legacy path works ─────────────────────────

def test_v2_disabled_legacy_path() -> None:
    """When V2 is disabled, old CVD structure path should work normally."""
    strat = _strategy(
        entry_reclaim_v2_enabled=False,
        entry_cvd_structure_mode="DIVERGENCE_OR_ABSORPTION",
        entry_cvd_divergence_enabled=True,
        entry_cvd_absorption_enabled=True,
    )
    # Legacy absorption confirmed should pass
    strat.state.lower_cvd_absorption_confirmed = True
    assert strat._lower_cvd_structure_ok() is True

    # Legacy divergence confirmed should pass
    strat.state.lower_cvd_divergence_confirmed = True
    assert strat._lower_cvd_structure_ok() is True


# ── Test 8: tracker reset on lower armed reset ──────────────────────

def test_tracker_reset_on_lower_armed_reset() -> None:
    strat = _strategy(min_outside_pct=0.001)
    boll = _boll()

    # Set up outside observation + extreme
    p1 = boll.lower * 0.998
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))
    p2 = boll.lower * 0.994
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))

    assert strat._lower_orderflow.initialised is True
    assert strat.state.lower_first_extreme_price is not None

    # Reset
    strat._reset_lower_armed()

    assert strat._lower_orderflow.initialised is False
    assert strat._lower_orderflow.new_extreme_count == 0
    assert strat.state.lower_outside_observed is False
    assert strat.state.lower_first_extreme_price is None
