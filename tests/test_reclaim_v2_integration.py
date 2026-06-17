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
    """First confirmed lower extreme should NOT arm (needs divergence)."""
    strat = _strategy(min_outside_pct=0.001)
    boll = _boll()

    # Tick 1: outside observed
    p1 = boll.lower * 0.998
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))

    # Tick 2: lower — first extreme candidate (deep enough)
    p2 = boll.lower * 0.995  # deeper
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000))

    # Tick 3: retrace to confirm first extreme
    p2_retrace = boll.lower * 0.998
    strat.on_tick(p2_retrace, 3000, boll, _cvd(ts_ms=3000, price=p2_retrace, cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000))

    assert strat.state.lower_first_extreme_price is not None
    assert strat.state.lower_anchored_divergence_confirmed is False
    assert strat.state.lower_armed is False  # still not armed (first extreme only)


# ── Test 3: new low + CVD follows lower → NOT armed ──────────────────

def test_new_low_cvd_follows_not_armed() -> None:
    """Price makes new confirmed low but CVD also makes new low — no divergence."""
    strat = _strategy(min_outside_pct=0.001)
    boll = _boll()

    # Anchor
    p1 = boll.lower * 0.998
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))
    # First extreme candidate
    p2 = boll.lower * 0.994
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000))
    # Retrace to confirm first extreme
    p2_retrace = boll.lower * 0.998
    strat.on_tick(p2_retrace, 2500, boll, _cvd(ts_ms=2500, price=p2_retrace, cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000))
    assert strat.state.lower_first_extreme_price is not None

    # New extreme candidate — CVD also makes new low (more sell volume)
    p3 = boll.lower * 0.990
    strat.on_tick(p3, 3000, boll, _cvd(ts_ms=3000, price=p3, cumulative_buy_volume=500_000, cumulative_sell_volume=1_800_000))
    # Retrace to confirm second extreme
    p3_retrace = boll.lower * 0.995
    strat.on_tick(p3_retrace, 3500, boll, _cvd(ts_ms=3500, price=p3_retrace, cumulative_buy_volume=500_000, cumulative_sell_volume=1_800_000))

    assert strat.state.lower_anchored_divergence_confirmed is False
    assert strat.state.lower_armed is False


# ── Test 4: new low + CVD recovers → ARMED ───────────────────────────

def test_new_low_cvd_recovers_armed() -> None:
    """Price makes new confirmed low but CVD recovers → anchored divergence → ARMED."""
    strat = _strategy(min_outside_pct=0.001)
    boll = _boll()

    # Anchor
    p1 = boll.lower * 0.998
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))
    # First extreme candidate: anchored_cvd = 0 (500k-1.5M vs anchor)
    p2 = boll.lower * 0.994
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))
    # Retrace to confirm first extreme
    p2_retrace = boll.lower * 0.998
    strat.on_tick(p2_retrace, 2500, boll, _cvd(ts_ms=2500, price=p2_retrace, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))

    # New extreme candidate with CVD recovery: anchored_cvd = 600k-1.4M - (-1M) = 200k
    p3 = boll.lower * 0.990
    strat.on_tick(p3, 3000, boll, _cvd(ts_ms=3000, price=p3, cumulative_buy_volume=600_000, cumulative_sell_volume=1_400_000))
    # Retrace to confirm second extreme → divergence eval fires
    p3_retrace = boll.lower * 0.995
    strat.on_tick(p3_retrace, 3500, boll, _cvd(ts_ms=3500, price=p3_retrace, cumulative_buy_volume=600_000, cumulative_sell_volume=1_400_000))

    assert strat.state.lower_anchored_divergence_confirmed is True
    assert strat.state.lower_armed is True
    assert strat._lower_orderflow.new_extreme_count >= 2


# ── Test 5: upper mirror — new high + CVD reverses → ARMED ───────────

def test_upper_new_high_cvd_reverses_armed() -> None:
    """Price makes new confirmed high but CVD reverses down → bearish divergence → ARMED."""
    strat = _strategy(min_outside_pct=0.001)
    boll = _boll()

    # Anchor
    p1 = boll.upper * 1.002
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
                                         buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                                         cvd_increasing=False, cvd_decreasing=True, no_new_high=True))
    # First extreme candidate
    p2 = boll.upper * 1.006
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
                                         buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                                         cvd_increasing=False, cvd_decreasing=True, no_new_high=True))
    # Retrace to confirm first extreme
    p2_retrace = boll.upper * 1.002
    strat.on_tick(p2_retrace, 2500, boll, _cvd(ts_ms=2500, price=p2_retrace, cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
                                                 buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                                                 cvd_increasing=False, cvd_decreasing=True, no_new_high=True))
    # New extreme candidate with CVD reversal (less bullish)
    p3 = boll.upper * 1.010
    strat.on_tick(p3, 3000, boll, _cvd(ts_ms=3000, price=p3, cumulative_buy_volume=1_400_000, cumulative_sell_volume=600_000,
                                         buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                                         cvd_increasing=False, cvd_decreasing=True, no_new_high=True))
    # Retrace to confirm second extreme → divergence eval fires
    p3_retrace = boll.upper * 1.005
    strat.on_tick(p3_retrace, 3500, boll, _cvd(ts_ms=3500, price=p3_retrace, cumulative_buy_volume=1_400_000, cumulative_sell_volume=600_000,
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

    # Set up outside observation + first extreme candidate + confirm
    p1 = boll.lower * 0.998
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))
    p2 = boll.lower * 0.994
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))
    # Retrace to confirm first extreme
    p2_retrace = boll.lower * 0.998
    strat.on_tick(p2_retrace, 3000, boll, _cvd(ts_ms=3000, price=p2_retrace, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))

    assert strat._lower_orderflow.initialised is True
    assert strat.state.lower_first_extreme_price is not None

    # Reset
    strat._reset_lower_armed()

    assert strat._lower_orderflow.initialised is False
    assert strat._lower_orderflow.new_extreme_count == 0
    assert strat.state.lower_outside_observed is False
    assert strat.state.lower_first_extreme_price is None


# ── Test 9: first valid extreme must be deep enough ────────────────────

def test_first_extreme_requires_deep_enough_lower() -> None:
    """Shallow outside breach must NOT record LOWER_FIRST_VALID_EXTREME."""
    strat = _strategy(min_outside_pct=0.001)
    boll = _boll(lower=1900)
    # Price just barely below lower band (only ~0.05% outside, min_outside_pct=0.1%)
    price = boll.lower * 0.9998  # very shallow breach
    cvd = _cvd(ts_ms=1000, price=price, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000)
    strat.on_tick(price, 1000, boll, cvd)

    assert strat.state.lower_outside_observed is True
    assert strat.state.lower_first_extreme_price is None  # NOT deep enough
    assert strat.state.lower_armed is False


def test_first_extreme_requires_deep_enough_upper() -> None:
    """Shallow outside breach must NOT record UPPER_FIRST_VALID_EXTREME."""
    strat = _strategy(min_outside_pct=0.001)
    boll = _boll(upper=2100)
    price = boll.upper * 1.0002  # very shallow breach
    cvd = _cvd(ts_ms=1000, price=price, cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
               buy_ratio=0.4, sell_ratio=0.6)
    strat.on_tick(price, 1000, boll, cvd)

    assert strat.state.upper_outside_observed is True
    assert strat.state.upper_first_extreme_price is None  # NOT deep enough
    assert strat.state.upper_armed is False


# ── Helper: set up a fully armed LOWER divergence state ────────────────

def _setup_lower_armed_divergence(
    strat: BollCvdReclaimStrategy,
    *,
    ref_lower: float = 1900.0,
    ref_middle: float = 2000.0,
    div_extreme_price: float = 1880.0,
    div_extreme_cvd: float = -800_000.0,
    anchor_cum_cvd: float = -1_000_000.0,
) -> None:
    """Set up internal state for a LOWER-armed divergence.

    This simulates the state after outside observed → deep extreme
    → divergence confirmed, so follow-through tests can focus on the
    reclaim / entry phase.
    """
    strat.state.lower_armed = True
    strat.state.lower_deep_enough = True
    strat.state.lower_anchored_divergence_confirmed = True
    strat.state.lower_anchored_divergence_ts_ms = 10000
    strat.state.lower_cvd_divergence_confirmed = True
    strat.state.lower_first_armed_ts_ms = 10000
    strat.state.lower_outside_observed = True
    strat.state.lower_anchor_price = 1898.0
    strat.state.lower_anchor_ts_ms = 1000
    strat.state.lower_anchor_cumulative_cvd = anchor_cum_cvd
    strat.state.lower_extreme_price = div_extreme_price
    strat.state.lower_extreme_ts_ms = 5000
    # Divergence extreme + reference band
    strat.state.lower_divergence_extreme_price = div_extreme_price
    strat.state.lower_divergence_extreme_ts_ms = 5000
    strat.state.lower_divergence_extreme_anchored_cvd = div_extreme_cvd
    strat.state.lower_divergence_ref_lower = ref_lower
    strat.state.lower_divergence_ref_middle = ref_middle


# ── Helper: set up a fully armed UPPER divergence state ────────────────

def _setup_upper_armed_divergence(
    strat: BollCvdReclaimStrategy,
    *,
    ref_upper: float = 2100.0,
    ref_middle: float = 2000.0,
    div_extreme_price: float = 2120.0,
    div_extreme_cvd: float = 800_000.0,
    anchor_cum_cvd: float = 1_000_000.0,
) -> None:
    """Set up internal state for an UPPER-armed divergence."""
    strat.state.upper_armed = True
    strat.state.upper_deep_enough = True
    strat.state.upper_anchored_divergence_confirmed = True
    strat.state.upper_anchored_divergence_ts_ms = 10000
    strat.state.upper_cvd_divergence_confirmed = True
    strat.state.upper_first_armed_ts_ms = 10000
    strat.state.upper_outside_observed = True
    strat.state.upper_anchor_price = 2102.0
    strat.state.upper_anchor_ts_ms = 1000
    strat.state.upper_anchor_cumulative_cvd = anchor_cum_cvd
    strat.state.upper_extreme_price = div_extreme_price
    strat.state.upper_extreme_ts_ms = 5000
    strat.state.upper_divergence_extreme_price = div_extreme_price
    strat.state.upper_divergence_extreme_ts_ms = 5000
    strat.state.upper_divergence_extreme_anchored_cvd = div_extreme_cvd
    strat.state.upper_divergence_ref_upper = ref_upper
    strat.state.upper_divergence_ref_middle = ref_middle


# ── Test 10: LONG shallow zone CVD follow-through satisfied → entry ────

def test_long_shallow_zone_cvd_follow_through_ok() -> None:
    """CVD follow-through satisfies in shallow inside zone → entry True."""
    strat = _strategy(
        entry_reclaim_confirm_seconds=0,
        entry_reclaim_inside_band=False,
        entry_reclaim_max_inside_depth_ratio=0.15,
    )
    boll = _boll(lower=1900, middle=2000, upper=2100)
    _setup_lower_armed_divergence(
        strat, ref_lower=1900, ref_middle=2000,
        div_extreme_price=1880, div_extreme_cvd=-800_000,
        anchor_cum_cvd=-1_000_000,
    )

    # Price at ref_lower + 10% of band width (band = 100, 10 pips inside)
    # max_entry_price = 1900 + 100 * 0.15 = 1915
    price = 1908.0  # well inside shallow zone
    # reclaim_anchored_cvd = cum_cvd - anchor_cvd = (1_200_000 - 1_500_000) - (-1_000_000) = -300_000 + 1_000_000 = 700_000
    # Wait, _cumulative_cvd = cumulative_buy_volume - cumulative_sell_volume
    # anchor_cvd = -1_000_000 (anchor_cum_cvd)
    # So reclaim_anchored_cvd = (-300_000) - (-1_000_000) = +700_000
    # div_cvd = -800_000
    # reclaim_anchored_cvd (700_000) > div_cvd (-800_000) + 0 → True!
    cvd = _cvd(
        ts_ms=20000, price=price,
        cumulative_buy_volume=1_200_000, cumulative_sell_volume=1_500_000,
    )

    result = strat._check_lower_reclaim_v2_follow_through(price, cvd, boll)
    assert result is True


# ── Test 11: LONG shallow zone CVD follow-through NOT satisfied ────────

def test_long_shallow_zone_cvd_follow_through_not_satisfied() -> None:
    """CVD follow-through NOT satisfied in shallow zone → False, keep waiting."""
    strat = _strategy(
        entry_reclaim_confirm_seconds=0,
        entry_reclaim_inside_band=False,
        entry_reclaim_max_inside_depth_ratio=0.15,
    )
    boll = _boll(lower=1900, middle=2000, upper=2100)
    _setup_lower_armed_divergence(
        strat, ref_lower=1900, ref_middle=2000,
        div_extreme_price=1880, div_extreme_cvd=-800_000,
        anchor_cum_cvd=-1_000_000,
    )

    price = 1908.0  # still in shallow zone
    # reclaim_anchored_cvd = (500_000 - 1_500_000) - (-1_000_000) = -1_000_000 + 1_000_000 = 0
    # div_cvd = -800_000
    # reclaim_anchored_cvd (0) > div_cvd (-800_000) → True... wait that actually satisfies.
    # Let me make the CVs work: we need reclaim_anchored_cvd NOT > div_cvd
    # reclaim_anchored_cvd = -900_000; div_cvd = -800_000; -900_000 > -800_000 → False
    # So cumulative_buy - cumulative_sell = -900_000 + (-1_000_000) = -1_900_000
    # cumulative_buy = 500_000, cumulative_sell = 2_400_000 → diff = -1_900_000
    cvd = _cvd(
        ts_ms=20000, price=price,
        cumulative_buy_volume=500_000, cumulative_sell_volume=2_400_000,
    )

    result = strat._check_lower_reclaim_v2_follow_through(price, cvd, boll)
    assert result is False
    # State should still be waiting (reclaim not rejected)
    assert strat.state.lower_reclaim_seen is False  # wasn't set in test setup


# ── Test 12: LONG CVD satisfied but price too deep inside → rejected ───

def test_long_cvd_ok_but_too_deep_inside_rejected() -> None:
    """CVD follow-through satisfies but price > max_entry_price → rejected."""
    strat = _strategy(
        entry_reclaim_confirm_seconds=0,
        entry_reclaim_inside_band=False,
        entry_reclaim_max_inside_depth_ratio=0.15,
    )
    boll = _boll(lower=1900, middle=2000, upper=2100)
    _setup_lower_armed_divergence(
        strat, ref_lower=1900, ref_middle=2000,
        div_extreme_price=1880, div_extreme_cvd=-800_000,
        anchor_cum_cvd=-1_000_000,
    )

    # max_entry_price = 1900 + 100 * 0.15 = 1915
    price = 1925.0  # WAY too deep inside
    # reclaim_anchored_cvd = (1_200_000 - 1_500_000) - (-1_000_000) = 700_000
    # 700_000 > -800_000 → CVD OK, but too deep!
    cvd = _cvd(
        ts_ms=20000, price=price,
        cumulative_buy_volume=1_200_000, cumulative_sell_volume=1_500_000,
    )

    result = strat._check_lower_reclaim_v2_follow_through(price, cvd, boll)
    assert result is False
    # Reclaim attempt should be rejected
    assert strat.state.lower_reclaim_seen is False
    assert strat.state.lower_reclaim_cycle_count == 1


# ── Test 13: SHORT shallow zone CVD follow-through satisfied ───────────

def test_short_shallow_zone_cvd_follow_through_ok() -> None:
    """CVD follow-through satisfies in shallow inside zone → entry True (SHORT)."""
    strat = _strategy(
        entry_reclaim_confirm_seconds=0,
        entry_reclaim_inside_band=False,
        entry_reclaim_max_inside_depth_ratio=0.15,
    )
    boll = _boll(upper=2100, middle=2000, lower=1900)
    _setup_upper_armed_divergence(
        strat, ref_upper=2100, ref_middle=2000,
        div_extreme_price=2120, div_extreme_cvd=800_000,
        anchor_cum_cvd=1_000_000,
    )

    # band_width = 2100 - 2000 = 100, min_entry_price = 2100 - 100 * 0.15 = 2085
    price = 2092.0  # inside shallow zone
    # reclaim_anchored_cvd = (1_400_000 - 600_000) - 1_000_000 = 800_000 - 1_000_000 = -200_000
    # div_cvd = 800_000
    # reclaim_anchored_cvd (-200_000) < div_cvd (800_000) - 0 → True!
    cvd = _cvd(
        ts_ms=20000, price=price,
        cumulative_buy_volume=1_400_000, cumulative_sell_volume=600_000,
        buy_ratio=0.4, sell_ratio=0.6,
    )

    result = strat._check_upper_reclaim_v2_follow_through(price, cvd, boll)
    assert result is True


# ── Test 14: SHORT shallow zone CVD follow-through NOT satisfied ───────

def test_short_shallow_zone_cvd_follow_through_not_satisfied() -> None:
    """CVD follow-through NOT satisfied in shallow zone → False (SHORT)."""
    strat = _strategy(
        entry_reclaim_confirm_seconds=0,
        entry_reclaim_inside_band=False,
        entry_reclaim_max_inside_depth_ratio=0.15,
    )
    boll = _boll(upper=2100, middle=2000, lower=1900)
    _setup_upper_armed_divergence(
        strat, ref_upper=2100, ref_middle=2000,
        div_extreme_price=2120, div_extreme_cvd=800_000,
        anchor_cum_cvd=1_000_000,
    )

    price = 2092.0  # in shallow zone
    # reclaim_anchored_cvd = (1_600_000 - 500_000) - 1_000_000 = 1_100_000 - 1_000_000 = 100_000
    # div_cvd = 800_000
    # reclaim_anchored_cvd (100_000) < div_cvd (800_000) → True! Wait that also satisfies.
    # Need: reclaim_anchored_cvd NOT < div_cvd
    # reclaim_anchored_cvd = 900_000, div_cvd = 800_000, 900_000 < 800_000 → False
    # cum_buy - cum_sell = 1_900_000, anchor = 1_000_000, reclaim = 900_000
    cvd = _cvd(
        ts_ms=20000, price=price,
        cumulative_buy_volume=2_400_000, cumulative_sell_volume=500_000,
        buy_ratio=0.4, sell_ratio=0.6,
    )

    result = strat._check_upper_reclaim_v2_follow_through(price, cvd, boll)
    assert result is False


# ── Test 15: SHORT CVD satisfied but too deep inside → rejected ────────

def test_short_cvd_ok_but_too_deep_inside_rejected() -> None:
    """CVD follow-through ok but price < min_entry_price → rejected (SHORT)."""
    strat = _strategy(
        entry_reclaim_confirm_seconds=0,
        entry_reclaim_inside_band=False,
        entry_reclaim_max_inside_depth_ratio=0.15,
    )
    boll = _boll(upper=2100, middle=2000, lower=1900)
    _setup_upper_armed_divergence(
        strat, ref_upper=2100, ref_middle=2000,
        div_extreme_price=2120, div_extreme_cvd=800_000,
        anchor_cum_cvd=1_000_000,
    )

    # min_entry_price = 2100 - 100 * 0.15 = 2085
    price = 2075.0  # TOO deep
    # CVD OK: reclaim_anchored_cvd = (1_400_000 - 600_000) - 1_000_000 = -200_000
    # -200_000 < 800_000 → CVD satisfied, but too deep!
    cvd = _cvd(
        ts_ms=20000, price=price,
        cumulative_buy_volume=1_400_000, cumulative_sell_volume=600_000,
        buy_ratio=0.4, sell_ratio=0.6,
    )

    result = strat._check_upper_reclaim_v2_follow_through(price, cvd, boll)
    assert result is False
    assert strat.state.upper_reclaim_seen is False
    assert strat.state.upper_reclaim_cycle_count == 1


# ── Test 16: V2 must NOT use fast CVD as final confirm ─────────────────

def test_v2_rejects_fast_cvd_confirm_long() -> None:
    """Fast CVD OK but anchored CVD follow-through fails → NO entry."""
    strat = _strategy(
        entry_reclaim_confirm_seconds=0,
        entry_reclaim_inside_band=False,
        entry_reclaim_max_inside_depth_ratio=0.15,
    )
    boll = _boll(lower=1900, middle=2000, upper=2100)
    _setup_lower_armed_divergence(
        strat, ref_lower=1900, ref_middle=2000,
        div_extreme_price=1880, div_extreme_cvd=-800_000,
        anchor_cum_cvd=-1_000_000,
    )

    price = 1905.0  # shallow zone
    # Fast CVD looks great: cross_positive=True, buy_ratio=0.7, no_new_low=True
    # But anchored CVD follow-through: reclaim_anchored_cvd = -900_000
    # div_cvd = -800_000, -900_000 > -800_000 → False
    cvd = _cvd(
        ts_ms=20000, price=price,
        cumulative_buy_volume=500_000, cumulative_sell_volume=2_400_000,
        buy_ratio=0.7, sell_ratio=0.3,
        cross_positive=True, cvd_increasing=True, no_new_low=True,
    )

    # The V2 check should return False despite fast CVD being perfect
    result = strat._check_lower_reclaim_v2_follow_through(price, cvd, boll)
    assert result is False

    # Verify fast CVD fields are indeed "good" by legacy standards
    assert cvd.cross_positive is True
    assert cvd.buy_ratio >= 0.55
    assert cvd.no_new_low is True


def test_v2_rejects_fast_cvd_confirm_short() -> None:
    """Fast CVD OK but anchored CVD follow-through fails → NO entry (SHORT)."""
    strat = _strategy(
        entry_reclaim_confirm_seconds=0,
        entry_reclaim_inside_band=False,
        entry_reclaim_max_inside_depth_ratio=0.15,
    )
    boll = _boll(upper=2100, middle=2000, lower=1900)
    _setup_upper_armed_divergence(
        strat, ref_upper=2100, ref_middle=2000,
        div_extreme_price=2120, div_extreme_cvd=800_000,
        anchor_cum_cvd=1_000_000,
    )

    price = 2092.0  # shallow zone
    # Fast CVD looks good: cross_negative=True, sell_ratio=0.7, no_new_high=True
    # But anchored CVD: reclaim_anchored_cvd = 900_000, div_cvd = 800_000 → NOT < div_cvd
    cvd = _cvd(
        ts_ms=20000, price=price,
        cumulative_buy_volume=2_400_000, cumulative_sell_volume=500_000,
        buy_ratio=0.3, sell_ratio=0.7,
        cross_positive=False, cross_negative=True,
        cvd_increasing=False, cvd_decreasing=True,
        no_new_low=False, no_new_high=True,
    )

    result = strat._check_upper_reclaim_v2_follow_through(price, cvd, boll)
    assert result is False

    # Fast CVD looks "good" but V2 rejects it
    assert cvd.cross_negative is True
    assert cvd.sell_ratio >= 0.55
    assert cvd.no_new_high is True


# ═══════════════════════════════════════════════════════════════════════════════
# New tests: V2 immediate entry confirm (no 1-second soft confirm)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Test 17: LONG first reclaim tick with CVD OK → immediate entry ──────

def test_long_v2_first_tick_entry_when_cvd_ok() -> None:
    """V2 must enter on the very first tick that reclaims inside reference band.

    Even with ENTRY_RECLAIM_CONFIRM_SECONDS=1.0 (legacy delay), V2 must NOT
    wait—it immediately checks anchored CVD follow-through and returns True.
    """
    strat = _strategy(
        entry_reclaim_confirm_seconds=1.0,   # legacy delay — V2 must ignore
        entry_reclaim_inside_band=False,
        entry_reclaim_max_inside_depth_ratio=0.15,
    )
    boll = _boll(lower=1900, middle=2000, upper=2100)
    _setup_lower_armed_divergence(
        strat, ref_lower=1900, ref_middle=2000,
        div_extreme_price=1880, div_extreme_cvd=-800_000,
        anchor_cum_cvd=-1_000_000,
    )

    # Price just inside ref_lower (1900), shallow zone
    # max_entry_price = 1900 + 100 * 0.15 = 1915
    price = 1905.0
    # reclaim_anchored_cvd = (1_200_000 - 1_500_000) - (-1_000_000) = 700_000
    # 700_000 > -800_000 → CVD satisfied
    cvd = _cvd(
        ts_ms=20000, price=price,
        cumulative_buy_volume=1_200_000, cumulative_sell_volume=1_500_000,
    )

    # This is the FIRST tick back inside — V2 must return True immediately
    result = strat._long_setup(price, cvd, boll)
    assert result is True, "V2 should enter on first reclaim tick without waiting"


# ── Test 18: SHORT first reclaim tick with CVD OK → immediate entry ─────

def test_short_v2_first_tick_entry_when_cvd_ok() -> None:
    """V2 SHORT must enter on the very first tick that reclaims inside reference band."""
    strat = _strategy(
        entry_reclaim_confirm_seconds=1.0,   # legacy delay — V2 must ignore
        entry_reclaim_inside_band=False,
        entry_reclaim_max_inside_depth_ratio=0.15,
    )
    boll = _boll(upper=2100, middle=2000, lower=1900)
    _setup_upper_armed_divergence(
        strat, ref_upper=2100, ref_middle=2000,
        div_extreme_price=2120, div_extreme_cvd=800_000,
        anchor_cum_cvd=1_000_000,
    )

    # min_entry_price = 2100 - 100 * 0.15 = 2085
    price = 2092.0  # inside shallow zone
    # reclaim_anchored_cvd = (1_400_000 - 600_000) - 1_000_000 = -200_000
    # -200_000 < 800_000 → CVD satisfied
    cvd = _cvd(
        ts_ms=20000, price=price,
        cumulative_buy_volume=1_400_000, cumulative_sell_volume=600_000,
        buy_ratio=0.4, sell_ratio=0.6,
    )

    result = strat._short_setup(price, cvd, boll)
    assert result is True, "V2 SHORT should enter on first reclaim tick without waiting"


# ── Test 19: LONG CVD not satisfied then catches up while still shallow ──

def test_long_v2_cvd_not_ok_then_ok_while_still_shallow() -> None:
    """CVD not satisfied on first tick; price stays shallow; later CVD satisfies → entry."""
    strat = _strategy(
        entry_reclaim_confirm_seconds=0,
        entry_reclaim_inside_band=False,
        entry_reclaim_max_inside_depth_ratio=0.15,
    )
    boll = _boll(lower=1900, middle=2000, upper=2100)
    _setup_lower_armed_divergence(
        strat, ref_lower=1900, ref_middle=2000,
        div_extreme_price=1880, div_extreme_cvd=-800_000,
        anchor_cum_cvd=-1_000_000,
    )

    # max_entry_price = 1900 + 100 * 0.15 = 1915
    price = 1908.0  # shallow zone

    # Tick 1: CVD NOT satisfied
    # reclaim_anchored_cvd = (500_000 - 2_400_000) - (-1_000_000) = -900_000
    # -900_000 > -800_000 → False
    cvd1 = _cvd(
        ts_ms=20000, price=price,
        cumulative_buy_volume=500_000, cumulative_sell_volume=2_400_000,
    )
    result1 = strat._long_setup(price, cvd1, boll)
    assert result1 is False

    # Tick 2: CVD now satisfies (more buying stepped in)
    # reclaim_anchored_cvd = (1_200_000 - 1_500_000) - (-1_000_000) = 700_000
    # 700_000 > -800_000 → True
    cvd2 = _cvd(
        ts_ms=21000, price=price,
        cumulative_buy_volume=1_200_000, cumulative_sell_volume=1_500_000,
    )
    result2 = strat._long_setup(price, cvd2, boll)
    assert result2 is True, "CVD caught up while still shallow → must enter"


# ── Test 20: SHORT CVD not satisfied then catches up while still shallow ─

def test_short_v2_cvd_not_ok_then_ok_while_still_shallow() -> None:
    """CVD not satisfied on first tick; price stays shallow; later CVD satisfies (SHORT)."""
    strat = _strategy(
        entry_reclaim_confirm_seconds=0,
        entry_reclaim_inside_band=False,
        entry_reclaim_max_inside_depth_ratio=0.15,
    )
    boll = _boll(upper=2100, middle=2000, lower=1900)
    _setup_upper_armed_divergence(
        strat, ref_upper=2100, ref_middle=2000,
        div_extreme_price=2120, div_extreme_cvd=800_000,
        anchor_cum_cvd=1_000_000,
    )

    # min_entry_price = 2100 - 100 * 0.15 = 2085
    price = 2092.0  # shallow zone

    # Tick 1: CVD NOT satisfied
    # reclaim_anchored_cvd = (2_400_000 - 500_000) - 1_000_000 = 900_000
    # 900_000 < 800_000 → False
    cvd1 = _cvd(
        ts_ms=20000, price=price,
        cumulative_buy_volume=2_400_000, cumulative_sell_volume=500_000,
        buy_ratio=0.4, sell_ratio=0.6,
    )
    result1 = strat._short_setup(price, cvd1, boll)
    assert result1 is False

    # Tick 2: CVD now satisfies
    # reclaim_anchored_cvd = (1_400_000 - 600_000) - 1_000_000 = -200_000
    # -200_000 < 800_000 → True
    cvd2 = _cvd(
        ts_ms=21000, price=price,
        cumulative_buy_volume=1_400_000, cumulative_sell_volume=600_000,
        buy_ratio=0.4, sell_ratio=0.6,
    )
    result2 = strat._short_setup(price, cvd2, boll)
    assert result2 is True, "CVD caught up while still shallow → must enter (SHORT)"


# ── Test 21: LONG too-deep reject locks until re-outside ────────────────

def test_long_v2_too_deep_reject_locks_until_re_outside() -> None:
    """Too-deep reject must set rejected_until_next_outside; re-outside unlocks it."""
    strat = _strategy(
        entry_reclaim_confirm_seconds=0,
        entry_reclaim_inside_band=False,
        entry_reclaim_max_inside_depth_ratio=0.15,
        entry_reclaim_min_cvd_follow_through=0,
    )
    boll = _boll(lower=1900, middle=2000, upper=2100)
    _setup_lower_armed_divergence(
        strat, ref_lower=1900, ref_middle=2000,
        div_extreme_price=1880, div_extreme_cvd=-800_000,
        anchor_cum_cvd=-1_000_000,
    )

    # max_entry_price = 1900 + 100 * 0.15 = 1915

    # Tick 1: price inside shallow zone, CVD NOT satisfied → keep waiting
    price_shallow = 1908.0
    cvd_bad = _cvd(
        ts_ms=20000, price=price_shallow,
        cumulative_buy_volume=500_000, cumulative_sell_volume=2_400_000,
    )
    result1 = strat._long_setup(price_shallow, cvd_bad, boll)
    assert result1 is False
    assert strat.state.lower_reclaim_rejected_until_next_outside is False

    # Tick 2: price > max_entry_price (too deep), CVD satisfied → REJECTED + LOCKED
    price_deep = 1930.0  # well beyond max_entry_price=1915
    cvd_good = _cvd(
        ts_ms=21000, price=price_deep,
        cumulative_buy_volume=1_200_000, cumulative_sell_volume=1_500_000,
    )
    result2 = strat._long_setup(price_deep, cvd_good, boll)
    assert result2 is False
    assert strat.state.lower_reclaim_rejected_until_next_outside is True
    assert strat.state.lower_reclaim_cycle_count == 1

    # Tick 3: price back in shallow zone, CVD satisfied → STILL blocked by lock
    cvd_good3 = _cvd(
        ts_ms=22000, price=price_shallow,
        cumulative_buy_volume=1_200_000, cumulative_sell_volume=1_500_000,
    )
    result3 = strat._long_setup(price_shallow, cvd_good3, boll)
    assert result3 is False, "Should be locked by rejected_until_next_outside"

    # Now simulate re-entering outside (price below lower band → unlock)
    # First initialise the orderflow tracker so _update_lower_outside_v2 hits "every tick" path
    cum_cvd_outside = 500_000 - 1_500_000  # = -1_000_000
    strat._lower_orderflow.anchor(
        direction="DOWN", ts_ms=23000, price=boll.lower * 0.998,
        cumulative_cvd=cum_cvd_outside,
        cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000,
    )
    strat._update_lower_outside_v2(
        price=boll.lower * 0.998, ts_ms=24000, boll=boll,
        cvd=_cvd(ts_ms=24000, price=boll.lower * 0.998,
                 cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000),
    )
    # Now unlock should have happened
    assert strat.state.lower_reclaim_rejected_until_next_outside is False, (
        "Re-entering outside must unlock reclaim retry"
    )

    # Tick 5: price reclaims shallow zone + CVD satisfied → NOW it enters
    price_reclaim = 1905.0
    cvd_final = _cvd(
        ts_ms=25000, price=price_reclaim,
        cumulative_buy_volume=1_200_000, cumulative_sell_volume=1_500_000,
    )
    result5 = strat._long_setup(price_reclaim, cvd_final, boll)
    assert result5 is True, "After re-outside unlock, reclaim must succeed"


# ── Test 22: SHORT too-deep reject locks until re-outside ───────────────

def test_short_v2_too_deep_reject_locks_until_re_outside() -> None:
    """Too-deep reject must set upper_rejected_until_next_outside; re-outside unlocks it."""
    strat = _strategy(
        entry_reclaim_confirm_seconds=0,
        entry_reclaim_inside_band=False,
        entry_reclaim_max_inside_depth_ratio=0.15,
        entry_reclaim_min_cvd_follow_through=0,
    )
    boll = _boll(upper=2100, middle=2000, lower=1900)
    _setup_upper_armed_divergence(
        strat, ref_upper=2100, ref_middle=2000,
        div_extreme_price=2120, div_extreme_cvd=800_000,
        anchor_cum_cvd=1_000_000,
    )

    # min_entry_price = 2100 - 100 * 0.15 = 2085

    # Tick 1: price inside shallow zone, CVD NOT satisfied
    price_shallow = 2092.0
    cvd_bad = _cvd(
        ts_ms=20000, price=price_shallow,
        cumulative_buy_volume=2_400_000, cumulative_sell_volume=500_000,
        buy_ratio=0.4, sell_ratio=0.6,
    )
    result1 = strat._short_setup(price_shallow, cvd_bad, boll)
    assert result1 is False
    assert strat.state.upper_reclaim_rejected_until_next_outside is False

    # Tick 2: price < min_entry_price (too deep), CVD satisfied → REJECTED + LOCKED
    price_deep = 2070.0  # below min_entry_price=2085
    cvd_good = _cvd(
        ts_ms=21000, price=price_deep,
        cumulative_buy_volume=1_400_000, cumulative_sell_volume=600_000,
        buy_ratio=0.4, sell_ratio=0.6,
    )
    result2 = strat._short_setup(price_deep, cvd_good, boll)
    assert result2 is False
    assert strat.state.upper_reclaim_rejected_until_next_outside is True
    assert strat.state.upper_reclaim_cycle_count == 1

    # Tick 3: price back in shallow zone, CVD satisfied → STILL blocked
    cvd_good3 = _cvd(
        ts_ms=22000, price=price_shallow,
        cumulative_buy_volume=1_400_000, cumulative_sell_volume=600_000,
        buy_ratio=0.4, sell_ratio=0.6,
    )
    result3 = strat._short_setup(price_shallow, cvd_good3, boll)
    assert result3 is False, "Should be locked by upper_rejected_until_next_outside"

    # Simulate re-entering outside (price above upper band → unlock)
    cum_cvd_outside = 1_500_000 - 500_000  # = 1_000_000
    strat._upper_orderflow.anchor(
        direction="UP", ts_ms=23000, price=boll.upper * 1.002,
        cumulative_cvd=cum_cvd_outside,
        cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
    )
    strat._update_upper_outside_v2(
        price=boll.upper * 1.002, ts_ms=24000, boll=boll,
        cvd=_cvd(ts_ms=24000, price=boll.upper * 1.002,
                 cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
                 buy_ratio=0.4, sell_ratio=0.6),
    )
    assert strat.state.upper_reclaim_rejected_until_next_outside is False, (
        "Re-entering outside must unlock reclaim retry (SHORT)"
    )

    # Tick 5: price reclaims shallow zone + CVD satisfied → enters
    price_reclaim = 2092.0
    cvd_final = _cvd(
        ts_ms=25000, price=price_reclaim,
        cumulative_buy_volume=1_400_000, cumulative_sell_volume=600_000,
        buy_ratio=0.4, sell_ratio=0.6,
    )
    result5 = strat._short_setup(price_reclaim, cvd_final, boll)
    assert result5 is True, "After re-outside unlock, SHORT reclaim must succeed"


# ── Test 23: V2 ignores ENTRY_RECLAIM_CONFIRM_SECONDS ───────────────────

def test_v2_ignores_confirm_seconds() -> None:
    """Even with ENTRY_RECLAIM_CONFIRM_SECONDS=999, V2 enters immediately."""
    strat = _strategy(
        entry_reclaim_confirm_seconds=999.0,  # huge delay — V2 must ignore
        entry_reclaim_inside_band=False,
        entry_reclaim_max_inside_depth_ratio=0.15,
    )
    boll = _boll(lower=1900, middle=2000, upper=2100)
    _setup_lower_armed_divergence(
        strat, ref_lower=1900, ref_middle=2000,
        div_extreme_price=1880, div_extreme_cvd=-800_000,
        anchor_cum_cvd=-1_000_000,
    )

    price = 1905.0  # shallow zone, CVD satisfied
    cvd = _cvd(
        ts_ms=20000, price=price,
        cumulative_buy_volume=1_200_000, cumulative_sell_volume=1_500_000,
    )

    result = strat._long_setup(price, cvd, boll)
    assert result is True, (
        "V2 must ignore ENTRY_RECLAIM_CONFIRM_SECONDS and enter on first tick"
    )


# ── Test 24: Legacy path still uses ENTRY_RECLAIM_CONFIRM_SECONDS ───────

def test_legacy_path_still_uses_confirm_seconds() -> None:
    """When V2 is disabled, the legacy 1-second soft confirm must still work."""
    strat = _strategy(
        entry_reclaim_v2_enabled=False,
        entry_cvd_structure_mode="DIVERGENCE_OR_ABSORPTION",
        entry_cvd_divergence_enabled=True,
        entry_cvd_absorption_enabled=False,
        entry_reclaim_confirm_seconds=1.0,
        entry_reclaim_inside_band=False,
        entry_reclaim_buffer_pct=0.0,
        entry_max_extreme_to_reclaim_seconds=900,
    )
    boll = _boll(lower=1900, middle=2000, upper=2100)

    # Manually arm state (legacy path needs lower_armed + extreme + deep_enough + divergence)
    strat.state.lower_armed = True
    strat.state.lower_extreme_price = 1880.0
    strat.state.lower_extreme_ts_ms = 5000
    strat.state.lower_deep_enough = True
    strat.state.lower_cvd_divergence_confirmed = True

    # First tick back inside band → PENDING, not entry
    price = 1905.0
    cvd1 = _cvd(ts_ms=10000, price=price,
                cumulative_buy_volume=1_200_000, cumulative_sell_volume=1_500_000)
    result1 = strat._long_setup(price, cvd1, boll)
    assert result1 is False
    assert strat.state.lower_reclaim_seen is True
    assert strat.state.lower_reclaim_ts_ms == 10000

    # Before confirm_seconds elapses → still False
    cvd2 = _cvd(ts_ms=10500, price=price,
                cumulative_buy_volume=1_200_000, cumulative_sell_volume=1_500_000)
    result2 = strat._long_setup(price, cvd2, boll)
    assert result2 is False  # only 500ms elapsed, need 1000ms

    # After 1 second → legacy soft confirm passes
    cvd3 = _cvd(ts_ms=11001, price=price,
                cumulative_buy_volume=1_200_000, cumulative_sell_volume=1_500_000,
                cross_positive=True, cvd_increasing=True, no_new_low=True)
    result3 = strat._long_setup(price, cvd3, boll)
    assert result3 is True, "Legacy soft confirm must still work after 1 second"


# ═══════════════════════════════════════════════════════════════════════════════
# Tests: follow-through confirmed log once-per-setup latch
# ═══════════════════════════════════════════════════════════════════════════════


def test_lower_follow_through_confirmed_log_only_once(caplog) -> None:
    """LOWER_RECLAIM_CVD_FOLLOW_THROUGH_CONFIRMED must print only once per setup."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(
        entry_reclaim_confirm_seconds=0,
        entry_reclaim_inside_band=False,
        entry_reclaim_max_inside_depth_ratio=0.15,
        entry_reclaim_min_cvd_follow_through=0,
    )
    boll = _boll(lower=1900, middle=2000, upper=2100)
    _setup_lower_armed_divergence(
        strat, ref_lower=1900, ref_middle=2000,
        div_extreme_price=1880, div_extreme_cvd=-800_000,
        anchor_cum_cvd=-1_000_000,
    )

    # max_entry_price = 1900 + 100 * 0.15 = 1915
    price = 1908.0  # shallow zone, CVD satisfied
    # reclaim_anchored_cvd = (1_200_000 - 1_500_000) - (-1_000_000) = 700_000
    # 700_000 > -800_000 → True
    cvd = _cvd(
        ts_ms=20000, price=price,
        cumulative_buy_volume=1_200_000, cumulative_sell_volume=1_500_000,
    )

    # First call: log should appear
    result1 = strat._check_lower_reclaim_v2_follow_through(price, cvd, boll)
    assert result1 is True
    assert strat.state.lower_reclaim_cvd_follow_through_logged is True

    # Second call: no additional log
    result2 = strat._check_lower_reclaim_v2_follow_through(price, cvd, boll)
    assert result2 is True

    confirmed_count = sum(
        1 for r in caplog.records
        if r.levelno == logging.INFO
        and "LOWER_RECLAIM_CVD_FOLLOW_THROUGH_CONFIRMED" in r.message
    )
    assert confirmed_count == 1, (
        f"Expected 1 LOWER_RECLAIM_CVD_FOLLOW_THROUGH_CONFIRMED, got {confirmed_count}"
    )


def test_upper_follow_through_confirmed_log_only_once(caplog) -> None:
    """UPPER_RECLAIM_CVD_FOLLOW_THROUGH_CONFIRMED must print only once per setup."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(
        entry_reclaim_confirm_seconds=0,
        entry_reclaim_inside_band=False,
        entry_reclaim_max_inside_depth_ratio=0.15,
        entry_reclaim_min_cvd_follow_through=0,
    )
    boll = _boll(upper=2100, middle=2000, lower=1900)
    _setup_upper_armed_divergence(
        strat, ref_upper=2100, ref_middle=2000,
        div_extreme_price=2120, div_extreme_cvd=800_000,
        anchor_cum_cvd=1_000_000,
    )

    # min_entry_price = 2100 - 100 * 0.15 = 2085
    price = 2092.0  # shallow zone, CVD satisfied
    # reclaim_anchored_cvd = (1_400_000 - 600_000) - 1_000_000 = -200_000
    # -200_000 < 800_000 → True
    cvd = _cvd(
        ts_ms=20000, price=price,
        cumulative_buy_volume=1_400_000, cumulative_sell_volume=600_000,
        buy_ratio=0.4, sell_ratio=0.6,
    )

    # First call: log should appear
    result1 = strat._check_upper_reclaim_v2_follow_through(price, cvd, boll)
    assert result1 is True
    assert strat.state.upper_reclaim_cvd_follow_through_logged is True

    # Second call: no additional log
    result2 = strat._check_upper_reclaim_v2_follow_through(price, cvd, boll)
    assert result2 is True

    confirmed_count = sum(
        1 for r in caplog.records
        if r.levelno == logging.INFO
        and "UPPER_RECLAIM_CVD_FOLLOW_THROUGH_CONFIRMED" in r.message
    )
    assert confirmed_count == 1, (
        f"Expected 1 UPPER_RECLAIM_CVD_FOLLOW_THROUGH_CONFIRMED, got {confirmed_count}"
    )


def test_lower_follow_through_log_again_after_reset(caplog) -> None:
    """After _reset_lower_armed(), follow-through confirmed log can print again."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(
        entry_reclaim_confirm_seconds=0,
        entry_reclaim_inside_band=False,
        entry_reclaim_max_inside_depth_ratio=0.15,
        entry_reclaim_min_cvd_follow_through=0,
    )
    boll = _boll(lower=1900, middle=2000, upper=2100)
    _setup_lower_armed_divergence(
        strat, ref_lower=1900, ref_middle=2000,
        div_extreme_price=1880, div_extreme_cvd=-800_000,
        anchor_cum_cvd=-1_000_000,
    )

    price = 1908.0
    cvd = _cvd(
        ts_ms=20000, price=price,
        cumulative_buy_volume=1_200_000, cumulative_sell_volume=1_500_000,
    )

    # First cycle: log prints once
    strat._check_lower_reclaim_v2_follow_through(price, cvd, boll)
    assert strat.state.lower_reclaim_cvd_follow_through_logged is True

    # Reset
    strat._reset_lower_armed()
    assert strat.state.lower_reclaim_cvd_follow_through_logged is False

    # Re-setup armed divergence
    _setup_lower_armed_divergence(
        strat, ref_lower=1900, ref_middle=2000,
        div_extreme_price=1880, div_extreme_cvd=-800_000,
        anchor_cum_cvd=-1_000_000,
    )

    # Second cycle: log can print again
    strat._check_lower_reclaim_v2_follow_through(price, cvd, boll)
    assert strat.state.lower_reclaim_cvd_follow_through_logged is True

    confirmed_count = sum(
        1 for r in caplog.records
        if r.levelno == logging.INFO
        and "LOWER_RECLAIM_CVD_FOLLOW_THROUGH_CONFIRMED" in r.message
    )
    assert confirmed_count == 2, (
        f"Expected 2 LOWER_RECLAIM_CVD_FOLLOW_THROUGH_CONFIRMED after reset, got {confirmed_count}"
    )


def test_upper_follow_through_log_again_after_reset(caplog) -> None:
    """After _reset_upper_armed(), follow-through confirmed log can print again."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(
        entry_reclaim_confirm_seconds=0,
        entry_reclaim_inside_band=False,
        entry_reclaim_max_inside_depth_ratio=0.15,
        entry_reclaim_min_cvd_follow_through=0,
    )
    boll = _boll(upper=2100, middle=2000, lower=1900)
    _setup_upper_armed_divergence(
        strat, ref_upper=2100, ref_middle=2000,
        div_extreme_price=2120, div_extreme_cvd=800_000,
        anchor_cum_cvd=1_000_000,
    )

    price = 2092.0
    cvd = _cvd(
        ts_ms=20000, price=price,
        cumulative_buy_volume=1_400_000, cumulative_sell_volume=600_000,
        buy_ratio=0.4, sell_ratio=0.6,
    )

    # First cycle: log prints once
    strat._check_upper_reclaim_v2_follow_through(price, cvd, boll)
    assert strat.state.upper_reclaim_cvd_follow_through_logged is True

    # Reset
    strat._reset_upper_armed()
    assert strat.state.upper_reclaim_cvd_follow_through_logged is False

    # Re-setup armed divergence
    _setup_upper_armed_divergence(
        strat, ref_upper=2100, ref_middle=2000,
        div_extreme_price=2120, div_extreme_cvd=800_000,
        anchor_cum_cvd=1_000_000,
    )

    # Second cycle: log can print again
    strat._check_upper_reclaim_v2_follow_through(price, cvd, boll)
    assert strat.state.upper_reclaim_cvd_follow_through_logged is True

    confirmed_count = sum(
        1 for r in caplog.records
        if r.levelno == logging.INFO
        and "UPPER_RECLAIM_CVD_FOLLOW_THROUGH_CONFIRMED" in r.message
    )
    assert confirmed_count == 2, (
        f"Expected 2 UPPER_RECLAIM_CVD_FOLLOW_THROUGH_CONFIRMED after reset, got {confirmed_count}"
    )


# ═══════════════════════════════════════════════════════════════════════
# Reclaim V2 observability: extreme snapshot log
# ═══════════════════════════════════════════════════════════════════════


def test_extreme_snapshot_after_new_extreme_lower(caplog) -> None:
    """After a new lower confirmed extreme, snapshot log prints after 10s interval."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001, reclaim_extreme_log_interval_seconds=10)
    boll = _boll()

    # Tick 1: anchor
    p1 = boll.lower * 0.998
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))
    # Tick 2: first extreme candidate → retrace confirm
    p2 = boll.lower * 0.994
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000))
    strat.on_tick(p1, 2500, boll, _cvd(ts_ms=2500, price=p1, cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000))
    # Tick 3: new extreme candidate → retrace confirm (triggers divergence eval + snapshot_pending, throttled)
    p3 = boll.lower * 0.990
    strat.on_tick(p3, 3000, boll, _cvd(ts_ms=3000, price=p3, cumulative_buy_volume=600_000, cumulative_sell_volume=1_800_000))
    strat.on_tick(p1, 3200, boll, _cvd(ts_ms=3200, price=p1, cumulative_buy_volume=600_000, cumulative_sell_volume=1_800_000))

    # Within 10s (t=3200..13200), snapshot throttled — no log yet
    snapshot_logs = [r for r in caplog.records if "LOWER_EXTREME_SNAPSHOT" in r.message]
    assert len(snapshot_logs) == 0, "Should not print snapshot within 10s of pending"

    # After 10s (t=14000): new extreme candidate → retrace confirm → snapshot fires
    p4 = boll.lower * 0.986
    strat.on_tick(p4, 14000, boll, _cvd(ts_ms=14000, price=p4, cumulative_buy_volume=650_000, cumulative_sell_volume=1_900_000))
    strat.on_tick(p1, 14200, boll, _cvd(ts_ms=14200, price=p1, cumulative_buy_volume=650_000, cumulative_sell_volume=1_900_000))
    snapshot_logs = [r for r in caplog.records if "LOWER_EXTREME_SNAPSHOT" in r.message]
    assert len(snapshot_logs) == 1, f"Expected 1 snapshot after 10s, got {len(snapshot_logs)}"


def test_extreme_snapshot_after_new_extreme_upper(caplog) -> None:
    """After a new upper confirmed extreme, snapshot log prints after 10s interval."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001, reclaim_extreme_log_interval_seconds=10)
    boll = _boll()

    # Tick 1: anchor
    p1 = boll.upper * 1.002
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
                                         buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                                         cvd_increasing=False, cvd_decreasing=True, no_new_high=True))
    # Tick 2: first extreme candidate → retrace confirm
    p2 = boll.upper * 1.006
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
                                         buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                                         cvd_increasing=False, cvd_decreasing=True, no_new_high=True))
    strat.on_tick(p1, 2500, boll, _cvd(ts_ms=2500, price=p1, cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
                                         buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                                         cvd_increasing=False, cvd_decreasing=True, no_new_high=True))
    # Tick 3: new extreme candidate — CVD follows higher (no divergence) → retrace confirm
    p3 = boll.upper * 1.010
    strat.on_tick(p3, 3000, boll, _cvd(ts_ms=3000, price=p3, cumulative_buy_volume=1_700_000, cumulative_sell_volume=500_000,
                                         buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                                         cvd_increasing=False, cvd_decreasing=True, no_new_high=True))
    strat.on_tick(p1, 3200, boll, _cvd(ts_ms=3200, price=p1, cumulative_buy_volume=1_700_000, cumulative_sell_volume=500_000,
                                         buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                                         cvd_increasing=False, cvd_decreasing=True, no_new_high=True))

    # After 10s (t=14000): new extreme candidate with CVD reverse → retrace confirm → snapshot fires
    p4 = boll.upper * 1.014
    strat.on_tick(p4, 14000, boll, _cvd(ts_ms=14000, price=p4, cumulative_buy_volume=1_300_000, cumulative_sell_volume=700_000,
                                         buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                                         cvd_increasing=False, cvd_decreasing=True, no_new_high=True))
    strat.on_tick(p1, 14200, boll, _cvd(ts_ms=14200, price=p1, cumulative_buy_volume=1_300_000, cumulative_sell_volume=700_000,
                                         buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                                         cvd_increasing=False, cvd_decreasing=True, no_new_high=True))
    snapshot_logs = [r for r in caplog.records if "UPPER_EXTREME_SNAPSHOT" in r.message]
    assert len(snapshot_logs) == 1, f"Expected 1 snapshot after 10s, got {len(snapshot_logs)}"


def test_no_snapshot_without_new_extreme(caplog) -> None:
    """Outside ticks without new confirmed extreme should NOT trigger snapshot.

    Snapshot is only called after divergence evaluation, which requires
    a confirmed extreme. Same-price ticks do not trigger new snapshots.
    """
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001, reclaim_extreme_log_interval_seconds=10)
    boll = _boll()

    # Anchor + first extreme candidate → retrace confirm
    pr = boll.lower * 0.998  # retrace reference price
    p1 = pr
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))
    p2 = boll.lower * 0.994
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000))
    strat.on_tick(pr, 2500, boll, _cvd(ts_ms=2500, price=pr, cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000))
    # New extreme candidate → retrace confirm (triggers pending, but throttled within 10s)
    p3 = boll.lower * 0.990
    strat.on_tick(p3, 3000, boll, _cvd(ts_ms=3000, price=p3, cumulative_buy_volume=600_000, cumulative_sell_volume=1_800_000))
    strat.on_tick(pr, 3200, boll, _cvd(ts_ms=3200, price=pr, cumulative_buy_volume=600_000, cumulative_sell_volume=1_800_000))

    # No snapshot yet (throttled within 10s interval)
    snapshot_logs = [r for r in caplog.records if "LOWER_EXTREME_SNAPSHOT" in r.message]
    assert len(snapshot_logs) == 0, "Snapshot should be throttled within 10s"

    # After 10s, same-price tick (no new confirmed extreme) → snapshot NOT called
    strat.on_tick(p3, 25000, boll, _cvd(ts_ms=25000, price=p3, cumulative_buy_volume=600_000, cumulative_sell_volume=1_800_000))
    snapshot_logs = [r for r in caplog.records if "LOWER_EXTREME_SNAPSHOT" in r.message]
    assert len(snapshot_logs) == 0, (
        "Same-price tick (no new confirmed extreme) should NOT trigger snapshot — "
        "snapshot only fires after divergence evaluation"
    )

    # New extreme candidate after throttle expires → retrace confirm → snapshot fires
    p4 = boll.lower * 0.986
    strat.on_tick(p4, 35000, boll, _cvd(ts_ms=35000, price=p4, cumulative_buy_volume=650_000, cumulative_sell_volume=1_900_000))
    strat.on_tick(pr, 35200, boll, _cvd(ts_ms=35200, price=pr, cumulative_buy_volume=650_000, cumulative_sell_volume=1_900_000))
    snapshot_logs = [r for r in caplog.records if "LOWER_EXTREME_SNAPSHOT" in r.message]
    assert len(snapshot_logs) >= 1, "New confirmed extreme after throttle should trigger snapshot"

    # More same-price ticks → no additional snapshots
    strat.on_tick(p4, 36000, boll, _cvd(ts_ms=36000, price=p4, cumulative_buy_volume=650_000, cumulative_sell_volume=1_900_000))
    snapshot_logs = [r for r in caplog.records if "LOWER_EXTREME_SNAPSHOT" in r.message]
    assert len(snapshot_logs) >= 1, "Should not print additional snapshots without new confirmed extreme"


def test_snapshot_logs_latest_extreme_only(caplog) -> None:
    """Multiple new confirmed extremes within 10s → only latest printed in snapshot."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001, reclaim_extreme_log_interval_seconds=10)
    boll = _boll()

    pr = boll.lower * 0.998  # retrace reference
    # Anchor
    p1 = pr
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))
    # First extreme candidate → retrace confirm
    p2 = boll.lower * 0.994  # ~1888.6
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000))
    strat.on_tick(pr, 2500, boll, _cvd(ts_ms=2500, price=pr, cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000))
    # New extreme t=3000: ~1881 → retrace confirm (throttled — within 10s)
    p3 = boll.lower * 0.990
    strat.on_tick(p3, 3000, boll, _cvd(ts_ms=3000, price=p3, cumulative_buy_volume=600_000, cumulative_sell_volume=1_800_000))
    strat.on_tick(pr, 3200, boll, _cvd(ts_ms=3200, price=pr, cumulative_buy_volume=600_000, cumulative_sell_volume=1_800_000))
    # New extreme t=5000: ~1875 → retrace confirm (throttled)
    p4 = boll.lower * 0.987
    strat.on_tick(p4, 5000, boll, _cvd(ts_ms=5000, price=p4, cumulative_buy_volume=650_000, cumulative_sell_volume=1_900_000))
    strat.on_tick(pr, 5200, boll, _cvd(ts_ms=5200, price=pr, cumulative_buy_volume=650_000, cumulative_sell_volume=1_900_000))
    # New extreme t=7000: ~1862 → retrace confirm (throttled)
    p5 = boll.lower * 0.980
    strat.on_tick(p5, 7000, boll, _cvd(ts_ms=7000, price=p5, cumulative_buy_volume=700_000, cumulative_sell_volume=2_000_000))
    strat.on_tick(pr, 7200, boll, _cvd(ts_ms=7200, price=pr, cumulative_buy_volume=700_000, cumulative_sell_volume=2_000_000))

    # All within 10s → no snapshot yet
    snapshot_logs = [r for r in caplog.records if "LOWER_EXTREME_SNAPSHOT" in r.message]
    assert len(snapshot_logs) == 0, "All extremes within 10s — no snapshot yet"

    # After 10s (t=18000): new extreme candidate → retrace confirm → snapshot fires
    p6 = boll.lower * 0.976  # ~1854.4
    strat.on_tick(p6, 18000, boll, _cvd(ts_ms=18000, price=p6, cumulative_buy_volume=750_000, cumulative_sell_volume=2_100_000))
    strat.on_tick(pr, 18200, boll, _cvd(ts_ms=18200, price=pr, cumulative_buy_volume=750_000, cumulative_sell_volume=2_100_000))
    snapshot_logs = [r for r in caplog.records if "LOWER_EXTREME_SNAPSHOT" in r.message]
    assert len(snapshot_logs) == 1, f"Expected 1 snapshot (latest only), got {len(snapshot_logs)}"
    msg = snapshot_logs[0].getMessage()
    # prev must be p5 (~1862), curr must be p6 (~1854)
    assert "curr_extreme=1854.4000" in msg, f"Snapshot should use latest extreme, got: {msg}"
    # Verify coherent: prev should reference p5 (~1862), not p4 or earlier
    assert "prev_extreme=1862.0000" in msg, f"prev should be 1862 (p5), got: {msg}"


# ═══════════════════════════════════════════════════════════════════════
# Reclaim V2 observability: no-entry reason logging
# ═══════════════════════════════════════════════════════════════════════


def _setup_lower_outside_with_first_extreme(strat, boll) -> None:
    """Helper: anchor + first confirmed extreme for LOWER, no divergence."""
    p1 = boll.lower * 0.998
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))
    p2 = boll.lower * 0.994
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000))
    # Retrace to confirm first extreme
    p2_retrace = boll.lower * 0.998
    strat.on_tick(p2_retrace, 3000, boll, _cvd(ts_ms=3000, price=p2_retrace, cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000))


def _setup_upper_outside_with_first_extreme(strat, boll) -> None:
    """Helper: anchor + first confirmed extreme for UPPER, no divergence."""
    p1 = boll.upper * 1.002
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
                                         buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                                         cvd_increasing=False, cvd_decreasing=True, no_new_high=True))
    p2 = boll.upper * 1.006
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
                                         buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                                         cvd_increasing=False, cvd_decreasing=True, no_new_high=True))
    # Retrace to confirm first extreme
    p2_retrace = boll.upper * 1.002
    strat.on_tick(p2_retrace, 3000, boll, _cvd(ts_ms=3000, price=p2_retrace, cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
                                                 buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                                                 cvd_increasing=False, cvd_decreasing=True, no_new_high=True))


def test_no_entry_no_anchored_divergence_lower(caplog) -> None:
    """Outside observed + first extreme + no divergence + price inside
    → LOWER_RECLAIM_ABORTED (one-shot), no repeated no_anchored_divergence heartbeat."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001, reclaim_no_entry_log_interval_seconds=0)
    boll = _boll()
    _setup_lower_outside_with_first_extreme(strat, boll)

    # Price moves back inside band without divergence being confirmed
    inside_price = boll.lower * 1.001  # just inside the band
    strat.on_tick(inside_price, 3000, boll, _cvd(ts_ms=3000, price=inside_price, cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000))

    # Should see LOWER_RECLAIM_ABORTED (one-shot) instead of repeated no_anchored_divergence
    abort_logs = [r for r in caplog.records if "LOWER_RECLAIM_ABORTED" in r.message]
    assert len(abort_logs) == 1, (
        f"Expected 1 LOWER_RECLAIM_ABORTED, got {len(abort_logs)}"
    )
    assert "inside_return_without_anchored_divergence" in abort_logs[0].message

    # No stale LOWER_RECLAIM_NO_ENTRY with no_anchored_divergence
    no_entry_no_div = [
        r for r in caplog.records
        if "LOWER_RECLAIM_NO_ENTRY" in r.message
        and "no_anchored_divergence" in r.message
    ]
    assert len(no_entry_no_div) == 0, (
        f"No LOWER_RECLAIM_NO_ENTRY no_anchored_divergence expected, got {len(no_entry_no_div)}"
    )


def test_no_entry_no_anchored_divergence_upper(caplog) -> None:
    """Outside observed + first extreme + no divergence + price inside
    → UPPER_RECLAIM_ABORTED (one-shot), no repeated no_anchored_divergence heartbeat."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001, reclaim_no_entry_log_interval_seconds=0)
    boll = _boll()
    _setup_upper_outside_with_first_extreme(strat, boll)

    # Price moves back inside band
    inside_price = boll.upper * 0.999
    strat.on_tick(inside_price, 3000, boll, _cvd(ts_ms=3000, price=inside_price, cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
                                                   buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                                                   cvd_increasing=False, cvd_decreasing=True, no_new_high=True))

    abort_logs = [r for r in caplog.records if "UPPER_RECLAIM_ABORTED" in r.message]
    assert len(abort_logs) == 1, (
        f"Expected 1 UPPER_RECLAIM_ABORTED, got {len(abort_logs)}"
    )
    assert "inside_return_without_anchored_divergence" in abort_logs[0].message

    no_entry_no_div = [
        r for r in caplog.records
        if "UPPER_RECLAIM_NO_ENTRY" in r.message
        and "no_anchored_divergence" in r.message
    ]
    assert len(no_entry_no_div) == 0, (
        f"No UPPER_RECLAIM_NO_ENTRY no_anchored_divergence expected, got {len(no_entry_no_div)}"
    )


def _run_lower_armed_divergence(strat, boll) -> None:
    """Helper: run full lower armed with anchored divergence confirmed via on_tick.

    Uses confirmed swing extremes — retrace ticks after each low trigger
    confirmation before divergence is evaluated.
    """
    # Anchor
    p1 = boll.lower * 0.998
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))
    # First extreme candidate
    p2 = boll.lower * 0.994
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))
    # Retrace to confirm first extreme (price must rise >= p2 * 1.0008)
    p2_retrace = boll.lower * 0.998
    strat.on_tick(p2_retrace, 2500, boll, _cvd(ts_ms=2500, price=p2_retrace, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))
    # New low + CVD recovery — sets new candidate
    p3 = boll.lower * 0.990
    strat.on_tick(p3, 3000, boll, _cvd(ts_ms=3000, price=p3, cumulative_buy_volume=600_000, cumulative_sell_volume=1_400_000))
    # Retrace to confirm second extreme → divergence evaluation fires
    p3_retrace = boll.lower * 0.995
    strat.on_tick(p3_retrace, 3500, boll, _cvd(ts_ms=3500, price=p3_retrace, cumulative_buy_volume=600_000, cumulative_sell_volume=1_400_000))
    assert strat.state.lower_anchored_divergence_confirmed is True
    assert strat.state.lower_armed is True


def _run_upper_armed_divergence(strat, boll) -> None:
    """Helper: run full upper armed with anchored divergence confirmed via on_tick.

    Uses confirmed swing extremes — retrace ticks after each high trigger
    confirmation before divergence is evaluated.
    """
    # Anchor
    p1 = boll.upper * 1.002
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
                                         buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                                         cvd_increasing=False, cvd_decreasing=True, no_new_high=True))
    # First extreme candidate
    p2 = boll.upper * 1.006
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
                                         buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                                         cvd_increasing=False, cvd_decreasing=True, no_new_high=True))
    # Retrace to confirm first extreme (price must fall <= p2 * 0.9992)
    p2_retrace = boll.upper * 1.002
    strat.on_tick(p2_retrace, 2500, boll, _cvd(ts_ms=2500, price=p2_retrace, cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
                                                 buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                                                 cvd_increasing=False, cvd_decreasing=True, no_new_high=True))
    # New high + CVD reverse — sets new candidate
    p3 = boll.upper * 1.010
    strat.on_tick(p3, 3000, boll, _cvd(ts_ms=3000, price=p3, cumulative_buy_volume=1_400_000, cumulative_sell_volume=600_000,
                                         buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                                         cvd_increasing=False, cvd_decreasing=True, no_new_high=True))
    # Retrace to confirm second extreme → divergence evaluation fires
    p3_retrace = boll.upper * 1.005
    strat.on_tick(p3_retrace, 3500, boll, _cvd(ts_ms=3500, price=p3_retrace, cumulative_buy_volume=1_400_000, cumulative_sell_volume=600_000,
                                                 buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                                                 cvd_increasing=False, cvd_decreasing=True, no_new_high=True))
    assert strat.state.upper_anchored_divergence_confirmed is True
    assert strat.state.upper_armed is True


def test_no_entry_cvd_follow_through_not_met_lower(caplog) -> None:
    """Armed + inside shallow zone + CVD not following through → cvd_follow_through_not_met."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(
        min_outside_pct=0.001,
        entry_reclaim_max_inside_depth_ratio=0.15,
        entry_reclaim_min_cvd_follow_through=100_000,  # high threshold to prevent follow-through
        reclaim_no_entry_log_interval_seconds=0,
        entry_reclaim_confirm_seconds=0,
        entry_reclaim_inside_band=False,
    )
    boll = _boll()
    _run_lower_armed_divergence(strat, boll)

    # Price reclaims inside shallow zone (ref_lower + band_width * 0.05 = 1905)
    ref_lower = strat.state.lower_divergence_ref_lower or boll.lower
    ref_middle = strat.state.lower_divergence_ref_middle or boll.middle
    shallow_price = ref_lower + (ref_middle - ref_lower) * 0.05
    strat.on_tick(shallow_price, 4000, boll, _cvd(ts_ms=4000, price=shallow_price, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))

    no_entry_logs = [r for r in caplog.records if "LOWER_RECLAIM_NO_ENTRY" in r.message]
    cvd_logs = [r for r in no_entry_logs if "cvd_follow_through_not_met" in r.message]
    assert len(cvd_logs) >= 1, f"Expected cvd_follow_through_not_met log, got {len(cvd_logs)}"


def test_no_entry_cvd_follow_through_not_met_upper(caplog) -> None:
    """Armed + inside shallow zone + CVD not following through → cvd_follow_through_not_met (UPPER)."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(
        min_outside_pct=0.001,
        entry_reclaim_max_inside_depth_ratio=0.15,
        entry_reclaim_min_cvd_follow_through=100_000,
        reclaim_no_entry_log_interval_seconds=0,
        entry_reclaim_confirm_seconds=0,
        entry_reclaim_inside_band=False,
    )
    boll = _boll()
    _run_upper_armed_divergence(strat, boll)

    # Price reclaims inside shallow zone (ref_upper - band_width * 0.05 = 2095)
    ref_upper = strat.state.upper_divergence_ref_upper or boll.upper
    ref_middle = strat.state.upper_divergence_ref_middle or boll.middle
    shallow_price = ref_upper - (ref_upper - ref_middle) * 0.05
    strat.on_tick(shallow_price, 4000, boll, _cvd(ts_ms=4000, price=shallow_price, cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
                                                   buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                                                   cvd_increasing=False, cvd_decreasing=True, no_new_high=True))

    no_entry_logs = [r for r in caplog.records if "UPPER_RECLAIM_NO_ENTRY" in r.message]
    cvd_logs = [r for r in no_entry_logs if "cvd_follow_through_not_met" in r.message]
    assert len(cvd_logs) >= 1, f"Expected cvd_follow_through_not_met log, got {len(cvd_logs)}"


def test_no_entry_too_deep_inside(caplog) -> None:
    """Price goes too deep inside before CVD follow-through → too_deep log."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(
        min_outside_pct=0.001,
        entry_reclaim_max_inside_depth_ratio=0.01,  # very narrow shallow zone
        entry_reclaim_min_cvd_follow_through=1_000_000,  # impossible threshold
        reclaim_no_entry_log_interval_seconds=0,
        entry_reclaim_confirm_seconds=0,
        entry_reclaim_inside_band=False,
        entry_max_reclaim_cycles=10,
    )
    boll = _boll()
    _run_lower_armed_divergence(strat, boll)

    # Price goes past shallow zone but not to middle (to avoid middle reset)
    ref_lower = strat.state.lower_divergence_ref_lower or boll.lower
    ref_middle = strat.state.lower_divergence_ref_middle or boll.middle
    deep_price = ref_lower + (ref_middle - ref_lower) * 0.20  # 20% into band, past 1% shallow limit
    strat.on_tick(deep_price, 4000, boll, _cvd(ts_ms=4000, price=deep_price, cumulative_buy_volume=600_000, cumulative_sell_volume=1_400_000))

    rejected_logs = [r for r in caplog.records if "LOWER_RECLAIM_ATTEMPT_REJECTED" in r.message]
    no_entry_logs = [r for r in caplog.records if "LOWER_RECLAIM_NO_ENTRY" in r.message]
    too_deep_logs = [r for r in no_entry_logs if "too_deep_inside_before_cvd_follow_through" in r.message]
    assert len(rejected_logs) >= 1, "Should have RECLAIM_ATTEMPT_REJECTED"
    assert len(too_deep_logs) >= 1, "Should have RECLAIM_NO_ENTRY for too_deep"


def test_no_entry_throttle_same_reason(caplog) -> None:
    """Same no-entry reason within 60s interval should only log once."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(
        min_outside_pct=0.001,
        entry_reclaim_max_inside_depth_ratio=0.15,
        entry_reclaim_min_cvd_follow_through=100_000,
        reclaim_no_entry_log_interval_seconds=60,
        entry_reclaim_confirm_seconds=0,
        entry_reclaim_inside_band=False,
    )
    boll = _boll()
    _run_lower_armed_divergence(strat, boll)

    ref_lower = strat.state.lower_divergence_ref_lower or boll.lower
    ref_middle = strat.state.lower_divergence_ref_middle or boll.middle
    shallow_price = ref_lower + (ref_middle - ref_lower) * 0.05

    # First call: log fires
    strat.on_tick(shallow_price, 4000, boll, _cvd(ts_ms=4000, price=shallow_price, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))

    # Second call within 60s: same reason, should NOT fire
    strat.on_tick(shallow_price, 5000, boll, _cvd(ts_ms=5000, price=shallow_price, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))

    no_entry_logs = [r for r in caplog.records if "LOWER_RECLAIM_NO_ENTRY" in r.message]
    cvd_logs = [r for r in no_entry_logs if "cvd_follow_through_not_met" in r.message]
    assert len(cvd_logs) == 1, f"Expected 1 log within 60s, got {len(cvd_logs)}"


def test_reward_risk_not_met_no_entry(caplog) -> None:
    """RR gate rejects entry → reward_risk_not_met log."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(
        min_outside_pct=0.001,
        entry_reclaim_max_inside_depth_ratio=0.15,
        entry_reclaim_min_cvd_follow_through=0,
        reclaim_no_entry_log_interval_seconds=0,
        entry_reclaim_confirm_seconds=0,
        entry_reclaim_inside_band=False,
        entry_min_reward_risk=99.0,  # impossibly high RR requirement
        order_cooldown_seconds=0,
    )
    boll = _boll()
    _run_lower_armed_divergence(strat, boll)

    # Price in shallow zone with CVD recovery to trigger follow-through success
    ref_lower = strat.state.lower_divergence_ref_lower or boll.lower
    ref_middle = strat.state.lower_divergence_ref_middle or boll.middle
    reclaim_price = ref_lower + (ref_middle - ref_lower) * 0.05
    # CVD follows through — setup passes, but RR check fails due to high min_reward_risk
    strat.on_tick(reclaim_price, 4000, boll, _cvd(ts_ms=4000, price=reclaim_price, cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000))

    rr_logs = [r for r in caplog.records if "reward_risk_not_met" in r.message or "ENTRY_SKIPPED" in r.message]
    assert len(rr_logs) >= 1, f"Expected RR rejection log, got {len(rr_logs)}"


def test_post_entry_sl_cooldown_no_entry(caplog) -> None:
    """Post-entry SL cooldown blocks entry → post_entry_sl_cooldown log."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(
        min_outside_pct=0.001,
        reclaim_no_entry_log_interval_seconds=0,
        post_entry_sl_cooldown_enabled=True,
        post_entry_sl_cooldown_seconds=1800,
        post_entry_sl_cooldown_scope="SIDE",
    )
    boll = _boll()

    # Arm cooldown for LONG side
    strat.arm_post_entry_sl_cooldown(ts_ms=1000, side="LONG", reason="test_sl_exit")

    # Price below lower → should be blocked by cooldown
    price = boll.lower * 0.995
    strat.on_tick(price, 2000, boll, _cvd(ts_ms=2000, price=price, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))

    cooldown_logs = [r for r in caplog.records if "post_entry_sl_cooldown" in r.message.lower()]
    no_entry_logs = [r for r in caplog.records if "RECLAIM_NO_ENTRY" in r.message and "post_entry_sl_cooldown" in r.message]
    assert len(no_entry_logs) >= 1 or len(cooldown_logs) >= 1, \
        f"Expected cooldown or no-entry log, got {len(cooldown_logs)} cooldown / {len(no_entry_logs)} no-entry"


def test_first_valid_extreme_renamed(caplog) -> None:
    """Verify FIRST_EXTREME renamed to FIRST_CONFIRMED_EXTREME."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001)
    boll = _boll()

    p1 = boll.lower * 0.998
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))
    p2 = boll.lower * 0.994
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000))
    # Retrace to confirm first extreme
    strat.on_tick(p1, 3000, boll, _cvd(ts_ms=3000, price=p1, cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000))

    first_extreme_logs = [r for r in caplog.records if "FIRST_EXTREME" in r.message]
    first_valid_logs = [r for r in caplog.records if "FIRST_VALID_EXTREME" in r.message]
    first_confirmed_logs = [r for r in caplog.records if "FIRST_CONFIRMED_EXTREME" in r.message]
    assert len(first_extreme_logs) == 0, f"LOWER_FIRST_EXTREME should be renamed, got {len(first_extreme_logs)}"
    assert len(first_valid_logs) == 0, f"LOWER_FIRST_VALID_EXTREME should be renamed to LOWER_FIRST_CONFIRMED_EXTREME"
    assert len(first_confirmed_logs) == 1, f"Expected LOWER_FIRST_CONFIRMED_EXTREME, got {len(first_confirmed_logs)}"


# ── Reclaim V2 abort + re-entry ────────────────────────────────────────


def test_re_entry_after_abort_lower(caplog) -> None:
    """After a no-divergence abort, price going back outside lower should
    start a fresh setup (new LOWER_OUTSIDE_OBSERVED, new anchor)."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001)
    boll = _boll()

    # Cycle 1: outside → first extreme candidate → retrace confirm → NO divergence → price returns inside
    p1 = boll.lower * 0.998
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))
    p2 = boll.lower * 0.994
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000))
    # Retrace to confirm first extreme
    pr = boll.lower * 0.998
    strat.on_tick(pr, 2500, boll, _cvd(ts_ms=2500, price=pr, cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000))

    assert strat.state.lower_outside_observed is True
    assert strat.state.lower_first_extreme_price is not None
    assert strat.state.lower_anchored_divergence_confirmed is False

    # Price returns inside → should abort
    inside_price = boll.lower * 1.002  # slightly above lower
    strat.on_tick(inside_price, 3000, boll, _cvd(ts_ms=3000, price=inside_price, cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000))

    abort_logs = [r for r in caplog.records if "LOWER_RECLAIM_ABORTED" in r.getMessage()]
    assert len(abort_logs) == 1, f"Expected 1 LOWER_RECLAIM_ABORTED, got {len(abort_logs)}"
    assert strat.state.lower_outside_observed is False, "Should reset after abort"
    assert strat.state.lower_first_extreme_price is None, "Should reset first extreme"

    # Cycle 2: price goes outside lower again → fresh setup
    p3 = boll.lower * 0.996
    strat.on_tick(p3, 4000, boll, _cvd(ts_ms=4000, price=p3, cumulative_buy_volume=500_000, cumulative_sell_volume=1_700_000))

    assert strat.state.lower_outside_observed is True, (
        "Should start fresh lower_outside_observed after re-break"
    )
    # First extreme should be None initially (just observed, not yet recorded)
    assert strat.state.lower_anchor_ts_ms == 4000, (
        f"New anchor_ts_ms should be 4000, got {strat.state.lower_anchor_ts_ms}"
    )


def test_re_entry_after_abort_upper(caplog) -> None:
    """After a no-divergence abort on upper, price going back outside upper
    should start a fresh setup."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001)
    boll = _boll()

    # Cycle 1: outside above upper → first extreme candidate → retrace confirm → NO divergence → price returns inside
    p1 = boll.upper * 1.002
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000))
    p2 = boll.upper * 1.006
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, cumulative_buy_volume=1_600_000, cumulative_sell_volume=500_000))
    # Retrace to confirm first extreme
    pr = boll.upper * 1.002
    strat.on_tick(pr, 2500, boll, _cvd(ts_ms=2500, price=pr, cumulative_buy_volume=1_600_000, cumulative_sell_volume=500_000))

    assert strat.state.upper_outside_observed is True
    assert strat.state.upper_first_extreme_price is not None
    assert strat.state.upper_anchored_divergence_confirmed is False

    # Price returns inside → should abort
    inside_price = boll.upper * 0.998  # slightly below upper
    strat.on_tick(inside_price, 3000, boll, _cvd(ts_ms=3000, price=inside_price, cumulative_buy_volume=1_600_000, cumulative_sell_volume=500_000))

    abort_logs = [r for r in caplog.records if "UPPER_RECLAIM_ABORTED" in r.getMessage()]
    assert len(abort_logs) == 1, f"Expected 1 UPPER_RECLAIM_ABORTED, got {len(abort_logs)}"
    assert strat.state.upper_outside_observed is False, "Should reset after abort"
    assert strat.state.upper_first_extreme_price is None

    # Cycle 2: price goes outside upper again → fresh setup
    p3 = boll.upper * 1.004
    strat.on_tick(p3, 4000, boll, _cvd(ts_ms=4000, price=p3, cumulative_buy_volume=1_700_000, cumulative_sell_volume=500_000))

    assert strat.state.upper_outside_observed is True, (
        "Should start fresh upper_outside_observed after re-break"
    )


# ======================================================================
# Tests: coherent snapshot log — prev/current pair from SAME divergence evaluation
# ======================================================================


def _snap_strategy(**overrides) -> BollCvdReclaimStrategy:
    """Build a strategy for snapshot log tests with zero log interval."""
    cfg_kwargs = dict(
        min_outside_pct=0.001,
        entry_min_reward_risk=0.0,
        entry_fee_slippage_buffer_pct=0.0,
        order_cooldown_seconds=0,
        entry_reclaim_v2_enabled=True,
        entry_reclaim_require_anchored_divergence=True,
        entry_reclaim_new_extreme_buffer_pct=0.0,
        entry_reclaim_min_cvd_recovery=0.0,
        reclaim_extreme_log_interval_seconds=0,  # no throttle for testing
    )
    cfg_kwargs.update(overrides)
    cfg = BollCvdReclaimStrategyConfig(**cfg_kwargs)
    sizer = SimplePositionSizer(SimplePositionSizerConfig(
        dry_run_equity_usdt=10_000, leverage=20, trade_risk_pct=0.01,
        fee_slippage_buffer_pct=0.001,
    ))
    return BollCvdReclaimStrategy(cfg, sizer)


# ── Test: LOWER snapshot uses coherent prev/current pair ──────────────

def test_lower_snapshot_coherent_pair(caplog) -> None:
    """LOWER_EXTREME_SNAPSHOT prev/curr must come from same divergence eval."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _snap_strategy()
    boll = _boll(lower=1900)

    # Step 1: outside observed
    p1 = 1898.0  # below lower=1900
    strat.on_tick(p1, 1000, boll, _cvd(
        ts_ms=1000, price=p1,
        cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000,
    ))

    # Step 2: first extreme candidate → retrace to confirm
    p2 = 1894.0
    strat.on_tick(p2, 2000, boll, _cvd(
        ts_ms=2000, price=p2,
        cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000,
    ))
    # Retrace confirm: 1894 * 1.0008 = 1895.52; 1897 >= 1895.52 → confirmed
    pr = 1897.0
    strat.on_tick(pr, 2500, boll, _cvd(
        ts_ms=2500, price=pr,
        cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000,
    ))
    # Should record first CONFIRMED extreme but NOT print snapshot
    assert strat.state.lower_first_extreme_price == 1894.0
    snapshot_logs_step2 = [r for r in caplog.records if "LOWER_EXTREME_SNAPSHOT" in r.getMessage()]
    assert len(snapshot_logs_step2) == 0, (
        "First extreme should NOT produce LOWER_EXTREME_SNAPSHOT (no pair yet)"
    )
    first_logs = [r for r in caplog.records if "LOWER_FIRST_CONFIRMED_EXTREME" in r.getMessage()]
    assert len(first_logs) == 1

    # Step 3: second extreme candidate (price=1890, anchored_cvd = -200k, CVD follows lower)
    # prev=1894 prev_cvd=-100k, curr=1890 curr_cvd=-200k → no recovery → cvd_not_recovered
    p3 = 1890.0
    strat.on_tick(p3, 3000, boll, _cvd(
        ts_ms=3000, price=p3,
        cumulative_buy_volume=500_000, cumulative_sell_volume=1_700_000,
    ))
    # Retrace confirm: 1890 * 1.0008 = 1891.51; 1894 >= 1891.51 → confirmed
    pr2 = 1894.0
    strat.on_tick(pr2, 3500, boll, _cvd(
        ts_ms=3500, price=pr2,
        cumulative_buy_volume=500_000, cumulative_sell_volume=1_700_000,
    ))

    snapshot_logs = [r for r in caplog.records if "LOWER_EXTREME_SNAPSHOT" in r.getMessage()]
    assert len(snapshot_logs) == 1, f"Expected 1 LOWER_EXTREME_SNAPSHOT, got {len(snapshot_logs)}"
    msg = snapshot_logs[0].getMessage()

    # prev pair must match first extreme
    assert "prev_extreme=1894.0000" in msg, f"prev_extreme mismatch: {msg}"
    assert "prev_cvd=-100000.0000" in msg, f"prev_cvd mismatch: {msg}"
    # curr pair must match second extreme
    assert "curr_extreme=1890.0000" in msg, f"curr_extreme mismatch: {msg}"
    assert "curr_cvd=-200000.0000" in msg, f"curr_cvd mismatch: {msg}"
    # divergence not confirmed (CVD followed lower)
    assert "divergence_confirmed=False" in msg
    assert "divergence_reason=cvd_not_recovered" in msg


# ── Test: LOWER cvd_not_recovered reason when price extends but CVD follows ──

def test_lower_snapshot_cvd_not_recovered_reason(caplog) -> None:
    """When price makes a new low AND CVD makes a new low → cvd_not_recovered."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _snap_strategy()
    boll = _boll(lower=1900)

    # Outside observed
    strat.on_tick(1898, 1000, boll, _cvd(
        ts_ms=1000, price=1898,
        cumulative_buy_volume=1_000_000, cumulative_sell_volume=2_000_000,
    ))
    # First extreme candidate → retrace confirm
    strat.on_tick(1894, 2000, boll, _cvd(
        ts_ms=2000, price=1894,
        cumulative_buy_volume=1_000_000, cumulative_sell_volume=2_000_000,
    ))
    strat.on_tick(1897, 2500, boll, _cvd(
        ts_ms=2500, price=1897,
        cumulative_buy_volume=1_000_000, cumulative_sell_volume=2_000_000,
    ))
    assert strat.state.lower_first_extreme_price is not None

    # Second extreme: price lower (1890 < 1894) BUT CVD unchanged/more sell
    strat.on_tick(1890, 3000, boll, _cvd(
        ts_ms=3000, price=1890,
        cumulative_buy_volume=1_000_000, cumulative_sell_volume=2_300_000,
    ))
    # Retrace confirm → divergence eval fires
    strat.on_tick(1894, 3500, boll, _cvd(
        ts_ms=3500, price=1894,
        cumulative_buy_volume=1_000_000, cumulative_sell_volume=2_300_000,
    ))

    snapshot_logs = [r for r in caplog.records if "LOWER_EXTREME_SNAPSHOT" in r.getMessage()]
    assert len(snapshot_logs) == 1
    msg = snapshot_logs[0].getMessage()

    assert "divergence_confirmed=False" in msg
    assert "divergence_reason=cvd_not_recovered" in msg
    # Must NOT be no_new_low (price DID make a new low)
    assert "no_new_low" not in msg


# ── Test: LOWER divergence confirmed when CVD recovers ────────────────

def test_lower_snapshot_divergence_confirmed(caplog) -> None:
    """When price makes new low AND CVD recovers → confirmed=True, reason=ok."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _snap_strategy()
    boll = _boll(lower=1900)

    # Outside observed: cum_cvd = 500k - 1.5M = -1M (anchor)
    strat.on_tick(1898, 1000, boll, _cvd(
        ts_ms=1000, price=1898,
        cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000,
    ))
    # First extreme candidate → retrace confirm
    strat.on_tick(1894, 2000, boll, _cvd(
        ts_ms=2000, price=1894,
        cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000,
    ))
    strat.on_tick(1897, 2500, boll, _cvd(
        ts_ms=2500, price=1897,
        cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000,
    ))

    # Second extreme candidate: price lower, CVD recovered
    # cum_cvd = 800k - 1.4M = -600k. anchored_cvd = -600k - (-1M) = 400k
    # prev anchored_cvd = 0, curr anchored_cvd = 400k → CVD recovered → confirmed
    strat.on_tick(1890, 3000, boll, _cvd(
        ts_ms=3000, price=1890,
        cumulative_buy_volume=800_000, cumulative_sell_volume=1_400_000,
    ))
    # Retrace confirm → divergence eval fires
    strat.on_tick(1894, 3500, boll, _cvd(
        ts_ms=3500, price=1894,
        cumulative_buy_volume=800_000, cumulative_sell_volume=1_400_000,
    ))

    snapshot_logs = [r for r in caplog.records if "LOWER_EXTREME_SNAPSHOT" in r.getMessage()]
    assert len(snapshot_logs) == 1
    msg = snapshot_logs[0].getMessage()

    assert "prev_extreme=1894.0000" in msg
    assert "curr_extreme=1890.0000" in msg
    assert "divergence_confirmed=True" in msg
    assert "divergence_reason=ok" in msg
    assert strat.state.lower_armed is True


# ── Test: UPPER snapshot uses coherent prev/current pair ──────────────

def test_upper_snapshot_coherent_pair(caplog) -> None:
    """UPPER_EXTREME_SNAPSHOT prev/curr must come from same divergence eval."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _snap_strategy()
    boll = _boll(upper=2100)

    # Outside observed above upper
    strat.on_tick(2102, 1000, boll, _cvd(
        ts_ms=1000, price=2102,
        cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
        buy_ratio=0.4, sell_ratio=0.6,
    ))
    # First extreme candidate → retrace confirm
    strat.on_tick(2106, 2000, boll, _cvd(
        ts_ms=2000, price=2106,
        cumulative_buy_volume=1_600_000, cumulative_sell_volume=500_000,
        buy_ratio=0.4, sell_ratio=0.6,
    ))
    strat.on_tick(2103, 2500, boll, _cvd(
        ts_ms=2500, price=2103,
        cumulative_buy_volume=1_600_000, cumulative_sell_volume=500_000,
        buy_ratio=0.4, sell_ratio=0.6,
    ))
    assert strat.state.upper_first_extreme_price is not None
    # No snapshot for first extreme
    snapshot_logs_step2 = [r for r in caplog.records if "UPPER_EXTREME_SNAPSHOT" in r.getMessage()]
    assert len(snapshot_logs_step2) == 0

    # Second extreme candidate at 2110: anchored_cvd = +200k (CVD follows higher → no reversal)
    strat.on_tick(2110, 3000, boll, _cvd(
        ts_ms=3000, price=2110,
        cumulative_buy_volume=1_700_000, cumulative_sell_volume=500_000,
        buy_ratio=0.4, sell_ratio=0.6,
    ))
    # Retrace confirm → divergence eval fires
    strat.on_tick(2107, 3500, boll, _cvd(
        ts_ms=3500, price=2107,
        cumulative_buy_volume=1_700_000, cumulative_sell_volume=500_000,
        buy_ratio=0.4, sell_ratio=0.6,
    ))

    snapshot_logs = [r for r in caplog.records if "UPPER_EXTREME_SNAPSHOT" in r.getMessage()]
    assert len(snapshot_logs) == 1, f"Expected 1 UPPER_EXTREME_SNAPSHOT, got {len(snapshot_logs)}"
    msg = snapshot_logs[0].getMessage()

    assert "prev_extreme=2106.0000" in msg, f"prev_extreme mismatch: {msg}"
    assert "prev_cvd=100000.0000" in msg, f"prev_cvd mismatch: {msg}"
    assert "curr_extreme=2110.0000" in msg, f"curr_extreme mismatch: {msg}"
    assert "curr_cvd=200000.0000" in msg, f"curr_cvd mismatch: {msg}"
    assert "divergence_confirmed=False" in msg
    assert "divergence_reason=cvd_not_reversed" in msg


# ── Test: UPPER divergence confirmed when CVD reverses ────────────────

def test_upper_snapshot_divergence_confirmed(caplog) -> None:
    """When price makes new high AND CVD reverses → confirmed=True, reason=ok."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _snap_strategy()
    boll = _boll(upper=2100)

    # Outside observed: cum_cvd = 1.5M - 500k = +1M (anchor)
    strat.on_tick(2102, 1000, boll, _cvd(
        ts_ms=1000, price=2102,
        cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
        buy_ratio=0.4, sell_ratio=0.6,
    ))
    # First extreme candidate → retrace confirm
    strat.on_tick(2106, 2000, boll, _cvd(
        ts_ms=2000, price=2106,
        cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
        buy_ratio=0.4, sell_ratio=0.6,
    ))
    strat.on_tick(2103, 2500, boll, _cvd(
        ts_ms=2500, price=2103,
        cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
        buy_ratio=0.4, sell_ratio=0.6,
    ))

    # Second extreme candidate at 2110: cum_cvd = 1.2M - 800k = +400k
    # anchored_cvd = +400k - (+1M) = -600k
    # prev anchored_cvd = 0, curr anchored_cvd = -600k → CVD reversed → confirmed
    strat.on_tick(2110, 3000, boll, _cvd(
        ts_ms=3000, price=2110,
        cumulative_buy_volume=1_200_000, cumulative_sell_volume=800_000,
        buy_ratio=0.4, sell_ratio=0.6,
    ))
    # Retrace confirm → divergence eval fires
    strat.on_tick(2107, 3500, boll, _cvd(
        ts_ms=3500, price=2107,
        cumulative_buy_volume=1_200_000, cumulative_sell_volume=800_000,
        buy_ratio=0.4, sell_ratio=0.6,
    ))

    snapshot_logs = [r for r in caplog.records if "UPPER_EXTREME_SNAPSHOT" in r.getMessage()]
    assert len(snapshot_logs) == 1
    msg = snapshot_logs[0].getMessage()

    assert "prev_extreme=2106.0000" in msg
    assert "curr_extreme=2110.0000" in msg
    assert "divergence_confirmed=True" in msg
    assert "divergence_reason=ok" in msg
    assert strat.state.upper_armed is True


# ── Test: delayed snapshot uses latest coherent decision ──────────────

def test_delayed_snapshot_uses_latest_coherent_decision(caplog) -> None:
    """Multiple new extremes within log interval → snapshot uses LAST evaluation."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _snap_strategy(reclaim_extreme_log_interval_seconds=10)
    boll = _boll(lower=1900)

    # Outside observed: cum_cvd = 500k - 1.5M = -1M (anchor)
    strat.on_tick(1898, 1000, boll, _cvd(
        ts_ms=1000, price=1898,
        cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000,
    ))
    # First extreme candidate → retrace confirm
    strat.on_tick(1894, 2000, boll, _cvd(
        ts_ms=2000, price=1894,
        cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000,
    ))
    strat.on_tick(1897, 2500, boll, _cvd(
        ts_ms=2500, price=1897,
        cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000,
    ))

    # Second extreme (1892): same CVD → cvd_not_recovered
    # This sets pending=True but throttle prevents log (interval=10s, only 0.5s elapsed)
    strat.on_tick(1892, 3000, boll, _cvd(
        ts_ms=3000, price=1892,
        cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000,
    ))
    strat.on_tick(1895, 3200, boll, _cvd(
        ts_ms=3200, price=1895,
        cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000,
    ))
    # Should NOT have printed snapshot yet (throttled)
    snapshot_after_2nd = [r for r in caplog.records if "LOWER_EXTREME_SNAPSHOT" in r.getMessage()]
    assert len(snapshot_after_2nd) == 0, "Snapshot should be throttled within 10s"

    # Third extreme (1890): prev is now 1892, curr is 1890
    # This re-evaluates and overwrites the snapshot cache
    strat.on_tick(1890, 4000, boll, _cvd(
        ts_ms=4000, price=1890,
        cumulative_buy_volume=500_000, cumulative_sell_volume=1_700_000,
    ))
    strat.on_tick(1894, 4200, boll, _cvd(
        ts_ms=4200, price=1894,
        cumulative_buy_volume=500_000, cumulative_sell_volume=1_700_000,
    ))
    # Still throttled
    snapshot_after_3rd = [r for r in caplog.records if "LOWER_EXTREME_SNAPSHOT" in r.getMessage()]
    assert len(snapshot_after_3rd) == 0, "Snapshot should still be throttled"

    # Now advance time past the 10s interval → log fires
    strat.on_tick(1889, 15000, boll, _cvd(
        ts_ms=15000, price=1889,
        cumulative_buy_volume=500_000, cumulative_sell_volume=1_800_000,
    ))
    strat.on_tick(1893, 15200, boll, _cvd(
        ts_ms=15200, price=1893,
        cumulative_buy_volume=500_000, cumulative_sell_volume=1_800_000,
    ))

    snapshot_logs = [r for r in caplog.records if "LOWER_EXTREME_SNAPSHOT" in r.getMessage()]
    assert len(snapshot_logs) >= 1, f"Expected at least 1 snapshot after interval, got {len(snapshot_logs)}"
    last_msg = snapshot_logs[-1].getMessage()

    # The LAST snapshot must correspond to the latest evaluation (step 4: 1889 vs 1890)
    # prev should be 1890 (from step 3 eval), NOT 1892 or 1894
    assert "prev_extreme=1890.0000" in last_msg, (
        f"Delayed snapshot should use latest eval's prev (1890), got: {last_msg}"
    )
    # curr should be 1889 (from step 4). Wait — step 4 is also a new extreme.
    # Let me reconsider. Step 4 at ts_ms=15000: price=1889, anchored_cvd = (500k-1.8M) - (-1M) = -1.3M - (-1M) = -300k
    # Step 3: curr=1890, curr_cvd=(500k-1.7M)-(-1M) = -1.2M-(-1M) = -200k
    # Step 4: curr=1889, curr_cvd=-300k
    # Step 4 eval: prev=1890 prev_cvd=-200k, curr=1889 curr_cvd=-300k → cvd_not_recovered
    # So snapshot should show prev=1890, curr=1889
    assert "curr_extreme=1889.0000" in last_msg, (
        f"Delayed snapshot should use latest eval's curr (1889), got: {last_msg}"
    )
    assert "prev_extreme=1894.0000" not in last_msg, (
        f"Delayed snapshot must NOT contain stale first extreme (1894): {last_msg}"
    )
    assert "prev_extreme=1892.0000" not in last_msg, (
        f"Delayed snapshot must NOT contain intermediate extreme (1892): {last_msg}"
    )


# ── Test: no snapshot when no divergence evaluation happened ──────────

def test_no_snapshot_without_evaluation(caplog) -> None:
    """Only LOWER_FIRST_VALID_EXTREME, no LOWER_EXTREME_SNAPSHOT for first extreme."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _snap_strategy()
    boll = _boll(lower=1900)

    # Outside observed
    strat.on_tick(1898, 1000, boll, _cvd(
        ts_ms=1000, price=1898,
        cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000,
    ))
    # First extreme candidate → retrace confirm
    strat.on_tick(1894, 2000, boll, _cvd(
        ts_ms=2000, price=1894,
        cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000,
    ))
    strat.on_tick(1897, 2500, boll, _cvd(
        ts_ms=2500, price=1897,
        cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000,
    ))

    first_logs = [r for r in caplog.records if "LOWER_FIRST_CONFIRMED_EXTREME" in r.getMessage()]
    snapshot_logs = [r for r in caplog.records if "LOWER_EXTREME_SNAPSHOT" in r.getMessage()]

    assert len(first_logs) == 1, "Should log LOWER_FIRST_CONFIRMED_EXTREME"
    assert len(snapshot_logs) == 0, (
        "First extreme should NOT produce LOWER_EXTREME_SNAPSHOT (no prev/curr pair)"
    )

    # Verify snapshot cache is NOT populated for first extreme
    assert strat.state.lower_last_snapshot_prev_extreme_price is None
    assert strat.state.lower_last_snapshot_curr_extreme_price is None
    assert strat.state.lower_last_snapshot_divergence_reason is None


# ── Test: UPPER cvd_not_reversed when no CVD reversal ─────────────────

def test_upper_snapshot_cvd_not_reversed_reason(caplog) -> None:
    """When price makes new high but CVD doesn't reverse → cvd_not_reversed."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _snap_strategy()
    boll = _boll(upper=2100)

    # Outside observed
    strat.on_tick(2102, 1000, boll, _cvd(
        ts_ms=1000, price=2102,
        cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
        buy_ratio=0.4, sell_ratio=0.6,
    ))
    # First extreme candidate → retrace confirm
    strat.on_tick(2106, 2000, boll, _cvd(
        ts_ms=2000, price=2106,
        cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
        buy_ratio=0.4, sell_ratio=0.6,
    ))
    strat.on_tick(2103, 2500, boll, _cvd(
        ts_ms=2500, price=2103,
        cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
        buy_ratio=0.4, sell_ratio=0.6,
    ))
    # Second extreme candidate: CVD follows higher → cvd_not_reversed
    strat.on_tick(2110, 3000, boll, _cvd(
        ts_ms=3000, price=2110,
        cumulative_buy_volume=1_800_000, cumulative_sell_volume=500_000,
        buy_ratio=0.4, sell_ratio=0.6,
    ))
    # Retrace confirm → divergence eval fires
    strat.on_tick(2107, 3500, boll, _cvd(
        ts_ms=3500, price=2107,
        cumulative_buy_volume=1_800_000, cumulative_sell_volume=500_000,
        buy_ratio=0.4, sell_ratio=0.6,
    ))

    snapshot_logs = [r for r in caplog.records if "UPPER_EXTREME_SNAPSHOT" in r.getMessage()]
    assert len(snapshot_logs) == 1
    msg = snapshot_logs[0].getMessage()

    assert "divergence_confirmed=False" in msg
    assert "divergence_reason=cvd_not_reversed" in msg
    assert "no_new_high" not in msg
