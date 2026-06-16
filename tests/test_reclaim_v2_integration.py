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


# ── Test 9: first valid extreme must be deep enough ────────────────────

def test_first_extreme_requires_deep_enough_lower() -> None:
    """Shallow outside breach must NOT record LOWER_FIRST_EXTREME."""
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
    """Shallow outside breach must NOT record UPPER_FIRST_EXTREME."""
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
