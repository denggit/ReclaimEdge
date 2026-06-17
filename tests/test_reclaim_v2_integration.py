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
    """After a new lower extreme, snapshot log prints after 10s interval."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001, reclaim_extreme_log_interval_seconds=10)
    boll = _boll()

    # Tick 1: anchor
    p1 = boll.lower * 0.998
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))
    # Tick 2: first extreme
    p2 = boll.lower * 0.994
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000))
    # Tick 3: new extreme (t=3000) — triggers snapshot_pending
    p3 = boll.lower * 0.990
    strat.on_tick(p3, 3000, boll, _cvd(ts_ms=3000, price=p3, cumulative_buy_volume=600_000, cumulative_sell_volume=1_800_000))

    # Within 10s (t=3000..13000), no snapshot yet — still inside interval
    # At t=4000: same extreme, but no new extreme, still shouldn't log
    strat.on_tick(p3, 4000, boll, _cvd(ts_ms=4000, price=p3, cumulative_buy_volume=600_000, cumulative_sell_volume=1_800_000))
    snapshot_logs = [r for r in caplog.records if "LOWER_EXTREME_SNAPSHOT" in r.message]
    assert len(snapshot_logs) == 0, "Should not print snapshot within 10s of pending"

    # After 10s (t=14000): snapshot should print
    strat.on_tick(p3, 14000, boll, _cvd(ts_ms=14000, price=p3, cumulative_buy_volume=600_000, cumulative_sell_volume=1_800_000))
    snapshot_logs = [r for r in caplog.records if "LOWER_EXTREME_SNAPSHOT" in r.message]
    assert len(snapshot_logs) == 1, f"Expected 1 snapshot after 10s, got {len(snapshot_logs)}"


def test_extreme_snapshot_after_new_extreme_upper(caplog) -> None:
    """After a new upper extreme, snapshot log prints after 10s interval."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001, reclaim_extreme_log_interval_seconds=10)
    boll = _boll()

    # Tick 1: anchor
    p1 = boll.upper * 1.002
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
                                         buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                                         cvd_increasing=False, cvd_decreasing=True, no_new_high=True))
    # Tick 2: first extreme
    p2 = boll.upper * 1.006
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
                                         buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                                         cvd_increasing=False, cvd_decreasing=True, no_new_high=True))
    # Tick 3: new extreme
    p3 = boll.upper * 1.010
    strat.on_tick(p3, 3000, boll, _cvd(ts_ms=3000, price=p3, cumulative_buy_volume=1_400_000, cumulative_sell_volume=600_000,
                                         buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                                         cvd_increasing=False, cvd_decreasing=True, no_new_high=True))

    # After 10s: snapshot should print
    strat.on_tick(p3, 14000, boll, _cvd(ts_ms=14000, price=p3, cumulative_buy_volume=1_400_000, cumulative_sell_volume=600_000,
                                         buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                                         cvd_increasing=False, cvd_decreasing=True, no_new_high=True))
    snapshot_logs = [r for r in caplog.records if "UPPER_EXTREME_SNAPSHOT" in r.message]
    assert len(snapshot_logs) == 1, f"Expected 1 snapshot after 10s, got {len(snapshot_logs)}"


def test_no_snapshot_without_new_extreme(caplog) -> None:
    """Outside ticks without new extreme should NOT trigger snapshot."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001, reclaim_extreme_log_interval_seconds=10)
    boll = _boll()

    # Anchor + first extreme + new extreme (triggers pending)
    p1 = boll.lower * 0.998
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))
    p2 = boll.lower * 0.994
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000))
    p3 = boll.lower * 0.990
    strat.on_tick(p3, 3000, boll, _cvd(ts_ms=3000, price=p3, cumulative_buy_volume=600_000, cumulative_sell_volume=1_800_000))

    # After 10s, snapshot prints once
    strat.on_tick(p3, 14000, boll, _cvd(ts_ms=14000, price=p3, cumulative_buy_volume=600_000, cumulative_sell_volume=1_800_000))
    snapshot_logs = [r for r in caplog.records if "LOWER_EXTREME_SNAPSHOT" in r.message]
    assert len(snapshot_logs) == 1

    # More outside ticks without new extreme → no more snapshots
    strat.on_tick(p3, 25000, boll, _cvd(ts_ms=25000, price=p3, cumulative_buy_volume=600_000, cumulative_sell_volume=1_800_000))
    strat.on_tick(p3, 26000, boll, _cvd(ts_ms=26000, price=p3, cumulative_buy_volume=600_000, cumulative_sell_volume=1_800_000))
    snapshot_logs = [r for r in caplog.records if "LOWER_EXTREME_SNAPSHOT" in r.message]
    assert len(snapshot_logs) == 1, "Should not print additional snapshots without new extreme"


def test_snapshot_logs_latest_extreme_only(caplog) -> None:
    """Multiple new extremes within 10s → only latest printed in snapshot."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001, reclaim_extreme_log_interval_seconds=10)
    boll = _boll()

    # Anchor
    p1 = boll.lower * 0.998
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))
    # First extreme: 1890.6
    p2 = boll.lower * 0.994  # ~1888.6
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000))
    # New extreme t=3000: price 1881
    p3 = boll.lower * 0.990  # ~1881
    strat.on_tick(p3, 3000, boll, _cvd(ts_ms=3000, price=p3, cumulative_buy_volume=600_000, cumulative_sell_volume=1_800_000))
    # New extreme t=5000: price 1875 (even lower)
    p4 = boll.lower * 0.987  # ~1875.3
    strat.on_tick(p4, 5000, boll, _cvd(ts_ms=5000, price=p4, cumulative_buy_volume=650_000, cumulative_sell_volume=1_900_000))
    # New extreme t=7000: price 1862 (lowest)
    p5 = boll.lower * 0.980  # ~1862
    strat.on_tick(p5, 7000, boll, _cvd(ts_ms=7000, price=p5, cumulative_buy_volume=700_000, cumulative_sell_volume=2_000_000))

    # After 10s (t=14000): only one snapshot with latest extreme
    strat.on_tick(p5, 14000, boll, _cvd(ts_ms=14000, price=p5, cumulative_buy_volume=700_000, cumulative_sell_volume=2_000_000))
    snapshot_logs = [r for r in caplog.records if "LOWER_EXTREME_SNAPSHOT" in r.message]
    assert len(snapshot_logs) == 1, f"Expected 1 snapshot (latest only), got {len(snapshot_logs)}"
    # The new_extreme_count should reflect all extremes seen
    assert "new_extreme_count=3" in snapshot_logs[0].message or "new_extreme_count=4" in snapshot_logs[0].message


# ═══════════════════════════════════════════════════════════════════════
# Reclaim V2 observability: no-entry reason logging
# ═══════════════════════════════════════════════════════════════════════


def _setup_lower_outside_with_first_extreme(strat, boll) -> None:
    """Helper: anchor + first valid extreme for LOWER, no divergence."""
    p1 = boll.lower * 0.998
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))
    p2 = boll.lower * 0.994
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000))


def _setup_upper_outside_with_first_extreme(strat, boll) -> None:
    """Helper: anchor + first valid extreme for UPPER, no divergence."""
    p1 = boll.upper * 1.002
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
                                         buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                                         cvd_increasing=False, cvd_decreasing=True, no_new_high=True))
    p2 = boll.upper * 1.006
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
                                         buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                                         cvd_increasing=False, cvd_decreasing=True, no_new_high=True))


def test_no_entry_no_anchored_divergence_lower(caplog) -> None:
    """Outside observed + first extreme + no divergence + price inside → no_anchored_divergence log."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001, reclaim_no_entry_log_interval_seconds=0)
    boll = _boll()
    _setup_lower_outside_with_first_extreme(strat, boll)

    # Price moves back inside band without divergence being confirmed
    inside_price = boll.lower * 1.001  # just inside the band
    strat.on_tick(inside_price, 3000, boll, _cvd(ts_ms=3000, price=inside_price, cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000))

    no_entry_logs = [r for r in caplog.records if "LOWER_RECLAIM_NO_ENTRY" in r.message]
    assert len(no_entry_logs) >= 1, f"Expected no-entry log, got {len(no_entry_logs)}"
    assert "no_anchored_divergence" in no_entry_logs[0].message


def test_no_entry_no_anchored_divergence_upper(caplog) -> None:
    """Outside observed + first extreme + no divergence + price inside → no_anchored_divergence log (UPPER)."""
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

    no_entry_logs = [r for r in caplog.records if "UPPER_RECLAIM_NO_ENTRY" in r.message]
    assert len(no_entry_logs) >= 1, f"Expected no-entry log, got {len(no_entry_logs)}"
    assert "no_anchored_divergence" in no_entry_logs[0].message


def _run_lower_armed_divergence(strat, boll) -> None:
    """Helper: run full lower armed with anchored divergence confirmed via on_tick."""
    p1 = boll.lower * 0.998
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))
    p2 = boll.lower * 0.994
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))
    # New low + CVD recovery → divergence confirmed
    p3 = boll.lower * 0.990
    strat.on_tick(p3, 3000, boll, _cvd(ts_ms=3000, price=p3, cumulative_buy_volume=600_000, cumulative_sell_volume=1_400_000))
    assert strat.state.lower_anchored_divergence_confirmed is True
    assert strat.state.lower_armed is True


def _run_upper_armed_divergence(strat, boll) -> None:
    """Helper: run full upper armed with anchored divergence confirmed via on_tick."""
    p1 = boll.upper * 1.002
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
                                         buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                                         cvd_increasing=False, cvd_decreasing=True, no_new_high=True))
    p2 = boll.upper * 1.006
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
                                         buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                                         cvd_increasing=False, cvd_decreasing=True, no_new_high=True))
    # New high + CVD reverse → divergence confirmed
    p3 = boll.upper * 1.010
    strat.on_tick(p3, 3000, boll, _cvd(ts_ms=3000, price=p3, cumulative_buy_volume=1_400_000, cumulative_sell_volume=600_000,
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
    """Verify FIRST_EXTREME renamed to FIRST_VALID_EXTREME."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001)
    boll = _boll()

    p1 = boll.lower * 0.998
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))
    p2 = boll.lower * 0.994
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000))

    first_extreme_logs = [r for r in caplog.records if "FIRST_EXTREME" in r.message]
    first_valid_logs = [r for r in caplog.records if "FIRST_VALID_EXTREME" in r.message]
    assert len(first_extreme_logs) == 0, f"LOWER_FIRST_EXTREME should be renamed, got {len(first_extreme_logs)}"
    assert len(first_valid_logs) == 1, f"Expected LOWER_FIRST_VALID_EXTREME, got {len(first_valid_logs)}"
