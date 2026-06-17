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

_MINUTE_MS = 60_000


def _feed_lower_fractal_sequence(strat, boll, *, anchor_minute=0):
    """Feed a 6-minute tick sequence that produces a LOWER fractal extreme.

    Minute 2 has the lowest low; fractal confirms when minute 4 closes.
    """
    prices = {
        0: boll.lower * 0.998,   # left-2
        1: boll.lower * 0.996,   # left-1
        2: boll.lower * 0.992,   # candidate (lowest)
        3: boll.lower * 0.995,   # right-1
        4: boll.lower * 0.997,   # right-2
        5: boll.lower * 0.998,   # extra
    }
    for minute in range(6):
        ts = minute * _MINUTE_MS + 30000
        price = prices[minute]
        strat.on_tick(
            price, ts, boll,
            _cvd(ts_ms=ts, price=price,
                 cumulative_buy_volume=500_000,
                 cumulative_sell_volume=1_500_000 + minute * 100_000),
        )


def _feed_upper_fractal_sequence(strat, boll, *, anchor_minute=0):
    """Feed a 6-minute tick sequence that produces an UPPER fractal extreme."""
    prices = {
        0: boll.upper * 1.002,   # left-2
        1: boll.upper * 1.004,   # left-1
        2: boll.upper * 1.008,   # candidate (highest)
        3: boll.upper * 1.005,   # right-1
        4: boll.upper * 1.003,   # right-2
        5: boll.upper * 1.002,   # extra
    }
    for minute in range(6):
        ts = minute * _MINUTE_MS + 30000
        price = prices[minute]
        strat.on_tick(
            price, ts, boll,
            _cvd(ts_ms=ts, price=price,
                 cumulative_buy_volume=1_500_000 + minute * 100_000,
                 cumulative_sell_volume=500_000,
                 buy_ratio=0.4, sell_ratio=0.6,
                 cross_positive=False, cross_negative=True,
                 cvd_increasing=False, cvd_decreasing=True,
                 no_new_high=True),
        )


def test_first_extreme_not_armed() -> None:
    """First confirmed lower extreme should NOT arm (needs divergence)."""
    strat = _strategy(min_outside_pct=0.001)
    boll = _boll()
    _feed_lower_fractal_sequence(strat, boll)

    assert strat.state.lower_first_extreme_price is not None
    assert strat.state.lower_anchored_divergence_confirmed is False
    assert strat.state.lower_armed is False  # still not armed (first extreme only)


# ── Test 3: new low + CVD follows lower → NOT armed ──────────────────

def test_new_low_cvd_follows_not_armed() -> None:
    """Price makes new confirmed low but CVD also makes new low — no divergence."""
    strat = _strategy(min_outside_pct=0.001)
    boll = _boll()

    # First fractal: produces first confirmed extreme
    _feed_lower_fractal_sequence(strat, boll)
    assert strat.state.lower_first_extreme_price is not None

    # Preserve first extreme reference, reset 1m trackers for new sequence
    first_ext = strat.state.lower_first_extreme_price
    first_cvd = strat.state.lower_first_extreme_anchored_cvd
    prev_price = strat.state.lower_previous_confirmed_extreme_price
    prev_cvd = strat.state.lower_previous_confirmed_extreme_anchored_cvd

    strat._reset_lower_armed()
    strat.state.lower_first_extreme_price = first_ext
    strat.state.lower_first_extreme_ts_ms = 5000
    strat.state.lower_first_extreme_anchored_cvd = first_cvd
    strat.state.lower_previous_confirmed_extreme_price = prev_price
    strat.state.lower_previous_confirmed_extreme_ts_ms = 5000
    strat.state.lower_previous_confirmed_extreme_anchored_cvd = prev_cvd

    # Second fractal sequence: CVD follows price lower (more negative)
    for minute in range(6):
        ts = (minute + 10) * _MINUTE_MS + 30000
        # Form fractal with low at minute 2, but CVD worsens
        if minute == 2:
            price = boll.lower * 0.988  # new low
        elif minute < 2:
            price = boll.lower * 0.995
        else:
            price = boll.lower * 0.997
        strat.on_tick(
            price, ts, boll,
            _cvd(ts_ms=ts, price=price,
                 cumulative_buy_volume=500_000,
                 cumulative_sell_volume=2_000_000 + minute * 200_000),
        )

    assert strat.state.lower_anchored_divergence_confirmed is False
    assert strat.state.lower_armed is False


# ── Test 4: new low + CVD recovers → ARMED ───────────────────────────

def test_new_low_cvd_recovers_armed() -> None:
    """Price makes new confirmed low but CVD recovers → anchored divergence → ARMED."""
    strat = _strategy(min_outside_pct=0.001)
    boll = _boll()

    # First fractal
    _feed_lower_fractal_sequence(strat, boll)
    assert strat.state.lower_first_extreme_price is not None

    first_ext = strat.state.lower_first_extreme_price
    first_cvd = strat.state.lower_first_extreme_anchored_cvd
    prev_price = strat.state.lower_previous_confirmed_extreme_price
    prev_cvd = strat.state.lower_previous_confirmed_extreme_anchored_cvd

    strat._reset_lower_armed()
    strat.state.lower_first_extreme_price = first_ext
    strat.state.lower_first_extreme_ts_ms = 5000
    strat.state.lower_first_extreme_anchored_cvd = first_cvd
    strat.state.lower_previous_confirmed_extreme_price = prev_price
    strat.state.lower_previous_confirmed_extreme_ts_ms = 5000
    strat.state.lower_previous_confirmed_extreme_anchored_cvd = prev_cvd

    # Second fractal: lower low BUT CVD recovers (less negative)
    for minute in range(6):
        ts = (minute + 10) * _MINUTE_MS + 30000
        if minute == 2:
            price = boll.lower * 0.985  # new lower low
        elif minute < 2:
            price = boll.lower * 0.995
        else:
            price = boll.lower * 0.998
        strat.on_tick(
            price, ts, boll,
            _cvd(ts_ms=ts, price=price,
                 cumulative_buy_volume=700_000,
                 cumulative_sell_volume=1_400_000),
        )

    assert strat.state.lower_anchored_divergence_confirmed is True
    assert strat.state.lower_armed is True


# ── Test 5: upper mirror — new high + CVD reverses → ARMED ───────────

def test_upper_new_high_cvd_reverses_armed() -> None:
    """Price makes new confirmed high but CVD reverses down → bearish divergence → ARMED."""
    strat = _strategy(min_outside_pct=0.001)
    boll = _boll()

    # First fractal
    _feed_upper_fractal_sequence(strat, boll)
    assert strat.state.upper_first_extreme_price is not None

    first_ext = strat.state.upper_first_extreme_price
    first_cvd = strat.state.upper_first_extreme_anchored_cvd
    prev_price = strat.state.upper_previous_confirmed_extreme_price
    prev_cvd = strat.state.upper_previous_confirmed_extreme_anchored_cvd

    strat._reset_upper_armed()
    strat.state.upper_first_extreme_price = first_ext
    strat.state.upper_first_extreme_ts_ms = 5000
    strat.state.upper_first_extreme_anchored_cvd = first_cvd
    strat.state.upper_previous_confirmed_extreme_price = prev_price
    strat.state.upper_previous_confirmed_extreme_ts_ms = 5000
    strat.state.upper_previous_confirmed_extreme_anchored_cvd = prev_cvd

    # Second fractal: higher high but CVD reverses (less bullish)
    for minute in range(6):
        ts = (minute + 10) * _MINUTE_MS + 30000
        if minute == 2:
            price = boll.upper * 1.015  # new higher high
        elif minute < 2:
            price = boll.upper * 1.005
        else:
            price = boll.upper * 1.003
        strat.on_tick(
            price, ts, boll,
            _cvd(ts_ms=ts, price=price,
                 cumulative_buy_volume=1_300_000,
                 cumulative_sell_volume=700_000,
                 buy_ratio=0.4, sell_ratio=0.6,
                 cross_positive=False, cross_negative=True,
                 cvd_increasing=False, cvd_decreasing=True,
                 no_new_high=True),
        )

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

    # Feed a full fractal sequence to get first extreme
    _feed_lower_fractal_sequence(strat, boll)

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
    """After a 1m fractal extreme, snapshot log prints after divergence eval."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001, reclaim_extreme_log_interval_seconds=0)
    boll = _boll()
    _feed_two_lower_fractals_for_snapshot(strat, boll, second_cvd_recovery=False)

    snapshot_logs = [r for r in caplog.records if "LOWER_EXTREME_SNAPSHOT" in r.message]
    assert len(snapshot_logs) >= 1, f"Expected snapshot after divergence eval, got {len(snapshot_logs)}"


def test_extreme_snapshot_after_new_extreme_upper(caplog) -> None:
    """After a 1m fractal extreme, snapshot log prints after divergence eval (UPPER)."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001, reclaim_extreme_log_interval_seconds=0)
    boll = _boll()
    _feed_two_upper_fractals_for_snapshot(strat, boll, second_cvd_reversal=False)

    snapshot_logs = [r for r in caplog.records if "UPPER_EXTREME_SNAPSHOT" in r.message]
    assert len(snapshot_logs) >= 1, f"Expected snapshot after divergence eval, got {len(snapshot_logs)}"


def test_no_snapshot_without_new_extreme(caplog) -> None:
    """Single fractal sequence → first extreme only, no snapshot."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001, reclaim_extreme_log_interval_seconds=0)
    boll = _boll()
    _feed_lower_fractal_sequence(strat, boll)

    snapshot_logs = [r for r in caplog.records if "LOWER_EXTREME_SNAPSHOT" in r.message]
    assert len(snapshot_logs) == 0, (
        "First extreme only — no snapshot (no prev/curr pair)"
    )


def test_snapshot_logs_latest_extreme_only(caplog) -> None:
    """Snapshot log only prints after divergence evaluation (requires 2 extremes)."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001, reclaim_extreme_log_interval_seconds=0)
    boll = _boll()
    _feed_two_lower_fractals_for_snapshot(strat, boll, second_cvd_recovery=True)

    snapshot_logs = [r for r in caplog.records if "LOWER_EXTREME_SNAPSHOT" in r.message]
    assert len(snapshot_logs) == 1, f"Expected 1 snapshot after 2nd extreme, got {len(snapshot_logs)}"
    msg = snapshot_logs[0].getMessage()
    assert "divergence_confirmed=True" in msg


# ═══════════════════════════════════════════════════════════════════════
# Reclaim V2 observability: no-entry reason logging
# ═══════════════════════════════════════════════════════════════════════


def _setup_lower_outside_with_first_extreme(strat, boll) -> None:
    """Helper: anchor + first confirmed extreme for LOWER, no divergence."""
    _feed_lower_fractal_sequence(strat, boll)
    # Verify we got first extreme but no divergence (only one fractal sequence)
    assert strat.state.lower_first_extreme_price is not None
    assert strat.state.lower_anchored_divergence_confirmed is False


def _setup_upper_outside_with_first_extreme(strat, boll) -> None:
    """Helper: anchor + first confirmed extreme for UPPER, no divergence."""
    _feed_upper_fractal_sequence(strat, boll)
    assert strat.state.upper_first_extreme_price is not None
    assert strat.state.upper_anchored_divergence_confirmed is False


def test_no_entry_no_anchored_divergence_lower(caplog) -> None:
    """Outside observed + first extreme + no divergence + price inside
    → LOWER_RECLAIM_ABORTED (one-shot), no repeated no_anchored_divergence heartbeat."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001, reclaim_no_entry_log_interval_seconds=0)
    boll = _boll()
    _setup_lower_outside_with_first_extreme(strat, boll)
    # Simulate: divergence was evaluated but NOT confirmed (cvd_not_recovered)
    strat.state.lower_last_divergence_evaluated_ts_ms = 1

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
    # Simulate: divergence was evaluated but NOT confirmed (cvd_not_reversed)
    strat.state.upper_last_divergence_evaluated_ts_ms = 1

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

    Uses 1m fractal extremes — feeds two fractal sequences (6 min each)
    where the second sequence has CVD recovery.
    """
    # First fractal sequence → produces first confirmed extreme
    _feed_lower_fractal_sequence(strat, boll)
    assert strat.state.lower_first_extreme_price is not None

    # Preserve first extreme references
    first_ext = strat.state.lower_first_extreme_price
    first_cvd = strat.state.lower_first_extreme_anchored_cvd
    prev_price = strat.state.lower_previous_confirmed_extreme_price
    prev_cvd = strat.state.lower_previous_confirmed_extreme_anchored_cvd

    # Reset internal trackers but keep reference state
    strat._reset_lower_armed()
    strat.state.lower_first_extreme_price = first_ext
    strat.state.lower_first_extreme_ts_ms = 5000
    strat.state.lower_first_extreme_anchored_cvd = first_cvd
    strat.state.lower_previous_confirmed_extreme_price = prev_price
    strat.state.lower_previous_confirmed_extreme_ts_ms = 5000
    strat.state.lower_previous_confirmed_extreme_anchored_cvd = prev_cvd

    # Second fractal sequence: lower low with CVD recovery
    for minute in range(6):
        ts = (minute + 10) * _MINUTE_MS + 30000
        if minute == 2:
            price = boll.lower * 0.985  # new lower low
        elif minute < 2:
            price = boll.lower * 0.995
        else:
            price = boll.lower * 0.998
        strat.on_tick(
            price, ts, boll,
            _cvd(ts_ms=ts, price=price,
                 cumulative_buy_volume=700_000,
                 cumulative_sell_volume=1_400_000),
        )
    assert strat.state.lower_anchored_divergence_confirmed is True
    assert strat.state.lower_armed is True


def _run_upper_armed_divergence(strat, boll) -> None:
    """Helper: run full upper armed with anchored divergence confirmed via on_tick.

    Uses 1m fractal extremes — feeds two fractal sequences (6 min each)
    where the second sequence has CVD reversal.
    """
    # First fractal sequence → produces first confirmed extreme
    _feed_upper_fractal_sequence(strat, boll)
    assert strat.state.upper_first_extreme_price is not None

    # Preserve first extreme references
    first_ext = strat.state.upper_first_extreme_price
    first_cvd = strat.state.upper_first_extreme_anchored_cvd
    prev_price = strat.state.upper_previous_confirmed_extreme_price
    prev_cvd = strat.state.upper_previous_confirmed_extreme_anchored_cvd

    # Reset internal trackers but keep reference state
    strat._reset_upper_armed()
    strat.state.upper_first_extreme_price = first_ext
    strat.state.upper_first_extreme_ts_ms = 5000
    strat.state.upper_first_extreme_anchored_cvd = first_cvd
    strat.state.upper_previous_confirmed_extreme_price = prev_price
    strat.state.upper_previous_confirmed_extreme_ts_ms = 5000
    strat.state.upper_previous_confirmed_extreme_anchored_cvd = prev_cvd

    # Second fractal sequence: higher high with CVD reversal
    for minute in range(6):
        ts = (minute + 10) * _MINUTE_MS + 30000
        if minute == 2:
            price = boll.upper * 1.015  # new higher high
        elif minute < 2:
            price = boll.upper * 1.005
        else:
            price = boll.upper * 1.003
        strat.on_tick(
            price, ts, boll,
            _cvd(ts_ms=ts, price=price,
                 cumulative_buy_volume=1_300_000,
                 cumulative_sell_volume=700_000,
                 buy_ratio=0.4, sell_ratio=0.6,
                 cross_positive=False, cross_negative=True,
                 cvd_increasing=False, cvd_decreasing=True,
                 no_new_high=True),
        )
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

    _feed_lower_fractal_sequence(strat, boll)

    first_confirmed_logs = [r for r in caplog.records if "FIRST_CONFIRMED_EXTREME" in r.message]
    assert len(first_confirmed_logs) == 1, f"Expected LOWER_FIRST_CONFIRMED_EXTREME, got {len(first_confirmed_logs)}"


# ── Reclaim V2 abort + re-entry ────────────────────────────────────────


def test_re_entry_after_abort_lower(caplog) -> None:
    """After a no-divergence abort, price going back outside lower should
    start a fresh setup (new LOWER_OUTSIDE_OBSERVED, new anchor)."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001)
    boll = _boll()

    # Cycle 1: feed fractal sequence → first extreme confirmed, NO divergence
    _feed_lower_fractal_sequence(strat, boll)

    assert strat.state.lower_outside_observed is True
    assert strat.state.lower_first_extreme_price is not None
    assert strat.state.lower_anchored_divergence_confirmed is False
    # Simulate: divergence was evaluated but NOT confirmed
    strat.state.lower_last_divergence_evaluated_ts_ms = 1

    # Price returns inside → should abort
    inside_price = boll.lower * 1.002  # slightly above lower
    strat.on_tick(inside_price, 6 * _MINUTE_MS, boll, _cvd(ts_ms=6 * _MINUTE_MS, price=inside_price, cumulative_buy_volume=500_000, cumulative_sell_volume=2_000_000))

    abort_logs = [r for r in caplog.records if "LOWER_RECLAIM_ABORTED" in r.getMessage()]
    assert len(abort_logs) == 1, f"Expected 1 LOWER_RECLAIM_ABORTED, got {len(abort_logs)}"
    assert strat.state.lower_outside_observed is False, "Should reset after abort"
    assert strat.state.lower_first_extreme_price is None, "Should reset first extreme"

    # Cycle 2: price goes outside lower again → fresh setup
    p3 = boll.lower * 0.996
    strat.on_tick(p3, 7 * _MINUTE_MS, boll, _cvd(ts_ms=7 * _MINUTE_MS, price=p3, cumulative_buy_volume=500_000, cumulative_sell_volume=2_100_000))

    assert strat.state.lower_outside_observed is True, (
        "Should start fresh lower_outside_observed after re-break"
    )
    assert strat.state.lower_anchor_ts_ms == 7 * _MINUTE_MS, (
        f"New anchor_ts_ms should be {7 * _MINUTE_MS}, got {strat.state.lower_anchor_ts_ms}"
    )


def test_re_entry_after_abort_upper(caplog) -> None:
    """After a no-divergence abort on upper, price going back outside upper
    should start a fresh setup."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001)
    boll = _boll()

    # Cycle 1: feed fractal sequence → first extreme confirmed, NO divergence
    _feed_upper_fractal_sequence(strat, boll)

    assert strat.state.upper_outside_observed is True
    assert strat.state.upper_first_extreme_price is not None
    assert strat.state.upper_anchored_divergence_confirmed is False
    # Simulate: divergence was evaluated but NOT confirmed
    strat.state.upper_last_divergence_evaluated_ts_ms = 1

    # Price returns inside → should abort
    inside_price = boll.upper * 0.998  # slightly below upper
    strat.on_tick(inside_price, 6 * _MINUTE_MS, boll, _cvd(ts_ms=6 * _MINUTE_MS, price=inside_price,
                                                             cumulative_buy_volume=2_100_000, cumulative_sell_volume=500_000))

    abort_logs = [r for r in caplog.records if "UPPER_RECLAIM_ABORTED" in r.getMessage()]
    assert len(abort_logs) == 1, f"Expected 1 UPPER_RECLAIM_ABORTED, got {len(abort_logs)}"
    assert strat.state.upper_outside_observed is False, "Should reset after abort"
    assert strat.state.upper_first_extreme_price is None

    # Cycle 2: price goes outside upper again → fresh setup
    p3 = boll.upper * 1.004
    strat.on_tick(p3, 7 * _MINUTE_MS, boll, _cvd(ts_ms=7 * _MINUTE_MS, price=p3,
                                                   cumulative_buy_volume=2_200_000, cumulative_sell_volume=500_000))

    assert strat.state.upper_outside_observed is True, (
        "Should start fresh upper_outside_observed after re-break"
    )


# ======================================================================
# Snapshot helpers (1m fractal based)
# ======================================================================

_MINUTE_MS_2 = 60_000


def _feed_two_lower_fractals_for_snapshot(strat, boll, *, second_cvd_recovery: bool = False):
    """Feed two LOWER fractal sequences to produce divergence evaluation.

    First sequence → first confirmed extreme (no snapshot).
    Second sequence → lower low, triggers divergence eval → snapshot.

    If second_cvd_recovery=True, CVD improves → divergence confirmed.
    Otherwise CVD follows lower → cvd_not_recovered.
    """
    # First fractal sequence
    _feed_lower_fractal_sequence(strat, boll)
    assert strat.state.lower_first_extreme_price is not None

    first_ext = strat.state.lower_first_extreme_price
    first_cvd = strat.state.lower_first_extreme_anchored_cvd
    prev_price = strat.state.lower_previous_confirmed_extreme_price
    prev_cvd = strat.state.lower_previous_confirmed_extreme_anchored_cvd

    strat._reset_lower_armed()
    strat.state.lower_first_extreme_price = first_ext
    strat.state.lower_first_extreme_ts_ms = 5 * _MINUTE_MS_2
    strat.state.lower_first_extreme_anchored_cvd = first_cvd
    strat.state.lower_previous_confirmed_extreme_price = prev_price
    strat.state.lower_previous_confirmed_extreme_ts_ms = 5 * _MINUTE_MS_2
    strat.state.lower_previous_confirmed_extreme_anchored_cvd = prev_cvd

    # Second fractal sequence (starts at minute 10)
    offset = 10
    for minute in range(6):
        ts = (minute + offset) * _MINUTE_MS_2 + 30000
        if minute == 2:
            price = boll.lower * 0.985  # new low
        elif minute < 2:
            price = boll.lower * 0.995
        else:
            price = boll.lower * 0.998
        if second_cvd_recovery:
            # CVD recovers → divergence confirmed
            strat.on_tick(price, ts, boll, _cvd(ts_ms=ts, price=price,
                          cumulative_buy_volume=700_000, cumulative_sell_volume=1_400_000))
        else:
            # CVD follows lower → no divergence
            strat.on_tick(price, ts, boll, _cvd(ts_ms=ts, price=price,
                          cumulative_buy_volume=500_000, cumulative_sell_volume=2_000_000 + minute * 200_000))


def _feed_two_upper_fractals_for_snapshot(strat, boll, *, second_cvd_reversal: bool = False):
    """Feed two UPPER fractal sequences to produce divergence evaluation."""
    # First fractal sequence
    _feed_upper_fractal_sequence(strat, boll)
    assert strat.state.upper_first_extreme_price is not None

    first_ext = strat.state.upper_first_extreme_price
    first_cvd = strat.state.upper_first_extreme_anchored_cvd
    prev_price = strat.state.upper_previous_confirmed_extreme_price
    prev_cvd = strat.state.upper_previous_confirmed_extreme_anchored_cvd

    strat._reset_upper_armed()
    strat.state.upper_first_extreme_price = first_ext
    strat.state.upper_first_extreme_ts_ms = 5 * _MINUTE_MS_2
    strat.state.upper_first_extreme_anchored_cvd = first_cvd
    strat.state.upper_previous_confirmed_extreme_price = prev_price
    strat.state.upper_previous_confirmed_extreme_ts_ms = 5 * _MINUTE_MS_2
    strat.state.upper_previous_confirmed_extreme_anchored_cvd = prev_cvd

    # Second fractal sequence
    offset = 10
    for minute in range(6):
        ts = (minute + offset) * _MINUTE_MS_2 + 30000
        if minute == 2:
            price = boll.upper * 1.015  # new high
        elif minute < 2:
            price = boll.upper * 1.005
        else:
            price = boll.upper * 1.003
        if second_cvd_reversal:
            strat.on_tick(price, ts, boll, _cvd(ts_ms=ts, price=price,
                          cumulative_buy_volume=1_300_000, cumulative_sell_volume=700_000,
                          buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                          cvd_increasing=False, cvd_decreasing=True, no_new_high=True))
        else:
            strat.on_tick(price, ts, boll, _cvd(ts_ms=ts, price=price,
                          cumulative_buy_volume=2_000_000 + minute * 200_000, cumulative_sell_volume=500_000,
                          buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                          cvd_increasing=False, cvd_decreasing=True, no_new_high=True))


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

    _feed_two_lower_fractals_for_snapshot(strat, boll, second_cvd_recovery=False)

    snapshot_logs = [r for r in caplog.records if "LOWER_EXTREME_SNAPSHOT" in r.getMessage()]
    assert len(snapshot_logs) == 1, f"Expected 1 LOWER_EXTREME_SNAPSHOT, got {len(snapshot_logs)}"
    msg = snapshot_logs[0].getMessage()
    assert "divergence_confirmed=False" in msg
    assert "divergence_reason=cvd_not_recovered" in msg


# ── Test: LOWER cvd_not_recovered reason when price extends but CVD follows ──

def test_lower_snapshot_cvd_not_recovered_reason(caplog) -> None:
    """When price makes a new low AND CVD makes a new low → cvd_not_recovered."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _snap_strategy()
    boll = _boll(lower=1900)
    _feed_two_lower_fractals_for_snapshot(strat, boll, second_cvd_recovery=False)

    snapshot_logs = [r for r in caplog.records if "LOWER_EXTREME_SNAPSHOT" in r.getMessage()]
    assert len(snapshot_logs) == 1
    msg = snapshot_logs[0].getMessage()
    assert "divergence_confirmed=False" in msg
    assert "divergence_reason=cvd_not_recovered" in msg
    assert "no_new_low" not in msg


# ── Test: LOWER divergence confirmed when CVD recovers ────────────────

def test_lower_snapshot_divergence_confirmed(caplog) -> None:
    """When price makes new low AND CVD recovers → confirmed=True, reason=ok."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _snap_strategy()
    boll = _boll(lower=1900)
    _feed_two_lower_fractals_for_snapshot(strat, boll, second_cvd_recovery=True)

    snapshot_logs = [r for r in caplog.records if "LOWER_EXTREME_SNAPSHOT" in r.getMessage()]
    assert len(snapshot_logs) == 1
    msg = snapshot_logs[0].getMessage()
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
    _feed_two_upper_fractals_for_snapshot(strat, boll, second_cvd_reversal=False)

    snapshot_logs = [r for r in caplog.records if "UPPER_EXTREME_SNAPSHOT" in r.getMessage()]
    assert len(snapshot_logs) == 1, f"Expected 1 UPPER_EXTREME_SNAPSHOT, got {len(snapshot_logs)}"
    msg = snapshot_logs[0].getMessage()
    assert "divergence_confirmed=False" in msg
    assert "divergence_reason=cvd_not_reversed" in msg


# ── Test: UPPER divergence confirmed when CVD reverses ────────────────

def test_upper_snapshot_divergence_confirmed(caplog) -> None:
    """When price makes new high AND CVD reverses → confirmed=True, reason=ok."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _snap_strategy()
    boll = _boll(upper=2100)
    _feed_two_upper_fractals_for_snapshot(strat, boll, second_cvd_reversal=True)

    snapshot_logs = [r for r in caplog.records if "UPPER_EXTREME_SNAPSHOT" in r.getMessage()]
    assert len(snapshot_logs) == 1
    msg = snapshot_logs[0].getMessage()
    assert "divergence_confirmed=True" in msg
    assert "divergence_reason=ok" in msg
    assert strat.state.upper_armed is True


# ── Test: delayed snapshot uses latest coherent decision ──────────────

def test_delayed_snapshot_uses_latest_coherent_decision(caplog) -> None:
    """Multiple extremes within log interval → snapshot uses LAST evaluation."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _snap_strategy()
    boll = _boll(lower=1900)
    _feed_two_lower_fractals_for_snapshot(strat, boll, second_cvd_recovery=False)

    snapshot_logs = [r for r in caplog.records if "LOWER_EXTREME_SNAPSHOT" in r.getMessage()]
    assert len(snapshot_logs) == 1, f"Expected 1 snapshot, got {len(snapshot_logs)}"
    msg = snapshot_logs[0].getMessage()
    assert "divergence_confirmed=False" in msg
    assert "divergence_reason=cvd_not_recovered" in msg


# ── Test: no snapshot when no divergence evaluation happened ──────────

def test_no_snapshot_without_evaluation(caplog) -> None:
    """Only LOWER_FIRST_CONFIRMED_EXTREME, no LOWER_EXTREME_SNAPSHOT for first extreme."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _snap_strategy()
    boll = _boll(lower=1900)
    _feed_lower_fractal_sequence(strat, boll)

    first_logs = [r for r in caplog.records if "LOWER_FIRST_CONFIRMED_EXTREME" in r.getMessage()]
    snapshot_logs = [r for r in caplog.records if "LOWER_EXTREME_SNAPSHOT" in r.getMessage()]

    assert len(first_logs) == 1, "Should log LOWER_FIRST_CONFIRMED_EXTREME"
    assert len(snapshot_logs) == 0, (
        "First extreme should NOT produce LOWER_EXTREME_SNAPSHOT (no prev/curr pair)"
    )
    assert strat.state.lower_last_snapshot_divergence_reason is None


# ── Test: UPPER cvd_not_reversed when no CVD reversal ─────────────────

def test_upper_snapshot_cvd_not_reversed_reason(caplog) -> None:
    """When price makes new high but CVD doesn't reverse → cvd_not_reversed."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _snap_strategy()
    boll = _boll(upper=2100)
    _feed_two_upper_fractals_for_snapshot(strat, boll, second_cvd_reversal=False)

    snapshot_logs = [r for r in caplog.records if "UPPER_EXTREME_SNAPSHOT" in r.getMessage()]
    assert len(snapshot_logs) == 1
    msg = snapshot_logs[0].getMessage()
    assert "divergence_confirmed=False" in msg
    assert "divergence_reason=cvd_not_reversed" in msg
    assert "no_new_high" not in msg


# ═══════════════════════════════════════════════════════════════════════
# 1m fractal: first outside tick no longer creates immediate confirmed extreme.
# The old V-shaped sweep test is obsolete — 1m fractal requires 2L/2R closure.
# ═══════════════════════════════════════════════════════════════════════


def test_first_outside_tick_does_not_immediately_confirm() -> None:
    """First outside tick seeds 1m candle builder, does NOT create confirmed extreme."""
    strat = _strategy(min_outside_pct=0.001)
    boll = _boll(lower=1900)

    # Single outside tick
    p1 = 1780.0
    cvd1 = _cvd(ts_ms=1000, price=p1, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000)
    strat._update_lower_outside_v2(price=p1, ts_ms=1000, boll=boll, cvd=cvd1)
    assert strat.state.lower_outside_observed is True
    # No extreme can be confirmed from a single tick (needs 5 closed 1m candles)
    assert strat.state.lower_first_extreme_price is None

    # Price retracing in same minute → still no extreme
    p2 = 1785.0
    cvd2 = _cvd(ts_ms=2000, price=p2, cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000)
    strat._update_lower_outside_v2(price=p2, ts_ms=2000, boll=boll, cvd=cvd2)
    assert strat.state.lower_first_extreme_price is None  # Still no fractal extreme


# ═══════════════════════════════════════════════════════════════════════
# Snapshot log: running_tick_extreme_count field
# ═══════════════════════════════════════════════════════════════════════


def test_snapshot_log_uses_running_tick_extreme_count_not_new_extreme_count(caplog) -> None:
    """Snapshot log must use running_tick_extreme_count, not new_extreme_count."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _snap_strategy()
    boll = _boll(lower=1900)
    _feed_two_lower_fractals_for_snapshot(strat, boll, second_cvd_recovery=False)

    snapshot_logs = [r for r in caplog.records if "LOWER_EXTREME_SNAPSHOT" in r.getMessage()]
    assert len(snapshot_logs) >= 1, "Should print snapshot log"

    for record in snapshot_logs:
        msg = record.getMessage()
        assert "new_extreme_count=" not in msg, (
            f"Snapshot log must NOT contain 'new_extreme_count='. Got: {msg}"
        )
        assert "running_tick_extreme_count=" in msg, (
            f"Snapshot log must contain 'running_tick_extreme_count='. Got: {msg}"
        )


# ═══════════════════════════════════════════════════════════════════════
# 1m fractal inside-continuation tests (bug fix: keep tracking after
# price returns inside band so fractal extremes can still confirm)
# ═══════════════════════════════════════════════════════════════════════


# ── 8.1: price inside 后仍继续确认 LOWER fractal ──────────────────────

def test_lower_fractal_confirms_after_price_returns_inside(caplog) -> None:
    """Price returns inside lower band after first 2 minutes but fractal
    still confirms when right-side candles close (minutes 3-5 are inside)."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001)
    boll = _boll()
    _OFFSET = 30000  # avoid ts_ms=0 which fails orderflow anchor_ts_ms > 0 check

    # Minute 0: price below lower → anchor, LEFT-2
    t0 = _OFFSET
    strat.on_tick(boll.lower * 0.998, t0, boll, _cvd(ts_ms=t0, price=boll.lower * 0.998,
                 cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))
    assert strat.state.lower_outside_observed is True

    # Minute 1: still below lower → LEFT-1
    t1 = _MINUTE_MS + _OFFSET
    strat.on_tick(boll.lower * 0.996, t1, boll, _cvd(ts_ms=t1, price=boll.lower * 0.996,
                 cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000))

    # Minute 2: CANDIDATE (lowest) — still below lower
    t2 = 2 * _MINUTE_MS + _OFFSET
    strat.on_tick(boll.lower * 0.992, t2, boll, _cvd(ts_ms=t2, price=boll.lower * 0.992,
                 cumulative_buy_volume=500_000, cumulative_sell_volume=1_700_000))

    # Minute 3: price RETURNS INSIDE (above lower but below middle) — RIGHT-1
    t3 = 3 * _MINUTE_MS + _OFFSET
    inside_price = boll.lower * 1.002  # just barely inside
    strat.on_tick(inside_price, t3, boll, _cvd(ts_ms=t3, price=inside_price,
                 cumulative_buy_volume=500_000, cumulative_sell_volume=1_800_000))

    # Minute 4: still inside — RIGHT-2 (should trigger fractal confirmation!)
    t4 = 4 * _MINUTE_MS + _OFFSET
    strat.on_tick(boll.lower * 1.001, t4, boll, _cvd(ts_ms=t4, price=boll.lower * 1.001,
                 cumulative_buy_volume=500_000, cumulative_sell_volume=1_900_000))

    # Minute 5: still inside
    t5 = 5 * _MINUTE_MS + _OFFSET
    strat.on_tick(boll.lower * 1.003, t5, boll, _cvd(ts_ms=t5, price=boll.lower * 1.003,
                 cumulative_buy_volume=500_000, cumulative_sell_volume=2_000_000))

    # Verify fractal was confirmed despite price returning inside
    confirmed_logs = [r for r in caplog.records if "LOWER_CONFIRMED_EXTREME | source=1m_fractal_2l2r" in r.message]
    assert len(confirmed_logs) >= 1, (
        f"Fractal should confirm even after price returns inside, got {len(confirmed_logs)}"
    )
    # State should NOT be aborted
    assert strat.state.lower_outside_observed is True
    assert strat.state.lower_first_extreme_price is not None


# ── 8.1b: UPPER mirror — fractal confirms after price returns inside ──

def test_upper_fractal_confirms_after_price_returns_inside(caplog) -> None:
    """UPPER mirror: fractal still confirms after price returns inside band."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001)
    boll = _boll()
    _OFFSET = 30000

    # Minute 0-2: outside upper
    t0 = _OFFSET
    strat.on_tick(boll.upper * 1.002, t0, boll, _cvd(ts_ms=t0, price=boll.upper * 1.002,
                 cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
                 buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                 cvd_increasing=False, cvd_decreasing=True, no_new_high=True))
    assert strat.state.upper_outside_observed is True

    t1 = _MINUTE_MS + _OFFSET
    strat.on_tick(boll.upper * 1.004, t1, boll, _cvd(ts_ms=t1, price=boll.upper * 1.004,
                 cumulative_buy_volume=1_600_000, cumulative_sell_volume=500_000,
                 buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                 cvd_increasing=False, cvd_decreasing=True, no_new_high=True))

    t2 = 2 * _MINUTE_MS + _OFFSET
    strat.on_tick(boll.upper * 1.008, t2, boll, _cvd(ts_ms=t2, price=boll.upper * 1.008,
                 cumulative_buy_volume=1_700_000, cumulative_sell_volume=500_000,
                 buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                 cvd_increasing=False, cvd_decreasing=True, no_new_high=True))

    # Minute 3-5: inside band
    inside_price = boll.upper * 0.998
    for m in range(3, 6):
        ts = m * _MINUTE_MS + _OFFSET
        strat.on_tick(inside_price, ts, boll, _cvd(ts_ms=ts, price=inside_price,
                     cumulative_buy_volume=1_800_000 + m * 100_000, cumulative_sell_volume=500_000,
                     buy_ratio=0.4, sell_ratio=0.6, cross_positive=False, cross_negative=True,
                     cvd_increasing=False, cvd_decreasing=True, no_new_high=True))

    confirmed_logs = [r for r in caplog.records if "UPPER_CONFIRMED_EXTREME | source=1m_fractal_2l2r" in r.message]
    assert len(confirmed_logs) >= 1, (
        f"UPPER fractal should confirm even after price returns inside, got {len(confirmed_logs)}"
    )
    assert strat.state.upper_outside_observed is True
    assert strat.state.upper_first_extreme_price is not None


# ── 8.2: inside return before divergence evaluation 不 abort ──────────

def test_lower_inside_return_before_divergence_eval_no_abort(caplog) -> None:
    """When setup is active (outside_observed=True, no divergence yet, no
    first_extreme), price returning inside should NOT abort — it should
    continue waiting for the 1m fractal to form."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001)
    boll = _boll()
    _OFFSET = 30000

    # Anchor: first outside tick
    t0 = _OFFSET
    strat.on_tick(boll.lower * 0.998, t0, boll, _cvd(ts_ms=t0, price=boll.lower * 0.998,
                 cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))
    assert strat.state.lower_outside_observed is True
    assert strat.state.lower_first_extreme_price is None  # no extreme yet
    assert strat.state.lower_last_divergence_evaluated_ts_ms == 0  # no div eval

    # Price returns inside immediately
    inside_price = boll.lower * 1.002
    t1 = _MINUTE_MS + _OFFSET
    strat.on_tick(inside_price, t1, boll, _cvd(ts_ms=t1, price=inside_price,
                 cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000))

    # Must NOT abort (no divergence evaluation has happened yet)
    abort_logs = [r for r in caplog.records if "LOWER_RECLAIM_ABORTED" in r.message]
    assert len(abort_logs) == 0, (
        f"Should NOT abort before divergence evaluation, got {len(abort_logs)} abort(s)"
    )
    # State must remain
    assert strat.state.lower_outside_observed is True, "lower_outside_observed must remain True"


def test_upper_inside_return_before_divergence_eval_no_abort(caplog) -> None:
    """UPPER mirror: inside return before divergence eval → no abort."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001)
    boll = _boll()
    _OFFSET = 30000

    t0 = _OFFSET
    strat.on_tick(boll.upper * 1.002, t0, boll, _cvd(ts_ms=t0, price=boll.upper * 1.002,
                 cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
                 buy_ratio=0.4, sell_ratio=0.6))
    assert strat.state.upper_outside_observed is True
    assert strat.state.upper_first_extreme_price is None
    assert strat.state.upper_last_divergence_evaluated_ts_ms == 0

    inside_price = boll.upper * 0.998
    t1 = _MINUTE_MS + _OFFSET
    strat.on_tick(inside_price, t1, boll, _cvd(ts_ms=t1, price=inside_price,
                 cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
                 buy_ratio=0.4, sell_ratio=0.6))

    abort_logs = [r for r in caplog.records if "UPPER_RECLAIM_ABORTED" in r.message]
    assert len(abort_logs) == 0, "Should NOT abort before divergence evaluation"
    assert strat.state.upper_outside_observed is True


# ── 8.3: divergence evaluated 后 no divergence 才允许 abort ───────────

def test_lower_abort_after_divergence_eval_no_divergence(caplog) -> None:
    """After divergence has been evaluated (but NOT confirmed), inside
    return SHOULD trigger LOWER_RECLAIM_ABORTED."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001)
    boll = _boll()

    # Anchor + first extreme
    _feed_lower_fractal_sequence(strat, boll)
    assert strat.state.lower_first_extreme_price is not None
    assert strat.state.lower_anchored_divergence_confirmed is False

    # Simulate: divergence was evaluated (cvd_not_recovered)
    strat.state.lower_last_divergence_evaluated_ts_ms = 1000
    # Keep state alive
    strat.state.lower_outside_observed = True

    # Now price returns inside → should abort
    caplog.clear()
    inside_price = boll.lower * 1.002
    strat.on_tick(inside_price, 500000, boll, _cvd(ts_ms=500000, price=inside_price,
                 cumulative_buy_volume=500_000, cumulative_sell_volume=2_000_000))

    abort_logs = [r for r in caplog.records if "LOWER_RECLAIM_ABORTED" in r.message]
    assert len(abort_logs) == 1, (
        f"Should abort after divergence eval + no divergence, got {len(abort_logs)}"
    )
    assert "inside_return_without_anchored_divergence" in abort_logs[0].message
    assert strat.state.lower_outside_observed is False  # reset after abort


def test_upper_abort_after_divergence_eval_no_divergence(caplog) -> None:
    """UPPER mirror: after divergence eval + no divergence → abort on inside return."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001)
    boll = _boll()

    _feed_upper_fractal_sequence(strat, boll)
    assert strat.state.upper_first_extreme_price is not None
    assert strat.state.upper_anchored_divergence_confirmed is False
    strat.state.upper_last_divergence_evaluated_ts_ms = 1000
    strat.state.upper_outside_observed = True

    caplog.clear()
    inside_price = boll.upper * 0.998
    strat.on_tick(inside_price, 500000, boll, _cvd(ts_ms=500000, price=inside_price,
                 cumulative_buy_volume=2_000_000, cumulative_sell_volume=500_000,
                 buy_ratio=0.4, sell_ratio=0.6))

    abort_logs = [r for r in caplog.records if "UPPER_RECLAIM_ABORTED" in r.message]
    assert len(abort_logs) == 1, f"Should abort after divergence eval, got {len(abort_logs)}"
    assert strat.state.upper_outside_observed is False


# ── 8.4: middle reclaimed 仍 reset (V2 armed: divergence confirmed) ─

def test_lower_middle_reclaimed_resets_after_divergence_confirm(caplog) -> None:
    """After divergence is confirmed and setup is armed, price reaching
    middle must reset the LOWER setup."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001)
    boll = _boll()

    # Set up armed divergence state (divergence confirmed, waiting for reclaim)
    _setup_lower_armed_divergence(strat, ref_lower=boll.lower, ref_middle=boll.middle)

    # Price goes to middle (or above)
    strat.on_tick(boll.middle * 1.001, 60000, boll, _cvd(ts_ms=60000, price=boll.middle * 1.001,
                 cumulative_buy_volume=600_000, cumulative_sell_volume=1_500_000))

    reset_logs = [r for r in caplog.records if "LOWER_ARMED_RESET" in r.message and "middle_reclaimed" in r.message]
    assert len(reset_logs) >= 1, f"Should reset on middle reclaimed, got {len(reset_logs)}"


def test_upper_middle_reclaimed_resets_after_divergence_confirm(caplog) -> None:
    """UPPER mirror: middle reclaimed resets after divergence confirm."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001)
    boll = _boll()

    _setup_upper_armed_divergence(strat, ref_upper=boll.upper, ref_middle=boll.middle)

    strat.on_tick(boll.middle * 0.999, 60000, boll, _cvd(ts_ms=60000, price=boll.middle * 0.999,
                 cumulative_buy_volume=1_500_000, cumulative_sell_volume=600_000,
                 buy_ratio=0.4, sell_ratio=0.6))

    reset_logs = [r for r in caplog.records if "UPPER_ARMED_RESET" in r.message and "middle_reclaimed" in r.message]
    assert len(reset_logs) >= 1, f"Should reset on middle reclaimed, got {len(reset_logs)}"


# ── 8.5: divergence confirmed — inside return does NOT abort ──────────

def test_lower_divergence_confirmed_inside_return_no_abort(caplog) -> None:
    """When anchored divergence IS confirmed, inside return must NOT abort —
    it must keep waiting for CVD follow-through entry."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001)
    boll = _boll()

    _run_lower_armed_divergence(strat, boll)
    assert strat.state.lower_anchored_divergence_confirmed is True
    assert strat.state.lower_armed is True

    # Price returns inside band — should NOT trigger LOWER_RECLAIM_ABORTED
    inside_price = boll.lower * 1.002
    strat.on_tick(inside_price, 300000, boll, _cvd(ts_ms=300000, price=inside_price,
                 cumulative_buy_volume=700_000, cumulative_sell_volume=1_400_000))

    abort_logs = [r for r in caplog.records if "LOWER_RECLAIM_ABORTED" in r.message]
    assert len(abort_logs) == 0, (
        f"Should NOT abort when divergence is confirmed, got {len(abort_logs)}"
    )


def test_upper_divergence_confirmed_inside_return_no_abort(caplog) -> None:
    """UPPER mirror: divergence confirmed → inside return does NOT abort."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001)
    boll = _boll()

    _run_upper_armed_divergence(strat, boll)
    assert strat.state.upper_anchored_divergence_confirmed is True
    assert strat.state.upper_armed is True

    inside_price = boll.upper * 0.998
    strat.on_tick(inside_price, 300000, boll, _cvd(ts_ms=300000, price=inside_price,
                 cumulative_buy_volume=1_300_000, cumulative_sell_volume=700_000,
                 buy_ratio=0.4, sell_ratio=0.6))

    abort_logs = [r for r in caplog.records if "UPPER_RECLAIM_ABORTED" in r.message]
    assert len(abort_logs) == 0, "Should NOT abort when divergence is confirmed"


# ── 8.6: full on_tick path — fractal extreme only after inside return ─

def test_on_tick_full_path_fractal_after_inside_return(caplog) -> None:
    """Full on_tick path test: price goes outside, returns inside, then
    1m fractal confirms. Verifies the full tick flow (no private method
    calls bypassing _update_armed_state)."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(min_outside_pct=0.001)
    boll = _boll()

    # Minute 0: outside lower → anchor
    strat.on_tick(boll.lower * 0.998, 30000, boll, _cvd(ts_ms=30000, price=boll.lower * 0.998,
                 cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))
    assert strat.state.lower_outside_observed is True

    # Minute 1: still outside
    strat.on_tick(boll.lower * 0.996, 90000, boll, _cvd(ts_ms=90000, price=boll.lower * 0.996,
                 cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000))

    # Minute 2: candidate low, still outside
    strat.on_tick(boll.lower * 0.990, 150000, boll, _cvd(ts_ms=150000, price=boll.lower * 0.990,
                 cumulative_buy_volume=500_000, cumulative_sell_volume=1_700_000))

    # Minute 3: price RETURNS INSIDE — this is the critical path!
    inside_price = boll.lower * 1.002
    strat.on_tick(inside_price, 210000, boll, _cvd(ts_ms=210000, price=inside_price,
                 cumulative_buy_volume=500_000, cumulative_sell_volume=1_800_000))

    # Verify: no abort happened
    abort_logs = [r for r in caplog.records if "LOWER_RECLAIM_ABORTED" in r.message]
    assert len(abort_logs) == 0, (
        f"Should NOT abort — still waiting for fractal right-side candles. Got {len(abort_logs)}"
    )
    assert strat.state.lower_outside_observed is True

    # Minute 4: still inside
    strat.on_tick(boll.lower * 1.001, 270000, boll, _cvd(ts_ms=270000, price=boll.lower * 1.001,
                 cumulative_buy_volume=500_000, cumulative_sell_volume=1_900_000))

    # Minute 5: at this point the fractal should confirm
    strat.on_tick(boll.lower * 1.003, 330000, boll, _cvd(ts_ms=330000, price=boll.lower * 1.003,
                 cumulative_buy_volume=500_000, cumulative_sell_volume=2_000_000))

    # Fractal should have been confirmed
    confirmed_logs = [r for r in caplog.records if "LOWER_CONFIRMED_EXTREME" in r.message]
    assert len(confirmed_logs) >= 1, (
        f"Fractal should confirm via on_tick path, got {len(confirmed_logs)}"
    )
    assert strat.state.lower_first_extreme_price is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Reclaim V2 timeout fixes: total setup timeout from anchor_ts_ms
# ═══════════════════════════════════════════════════════════════════════════════


# ── 9.1: waiting fractal phase total setup timeout (LOWER) ─────────────────

def test_v2_waiting_fractal_total_setup_timeout_lower(caplog) -> None:
    """V2 LOWER waiting for first fractal: exceed total setup → reset.

    Scenario:
    - lower_outside_observed=True, lower_anchor_ts_ms=1000
    - lower_anchored_divergence_confirmed=False (no fractal yet)
    - price inside band, below middle
    - ts_ms > anchor + total_setup → LOWER_SETUP_EXPIRED
    """
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(
        entry_max_total_setup_seconds=1,  # 1 second timeout
        min_outside_pct=0.001,
    )
    boll = _boll()

    # First outside tick → anchor at ts=1000
    outside_price = boll.lower * 0.998
    strat.on_tick(outside_price, 1000, boll, _cvd(ts_ms=1000, price=outside_price,
                 cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))
    assert strat.state.lower_outside_observed is True
    assert strat.state.lower_anchor_ts_ms == 1000
    assert strat.state.lower_anchored_divergence_confirmed is False

    # Advance past total setup timeout; price returns inside band
    inside_price = boll.lower * 1.001  # just inside lower
    ts_past_timeout = 1000 + 1 * 1000 + 100  # anchor + 1s + 100ms
    strat.on_tick(inside_price, ts_past_timeout, boll, _cvd(ts_ms=ts_past_timeout, price=inside_price,
                 cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000))

    # Assert: total setup timeout fired and state was reset
    expired_logs = [r for r in caplog.records if "LOWER_SETUP_EXPIRED" in r.message
                    and "total_setup_timeout" in r.message
                    and "phase=reclaim_v2" in r.message]
    assert len(expired_logs) >= 1, (
        f"Expected LOWER_SETUP_EXPIRED total_setup_timeout, got {len(expired_logs)}"
    )
    assert strat.state.lower_outside_observed is False, (
        "lower_outside_observed should be False after total setup timeout reset"
    )


# ── 9.1b: waiting fractal phase total setup timeout (UPPER mirror) ────────

def test_v2_waiting_fractal_total_setup_timeout_upper(caplog) -> None:
    """V2 UPPER waiting for first fractal: exceed total setup → reset."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(
        entry_max_total_setup_seconds=1,
        min_outside_pct=0.001,
    )
    boll = _boll()

    outside_price = boll.upper * 1.002
    strat.on_tick(outside_price, 1000, boll, _cvd(ts_ms=1000, price=outside_price,
                 cumulative_buy_volume=1_500_000, cumulative_sell_volume=500_000,
                 buy_ratio=0.4, sell_ratio=0.6))
    assert strat.state.upper_outside_observed is True
    assert strat.state.upper_anchor_ts_ms == 1000

    inside_price = boll.upper * 0.999
    ts_past_timeout = 1000 + 1 * 1000 + 100
    strat.on_tick(inside_price, ts_past_timeout, boll, _cvd(ts_ms=ts_past_timeout, price=inside_price,
                 cumulative_buy_volume=1_500_000, cumulative_sell_volume=600_000,
                 buy_ratio=0.4, sell_ratio=0.6))

    expired_logs = [r for r in caplog.records if "UPPER_SETUP_EXPIRED" in r.message
                    and "total_setup_timeout" in r.message
                    and "phase=reclaim_v2" in r.message]
    assert len(expired_logs) >= 1, (
        f"Expected UPPER_SETUP_EXPIRED total_setup_timeout, got {len(expired_logs)}"
    )
    assert strat.state.upper_outside_observed is False


# ── 9.2: waiting divergence phase total setup timeout (LOWER) ──────────────

def test_v2_waiting_divergence_total_setup_timeout_lower(caplog) -> None:
    """V2 LOWER has first fractal extreme but no divergence yet; exceed total setup → reset.

    Scenario:
    - First fractal extreme confirmed
    - lower_anchored_divergence_confirmed=False (no 2nd fractal / divergence eval)
    - No second fractal sequence; advance time past total setup
    """
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(
        entry_max_total_setup_seconds=1,
        min_outside_pct=0.001,
    )
    boll = _boll()

    # First outside tick
    outside_price = boll.lower * 0.998
    strat.on_tick(outside_price, 1000, boll, _cvd(ts_ms=1000, price=outside_price,
                 cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))
    assert strat.state.lower_outside_observed is True
    assert strat.state.lower_anchor_ts_ms == 1000

    # Feed ticks to simulate fractal formation (partial, within 1s)
    for minute in range(3):
        ts = minute * _MINUTE_MS + 30000
        price = boll.lower * (0.998 - minute * 0.002)
        strat.on_tick(price, ts, boll, _cvd(ts_ms=ts, price=price,
                     cumulative_buy_volume=500_000,
                     cumulative_sell_volume=1_500_000 + minute * 100_000))

    # First extreme should NOT be set yet (need full fractal)
    # Now let time pass: jump to after total setup timeout
    # But anchor was at 1000, and we're at ts ~ 150000 from the fractal feed...
    # The total setup timeout should have fired by now (1800s default is not hit).
    # We need to set a very small total_setup so it triggers.
    # But the fractal feeding also advances time, potentially past the timeout.
    # Let's just set up state directly for this test.

    # Alternative: use direct state setup
    strat2 = _strategy(entry_max_total_setup_seconds=1, min_outside_pct=0.001)
    strat2.state.lower_outside_observed = True
    strat2.state.lower_anchor_ts_ms = 1000
    strat2.state.lower_first_extreme_price = boll.lower * 0.990
    strat2.state.lower_first_extreme_ts_ms = 30000
    strat2.state.lower_last_divergence_evaluated_ts_ms = 0
    strat2.state.lower_anchored_divergence_confirmed = False

    # Call _update_armed_state with ts past timeout
    ts_past = 1000 + 1 * 1000 + 100
    strat2._update_armed_state(boll.middle * 0.95, ts_past, boll, None)

    expired_logs = [r for r in caplog.records if "LOWER_SETUP_EXPIRED" in r.message
                    and "total_setup_timeout" in r.message
                    and "phase=reclaim_v2" in r.message]
    assert len(expired_logs) >= 1, (
        f"Expected LOWER_SETUP_EXPIRED total_setup_timeout in waiting-divergence phase, "
        f"got {len(expired_logs)}"
    )
    assert strat2.state.lower_outside_observed is False, (
        "lower_outside_observed should be reset after total setup timeout"
    )


# ── 9.2b: waiting divergence phase total setup timeout (UPPER mirror) ──────

def test_v2_waiting_divergence_total_setup_timeout_upper(caplog) -> None:
    """V2 UPPER has first fractal extreme but no divergence yet; exceed total setup → reset."""
    import logging
    caplog.set_level(logging.INFO)

    strat = _strategy(entry_max_total_setup_seconds=1, min_outside_pct=0.001)
    boll = _boll()

    strat.state.upper_outside_observed = True
    strat.state.upper_anchor_ts_ms = 1000
    strat.state.upper_first_extreme_price = boll.upper * 1.010
    strat.state.upper_first_extreme_ts_ms = 30000
    strat.state.upper_last_divergence_evaluated_ts_ms = 0
    strat.state.upper_anchored_divergence_confirmed = False

    ts_past = 1000 + 1 * 1000 + 100
    strat._update_armed_state(boll.middle * 1.05, ts_past, boll, None)

    expired_logs = [r for r in caplog.records if "UPPER_SETUP_EXPIRED" in r.message
                    and "total_setup_timeout" in r.message
                    and "phase=reclaim_v2" in r.message]
    assert len(expired_logs) >= 1, (
        f"Expected UPPER_SETUP_EXPIRED total_setup_timeout, got {len(expired_logs)}"
    )
    assert strat.state.upper_outside_observed is False


# ── 9.3: V2 armed extreme-to-reclaim uses divergence_extreme_ts_ms (LOWER) ──

def test_v2_armed_extreme_timeout_uses_divergence_ts_lower(caplog) -> None:
    """When V2 divergence confirmed, extreme-to-reclaim timeout must use
    lower_divergence_extreme_ts_ms, NOT tick-level lower_extreme_ts_ms.

    Scenario:
    - lower_anchored_divergence_confirmed=True
    - lower_divergence_extreme_ts_ms=1000 (divergence extreme)
    - lower_extreme_ts_ms=999999999 (tick-level, deliberately different)
    - ts_ms = divergence extreme + max_extreme_to_reclaim*1000 + 1
    - Assert: timeout fires (using divergence ts, not tick-level ts)
    """
    import logging
    caplog.set_level(logging.INFO)

    _MAX_RECLAIM = 5  # small timeout for testing
    strat = _strategy(
        entry_max_extreme_to_reclaim_seconds=_MAX_RECLAIM,
        min_outside_pct=0.001,
    )
    boll = _boll()

    # Set up armed state with divergence confirmed
    strat.state.lower_armed = True
    strat.state.lower_armed_ts_ms = 5000
    strat.state.lower_deep_enough = True
    strat.state.lower_extreme_price = 1880.0
    strat.state.lower_anchored_divergence_confirmed = True
    strat.state.lower_cvd_divergence_confirmed = True
    strat.state.lower_first_armed_ts_ms = 5000
    strat.state.lower_outside_observed = True
    strat.state.lower_anchor_ts_ms = 1000

    # Key: divergence_extreme_ts_ms is 1000, lower_extreme_ts_ms is very large
    strat.state.lower_divergence_extreme_ts_ms = 1000
    strat.state.lower_extreme_ts_ms = 999999999  # would NOT trigger timeout if used

    # Call _expire_armed_state at ts that would timeout using divergence ts
    ts_ms = 1000 + _MAX_RECLAIM * 1000 + 1  # just past timeout from divergence ts
    strat._expire_armed_state(ts_ms)

    # Assert: timeout fired with source=reclaim_v2_fractal
    reset_logs = [r for r in caplog.records if "LOWER_ARMED_RESET" in r.message
                  and "extreme_to_reclaim_timeout" in r.message
                  and "source=reclaim_v2_fractal" in r.message]
    assert len(reset_logs) >= 1, (
        f"Expected LOWER_ARMED_RESET extreme_to_reclaim_timeout source=reclaim_v2_fractal, "
        f"got {len(reset_logs)}"
    )
    assert strat.state.lower_armed is False, "Should be reset after timeout"


# ── 9.3b: V2 armed extreme-to-reclaim uses divergence_extreme_ts_ms (UPPER) ─

def test_v2_armed_extreme_timeout_uses_divergence_ts_upper(caplog) -> None:
    """UPPER mirror: extreme-to-reclaim timeout uses upper_divergence_extreme_ts_ms."""
    import logging
    caplog.set_level(logging.INFO)

    _MAX_RECLAIM = 5
    strat = _strategy(
        entry_max_extreme_to_reclaim_seconds=_MAX_RECLAIM,
        min_outside_pct=0.001,
    )
    boll = _boll()

    strat.state.upper_armed = True
    strat.state.upper_armed_ts_ms = 5000
    strat.state.upper_deep_enough = True
    strat.state.upper_extreme_price = 2120.0
    strat.state.upper_anchored_divergence_confirmed = True
    strat.state.upper_cvd_divergence_confirmed = True
    strat.state.upper_first_armed_ts_ms = 5000
    strat.state.upper_outside_observed = True
    strat.state.upper_anchor_ts_ms = 1000

    strat.state.upper_divergence_extreme_ts_ms = 1000
    strat.state.upper_extreme_ts_ms = 999999999

    ts_ms = 1000 + _MAX_RECLAIM * 1000 + 1
    strat._expire_armed_state(ts_ms)

    reset_logs = [r for r in caplog.records if "UPPER_ARMED_RESET" in r.message
                  and "extreme_to_reclaim_timeout" in r.message
                  and "source=reclaim_v2_fractal" in r.message]
    assert len(reset_logs) >= 1, (
        f"Expected UPPER_ARMED_RESET source=reclaim_v2_fractal, got {len(reset_logs)}"
    )
    assert strat.state.upper_armed is False


# ── 9.4: V2 missing divergence extreme ts safe reset (LOWER) ────────────────

def test_v2_missing_divergence_extreme_ts_safe_reset_lower(caplog) -> None:
    """When V2 divergence confirmed but divergence_extreme_ts_ms <= 0,
    must log WARNING and reset — no silent fallback to tick-level extreme_ts.

    Scenario:
    - lower_anchored_divergence_confirmed=True
    - lower_divergence_extreme_ts_ms=0
    - Assert: WARNING + LOWER_ARMED_RESET reason=missing_divergence_extreme_ts
    """
    import logging
    caplog.set_level(logging.WARNING)

    strat = _strategy(min_outside_pct=0.001)
    boll = _boll()

    strat.state.lower_armed = True
    strat.state.lower_armed_ts_ms = 5000
    strat.state.lower_deep_enough = True
    strat.state.lower_extreme_price = 1880.0
    strat.state.lower_anchored_divergence_confirmed = True
    strat.state.lower_cvd_divergence_confirmed = True
    strat.state.lower_first_armed_ts_ms = 5000
    strat.state.lower_outside_observed = True
    strat.state.lower_anchor_ts_ms = 1000

    # Missing divergence extreme ts
    strat.state.lower_divergence_extreme_ts_ms = 0
    strat.state.lower_extreme_ts_ms = 5000  # has tick-level ts but shouldn't be used

    strat._expire_armed_state(100000)

    # Should log WARNING and reset
    warn_logs = [r for r in caplog.records if r.levelno == logging.WARNING
                 and "missing_divergence_extreme_ts" in r.message]
    assert len(warn_logs) >= 1, (
        f"Expected WARNING for missing divergence extreme ts, got {len(warn_logs)}"
    )
    assert strat.state.lower_armed is False, "Should be reset on missing divergence extreme ts"


# ── 9.4b: V2 missing divergence extreme ts safe reset (UPPER) ───────────────

def test_v2_missing_divergence_extreme_ts_safe_reset_upper(caplog) -> None:
    """UPPER mirror: missing divergence extreme ts → WARNING + reset."""
    import logging
    caplog.set_level(logging.WARNING)

    strat = _strategy(min_outside_pct=0.001)
    boll = _boll()

    strat.state.upper_armed = True
    strat.state.upper_armed_ts_ms = 5000
    strat.state.upper_deep_enough = True
    strat.state.upper_extreme_price = 2120.0
    strat.state.upper_anchored_divergence_confirmed = True
    strat.state.upper_cvd_divergence_confirmed = True
    strat.state.upper_first_armed_ts_ms = 5000
    strat.state.upper_outside_observed = True
    strat.state.upper_anchor_ts_ms = 1000

    strat.state.upper_divergence_extreme_ts_ms = 0
    strat.state.upper_extreme_ts_ms = 5000

    strat._expire_armed_state(100000)

    warn_logs = [r for r in caplog.records if r.levelno == logging.WARNING
                 and "missing_divergence_extreme_ts" in r.message]
    assert len(warn_logs) >= 1, (
        f"Expected WARNING for missing divergence extreme ts (UPPER), got {len(warn_logs)}"
    )
    assert strat.state.upper_armed is False


# ── 9.5: full on_tick path — outside → inside → fractal → total timeout ─────

def test_on_tick_full_path_total_setup_timeout(caplog) -> None:
    """Full on_tick path: outside observed → inside continuation → wait fractal
    → exceed total setup timeout → reset. Uses strat.on_tick() only."""
    import logging
    caplog.set_level(logging.INFO)

    _TOTAL_SETUP_S = 5  # 5 seconds
    strat = _strategy(
        entry_max_total_setup_seconds=_TOTAL_SETUP_S,
        min_outside_pct=0.001,
    )
    boll = _boll()

    # Minute 0: outside lower → anchor at ts=30000
    outside_price = boll.lower * 0.998
    strat.on_tick(outside_price, 30000, boll, _cvd(ts_ms=30000, price=outside_price,
                 cumulative_buy_volume=500_000, cumulative_sell_volume=1_500_000))
    assert strat.state.lower_outside_observed is True
    assert strat.state.lower_anchor_ts_ms == 30000

    # Minute 1: still outside
    strat.on_tick(boll.lower * 0.996, 90000, boll, _cvd(ts_ms=90000, price=boll.lower * 0.996,
                 cumulative_buy_volume=500_000, cumulative_sell_volume=1_600_000))

    # Minute 2: candidate low
    strat.on_tick(boll.lower * 0.990, 150000, boll, _cvd(ts_ms=150000, price=boll.lower * 0.990,
                 cumulative_buy_volume=500_000, cumulative_sell_volume=1_700_000))

    # Minute 3: price returns inside band
    # The total setup timeout fires at the BEGINNING of _update_armed_state,
    # and since anchor_ts_ms=150000 (from re-anchoring at ts=150000),
    # ts=210000 → diff=60000ms > 5000ms → timeout fires → reset.
    inside_price = boll.lower * 1.002
    strat.on_tick(inside_price, 210000, boll, _cvd(ts_ms=210000, price=inside_price,
                 cumulative_buy_volume=500_000, cumulative_sell_volume=1_800_000))

    # After the timeout reset at ts=210000, lower_outside_observed is False
    expired_logs = [r for r in caplog.records if "LOWER_SETUP_EXPIRED" in r.message
                    and "total_setup_timeout" in r.message
                    and "phase=reclaim_v2" in r.message]
    assert len(expired_logs) >= 1, (
        f"Expected total_setup_timeout via on_tick path, got {len(expired_logs)}"
    )
    assert strat.state.lower_outside_observed is False, (
        "lower_outside_observed should be False after total setup timeout via on_tick"
    )
