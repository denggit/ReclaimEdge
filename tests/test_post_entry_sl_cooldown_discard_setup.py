"""Tests for post-entry SL cooldown setup discard behaviour.

Covers:
1. SIDE LONG cooldown → lower setup discarded, upper preserved
2. SIDE SHORT cooldown → upper setup discarded, lower preserved
3. GLOBAL cooldown → both setups discarded
4. Cooldown expired → must wait for fresh setup (old setup not reused)
5. Log throttling — DISCARDED log ≤1 per 60s per key
6. _post_entry_sl_cooldown_ok is a pure gate (no ACTIVE logging)
"""

from __future__ import annotations

import logging

import pytest

from src.indicators.cvd_tracker import CvdSnapshot
from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
)
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig


# ======================================================================
# Helpers
# ======================================================================


def _boll(
    middle: float = 2000,
    upper: float = 2100,
    lower: float = 1900,
    alert_switch_on: bool = True,
) -> BollSnapshot:
    return BollSnapshot(
        inst_id="ETH-USDT-SWAP",
        candle_ts_ms=1000,
        close=middle,
        middle=middle,
        upper=upper,
        lower=lower,
        upper_distance_pct=0.0,
        lower_distance_pct=0.001,
        alert_switch_on=alert_switch_on,
        live_mode=True,
    )


def _cvd(
    ts_ms: int = 1000,
    price: float = 1901,
    fast_cvd: float = 1.0,
    previous_fast_cvd: float = 0.0,
    buy_ratio: float = 0.7,
    sell_ratio: float = 0.3,
    cross_positive: bool = True,
    cross_negative: bool = False,
    cvd_increasing: bool = True,
    cvd_decreasing: bool = False,
    no_new_low: bool = True,
    no_new_high: bool = True,
    up_burst: bool = False,
    down_burst: bool = False,
) -> CvdSnapshot:
    return CvdSnapshot(
        ts_ms=ts_ms,
        price=price,
        side="buy" if buy_ratio >= sell_ratio else "sell",
        size=1.0,
        signed_delta=1.0,
        total_cvd=10.0,
        fast_cvd=fast_cvd,
        previous_fast_cvd=previous_fast_cvd,
        buy_volume=70.0,
        sell_volume=30.0,
        buy_ratio=buy_ratio,
        sell_ratio=sell_ratio,
        cross_positive=cross_positive,
        cross_negative=cross_negative,
        cvd_increasing=cvd_increasing,
        cvd_decreasing=cvd_decreasing,
        no_new_low=no_new_low,
        no_new_high=no_new_high,
        window_low=1897.0,
        window_high=1905.0,
        burst_net_move_pct=0.0,
        burst_range_pct=0.002,
        baseline_range_pct=0.001,
        burst_move_ratio=2.0,
        burst_volume=10.0,
        baseline_volume=5.0,
        burst_volume_ratio=2.0,
        up_burst=up_burst,
        down_burst=down_burst,
    )


def _make_strategy(
    post_entry_sl_cooldown_enabled=True,
    post_entry_sl_cooldown_seconds=1800,
    post_entry_sl_cooldown_scope="SIDE",
    **extra,
) -> BollCvdReclaimStrategy:
    cfg = BollCvdReclaimStrategyConfig(
        post_entry_sl_cooldown_enabled=post_entry_sl_cooldown_enabled,
        post_entry_sl_cooldown_seconds=post_entry_sl_cooldown_seconds,
        post_entry_sl_cooldown_scope=post_entry_sl_cooldown_scope,
        min_outside_pct=0.001,
        entry_min_reward_risk=0.0,
        entry_fee_slippage_buffer_pct=0.0,
        order_cooldown_seconds=0,
        entry_cvd_divergence_enabled=False,
        entry_cvd_absorption_enabled=False,
        entry_reclaim_confirm_seconds=0,
        entry_reclaim_inside_band=True,
        **extra,
    )
    sizer = SimplePositionSizer(
        SimplePositionSizerConfig(
            dry_run_equity_usdt=10_000,
            leverage=20,
            trade_risk_pct=0.01,
            fee_slippage_buffer_pct=0.001,
        )
    )
    return BollCvdReclaimStrategy(cfg, sizer)


def _arm_cooldown(
    strat: BollCvdReclaimStrategy,
    ts_ms: int = 100_000,
    side: str = "LONG",
    reason: str = "entry_protective_sl_flat",
) -> None:
    strat.arm_post_entry_sl_cooldown(ts_ms, side, reason)


# ======================================================================
# 1. SIDE LONG cooldown discards lower setup
# ======================================================================


def test_side_long_cooldown_discards_lower_setup(caplog):
    """When SCOPE=SIDE and cooldown_side=LONG, lower (LONG) setup is discarded."""
    strat = _make_strategy(post_entry_sl_cooldown_scope="SIDE")
    _arm_cooldown(strat, ts_ms=100_000, side="LONG", reason="entry_protective_sl_flat")
    caplog.set_level(logging.INFO)

    boll = _boll()
    # Price breaks below lower band → arms lower setup
    price_below = boll.lower * 0.9985
    strat.on_tick(price_below, 100_001, boll, _cvd(ts_ms=100_001, price=price_below))

    # Lower setup must be discarded
    assert strat.state.lower_armed is False, "lower setup should be discarded"
    assert strat.state.lower_extreme_price is None
    assert strat.state.lower_reclaim_seen is False

    # DISCARDED log must appear exactly once
    discard_count = sum(
        1 for r in caplog.records if "POST_ENTRY_SL_COOLDOWN_SETUP_DISCARDED" in r.getMessage()
    )
    assert discard_count == 1, f"Expected 1 DISCARDED log, got {discard_count}"


# ======================================================================
# 2. SIDE LONG cooldown preserves upper setup
# ======================================================================


def test_side_long_cooldown_preserves_upper_setup(caplog):
    """When SCOPE=SIDE and cooldown_side=LONG, upper (SHORT) setup is NOT discarded."""
    strat = _make_strategy(post_entry_sl_cooldown_scope="SIDE")
    _arm_cooldown(strat, ts_ms=100_000, side="LONG", reason="entry_protective_sl_flat")
    caplog.set_level(logging.INFO)

    boll = _boll()
    # Price breaks above upper band → arms upper setup
    price_above = boll.upper * 1.0015
    strat.on_tick(price_above, 100_001, boll,
                  _cvd(ts_ms=100_001, price=price_above,
                       cross_positive=False, cross_negative=False,
                       cvd_increasing=False, cvd_decreasing=False))

    # Upper setup must be preserved (SHORT not blocked by LONG cooldown with SIDE scope)
    assert strat.state.upper_armed is True, "upper setup should be preserved"

    # No DISCARDED log for SHORT
    discard_logs = [
        r for r in caplog.records
        if "POST_ENTRY_SL_COOLDOWN_SETUP_DISCARDED" in r.getMessage() and "SHORT" in r.getMessage()
    ]
    assert len(discard_logs) == 0, "SHORT discard log should not appear"


# ======================================================================
# 3. SIDE SHORT cooldown discards upper setup
# ======================================================================


def test_side_short_cooldown_discards_upper_setup(caplog):
    """When SCOPE=SIDE and cooldown_side=SHORT, upper (SHORT) setup is discarded."""
    strat = _make_strategy(post_entry_sl_cooldown_scope="SIDE")
    _arm_cooldown(strat, ts_ms=100_000, side="SHORT", reason="entry_protective_sl_flat")
    caplog.set_level(logging.INFO)

    boll = _boll()
    # Price breaks above upper band
    price_above = boll.upper * 1.0015
    strat.on_tick(price_above, 100_001, boll,
                  _cvd(ts_ms=100_001, price=price_above,
                       cross_positive=False, cross_negative=False,
                       cvd_increasing=False, cvd_decreasing=False))

    # Upper setup must be discarded
    assert strat.state.upper_armed is False, "upper setup should be discarded"
    assert strat.state.upper_extreme_price is None
    assert strat.state.upper_reclaim_seen is False

    # DISCARDED log must appear
    discard_count = sum(
        1 for r in caplog.records if "POST_ENTRY_SL_COOLDOWN_SETUP_DISCARDED" in r.getMessage()
    )
    assert discard_count == 1


# ======================================================================
# 4. GLOBAL cooldown discards both setups
# ======================================================================


def test_global_cooldown_discards_lower_setup(caplog):
    """When SCOPE=GLOBAL, lower setup is discarded."""
    strat = _make_strategy(post_entry_sl_cooldown_scope="GLOBAL")
    _arm_cooldown(strat, ts_ms=100_000, side="LONG", reason="entry_protective_sl_flat")
    caplog.set_level(logging.INFO)

    boll = _boll()
    price_below = boll.lower * 0.9985
    strat.on_tick(price_below, 100_001, boll, _cvd(ts_ms=100_001, price=price_below))

    assert strat.state.lower_armed is False, "GLOBAL cooldown must discard lower setup"
    discard_count = sum(
        1 for r in caplog.records if "POST_ENTRY_SL_COOLDOWN_SETUP_DISCARDED" in r.getMessage()
    )
    assert discard_count == 1


def test_global_cooldown_discards_upper_setup(caplog):
    """When SCOPE=GLOBAL, upper setup is also discarded."""
    strat = _make_strategy(post_entry_sl_cooldown_scope="GLOBAL")
    _arm_cooldown(strat, ts_ms=100_000, side="LONG", reason="entry_protective_sl_flat")
    caplog.set_level(logging.INFO)

    boll = _boll()
    price_above = boll.upper * 1.0015
    strat.on_tick(price_above, 100_001, boll,
                  _cvd(ts_ms=100_001, price=price_above,
                       cross_positive=False, cross_negative=False,
                       cvd_increasing=False, cvd_decreasing=False))

    assert strat.state.upper_armed is False, "GLOBAL cooldown must discard upper setup"
    discard_count = sum(
        1 for r in caplog.records if "POST_ENTRY_SL_COOLDOWN_SETUP_DISCARDED" in r.getMessage()
    )
    assert discard_count == 1  # First on_tick already armed → one discard


# ======================================================================
# 5. Cooldown expired → must wait for fresh setup
# ======================================================================


def test_cooldown_expired_no_stale_setup_reuse():
    """After cooldown expires, old setup must NOT be reused.

    1. Cooldown active, price < lower → lower setup discarded
    2. Advance ts_ms past cooldown expiry
    3. Price is now inside band → lower must still be un-armed
    4. THEN a new price < lower tick must arm a fresh setup
    """
    strat = _make_strategy(post_entry_sl_cooldown_scope="SIDE")
    ts_arm = 100_000
    _arm_cooldown(strat, ts_ms=ts_arm, side="LONG", reason="entry_protective_sl_flat")
    boll = _boll()

    # Step 1: price < lower during cooldown → setup discarded
    price_below = boll.lower * 0.9985
    strat.on_tick(price_below, ts_arm + 1000, boll, _cvd(ts_ms=ts_arm + 1000, price=price_below))
    assert strat.state.lower_armed is False

    # Step 2: advance past cooldown (1800s later)
    ts_post_cooldown = ts_arm + 1_800_001
    # Price is inside band (middle)
    strat.on_tick(float(boll.middle), ts_post_cooldown, boll, _cvd(ts_ms=ts_post_cooldown, price=float(boll.middle)))
    assert strat.state.lower_armed is False, (
        "After cooldown expiry, stale setup must not reappear"
    )

    # Step 3: fresh price < lower must arm a new setup
    ts_fresh = ts_arm + 1_800_100
    price_fresh = boll.lower * 0.9985
    strat.on_tick(price_fresh, ts_fresh, boll, _cvd(ts_ms=ts_fresh, price=price_fresh))
    assert strat.state.lower_armed is True, (
        "After cooldown expiry, fresh out-of-band tick must arm new setup"
    )


# ======================================================================
# 6. Log throttling — DISCARDED log ≤ 1 per 60s per key
# ======================================================================


def test_discard_log_throttled_one_per_60s(caplog):
    """POST_ENTRY_SL_COOLDOWN_SETUP_DISCARDED logs at most once per 60s per key."""
    strat = _make_strategy(post_entry_sl_cooldown_scope="SIDE")
    _arm_cooldown(strat, ts_ms=100_000, side="LONG", reason="entry_protective_sl_flat")
    caplog.set_level(logging.INFO)

    boll = _boll()
    price_below = boll.lower * 0.9985

    # 100 ticks within 60s — each re-arms lower then gets discarded
    for i in range(100):
        ts = 100_000 + i
        # Each tick: _update_armed_state arms lower, then _discard resets it
        strat.on_tick(price_below, ts, boll, _cvd(ts_ms=ts, price=price_below))
        # Manually re-arm for next iteration (since discard resets it)
        # Actually, _update_armed_state will re-arm it each tick since price < lower.
        # But _discard resets it immediately. So we just need to check the log count.

    discard_count = sum(
        1 for r in caplog.records if "POST_ENTRY_SL_COOLDOWN_SETUP_DISCARDED" in r.getMessage()
    )
    assert discard_count == 1, (
        f"DISCARDED log should appear once in 60s window, got {discard_count}"
    )


def test_discard_log_allows_after_60s(caplog):
    """After 60s, same key can log DISCARDED again."""
    strat = _make_strategy(post_entry_sl_cooldown_scope="SIDE")
    _arm_cooldown(strat, ts_ms=100_000, side="LONG", reason="entry_protective_sl_flat")
    caplog.set_level(logging.INFO)

    boll = _boll()
    price_below = boll.lower * 0.9985

    # First tick at t=100000
    strat.on_tick(price_below, 100_000, boll, _cvd(ts_ms=100_000, price=price_below))

    # Second tick at t=160001 (60s + 1ms later)
    strat.on_tick(price_below, 160_001, boll, _cvd(ts_ms=160_001, price=price_below))

    discard_count = sum(
        1 for r in caplog.records if "POST_ENTRY_SL_COOLDOWN_SETUP_DISCARDED" in r.getMessage()
    )
    assert discard_count == 2, (
        f"Should log DISCARDED again after 60s, got {discard_count}"
    )


def test_no_active_log_from_gate(caplog):
    """_post_entry_sl_cooldown_ok must be a pure gate — no ACTIVE logging."""
    strat = _make_strategy(post_entry_sl_cooldown_scope="SIDE")
    _arm_cooldown(strat, ts_ms=100_000, side="LONG", reason="entry_protective_sl_flat")
    caplog.set_level(logging.INFO)

    # Call the gate many times
    for _ in range(10):
        strat._post_entry_sl_cooldown_ok("LONG", 100_001)

    active_count = sum(
        1 for r in caplog.records if "POST_ENTRY_SL_COOLDOWN_ACTIVE" in r.getMessage()
    )
    assert active_count == 0, (
        f"_post_entry_sl_cooldown_ok must not log ACTIVE, got {active_count}"
    )


# ======================================================================
# 7. Call-site tests: on_tick correctly gates setups
# ======================================================================


def test_on_tick_does_not_produce_entry_intent_during_cooldown(caplog):
    """Full reclaim cycle is blocked during cooldown — no OPEN intent produced."""
    strat = _make_strategy(post_entry_sl_cooldown_scope="SIDE")
    ts_arm = 100_000
    _arm_cooldown(strat, ts_ms=ts_arm, side="LONG", reason="entry_protective_sl_flat")
    caplog.set_level(logging.INFO)

    boll = _boll()
    # Arm lower band
    price1 = boll.lower * 0.9985
    strat.on_tick(price1, ts_arm + 1000, boll, _cvd(ts_ms=ts_arm + 1000, price=price1))
    # Lower setup should be discarded
    assert strat.state.lower_armed is False

    # Try to enter on reclaim — should be blocked (no armed state to trigger)
    reclaim_price = boll.lower * 1.001
    cvd_reclaim = _cvd(ts_ms=ts_arm + 5000, price=reclaim_price,
                       cross_positive=True, cvd_increasing=True,
                       buy_ratio=0.7, sell_ratio=0.3, no_new_low=True)
    intents = strat.on_tick(reclaim_price, ts_arm + 5000, boll, cvd_reclaim)
    open_intents = [i for i in intents if i.intent_type in ("OPEN_LONG", "OPEN_SHORT")]
    assert len(open_intents) == 0, "No OPEN intent should be produced during cooldown"


def test_on_tick_allows_opposite_side_during_side_cooldown():
    """SIDE scope: opposite side entry is NOT blocked."""
    strat = _make_strategy(post_entry_sl_cooldown_scope="SIDE")
    ts_arm = 100_000
    _arm_cooldown(strat, ts_ms=ts_arm, side="LONG", reason="entry_protective_sl_flat")

    boll = _boll()
    # Arm upper band for SHORT (opposite side)
    price_above = boll.upper * 1.0015
    strat.on_tick(price_above, ts_arm + 1000, boll,
                  _cvd(ts_ms=ts_arm + 1000, price=price_above,
                       cross_positive=False, cross_negative=False,
                       cvd_increasing=False, cvd_decreasing=False))
    # Upper setup should be preserved
    assert strat.state.upper_armed is True, "Opposite side setup must be preserved"

    # Enter SHORT on reclaim
    reclaim_price = boll.upper * 0.999
    cvd_short = _cvd(ts_ms=ts_arm + 5000, price=reclaim_price,
                     cross_positive=False, cross_negative=True,
                     cvd_increasing=False, cvd_decreasing=True,
                     buy_ratio=0.3, sell_ratio=0.7, no_new_high=True)
    intents = strat.on_tick(reclaim_price, ts_arm + 5000, boll, cvd_short)
    short_intents = [i for i in intents if i.side == "SHORT"]
    assert len(short_intents) >= 1, "Opposite side entry should be allowed"


def test_blocks_side_returns_false_when_cooldown_disabled():
    """_post_entry_sl_cooldown_blocks_side returns False when cooldown disabled."""
    strat = _make_strategy(post_entry_sl_cooldown_enabled=False)
    _arm_cooldown(strat, ts_ms=100_000, side="LONG", reason="entry_protective_sl_flat")

    assert strat._post_entry_sl_cooldown_blocks_side("LONG", 100_001) is False
    assert strat._post_entry_sl_cooldown_blocks_side("SHORT", 100_001) is False


def test_blocks_side_returns_true_only_for_same_side():
    """SIDE scope: blocks_side returns True only for matching side."""
    strat = _make_strategy(post_entry_sl_cooldown_scope="SIDE")
    _arm_cooldown(strat, ts_ms=100_000, side="LONG", reason="entry_protective_sl_flat")

    assert strat._post_entry_sl_cooldown_blocks_side("LONG", 100_001) is True
    assert strat._post_entry_sl_cooldown_blocks_side("SHORT", 100_001) is False


def test_blocks_side_returns_true_for_both_in_global():
    """GLOBAL scope: blocks_side returns True for both sides."""
    strat = _make_strategy(post_entry_sl_cooldown_scope="GLOBAL")
    _arm_cooldown(strat, ts_ms=100_000, side="LONG", reason="entry_protective_sl_flat")

    assert strat._post_entry_sl_cooldown_blocks_side("LONG", 100_001) is True
    assert strat._post_entry_sl_cooldown_blocks_side("SHORT", 100_001) is True


def test_blocks_side_returns_false_after_expiry():
    """After cooldown expires, blocks_side returns False and clears state."""
    strat = _make_strategy(post_entry_sl_cooldown_scope="SIDE")
    _arm_cooldown(strat, ts_ms=100_000, side="LONG", reason="entry_protective_sl_flat")

    assert strat._post_entry_sl_cooldown_blocks_side("LONG", 100_000 + 1_800_001) is False
    assert strat.state.post_entry_sl_cooldown_until_ts_ms == 0
    assert strat.state.post_entry_sl_cooldown_side is None
    assert strat.state.post_entry_sl_cooldown_reason is None


def test_discard_no_armed_state_no_log(caplog):
    """When nothing is armed, discard produces no log."""
    strat = _make_strategy(post_entry_sl_cooldown_scope="SIDE")
    _arm_cooldown(strat, ts_ms=100_000, side="LONG", reason="entry_protective_sl_flat")
    caplog.set_level(logging.INFO)

    # No setup armed (price near middle)
    strat._discard_cooldown_blocked_setups(
        long_blocked=True, short_blocked=False, ts_ms=100_001,
    )
    discard_count = sum(
        1 for r in caplog.records if "POST_ENTRY_SL_COOLDOWN_SETUP_DISCARDED" in r.getMessage()
    )
    assert discard_count == 0, "No log when nothing is armed"


# ======================================================================
# Source-level prevention: no LOWER_ARMED / UPPER_ARMED during cooldown
# ======================================================================


def test_long_cooldown_no_lower_armed_log(caplog):
    """When LONG is blocked by cooldown and price < lower, LOWER_ARMED must NOT appear."""
    strat = _make_strategy(post_entry_sl_cooldown_scope="SIDE")
    _arm_cooldown(strat, ts_ms=100_000, side="LONG", reason="entry_protective_sl_flat")
    caplog.set_level(logging.INFO)

    boll = _boll()
    price_below = boll.lower * 0.9985
    strat.on_tick(price_below, 100_001, boll, _cvd(ts_ms=100_001, price=price_below))

    # State must be unarmed
    assert strat.state.lower_armed is False, "LONG cooldown must prevent lower_armed"
    assert strat.state.lower_extreme_price is None

    # LOWER_ARMED log must NOT appear
    armed_log_count = sum(
        1 for r in caplog.records if "LOWER_ARMED" in r.getMessage()
    )
    assert armed_log_count == 0, (
        f"LOWER_ARMED must not appear during cooldown, got {armed_log_count}"
    )

    # DISCARDED log must appear
    discard_count = sum(
        1 for r in caplog.records if "POST_ENTRY_SL_COOLDOWN_SETUP_DISCARDED" in r.getMessage()
    )
    assert discard_count == 1, (
        f"DISCARDED log must appear when setup is blocked, got {discard_count}"
    )


def test_short_cooldown_no_upper_armed_log(caplog):
    """When SHORT is blocked by cooldown and price > upper, UPPER_ARMED must NOT appear."""
    strat = _make_strategy(post_entry_sl_cooldown_scope="SIDE")
    _arm_cooldown(strat, ts_ms=100_000, side="SHORT", reason="entry_protective_sl_flat")
    caplog.set_level(logging.INFO)

    boll = _boll()
    price_above = boll.upper * 1.0015
    strat.on_tick(price_above, 100_001, boll,
                  _cvd(ts_ms=100_001, price=price_above,
                       cross_positive=False, cross_negative=False,
                       cvd_increasing=False, cvd_decreasing=False))

    # State must be unarmed
    assert strat.state.upper_armed is False, "SHORT cooldown must prevent upper_armed"
    assert strat.state.upper_extreme_price is None

    # UPPER_ARMED log must NOT appear
    armed_log_count = sum(
        1 for r in caplog.records if "UPPER_ARMED" in r.getMessage()
    )
    assert armed_log_count == 0, (
        f"UPPER_ARMED must not appear during cooldown, got {armed_log_count}"
    )

    # DISCARDED log must appear
    discard_count = sum(
        1 for r in caplog.records if "POST_ENTRY_SL_COOLDOWN_SETUP_DISCARDED" in r.getMessage()
    )
    assert discard_count == 1


def test_global_cooldown_no_armed_log_either_side(caplog):
    """GLOBAL cooldown must prevent both LOWER_ARMED and UPPER_ARMED."""
    strat = _make_strategy(post_entry_sl_cooldown_scope="GLOBAL")
    _arm_cooldown(strat, ts_ms=100_000, side="LONG", reason="entry_protective_sl_flat")
    caplog.set_level(logging.INFO)

    boll = _boll()
    price_below = boll.lower * 0.9985
    strat.on_tick(price_below, 100_001, boll, _cvd(ts_ms=100_001, price=price_below))

    # Neither side should be armed
    assert strat.state.lower_armed is False
    assert strat.state.upper_armed is False

    # Neither ARMED log should appear
    lower_armed = sum(1 for r in caplog.records if "LOWER_ARMED" in r.getMessage())
    upper_armed = sum(1 for r in caplog.records if "UPPER_ARMED" in r.getMessage())
    assert lower_armed == 0, f"No LOWER_ARMED in GLOBAL cooldown, got {lower_armed}"
    assert upper_armed == 0, f"No UPPER_ARMED in GLOBAL cooldown, got {upper_armed}"


# ======================================================================
# 100 ticks — no log spamming
# ======================================================================


def test_100_ticks_no_armed_log_spam(caplog):
    """100 ticks of price < lower during LONG cooldown: no LOWER_ARMED, ≤1 DISCARDED."""
    strat = _make_strategy(post_entry_sl_cooldown_scope="SIDE")
    _arm_cooldown(strat, ts_ms=100_000, side="LONG", reason="entry_protective_sl_flat")
    caplog.set_level(logging.INFO)

    boll = _boll()
    price_below = boll.lower * 0.9985

    for i in range(100):
        ts = 100_000 + i
        strat.on_tick(price_below, ts, boll, _cvd(ts_ms=ts, price=price_below))
        assert strat.state.lower_armed is False, f"Tick {i}: lower_armed must stay False"

    # LOWER_ARMED must NEVER appear
    armed_count = sum(
        1 for r in caplog.records if "LOWER_ARMED" in r.getMessage()
    )
    assert armed_count == 0, (
        f"LOWER_ARMED must not appear in 100 cooldown ticks, got {armed_count}"
    )

    # DISCARDED throttled to ≤1 in 60s window
    discard_count = sum(
        1 for r in caplog.records if "POST_ENTRY_SL_COOLDOWN_SETUP_DISCARDED" in r.getMessage()
    )
    assert discard_count == 1, (
        f"DISCARDED throttled to 1 in 60s, got {discard_count}"
    )
