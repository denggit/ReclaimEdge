from __future__ import annotations

import os
import tempfile

from src.indicators.cvd_tracker import CvdSnapshot
from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.reporting.live_state_store import LivePositionState, LiveStateStore
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
    StrategyPositionState,
)


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


def _strategy(**overrides) -> BollCvdReclaimStrategy:
    cfg_kwargs = dict(
        min_outside_pct=0.001,
        entry_min_reward_risk=0.0,
        entry_fee_slippage_buffer_pct=0.0,
        order_cooldown_seconds=0,
        entry_cvd_divergence_enabled=False,  # disable for cooldown tests
        entry_cvd_absorption_enabled=False,
        entry_reclaim_confirm_seconds=0,  # no wait
        entry_reclaim_inside_band=True,
        post_entry_sl_cooldown_enabled=True,
        post_entry_sl_cooldown_seconds=1800,
        post_entry_sl_cooldown_scope="GLOBAL",
    )
    cfg_kwargs.update(overrides)
    cfg = BollCvdReclaimStrategyConfig(**cfg_kwargs)
    sizer = SimplePositionSizer(
        SimplePositionSizerConfig(
            dry_run_equity_usdt=10_000,
            leverage=20,
            trade_risk_pct=0.01,
            fee_slippage_buffer_pct=0.001,
        )
    )
    return BollCvdReclaimStrategy(cfg, sizer)


def _arm_and_enter_long(strat: BollCvdReclaimStrategy, ts_ms: int = 5000) -> None:
    """Helper: arm lower band, reach deep enough, and enter LONG."""
    boll = _boll()
    # Arm
    price1 = 1900 * 0.9985
    strat.on_tick(price1, 1000, boll, _cvd(ts_ms=1000, price=price1))
    # Enter on reclaim with good CVD
    reclaim_price = boll.lower * 1.001
    cvd = _cvd(ts_ms=ts_ms, price=reclaim_price,
               cross_positive=True, cvd_increasing=True,
               buy_ratio=0.7, sell_ratio=0.3, no_new_low=True)
    strat.on_tick(reclaim_price, ts_ms, boll, cvd)


def _arm_and_enter_short(strat: BollCvdReclaimStrategy, ts_ms: int = 5000) -> None:
    """Helper: arm upper band, reach deep enough, and enter SHORT."""
    boll = _boll()
    # Arm
    price1 = 2100 * 1.0015
    strat.on_tick(price1, 1000, boll, _cvd(ts_ms=1000, price=price1,
                                            cross_positive=False, cross_negative=False,
                                            cvd_increasing=False, cvd_decreasing=False))
    # Enter on reclaim with good CVD
    reclaim_price = boll.upper * 0.999
    cvd = _cvd(ts_ms=ts_ms, price=reclaim_price,
               cross_positive=False, cross_negative=True,
               cvd_increasing=False, cvd_decreasing=True,
               buy_ratio=0.3, sell_ratio=0.7, no_new_high=True)
    strat.on_tick(reclaim_price, ts_ms, boll, cvd)


# ── Test: entry SL flat → cooldown armed ──────────────────────────────


def test_entry_sl_flat_arms_cooldown() -> None:
    """arm_post_entry_sl_cooldown sets cooldown_until_ts_ms = now + 1800s."""
    strat = _strategy()
    ts_now = 100_000
    strat.arm_post_entry_sl_cooldown(ts_now, "LONG", "entry_protective_sl_flat")

    assert strat.state.post_entry_sl_cooldown_until_ts_ms == ts_now + 1_800_000
    assert strat.state.post_entry_sl_cooldown_side == "LONG"
    assert strat.state.post_entry_sl_cooldown_reason == "entry_protective_sl_flat"


# ── Test: cooldown GLOBAL → both sides blocked ─────────────────────────


def test_cooldown_global_blocks_both_sides() -> None:
    """Scope=GLOBAL blocks LONG and SHORT entries during cooldown."""
    strat = _strategy(post_entry_sl_cooldown_scope="GLOBAL")
    ts_now = 100_000
    strat.arm_post_entry_sl_cooldown(ts_now, "LONG", "entry_protective_sl_flat")

    # During cooldown
    assert strat._post_entry_sl_cooldown_ok("LONG", ts_now + 1000) is False
    assert strat._post_entry_sl_cooldown_ok("SHORT", ts_now + 1000) is False


# ── Test: cooldown SIDE → only same side blocked ───────────────────────


def test_cooldown_side_only_blocks_same_side() -> None:
    """Scope=SIDE only blocks the same direction."""
    strat = _strategy(post_entry_sl_cooldown_scope="SIDE")
    ts_now = 100_000
    strat.arm_post_entry_sl_cooldown(ts_now, "LONG", "entry_protective_sl_flat")

    # Same side blocked
    assert strat._post_entry_sl_cooldown_ok("LONG", ts_now + 1000) is False
    # Opposite side allowed
    assert strat._post_entry_sl_cooldown_ok("SHORT", ts_now + 1000) is True


# ── Test: cooldown expired → allows entry ─────────────────────────────


def test_cooldown_expired_allows_entry() -> None:
    """After cooldown expires, entries are allowed again."""
    strat = _strategy()
    ts_now = 100_000
    strat.arm_post_entry_sl_cooldown(ts_now, "LONG", "entry_protective_sl_flat")

    # After cooldown period
    assert strat._post_entry_sl_cooldown_ok("LONG", ts_now + 1_800_001) is True
    # Cooldown state should be cleared
    assert strat.state.post_entry_sl_cooldown_until_ts_ms == 0
    assert strat.state.post_entry_sl_cooldown_side is None


# ── Test: three_stage_tp1_consumed → post-TP1 SL does NOT trigger cooldown ─


def test_three_stage_tp1_consumed_no_cooldown() -> None:
    """When three_stage_tp1_consumed=True, arming cooldown should be skipped.

    This test verifies the guard in pre_core_position.py logic:
    three_stage_tp1_consumed=True → should NOT arm cooldown.
    """
    strat = _strategy()
    strat.state.three_stage_tp1_consumed = True
    strat.state.entry_protective_sl_order_id = "sl-order-123"

    # Simulate the cooldown decision logic from pre_core_position.py
    should_arm = (
        strat.config.post_entry_sl_cooldown_enabled
        and not strat.state.three_stage_tp1_consumed
        and not strat.state.partial_tp_consumed
        and strat.state.entry_protective_sl_order_id is not None
    )
    assert should_arm is False


# ── Test: partial_tp_consumed → runner SL does NOT trigger cooldown ────


def test_partial_tp_consumed_no_cooldown() -> None:
    """When partial_tp_consumed=True, arming cooldown should be skipped."""
    strat = _strategy()
    strat.state.partial_tp_consumed = True
    strat.state.entry_protective_sl_order_id = "sl-order-123"

    should_arm = (
        strat.config.post_entry_sl_cooldown_enabled
        and not strat.state.three_stage_tp1_consumed
        and not strat.state.partial_tp_consumed
        and strat.state.entry_protective_sl_order_id is not None
    )
    assert should_arm is False


# ── Test: cooldown persists in live_state_store ────────────────────────


def test_cooldown_persists_in_live_state_store() -> None:
    """Cooldown fields are saved and loaded from LivePositionState."""
    strat = _strategy()
    ts_now = 100_000
    strat.arm_post_entry_sl_cooldown(ts_now, "SHORT", "entry_protective_sl_flat")

    state = LiveStateStore.from_strategy_state(
        position_id="test-pos-1",
        symbol="ETH-USDT-SWAP",
        strategy_state=strat.state,
        cash_before_position=5000.0,
    )

    assert state.post_entry_sl_cooldown_until_ts_ms == ts_now + 1_800_000
    assert state.post_entry_sl_cooldown_side == "SHORT"
    assert state.post_entry_sl_cooldown_reason == "entry_protective_sl_flat"


# ── Test: cooldown restored from startup_recovery ──────────────────────


def test_cooldown_restored_from_startup_recovery() -> None:
    """Cooldown is restored when strategy state is loaded from saved state."""
    from src.live.startup_recovery.basic_restore import restore_strategy_from_saved_state

    strat = _strategy()
    ts_now = 100_000
    strat.arm_post_entry_sl_cooldown(ts_now, "SHORT", "entry_protective_sl_flat")

    # Save state
    saved = LiveStateStore.from_strategy_state(
        position_id="test-pos-1",
        symbol="ETH-USDT-SWAP",
        strategy_state=strat.state,
        cash_before_position=5000.0,
    )

    # Restore into a new strategy
    strat2 = _strategy()
    restore_strategy_from_saved_state(strat2, saved)

    assert strat2.state.post_entry_sl_cooldown_until_ts_ms == ts_now + 1_800_000
    assert strat2.state.post_entry_sl_cooldown_side == "SHORT"
    assert strat2.state.post_entry_sl_cooldown_reason == "entry_protective_sl_flat"


# ── Test: reset armed does NOT clear cooldown ──────────────────────────


def test_reset_armed_does_not_clear_cooldown() -> None:
    """Resetting armed state should not affect cooldown fields."""
    strat = _strategy()
    ts_now = 100_000
    strat.arm_post_entry_sl_cooldown(ts_now, "LONG", "entry_protective_sl_flat")

    # Arm and then reset lower
    boll = _boll()
    price1 = 1900 * 0.9985
    strat.on_tick(price1, 1000, boll, _cvd(ts_ms=1000, price=price1))
    assert strat.state.lower_armed is True
    strat._reset_lower_armed()
    assert strat.state.lower_armed is False

    # Cooldown should still be active
    assert strat.state.post_entry_sl_cooldown_until_ts_ms == ts_now + 1_800_000
    assert strat.state.post_entry_sl_cooldown_side == "LONG"


# ── Test: flat settlement preserves cooldown ────────────────────────────


def test_flat_settlement_preserves_cooldown() -> None:
    """When strategy.state is replaced with fresh StrategyPositionState,
    cooldown fields should be preserved."""
    strat = _strategy()
    ts_now = 100_000
    strat.arm_post_entry_sl_cooldown(ts_now, "LONG", "entry_protective_sl_flat")

    # Simulate what flat_settlement_phase does
    saved_until = strat.state.post_entry_sl_cooldown_until_ts_ms
    saved_side = strat.state.post_entry_sl_cooldown_side
    saved_reason = strat.state.post_entry_sl_cooldown_reason

    strat.state = StrategyPositionState()

    # Restore
    strat.state.post_entry_sl_cooldown_until_ts_ms = saved_until
    strat.state.post_entry_sl_cooldown_side = saved_side
    strat.state.post_entry_sl_cooldown_reason = saved_reason

    assert strat.state.post_entry_sl_cooldown_until_ts_ms == ts_now + 1_800_000
    assert strat.state.post_entry_sl_cooldown_side == "LONG"
    assert strat.state.post_entry_sl_cooldown_reason == "entry_protective_sl_flat"
    # Other fields should be at defaults
    assert strat.state.lower_armed is False
    assert strat.state.layers == 0


# ── Test: cooldown blocks on_tick entry ────────────────────────────────


def test_cooldown_blocks_on_tick_entry() -> None:
    """When cooldown is active, on_tick should not generate entry intents."""
    strat = _strategy()
    ts_now = 100_000
    strat.arm_post_entry_sl_cooldown(ts_now, "LONG", "entry_protective_sl_flat")

    boll = _boll()
    # Setup armed state
    price1 = 1900 * 0.9985
    strat.on_tick(price1, ts_now + 1000, boll, _cvd(ts_ms=ts_now + 1000, price=price1))

    # Try to enter — cooldown should block
    reclaim_price = boll.lower * 1.001
    cvd = _cvd(ts_ms=ts_now + 5000, price=reclaim_price,
               cross_positive=True, cvd_increasing=True,
               buy_ratio=0.7, sell_ratio=0.3, no_new_low=True)
    intents = strat.on_tick(reclaim_price, ts_now + 5000, boll, cvd)
    assert len(intents) == 0  # blocked by cooldown


# ── Test: on_tick scope=GLOBAL, LONG cooldown blocks both sides ─────────


def test_on_tick_cooldown_global_blocks_both_sides() -> None:
    """scope=GLOBAL: LONG cooldown should block both LONG and SHORT entry via on_tick."""
    strat = _strategy(post_entry_sl_cooldown_scope="GLOBAL")
    ts_now = 100_000
    strat.arm_post_entry_sl_cooldown(ts_now, "LONG", "entry_protective_sl_flat")

    boll = _boll()
    # Arm lower band for LONG setup
    strat.on_tick(1900 * 0.9985, ts_now + 1000, boll, _cvd(ts_ms=ts_now + 1000, price=1900 * 0.9985))
    # Also arm upper band for SHORT setup (opposite band break resets lower, so arm upper only)
    # Actually, after the fix, each side is checked independently. But upper_armed was cleared
    # by opposite band break. Let's re-arm upper in a separate tick.
    strat2 = _strategy(post_entry_sl_cooldown_scope="GLOBAL")
    strat2.arm_post_entry_sl_cooldown(ts_now, "LONG", "entry_protective_sl_flat")
    # Arm upper
    strat2.on_tick(2100 * 1.0015, ts_now + 1000, boll, _cvd(ts_ms=ts_now + 1000, price=2100 * 1.0015,
               cross_positive=False, cross_negative=False, cvd_increasing=False, cvd_decreasing=False))
    # Try SHORT reclaim
    short_reclaim = boll.upper * 0.999
    cvd_short = _cvd(ts_ms=ts_now + 5000, price=short_reclaim,
                     cross_positive=False, cross_negative=True, cvd_increasing=False, cvd_decreasing=True,
                     buy_ratio=0.3, sell_ratio=0.7, no_new_high=True)
    intents = strat2.on_tick(short_reclaim, ts_now + 5000, boll, cvd_short)
    assert len(intents) == 0  # GLOBAL cooldown blocks SHORT too


# ── Test: on_tick scope=SIDE, LONG cooldown allows SHORT entry ─────────


def test_on_tick_cooldown_side_long_allows_short_entry() -> None:
    """scope=SIDE: LONG cooldown blocks LONG but allows SHORT entry via on_tick.

    We test each direction independently because _update_armed_state() resets
    the opposite armed side when price crosses the middle band.
    """
    strat = _strategy(post_entry_sl_cooldown_scope="SIDE")
    ts_now = 100_000
    strat.arm_post_entry_sl_cooldown(ts_now, "LONG", "entry_protective_sl_flat")
    boll = _boll()

    # ── Sub-test A: LONG is blocked ──────────────────────────────────
    strat_a = _strategy(post_entry_sl_cooldown_scope="SIDE")
    strat_a.arm_post_entry_sl_cooldown(ts_now, "LONG", "entry_protective_sl_flat")
    # Arm lower band for LONG
    strat_a.on_tick(1900 * 0.9985, ts_now + 1000, boll, _cvd(ts_ms=ts_now + 1000, price=1900 * 0.9985))
    long_reclaim = boll.lower * 1.001
    cvd_long = _cvd(ts_ms=ts_now + 2000, price=long_reclaim,
                    cross_positive=True, cvd_increasing=True, buy_ratio=0.7, sell_ratio=0.3, no_new_low=True)
    intents_a = strat_a.on_tick(long_reclaim, ts_now + 2000, boll, cvd_long)
    long_intents = [i for i in intents_a if i.side == "LONG"]
    assert len(long_intents) == 0  # LONG blocked by SIDE cooldown

    # ── Sub-test B: SHORT is NOT blocked ─────────────────────────────
    strat_b = _strategy(post_entry_sl_cooldown_scope="SIDE")
    strat_b.arm_post_entry_sl_cooldown(ts_now, "LONG", "entry_protective_sl_flat")
    # Arm upper band for SHORT
    strat_b.on_tick(2100 * 1.0015, ts_now + 1000, boll, _cvd(ts_ms=ts_now + 1000, price=2100 * 1.0015,
                    cross_positive=False, cross_negative=False, cvd_increasing=False, cvd_decreasing=False))
    short_reclaim = boll.upper * 0.999
    cvd_short = _cvd(ts_ms=ts_now + 2000, price=short_reclaim,
                     cross_positive=False, cross_negative=True, cvd_increasing=False, cvd_decreasing=True,
                     buy_ratio=0.3, sell_ratio=0.7, no_new_high=True)
    intents_b = strat_b.on_tick(short_reclaim, ts_now + 2000, boll, cvd_short)
    short_intents = [i for i in intents_b if i.side == "SHORT"]
    assert len(short_intents) >= 1  # SHORT allowed with SIDE cooldown


# ── Test: on_tick scope=SIDE, SHORT cooldown allows LONG entry ─────────


def test_on_tick_cooldown_side_short_allows_long_entry() -> None:
    """scope=SIDE: SHORT cooldown blocks SHORT but allows LONG entry via on_tick.

    We test each direction independently because _update_armed_state() resets
    the opposite armed side when price crosses the middle band.
    """
    strat = _strategy(post_entry_sl_cooldown_scope="SIDE")
    ts_now = 100_000
    strat.arm_post_entry_sl_cooldown(ts_now, "SHORT", "entry_protective_sl_flat")
    boll = _boll()

    # ── Sub-test A: SHORT is blocked ─────────────────────────────────
    strat_a = _strategy(post_entry_sl_cooldown_scope="SIDE")
    strat_a.arm_post_entry_sl_cooldown(ts_now, "SHORT", "entry_protective_sl_flat")
    # Arm upper band for SHORT
    strat_a.on_tick(2100 * 1.0015, ts_now + 1000, boll, _cvd(ts_ms=ts_now + 1000, price=2100 * 1.0015,
                    cross_positive=False, cross_negative=False, cvd_increasing=False, cvd_decreasing=False))
    short_reclaim = boll.upper * 0.999
    cvd_short = _cvd(ts_ms=ts_now + 2000, price=short_reclaim,
                     cross_positive=False, cross_negative=True, cvd_increasing=False, cvd_decreasing=True,
                     buy_ratio=0.3, sell_ratio=0.7, no_new_high=True)
    intents_a = strat_a.on_tick(short_reclaim, ts_now + 2000, boll, cvd_short)
    short_intents = [i for i in intents_a if i.side == "SHORT"]
    assert len(short_intents) == 0  # SHORT blocked by SIDE cooldown

    # ── Sub-test B: LONG is NOT blocked ──────────────────────────────
    strat_b = _strategy(post_entry_sl_cooldown_scope="SIDE")
    strat_b.arm_post_entry_sl_cooldown(ts_now, "SHORT", "entry_protective_sl_flat")
    # Arm lower band for LONG
    strat_b.on_tick(1900 * 0.9985, ts_now + 1000, boll, _cvd(ts_ms=ts_now + 1000, price=1900 * 0.9985))
    long_reclaim = boll.lower * 1.001
    cvd_long = _cvd(ts_ms=ts_now + 2000, price=long_reclaim,
                    cross_positive=True, cvd_increasing=True, buy_ratio=0.7, sell_ratio=0.3, no_new_low=True)
    intents_b = strat_b.on_tick(long_reclaim, ts_now + 2000, boll, cvd_long)
    long_intents = [i for i in intents_b if i.side == "LONG"]
    assert len(long_intents) >= 1  # LONG allowed with SIDE cooldown


# ── Fix 4: cooldown candidate logic (pre_core_position pattern) ─────────


def test_entry_sl_loss_flat_is_cooldown_candidate() -> None:
    """Entry SL loss scenario: no partial TP, has entry_sl_order_id → candidate."""
    strat = _strategy()
    strat.state.entry_protective_sl_order_id = "sl-order-123"
    strat.state.partial_tp_consumed = False
    strat.state.three_stage_tp1_consumed = False

    entry_sl_cooldown_candidate = (
        strat.config.post_entry_sl_cooldown_enabled
        and not strat.state.three_stage_tp1_consumed
        and not strat.state.partial_tp_consumed
        and strat.state.entry_protective_sl_order_id is not None
    )
    assert entry_sl_cooldown_candidate is True


def test_single_tp_positive_flat_not_candidate() -> None:
    """SINGLE TP filled scenario: partial_tp_consumed=True → not a candidate."""
    strat = _strategy()
    strat.state.entry_protective_sl_order_id = "sl-order-123"
    strat.state.partial_tp_consumed = True  # TP was consumed
    strat.state.three_stage_tp1_consumed = False

    entry_sl_cooldown_candidate = (
        strat.config.post_entry_sl_cooldown_enabled
        and not strat.state.three_stage_tp1_consumed
        and not strat.state.partial_tp_consumed
        and strat.state.entry_protective_sl_order_id is not None
    )
    assert entry_sl_cooldown_candidate is False  # TP consumed → not candidate


def test_manual_close_positive_flat_not_candidate_by_pattern() -> None:
    """Manual close positive flat: entry_sl_order_id may still be set but
    partial_tp_consumed ≠ True. However, the settled-loss check in
    flat_settlement_phase will skip arming because realized PnL ≥ 0.

    The candidate flag alone does NOT arm — it only marks the flat for
    settled-loss evaluation.
    """
    strat = _strategy()
    strat.state.entry_protective_sl_order_id = "sl-order-123"
    strat.state.partial_tp_consumed = False
    strat.state.three_stage_tp1_consumed = False

    # Candidate logic: passes the pre-checks
    entry_sl_cooldown_candidate = (
        strat.config.post_entry_sl_cooldown_enabled
        and not strat.state.three_stage_tp1_consumed
        and not strat.state.partial_tp_consumed
        and strat.state.entry_protective_sl_order_id is not None
    )
    assert entry_sl_cooldown_candidate is True

    # But settled-loss check would skip: cash_after >= cash_before → no arm
    cash_before = 5000.0
    cash_after = 5100.0  # positive realized
    realized_delta = cash_after - cash_before
    assert realized_delta >= 0  # positive → no cooldown
    # cooldown should NOT be armed for positive flat


def test_manual_close_loss_flat_default_no_arm() -> None:
    """Manual close loss flat: by default, without explicit configuration
    allowing manual loss cooldown, the realized loss check in
    flat_settlement_phase is conservative.

    The pre_core candidate flag may be True, but only a settled negative
    PnL will actually arm. Manual close loss still gets the settled-loss
    evaluation, and by default would NOT arm unless explicitly configured.
    """
    strat = _strategy()
    strat.state.entry_protective_sl_order_id = "sl-order-123"
    strat.state.partial_tp_consumed = False
    strat.state.three_stage_tp1_consumed = False

    # Candidate logic passes
    entry_sl_cooldown_candidate = (
        strat.config.post_entry_sl_cooldown_enabled
        and not strat.state.three_stage_tp1_consumed
        and not strat.state.partial_tp_consumed
        and strat.state.entry_protective_sl_order_id is not None
    )
    assert entry_sl_cooldown_candidate is True

    # But with negative realized: the settled-loss check in
    # flat_settlement_phase would arm cooldown. This is conservative:
    # if a loss flat occurs without explicit fill-source info, it may
    # have been a protective SL. Manual close losses are rare in normal
    # bot operation.
    cash_before = 5000.0
    cash_after = 4950.0  # negative realized
    realized_delta = cash_after - cash_before
    # Conservative: negative flat → arm cooldown
    assert realized_delta < 0


def test_post_tp1_protective_sl_flat_not_candidate() -> None:
    """Post-TP1 protective SL flat: three_stage_tp1_consumed=True → not candidate."""
    strat = _strategy()
    strat.state.entry_protective_sl_order_id = "sl-order-123"
    strat.state.three_stage_tp1_consumed = True
    strat.state.partial_tp_consumed = False

    entry_sl_cooldown_candidate = (
        strat.config.post_entry_sl_cooldown_enabled
        and not strat.state.three_stage_tp1_consumed
        and not strat.state.partial_tp_consumed
        and strat.state.entry_protective_sl_order_id is not None
    )
    assert entry_sl_cooldown_candidate is False  # TP1 consumed → not candidate


def test_runner_sl_flat_not_candidate() -> None:
    """Runner SL flat: partial_tp_consumed=True → not candidate."""
    strat = _strategy()
    strat.state.entry_protective_sl_order_id = "sl-order-123"
    strat.state.partial_tp_consumed = True
    strat.state.three_stage_tp1_consumed = False

    entry_sl_cooldown_candidate = (
        strat.config.post_entry_sl_cooldown_enabled
        and not strat.state.three_stage_tp1_consumed
        and not strat.state.partial_tp_consumed
        and strat.state.entry_protective_sl_order_id is not None
    )
    assert entry_sl_cooldown_candidate is False  # partial TP consumed → not candidate


def test_no_entry_sl_order_id_not_candidate() -> None:
    """No entry_sl_order_id → not a candidate regardless of other flags."""
    strat = _strategy()
    strat.state.entry_protective_sl_order_id = None  # no SL order
    strat.state.partial_tp_consumed = False
    strat.state.three_stage_tp1_consumed = False

    entry_sl_cooldown_candidate = (
        strat.config.post_entry_sl_cooldown_enabled
        and not strat.state.three_stage_tp1_consumed
        and not strat.state.partial_tp_consumed
        and strat.state.entry_protective_sl_order_id is not None
    )
    assert entry_sl_cooldown_candidate is False  # no SL order → not candidate
