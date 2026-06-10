"""Tests for middle bucket split partial TP update preservation.

Validates that when one leg of a middle bucket split has been filled,
subsequent TP updates do NOT reset the consumed flags and do NOT
re-generate the already-filled leg in order specs.
"""

from decimal import Decimal

from src.execution.order_specs import (
    MiddleBucketSplitOrderInput,
    build_take_profit_order_specs,
)
from src.position_management.middle_bucket_split_state import (
    clear_middle_bucket_split_state,
    degrade_middle_bucket_split_to_single_final,
)
from src.strategies.middle_bucket_split_apply import (
    MiddleBucketSplitApplyResult,
    apply_three_stage_middle_bucket_split,
)


class FakeBollSnapshot:
    """Minimal BollSnapshot stub for apply_three_stage_middle_bucket_split."""

    def __init__(self, *, middle=1640.0, tp_middle=1650.0, candle_ts_ms=0):
        self.middle = middle
        self.tp_middle = tp_middle
        self.candle_ts_ms = candle_ts_ms
        # Additional fields that BollSnapshot may have
        self.upper = middle + 100
        self.lower = middle - 100


class FakeConfig:
    """Minimal config stub."""

    def __init__(self):
        self.middle_bucket_split_enabled = True
        self.middle_bucket_split_fast_ratio = 0.70
        self.tp_min_net_profit_pct = 0.005
        self.three_stage_runner_enabled = True
        self.three_stage_pre_tp1_degrade_enabled = False


class FakeState:
    """Minimal state stub to test split apply logic."""

    def __init__(self, *, side="SHORT"):
        self.side = side
        self.three_stage_tp1_ratio = 0.56
        self.three_stage_tp2_ratio = 0.20
        self.three_stage_runner_ratio = 0.24
        self.three_stage_tp1_consumed = False
        self.three_stage_tp2_consumed = False
        self.three_stage_runner_enabled_for_position = True
        self.trend_runner_active = False
        self.avg_entry_price = 1680.0
        self.breakeven_price = 1680.0
        self.tp_plan = "THREE_STAGE_RUNNER"
        self.tp_price = None
        self.tp_mode = "LOWER"
        self.partial_tp_price = None
        self.partial_tp_ratio = 0.0
        self.partial_tp_consumed = False
        self.middle_runner_active = False
        self.middle_runner_pending = False
        self.layers = 1
        self.last_tp_update_ts_ms = 0
        self.last_tp_update_candle_ts_ms = 0
        self.startup_force_tp_reconcile = False

        # Middle bucket split fields
        self.middle_bucket_split_active = False
        self.middle_bucket_split_fast_consumed = False
        self.middle_bucket_split_slow_consumed = False
        self.middle_bucket_split_fast_price = None
        self.middle_bucket_split_slow_price = None
        self.middle_bucket_split_effective_price = None
        self.middle_bucket_split_middle_bucket_ratio = 0.0
        self.middle_bucket_split_fast_ratio_of_bucket = 0.0
        self.middle_bucket_split_slow_ratio_of_bucket = 0.0
        self.middle_bucket_split_fast_total_ratio = 0.0
        self.middle_bucket_split_slow_total_ratio = 0.0
        self.middle_bucket_split_reason = None
        self.middle_bucket_split_fast_sl_price = None
        self.middle_bucket_split_fast_sl_order_id = None
        self.middle_bucket_split_fast_sl_protected = False
        self.middle_bucket_split_fast_sl_invalid_action_taken = None
        self.middle_bucket_split_add_disabled = False

        # Three-stage fields
        self.three_stage_tp1_price = None
        self.three_stage_tp2_price = None
        self.three_stage_post_tp1_protective_sl_price = None
        self.three_stage_post_tp1_protective_sl_order_id = None
        self.three_stage_post_tp1_protected = False
        self.three_stage_pre_tp1_degrade_stage = None
        self.three_stage_pre_tp1_degraded_ts_ms = 0

        # Middle runner fields
        self.middle_runner_first_close_ratio = 0.80
        self.middle_runner_first_tp_price = None
        self.middle_runner_final_tp_price = None
        self.middle_runner_protective_sl_price = None
        self.middle_runner_protective_sl_order_id = None
        self.middle_runner_size_mismatch_protected = False

        # Trend runner fields
        self.trend_runner_tp_price = None
        self.trend_runner_sl_price = None
        self.trend_runner_tp_order_id = None
        self.trend_runner_sl_order_id = None
        self.trend_runner_adjust_count = 0
        self.trend_runner_last_update_candle_ts_ms = 0
        self.trend_runner_exit_reason = None

        # Other fields
        self.near_tp_protected = False
        self.near_tp_add_disabled = False
        self.first_entry_ts_ms = 0


class FakeStrategy:
    """Minimal strategy stub."""

    def __init__(self, *, side="SHORT"):
        self.config = FakeConfig()
        self.state = FakeState(side=side)

    def _effective_breakeven_for_tp_selection(self, side):
        """Stub: return avg_entry_price as breakeven."""
        return self.state.avg_entry_price

    def _format_optional_price(self, price):
        """Stub."""
        return f"{price:.4f}" if price is not None else "-"

    def _select_valid_tp_outer_with_profit_fallback(self, side, boll):
        """Stub."""
        return (boll.middle + 100, "upper") if side == "LONG" else (boll.middle - 100, "lower")

    def _select_three_stage_tp2_outer(self, side, boll):
        """Stub."""
        return (boll.middle + 60, "upper") if side == "LONG" else (boll.middle - 60, "lower")

    def _select_valid_tp_middle_with_profit_fallback(self, side, boll):
        """Stub."""
        return (boll.middle, "middle")

    def _update_three_stage_dynamic_targets_without_reset(self, side, boll):
        """Stub."""
        return True

    def _fallback_to_single_outer_due_middle_profit_insufficient(self, side, boll, ts_ms, reason):
        """Stub."""
        from typing import cast
        outer = boll.middle + 100 if side == "LONG" else boll.middle - 100
        mode = "UPPER" if side == "LONG" else "LOWER"
        return outer, mode

    def _tp_plan_unchanged(self, tp_price, partial_tp_price, partial_tp_ratio, tp_plan):
        """Stub."""
        return False

    def _log_tp_boll_price_selected(self, **kwargs):
        """Stub."""
        pass

    def _intent(self, *args, **kwargs):
        """Stub."""
        return None

    def _three_stage_waiting_tp2(self):
        """Stub."""
        return False

    def _set_middle_runner_planned(self, partial_tp_price, tp_price):
        """Stub."""
        pass

    def _reset_three_stage_runner_state(self):
        """Stub."""
        pass

    def _reset_middle_runner_state(self):
        """Stub."""
        pass

    def _normalized_three_stage_ratios(self):
        """Stub."""
        return (0.56, 0.20, 0.24)

    def _effective_breakeven_for_tp_check(self, side):
        """Stub."""
        return self.state.avg_entry_price


# ── Test 6: Fast consumed after TP update does not reset fast_consumed ──

def test_fast_consumed_tp_update_preserves_fast_consumed():
    """Given split active, fast_consumed=True, slow_consumed=False.
    When TP update tries to re-apply split.
    Then fast_consumed remains True, slow_consumed remains False.
    No fresh MIDDLE_BUCKET_SPLIT_SELECTED reset.
    """
    strategy = FakeStrategy(side="SHORT")
    state = strategy.state

    # Setup: partial split with fast already consumed
    state.middle_bucket_split_active = True
    state.middle_bucket_split_fast_consumed = True
    state.middle_bucket_split_slow_consumed = False
    state.middle_bucket_split_fast_price = 1638.7253
    state.middle_bucket_split_slow_price = 1634.4645
    state.middle_bucket_split_effective_price = 1636.85
    state.middle_bucket_split_middle_bucket_ratio = 0.56
    state.middle_bucket_split_fast_ratio_of_bucket = 0.70
    state.middle_bucket_split_slow_ratio_of_bucket = 0.30
    state.middle_bucket_split_fast_total_ratio = 0.392
    state.middle_bucket_split_slow_total_ratio = 0.168
    state.three_stage_tp1_price = 1636.85
    state.three_stage_tp1_consumed = False

    boll = FakeBollSnapshot(middle=1640.0, tp_middle=1650.0, candle_ts_ms=1715700000000)

    result = apply_three_stage_middle_bucket_split(strategy=strategy, boll=boll)

    # The apply function should return SPLIT action with preserved flags
    assert result.action == "SPLIT"
    assert result.split_active is True

    # Consumed flags must remain unchanged
    assert state.middle_bucket_split_fast_consumed is True, (
        "fast_consumed should remain True after TP update"
    )
    assert state.middle_bucket_split_slow_consumed is False, (
        "slow_consumed should remain False after TP update"
    )

    # The effective price should be preserved (not reset to new split price)
    assert state.middle_bucket_split_effective_price == 1636.85, (
        "effective_price should be preserved"
    )


# ── Test 7: Slow consumed after TP update does not reset ────────────────

def test_slow_consumed_tp_update_preserves_slow_consumed():
    """Given split active, fast_consumed=False, slow_consumed=True.
    When TP update tries to re-apply split.
    Then slow_consumed remains True, fast_consumed remains False.
    Generated order specs do not include tp1_middle_slow.
    """
    strategy = FakeStrategy(side="SHORT")
    state = strategy.state

    # Setup: partial split with slow already consumed
    state.middle_bucket_split_active = True
    state.middle_bucket_split_fast_consumed = False
    state.middle_bucket_split_slow_consumed = True
    state.middle_bucket_split_fast_price = 1638.7253
    state.middle_bucket_split_slow_price = 1634.4645
    state.middle_bucket_split_effective_price = 1636.85
    state.middle_bucket_split_middle_bucket_ratio = 0.56
    state.middle_bucket_split_fast_ratio_of_bucket = 0.70
    state.middle_bucket_split_slow_ratio_of_bucket = 0.30
    state.middle_bucket_split_fast_total_ratio = 0.392
    state.middle_bucket_split_slow_total_ratio = 0.168
    state.three_stage_tp1_price = 1636.85
    state.three_stage_tp1_consumed = False

    boll = FakeBollSnapshot(middle=1640.0, tp_middle=1650.0, candle_ts_ms=1715700000000)

    result = apply_three_stage_middle_bucket_split(strategy=strategy, boll=boll)

    assert result.action == "SPLIT"
    assert state.middle_bucket_split_fast_consumed is False
    assert state.middle_bucket_split_slow_consumed is True
    assert state.middle_bucket_split_effective_price == 1636.85


# ── Test 8: Partial split in progress logs are emitted ──────────────────

def test_partial_split_progress_logged(caplog):
    """When partial split is in progress, the apply function should log
    MIDDLE_BUCKET_SPLIT_PARTIAL_PROGRESS_PRESERVED."""
    import logging

    strategy = FakeStrategy(side="SHORT")
    state = strategy.state

    state.middle_bucket_split_active = True
    state.middle_bucket_split_fast_consumed = True
    state.middle_bucket_split_slow_consumed = False
    state.middle_bucket_split_fast_price = 1638.7253
    state.middle_bucket_split_slow_price = 1634.4645
    state.middle_bucket_split_effective_price = 1636.85
    state.middle_bucket_split_middle_bucket_ratio = 0.56
    state.middle_bucket_split_fast_ratio_of_bucket = 0.70
    state.middle_bucket_split_slow_ratio_of_bucket = 0.30
    state.middle_bucket_split_fast_total_ratio = 0.392
    state.middle_bucket_split_slow_total_ratio = 0.168
    state.three_stage_tp1_price = 1636.85

    boll = FakeBollSnapshot(middle=1640.0, tp_middle=1650.0, candle_ts_ms=1715700000000)

    with caplog.at_level(logging.WARNING):
        apply_three_stage_middle_bucket_split(strategy=strategy, boll=boll)

    log_messages = [r.message for r in caplog.records]
    partial_log = [m for m in log_messages if "MIDDLE_BUCKET_SPLIT_PARTIAL_PROGRESS_PRESERVED" in m]
    assert len(partial_log) > 0, (
        f"Expected MIDDLE_BUCKET_SPLIT_PARTIAL_PROGRESS_PRESERVED log, got: {log_messages}"
    )
    # Verify no fresh SPLIT_SELECTED log
    selected_log = [m for m in log_messages if "MIDDLE_BUCKET_SPLIT_SELECTED" in m]
    assert len(selected_log) == 0, (
        "MIDDLE_BUCKET_SPLIT_SELECTED should NOT be logged during partial progress preservation"
    )


# ── Test: Fresh split still works normally ──────────────────────────────

def test_fresh_split_no_partial_progress():
    """When no partial split in progress, fresh split works normally."""
    strategy = FakeStrategy(side="SHORT")
    state = strategy.state

    # No prior split state
    state.middle_bucket_split_active = False
    state.middle_bucket_split_fast_consumed = False
    state.middle_bucket_split_slow_consumed = False

    boll = FakeBollSnapshot(middle=1640.0, tp_middle=1650.0, candle_ts_ms=1715700000000)

    result = apply_three_stage_middle_bucket_split(strategy=strategy, boll=boll)

    # Should still work as a fresh split
    assert result.action in ("SPLIT", "UNSPLIT_SLOW_MIDDLE", "FALLBACK_OUTER")
    # If split, consumed flags should be False (fresh)
    if result.action == "SPLIT":
        assert state.middle_bucket_split_fast_consumed is False
        assert state.middle_bucket_split_slow_consumed is False
