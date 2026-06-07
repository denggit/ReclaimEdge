"""Tests for middle bucket split state/order consistency.

Verifies that when split sub-legs are too small, the execution layer
disables the split and the strategy state is cleared to match actual orders.
"""

from __future__ import annotations

from decimal import Decimal
from unittest import mock

import pytest

from src.execution.middle_bucket_split_size import (
    MiddleBucketSplitSizeCheck,
    check_middle_runner_bucket_split_size,
    check_three_stage_middle_bucket_split_size,
)
from src.execution.order_specs import (
    MiddleBucketSplitOrderInput,
    build_take_profit_order_specs,
    round_contracts_down,
)


# ── Size check pure function tests ─────────────────────────────────────

class TestThreeStageSizeCheck:
    """Tests for check_three_stage_middle_bucket_split_size()."""

    def test_split_ok(self):
        """position=100, tp1_ratio=0.70, fast_ratio=0.70, min=1.
        tp1=70, fast=49, slow=21 → both >= 1 → ok."""
        result = check_three_stage_middle_bucket_split_size(
            position_contracts=Decimal("100"),
            min_contracts=Decimal("1"),
            contract_precision=Decimal("1"),
            three_stage_tp1_ratio=Decimal("0.70"),
            fast_ratio_of_bucket=Decimal("0.70"),
        )
        assert result.ok is True
        assert result.reason == "ok"
        assert result.tp1_total_contracts == Decimal("70")
        assert result.fast_contracts == Decimal("49")
        assert result.slow_contracts == Decimal("21")

    def test_subleg_too_small(self):
        """position=100, tp1=0.70, fast_ratio=0.99, min=10.
        tp1=70, fast=69, slow=1 < 10 → not ok."""
        result = check_three_stage_middle_bucket_split_size(
            position_contracts=Decimal("100"),
            min_contracts=Decimal("10"),
            contract_precision=Decimal("1"),
            three_stage_tp1_ratio=Decimal("0.70"),
            fast_ratio_of_bucket=Decimal("0.99"),
        )
        assert result.ok is False
        assert result.reason == "subleg_too_small"
        assert result.tp1_total_contracts == Decimal("70")
        assert result.fast_contracts == Decimal("69")
        assert result.slow_contracts == Decimal("1")
        assert result.min_contracts == Decimal("10")

    def test_invalid_ratios(self):
        """tp1_ratio=0 → invalid_ratios."""
        result = check_three_stage_middle_bucket_split_size(
            position_contracts=Decimal("100"),
            min_contracts=Decimal("1"),
            contract_precision=Decimal("1"),
            three_stage_tp1_ratio=Decimal("0"),
            fast_ratio_of_bucket=Decimal("0.70"),
        )
        assert result.ok is False
        assert result.reason == "invalid_ratios"

    def test_matches_order_specs_rounding(self):
        """Verify the size check uses the same rounding as order_specs."""
        result = check_three_stage_middle_bucket_split_size(
            position_contracts=Decimal("123.456"),
            min_contracts=Decimal("1"),
            contract_precision=Decimal("0.01"),
            three_stage_tp1_ratio=Decimal("0.70"),
            fast_ratio_of_bucket=Decimal("0.70"),
        )
        # Manually compute the expected values using same rounding
        _rnd = lambda c: round_contracts_down(contracts=c, contract_precision=Decimal("0.01"))
        expected_tp1 = _rnd(Decimal("123.456") * Decimal("0.70"))
        expected_fast = _rnd(expected_tp1 * Decimal("0.70"))
        expected_slow = expected_tp1 - expected_fast
        assert result.tp1_total_contracts == expected_tp1
        assert result.fast_contracts == expected_fast
        assert result.slow_contracts == expected_slow


class TestMiddleRunnerSizeCheck:
    """Tests for check_middle_runner_bucket_split_size()."""

    def test_split_ok(self):
        """position=100, partial=0.80, fast_ratio=0.70, min=1.
        partial=80, fast=56, slow=24 → both >= 1 → ok."""
        result = check_middle_runner_bucket_split_size(
            position_contracts=Decimal("100"),
            min_contracts=Decimal("1"),
            contract_precision=Decimal("1"),
            partial_tp_ratio=Decimal("0.80"),
            fast_ratio_of_bucket=Decimal("0.70"),
        )
        assert result.ok is True
        assert result.reason == "ok"
        assert result.tp1_total_contracts == Decimal("80")
        assert result.fast_contracts == Decimal("56")
        assert result.slow_contracts == Decimal("24")

    def test_subleg_too_small(self):
        """position=100, partial=0.80, fast_ratio=0.99, min=10.
        partial=80, fast=79, slow=1 < 10 → not ok."""
        result = check_middle_runner_bucket_split_size(
            position_contracts=Decimal("100"),
            min_contracts=Decimal("10"),
            contract_precision=Decimal("1"),
            partial_tp_ratio=Decimal("0.80"),
            fast_ratio_of_bucket=Decimal("0.99"),
        )
        assert result.ok is False
        assert result.reason == "subleg_too_small"


# ── Order specs subleg too small context tests ─────────────────────────

class TestOrderSpecsSublegTooSmallContext:
    """Verify order_specs fallback_context when split subleg is too small."""

    def test_three_stage_context_includes_all_fields(self):
        split = MiddleBucketSplitOrderInput(
            active=True,
            fast_price=1650.0,
            slow_price=1640.0,
            effective_price=1647.0,
            middle_bucket_ratio=Decimal("0.70"),
            fast_ratio_of_bucket=Decimal("0.99"),
            slow_ratio_of_bucket=Decimal("0.01"),
            fast_total_ratio=Decimal("0.693"),
            slow_total_ratio=Decimal("0.007"),
        )
        decision = build_take_profit_order_specs(
            position_contracts=Decimal("100"),
            min_contracts=Decimal("10"),
            contract_precision=Decimal("1"),
            tp_plan="THREE_STAGE_RUNNER",
            final_tp_price=1700.0,
            partial_tp_price=None,
            partial_tp_ratio=Decimal("0"),
            partial_tp_consumed=False,
            middle_runner_active=False,
            three_stage_tp1_price=1647.0,
            three_stage_tp2_price=1700.0,
            three_stage_tp1_ratio=Decimal("0.70"),
            three_stage_tp2_ratio=Decimal("0.20"),
            three_stage_tp1_consumed=False,
            three_stage_tp2_consumed=False,
            three_stage_runner_ratio=Decimal("0.10"),
            middle_bucket_split=split,
        )

        assert decision.fallback_reason == "MIDDLE_BUCKET_SPLIT_SUBLEG_TOO_SMALL_UNSPLIT"
        ctx = decision.fallback_context
        assert ctx is not None
        assert ctx["split_active"] is True
        assert isinstance(ctx["fast_contracts"], Decimal)
        assert isinstance(ctx["slow_contracts"], Decimal)
        assert ctx["min_contracts"] == Decimal("10")
        # slow should be less than min_contracts (the reason for fallback)
        assert ctx["slow_contracts"] < ctx["min_contracts"]

    def test_middle_runner_context_includes_all_fields(self):
        split = MiddleBucketSplitOrderInput(
            active=True,
            fast_price=1650.0,
            slow_price=1640.0,
            effective_price=1647.0,
            middle_bucket_ratio=Decimal("0.80"),
            fast_ratio_of_bucket=Decimal("0.99"),
            slow_ratio_of_bucket=Decimal("0.01"),
            fast_total_ratio=Decimal("0.792"),
            slow_total_ratio=Decimal("0.008"),
        )
        decision = build_take_profit_order_specs(
            position_contracts=Decimal("100"),
            min_contracts=Decimal("10"),
            contract_precision=Decimal("1"),
            tp_plan="MIDDLE_RUNNER",
            final_tp_price=1700.0,
            partial_tp_price=1647.0,
            partial_tp_ratio=Decimal("0.80"),
            partial_tp_consumed=False,
            middle_runner_active=False,
            three_stage_tp1_price=None,
            three_stage_tp2_price=None,
            three_stage_tp1_ratio=Decimal("0"),
            three_stage_tp2_ratio=Decimal("0"),
            three_stage_tp1_consumed=False,
            three_stage_tp2_consumed=False,
            three_stage_runner_ratio=Decimal("0"),
            middle_bucket_split=split,
        )

        assert decision.fallback_reason == "MIDDLE_BUCKET_SPLIT_SUBLEG_TOO_SMALL_UNSPLIT"
        ctx = decision.fallback_context
        assert ctx is not None
        assert ctx["split_active"] is True
        assert "fast_contracts" in ctx
        assert "slow_contracts" in ctx
        assert ctx["min_contracts"] == Decimal("10")

    def test_no_split_active_no_fallback_context(self):
        """When split is not active, fallback_context is None."""
        decision = build_take_profit_order_specs(
            position_contracts=Decimal("100"),
            min_contracts=Decimal("1"),
            contract_precision=Decimal("1"),
            tp_plan="THREE_STAGE_RUNNER",
            final_tp_price=1700.0,
            partial_tp_price=None,
            partial_tp_ratio=Decimal("0"),
            partial_tp_consumed=False,
            middle_runner_active=False,
            three_stage_tp1_price=1640.0,
            three_stage_tp2_price=1700.0,
            three_stage_tp1_ratio=Decimal("0.70"),
            three_stage_tp2_ratio=Decimal("0.20"),
            three_stage_tp1_consumed=False,
            three_stage_tp2_consumed=False,
            three_stage_runner_ratio=Decimal("0.10"),
            middle_bucket_split=None,
        )
        assert decision.fallback_reason is None
        assert decision.fallback_context is None


# ── LiveTradeResult split status tests ─────────────────────────────────

class TestLiveTradeResultSplitStatus:
    """Verify LiveTradeResult carries split execution status."""

    def test_new_fields_exist_with_defaults(self):
        """New fields exist and default to None."""
        from src.execution.trader import LiveTradeResult
        result = LiveTradeResult(
            ok=True,
            action="UPDATE_TP",
            order_id=None,
            tp_order_id="123",
            contracts="10",
            tp_price="1700.0",
            message="test",
        )
        assert result.middle_bucket_split_executed is None
        assert result.middle_bucket_split_disabled_reason is None

    def test_split_executed_true(self):
        """When split was active and succeeded."""
        from src.execution.trader import LiveTradeResult
        result = LiveTradeResult(
            ok=True,
            action="UPDATE_TP",
            order_id=None,
            tp_order_id="123",
            contracts="10",
            tp_price="1700.0",
            message="test",
            middle_bucket_split_executed=True,
            middle_bucket_split_disabled_reason=None,
        )
        assert result.middle_bucket_split_executed is True
        assert result.middle_bucket_split_disabled_reason is None

    def test_split_disabled_subleg_too_small(self):
        """When split was disabled due to subleg too small."""
        from src.execution.trader import LiveTradeResult
        result = LiveTradeResult(
            ok=True,
            action="UPDATE_TP",
            order_id=None,
            tp_order_id="123",
            contracts="10",
            tp_price="1700.0",
            message="test",
            middle_bucket_split_executed=False,
            middle_bucket_split_disabled_reason="subleg_too_small",
        )
        assert result.middle_bucket_split_executed is False
        assert result.middle_bucket_split_disabled_reason == "subleg_too_small"
