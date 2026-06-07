"""Tests for order_specs.py middle bucket split integration."""

from decimal import Decimal

from src.execution.order_specs import (
    MiddleBucketSplitOrderInput,
    TakeProfitOrderSpec,
    build_take_profit_order_specs,
)


def _make_split_input(
    fast_price=1650.0,
    slow_price=1640.0,
    middle_bucket_ratio=Decimal("0.70"),
    fast_ratio=Decimal("0.70"),
):
    """Helper to create a MiddleBucketSplitOrderInput with computed ratios."""
    fast_total = middle_bucket_ratio * fast_ratio
    slow_total = middle_bucket_ratio * (Decimal("1") - fast_ratio)
    return MiddleBucketSplitOrderInput(
        active=True,
        fast_price=fast_price,
        slow_price=slow_price,
        effective_price=fast_price * float(fast_ratio) + slow_price * (1 - float(fast_ratio)),
        middle_bucket_ratio=middle_bucket_ratio,
        fast_ratio_of_bucket=fast_ratio,
        slow_ratio_of_bucket=Decimal("1") - fast_ratio,
        fast_total_ratio=fast_total,
        slow_total_ratio=slow_total,
    )


class TestThreeStageSplit:
    """Three-Stage TP with middle bucket split active."""

    def test_split_generates_fast_slow_tp2(self):
        """position=100, tp1=0.70, tp2=0.20, runner=0.10, fast_ratio=0.70.
        Expected: tp1_total=70, fast=49, slow=21, tp2=20, runner implicit=10.
        """
        split = _make_split_input(
            fast_price=1650.0,
            slow_price=1640.0,
            middle_bucket_ratio=Decimal("0.70"),
            fast_ratio=Decimal("0.70"),
        )
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
            three_stage_tp1_price=1647.0,  # effective
            three_stage_tp2_price=1700.0,
            three_stage_tp1_ratio=Decimal("0.70"),
            three_stage_tp2_ratio=Decimal("0.20"),
            three_stage_tp1_consumed=False,
            three_stage_tp2_consumed=False,
            three_stage_runner_ratio=Decimal("0.10"),
            middle_bucket_split=split,
        )

        specs = decision.specs
        labels = [s.label for s in specs]
        assert "tp1_middle_fast" in labels
        assert "tp1_middle_slow" in labels
        assert "tp2_outer" in labels

        # fast = round(70 * 0.70) = 49
        # slow = 70 - 49 = 21
        # tp2 = round(100 * 0.20) = 20
        fast_spec = [s for s in specs if s.label == "tp1_middle_fast"][0]
        slow_spec = [s for s in specs if s.label == "tp1_middle_slow"][0]
        tp2_spec = [s for s in specs if s.label == "tp2_outer"][0]
        assert fast_spec.contracts == Decimal("49")
        assert slow_spec.contracts == Decimal("21")
        assert tp2_spec.contracts == Decimal("20")
        assert fast_spec.price == 1650.0
        assert slow_spec.price == 1640.0
        assert tp2_spec.price == 1700.0

    def test_different_tp1_ratio_not_hardcoded(self):
        """With THREE_STAGE_TP1_RATIO=0.60 instead of 0.70.
        fast = 0.60 * 0.70 * 100 = 42, slow = 0.60 * 0.30 * 100 = 18.
        """
        split = _make_split_input(
            fast_price=1650.0,
            slow_price=1640.0,
            middle_bucket_ratio=Decimal("0.60"),
            fast_ratio=Decimal("0.70"),
        )
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
            three_stage_tp1_price=1647.0,
            three_stage_tp2_price=1700.0,
            three_stage_tp1_ratio=Decimal("0.60"),
            three_stage_tp2_ratio=Decimal("0.20"),
            three_stage_tp1_consumed=False,
            three_stage_tp2_consumed=False,
            three_stage_runner_ratio=Decimal("0.20"),
            middle_bucket_split=split,
        )

        specs = decision.specs
        fast_spec = [s for s in specs if s.label == "tp1_middle_fast"][0]
        slow_spec = [s for s in specs if s.label == "tp1_middle_slow"][0]
        # fast = round(60 * 0.70) = 42
        # slow = 60 - 42 = 18
        assert fast_spec.contracts == Decimal("42")
        assert slow_spec.contracts == Decimal("18")

    def test_split_not_active_old_behavior_unchanged(self):
        """When split is not active, old behavior is preserved."""
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

        specs = decision.specs
        labels = [s.label for s in specs]
        assert "tp1_middle" in labels
        assert "tp2_outer" in labels
        assert "tp1_middle_fast" not in labels

    def test_split_fallback_when_sub_leg_too_small(self):
        """When fast or slow contracts < min_contracts, fallback to unsplit tp1_middle."""
        split = _make_split_input(
            fast_price=1650.0,
            slow_price=1640.0,
            middle_bucket_ratio=Decimal("0.70"),
            fast_ratio=Decimal("0.99"),  # slow would be 1 contract
        )
        decision = build_take_profit_order_specs(
            position_contracts=Decimal("100"),
            min_contracts=Decimal("10"),  # high min_contracts to trigger fallback
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

        specs = decision.specs
        labels = [s.label for s in specs]
        # Should fallback to unsplit: tp1_middle + tp2_outer, not single final
        assert "tp1_middle" in labels
        assert "tp1_middle_fast" not in labels

        # Verify enriched fallback context
        assert decision.fallback_reason == "MIDDLE_BUCKET_SPLIT_SUBLEG_TOO_SMALL_UNSPLIT"
        assert decision.fallback_context is not None
        ctx = decision.fallback_context
        assert ctx["split_active"] is True
        assert "fast_contracts" in ctx
        assert "slow_contracts" in ctx
        assert "min_contracts" in ctx
        assert ctx["min_contracts"] == Decimal("10")
        # fast = round(70 * 0.99) = 69, slow = 70 - 69 = 1 < 10
        assert int(ctx["slow_contracts"]) < int(ctx["min_contracts"])


class TestMiddleRunnerSplit:
    """Middle Runner TP with middle bucket split active."""

    def test_split_generates_middle_fast_slow_runner(self):
        """position=100, partial=0.80, fast_ratio=0.70.
        Expected: middle_total=80, fast=56, slow=24, runner=20.
        """
        split = _make_split_input(
            fast_price=1650.0,
            slow_price=1640.0,
            middle_bucket_ratio=Decimal("0.80"),
            fast_ratio=Decimal("0.70"),
        )
        decision = build_take_profit_order_specs(
            position_contracts=Decimal("100"),
            min_contracts=Decimal("1"),
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

        specs = decision.specs
        labels = [s.label for s in specs]
        assert "middle_fast" in labels
        assert "middle_slow" in labels
        assert "runner" in labels

        fast_spec = [s for s in specs if s.label == "middle_fast"][0]
        slow_spec = [s for s in specs if s.label == "middle_slow"][0]
        runner_spec = [s for s in specs if s.label == "runner"][0]
        # fast = round(80 * 0.70) = 56
        # slow = 80 - 56 = 24
        # runner = 100 - 80 = 20
        assert fast_spec.contracts == Decimal("56")
        assert slow_spec.contracts == Decimal("24")
        assert runner_spec.contracts == Decimal("20")

    def test_different_first_close_ratio_not_hardcoded(self):
        """With FIRST_CLOSE_RATIO=0.75: middle_total=75, fast=round(75*0.70)=52, slow=23."""
        split = _make_split_input(
            fast_price=1650.0,
            slow_price=1640.0,
            middle_bucket_ratio=Decimal("0.75"),
            fast_ratio=Decimal("0.70"),
        )
        decision = build_take_profit_order_specs(
            position_contracts=Decimal("100"),
            min_contracts=Decimal("1"),
            contract_precision=Decimal("1"),
            tp_plan="MIDDLE_RUNNER",
            final_tp_price=1700.0,
            partial_tp_price=1647.0,
            partial_tp_ratio=Decimal("0.75"),
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

        specs = decision.specs
        fast_spec = [s for s in specs if s.label == "middle_fast"][0]
        slow_spec = [s for s in specs if s.label == "middle_slow"][0]
        # fast = round(75 * 0.70) = 52 (round down from 52.5)
        # slow = 75 - 52 = 23
        assert fast_spec.contracts == Decimal("52")
        assert slow_spec.contracts == Decimal("23")

    def test_split_not_active_middle_runner_old_behavior(self):
        """When split is not active for Middle Runner, old labels preserved."""
        decision = build_take_profit_order_specs(
            position_contracts=Decimal("100"),
            min_contracts=Decimal("1"),
            contract_precision=Decimal("1"),
            tp_plan="MIDDLE_RUNNER",
            final_tp_price=1700.0,
            partial_tp_price=1640.0,
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
            middle_bucket_split=None,
        )

        specs = decision.specs
        labels = [s.label for s in specs]
        assert "middle" in labels
        assert "runner" in labels
        assert "middle_fast" not in labels

    def test_split_fallback_when_sub_leg_too_small_middle_runner(self):
        """Middle Runner: when fast or slow contracts < min_contracts, fallback to unsplit."""
        split = _make_split_input(
            fast_price=1650.0,
            slow_price=1640.0,
            middle_bucket_ratio=Decimal("0.80"),
            fast_ratio=Decimal("0.99"),  # slow = ~1 contract
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

        specs = decision.specs
        labels = [s.label for s in specs]
        assert "middle" in labels
        assert "middle_fast" not in labels

        assert decision.fallback_reason == "MIDDLE_BUCKET_SPLIT_SUBLEG_TOO_SMALL_UNSPLIT"
        assert decision.fallback_context is not None
        ctx = decision.fallback_context
        assert ctx["split_active"] is True
        assert "fast_contracts" in ctx
        assert "slow_contracts" in ctx
        assert "min_contracts" in ctx
        assert ctx["min_contracts"] == Decimal("10")
