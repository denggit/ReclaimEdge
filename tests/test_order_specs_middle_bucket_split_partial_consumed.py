"""Tests for order_specs.py middle bucket split partial consumed handling.

Validates that when one leg of a middle bucket split has been filled
(consumed), the order_specs correctly generates only the unconsumed
leg + tp2_outer, and never re-generates the already-filled leg.
"""

from decimal import Decimal

from src.execution.order_specs import (
    MiddleBucketSplitOrderInput,
    build_take_profit_order_specs,
)


def _make_split_input(
    fast_price=1650.0,
    slow_price=1640.0,
    middle_bucket_ratio=Decimal("0.70"),
    fast_ratio=Decimal("0.70"),
    fast_consumed=False,
    slow_consumed=False,
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
        fast_consumed=fast_consumed,
        slow_consumed=slow_consumed,
    )


# ── Test 1: Fresh split ────────────────────────────────────────────────

def test_fresh_split_all_three_labels():
    """Given split active, fast_consumed=False, slow_consumed=False.
    Then labels == ["tp1_middle_fast", "tp1_middle_slow", "tp2_outer"].
    """
    split = _make_split_input(
        fast_price=1650.0,
        slow_price=1640.0,
        middle_bucket_ratio=Decimal("0.70"),
        fast_ratio=Decimal("0.70"),
        fast_consumed=False,
        slow_consumed=False,
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
        three_stage_tp1_ratio=Decimal("0.70"),
        three_stage_tp2_ratio=Decimal("0.20"),
        three_stage_tp1_consumed=False,
        three_stage_tp2_consumed=False,
        three_stage_runner_ratio=Decimal("0.10"),
        middle_bucket_split=split,
    )

    labels = [s.label for s in decision.specs]
    assert labels == ["tp1_middle_fast", "tp1_middle_slow", "tp2_outer"], f"Unexpected labels: {labels}"


# ── Test 2: Fast consumed only ─────────────────────────────────────────

def test_fast_consumed_only_no_fast_label():
    """Given split active, fast_consumed=True, slow_consumed=False.
    Then labels == ["tp1_middle_slow", "tp2_outer"].
    Assert "tp1_middle_fast" NOT in labels.
    """
    split = _make_split_input(
        fast_price=1650.0,
        slow_price=1640.0,
        middle_bucket_ratio=Decimal("0.70"),
        fast_ratio=Decimal("0.70"),
        fast_consumed=True,
        slow_consumed=False,
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
        three_stage_tp1_ratio=Decimal("0.70"),
        three_stage_tp2_ratio=Decimal("0.20"),
        three_stage_tp1_consumed=False,
        three_stage_tp2_consumed=False,
        three_stage_runner_ratio=Decimal("0.10"),
        middle_bucket_split=split,
    )

    labels = [s.label for s in decision.specs]
    assert "tp1_middle_fast" not in labels, f"tp1_middle_fast should not be generated when fast is already consumed. Labels: {labels}"
    assert "tp1_middle_slow" in labels, f"tp1_middle_slow should be present. Labels: {labels}"
    assert "tp2_outer" in labels, f"tp2_outer should be present. Labels: {labels}"
    assert len(labels) == 2, f"Expected exactly 2 labels, got: {labels}"


# ── Test 3: Slow consumed only ─────────────────────────────────────────

def test_slow_consumed_only_no_slow_label():
    """Given split active, fast_consumed=False, slow_consumed=True.
    Then labels == ["tp1_middle_fast", "tp2_outer"].
    Assert "tp1_middle_slow" NOT in labels.
    """
    split = _make_split_input(
        fast_price=1650.0,
        slow_price=1640.0,
        middle_bucket_ratio=Decimal("0.70"),
        fast_ratio=Decimal("0.70"),
        fast_consumed=False,
        slow_consumed=True,
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
        three_stage_tp1_ratio=Decimal("0.70"),
        three_stage_tp2_ratio=Decimal("0.20"),
        three_stage_tp1_consumed=False,
        three_stage_tp2_consumed=False,
        three_stage_runner_ratio=Decimal("0.10"),
        middle_bucket_split=split,
    )

    labels = [s.label for s in decision.specs]
    assert "tp1_middle_slow" not in labels, f"tp1_middle_slow should not be generated when slow is already consumed. Labels: {labels}"
    assert "tp1_middle_fast" in labels, f"tp1_middle_fast should be present. Labels: {labels}"
    assert "tp2_outer" in labels, f"tp2_outer should be present. Labels: {labels}"
    assert len(labels) == 2, f"Expected exactly 2 labels, got: {labels}"


# ── Test 4: Both consumed and three_stage_tp1_consumed=True ─────────────

def test_both_consumed_tp1_consumed_tp2_only():
    """Given split both consumed and three_stage_tp1_consumed=True.
    Then labels == ["tp2_outer"] (Case A: after TP1 consumed, TP2 pending).
    """
    split = _make_split_input(
        fast_price=1650.0,
        slow_price=1640.0,
        middle_bucket_ratio=Decimal("0.70"),
        fast_ratio=Decimal("0.70"),
        fast_consumed=True,
        slow_consumed=True,
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
        three_stage_tp1_ratio=Decimal("0.70"),
        three_stage_tp2_ratio=Decimal("0.20"),
        three_stage_tp1_consumed=True,
        three_stage_tp2_consumed=False,
        three_stage_runner_ratio=Decimal("0.10"),
        middle_bucket_split=split,
    )

    labels = [s.label for s in decision.specs]
    # Case A: tp1 consumed → only tp2_outer
    assert labels == ["tp2_outer"], f"Expected only tp2_outer when both consumed and tp1_consumed=True. Got: {labels}"


# ── Test 5: Both consumed but three_stage_tp1_consumed=False ────────────

def test_both_consumed_but_tp1_not_consumed_safe_fallback():
    """Given split both consumed but three_stage_tp1_consumed=False.
    Then safe fallback with fallback_reason.
    """
    split = _make_split_input(
        fast_price=1650.0,
        slow_price=1640.0,
        middle_bucket_ratio=Decimal("0.70"),
        fast_ratio=Decimal("0.70"),
        fast_consumed=True,
        slow_consumed=True,
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
        three_stage_tp1_ratio=Decimal("0.70"),
        three_stage_tp2_ratio=Decimal("0.20"),
        three_stage_tp1_consumed=False,
        three_stage_tp2_consumed=False,
        three_stage_runner_ratio=Decimal("0.10"),
        middle_bucket_split=split,
    )

    assert decision.fallback_reason == "MIDDLE_BUCKET_SPLIT_BOTH_CONSUMED_BUT_TP1_NOT_CONSUMED", \
        f"Expected fallback reason, got: {decision.fallback_reason}"
    assert decision.fallback_context is not None
    assert decision.fallback_context["fast_consumed"] is True
    assert decision.fallback_context["slow_consumed"] is True
    assert decision.fallback_context["three_stage_tp1_consumed"] is False


# ── Test: Partial consumed with split not active (old behavior) ────────

def test_partial_consumed_split_not_active_old_behavior():
    """When split is not active, consumed flags are ignored — old behavior preserved."""
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

    labels = [s.label for s in decision.specs]
    assert "tp1_middle" in labels
    assert "tp2_outer" in labels
    assert "tp1_middle_fast" not in labels
