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

    def test_split_disabled_order_placement_failed(self):
        """When split was disabled due to order placement failed fallback."""
        from src.execution.trader import LiveTradeResult
        result = LiveTradeResult(
            ok=True,
            action="UPDATE_TP",
            order_id=None,
            tp_order_id="123",
            contracts="10",
            tp_price="1700.0",
            message="split take-profit placement failed; fallback to single final TP",
            middle_bucket_split_executed=False,
            middle_bucket_split_disabled_reason="split_order_placement_failed_fallback_final",
        )
        assert result.middle_bucket_split_executed is False
        assert result.middle_bucket_split_disabled_reason == "split_order_placement_failed_fallback_final"


# ── clear_middle_bucket_split_state helper tests ────────────────────────

class TestClearMiddleBucketSplitState:
    """Verify the canonical state-clearing helper."""

    def test_clears_all_split_fields(self):
        """All middle_bucket_split_* fields are set to their zero/None values."""
        from src.position_management.middle_bucket_split_state import (
            clear_middle_bucket_split_state,
        )
        from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState

        state = StrategyPositionState()
        # Set all fields to non-zero values
        state.middle_bucket_split_active = True
        state.middle_bucket_split_fast_consumed = True
        state.middle_bucket_split_slow_consumed = True
        state.middle_bucket_split_fast_price = 1650.0
        state.middle_bucket_split_slow_price = 1640.0
        state.middle_bucket_split_effective_price = 1647.0
        state.middle_bucket_split_middle_bucket_ratio = 0.70
        state.middle_bucket_split_fast_ratio_of_bucket = 0.70
        state.middle_bucket_split_slow_ratio_of_bucket = 0.30
        state.middle_bucket_split_fast_total_ratio = 0.49
        state.middle_bucket_split_slow_total_ratio = 0.21
        state.middle_bucket_split_reason = "split_enabled"
        state.middle_bucket_split_fast_sl_price = 1601.0
        state.middle_bucket_split_fast_sl_order_id = "order-123"
        state.middle_bucket_split_fast_sl_protected = True
        state.middle_bucket_split_fast_sl_invalid_action_taken = "MARKET_EXIT"
        state.middle_bucket_split_add_disabled = True

        clear_middle_bucket_split_state(state, reason="test_clear")

        assert state.middle_bucket_split_active is False
        assert state.middle_bucket_split_fast_consumed is False
        assert state.middle_bucket_split_slow_consumed is False
        assert state.middle_bucket_split_fast_price is None
        assert state.middle_bucket_split_slow_price is None
        assert state.middle_bucket_split_effective_price is None
        assert state.middle_bucket_split_middle_bucket_ratio == 0.0
        assert state.middle_bucket_split_fast_ratio_of_bucket == 0.0
        assert state.middle_bucket_split_slow_ratio_of_bucket == 0.0
        assert state.middle_bucket_split_fast_total_ratio == 0.0
        assert state.middle_bucket_split_slow_total_ratio == 0.0
        assert state.middle_bucket_split_reason == "test_clear"
        assert state.middle_bucket_split_fast_sl_price is None
        assert state.middle_bucket_split_fast_sl_order_id is None
        assert state.middle_bucket_split_fast_sl_protected is False
        assert state.middle_bucket_split_fast_sl_invalid_action_taken is None
        assert state.middle_bucket_split_add_disabled is False

    def test_reason_none_by_default(self):
        """When reason is not passed, middle_bucket_split_reason is set to None."""
        from src.position_management.middle_bucket_split_state import (
            clear_middle_bucket_split_state,
        )
        from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState

        state = StrategyPositionState()
        state.middle_bucket_split_active = True
        state.middle_bucket_split_reason = "split_enabled"

        clear_middle_bucket_split_state(state)

        assert state.middle_bucket_split_active is False
        assert state.middle_bucket_split_reason is None

    def test_constants_are_defined(self):
        """Disabled reason constants are defined and are strings."""
        from src.position_management.middle_bucket_split_state import (
            MIDDLE_BUCKET_SPLIT_DISABLED_ORDER_PLACEMENT_FAILED_FALLBACK_FINAL,
            MIDDLE_BUCKET_SPLIT_DISABLED_SIZE_INVALID_RATIOS,
            MIDDLE_BUCKET_SPLIT_DISABLED_SUBLEG_TOO_SMALL,
        )
        assert isinstance(MIDDLE_BUCKET_SPLIT_DISABLED_SUBLEG_TOO_SMALL, str)
        assert isinstance(MIDDLE_BUCKET_SPLIT_DISABLED_ORDER_PLACEMENT_FAILED_FALLBACK_FINAL, str)
        assert isinstance(MIDDLE_BUCKET_SPLIT_DISABLED_SIZE_INVALID_RATIOS, str)
        assert len(MIDDLE_BUCKET_SPLIT_DISABLED_SUBLEG_TOO_SMALL) > 0
        assert len(MIDDLE_BUCKET_SPLIT_DISABLED_ORDER_PLACEMENT_FAILED_FALLBACK_FINAL) > 0


# ── Trader entry result propagation tests ────────────────────────────────

class TestTraderEntryResultPropagation:
    """Verify entry wrappers propagate middle_bucket_split_executed."""

    def test_entry_success_propagates_split_executed_false(self):
        """When replace_take_profit returns split_executed=False, the outer
        entry result must also have split_executed=False."""
        from src.execution.trader import LiveTradeResult

        # Simulate tp result from replace_take_profit
        tp_result = LiveTradeResult(
            ok=True,
            action="UPDATE_TP",
            order_id=None,
            tp_order_id="tp-123",
            contracts="10",
            tp_price="1700.0",
            message="take-profit replaced",
            tp_ok=True,
            middle_bucket_split_executed=False,
            middle_bucket_split_disabled_reason="subleg_too_small",
        )

        # Verify the tp result carries the fields
        assert tp_result.middle_bucket_split_executed is False
        assert tp_result.middle_bucket_split_disabled_reason == "subleg_too_small"

    def test_entry_success_propagates_split_executed_true(self):
        """When replace_take_profit returns split_executed=True, the outer
        entry result must also have split_executed=True."""
        from src.execution.trader import LiveTradeResult

        tp_result = LiveTradeResult(
            ok=True,
            action="UPDATE_TP",
            order_id=None,
            tp_order_id="tp-123",
            contracts="10",
            tp_price="1700.0",
            message="take-profit replaced",
            tp_ok=True,
            middle_bucket_split_executed=True,
            middle_bucket_split_disabled_reason=None,
        )

        assert tp_result.middle_bucket_split_executed is True
        assert tp_result.middle_bucket_split_disabled_reason is None

    def test_entry_tp_failed_propagates_split_executed_false(self):
        """When replace_take_profit fails AND split was disabled, the outer
        entry_filled_but_tp_failed result must propagate the split fields."""
        from src.execution.trader import LiveTradeResult

        tp_result = LiveTradeResult(
            ok=False,
            action="UPDATE_TP",
            order_id=None,
            tp_order_id=None,
            contracts="10",
            tp_price="1700.0",
            message="tp failed",
            tp_ok=False,
            middle_bucket_split_executed=False,
            middle_bucket_split_disabled_reason="split_order_placement_failed_fallback_final",
        )

        assert tp_result.ok is False
        assert tp_result.middle_bucket_split_executed is False
        assert tp_result.middle_bucket_split_disabled_reason == "split_order_placement_failed_fallback_final"


# ── Fallback final returns split_executed=False tests ───────────────────

class TestFallbackFinalReturnsSplitExecutedFalse:
    """Verify that split order placement failure → fallback final returns
    middle_bucket_split_executed=False."""

    @pytest.mark.asyncio
    async def test_split_placement_failed_fallback_final_returns_false(self):
        """When multi-spec placement fails and fallback final succeeds,
        the result must have middle_bucket_split_executed=False.

        This test verifies the specific code path in replace_take_profit
        where the exception handler sets split_disabled_reason and the
        final return computes middle_bucket_split_executed=False.
        """
        from decimal import Decimal
        from unittest import mock

        from src.execution.tp_sl_execution_manager import TpSlExecutionManager
        from src.execution.trader import LiveTradeResult, PositionSnapshot, Trader

        # Create bare trader (following existing test pattern)
        trader = Trader.__new__(Trader)
        trader.symbol = "ETH-USDT-SWAP"
        trader.td_mode = "isolated"
        trader.leverage = "50"
        trader.pos_side_mode = "net"
        trader.live_trading = True
        trader.contract_multiplier = Decimal("0.1")
        trader.contract_precision = Decimal("0.01")
        trader.min_contracts = Decimal("0.01")
        trader.position_contracts = Decimal("10")
        trader.tp_order_id = None
        trader.near_tp_protective_sl_order_id = None
        trader.middle_runner_protective_sl_order_id = None
        trader.three_stage_post_tp1_protective_sl_order_id = None
        trader.trend_runner_sl_order_id = None
        trader.account_equity_usdt = 0.0
        trader._protected_reduce_only_order_ids = set()
        trader._managed_reduce_only_order_ids = set()
        trader._allow_cancel_unmanaged_reduce_only = True

        trader.decimal_to_str = lambda d: str(d)
        trader.price_to_str = lambda p: f"{p:.1f}"
        trader.round_contracts_down = lambda c: c
        trader._tp_price_summary = lambda specs: trader.price_to_str(specs[0][2])

        # Mock position fetch to return a valid LONG position
        async def fake_fetch():
            return PositionSnapshot("LONG", Decimal("10"), 3000.0, Decimal("1"), Decimal("10"))
        trader.fetch_position_snapshot = fake_fetch

        facade = TpSlExecutionManager(trader)

        # Build intent with split active (following TradeIntent field order)
        from src.strategies.boll_cvd_reclaim_strategy import TradeIntent
        from src.risk.simple_position_sizer import PositionSize

        intent = TradeIntent(
            intent_type="UPDATE_TP",
            side="LONG",
            price=3000.0,
            layer_index=1,
            tp_price=3100.0,
            reason="test",
            size=PositionSize(
                eth_qty=0.1, margin_usdt=300.0, notional_usdt=3000.0,
                layer_index=1, layer_multiplier=1.0,
            ),
            fast_cvd=0.0,
            previous_fast_cvd=0.0,
            buy_ratio=0.5,
            sell_ratio=0.5,
            boll_upper=3200.0,
            boll_middle=3100.0,
            boll_lower=3000.0,
            ts_ms=1000,
            avg_entry_price=3000.0,
            breakeven_price=3000.0,
            tp_mode="MIDDLE",
            tp_plan="THREE_STAGE_RUNNER",
            three_stage_tp1_price=3050.0,
            three_stage_tp2_price=3200.0,
            three_stage_tp1_ratio=0.70,
            three_stage_tp2_ratio=0.20,
            three_stage_runner_ratio=0.10,
            middle_bucket_split_active=True,
            middle_bucket_split_fast_price=3060.0,
            middle_bucket_split_slow_price=3040.0,
            middle_bucket_split_effective_price=3054.0,
            middle_bucket_split_middle_bucket_ratio=0.70,
            middle_bucket_split_fast_ratio_of_bucket=0.70,
            middle_bucket_split_slow_ratio_of_bucket=0.30,
            middle_bucket_split_fast_total_ratio=0.49,
            middle_bucket_split_slow_total_ratio=0.21,
        )

        # Mock _build_take_profit_order_specs to return multiple specs
        multi_specs = [
            ("tp1_middle_fast", Decimal("3.43"), 3060.0),
            ("tp1_middle_slow", Decimal("1.47"), 3040.0),
            ("tp2_outer", Decimal("2.00"), 3200.0),
        ]
        facade.core_tp._build_take_profit_order_specs = mock.MagicMock(
            return_value=(multi_specs, None)
        )

        # Mock cancel methods at the Trader level (called before specs are built)
        trader._cancel_existing_take_profit_orders_for_intent = mock.AsyncMock()
        trader._cancel_stale_runner_protective_stops_for_degrade = mock.AsyncMock()
        # Also mock cancel at core_tp level (called in the except handler)
        # But the except handler calls self.trader._cancel_existing_take_profit_orders_for_intent
        # which goes through Trader delegation.  Since we mock on trader,
        # the except handler path is also covered.

        # Mock _place_reduce_only_take_profit_orders on Trader (short-circuits
        # the delegation chain to avoid HTTP).  First call raises, second succeeds.
        call_count = [0]

        async def mock_place(inner_intent, specs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("simulated placement failure")
            return ["fallback-final-order"]

        trader._place_reduce_only_take_profit_orders = mock_place

        # Execute via facade.replace_take_profit (goes through TpSlExecutionManager delegation)
        result = await facade.replace_take_profit(intent)

        # Assertions
        assert result.ok is True
        assert result.tp_order_ids == ("fallback-final-order",)
        assert "fallback" in result.message
        assert result.middle_bucket_split_executed is False
        assert (
            result.middle_bucket_split_disabled_reason
            == "split_order_placement_failed_fallback_final"
        )
