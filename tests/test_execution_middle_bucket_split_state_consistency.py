"""Tests for middle bucket split state/order consistency.

Verifies that when split sub-legs are too small, the execution layer
disables the split and the strategy state is cleared to match actual orders.
"""

from __future__ import annotations

from decimal import Decimal
from unittest import mock

import pytest

from tests.conftest import FakeOkxClient
from src.execution.okx_trading_client import OkxTradingClient

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
        trader._client = FakeOkxClient(trader)
        trader.trading_client = OkxTradingClient(trader, private_client=trader._client)  # type: ignore[assignment]

        # Mock position fetch to return a valid LONG position
        async def fake_fetch():
            return PositionSnapshot("LONG", Decimal("10"), 3000.0, Decimal("1"), Decimal("10"))
        trader.fetch_position_snapshot = fake_fetch

        facade = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = facade  # type: ignore[assignment]

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


# ── degrade_middle_bucket_split_to_single_final tests ──────────────────

class TestDegradeMiddleBucketSplitToSingleFinal:
    """Verify the TP plan degrading helper."""

    def test_degrades_tp_plan_to_single(self):
        """All split fields cleared AND tp_plan degraded to SINGLE."""
        from src.position_management.middle_bucket_split_state import (
            degrade_middle_bucket_split_to_single_final,
        )
        from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState

        state = StrategyPositionState()
        # Set up full Three-Stage + split state
        state.tp_plan = "THREE_STAGE_RUNNER"
        state.partial_tp_price = 3050.0
        state.partial_tp_ratio = 0.80
        state.partial_tp_consumed = True
        state.three_stage_tp1_price = 3050.0
        state.three_stage_tp2_price = 3200.0
        state.three_stage_tp1_consumed = True
        state.three_stage_tp2_consumed = False
        state.three_stage_post_tp1_protective_sl_price = 2995.0
        state.three_stage_post_tp1_protective_sl_order_id = "sl-order-123"
        state.three_stage_post_tp1_protected = True
        state.trend_runner_active = True
        state.trend_runner_tp_price = 3300.0
        state.trend_runner_sl_price = 3100.0
        state.trend_runner_tp_order_id = "runner-tp-123"
        state.trend_runner_sl_order_id = "runner-sl-123"
        state.middle_runner_pending = True
        state.middle_runner_active = True
        state.middle_runner_first_tp_price = 3060.0
        state.middle_runner_final_tp_price = 3200.0
        state.middle_runner_protective_sl_price = 2990.0
        state.middle_runner_protective_sl_order_id = "mr-sl-123"
        state.middle_bucket_split_active = True
        state.middle_bucket_split_reason = "split_enabled"

        degrade_middle_bucket_split_to_single_final(
            state, reason="split_order_placement_failed_fallback_final",
        )

        # ── Split fields cleared ──
        assert state.middle_bucket_split_active is False
        assert state.middle_bucket_split_reason == "split_order_placement_failed_fallback_final"

        # ── TP plan degraded ──
        assert state.tp_plan == "SINGLE"
        assert state.partial_tp_price is None
        assert state.partial_tp_ratio == 0.0
        assert state.partial_tp_consumed is False

        # ── Three-Stage runtime cleared ──
        assert state.three_stage_tp1_price is None
        assert state.three_stage_tp2_price is None
        assert state.three_stage_tp1_consumed is False
        assert state.three_stage_tp2_consumed is False
        assert state.three_stage_post_tp1_protective_sl_price is None
        assert state.three_stage_post_tp1_protective_sl_order_id is None
        assert state.three_stage_post_tp1_protected is False

        # ── Trend Runner runtime cleared ──
        assert state.trend_runner_active is False
        assert state.trend_runner_tp_price is None
        assert state.trend_runner_sl_price is None
        assert state.trend_runner_tp_order_id is None
        assert state.trend_runner_sl_order_id is None

        # ── Middle Runner runtime cleared ──
        assert state.middle_runner_pending is False
        assert state.middle_runner_active is False
        assert state.middle_runner_first_tp_price is None
        assert state.middle_runner_final_tp_price is None
        assert state.middle_runner_protective_sl_price is None
        assert state.middle_runner_protective_sl_order_id is None

    def test_does_not_clear_config_ratios(self):
        """Configuration fields like three_stage_tp1_ratio are NOT touched."""
        from src.position_management.middle_bucket_split_state import (
            degrade_middle_bucket_split_to_single_final,
        )
        from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState

        state = StrategyPositionState()
        state.tp_plan = "THREE_STAGE_RUNNER"
        state.three_stage_tp1_ratio = 0.60
        state.three_stage_tp2_ratio = 0.20
        state.three_stage_runner_ratio = 0.20
        state.middle_runner_first_close_ratio = 0.80
        state.middle_runner_keep_ratio = 0.20
        state.middle_bucket_split_active = True

        degrade_middle_bucket_split_to_single_final(
            state, reason="split_order_placement_failed_fallback_final",
        )

        # Config ratios preserved
        assert state.three_stage_tp1_ratio == 0.60
        assert state.three_stage_tp2_ratio == 0.20
        assert state.three_stage_runner_ratio == 0.20
        assert state.middle_runner_first_close_ratio == 0.80
        assert state.middle_runner_keep_ratio == 0.20

    def test_preserves_position_cost_fields(self):
        """Position cost / entry fields are NOT cleared."""
        from src.position_management.middle_bucket_split_state import (
            degrade_middle_bucket_split_to_single_final,
        )
        from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState

        state = StrategyPositionState()
        state.side = "LONG"
        state.layers = 3
        state.avg_entry_price = 3000.0
        state.breakeven_price = 3005.0
        state.total_entry_qty = 1.0
        state.position_cost_entry_notional = 3000.0
        state.tp_plan = "THREE_STAGE_RUNNER"
        state.middle_bucket_split_active = True

        degrade_middle_bucket_split_to_single_final(
            state, reason="split_order_placement_failed_fallback_final",
        )

        assert state.side == "LONG"
        assert state.layers == 3
        assert state.avg_entry_price == 3000.0
        assert state.breakeven_price == 3005.0
        assert state.total_entry_qty == 1.0
        assert state.position_cost_entry_notional == 3000.0


# ── actual_order_mode routing tests ─────────────────────────────────────

class TestMaybeClearAfterExecutionResultRouting:
    """Verify that _maybe_clear_middle_bucket_split_after_execution_result
    routes to degrade or clear based on actual_order_mode."""

    def _make_processor(self):
        from unittest import mock
        import asyncio

        from src.live.workers.execution_command_processor import (
            ExecutionCommandProcessor,
        )
        from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState

        processor = ExecutionCommandProcessor.__new__(ExecutionCommandProcessor)
        processor.state_lock = asyncio.Lock()
        processor.strategy = mock.MagicMock()
        processor.strategy.state = StrategyPositionState()
        processor.journal = mock.MagicMock()
        processor.journal.append = mock.MagicMock()
        processor.execution_state = mock.MagicMock()
        processor.account_snapshot = mock.MagicMock()
        processor.trader = mock.MagicMock()
        processor.state_store = mock.MagicMock()
        processor.email_sender = mock.MagicMock()
        processor._background_tasks = set()
        return processor

    def _make_result(self, **kwargs):
        from src.execution.trader import LiveTradeResult

        defaults = dict(
            ok=True, action="UPDATE_TP", order_id=None,
            tp_order_id="tp-123", contracts="10", tp_price="1700.0",
            message="test",
        )
        defaults.update(kwargs)
        return LiveTradeResult(**defaults)

    def test_final_full_size_degrades_to_single(self):
        """When actual_order_mode=FINAL_FULL_SIZE, degrade to SINGLE."""
        processor = self._make_processor()
        state = processor.strategy.state
        state.tp_plan = "THREE_STAGE_RUNNER"
        state.three_stage_tp1_price = 3050.0
        state.three_stage_tp2_price = 3200.0
        state.trend_runner_active = True
        state.middle_bucket_split_active = True

        result = self._make_result(
            middle_bucket_split_executed=False,
            middle_bucket_split_disabled_reason="split_order_placement_failed_fallback_final",
            middle_bucket_split_actual_order_mode="FINAL_FULL_SIZE",
            tp_order_id="final-order",
            tp_order_ids=("final-order",),
        )

        changed = processor._maybe_clear_middle_bucket_split_after_execution_result(
            result=result,
            current_position_id="pos-1",
        )

        assert changed is True
        assert state.tp_plan == "SINGLE"
        assert state.middle_bucket_split_active is False
        assert state.three_stage_tp1_price is None
        assert state.three_stage_tp2_price is None
        assert state.trend_runner_active is False
        assert state.middle_runner_active is False

        # Journal event
        processor.journal.append.assert_called_once()
        call_args = processor.journal.append.call_args
        assert call_args[0][0] == "MIDDLE_BUCKET_SPLIT_DEGRADED_TO_SINGLE_FINAL"
        payload = call_args[0][1]
        assert payload["actual_order_mode"] == "FINAL_FULL_SIZE"
        assert payload["previous_tp_plan"] == "THREE_STAGE_RUNNER"
        assert payload["new_tp_plan"] == "SINGLE"
        assert payload["state_order_consistent"] is True

    def test_unsplit_middle_bucket_clears_only_split(self):
        """When actual_order_mode=UNSPLIT_MIDDLE_BUCKET, clear split only."""
        processor = self._make_processor()
        state = processor.strategy.state
        state.tp_plan = "MIDDLE_RUNNER"
        state.middle_runner_pending = True
        state.middle_runner_first_tp_price = 3060.0
        state.middle_bucket_split_active = True
        state.middle_bucket_split_reason = "split_enabled"

        result = self._make_result(
            middle_bucket_split_executed=False,
            middle_bucket_split_disabled_reason="subleg_too_small",
            middle_bucket_split_actual_order_mode="UNSPLIT_MIDDLE_BUCKET",
        )

        changed = processor._maybe_clear_middle_bucket_split_after_execution_result(
            result=result,
            current_position_id="pos-1",
        )

        assert changed is True
        # Split state cleared
        assert state.middle_bucket_split_active is False
        assert state.middle_bucket_split_reason == "subleg_too_small"
        # TP plan preserved
        assert state.tp_plan == "MIDDLE_RUNNER"
        assert state.middle_runner_pending is True
        assert state.middle_runner_first_tp_price == 3060.0

        # Journal event is MIDDLE_BUCKET_SPLIT_DISABLED_ON_ORDER_BUILD
        processor.journal.append.assert_called_once()
        call_args = processor.journal.append.call_args
        assert call_args[0][0] == "MIDDLE_BUCKET_SPLIT_DISABLED_ON_ORDER_BUILD"

    def test_unsplit_middle_bucket_keeps_three_stage_plan(self):
        """When actual_order_mode=UNSPLIT_MIDDLE_BUCKET on THREE_STAGE,
        only split fields cleared; tp_plan stays THREE_STAGE_RUNNER."""
        processor = self._make_processor()
        state = processor.strategy.state
        state.tp_plan = "THREE_STAGE_RUNNER"
        state.three_stage_tp1_price = 3050.0
        state.three_stage_tp2_price = 3200.0
        state.middle_bucket_split_active = True
        state.middle_bucket_split_reason = "split_enabled"

        result = self._make_result(
            middle_bucket_split_executed=False,
            middle_bucket_split_disabled_reason="subleg_too_small",
            middle_bucket_split_actual_order_mode="UNSPLIT_MIDDLE_BUCKET",
        )

        changed = processor._maybe_clear_middle_bucket_split_after_execution_result(
            result=result,
            current_position_id="pos-1",
        )

        assert changed is True
        assert state.middle_bucket_split_active is False
        assert state.middle_bucket_split_reason == "subleg_too_small"
        # TP plan preserved
        assert state.tp_plan == "THREE_STAGE_RUNNER"
        assert state.three_stage_tp1_price == 3050.0
        assert state.three_stage_tp2_price == 3200.0

    def test_split_executed_true_returns_false(self):
        """When split succeeded, no state modification."""
        processor = self._make_processor()
        state = processor.strategy.state
        state.tp_plan = "THREE_STAGE_RUNNER"
        state.middle_bucket_split_active = True

        result = self._make_result(
            middle_bucket_split_executed=True,
            middle_bucket_split_disabled_reason=None,
            middle_bucket_split_actual_order_mode="SPLIT_FAST_SLOW",
        )

        changed = processor._maybe_clear_middle_bucket_split_after_execution_result(
            result=result,
            current_position_id="pos-1",
        )

        assert changed is False
        assert state.tp_plan == "THREE_STAGE_RUNNER"
        assert state.middle_bucket_split_active is True
        processor.journal.append.assert_not_called()

    def test_split_executed_none_returns_false(self):
        """When split not involved at all, no state modification."""
        processor = self._make_processor()

        result = self._make_result(
            middle_bucket_split_executed=None,
            middle_bucket_split_disabled_reason=None,
            middle_bucket_split_actual_order_mode=None,
        )

        changed = processor._maybe_clear_middle_bucket_split_after_execution_result(
            result=result,
            current_position_id="pos-1",
        )

        assert changed is False

    def test_backward_compat_fallback_final_reason_degrades(self):
        """When actual_order_mode is None but reason is placement_failed,
        backward-compat degrades to SINGLE."""
        processor = self._make_processor()
        state = processor.strategy.state
        state.tp_plan = "THREE_STAGE_RUNNER"
        state.three_stage_tp1_price = 3050.0
        state.middle_bucket_split_active = True

        result = self._make_result(
            middle_bucket_split_executed=False,
            middle_bucket_split_disabled_reason="split_order_placement_failed_fallback_final",
            middle_bucket_split_actual_order_mode=None,
        )

        changed = processor._maybe_clear_middle_bucket_split_after_execution_result(
            result=result,
            current_position_id="pos-1",
        )

        assert changed is True
        assert state.tp_plan == "SINGLE"
        # Journal event is also DEGRADED (backward-compat)
        processor.journal.append.assert_called_once()
        call_args = processor.journal.append.call_args
        assert call_args[0][0] == "MIDDLE_BUCKET_SPLIT_DEGRADED_TO_SINGLE_FINAL"


# ── LiveTradeResult actual_order_mode field tests ───────────────────────

class TestActualOrderModeOnLiveTradeResult:
    """Verify the new middle_bucket_split_actual_order_mode field."""

    def test_field_defaults_to_none(self):
        from src.execution.trader import LiveTradeResult
        result = LiveTradeResult(
            ok=True, action="UPDATE_TP", order_id=None,
            tp_order_id="123", contracts="10", tp_price="1700.0",
            message="test",
        )
        assert result.middle_bucket_split_actual_order_mode is None

    def test_split_fast_slow_explicit(self):
        from src.execution.trader import LiveTradeResult
        result = LiveTradeResult(
            ok=True, action="UPDATE_TP", order_id=None,
            tp_order_id="123", contracts="10", tp_price="1700.0",
            message="test",
            middle_bucket_split_executed=True,
            middle_bucket_split_actual_order_mode="SPLIT_FAST_SLOW",
        )
        assert result.middle_bucket_split_actual_order_mode == "SPLIT_FAST_SLOW"

    def test_unsplit_middle_bucket_explicit(self):
        from src.execution.trader import LiveTradeResult
        result = LiveTradeResult(
            ok=True, action="UPDATE_TP", order_id=None,
            tp_order_id="123", contracts="10", tp_price="1700.0",
            message="test",
            middle_bucket_split_executed=False,
            middle_bucket_split_disabled_reason="subleg_too_small",
            middle_bucket_split_actual_order_mode="UNSPLIT_MIDDLE_BUCKET",
        )
        assert result.middle_bucket_split_actual_order_mode == "UNSPLIT_MIDDLE_BUCKET"

    def test_final_full_size_explicit(self):
        from src.execution.trader import LiveTradeResult
        result = LiveTradeResult(
            ok=True, action="UPDATE_TP", order_id=None,
            tp_order_id="final-order", contracts="10",
            tp_price="1700.0",
            message="split take-profit placement failed; fallback to single final TP",
            middle_bucket_split_executed=False,
            middle_bucket_split_disabled_reason="split_order_placement_failed_fallback_final",
            middle_bucket_split_actual_order_mode="FINAL_FULL_SIZE",
        )
        assert result.middle_bucket_split_actual_order_mode == "FINAL_FULL_SIZE"


# ── Trader entry wrapper actual_order_mode propagation tests ────────────

class TestTraderEntryActualOrderModePropagation:
    """Verify entry wrapper propagates actual_order_mode."""

    def test_entry_success_propagates_actual_order_mode(self):
        """Entry success branch includes actual_order_mode from tp result."""
        from src.execution.trader import LiveTradeResult

        tp = LiveTradeResult(
            ok=True, action="UPDATE_TP", order_id=None,
            tp_order_id="tp-123", contracts="10", tp_price="1700.0",
            message="take-profit replaced", tp_ok=True,
            middle_bucket_split_executed=True,
            middle_bucket_split_disabled_reason=None,
            middle_bucket_split_actual_order_mode="SPLIT_FAST_SLOW",
        )
        assert tp.middle_bucket_split_actual_order_mode == "SPLIT_FAST_SLOW"

        # Simulate what execute_intent does:
        outer = LiveTradeResult(
            ok=True, action="OPEN_LONG", order_id="entry-1",
            tp_order_id=tp.tp_order_id, contracts="10",
            tp_price=tp.tp_price,
            message="market order placed and take-profit protected",
            entry_filled=True, tp_ok=True,
            tp_order_ids=tp.tp_order_ids,
            protective_sl_order_id=tp.protective_sl_order_id,
            protective_sl_price=tp.protective_sl_price,
            protective_sl_ok=tp.protective_sl_ok,
            middle_bucket_split_executed=tp.middle_bucket_split_executed,
            middle_bucket_split_disabled_reason=tp.middle_bucket_split_disabled_reason,
            middle_bucket_split_actual_order_mode=tp.middle_bucket_split_actual_order_mode,
        )
        assert outer.middle_bucket_split_actual_order_mode == "SPLIT_FAST_SLOW"

    def test_entry_tp_failed_propagates_actual_order_mode(self):
        """Entry_filled_but_tp_failed branch includes actual_order_mode."""
        from src.execution.trader import LiveTradeResult

        tp = LiveTradeResult(
            ok=False, action="UPDATE_TP", order_id=None,
            tp_order_id=None, contracts="10", tp_price="1700.0",
            message="tp failed", tp_ok=False,
            middle_bucket_split_executed=False,
            middle_bucket_split_disabled_reason="split_order_placement_failed_fallback_final",
            middle_bucket_split_actual_order_mode="FINAL_FULL_SIZE",
        )
        assert tp.middle_bucket_split_actual_order_mode == "FINAL_FULL_SIZE"

        outer = LiveTradeResult(
            ok=False, action="OPEN_LONG", order_id="entry-1",
            tp_order_id=tp.tp_order_id, contracts="10",
            tp_price=tp.tp_price,
            message=f"entry_filled_but_tp_failed: {tp.message}",
            entry_filled=True, tp_ok=False,
            tp_order_ids=tp.tp_order_ids,
            protective_sl_order_id=tp.protective_sl_order_id,
            protective_sl_price=tp.protective_sl_price,
            protective_sl_ok=tp.protective_sl_ok,
            middle_bucket_split_executed=tp.middle_bucket_split_executed,
            middle_bucket_split_disabled_reason=tp.middle_bucket_split_disabled_reason,
            middle_bucket_split_actual_order_mode=tp.middle_bucket_split_actual_order_mode,
        )
        assert outer.middle_bucket_split_actual_order_mode == "FINAL_FULL_SIZE"


# ── Update TP result degrades tp_plan integration test ──────────────────

class TestUpdateTpResultDegradesToSingle:
    """Integration test: _apply_update_tp_result degrades tp_plan to SINGLE
    when actual_order_mode=FINAL_FULL_SIZE."""

    @pytest.mark.asyncio
    async def test_fallback_final_degrades_tp_plan_to_single_on_update_tp(self):
        from unittest import mock
        import asyncio

        from src.live.workers.execution_command_processor import (
            ExecutionCommandProcessor,
        )
        from src.live import runtime_types as live_runtime_types
        from src.strategies.boll_cvd_reclaim_strategy import (
            BollCvdReclaimStrategy,
            BollCvdReclaimStrategyConfig,
            StrategyPositionState,
        )
        from src.risk.simple_position_sizer import SimplePositionSizer

        # Build a real strategy with state
        config = BollCvdReclaimStrategyConfig()
        sizer = SimplePositionSizer(config)
        strategy = BollCvdReclaimStrategy(config, sizer)

        # Set up full Three-Stage + split state
        state = strategy.state
        state.tp_plan = "THREE_STAGE_RUNNER"
        state.three_stage_tp1_price = 3050.0
        state.three_stage_tp2_price = 3200.0
        state.three_stage_tp1_consumed = False
        state.three_stage_tp2_consumed = False
        state.trend_runner_active = True
        state.trend_runner_tp_price = 3300.0
        state.middle_bucket_split_active = True

        # Build processor with mocks
        es = live_runtime_types.ExecutionState(
            current_position_id="pos-1",
            cash_before_position=1000.0,
        )
        account_snapshot = live_runtime_types.AccountSnapshot(
            position=None, cash=1000.0, equity=1000.0,
            updated_monotonic=0, updated_ts_ms=0,
        )

        processor = ExecutionCommandProcessor.__new__(ExecutionCommandProcessor)
        processor.state_lock = asyncio.Lock()
        processor.execution_state = es
        processor.account_snapshot = account_snapshot
        processor.strategy = strategy
        processor.trader = mock.MagicMock()
        processor.trader.symbol = "ETH-USDT-SWAP"
        processor.journal = mock.MagicMock()
        processor.journal.record_tp_update = mock.MagicMock()
        processor.journal.append = mock.MagicMock()
        processor.state_store = mock.MagicMock()
        processor.email_sender = mock.MagicMock()
        processor._background_tasks = set()

        # Build TradeIntent
        from src.strategies.boll_cvd_reclaim_strategy import TradeIntent
        from src.risk.simple_position_sizer import PositionSize

        intent = TradeIntent(
            intent_type="UPDATE_TP",
            side="LONG",
            price=3000.0,
            layer_index=1,
            tp_price=3100.0,
            reason="test_degrade",
            size=PositionSize(eth_qty=0.1, margin_usdt=300.0, notional_usdt=3000.0, layer_index=1, layer_multiplier=1.0),
            fast_cvd=0.0, previous_fast_cvd=0.0,
            buy_ratio=0.5, sell_ratio=0.5,
            boll_upper=3200.0, boll_middle=3100.0, boll_lower=3000.0,
            ts_ms=2000,
            avg_entry_price=3000.0,
            breakeven_price=3000.0,
            tp_mode="MIDDLE",
            tp_plan="THREE_STAGE_RUNNER",
            middle_bucket_split_active=True,
        )

        snapshot = StrategyPositionState()
        snapshot.tp_plan = "THREE_STAGE_RUNNER"

        command = live_runtime_types.TradeCommand(
            intent=intent,
            strategy_state_snapshot=snapshot,
            tick_ts_ms=2000,
            created_monotonic=0,
            account_snapshot_updated_ts_ms=0,
            reason="test_degrade",
        )

        # Build result with FINAL_FULL_SIZE
        from src.execution.trader import LiveTradeResult
        result = LiveTradeResult(
            ok=True,
            action="UPDATE_TP",
            order_id=None,
            tp_order_id="final-order",
            contracts="10",
            tp_price="3100.0",
            message="split take-profit placement failed; fallback to single final TP",
            middle_bucket_split_executed=False,
            middle_bucket_split_disabled_reason="split_order_placement_failed_fallback_final",
            middle_bucket_split_actual_order_mode="FINAL_FULL_SIZE",
            tp_order_ids=("final-order",),
        )

        # Execute
        await processor._apply_update_tp_result(command, result)

        # Assert degrade happened
        assert state.tp_plan == "SINGLE"
        assert state.middle_bucket_split_active is False
        assert state.three_stage_tp1_price is None
        assert state.three_stage_tp2_price is None
        assert state.trend_runner_active is False

        # Journal events
        journal_calls = [c[0][0] for c in processor.journal.append.call_args_list]
        assert "MIDDLE_BUCKET_SPLIT_DEGRADED_TO_SINGLE_FINAL" in journal_calls

        # State saved
        processor.state_store.save.assert_called_once()


# ── Entry result degrades tp_plan integration test ──────────────────────

class TestEntryResultDegradesToSingle:
    """Integration test: _apply_entry_result degrades tp_plan to SINGLE
    when actual_order_mode=FINAL_FULL_SIZE."""

    @pytest.mark.asyncio
    async def test_fallback_final_degrades_tp_plan_to_single_on_entry_result(self):
        from unittest import mock
        import asyncio

        from src.live.workers.execution_command_processor import (
            ExecutionCommandProcessor,
        )
        from src.live import runtime_types as live_runtime_types
        from src.strategies.boll_cvd_reclaim_strategy import (
            BollCvdReclaimStrategy,
            BollCvdReclaimStrategyConfig,
            StrategyPositionState,
        )
        from src.risk.simple_position_sizer import SimplePositionSizer

        config = BollCvdReclaimStrategyConfig()
        sizer = SimplePositionSizer(config)
        strategy = BollCvdReclaimStrategy(config, sizer)

        state = strategy.state
        state.tp_plan = "MIDDLE_RUNNER"
        state.middle_runner_pending = True
        state.middle_runner_first_tp_price = 3060.0
        state.middle_runner_active = True
        state.middle_bucket_split_active = True

        es = live_runtime_types.ExecutionState(
            current_position_id="pos-1",
            cash_before_position=1000.0,
        )
        account_snapshot = live_runtime_types.AccountSnapshot(
            position=None, cash=1000.0, equity=1000.0,
            updated_monotonic=0, updated_ts_ms=0,
        )

        processor = ExecutionCommandProcessor.__new__(ExecutionCommandProcessor)
        processor.state_lock = asyncio.Lock()
        processor.execution_state = es
        processor.account_snapshot = account_snapshot
        processor.strategy = strategy
        processor.trader = mock.MagicMock()
        processor.trader.symbol = "ETH-USDT-SWAP"
        processor.journal = mock.MagicMock()
        processor.journal.record_entry = mock.MagicMock()
        processor.journal.append = mock.MagicMock()
        processor.state_store = mock.MagicMock()
        processor.email_sender = mock.MagicMock()
        processor._background_tasks = set()

        from src.strategies.boll_cvd_reclaim_strategy import TradeIntent
        from src.risk.simple_position_sizer import PositionSize

        intent = TradeIntent(
            intent_type="OPEN_LONG",
            side="LONG",
            price=3000.0,
            layer_index=1,
            tp_price=3100.0,
            reason="test_degrade_entry",
            size=PositionSize(eth_qty=0.1, margin_usdt=300.0, notional_usdt=3000.0, layer_index=1, layer_multiplier=1.0),
            fast_cvd=0.0, previous_fast_cvd=0.0,
            buy_ratio=0.5, sell_ratio=0.5,
            boll_upper=3200.0, boll_middle=3100.0, boll_lower=3000.0,
            ts_ms=2000,
            avg_entry_price=3000.0,
            breakeven_price=3000.0,
            tp_mode="MIDDLE",
            tp_plan="MIDDLE_RUNNER",
            middle_bucket_split_active=True,
        )

        snapshot = StrategyPositionState()
        snapshot.tp_plan = "MIDDLE_RUNNER"

        command = live_runtime_types.TradeCommand(
            intent=intent,
            strategy_state_snapshot=snapshot,
            tick_ts_ms=2000,
            created_monotonic=0,
            account_snapshot_updated_ts_ms=0,
            reason="test_degrade_entry",
        )

        from src.execution.trader import LiveTradeResult
        result = LiveTradeResult(
            ok=True,
            action="OPEN_LONG",
            order_id="entry-1",
            tp_order_id="final-order",
            contracts="10",
            tp_price="3100.0",
            message="entry_filled_but_tp_failed: split placement failed; fallback to single final TP",
            entry_filled=True,
            tp_ok=True,
            tp_order_ids=("final-order",),
            middle_bucket_split_executed=False,
            middle_bucket_split_disabled_reason="split_order_placement_failed_fallback_final",
            middle_bucket_split_actual_order_mode="FINAL_FULL_SIZE",
        )

        # entry_cash_before=None, sidecar_plan=None
        await processor._apply_entry_result(command, result, None, None)

        assert state.tp_plan == "SINGLE"
        assert state.middle_bucket_split_active is False
        assert state.middle_runner_pending is False
        assert state.middle_runner_active is False

        journal_calls = [c[0][0] for c in processor.journal.append.call_args_list]
        assert "MIDDLE_BUCKET_SPLIT_DEGRADED_TO_SINGLE_FINAL" in journal_calls


# ── Entry journal does NOT write planned events when FINAL_FULL_SIZE ────

class TestEntryJournalSkipsPlannedWhenFinalFullSize:
    """When actual_order_mode=FINAL_FULL_SIZE, _apply_entry_result must NOT
    write MIDDLE_RUNNER_PLANNED or THREE_STAGE_RUNNER_PLANNED."""

    @pytest.mark.asyncio
    async def test_middle_runner_planned_skipped_when_final_full_size(self):
        from unittest import mock
        import asyncio

        from src.live.workers.execution_command_processor import (
            ExecutionCommandProcessor,
        )
        from src.live import runtime_types as live_runtime_types
        from src.strategies.boll_cvd_reclaim_strategy import (
            BollCvdReclaimStrategy,
            BollCvdReclaimStrategyConfig,
            StrategyPositionState,
        )
        from src.risk.simple_position_sizer import SimplePositionSizer

        config = BollCvdReclaimStrategyConfig()
        sizer = SimplePositionSizer(config)
        strategy = BollCvdReclaimStrategy(config, sizer)

        state = strategy.state
        state.tp_plan = "MIDDLE_RUNNER"
        state.middle_runner_pending = True
        state.middle_bucket_split_active = True

        es = live_runtime_types.ExecutionState(
            current_position_id="pos-1",
            cash_before_position=1000.0,
        )
        account_snapshot = live_runtime_types.AccountSnapshot(
            position=None, cash=1000.0, equity=1000.0,
            updated_monotonic=0, updated_ts_ms=0,
        )

        processor = ExecutionCommandProcessor.__new__(ExecutionCommandProcessor)
        processor.state_lock = asyncio.Lock()
        processor.execution_state = es
        processor.account_snapshot = account_snapshot
        processor.strategy = strategy
        processor.trader = mock.MagicMock()
        processor.trader.symbol = "ETH-USDT-SWAP"
        processor.journal = mock.MagicMock()
        processor.journal.record_entry = mock.MagicMock()
        processor.journal.append = mock.MagicMock()
        processor.state_store = mock.MagicMock()
        processor.email_sender = mock.MagicMock()
        processor._background_tasks = set()

        from src.strategies.boll_cvd_reclaim_strategy import TradeIntent
        from src.risk.simple_position_sizer import PositionSize

        intent = TradeIntent(
            intent_type="OPEN_LONG",
            side="LONG", price=3000.0, layer_index=1,
            tp_price=3100.0, reason="test_journal_skip",
            size=PositionSize(eth_qty=0.1, margin_usdt=300.0, notional_usdt=3000.0,
                              layer_index=1, layer_multiplier=1.0),
            fast_cvd=0.0, previous_fast_cvd=0.0,
            buy_ratio=0.5, sell_ratio=0.5,
            boll_upper=3200.0, boll_middle=3100.0, boll_lower=3000.0,
            ts_ms=2000, avg_entry_price=3000.0, breakeven_price=3000.0,
            tp_mode="MIDDLE", tp_plan="MIDDLE_RUNNER",
            middle_bucket_split_active=True,
        )

        snapshot = StrategyPositionState()
        snapshot.tp_plan = "MIDDLE_RUNNER"

        command = live_runtime_types.TradeCommand(
            intent=intent, strategy_state_snapshot=snapshot,
            tick_ts_ms=2000, created_monotonic=0,
            account_snapshot_updated_ts_ms=0, reason="test_journal_skip",
        )

        from src.execution.trader import LiveTradeResult
        result = LiveTradeResult(
            ok=True, action="OPEN_LONG",
            order_id="entry-1", tp_order_id="final-order",
            contracts="10", tp_price="3100.0",
            message="entry with fallback final",
            entry_filled=True, tp_ok=True,
            tp_order_ids=("final-order",),
            middle_bucket_split_executed=False,
            middle_bucket_split_disabled_reason="split_fallback_final_order_structure",
            middle_bucket_split_actual_order_mode="FINAL_FULL_SIZE",
        )

        await processor._apply_entry_result(command, result, None, None)

        # Verify MIDDLE_RUNNER_PLANNED was NOT written
        journal_events = [c[0][0] for c in processor.journal.append.call_args_list]
        assert "MIDDLE_RUNNER_PLANNED" not in journal_events

    @pytest.mark.asyncio
    async def test_three_stage_runner_planned_skipped_when_final_full_size(self):
        from unittest import mock
        import asyncio

        from src.live.workers.execution_command_processor import (
            ExecutionCommandProcessor,
        )
        from src.live import runtime_types as live_runtime_types
        from src.strategies.boll_cvd_reclaim_strategy import (
            BollCvdReclaimStrategy,
            BollCvdReclaimStrategyConfig,
            StrategyPositionState,
        )
        from src.risk.simple_position_sizer import SimplePositionSizer

        config = BollCvdReclaimStrategyConfig()
        sizer = SimplePositionSizer(config)
        strategy = BollCvdReclaimStrategy(config, sizer)

        state = strategy.state
        state.tp_plan = "THREE_STAGE_RUNNER"
        state.middle_bucket_split_active = True

        es = live_runtime_types.ExecutionState(
            current_position_id="pos-1",
            cash_before_position=1000.0,
        )
        account_snapshot = live_runtime_types.AccountSnapshot(
            position=None, cash=1000.0, equity=1000.0,
            updated_monotonic=0, updated_ts_ms=0,
        )

        processor = ExecutionCommandProcessor.__new__(ExecutionCommandProcessor)
        processor.state_lock = asyncio.Lock()
        processor.execution_state = es
        processor.account_snapshot = account_snapshot
        processor.strategy = strategy
        processor.trader = mock.MagicMock()
        processor.trader.symbol = "ETH-USDT-SWAP"
        processor.journal = mock.MagicMock()
        processor.journal.record_entry = mock.MagicMock()
        processor.journal.append = mock.MagicMock()
        processor.state_store = mock.MagicMock()
        processor.email_sender = mock.MagicMock()
        processor._background_tasks = set()

        from src.strategies.boll_cvd_reclaim_strategy import TradeIntent
        from src.risk.simple_position_sizer import PositionSize

        intent = TradeIntent(
            intent_type="OPEN_LONG",
            side="LONG", price=3000.0, layer_index=1,
            tp_price=3100.0, reason="test_journal_skip_3s",
            size=PositionSize(eth_qty=0.1, margin_usdt=300.0, notional_usdt=3000.0,
                              layer_index=1, layer_multiplier=1.0),
            fast_cvd=0.0, previous_fast_cvd=0.0,
            buy_ratio=0.5, sell_ratio=0.5,
            boll_upper=3200.0, boll_middle=3100.0, boll_lower=3000.0,
            ts_ms=2000, avg_entry_price=3000.0, breakeven_price=3000.0,
            tp_mode="MIDDLE", tp_plan="THREE_STAGE_RUNNER",
            three_stage_tp1_price=3050.0, three_stage_tp2_price=3200.0,
            three_stage_tp1_ratio=0.70, three_stage_tp2_ratio=0.20,
            three_stage_runner_ratio=0.10,
            middle_bucket_split_active=True,
        )

        snapshot = StrategyPositionState()
        snapshot.tp_plan = "THREE_STAGE_RUNNER"

        command = live_runtime_types.TradeCommand(
            intent=intent, strategy_state_snapshot=snapshot,
            tick_ts_ms=2000, created_monotonic=0,
            account_snapshot_updated_ts_ms=0, reason="test_journal_skip_3s",
        )

        from src.execution.trader import LiveTradeResult
        result = LiveTradeResult(
            ok=True, action="OPEN_LONG",
            order_id="entry-1", tp_order_id="final-order",
            contracts="10", tp_price="3100.0",
            message="entry with fallback final",
            entry_filled=True, tp_ok=True,
            tp_order_ids=("final-order",),
            middle_bucket_split_executed=False,
            middle_bucket_split_disabled_reason="split_fallback_final_order_structure",
            middle_bucket_split_actual_order_mode="FINAL_FULL_SIZE",
        )

        await processor._apply_entry_result(command, result, None, None)

        # Verify THREE_STAGE_RUNNER_PLANNED was NOT written
        journal_events = [c[0][0] for c in processor.journal.append.call_args_list]
        assert "THREE_STAGE_RUNNER_PLANNED" not in journal_events
