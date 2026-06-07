"""Tests for middle_bucket_split_actual_order_mode in CoreTakeProfitManager.

Verifies that LiveTradeResult carries the correct actual_order_mode value
based on whether the split succeeded, fell back to unsplit, or fell back
to a full-size final TP.
"""

from __future__ import annotations

from decimal import Decimal
from unittest import mock

import pytest


class TestSplitNormalReturnsSplitFastSlow:
    """When split succeeds, actual_order_mode == SPLIT_FAST_SLOW."""

    @pytest.mark.asyncio
    async def test_split_normal_returns_split_fast_slow(self):
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager
        from src.execution.trader import PositionSnapshot, Trader

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

        async def fake_fetch():
            return PositionSnapshot("LONG", Decimal("10"), 3000.0, Decimal("1"), Decimal("10"))
        trader.fetch_position_snapshot = fake_fetch

        facade = TpSlExecutionManager(trader)

        from src.strategies.boll_cvd_reclaim_strategy import TradeIntent
        from src.risk.simple_position_sizer import PositionSize

        intent = TradeIntent(
            intent_type="UPDATE_TP",
            side="LONG",
            price=3000.0,
            layer_index=1,
            tp_price=3100.0,
            reason="test_split_normal",
            size=PositionSize(
                eth_qty=0.1, margin_usdt=300.0, notional_usdt=3000.0,
                layer_index=1, layer_multiplier=1.0,
            ),
            fast_cvd=0.0, previous_fast_cvd=0.0,
            buy_ratio=0.5, sell_ratio=0.5,
            boll_upper=3200.0, boll_middle=3100.0, boll_lower=3000.0,
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

        # Build specs with split (3 labels)
        multi_specs = [
            ("tp1_middle_fast", Decimal("3.43"), 3060.0),
            ("tp1_middle_slow", Decimal("1.47"), 3040.0),
            ("tp2_outer", Decimal("2.00"), 3200.0),
        ]
        facade.core_tp._build_take_profit_order_specs = mock.MagicMock(
            return_value=(multi_specs, None)  # split_disabled_reason=None → split succeeded
        )

        trader._cancel_existing_take_profit_orders_for_intent = mock.AsyncMock()
        trader._cancel_stale_runner_protective_stops_for_degrade = mock.AsyncMock()
        trader._place_reduce_only_take_profit_orders = mock.AsyncMock(
            return_value=["fast-order", "slow-order", "outer-order"]
        )

        result = await facade.replace_take_profit(intent)

        assert result.ok is True
        assert result.middle_bucket_split_executed is True
        assert result.middle_bucket_split_actual_order_mode == "SPLIT_FAST_SLOW"
        assert result.middle_bucket_split_disabled_reason is None


class TestSublegTooSmallReturnsUnsplitMiddleBucket:
    """When subleg too small, actual_order_mode == UNSPLIT_MIDDLE_BUCKET."""

    @pytest.mark.asyncio
    async def test_subleg_too_small_returns_unsplit_middle_bucket(self):
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager
        from src.execution.trader import PositionSnapshot, Trader

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

        async def fake_fetch():
            return PositionSnapshot("LONG", Decimal("10"), 3000.0, Decimal("1"), Decimal("10"))
        trader.fetch_position_snapshot = fake_fetch

        facade = TpSlExecutionManager(trader)

        from src.strategies.boll_cvd_reclaim_strategy import TradeIntent
        from src.risk.simple_position_sizer import PositionSize

        intent = TradeIntent(
            intent_type="UPDATE_TP",
            side="LONG",
            price=3000.0,
            layer_index=1,
            tp_price=3100.0,
            reason="test_subleg_too_small",
            size=PositionSize(
                eth_qty=0.1, margin_usdt=300.0, notional_usdt=3000.0,
                layer_index=1, layer_multiplier=1.0,
            ),
            fast_cvd=0.0, previous_fast_cvd=0.0,
            buy_ratio=0.5, sell_ratio=0.5,
            boll_upper=3200.0, boll_middle=3100.0, boll_lower=3000.0,
            ts_ms=1000,
            avg_entry_price=3000.0,
            breakeven_price=3000.0,
            tp_mode="MIDDLE",
            tp_plan="MIDDLE_RUNNER",
            partial_tp_price=3050.0,
            partial_tp_ratio=0.80,
            middle_bucket_split_active=True,
            middle_bucket_split_fast_price=3060.0,
            middle_bucket_split_slow_price=3040.0,
            middle_bucket_split_effective_price=3054.0,
            middle_bucket_split_middle_bucket_ratio=0.80,
            middle_bucket_split_fast_ratio_of_bucket=0.70,
            middle_bucket_split_slow_ratio_of_bucket=0.30,
            middle_bucket_split_fast_total_ratio=0.56,
            middle_bucket_split_slow_total_ratio=0.24,
        )

        # Build specs with unsplit middle bucket (split_disabled=subleg_too_small)
        unsplit_specs = [
            ("tp1_middle", Decimal("8"), 3050.0),
            ("final", Decimal("2"), 3100.0),
        ]
        facade.core_tp._build_take_profit_order_specs = mock.MagicMock(
            return_value=(unsplit_specs, "subleg_too_small")
        )

        trader._cancel_existing_take_profit_orders_for_intent = mock.AsyncMock()
        trader._cancel_stale_runner_protective_stops_for_degrade = mock.AsyncMock()
        trader._place_reduce_only_take_profit_orders = mock.AsyncMock(
            return_value=["middle-order", "final-order"]
        )

        result = await facade.replace_take_profit(intent)

        assert result.ok is True
        assert result.middle_bucket_split_executed is False
        assert result.middle_bucket_split_disabled_reason == "subleg_too_small"
        assert result.middle_bucket_split_actual_order_mode == "UNSPLIT_MIDDLE_BUCKET"


class TestPlacementFailedReturnsFinalFullSize:
    """When placement fails and fallback is full-size final,
    actual_order_mode == FINAL_FULL_SIZE."""

    @pytest.mark.asyncio
    async def test_placement_failed_returns_final_full_size(self):
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager
        from src.execution.trader import PositionSnapshot, Trader

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

        async def fake_fetch():
            return PositionSnapshot("LONG", Decimal("10"), 3000.0, Decimal("1"), Decimal("10"))
        trader.fetch_position_snapshot = fake_fetch

        facade = TpSlExecutionManager(trader)

        from src.strategies.boll_cvd_reclaim_strategy import TradeIntent
        from src.risk.simple_position_sizer import PositionSize

        intent = TradeIntent(
            intent_type="UPDATE_TP",
            side="LONG",
            price=3000.0,
            layer_index=1,
            tp_price=3100.0,
            reason="test_placement_failed",
            size=PositionSize(
                eth_qty=0.1, margin_usdt=300.0, notional_usdt=3000.0,
                layer_index=1, layer_multiplier=1.0,
            ),
            fast_cvd=0.0, previous_fast_cvd=0.0,
            buy_ratio=0.5, sell_ratio=0.5,
            boll_upper=3200.0, boll_middle=3100.0, boll_lower=3000.0,
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

        # Return multiple specs so except handler triggers split→fallback logic
        multi_specs = [
            ("tp1_middle_fast", Decimal("3.43"), 3060.0),
            ("tp1_middle_slow", Decimal("1.47"), 3040.0),
            ("tp2_outer", Decimal("2.00"), 3200.0),
        ]
        facade.core_tp._build_take_profit_order_specs = mock.MagicMock(
            return_value=(multi_specs, None)
        )

        trader._cancel_existing_take_profit_orders_for_intent = mock.AsyncMock()
        trader._cancel_stale_runner_protective_stops_for_degrade = mock.AsyncMock()

        # First placement raises, second (fallback final) succeeds
        call_count = [0]

        async def mock_place(inner_intent, specs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("simulated placement failure")
            return ["fallback-final-order"]

        trader._place_reduce_only_take_profit_orders = mock_place

        result = await facade.replace_take_profit(intent)

        assert result.ok is True
        assert result.middle_bucket_split_executed is False
        assert (
            result.middle_bucket_split_disabled_reason
            == "split_order_placement_failed_fallback_final"
        )
        assert result.middle_bucket_split_actual_order_mode == "FINAL_FULL_SIZE"
        assert "fallback" in result.message


class TestNoSplitActiveReturnsNone:
    """When split is not active, actual_order_mode is None."""

    @pytest.mark.asyncio
    async def test_no_split_active_returns_none(self):
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager
        from src.execution.trader import PositionSnapshot, Trader

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

        async def fake_fetch():
            return PositionSnapshot("LONG", Decimal("10"), 3000.0, Decimal("1"), Decimal("10"))
        trader.fetch_position_snapshot = fake_fetch

        facade = TpSlExecutionManager(trader)

        from src.strategies.boll_cvd_reclaim_strategy import TradeIntent
        from src.risk.simple_position_sizer import PositionSize

        intent = TradeIntent(
            intent_type="UPDATE_TP",
            side="LONG",
            price=3000.0,
            layer_index=1,
            tp_price=3100.0,
            reason="test_no_split",
            size=PositionSize(
                eth_qty=0.1, margin_usdt=300.0, notional_usdt=3000.0,
                layer_index=1, layer_multiplier=1.0,
            ),
            fast_cvd=0.0, previous_fast_cvd=0.0,
            buy_ratio=0.5, sell_ratio=0.5,
            boll_upper=3200.0, boll_middle=3100.0, boll_lower=3000.0,
            ts_ms=1000,
            avg_entry_price=3000.0,
            breakeven_price=3000.0,
            tp_mode="MIDDLE",
            tp_plan="SINGLE",
            middle_bucket_split_active=False,
        )

        specs = [("final", Decimal("10"), 3100.0)]
        facade.core_tp._build_take_profit_order_specs = mock.MagicMock(
            return_value=(specs, None)
        )

        trader._cancel_existing_take_profit_orders_for_intent = mock.AsyncMock()
        trader._cancel_stale_runner_protective_stops_for_degrade = mock.AsyncMock()
        trader._place_reduce_only_take_profit_orders = mock.AsyncMock(
            return_value=["final-order"]
        )

        result = await facade.replace_take_profit(intent)

        assert result.ok is True
        assert result.middle_bucket_split_executed is None
        assert result.middle_bucket_split_actual_order_mode is None
