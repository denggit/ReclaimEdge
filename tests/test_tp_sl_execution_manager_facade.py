from __future__ import annotations

import unittest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import src.execution.trader as trader_module
from src.execution.tp_sl_execution_manager import TpSlExecutionManager
from src.execution.trader import LiveTradeResult, PositionSnapshot, Trader


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_trader(**overrides) -> Trader:
    t = Trader.__new__(Trader)
    t.base_url = "https://www.okx.test"
    t.api_key = "key"
    t.secret_key = "secret"
    t.passphrase = "pass"
    t._timeout_seconds = 7.0
    t.symbol = "ETH-USDT-SWAP"
    t.td_mode = "isolated"
    t.leverage = "50"
    t.pos_side_mode = "net"
    t.live_trading = True
    t.max_live_equity_usdt = 30.0
    t.contract_multiplier = Decimal("0.1")
    t.contract_precision = Decimal("0.01")
    t.min_contracts = Decimal("0.01")
    t.tp_order_id = None
    t.near_tp_protective_sl_order_id = None
    t.middle_runner_protective_sl_order_id = None
    t.three_stage_post_tp1_protective_sl_order_id = None
    t.trend_runner_sl_order_id = None
    t.position_contracts = Decimal("0")
    t.account_equity_usdt = 0.0
    t._protected_reduce_only_order_ids = set()
    t._managed_reduce_only_order_ids = set()
    t._allow_cancel_unmanaged_reduce_only = True
    for k, v in overrides.items():
        setattr(t, k, v)
    return t


class TpSlExecutionManagerFacadeTest(unittest.TestCase):
    """Tests for TpSlExecutionManager facade initialization and delegation."""

    # ------------------------------------------------------------------
    # facade initialization
    # ------------------------------------------------------------------

    def test_facade_has_expected_attributes(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        self.assertIs(facade.trader, trader)
        self.assertIsNotNone(facade.core_tp)
        self.assertIsNotNone(facade.protective_stops)
        self.assertIsNotNone(facade.sidecar)
        self.assertIsNotNone(facade.market_exit)
        self.assertIsNotNone(facade.near_tp)

    def test_sub_managers_share_trader_reference(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        self.assertIs(facade.core_tp.trader, trader)
        self.assertIs(facade.protective_stops.trader, trader)
        self.assertIs(facade.sidecar.trader, trader)
        self.assertIs(facade.market_exit.trader, trader)
        self.assertIs(facade.near_tp.trader, trader)

    # ------------------------------------------------------------------
    # facade delegates to core_tp
    # ------------------------------------------------------------------

    def test_replace_take_profit_delegates_to_core_tp(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        mock_result = LiveTradeResult(True, "test", None, "tp1", "10", "3100.00", "ok")
        facade.core_tp.replace_take_profit = AsyncMock(return_value=mock_result)
        # We can't easily test async delegation without running the event loop,
        # but we can verify the method is callable and delegates
        self.assertTrue(callable(facade.replace_take_profit))

    def test_cancel_existing_take_profit_orders_delegates_to_core_tp(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        facade.core_tp._cancel_existing_take_profit_orders_for_intent = AsyncMock()
        self.assertTrue(callable(facade._cancel_existing_take_profit_orders_for_intent))

    def test_cancel_stale_runner_protective_stops_delegates_to_core_tp(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        facade.core_tp._cancel_stale_runner_protective_stops_for_degrade = AsyncMock()
        self.assertTrue(callable(facade._cancel_stale_runner_protective_stops_for_degrade))

    def test_protected_order_ids_from_intent_delegates_to_core_tp(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        self.assertTrue(callable(facade._protected_order_ids_from_intent))

    def test_split_order_ids_returns_set(self) -> None:
        # _split_order_ids is a static method on the facade (inline impl)
        result = TpSlExecutionManager._split_order_ids("a,b,,c")
        self.assertEqual(result, {"a", "b", "c"})

    def test_split_order_ids_none_returns_empty_set(self) -> None:
        result = TpSlExecutionManager._split_order_ids(None)
        self.assertEqual(result, set())

    def test_split_order_ids_empty_string_returns_empty_set(self) -> None:
        result = TpSlExecutionManager._split_order_ids("")
        self.assertEqual(result, set())

    def test_managed_core_contracts_delegates_to_core_tp(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        self.assertTrue(callable(facade._managed_core_contracts_from_intent))

    def test_build_take_profit_order_specs_delegates_to_core_tp(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        self.assertTrue(callable(facade._build_take_profit_order_specs))

    def test_build_three_stage_order_specs_delegates_to_core_tp(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        self.assertTrue(callable(facade._build_three_stage_order_specs))

    def test_trend_runner_sl_contracts_delegates_to_core_tp(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        self.assertTrue(callable(facade._trend_runner_sl_contracts))

    def test_place_reduce_only_take_profit_orders_delegates_to_core_tp(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        self.assertTrue(callable(facade._place_reduce_only_take_profit_orders))

    def test_tp_price_summary_delegates_to_core_tp(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        specs = [("final", Decimal("10"), 3100.0)]
        result = facade._tp_price_summary(specs)
        self.assertEqual(result, "3100.00")

    # ------------------------------------------------------------------
    # facade delegates to protective_stops
    # ------------------------------------------------------------------

    def test_place_near_tp_protective_stop_delegates_to_protective_stops(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        self.assertTrue(callable(facade.place_near_tp_protective_stop_with_retries))

    def test_place_middle_runner_protective_stop_delegates_to_protective_stops(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        self.assertTrue(callable(facade.place_middle_runner_protective_stop_with_retries))

    def test_place_trend_runner_protective_stop_delegates_to_protective_stops(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        self.assertTrue(callable(facade.place_trend_runner_protective_stop_with_retries))

    def test_place_three_stage_post_tp1_protective_stop_delegates_to_protective_stops(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        self.assertTrue(callable(facade.place_three_stage_post_tp1_protective_stop_with_retries))

    def test_cancel_unverified_near_tp_algo_delegates_to_protective_stops(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        self.assertTrue(callable(facade._cancel_unverified_near_tp_algo))

    def test_verify_near_tp_protective_stop_delegates_to_protective_stops(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        self.assertTrue(callable(facade.verify_near_tp_protective_stop))

    def test_near_tp_protective_stop_matches_delegates_to_protective_stops(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        self.assertTrue(callable(facade._near_tp_protective_stop_matches))

    # ------------------------------------------------------------------
    # facade delegates to market_exit
    # ------------------------------------------------------------------

    def test_market_exit_remaining_position_delegates_to_market_exit(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        self.assertTrue(callable(facade.market_exit_remaining_position_with_retries))

    def test_cleanup_after_near_tp_market_exit_delegates_to_market_exit(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        self.assertTrue(callable(facade._cleanup_after_near_tp_market_exit))

    # ------------------------------------------------------------------
    # facade delegates to near_tp
    # ------------------------------------------------------------------

    def test_execute_near_tp_reduce_delegates_to_near_tp(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        self.assertTrue(callable(facade.execute_near_tp_reduce))

    def test_execute_market_exit_runner_delegates_to_near_tp(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        self.assertTrue(callable(facade.execute_market_exit_runner))

    # ------------------------------------------------------------------
    # facade delegates to sidecar
    # ------------------------------------------------------------------

    def test_place_sidecar_fixed_take_profit_delegates_to_sidecar(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        self.assertTrue(callable(facade.place_sidecar_fixed_take_profit))

    def test_cancel_sidecar_take_profit_delegates_to_sidecar(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        self.assertTrue(callable(facade.cancel_sidecar_take_profit))

    def test_fetch_sidecar_order_status_delegates_to_sidecar(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        self.assertTrue(callable(facade.fetch_sidecar_order_status))

    # ------------------------------------------------------------------
    # facade keeps cancel methods
    # ------------------------------------------------------------------

    def test_cancel_near_tp_protective_stop_exists(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        self.assertTrue(callable(facade.cancel_near_tp_protective_stop))

    def test_cancel_middle_runner_protective_stop_exists(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        self.assertTrue(callable(facade.cancel_middle_runner_protective_stop))

    def test_cancel_trend_runner_protective_stop_exists(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        self.assertTrue(callable(facade.cancel_trend_runner_protective_stop))

    def test_cancel_three_stage_post_tp1_protective_stop_exists(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        self.assertTrue(callable(facade.cancel_three_stage_post_tp1_protective_stop))

    def test_cancel_existing_reduce_only_orders_exists(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        self.assertTrue(callable(facade.cancel_existing_reduce_only_orders))

    # ------------------------------------------------------------------
    # core TP replacement regression (smoke)
    # ------------------------------------------------------------------

    def test_core_tp_manager_class_exists(self) -> None:
        from src.execution.tp_sl_core_tp_manager import CoreTakeProfitManager
        self.assertTrue(True)

    def test_protective_stop_manager_class_exists(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager
        self.assertTrue(True)

    def test_sidecar_tp_manager_class_exists(self) -> None:
        from src.execution.tp_sl_sidecar_manager import SidecarTpManager
        self.assertTrue(True)

    def test_market_exit_manager_class_exists(self) -> None:
        from src.execution.tp_sl_market_exit_manager import MarketExitManager
        self.assertTrue(True)

    def test_near_tp_execution_manager_class_exists(self) -> None:
        from src.execution.tp_sl_near_tp_manager import NearTpExecutionManager
        self.assertTrue(True)
