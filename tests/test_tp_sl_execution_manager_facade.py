from __future__ import annotations

import unittest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from tests.conftest import FakeOkxClient
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
    t._client = FakeOkxClient(t)
    for k, v in overrides.items():
        setattr(t, k, v)
    return t


# ---------------------------------------------------------------------------
# sync facade attribute / helper tests (no event loop needed)
# ---------------------------------------------------------------------------

class TpSlExecutionManagerFacadeTest(unittest.TestCase):
    """Tests for TpSlExecutionManager facade initialization and sync helpers."""

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
    # _split_order_ids (static method)
    # ------------------------------------------------------------------

    def test_split_order_ids_returns_set(self) -> None:
        result = TpSlExecutionManager._split_order_ids("a,b,,c")
        self.assertEqual(result, {"a", "b", "c"})

    def test_split_order_ids_none_returns_empty_set(self) -> None:
        result = TpSlExecutionManager._split_order_ids(None)
        self.assertEqual(result, set())

    def test_split_order_ids_empty_string_returns_empty_set(self) -> None:
        result = TpSlExecutionManager._split_order_ids("")
        self.assertEqual(result, set())

    # ------------------------------------------------------------------
    # _tp_price_summary (sync delegation)
    # ------------------------------------------------------------------

    def test_tp_price_summary_delegates_to_core_tp(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        specs = [("final", Decimal("10"), 3100.0)]
        mock_result = "3100.00"
        facade.core_tp._tp_price_summary = MagicMock(return_value=mock_result)
        result = facade._tp_price_summary(specs)
        self.assertEqual(result, mock_result)
        facade.core_tp._tp_price_summary.assert_called_once_with(specs)

    # ------------------------------------------------------------------
    # _protected_order_ids_from_intent (sync delegation)
    # ------------------------------------------------------------------

    def test_protected_order_ids_from_intent_delegates_to_core_tp(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        mock_ids = {"a", "b"}
        facade.core_tp._protected_order_ids_from_intent = MagicMock(return_value=mock_ids)
        result = facade._protected_order_ids_from_intent(None)
        self.assertEqual(result, mock_ids)
        facade.core_tp._protected_order_ids_from_intent.assert_called_once_with(None)

    # ------------------------------------------------------------------
    # _managed_core_contracts_from_intent (sync delegation)
    # ------------------------------------------------------------------

    def test_managed_core_contracts_delegates_to_core_tp(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        facade.core_tp._managed_core_contracts_from_intent = MagicMock(return_value=Decimal("5"))
        result = facade._managed_core_contracts_from_intent(None)
        self.assertEqual(result, Decimal("5"))
        facade.core_tp._managed_core_contracts_from_intent.assert_called_once_with(None)

    # ------------------------------------------------------------------
    # _build_take_profit_order_specs (sync delegation)
    # ------------------------------------------------------------------

    def test_build_take_profit_order_specs_delegates_to_core_tp(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        mock_specs = [("final", Decimal("10"), 3100.0)]
        facade.core_tp._build_take_profit_order_specs_public = MagicMock(return_value=mock_specs)
        result = facade._build_take_profit_order_specs(None)
        self.assertEqual(result, mock_specs)
        facade.core_tp._build_take_profit_order_specs_public.assert_called_once_with(None)

    # ------------------------------------------------------------------
    # _build_three_stage_order_specs (sync delegation)
    # ------------------------------------------------------------------

    def test_build_three_stage_order_specs_delegates_to_core_tp(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        mock_specs = [("partial", Decimal("3"), 3050.0), ("final", Decimal("7"), 3100.0)]
        facade.core_tp._build_three_stage_order_specs_public = MagicMock(return_value=mock_specs)
        result = facade._build_three_stage_order_specs(None)
        self.assertEqual(result, mock_specs)
        facade.core_tp._build_three_stage_order_specs_public.assert_called_once_with(None)

    # ------------------------------------------------------------------
    # _trend_runner_sl_contracts (sync delegation)
    # ------------------------------------------------------------------

    def test_trend_runner_sl_contracts_delegates_to_core_tp(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        facade.core_tp._trend_runner_sl_contracts = MagicMock(return_value=Decimal("5"))
        result = facade._trend_runner_sl_contracts(None, Decimal("10"))
        self.assertEqual(result, Decimal("5"))
        facade.core_tp._trend_runner_sl_contracts.assert_called_once_with(None, Decimal("10"))

    # ------------------------------------------------------------------
    # _near_tp_protective_stop_matches (sync delegation)
    # ------------------------------------------------------------------

    def test_near_tp_protective_stop_matches_delegates_to_protective_stops(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        facade.protective_stops._near_tp_protective_stop_matches = MagicMock(return_value=True)
        item = {"algoId": "algo-1"}
        result = facade._near_tp_protective_stop_matches(item, "algo-1", "LONG", Decimal("10"), 3000.0)
        self.assertTrue(result)
        facade.protective_stops._near_tp_protective_stop_matches.assert_called_once_with(
            item, "algo-1", "LONG", Decimal("10"), 3000.0)

    # ------------------------------------------------------------------
    # class existence (smoke)
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


# ---------------------------------------------------------------------------
# async delegation tests (assert real await + called_with)
# ---------------------------------------------------------------------------

class TpSlExecutionManagerFacadeAsyncTest(unittest.IsolatedAsyncioTestCase):
    """Async tests that prove facade methods truly delegate to sub-managers."""

    # ------------------------------------------------------------------
    # core_tp delegation
    # ------------------------------------------------------------------

    async def test_replace_take_profit_delegates_to_core_tp(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        mock_result = LiveTradeResult(True, "test", None, "tp1", "10", "3100.00", "ok")
        facade.core_tp.replace_take_profit = AsyncMock(return_value=mock_result)
        result = await facade.replace_take_profit(None)
        self.assertIs(result, mock_result)
        facade.core_tp.replace_take_profit.assert_awaited_once_with(None)

    async def test_cancel_existing_take_profit_orders_delegates_to_core_tp(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        facade.core_tp._cancel_existing_take_profit_orders_for_intent = AsyncMock()
        await facade._cancel_existing_take_profit_orders_for_intent(None)
        facade.core_tp._cancel_existing_take_profit_orders_for_intent.assert_awaited_once_with(None)

    async def test_cancel_stale_runner_protective_stops_delegates_to_core_tp(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        facade.core_tp._cancel_stale_runner_protective_stops_for_degrade = AsyncMock()
        await facade._cancel_stale_runner_protective_stops_for_degrade(None)
        facade.core_tp._cancel_stale_runner_protective_stops_for_degrade.assert_awaited_once_with(None)

    async def test_place_reduce_only_take_profit_orders_delegates_to_core_tp(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        mock_order_ids = ["ord-1", "ord-2"]
        facade.core_tp._place_reduce_only_take_profit_orders = AsyncMock(return_value=mock_order_ids)
        specs = [("final", Decimal("10"), 3100.0)]
        result = await facade._place_reduce_only_take_profit_orders(None, specs)
        self.assertEqual(result, mock_order_ids)
        facade.core_tp._place_reduce_only_take_profit_orders.assert_awaited_once_with(None, specs)

    # ------------------------------------------------------------------
    # near_tp delegation
    # ------------------------------------------------------------------

    async def test_execute_near_tp_reduce_delegates_to_near_tp(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        mock_result = LiveTradeResult(True, "test", None, None, "5", "3050.00", "ok")
        facade.near_tp.execute_near_tp_reduce = AsyncMock(return_value=mock_result)
        result = await facade.execute_near_tp_reduce(None)
        self.assertIs(result, mock_result)
        facade.near_tp.execute_near_tp_reduce.assert_awaited_once_with(None)

    async def test_execute_market_exit_runner_delegates_to_near_tp(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        mock_result = LiveTradeResult(True, "test", None, None, "0", "0.00", "ok")
        facade.near_tp.execute_market_exit_runner = AsyncMock(return_value=mock_result)
        result = await facade.execute_market_exit_runner(None)
        self.assertIs(result, mock_result)
        facade.near_tp.execute_market_exit_runner.assert_awaited_once_with(None)

    # ------------------------------------------------------------------
    # protective_stops delegation
    # ------------------------------------------------------------------

    async def test_place_near_tp_protective_stop_delegates_to_protective_stops(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        mock_result = (True, "algo-1", "protective_sl_placed")
        facade.protective_stops.place_near_tp_protective_stop_with_retries = AsyncMock(return_value=mock_result)
        ok, algo_id, msg = await facade.place_near_tp_protective_stop_with_retries(
            "LONG", Decimal("10"), 3050.0, 3, 0.1)
        self.assertTrue(ok)
        self.assertEqual(algo_id, "algo-1")
        self.assertEqual(msg, "protective_sl_placed")
        facade.protective_stops.place_near_tp_protective_stop_with_retries.assert_awaited_once_with(
            "LONG", Decimal("10"), 3050.0, 3, 0.1)

    async def test_place_middle_runner_protective_stop_delegates_to_protective_stops(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        mock_result = (True, "algo-mid", "placed")
        facade.protective_stops.place_middle_runner_protective_stop_with_retries = AsyncMock(return_value=mock_result)
        ok, algo_id, msg = await facade.place_middle_runner_protective_stop_with_retries(
            "SHORT", Decimal("5"), 2900.0, 2, 0.2)
        self.assertTrue(ok)
        facade.protective_stops.place_middle_runner_protective_stop_with_retries.assert_awaited_once_with(
            "SHORT", Decimal("5"), 2900.0, 2, 0.2)

    async def test_place_trend_runner_protective_stop_delegates_to_protective_stops(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        mock_result = (True, "algo-trend", "placed")
        facade.protective_stops.place_trend_runner_protective_stop_with_retries = AsyncMock(return_value=mock_result)
        ok, algo_id, msg = await facade.place_trend_runner_protective_stop_with_retries(
            "LONG", Decimal("8"), 3200.0, 3, 0.1)
        self.assertTrue(ok)
        facade.protective_stops.place_trend_runner_protective_stop_with_retries.assert_awaited_once_with(
            "LONG", Decimal("8"), 3200.0, 3, 0.1)

    async def test_place_three_stage_post_tp1_protective_stop_delegates_to_protective_stops(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        mock_result = (True, "algo-3s", "placed")
        facade.protective_stops.place_three_stage_post_tp1_protective_stop_with_retries = AsyncMock(
            return_value=mock_result)
        ok, algo_id, msg = await facade.place_three_stage_post_tp1_protective_stop_with_retries(
            "LONG", Decimal("3"), 3150.0, 2, 0.1)
        self.assertTrue(ok)
        facade.protective_stops.place_three_stage_post_tp1_protective_stop_with_retries.assert_awaited_once_with(
            "LONG", Decimal("3"), 3150.0, 2, 0.1)

    async def test_cancel_unverified_near_tp_algo_delegates_to_protective_stops(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        facade.protective_stops._cancel_unverified_near_tp_algo = AsyncMock()
        await facade._cancel_unverified_near_tp_algo("algo-x", phase="test")
        facade.protective_stops._cancel_unverified_near_tp_algo.assert_awaited_once_with(
            "algo-x", phase="test")

    async def test_verify_near_tp_protective_stop_delegates_to_protective_stops(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        facade.protective_stops.verify_near_tp_protective_stop = AsyncMock(return_value=True)
        result = await facade.verify_near_tp_protective_stop("algo-1", "LONG", Decimal("10"), 3000.0)
        self.assertTrue(result)
        facade.protective_stops.verify_near_tp_protective_stop.assert_awaited_once_with(
            "algo-1", "LONG", Decimal("10"), 3000.0)

    # ------------------------------------------------------------------
    # market_exit delegation
    # ------------------------------------------------------------------

    async def test_market_exit_remaining_position_delegates_to_market_exit(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        mock_result = (True, "market_exit_order_id=exit-1")
        facade.market_exit.market_exit_remaining_position_with_retries = AsyncMock(return_value=mock_result)
        ok, msg = await facade.market_exit_remaining_position_with_retries("LONG", 3)
        self.assertTrue(ok)
        self.assertIn("exit-1", msg)
        facade.market_exit.market_exit_remaining_position_with_retries.assert_awaited_once_with("LONG", 3, context="generic", retry_interval_seconds=None)

    async def test_cleanup_after_near_tp_market_exit_delegates_to_market_exit(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        facade.market_exit._cleanup_after_near_tp_market_exit = AsyncMock()
        await facade._cleanup_after_near_tp_market_exit()
        facade.market_exit._cleanup_after_near_tp_market_exit.assert_awaited_once()

    # ------------------------------------------------------------------
    # sidecar delegation
    # ------------------------------------------------------------------

    async def test_place_sidecar_fixed_take_profit_delegates_to_sidecar(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        mock_order_id = "tp-sid-1"
        facade.sidecar.place_sidecar_fixed_take_profit = AsyncMock(return_value=mock_order_id)
        result = await facade.place_sidecar_fixed_take_profit(
            side="LONG", contracts="10", tp_price=3100.0, client_order_id="cl-123")
        self.assertEqual(result, mock_order_id)
        facade.sidecar.place_sidecar_fixed_take_profit.assert_awaited_once_with(
            side="LONG", contracts="10", tp_price=3100.0, client_order_id="cl-123")

    async def test_cancel_sidecar_take_profit_delegates_to_sidecar(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        facade.sidecar.cancel_sidecar_take_profit = AsyncMock(return_value=True)
        result = await facade.cancel_sidecar_take_profit("tp-1")
        self.assertTrue(result)
        facade.sidecar.cancel_sidecar_take_profit.assert_awaited_once_with("tp-1")

    async def test_fetch_sidecar_order_status_delegates_to_sidecar(self) -> None:
        trader = make_trader()
        facade = TpSlExecutionManager(trader)
        mock_status = {"ordId": "tp-1", "state": "filled"}
        facade.sidecar.fetch_sidecar_order_status = AsyncMock(return_value=mock_status)
        result = await facade.fetch_sidecar_order_status("tp-1")
        self.assertEqual(result, mock_status)
        facade.sidecar.fetch_sidecar_order_status.assert_awaited_once_with("tp-1")

    # ------------------------------------------------------------------
    # facade keeps cancel methods (callable check — these are inline methods)
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


if __name__ == "__main__":
    unittest.main()
