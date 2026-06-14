from __future__ import annotations

import unittest
from decimal import Decimal

import src.execution.trader as trader_module
from src.exchanges.models import (
    BrokerOrder,
    BrokerOrderSide,
    BrokerOrderStatus,
    BrokerOrderType,
    BrokerPositionSide,
    BrokerQuantityUnit,
    ExchangeName,
)
from src.execution.tp_sl_execution_manager import TpSlExecutionManager
from src.execution.trader import Trader
from src.risk.simple_position_sizer import PositionSize
from src.strategies.boll_cvd_reclaim_strategy import TradeIntent


def make_intent(**overrides) -> TradeIntent:
    kwargs = dict(
        intent_type="UPDATE_TP",
        side="LONG",
        price=3000.0,
        layer_index=1,
        tp_price=3100.0,
        reason="test",
        size=PositionSize(30.0, 1500.0, 0.5, 1, 1.0),
        fast_cvd=0.0,
        previous_fast_cvd=0.0,
        buy_ratio=0.0,
        sell_ratio=0.0,
        boll_upper=3100.0,
        boll_middle=3000.0,
        boll_lower=2900.0,
        ts_ms=1000,
        avg_entry_price=3000.0,
        breakeven_price=3003.0,
        tp_mode="MIDDLE",
    )
    kwargs.update(overrides)
    return TradeIntent(**kwargs)  # type: ignore[arg-type]


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


class TpSlExecutionManagerTest(unittest.IsolatedAsyncioTestCase):
    """Tests for TpSlExecutionManager helper methods and wrapper delegation."""

    # ------------------------------------------------------------------
    # _tp_price_summary
    # ------------------------------------------------------------------

    def test_tp_price_summary_single_spec_returns_price(self) -> None:
        trader = make_trader()
        manager = TpSlExecutionManager(trader)
        specs = [("final", Decimal("10"), 3100.0)]
        result = manager._tp_price_summary(specs)
        self.assertEqual(result, "3100.00")

    def test_tp_price_summary_multi_spec_returns_labeled_prices(self) -> None:
        trader = make_trader()
        manager = TpSlExecutionManager(trader)
        specs = [("partial", Decimal("5"), 3050.0), ("final", Decimal("5"), 3100.0)]
        result = manager._tp_price_summary(specs)
        self.assertEqual(result, "partial:3050.00,final:3100.00")

    # ------------------------------------------------------------------
    # _split_order_ids
    # ------------------------------------------------------------------

    def test_split_order_ids_none_returns_empty_set(self) -> None:
        result = TpSlExecutionManager._split_order_ids(None)
        self.assertEqual(result, set())

    def test_split_order_ids_empty_string_returns_empty_set(self) -> None:
        result = TpSlExecutionManager._split_order_ids("")
        self.assertEqual(result, set())

    def test_split_order_ids_comma_separated_returns_set(self) -> None:
        result = TpSlExecutionManager._split_order_ids("a,b,c")
        self.assertEqual(result, {"a", "b", "c"})

    def test_split_order_ids_single_value_returns_singleton_set(self) -> None:
        result = TpSlExecutionManager._split_order_ids("single-id")
        self.assertEqual(result, {"single-id"})

    def test_split_order_ids_strips_whitespace(self) -> None:
        result = TpSlExecutionManager._split_order_ids(" a , b , c ")
        self.assertEqual(result, {"a", "b", "c"})

    # ------------------------------------------------------------------
    # _protected_order_ids_from_intent
    # ------------------------------------------------------------------

    def test_protected_order_ids_includes_intent_protected_order_ids(self) -> None:
        trader = make_trader()
        manager = TpSlExecutionManager(trader)
        intent = make_intent(protected_order_ids=("sidecar-tp", "custom-id"))
        ids = manager._protected_order_ids_from_intent(intent)
        self.assertIn("sidecar-tp", ids)
        self.assertIn("custom-id", ids)

    def test_protected_order_ids_includes_intent_near_tp_sl_order_id(self) -> None:
        trader = make_trader()
        manager = TpSlExecutionManager(trader)
        intent = make_intent()
        object.__setattr__(intent, "near_tp_protective_sl_order_id", "near-sl-id")
        ids = manager._protected_order_ids_from_intent(intent)
        self.assertIn("near-sl-id", ids)

    def test_protected_order_ids_includes_trader_current_sl_order_ids(self) -> None:
        trader = make_trader(
            near_tp_protective_sl_order_id="current-near-sl",
            middle_runner_protective_sl_order_id="current-mid-sl",
            trend_runner_sl_order_id="current-trend-sl",
        )
        manager = TpSlExecutionManager(trader)
        intent = make_intent()
        ids = manager._protected_order_ids_from_intent(intent)
        self.assertIn("current-near-sl", ids)
        self.assertIn("current-mid-sl", ids)
        self.assertIn("current-trend-sl", ids)

    def test_protected_order_ids_excludes_none_values(self) -> None:
        trader = make_trader()
        manager = TpSlExecutionManager(trader)
        intent = make_intent()
        ids = manager._protected_order_ids_from_intent(intent)
        self.assertNotIn(None, ids)

    # ------------------------------------------------------------------
    # _managed_core_contracts_from_intent
    # ------------------------------------------------------------------

    def test_managed_core_contracts_none_returns_none(self) -> None:
        trader = make_trader()
        manager = TpSlExecutionManager(trader)
        intent = make_intent(managed_core_contracts=None)
        result = manager._managed_core_contracts_from_intent(intent)
        self.assertIsNone(result)

    def test_managed_core_contracts_empty_string_returns_none(self) -> None:
        trader = make_trader()
        manager = TpSlExecutionManager(trader)
        intent = make_intent(managed_core_contracts="")
        result = manager._managed_core_contracts_from_intent(intent)
        self.assertIsNone(result)

    def test_managed_core_contracts_valid_returns_decimal(self) -> None:
        trader = make_trader()
        manager = TpSlExecutionManager(trader)
        intent = make_intent(managed_core_contracts="10")
        result = manager._managed_core_contracts_from_intent(intent)
        self.assertEqual(result, Decimal("10"))

    def test_managed_core_contracts_invalid_string_raises_runtime_error(self) -> None:
        trader = make_trader()
        manager = TpSlExecutionManager(trader)
        intent = make_intent(managed_core_contracts="not_a_number")
        with self.assertRaises(RuntimeError) as ctx:
            manager._managed_core_contracts_from_intent(intent)
        self.assertIn("invalid managed_core_contracts", str(ctx.exception))

    def test_managed_core_contracts_below_min_raises_runtime_error(self) -> None:
        trader = make_trader(min_contracts=Decimal("1"))
        manager = TpSlExecutionManager(trader)
        intent = make_intent(managed_core_contracts="0.001")
        with self.assertRaises(RuntimeError) as ctx:
            manager._managed_core_contracts_from_intent(intent)
        self.assertIn("managed_core_contracts below min_contracts", str(ctx.exception))

    def test_managed_core_contracts_rounded_down(self) -> None:
        trader = make_trader()
        manager = TpSlExecutionManager(trader)
        intent = make_intent(managed_core_contracts="10.005")
        result = manager._managed_core_contracts_from_intent(intent)
        self.assertEqual(result, Decimal("10.00"))

    # ------------------------------------------------------------------
    # cancel_existing_reduce_only_orders
    # ------------------------------------------------------------------

    async def test_cancel_reduce_only_skips_protected_orders(self) -> None:
        trader = make_trader()
        trader._protected_reduce_only_order_ids = {"sidecar-tp"}
        requests = []

        async def fake_fetch_broker_open():  # type: ignore[no-untyped-def]
            return (
                BrokerOrder(
                    exchange=ExchangeName.OKX,
                    symbol=trader.symbol,
                    order_id="core-old",
                    client_order_id=None,
                    side=BrokerOrderSide.SELL,
                    position_side=BrokerPositionSide.LONG,
                    order_type=BrokerOrderType.LIMIT,
                    status=BrokerOrderStatus.OPEN,
                    price=Decimal("3100"),
                    quantity=Decimal("1"),
                    quantity_unit=BrokerQuantityUnit.CONTRACTS,
                    reduce_only=True,
                    raw={"instId": trader.symbol, "reduceOnly": "true", "ordId": "core-old"},
                ),
                BrokerOrder(
                    exchange=ExchangeName.OKX,
                    symbol=trader.symbol,
                    order_id="sidecar-tp",
                    client_order_id=None,
                    side=BrokerOrderSide.SELL,
                    position_side=BrokerPositionSide.LONG,
                    order_type=BrokerOrderType.LIMIT,
                    status=BrokerOrderStatus.OPEN,
                    price=Decimal("3100"),
                    quantity=Decimal("1"),
                    quantity_unit=BrokerQuantityUnit.CONTRACTS,
                    reduce_only=True,
                    raw={"instId": trader.symbol, "reduceOnly": "true", "ordId": "sidecar-tp"},
                ),
            )

        async def fake_request(method, path, body):  # type: ignore[no-untyped-def]
            requests.append((method, path, body))
            return {"code": "0", "data": []}

        trader.fetch_broker_open_orders = fake_fetch_broker_open
        trader.request = fake_request  # type: ignore[method-assign]

        manager = TpSlExecutionManager(trader)
        await manager.cancel_existing_reduce_only_orders()

        cancelled_ids = [r[2].get("ordId") for r in requests if "/cancel-order" in r[1]]
        self.assertIn("core-old", cancelled_ids)
        self.assertNotIn("sidecar-tp", cancelled_ids)

    async def test_cancel_reduce_only_unmanaged_with_managed_set_raises(self) -> None:
        trader = make_trader()
        trader._managed_reduce_only_order_ids = {"known-id"}
        trader._protected_reduce_only_order_ids = set()

        async def fake_fetch_broker_open():  # type: ignore[no-untyped-def]
            return (
                BrokerOrder(
                    exchange=ExchangeName.OKX,
                    symbol=trader.symbol,
                    order_id="unknown-id",
                    client_order_id=None,
                    side=BrokerOrderSide.SELL,
                    position_side=BrokerPositionSide.LONG,
                    order_type=BrokerOrderType.LIMIT,
                    status=BrokerOrderStatus.OPEN,
                    price=Decimal("3100"),
                    quantity=Decimal("1"),
                    quantity_unit=BrokerQuantityUnit.CONTRACTS,
                    reduce_only=True,
                    raw={"instId": trader.symbol, "reduceOnly": "true", "ordId": "unknown-id"},
                ),
            )

        trader.fetch_broker_open_orders = fake_fetch_broker_open

        manager = TpSlExecutionManager(trader)
        with self.assertRaises(RuntimeError) as ctx:
            await manager.cancel_existing_reduce_only_orders()
        self.assertIn("reduce_only_order_identity_unknown", str(ctx.exception))

    # ------------------------------------------------------------------
    # cancel_sidecar_take_profit
    # ------------------------------------------------------------------

    async def test_cancel_sidecar_take_profit_none_returns_true(self) -> None:
        trader = make_trader()
        manager = TpSlExecutionManager(trader)
        result = await manager.cancel_sidecar_take_profit(None)
        self.assertTrue(result)

    async def test_cancel_sidecar_take_profit_already_absent_returns_true(self) -> None:
        trader = make_trader()

        async def fake_request(method, path, body):  # type: ignore[no-untyped-def]
            raise RuntimeError("order does not exist")

        trader.request = fake_request  # type: ignore[method-assign]
        manager = TpSlExecutionManager(trader)
        result = await manager.cancel_sidecar_take_profit("tp-1")
        self.assertTrue(result)

    async def test_cancel_sidecar_take_profit_success(self) -> None:
        trader = make_trader()

        async def fake_request(method, path, body):  # type: ignore[no-untyped-def]
            return {"code": "0", "data": []}

        trader.request = fake_request  # type: ignore[method-assign]
        manager = TpSlExecutionManager(trader)
        result = await manager.cancel_sidecar_take_profit("tp-1")
        self.assertTrue(result)

    # ------------------------------------------------------------------
    # place_near_tp_protective_stop_with_retries — happy path
    # ------------------------------------------------------------------

    async def test_protective_sl_retries_happy_path(self) -> None:
        trader = make_trader()
        requests = []

        async def fake_request(method, path, body):  # type: ignore[no-untyped-def]
            requests.append((method, path, dict(body)))
            return {"code": "0", "data": [{"algoId": "algo-1"}]}

        async def fake_verify(_algo_id, _side, _contracts, _stop_price):  # type: ignore[no-untyped-def]
            return True

        trader.request = fake_request  # type: ignore[method-assign]
        trader.verify_near_tp_protective_stop = fake_verify  # type: ignore[method-assign]

        manager = TpSlExecutionManager(trader)
        ok, algo_id, message = await manager.place_near_tp_protective_stop_with_retries(
            "LONG", Decimal("10"), 3050.0, 3, 0.1
        )

        self.assertTrue(ok)
        self.assertEqual(algo_id, "algo-1")
        self.assertEqual(message, "protective_sl_placed")

    # ------------------------------------------------------------------
    # place_near_tp_protective_stop_with_retries — fallback path
    # ------------------------------------------------------------------

    async def test_protective_sl_retries_fallback_path(self) -> None:
        trader = make_trader()
        requests = []
        verify_count = [0]

        async def fake_request(method, path, body):  # type: ignore[no-untyped-def]
            requests.append((method, path, dict(body)))
            return {"code": "0", "data": [{"algoId": f"algo-{len(requests)}"}]}

        async def fake_verify(_algo_id, _side, _contracts, _stop_price):  # type: ignore[no-untyped-def]
            verify_count[0] += 1
            # Primary retries fail verification; fallback succeeds on first try
            return verify_count[0] > 3  # 3 primary retries fail, fallback succeeds

        async def fake_cancel_near_tp(_order_id):  # type: ignore[no-untyped-def]
            return True

        trader.request = fake_request  # type: ignore[method-assign]
        trader.verify_near_tp_protective_stop = fake_verify  # type: ignore[method-assign]
        trader.cancel_near_tp_protective_stop = fake_cancel_near_tp  # type: ignore[method-assign]

        manager = TpSlExecutionManager(trader)
        ok, algo_id, message = await manager.place_near_tp_protective_stop_with_retries(
            "LONG", Decimal("10"), 3050.0, 3, 0.0
        )

        self.assertTrue(ok)
        self.assertEqual(message, "fallback_conditional_close_placed")

    # ------------------------------------------------------------------
    # market_exit_remaining_position_with_retries
    # ------------------------------------------------------------------

    async def test_market_exit_already_flat_returns_true(self) -> None:
        trader = make_trader()

        async def fake_fetch_snapshot():  # type: ignore[no-untyped-def]
            return trader_module.PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0"))

        async def fake_cancel_reduce_only():  # type: ignore[no-untyped-def]
            return None

        trader.fetch_position_snapshot = fake_fetch_snapshot
        trader.cancel_existing_reduce_only_orders = fake_cancel_reduce_only  # type: ignore[method-assign]

        manager = TpSlExecutionManager(trader)
        ok, message = await manager.market_exit_remaining_position_with_retries("LONG", 3)

        self.assertTrue(ok)
        self.assertEqual(message, "already_flat")

    async def test_market_exit_normal_close_success(self) -> None:
        trader = make_trader(min_contracts=Decimal("0.01"))
        call_count = [0]

        async def fake_fetch_snapshot():  # type: ignore[no-untyped-def]
            call_count[0] += 1
            if call_count[0] == 1:
                return trader_module.PositionSnapshot("LONG", Decimal("10"), 3000.0, 1.0, Decimal("10"))
            # After order, position is flat
            return trader_module.PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0"))

        async def fake_request(method, path, body):  # type: ignore[no-untyped-def]
            return {"code": "0", "data": [{"ordId": "exit-1"}]}

        trader.fetch_position_snapshot = fake_fetch_snapshot
        trader.request = fake_request  # type: ignore[method-assign]
        trader.cancel_existing_reduce_only_orders = (  # type: ignore[method-assign]
            lambda: None
        )

        manager = TpSlExecutionManager(trader)
        ok, message = await manager.market_exit_remaining_position_with_retries("LONG", 3)

        self.assertTrue(ok)
        self.assertIn("market_exit_order_id=exit-1", message)

    async def test_market_exit_not_flat_after_order_retries(self) -> None:
        trader = make_trader(min_contracts=Decimal("0.01"))

        async def fake_fetch_snapshot():  # type: ignore[no-untyped-def]
            # Position never goes flat
            return trader_module.PositionSnapshot("LONG", Decimal("10"), 3000.0, 1.0, Decimal("10"))

        async def fake_request(method, path, body):  # type: ignore[no-untyped-def]
            return {"code": "0", "data": [{"ordId": "exit-1"}]}

        trader.fetch_position_snapshot = fake_fetch_snapshot
        trader.request = fake_request  # type: ignore[method-assign]

        manager = TpSlExecutionManager(trader)
        ok, message = await manager.market_exit_remaining_position_with_retries("LONG", 2)

        self.assertFalse(ok)
        self.assertIn("not_flat_after_order", message)

    # ------------------------------------------------------------------
    # Trader wrapper delegation tests
    # ------------------------------------------------------------------

    async def test_trader_replace_take_profit_delegates_to_manager(self) -> None:
        trader = make_trader()
        delegation_called = [False]
        original_mgr_replace = trader._tp_sl_manager.replace_take_profit

        async def tracking_replace(intent):  # type: ignore[no-untyped-def]
            delegation_called[0] = True
            return trader_module.LiveTradeResult(
                ok=True,
                action="test",
                order_id=None,
                tp_order_id=None,
                contracts="0",
                tp_price="0.00",
                message="delegated",
            )

        trader._tp_sl_manager.replace_take_profit = tracking_replace  # type: ignore[method-assign]
        try:
            intent = make_intent()
            await trader.replace_take_profit(intent)
            self.assertTrue(delegation_called[0])
        finally:
            trader._tp_sl_manager.replace_take_profit = original_mgr_replace  # type: ignore[method-assign]

    async def test_trader_execute_near_tp_reduce_delegates_to_manager(self) -> None:
        trader = make_trader()
        delegation_called = [False]

        async def tracking_execute(intent):  # type: ignore[no-untyped-def]
            delegation_called[0] = True
            return trader_module.LiveTradeResult(
                ok=True, action="test", order_id=None, tp_order_id=None, contracts="0", tp_price="0.00",
                message="delegated"
            )

        original = trader._tp_sl_manager.execute_near_tp_reduce
        trader._tp_sl_manager.execute_near_tp_reduce = tracking_execute  # type: ignore[method-assign]
        try:
            intent = make_intent(intent_type="NEAR_TP_REDUCE")
            await trader.execute_near_tp_reduce(intent)
            self.assertTrue(delegation_called[0])
        finally:
            trader._tp_sl_manager.execute_near_tp_reduce = original  # type: ignore[method-assign]

    async def test_trader_execute_market_exit_runner_delegates_to_manager(self) -> None:
        trader = make_trader()
        delegation_called = [False]

        async def tracking_execute(intent):  # type: ignore[no-untyped-def]
            delegation_called[0] = True
            return trader_module.LiveTradeResult(
                ok=True, action="test", order_id=None, tp_order_id=None, contracts="0", tp_price="0.00",
                message="delegated"
            )

        original = trader._tp_sl_manager.execute_market_exit_runner
        trader._tp_sl_manager.execute_market_exit_runner = tracking_execute  # type: ignore[method-assign]
        try:
            intent = make_intent(intent_type="MARKET_EXIT_RUNNER")
            await trader.execute_market_exit_runner(intent)
            self.assertTrue(delegation_called[0])
        finally:
            trader._tp_sl_manager.execute_market_exit_runner = original  # type: ignore[method-assign]

    async def test_trader_cancel_existing_reduce_only_orders_delegates_to_manager(self) -> None:
        trader = make_trader()
        delegation_called = [False]

        async def tracking_cancel():  # type: ignore[no-untyped-def]
            delegation_called[0] = True

        original = trader._tp_sl_manager.cancel_existing_reduce_only_orders
        trader._tp_sl_manager.cancel_existing_reduce_only_orders = tracking_cancel  # type: ignore[method-assign]
        try:
            await trader.cancel_existing_reduce_only_orders()
            self.assertTrue(delegation_called[0])
        finally:
            trader._tp_sl_manager.cancel_existing_reduce_only_orders = original  # type: ignore[method-assign]

    async def test_trader_place_near_tp_protective_stop_with_retries_delegates_to_manager(self) -> None:
        trader = make_trader()
        delegation_called = [False]

        async def tracking_place(side, contracts, stop_price, retry_count,
                                 retry_interval_seconds):  # type: ignore[no-untyped-def]
            delegation_called[0] = True
            return True, "algo-1", "placed"

        original = trader._tp_sl_manager.place_near_tp_protective_stop_with_retries
        trader._tp_sl_manager.place_near_tp_protective_stop_with_retries = tracking_place  # type: ignore[method-assign]
        try:
            await trader.place_near_tp_protective_stop_with_retries("LONG", Decimal("10"), 3050.0, 3, 0.1)
            self.assertTrue(delegation_called[0])
        finally:
            trader._tp_sl_manager.place_near_tp_protective_stop_with_retries = original  # type: ignore[method-assign]

    # ------------------------------------------------------------------
    # sidecar fixed TP — sanitize client_order_id
    # ------------------------------------------------------------------

    async def test_sidecar_fixed_tp_sanitizes_client_order_id(self) -> None:
        trader = make_trader()
        requests = []

        async def fake_request(method, path, body):  # type: ignore[no-untyped-def]
            requests.append((method, path, dict(body)))
            return {"data": [{"ordId": "tp-1"}]}

        trader.request = fake_request  # type: ignore[method-assign]

        manager = TpSlExecutionManager(trader)
        order_id = await manager.place_sidecar_fixed_take_profit(
            side="LONG",
            contracts="0.69",
            tp_price=3012.0,
            client_order_id="SC-97644895de-L1-47229",
        )

        self.assertEqual(order_id, "tp-1")
        self.assertEqual(requests[0][2]["clOrdId"], "SC97644895deL147229")
        self.assertLessEqual(len(requests[0][2]["clOrdId"]), 32)

    # ------------------------------------------------------------------
    # Regression: LiveTradeResult / PositionSnapshot runtime import
    # ------------------------------------------------------------------

    async def test_execute_near_tp_reduce_no_position_returns_LiveTradeResult(self) -> None:
        """execute_near_tp_reduce with no position returns LiveTradeResult (runtime import check)."""
        trader = make_trader()

        async def fake_fetch_snapshot():  # type: ignore[no-untyped-def]
            return trader_module.PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0"))

        trader.fetch_position_snapshot = fake_fetch_snapshot

        manager = TpSlExecutionManager(trader)
        intent = make_intent(intent_type="NEAR_TP_REDUCE")
        result = await manager.execute_near_tp_reduce(intent)

        self.assertIsInstance(result, trader_module.LiveTradeResult)
        self.assertFalse(result.ok)
        self.assertEqual(result.message, "no position")

    async def test_replace_take_profit_no_position_returns_LiveTradeResult(self) -> None:
        """replace_take_profit with no net position returns LiveTradeResult (runtime import check)."""
        trader = make_trader()

        async def fake_fetch_snapshot():  # type: ignore[no-untyped-def]
            return trader_module.PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0"))

        trader.fetch_position_snapshot = fake_fetch_snapshot

        manager = TpSlExecutionManager(trader)
        intent = make_intent()
        result = await manager.replace_take_profit(intent)

        self.assertIsInstance(result, trader_module.LiveTradeResult)
        self.assertFalse(result.ok)
        self.assertEqual(result.message, "no position to protect")


if __name__ == "__main__":
    unittest.main()
