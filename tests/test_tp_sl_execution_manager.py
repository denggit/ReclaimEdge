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
from src.execution.okx_trading_client import OkxTradingClient


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
    from tests.conftest import FakeOkxClient

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
    t.middle_runner_protective_sl_order_id = None
    t.three_stage_post_tp1_protective_sl_order_id = None
    t.trend_runner_sl_order_id = None
    t.position_contracts = Decimal("0")
    t.account_equity_usdt = 0.0
    t._protected_reduce_only_order_ids = set()
    t._managed_reduce_only_order_ids = set()
    t._allow_cancel_unmanaged_reduce_only = True
    t._client = FakeOkxClient(t)
    t.trading_client = OkxTradingClient(t, private_client=t._client)  # type: ignore[assignment]
    t._tp_sl_manager = TpSlExecutionManager(t, trading_client=t.trading_client)  # type: ignore[arg-type]
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
        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]
        specs = [("final", Decimal("10"), 3100.0)]
        result = manager._tp_price_summary(specs)
        self.assertEqual(result, "3100.00")

    def test_tp_price_summary_multi_spec_returns_labeled_prices(self) -> None:
        trader = make_trader()
        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]
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
        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]
        intent = make_intent(protected_order_ids=("sidecar-tp", "custom-id"))
        ids = manager._protected_order_ids_from_intent(intent)
        self.assertIn("sidecar-tp", ids)
        self.assertIn("custom-id", ids)

    def test_protected_order_ids_includes_trader_current_sl_order_ids(self) -> None:
        trader = make_trader(
            middle_runner_protective_sl_order_id="current-mid-sl",
            trend_runner_sl_order_id="current-trend-sl",
        )
        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]
        intent = make_intent()
        ids = manager._protected_order_ids_from_intent(intent)
        self.assertIn("current-mid-sl", ids)
        self.assertIn("current-trend-sl", ids)

    def test_protected_order_ids_excludes_none_values(self) -> None:
        trader = make_trader()
        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]
        intent = make_intent()
        ids = manager._protected_order_ids_from_intent(intent)
        self.assertNotIn(None, ids)

    # ------------------------------------------------------------------
    # _managed_core_contracts_from_intent
    # ------------------------------------------------------------------

    def test_managed_core_contracts_none_returns_none(self) -> None:
        trader = make_trader()
        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]
        intent = make_intent(managed_core_contracts=None)
        result = manager._managed_core_contracts_from_intent(intent)
        self.assertIsNone(result)

    def test_managed_core_contracts_empty_string_returns_none(self) -> None:
        trader = make_trader()
        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]
        intent = make_intent(managed_core_contracts="")
        result = manager._managed_core_contracts_from_intent(intent)
        self.assertIsNone(result)

    def test_managed_core_contracts_valid_returns_decimal(self) -> None:
        trader = make_trader()
        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]
        intent = make_intent(managed_core_contracts="10")
        result = manager._managed_core_contracts_from_intent(intent)
        self.assertEqual(result, Decimal("10"))

    def test_managed_core_contracts_invalid_string_raises_runtime_error(self) -> None:
        trader = make_trader()
        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]
        intent = make_intent(managed_core_contracts="not_a_number")
        with self.assertRaises(RuntimeError) as ctx:
            manager._managed_core_contracts_from_intent(intent)
        self.assertIn("invalid managed_core_contracts", str(ctx.exception))

    def test_managed_core_contracts_below_min_raises_runtime_error(self) -> None:
        trader = make_trader(min_contracts=Decimal("1"))
        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]
        intent = make_intent(managed_core_contracts="0.001")
        with self.assertRaises(RuntimeError) as ctx:
            manager._managed_core_contracts_from_intent(intent)
        self.assertIn("managed_core_contracts below min_contracts", str(ctx.exception))

    def test_managed_core_contracts_rounded_down(self) -> None:
        trader = make_trader()
        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]
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

        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]
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

        manager = TpSlExecutionManager(trader, trading_client=trader.trading_client)
        trader._tp_sl_manager = manager  # type: ignore[assignment]
        with self.assertRaises(RuntimeError) as ctx:
            await manager.cancel_existing_reduce_only_orders()
        self.assertIn("reduce_only_order_identity_unknown", str(ctx.exception))

    # ------------------------------------------------------------------
    # cancel_sidecar_take_profit
    # ------------------------------------------------------------------

