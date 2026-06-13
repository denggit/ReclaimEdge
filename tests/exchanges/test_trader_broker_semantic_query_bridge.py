#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_trader_broker_semantic_query_bridge.py
@Description: Tests for Trader read-only broker semantic query bridge.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from src.execution.trader import Trader
from src.exchanges.models import (
    BrokerOrder,
    BrokerOrderSide,
    BrokerOrderStatus,
    BrokerOrderType,
    BrokerPosition,
    BrokerPositionSide,
    BrokerQuantityUnit,
    ExchangeName,
)
from src.exchanges.semantic_models import (
    BrokerSemanticAction,
    BrokerSemanticOrderRole,
    BrokerSemanticResult,
)


class FakeSemanticExecutor:
    def __init__(self) -> None:
        self.calls = []
        self.open_orders = ()
        self.algo_orders = ()
        self.recovered_orders = ()
        self.position = None
        self.fail_actions = set()
        self.message = "boom"

    async def fetch_open_orders(self, *, symbol: str):
        self.calls.append(("fetch_open_orders", symbol))
        return BrokerSemanticResult(
            exchange=ExchangeName.OKX,
            symbol=symbol,
            action=BrokerSemanticAction.FETCH_OPEN_ORDERS,
            role=BrokerSemanticOrderRole.RECOVERY,
            ok=BrokerSemanticAction.FETCH_OPEN_ORDERS not in self.fail_actions,
            message=self.message if BrokerSemanticAction.FETCH_OPEN_ORDERS in self.fail_actions else "",
            orders=tuple(self.open_orders),
        )

    async def fetch_algo_orders(self, *, symbol: str):
        self.calls.append(("fetch_algo_orders", symbol))
        return BrokerSemanticResult(
            exchange=ExchangeName.OKX,
            symbol=symbol,
            action=BrokerSemanticAction.FETCH_ALGO_ORDERS,
            role=BrokerSemanticOrderRole.RECOVERY,
            ok=BrokerSemanticAction.FETCH_ALGO_ORDERS not in self.fail_actions,
            message=self.message if BrokerSemanticAction.FETCH_ALGO_ORDERS in self.fail_actions else "",
            orders=tuple(self.algo_orders),
        )

    async def recover_open_orders(self, *, symbol: str):
        self.calls.append(("recover_open_orders", symbol))
        return BrokerSemanticResult(
            exchange=ExchangeName.OKX,
            symbol=symbol,
            action=BrokerSemanticAction.RECOVER_OPEN_ORDERS,
            role=BrokerSemanticOrderRole.RECOVERY,
            ok=BrokerSemanticAction.RECOVER_OPEN_ORDERS not in self.fail_actions,
            message=self.message if BrokerSemanticAction.RECOVER_OPEN_ORDERS in self.fail_actions else "",
            orders=tuple(self.recovered_orders),
        )

    async def fetch_position(self, *, symbol: str):
        self.calls.append(("fetch_position", symbol))
        return BrokerSemanticResult(
            exchange=ExchangeName.OKX,
            symbol=symbol,
            action=BrokerSemanticAction.FETCH_POSITION,
            role=BrokerSemanticOrderRole.RECOVERY,
            ok=BrokerSemanticAction.FETCH_POSITION not in self.fail_actions,
            message=self.message if BrokerSemanticAction.FETCH_POSITION in self.fail_actions else "",
            position=self.position,
        )


def _trader(fake_executor: FakeSemanticExecutor) -> Trader:
    trader = object.__new__(Trader)
    trader.symbol = "ETH-USDT-SWAP"
    trader._broker_client = None
    trader._broker_semantic_executor = fake_executor
    return trader


def _order(
    order_id: str,
    *,
    raw: dict[str, str] | None = None,
    metadata: dict[str, str] | None = None,
) -> BrokerOrder:
    return BrokerOrder(
        exchange=ExchangeName.OKX,
        symbol="ETH-USDT-SWAP",
        order_id=order_id,
        client_order_id=None,
        side=BrokerOrderSide.SELL,
        position_side=BrokerPositionSide.LONG,
        order_type=BrokerOrderType.LIMIT,
        status=BrokerOrderStatus.OPEN,
        price=Decimal("3500"),
        quantity=Decimal("10"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
        reduce_only=True,
        raw=raw or {"ordId": order_id},
        metadata=metadata or {"source": "ordinary"},
    )


@pytest.mark.asyncio
async def test_fetch_broker_open_orders_returns_broker_order_tuple() -> None:
    fake = FakeSemanticExecutor()
    order = _order("ord-1", raw={"ordId": "ord-1"})
    fake.open_orders = (order,)

    orders = await _trader(fake).fetch_broker_open_orders()

    assert orders == (order,)
    assert fake.calls == [("fetch_open_orders", "ETH-USDT-SWAP")]


@pytest.mark.asyncio
async def test_fetch_broker_algo_orders_returns_broker_order_tuple() -> None:
    fake = FakeSemanticExecutor()
    algo_order = _order(
        "algo-1",
        raw={"algoId": "algo-1"},
        metadata={"source": "algo"},
    )
    fake.algo_orders = (algo_order,)

    orders = await _trader(fake).fetch_broker_algo_orders()

    assert orders == (algo_order,)
    assert fake.calls == [("fetch_algo_orders", "ETH-USDT-SWAP")]


@pytest.mark.asyncio
async def test_recover_broker_open_orders_returns_ordinary_and_algo_orders() -> None:
    fake = FakeSemanticExecutor()
    ordinary_order = _order("ord-1", raw={"ordId": "ord-1"})
    algo_order = _order(
        "algo-1",
        raw={"algoId": "algo-1"},
        metadata={"source": "algo"},
    )
    fake.recovered_orders = (ordinary_order, algo_order)

    orders = await _trader(fake).recover_broker_open_orders()

    assert orders == (ordinary_order, algo_order)
    assert fake.calls == [("recover_open_orders", "ETH-USDT-SWAP")]


@pytest.mark.asyncio
async def test_fetch_broker_position_returns_position() -> None:
    fake = FakeSemanticExecutor()
    position = BrokerPosition(
        exchange=ExchangeName.OKX,
        symbol="ETH-USDT-SWAP",
        position_side=BrokerPositionSide.LONG,
        quantity=Decimal("10"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
        average_entry_price=Decimal("3300"),
    )
    fake.position = position

    result = await _trader(fake).fetch_broker_position()

    assert result == position
    assert fake.calls == [("fetch_position", "ETH-USDT-SWAP")]


@pytest.mark.asyncio
async def test_fetch_broker_open_order_raws_returns_dict_copies() -> None:
    fake = FakeSemanticExecutor()
    order = _order("ord-1", raw={"ordId": "ord-1"})
    fake.open_orders = (order,)

    raws = await _trader(fake).fetch_broker_open_order_raws()

    assert raws == [{"ordId": "ord-1"}]
    assert raws[0] is not order.raw


@pytest.mark.asyncio
async def test_fetch_broker_algo_order_raws_returns_dict_copies() -> None:
    fake = FakeSemanticExecutor()
    order = _order(
        "algo-1",
        raw={"algoId": "algo-1"},
        metadata={"source": "algo"},
    )
    fake.algo_orders = (order,)

    raws = await _trader(fake).fetch_broker_algo_order_raws()

    assert raws == [{"algoId": "algo-1"}]
    assert raws[0] is not order.raw


@pytest.mark.asyncio
async def test_recover_broker_open_order_raws_returns_dict_copies() -> None:
    fake = FakeSemanticExecutor()
    ordinary_order = _order("ord-1", raw={"ordId": "ord-1"})
    algo_order = _order(
        "algo-1",
        raw={"algoId": "algo-1"},
        metadata={"source": "algo"},
    )
    fake.recovered_orders = (ordinary_order, algo_order)

    raws = await _trader(fake).recover_broker_open_order_raws()

    assert raws == [{"ordId": "ord-1"}, {"algoId": "algo-1"}]
    assert raws[0] is not ordinary_order.raw
    assert raws[1] is not algo_order.raw


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "method_name"),
    [
        (BrokerSemanticAction.FETCH_OPEN_ORDERS, "fetch_broker_open_orders"),
        (BrokerSemanticAction.FETCH_ALGO_ORDERS, "fetch_broker_algo_orders"),
        (BrokerSemanticAction.RECOVER_OPEN_ORDERS, "recover_broker_open_orders"),
        (BrokerSemanticAction.FETCH_POSITION, "fetch_broker_position"),
    ],
)
async def test_broker_query_bridge_raises_on_failed_semantic_result(
    action: BrokerSemanticAction,
    method_name: str,
) -> None:
    fake = FakeSemanticExecutor()
    fake.fail_actions.add(action)
    trader = _trader(fake)

    with pytest.raises(RuntimeError, match="boom"):
        await getattr(trader, method_name)()


def test_legacy_raw_query_paths_are_not_replaced_by_broker_bridge() -> None:
    text = Path("src/execution/trader.py").read_text()
    forbidden_bridge_symbols = [
        "broker_semantic_executor",
        "fetch_broker_open_orders",
        "fetch_broker_algo_orders",
        "recover_broker_open_orders",
    ]
    expected_endpoints = {
        "fetch_pending_orders": "/api/v5/trade/orders-pending",
        "fetch_pending_algo_orders": "/api/v5/trade/orders-algo-pending",
        "fetch_position_snapshot": "/api/v5/account/positions",
    }

    for method_name, endpoint in expected_endpoints.items():
        block = text.split(f"async def {method_name}", 1)[1].split(
            "\n    async def ",
            1,
        )[0]
        assert endpoint in block
        for symbol in forbidden_bridge_symbols:
            assert symbol not in block
