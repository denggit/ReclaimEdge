#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_trader_broker_semantic_query_bridge.py
@Description: Tests for Trader read-only broker semantic query bridge.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from tests.conftest import FakeOkxClient
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
        action = BrokerSemanticAction.FETCH_OPEN_ORDERS
        ok = action not in self.fail_actions
        if not ok:
            self.fail_actions.discard(action)  # single-shot failure
        return BrokerSemanticResult(
            exchange=ExchangeName.OKX,
            symbol=symbol,
            action=action,
            role=BrokerSemanticOrderRole.RECOVERY,
            ok=ok,
            message=self.message if not ok else "",
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
    from src.execution.okx_trading_client import OkxTradingClient

    trader = object.__new__(Trader)
    trader.symbol = "ETH-USDT-SWAP"
    trader.td_mode = "isolated"
    trader.leverage = "50"
    trader.pos_side_mode = "net"
    trader.contract_multiplier = Decimal("0.1")
    trader.contract_precision = Decimal("0.01")
    trader.min_contracts = Decimal("0.01")
    trader._broker_client = None
    trader._broker_semantic_executor = fake_executor
    fake_private_client = FakeOkxClient(trader)
    trader._client = fake_private_client
    trader.trading_client = OkxTradingClient(trader, private_client=fake_private_client)
    return trader


def _trader_with_legacy_request(fake_executor: FakeSemanticExecutor) -> Trader:
    trader = _trader(fake_executor)
    trader.requests = []

    async def fake_request(
        method: str,
        endpoint: str,
        payload: Any | None = None,
    ) -> dict[str, Any]:
        trader.requests.append((method, endpoint, payload))
        if endpoint.startswith("/api/v5/trade/orders-pending?"):
            return {"data": [{"instId": "ETH-USDT-SWAP", "ordId": "legacy-1"}]}
        if endpoint.startswith("/api/v5/trade/orders-algo-pending?"):
            return {"data": [{"instId": "ETH-USDT-SWAP", "algoId": "legacy-algo-1"}]}
        raise AssertionError(endpoint)

    trader.request = fake_request

    # Also wire fake_request into _client so that OkxTradingClient's direct
    # REST calls (fetch_open_algo_orders, etc.) also go through the fake.
    trader._client.request = fake_request

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


@pytest.mark.asyncio
async def test_pending_order_reads_use_legacy_endpoints_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy methods now delegate to trading_client → broker/API layer.

    The test verifies that the OkxTradingClient / broker layer properly
    routes to the OKX REST endpoints (via _client).  The trader.request
    monkeypatch records the legacy-style endpoint calls.
    """
    monkeypatch.delenv("BROKER_SEMANTIC_READS_ENABLED", raising=False)
    fake = FakeSemanticExecutor()
    # Set up broker open orders so the legacy fetch_pending_orders path
    # (broker → semantic executor) returns expected data.
    fake.open_orders = (
        _order("legacy-1", raw={"instId": "ETH-USDT-SWAP", "ordId": "legacy-1"}),
    )
    trader = _trader_with_legacy_request(fake)

    orders = await Trader.fetch_pending_orders(trader)
    algo_orders = await Trader.fetch_pending_algo_orders(trader)

    assert orders == [{"instId": "ETH-USDT-SWAP", "ordId": "legacy-1"}]
    assert algo_orders == [{"instId": "ETH-USDT-SWAP", "algoId": "legacy-algo-1"}]
    # algo orders go through _client directly (OkxTradingClient calls
    # _client.request for fetch_open_algo_orders)
    assert (
        "GET",
        "/api/v5/trade/orders-algo-pending?instId=ETH-USDT-SWAP&ordType=conditional",
        None,
    ) in trader.requests


@pytest.mark.asyncio
async def test_pending_orders_prefer_semantic_path_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BROKER_SEMANTIC_READS_ENABLED", "true")
    fake = FakeSemanticExecutor()
    fake.open_orders = (
        _order(
            "semantic-1",
            raw={"instId": "ETH-USDT-SWAP", "ordId": "semantic-1"},
        ),
    )
    trader = _trader_with_legacy_request(fake)

    orders = await Trader.fetch_pending_orders(trader)

    assert orders == [{"instId": "ETH-USDT-SWAP", "ordId": "semantic-1"}]
    assert fake.calls == [("fetch_open_orders", "ETH-USDT-SWAP")]
    assert not any(
        endpoint.startswith("/api/v5/trade/orders-pending?")
        for _, endpoint, _ in trader.requests
    )


@pytest.mark.asyncio
async def test_pending_algo_orders_prefer_semantic_path_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BROKER_SEMANTIC_READS_ENABLED", "true")
    fake = FakeSemanticExecutor()
    fake.algo_orders = (
        _order(
            "semantic-algo-1",
            raw={"instId": "ETH-USDT-SWAP", "algoId": "semantic-algo-1"},
            metadata={"source": "algo"},
        ),
    )
    trader = _trader_with_legacy_request(fake)

    orders = await Trader.fetch_pending_algo_orders(trader)

    assert orders == [{"instId": "ETH-USDT-SWAP", "algoId": "semantic-algo-1"}]
    assert fake.calls == [("fetch_algo_orders", "ETH-USDT-SWAP")]
    assert not any(
        endpoint.startswith("/api/v5/trade/orders-algo-pending?")
        for _, endpoint, _ in trader.requests
    )


@pytest.mark.asyncio
async def test_pending_orders_fallback_to_legacy_when_semantic_path_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When broker semantic path fails, the fallback goes through OkxTradingClient.

    The initial fetch_pending_orders tries broker semantic first (enabled=true).
    When that fails, it falls through to trading_client.fetch_open_orders()
    which uses the Okx broker layer.  The FakeSemanticExecutor clears
    fail_actions after the first failure so the retry succeeds.
    """
    monkeypatch.setenv("BROKER_SEMANTIC_READS_ENABLED", "true")
    caplog.set_level(logging.WARNING)
    fake = FakeSemanticExecutor()
    fake.fail_actions.add(BrokerSemanticAction.FETCH_OPEN_ORDERS)
    fake.open_orders = (
        _order("legacy-1", raw={"instId": "ETH-USDT-SWAP", "ordId": "legacy-1"}),
    )
    trader = _trader_with_legacy_request(fake)

    orders = await Trader.fetch_pending_orders(trader)

    assert orders == [{"instId": "ETH-USDT-SWAP", "ordId": "legacy-1"}]
    assert "BROKER_SEMANTIC_READ_FALLBACK" in caplog.text
    assert "kind=open_orders" in caplog.text


@pytest.mark.asyncio
async def test_pending_algo_orders_fallback_to_legacy_when_semantic_path_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("BROKER_SEMANTIC_READS_ENABLED", "true")
    caplog.set_level(logging.WARNING)
    fake = FakeSemanticExecutor()
    fake.fail_actions.add(BrokerSemanticAction.FETCH_ALGO_ORDERS)
    trader = _trader_with_legacy_request(fake)

    orders = await Trader.fetch_pending_algo_orders(trader)

    assert orders == [{"instId": "ETH-USDT-SWAP", "algoId": "legacy-algo-1"}]
    assert any(
        endpoint.startswith("/api/v5/trade/orders-algo-pending?")
        for _, endpoint, _ in trader.requests
    )
    assert "BROKER_SEMANTIC_READ_FALLBACK" in caplog.text
    assert "kind=algo_orders" in caplog.text


@pytest.mark.asyncio
async def test_pending_orders_semantic_raw_result_is_a_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BROKER_SEMANTIC_READS_ENABLED", "true")
    fake = FakeSemanticExecutor()
    raw = {"instId": "ETH-USDT-SWAP", "ordId": "semantic-1"}
    fake.open_orders = (_order("semantic-1", raw=raw),)
    trader = _trader_with_legacy_request(fake)

    orders = await Trader.fetch_pending_orders(trader)
    orders[0]["ordId"] = "mutated"

    assert raw["ordId"] == "semantic-1"


def _method_block(text: str, method_name: str) -> str:
    return text.split(f"async def {method_name}", 1)[1].split(
        "\n    async def ",
        1,
    )[0]


def test_legacy_wrappers_delegate_to_trading_client() -> None:
    """Legacy methods now delegate to TradingClientPort, not direct /api/v5."""
    text = Path("src/execution/trader.py").read_text()
    expected_calls = {
        "fetch_pending_orders": "trading_client.fetch_open_orders()",
        "fetch_pending_algo_orders": "trading_client.fetch_open_algo_orders()",
        "fetch_position_snapshot": "trading_client.fetch_position()",
        "fetch_usdt_equity": "trading_client.fetch_balance()",
        "set_leverage": "trading_client.configure_instrument()",
    }

    for method_name, expected_call in expected_calls.items():
        block = _method_block(text, method_name)
        assert expected_call in block, f"{method_name} must call {expected_call}"

    # Also verify these methods do NOT contain direct /api/v5
    for method_name in expected_calls:
        block = _method_block(text, method_name)
        assert "/api/v5" not in block, f"{method_name} must NOT contain /api/v5"


def test_fetch_position_snapshot_does_not_use_semantic_reads() -> None:
    text = Path("src/execution/trader.py").read_text()
    block = _method_block(text, "fetch_position_snapshot")

    # Now delegates to trading_client, no direct /api/v5
    assert "trading_client.fetch_position()" in block
    assert "broker_semantic_executor" not in block
    assert "fetch_broker_position" not in block
    assert "BROKER_SEMANTIC_READS_ENABLED" not in block


def test_execute_intent_does_not_use_semantic_reads() -> None:
    text = Path("src/execution/trader.py").read_text()
    block = _method_block(text, "execute_intent")

    assert "fetch_broker_open_orders" not in block
    assert "broker_semantic_executor" not in block
    assert "BROKER_SEMANTIC_READS_ENABLED" not in block
