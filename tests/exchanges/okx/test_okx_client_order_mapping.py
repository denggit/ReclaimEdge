#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_okx_client_order_mapping.py
@Description: Tests for OkxBrokerClient order body and DTO mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pytest

from src.exchanges.errors import ExchangeError, ExchangeErrorKind
from src.exchanges.models import (
    BrokerOrderRequest,
    BrokerOrderSide,
    BrokerOrderType,
    BrokerPositionSide,
    BrokerQuantityUnit,
    ExchangeName,
)
from src.exchanges.okx.client import OkxBrokerClient


@dataclass(frozen=True)
class LegacyPositionSnapshot:
    side: str | None
    contracts: Decimal
    avg_entry_price: float


class FakeTrader:
    symbol = "ETH-USDT-SWAP"
    td_mode = "isolated"
    pos_side_mode = "long_short"

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, Any | None]] = []
        self.pending_orders: list[dict[str, Any]] = []
        self.pending_algo_orders: list[dict[str, Any]] = []
        self.position_snapshot: LegacyPositionSnapshot | None = None

    async def request(
        self,
        method: str,
        endpoint: str,
        payload: Any | None = None,
    ) -> dict[str, Any]:
        self.requests.append((method, endpoint, payload))
        if endpoint == "/api/v5/trade/order":
            return {"data": [{"ordId": "order-1"}]}
        if endpoint == "/api/v5/trade/order-algo":
            return {"data": [{"algoId": "algo-1"}]}
        if endpoint == "/api/v5/trade/cancel-order":
            return {"data": [{"ordId": payload["ordId"], "sCode": "0"}]}
        if endpoint == "/api/v5/trade/cancel-algos":
            return {"data": [{"algoId": payload[0]["algoId"], "sCode": "0"}]}
        raise AssertionError(f"unexpected request {method} {endpoint} {payload}")

    async def fetch_pending_orders(self) -> list[dict[str, Any]]:
        return list(self.pending_orders)

    async def fetch_pending_algo_orders(self) -> list[dict[str, Any]]:
        return list(self.pending_algo_orders)

    async def fetch_position_snapshot(self) -> LegacyPositionSnapshot | None:
        return self.position_snapshot

    @staticmethod
    def decimal_to_str(value: Decimal) -> str:
        return format(value.normalize(), "f")

    @staticmethod
    def price_to_str(price: float) -> str:
        return f"{price:.2f}"

    @staticmethod
    def extract_order_id(res: dict[str, Any]) -> str:
        return str(res["data"][0]["ordId"])

    @staticmethod
    def extract_algo_id(res: dict[str, Any]) -> str:
        item = res["data"][0]
        return str(item.get("algoId") or item.get("ordId"))


@pytest.fixture
def fake_trader() -> FakeTrader:
    return FakeTrader()


@pytest.fixture
def client(fake_trader: FakeTrader) -> OkxBrokerClient:
    return OkxBrokerClient(trader=fake_trader)


@pytest.mark.asyncio
async def test_market_entry_order_uses_market_body(
    fake_trader: FakeTrader,
    client: OkxBrokerClient,
) -> None:
    request = BrokerOrderRequest(
        exchange=ExchangeName.OKX,
        symbol="ETH-USDT-SWAP",
        side=BrokerOrderSide.BUY,
        position_side=BrokerPositionSide.LONG,
        order_type=BrokerOrderType.MARKET,
        quantity=Decimal("12"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
        reduce_only=False,
    )

    result = await client.place_order(request)

    assert result.ok is True
    assert result.order_id == "order-1"
    method, endpoint, payload = fake_trader.requests[-1]
    assert method == "POST"
    assert endpoint == "/api/v5/trade/order"
    assert payload["instId"] == "ETH-USDT-SWAP"
    assert payload["tdMode"] == "isolated"
    assert payload["side"] == "buy"
    assert payload["ordType"] == "market"
    assert payload["sz"] == "12"
    assert payload["posSide"] == "long"
    assert "reduceOnly" not in payload


@pytest.mark.asyncio
async def test_limit_reduce_only_tp_order_uses_tp_body(
    fake_trader: FakeTrader,
    client: OkxBrokerClient,
) -> None:
    request = BrokerOrderRequest(
        exchange=ExchangeName.OKX,
        symbol="ETH-USDT-SWAP",
        side=BrokerOrderSide.SELL,
        position_side=BrokerPositionSide.LONG,
        order_type=BrokerOrderType.LIMIT,
        quantity=Decimal("10"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
        price=Decimal("3500"),
        reduce_only=True,
        client_order_id="tp-client-1",
    )

    result = await client.place_order(request)

    assert result.order_id == "order-1"
    payload = fake_trader.requests[-1][2]
    assert payload["side"] == "sell"
    assert payload["ordType"] == "limit"
    assert payload["px"] in {"3500.00", "3500"}
    assert payload["sz"] == "10"
    assert payload["reduceOnly"] == "true"
    assert payload["clOrdId"] == "tp-client-1"
    assert payload["posSide"] == "long"


@pytest.mark.asyncio
async def test_market_reduce_only_order_uses_close_body(
    fake_trader: FakeTrader,
    client: OkxBrokerClient,
) -> None:
    request = BrokerOrderRequest(
        exchange=ExchangeName.OKX,
        symbol="ETH-USDT-SWAP",
        side=BrokerOrderSide.SELL,
        position_side=BrokerPositionSide.LONG,
        order_type=BrokerOrderType.MARKET,
        quantity=Decimal("10"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
        reduce_only=True,
    )

    await client.place_order(request)

    payload = fake_trader.requests[-1][2]
    assert payload["side"] == "sell"
    assert payload["ordType"] == "market"
    assert payload["reduceOnly"] == "true"


@pytest.mark.asyncio
async def test_stop_market_protective_sl_order_uses_algo_body(
    fake_trader: FakeTrader,
    client: OkxBrokerClient,
) -> None:
    request = BrokerOrderRequest(
        exchange=ExchangeName.OKX,
        symbol="ETH-USDT-SWAP",
        side=BrokerOrderSide.SELL,
        position_side=BrokerPositionSide.LONG,
        order_type=BrokerOrderType.STOP_MARKET,
        quantity=Decimal("10"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
        trigger_price=Decimal("3400"),
        reduce_only=True,
    )

    result = await client.place_order(request)

    assert result.order_id == "algo-1"
    method, endpoint, payload = fake_trader.requests[-1]
    assert method == "POST"
    assert endpoint == "/api/v5/trade/order-algo"
    assert payload["ordType"] == "conditional"
    assert payload["side"] == "sell"
    assert payload["slTriggerPx"] in {"3400.00", "3400"}
    assert payload["slOrdPx"] == "-1"
    assert payload["reduceOnly"] == "true"
    assert payload["posSide"] == "long"


@pytest.mark.asyncio
async def test_cancel_order_uses_cancel_order_body(
    fake_trader: FakeTrader,
    client: OkxBrokerClient,
) -> None:
    result = await client.cancel_order("ETH-USDT-SWAP", "order-1")

    assert fake_trader.requests[-1] == (
        "POST",
        "/api/v5/trade/cancel-order",
        {"instId": "ETH-USDT-SWAP", "ordId": "order-1"},
    )
    assert result.ok is True
    assert result.order_id == "order-1"


@pytest.mark.asyncio
async def test_cancel_algo_order_uses_cancel_algo_body(
    fake_trader: FakeTrader,
    client: OkxBrokerClient,
) -> None:
    result = await client.cancel_algo_order("ETH-USDT-SWAP", "algo-1")

    assert fake_trader.requests[-1] == (
        "POST",
        "/api/v5/trade/cancel-algos",
        [{"instId": "ETH-USDT-SWAP", "algoId": "algo-1"}],
    )
    assert result.ok is True
    assert result.order_id == "algo-1"


@pytest.mark.asyncio
async def test_fetch_open_orders_filters_symbol_and_maps_raw(
    fake_trader: FakeTrader,
    client: OkxBrokerClient,
) -> None:
    fake_trader.pending_orders = [
        {
            "instId": "ETH-USDT-SWAP",
            "ordId": "123",
            "side": "sell",
            "posSide": "long",
            "ordType": "limit",
            "state": "live",
            "px": "3500",
            "sz": "10",
            "reduceOnly": "true",
        },
        {
            "instId": "BTC-USDT-SWAP",
            "ordId": "btc-1",
            "side": "sell",
            "posSide": "long",
            "ordType": "limit",
            "state": "live",
            "px": "70000",
            "sz": "1",
            "reduceOnly": "true",
        },
    ]

    orders = await client.fetch_open_orders("ETH-USDT-SWAP")

    assert isinstance(orders, tuple)
    assert len(orders) == 1
    assert orders[0].order_id == "123"
    assert orders[0].raw["ordId"] == "123"


@pytest.mark.asyncio
async def test_fetch_algo_orders_filters_symbol_and_maps_raw(
    fake_trader: FakeTrader,
    client: OkxBrokerClient,
) -> None:
    raw = {
        "instId": "ETH-USDT-SWAP",
        "algoId": "algo-123",
        "side": "sell",
        "posSide": "long",
        "ordType": "conditional",
        "state": "live",
        "sz": "10",
        "slTriggerPx": "3400",
        "slOrdPx": "-1",
        "reduceOnly": "true",
    }
    fake_trader.pending_algo_orders = [
        raw,
        {
            "instId": "BTC-USDT-SWAP",
            "algoId": "btc-algo",
            "side": "sell",
            "posSide": "long",
            "ordType": "conditional",
        },
    ]

    orders = await client.fetch_algo_orders("ETH-USDT-SWAP")

    assert len(orders) == 1
    assert orders[0].order_id == "algo-123"
    assert orders[0].metadata["source"] == "algo"
    assert orders[0].raw == raw


@pytest.mark.asyncio
async def test_fetch_position_bridges_legacy_position_snapshot(
    fake_trader: FakeTrader,
    client: OkxBrokerClient,
) -> None:
    fake_trader.position_snapshot = LegacyPositionSnapshot(
        side="LONG",
        contracts=Decimal("10"),
        avg_entry_price=3500.5,
    )

    position = await client.fetch_position("ETH-USDT-SWAP")

    assert position is not None
    assert position.position_side == BrokerPositionSide.LONG
    assert position.quantity == Decimal("10")
    assert position.quantity_unit == BrokerQuantityUnit.CONTRACTS
    assert position.average_entry_price == Decimal("3500.5")
    assert position.raw == {"source": "legacy_position_snapshot"}


@pytest.mark.asyncio
async def test_fetch_position_returns_none_for_flat_snapshot(
    fake_trader: FakeTrader,
    client: OkxBrokerClient,
) -> None:
    fake_trader.position_snapshot = LegacyPositionSnapshot(
        side=None,
        contracts=Decimal("0"),
        avg_entry_price=0.0,
    )

    assert await client.fetch_position("ETH-USDT-SWAP") is None


@pytest.mark.asyncio
async def test_unsupported_order_type_or_side_raises_exchange_error(
    client: OkxBrokerClient,
) -> None:
    request = BrokerOrderRequest(
        exchange=ExchangeName.OKX,
        symbol="ETH-USDT-SWAP",
        side=BrokerOrderSide.BUY,
        position_side=BrokerPositionSide.UNKNOWN,
        order_type=BrokerOrderType.UNKNOWN,
        quantity=Decimal("1"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
    )

    with pytest.raises(ExchangeError) as exc_info:
        await client.place_order(request)

    assert exc_info.value.kind == ExchangeErrorKind.UNSUPPORTED_OPERATION


@pytest.mark.asyncio
async def test_limit_reduce_only_missing_price_raises_invalid_price(
    client: OkxBrokerClient,
) -> None:
    request = BrokerOrderRequest(
        exchange=ExchangeName.OKX,
        symbol="ETH-USDT-SWAP",
        side=BrokerOrderSide.SELL,
        position_side=BrokerPositionSide.LONG,
        order_type=BrokerOrderType.LIMIT,
        quantity=Decimal("1"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
        reduce_only=True,
    )

    with pytest.raises(ExchangeError) as exc_info:
        await client.place_order(request)

    assert exc_info.value.kind == ExchangeErrorKind.INVALID_PRICE


@pytest.mark.asyncio
async def test_stop_market_missing_trigger_price_raises_invalid_price(
    client: OkxBrokerClient,
) -> None:
    request = BrokerOrderRequest(
        exchange=ExchangeName.OKX,
        symbol="ETH-USDT-SWAP",
        side=BrokerOrderSide.SELL,
        position_side=BrokerPositionSide.LONG,
        order_type=BrokerOrderType.STOP_MARKET,
        quantity=Decimal("1"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
        reduce_only=True,
    )

    with pytest.raises(ExchangeError) as exc_info:
        await client.place_order(request)

    assert exc_info.value.kind == ExchangeErrorKind.INVALID_PRICE


@pytest.mark.asyncio
async def test_quantity_less_than_or_equal_to_zero_raises_invalid_size(
    client: OkxBrokerClient,
) -> None:
    request = BrokerOrderRequest(
        exchange=ExchangeName.OKX,
        symbol="ETH-USDT-SWAP",
        side=BrokerOrderSide.BUY,
        position_side=BrokerPositionSide.LONG,
        order_type=BrokerOrderType.MARKET,
        quantity=Decimal("0"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
    )

    with pytest.raises(ExchangeError) as exc_info:
        await client.place_order(request)

    assert exc_info.value.kind == ExchangeErrorKind.INVALID_ORDER_SIZE
