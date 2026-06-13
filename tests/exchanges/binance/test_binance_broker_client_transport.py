#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_binance_broker_client_transport.py
@Description: Verify BinanceBrokerClient with an injected FakeBinanceTransport
              correctly builds signed requests, dispatches them, and maps
              responses (or errors) back into Broker* DTOs.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.exchanges.binance import BinanceBrokerClient
from src.exchanges.binance.signing import (
    BINANCE_USDM_OPEN_ORDERS_PATH,
    BINANCE_USDM_ORDER_PATH,
    BINANCE_USDM_POSITION_RISK_PATH,
)
from src.exchanges.binance.transport import BinanceTransportResponse
from src.exchanges.errors import ExchangeError, ExchangeErrorKind
from src.exchanges.models import (
    BrokerCancelResult,
    BrokerOrder,
    BrokerOrderRequest,
    BrokerOrderResult,
    BrokerOrderSide,
    BrokerOrderStatus,
    BrokerOrderType,
    BrokerPositionSide,
    BrokerQuantityUnit,
    ExchangeName,
)


# ---------------------------------------------------------------------------
# Fake transport
# ---------------------------------------------------------------------------


class FakeBinanceTransport:
    """Records every request and returns a canned response."""

    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.requests: list = []

    async def send(self, request):
        self.requests.append(request)
        return BinanceTransportResponse(
            status_code=self.status_code,
            payload=self.payload,
            headers={},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(transport=None, **kwargs):
    return BinanceBrokerClient(
        api_key="test-api-key",
        api_secret="test-api-secret",
        transport=transport,
        **kwargs,
    )


def _market_buy_request():
    return BrokerOrderRequest(
        exchange=ExchangeName.BINANCE,
        symbol="ETHUSDT",
        side=BrokerOrderSide.BUY,
        position_side=BrokerPositionSide.LONG,
        order_type=BrokerOrderType.MARKET,
        quantity=Decimal("1"),
        quantity_unit=BrokerQuantityUnit.BASE_ASSET,
    )


def _minimal_order_payload(**overrides):
    data = {
        "symbol": "ETHUSDT",
        "orderId": 123456789,
        "clientOrderId": "cid-abc-001",
        "side": "BUY",
        "positionSide": "LONG",
        "type": "MARKET",
        "status": "NEW",
        "price": "0",
        "origQty": "1.0",
        "executedQty": "0",
        "avgPrice": "0",
        "reduceOnly": False,
    }
    data.update(overrides)
    return data


# ===================================================================
# place_order
# ===================================================================


@pytest.mark.asyncio
async def test_place_order_sends_post_to_order_path() -> None:
    transport = FakeBinanceTransport(_minimal_order_payload())
    client = _make_client(transport)

    await client.place_order(_market_buy_request())

    assert len(transport.requests) == 1
    req = transport.requests[0]
    assert req.method == "POST"
    assert req.path == BINANCE_USDM_ORDER_PATH


@pytest.mark.asyncio
async def test_place_order_includes_api_key_header() -> None:
    transport = FakeBinanceTransport(_minimal_order_payload())
    client = _make_client(transport)

    await client.place_order(_market_buy_request())

    req = transport.requests[0]
    assert req.headers.get("X-MBX-APIKEY") == "test-api-key"


@pytest.mark.asyncio
async def test_place_order_maps_payload_to_broker_order_result() -> None:
    transport = FakeBinanceTransport(_minimal_order_payload())
    client = _make_client(transport)

    result = await client.place_order(_market_buy_request())

    assert isinstance(result, BrokerOrderResult)
    assert result.exchange == ExchangeName.BINANCE
    assert result.ok is True
    assert result.order_id == "123456789"
    assert result.client_order_id == "cid-abc-001"
    assert result.order is not None
    assert result.order.symbol == "ETHUSDT"
    assert result.order.side == BrokerOrderSide.BUY
    assert result.order.position_side == BrokerPositionSide.LONG
    assert result.order.order_type == BrokerOrderType.MARKET
    assert result.order.status == BrokerOrderStatus.OPEN


# ===================================================================
# cancel_order
# ===================================================================


@pytest.mark.asyncio
async def test_cancel_order_sends_delete_to_order_path() -> None:
    transport = FakeBinanceTransport({
        "symbol": "ETHUSDT",
        "orderId": 111222333,
        "clientOrderId": "cid-cancel-1",
        "status": "CANCELED",
    })
    client = _make_client(transport)

    await client.cancel_order("ETHUSDT", "111222333")

    assert len(transport.requests) == 1
    req = transport.requests[0]
    assert req.method == "DELETE"
    assert req.path == BINANCE_USDM_ORDER_PATH


@pytest.mark.asyncio
async def test_cancel_order_maps_payload_to_broker_cancel_result() -> None:
    transport = FakeBinanceTransport({
        "symbol": "ETHUSDT",
        "orderId": 111222333,
        "clientOrderId": "cid-cancel-1",
        "status": "CANCELED",
    })
    client = _make_client(transport)

    result = await client.cancel_order("ETHUSDT", "111222333")

    assert isinstance(result, BrokerCancelResult)
    assert result.exchange == ExchangeName.BINANCE
    assert result.ok is True
    assert result.order_id == "111222333"
    assert result.client_order_id == "cid-cancel-1"


# ===================================================================
# fetch_open_orders
# ===================================================================


@pytest.mark.asyncio
async def test_fetch_open_orders_sends_get_to_open_orders_path() -> None:
    transport = FakeBinanceTransport([])
    client = _make_client(transport)

    await client.fetch_open_orders("ETHUSDT")

    assert len(transport.requests) == 1
    req = transport.requests[0]
    assert req.method == "GET"
    assert req.path == BINANCE_USDM_OPEN_ORDERS_PATH


@pytest.mark.asyncio
async def test_fetch_open_orders_maps_list_into_broker_orders() -> None:
    transport = FakeBinanceTransport([
        _minimal_order_payload(orderId=1),
        _minimal_order_payload(orderId=2, side="SELL", positionSide="SHORT"),
    ])
    client = _make_client(transport)

    orders = await client.fetch_open_orders("ETHUSDT")

    assert isinstance(orders, list)
    assert len(orders) == 2
    assert all(isinstance(o, BrokerOrder) for o in orders)
    assert orders[0].order_id == "1"
    assert orders[0].side == BrokerOrderSide.BUY
    assert orders[1].order_id == "2"
    assert orders[1].side == BrokerOrderSide.SELL
    assert orders[1].position_side == BrokerPositionSide.SHORT


# ===================================================================
# Error mapping
# ===================================================================


@pytest.mark.asyncio
async def test_http_401_raises_auth_error() -> None:
    transport = FakeBinanceTransport(
        {"code": -2015, "msg": "Invalid API-key."},
        status_code=401,
    )
    client = _make_client(transport)

    with pytest.raises(ExchangeError) as exc_info:
        await client.place_order(_market_buy_request())

    err = exc_info.value
    assert err.kind == ExchangeErrorKind.AUTH_ERROR


@pytest.mark.asyncio
async def test_binance_error_code_order_not_found() -> None:
    transport = FakeBinanceTransport(
        {"code": -2011, "msg": "Order does not exist."},
        status_code=400,
    )
    client = _make_client(transport)

    with pytest.raises(ExchangeError) as exc_info:
        await client.cancel_order("ETHUSDT", "999999")

    err = exc_info.value
    assert err.kind == ExchangeErrorKind.ORDER_NOT_FOUND


# ===================================================================
# Pre-send validation
# ===================================================================


@pytest.mark.asyncio
async def test_invalid_symbol_rejects_before_transport_send() -> None:
    transport = FakeBinanceTransport({})
    client = _make_client(transport)

    with pytest.raises(ValueError, match="Unsupported Binance symbol"):
        await client.cancel_order("BTCUSDT", "123")

    assert len(transport.requests) == 0


@pytest.mark.asyncio
async def test_missing_order_id_rejects_before_transport_send() -> None:
    transport = FakeBinanceTransport({})
    client = _make_client(transport)

    with pytest.raises(ValueError, match="order_id must not be empty"):
        await client.cancel_order("ETHUSDT", "")

    assert len(transport.requests) == 0


# ===================================================================
# Transport readiness (no-arg shell)
# ===================================================================


@pytest.mark.asyncio
async def test_no_arg_client_raises_unsupported() -> None:
    client = BinanceBrokerClient()

    with pytest.raises(ExchangeError) as exc_info:
        await client.place_order(_market_buy_request())

    assert exc_info.value.kind == ExchangeErrorKind.UNSUPPORTED_OPERATION
    assert "place_order" in exc_info.value.message


@pytest.mark.asyncio
async def test_partial_credentials_raises_unsupported() -> None:
    # api_key only, no secret and no transport
    client = BinanceBrokerClient(api_key="k")

    with pytest.raises(ExchangeError) as exc_info:
        await client.fetch_open_orders("ETHUSDT")

    assert exc_info.value.kind == ExchangeErrorKind.UNSUPPORTED_OPERATION


@pytest.mark.asyncio
async def test_transport_without_credentials_raises_unsupported() -> None:
    transport = FakeBinanceTransport({})
    client = BinanceBrokerClient(transport=transport)

    with pytest.raises(ExchangeError) as exc_info:
        await client.fetch_position("ETHUSDT")

    assert exc_info.value.kind == ExchangeErrorKind.UNSUPPORTED_OPERATION
