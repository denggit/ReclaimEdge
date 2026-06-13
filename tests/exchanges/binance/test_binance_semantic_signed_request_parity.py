#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_binance_semantic_signed_request_parity.py
@Description: End-to-end parity tests from BinanceBrokerSemanticExecutor
              through to signed BinanceSignedRequest captured by a
              FakeBinanceTransport.

Verifies the full chain:

    BinanceBrokerSemanticExecutor
      -> BinanceBrokerClient
      -> broker_order_request_to_binance_params
      -> build_signed_request
      -> FakeBinanceTransport captures BinanceSignedRequest
"""

from __future__ import annotations

from decimal import Decimal
from urllib.parse import parse_qs

import pytest

from src.exchanges.binance import BinanceBrokerClient, BinanceBrokerSemanticExecutor
from src.exchanges.binance.transport import BinanceTransportResponse
from src.exchanges.models import BrokerPositionSide, BrokerQuantityUnit


# ---------------------------------------------------------------------------
# Fake transport
# ---------------------------------------------------------------------------


class FakeBinanceTransport:
    """Records every request and returns a canned response."""

    def __init__(self, payload):
        self.payload = payload
        self.requests: list = []

    async def send(self, request):
        self.requests.append(request)
        return BinanceTransportResponse(
            status_code=200,
            payload=self.payload,
            headers={},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_executor(payload):
    """Create a BinanceBrokerSemanticExecutor wired to a FakeBinanceTransport."""
    transport = FakeBinanceTransport(payload)
    client = BinanceBrokerClient(
        api_key="test-key",
        api_secret="test-secret",
        transport=transport,
    )
    executor = BinanceBrokerSemanticExecutor(client)
    return executor, transport


def _order_payload(
    *,
    side="BUY",
    position_side="LONG",
    order_type="MARKET",
    status="NEW",
    price="0",
    orig_qty="0.1",
    stop_price="0",
):
    """Build a minimal Binance order response payload."""
    return {
        "symbol": "ETHUSDT",
        "orderId": 123456,
        "clientOrderId": "cid-test",
        "side": side,
        "positionSide": position_side,
        "type": order_type,
        "status": status,
        "price": price,
        "origQty": orig_qty,
        "executedQty": "0",
        "avgPrice": "0",
        "reduceOnly": False,
        "stopPrice": stop_price,
    }


def _captured_params(transport):
    """Extract the single captured signed request and parse its query string."""
    assert len(transport.requests) == 1
    req = transport.requests[0]
    parsed = parse_qs(req.query_string)
    return req, {key: values[-1] for key, values in parsed.items()}


def _assert_common_signed_request(req, params):
    """Assertions shared by every semantic → signed request test."""
    assert req.method == "POST"
    assert req.path == "/fapi/v1/order"
    assert req.headers["X-MBX-APIKEY"] == "test-key"
    assert "timestamp" in params
    assert "recvWindow" in params
    assert "signature" in params
    assert "reduceOnly" not in params


# ===================================================================
# open_position
# ===================================================================


@pytest.mark.asyncio
async def test_semantic_open_position_long_builds_buy_long_market_signed_request() -> None:
    """Semantic open_position LONG → BUY LONG MARKET signed request."""
    executor, transport = _make_executor(
        _order_payload(side="BUY", position_side="LONG")
    )

    await executor.open_position(
        symbol="ETHUSDT",
        side=BrokerPositionSide.LONG,
        quantity=Decimal("2"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
    )

    req, params = _captured_params(transport)
    _assert_common_signed_request(req, params)

    assert params["symbol"] == "ETHUSDT"
    assert params["side"] == "BUY"
    assert params["positionSide"] == "LONG"
    assert params["type"] == "MARKET"
    assert params["quantity"] == "0.2"


@pytest.mark.asyncio
async def test_semantic_open_position_short_builds_sell_short_market_signed_request() -> None:
    """Semantic open_position SHORT → SELL SHORT MARKET signed request."""
    executor, transport = _make_executor(
        _order_payload(side="SELL", position_side="SHORT")
    )

    await executor.open_position(
        symbol="ETHUSDT",
        side=BrokerPositionSide.SHORT,
        quantity=Decimal("2"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
    )

    req, params = _captured_params(transport)
    _assert_common_signed_request(req, params)

    assert params["symbol"] == "ETHUSDT"
    assert params["side"] == "SELL"
    assert params["positionSide"] == "SHORT"
    assert params["type"] == "MARKET"
    assert params["quantity"] == "0.2"


# ===================================================================
# place_reduce_only_tp
# ===================================================================


@pytest.mark.asyncio
async def test_semantic_tp_long_builds_sell_long_limit_signed_request() -> None:
    """Semantic reduce-only TP LONG → SELL LONG LIMIT price/timeInForce."""
    executor, transport = _make_executor(
        _order_payload(side="SELL", position_side="LONG", order_type="LIMIT", price="3550")
    )

    await executor.place_reduce_only_tp(
        symbol="ETHUSDT",
        side=BrokerPositionSide.LONG,
        quantity=Decimal("2"),
        trigger_price=Decimal("3550"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
    )

    req, params = _captured_params(transport)
    _assert_common_signed_request(req, params)

    assert params["symbol"] == "ETHUSDT"
    assert params["side"] == "SELL"
    assert params["positionSide"] == "LONG"
    assert params["type"] == "LIMIT"
    assert params["quantity"] == "0.2"
    assert params["price"] == "3550"
    assert params["timeInForce"] == "GTC"


@pytest.mark.asyncio
async def test_semantic_tp_short_builds_buy_short_limit_signed_request() -> None:
    """Semantic reduce-only TP SHORT → BUY SHORT LIMIT price/timeInForce."""
    executor, transport = _make_executor(
        _order_payload(side="BUY", position_side="SHORT", order_type="LIMIT", price="2550")
    )

    await executor.place_reduce_only_tp(
        symbol="ETHUSDT",
        side=BrokerPositionSide.SHORT,
        quantity=Decimal("2"),
        trigger_price=Decimal("2550"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
    )

    req, params = _captured_params(transport)
    _assert_common_signed_request(req, params)

    assert params["symbol"] == "ETHUSDT"
    assert params["side"] == "BUY"
    assert params["positionSide"] == "SHORT"
    assert params["type"] == "LIMIT"
    assert params["quantity"] == "0.2"
    assert params["price"] == "2550"
    assert params["timeInForce"] == "GTC"


# ===================================================================
# place_protective_stop
# ===================================================================


@pytest.mark.asyncio
async def test_semantic_protective_stop_long_builds_sell_long_stop_market_signed_request() -> None:
    """Semantic protective stop LONG → SELL LONG STOP_MARKET stopPrice."""
    executor, transport = _make_executor(
        _order_payload(
            side="SELL", position_side="LONG", order_type="STOP_MARKET", stop_price="2950"
        )
    )

    await executor.place_protective_stop(
        symbol="ETHUSDT",
        side=BrokerPositionSide.LONG,
        quantity=Decimal("2"),
        trigger_price=Decimal("2950"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
    )

    req, params = _captured_params(transport)
    _assert_common_signed_request(req, params)

    assert params["symbol"] == "ETHUSDT"
    assert params["side"] == "SELL"
    assert params["positionSide"] == "LONG"
    assert params["type"] == "STOP_MARKET"
    assert params["quantity"] == "0.2"
    assert params["stopPrice"] == "2950"


@pytest.mark.asyncio
async def test_semantic_protective_stop_short_builds_buy_short_stop_market_signed_request() -> None:
    """Semantic protective stop SHORT → BUY SHORT STOP_MARKET stopPrice."""
    executor, transport = _make_executor(
        _order_payload(
            side="BUY", position_side="SHORT", order_type="STOP_MARKET", stop_price="3350"
        )
    )

    await executor.place_protective_stop(
        symbol="ETHUSDT",
        side=BrokerPositionSide.SHORT,
        quantity=Decimal("2"),
        trigger_price=Decimal("3350"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
    )

    req, params = _captured_params(transport)
    _assert_common_signed_request(req, params)

    assert params["symbol"] == "ETHUSDT"
    assert params["side"] == "BUY"
    assert params["positionSide"] == "SHORT"
    assert params["type"] == "STOP_MARKET"
    assert params["quantity"] == "0.2"
    assert params["stopPrice"] == "3350"


# ===================================================================
# market_exit
# ===================================================================


@pytest.mark.asyncio
async def test_semantic_market_exit_long_builds_sell_long_market_signed_request() -> None:
    """Semantic market_exit LONG → SELL LONG MARKET signed request."""
    executor, transport = _make_executor(
        _order_payload(side="SELL", position_side="LONG")
    )

    await executor.market_exit(
        symbol="ETHUSDT",
        side=BrokerPositionSide.LONG,
        quantity=Decimal("2"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
    )

    req, params = _captured_params(transport)
    _assert_common_signed_request(req, params)

    assert params["symbol"] == "ETHUSDT"
    assert params["side"] == "SELL"
    assert params["positionSide"] == "LONG"
    assert params["type"] == "MARKET"
    assert params["quantity"] == "0.2"


@pytest.mark.asyncio
async def test_semantic_market_exit_short_builds_buy_short_market_signed_request() -> None:
    """Semantic market_exit SHORT → BUY SHORT MARKET signed request."""
    executor, transport = _make_executor(
        _order_payload(side="BUY", position_side="SHORT")
    )

    await executor.market_exit(
        symbol="ETHUSDT",
        side=BrokerPositionSide.SHORT,
        quantity=Decimal("2"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
    )

    req, params = _captured_params(transport)
    _assert_common_signed_request(req, params)

    assert params["symbol"] == "ETHUSDT"
    assert params["side"] == "BUY"
    assert params["positionSide"] == "SHORT"
    assert params["type"] == "MARKET"
    assert params["quantity"] == "0.2"
