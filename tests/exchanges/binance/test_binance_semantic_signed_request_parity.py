#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_binance_semantic_signed_request_parity.py
@Description: End-to-end parity tests from BinanceBrokerSemanticExecutor
              through to signed BinanceSignedRequest captured by a
              FakeBinanceTransport.

Verifies the full chain for One-way / net position mode:

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
        position_mode="net",
    )
    executor = BinanceBrokerSemanticExecutor(client)
    return executor, transport


def _make_executor_with_algo(payload):
    """Create a BinanceBrokerSemanticExecutor with a fake algo client.

    The algo client uses its own FakeBinanceTransport for capturing
    algo-order signed requests.
    """
    from src.exchanges.binance.algo_orders import BinanceAlgoOrderClient

    transport = FakeBinanceTransport(payload)
    algo_transport = FakeBinanceTransport(payload)
    client = BinanceBrokerClient(
        api_key="test-key",
        api_secret="test-secret",
        transport=transport,
        position_mode="net",
    )
    algo_client = BinanceAlgoOrderClient(
        api_key="test-key",
        api_secret="test-secret",
        transport=algo_transport,
    )
    executor = BinanceBrokerSemanticExecutor(client, algo_client=algo_client)
    return executor, algo_transport


def _order_payload(
    *,
    side="BUY",
    position_side="BOTH",
    order_type="MARKET",
    status="NEW",
    price="0",
    orig_qty="0.1",
    stop_price="0",
):
    """Build a minimal Binance order response payload (One-way mode)."""
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
    """Assertions shared by every semantic -> signed request test."""
    assert req.method == "POST"
    assert req.path == "/fapi/v1/order"
    assert req.headers["X-MBX-APIKEY"] == "test-key"
    assert "timestamp" in params
    assert "recvWindow" in params
    assert "signature" in params
    # In One-way mode, positionSide is never emitted
    assert "positionSide" not in params


def _assert_reduce_only(params):
    """Assert reduceOnly is present in close-order params."""
    assert params["reduceOnly"] == "true"


def _assert_no_reduce_only(params):
    """Assert reduceOnly is absent in open-order params."""
    assert "reduceOnly" not in params


# ===================================================================
# open_position (One-way: no positionSide, no reduceOnly)
# ===================================================================


@pytest.mark.asyncio
async def test_semantic_open_position_long_builds_buy_market_signed_request() -> None:
    executor, transport = _make_executor(
        _order_payload(side="BUY", position_side="BOTH")
    )

    await executor.open_position(
        symbol="ETHUSDT",
        side=BrokerPositionSide.LONG,
        quantity=Decimal("2"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
    )

    req, params = _captured_params(transport)
    _assert_common_signed_request(req, params)
    _assert_no_reduce_only(params)

    assert params["symbol"] == "ETHUSDT"
    assert params["side"] == "BUY"
    assert params["type"] == "MARKET"
    assert params["quantity"] == "0.2"


@pytest.mark.asyncio
async def test_semantic_open_position_short_builds_sell_market_signed_request() -> None:
    executor, transport = _make_executor(
        _order_payload(side="SELL", position_side="BOTH")
    )

    await executor.open_position(
        symbol="ETHUSDT",
        side=BrokerPositionSide.SHORT,
        quantity=Decimal("2"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
    )

    req, params = _captured_params(transport)
    _assert_common_signed_request(req, params)
    _assert_no_reduce_only(params)

    assert params["symbol"] == "ETHUSDT"
    assert params["side"] == "SELL"
    assert params["type"] == "MARKET"
    assert params["quantity"] == "0.2"


# ===================================================================
# place_reduce_only_tp (One-way: SELL/BUY + reduceOnly="true")
# ===================================================================


@pytest.mark.asyncio
async def test_semantic_tp_long_builds_sell_limit_reduce_only_signed_request() -> None:
    executor, transport = _make_executor(
        _order_payload(side="SELL", position_side="BOTH", order_type="LIMIT", price="3550")
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
    _assert_reduce_only(params)

    assert params["symbol"] == "ETHUSDT"
    assert params["side"] == "SELL"
    assert params["type"] == "LIMIT"
    assert params["quantity"] == "0.2"
    assert params["price"] == "3550"
    assert params["timeInForce"] == "GTC"


@pytest.mark.asyncio
async def test_semantic_tp_short_builds_buy_limit_reduce_only_signed_request() -> None:
    executor, transport = _make_executor(
        _order_payload(side="BUY", position_side="BOTH", order_type="LIMIT", price="2550")
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
    _assert_reduce_only(params)

    assert params["symbol"] == "ETHUSDT"
    assert params["side"] == "BUY"
    assert params["type"] == "LIMIT"
    assert params["quantity"] == "0.2"
    assert params["price"] == "2550"
    assert params["timeInForce"] == "GTC"


# ===================================================================
# place_protective_stop (One-way: SELL/BUY + reduceOnly="true")
# ===================================================================


@pytest.mark.asyncio
async def test_semantic_protective_stop_long_builds_sell_stop_market_reduce_only_signed_request() -> None:
    executor, transport = _make_executor_with_algo(
        _order_payload(
            side="SELL", position_side="BOTH", order_type="STOP_MARKET", stop_price="2950"
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
    # Protective stop uses Algo Order API, not /fapi/v1/order
    assert req.method == "POST"
    assert req.path == "/fapi/v1/algoOrder"

    # Algo order uses different param names
    assert params["symbol"] == "ETHUSDT"
    assert params["side"] == "SELL"
    assert params["algoType"] == "CONDITIONAL"
    assert params["type"] == "STOP_MARKET"
    assert params["reduceOnly"] == "true"
    # Quantity is base-asset: 2 contracts * 0.1 = 0.2 ETH
    assert params["quantity"] == "0.2"
    assert params["triggerPrice"] == "2950"


@pytest.mark.asyncio
async def test_semantic_protective_stop_short_builds_buy_stop_market_reduce_only_signed_request() -> None:
    executor, transport = _make_executor_with_algo(
        _order_payload(
            side="BUY", position_side="BOTH", order_type="STOP_MARKET", stop_price="3350"
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
    # Protective stop uses Algo Order API, not /fapi/v1/order
    assert req.method == "POST"
    assert req.path == "/fapi/v1/algoOrder"

    assert params["symbol"] == "ETHUSDT"
    assert params["side"] == "BUY"
    assert params["algoType"] == "CONDITIONAL"
    assert params["type"] == "STOP_MARKET"
    assert params["reduceOnly"] == "true"
    assert params["quantity"] == "0.2"
    assert params["triggerPrice"] == "3350"


# ===================================================================
# market_exit (One-way: SELL/BUY + reduceOnly="true")
# ===================================================================


@pytest.mark.asyncio
async def test_semantic_market_exit_long_builds_sell_market_reduce_only_signed_request() -> None:
    executor, transport = _make_executor(
        _order_payload(side="SELL", position_side="BOTH")
    )

    await executor.market_exit(
        symbol="ETHUSDT",
        side=BrokerPositionSide.LONG,
        quantity=Decimal("2"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
    )

    req, params = _captured_params(transport)
    _assert_common_signed_request(req, params)
    _assert_reduce_only(params)

    assert params["symbol"] == "ETHUSDT"
    assert params["side"] == "SELL"
    assert params["type"] == "MARKET"
    assert params["quantity"] == "0.2"


@pytest.mark.asyncio
async def test_semantic_market_exit_short_builds_buy_market_reduce_only_signed_request() -> None:
    executor, transport = _make_executor(
        _order_payload(side="BUY", position_side="BOTH")
    )

    await executor.market_exit(
        symbol="ETHUSDT",
        side=BrokerPositionSide.SHORT,
        quantity=Decimal("2"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
    )

    req, params = _captured_params(transport)
    _assert_common_signed_request(req, params)
    _assert_reduce_only(params)

    assert params["symbol"] == "ETHUSDT"
    assert params["side"] == "BUY"
    assert params["type"] == "MARKET"
    assert params["quantity"] == "0.2"
