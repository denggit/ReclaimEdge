#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_binance_broker_client_position_fetch.py
@Description: Verify BinanceBrokerClient.fetch_position with injected transport.

Covers hedge-mode position fetch scenarios including zero, single-side,
dual-side (unsupported) and error payloads.
"""

from __future__ import annotations

import pytest

from src.exchanges.binance import BinanceBrokerClient
from src.exchanges.binance.signing import BINANCE_USDM_POSITION_RISK_PATH
from src.exchanges.binance.transport import BinanceTransportResponse
from src.exchanges.errors import ExchangeError, ExchangeErrorKind
from src.exchanges.models import BrokerPositionSide, ExchangeName


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


def _make_client(transport):
    return BinanceBrokerClient(
        api_key="test-api-key",
        api_secret="test-api-secret",
        transport=transport,
    )


# ---------------------------------------------------------------------------
# Position payload helpers
# ---------------------------------------------------------------------------


def _position_item(**overrides):
    data = {
        "symbol": "ETHUSDT",
        "positionAmt": "0",
        "entryPrice": "0",
        "markPrice": "0",
        "unRealizedProfit": "0",
        "leverage": "1",
        "positionSide": "BOTH",
    }
    data.update(overrides)
    return data


# ===================================================================
# Path and method
# ===================================================================


@pytest.mark.asyncio
async def test_fetch_position_sends_get_to_position_risk_path() -> None:
    transport = FakeBinanceTransport([_position_item()])
    client = _make_client(transport)

    await client.fetch_position("ETHUSDT")

    assert len(transport.requests) == 1
    req = transport.requests[0]
    assert req.method == "GET"
    assert req.path == BINANCE_USDM_POSITION_RISK_PATH


# ===================================================================
# Zero / no position
# ===================================================================


@pytest.mark.asyncio
async def test_zero_long_and_zero_short_returns_none() -> None:
    transport = FakeBinanceTransport([_position_item(positionAmt="0", positionSide="BOTH")])
    client = _make_client(transport)

    result = await client.fetch_position("ETHUSDT")

    assert result is None


# ===================================================================
# Single active position
# ===================================================================


@pytest.mark.asyncio
async def test_one_active_long_returns_broker_position_long() -> None:
    transport = FakeBinanceTransport([
        _position_item(
            positionAmt="1.5",
            entryPrice="3100.50",
            markPrice="3150.00",
            unRealizedProfit="74.25",
            leverage="10",
            positionSide="LONG",
        )
    ])
    client = _make_client(transport)

    result = await client.fetch_position("ETHUSDT")

    assert result is not None
    assert result.exchange == ExchangeName.BINANCE
    assert result.symbol == "ETHUSDT"
    assert result.position_side == BrokerPositionSide.LONG
    assert result.average_entry_price is not None


@pytest.mark.asyncio
async def test_one_active_short_returns_broker_position_short() -> None:
    transport = FakeBinanceTransport([
        _position_item(
            positionAmt="-2.0",
            entryPrice="3200.00",
            markPrice="3180.00",
            unRealizedProfit="40.00",
            leverage="5",
            positionSide="SHORT",
        )
    ])
    client = _make_client(transport)

    result = await client.fetch_position("ETHUSDT")

    assert result is not None
    assert result.exchange == ExchangeName.BINANCE
    assert result.position_side == BrokerPositionSide.SHORT


# ===================================================================
# Simultaneous hedge positions
# ===================================================================


@pytest.mark.asyncio
async def test_simultaneous_long_and_short_raises_unsupported_operation() -> None:
    transport = FakeBinanceTransport([
        _position_item(positionAmt="1.5", positionSide="LONG"),
        _position_item(positionAmt="-0.8", positionSide="SHORT"),
    ])
    client = _make_client(transport)

    with pytest.raises(ExchangeError) as exc_info:
        await client.fetch_position("ETHUSDT")

    err = exc_info.value
    assert err.kind == ExchangeErrorKind.UNSUPPORTED_OPERATION
    assert "simultaneous" in err.message.lower()


# ===================================================================
# Non-list payload
# ===================================================================


@pytest.mark.asyncio
async def test_non_list_payload_raises_exchange_error() -> None:
    transport = FakeBinanceTransport({"not": "a list"})
    client = _make_client(transport)

    with pytest.raises(ExchangeError) as exc_info:
        await client.fetch_position("ETHUSDT")

    err = exc_info.value
    assert "payload is not a list" in err.message.lower()
