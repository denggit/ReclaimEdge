#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_binance_broker_client_shell.py
@Description: Verify that BinanceBrokerClient shell satisfies the BrokerClient
              contract without performing real network requests.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.exchanges.binance import BinanceBrokerClient
from src.exchanges.base import BrokerClient
from src.exchanges.errors import ExchangeError, ExchangeErrorKind
from src.exchanges.models import (
    BrokerOrderRequest,
    BrokerOrderSide,
    BrokerOrderType,
    BrokerPositionSide,
    BrokerQuantityUnit,
    ExchangeName,
)


@pytest.mark.asyncio
async def test_binance_client_exchange_name_and_port() -> None:
    client = BinanceBrokerClient()

    assert isinstance(client, BrokerClient)
    assert client.exchange == ExchangeName.BINANCE


async def _assert_unsupported(coro, operation: str) -> None:
    with pytest.raises(ExchangeError) as exc_info:
        await coro

    err = exc_info.value
    assert err.exchange == ExchangeName.BINANCE
    assert err.kind == ExchangeErrorKind.UNSUPPORTED_OPERATION
    assert operation in err.message
    assert err.raw.get("operation") == operation


@pytest.mark.asyncio
async def test_binance_client_methods_raise_unsupported() -> None:
    client = BinanceBrokerClient()

    request = BrokerOrderRequest(
        exchange=ExchangeName.BINANCE,
        symbol="ETHUSDT",
        side=BrokerOrderSide.BUY,
        position_side=BrokerPositionSide.LONG,
        order_type=BrokerOrderType.MARKET,
        quantity=Decimal("1"),
        quantity_unit=BrokerQuantityUnit.CONTRACTS,
    )

    await _assert_unsupported(client.place_order(request), "place_order")
    await _assert_unsupported(client.cancel_order("ETHUSDT", "order-1"), "cancel_order")
    await _assert_unsupported(client.fetch_open_orders("ETHUSDT"), "fetch_open_orders")
    await _assert_unsupported(client.fetch_position("ETHUSDT"), "fetch_position")
