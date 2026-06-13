#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_binance_mapper_orders.py
@Description: Unit tests for Binance order response mapping.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.exchanges.binance.mapper import map_binance_order
from src.exchanges.models import (
    BrokerOrderSide,
    BrokerOrderStatus,
    BrokerOrderType,
    BrokerPositionSide,
    BrokerQuantityUnit,
    ExchangeName,
)


# ---------------------------------------------------------------------------
# Happy path — recognised fields
# ---------------------------------------------------------------------------

def test_map_new_limit_reduce_only_order() -> None:
    """A NEW LIMIT buy order with reduceOnly=True maps correctly."""
    raw = {
        "symbol": "ETHUSDT",
        "orderId": 123456789,
        "clientOrderId": "test-client-001",
        "side": "BUY",
        "positionSide": "LONG",
        "type": "LIMIT",
        "status": "NEW",
        "price": "3100.50",
        "origQty": "0.100",
        "executedQty": "0",
        "avgPrice": "0",
        "reduceOnly": True,
        "stopPrice": "0",
    }

    order = map_binance_order(raw)

    assert order.exchange == ExchangeName.BINANCE
    assert order.symbol == "ETHUSDT"
    assert order.order_id == "123456789"
    assert order.client_order_id == "test-client-001"
    assert order.side == BrokerOrderSide.BUY
    assert order.position_side == BrokerPositionSide.LONG
    assert order.order_type == BrokerOrderType.LIMIT
    assert order.status == BrokerOrderStatus.OPEN
    assert order.price == Decimal("3100.50")
    assert order.quantity == Decimal("0.100")
    assert order.quantity_unit == BrokerQuantityUnit.BASE_ASSET
    assert order.filled_quantity == Decimal("0")
    assert order.average_price == Decimal("0")
    assert order.reduce_only is True
    assert order.trigger_price == Decimal("0")
    assert order.raw == raw


def test_map_filled_take_profit_market_order() -> None:
    """A FILLED TAKE_PROFIT_MARKET sell order maps correctly."""
    raw = {
        "symbol": "ETHUSDT",
        "orderId": 987654321,
        "clientOrderId": "tp-client-002",
        "side": "SELL",
        "positionSide": "LONG",
        "type": "TAKE_PROFIT_MARKET",
        "status": "FILLED",
        "price": "3150.00",
        "origQty": "0.050",
        "executedQty": "0.050",
        "avgPrice": "3150.25",
        "reduceOnly": True,
        "stopPrice": "3150.00",
    }

    order = map_binance_order(raw)

    assert order.exchange == ExchangeName.BINANCE
    assert order.symbol == "ETHUSDT"
    assert order.side == BrokerOrderSide.SELL
    assert order.order_type == BrokerOrderType.TAKE_PROFIT_MARKET
    assert order.status == BrokerOrderStatus.FILLED
    assert order.quantity == Decimal("0.050")
    assert order.filled_quantity == Decimal("0.050")
    assert order.average_price == Decimal("3150.25")
    assert order.trigger_price == Decimal("3150.00")
    assert order.reduce_only is True
    assert order.quantity_unit == BrokerQuantityUnit.BASE_ASSET


def test_map_partially_filled_stop_market_order() -> None:
    """A PARTIALLY_FILLED STOP_MARKET order maps correctly."""
    raw = {
        "symbol": "ETHUSDT",
        "orderId": 555555,
        "clientOrderId": "sl-client-003",
        "side": "SELL",
        "positionSide": "SHORT",
        "type": "STOP_MARKET",
        "status": "PARTIALLY_FILLED",
        "price": "0",
        "origQty": "0.200",
        "executedQty": "0.100",
        "avgPrice": "3000.00",
        "reduceOnly": True,
        "stopPrice": "2990.00",
    }

    order = map_binance_order(raw)

    assert order.order_type == BrokerOrderType.STOP_MARKET
    assert order.status == BrokerOrderStatus.PARTIALLY_FILLED
    assert order.position_side == BrokerPositionSide.SHORT
    assert order.quantity == Decimal("0.200")
    assert order.filled_quantity == Decimal("0.100")
    assert order.trigger_price == Decimal("2990.00")
    assert order.quantity_unit == BrokerQuantityUnit.BASE_ASSET


def test_map_market_order_canceled() -> None:
    """A CANCELED MARKET order maps correctly."""
    raw = {
        "symbol": "ETHUSDT",
        "orderId": 111222,
        "clientOrderId": "cancel-test",
        "side": "BUY",
        "positionSide": "LONG",
        "type": "MARKET",
        "status": "CANCELED",
        "price": "0",
        "origQty": "0.300",
        "executedQty": "0",
        "avgPrice": "0",
        "reduceOnly": False,
        "stopPrice": "0",
    }

    order = map_binance_order(raw)

    assert order.order_type == BrokerOrderType.MARKET
    assert order.status == BrokerOrderStatus.CANCELED
    assert order.reduce_only is False


def test_map_cancelled_spelling_variant() -> None:
    """Binance uses double-L 'CANCELLED' spelling — must map to CANCELED."""
    raw = {
        "symbol": "ETHUSDT",
        "orderId": 444333,
        "clientOrderId": "cxl-test",
        "side": "SELL",
        "positionSide": "SHORT",
        "type": "LIMIT",
        "status": "CANCELLED",
        "price": "3200.00",
        "origQty": "0.010",
        "executedQty": "0",
        "avgPrice": "0",
        "reduceOnly": False,
        "stopPrice": "0",
    }

    order = map_binance_order(raw)
    assert order.status == BrokerOrderStatus.CANCELED


def test_map_rejected_order() -> None:
    """A REJECTED order maps correctly."""
    raw = {
        "symbol": "ETHUSDT",
        "orderId": 777888,
        "clientOrderId": "rej-test",
        "side": "BUY",
        "positionSide": "LONG",
        "type": "LIMIT",
        "status": "REJECTED",
        "price": "99999.00",
        "origQty": "1.000",
        "executedQty": "0",
        "avgPrice": "0",
        "reduceOnly": False,
        "stopPrice": "0",
    }

    order = map_binance_order(raw)
    assert order.status == BrokerOrderStatus.REJECTED


def test_map_expired_order() -> None:
    """An EXPIRED order maps correctly."""
    raw = {
        "symbol": "ETHUSDT",
        "orderId": 999000,
        "clientOrderId": "exp-test",
        "side": "SELL",
        "positionSide": "SHORT",
        "type": "LIMIT",
        "status": "EXPIRED",
        "price": "5000.00",
        "origQty": "0.500",
        "executedQty": "0",
        "avgPrice": "0",
        "reduceOnly": False,
        "stopPrice": "0",
    }

    order = map_binance_order(raw)
    assert order.status == BrokerOrderStatus.EXPIRED


# ---------------------------------------------------------------------------
# Symbol guard
# ---------------------------------------------------------------------------

def test_unsupported_symbol_btcusdt_raises_value_error() -> None:
    """BTCUSDT is not supported yet — must raise ValueError."""
    raw = {
        "symbol": "BTCUSDT",
        "orderId": 1,
        "clientOrderId": "bad",
        "side": "BUY",
        "positionSide": "LONG",
        "type": "LIMIT",
        "status": "NEW",
        "price": "50000.00",
        "origQty": "0.010",
        "executedQty": "0",
        "avgPrice": "0",
        "reduceOnly": False,
        "stopPrice": "0",
    }

    with pytest.raises(ValueError, match="Unsupported Binance symbol"):
        map_binance_order(raw)


# ---------------------------------------------------------------------------
# Unknown enum fallbacks
# ---------------------------------------------------------------------------

def test_unknown_side_falls_back_to_unknown() -> None:
    raw = {
        "symbol": "ETHUSDT",
        "orderId": 1,
        "clientOrderId": "x",
        "side": "LONG",
        "positionSide": "LONG",
        "type": "LIMIT",
        "status": "NEW",
        "price": "100",
        "origQty": "0.1",
        "executedQty": "0",
        "avgPrice": "0",
        "reduceOnly": False,
        "stopPrice": "0",
    }

    order = map_binance_order(raw)
    assert order.side == BrokerOrderSide.UNKNOWN


def test_unknown_type_falls_back_to_unknown() -> None:
    raw = {
        "symbol": "ETHUSDT",
        "orderId": 1,
        "clientOrderId": "x",
        "side": "BUY",
        "positionSide": "LONG",
        "type": "STOP_LIMIT",
        "status": "NEW",
        "price": "100",
        "origQty": "0.1",
        "executedQty": "0",
        "avgPrice": "0",
        "reduceOnly": False,
        "stopPrice": "0",
    }

    order = map_binance_order(raw)
    assert order.order_type == BrokerOrderType.UNKNOWN


def test_unknown_status_falls_back_to_unknown() -> None:
    raw = {
        "symbol": "ETHUSDT",
        "orderId": 1,
        "clientOrderId": "x",
        "side": "BUY",
        "positionSide": "LONG",
        "type": "LIMIT",
        "status": "PENDING_NEW",
        "price": "100",
        "origQty": "0.1",
        "executedQty": "0",
        "avgPrice": "0",
        "reduceOnly": False,
        "stopPrice": "0",
    }

    order = map_binance_order(raw)
    assert order.status == BrokerOrderStatus.UNKNOWN


def test_none_side_falls_back_to_unknown() -> None:
    """When side key is missing entirely, should resolve to UNKNOWN."""
    raw = {
        "symbol": "ETHUSDT",
        "orderId": 1,
        "clientOrderId": "x",
        "positionSide": "LONG",
        "type": "LIMIT",
        "status": "NEW",
        "price": "100",
        "origQty": "0.1",
        "executedQty": "0",
        "avgPrice": "0",
        "reduceOnly": False,
        "stopPrice": "0",
    }

    order = map_binance_order(raw)
    assert order.side == BrokerOrderSide.UNKNOWN


# ---------------------------------------------------------------------------
# quantity_unit guarantee
# ---------------------------------------------------------------------------

def test_quantity_unit_is_always_base_asset() -> None:
    raw = {
        "symbol": "ETHUSDT",
        "orderId": 1,
        "clientOrderId": "qty-test",
        "side": "BUY",
        "positionSide": "LONG",
        "type": "LIMIT",
        "status": "NEW",
        "price": "3000.00",
        "origQty": "2.500",
        "executedQty": "0",
        "avgPrice": "0",
        "reduceOnly": False,
        "stopPrice": "0",
    }

    order = map_binance_order(raw)

    assert order.quantity_unit == BrokerQuantityUnit.BASE_ASSET
    assert order.quantity_unit != BrokerQuantityUnit.CONTRACTS
    assert order.quantity_unit != BrokerQuantityUnit.QUOTE_ASSET


# ---------------------------------------------------------------------------
# Edge cases — missing optional fields
# ---------------------------------------------------------------------------

def test_map_order_with_minimal_fields() -> None:
    """Order with only required fields present."""
    raw = {
        "symbol": "ETHUSDT",
        "orderId": 1,
        "side": "BUY",
        "type": "MARKET",
        "status": "FILLED",
    }

    order = map_binance_order(raw)

    assert order.exchange == ExchangeName.BINANCE
    assert order.symbol == "ETHUSDT"
    assert order.client_order_id is None
    assert order.price is None
    assert order.quantity is None
    assert order.filled_quantity is None
    assert order.average_price is None
    assert order.trigger_price is None
    assert order.reduce_only is False


def test_map_order_with_null_fields() -> None:
    """Order with None values for optional fields."""
    raw = {
        "symbol": "ETHUSDT",
        "orderId": 2,
        "clientOrderId": None,
        "side": "SELL",
        "positionSide": None,
        "type": "LIMIT",
        "status": "NEW",
        "price": None,
        "origQty": None,
        "executedQty": None,
        "avgPrice": None,
        "reduceOnly": None,
        "stopPrice": None,
    }

    order = map_binance_order(raw)

    assert order.client_order_id is None
    assert order.price is None
    assert order.quantity is None
    assert order.filled_quantity is None
    assert order.average_price is None
    assert order.reduce_only is False
    assert order.trigger_price is None


def test_map_order_falls_back_to_quantity_field() -> None:
    """When origQty is missing, fall back to quantity field."""
    raw = {
        "symbol": "ETHUSDT",
        "orderId": 3,
        "side": "BUY",
        "type": "MARKET",
        "status": "NEW",
        "quantity": "0.750",
    }

    order = map_binance_order(raw)
    assert order.quantity == Decimal("0.750")
    assert order.quantity_unit == BrokerQuantityUnit.BASE_ASSET


def test_map_order_reduce_only_string_true() -> None:
    """reduceOnly may come as string 'true' from Binance."""
    raw = {
        "symbol": "ETHUSDT",
        "orderId": 4,
        "side": "SELL",
        "type": "LIMIT",
        "status": "NEW",
        "reduceOnly": "true",
    }

    order = map_binance_order(raw)
    assert order.reduce_only is True


def test_map_order_reduce_only_string_1() -> None:
    """reduceOnly may come as string '1' from Binance."""
    raw = {
        "symbol": "ETHUSDT",
        "orderId": 5,
        "side": "SELL",
        "type": "LIMIT",
        "status": "NEW",
        "reduceOnly": "1",
    }

    order = map_binance_order(raw)
    assert order.reduce_only is True
