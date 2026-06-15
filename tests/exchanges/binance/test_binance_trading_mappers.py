#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/15
@File       : test_binance_trading_mappers.py
@Description: Tests for Binance trading mappers — pure functions that map
              Binance API responses to TradingClientPort DTOs.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.exchanges.binance.trading_mappers import (
    map_binance_algo_order_to_snapshot,
    map_binance_balance_to_snapshot,
    map_binance_order_to_snapshot,
    map_binance_order_to_status_snapshot,
    map_binance_position_to_snapshot,
    map_binance_side_to_port_side,
    map_binance_status_to_port_status,
)


# ---------------------------------------------------------------------------
# side mapping
# ---------------------------------------------------------------------------


def test_map_side_buy() -> None:
    assert map_binance_side_to_port_side("BUY") == "BUY"
    assert map_binance_side_to_port_side("buy") == "BUY"
    assert map_binance_side_to_port_side("Buy") == "BUY"


def test_map_side_sell() -> None:
    assert map_binance_side_to_port_side("SELL") == "SELL"
    assert map_binance_side_to_port_side("sell") == "SELL"


def test_map_side_unknown_raises() -> None:
    with pytest.raises(ValueError):
        map_binance_side_to_port_side("LONG")
    with pytest.raises(ValueError):
        map_binance_side_to_port_side("")
    with pytest.raises(ValueError):
        map_binance_side_to_port_side(None)


# ---------------------------------------------------------------------------
# status mapping
# ---------------------------------------------------------------------------


def test_map_status_new_is_open() -> None:
    assert map_binance_status_to_port_status("NEW") == "OPEN"


def test_map_status_partially_filled_is_open() -> None:
    assert map_binance_status_to_port_status("PARTIALLY_FILLED") == "OPEN"


def test_map_status_filled() -> None:
    assert map_binance_status_to_port_status("FILLED") == "FILLED"


def test_map_status_canceled() -> None:
    assert map_binance_status_to_port_status("CANCELED") == "CANCELED"
    assert map_binance_status_to_port_status("CANCELLED") == "CANCELED"


def test_map_status_rejected() -> None:
    assert map_binance_status_to_port_status("REJECTED") == "REJECTED"


def test_map_status_expired() -> None:
    assert map_binance_status_to_port_status("EXPIRED") == "EXPIRED"


def test_map_status_unknown() -> None:
    assert map_binance_status_to_port_status("UNKNOWN_STATUS") == "UNKNOWN"
    assert map_binance_status_to_port_status("") == "UNKNOWN"
    assert map_binance_status_to_port_status(None) == "UNKNOWN"


# ---------------------------------------------------------------------------
# balance mapper
# ---------------------------------------------------------------------------


def test_map_balance_correct() -> None:
    raw = {
        "asset": "USDT",
        "balance": "1000.50",
        "crossWalletBalance": "1000.50",
        "availableBalance": "950.25",
    }
    snap = map_binance_balance_to_snapshot(raw, margin_asset="USDT")
    assert snap.asset == "USDT"
    assert snap.total == Decimal("1000.50")
    assert snap.available == Decimal("950.25")
    assert isinstance(snap.total, Decimal)
    assert isinstance(snap.available, Decimal)


def test_map_balance_zero() -> None:
    raw = {
        "asset": "USDT",
        "balance": "0",
        "availableBalance": "0",
    }
    snap = map_binance_balance_to_snapshot(raw, margin_asset="USDT")
    assert snap.total == Decimal("0")
    assert snap.available == Decimal("0")


def test_map_balance_missing_fields() -> None:
    raw: dict = {}
    snap = map_binance_balance_to_snapshot(raw, margin_asset="USDT")
    assert snap.total == Decimal("0")
    assert snap.available is None


def test_map_balance_raw_preserved() -> None:
    raw = {"asset": "USDT", "balance": "500", "availableBalance": "400"}
    snap = map_binance_balance_to_snapshot(raw, margin_asset="USDT")
    assert snap.raw == raw


# ---------------------------------------------------------------------------
# position mapper
# ---------------------------------------------------------------------------


def test_map_position_long() -> None:
    raw = {
        "symbol": "ETHUSDT",
        "positionAmt": "1.5",
        "entryPrice": "3000.00",
        "positionSide": "BOTH",
    }
    snap = map_binance_position_to_snapshot(raw)
    assert snap.side == "LONG"
    assert snap.qty == Decimal("1.5")
    assert snap.avg_entry_price == Decimal("3000.00")


def test_map_position_short() -> None:
    raw = {
        "symbol": "ETHUSDT",
        "positionAmt": "-2.0",
        "entryPrice": "3100.50",
    }
    snap = map_binance_position_to_snapshot(raw)
    assert snap.side == "SHORT"
    assert snap.qty == Decimal("2.0")
    assert snap.avg_entry_price == Decimal("3100.50")


def test_map_position_zero_no_position() -> None:
    raw = {
        "symbol": "ETHUSDT",
        "positionAmt": "0",
        "entryPrice": "3000.00",
    }
    snap = map_binance_position_to_snapshot(raw)
    assert snap.side is None
    assert snap.qty == Decimal("0")
    assert snap.avg_entry_price is None


def test_map_position_missing_entry_price() -> None:
    raw = {
        "symbol": "ETHUSDT",
        "positionAmt": "1.0",
    }
    snap = map_binance_position_to_snapshot(raw)
    assert snap.side == "LONG"
    assert snap.avg_entry_price is None


def test_map_position_integer_amt() -> None:
    raw = {
        "symbol": "ETHUSDT",
        "positionAmt": 1,
        "entryPrice": 3000,
    }
    snap = map_binance_position_to_snapshot(raw)
    assert snap.qty == Decimal("1")
    assert snap.avg_entry_price == Decimal("3000")


def test_map_position_raw_preserved() -> None:
    raw = {"symbol": "ETHUSDT", "positionAmt": "0.5"}
    snap = map_binance_position_to_snapshot(raw)
    assert snap.raw == raw


# ---------------------------------------------------------------------------
# order mapper
# ---------------------------------------------------------------------------


def test_map_order_basic() -> None:
    raw = {
        "symbol": "ETHUSDT",
        "orderId": 123456,
        "clientOrderId": "cid-001",
        "side": "BUY",
        "type": "LIMIT",
        "price": "3000.00",
        "origQty": "1.0",
        "reduceOnly": False,
        "stopPrice": "0",
    }
    snap = map_binance_order_to_snapshot(raw)
    assert snap.order_id == "123456"
    assert snap.client_order_id == "cid-001"
    assert snap.side == "BUY"
    assert snap.qty == Decimal("1.0")
    assert snap.price == Decimal("3000.00")
    assert snap.trigger_price is None
    assert snap.reduce_only is False


def test_map_order_with_trigger() -> None:
    raw = {
        "symbol": "ETHUSDT",
        "orderId": 789,
        "clientOrderId": "cid-stop",
        "side": "SELL",
        "type": "STOP_MARKET",
        "stopPrice": "2900.00",
        "origQty": "0.5",
        "reduceOnly": True,
    }
    snap = map_binance_order_to_snapshot(raw)
    assert snap.side == "SELL"
    assert snap.qty == Decimal("0.5")
    assert snap.trigger_price == Decimal("2900.00")
    assert snap.reduce_only is True


def test_map_order_reduce_only_true_string() -> None:
    raw = {
        "symbol": "ETHUSDT",
        "orderId": 1,
        "side": "SELL",
        "origQty": "1",
        "reduceOnly": "true",
    }
    snap = map_binance_order_to_snapshot(raw)
    assert snap.reduce_only is True


def test_map_order_reduce_only_false_string() -> None:
    raw = {
        "symbol": "ETHUSDT",
        "orderId": 1,
        "side": "BUY",
        "origQty": "1",
        "reduceOnly": "false",
    }
    snap = map_binance_order_to_snapshot(raw)
    assert snap.reduce_only is False


def test_map_order_missing_ids() -> None:
    raw = {"symbol": "ETHUSDT", "side": "BUY", "origQty": "1"}
    snap = map_binance_order_to_snapshot(raw)
    assert snap.order_id is None
    assert snap.client_order_id is None


def test_map_order_raw_preserved() -> None:
    raw = {"symbol": "ETHUSDT", "orderId": 1, "side": "BUY", "origQty": "1"}
    snap = map_binance_order_to_snapshot(raw)
    assert snap.raw == raw


# ---------------------------------------------------------------------------
# order status mapper
# ---------------------------------------------------------------------------


def test_map_order_status_filled() -> None:
    raw = {
        "symbol": "ETHUSDT",
        "orderId": 123,
        "clientOrderId": "cid-1",
        "status": "FILLED",
        "executedQty": "1.0",
        "avgPrice": "3000.00",
    }
    snap = map_binance_order_to_status_snapshot(raw)
    assert snap.status == "FILLED"
    assert snap.filled_qty == Decimal("1.0")
    assert snap.avg_fill_price == Decimal("3000.00")


def test_map_order_status_open() -> None:
    raw = {
        "symbol": "ETHUSDT",
        "orderId": 456,
        "status": "NEW",
    }
    snap = map_binance_order_to_status_snapshot(raw)
    assert snap.status == "OPEN"


def test_map_order_status_partial() -> None:
    raw = {
        "symbol": "ETHUSDT",
        "orderId": 789,
        "status": "PARTIALLY_FILLED",
        "executedQty": "0.5",
        "avgPrice": "2999.00",
    }
    snap = map_binance_order_to_status_snapshot(raw)
    assert snap.status == "OPEN"


def test_map_order_status_canceled() -> None:
    raw = {"symbol": "ETHUSDT", "orderId": 1, "status": "CANCELED"}
    snap = map_binance_order_to_status_snapshot(raw)
    assert snap.status == "CANCELED"


def test_map_order_status_rejected() -> None:
    raw = {"symbol": "ETHUSDT", "orderId": 1, "status": "REJECTED"}
    snap = map_binance_order_to_status_snapshot(raw)
    assert snap.status == "REJECTED"


def test_map_order_status_expired() -> None:
    raw = {"symbol": "ETHUSDT", "orderId": 1, "status": "EXPIRED"}
    snap = map_binance_order_to_status_snapshot(raw)
    assert snap.status == "EXPIRED"


def test_map_order_status_unknown() -> None:
    raw = {"symbol": "ETHUSDT", "orderId": 1, "status": "WEIRD"}
    snap = map_binance_order_to_status_snapshot(raw)
    assert snap.status == "UNKNOWN"


def test_map_order_status_missing_ids() -> None:
    raw = {"symbol": "ETHUSDT", "status": "FILLED"}
    snap = map_binance_order_to_status_snapshot(raw)
    assert snap.order_id is None
    assert snap.client_order_id is None


def test_map_order_status_raw_preserved() -> None:
    raw = {"symbol": "ETHUSDT", "orderId": 1, "status": "FILLED"}
    snap = map_binance_order_to_status_snapshot(raw)
    assert snap.raw == raw


# ---------------------------------------------------------------------------
# algo order mapper
# ---------------------------------------------------------------------------


def test_map_algo_order_stop_market() -> None:
    raw = {
        "symbol": "ETHUSDT",
        "orderId": 100,
        "clientOrderId": "algo-cid-1",
        "side": "SELL",
        "type": "STOP_MARKET",
        "stopPrice": "2800.00",
        "origQty": "0.5",
        "status": "NEW",
    }
    snap = map_binance_algo_order_to_snapshot(raw)
    assert snap.order_id == "100"
    assert snap.client_order_id == "algo-cid-1"
    assert snap.side == "SELL"
    assert snap.qty == Decimal("0.5")
    assert snap.trigger_price == Decimal("2800.00")
    assert snap.status == "OPEN"


def test_map_algo_order_take_profit() -> None:
    raw = {
        "symbol": "ETHUSDT",
        "orderId": 200,
        "clientOrderId": "tp-cid-1",
        "side": "SELL",
        "type": "TAKE_PROFIT_MARKET",
        "stopPrice": "3200.00",
        "origQty": "1.0",
        "status": "NEW",
    }
    snap = map_binance_algo_order_to_snapshot(raw)
    assert snap.trigger_price == Decimal("3200.00")
    assert snap.status == "OPEN"


def test_map_algo_order_missing_trigger() -> None:
    raw = {
        "symbol": "ETHUSDT",
        "orderId": 300,
        "side": "SELL",
        "type": "STOP_MARKET",
        "stopPrice": "0",
        "origQty": "1.0",
        "status": "NEW",
    }
    snap = map_binance_algo_order_to_snapshot(raw)
    assert snap.trigger_price is None


def test_map_algo_order_raw_preserved() -> None:
    raw = {
        "symbol": "ETHUSDT",
        "orderId": 400,
        "side": "SELL",
        "type": "STOP_MARKET",
        "stopPrice": "2800.00",
        "origQty": "1.0",
        "status": "NEW",
    }
    snap = map_binance_algo_order_to_snapshot(raw)
    assert snap.raw == raw
