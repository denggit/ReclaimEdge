#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_okx_mapper.py
@Description: Tests for pure OKX raw-to-broker DTO mappers.
"""

from __future__ import annotations

from dataclasses import fields
from decimal import Decimal
from pathlib import Path

import pytest

from src.exchanges.models import (
    BrokerOrder,
    BrokerOrderSide,
    BrokerOrderStatus,
    BrokerOrderType,
    BrokerPositionSide,
    BrokerQuantityUnit,
    ExchangeName,
)
from src.exchanges.okx.mapper import (
    broker_balance_from_okx_balance_detail,
    broker_order_from_okx_pending_algo_order,
    broker_order_from_okx_pending_order,
    broker_position_from_okx_position,
)


def test_pending_order_maps_okx_fields_to_broker_order() -> None:
    raw = {
        "instId": "ETH-USDT-SWAP",
        "ordId": "123",
        "clOrdId": "client-123",
        "side": "sell",
        "posSide": "long",
        "ordType": "limit",
        "state": "live",
        "px": "3500.1",
        "sz": "12",
        "accFillSz": "3",
        "avgPx": "3499.5",
        "reduceOnly": "true",
    }

    order = broker_order_from_okx_pending_order(raw)

    assert order.exchange == ExchangeName.OKX
    assert order.symbol == "ETH-USDT-SWAP"
    assert order.order_id == "123"
    assert order.client_order_id == "client-123"
    assert order.side == BrokerOrderSide.SELL
    assert order.position_side == BrokerPositionSide.LONG
    assert order.order_type == BrokerOrderType.LIMIT
    assert order.status == BrokerOrderStatus.OPEN
    assert order.price == Decimal("3500.1")
    assert order.quantity == Decimal("12")
    assert order.filled_quantity == Decimal("3")
    assert order.average_price == Decimal("3499.5")
    assert order.reduce_only is True
    assert order.quantity_unit == BrokerQuantityUnit.CONTRACTS
    assert order.raw == raw
    assert order.metadata["source"] == "ordinary"


def test_pending_algo_order_maps_protective_sl_to_stop_market() -> None:
    raw = {
        "instId": "ETH-USDT-SWAP",
        "algoId": "algo-1",
        "algoClOrdId": "algo-client-1",
        "side": "sell",
        "posSide": "long",
        "ordType": "conditional",
        "sz": "12",
        "slTriggerPx": "3450.5",
        "slOrdPx": "-1",
        "reduceOnly": "true",
    }

    order = broker_order_from_okx_pending_algo_order(raw)

    assert order.exchange == ExchangeName.OKX
    assert order.order_id == "algo-1"
    assert order.client_order_id == "algo-client-1"
    assert order.order_type == BrokerOrderType.STOP_MARKET
    assert order.status == BrokerOrderStatus.OPEN
    assert order.trigger_price == Decimal("3450.5")
    assert order.quantity == Decimal("12")
    assert order.reduce_only is True
    assert order.metadata["source"] == "algo"
    assert order.metadata["okx_order_id_field"] == "algoId"
    assert order.raw == raw


def test_pending_algo_order_falls_back_to_ord_id_when_algo_id_missing() -> None:
    raw = {
        "instId": "ETH-USDT-SWAP",
        "ordId": "fallback-ord",
        "side": "buy",
        "posSide": "short",
        "ordType": "conditional",
        "sz": "2",
        "slTriggerPx": "3600",
        "slOrdPx": "-1",
    }

    order = broker_order_from_okx_pending_algo_order(raw)

    assert order.order_id == "fallback-ord"
    assert order.metadata["okx_order_id_field"] == "ordId"


def test_position_maps_okx_fields_to_broker_position() -> None:
    raw = {
        "instId": "ETH-USDT-SWAP",
        "posSide": "short",
        "pos": "-7",
        "avgPx": "3500",
        "markPx": "3490",
        "upl": "12.5",
        "lever": "20",
    }

    position = broker_position_from_okx_position(raw)

    assert position.exchange == ExchangeName.OKX
    assert position.symbol == "ETH-USDT-SWAP"
    assert position.position_side == BrokerPositionSide.SHORT
    assert position.quantity == Decimal("7")
    assert position.quantity_unit == BrokerQuantityUnit.CONTRACTS
    assert position.average_entry_price == Decimal("3500")
    assert position.mark_price == Decimal("3490")
    assert position.unrealized_pnl == Decimal("12.5")
    assert position.leverage == Decimal("20")
    assert position.raw == raw
    assert position.metadata["source"] == "position"


@pytest.mark.parametrize(
    ("pos", "expected_side"),
    [
        ("5", BrokerPositionSide.LONG),
        ("-5", BrokerPositionSide.SHORT),
        ("0", BrokerPositionSide.NET),
    ],
)
def test_position_side_is_inferred_from_pos_when_pos_side_missing(
    pos: str,
    expected_side: BrokerPositionSide,
) -> None:
    raw = {
        "instId": "ETH-USDT-SWAP",
        "pos": pos,
    }

    position = broker_position_from_okx_position(raw)

    assert position.position_side == expected_side


def test_balance_detail_maps_okx_fields_to_broker_balance() -> None:
    raw = {
        "ccy": "USDT",
        "eq": "1000.5",
        "availEq": "900.1",
        "frozenBal": "50",
    }

    balance = broker_balance_from_okx_balance_detail(raw)

    assert balance.exchange == ExchangeName.OKX
    assert balance.asset == "USDT"
    assert balance.total == Decimal("1000.5")
    assert balance.available == Decimal("900.1")
    assert balance.frozen == Decimal("50")
    assert balance.raw == raw


def test_invalid_decimal_values_do_not_raise_and_map_to_none() -> None:
    raw = {
        "instId": "ETH-USDT-SWAP",
        "ordId": "123",
        "side": "sell",
        "posSide": "long",
        "ordType": "limit",
        "state": "live",
        "px": "bad-price",
        "sz": "bad-size",
    }

    order = broker_order_from_okx_pending_order(raw)

    assert order.price is None
    assert order.quantity is None


def test_broker_order_does_not_expose_okx_private_fields() -> None:
    field_names = {field.name for field in fields(BrokerOrder)}

    assert "ordId" not in field_names
    assert "algoId" not in field_names
    assert "instId" not in field_names
    assert "posSide" not in field_names
    assert "tdMode" not in field_names
    assert "slTriggerPx" not in field_names
    assert "slOrdPx" not in field_names


def test_okx_mapper_functions_are_not_wired_into_live_paths() -> None:
    file_names = [
        "scripts/run_boll_cvd_live.py",
        "src/execution/trader.py",
        "src/execution/tp_sl_execution_manager.py",
        "src/live/workers/execution_command_processor.py",
    ]
    forbidden_symbols = [
        "broker_order_from_okx_pending_order",
        "broker_order_from_okx_pending_algo_order",
        "broker_position_from_okx_position",
    ]

    for file_name in file_names:
        path = Path(file_name)
        if not path.exists():
            continue
        text = path.read_text()
        for symbol in forbidden_symbols:
            assert symbol not in text
