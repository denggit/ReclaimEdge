#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_binance_mapper_positions.py
@Description: Unit tests for Binance position response mapping.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.exchanges.binance.mapper import map_binance_position
from src.exchanges.models import BrokerPositionSide, BrokerQuantityUnit, ExchangeName


# ---------------------------------------------------------------------------
# Hedge mode positions
# ---------------------------------------------------------------------------

def test_map_long_hedge_position() -> None:
    """Hedge-mode LONG position maps correctly."""
    raw = {
        "symbol": "ETHUSDT",
        "positionSide": "LONG",
        "positionAmt": "0.500",
        "entryPrice": "3000.00",
        "markPrice": "3100.00",
        "unRealizedProfit": "50.00",
        "leverage": "10",
    }

    pos = map_binance_position(raw)

    assert pos is not None
    assert pos.exchange == ExchangeName.BINANCE
    assert pos.symbol == "ETHUSDT"
    assert pos.position_side == BrokerPositionSide.LONG
    assert pos.quantity == Decimal("0.500")
    assert pos.quantity_unit == BrokerQuantityUnit.BASE_ASSET
    assert pos.average_entry_price == Decimal("3000.00")
    assert pos.mark_price == Decimal("3100.00")
    assert pos.unrealized_pnl == Decimal("50.00")
    assert pos.leverage == Decimal("10")
    assert pos.raw == raw


def test_map_short_hedge_position() -> None:
    """Hedge-mode SHORT position maps correctly (quantity is abs)."""
    raw = {
        "symbol": "ETHUSDT",
        "positionSide": "SHORT",
        "positionAmt": "-0.300",
        "entryPrice": "3050.00",
        "markPrice": "3020.00",
        "unRealizedProfit": "9.00",
        "leverage": "5",
    }

    pos = map_binance_position(raw)

    assert pos is not None
    assert pos.position_side == BrokerPositionSide.SHORT
    assert pos.quantity == Decimal("0.300")  # abs(-0.300)
    assert pos.quantity_unit == BrokerQuantityUnit.BASE_ASSET
    assert pos.average_entry_price == Decimal("3050.00")


# ---------------------------------------------------------------------------
# One-way / net mode (BOTH positionSide)
# ---------------------------------------------------------------------------

def test_map_both_positive_amt_maps_to_long() -> None:
    """One-way mode with positive positionAmt -> LONG."""
    raw = {
        "symbol": "ETHUSDT",
        "positionSide": "BOTH",
        "positionAmt": "0.750",
        "entryPrice": "2900.00",
        "markPrice": "2950.00",
        "unRealizedProfit": "37.50",
        "leverage": "10",
    }

    pos = map_binance_position(raw)

    assert pos is not None
    assert pos.position_side == BrokerPositionSide.LONG
    assert pos.quantity == Decimal("0.750")
    assert pos.quantity_unit == BrokerQuantityUnit.BASE_ASSET


def test_map_both_negative_amt_maps_to_short() -> None:
    """One-way mode with negative positionAmt -> SHORT."""
    raw = {
        "symbol": "ETHUSDT",
        "positionSide": "BOTH",
        "positionAmt": "-0.200",
        "entryPrice": "3100.00",
        "markPrice": "3050.00",
        "unRealizedProfit": "10.00",
        "leverage": "5",
    }

    pos = map_binance_position(raw)

    assert pos is not None
    assert pos.position_side == BrokerPositionSide.SHORT
    assert pos.quantity == Decimal("0.200")
    assert pos.quantity_unit == BrokerQuantityUnit.BASE_ASSET


# ---------------------------------------------------------------------------
# Zero position
# ---------------------------------------------------------------------------

def test_zero_position_amt_returns_none() -> None:
    """positionAmt == 0 means no open position — must return None."""
    raw = {
        "symbol": "ETHUSDT",
        "positionSide": "LONG",
        "positionAmt": "0",
        "entryPrice": "3000.00",
        "markPrice": "3100.00",
        "unRealizedProfit": "0.00",
        "leverage": "10",
    }

    result = map_binance_position(raw)
    assert result is None


def test_zero_position_amt_decimal_returns_none() -> None:
    """positionAmt == 0 as Decimal still returns None."""
    raw = {
        "symbol": "ETHUSDT",
        "positionSide": "BOTH",
        "positionAmt": "0.000",
        "entryPrice": "3000.00",
        "markPrice": "3100.00",
        "unRealizedProfit": "0.00",
        "leverage": "10",
    }

    result = map_binance_position(raw)
    assert result is None


# ---------------------------------------------------------------------------
# Symbol guard
# ---------------------------------------------------------------------------

def test_btcusdt_position_raises_value_error() -> None:
    """BTCUSDT is not supported yet — must raise ValueError."""
    raw = {
        "symbol": "BTCUSDT",
        "positionSide": "LONG",
        "positionAmt": "0.100",
        "entryPrice": "50000.00",
        "markPrice": "51000.00",
        "unRealizedProfit": "100.00",
        "leverage": "10",
    }

    with pytest.raises(ValueError, match="Unsupported Binance symbol"):
        map_binance_position(raw)


# ---------------------------------------------------------------------------
# quantity_unit guarantee
# ---------------------------------------------------------------------------

def test_quantity_unit_is_always_base_asset() -> None:
    raw = {
        "symbol": "ETHUSDT",
        "positionSide": "LONG",
        "positionAmt": "1.000",
        "entryPrice": "3000.00",
        "markPrice": "3100.00",
        "unRealizedProfit": "100.00",
        "leverage": "10",
    }

    pos = map_binance_position(raw)

    assert pos is not None
    assert pos.quantity_unit == BrokerQuantityUnit.BASE_ASSET
    assert pos.quantity_unit != BrokerQuantityUnit.CONTRACTS
    assert pos.quantity_unit != BrokerQuantityUnit.QUOTE_ASSET


# ---------------------------------------------------------------------------
# Edge cases — missing optional fields
# ---------------------------------------------------------------------------

def test_map_position_with_minimal_fields() -> None:
    """Position with only required fields present."""
    raw = {
        "symbol": "ETHUSDT",
        "positionSide": "LONG",
        "positionAmt": "0.250",
    }

    pos = map_binance_position(raw)

    assert pos is not None
    assert pos.average_entry_price is None
    assert pos.mark_price is None
    assert pos.unrealized_pnl is None
    assert pos.leverage is None
    assert pos.quantity_unit == BrokerQuantityUnit.BASE_ASSET


def test_map_position_with_null_fields() -> None:
    """Position with None values for optional fields."""
    raw = {
        "symbol": "ETHUSDT",
        "positionSide": "SHORT",
        "positionAmt": "-0.500",
        "entryPrice": None,
        "markPrice": None,
        "unRealizedProfit": None,
        "leverage": None,
    }

    pos = map_binance_position(raw)

    assert pos is not None
    assert pos.average_entry_price is None
    assert pos.mark_price is None
    assert pos.unrealized_pnl is None
    assert pos.leverage is None


def test_map_position_unknown_position_side() -> None:
    """Unknown positionSide falls back to UNKNOWN."""
    raw = {
        "symbol": "ETHUSDT",
        "positionSide": "XYZ",
        "positionAmt": "0.100",
    }

    pos = map_binance_position(raw)
    assert pos is not None
    assert pos.position_side == BrokerPositionSide.UNKNOWN
