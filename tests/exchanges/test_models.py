#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/12
@File       : test_models.py
@Description: Tests for src.exchanges.models – generic broker DTOs.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from src.exchanges.models import (
    BrokerCancelResult,
    BrokerMarketType,
    BrokerOrder,
    BrokerOrderRequest,
    BrokerOrderResult,
    BrokerOrderSide,
    BrokerOrderStatus,
    BrokerOrderType,
    BrokerPosition,
    BrokerPositionSide,
    BrokerQuantityUnit,
    BrokerSymbol,
    ExchangeName,
)


# ---------------------------------------------------------------------------
# BrokerOrder
# ---------------------------------------------------------------------------


class TestBrokerOrder:
    def test_can_be_constructed(self) -> None:
        order = BrokerOrder(
            exchange=ExchangeName.OKX,
            symbol="BTC-USDT-SWAP",
            order_id="12345",
            client_order_id="cid-1",
            side=BrokerOrderSide.BUY,
            position_side=BrokerPositionSide.LONG,
            order_type=BrokerOrderType.LIMIT,
            status=BrokerOrderStatus.OPEN,
            price=Decimal("50000.00"),
            quantity=Decimal("1.0"),
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
        )
        assert order.exchange == ExchangeName.OKX
        assert order.symbol == "BTC-USDT-SWAP"
        assert order.order_id == "12345"
        assert order.side == BrokerOrderSide.BUY
        assert order.position_side == BrokerPositionSide.LONG
        assert order.price == Decimal("50000.00")
        assert order.quantity == Decimal("1.0")

    def test_is_frozen(self) -> None:
        order = BrokerOrder(
            exchange=ExchangeName.OKX,
            symbol="BTC-USDT-SWAP",
            order_id="12345",
            client_order_id="cid-1",
            side=BrokerOrderSide.SELL,
            position_side=BrokerPositionSide.SHORT,
            order_type=BrokerOrderType.MARKET,
            status=BrokerOrderStatus.FILLED,
            price=Decimal("50000.00"),
            quantity=Decimal("1.0"),
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
        )
        with pytest.raises(FrozenInstanceError):
            order.order_id = "99999"  # type: ignore[misc]

    def test_raw_stores_exchange_fields_but_generic_fields_are_clean(self) -> None:
        """OKX-specific fields must live in raw, not as first-class attrs."""
        order = BrokerOrder(
            exchange=ExchangeName.OKX,
            symbol="ETH-USDT-SWAP",
            order_id="order-1",
            client_order_id="cid-1",
            side=BrokerOrderSide.SELL,
            position_side=BrokerPositionSide.SHORT,
            order_type=BrokerOrderType.LIMIT,
            status=BrokerOrderStatus.OPEN,
            price=Decimal("3000.00"),
            quantity=Decimal("2.0"),
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
            raw={
                "ordId": "okx-ord-123",
                "algoId": "okx-algo-456",
                "instId": "ETH-USDT-SWAP",
                "posSide": "short",
                "tdMode": "cross",
            },
        )
        # raw carries the exchange-private data
        assert order.raw["ordId"] == "okx-ord-123"
        assert order.raw["algoId"] == "okx-algo-456"
        # but the generic model does NOT have ordId / algoId / instId attrs
        assert not hasattr(order, "ordId")
        assert not hasattr(order, "algoId")
        assert not hasattr(order, "instId")
        assert not hasattr(order, "posSide")
        assert not hasattr(order, "tdMode")

    def test_defaults(self) -> None:
        order = BrokerOrder(
            exchange=ExchangeName.BINANCE,
            symbol="BTCUSDT",
            order_id=None,
            client_order_id=None,
            side=BrokerOrderSide.BUY,
            position_side=BrokerPositionSide.LONG,
            order_type=BrokerOrderType.MARKET,
            status=BrokerOrderStatus.UNKNOWN,
            price=None,
            quantity=None,
            quantity_unit=None,
        )
        assert order.filled_quantity is None
        assert order.average_price is None
        assert order.reduce_only is False
        assert order.trigger_price is None
        assert order.raw == {}
        assert order.metadata == {}

    def test_raw_and_metadata_default_to_empty_dict(self) -> None:
        order = BrokerOrder(
            exchange=ExchangeName.OKX,
            symbol="BTC-USDT-SWAP",
            order_id=None,
            client_order_id=None,
            side=BrokerOrderSide.BUY,
            position_side=BrokerPositionSide.LONG,
            order_type=BrokerOrderType.MARKET,
            status=BrokerOrderStatus.UNKNOWN,
            price=None,
            quantity=None,
            quantity_unit=None,
        )
        assert isinstance(order.raw, dict)
        assert len(order.raw) == 0
        assert isinstance(order.metadata, dict)
        assert len(order.metadata) == 0


# ---------------------------------------------------------------------------
# BrokerOrderRequest
# ---------------------------------------------------------------------------


class TestBrokerOrderRequest:
    def test_quantity_unit_contracts(self) -> None:
        req = BrokerOrderRequest(
            exchange=ExchangeName.OKX,
            symbol="BTC-USDT-SWAP",
            side=BrokerOrderSide.BUY,
            position_side=BrokerPositionSide.LONG,
            order_type=BrokerOrderType.LIMIT,
            quantity=Decimal("1"),
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
            price=Decimal("50000"),
        )
        assert req.quantity_unit == BrokerQuantityUnit.CONTRACTS

    def test_quantity_unit_base_asset(self) -> None:
        req = BrokerOrderRequest(
            exchange=ExchangeName.BINANCE,
            symbol="BTCUSDT",
            side=BrokerOrderSide.BUY,
            position_side=BrokerPositionSide.LONG,
            order_type=BrokerOrderType.LIMIT,
            quantity=Decimal("0.01"),
            quantity_unit=BrokerQuantityUnit.BASE_ASSET,
            price=Decimal("50000"),
        )
        assert req.quantity_unit == BrokerQuantityUnit.BASE_ASSET

    def test_reduce_only_and_close_position_are_independent(self) -> None:
        req = BrokerOrderRequest(
            exchange=ExchangeName.BINANCE,
            symbol="BTCUSDT",
            side=BrokerOrderSide.SELL,
            position_side=BrokerPositionSide.LONG,
            order_type=BrokerOrderType.MARKET,
            quantity=Decimal("0.01"),
            quantity_unit=BrokerQuantityUnit.BASE_ASSET,
            reduce_only=True,
            close_position=True,
        )
        assert req.reduce_only is True
        assert req.close_position is True

    def test_no_okx_private_fields(self) -> None:
        """BrokerOrderRequest must NOT expose tdMode / instId / algoClOrdId."""
        req = BrokerOrderRequest(
            exchange=ExchangeName.OKX,
            symbol="BTC-USDT-SWAP",
            side=BrokerOrderSide.BUY,
            position_side=BrokerPositionSide.LONG,
            order_type=BrokerOrderType.LIMIT,
            quantity=Decimal("1"),
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
        )
        assert not hasattr(req, "tdMode")
        assert not hasattr(req, "instId")
        assert not hasattr(req, "algoClOrdId")


# ---------------------------------------------------------------------------
# BrokerPosition
# ---------------------------------------------------------------------------


class TestBrokerPosition:
    def test_long_position(self) -> None:
        pos = BrokerPosition(
            exchange=ExchangeName.OKX,
            symbol="BTC-USDT-SWAP",
            position_side=BrokerPositionSide.LONG,
            quantity=Decimal("2"),
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
            average_entry_price=Decimal("49500"),
        )
        assert pos.position_side == BrokerPositionSide.LONG
        assert pos.quantity == Decimal("2")

    def test_short_position(self) -> None:
        pos = BrokerPosition(
            exchange=ExchangeName.OKX,
            symbol="BTC-USDT-SWAP",
            position_side=BrokerPositionSide.SHORT,
            quantity=Decimal("3"),
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
            average_entry_price=Decimal("51000"),
        )
        assert pos.position_side == BrokerPositionSide.SHORT
        assert pos.quantity == Decimal("3")

    def test_net_position(self) -> None:
        pos = BrokerPosition(
            exchange=ExchangeName.BINANCE,
            symbol="BTCUSDT",
            position_side=BrokerPositionSide.NET,
            quantity=Decimal("1.5"),
            quantity_unit=BrokerQuantityUnit.BASE_ASSET,
        )
        assert pos.position_side == BrokerPositionSide.NET


# ---------------------------------------------------------------------------
# BrokerOrderResult / BrokerCancelResult
# ---------------------------------------------------------------------------


class TestBrokerOrderResult:
    def test_success_result(self) -> None:
        result = BrokerOrderResult(
            exchange=ExchangeName.OKX,
            symbol="BTC-USDT-SWAP",
            ok=True,
            order_id="okx-123",
            message="filled",
        )
        assert result.ok is True
        assert result.order_id == "okx-123"

    def test_failure_result(self) -> None:
        result = BrokerOrderResult(
            exchange=ExchangeName.OKX,
            symbol="BTC-USDT-SWAP",
            ok=False,
            message="insufficient margin",
            raw={"code": "1", "msg": "insufficient margin"},
        )
        assert result.ok is False
        assert result.order_id is None
        assert "insufficient margin" in result.message


class TestBrokerCancelResult:
    def test_success(self) -> None:
        result = BrokerCancelResult(
            exchange=ExchangeName.OKX,
            symbol="BTC-USDT-SWAP",
            ok=True,
            order_id="okx-456",
        )
        assert result.ok is True
        assert result.order_id == "okx-456"

    def test_defaults(self) -> None:
        result = BrokerCancelResult(
            exchange=ExchangeName.OKX,
            symbol="BTC-USDT-SWAP",
            ok=False,
        )
        assert result.message == ""
        assert result.raw == {}


# ---------------------------------------------------------------------------
# BrokerSymbol
# ---------------------------------------------------------------------------


class TestBrokerSymbol:
    def test_swap_symbol(self) -> None:
        sym = BrokerSymbol(
            exchange=ExchangeName.OKX,
            raw_symbol="BTC-USDT-SWAP",
            base_asset="BTC",
            quote_asset="USDT",
            market_type=BrokerMarketType.SWAP,
        )
        assert sym.base_asset == "BTC"
        assert sym.quote_asset == "USDT"
