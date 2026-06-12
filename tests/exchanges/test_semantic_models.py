#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/12
@File       : test_semantic_models.py
@Description: Tests for src.exchanges.semantic_models.
"""

from __future__ import annotations

from decimal import Decimal

from src.exchanges.models import (
    BrokerOrder,
    BrokerOrderSide,
    BrokerOrderStatus,
    BrokerOrderType,
    BrokerPosition,
    BrokerPositionSide,
    BrokerQuantityUnit,
    ExchangeName,
)
from src.exchanges.semantic_models import (
    BrokerSemanticAction,
    BrokerSemanticOrderRole,
    BrokerSemanticRequest,
    BrokerSemanticResult,
)


# ---------------------------------------------------------------------------
# BrokerSemanticRequest
# ---------------------------------------------------------------------------


class TestBrokerSemanticRequest:
    def test_place_reduce_only_tp(self) -> None:
        req = BrokerSemanticRequest(
            exchange=ExchangeName.OKX,
            symbol="BTC-USDT-SWAP",
            action=BrokerSemanticAction.PLACE_REDUCE_ONLY_TP,
            role=BrokerSemanticOrderRole.CORE_TP,
            side=BrokerPositionSide.LONG,
            quantity=Decimal("1"),
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
            trigger_price=Decimal("52000"),
            reduce_only=True,
        )
        assert req.action == BrokerSemanticAction.PLACE_REDUCE_ONLY_TP
        assert req.role == BrokerSemanticOrderRole.CORE_TP
        assert req.reduce_only is True
        assert req.trigger_price == Decimal("52000")

    def test_place_protective_stop(self) -> None:
        req = BrokerSemanticRequest(
            exchange=ExchangeName.OKX,
            symbol="ETH-USDT-SWAP",
            action=BrokerSemanticAction.PLACE_PROTECTIVE_STOP,
            role=BrokerSemanticOrderRole.PROTECTIVE_SL,
            side=BrokerPositionSide.SHORT,
            quantity=Decimal("2"),
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
            trigger_price=Decimal("2800"),
            reduce_only=True,
        )
        assert req.action == BrokerSemanticAction.PLACE_PROTECTIVE_STOP
        assert req.role == BrokerSemanticOrderRole.PROTECTIVE_SL
        assert req.side == BrokerPositionSide.SHORT

    def test_market_exit_runner(self) -> None:
        req = BrokerSemanticRequest(
            exchange=ExchangeName.OKX,
            symbol="BTC-USDT-SWAP",
            action=BrokerSemanticAction.MARKET_EXIT_RUNNER,
            role=BrokerSemanticOrderRole.MARKET_EXIT,
            side=BrokerPositionSide.LONG,
            quantity=Decimal("0.5"),
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
            reduce_only=True,
        )
        assert req.action == BrokerSemanticAction.MARKET_EXIT_RUNNER
        assert req.reduce_only is True

    def test_all_actions_are_distinct(self) -> None:
        values = [e.value for e in BrokerSemanticAction]
        assert len(values) == len(set(values))

    def test_all_roles_are_distinct(self) -> None:
        values = [e.value for e in BrokerSemanticOrderRole]
        assert len(values) == len(set(values))

    def test_metadata_defaults_to_empty(self) -> None:
        req = BrokerSemanticRequest(
            exchange=ExchangeName.OKX,
            symbol="BTC-USDT-SWAP",
            action=BrokerSemanticAction.FETCH_POSITION,
            role=BrokerSemanticOrderRole.UNKNOWN,
        )
        assert req.metadata == {}


# ---------------------------------------------------------------------------
# BrokerSemanticResult
# ---------------------------------------------------------------------------


class TestBrokerSemanticResult:
    def test_can_carry_broker_order(self) -> None:
        order = BrokerOrder(
            exchange=ExchangeName.OKX,
            symbol="BTC-USDT-SWAP",
            order_id="123",
            client_order_id="cid-1",
            side=BrokerOrderSide.BUY,
            position_side=BrokerPositionSide.LONG,
            order_type=BrokerOrderType.LIMIT,
            status=BrokerOrderStatus.OPEN,
            price=Decimal("50000"),
            quantity=Decimal("1"),
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
        )
        result = BrokerSemanticResult(
            exchange=ExchangeName.OKX,
            symbol="BTC-USDT-SWAP",
            action=BrokerSemanticAction.OPEN_POSITION,
            role=BrokerSemanticOrderRole.ENTRY,
            ok=True,
            message="ok",
            order=order,
        )
        assert result.ok is True
        assert result.order is order
        assert result.order.order_id == "123"

    def test_can_carry_position(self) -> None:
        pos = BrokerPosition(
            exchange=ExchangeName.OKX,
            symbol="BTC-USDT-SWAP",
            position_side=BrokerPositionSide.LONG,
            quantity=Decimal("2"),
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
        )
        result = BrokerSemanticResult(
            exchange=ExchangeName.OKX,
            symbol="BTC-USDT-SWAP",
            action=BrokerSemanticAction.FETCH_POSITION,
            role=BrokerSemanticOrderRole.UNKNOWN,
            ok=True,
            position=pos,
        )
        assert result.position is pos
        assert result.position.quantity == Decimal("2")

    def test_orders_is_tuple(self) -> None:
        result = BrokerSemanticResult(
            exchange=ExchangeName.OKX,
            symbol="BTC-USDT-SWAP",
            action=BrokerSemanticAction.FETCH_OPEN_ORDERS,
            role=BrokerSemanticOrderRole.UNKNOWN,
            ok=True,
            orders=(),
        )
        assert isinstance(result.orders, tuple)
        assert len(result.orders) == 0

    def test_orders_can_hold_multiple_orders(self) -> None:
        o1 = BrokerOrder(
            exchange=ExchangeName.OKX,
            symbol="BTC-USDT-SWAP",
            order_id="1",
            client_order_id=None,
            side=BrokerOrderSide.BUY,
            position_side=BrokerPositionSide.LONG,
            order_type=BrokerOrderType.LIMIT,
            status=BrokerOrderStatus.OPEN,
            price=Decimal("50000"),
            quantity=Decimal("1"),
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
        )
        o2 = BrokerOrder(
            exchange=ExchangeName.OKX,
            symbol="BTC-USDT-SWAP",
            order_id="2",
            client_order_id=None,
            side=BrokerOrderSide.BUY,
            position_side=BrokerPositionSide.LONG,
            order_type=BrokerOrderType.LIMIT,
            status=BrokerOrderStatus.OPEN,
            price=Decimal("51000"),
            quantity=Decimal("1"),
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
        )
        result = BrokerSemanticResult(
            exchange=ExchangeName.OKX,
            symbol="BTC-USDT-SWAP",
            action=BrokerSemanticAction.FETCH_OPEN_ORDERS,
            role=BrokerSemanticOrderRole.UNKNOWN,
            ok=True,
            orders=(o1, o2),
        )
        assert len(result.orders) == 2
        assert result.orders[0].order_id == "1"
        assert result.orders[1].order_id == "2"
