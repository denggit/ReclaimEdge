#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : semantic_executor.py
@Description: Binance semantic executor adapter.

This module translates high-level broker semantic requests into generic
broker order requests or direct broker-client calls.  It does not instantiate
Trader or the exchangeʼs own broker client and is not wired into live trading
paths.

Key difference from OKX:
- Binance USD‑M Futures manages STOP_MARKET / TAKE_PROFIT_MARKET through the
  ordinary order endpoint, so there is no separate algo-order path.
- ``place_protective_stop`` maps to ``broker_client.place_order(STOP_MARKET)``
  instead of a dedicated algo stop method.
- ``cancel_protective_stop`` maps to ``broker_client.cancel_order()``.
- ``fetch_algo_orders`` always returns an empty tuple because Binance algo
  orders are indistinguishable from regular open orders.
- ``recover_open_orders`` only reads ordinary open orders.
"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import Any

from src.exchanges.errors import ExchangeError, ExchangeErrorKind
from src.exchanges.models import (
    BrokerCancelResult,
    BrokerOrder,
    BrokerOrderRequest,
    BrokerOrderResult,
    BrokerOrderSide,
    BrokerOrderType,
    BrokerPositionSide,
    BrokerQuantityUnit,
    ExchangeName,
)
from src.exchanges.semantic_models import (
    BrokerSemanticAction,
    BrokerSemanticRequest,
    BrokerSemanticResult,
)
from src.exchanges.semantics import BrokerSemanticExecutor


class BinanceBrokerSemanticExecutor(BrokerSemanticExecutor):
    """Binance implementation of the broker semantic executor port."""

    def __init__(self, broker_client: Any) -> None:
        self._broker_client = broker_client

    @property
    def exchange(self) -> ExchangeName:
        return ExchangeName.BINANCE

    async def execute(self, request: BrokerSemanticRequest) -> BrokerSemanticResult:
        if request.exchange != ExchangeName.BINANCE:
            raise _exchange_error(
                ExchangeErrorKind.UNSUPPORTED_OPERATION,
                f"BinanceBrokerSemanticExecutor cannot execute {request.exchange.value}",
            )

        if request.action in {
            BrokerSemanticAction.OPEN_POSITION,
            BrokerSemanticAction.ADD_POSITION,
            BrokerSemanticAction.SIDECAR_ENTRY,
        }:
            return await self._place_open_order(request)

        if request.action in {
            BrokerSemanticAction.PLACE_REDUCE_ONLY_TP,
            BrokerSemanticAction.SIDECAR_TP,
        }:
            return await self._place_reduce_only_tp(request)

        if request.action == BrokerSemanticAction.PLACE_PROTECTIVE_STOP:
            return await self._place_protective_stop(request)

        if request.action in {
            BrokerSemanticAction.MARKET_EXIT,
            BrokerSemanticAction.MARKET_EXIT_RUNNER,
        }:
            return await self._place_market_exit(request)

        if request.action in {
            BrokerSemanticAction.CANCEL_ORDER,
            BrokerSemanticAction.CANCEL_REDUCE_ONLY_TP,
            BrokerSemanticAction.CANCEL_PROTECTIVE_STOP,
        }:
            order_id = _require_order_id(request)
            cancel_result = await self._broker_client.cancel_order(
                request.symbol,
                order_id,
            )
            return _semantic_result_from_cancel_result(
                request=request,
                cancel_result=cancel_result,
            )

        if request.action == BrokerSemanticAction.CANCEL_ALL_OPEN_ORDERS:
            raise _exchange_error(
                ExchangeErrorKind.UNSUPPORTED_OPERATION,
                "CANCEL_ALL_OPEN_ORDERS is disabled for Binance until reduce-only identity safety is enforced",
            )

        if request.action == BrokerSemanticAction.FETCH_POSITION:
            position = await self._broker_client.fetch_position(request.symbol)
            return BrokerSemanticResult(
                exchange=request.exchange,
                symbol=request.symbol,
                action=request.action,
                role=request.role,
                ok=True,
                position=position,
                message="" if position is not None else "no_position",
            )

        if request.action == BrokerSemanticAction.FETCH_OPEN_ORDERS:
            orders = await self._broker_client.fetch_open_orders(request.symbol)
            return BrokerSemanticResult(
                exchange=request.exchange,
                symbol=request.symbol,
                action=request.action,
                role=request.role,
                ok=True,
                orders=tuple(orders),
            )

        if request.action == BrokerSemanticAction.FETCH_ALGO_ORDERS:
            return BrokerSemanticResult(
                exchange=request.exchange,
                symbol=request.symbol,
                action=request.action,
                role=request.role,
                ok=True,
                orders=(),
                message="binance_algo_orders_are_regular_open_orders",
            )

        if request.action == BrokerSemanticAction.RECOVER_OPEN_ORDERS:
            ordinary_orders = await self._broker_client.fetch_open_orders(request.symbol)
            recovered = tuple(
                _with_metadata_source(order, "ordinary")
                for order in ordinary_orders
            )
            return BrokerSemanticResult(
                exchange=request.exchange,
                symbol=request.symbol,
                action=request.action,
                role=request.role,
                ok=True,
                orders=recovered,
                message="recovered_open_orders",
            )

        raise _exchange_error(
            ExchangeErrorKind.UNSUPPORTED_OPERATION,
            f"Unsupported Binance broker semantic action: {request.action}",
        )

    async def _place_open_order(
        self,
        request: BrokerSemanticRequest,
    ) -> BrokerSemanticResult:
        position_side = _require_side(request)
        quantity = _require_quantity(request)
        order_request = BrokerOrderRequest(
            exchange=ExchangeName.BINANCE,
            symbol=request.symbol,
            side=_open_order_side(position_side),
            position_side=position_side,
            order_type=BrokerOrderType.MARKET,
            quantity=quantity,
            quantity_unit=request.quantity_unit or BrokerQuantityUnit.CONTRACTS,
            reduce_only=False,
            client_order_id=request.client_order_id,
            metadata=request.metadata,
        )
        order_result = await self._broker_client.place_order(order_request)
        return _semantic_result_from_order_result(
            request=request,
            order_result=order_result,
        )

    async def _place_reduce_only_tp(
        self,
        request: BrokerSemanticRequest,
    ) -> BrokerSemanticResult:
        position_side = _require_side(request)
        quantity = _require_quantity(request)
        price = _require_price(request)
        order_request = BrokerOrderRequest(
            exchange=ExchangeName.BINANCE,
            symbol=request.symbol,
            side=_close_order_side(position_side),
            position_side=position_side,
            order_type=BrokerOrderType.LIMIT,
            quantity=quantity,
            quantity_unit=request.quantity_unit or BrokerQuantityUnit.CONTRACTS,
            price=price,
            reduce_only=True,
            client_order_id=request.client_order_id,
            metadata=request.metadata,
        )
        order_result = await self._broker_client.place_order(order_request)
        return _semantic_result_from_order_result(
            request=request,
            order_result=order_result,
        )

    async def _place_protective_stop(
        self,
        request: BrokerSemanticRequest,
    ) -> BrokerSemanticResult:
        position_side = _require_side(request)
        quantity = _require_quantity(request)
        trigger_price = _require_trigger_price(request)
        order_request = BrokerOrderRequest(
            exchange=ExchangeName.BINANCE,
            symbol=request.symbol,
            side=_close_order_side(position_side),
            position_side=position_side,
            order_type=BrokerOrderType.STOP_MARKET,
            quantity=quantity,
            quantity_unit=request.quantity_unit or BrokerQuantityUnit.CONTRACTS,
            trigger_price=trigger_price,
            reduce_only=True,
            client_order_id=request.client_order_id,
            metadata=request.metadata,
        )
        order_result = await self._broker_client.place_order(order_request)
        return _semantic_result_from_order_result(
            request=request,
            order_result=order_result,
        )

    async def _place_market_exit(
        self,
        request: BrokerSemanticRequest,
    ) -> BrokerSemanticResult:
        position_side = _require_side(request)
        quantity = _require_quantity(request)
        order_request = BrokerOrderRequest(
            exchange=ExchangeName.BINANCE,
            symbol=request.symbol,
            side=_close_order_side(position_side),
            position_side=position_side,
            order_type=BrokerOrderType.MARKET,
            quantity=quantity,
            quantity_unit=request.quantity_unit or BrokerQuantityUnit.CONTRACTS,
            reduce_only=True,
            client_order_id=request.client_order_id,
            metadata=request.metadata,
        )
        order_result = await self._broker_client.place_order(order_request)
        return _semantic_result_from_order_result(
            request=request,
            order_result=order_result,
        )


def _require_side(request: BrokerSemanticRequest) -> BrokerPositionSide:
    if request.side is None:
        raise _exchange_error(
            ExchangeErrorKind.EXCHANGE_REJECTED,
            f"{request.action.value} requires side",
        )
    return request.side


def _require_quantity(request: BrokerSemanticRequest) -> Decimal:
    if request.quantity is None or request.quantity <= 0:
        raise _exchange_error(
            ExchangeErrorKind.INVALID_ORDER_SIZE,
            f"{request.action.value} requires positive quantity",
        )
    return request.quantity


def _require_price(request: BrokerSemanticRequest) -> Decimal:
    if request.price is None:
        raise _exchange_error(
            ExchangeErrorKind.INVALID_PRICE,
            f"{request.action.value} requires price",
        )
    return request.price


def _require_trigger_price(request: BrokerSemanticRequest) -> Decimal:
    if request.trigger_price is None:
        raise _exchange_error(
            ExchangeErrorKind.INVALID_PRICE,
            f"{request.action.value} requires trigger_price",
        )
    return request.trigger_price


def _require_order_id(request: BrokerSemanticRequest) -> str:
    if not request.order_id:
        raise _exchange_error(
            ExchangeErrorKind.EXCHANGE_REJECTED,
            f"{request.action.value} requires order_id",
        )
    return request.order_id


def _open_order_side(position_side: BrokerPositionSide) -> BrokerOrderSide:
    if position_side == BrokerPositionSide.LONG:
        return BrokerOrderSide.BUY
    if position_side == BrokerPositionSide.SHORT:
        return BrokerOrderSide.SELL
    raise _exchange_error(
        ExchangeErrorKind.UNSUPPORTED_OPERATION,
        f"Unsupported position side for open order: {position_side.value}",
    )


def _close_order_side(position_side: BrokerPositionSide) -> BrokerOrderSide:
    if position_side == BrokerPositionSide.LONG:
        return BrokerOrderSide.SELL
    if position_side == BrokerPositionSide.SHORT:
        return BrokerOrderSide.BUY
    raise _exchange_error(
        ExchangeErrorKind.UNSUPPORTED_OPERATION,
        f"Unsupported position side for close order: {position_side.value}",
    )


def _semantic_result_from_order_result(
    *,
    request: BrokerSemanticRequest,
    order_result: BrokerOrderResult,
) -> BrokerSemanticResult:
    return BrokerSemanticResult(
        exchange=request.exchange,
        symbol=request.symbol,
        action=request.action,
        role=request.role,
        ok=order_result.ok,
        message=order_result.message,
        order=order_result.order,
        order_id=order_result.order_id,
        client_order_id=order_result.client_order_id,
        raw=order_result.raw,
    )


def _semantic_result_from_cancel_result(
    *,
    request: BrokerSemanticRequest,
    cancel_result: BrokerCancelResult,
) -> BrokerSemanticResult:
    return BrokerSemanticResult(
        exchange=request.exchange,
        symbol=request.symbol,
        action=request.action,
        role=request.role,
        ok=cancel_result.ok,
        order_id=cancel_result.order_id,
        client_order_id=cancel_result.client_order_id,
        message=cancel_result.message,
        raw=cancel_result.raw,
    )


def _with_metadata_source(order: BrokerOrder, source: str) -> BrokerOrder:
    return replace(
        order,
        metadata={
            **dict(order.metadata),
            "source": source,
        },
    )


def _exchange_error(
    kind: ExchangeErrorKind,
    message: str,
) -> ExchangeError:
    return ExchangeError(
        exchange=ExchangeName.BINANCE,
        kind=kind,
        message=message,
    )


__all__ = ["BinanceBrokerSemanticExecutor"]
