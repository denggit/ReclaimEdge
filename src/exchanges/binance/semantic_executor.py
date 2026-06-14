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
- Binance USD‑M Futures rejects STOP_MARKET / TAKE_PROFIT_MARKET through the
  ordinary order endpoint (error -4120).  Protective stops MUST use the
  Algo Order API (``POST /fapi/v1/algoOrder`` with ``algoType=CONDITIONAL``).
- An optional ``algo_client`` (``BinanceAlgoOrderClient``) is injected for
  protective stop operations.  Without it, ``PLACE_PROTECTIVE_STOP`` raises
  a clear error — it never falls back to regular ``STOP_MARKET``.
- ``FETCH_ALGO_ORDERS`` returns actual open algo orders when the algo client
  is available.
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
    BrokerOrderStatus,
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

    def __init__(
        self,
        broker_client: Any,
        *,
        algo_client: Any | None = None,
    ) -> None:
        self._broker_client = broker_client
        self._algo_client = algo_client

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

        if request.action == BrokerSemanticAction.CANCEL_PROTECTIVE_STOP:
            return await self._cancel_protective_stop(request)

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
            return await self._fetch_algo_orders(request)

        if request.action == BrokerSemanticAction.RECOVER_OPEN_ORDERS:
            return await self._recover_open_orders(request)

        raise _exchange_error(
            ExchangeErrorKind.UNSUPPORTED_OPERATION,
            f"Unsupported Binance broker semantic action: {request.action}",
        )

    # ------------------------------------------------------------------
    # Open / add / TP / market exit (unchanged — regular order path)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Protective stop — Algo Order API
    # ------------------------------------------------------------------

    async def _place_protective_stop(
        self,
        request: BrokerSemanticRequest,
    ) -> BrokerSemanticResult:
        if self._algo_client is None:
            raise _exchange_error(
                ExchangeErrorKind.UNSUPPORTED_OPERATION,
                "PLACE_PROTECTIVE_STOP requires BinanceAlgoOrderClient; "
                "regular STOP_MARKET is rejected by Binance (error -4120)",
            )

        position_side = _require_side(request)
        quantity = _require_quantity(request)
        trigger_price = _require_trigger_price(request)

        # Convert contracts → base-asset quantity for algo order
        from src.exchanges.binance.request_mapper import (
            BINANCE_ETH_CONTRACT_SIZE_BASE,
        )

        base_qty = quantity * BINANCE_ETH_CONTRACT_SIZE_BASE

        side = "SELL" if position_side == BrokerPositionSide.LONG else "BUY"
        client_algo_id = request.client_order_id or ""

        result = await self._algo_client.place_stop_loss(
            symbol=request.symbol,
            side=side,
            quantity=base_qty,
            trigger_price=trigger_price,
            client_algo_id=client_algo_id,
        )

        return BrokerSemanticResult(
            exchange=request.exchange,
            symbol=request.symbol,
            action=request.action,
            role=request.role,
            ok=result.ok,
            order_id=result.order_id,
            client_order_id=result.client_order_id,
            message=result.message,
            raw=result.raw,
        )

    async def _cancel_protective_stop(
        self,
        request: BrokerSemanticRequest,
    ) -> BrokerSemanticResult:
        if self._algo_client is None:
            raise _exchange_error(
                ExchangeErrorKind.UNSUPPORTED_OPERATION,
                "CANCEL_PROTECTIVE_STOP requires BinanceAlgoOrderClient",
            )

        client_algo_id = request.order_id
        if not client_algo_id:
            raise _exchange_error(
                ExchangeErrorKind.EXCHANGE_REJECTED,
                "CANCEL_PROTECTIVE_STOP requires order_id (clientAlgoId)",
            )

        result = await self._algo_client.cancel_algo_order(
            symbol=request.symbol,
            client_algo_id=client_algo_id,
        )

        return BrokerSemanticResult(
            exchange=request.exchange,
            symbol=request.symbol,
            action=request.action,
            role=request.role,
            ok=result.ok,
            order_id=result.order_id,
            client_order_id=result.client_order_id,
            message=result.message,
            raw=result.raw,
        )

    async def _fetch_algo_orders(
        self,
        request: BrokerSemanticRequest,
    ) -> BrokerSemanticResult:
        if self._algo_client is None:
            return BrokerSemanticResult(
                exchange=request.exchange,
                symbol=request.symbol,
                action=request.action,
                role=request.role,
                ok=True,
                orders=(),
                message="binance_algo_client_not_configured",
            )

        raw_orders = await self._algo_client.fetch_open_algo_orders(
            symbol=request.symbol,
        )

        mapped: list[BrokerOrder] = []
        for raw in raw_orders:
            mapped.append(_map_algo_order_to_broker_order(raw))

        return BrokerSemanticResult(
            exchange=request.exchange,
            symbol=request.symbol,
            action=request.action,
            role=request.role,
            ok=True,
            orders=tuple(mapped),
        )

    async def _recover_open_orders(
        self,
        request: BrokerSemanticRequest,
    ) -> BrokerSemanticResult:
        ordinary_orders = await self._broker_client.fetch_open_orders(request.symbol)
        recovered = tuple(
            _with_metadata_source(order, "ordinary")
            for order in ordinary_orders
        )

        # Also include algo orders if the algo client is available
        algo_orders: tuple[BrokerOrder, ...] = ()
        if self._algo_client is not None:
            raw_algo = await self._algo_client.fetch_open_algo_orders(
                symbol=request.symbol,
            )
            algo_orders = tuple(
                _with_metadata_source(_map_algo_order_to_broker_order(raw), "algo")
                for raw in raw_algo
            )

        all_orders = recovered + algo_orders

        return BrokerSemanticResult(
            exchange=request.exchange,
            symbol=request.symbol,
            action=request.action,
            role=request.role,
            ok=True,
            orders=all_orders,
            message="recovered_open_orders",
        )


# ---------------------------------------------------------------------------
# Algo order → BrokerOrder mapper
# ---------------------------------------------------------------------------


def _map_algo_order_to_broker_order(raw: dict[str, Any]) -> BrokerOrder:
    """Map a raw Binance algo order dict into a BrokerOrder DTO."""
    from src.exchanges.binance.mapper import (
        BINANCE_ETH_USDT_SYMBOL,
        map_binance_order_side,
    )

    symbol = str(raw.get("symbol", BINANCE_ETH_USDT_SYMBOL))
    algo_id = raw.get("algoId")
    client_algo_id = raw.get("clientAlgoId")

    order_type_str = str(raw.get("orderType") or raw.get("type") or "").upper()
    if order_type_str == "STOP_MARKET":
        order_type = BrokerOrderType.STOP_MARKET
    elif order_type_str == "TAKE_PROFIT_MARKET":
        order_type = BrokerOrderType.TAKE_PROFIT_MARKET
    elif order_type_str == "MARKET":
        order_type = BrokerOrderType.MARKET
    elif order_type_str == "LIMIT":
        order_type = BrokerOrderType.LIMIT
    else:
        order_type = BrokerOrderType.UNKNOWN

    quantity = Decimal(str(raw.get("quantity") or "0"))

    return BrokerOrder(
        exchange=ExchangeName.BINANCE,
        symbol=symbol,
        order_id=str(algo_id) if algo_id is not None else None,
        client_order_id=str(client_algo_id) if client_algo_id is not None else None,
        side=map_binance_order_side(raw.get("side")),
        position_side=BrokerPositionSide.UNKNOWN,
        order_type=order_type,
        status=BrokerOrderStatus.OPEN,
        price=None,
        quantity=quantity,
        quantity_unit=BrokerQuantityUnit.BASE_ASSET,
        reduce_only=True,
        trigger_price=Decimal(str(raw.get("triggerPrice") or "0")) if raw.get("triggerPrice") else None,
        raw=dict(raw),
        metadata={"source": "algo"},
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


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
