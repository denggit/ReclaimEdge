#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : trading_client_port.py
@Description: Trading client port — the final exchange-facing trading interface.

This module defines the Protocol that every exchange trading adapter must satisfy.
It does NOT import any concrete exchange implementation or the Trader class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class BalanceSnapshot:
    asset: str
    total: Decimal
    available: Decimal | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PositionSnapshot:
    side: str | None
    qty: Decimal
    avg_entry_price: Decimal | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_position(self) -> bool:
        return self.side is not None and self.qty > 0


@dataclass(frozen=True)
class OrderSnapshot:
    order_id: str | None
    client_order_id: str | None
    side: str
    qty: Decimal
    price: Decimal | None = None
    trigger_price: Decimal | None = None
    reduce_only: bool = False
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderResult:
    ok: bool
    order_id: str | None = None
    client_order_id: str | None = None
    message: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CancelResult:
    ok: bool
    order_id: str | None = None
    client_order_id: str | None = None
    message: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderStatusSnapshot:
    order_id: str | None
    client_order_id: str | None
    status: str
    filled_qty: Decimal | None = None
    avg_fill_price: Decimal | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlgoOrderSnapshot:
    order_id: str | None
    client_order_id: str | None
    side: str | None = None
    qty: Decimal | None = None
    trigger_price: Decimal | None = None
    status: str = "OPEN"
    raw: Mapping[str, Any] = field(default_factory=dict)


class TradingClientPort(Protocol):
    async def configure_instrument(self) -> None:
        ...

    async def fetch_balance(self) -> BalanceSnapshot:
        ...

    async def fetch_position(self) -> PositionSnapshot:
        ...

    async def fetch_open_orders(self) -> list[OrderSnapshot]:
        ...

    async def fetch_order_status(
        self,
        *,
        order_id: str | None = None,
        client_order_id: str | None = None,
    ) -> OrderStatusSnapshot:
        ...

    async def fetch_open_algo_orders(self) -> tuple[AlgoOrderSnapshot, ...]:
        ...

    async def place_market_order(
        self,
        *,
        side: str,
        qty: Decimal,
        reduce_only: bool,
        client_order_id: str,
    ) -> OrderResult:
        ...

    async def place_limit_order(
        self,
        *,
        side: str,
        qty: Decimal,
        price: Decimal,
        reduce_only: bool,
        client_order_id: str,
    ) -> OrderResult:
        ...

    async def place_stop_market_order(
        self,
        *,
        side: str,
        qty: Decimal | None,
        trigger_price: Decimal,
        reduce_only: bool,
        client_order_id: str,
    ) -> OrderResult:
        ...

    async def cancel_order(
        self,
        *,
        order_id: str | None = None,
        client_order_id: str | None = None,
    ) -> CancelResult:
        ...
