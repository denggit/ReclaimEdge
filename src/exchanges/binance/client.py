#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : client.py
@Description: Binance broker adapter shell.

This class intentionally does NOT perform real network requests yet.
It only satisfies the BrokerClient port so future Binance futures support
can be added behind the same exchange-agnostic interface.
"""

from __future__ import annotations

from typing import Sequence

from src.exchanges.base import BrokerClient
from src.exchanges.binance.errors import binance_unsupported
from src.exchanges.models import (
    BrokerCancelResult,
    BrokerOrder,
    BrokerOrderRequest,
    BrokerOrderResult,
    BrokerPosition,
    ExchangeName,
)


class BinanceBrokerClient(BrokerClient):
    """Binance broker adapter shell.

    This class intentionally does NOT perform real network requests yet.
    It only satisfies the BrokerClient port so future Binance futures support
    can be added behind the same exchange-agnostic interface.
    """

    @property
    def exchange(self) -> ExchangeName:
        return ExchangeName.BINANCE

    async def place_order(self, request: BrokerOrderRequest) -> BrokerOrderResult:
        raise binance_unsupported("place_order")

    async def cancel_order(self, symbol: str, order_id: str) -> BrokerCancelResult:
        raise binance_unsupported("cancel_order")

    async def fetch_open_orders(self, symbol: str) -> Sequence[BrokerOrder]:
        raise binance_unsupported("fetch_open_orders")

    async def fetch_position(self, symbol: str) -> BrokerPosition | None:
        raise binance_unsupported("fetch_position")


__all__ = ["BinanceBrokerClient"]
