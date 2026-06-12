#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/12
@File       : capabilities.py
@Description: Static exchange capability descriptors.

These describe what an exchange *can* do without making any network calls.
They are pure data and never contain business logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from src.exchanges.models import BrokerQuantityUnit, ExchangeName


@dataclass(frozen=True)
class ExchangeCapabilities:
    exchange: ExchangeName
    supports_hedge_mode: bool
    supports_net_mode: bool
    supports_reduce_only: bool
    supports_conditional_orders: bool
    supports_close_position: bool
    supports_client_order_id: bool
    default_quantity_unit: BrokerQuantityUnit
    metadata: Mapping[str, Any] = field(default_factory=dict)


def okx_default_capabilities() -> ExchangeCapabilities:
    """Return the static capabilities for OKX (swap / futures)."""
    return ExchangeCapabilities(
        exchange=ExchangeName.OKX,
        supports_hedge_mode=True,
        supports_net_mode=True,
        supports_reduce_only=True,
        supports_conditional_orders=True,
        supports_close_position=False,
        supports_client_order_id=True,
        default_quantity_unit=BrokerQuantityUnit.CONTRACTS,
    )


def binance_usdm_default_capabilities() -> ExchangeCapabilities:
    """Return the static capabilities for Binance USDⓈ-M futures."""
    return ExchangeCapabilities(
        exchange=ExchangeName.BINANCE,
        supports_hedge_mode=True,
        supports_net_mode=True,
        supports_reduce_only=True,
        supports_conditional_orders=True,
        supports_close_position=True,
        supports_client_order_id=True,
        default_quantity_unit=BrokerQuantityUnit.BASE_ASSET,
    )
