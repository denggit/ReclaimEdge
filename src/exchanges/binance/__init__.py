#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : __init__.py
@Description: Binance exchange adapter package.

Real broker execution is deferred to a later task.
"""

from src.exchanges.binance.client import BinanceBrokerClient
from src.exchanges.binance.mapper import (
    BINANCE_ETH_USDT_SYMBOL,
    assert_binance_ethusdt_symbol,
    map_binance_error,
    map_binance_order,
    map_binance_order_side,
    map_binance_order_status,
    map_binance_order_type,
    map_binance_position,
    map_binance_position_side,
)

__all__ = [
    "BINANCE_ETH_USDT_SYMBOL",
    "BinanceBrokerClient",
    "assert_binance_ethusdt_symbol",
    "map_binance_error",
    "map_binance_order",
    "map_binance_order_side",
    "map_binance_order_status",
    "map_binance_order_type",
    "map_binance_position",
    "map_binance_position_side",
]
