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
from src.exchanges.binance.request_mapper import (
    BINANCE_ETH_CONTRACT_SIZE_BASE,
    broker_order_request_to_binance_params,
    broker_order_side_to_binance,
    broker_order_type_to_binance,
    broker_position_side_to_binance,
    broker_quantity_to_binance_base_quantity,
)

__all__ = [
    "BINANCE_ETH_CONTRACT_SIZE_BASE",
    "BINANCE_ETH_USDT_SYMBOL",
    "BinanceBrokerClient",
    "assert_binance_ethusdt_symbol",
    "broker_order_request_to_binance_params",
    "broker_order_side_to_binance",
    "broker_order_type_to_binance",
    "broker_position_side_to_binance",
    "broker_quantity_to_binance_base_quantity",
    "map_binance_error",
    "map_binance_order",
    "map_binance_order_side",
    "map_binance_order_status",
    "map_binance_order_type",
    "map_binance_position",
    "map_binance_position_side",
]
