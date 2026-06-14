#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : runtime_bundle.py
@Description: Live runtime bundle — holds all exchange-adapted infrastructure.

This module defines the ``LiveRuntimeBundle`` dataclass that groups together
the exchange runtime config, market data client, trading client, and trader
instance for a live trading session.

It does NOT import any concrete exchange implementation, strategy, or risk module.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.data_feed.market_data_client_port import MarketDataClientPort
from src.execution.trading_client_port import TradingClientPort
from src.exchanges.runtime_config import ExchangeRuntimeConfig


@dataclass(frozen=True)
class LiveRuntimeBundle:
    """Frozen bundle of all exchange-adapted infrastructure for live trading.

    Attributes:
        runtime_config: Canonical exchange runtime configuration.
        market_data_client: Exchange market data adapter.
        trading_client: Exchange trading adapter.
        trader: Live trader instance (execution facade).
    """

    runtime_config: ExchangeRuntimeConfig
    market_data_client: MarketDataClientPort
    trading_client: TradingClientPort
    trader: object  # Trader — avoids circular import
