#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : binance_live_preflight.py
@Description: Binance live preflight — backward-compatible re-export wrapper.

The actual implementation has moved to ``src/exchanges/binance/live_preflight.py``.
This module exists only to avoid breaking existing imports.
New code should import from ``src.exchanges.binance.live_preflight`` directly.
"""

from __future__ import annotations

from src.exchanges.binance.live_preflight import (
    BINANCE_LIVE_CONFIRMATION_PHRASE,
    BINANCE_LIVE_HARD_MAX_LEVERAGE,
    BINANCE_LIVE_HARD_MAX_ORDER_NOTIONAL_USDT,
    BINANCE_LIVE_HARD_MAX_POSITION_NOTIONAL_USDT,
    BinanceLivePreflightConfig,
    BinanceLivePreflightReport,
    LIVE_CONFIRMATION_PHRASE,
    build_binance_live_preflight_report,
    format_binance_live_blocked_message,
    load_binance_live_preflight_config,
)

__all__ = [
    "LIVE_CONFIRMATION_PHRASE",
    "BINANCE_LIVE_CONFIRMATION_PHRASE",
    "BINANCE_LIVE_HARD_MAX_LEVERAGE",
    "BINANCE_LIVE_HARD_MAX_ORDER_NOTIONAL_USDT",
    "BINANCE_LIVE_HARD_MAX_POSITION_NOTIONAL_USDT",
    "BinanceLivePreflightConfig",
    "BinanceLivePreflightReport",
    "build_binance_live_preflight_report",
    "format_binance_live_blocked_message",
    "load_binance_live_preflight_config",
]
