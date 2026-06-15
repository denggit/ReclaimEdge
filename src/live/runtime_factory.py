#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : runtime_factory.py
@Description: Live runtime factory — creates the exchange-adapted runtime bundle.

This module wires together the exchange runtime config, market data client,
trading client, and trader for the configured exchange.

It does NOT import or instantiate any exchange-specific concrete class.
All exchange-specific wiring is delegated to the respective exchange adapter
(e.g. ``src/exchanges/okx/runtime_adapter.py``).

Currently only OKX is supported for live trading.
Binance live trading is explicitly blocked.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from src.exchanges.models import ExchangeName
from src.exchanges.runtime_config import load_unified_runtime_config
from src.live.runtime_bundle import LiveRuntimeBundle


def create_runtime_bundle(
    env: Mapping[str, str] | None = None,
) -> LiveRuntimeBundle:
    """Create a fully-wired live runtime bundle for the configured exchange.

    Parameters
    ----------
    env:
        Optional mapping of environment variables.  When ``None`` (the
        default) the real ``os.environ`` is used.

    Returns
    -------
    LiveRuntimeBundle
        A frozen bundle containing config, market_data_client, trading_client,
        and trader.

    Raises
    ------
    RuntimeError
        When ``EXCHANGE=binance`` — Binance live trading is not wired yet.
    ValueError
        When ``EXCHANGE`` is set to an unsupported value or configuration is
        invalid.
    """
    values = os.environ if env is None else env

    config = load_unified_runtime_config(values)

    if config.exchange == ExchangeName.OKX:
        from src.exchanges.okx.runtime_adapter import create_okx_runtime_bundle
        return create_okx_runtime_bundle(config, values)

    if config.exchange == ExchangeName.BINANCE:
        from src.live.binance_live_preflight import (
            build_binance_live_preflight_report,
            format_binance_live_blocked_message,
        )

        report = build_binance_live_preflight_report(
            values,
            orders_globally_enabled=False,
        )
        raise RuntimeError(format_binance_live_blocked_message(report))

    raise RuntimeError(f"Unsupported exchange for live runtime: {config.exchange}")
