#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : runtime_factory.py
@Description: Live runtime factory — creates the exchange-adapted runtime bundle.

This module wires together the exchange runtime config, market data client,
trading client, and trader for the configured exchange.

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
        return _create_okx_bundle(config)

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


# ======================================================================
# OKX bundle wiring
# ======================================================================


def _create_okx_bundle(config) -> LiveRuntimeBundle:
    """Wire the OKX live runtime bundle.

    Creates:
    - Trader (execution facade)
    - OkxTradingClient (trading adapter, wraps Trader)
    - OkxMarketDataClient (market data adapter, standalone)
    """
    from src.data_feed.okx_market_data_client import (
        OkxMarketDataClient,
        OkxMarketDataClientConfig,
    )
    from src.execution.okx_trading_client import OkxTradingClient
    from src.execution.trader import Trader

    trader = Trader()
    trading_client = OkxTradingClient(trader)

    market_data_config = OkxMarketDataClientConfig(
        inst_id=config.okx_inst_id,
        bar=config.kline_interval,
    )
    market_data_client = OkxMarketDataClient(market_data_config)

    return LiveRuntimeBundle(
        runtime_config=config,
        market_data_client=market_data_client,
        trading_client=trading_client,
        trader=trader,
    )
