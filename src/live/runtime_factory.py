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
        return _create_okx_bundle(config, values)

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


def _create_okx_bundle(config, values: Mapping[str, str]) -> LiveRuntimeBundle:
    """Wire the OKX live runtime bundle.

    Creates:
    - OkxPrivateClient (private REST client with signing)
    - PrivateWriteRateLimiter (rate limiter for private writes)
    - Trader (execution facade) with injected runtime settings
    - OkxTradingClient (trading adapter, owns OkxPrivateClient)
    - OkxBrokerClient + OkxBrokerSemanticExecutor (legacy broker path)
    - OkxMarketDataClient (market data adapter, standalone)
    - Calls trader.bind_*() to wire all adapters
    """
    from config.env_loader import OKX_CONFIG
    from src.data_feed.okx_market_data_client import (
        OkxMarketDataClient,
        OkxMarketDataClientConfig,
    )
    from src.execution.okx_private_client import (
        OkxPrivateClient,
        OkxPrivateClientConfig,
        PrivateWriteRateLimiter,
    )
    from src.execution.okx_trading_client import OkxTradingClient
    from src.execution.trader import Trader, TraderRuntimeSettings
    from src.exchanges.okx.client import OkxBrokerClient
    from src.exchanges.okx.semantic_executor import OkxBrokerSemanticExecutor

    # --- credential validation (moved from Trader.__init__) ---
    api_key = OKX_CONFIG.get("api_key", "")
    secret_key = OKX_CONFIG.get("secret_key", "")
    passphrase = OKX_CONFIG.get("passphrase", "")
    if not api_key or not secret_key or not passphrase:
        raise ValueError(
            "OKX API config is incomplete. "
            "Check EXCHANGE_API_KEY/EXCHANGE_API_SECRET/EXCHANGE_API_PASSPHRASE "
            "or legacy OKX_API_KEY/OKX_SECRET_KEY/OKX_PASSPHASE."
        )

    # --- private REST client (owned by OkxTradingClient) ---
    private_client = OkxPrivateClient(
        OkxPrivateClientConfig(
            base_url=values.get("OKX_BASE_URL", "https://www.okx.com"),
            api_key=api_key,
            secret_key=secret_key,
            passphrase=passphrase,
            timeout_seconds=float(values.get("OKX_PRIVATE_REST_TIMEOUT_SECONDS", "10")),
        )
    )
    rate_limiter = PrivateWriteRateLimiter()

    # --- trader (execution facade) ---
    settings = TraderRuntimeSettings(
        symbol=config.okx_inst_id,
        base_url=values.get("OKX_BASE_URL", "https://www.okx.com"),
        td_mode=config.margin_mode,
        pos_side_mode=config.position_mode,
        leverage=values.get("LEVERAGE", "50"),
        live_trading=True,
        max_live_equity_usdt=float(values.get("MAX_LIVE_EQUITY_USDT", "30")),
    )
    trader = Trader(settings=settings)

    # --- trading client (owns private client) ---
    trading_client = OkxTradingClient(
        trader,
        private_client=private_client,
        rate_limiter=rate_limiter,
    )
    trader.bind_trading_client(trading_client)

    # --- broker semantic executor (legacy, behind feature flags) ---
    broker_client = OkxBrokerClient(
        trader,
        private_client=private_client,
        rate_limiter=rate_limiter,
    )
    broker_semantic_executor = OkxBrokerSemanticExecutor(broker_client)
    trader.bind_broker_semantic_executor(broker_semantic_executor)

    # --- market data client (standalone) ---
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
