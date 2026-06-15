#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : runtime_factory.py
@Description: Live runtime factory — creates the exchange-adapted runtime bundle.

This module wires together the exchange runtime config and exchange adapters
into a LiveRuntimeBundle.  It does NOT import or instantiate any
exchange-specific concrete class or any exchange-specific adapter module.
All exchange dispatch is delegated to the generic adapter factory.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from src.exchanges.runtime_adapter_factory import create_exchange_runtime_adapters
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
        When the exchange is not supported for live trading (e.g. Binance).
    ValueError
        When the exchange runtime configuration is invalid.
    """
    values = os.environ if env is None else env

    config = load_unified_runtime_config(values)
    adapters = create_exchange_runtime_adapters(config, values)

    return LiveRuntimeBundle(
        runtime_config=config,
        market_data_client=adapters.market_data_client,
        trading_client=adapters.trading_client,
        trader=adapters.trader,
    )
