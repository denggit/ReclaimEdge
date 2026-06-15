#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/15
@File       : runtime_adapter.py
@Description: Binance runtime adapter — Binance live trading is blocked by build.

This module exists so that the generic exchange adapter factory can dispatch
to it.  It always raises ``RuntimeError`` because Binance live trading has
not been wired yet.
"""

from __future__ import annotations

from collections.abc import Mapping

from src.exchanges.binance.live_preflight import (
    build_binance_live_preflight_report,
    format_binance_live_blocked_message,
)
from src.exchanges.runtime_adapters import ExchangeRuntimeAdapters
from src.exchanges.runtime_config import ExchangeRuntimeConfig


def create_binance_runtime_adapters(
    config: ExchangeRuntimeConfig,
    env: Mapping[str, str],
) -> ExchangeRuntimeAdapters:
    """Binance live trading is explicitly blocked — always raises RuntimeError.

    Parameters
    ----------
    config:
        Canonical exchange runtime configuration.
    env:
        Environment variable mapping.

    Returns
    -------
    ExchangeRuntimeAdapters
        Never returns normally.

    Raises
    ------
    RuntimeError
        Always — Binance live trading is not wired.
    """
    report = build_binance_live_preflight_report(
        env,
        orders_globally_enabled=False,
    )
    raise RuntimeError(format_binance_live_blocked_message(report))
