#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : runtime_config.py
@Description: Canonical exchange runtime configuration loaded from environment.

This module is NOT wired into live trading paths.  It provides a single
configuration object that downstream code may consume without touching
os.environ directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping

from src.exchanges.models import ExchangeName


_TRUE_VALUES = {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class ExchangeRuntimeConfig:
    """Canonical per-process exchange runtime configuration.

    All secrets use ``repr=False`` so that accidental logging never leaks them.
    """

    exchange: ExchangeName
    trade_asset: str
    quote_asset: str
    market_type: str
    api_key: str = field(repr=False)
    api_secret: str = field(repr=False)
    api_passphrase: str = field(default="", repr=False)
    leverage: int = 20
    margin_mode: str = "isolated"
    position_mode: str = "hedge"

    # -- derived properties ---------------------------------------------------

    @property
    def canonical_symbol(self) -> str:
        """Return the canonical trading symbol, e.g. ``ETH-USDT-PERP``."""
        return f"{self.trade_asset}-{self.quote_asset}-PERP"

    @property
    def is_okx(self) -> bool:
        return self.exchange == ExchangeName.OKX

    @property
    def is_binance(self) -> bool:
        return self.exchange == ExchangeName.BINANCE


def load_exchange_runtime_config_from_env(
    env: Mapping[str, str] | None = None,
) -> ExchangeRuntimeConfig:
    """Build an :class:`ExchangeRuntimeConfig` from environment variables.

    Parameters
    ----------
    env:
        An optional mapping used instead of ``os.environ`` (useful in tests).

    Returns
    -------
    ExchangeRuntimeConfig

    Raises
    ------
    ValueError
        If a value is invalid (unsupported exchange, non-perpetual market type,
        non-integer leverage, etc.).
    """
    values = os.environ if env is None else env

    # -- exchange -------------------------------------------------------------

    exchange_raw = str(values.get("EXCHANGE", "okx")).strip().lower()
    try:
        exchange = ExchangeName(exchange_raw)
    except ValueError as exc:
        raise ValueError(f"Unsupported EXCHANGE: {exchange_raw}") from exc

    # -- trading pair ---------------------------------------------------------

    trade_asset = str(values.get("TRADE_ASSET", "ETH")).strip().upper()
    quote_asset = str(values.get("QUOTE_ASSET", "USDT")).strip().upper()
    market_type = str(values.get("MARKET_TYPE", "PERPETUAL")).strip().upper()

    if not trade_asset:
        raise ValueError("TRADE_ASSET must not be empty")
    if not quote_asset:
        raise ValueError("QUOTE_ASSET must not be empty")
    if market_type != "PERPETUAL":
        raise ValueError(f"Unsupported MARKET_TYPE: {market_type}")

    # -- leverage / margin / position -----------------------------------------

    leverage_raw = str(values.get("LEVERAGE", "20")).strip()
    try:
        leverage = int(leverage_raw)
    except ValueError as exc:
        raise ValueError(f"LEVERAGE must be an integer: {leverage_raw}") from exc
    if leverage <= 0:
        raise ValueError("LEVERAGE must be positive")

    margin_mode = str(values.get("MARGIN_MODE", "isolated")).strip().lower()
    if margin_mode not in {"isolated", "cross"}:
        raise ValueError(f"Unsupported MARGIN_MODE: {margin_mode}")

    position_mode = str(values.get("POSITION_MODE", "hedge")).strip().lower()
    if position_mode not in {"hedge", "net"}:
        raise ValueError(f"Unsupported POSITION_MODE: {position_mode}")

    # -- credentials ----------------------------------------------------------

    return ExchangeRuntimeConfig(
        exchange=exchange,
        trade_asset=trade_asset,
        quote_asset=quote_asset,
        market_type=market_type,
        api_key=str(values.get("EXCHANGE_API_KEY", "")),
        api_secret=str(values.get("EXCHANGE_API_SECRET", "")),
        api_passphrase=str(values.get("EXCHANGE_API_PASSPHRASE", "")),
        leverage=leverage,
        margin_mode=margin_mode,
        position_mode=position_mode,
    )
