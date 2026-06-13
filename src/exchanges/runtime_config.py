#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : runtime_config.py
@Description: Unified exchange runtime configuration loaded from environment.

This module provides a single configuration object for both OKX and Binance.
The only differences between platforms are ``EXCHANGE`` and the API credentials;
all other parameters (trade asset, quote asset, market type, margin mode,
position mode, leverage, kline interval) are identical.

Legacy OKX-specific environment variables (OKX_TD_MODE, OKX_POS_SIDE_MODE,
OKX_INST_ID, OKX_BAR) are NOT consumed by this module.  They remain available
for backward-compatible raw config paths only.

This module is NOT wired into live trading paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping

from src.exchanges.models import ExchangeName


# ---------------------------------------------------------------------------
# Supported-value constants
# ---------------------------------------------------------------------------

SUPPORTED_TRADE_ASSET = "ETH"
SUPPORTED_QUOTE_ASSET = "USDT"
SUPPORTED_MARKET_TYPE = "PERPETUAL"
SUPPORTED_MARGIN_MODE = "isolated"
SUPPORTED_POSITION_MODE = "net"
SUPPORTED_LEVERAGE = 20
SUPPORTED_KLINE_INTERVAL = "15m"
SUPPORTED_CANONICAL_SYMBOL = "ETH-USDT-PERP"
SUPPORTED_OKX_INST_ID = "ETH-USDT-SWAP"
SUPPORTED_BINANCE_SYMBOL = "ETHUSDT"


@dataclass(frozen=True)
class ExchangeRuntimeConfig:
    """Canonical per-process exchange runtime configuration.

    All secrets use ``repr=False`` so that accidental logging never leaks them.

    Derived properties
    ------------------
    canonical_symbol : str
        ``ETH-USDT-PERP`` — the canonical trading symbol.
    okx_inst_id : str
        ``ETH-USDT-SWAP`` — the OKX instrument ID.
    binance_symbol : str
        ``ETHUSDT`` — the Binance trading symbol.
    """

    exchange: ExchangeName
    trade_asset: str
    quote_asset: str
    market_type: str
    leverage: int = 20
    margin_mode: str = "isolated"
    position_mode: str = "net"
    kline_interval: str = "15m"
    api_key: str = field(repr=False, default="")
    api_secret: str = field(repr=False, default="")
    api_passphrase: str = field(default="", repr=False)

    # -- derived properties ---------------------------------------------------

    @property
    def canonical_symbol(self) -> str:
        """Return the canonical trading symbol, e.g. ``ETH-USDT-PERP``."""
        return f"{self.trade_asset}-{self.quote_asset}-PERP"

    @property
    def okx_inst_id(self) -> str:
        """Return the OKX instrument ID, e.g. ``ETH-USDT-SWAP``."""
        return f"{self.trade_asset}-{self.quote_asset}-SWAP"

    @property
    def binance_symbol(self) -> str:
        """Return the Binance trading symbol, e.g. ``ETHUSDT``."""
        return f"{self.trade_asset}{self.quote_asset}"

    @property
    def is_okx(self) -> bool:
        return self.exchange == ExchangeName.OKX

    @property
    def is_binance(self) -> bool:
        return self.exchange == ExchangeName.BINANCE


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_unified_runtime_config(
    env: Mapping[str, str] | None = None,
) -> ExchangeRuntimeConfig:
    """Build an :class:`ExchangeRuntimeConfig` from environment variables.

    OKX and Binance share every parameter except ``EXCHANGE`` and the three
    API credentials.  Legacy OKX-specific environment variables (OKX_TD_MODE,
    OKX_POS_SIDE_MODE, OKX_INST_ID, OKX_BAR) are intentionally NOT consumed.

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
        If any value is invalid (unsupported exchange, non-ETH asset,
        non-USDT quote, non-PERPETUAL market type, non-positive leverage,
        margin mode other than isolated, position mode other than net,
        kline interval other than 15m, etc.).
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

    if trade_asset != SUPPORTED_TRADE_ASSET:
        raise ValueError(
            f"Unsupported TRADE_ASSET: {trade_asset} "
            f"(only {SUPPORTED_TRADE_ASSET} is supported)"
        )
    if quote_asset != SUPPORTED_QUOTE_ASSET:
        raise ValueError(
            f"Unsupported QUOTE_ASSET: {quote_asset} "
            f"(only {SUPPORTED_QUOTE_ASSET} is supported)"
        )
    if market_type != SUPPORTED_MARKET_TYPE:
        raise ValueError(
            f"Unsupported MARKET_TYPE: {market_type} "
            f"(only {SUPPORTED_MARKET_TYPE} is supported)"
        )

    # -- leverage / margin / position / kline ----------------------------------

    leverage_raw = str(values.get("LEVERAGE", str(SUPPORTED_LEVERAGE))).strip()
    try:
        leverage = int(leverage_raw)
    except ValueError as exc:
        raise ValueError(f"LEVERAGE must be an integer: {leverage_raw}") from exc
    if leverage <= 0:
        raise ValueError("LEVERAGE must be positive")

    margin_mode = str(values.get("MARGIN_MODE", SUPPORTED_MARGIN_MODE)).strip().lower()
    if margin_mode != SUPPORTED_MARGIN_MODE:
        raise ValueError(
            f"Unsupported MARGIN_MODE: {margin_mode} "
            f"(only {SUPPORTED_MARGIN_MODE} is supported)"
        )

    position_mode = str(values.get("POSITION_MODE", SUPPORTED_POSITION_MODE)).strip().lower()
    if position_mode != SUPPORTED_POSITION_MODE:
        raise ValueError(
            f"Unsupported POSITION_MODE: {position_mode} "
            f"(only {SUPPORTED_POSITION_MODE} is supported)"
        )

    kline_interval = str(values.get("KLINE_INTERVAL", SUPPORTED_KLINE_INTERVAL)).strip().lower()
    if kline_interval != SUPPORTED_KLINE_INTERVAL:
        raise ValueError(
            f"Unsupported KLINE_INTERVAL: {kline_interval} "
            f"(only {SUPPORTED_KLINE_INTERVAL} is supported)"
        )

    # -- credentials ----------------------------------------------------------

    return ExchangeRuntimeConfig(
        exchange=exchange,
        trade_asset=trade_asset,
        quote_asset=quote_asset,
        market_type=market_type,
        leverage=leverage,
        margin_mode=margin_mode,
        position_mode=position_mode,
        kline_interval=kline_interval,
        api_key=str(values.get("EXCHANGE_API_KEY", "")),
        api_secret=str(values.get("EXCHANGE_API_SECRET", "")),
        api_passphrase=str(values.get("EXCHANGE_API_PASSPHRASE", "")),
    )


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------

load_exchange_runtime_config_from_env = load_unified_runtime_config
