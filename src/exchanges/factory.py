#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/12
@File       : factory.py
@Description: Exchange adapter runtime factory / selector.

Provides ``normalize_exchange_name``, ``build_broker_client``, and
``build_broker_semantic_executor`` to let the live runtime select the
correct exchange adapter based on configuration.

The factory defaults to OKX.  Binance requires an explicit ``exchange=binance``
flag **and** an injected transport (unless ``allow_binance_without_transport``
is explicitly set to ``True`` — this is only for testing / shell construction).

The factory does **not** read environment variables, does **not** instantiate
real HTTP transports, and is **not** wired into the live entrypoint yet.
"""

from __future__ import annotations

from typing import Any

from src.exchanges.base import BrokerClient
from src.exchanges.binance.client import BinanceBrokerClient
from src.exchanges.binance.semantic_executor import BinanceBrokerSemanticExecutor
from src.exchanges.models import ExchangeName
from src.exchanges.okx.client import OkxBrokerClient
from src.exchanges.okx.semantic_executor import OkxBrokerSemanticExecutor
from src.exchanges.semantics import BrokerSemanticExecutor


# ---------------------------------------------------------------------------
# Legacy placeholder — keep for backward compatibility
# ---------------------------------------------------------------------------


def unsupported_exchange_message(exchange: ExchangeName) -> str:
    """Return a human-readable message for an unsupported exchange."""
    return f"Exchange adapter is not wired yet: {exchange.value}"


# ---------------------------------------------------------------------------
# Exchange name normalization
# ---------------------------------------------------------------------------


def normalize_exchange_name(exchange: str | ExchangeName | None) -> ExchangeName:
    """Normalize an exchange identifier to an ``ExchangeName`` enum value.

    Args:
        exchange: One of ``None``, a string, or an ``ExchangeName`` instance.

    Returns:
        The resolved ``ExchangeName``.  ``None`` defaults to ``OKX``.

    Raises:
        ValueError: If the string does not map to a supported exchange.
    """
    if exchange is None:
        return ExchangeName.OKX
    if isinstance(exchange, ExchangeName):
        return exchange

    normalized = str(exchange).strip().lower()
    if normalized == ExchangeName.OKX.value:
        return ExchangeName.OKX
    if normalized == ExchangeName.BINANCE.value:
        return ExchangeName.BINANCE

    raise ValueError(f"Unsupported exchange: {exchange!r}")


# ---------------------------------------------------------------------------
# Broker client construction
# ---------------------------------------------------------------------------


def build_broker_client(
    *,
    exchange: str | ExchangeName | None = None,
    okx_client: Any | None = None,
    binance_api_key: str | None = None,
    binance_api_secret: str | None = None,
    binance_transport: Any | None = None,
    binance_base_url: str | None = None,
    allow_binance_without_transport: bool = False,
) -> BrokerClient:
    """Build a ``BrokerClient`` for the resolved exchange.

    Args:
        exchange: Exchange identifier.  ``None`` defaults to OKX.
        okx_client: Required when building the OKX client — the injected
            trader-like object that ``OkxBrokerClient`` wraps.
        binance_api_key: Binance API key (optional for shell construction).
        binance_api_secret: Binance API secret (optional for shell
            construction).
        binance_transport: Required for Binance unless
            ``allow_binance_without_transport`` is ``True``.  Must satisfy
            the ``BinanceHttpTransport`` protocol.
        binance_base_url: Override the default Binance USDⓈ-M base URL.
        allow_binance_without_transport: When ``True``, a Binance broker
            client is returned without a transport (shell mode — every
            method raises ``UNSUPPORTED_OPERATION``).

    Returns:
        A ``BrokerClient`` ready for use.

    Raises:
        ValueError: If required arguments are missing or the exchange is
            unsupported.
    """
    exchange_name = normalize_exchange_name(exchange)

    if exchange_name == ExchangeName.OKX:
        if okx_client is None:
            raise ValueError("okx_client is required to build OKX broker client")
        return OkxBrokerClient(okx_client)

    if exchange_name == ExchangeName.BINANCE:
        if not allow_binance_without_transport and binance_transport is None:
            raise ValueError(
                "binance_transport is required unless allow_binance_without_transport=True"
            )
        return BinanceBrokerClient(
            api_key=binance_api_key,
            api_secret=binance_api_secret,
            transport=binance_transport,
            base_url=binance_base_url,
        )

    raise ValueError(f"Unsupported exchange: {exchange_name!r}")


# ---------------------------------------------------------------------------
# Broker semantic executor construction
# ---------------------------------------------------------------------------


def build_broker_semantic_executor(
    broker_client: BrokerClient,
    *,
    exchange: str | ExchangeName | None = None,
) -> BrokerSemanticExecutor:
    """Build a ``BrokerSemanticExecutor`` that wraps *broker_client*.

    Args:
        broker_client: An already-constructed ``BrokerClient``.
        exchange: Exchange identifier used to pick the correct executor
            class.  ``None`` defaults to OKX.

    Returns:
        A ``BrokerSemanticExecutor`` bound to *broker_client*.

    Raises:
        ValueError: If the exchange is unsupported.
    """
    exchange_name = normalize_exchange_name(exchange)

    if exchange_name == ExchangeName.OKX:
        return OkxBrokerSemanticExecutor(broker_client)

    if exchange_name == ExchangeName.BINANCE:
        return BinanceBrokerSemanticExecutor(broker_client)

    raise ValueError(f"Unsupported exchange: {exchange_name!r}")
