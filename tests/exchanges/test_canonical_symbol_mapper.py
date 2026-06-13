#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_canonical_symbol_mapper.py
@Description: Unit tests for the canonical symbol mapper.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.exchanges.models import ExchangeName
from src.exchanges.runtime_config import load_exchange_runtime_config_from_env
from src.exchanges.symbols import (
    assert_supported_canonical_symbol,
    raw_symbol_for_exchange,
    raw_symbol_from_runtime_config,
)


# ---------------------------------------------------------------------------
# 1. OKX mapping
# ---------------------------------------------------------------------------


def test_raw_symbol_for_okx_eth_usdt_perpetual() -> None:
    assert raw_symbol_for_exchange(
        exchange=ExchangeName.OKX,
        canonical_symbol="ETH-USDT-PERP",
    ) == "ETH-USDT-SWAP"


# ---------------------------------------------------------------------------
# 2. Binance mapping
# ---------------------------------------------------------------------------


def test_raw_symbol_for_binance_eth_usdt_perpetual() -> None:
    assert raw_symbol_for_exchange(
        exchange=ExchangeName.BINANCE,
        canonical_symbol="ETH-USDT-PERP",
    ) == "ETHUSDT"


# ---------------------------------------------------------------------------
# 3. Runtime config mapping – OKX
# ---------------------------------------------------------------------------


def test_raw_symbol_from_okx_runtime_config() -> None:
    config = load_exchange_runtime_config_from_env(
        {
            "EXCHANGE": "okx",
            "TRADE_ASSET": "ETH",
            "QUOTE_ASSET": "USDT",
            "MARKET_TYPE": "PERPETUAL",
        }
    )

    assert config.canonical_symbol == "ETH-USDT-PERP"
    assert raw_symbol_from_runtime_config(config) == "ETH-USDT-SWAP"


# ---------------------------------------------------------------------------
# 4. Runtime config mapping – Binance
# ---------------------------------------------------------------------------


def test_raw_symbol_from_binance_runtime_config() -> None:
    config = load_exchange_runtime_config_from_env(
        {
            "EXCHANGE": "binance",
            "TRADE_ASSET": "ETH",
            "QUOTE_ASSET": "USDT",
            "MARKET_TYPE": "PERPETUAL",
        }
    )

    assert config.canonical_symbol == "ETH-USDT-PERP"
    assert raw_symbol_from_runtime_config(config) == "ETHUSDT"


# ---------------------------------------------------------------------------
# 5. Unsupported canonical symbol
# ---------------------------------------------------------------------------


def test_raw_symbol_rejects_unsupported_canonical_symbol() -> None:
    with pytest.raises(ValueError, match="Unsupported canonical symbol"):
        raw_symbol_for_exchange(
            exchange=ExchangeName.OKX,
            canonical_symbol="BTC-USDT-PERP",
        )


# ---------------------------------------------------------------------------
# 6. Unsupported exchange
# ---------------------------------------------------------------------------


def test_raw_symbol_rejects_unsupported_exchange() -> None:
    with pytest.raises(ValueError, match="Unsupported exchange"):
        raw_symbol_for_exchange(
            exchange=ExchangeName.BYBIT,
            canonical_symbol="ETH-USDT-PERP",
        )


# ---------------------------------------------------------------------------
# 7. assert_supported_canonical_symbol – direct
# ---------------------------------------------------------------------------


def test_assert_supported_canonical_symbol_passes_for_eth_usdt_perp() -> None:
    # Must not raise
    assert_supported_canonical_symbol("ETH-USDT-PERP")


def test_assert_supported_canonical_symbol_raises_for_unknown() -> None:
    with pytest.raises(ValueError, match="Unsupported canonical symbol"):
        assert_supported_canonical_symbol("BTC-USDT-PERP")


# ---------------------------------------------------------------------------
# 8. Source-level boundary – no live / execution / config dependency
# ---------------------------------------------------------------------------


def test_symbol_mapper_has_no_live_execution_or_config_dependency() -> None:
    text = Path("src/exchanges/symbols.py").read_text(encoding="utf-8")

    forbidden = [
        "src.execution",
        "src.live",
        "src.strategies",
        "src.risk",
        "src.reporting",
        "config.",
        "OKX_CONFIG",
        "requests",
        "aiohttp",
        "httpx",
        "websockets",
    ]

    for token in forbidden:
        assert token not in text, f"{token} should not appear in symbols.py"
