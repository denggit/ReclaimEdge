#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_exchange_runtime_config.py
@Description: Unit tests for ExchangeRuntimeConfig and its env loader.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.exchanges.models import ExchangeName
from src.exchanges.runtime_config import (
    ExchangeRuntimeConfig,
    load_exchange_runtime_config_from_env,
)


# ---------------------------------------------------------------------------
# 1. Defaults
# ---------------------------------------------------------------------------


def test_load_exchange_runtime_config_defaults_to_okx_eth_perpetual() -> None:
    config = load_exchange_runtime_config_from_env({})

    assert config.exchange == ExchangeName.OKX
    assert config.trade_asset == "ETH"
    assert config.quote_asset == "USDT"
    assert config.market_type == "PERPETUAL"
    assert config.canonical_symbol == "ETH-USDT-PERP"
    assert config.leverage == 20
    assert config.margin_mode == "isolated"
    assert config.position_mode == "hedge"
    assert config.api_key == ""
    assert config.api_secret == ""
    assert config.api_passphrase == ""


# ---------------------------------------------------------------------------
# 2. OKX full config
# ---------------------------------------------------------------------------


def test_load_exchange_runtime_config_okx() -> None:
    config = load_exchange_runtime_config_from_env(
        {
            "EXCHANGE": "okx",
            "TRADE_ASSET": "ETH",
            "QUOTE_ASSET": "USDT",
            "MARKET_TYPE": "PERPETUAL",
            "EXCHANGE_API_KEY": "okx-key",
            "EXCHANGE_API_SECRET": "okx-secret",
            "EXCHANGE_API_PASSPHRASE": "okx-pass",
            "LEVERAGE": "20",
            "MARGIN_MODE": "isolated",
            "POSITION_MODE": "hedge",
        }
    )

    assert config.exchange == ExchangeName.OKX
    assert config.is_okx is True
    assert config.is_binance is False
    assert config.canonical_symbol == "ETH-USDT-PERP"
    assert config.api_key == "okx-key"
    assert config.api_secret == "okx-secret"
    assert config.api_passphrase == "okx-pass"


# ---------------------------------------------------------------------------
# 3. Binance config – same canonical symbol
# ---------------------------------------------------------------------------


def test_load_exchange_runtime_config_binance_same_canonical_symbol() -> None:
    config = load_exchange_runtime_config_from_env(
        {
            "EXCHANGE": "binance",
            "TRADE_ASSET": "ETH",
            "QUOTE_ASSET": "USDT",
            "MARKET_TYPE": "PERPETUAL",
            "EXCHANGE_API_KEY": "binance-key",
            "EXCHANGE_API_SECRET": "binance-secret",
            "EXCHANGE_API_PASSPHRASE": "",
            "LEVERAGE": "20",
            "MARGIN_MODE": "isolated",
            "POSITION_MODE": "hedge",
        }
    )

    assert config.exchange == ExchangeName.BINANCE
    assert config.is_okx is False
    assert config.is_binance is True
    assert config.canonical_symbol == "ETH-USDT-PERP"
    assert config.api_key == "binance-key"
    assert config.api_secret == "binance-secret"
    assert config.api_passphrase == ""


# ---------------------------------------------------------------------------
# 4. Secret leak guard
# ---------------------------------------------------------------------------


def test_exchange_runtime_config_repr_does_not_leak_secrets() -> None:
    config = load_exchange_runtime_config_from_env(
        {
            "EXCHANGE_API_KEY": "super-key",
            "EXCHANGE_API_SECRET": "super-secret",
            "EXCHANGE_API_PASSPHRASE": "super-pass",
        }
    )

    text = repr(config)

    assert "super-key" not in text
    assert "super-secret" not in text
    assert "super-pass" not in text


# ---------------------------------------------------------------------------
# 5. Invalid exchange
# ---------------------------------------------------------------------------


def test_load_exchange_runtime_config_rejects_unknown_exchange() -> None:
    with pytest.raises(ValueError, match="Unsupported EXCHANGE"):
        load_exchange_runtime_config_from_env({"EXCHANGE": "coinbase"})


# ---------------------------------------------------------------------------
# 6. Invalid market type
# ---------------------------------------------------------------------------


def test_load_exchange_runtime_config_rejects_non_perpetual_market_type() -> None:
    with pytest.raises(ValueError, match="Unsupported MARKET_TYPE"):
        load_exchange_runtime_config_from_env({"MARKET_TYPE": "SPOT"})


# ---------------------------------------------------------------------------
# 7. Invalid leverage
# ---------------------------------------------------------------------------


def test_load_exchange_runtime_config_rejects_invalid_leverage() -> None:
    with pytest.raises(ValueError, match="LEVERAGE"):
        load_exchange_runtime_config_from_env({"LEVERAGE": "abc"})

    with pytest.raises(ValueError, match="LEVERAGE must be positive"):
        load_exchange_runtime_config_from_env({"LEVERAGE": "0"})


# ---------------------------------------------------------------------------
# 8. Boundary – no live / execution / config dependency
# ---------------------------------------------------------------------------


def test_exchange_runtime_config_has_no_live_execution_or_config_dependency() -> None:
    """runtime_config.py must not import or reference live / execution / config
    / network modules.  The check is line-based to avoid matching English prose
    (e.g. "configuration" containing "config.")."""
    lines = Path("src/exchanges/runtime_config.py").read_text(encoding="utf-8").splitlines()

    forbidden_imports = [
        "src.execution",
        "src.live",
        "src.strategies",
        "src.risk",
        "src.reporting",
        "from config",
        "import config",
        "OKX_CONFIG",
    ]

    for line in lines:
        stripped = line.strip()
        # only look at import lines
        if not stripped.startswith(("import ", "from ")):
            continue
        for token in forbidden_imports:
            assert token not in stripped, (
                f"runtime_config.py MUST NOT import/reference '{token}': {stripped}"
            )


# ---------------------------------------------------------------------------
# 9. Reject non-ETH trade asset
# ---------------------------------------------------------------------------


def test_load_exchange_runtime_config_rejects_non_eth_trade_asset() -> None:
    with pytest.raises(ValueError, match="Unsupported TRADE_ASSET"):
        load_exchange_runtime_config_from_env({"TRADE_ASSET": "BTC"})


# ---------------------------------------------------------------------------
# 10. Reject non-USDT quote asset
# ---------------------------------------------------------------------------


def test_load_exchange_runtime_config_rejects_non_usdt_quote_asset() -> None:
    with pytest.raises(ValueError, match="Unsupported QUOTE_ASSET"):
        load_exchange_runtime_config_from_env({"QUOTE_ASSET": "USDC"})


# ---------------------------------------------------------------------------
# 11. Source-level – no _TRUE_VALUES dead code
# ---------------------------------------------------------------------------


def test_exchange_runtime_config_has_no_unused_true_values_constant() -> None:
    text = Path("src/exchanges/runtime_config.py").read_text(encoding="utf-8")
    assert "_TRUE_VALUES" not in text
