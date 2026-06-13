#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_unified_eth_runtime_config.py
@Description: Comprehensive tests for the unified ETH perpetual runtime config.

Covers load_unified_runtime_config and ExchangeRuntimeConfig with every
supported and rejected value.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.exchanges.models import ExchangeName
from src.exchanges.runtime_config import (
    SUPPORTED_BINANCE_SYMBOL,
    SUPPORTED_CANONICAL_SYMBOL,
    SUPPORTED_KLINE_INTERVAL,
    SUPPORTED_LEVERAGE,
    SUPPORTED_MARGIN_MODE,
    SUPPORTED_MARKET_TYPE,
    SUPPORTED_OKX_INST_ID,
    SUPPORTED_POSITION_MODE,
    SUPPORTED_QUOTE_ASSET,
    SUPPORTED_TRADE_ASSET,
    ExchangeRuntimeConfig,
    load_unified_runtime_config,
)


# ---------------------------------------------------------------------------
# 1. Defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_default_exchange_is_okx(self) -> None:
        config = load_unified_runtime_config({})
        assert config.exchange == ExchangeName.OKX

    def test_default_trade_asset_is_eth(self) -> None:
        config = load_unified_runtime_config({})
        assert config.trade_asset == SUPPORTED_TRADE_ASSET

    def test_default_quote_asset_is_usdt(self) -> None:
        config = load_unified_runtime_config({})
        assert config.quote_asset == SUPPORTED_QUOTE_ASSET

    def test_default_market_type_is_perpetual(self) -> None:
        config = load_unified_runtime_config({})
        assert config.market_type == SUPPORTED_MARKET_TYPE

    def test_default_margin_mode_is_isolated(self) -> None:
        config = load_unified_runtime_config({})
        assert config.margin_mode == SUPPORTED_MARGIN_MODE

    def test_default_position_mode_is_net(self) -> None:
        config = load_unified_runtime_config({})
        assert config.position_mode == SUPPORTED_POSITION_MODE

    def test_default_leverage_is_20(self) -> None:
        config = load_unified_runtime_config({})
        assert config.leverage == SUPPORTED_LEVERAGE

    def test_default_kline_interval_is_15m(self) -> None:
        config = load_unified_runtime_config({})
        assert config.kline_interval == SUPPORTED_KLINE_INTERVAL

    def test_default_api_credentials_are_empty(self) -> None:
        config = load_unified_runtime_config({})
        assert config.api_key == ""
        assert config.api_secret == ""
        assert config.api_passphrase == ""


# ---------------------------------------------------------------------------
# 2. Derived properties
# ---------------------------------------------------------------------------


class TestDerivedProperties:
    def test_canonical_symbol_okx(self) -> None:
        config = load_unified_runtime_config({"EXCHANGE": "okx"})
        assert config.canonical_symbol == SUPPORTED_CANONICAL_SYMBOL

    def test_canonical_symbol_binance(self) -> None:
        config = load_unified_runtime_config({"EXCHANGE": "binance"})
        assert config.canonical_symbol == SUPPORTED_CANONICAL_SYMBOL

    def test_okx_inst_id(self) -> None:
        config = load_unified_runtime_config({"EXCHANGE": "okx"})
        assert config.okx_inst_id == SUPPORTED_OKX_INST_ID

    def test_binance_symbol(self) -> None:
        config = load_unified_runtime_config({"EXCHANGE": "binance"})
        assert config.binance_symbol == SUPPORTED_BINANCE_SYMBOL

    def test_is_okx(self) -> None:
        config = load_unified_runtime_config({"EXCHANGE": "okx"})
        assert config.is_okx is True
        assert config.is_binance is False

    def test_is_binance(self) -> None:
        config = load_unified_runtime_config({"EXCHANGE": "binance"})
        assert config.is_okx is False
        assert config.is_binance is True


# ---------------------------------------------------------------------------
# 3. OKX full config
# ---------------------------------------------------------------------------


class TestOkxConfig:
    def test_okx_full_config(self) -> None:
        config = load_unified_runtime_config({
            "EXCHANGE": "okx",
            "TRADE_ASSET": "ETH",
            "QUOTE_ASSET": "USDT",
            "MARKET_TYPE": "PERPETUAL",
            "MARGIN_MODE": "isolated",
            "POSITION_MODE": "net",
            "LEVERAGE": "10",
            "KLINE_INTERVAL": "15m",
            "EXCHANGE_API_KEY": "okx-key",
            "EXCHANGE_API_SECRET": "okx-secret",
            "EXCHANGE_API_PASSPHRASE": "okx-pass",
        })
        assert config.exchange == ExchangeName.OKX
        assert config.okx_inst_id == "ETH-USDT-SWAP"
        assert config.leverage == 10
        assert config.api_key == "okx-key"
        assert config.api_secret == "okx-secret"
        assert config.api_passphrase == "okx-pass"


# ---------------------------------------------------------------------------
# 4. Binance full config
# ---------------------------------------------------------------------------


class TestBinanceConfig:
    def test_binance_full_config(self) -> None:
        config = load_unified_runtime_config({
            "EXCHANGE": "binance",
            "TRADE_ASSET": "ETH",
            "QUOTE_ASSET": "USDT",
            "MARKET_TYPE": "PERPETUAL",
            "MARGIN_MODE": "isolated",
            "POSITION_MODE": "net",
            "LEVERAGE": "5",
            "KLINE_INTERVAL": "15m",
            "EXCHANGE_API_KEY": "binance-key",
            "EXCHANGE_API_SECRET": "binance-secret",
        })
        assert config.exchange == ExchangeName.BINANCE
        assert config.binance_symbol == "ETHUSDT"
        assert config.leverage == 5
        assert config.api_key == "binance-key"
        assert config.api_secret == "binance-secret"

    def test_binance_okx_inst_id_still_computed(self) -> None:
        """okx_inst_id is always derived, even when exchange is binance."""
        config = load_unified_runtime_config({"EXCHANGE": "binance"})
        assert config.okx_inst_id == "ETH-USDT-SWAP"

    def test_okx_binance_symbol_still_computed(self) -> None:
        """binance_symbol is always derived, even when exchange is okx."""
        config = load_unified_runtime_config({"EXCHANGE": "okx"})
        assert config.binance_symbol == "ETHUSDT"


# ---------------------------------------------------------------------------
# 5. Rejections
# ---------------------------------------------------------------------------


class TestRejections:
    def test_rejects_btc_trade_asset(self) -> None:
        with pytest.raises(ValueError, match="Unsupported TRADE_ASSET"):
            load_unified_runtime_config({"TRADE_ASSET": "BTC"})

    def test_rejects_btc_trade_asset_lowercase(self) -> None:
        with pytest.raises(ValueError, match="Unsupported TRADE_ASSET"):
            load_unified_runtime_config({"TRADE_ASSET": "btc"})

    def test_rejects_non_usdt_quote_asset(self) -> None:
        with pytest.raises(ValueError, match="Unsupported QUOTE_ASSET"):
            load_unified_runtime_config({"QUOTE_ASSET": "USD"})

    def test_rejects_spot_market_type(self) -> None:
        with pytest.raises(ValueError, match="Unsupported MARKET_TYPE"):
            load_unified_runtime_config({"MARKET_TYPE": "SPOT"})

    def test_rejects_futures_market_type(self) -> None:
        with pytest.raises(ValueError, match="Unsupported MARKET_TYPE"):
            load_unified_runtime_config({"MARKET_TYPE": "FUTURES"})

    def test_rejects_cross_margin_mode(self) -> None:
        with pytest.raises(ValueError, match="Unsupported MARGIN_MODE"):
            load_unified_runtime_config({"MARGIN_MODE": "cross"})

    def test_rejects_hedge_position_mode(self) -> None:
        with pytest.raises(ValueError, match="Unsupported POSITION_MODE"):
            load_unified_runtime_config({"POSITION_MODE": "hedge"})

    def test_rejects_1m_kline_interval(self) -> None:
        with pytest.raises(ValueError, match="Unsupported KLINE_INTERVAL"):
            load_unified_runtime_config({"KLINE_INTERVAL": "1m"})

    def test_rejects_5m_kline_interval(self) -> None:
        with pytest.raises(ValueError, match="Unsupported KLINE_INTERVAL"):
            load_unified_runtime_config({"KLINE_INTERVAL": "5m"})

    def test_rejects_unknown_exchange(self) -> None:
        with pytest.raises(ValueError, match="Unsupported EXCHANGE"):
            load_unified_runtime_config({"EXCHANGE": "coinbase"})

    def test_rejects_non_integer_leverage(self) -> None:
        with pytest.raises(ValueError, match="LEVERAGE must be an integer"):
            load_unified_runtime_config({"LEVERAGE": "abc"})

    def test_rejects_zero_leverage(self) -> None:
        with pytest.raises(ValueError, match="LEVERAGE must be positive"):
            load_unified_runtime_config({"LEVERAGE": "0"})

    def test_rejects_negative_leverage(self) -> None:
        with pytest.raises(ValueError, match="LEVERAGE must be positive"):
            load_unified_runtime_config({"LEVERAGE": "-1"})


# ---------------------------------------------------------------------------
# 6. OKX raw symbol
# ---------------------------------------------------------------------------


class TestOkxRawSymbol:
    def test_okx_inst_id_is_eth_usdt_swap(self) -> None:
        config = load_unified_runtime_config({"EXCHANGE": "okx"})
        assert config.okx_inst_id == "ETH-USDT-SWAP"


# ---------------------------------------------------------------------------
# 7. Binance raw symbol
# ---------------------------------------------------------------------------


class TestBinanceRawSymbol:
    def test_binance_symbol_is_ethusdt(self) -> None:
        config = load_unified_runtime_config({"EXCHANGE": "binance"})
        assert config.binance_symbol == "ETHUSDT"


# ---------------------------------------------------------------------------
# 8. Secret leak guard
# ---------------------------------------------------------------------------


class TestSecretLeakGuard:
    def test_repr_does_not_leak_secrets(self) -> None:
        config = load_unified_runtime_config({
            "EXCHANGE_API_KEY": "super-key",
            "EXCHANGE_API_SECRET": "super-secret",
            "EXCHANGE_API_PASSPHRASE": "super-pass",
        })
        text = repr(config)
        assert "super-key" not in text
        assert "super-secret" not in text
        assert "super-pass" not in text


# ---------------------------------------------------------------------------
# 9. Boundary – no forbidden imports
# ---------------------------------------------------------------------------


class TestNoForbiddenImports:
    def test_runtime_config_has_no_live_imports(self) -> None:
        lines = (
            Path("src/exchanges/runtime_config.py")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        forbidden_imports = [
            "src.execution",
            "src.live",
            "src.strategies",
            "src.risk",
            "src.reporting",
            "src.data_feed",
            "src.account_sync",
            "from config",
            "import config",
            "aiohttp",
            "websockets",
            "requests",
            "OKX_CONFIG",
        ]
        for line in lines:
            stripped = line.strip()
            if not stripped.startswith(("import ", "from ")):
                continue
            for token in forbidden_imports:
                assert token not in stripped, (
                    f"runtime_config.py MUST NOT import '{token}': {stripped}"
                )

    def test_runtime_config_does_not_import_binance_http_transport(self) -> None:
        text = Path("src/exchanges/runtime_config.py").read_text(encoding="utf-8")
        assert "BinanceHttpTransport" not in text
        assert "AiohttpBinanceTransport" not in text
        assert "binance.transport" not in text


# ---------------------------------------------------------------------------
# 10. OKX_* legacy env vars are NOT consumed by the loader
# ---------------------------------------------------------------------------


class TestNoOkxLegacyEnvConsumption:
    def test_okx_legacy_env_vars_not_consumed(self) -> None:
        """The unified loader does not read OKX_* legacy env vars."""
        text = Path("src/exchanges/runtime_config.py").read_text(encoding="utf-8")
        legacy = {"OKX_TD_MODE", "OKX_POS_SIDE_MODE", "OKX_INST_ID", "OKX_BAR"}

        # Only look at lines inside the loader function body
        in_docstring = False
        in_loader = False
        for line in text.splitlines():
            stripped = line.strip()

            # Track docstring boundaries
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_docstring = not in_docstring
                continue

            if in_docstring:
                continue

            if stripped.startswith("#"):
                continue

            if "def load_unified_runtime_config" in stripped:
                in_loader = True
                continue
            if in_loader and "def load_exchange_runtime_config_from_env" in stripped:
                in_loader = False
                continue

            if not in_loader:
                continue

            for var in legacy:
                if var in stripped:
                    raise AssertionError(
                        f"Loader body must not reference {var}: {stripped}"
                    )

    def test_unified_loader_does_not_consume_okx_env_vars(self) -> None:
        """OKX_* vars may appear in docstrings but not as env keys in code."""
        text = Path("src/exchanges/runtime_config.py").read_text(encoding="utf-8")

        # Find the loader function body
        loader_start = text.find("def load_unified_runtime_config")
        loader_end = text.find("def load_exchange_runtime_config_from_env")
        if loader_end == -1:
            loader_end = len(text)
        loader_body = text[loader_start:loader_end]

        legacy = {"OKX_TD_MODE", "OKX_POS_SIDE_MODE", "OKX_INST_ID", "OKX_BAR"}
        for var in legacy:
            # Only flag if the var appears in a values.get() or similar
            if f'values.get("{var}"' in loader_body or f"values.get('{var}')" in loader_body:
                raise AssertionError(
                    f"Unified config loader must not consume OKX_* env var: {var}"
                )
            if f'os.environ.get("{var}"' in loader_body or f"os.environ.get('{var}')" in loader_body:
                raise AssertionError(
                    f"Unified config loader must not consume OKX_* env var: {var}"
                )


# ---------------------------------------------------------------------------
# 11. Supported-value constants
# ---------------------------------------------------------------------------


class TestSupportedConstants:
    def test_supported_trade_asset_is_eth(self) -> None:
        assert SUPPORTED_TRADE_ASSET == "ETH"

    def test_supported_quote_asset_is_usdt(self) -> None:
        assert SUPPORTED_QUOTE_ASSET == "USDT"

    def test_supported_market_type_is_perpetual(self) -> None:
        assert SUPPORTED_MARKET_TYPE == "PERPETUAL"

    def test_supported_margin_mode_is_isolated(self) -> None:
        assert SUPPORTED_MARGIN_MODE == "isolated"

    def test_supported_position_mode_is_net(self) -> None:
        assert SUPPORTED_POSITION_MODE == "net"

    def test_supported_leverage_is_20(self) -> None:
        assert SUPPORTED_LEVERAGE == 20

    def test_supported_kline_interval_is_15m(self) -> None:
        assert SUPPORTED_KLINE_INTERVAL == "15m"

    def test_supported_canonical_symbol(self) -> None:
        assert SUPPORTED_CANONICAL_SYMBOL == "ETH-USDT-PERP"

    def test_supported_okx_inst_id(self) -> None:
        assert SUPPORTED_OKX_INST_ID == "ETH-USDT-SWAP"

    def test_supported_binance_symbol(self) -> None:
        assert SUPPORTED_BINANCE_SYMBOL == "ETHUSDT"


# ---------------------------------------------------------------------------
# 12. Missing env keys use defaults
# ---------------------------------------------------------------------------


class TestMissingKeyDefaults:
    def test_missing_exchange_defaults_to_okx(self) -> None:
        config = load_unified_runtime_config({})
        assert config.exchange == ExchangeName.OKX

    def test_missing_trade_asset_defaults_to_eth(self) -> None:
        config = load_unified_runtime_config({})
        assert config.trade_asset == SUPPORTED_TRADE_ASSET

    def test_empty_string_exchange_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported EXCHANGE"):
            load_unified_runtime_config({"EXCHANGE": ""})

    def test_empty_string_trade_asset_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported TRADE_ASSET"):
            load_unified_runtime_config({"TRADE_ASSET": ""})
