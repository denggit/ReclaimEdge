#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_binance_signal_only_runtime_boundaries.py
@Description: Boundary tests — the Binance signal-only runtime must NOT import
              strategy, execution, broker, signing, live workers, OKX modules,
              or any order-placing / secret-reading APIs.  It must only support
              ETH-USDT-PERP / ETHUSDT / 15m.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.live.binance_signal_only_runtime import (
    BinanceSignalOnlyConfig,
    load_binance_signal_only_config,
)

# ======================================================================
# Path to runtime source
# ======================================================================

_RUNTIME_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "live"
    / "binance_signal_only_runtime.py"
)

# Also check the config loader separately
_CONFIG_MODULE = "src.live.binance_signal_only_runtime"


def _read_runtime_text() -> str:
    return _RUNTIME_PATH.read_text(encoding="utf-8")


# ======================================================================
# File existence / compilation
# ======================================================================


def test_runtime_file_exists() -> None:
    assert _RUNTIME_PATH.exists(), f"Runtime file not found at {_RUNTIME_PATH}"
    assert _RUNTIME_PATH.is_file()


def test_runtime_file_compiles() -> None:
    text = _read_runtime_text()
    compile(text, str(_RUNTIME_PATH), "exec")


# ======================================================================
# Forbidden imports
# ======================================================================


class TestForbiddenImports:
    """The signal-only runtime must NOT import forbidden modules."""

    def test_does_not_import_execution(self) -> None:
        text = _read_runtime_text()
        assert "src.execution" not in text

    def test_does_not_import_binance_broker(self) -> None:
        text = _read_runtime_text()
        assert "src.exchanges.binance.broker" not in text
        _BROKER = "Binance" + "BrokerClient"
        assert _BROKER not in text

    def test_does_not_import_signed_request(self) -> None:
        text = _read_runtime_text()
        assert "src.exchanges.binance.signing" not in text
        assert "build_signed_request" not in text

    def test_does_not_import_semantic_executor(self) -> None:
        text = _read_runtime_text()
        assert "src.exchanges.binance.semantic" not in text

    def test_does_not_import_okx(self) -> None:
        text = _read_runtime_text()
        assert "src.exchanges.okx" not in text

    def test_does_not_import_live_workers(self) -> None:
        text = _read_runtime_text()
        assert "src.live.workers" not in text

    def test_does_not_import_execution_worker(self) -> None:
        text = _read_runtime_text()
        assert "execution_worker" not in text

    def test_does_not_import_account_sync(self) -> None:
        text = _read_runtime_text()
        assert "account_position_sync_worker" not in text

    def test_does_not_import_trader(self) -> None:
        text = _read_runtime_text()
        assert "src.execution.trader" not in text

    def test_does_not_import_startup_recovery(self) -> None:
        text = _read_runtime_text()
        assert "src.live.startup_recovery" not in text

    def test_does_not_import_reporting(self) -> None:
        text = _read_runtime_text()
        assert "src.reporting" not in text

    def test_does_not_import_position_management(self) -> None:
        text = _read_runtime_text()
        assert "src.position_management" not in text

    def test_does_not_import_delayed_market_exit(self) -> None:
        text = _read_runtime_text()
        assert "src.live.delayed_market_exit" not in text


# ======================================================================
# No secret / API key references
# ======================================================================


class TestNoApiKeyOrEnv:
    """The signal-only runtime must not read API credentials."""

    def test_does_not_reference_api_key_env(self) -> None:
        text = _read_runtime_text()
        # Source code must not contain secret env var names
        assert "EXCHANGE_API_KEY" not in text
        assert "EXCHANGE_API_SECRET" not in text
        assert "EXCHANGE_API_PASSPHRASE" not in text
        assert "BINANCE_API_KEY" not in text
        assert "BINANCE_SECRET_KEY" not in text

    def test_does_not_call_os_getenv_on_secrets(self) -> None:
        text = _read_runtime_text()
        # The whitelist approach means secrets are never read individually
        # But check that the source doesn't use os.getenv on secret keys
        for secret in ["EXCHANGE_API_KEY", "EXCHANGE_API_SECRET",
                       "EXCHANGE_API_PASSPHRASE", "BINANCE_API_KEY",
                       "BINANCE_SECRET_KEY"]:
            assert secret not in text, f"Secret key '{secret}' found in runtime source"


# ======================================================================
# No order / trading logic
# ======================================================================


class TestNoOrderPlacement:
    """The signal-only runtime must NOT contain order-placing logic."""

    def test_no_place_order(self) -> None:
        text = _read_runtime_text()
        assert "place_order" not in text

    def test_no_cancel_order(self) -> None:
        text = _read_runtime_text()
        assert "cancel_order" not in text

    def test_no_position_side(self) -> None:
        text = _read_runtime_text()
        assert "positionSide" not in text
        assert "POSITION_SIDE" not in text
        assert "position_side" not in text

    def test_no_broker_creation(self) -> None:
        text = _read_runtime_text()
        _BROKER = "Binance" + "BrokerClient("
        assert _BROKER not in text
        assert "Trader()" not in text


# ======================================================================
# No BTC / SPOT / multi-symbol
# ======================================================================


class TestNoBtcOrMultiSymbol:
    """The runtime must only support ETH."""

    def test_no_btc_reference(self) -> None:
        text = _read_runtime_text()
        assert "BTC" not in text

    def test_no_spot_reference(self) -> None:
        text = _read_runtime_text()
        assert "SPOT" not in text


# ======================================================================
# Symbol / interval restrictions
# ======================================================================


class TestSymbolRestrictions:
    """Only ETH-USDT-PERP / ETHUSDT / 15m are accepted."""

    def test_btc_trade_asset_rejected(self) -> None:
        import os
        from unittest import mock

        env = {
            "EXCHANGE": "binance",
            "TRADE_ASSET": "BTC",
            "QUOTE_ASSET": "USDT",
            "MARKET_TYPE": "PERPETUAL",
            "KLINE_INTERVAL": "15m",
            "BINANCE_SIGNAL_ONLY": "true",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError):
                load_binance_signal_only_config()

    def test_1m_kline_rejected(self) -> None:
        import os
        from unittest import mock

        env = {
            "EXCHANGE": "binance",
            "TRADE_ASSET": "ETH",
            "QUOTE_ASSET": "USDT",
            "MARKET_TYPE": "PERPETUAL",
            "KLINE_INTERVAL": "1m",
            "BINANCE_SIGNAL_ONLY": "true",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError):
                load_binance_signal_only_config()

    def test_5m_kline_rejected(self) -> None:
        import os
        from unittest import mock

        env = {
            "EXCHANGE": "binance",
            "TRADE_ASSET": "ETH",
            "QUOTE_ASSET": "USDT",
            "MARKET_TYPE": "PERPETUAL",
            "KLINE_INTERVAL": "5m",
            "BINANCE_SIGNAL_ONLY": "true",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError):
                load_binance_signal_only_config()

    def test_okx_exchange_rejected(self) -> None:
        import os
        from unittest import mock

        env = {
            "EXCHANGE": "okx",
            "TRADE_ASSET": "ETH",
            "QUOTE_ASSET": "USDT",
            "MARKET_TYPE": "PERPETUAL",
            "KLINE_INTERVAL": "15m",
            "BINANCE_SIGNAL_ONLY": "true",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError):
                load_binance_signal_only_config()


# ======================================================================
# Config immutability
# ======================================================================


class TestConfigFrozen:
    """Config is frozen."""

    def test_frozen_prevents_mutation(self) -> None:
        config = BinanceSignalOnlyConfig(
            canonical_symbol="ETH-USDT-PERP",
            raw_symbol="ETHUSDT",
            kline_interval="15m",
            duration_seconds=3600.0,
            max_events=100000,
            heartbeat_seconds=30.0,
            candle_limit=100,
            boll_window=20,
            boll_std_multiplier=2.0,
            band_distance_threshold_pct=0.005,
            tp_boll_enabled=True,
            tp_boll_window=15,
        )
        with pytest.raises(Exception):
            config.duration_seconds = 10.0  # type: ignore[misc]


# ======================================================================
# No monitor import (only data model + calculator)
# ======================================================================

class TestNoMonitorInstantiation:
    """The runtime must not instantiate BollBandBreakoutMonitor."""

    def test_no_monitor_class_in_source(self) -> None:
        text = _read_runtime_text()
        assert "BollBandBreakoutMonitor(" not in text
        assert "BollBandBreakoutMonitorConfig" not in text
        # BollSnapshot is imported for type/dataclass use only — that's OK
        assert "BollSnapshot" in text

    def test_no_okx_public_client(self) -> None:
        text = _read_runtime_text()
        assert "OkxPublicMarketClient" not in text
