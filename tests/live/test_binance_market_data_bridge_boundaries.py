#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_binance_market_data_bridge_boundaries.py
@Description: Boundary tests — the Binance signal bridge must NOT import
              strategy, execution, broker, signing, live workers, or any
              order-placing modules.  It must only support ETH-USDT-PERP
              / ETHUSDT / 15m.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.live.binance_market_data_bridge import (
    SUPPORTED_CANONICAL_SYMBOL,
    SUPPORTED_INTERVAL,
    SUPPORTED_RAW_SYMBOL,
    BinanceMarketDataSignalBridge,
)

# ======================================================================
# Path to bridge source
# ======================================================================

_BRIDGE_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "live" / "binance_market_data_bridge.py"
)


def _read_bridge_text() -> str:
    return _BRIDGE_PATH.read_text(encoding="utf-8")


# ======================================================================
# File existence / compilation
# ======================================================================


def test_bridge_file_exists() -> None:
    assert _BRIDGE_PATH.exists(), f"Bridge file not found at {_BRIDGE_PATH}"
    assert _BRIDGE_PATH.is_file()


def test_bridge_file_compiles() -> None:
    text = _read_bridge_text()
    compile(text, str(_BRIDGE_PATH), "exec")


# ======================================================================
# Symbol / interval restrictions
# ======================================================================


class TestSymbolRestrictions:
    """Only ETH-USDT-PERP / ETHUSDT / 15m are allowed."""

    def test_canonical_symbol_btc_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported canonical_symbol"):
            BinanceMarketDataSignalBridge(canonical_symbol="BTC-USDT-PERP")

    def test_canonical_symbol_btc_perp_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported canonical_symbol"):
            BinanceMarketDataSignalBridge(canonical_symbol="BTC-USDT-PERP")

    def test_raw_symbol_btc_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported raw_symbol"):
            BinanceMarketDataSignalBridge(raw_symbol="BTCUSDT")

    def test_raw_symbol_btc_perp_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported raw_symbol"):
            BinanceMarketDataSignalBridge(raw_symbol="BTCUSDT")

    def test_interval_1m_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported interval"):
            BinanceMarketDataSignalBridge(interval="1m")

    def test_interval_5m_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported interval"):
            BinanceMarketDataSignalBridge(interval="5m")

    def test_interval_1h_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported interval"):
            BinanceMarketDataSignalBridge(interval="1h")

    def test_interval_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported interval"):
            BinanceMarketDataSignalBridge(interval="")

    def test_spot_symbol_rejected(self) -> None:
        """Spot-style symbols must be rejected."""
        with pytest.raises(ValueError, match="Unsupported canonical_symbol"):
            BinanceMarketDataSignalBridge(canonical_symbol="ETH-USDT")

    def test_empty_canonical_symbol_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported canonical_symbol"):
            BinanceMarketDataSignalBridge(canonical_symbol="")

    def test_empty_raw_symbol_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported raw_symbol"):
            BinanceMarketDataSignalBridge(raw_symbol="")


# ======================================================================
# Forbidden imports — must NOT import execution, strategy, broker, etc.
# ======================================================================


class TestForbiddenImports:
    """The bridge source must not import any forbidden modules."""

    def test_does_not_import_strategy(self) -> None:
        text = _read_bridge_text()
        assert "src.strategies" not in text

    def test_does_not_import_execution(self) -> None:
        text = _read_bridge_text()
        assert "src.execution" not in text

    def test_does_not_import_broker_client(self) -> None:
        text = _read_bridge_text()
        assert "src.exchanges.binance.broker" not in text
        _BROKER = "Binance" + "BrokerClient"
        assert _BROKER not in text

    def test_does_not_import_signed_request_builder(self) -> None:
        text = _read_bridge_text()
        assert "src.exchanges.binance.signed" not in text
        assert "build_signed_request" not in text

    def test_does_not_import_semantic_executor(self) -> None:
        text = _read_bridge_text()
        assert "src.exchanges.binance.semantic" not in text

    def test_does_not_import_live_workers(self) -> None:
        text = _read_bridge_text()
        assert "src.live.workers" not in text

    def test_does_not_import_trader(self) -> None:
        text = _read_bridge_text()
        assert "src.execution.trader" not in text

    def test_does_not_import_live_smoke_test(self) -> None:
        text = _read_bridge_text()
        assert "scripts.binance_live_smoke_test" not in text

    def test_does_not_import_monitors(self) -> None:
        text = _read_bridge_text()
        assert "src.monitors" not in text

    def test_does_not_import_risk(self) -> None:
        text = _read_bridge_text()
        assert "src.risk" not in text

    def test_does_not_import_position_management(self) -> None:
        text = _read_bridge_text()
        assert "src.position_management" not in text

    def test_does_not_import_reporting(self) -> None:
        text = _read_bridge_text()
        assert "src.reporting" not in text

    def test_does_not_import_aiohttp_transport(self) -> None:
        text = _read_bridge_text()
        _TRANSPORT = "Aiohttp" + "BinanceTransport"
        assert _TRANSPORT not in text


# ======================================================================
# No API key / env reading
# ======================================================================


class TestNoApiKeyOrEnv:
    """The bridge must not read API credentials or env vars."""

    def test_does_not_reference_api_key_env(self) -> None:
        text = _read_bridge_text()
        assert "EXCHANGE_API_KEY" not in text
        assert "EXCHANGE_API_SECRET" not in text
        assert "EXCHANGE_API_PASSPHRASE" not in text
        assert "BINANCE_API_KEY" not in text
        assert "BINANCE_SECRET_KEY" not in text

    def test_does_not_call_getenv(self) -> None:
        text = _read_bridge_text()
        assert "os.environ" not in text
        assert "os.getenv" not in text
        assert "load_dotenv" not in text

    def test_does_not_read_okx_env_vars(self) -> None:
        text = _read_bridge_text()
        okx_vars = [
            "OKX_INST_ID",
            "OKX_BAR",
            "OKX_TD_MODE",
            "OKX_POS_SIDE_MODE",
        ]
        for var in okx_vars:
            assert var not in text, f"'{var}' must not appear in bridge source"


# ======================================================================
# No order / trading logic
# ======================================================================


class TestNoOrderPlacement:
    """The bridge must not contain any order-placing or trading logic."""

    def test_no_place_order(self) -> None:
        text = _read_bridge_text()
        assert "place_order" not in text

    def test_no_cancel_order(self) -> None:
        text = _read_bridge_text()
        assert "cancel_order" not in text

    def test_no_market_order_text(self) -> None:
        text = _read_bridge_text()
        assert "MARKET BUY" not in text
        assert "MARKET SELL" not in text

    def test_no_position_side(self) -> None:
        text = _read_bridge_text()
        assert "positionSide" not in text
        assert "POSITION_SIDE" not in text
        assert "position_side" not in text

    def test_no_live_confirmation(self) -> None:
        text = _read_bridge_text()
        assert "LIVE_SMOKE_TEST_CONFIRM" not in text
        assert "BINANCE_LIVE_SMOKE_TEST_CONFIRM" not in text
        assert "I_UNDERSTAND" not in text

    def test_no_broker_creation(self) -> None:
        text = _read_bridge_text()
        _BROKER = "Binance" + "BrokerClient("
        assert _BROKER not in text


# ======================================================================
# No WebSocket / network
# ======================================================================


class TestNoWebsocket:
    """The bridge must not connect to any WebSocket or network."""

    def test_no_websocket_import(self) -> None:
        text = _read_bridge_text()
        assert "import websocket" not in text
        assert "from websocket" not in text
        assert "aiohttp" not in text
        assert "ws_connect" not in text

    def test_no_ws_url(self) -> None:
        text = _read_bridge_text()
        assert "fstream.binance.com" not in text
        assert "wss://" not in text

    def test_no_asyncio_network(self) -> None:
        text = _read_bridge_text()
        assert "asyncio" not in text


# ======================================================================
# No BTC / multi-symbol
# ======================================================================


class TestNoBtcOrMultiSymbol:
    """The bridge must not reference BTC or multi-symbol support."""

    def test_no_btc_reference(self) -> None:
        text = _read_bridge_text()
        assert "BTC" not in text

    def test_no_spot_reference(self) -> None:
        text = _read_bridge_text()
        assert "SPOT" not in text

    def test_only_eth_in_source(self) -> None:
        text = _read_bridge_text()
        assert "ETH" in text  # Must reference ETH

    def test_no_multi_symbol_list(self) -> None:
        text = _read_bridge_text()
        # Should not have a list/dict of symbol mappings
        assert "SYMBOLS" not in text
        assert "symbols" not in text.split("SUPPORTED")[0] if "SUPPORTED" in text else True


# ======================================================================
# Decimal enforcement — no float conversion
# ======================================================================


class TestDecimalEnforcement:
    """The bridge must not convert Decimal to float."""

    def test_no_float_conversion(self) -> None:
        text = _read_bridge_text()
        # Must not have bare float() calls on price/quantity/OHLCV paths
        assert "float(" not in text


# ======================================================================
# Constants are hard-coded (not from env)
# ======================================================================


class TestHardcodedConstants:
    """Supported symbol/interval constants are hard-coded, not read from env."""

    def test_supported_constants_are_literals(self) -> None:
        assert SUPPORTED_CANONICAL_SYMBOL == "ETH-USDT-PERP"
        assert SUPPORTED_RAW_SYMBOL == "ETHUSDT"
        assert SUPPORTED_INTERVAL == "15m"

    def test_constants_are_strings(self) -> None:
        assert isinstance(SUPPORTED_CANONICAL_SYMBOL, str)
        assert isinstance(SUPPORTED_RAW_SYMBOL, str)
        assert isinstance(SUPPORTED_INTERVAL, str)
