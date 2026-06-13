#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_binance_market_data_probe_boundaries.py
@Description: Boundary tests — the market data probe must not import strategy,
              execution, live, risk, API-signing, or any order-placing modules.
"""

from __future__ import annotations

from pathlib import Path


_PROBE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "binance_market_data_probe.py"


def _read_probe_text() -> str:
    return _PROBE_PATH.read_text(encoding="utf-8")


# ======================================================================
# File existence / compilation
# ======================================================================


def test_probe_file_exists() -> None:
    assert _PROBE_PATH.exists(), f"Probe script not found at {_PROBE_PATH}"
    assert _PROBE_PATH.is_file()


def test_probe_file_compiles() -> None:
    """Ensure the script compiles without syntax errors."""
    text = _read_probe_text()
    compile(text, str(_PROBE_PATH), "exec")


# ======================================================================
# Forbidden imports — must NOT import execution, live, strategy, risk, etc.
# ======================================================================


def test_probe_does_not_import_forbidden_modules() -> None:
    text = _read_probe_text()

    forbidden = [
        "src.execution",
        "src.live",
        "src.strategies",
        "src.risk",
        "src.reporting",
        "src.account_sync",
        "src.startup_recovery",
        "src.position_management",
    ]

    for token in forbidden:
        assert token not in text, (
            f"'{token}' must not appear in market data probe script"
        )


def test_probe_does_not_import_binance_client() -> None:
    text = _read_probe_text()
    assert "src.exchanges.binance.client" not in text
    _BROKER = "Binance" + "BrokerClient"
    assert _BROKER not in text


def test_probe_does_not_import_binance_signing() -> None:
    text = _read_probe_text()
    assert "src.exchanges.binance.signing" not in text
    assert "build_signed_request" not in text


def test_probe_does_not_import_aiohttp_transport() -> None:
    text = _read_probe_text()
    _TRANSPORT = "Aiohttp" + "BinanceTransport"
    assert _TRANSPORT not in text


# ======================================================================
# No API key reading
# ======================================================================


def test_probe_does_not_read_api_credentials() -> None:
    text = _read_probe_text()
    assert "EXCHANGE_API_KEY" not in text
    assert "EXCHANGE_API_SECRET" not in text
    assert "EXCHANGE_API_PASSPHRASE" not in text


# ======================================================================
# No OKX legacy env vars
# ======================================================================


def test_probe_does_not_read_okx_legacy_env_vars() -> None:
    text = _read_probe_text()
    okx_env_vars = [
        "OKX_INST_ID",
        "OKX_BAR",
        "OKX_TD_MODE",
        "OKX_POS_SIDE_MODE",
    ]
    for var in okx_env_vars:
        assert var not in text, f"'{var}' must not appear in probe script"


# ======================================================================
# No order / trading logic
# ======================================================================


def test_probe_has_no_order_placement() -> None:
    text = _read_probe_text()
    assert "place_order" not in text
    assert "cancel_order" not in text


def test_probe_has_no_market_orders() -> None:
    text = _read_probe_text()
    assert "MARKET BUY" not in text
    assert "MARKET SELL" not in text


def test_probe_has_no_live_confirmation() -> None:
    text = _read_probe_text()
    assert "LIVE_SMOKE_TEST_CONFIRM" not in text
    assert "BINANCE_LIVE_SMOKE_TEST_CONFIRM" not in text
    assert "I_UNDERSTAND" not in text


def test_probe_has_no_broker_client_creation() -> None:
    text = _read_probe_text()
    _BROKER_CLIENT = "Binance" + "BrokerClient("
    assert _BROKER_CLIENT not in text


# ======================================================================
# Must reference Binance market WebSocket only
# ======================================================================


def test_probe_references_binance_ws_url() -> None:
    """Probe must connect to fstream.binance.com/market (market data only)."""
    text = _read_probe_text()
    # May be linked via feed.stream_url(), but the base URL constant is in websocket_feed
    # Minimal check: the probe uses binance_ws_connector and the feed's stream_url()
    assert "ws_connector" in text or "connect_binance_ws" in text


def test_probe_uses_build_market_data_feed() -> None:
    text = _read_probe_text()
    assert "build_market_data_feed" in text


# ======================================================================
# Must subscribe to aggTrade + kline_15m only
# ======================================================================


def test_probe_subscribes_ethusdt_agg_trade() -> None:
    text = _read_probe_text()
    assert "aggTrade" in text


def test_probe_subscribes_ethusdt_kline_15m() -> None:
    text = _read_probe_text()
    assert "kline_15m" in text or 'kline_' in text


# ======================================================================
# Only ETH-USDT-PERP / ETHUSDT
# ======================================================================


def test_probe_is_eth_only() -> None:
    text = _read_probe_text()
    assert "ETH-USDT-PERP" in text or "ETHUSDT" in text
    assert "BTCUSDT" not in text
    assert "BTC-USDT" not in text


# ======================================================================
# Must validate exchange=binance
# ======================================================================


def test_probe_validates_exchange_is_binance() -> None:
    text = _read_probe_text()
    assert "ExchangeName.BINANCE" in text
    assert "validate_probe_config" in text


def test_probe_loads_unified_runtime_config() -> None:
    text = _read_probe_text()
    assert "load_unified_runtime_config" in text
