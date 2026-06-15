#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_binance_live_preflight_boundaries.py
@Description: Boundary tests — the Binance live preflight module must NOT import
              strategy, execution, broker, signing, order-placing, or secret
              reading.  It must contain the expected public symbols.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ======================================================================
# Path to preflight source
# ======================================================================

_PREFLIGHT_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "exchanges"
    / "binance"
    / "live_preflight.py"
)


def _read_preflight_text() -> str:
    return _PREFLIGHT_PATH.read_text(encoding="utf-8")


# ======================================================================
# File existence / compilation
# ======================================================================


def test_preflight_file_exists() -> None:
    assert _PREFLIGHT_PATH.exists(), f"Preflight file not found at {_PREFLIGHT_PATH}"
    assert _PREFLIGHT_PATH.is_file()


def test_preflight_file_compiles() -> None:
    text = _read_preflight_text()
    compile(text, str(_PREFLIGHT_PATH), "exec")


# ======================================================================
# Forbidden — execution / strategy / exchange / order placement
# ======================================================================


class TestForbiddenImports:
    """The preflight must NOT import forbidden modules or symbols."""

    def test_does_not_import_src_execution(self) -> None:
        text = _read_preflight_text()
        assert "src.execution" not in text

    def test_does_not_import_src_strategies(self) -> None:
        text = _read_preflight_text()
        assert "src.strategies" not in text

    def test_does_not_import_exchange_adapter_internals(self) -> None:
        """The canonical preflight module must NOT import exchange adapter internals
        (broker client, semantic executor, or OKX paths)."""
        text = _read_preflight_text()
        assert "src.exchanges.binance.client" not in text
        assert "src.exchanges.binance.semantic_executor" not in text
        assert "src.exchanges.binance.runtime_adapter" not in text
        assert "src.exchanges.okx" not in text

    def test_does_not_import_binance_client(self) -> None:
        text = _read_preflight_text()
        assert "src.exchanges.binance.client" not in text
        assert "binance.client" not in text

    def test_does_not_import_semantic_executor(self) -> None:
        text = _read_preflight_text()
        assert "src.exchanges.binance.semantic_executor" not in text
        # class name checked in separate test below

    def test_does_not_import_signing(self) -> None:
        text = _read_preflight_text()
        assert "src.exchanges.binance.signing" not in text
        # build_signed_request checked in separate test below

    def test_does_not_contain_binance_broker_client(self) -> None:
        text = _read_preflight_text()
        assert "BinanceBrokerClient" not in text

    def test_does_not_contain_semantic_executor_class(self) -> None:
        text = _read_preflight_text()
        assert "BinanceBrokerSemanticExecutor" not in text

    def test_does_not_contain_build_signed_request(self) -> None:
        text = _read_preflight_text()
        assert "build_signed_request" not in text

    def test_does_not_contain_place_order(self) -> None:
        text = _read_preflight_text()
        assert "place_order" not in text

    def test_does_not_contain_cancel_order(self) -> None:
        text = _read_preflight_text()
        assert "cancel_order" not in text

    def test_does_not_contain_fetch_position(self) -> None:
        text = _read_preflight_text()
        assert "fetch_position" not in text

    def test_does_not_contain_fetch_open_orders(self) -> None:
        text = _read_preflight_text()
        assert "fetch_open_orders" not in text


# ======================================================================
# Forbidden — positionSide / BTC / SPOT
# ======================================================================


class TestNoBtcOrSpot:
    """The preflight must NOT reference BTC, SPOT, or positionSide."""

    def test_no_position_side(self) -> None:
        text = _read_preflight_text()
        assert "positionSide" not in text

    def test_no_btc(self) -> None:
        text = _read_preflight_text()
        assert "BTC" not in text

    def test_no_spot(self) -> None:
        text = _read_preflight_text()
        assert "SPOT" not in text


# ======================================================================
# Forbidden — secret / API key references
# ======================================================================


class TestNoApiKeyOrSecrets:
    """The preflight must NOT reference any API keys or secrets."""

    def test_no_exchange_api_key(self) -> None:
        text = _read_preflight_text()
        assert "EXCHANGE_API_KEY" not in text

    def test_no_exchange_api_secret(self) -> None:
        text = _read_preflight_text()
        assert "EXCHANGE_API_SECRET" not in text

    def test_no_exchange_api_passphrase(self) -> None:
        text = _read_preflight_text()
        assert "EXCHANGE_API_PASSPHRASE" not in text

    def test_no_binance_api_key(self) -> None:
        text = _read_preflight_text()
        assert "BINANCE_API_KEY" not in text

    def test_no_binance_secret_key(self) -> None:
        text = _read_preflight_text()
        assert "BINANCE_SECRET_KEY" not in text


# ======================================================================
# Required — must contain expected public symbols
# ======================================================================


class TestRequiredSymbols:
    """The preflight source must contain the expected public API."""

    def test_contains_config_class(self) -> None:
        text = _read_preflight_text()
        assert "BinanceLivePreflightConfig" in text

    def test_contains_report_class(self) -> None:
        text = _read_preflight_text()
        assert "BinanceLivePreflightReport" in text

    def test_contains_build_report(self) -> None:
        text = _read_preflight_text()
        assert "build_binance_live_preflight_report" in text

    def test_contains_format_message(self) -> None:
        text = _read_preflight_text()
        assert "format_binance_live_blocked_message" in text

    def test_contains_confirmation_constant(self) -> None:
        text = _read_preflight_text()
        assert "BINANCE_LIVE_CONFIRMATION" in text


# ======================================================================
# Side-effect free — no network, no dotenv, no queues
# ======================================================================


class TestNoSideEffects:
    """The preflight must be a pure module with no side effects."""

    def test_no_load_dotenv(self) -> None:
        text = _read_preflight_text()
        assert "load_dotenv" not in text

    def test_no_asyncio_queue(self) -> None:
        text = _read_preflight_text()
        assert "asyncio" not in text
        assert "Queue" not in text

    def test_no_http_or_websocket(self) -> None:
        text = _read_preflight_text()
        assert "aiohttp" not in text
        assert "websocket" not in text.lower()
        assert "httpx" not in text

    def test_does_not_import_config(self) -> None:
        text = _read_preflight_text()
        assert "from config" not in text
