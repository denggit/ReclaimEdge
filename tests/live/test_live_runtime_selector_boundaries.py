#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_live_runtime_selector_boundaries.py
@Description: Boundary tests — the live runtime selector must NOT import
              strategy, execution, broker, signing, order-placing, or secret
              reading.  It must contain the expected public symbols.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ======================================================================
# Path to selector source
# ======================================================================

_SELECTOR_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "live"
    / "live_runtime_selector.py"
)


def _read_selector_text() -> str:
    return _SELECTOR_PATH.read_text(encoding="utf-8")


# ======================================================================
# File existence / compilation
# ======================================================================


def test_selector_file_exists() -> None:
    assert _SELECTOR_PATH.exists(), f"Selector file not found at {_SELECTOR_PATH}"
    assert _SELECTOR_PATH.is_file()


def test_selector_file_compiles() -> None:
    text = _read_selector_text()
    compile(text, str(_SELECTOR_PATH), "exec")


# ======================================================================
# Forbidden — execution / strategy / exchange / order placement
# ======================================================================


class TestForbiddenImports:
    """The selector must NOT import forbidden modules or symbols."""

    def test_does_not_import_execution(self) -> None:
        text = _read_selector_text()
        assert "src.execution" not in text

    def test_does_not_import_strategies(self) -> None:
        text = _read_selector_text()
        assert "src.strategies" not in text

    def test_does_not_import_exchanges(self) -> None:
        text = _read_selector_text()
        assert "src.exchanges" not in text

    def test_does_not_import_monitors(self) -> None:
        text = _read_selector_text()
        assert "src.monitors" not in text

    def test_does_not_import_live_workers(self) -> None:
        text = _read_selector_text()
        assert "src.live.workers" not in text

    def test_does_not_import_trader(self) -> None:
        text = _read_selector_text()
        assert "Trader" not in text

    def test_does_not_contain_place_order(self) -> None:
        text = _read_selector_text()
        assert "place_order" not in text

    def test_does_not_contain_cancel_order(self) -> None:
        text = _read_selector_text()
        assert "cancel_order" not in text


# ======================================================================
# Forbidden — secret / API key references
# ======================================================================


class TestNoApiKeyOrSecrets:
    """The selector must NOT reference any API keys or secrets."""

    def test_no_exchange_api_key(self) -> None:
        text = _read_selector_text()
        assert "EXCHANGE_API_KEY" not in text

    def test_no_exchange_api_secret(self) -> None:
        text = _read_selector_text()
        assert "EXCHANGE_API_SECRET" not in text

    def test_no_exchange_api_passphrase(self) -> None:
        text = _read_selector_text()
        assert "EXCHANGE_API_PASSPHRASE" not in text

    def test_no_binance_api_key(self) -> None:
        text = _read_selector_text()
        assert "BINANCE_API_KEY" not in text

    def test_no_binance_secret_key(self) -> None:
        text = _read_selector_text()
        assert "BINANCE_SECRET_KEY" not in text


# ======================================================================
# Forbidden — BTC / SPOT / positionSide
# ======================================================================


class TestNoBtcOrSpot:
    """The selector must NOT reference BTC or spot trading."""

    def test_no_btc(self) -> None:
        text = _read_selector_text()
        assert "BTC" not in text

    def test_no_spot(self) -> None:
        text = _read_selector_text()
        assert "SPOT" not in text

    def test_no_position_side(self) -> None:
        text = _read_selector_text()
        assert "positionSide" not in text


# ======================================================================
# Required — must contain expected public symbols
# ======================================================================


class TestRequiredSymbols:
    """The selector source must contain the expected public API."""

    def test_contains_live_runtime_kind(self) -> None:
        text = _read_selector_text()
        assert "LiveRuntimeKind" in text

    def test_contains_live_runtime_selection(self) -> None:
        text = _read_selector_text()
        assert "LiveRuntimeSelection" in text

    def test_contains_select_live_runtime(self) -> None:
        text = _read_selector_text()
        assert "select_live_runtime" in text

    def test_contains_binance_signal_only_key(self) -> None:
        text = _read_selector_text()
        assert "BINANCE_SIGNAL_ONLY" in text


# ======================================================================
# Side-effect free — no network, no dotenv, no queues
# ======================================================================


class TestNoSideEffects:
    """The selector must be a pure module with no side effects."""

    def test_no_load_dotenv(self) -> None:
        text = _read_selector_text()
        assert "load_dotenv" not in text

    def test_no_asyncio_queue(self) -> None:
        text = _read_selector_text()
        assert "asyncio" not in text
        assert "Queue" not in text

    def test_no_http_or_websocket(self) -> None:
        text = _read_selector_text()
        assert "aiohttp" not in text
        assert "websocket" not in text.lower()
        assert "httpx" not in text

    def test_does_not_import_config(self) -> None:
        text = _read_selector_text()
        assert "from config" not in text

    def test_does_not_import_binance_signal_only_runtime(self) -> None:
        text = _read_selector_text()
        assert "binance_signal_only_runtime" not in text
