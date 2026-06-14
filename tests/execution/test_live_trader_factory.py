#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_live_trader_factory.py
@Description: Tests for create_live_trader factory — exchange routing and blocking.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.execution.live_trader_factory import create_live_trader


# ======================================================================
# Helpers
# ======================================================================


class FakeTrader:
    """Lightweight stand-in so we never construct a real OKX Trader."""
    broker_exchange_name = "okx"


# ======================================================================
# OKX path
# ======================================================================


class TestOkxPath:
    """EXCHANGE=okx (or missing) returns a Trader."""

    def test_exchange_missing_creates_trader(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "src.execution.live_trader_factory.Trader", FakeTrader,
        )
        trader = create_live_trader({})
        assert isinstance(trader, FakeTrader)

    def test_exchange_okx_creates_trader(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "src.execution.live_trader_factory.Trader", FakeTrader,
        )
        trader = create_live_trader({"EXCHANGE": "okx"})
        assert isinstance(trader, FakeTrader)

    def test_exchange_okx_uppercase(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "src.execution.live_trader_factory.Trader", FakeTrader,
        )
        trader = create_live_trader({"EXCHANGE": "OKX"})
        assert isinstance(trader, FakeTrader)


# ======================================================================
# Binance path — blocked
# ======================================================================


class TestBinancePathBlocked:
    """EXCHANGE=binance raises RuntimeError — not wired yet."""

    def test_binance_raises_runtime_error(self) -> None:
        with pytest.raises(RuntimeError) as exc_info:
            create_live_trader({"EXCHANGE": "binance"})
        msg = str(exc_info.value)
        assert "Binance live trading runtime is not wired yet" in msg
        assert "blocking_reasons=" in msg

    def test_binance_blocked_message_contains_signal_only_hint(self) -> None:
        with pytest.raises(RuntimeError) as exc_info:
            create_live_trader({"EXCHANGE": "binance"})
        msg = str(exc_info.value)
        assert "SIGNAL_ONLY=true" in msg

    def test_binance_blocked_contains_disabled_by_build(self) -> None:
        with pytest.raises(RuntimeError) as exc_info:
            create_live_trader({"EXCHANGE": "binance"})
        msg = str(exc_info.value)
        assert "binance_live_orders_disabled_by_build" in msg

    def test_binance_with_all_confirmations_still_blocked(self) -> None:
        """Even with all preflight env set, orders_globally_enabled=False blocks."""
        env = {
            "EXCHANGE": "binance",
            "SIGNAL_ONLY": "false",
            "LIVE_ENABLED": "true",
            "LIVE_ALLOW_ORDERS": "true",
            "LIVE_CONFIRMATION": "I_UNDERSTAND_BINANCE_LIVE_TRADING",
            "LIVE_MAX_ORDER_NOTIONAL_USDT": "5",
            "LIVE_MAX_POSITION_NOTIONAL_USDT": "20",
            "LIVE_LEVERAGE": "20",
        }
        with pytest.raises(RuntimeError) as exc_info:
            create_live_trader(env)
        msg = str(exc_info.value)
        assert "binance_live_orders_disabled_by_build" in msg


# ======================================================================
# Unsupported exchange
# ======================================================================


class TestUnsupportedExchange:
    """Unsupported EXCHANGE values raise ValueError."""

    def test_unsupported_exchange_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported exchange"):
            create_live_trader({"EXCHANGE": "bybit"})

    def test_unsupported_exchange_message_mentions_supported(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            create_live_trader({"EXCHANGE": "coinbase"})
        msg = str(exc_info.value)
        assert "okx" in msg
        assert "binance" in msg


# ======================================================================
# Source-level safety checks
# ======================================================================


class TestFactoryFileSafety:
    """The factory file must not directly import Binance clients."""

    _FACTORY_PATH: Path = Path("src/execution/live_trader_factory.py")

    def test_factory_file_has_no_binance_client_imports(self) -> None:
        text = self._FACTORY_PATH.read_text()
        assert "BinanceBrokerClient" not in text
        assert "BinanceBrokerSemanticExecutor" not in text

    def test_factory_file_has_no_signing(self) -> None:
        text = self._FACTORY_PATH.read_text()
        assert "build_signed_request" not in text

    def test_factory_file_has_no_order_methods(self) -> None:
        text = self._FACTORY_PATH.read_text()
        assert "place_order" not in text
        assert "cancel_order" not in text

    def test_factory_file_has_no_position_side(self) -> None:
        text = self._FACTORY_PATH.read_text()
        assert "positionSide" not in text

    def test_factory_file_has_no_btc_or_spot(self) -> None:
        text = self._FACTORY_PATH.read_text()
        assert "BTC" not in text
        assert "SPOT" not in text
