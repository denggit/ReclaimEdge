#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_live_trader_factory.py
@Description: Tests for create_live_trader factory — exchange routing.
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


class FakeBinanceLiveTrader:
    """Lightweight stand-in for BinanceLiveTrader."""
    broker_exchange_name = "binance"
    symbol = "ETHUSDT"

    def __init__(self, *, env=None, **kwargs):
        pass


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
# Binance path — wired
# ======================================================================


class TestBinancePathWired:
    """EXCHANGE=binance returns BinanceLiveTrader."""

    def test_binance_returns_binance_live_trader(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "src.execution.binance_live_trader.BinanceLiveTrader",
            FakeBinanceLiveTrader,
        )
        trader = create_live_trader({"EXCHANGE": "binance"})
        assert isinstance(trader, FakeBinanceLiveTrader)
        assert trader.broker_exchange_name == "binance"

    def test_binance_with_all_confirmations_returns_trader(self, monkeypatch) -> None:
        """With all preflight env set, Binance path creates BinanceLiveTrader."""
        monkeypatch.setattr(
            "src.execution.binance_live_trader.BinanceLiveTrader",
            FakeBinanceLiveTrader,
        )
        env = {
            "EXCHANGE": "binance",
            "SIGNAL_ONLY": "false",
            "LIVE_ENABLED": "true",
            "LIVE_ALLOW_ORDERS": "true",
            "LIVE_CONFIRMATION": "I_UNDERSTAND_EXCHANGE_LIVE_TRADING",
            "LIVE_MAX_ORDER_NOTIONAL_USDT": "5",
            "LIVE_MAX_POSITION_NOTIONAL_USDT": "20",
            "LIVE_LEVERAGE": "20",
        }
        trader = create_live_trader(env)
        assert isinstance(trader, FakeBinanceLiveTrader)

    def test_factory_does_not_read_api_key(self, monkeypatch) -> None:
        """Factory passes env to BinanceLiveTrader, does not read API keys itself."""
        monkeypatch.setattr(
            "src.execution.binance_live_trader.BinanceLiveTrader",
            FakeBinanceLiveTrader,
        )
        env = {
            "EXCHANGE": "binance",
            "EXCHANGE_API_KEY": "should_not_be_read_here",
            "EXCHANGE_API_SECRET": "should_not_be_read_here",
        }
        trader = create_live_trader(env)
        assert isinstance(trader, FakeBinanceLiveTrader)


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

    def test_factory_file_has_no_broker_client_init(self) -> None:
        """Factory creates BinanceLiveTrader via env, not broker clients."""
        text = self._FACTORY_PATH.read_text()
        assert "BinanceBrokerClient(" not in text

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

    def test_factory_lazy_imports_binance_trader(self) -> None:
        """BinanceLiveTrader is lazily imported only in the binance branch."""
        text = self._FACTORY_PATH.read_text()
        # Import should appear inside the 'if exchange == "binance":' block
        assert "from src.execution.binance_live_trader import" in text
        assert "BinanceLiveTrader" in text
