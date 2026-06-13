#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_binance_market_data_bridge.py
@Description: Functional tests for the Binance signal-only market data bridge.

All tests use in-memory canonical event objects — no network, no env, no API keys.
"""

from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path
from unittest import mock

import pytest

from src.data_feed.market_events import (
    MarketCandleEvent,
    MarketTradeEvent,
    MarketTradeSide,
)
from src.exchanges.models import ExchangeName
from src.live.binance_market_data_bridge import (
    SUPPORTED_CANONICAL_SYMBOL,
    SUPPORTED_INTERVAL,
    SUPPORTED_RAW_SYMBOL,
    BinanceMarketDataSignalBridge,
    BinanceSignalBridgeStats,
    BinanceSignalCandleInput,
    BinanceSignalTradeInput,
)

# ======================================================================
# Helpers
# ======================================================================


def _make_trade_event(
    *,
    canonical_symbol: str = SUPPORTED_CANONICAL_SYMBOL,
    raw_symbol: str = SUPPORTED_RAW_SYMBOL,
    price: str = "3000.00",
    quantity: str = "1.5",
    side: MarketTradeSide = MarketTradeSide.BUY,
    event_time_ms: int = 1700000000000,
) -> MarketTradeEvent:
    return MarketTradeEvent(
        exchange=ExchangeName.BINANCE,
        canonical_symbol=canonical_symbol,
        raw_symbol=raw_symbol,
        price=Decimal(price),
        quantity=Decimal(quantity),
        taker_side=side,
        event_time_ms=event_time_ms,
        is_aggregated=True,
        aggregation_window_ms=100,
    )


def _make_candle_event(
    *,
    canonical_symbol: str = SUPPORTED_CANONICAL_SYMBOL,
    raw_symbol: str = SUPPORTED_RAW_SYMBOL,
    timeframe: str = SUPPORTED_INTERVAL,
    open_price: str = "2990.00",
    high_price: str = "3020.00",
    low_price: str = "2980.00",
    close_price: str = "3010.00",
    volume: str = "100.5",
    is_closed: bool = False,
    open_time_ms: int = 1700000000000,
    close_time_ms: int = 1700000900000,
) -> MarketCandleEvent:
    return MarketCandleEvent(
        exchange=ExchangeName.BINANCE,
        canonical_symbol=canonical_symbol,
        raw_symbol=raw_symbol,
        timeframe=timeframe,
        open_time_ms=open_time_ms,
        close_time_ms=close_time_ms,
        open_price=Decimal(open_price),
        high_price=Decimal(high_price),
        low_price=Decimal(low_price),
        close_price=Decimal(close_price),
        volume=Decimal(volume),
        is_closed=is_closed,
    )


# ======================================================================
# Bridge construction
# ======================================================================


class TestBridgeConstruction:
    """Tests for ``BinanceMarketDataSignalBridge.__init__``."""

    def test_default_construction(self) -> None:
        bridge = BinanceMarketDataSignalBridge()
        assert bridge.canonical_symbol == SUPPORTED_CANONICAL_SYMBOL
        assert bridge.raw_symbol == SUPPORTED_RAW_SYMBOL
        assert bridge.interval == SUPPORTED_INTERVAL

    def test_explicit_params_accepted(self) -> None:
        bridge = BinanceMarketDataSignalBridge(
            canonical_symbol="ETH-USDT-PERP",
            raw_symbol="ETHUSDT",
            interval="15m",
        )
        assert bridge.canonical_symbol == "ETH-USDT-PERP"
        assert bridge.raw_symbol == "ETHUSDT"
        assert bridge.interval == "15m"

    def test_wrong_canonical_symbol_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported canonical_symbol"):
            BinanceMarketDataSignalBridge(canonical_symbol="BTC-USDT-PERP")

    def test_wrong_raw_symbol_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported raw_symbol"):
            BinanceMarketDataSignalBridge(raw_symbol="BTCUSDT")

    def test_wrong_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported interval"):
            BinanceMarketDataSignalBridge(interval="1m")

    def test_interval_5m_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported interval"):
            BinanceMarketDataSignalBridge(interval="5m")


# ======================================================================
# Trade event → BinanceSignalTradeInput
# ======================================================================


class TestTradeEventToSignal:
    """Tests that ``MarketTradeEvent`` → ``BinanceSignalTradeInput`` mapping works."""

    def test_basic_trade_conversion(self) -> None:
        bridge = BinanceMarketDataSignalBridge()
        event = _make_trade_event(price="3000.00", quantity="1.5", side=MarketTradeSide.BUY)

        result = bridge.handle_event(event)
        assert isinstance(result, BinanceSignalTradeInput)
        assert result.canonical_symbol == SUPPORTED_CANONICAL_SYMBOL
        assert result.raw_symbol == SUPPORTED_RAW_SYMBOL
        assert result.timestamp_ms == 1700000000000
        assert result.side == "BUY"
        assert result.price == Decimal("3000.00")
        assert result.quantity == Decimal("1.5")
        assert result.source == "binance_agg_trade"

    def test_sell_side_preserved(self) -> None:
        bridge = BinanceMarketDataSignalBridge()
        event = _make_trade_event(side=MarketTradeSide.SELL)

        result = bridge.handle_event(event)
        assert isinstance(result, BinanceSignalTradeInput)
        assert result.side == "SELL"

    def test_timestamp_preserved(self) -> None:
        bridge = BinanceMarketDataSignalBridge()
        event = _make_trade_event(event_time_ms=1718234567890)

        result = bridge.handle_event(event)
        assert isinstance(result, BinanceSignalTradeInput)
        assert result.timestamp_ms == 1718234567890

    def test_price_is_decimal(self) -> None:
        bridge = BinanceMarketDataSignalBridge()
        event = _make_trade_event(price="2999.99")

        result = bridge.handle_event(event)
        assert isinstance(result, BinanceSignalTradeInput)
        assert isinstance(result.price, Decimal)
        assert result.price == Decimal("2999.99")

    def test_quantity_is_decimal(self) -> None:
        bridge = BinanceMarketDataSignalBridge()
        event = _make_trade_event(quantity="0.001")

        result = bridge.handle_event(event)
        assert isinstance(result, BinanceSignalTradeInput)
        assert isinstance(result.quantity, Decimal)
        assert result.quantity == Decimal("0.001")

    def test_trade_increments_trade_events_stat(self) -> None:
        bridge = BinanceMarketDataSignalBridge()
        assert bridge.stats.trade_events == 0

        bridge.handle_event(_make_trade_event())
        assert bridge.stats.trade_events == 1

        bridge.handle_event(_make_trade_event())
        assert bridge.stats.trade_events == 2

    def test_trade_does_not_increment_candle_stats(self) -> None:
        bridge = BinanceMarketDataSignalBridge()
        bridge.handle_event(_make_trade_event())

        assert bridge.stats.candle_events == 0
        assert bridge.stats.closed_candle_events == 0


# ======================================================================
# Candle event → BinanceSignalCandleInput
# ======================================================================


class TestCandleEventToSignal:
    """Tests that ``MarketCandleEvent`` → ``BinanceSignalCandleInput`` mapping works."""

    def test_basic_candle_conversion(self) -> None:
        bridge = BinanceMarketDataSignalBridge()
        event = _make_candle_event(
            open_price="2990.00",
            high_price="3020.00",
            low_price="2980.00",
            close_price="3010.00",
            volume="100.5",
            is_closed=False,
        )

        result = bridge.handle_event(event)
        assert isinstance(result, BinanceSignalCandleInput)
        assert result.canonical_symbol == SUPPORTED_CANONICAL_SYMBOL
        assert result.raw_symbol == SUPPORTED_RAW_SYMBOL
        assert result.interval == SUPPORTED_INTERVAL
        assert result.timestamp_ms == 1700000000000
        assert result.open == Decimal("2990.00")
        assert result.high == Decimal("3020.00")
        assert result.low == Decimal("2980.00")
        assert result.close == Decimal("3010.00")
        assert result.volume == Decimal("100.5")
        assert result.closed is False
        assert result.source == "binance_kline"

    def test_closed_false_preserved(self) -> None:
        bridge = BinanceMarketDataSignalBridge()
        event = _make_candle_event(is_closed=False)

        result = bridge.handle_event(event)
        assert isinstance(result, BinanceSignalCandleInput)
        assert result.closed is False

    def test_closed_true_preserved(self) -> None:
        bridge = BinanceMarketDataSignalBridge()
        event = _make_candle_event(is_closed=True)

        result = bridge.handle_event(event)
        assert isinstance(result, BinanceSignalCandleInput)
        assert result.closed is True

    def test_candle_increments_candle_events_stat(self) -> None:
        bridge = BinanceMarketDataSignalBridge()
        assert bridge.stats.candle_events == 0

        bridge.handle_event(_make_candle_event(is_closed=False))
        assert bridge.stats.candle_events == 1

        bridge.handle_event(_make_candle_event(is_closed=True))
        assert bridge.stats.candle_events == 2

    def test_closed_candle_increments_closed_stat(self) -> None:
        bridge = BinanceMarketDataSignalBridge()
        assert bridge.stats.closed_candle_events == 0

        bridge.handle_event(_make_candle_event(is_closed=False))
        assert bridge.stats.closed_candle_events == 0

        bridge.handle_event(_make_candle_event(is_closed=True))
        assert bridge.stats.closed_candle_events == 1

        bridge.handle_event(_make_candle_event(is_closed=True))
        assert bridge.stats.closed_candle_events == 2

    def test_candle_does_not_increment_trade_stats(self) -> None:
        bridge = BinanceMarketDataSignalBridge()
        bridge.handle_event(_make_candle_event(is_closed=True))

        assert bridge.stats.trade_events == 0

    def test_all_ohlcv_fields_are_decimal(self) -> None:
        bridge = BinanceMarketDataSignalBridge()
        event = _make_candle_event()

        result = bridge.handle_event(event)
        assert isinstance(result, BinanceSignalCandleInput)
        assert isinstance(result.open, Decimal)
        assert isinstance(result.high, Decimal)
        assert isinstance(result.low, Decimal)
        assert isinstance(result.close, Decimal)
        assert isinstance(result.volume, Decimal)


# ======================================================================
# Stats
# ======================================================================


class TestBridgeStats:
    """Tests for ``BinanceSignalBridgeStats`` accumulation."""

    def test_initial_stats_are_all_zero(self) -> None:
        bridge = BinanceMarketDataSignalBridge()
        stats = bridge.stats
        assert stats.trade_events == 0
        assert stats.candle_events == 0
        assert stats.closed_candle_events == 0
        assert stats.ignored_events == 0
        assert stats.error_events == 0

    def test_mixed_events_cumulative_stats(self) -> None:
        bridge = BinanceMarketDataSignalBridge()

        bridge.handle_event(_make_trade_event())
        bridge.handle_event(_make_candle_event(is_closed=False))
        bridge.handle_event(_make_trade_event())
        bridge.handle_event(_make_candle_event(is_closed=True))
        bridge.handle_event(_make_candle_event(is_closed=True))

        stats = bridge.stats
        assert stats.trade_events == 2
        assert stats.candle_events == 3
        assert stats.closed_candle_events == 2
        assert stats.ignored_events == 0
        assert stats.error_events == 0

    def test_get_stats_returns_same_object(self) -> None:
        bridge = BinanceMarketDataSignalBridge()
        bridge.handle_event(_make_trade_event())
        s1 = bridge.stats
        s2 = bridge.get_stats()
        assert s1 is s2
        assert s1.trade_events == 1


# ======================================================================
# Unknown / mismatched events
# ======================================================================


class TestUnknownAndMismatchedEvents:
    """Tests that unmatched events return None and update ignored_events."""

    def test_trade_wrong_canonical_symbol_returns_none(self) -> None:
        bridge = BinanceMarketDataSignalBridge()
        event = _make_trade_event(canonical_symbol="BTC-USDT-PERP")

        result = bridge.handle_event(event)
        assert result is None
        assert bridge.stats.ignored_events == 1

    def test_trade_wrong_raw_symbol_returns_none(self) -> None:
        bridge = BinanceMarketDataSignalBridge()
        event = _make_trade_event(raw_symbol="BTCUSDT")

        result = bridge.handle_event(event)
        assert result is None
        assert bridge.stats.ignored_events == 1

    def test_candle_wrong_canonical_symbol_returns_none(self) -> None:
        bridge = BinanceMarketDataSignalBridge()
        event = _make_candle_event(canonical_symbol="BTC-USDT-PERP")

        result = bridge.handle_event(event)
        assert result is None
        assert bridge.stats.ignored_events == 1

    def test_candle_wrong_raw_symbol_returns_none(self) -> None:
        bridge = BinanceMarketDataSignalBridge()
        event = _make_candle_event(raw_symbol="BTCUSDT")

        result = bridge.handle_event(event)
        assert result is None
        assert bridge.stats.ignored_events == 1

    def test_candle_wrong_interval_returns_none(self) -> None:
        bridge = BinanceMarketDataSignalBridge()
        event = _make_candle_event(timeframe="1m")

        result = bridge.handle_event(event)
        assert result is None
        assert bridge.stats.ignored_events == 1

    def test_ignored_event_does_not_increment_trade_or_candle(self) -> None:
        bridge = BinanceMarketDataSignalBridge()
        bridge.handle_event(_make_trade_event(raw_symbol="BTCUSDT"))

        assert bridge.stats.trade_events == 0
        assert bridge.stats.candle_events == 0
        assert bridge.stats.ignored_events == 1


# ======================================================================
# No env / API key reading
# ======================================================================


class TestDoesNotReadEnvOrApiKey:
    """Verify the bridge does not read API credentials or environment variables."""

    def test_bridge_does_not_read_api_key_env(self) -> None:
        """The bridge source must NOT reference any API credential env vars."""
        src = (
            Path(__file__)
            .resolve()
            .parents[2]
            / "src"
            / "live"
            / "binance_market_data_bridge.py"
        ).read_text()

        forbidden = [
            "EXCHANGE_API_KEY",
            "EXCHANGE_API_SECRET",
            "EXCHANGE_API_PASSPHRASE",
            "BINANCE_API_KEY",
            "BINANCE_SECRET_KEY",
            "os.environ",
            "os.getenv",
            "load_dotenv",
        ]
        for token in forbidden:
            assert token not in src, (
                f"Bridge source must not reference '{token}'"
            )

    def test_bridge_does_not_read_api_credentials_dynamically(self) -> None:
        """Even with API key set in env, bridge works fine."""
        env = {
            "EXCHANGE_API_KEY": "fake-key",
            "EXCHANGE_API_SECRET": "fake-secret",
            "EXCHANGE_API_PASSPHRASE": "fake-passphrase",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            bridge = BinanceMarketDataSignalBridge()
            result = bridge.handle_event(_make_trade_event())
            assert isinstance(result, BinanceSignalTradeInput)


# ======================================================================
# Dataclass immutability / dunder methods
# ======================================================================


class TestDtoImmutability:
    """DTOs should be frozen and support equality."""

    def test_trade_input_is_frozen(self) -> None:
        a = BinanceSignalTradeInput(
            canonical_symbol="ETH-USDT-PERP",
            raw_symbol="ETHUSDT",
            timestamp_ms=1000,
            side="BUY",
            price=Decimal("3000"),
            quantity=Decimal("1"),
        )
        with pytest.raises(Exception):
            a.side = "SELL"  # type: ignore[misc]

    def test_candle_input_is_frozen(self) -> None:
        a = BinanceSignalCandleInput(
            canonical_symbol="ETH-USDT-PERP",
            raw_symbol="ETHUSDT",
            interval="15m",
            timestamp_ms=1000,
            open=Decimal("3000"),
            high=Decimal("3010"),
            low=Decimal("2990"),
            close=Decimal("3005"),
            volume=Decimal("100"),
            closed=False,
        )
        with pytest.raises(Exception):
            a.closed = True  # type: ignore[misc]

    def test_trade_input_equality(self) -> None:
        a = BinanceSignalTradeInput("ETH-USDT-PERP", "ETHUSDT", 1000, "BUY", Decimal("3000"), Decimal("1"))
        b = BinanceSignalTradeInput("ETH-USDT-PERP", "ETHUSDT", 1000, "BUY", Decimal("3000"), Decimal("1"))
        assert a == b

    def test_candle_input_equality(self) -> None:
        a = BinanceSignalCandleInput("ETH-USDT-PERP", "ETHUSDT", "15m", 1000,
                                     Decimal("3000"), Decimal("3010"), Decimal("2990"),
                                     Decimal("3005"), Decimal("100"), False)
        b = BinanceSignalCandleInput("ETH-USDT-PERP", "ETHUSDT", "15m", 1000,
                                     Decimal("3000"), Decimal("3010"), Decimal("2990"),
                                     Decimal("3005"), Decimal("100"), False)
        assert a == b


# ======================================================================
# Exports
# ======================================================================


class TestModuleExports:
    """Verify the bridge module exports the expected public names."""

    def test_expected_exports(self) -> None:
        from src.live import binance_market_data_bridge as m

        expected = {
            "BinanceMarketDataSignalBridge",
            "BinanceSignalBridgeStats",
            "BinanceSignalCandleInput",
            "BinanceSignalTradeInput",
            "BinanceSignalInput",
            "SUPPORTED_CANONICAL_SYMBOL",
            "SUPPORTED_RAW_SYMBOL",
            "SUPPORTED_INTERVAL",
        }
        for name in expected:
            assert hasattr(m, name), f"Module must export {name}"
