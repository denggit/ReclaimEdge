#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_binance_market_data_probe.py
@Description: Unit tests for the Binance market data probe script.

All tests use mocked / fake transports — no real network calls.
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest import mock

import pytest

from scripts.binance_market_data_probe import (
    CANONICAL_SYMBOL,
    BINANCE_SYMBOL,
    DEFAULT_MAX_EVENTS,
    DEFAULT_PROBE_SECONDS,
    ENV_MAX_EVENTS,
    ENV_PROBE_SECONDS,
    connect_binance_ws,
    read_max_events,
    read_probe_duration,
    run_probe,
    validate_probe_config,
)
from src.data_feed.market_events import (
    MarketCandleEvent,
    MarketTradeEvent,
    MarketTradeSide,
)
from src.data_feed.selector import build_market_data_feed
from src.exchanges.models import ExchangeName
from src.exchanges.runtime_config import (
    ExchangeRuntimeConfig,
    load_unified_runtime_config,
)

# ======================================================================
# Helpers — build minimal valid Binance event payloads
# ======================================================================


def make_agg_trade_payload(
    *,
    price: str = "3000.00",
    quantity: str = "1.5",
    event_time_ms: int = 1700000000000,
    trade_time_ms: int = 1700000000000,
    maker_side: bool = False,
    symbol: str = "ETHUSDT",
) -> dict[str, Any]:
    """Build a minimal Binance aggTrade payload (raw, without combined-stream wrapper)."""
    return {
        "e": "aggTrade",
        "E": event_time_ms,
        "s": symbol,
        "a": 12345,
        "p": price,
        "q": quantity,
        "f": 100,
        "l": 105,
        "T": trade_time_ms,
        "m": maker_side,
    }


def make_kline_payload(
    *,
    symbol: str = "ETHUSDT",
    interval: str = "15m",
    open_price: str = "2990.00",
    close_price: str = "3010.00",
    high_price: str = "3020.00",
    low_price: str = "2980.00",
    volume: str = "100.5",
    is_closed: bool = False,
    open_time_ms: int = 1700000000000,
    close_time_ms: int = 1700000900000,
    event_time_ms: int = 1700000900000,
) -> dict[str, Any]:
    """Build a minimal Binance kline payload (raw, without combined-stream wrapper)."""
    return {
        "e": "kline",
        "E": event_time_ms,
        "s": symbol,
        "k": {
            "t": open_time_ms,
            "T": close_time_ms,
            "s": symbol,
            "i": interval,
            "f": 100,
            "L": 200,
            "o": open_price,
            "c": close_price,
            "h": high_price,
            "l": low_price,
            "v": volume,
            "n": 1000,
            "x": is_closed,
            "q": "300000.00",
            "V": "50.0",
            "Q": "150000.00",
            "B": "0",
        },
    }


def wrap_payload(raw_payload: dict[str, Any], stream_name: str) -> dict[str, Any]:
    """Wrap a raw payload in the Binance combined-stream envelope."""
    return {"stream": stream_name, "data": raw_payload}


def make_valid_binance_rt(**overrides: Any) -> Any:
    """Return an ExchangeRuntimeConfig that passes probe validation."""
    env = {
        "EXCHANGE": "binance",
        "TRADE_ASSET": "ETH",
        "QUOTE_ASSET": "USDT",
        "MARKET_TYPE": "PERPETUAL",
        "MARGIN_MODE": "isolated",
        "POSITION_MODE": "net",
        "LEVERAGE": "20",
        "KLINE_INTERVAL": "15m",
    }
    env.update(overrides)
    return load_unified_runtime_config(env=env)


def build_feed_for_tests() -> Any:
    """Return a BinanceWebSocketMarketDataFeed via build_market_data_feed."""
    return build_market_data_feed(
        exchange=ExchangeName.BINANCE,
        canonical_symbol=CANONICAL_SYMBOL,
        raw_symbol=BINANCE_SYMBOL,
        kline_interval="15m",
        binance_ws_connector=connect_binance_ws,
    )


# ======================================================================
# Fake WebSocket connection for run_probe tests
# ======================================================================


class FakeBinanceWsConnection:
    """Fake async iterator that yields preloaded JSON strings and then stops.

    Compatible with the AiohttpBinanceWsConnection interface used by run_probe
    (__aiter__ / __anext__ yielding str, plus a close() method).
    """

    def __init__(self, json_strings: list[str]) -> None:
        self._strings = json_strings
        self._idx = 0

    def __aiter__(self) -> "FakeBinanceWsConnection":
        return self

    async def __anext__(self) -> str:
        if self._idx >= len(self._strings):
            raise StopAsyncIteration
        s = self._strings[self._idx]
        self._idx += 1
        return s

    async def close(self) -> None:
        pass


# ======================================================================
# Config validation tests
# ======================================================================


class TestValidateProbeConfig:
    """Tests for ``validate_probe_config``."""

    def test_binance_passes(self) -> None:
        rt = make_valid_binance_rt()
        symbol = validate_probe_config(rt)
        assert symbol == BINANCE_SYMBOL

    def test_exchange_okx_fails(self) -> None:
        rt = make_valid_binance_rt(EXCHANGE="okx")
        with pytest.raises(SystemExit) as exc_info:
            validate_probe_config(rt)
        assert exc_info.value.code == 1

    def test_canonical_symbol_wrong_fails(self) -> None:
        # Construct directly — load_unified_runtime_config rejects BTC first.
        rt = ExchangeRuntimeConfig(
            exchange=ExchangeName.BINANCE,
            trade_asset="BTC",
            quote_asset="USDT",
            market_type="PERPETUAL",
            leverage=20,
            margin_mode="isolated",
            position_mode="net",
            kline_interval="15m",
        )
        with pytest.raises(SystemExit) as exc_info:
            validate_probe_config(rt)
        assert exc_info.value.code == 1

    def test_binance_symbol_wrong_fails(self) -> None:
        rt = ExchangeRuntimeConfig(
            exchange=ExchangeName.BINANCE,
            trade_asset="ETH",
            quote_asset="USDC",
            market_type="PERPETUAL",
            leverage=20,
            margin_mode="isolated",
            position_mode="net",
            kline_interval="15m",
        )
        with pytest.raises(SystemExit) as exc_info:
            validate_probe_config(rt)
        assert exc_info.value.code == 1

    def test_kline_interval_wrong_fails(self) -> None:
        rt = ExchangeRuntimeConfig(
            exchange=ExchangeName.BINANCE,
            trade_asset="ETH",
            quote_asset="USDT",
            market_type="PERPETUAL",
            leverage=20,
            margin_mode="isolated",
            position_mode="net",
            kline_interval="5m",
        )
        with pytest.raises(SystemExit) as exc_info:
            validate_probe_config(rt)
        assert exc_info.value.code == 1


# ======================================================================
# Default / env value tests
# ======================================================================


class TestProbeDuration:
    """Tests for ``read_probe_duration``."""

    def test_default_60(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            assert read_probe_duration() == DEFAULT_PROBE_SECONDS

    def test_env_override(self) -> None:
        with mock.patch.dict(os.environ, {ENV_PROBE_SECONDS: "30"}, clear=True):
            assert read_probe_duration() == 30

    def test_zero_rejected(self) -> None:
        with mock.patch.dict(os.environ, {ENV_PROBE_SECONDS: "0"}, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                read_probe_duration()
            assert exc_info.value.code == 1

    def test_negative_rejected(self) -> None:
        with mock.patch.dict(os.environ, {ENV_PROBE_SECONDS: "-5"}, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                read_probe_duration()
            assert exc_info.value.code == 1

    def test_non_integer_rejected(self) -> None:
        with mock.patch.dict(os.environ, {ENV_PROBE_SECONDS: "abc"}, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                read_probe_duration()
            assert exc_info.value.code == 1


class TestMaxEvents:
    """Tests for ``read_max_events``."""

    def test_default_200(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            assert read_max_events() == DEFAULT_MAX_EVENTS

    def test_env_override(self) -> None:
        with mock.patch.dict(os.environ, {ENV_MAX_EVENTS: "50"}, clear=True):
            assert read_max_events() == 50

    def test_zero_rejected(self) -> None:
        with mock.patch.dict(os.environ, {ENV_MAX_EVENTS: "0"}, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                read_max_events()
            assert exc_info.value.code == 1

    def test_negative_rejected(self) -> None:
        with mock.patch.dict(os.environ, {ENV_MAX_EVENTS: "-1"}, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                read_max_events()
            assert exc_info.value.code == 1

    def test_non_integer_rejected(self) -> None:
        with mock.patch.dict(os.environ, {ENV_MAX_EVENTS: "xyz"}, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                read_max_events()
            assert exc_info.value.code == 1


# ======================================================================
# feed.map_message() tests (no network, no mock needed)
# ======================================================================


class TestFeedMapMessage:
    """Tests that ``feed.map_message()`` correctly produces events from JSON strings."""

    def test_agg_trade_event(self) -> None:
        feed = build_feed_for_tests()
        raw = make_agg_trade_payload(price="3000.00", quantity="1.5")
        wrapped = wrap_payload(raw, "ethusdt@aggTrade")
        message = json.dumps(wrapped)

        event = feed.map_message(message)
        assert isinstance(event, MarketTradeEvent)
        assert event.exchange == ExchangeName.BINANCE
        assert event.canonical_symbol == CANONICAL_SYMBOL
        assert event.raw_symbol == "ETHUSDT"
        assert str(event.price) == "3000.00"
        assert str(event.quantity) == "1.5"
        assert event.taker_side == MarketTradeSide.BUY
        assert event.is_aggregated is True

    def test_agg_trade_event_sell_side(self) -> None:
        feed = build_feed_for_tests()
        raw = make_agg_trade_payload(price="3100.00", quantity="2.0", maker_side=True)
        wrapped = wrap_payload(raw, "ethusdt@aggTrade")
        message = json.dumps(wrapped)

        event = feed.map_message(message)
        assert isinstance(event, MarketTradeEvent)
        assert event.taker_side == MarketTradeSide.SELL

    def test_kline_event(self) -> None:
        feed = build_feed_for_tests()
        raw = make_kline_payload(
            open_price="2990.00",
            high_price="3020.00",
            low_price="2980.00",
            close_price="3010.00",
            volume="100.5",
            is_closed=False,
        )
        wrapped = wrap_payload(raw, "ethusdt@kline_15m")
        message = json.dumps(wrapped)

        event = feed.map_message(message)
        assert isinstance(event, MarketCandleEvent)
        assert event.exchange == ExchangeName.BINANCE
        assert event.canonical_symbol == CANONICAL_SYMBOL
        assert event.raw_symbol == "ETHUSDT"
        assert event.timeframe == "15m"
        assert str(event.open_price) == "2990.00"
        assert str(event.high_price) == "3020.00"
        assert str(event.low_price) == "2980.00"
        assert str(event.close_price) == "3010.00"
        assert str(event.volume) == "100.5"
        assert event.is_closed is False

    def test_kline_closed(self) -> None:
        feed = build_feed_for_tests()
        raw = make_kline_payload(is_closed=True)
        wrapped = wrap_payload(raw, "ethusdt@kline_15m")
        message = json.dumps(wrapped)

        event = feed.map_message(message)
        assert isinstance(event, MarketCandleEvent)
        assert event.is_closed is True

    def test_unknown_event_type_returns_none(self) -> None:
        feed = build_feed_for_tests()
        raw = {
            "e": "bookTicker",
            "E": 1700000000000,
            "s": "ETHUSDT",
            "b": "3000.00",
            "a": "3000.01",
        }
        wrapped = wrap_payload(raw, "ethusdt@bookTicker")
        message = json.dumps(wrapped)

        event = feed.map_message(message)
        assert event is None


# ======================================================================
# run_probe tests with fake connection (mock connect_binance_ws)
# ======================================================================


def _make_json_strings(
    raw_payloads: list[dict[str, Any]],
    stream_names: list[str],
) -> list[str]:
    """Wrap raw payloads into combined-stream JSON text strings."""
    result: list[str] = []
    for i, raw in enumerate(raw_payloads):
        stream_name = stream_names[i % len(stream_names)]
        wrapped = wrap_payload(raw, stream_name)
        result.append(json.dumps(wrapped))
    return result


class TestRunProbeWithFakeConnection:
    """Tests for ``run_probe`` using a fake WebSocket connection.

    Each test patches ``scripts.binance_market_data_probe.connect_binance_ws``
    to return a ``FakeBinanceWsConnection`` with pre-baked JSON payloads.
    """

    PROBE_MODULE = "scripts.binance_market_data_probe.connect_binance_ws"

    @pytest.mark.asyncio
    async def test_trade_count_correct(self) -> None:
        """run_probe correctly counts trade events."""
        json_strings = _make_json_strings(
            [
                make_agg_trade_payload(),
                make_agg_trade_payload(price="3001.00"),
                make_kline_payload(),
            ],
            ["ethusdt@aggTrade", "ethusdt@aggTrade", "ethusdt@kline_15m"],
        )

        async def fake_connect(url: str) -> FakeBinanceWsConnection:
            return FakeBinanceWsConnection(json_strings)

        feed = build_feed_for_tests()
        with mock.patch(self.PROBE_MODULE, side_effect=fake_connect):
            stats = await run_probe(
                stream_url="wss://fake/market/stream",
                map_message=feed.map_message,
                duration_seconds=3600,
                max_events=100,
            )
        assert stats["trade_events"] == 2
        assert stats["candle_events"] == 1
        assert stats["errors"] == 0

    @pytest.mark.asyncio
    async def test_candle_count_correct(self) -> None:
        """run_probe correctly counts candle events."""
        json_strings = _make_json_strings(
            [
                make_kline_payload(),
                make_kline_payload(is_closed=True),
                make_kline_payload(),
            ],
            ["ethusdt@kline_15m"] * 3,
        )

        async def fake_connect(url: str) -> FakeBinanceWsConnection:
            return FakeBinanceWsConnection(json_strings)

        feed = build_feed_for_tests()
        with mock.patch(self.PROBE_MODULE, side_effect=fake_connect):
            stats = await run_probe(
                stream_url="wss://fake/market/stream",
                map_message=feed.map_message,
                duration_seconds=3600,
                max_events=100,
            )
        assert stats["trade_events"] == 0
        assert stats["candle_events"] == 3
        assert stats["closed_candle_events"] == 1

    @pytest.mark.asyncio
    async def test_closed_candle_count(self) -> None:
        """run_probe counts closed candles separately."""
        json_strings = _make_json_strings(
            [
                make_kline_payload(is_closed=True),
                make_kline_payload(is_closed=False),
                make_kline_payload(is_closed=True),
                make_kline_payload(is_closed=True),
            ],
            ["ethusdt@kline_15m"] * 4,
        )

        async def fake_connect(url: str) -> FakeBinanceWsConnection:
            return FakeBinanceWsConnection(json_strings)

        feed = build_feed_for_tests()
        with mock.patch(self.PROBE_MODULE, side_effect=fake_connect):
            stats = await run_probe(
                stream_url="wss://fake/market/stream",
                map_message=feed.map_message,
                duration_seconds=3600,
                max_events=100,
            )
        assert stats["closed_candle_events"] == 3

    @pytest.mark.asyncio
    async def test_max_events_exit(self) -> None:
        """run_probe exits after max_events is reached."""
        json_strings = _make_json_strings(
            [make_agg_trade_payload() for _ in range(10)],
            ["ethusdt@aggTrade"] * 10,
        )

        async def fake_connect(url: str) -> FakeBinanceWsConnection:
            return FakeBinanceWsConnection(json_strings)

        feed = build_feed_for_tests()
        with mock.patch(self.PROBE_MODULE, side_effect=fake_connect):
            stats = await run_probe(
                stream_url="wss://fake/market/stream",
                map_message=feed.map_message,
                duration_seconds=3600,
                max_events=3,
            )
        # Should stop at exactly 3 trade events
        assert stats["trade_events"] == 3
        assert stats["trade_events"] + stats["candle_events"] == 3

    @pytest.mark.asyncio
    async def test_malformed_payload_not_crash(self) -> None:
        """Malformed JSON payload increments errors but does not crash the loop."""
        json_strings = [
            "this is not json",
            *_make_json_strings(
                [make_agg_trade_payload()],
                ["ethusdt@aggTrade"],
            ),
        ]

        async def fake_connect(url: str) -> FakeBinanceWsConnection:
            return FakeBinanceWsConnection(json_strings)

        feed = build_feed_for_tests()
        with mock.patch(self.PROBE_MODULE, side_effect=fake_connect):
            stats = await run_probe(
                stream_url="wss://fake/market/stream",
                map_message=feed.map_message,
                duration_seconds=3600,
                max_events=10,
            )
        assert stats["errors"] >= 1
        # The valid trade after the malformed message should still be counted
        assert stats["trade_events"] == 1

    @pytest.mark.asyncio
    async def test_connection_failure_exits(self) -> None:
        """When connect_binance_ws raises, run_probe exits with SystemExit(1)."""

        async def failing_connect(url: str) -> FakeBinanceWsConnection:
            raise ConnectionError("simulated connection failure")

        feed = build_feed_for_tests()
        with mock.patch(self.PROBE_MODULE, side_effect=failing_connect):
            with pytest.raises(SystemExit) as exc_info:
                await run_probe(
                    stream_url="wss://fake/market/stream",
                    map_message=feed.map_message,
                    duration_seconds=60,
                    max_events=10,
                )
            assert exc_info.value.code == 1

    @pytest.mark.asyncio
    async def test_unknown_event_not_counted_as_error(self) -> None:
        """An unrecognised event type (map_message returns None) is not an error."""
        unrecognised_raw = {
            "e": "markPriceUpdate",
            "E": 1700000000000,
            "s": "ETHUSDT",
            "p": "3000.00",
        }
        json_strings = _make_json_strings(
            [unrecognised_raw, make_agg_trade_payload()],
            ["ethusdt@markPriceUpdate", "ethusdt@aggTrade"],
        )

        async def fake_connect(url: str) -> FakeBinanceWsConnection:
            return FakeBinanceWsConnection(json_strings)

        feed = build_feed_for_tests()
        with mock.patch(self.PROBE_MODULE, side_effect=fake_connect):
            stats = await run_probe(
                stream_url="wss://fake/market/stream",
                map_message=feed.map_message,
                duration_seconds=3600,
                max_events=100,
            )
        assert stats["errors"] == 0
        assert stats["trade_events"] == 1


# ======================================================================
# API key non-reading test
# ======================================================================


class TestDoesNotReadApiKey:
    """Verify the probe script does not read API credentials."""

    def test_validate_does_not_access_api_key(self) -> None:
        """validate_probe_config doesn't require api_key or api_secret."""
        env = {
            "EXCHANGE": "binance",
            "TRADE_ASSET": "ETH",
            "QUOTE_ASSET": "USDT",
            "MARKET_TYPE": "PERPETUAL",
            "MARGIN_MODE": "isolated",
            "POSITION_MODE": "net",
            "LEVERAGE": "20",
            "KLINE_INTERVAL": "15m",
        }
        rt = load_unified_runtime_config(env=env)
        assert rt.api_key == ""
        assert rt.api_secret == ""
        symbol = validate_probe_config(rt)
        assert symbol == BINANCE_SYMBOL

    def test_script_source_does_not_read_api_key_env(self) -> None:
        """The probe script source must NOT read any API credential env vars."""
        from pathlib import Path

        src = (
            Path(__file__)
            .resolve()
            .parents[2]
            / "scripts"
            / "binance_market_data_probe.py"
        ).read_text()
        assert "EXCHANGE_API_KEY" not in src
        assert "EXCHANGE_API_SECRET" not in src
        assert "EXCHANGE_API_PASSPHRASE" not in src
