from __future__ import annotations

import pytest

from src.data_feed.binance.adapter import BinanceMarketDataFeed
from src.data_feed.binance.websocket_feed import BinanceWebSocketMarketDataFeed
from src.data_feed.selector import build_market_data_feed


# ---------------------------------------------------------------------------
# Fake connector / connection
# ---------------------------------------------------------------------------


class FakeConnection:
    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        if False:
            yield ""


async def fake_connector(url: str) -> FakeConnection:
    return FakeConnection()


# ---------------------------------------------------------------------------
# selector builds BinanceWebSocketMarketDataFeed with fake connector
# ---------------------------------------------------------------------------


def test_selector_builds_ws_feed_with_fake_connector() -> None:
    feed = build_market_data_feed(
        exchange="binance",
        binance_ws_connector=fake_connector,
    )
    assert isinstance(feed, BinanceWebSocketMarketDataFeed)


# ---------------------------------------------------------------------------
# stream_names
# ---------------------------------------------------------------------------


def test_ws_feed_stream_names_default() -> None:
    feed = build_market_data_feed(
        exchange="binance",
        binance_ws_connector=fake_connector,
    )
    assert feed.stream_names() == ("ethusdt@aggTrade", "ethusdt@kline_15m")


# ---------------------------------------------------------------------------
# stream_url uses /market/stream?streams=
# ---------------------------------------------------------------------------


def test_ws_feed_stream_url_uses_market_stream() -> None:
    feed = build_market_data_feed(
        exchange="binance",
        binance_ws_connector=fake_connector,
    )
    url = feed.stream_url()
    assert "/market/stream?streams=" in url


# ---------------------------------------------------------------------------
# custom raw_symbol / kline_interval reflected in stream_names
# ---------------------------------------------------------------------------


def test_ws_feed_custom_raw_symbol_in_stream_names() -> None:
    feed = build_market_data_feed(
        exchange="binance",
        raw_symbol="btcusdt",
        binance_ws_connector=fake_connector,
    )
    names = feed.stream_names()
    assert names[0] == "btcusdt@aggTrade"


def test_ws_feed_custom_kline_interval_in_stream_names() -> None:
    feed = build_market_data_feed(
        exchange="binance",
        kline_interval="1m",
        binance_ws_connector=fake_connector,
    )
    names = feed.stream_names()
    assert names[1] == "ethusdt@kline_1m"


# ---------------------------------------------------------------------------
# shell mode only works with allow_binance_without_ws_connector=True
# ---------------------------------------------------------------------------


def test_binance_without_connector_raises_value_error() -> None:
    with pytest.raises(ValueError, match="binance_ws_connector is required"):
        build_market_data_feed(exchange="binance")


def test_binance_shell_mode_requires_allow_flag() -> None:
    feed = build_market_data_feed(
        exchange="binance",
        allow_binance_without_ws_connector=True,
    )
    assert isinstance(feed, BinanceMarketDataFeed)


def test_allow_flag_without_connector_returns_shell() -> None:
    """Explicit regression: allow flag must NOT require a connector."""
    feed = build_market_data_feed(
        exchange="binance",
        allow_binance_without_ws_connector=True,
    )
    assert isinstance(feed, BinanceMarketDataFeed)
    assert feed.raw_symbol == "ETHUSDT"


# ---------------------------------------------------------------------------
# connector wins over allow flag
# ---------------------------------------------------------------------------


def test_connector_wins_over_allow_flag() -> None:
    feed = build_market_data_feed(
        exchange="binance",
        binance_ws_connector=fake_connector,
        allow_binance_without_ws_connector=True,
    )
    assert isinstance(feed, BinanceWebSocketMarketDataFeed)
