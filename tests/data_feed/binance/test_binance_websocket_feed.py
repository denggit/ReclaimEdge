from __future__ import annotations

import json

import pytest

from src.data_feed.binance.websocket_feed import (
    BINANCE_USDM_WS_MARKET_BASE_URL,
    BinanceWebSocketMarketDataFeed,
    build_binance_combined_market_stream_url,
    decode_binance_ws_message,
    unwrap_binance_combined_stream_payload,
)
from src.data_feed.market_events import MarketCandleEvent, MarketTradeEvent
from src.exchanges.models import ExchangeName


# ---------------------------------------------------------------------------
# Fake connection / connector
# ---------------------------------------------------------------------------


class FakeConnection:
    def __init__(self, messages):
        self._messages = messages

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for message in self._messages:
            yield message


class FakeConnector:
    def __init__(self, messages):
        self.messages = messages
        self.urls = []

    async def __call__(self, url):
        self.urls.append(url)
        return FakeConnection(self.messages)


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------


def _agg_trade_direct_payload(**overrides):
    """Direct Binance aggTrade payload (not wrapped in combined stream)."""
    payload = {
        "e": "aggTrade",
        "E": 1710000000123,
        "s": "ETHUSDT",
        "a": 5933014,
        "p": "3100.50",
        "q": "1.25",
        "f": 100,
        "l": 105,
        "T": 1710000000111,
        "m": True,
    }
    payload.update(overrides)
    return payload


def _kline_direct_payload(**k_overrides):
    """Direct Binance kline payload (not wrapped in combined stream)."""
    kline = {
        "t": 1710000000000,
        "T": 1710000899999,
        "s": "ETHUSDT",
        "i": "15m",
        "o": "3100.00",
        "c": "3120.00",
        "h": "3130.00",
        "l": "3090.00",
        "v": "123.45",
        "x": True,
    }
    kline.update(k_overrides)
    payload = {
        "e": "kline",
        "E": 1710000900000,
        "s": "ETHUSDT",
        "k": kline,
    }
    return payload


def _combined_wrapper(stream_name: str, data: dict) -> dict:
    return {"stream": stream_name, "data": data}


# ---------------------------------------------------------------------------
# build_binance_combined_market_stream_url
# ---------------------------------------------------------------------------


def test_build_combined_stream_url_default():
    url = build_binance_combined_market_stream_url(
        ("ethusdt@aggTrade", "ethusdt@kline_15m"),
    )
    assert url == "wss://fstream.binance.com/market/stream?streams=ethusdt@aggTrade/ethusdt@kline_15m"


def test_build_combined_stream_url_custom_base_url():
    url = build_binance_combined_market_stream_url(
        ("btcusdt@aggTrade",),
        base_url="wss://fstream.binance.com/ws",
    )
    assert url == "wss://fstream.binance.com/ws/stream?streams=btcusdt@aggTrade"


def test_build_combined_stream_url_trailing_slash_on_base_url():
    url = build_binance_combined_market_stream_url(
        ("ethusdt@aggTrade",),
        base_url="wss://fstream.binance.com/market/",
    )
    assert url == "wss://fstream.binance.com/market/stream?streams=ethusdt@aggTrade"


def test_build_combined_stream_url_multiple_streams():
    url = build_binance_combined_market_stream_url(
        ("btcusdt@aggTrade", "btcusdt@kline_1m", "ethusdt@aggTrade"),
    )
    assert url == "wss://fstream.binance.com/market/stream?streams=btcusdt@aggTrade/btcusdt@kline_1m/ethusdt@aggTrade"


def test_build_combined_stream_url_empty_tuple_raises():
    with pytest.raises(ValueError, match="stream_names must not be empty"):
        build_binance_combined_market_stream_url(())


def test_build_combined_stream_url_empty_string_raises():
    with pytest.raises(ValueError, match="stream name must not be empty"):
        build_binance_combined_market_stream_url(("",))


def test_build_combined_stream_url_whitespace_only_raises():
    with pytest.raises(ValueError, match="stream name must not be empty"):
        build_binance_combined_market_stream_url(("   ",))


# ---------------------------------------------------------------------------
# decode_binance_ws_message
# ---------------------------------------------------------------------------


def test_decode_str_json_object():
    payload = decode_binance_ws_message('{"e":"aggTrade","s":"ETHUSDT"}')
    assert payload == {"e": "aggTrade", "s": "ETHUSDT"}


def test_decode_bytes_json_object():
    payload = decode_binance_ws_message(b'{"e":"aggTrade","s":"ETHUSDT"}')
    assert payload == {"e": "aggTrade", "s": "ETHUSDT"}


def test_decode_invalid_json_raises():
    with pytest.raises(ValueError, match="Invalid Binance websocket JSON payload"):
        decode_binance_ws_message("not json")


def test_decode_non_object_json_raises():
    with pytest.raises(ValueError, match="Binance websocket payload must be a JSON object"):
        decode_binance_ws_message("[1, 2, 3]")


# ---------------------------------------------------------------------------
# unwrap_binance_combined_stream_payload
# ---------------------------------------------------------------------------


def test_unwrap_combined_payload_returns_data():
    payload = _combined_wrapper(
        "ethusdt@aggTrade",
        _agg_trade_direct_payload(),
    )
    unwrapped = unwrap_binance_combined_stream_payload(payload)
    assert unwrapped["e"] == "aggTrade"
    assert unwrapped["s"] == "ETHUSDT"


def test_unwrap_direct_payload_returns_self():
    direct = _agg_trade_direct_payload()
    unwrapped = unwrap_binance_combined_stream_payload(direct)
    assert unwrapped is direct


def test_unwrap_payload_with_non_mapping_data_returns_self():
    payload = {"stream": "ethusdt@aggTrade", "data": "not-an-object"}
    result = unwrap_binance_combined_stream_payload(payload)
    assert result is payload


def test_unwrap_payload_with_null_data_returns_self():
    payload = {"stream": "ethusdt@aggTrade", "data": None}
    result = unwrap_binance_combined_stream_payload(payload)
    assert result is payload


# ---------------------------------------------------------------------------
# BinanceWebSocketMarketDataFeed — stream_names / stream_url
# ---------------------------------------------------------------------------


def test_feed_stream_names_defaults():
    feed = BinanceWebSocketMarketDataFeed(connector=FakeConnector([]))
    names = feed.stream_names()
    assert names == ("ethusdt@aggTrade", "ethusdt@kline_15m")


def test_feed_stream_names_custom_symbols():
    feed = BinanceWebSocketMarketDataFeed(
        connector=FakeConnector([]),
        raw_symbol="BTCUSDT",
        kline_interval="1m",
    )
    names = feed.stream_names()
    assert names == ("btcusdt@aggTrade", "btcusdt@kline_1m")


def test_feed_stream_url_default():
    feed = BinanceWebSocketMarketDataFeed(connector=FakeConnector([]))
    url = feed.stream_url()
    assert "wss://fstream.binance.com/market/stream?streams=" in url
    assert "ethusdt@aggTrade" in url
    assert "ethusdt@kline_15m" in url


def test_feed_stream_url_custom_base_url():
    feed = BinanceWebSocketMarketDataFeed(
        connector=FakeConnector([]),
        base_url="wss://test.binance.com/ws",
    )
    url = feed.stream_url()
    assert url.startswith("wss://test.binance.com/ws/stream?streams=")


def test_feed_canonical_symbol_property():
    feed = BinanceWebSocketMarketDataFeed(connector=FakeConnector([]))
    assert feed.canonical_symbol == "ETH-USDT-PERP"

    custom = BinanceWebSocketMarketDataFeed(
        connector=FakeConnector([]),
        canonical_symbol="BTC-USDT-PERP",
    )
    assert custom.canonical_symbol == "BTC-USDT-PERP"


def test_feed_raw_symbol_property():
    feed = BinanceWebSocketMarketDataFeed(connector=FakeConnector([]))
    assert feed.raw_symbol == "ETHUSDT"

    custom = BinanceWebSocketMarketDataFeed(
        connector=FakeConnector([]),
        raw_symbol="BTCUSDT",
    )
    assert custom.raw_symbol == "BTCUSDT"


# ---------------------------------------------------------------------------
# BinanceWebSocketMarketDataFeed — map_message (combined stream)
# ---------------------------------------------------------------------------


def test_map_message_combined_agg_trade_returns_market_trade_event():
    payload = _agg_trade_direct_payload()
    wrapped = _combined_wrapper("ethusdt@aggTrade", payload)
    raw_json = json.dumps(wrapped)

    feed = BinanceWebSocketMarketDataFeed(connector=FakeConnector([]))
    event = feed.map_message(raw_json)

    assert isinstance(event, MarketTradeEvent)
    assert event.exchange == ExchangeName.BINANCE
    assert event.canonical_symbol == "ETH-USDT-PERP"


def test_map_message_combined_kline_returns_market_candle_event():
    payload = _kline_direct_payload()
    wrapped = _combined_wrapper("ethusdt@kline_15m", payload)
    raw_json = json.dumps(wrapped)

    feed = BinanceWebSocketMarketDataFeed(connector=FakeConnector([]))
    event = feed.map_message(raw_json)

    assert isinstance(event, MarketCandleEvent)
    assert event.exchange == ExchangeName.BINANCE
    assert event.canonical_symbol == "ETH-USDT-PERP"
    assert event.timeframe == "15m"


# ---------------------------------------------------------------------------
# BinanceWebSocketMarketDataFeed — map_message (direct payload)
# ---------------------------------------------------------------------------


def test_map_message_direct_agg_trade_returns_market_trade_event():
    raw_json = json.dumps(_agg_trade_direct_payload())

    feed = BinanceWebSocketMarketDataFeed(connector=FakeConnector([]))
    event = feed.map_message(raw_json)

    assert isinstance(event, MarketTradeEvent)
    assert event.exchange == ExchangeName.BINANCE


def test_map_message_direct_kline_returns_market_candle_event():
    raw_json = json.dumps(_kline_direct_payload())

    feed = BinanceWebSocketMarketDataFeed(connector=FakeConnector([]))
    event = feed.map_message(raw_json)

    assert isinstance(event, MarketCandleEvent)
    assert event.exchange == ExchangeName.BINANCE


# ---------------------------------------------------------------------------
# BinanceWebSocketMarketDataFeed — map_message unsupported / None
# ---------------------------------------------------------------------------


def test_map_message_unsupported_returns_none():
    raw_json = json.dumps({"e": "bookTicker", "s": "ETHUSDT"})
    feed = BinanceWebSocketMarketDataFeed(connector=FakeConnector([]))
    event = feed.map_message(raw_json)
    assert event is None


def test_map_message_bytes_input():
    raw_bytes = json.dumps(_agg_trade_direct_payload()).encode("utf-8")
    feed = BinanceWebSocketMarketDataFeed(connector=FakeConnector([]))
    event = feed.map_message(raw_bytes)
    assert isinstance(event, MarketTradeEvent)


# ---------------------------------------------------------------------------
# BinanceWebSocketMarketDataFeed — map_payload
# ---------------------------------------------------------------------------


def test_map_payload_combined_agg_trade():
    wrapped = _combined_wrapper("ethusdt@aggTrade", _agg_trade_direct_payload())
    feed = BinanceWebSocketMarketDataFeed(connector=FakeConnector([]))
    event = feed.map_payload(wrapped)
    assert isinstance(event, MarketTradeEvent)


def test_map_payload_direct_kline():
    direct = _kline_direct_payload()
    feed = BinanceWebSocketMarketDataFeed(connector=FakeConnector([]))
    event = feed.map_payload(direct)
    assert isinstance(event, MarketCandleEvent)


def test_map_payload_unsupported_returns_none():
    feed = BinanceWebSocketMarketDataFeed(connector=FakeConnector([]))
    event = feed.map_payload({"e": "depthUpdate"})
    assert event is None


# ---------------------------------------------------------------------------
# BinanceWebSocketMarketDataFeed — events()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_events_calls_connector_with_stream_url():
    connector = FakeConnector([
        json.dumps(_agg_trade_direct_payload()),
    ])

    feed = BinanceWebSocketMarketDataFeed(connector=connector)
    events = [event async for event in feed.events()]

    assert len(connector.urls) == 1
    assert "wss://fstream.binance.com/market/stream?streams=" in connector.urls[0]


@pytest.mark.asyncio
async def test_events_yields_only_mapped_events():
    connector = FakeConnector([
        json.dumps(_agg_trade_direct_payload()),
        json.dumps({"e": "bookTicker"}),  # unsupported
        json.dumps(_kline_direct_payload()),
    ])

    feed = BinanceWebSocketMarketDataFeed(connector=connector)
    events = [event async for event in feed.events()]

    assert len(events) == 2
    assert isinstance(events[0], MarketTradeEvent)
    assert isinstance(events[1], MarketCandleEvent)


@pytest.mark.asyncio
async def test_events_empty_messages_yields_nothing():
    connector = FakeConnector([])
    feed = BinanceWebSocketMarketDataFeed(connector=connector)
    events = [event async for event in feed.events()]
    assert len(events) == 0


@pytest.mark.asyncio
async def test_events_all_unsupported_yields_nothing():
    connector = FakeConnector([
        json.dumps({"e": "bookTicker"}),
        json.dumps({"e": "depthUpdate"}),
    ])
    feed = BinanceWebSocketMarketDataFeed(connector=connector)
    events = [event async for event in feed.events()]
    assert len(events) == 0


@pytest.mark.asyncio
async def test_events_handles_bytes_messages():
    connector = FakeConnector([
        json.dumps(_agg_trade_direct_payload()).encode("utf-8"),
    ])
    feed = BinanceWebSocketMarketDataFeed(connector=connector)
    events = [event async for event in feed.events()]
    assert len(events) == 1
    assert isinstance(events[0], MarketTradeEvent)


# ---------------------------------------------------------------------------
# BinanceWebSocketMarketDataFeed — custom parameters
# ---------------------------------------------------------------------------


def test_feed_custom_canonical_symbol():
    feed = BinanceWebSocketMarketDataFeed(
        connector=FakeConnector([]),
        canonical_symbol="BTC-USDT-PERP",
    )
    payload = _agg_trade_direct_payload(s="BTCUSDT")
    event = feed.map_payload(payload)
    assert event is not None
    assert event.canonical_symbol == "BTC-USDT-PERP"


def test_feed_custom_raw_symbol_and_interval():
    feed = BinanceWebSocketMarketDataFeed(
        connector=FakeConnector([]),
        raw_symbol="BTCUSDT",
        kline_interval="1m",
    )
    names = feed.stream_names()
    assert names == ("btcusdt@aggTrade", "btcusdt@kline_1m")


def test_feed_custom_base_url():
    feed = BinanceWebSocketMarketDataFeed(
        connector=FakeConnector([]),
        base_url="wss://example.com/custom",
    )
    url = feed.stream_url()
    assert url.startswith("wss://example.com/custom/stream?streams=")


# ---------------------------------------------------------------------------
# BINANCE_USDM_WS_MARKET_BASE_URL constant
# ---------------------------------------------------------------------------


def test_base_url_constant():
    assert BINANCE_USDM_WS_MARKET_BASE_URL == "wss://fstream.binance.com/market"
