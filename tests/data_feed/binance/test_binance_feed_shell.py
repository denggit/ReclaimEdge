from __future__ import annotations

import pytest

from src.data_feed.binance.feed import (
    binance_agg_trade_stream_name,
    binance_default_market_stream_names,
    binance_kline_stream_name,
    map_binance_market_event,
    normalize_binance_stream_symbol,
    try_map_binance_market_event,
)
from src.data_feed.market_events import MarketCandleEvent, MarketTradeEvent
from src.exchanges.models import ExchangeName


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------


def _agg_trade_payload(**overrides):
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


def _kline_payload(**k_overrides):
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
    return {
        "e": "kline",
        "E": 1710000900000,
        "s": "ETHUSDT",
        "k": kline,
    }


# ---------------------------------------------------------------------------
# normalize_binance_stream_symbol
# ---------------------------------------------------------------------------


def test_normalize_binance_stream_symbol_lowercase():
    assert normalize_binance_stream_symbol("ETHUSDT") == "ethusdt"


def test_normalize_binance_stream_symbol_strip_whitespace():
    assert normalize_binance_stream_symbol(" ethusdt ") == "ethusdt"


def test_normalize_binance_stream_symbol_empty_raises():
    with pytest.raises(ValueError, match="raw_symbol must not be empty"):
        normalize_binance_stream_symbol("")


def test_normalize_binance_stream_symbol_none_raises():
    with pytest.raises(ValueError, match="raw_symbol must not be empty"):
        normalize_binance_stream_symbol(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# binance_agg_trade_stream_name
# ---------------------------------------------------------------------------


def test_agg_trade_stream_name_default():
    assert binance_agg_trade_stream_name() == "ethusdt@aggTrade"


def test_agg_trade_stream_name_btcusdt():
    assert binance_agg_trade_stream_name("BTCUSDT") == "btcusdt@aggTrade"


# ---------------------------------------------------------------------------
# binance_kline_stream_name
# ---------------------------------------------------------------------------


def test_kline_stream_name_default():
    assert binance_kline_stream_name() == "ethusdt@kline_15m"


def test_kline_stream_name_btcusdt_1m():
    assert binance_kline_stream_name("BTCUSDT", "1m") == "btcusdt@kline_1m"


def test_kline_stream_name_empty_interval_raises():
    with pytest.raises(ValueError, match="interval must not be empty"):
        binance_kline_stream_name("ETHUSDT", "")


# ---------------------------------------------------------------------------
# binance_default_market_stream_names
# ---------------------------------------------------------------------------


def test_default_market_stream_names():
    agg, kline = binance_default_market_stream_names()
    assert agg == "ethusdt@aggTrade"
    assert kline == "ethusdt@kline_15m"


# ---------------------------------------------------------------------------
# map_binance_market_event — aggTrade
# ---------------------------------------------------------------------------


def test_map_agg_trade_returns_market_trade_event():
    payload = _agg_trade_payload()
    event = map_binance_market_event(payload)
    assert isinstance(event, MarketTradeEvent)
    assert event.exchange == ExchangeName.BINANCE
    assert event.canonical_symbol == "ETH-USDT-PERP"


def test_map_agg_trade_passes_custom_canonical_symbol():
    payload = _agg_trade_payload()
    event = map_binance_market_event(
        payload,
        canonical_symbol="BTC-USDT-PERP",
    )
    assert event.canonical_symbol == "BTC-USDT-PERP"


# ---------------------------------------------------------------------------
# map_binance_market_event — kline
# ---------------------------------------------------------------------------


def test_map_kline_returns_market_candle_event():
    payload = _kline_payload()
    event = map_binance_market_event(payload)
    assert isinstance(event, MarketCandleEvent)
    assert event.exchange == ExchangeName.BINANCE
    assert event.canonical_symbol == "ETH-USDT-PERP"
    assert event.timeframe == "15m"


def test_map_kline_passes_custom_canonical_symbol():
    payload = _kline_payload()
    event = map_binance_market_event(
        payload,
        canonical_symbol="BTC-USDT-PERP",
    )
    assert event.canonical_symbol == "BTC-USDT-PERP"


# ---------------------------------------------------------------------------
# map_binance_market_event — unsupported
# ---------------------------------------------------------------------------


def test_map_unsupported_event_raises_value_error():
    payload = {"e": "bookTicker", "s": "ETHUSDT"}
    with pytest.raises(ValueError, match="Unsupported Binance market event type"):
        map_binance_market_event(payload)


def test_map_missing_event_type_raises_value_error():
    payload: dict = {}
    with pytest.raises(ValueError, match="Unsupported Binance market event type"):
        map_binance_market_event(payload)


# ---------------------------------------------------------------------------
# try_map_binance_market_event
# ---------------------------------------------------------------------------


def test_try_map_unsupported_event_returns_none():
    payload = {"e": "bookTicker", "s": "ETHUSDT"}
    result = try_map_binance_market_event(payload)
    assert result is None


def test_try_map_missing_event_type_returns_none():
    payload: dict = {}
    result = try_map_binance_market_event(payload)
    assert result is None


def test_try_map_agg_trade_returns_event():
    payload = _agg_trade_payload()
    event = try_map_binance_market_event(payload)
    assert isinstance(event, MarketTradeEvent)


def test_try_map_kline_returns_event():
    payload = _kline_payload()
    event = try_map_binance_market_event(payload)
    assert isinstance(event, MarketCandleEvent)
