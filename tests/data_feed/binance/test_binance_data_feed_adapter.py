from __future__ import annotations

from src.data_feed.binance.adapter import BinanceMarketDataFeed
from src.data_feed.market_events import MarketCandleEvent, MarketTradeEvent
from src.exchanges.models import ExchangeName


# ---------------------------------------------------------------------------
# Payload helpers (matching the pattern from test_binance_feed_shell.py)
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
# BinanceMarketDataFeed — exchange
# ---------------------------------------------------------------------------


def test_binance_data_feed_exchange_is_binance() -> None:
    feed = BinanceMarketDataFeed()
    assert feed.exchange == ExchangeName.BINANCE


# ---------------------------------------------------------------------------
# BinanceMarketDataFeed — default canonical_symbol
# ---------------------------------------------------------------------------


def test_binance_data_feed_default_canonical_symbol() -> None:
    feed = BinanceMarketDataFeed()
    assert feed.canonical_symbol == "ETH-USDT-PERP"


# ---------------------------------------------------------------------------
# BinanceMarketDataFeed — default raw_symbol
# ---------------------------------------------------------------------------


def test_binance_data_feed_default_raw_symbol() -> None:
    feed = BinanceMarketDataFeed()
    assert feed.raw_symbol == "ETHUSDT"


# ---------------------------------------------------------------------------
# BinanceMarketDataFeed — stream_names()
# ---------------------------------------------------------------------------


def test_binance_data_feed_default_stream_names() -> None:
    feed = BinanceMarketDataFeed()
    names = feed.stream_names()
    assert names == ("ethusdt@aggTrade", "ethusdt@kline_15m")


# ---------------------------------------------------------------------------
# BinanceMarketDataFeed — custom raw_symbol / kline_interval
# ---------------------------------------------------------------------------


def test_binance_data_feed_custom_raw_symbol() -> None:
    feed = BinanceMarketDataFeed(raw_symbol="BTCUSDT")
    assert feed.raw_symbol == "BTCUSDT"
    names = feed.stream_names()
    assert names == ("btcusdt@aggTrade", "btcusdt@kline_15m")


def test_binance_data_feed_custom_kline_interval() -> None:
    feed = BinanceMarketDataFeed(kline_interval="1m")
    names = feed.stream_names()
    assert names == ("ethusdt@aggTrade", "ethusdt@kline_1m")


# ---------------------------------------------------------------------------
# BinanceMarketDataFeed — map_message aggTrade returns MarketTradeEvent
# ---------------------------------------------------------------------------


def test_binance_data_feed_map_agg_trade_returns_market_trade_event() -> None:
    feed = BinanceMarketDataFeed()
    event = feed.map_message(_agg_trade_payload())
    assert isinstance(event, MarketTradeEvent)
    assert event.exchange == ExchangeName.BINANCE
    assert event.canonical_symbol == "ETH-USDT-PERP"


# ---------------------------------------------------------------------------
# BinanceMarketDataFeed — map_message kline returns MarketCandleEvent
# ---------------------------------------------------------------------------


def test_binance_data_feed_map_kline_returns_market_candle_event() -> None:
    feed = BinanceMarketDataFeed()
    event = feed.map_message(_kline_payload())
    assert isinstance(event, MarketCandleEvent)
    assert event.exchange == ExchangeName.BINANCE
    assert event.canonical_symbol == "ETH-USDT-PERP"


# ---------------------------------------------------------------------------
# BinanceMarketDataFeed — map_message unsupported returns None
# ---------------------------------------------------------------------------


def test_binance_data_feed_map_unsupported_returns_none() -> None:
    feed = BinanceMarketDataFeed()
    result = feed.map_message({"e": "bookTicker", "s": "ETHUSDT"})
    assert result is None
