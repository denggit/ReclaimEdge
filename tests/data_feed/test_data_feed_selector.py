from __future__ import annotations

import pytest

from src.data_feed.binance.adapter import BinanceMarketDataFeed
from src.data_feed.okx.adapter import OkxMarketDataFeed
from src.data_feed.selector import build_market_data_feed, normalize_exchange_name
from src.exchanges.models import ExchangeName


# ---------------------------------------------------------------------------
# normalize_exchange_name(None) -> OKX
# ---------------------------------------------------------------------------


def test_normalize_exchange_name_none_returns_okx() -> None:
    assert normalize_exchange_name(None) == ExchangeName.OKX


# ---------------------------------------------------------------------------
# normalize_exchange_name("okx") -> OKX
# ---------------------------------------------------------------------------


def test_normalize_exchange_name_str_okx_returns_okx() -> None:
    assert normalize_exchange_name("okx") == ExchangeName.OKX


# ---------------------------------------------------------------------------
# normalize_exchange_name("binance") -> BINANCE
# ---------------------------------------------------------------------------


def test_normalize_exchange_name_str_binance_returns_binance() -> None:
    assert normalize_exchange_name("binance") == ExchangeName.BINANCE


# ---------------------------------------------------------------------------
# normalize_exchange_name(ExchangeName.OKX) -> OKX
# ---------------------------------------------------------------------------


def test_normalize_exchange_name_enum_okx_returns_okx() -> None:
    assert normalize_exchange_name(ExchangeName.OKX) == ExchangeName.OKX


# ---------------------------------------------------------------------------
# normalize_exchange_name(ExchangeName.BINANCE) -> BINANCE
# ---------------------------------------------------------------------------


def test_normalize_exchange_name_enum_binance_returns_binance() -> None:
    assert normalize_exchange_name(ExchangeName.BINANCE) == ExchangeName.BINANCE


# ---------------------------------------------------------------------------
# normalize_exchange_name — unsupported exchange raises ValueError
# ---------------------------------------------------------------------------


def test_normalize_exchange_name_unsupported_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported data feed exchange"):
        normalize_exchange_name("bybit")


# ---------------------------------------------------------------------------
# build_market_data_feed() default returns OkxMarketDataFeed
# ---------------------------------------------------------------------------


def test_build_market_data_feed_default_returns_okx() -> None:
    feed = build_market_data_feed()
    assert isinstance(feed, OkxMarketDataFeed)
    assert feed.exchange == ExchangeName.OKX


# ---------------------------------------------------------------------------
# build_market_data_feed(exchange="okx") returns OkxMarketDataFeed
# ---------------------------------------------------------------------------


def test_build_market_data_feed_exchange_okx_returns_okx() -> None:
    feed = build_market_data_feed(exchange="okx")
    assert isinstance(feed, OkxMarketDataFeed)
    assert feed.exchange == ExchangeName.OKX


# ---------------------------------------------------------------------------
# build_market_data_feed(exchange="binance") returns BinanceMarketDataFeed
# ---------------------------------------------------------------------------


def test_build_market_data_feed_exchange_binance_returns_binance() -> None:
    feed = build_market_data_feed(exchange="binance")
    assert isinstance(feed, BinanceMarketDataFeed)
    assert feed.exchange == ExchangeName.BINANCE


# ---------------------------------------------------------------------------
# Binance selector default raw_symbol == ETHUSDT
# ---------------------------------------------------------------------------


def test_build_binance_feed_default_raw_symbol() -> None:
    feed = build_market_data_feed(exchange="binance")
    assert feed.raw_symbol == "ETHUSDT"


# ---------------------------------------------------------------------------
# OKX selector default raw_symbol == ETH-USDT-SWAP
# ---------------------------------------------------------------------------


def test_build_okx_feed_default_raw_symbol() -> None:
    feed = build_market_data_feed(exchange="okx")
    assert feed.raw_symbol == "ETH-USDT-SWAP"


# ---------------------------------------------------------------------------
# custom canonical_symbol / raw_symbol / kline_interval
# ---------------------------------------------------------------------------


def test_build_market_data_feed_custom_canonical_symbol() -> None:
    feed = build_market_data_feed(
        exchange="binance",
        canonical_symbol="BTC-USDT-PERP",
    )
    assert isinstance(feed, BinanceMarketDataFeed)
    assert feed.canonical_symbol == "BTC-USDT-PERP"


def test_build_market_data_feed_custom_raw_symbol_binance() -> None:
    feed = build_market_data_feed(
        exchange="binance",
        raw_symbol="BTCUSDT",
    )
    assert isinstance(feed, BinanceMarketDataFeed)
    assert feed.raw_symbol == "BTCUSDT"


def test_build_market_data_feed_custom_raw_symbol_okx() -> None:
    feed = build_market_data_feed(
        exchange="okx",
        raw_symbol="BTC-USDT-SWAP",
    )
    assert isinstance(feed, OkxMarketDataFeed)
    assert feed.raw_symbol == "BTC-USDT-SWAP"


def test_build_market_data_feed_custom_kline_interval() -> None:
    feed = build_market_data_feed(
        exchange="binance",
        kline_interval="1m",
    )
    assert isinstance(feed, BinanceMarketDataFeed)
    names = feed.stream_names()
    assert "kline_1m" in names[1]


# ---------------------------------------------------------------------------
# build_market_data_feed(exchange=ExchangeName.BINANCE) enum form
# ---------------------------------------------------------------------------


def test_build_market_data_feed_enum_binance() -> None:
    feed = build_market_data_feed(exchange=ExchangeName.BINANCE)
    assert isinstance(feed, BinanceMarketDataFeed)


# ---------------------------------------------------------------------------
# build_market_data_feed(exchange=ExchangeName.OKX) enum form
# ---------------------------------------------------------------------------


def test_build_market_data_feed_enum_okx() -> None:
    feed = build_market_data_feed(exchange=ExchangeName.OKX)
    assert isinstance(feed, OkxMarketDataFeed)
