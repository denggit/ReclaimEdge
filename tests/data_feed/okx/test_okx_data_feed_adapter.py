from __future__ import annotations

from src.data_feed.okx.adapter import (
    DEFAULT_OKX_CANONICAL_SYMBOL,
    DEFAULT_OKX_RAW_SYMBOL,
    OkxMarketDataFeed,
)
from src.exchanges.models import ExchangeName


# ---------------------------------------------------------------------------
# OkxMarketDataFeed — exchange
# ---------------------------------------------------------------------------


def test_okx_data_feed_exchange_is_okx() -> None:
    feed = OkxMarketDataFeed()
    assert feed.exchange == ExchangeName.OKX


# ---------------------------------------------------------------------------
# OkxMarketDataFeed — default canonical_symbol
# ---------------------------------------------------------------------------


def test_okx_data_feed_default_canonical_symbol() -> None:
    feed = OkxMarketDataFeed()
    assert feed.canonical_symbol == "ETH-USDT-PERP"
    assert feed.canonical_symbol == DEFAULT_OKX_CANONICAL_SYMBOL


# ---------------------------------------------------------------------------
# OkxMarketDataFeed — default raw_symbol
# ---------------------------------------------------------------------------


def test_okx_data_feed_default_raw_symbol() -> None:
    feed = OkxMarketDataFeed()
    assert feed.raw_symbol == "ETH-USDT-SWAP"
    assert feed.raw_symbol == DEFAULT_OKX_RAW_SYMBOL


# ---------------------------------------------------------------------------
# OkxMarketDataFeed — stream_names() == ()
# ---------------------------------------------------------------------------


def test_okx_data_feed_stream_names_is_empty() -> None:
    feed = OkxMarketDataFeed()
    assert feed.stream_names() == ()


# ---------------------------------------------------------------------------
# OkxMarketDataFeed — map_message(...) is None
# ---------------------------------------------------------------------------


def test_okx_data_feed_map_message_returns_none() -> None:
    feed = OkxMarketDataFeed()
    result = feed.map_message({"e": "aggTrade", "s": "ETH-USDT-SWAP"})
    assert result is None


# ---------------------------------------------------------------------------
# OkxMarketDataFeed — custom canonical_symbol / raw_symbol
# ---------------------------------------------------------------------------


def test_okx_data_feed_custom_canonical_symbol() -> None:
    feed = OkxMarketDataFeed(canonical_symbol="BTC-USDT-PERP")
    assert feed.canonical_symbol == "BTC-USDT-PERP"


def test_okx_data_feed_custom_raw_symbol() -> None:
    feed = OkxMarketDataFeed(raw_symbol="BTC-USDT-SWAP")
    assert feed.raw_symbol == "BTC-USDT-SWAP"
