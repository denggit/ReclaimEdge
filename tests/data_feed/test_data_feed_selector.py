from __future__ import annotations

import pytest

from src.data_feed.binance.adapter import BinanceMarketDataFeed
from src.data_feed.binance.websocket_feed import BinanceWebSocketMarketDataFeed
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
# build_market_data_feed(exchange="binance") raises ValueError without connector
# ---------------------------------------------------------------------------


def test_build_market_data_feed_exchange_binance_raises_without_connector() -> None:
    with pytest.raises(ValueError, match="binance_ws_connector is required"):
        build_market_data_feed(exchange="binance")


# ---------------------------------------------------------------------------
# Binance shell mode — allow_binance_without_ws_connector=True
# ---------------------------------------------------------------------------


def test_build_market_data_feed_binance_shell_mode() -> None:
    feed = build_market_data_feed(
        exchange="binance", allow_binance_without_ws_connector=True
    )
    assert isinstance(feed, BinanceMarketDataFeed)
    assert feed.exchange == ExchangeName.BINANCE


# ---------------------------------------------------------------------------
# Binance selector default raw_symbol == ETHUSDT
# ---------------------------------------------------------------------------


def test_build_binance_feed_default_raw_symbol() -> None:
    feed = build_market_data_feed(
        exchange="binance", allow_binance_without_ws_connector=True
    )
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


def test_non_eth_canonical_symbol_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Only ETH-USDT-PERP is supported"):
        build_market_data_feed(
            exchange="binance",
            canonical_symbol="BTC-USDT-PERP",
            allow_binance_without_ws_connector=True,
        )


def test_non_eth_binance_raw_symbol_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Only ETHUSDT is supported for Binance data feed"):
        build_market_data_feed(
            exchange="binance",
            raw_symbol="BTCUSDT",
            allow_binance_without_ws_connector=True,
        )


def test_non_eth_okx_raw_symbol_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Only ETH-USDT-SWAP is supported for OKX data feed"):
        build_market_data_feed(
            exchange="okx",
            raw_symbol="BTC-USDT-SWAP",
        )


def test_non_15m_kline_interval_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Only 15m kline interval is supported for Binance data feed"):
        build_market_data_feed(
            exchange="binance",
            kline_interval="1m",
            allow_binance_without_ws_connector=True,
        )


# ---------------------------------------------------------------------------
# build_market_data_feed(exchange=ExchangeName.BINANCE) enum form
# ---------------------------------------------------------------------------


def test_build_market_data_feed_enum_binance_shell_mode() -> None:
    feed = build_market_data_feed(
        exchange=ExchangeName.BINANCE, allow_binance_without_ws_connector=True
    )
    assert isinstance(feed, BinanceMarketDataFeed)


def test_build_market_data_feed_enum_binance_raises_without_connector() -> None:
    with pytest.raises(ValueError, match="binance_ws_connector is required"):
        build_market_data_feed(exchange=ExchangeName.BINANCE)


# ---------------------------------------------------------------------------
# build_market_data_feed(exchange=ExchangeName.OKX) enum form
# ---------------------------------------------------------------------------


def test_build_market_data_feed_enum_okx() -> None:
    feed = build_market_data_feed(exchange=ExchangeName.OKX)
    assert isinstance(feed, OkxMarketDataFeed)


# ---------------------------------------------------------------------------
# Binance websocket feed via selector with fake connector
# ---------------------------------------------------------------------------


class _FakeConnection:
    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        if False:
            yield ""


async def _fake_connector(url: str) -> _FakeConnection:
    return _FakeConnection()


def test_build_market_data_feed_binance_ws_connector_returns_ws_feed() -> None:
    feed = build_market_data_feed(
        exchange="binance",
        binance_ws_connector=_fake_connector,
    )
    assert isinstance(feed, BinanceWebSocketMarketDataFeed)


def test_build_market_data_feed_binance_ws_feed_stream_names() -> None:
    feed = build_market_data_feed(
        exchange="binance",
        binance_ws_connector=_fake_connector,
    )
    names = feed.stream_names()
    assert names == ("ethusdt@aggTrade", "ethusdt@kline_15m")


def test_build_market_data_feed_binance_ws_feed_stream_url() -> None:
    feed = build_market_data_feed(
        exchange="binance",
        binance_ws_connector=_fake_connector,
    )
    url = feed.stream_url()
    assert "/market/stream?streams=" in url


def test_build_market_data_feed_binance_ws_feed_custom_symbols_raises_value_error() -> None:
    """Non-ETH canonical_symbol must raise ValueError even with ws connector."""
    with pytest.raises(ValueError, match="Only ETH-USDT-PERP is supported"):
        build_market_data_feed(
            exchange="binance",
            canonical_symbol="BTC-USDT-PERP",
            raw_symbol="BTCUSDT",
            kline_interval="1m",
            binance_ws_connector=_fake_connector,
        )


def test_build_market_data_feed_binance_ws_connector_wins_over_shell() -> None:
    """When both connector and allow flag are set, connector takes precedence."""
    feed = build_market_data_feed(
        exchange="binance",
        binance_ws_connector=_fake_connector,
        allow_binance_without_ws_connector=True,
    )
    assert isinstance(feed, BinanceWebSocketMarketDataFeed)
