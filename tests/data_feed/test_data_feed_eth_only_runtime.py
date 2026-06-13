#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_data_feed_eth_only_runtime.py
@Description: Verify the data_feed selector is locked to ETH-USDT perpetual only.

All non-ETH canonical symbols, raw symbols, and non-15m kline intervals
must raise ``ValueError``.  Uses a fake connector — no real network calls.
"""

from __future__ import annotations

import pytest

from src.data_feed.binance.adapter import BinanceMarketDataFeed
from src.data_feed.binance.websocket_feed import BinanceWebSocketMarketDataFeed
from src.data_feed.okx.adapter import OkxMarketDataFeed
from src.data_feed.selector import (
    SUPPORTED_BINANCE_RAW_SYMBOL,
    SUPPORTED_CANONICAL_SYMBOL,
    SUPPORTED_KLINE_INTERVAL,
    SUPPORTED_OKX_RAW_SYMBOL,
    build_market_data_feed,
)
from src.exchanges.models import ExchangeName


# ---------------------------------------------------------------------------
# Fake websocket connector (no real I/O)
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
# 1. default (no args) returns OKX with ETH-USDT-PERP / ETH-USDT-SWAP
# ---------------------------------------------------------------------------


def test_default_okx_returns_eth_perp() -> None:
    feed = build_market_data_feed()
    assert isinstance(feed, OkxMarketDataFeed)
    assert feed.canonical_symbol == SUPPORTED_CANONICAL_SYMBOL
    assert feed.raw_symbol == SUPPORTED_OKX_RAW_SYMBOL


# ---------------------------------------------------------------------------
# 2. explicit OKX with canonical_symbol="ETH-USDT-PERP" works
# ---------------------------------------------------------------------------


def test_okx_explicit_eth_perp_canonical() -> None:
    feed = build_market_data_feed(exchange="okx", canonical_symbol="ETH-USDT-PERP")
    assert isinstance(feed, OkxMarketDataFeed)
    assert feed.canonical_symbol == SUPPORTED_CANONICAL_SYMBOL


# ---------------------------------------------------------------------------
# 3. explicit OKX with raw_symbol="ETH-USDT-SWAP" works
# ---------------------------------------------------------------------------


def test_okx_explicit_eth_swap_raw() -> None:
    feed = build_market_data_feed(exchange="okx", raw_symbol="ETH-USDT-SWAP")
    assert isinstance(feed, OkxMarketDataFeed)
    assert feed.raw_symbol == SUPPORTED_OKX_RAW_SYMBOL


# ---------------------------------------------------------------------------
# 4. OKX with canonical_symbol="BTC-USDT-PERP" raises ValueError
# ---------------------------------------------------------------------------


def test_okx_btc_canonical_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Only ETH-USDT-PERP is supported"):
        build_market_data_feed(exchange="okx", canonical_symbol="BTC-USDT-PERP")


# ---------------------------------------------------------------------------
# 5. OKX with raw_symbol="BTC-USDT-SWAP" raises ValueError
# ---------------------------------------------------------------------------


def test_okx_btc_raw_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Only ETH-USDT-SWAP is supported for OKX data feed"):
        build_market_data_feed(exchange="okx", raw_symbol="BTC-USDT-SWAP")


# ---------------------------------------------------------------------------
# 6. Binance shell mode returns ETH-USDT-PERP / ETHUSDT
# ---------------------------------------------------------------------------


def test_binance_shell_returns_eth_perp() -> None:
    feed = build_market_data_feed(
        exchange="binance", allow_binance_without_ws_connector=True
    )
    assert isinstance(feed, BinanceMarketDataFeed)
    assert feed.canonical_symbol == SUPPORTED_CANONICAL_SYMBOL
    assert feed.raw_symbol == SUPPORTED_BINANCE_RAW_SYMBOL


# ---------------------------------------------------------------------------
# 7. Binance with canonical_symbol="BTC-USDT-PERP" raises ValueError
# ---------------------------------------------------------------------------


def test_binance_btc_canonical_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Only ETH-USDT-PERP is supported"):
        build_market_data_feed(
            exchange="binance",
            canonical_symbol="BTC-USDT-PERP",
            allow_binance_without_ws_connector=True,
        )


# ---------------------------------------------------------------------------
# 8. Binance with raw_symbol="BTCUSDT" raises ValueError
# ---------------------------------------------------------------------------


def test_binance_btc_raw_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Only ETHUSDT is supported for Binance data feed"):
        build_market_data_feed(
            exchange="binance",
            raw_symbol="BTCUSDT",
            allow_binance_without_ws_connector=True,
        )


# ---------------------------------------------------------------------------
# 9. Binance with kline_interval="1m" raises ValueError
# ---------------------------------------------------------------------------


def test_binance_1m_interval_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Only 15m kline interval is supported for Binance data feed"):
        build_market_data_feed(
            exchange="binance",
            kline_interval="1m",
            allow_binance_without_ws_connector=True,
        )


# ---------------------------------------------------------------------------
# 10. Binance with kline_interval="15m" works
# ---------------------------------------------------------------------------


def test_binance_15m_interval_works() -> None:
    feed = build_market_data_feed(
        exchange="binance",
        kline_interval="15m",
        allow_binance_without_ws_connector=True,
    )
    assert isinstance(feed, BinanceMarketDataFeed)


# ---------------------------------------------------------------------------
# 11. Binance websocket feed with fake connector returns ETH streams
# ---------------------------------------------------------------------------


def test_binance_ws_with_fake_connector_returns_ws_feed() -> None:
    feed = build_market_data_feed(
        exchange="binance",
        binance_ws_connector=fake_connector,
    )
    assert isinstance(feed, BinanceWebSocketMarketDataFeed)
    assert feed.canonical_symbol == SUPPORTED_CANONICAL_SYMBOL


# ---------------------------------------------------------------------------
# 12. websocket feed stream_names == ("ethusdt@aggTrade", "ethusdt@kline_15m")
# ---------------------------------------------------------------------------


def test_binance_ws_stream_names_are_eth_only() -> None:
    feed = build_market_data_feed(
        exchange="binance",
        binance_ws_connector=fake_connector,
    )
    names = feed.stream_names()
    assert names == ("ethusdt@aggTrade", "ethusdt@kline_15m")


# ---------------------------------------------------------------------------
# Extra: OKX with kline_interval="1m" works (OKX doesn't validate kline_interval at selector level)
# ---------------------------------------------------------------------------


def test_okx_1m_interval_does_not_raise() -> None:
    """OKX does not validate kline_interval — only Binance does."""
    feed = build_market_data_feed(exchange="okx", kline_interval="1m")
    assert isinstance(feed, OkxMarketDataFeed)


# ---------------------------------------------------------------------------
# Extra: Binance with raw_symbol=None defaults to ETHUSDT
# ---------------------------------------------------------------------------


def test_binance_raw_symbol_none_defaults_to_ethusdt() -> None:
    feed = build_market_data_feed(
        exchange="binance",
        allow_binance_without_ws_connector=True,
    )
    assert feed.raw_symbol == SUPPORTED_BINANCE_RAW_SYMBOL


# ---------------------------------------------------------------------------
# Extra: Binance ws connector wins over allow flag (regression)
# ---------------------------------------------------------------------------


def test_binance_ws_connector_wins_over_allow_flag() -> None:
    feed = build_market_data_feed(
        exchange="binance",
        binance_ws_connector=fake_connector,
        allow_binance_without_ws_connector=True,
    )
    assert isinstance(feed, BinanceWebSocketMarketDataFeed)
