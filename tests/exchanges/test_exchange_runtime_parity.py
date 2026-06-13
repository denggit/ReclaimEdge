#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_exchange_runtime_parity.py
@Description: Runtime parity tests for data_feed vs broker selectors.

Verifies that the broker selector and data_feed selector agree on:
- Default exchange (OKX)
- Explicit exchange strings
- ``normalize_exchange_name`` semantics

Does **not** create real network connections or read environment variables.
"""

from __future__ import annotations

import pytest

from src.data_feed.binance.adapter import BinanceMarketDataFeed
from src.data_feed.okx.adapter import OkxMarketDataFeed
from src.data_feed.selector import (
    build_market_data_feed,
    normalize_exchange_name as normalize_data_feed_exchange_name,
)
from src.exchanges.factory import (
    build_broker_client,
    normalize_exchange_name as normalize_broker_exchange_name,
)
from src.exchanges.models import ExchangeName


# ---------------------------------------------------------------------------
# Fake helpers
# ---------------------------------------------------------------------------


class FakeOkxTrader:
    pass


class FakeBinanceTransport:
    async def send(self, request):
        raise AssertionError("transport should not be used during factory construction")


# ---------------------------------------------------------------------------
# data_feed selector defaults
# ---------------------------------------------------------------------------


class TestDataFeedSelectorDefaults:
    def test_data_feed_default_is_okx(self) -> None:
        feed = build_market_data_feed()
        assert isinstance(feed, OkxMarketDataFeed)

    def test_data_feed_explicit_binance_requires_connector(self) -> None:
        """Explicit binance without ws connector raises ValueError."""
        with pytest.raises(ValueError, match="binance_ws_connector is required"):
            build_market_data_feed(exchange="binance")


# ---------------------------------------------------------------------------
# broker selector defaults
# ---------------------------------------------------------------------------


class TestBrokerSelectorDefaults:
    def test_broker_default_is_okx(self) -> None:
        """When no exchange is specified, broker factory requires okx_client
        (confirming it routes to OKX)."""
        with pytest.raises(ValueError, match="okx_client is required"):
            build_broker_client()

    def test_broker_explicit_binance_requires_transport(self) -> None:
        """Explicit binance without transport raises ValueError."""
        with pytest.raises(ValueError, match="binance_transport is required"):
            build_broker_client(exchange="binance")

    def test_broker_explicit_binance_with_transport_succeeds(self) -> None:
        transport = FakeBinanceTransport()
        client = build_broker_client(
            exchange="binance",
            binance_transport=transport,
        )
        assert client.exchange == ExchangeName.BINANCE


# ---------------------------------------------------------------------------
# normalize_exchange_name parity
# ---------------------------------------------------------------------------


class TestNormalizeExchangeNameParity:
    """Verify data_feed and broker selectors agree on normalize_exchange_name."""

    def test_both_agree_on_none(self) -> None:
        assert normalize_data_feed_exchange_name(None) == ExchangeName.OKX
        assert normalize_broker_exchange_name(None) == ExchangeName.OKX

    def test_both_agree_on_okx_string(self) -> None:
        assert normalize_data_feed_exchange_name("okx") == ExchangeName.OKX
        assert normalize_broker_exchange_name("okx") == ExchangeName.OKX

    def test_both_agree_on_okx_with_whitespace(self) -> None:
        assert normalize_data_feed_exchange_name(" OKX ") == ExchangeName.OKX
        assert normalize_broker_exchange_name(" OKX ") == ExchangeName.OKX

    def test_both_agree_on_binance_string(self) -> None:
        assert normalize_data_feed_exchange_name("binance") == ExchangeName.BINANCE
        assert normalize_broker_exchange_name("binance") == ExchangeName.BINANCE

    def test_both_agree_on_binance_with_whitespace(self) -> None:
        assert normalize_data_feed_exchange_name(" BINANCE ") == ExchangeName.BINANCE
        assert normalize_broker_exchange_name(" BINANCE ") == ExchangeName.BINANCE

    def test_both_agree_on_enum_okx(self) -> None:
        assert normalize_data_feed_exchange_name(ExchangeName.OKX) == ExchangeName.OKX
        assert normalize_broker_exchange_name(ExchangeName.OKX) == ExchangeName.OKX

    def test_both_agree_on_enum_binance(self) -> None:
        assert normalize_data_feed_exchange_name(ExchangeName.BINANCE) == ExchangeName.BINANCE
        assert normalize_broker_exchange_name(ExchangeName.BINANCE) == ExchangeName.BINANCE

    def test_both_reject_unsupported(self) -> None:
        with pytest.raises(ValueError):
            normalize_data_feed_exchange_name("bybit")
        with pytest.raises(ValueError):
            normalize_broker_exchange_name("bybit")


# ---------------------------------------------------------------------------
# data_feed selector ETH-only constraints
# ---------------------------------------------------------------------------


class TestDataFeedEthOnlyConstraints:
    """Verify the data_feed selector locks to ETH-USDT perpetual only."""

    def test_default_is_okx_eth_perp(self) -> None:
        feed = build_market_data_feed()
        assert isinstance(feed, OkxMarketDataFeed)
        assert feed.canonical_symbol == "ETH-USDT-PERP"
        assert feed.raw_symbol == "ETH-USDT-SWAP"

    def test_okx_rejects_btc_canonical(self) -> None:
        with pytest.raises(ValueError, match="Only ETH-USDT-PERP is supported"):
            build_market_data_feed(exchange="okx", canonical_symbol="BTC-USDT-PERP")

    def test_okx_rejects_btc_raw(self) -> None:
        with pytest.raises(ValueError, match="Only ETH-USDT-SWAP is supported for OKX"):
            build_market_data_feed(exchange="okx", raw_symbol="BTC-USDT-SWAP")

    def test_binance_shell_returns_eth_perp(self) -> None:
        feed = build_market_data_feed(
            exchange="binance", allow_binance_without_ws_connector=True
        )
        assert isinstance(feed, BinanceMarketDataFeed)
        assert feed.canonical_symbol == "ETH-USDT-PERP"
        assert feed.raw_symbol == "ETHUSDT"

    def test_binance_rejects_btc_canonical(self) -> None:
        with pytest.raises(ValueError, match="Only ETH-USDT-PERP is supported"):
            build_market_data_feed(
                exchange="binance",
                canonical_symbol="BTC-USDT-PERP",
                allow_binance_without_ws_connector=True,
            )

    def test_binance_rejects_btc_raw(self) -> None:
        with pytest.raises(ValueError, match="Only ETHUSDT is supported for Binance"):
            build_market_data_feed(
                exchange="binance",
                raw_symbol="BTCUSDT",
                allow_binance_without_ws_connector=True,
            )

    def test_binance_rejects_1m_interval(self) -> None:
        with pytest.raises(ValueError, match="Only 15m kline interval is supported for Binance"):
            build_market_data_feed(
                exchange="binance",
                kline_interval="1m",
                allow_binance_without_ws_connector=True,
            )
