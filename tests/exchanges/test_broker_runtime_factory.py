#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : test_broker_runtime_factory.py
@Description: Tests for the exchange broker runtime factory / selector.

Does **not** create real network connections, read environment variables,
or wire into the live trading path.
"""

from __future__ import annotations

import pytest

from src.exchanges.base import BrokerClient
from src.exchanges.binance.client import BinanceBrokerClient
from src.exchanges.binance.semantic_executor import BinanceBrokerSemanticExecutor
from src.exchanges.factory import (
    build_broker_client,
    build_broker_semantic_executor,
    normalize_exchange_name,
    unsupported_exchange_message,
)
from src.exchanges.models import ExchangeName
from src.exchanges.okx.client import OkxBrokerClient
from src.exchanges.okx.semantic_executor import OkxBrokerSemanticExecutor
from src.exchanges.semantics import BrokerSemanticExecutor


# ---------------------------------------------------------------------------
# Fake helpers
# ---------------------------------------------------------------------------


class FakeOkxClient:
    """Minimal fake for testing OKX broker client construction."""

    pass


class FakeBinanceTransport:
    """A transport fake that must never be called during factory construction."""

    async def send(self, request):
        raise AssertionError("transport should not be used during factory construction")


# ---------------------------------------------------------------------------
# normalize_exchange_name
# ---------------------------------------------------------------------------


class TestNormalizeExchangeName:
    def test_none_defaults_to_okx(self) -> None:
        assert normalize_exchange_name(None) == ExchangeName.OKX

    def test_string_okx(self) -> None:
        assert normalize_exchange_name("okx") == ExchangeName.OKX

    def test_string_okx_with_whitespace(self) -> None:
        assert normalize_exchange_name(" OKX ") == ExchangeName.OKX

    def test_string_binance(self) -> None:
        assert normalize_exchange_name("binance") == ExchangeName.BINANCE

    def test_string_binance_with_whitespace(self) -> None:
        assert normalize_exchange_name(" BINANCE ") == ExchangeName.BINANCE

    def test_enum_okx(self) -> None:
        assert normalize_exchange_name(ExchangeName.OKX) == ExchangeName.OKX

    def test_enum_binance(self) -> None:
        assert normalize_exchange_name(ExchangeName.BINANCE) == ExchangeName.BINANCE

    def test_unsupported_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported exchange"):
            normalize_exchange_name("bybit")


# ---------------------------------------------------------------------------
# build_broker_client
# ---------------------------------------------------------------------------


class TestBuildBrokerClient:
    def test_default_exchange_requires_okx_client(self) -> None:
        """Default (None) exchange → OKX, but okx_client must be provided."""
        with pytest.raises(ValueError, match="okx_client is required"):
            build_broker_client()

    def test_exchange_okx_returns_okx_broker_client(self) -> None:
        fake_client = FakeOkxClient()
        client = build_broker_client(exchange="okx", okx_client=fake_client)
        assert isinstance(client, OkxBrokerClient)

    def test_exchange_okx_with_fake_okx_client(self) -> None:
        fake_client = FakeOkxClient()
        client = build_broker_client(exchange="okx", okx_client=fake_client)
        assert isinstance(client, BrokerClient)
        assert client.exchange == ExchangeName.OKX

    def test_exchange_binance_with_injected_transport(self) -> None:
        transport = FakeBinanceTransport()
        client = build_broker_client(
            exchange="binance",
            binance_api_key="fake-key",
            binance_api_secret="fake-secret",
            binance_transport=transport,
        )
        assert isinstance(client, BinanceBrokerClient)
        assert client.exchange == ExchangeName.BINANCE

    def test_exchange_binance_without_transport_raises_by_default(self) -> None:
        with pytest.raises(ValueError, match="binance_transport is required"):
            build_broker_client(exchange="binance")

    def test_exchange_binance_without_transport_allowed_with_flag(self) -> None:
        client = build_broker_client(
            exchange="binance",
            allow_binance_without_transport=True,
        )
        assert isinstance(client, BinanceBrokerClient)
        assert client.exchange == ExchangeName.BINANCE

    def test_exchange_binance_with_transport_and_flag_also_works(self) -> None:
        transport = FakeBinanceTransport()
        client = build_broker_client(
            exchange="binance",
            binance_transport=transport,
            allow_binance_without_transport=True,
        )
        assert isinstance(client, BinanceBrokerClient)

    def test_exchange_binance_with_fake_transport_does_not_trigger_send(self) -> None:
        """Factory construction must not call transport.send()."""
        transport = FakeBinanceTransport()
        client = build_broker_client(
            exchange="binance",
            binance_api_key="fake-key",
            binance_api_secret="fake-secret",
            binance_transport=transport,
            binance_base_url="https://example.com",
        )
        assert isinstance(client, BinanceBrokerClient)
        # If transport.send() were called during construction, the test
        # would have already failed via the AssertionError in the fake.


# ---------------------------------------------------------------------------
# build_broker_semantic_executor
# ---------------------------------------------------------------------------


class TestBuildBrokerSemanticExecutor:
    def test_exchange_okx_returns_okx_semantic_executor(self) -> None:
        fake_client = FakeOkxClient()
        broker = build_broker_client(exchange="okx", okx_client=fake_client)
        executor = build_broker_semantic_executor(broker, exchange="okx")
        assert isinstance(executor, OkxBrokerSemanticExecutor)
        assert isinstance(executor, BrokerSemanticExecutor)
        assert executor.exchange == ExchangeName.OKX

    def test_exchange_binance_returns_binance_semantic_executor(self) -> None:
        transport = FakeBinanceTransport()
        broker = build_broker_client(
            exchange="binance",
            binance_transport=transport,
        )
        executor = build_broker_semantic_executor(broker, exchange="binance")
        assert isinstance(executor, BinanceBrokerSemanticExecutor)
        assert isinstance(executor, BrokerSemanticExecutor)
        assert executor.exchange == ExchangeName.BINANCE

    def test_default_exchange_creates_okx_semantic_executor(self) -> None:
        """When exchange is None, semantic executor defaults to OKX."""
        fake_client = FakeOkxClient()
        broker = build_broker_client(exchange="okx", okx_client=fake_client)
        executor = build_broker_semantic_executor(broker)  # exchange=None
        assert isinstance(executor, OkxBrokerSemanticExecutor)
        assert executor.exchange == ExchangeName.OKX

    def test_unsupported_exchange_raises_value_error(self) -> None:
        fake_client = FakeOkxClient()
        broker = build_broker_client(exchange="okx", okx_client=fake_client)
        with pytest.raises(ValueError, match="Unsupported exchange"):
            build_broker_semantic_executor(broker, exchange="bybit")


# ---------------------------------------------------------------------------
# unsupported_exchange_message (legacy)
# ---------------------------------------------------------------------------


class TestUnsupportedExchangeMessage:
    def test_legacy_function_still_works(self) -> None:
        msg = unsupported_exchange_message(ExchangeName.BYBIT)
        assert "bybit" in msg
        assert "wired" in msg

    def test_legacy_function_okx(self) -> None:
        msg = unsupported_exchange_message(ExchangeName.OKX)
        assert "okx" in msg
