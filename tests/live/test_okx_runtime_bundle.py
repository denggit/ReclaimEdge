#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_okx_runtime_bundle.py
@Description: Tests for LiveRuntimeBundle — OKX runtime creation.
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from src.exchanges.models import ExchangeName
from src.exchanges.runtime_config import ExchangeRuntimeConfig
from src.live.runtime_bundle import LiveRuntimeBundle
from src.live.runtime_factory import create_runtime_bundle


class TestRuntimeBundleCreation:
    """EXCHANGE=okx creates the full runtime bundle with OKX adapters."""

    def test_okx_creates_bundle(self) -> None:
        env = {
            "EXCHANGE": "okx",
            "TRADE_ASSET": "ETH",
            "QUOTE_ASSET": "USDT",
            "MARKET_TYPE": "PERPETUAL",
            "EXCHANGE_API_KEY": "test-key",
            "EXCHANGE_API_SECRET": "test-secret",
            "EXCHANGE_API_PASSPHRASE": "test-pass",
            "LIVE_TRADING": "true",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            bundle = create_runtime_bundle(env)
            assert isinstance(bundle, LiveRuntimeBundle)
            assert bundle.runtime_config.exchange == ExchangeName.OKX
            assert bundle.runtime_config.okx_inst_id == "ETH-USDT-SWAP"
            assert bundle.runtime_config.kline_interval == "15m"
            assert bundle.market_data_client is not None
            assert bundle.trading_client is not None
            assert bundle.trader is not None

    def test_okx_bundle_has_correct_config_mapping(self) -> None:
        """Verify internal OKX adapter mapping from unified config."""
        env = {
            "EXCHANGE": "okx",
            "TRADE_ASSET": "ETH",
            "QUOTE_ASSET": "USDT",
            "MARKET_TYPE": "PERPETUAL",
            "MARGIN_MODE": "isolated",
            "POSITION_MODE": "net",
            "KLINE_INTERVAL": "15m",
            "EXCHANGE_API_KEY": "test-key",
            "EXCHANGE_API_SECRET": "test-secret",
            "EXCHANGE_API_PASSPHRASE": "test-pass",
            "LIVE_TRADING": "true",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            bundle = create_runtime_bundle(env)
            config = bundle.runtime_config
            # ETH + USDT + PERPETUAL -> ETH-USDT-SWAP
            assert config.okx_inst_id == "ETH-USDT-SWAP"
            # KLINE_INTERVAL=15m -> OKX bar=15m
            assert config.kline_interval == "15m"
            # MARGIN_MODE=isolated -> tdMode=isolated
            assert config.margin_mode == "isolated"
            # POSITION_MODE=net -> net mode
            assert config.position_mode == "net"

    def test_trader_has_broker_semantic_executor_bound(self) -> None:
        """Verify broker_semantic_executor is bound by the runtime factory."""
        env = {
            "EXCHANGE": "okx",
            "TRADE_ASSET": "ETH",
            "QUOTE_ASSET": "USDT",
            "MARKET_TYPE": "PERPETUAL",
            "EXCHANGE_API_KEY": "test-key",
            "EXCHANGE_API_SECRET": "test-secret",
            "EXCHANGE_API_PASSPHRASE": "test-pass",
            "LIVE_TRADING": "true",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            bundle = create_runtime_bundle(env)
            trader = bundle.trader
            # broker_semantic_executor should be bound (not raise)
            executor = trader.broker_semantic_executor
            assert executor is not None

    def test_trader_has_private_client_bound(self) -> None:
        """Verify _private_client is bound by the runtime factory."""
        env = {
            "EXCHANGE": "okx",
            "TRADE_ASSET": "ETH",
            "QUOTE_ASSET": "USDT",
            "MARKET_TYPE": "PERPETUAL",
            "EXCHANGE_API_KEY": "test-key",
            "EXCHANGE_API_SECRET": "test-secret",
            "EXCHANGE_API_PASSPHRASE": "test-pass",
            "LIVE_TRADING": "true",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            bundle = create_runtime_bundle(env)
            trader = bundle.trader
            # _private_client should be bound (not None)
            assert trader._private_client is not None


class TestRuntimeBundleBinanceBlocked:
    """EXCHANGE=binance is explicitly blocked."""

    def test_binance_raises_runtime_error(self) -> None:
        env = {
            "EXCHANGE": "binance",
            "LIVE_TRADING": "true",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            with pytest.raises(RuntimeError, match="Binance"):
                create_runtime_bundle(env)
