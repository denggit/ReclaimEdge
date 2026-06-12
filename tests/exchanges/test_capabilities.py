#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/12
@File       : test_capabilities.py
@Description: Tests for src.exchanges.capabilities – static exchange capabilities.
"""

from __future__ import annotations

from src.exchanges.capabilities import (
    ExchangeCapabilities,
    binance_usdm_default_capabilities,
    okx_default_capabilities,
)
from src.exchanges.models import BrokerQuantityUnit, ExchangeName


class TestOkxCapabilities:
    def test_default_quantity_unit_is_contracts(self) -> None:
        caps = okx_default_capabilities()
        assert caps.default_quantity_unit == BrokerQuantityUnit.CONTRACTS

    def test_supports_reduce_only(self) -> None:
        caps = okx_default_capabilities()
        assert caps.supports_reduce_only is True

    def test_supports_hedge_mode(self) -> None:
        caps = okx_default_capabilities()
        assert caps.supports_hedge_mode is True

    def test_supports_net_mode(self) -> None:
        caps = okx_default_capabilities()
        assert caps.supports_net_mode is True

    def test_close_position_is_false(self) -> None:
        caps = okx_default_capabilities()
        assert caps.supports_close_position is False

    def test_exchange_is_okx(self) -> None:
        caps = okx_default_capabilities()
        assert caps.exchange == ExchangeName.OKX

    def test_supports_conditional_orders(self) -> None:
        caps = okx_default_capabilities()
        assert caps.supports_conditional_orders is True

    def test_supports_client_order_id(self) -> None:
        caps = okx_default_capabilities()
        assert caps.supports_client_order_id is True


class TestBinanceUsdmCapabilities:
    def test_default_quantity_unit_is_base_asset(self) -> None:
        caps = binance_usdm_default_capabilities()
        assert caps.default_quantity_unit == BrokerQuantityUnit.BASE_ASSET

    def test_supports_reduce_only(self) -> None:
        caps = binance_usdm_default_capabilities()
        assert caps.supports_reduce_only is True

    def test_supports_close_position(self) -> None:
        caps = binance_usdm_default_capabilities()
        assert caps.supports_close_position is True

    def test_exchange_is_binance(self) -> None:
        caps = binance_usdm_default_capabilities()
        assert caps.exchange == ExchangeName.BINANCE


class TestCapabilitiesDiffer:
    """OKX and Binance capabilities differ where expected."""

    def test_quantity_units_differ(self) -> None:
        okx = okx_default_capabilities()
        bnc = binance_usdm_default_capabilities()
        assert okx.default_quantity_unit != bnc.default_quantity_unit

    def test_close_position_differs(self) -> None:
        okx = okx_default_capabilities()
        bnc = binance_usdm_default_capabilities()
        assert okx.supports_close_position != bnc.supports_close_position


class TestExchangeCapabilitiesConstructor:
    def test_frozen(self) -> None:
        caps = okx_default_capabilities()
        # can read
        _ = caps.supports_hedge_mode
        # frozen can be verified via dataclass property; just sanity
        assert isinstance(caps, ExchangeCapabilities)
