#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G09c tests for ``src.live.symbol_trader_config``.

These tests verify:
1. SymbolConfig -> TraderInstrumentMetadata mapping (ETH & BTC).
2. SymbolConfig -> TraderMarketSettings mapping (ETH & BTC).
3. Mappings are pure functions (no I/O, no env, no network).
4. Real TOML files from config/symbols/ produce expected values.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from config.symbol_config import (
    SymbolCapitalConfig,
    SymbolConfig,
    SymbolIdentityConfig,
    SymbolMarketConfig,
)
from src.execution.trader_types import TraderInstrumentMetadata, TraderMarketSettings
from src.live.symbol_trader_config import (
    build_trader_instrument_metadata,
    build_trader_market_settings,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _btc_symbol_config() -> SymbolConfig:
    """Minimal BTC-USDT-SWAP SymbolConfig matching the real TOML."""
    return SymbolConfig(
        symbol=SymbolIdentityConfig(inst_id="BTC-USDT-SWAP"),
        market=SymbolMarketConfig(
            contract_value=Decimal("0.01"),
            contract_precision=Decimal("0.01"),
            min_contracts=Decimal("0.01"),
            td_mode="isolated",
            pos_side_mode="net",
        ),
        capital=SymbolCapitalConfig(leverage=Decimal("15")),
    )


def _eth_symbol_config() -> SymbolConfig:
    """Minimal ETH-USDT-SWAP SymbolConfig matching the real TOML."""
    return SymbolConfig(
        symbol=SymbolIdentityConfig(inst_id="ETH-USDT-SWAP"),
        market=SymbolMarketConfig(
            contract_value=Decimal("0.1"),
            contract_precision=Decimal("0.01"),
            min_contracts=Decimal("0.01"),
            td_mode="isolated",
            pos_side_mode="net",
        ),
        capital=SymbolCapitalConfig(leverage=Decimal("15")),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. build_trader_instrument_metadata
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildTraderInstrumentMetadata:
    def test_btc_metadata_from_config(self) -> None:
        cfg = _btc_symbol_config()
        m = build_trader_instrument_metadata(cfg)

        assert isinstance(m, TraderInstrumentMetadata)
        assert m.inst_id == "BTC-USDT-SWAP"
        assert m.contract_multiplier == Decimal("0.01")
        assert m.contract_precision == Decimal("0.01")
        assert m.min_contracts == Decimal("0.01")

    def test_eth_metadata_from_config(self) -> None:
        cfg = _eth_symbol_config()
        m = build_trader_instrument_metadata(cfg)

        assert m.inst_id == "ETH-USDT-SWAP"
        assert m.contract_multiplier == Decimal("0.1")
        assert m.contract_precision == Decimal("0.01")
        assert m.min_contracts == Decimal("0.01")

    def test_contract_multiplier_equals_market_contract_value(self) -> None:
        """contract_multiplier must be the TOML market.contract_value."""
        cfg = _btc_symbol_config()
        m = build_trader_instrument_metadata(cfg)
        assert m.contract_multiplier == cfg.market.contract_value

    def test_contract_precision_equals_market_contract_precision(self) -> None:
        cfg = _btc_symbol_config()
        m = build_trader_instrument_metadata(cfg)
        assert m.contract_precision == cfg.market.contract_precision

    def test_min_contracts_equals_market_min_contracts(self) -> None:
        cfg = _btc_symbol_config()
        m = build_trader_instrument_metadata(cfg)
        assert m.min_contracts == cfg.market.min_contracts

    def test_inst_id_is_stripped(self) -> None:
        cfg = SymbolConfig(
            symbol=SymbolIdentityConfig(inst_id="  BTC-USDT-SWAP  "),
        )
        m = build_trader_instrument_metadata(cfg)
        # TraderInstrumentMetadata.__post_init__ strips inst_id
        assert m.inst_id == "BTC-USDT-SWAP"


# ═══════════════════════════════════════════════════════════════════════════
# 2. build_trader_market_settings
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildTraderMarketSettings:
    def test_btc_market_settings_from_config(self) -> None:
        cfg = _btc_symbol_config()
        s = build_trader_market_settings(cfg)

        assert isinstance(s, TraderMarketSettings)
        assert s.inst_id == "BTC-USDT-SWAP"
        assert s.td_mode == "isolated"
        assert s.pos_side_mode == "net"
        assert s.leverage == Decimal("15")

    def test_eth_market_settings_from_config(self) -> None:
        cfg = _eth_symbol_config()
        s = build_trader_market_settings(cfg)

        assert s.inst_id == "ETH-USDT-SWAP"
        assert s.td_mode == "isolated"
        assert s.pos_side_mode == "net"
        assert s.leverage == Decimal("15")

    def test_td_mode_equals_market_td_mode(self) -> None:
        cfg = _btc_symbol_config()
        s = build_trader_market_settings(cfg)
        assert s.td_mode == cfg.market.td_mode

    def test_pos_side_mode_equals_market_pos_side_mode(self) -> None:
        cfg = _btc_symbol_config()
        s = build_trader_market_settings(cfg)
        assert s.pos_side_mode == cfg.market.pos_side_mode

    def test_leverage_equals_capital_leverage(self) -> None:
        cfg = _btc_symbol_config()
        s = build_trader_market_settings(cfg)
        assert s.leverage == cfg.capital.leverage

    def test_inst_id_is_stripped(self) -> None:
        cfg = SymbolConfig(
            symbol=SymbolIdentityConfig(inst_id="  BTC-USDT-SWAP  "),
        )
        s = build_trader_market_settings(cfg)
        assert s.inst_id == "BTC-USDT-SWAP"


# ═══════════════════════════════════════════════════════════════════════════
# 3. Pure-function verification
# ═══════════════════════════════════════════════════════════════════════════


class TestSymbolTraderConfigPureFunctions:
    def test_no_io_or_env_in_module_source(self) -> None:
        """The module source must contain no I/O or env reads."""
        import src.live.symbol_trader_config as mod

        source_file = mod.__file__
        assert source_file is not None
        with open(source_file) as f:
            source = f.read()

        forbidden = [
            "os.getenv",
            "os.environ",
            "open(",
            "requests.",
            "aiohttp",
            "from src.execution.trader import",
        ]
        for token in forbidden:
            assert token not in source, (
                f"symbol_trader_config must not contain: {token!r}"
            )
