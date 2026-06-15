#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/15
@File       : test_trader_exchange_sizing.py
@Description: Tests for exchange-aware Trader sizing via TraderRuntimeSettings.

Covers:
  - OKX default sizing (0.1 / 0.01 / 0.01)
  - Binance sizing (1 / 0.001 / 0.001)
  - Trader.py does not hardcode ETHUSDT or Binance
  - symbol_allowlist gate works correctly
  - eth_qty_to_contracts correctness for both exchanges
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from src.execution.trader import Trader, TraderRuntimeSettings


# ======================================================================
# 1. OKX default sizing
# ======================================================================


class TestOkxDefaultSizing:
    """OKX default sizing is 0.1 / 0.01 / 0.01."""

    def test_okx_default_contract_multiplier(self) -> None:
        settings = TraderRuntimeSettings(
            symbol="ETH-USDT-SWAP",
            live_trading=True,
        )
        assert settings.contract_multiplier == Decimal("0.1")

    def test_okx_default_contract_precision(self) -> None:
        settings = TraderRuntimeSettings(
            symbol="ETH-USDT-SWAP",
            live_trading=True,
        )
        assert settings.contract_precision == Decimal("0.01")

    def test_okx_default_min_contracts(self) -> None:
        settings = TraderRuntimeSettings(
            symbol="ETH-USDT-SWAP",
            live_trading=True,
        )
        assert settings.min_contracts == Decimal("0.01")

    def test_okx_trader_sizing_values(self) -> None:
        settings = TraderRuntimeSettings(
            symbol="ETH-USDT-SWAP",
            live_trading=True,
            contract_multiplier=Decimal("0.1"),
            contract_precision=Decimal("0.01"),
            min_contracts=Decimal("0.01"),
        )
        trader = Trader(settings=settings)
        assert trader.contract_multiplier == Decimal("0.1")
        assert trader.contract_precision == Decimal("0.01")
        assert trader.min_contracts == Decimal("0.01")

    def test_okx_eth_qty_to_contracts(self) -> None:
        """OKX: 0.05 ETH → 0.5 contracts (multiplier 0.1)."""
        settings = TraderRuntimeSettings(
            symbol="ETH-USDT-SWAP",
            live_trading=True,
            contract_multiplier=Decimal("0.1"),
            contract_precision=Decimal("0.01"),
            min_contracts=Decimal("0.01"),
        )
        trader = Trader(settings=settings)
        contracts = trader.eth_qty_to_contracts(Decimal("0.05"))
        assert contracts == Decimal("0.5")

    def test_okx_eth_qty_to_contracts_edge(self) -> None:
        """OKX: 0.01 ETH → 0.1 contracts."""
        settings = TraderRuntimeSettings(
            symbol="ETH-USDT-SWAP",
            live_trading=True,
            contract_multiplier=Decimal("0.1"),
            contract_precision=Decimal("0.01"),
            min_contracts=Decimal("0.01"),
        )
        trader = Trader(settings=settings)
        contracts = trader.eth_qty_to_contracts(Decimal("0.01"))
        assert contracts == Decimal("0.1")


# ======================================================================
# 2. Binance sizing
# ======================================================================


class TestBinanceSizing:
    """Binance sizing is 1 / 0.001 / 0.001."""

    def test_binance_contract_multiplier_is_1(self) -> None:
        settings = TraderRuntimeSettings(
            symbol="ETHUSDT",
            symbol_allowlist=("ETHUSDT",),
            live_trading=True,
            contract_multiplier=Decimal("1"),
            contract_precision=Decimal("0.001"),
            min_contracts=Decimal("0.001"),
        )
        assert settings.contract_multiplier == Decimal("1")

    def test_binance_contract_precision_is_0001(self) -> None:
        settings = TraderRuntimeSettings(
            symbol="ETHUSDT",
            symbol_allowlist=("ETHUSDT",),
            live_trading=True,
            contract_multiplier=Decimal("1"),
            contract_precision=Decimal("0.001"),
            min_contracts=Decimal("0.001"),
        )
        assert settings.contract_precision == Decimal("0.001")

    def test_binance_min_contracts_is_0001(self) -> None:
        settings = TraderRuntimeSettings(
            symbol="ETHUSDT",
            symbol_allowlist=("ETHUSDT",),
            live_trading=True,
            contract_multiplier=Decimal("1"),
            contract_precision=Decimal("0.001"),
            min_contracts=Decimal("0.001"),
        )
        assert settings.min_contracts == Decimal("0.001")

    def test_binance_trader_sizing_values(self) -> None:
        """Core safety test: Binance sizing is correct in Trader."""
        settings = TraderRuntimeSettings(
            symbol="ETHUSDT",
            symbol_allowlist=("ETHUSDT",),
            live_trading=True,
            contract_multiplier=Decimal("1"),
            contract_precision=Decimal("0.001"),
            min_contracts=Decimal("0.001"),
        )
        trader = Trader(settings=settings)
        assert trader.contract_multiplier == Decimal("1")
        assert trader.contract_precision == Decimal("0.001")
        assert trader.min_contracts == Decimal("0.001")

    def test_binance_eth_qty_to_contracts(self) -> None:
        """Binance: 0.05 ETH → 0.05 contracts (multiplier 1, no scaling).
        This is the core safety test — Binance quantity must not be
        multiplied by 10.
        """
        settings = TraderRuntimeSettings(
            symbol="ETHUSDT",
            symbol_allowlist=("ETHUSDT",),
            live_trading=True,
            contract_multiplier=Decimal("1"),
            contract_precision=Decimal("0.001"),
            min_contracts=Decimal("0.001"),
        )
        trader = Trader(settings=settings)
        contracts = trader.eth_qty_to_contracts(Decimal("0.05"))
        assert contracts == Decimal("0.05")

    def test_binance_eth_qty_to_contracts_small(self) -> None:
        """Binance: 0.001 ETH → 0.001."""
        settings = TraderRuntimeSettings(
            symbol="ETHUSDT",
            symbol_allowlist=("ETHUSDT",),
            live_trading=True,
            contract_multiplier=Decimal("1"),
            contract_precision=Decimal("0.001"),
            min_contracts=Decimal("0.001"),
        )
        trader = Trader(settings=settings)
        contracts = trader.eth_qty_to_contracts(Decimal("0.001"))
        assert contracts == Decimal("0.001")

    def test_binance_rounds_down_to_precision(self) -> None:
        """Binance: 0.0019 ETH → 0.001 (rounded down to 0.001 precision)."""
        settings = TraderRuntimeSettings(
            symbol="ETHUSDT",
            symbol_allowlist=("ETHUSDT",),
            live_trading=True,
            contract_multiplier=Decimal("1"),
            contract_precision=Decimal("0.001"),
            min_contracts=Decimal("0.001"),
        )
        trader = Trader(settings=settings)
        contracts = trader.eth_qty_to_contracts(Decimal("0.0019"))
        assert contracts == Decimal("0.001")


# ======================================================================
# 3. symbol_allowlist gate
# ======================================================================


class TestSymbolAllowlistGate:
    """symbol_allowlist correctly gates which symbols are permitted."""

    def test_okx_symbol_in_default_allowlist(self) -> None:
        settings = TraderRuntimeSettings(
            symbol="ETH-USDT-SWAP",
            live_trading=True,
        )
        # Must not raise
        Trader(settings=settings)

    def test_binance_symbol_with_binance_allowlist(self) -> None:
        settings = TraderRuntimeSettings(
            symbol="ETHUSDT",
            symbol_allowlist=("ETHUSDT",),
            live_trading=True,
        )
        # Must not raise
        Trader(settings=settings)

    def test_binance_symbol_blocked_by_default_allowlist(self) -> None:
        """Default allowlist only has ETH-USDT-SWAP; ETHUSDT is rejected."""
        settings = TraderRuntimeSettings(
            symbol="ETHUSDT",
            live_trading=True,
            # Default symbol_allowlist is ("ETH-USDT-SWAP",) — does not
            # include ETHUSDT.
        )
        with pytest.raises(RuntimeError) as exc_info:
            Trader(settings=settings)
        msg = str(exc_info.value)
        assert "ETH-USDT-SWAP" in msg

    def test_unknown_symbol_blocked(self) -> None:
        settings = TraderRuntimeSettings(
            symbol="BTCUSDT",
            symbol_allowlist=("ETHUSDT",),
            live_trading=True,
        )
        with pytest.raises(RuntimeError) as exc_info:
            Trader(settings=settings)
        msg = str(exc_info.value)
        assert "BTCUSDT" in msg or "ETHUSDT" in msg

    def test_multi_symbol_allowlist(self) -> None:
        """Multi-symbol allowlist supports both OKX and Binance symbols."""
        settings = TraderRuntimeSettings(
            symbol="ETHUSDT",
            symbol_allowlist=("ETH-USDT-SWAP", "ETHUSDT"),
            live_trading=True,
        )
        # Must not raise — ETHUSDT is in the allowlist
        Trader(settings=settings)


# ======================================================================
# 4. Trader.py source-level check — no ETHUSDT or Binance hardcoding
# ======================================================================


class TestTraderPyNoBinanceHardcoding:
    """src/execution/trader.py must NOT contain ETHUSDT or Binance references."""

    TRADER_SOURCE = Path("src/execution/trader.py").read_text(encoding="utf-8")

    def test_no_ethusdt_hardcoded(self) -> None:
        """trader.py must not contain the string 'ETHUSDT'."""
        # Allow in comment lines only
        for i, line in enumerate(self.TRADER_SOURCE.split("\n"), 1):
            stripped = line.strip()
            if "ETHUSDT" in stripped and not stripped.startswith("#"):
                pytest.fail(
                    f"trader.py:{i} must not hardcode ETHUSDT: {stripped}"
                )

    def test_no_binance_hardcoded(self) -> None:
        """trader.py must not contain 'Binance' or 'binance'."""
        for i, line in enumerate(self.TRADER_SOURCE.split("\n"), 1):
            stripped = line.strip()
            if ("Binance" in stripped or "binance" in stripped) and not stripped.startswith("#"):
                pytest.fail(
                    f"trader.py:{i} must not hardcode Binance: {stripped}"
                )
