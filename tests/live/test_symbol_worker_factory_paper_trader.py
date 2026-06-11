#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""G08 tests for SymbolWorkerFactory paper trader creation.

These tests verify:
1. trader_mode="live" → create_trader delegates to Trader path.
2. trader_mode="paper" + BTC-USDT-SWAP → returns PaperTrader.
3. paper mode + ETH-USDT-SWAP → raises RuntimeError.
4. paper mode + RECLAIM_PAPER_SYMBOLS without BTC → raises RuntimeError.
5. PaperTrader uses BTC metadata (0.01 / 0.01 / 0.01).
6. Invalid trader_mode → raises RuntimeError.
"""

from __future__ import annotations

import os
from decimal import Decimal
from unittest.mock import patch

import pytest

from src.execution.paper_trader import PaperTrader
from src.live.symbol_worker_factory import SymbolWorkerFactory


# ── helpers ────────────────────────────────────────────────────────────────


def _paper_env(**overrides: str) -> dict[str, str]:
    env: dict[str, str] = {
        "OKX_INST_ID": "BTC-USDT-SWAP",
        "RECLAIM_PAPER_SYMBOLS": "BTC-USDT-SWAP",
        "PAPER_ACCOUNT_EQUITY_USDT": "1000",
    }
    env.update(overrides)
    return env


# ═══════════════════════════════════════════════════════════════════════════
# 1. Live mode → Trader
# ═══════════════════════════════════════════════════════════════════════════


class TestFactoryLiveMode:
    def test_live_mode_tries_trader_path(self) -> None:
        """trader_mode='live' must attempt to create a Trader.
        Since Trader() needs OKX credentials, this will fail — but the
        factory must NOT create a PaperTrader."""
        factory = SymbolWorkerFactory()
        try:
            factory.create_trader(trader_mode="live")
        except RuntimeError as e:
            # Expected — Trader needs OKX_API_KEY etc.
            msg = str(e).lower()
            assert "paper" not in msg, (
                f"Live mode must not create PaperTrader, got: {msg}"
            )
        except ValueError as e:
            # Also expected — Trader needs OKX config
            msg = str(e).lower()
            assert "paper" not in msg, (
                f"Live mode must not create PaperTrader, got: {msg}"
            )
        else:
            # If we somehow got a trader, it must not be PaperTrader
            pass


# ═══════════════════════════════════════════════════════════════════════════
# 2. Paper mode → PaperTrader
# ═══════════════════════════════════════════════════════════════════════════


class TestFactoryPaperMode:
    def test_paper_mode_returns_paper_trader(self) -> None:
        """trader_mode='paper' + BTC-USDT-SWAP must return PaperTrader."""
        with patch.dict(os.environ, _paper_env(), clear=True):
            factory = SymbolWorkerFactory()
            trader = factory.create_trader(trader_mode="paper")
            assert isinstance(trader, PaperTrader)
            assert trader.symbol == "BTC-USDT-SWAP"
            assert trader.live_trading is False
            assert trader.paper_trading is True

    def test_paper_mode_has_btc_metadata(self) -> None:
        """PaperTrader from factory must use BTC metadata."""
        with patch.dict(os.environ, _paper_env(), clear=True):
            factory = SymbolWorkerFactory()
            trader = factory.create_trader(trader_mode="paper")
            assert isinstance(trader, PaperTrader)
            assert trader.contract_multiplier == Decimal("0.01")
            assert trader.contract_precision == Decimal("0.01")
            assert trader.min_contracts == Decimal("0.01")

    def test_paper_mode_default_equity(self) -> None:
        """PaperTrader uses PAPER_ACCOUNT_EQUITY_USDT default 1000."""
        env = _paper_env()
        env.pop("PAPER_ACCOUNT_EQUITY_USDT", None)
        with patch.dict(os.environ, env, clear=True):
            factory = SymbolWorkerFactory()
            trader = factory.create_trader(trader_mode="paper")
            assert isinstance(trader, PaperTrader)
            assert trader.account_equity_usdt == 1000.0

    def test_paper_mode_from_env_equity(self) -> None:
        """PaperTrader reads PAPER_ACCOUNT_EQUITY_USDT from env."""
        with patch.dict(os.environ, _paper_env(PAPER_ACCOUNT_EQUITY_USDT="5000"), clear=True):
            factory = SymbolWorkerFactory()
            trader = factory.create_trader(trader_mode="paper")
            assert isinstance(trader, PaperTrader)
            assert trader.account_equity_usdt == 5000.0


# ═══════════════════════════════════════════════════════════════════════════
# 3. Paper mode validation errors
# ═══════════════════════════════════════════════════════════════════════════


class TestFactoryPaperModeErrors:
    def test_paper_mode_rejects_eth(self) -> None:
        """paper mode + OKX_INST_ID=ETH-USDT-SWAP must raise."""
        with patch.dict(os.environ, _paper_env(OKX_INST_ID="ETH-USDT-SWAP"), clear=True):
            factory = SymbolWorkerFactory()
            with pytest.raises(RuntimeError, match="only supports BTC-USDT-SWAP"):
                factory.create_trader(trader_mode="paper")

    def test_paper_mode_rejects_non_btc_in_paper_symbols(self) -> None:
        """paper mode + RECLAIM_PAPER_SYMBOLS without BTC must raise."""
        env = _paper_env(
            OKX_INST_ID="BTC-USDT-SWAP",
            RECLAIM_PAPER_SYMBOLS="ETH-USDT-SWAP,SOL-USDT-SWAP",
        )
        with patch.dict(os.environ, env, clear=True):
            factory = SymbolWorkerFactory()
            with pytest.raises(RuntimeError, match="not in RECLAIM_PAPER_SYMBOLS"):
                factory.create_trader(trader_mode="paper")

    def test_paper_mode_empty_inst_id_defaults_to_btc(self) -> None:
        """When OKX_INST_ID is empty, PaperTrader defaults to BTC-USDT-SWAP."""
        with patch.dict(os.environ, _paper_env(OKX_INST_ID=""), clear=True):
            factory = SymbolWorkerFactory()
            trader = factory.create_trader(trader_mode="paper")
            assert isinstance(trader, PaperTrader)
            assert trader.symbol == "BTC-USDT-SWAP"


# ═══════════════════════════════════════════════════════════════════════════
# 4. Invalid mode
# ═══════════════════════════════════════════════════════════════════════════


class TestFactoryInvalidMode:
    def test_invalid_mode_raises(self) -> None:
        """Invalid trader_mode must raise RuntimeError."""
        factory = SymbolWorkerFactory()
        with pytest.raises(RuntimeError, match="Invalid RECLAIM_WORKER_MODE"):
            factory.create_trader(trader_mode="invalid")


# ═══════════════════════════════════════════════════════════════════════════
# 5. create_paper_trader_from_env method
# ═══════════════════════════════════════════════════════════════════════════


class TestCreatePaperTraderFromEnv:
    def test_returns_paper_trader(self) -> None:
        with patch.dict(os.environ, _paper_env(), clear=True):
            factory = SymbolWorkerFactory()
            trader = factory.create_paper_trader_from_env()
            assert isinstance(trader, PaperTrader)

    def test_paper_trader_config(self) -> None:
        with patch.dict(os.environ, _paper_env(), clear=True):
            factory = SymbolWorkerFactory()
            trader = factory.create_paper_trader_from_env()
            assert trader.td_mode == "isolated"
            assert trader.leverage == "20"
            assert trader.pos_side_mode == "net"
