#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_trader_initialize_balance_port.py
@Description: Tests that Trader.initialize() reads balance through
              TradingClientPort.fetch_balance() instead of the legacy
              fetch_usdt_equity() method.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from src.execution.trader import Trader
from src.execution.trading_client_port import BalanceSnapshot


# ======================================================================
# Fake Trading Client
# ======================================================================


class FakeTradingClient:
    """A fake trading client that records balance reads and returns
    controlled snapshots.  Never touches the real OKX API."""

    def __init__(self):
        self.balance_reads = 0
        self.balance_total = Decimal("1234.56")

    async def fetch_balance(self):
        self.balance_reads += 1
        return BalanceSnapshot(
            asset="USDT",
            total=self.balance_total,
            available=None,
            raw={"fake": True},
        )


# ======================================================================
# Helper: build a minimal Trader with faked dependencies
# ======================================================================


def _make_trader(*, trading_client: FakeTradingClient) -> Trader:
    """Create a Trader via object.__new__ and inject only what
    initialize() needs.  The real __init__ is never called so no
    live API key checks run."""
    trader = object.__new__(Trader)
    trader.trading_client = trading_client

    # initialize() visible and internal dependencies
    trader.symbol = "ETH-USDT-SWAP"
    trader.td_mode = "isolated"
    trader.leverage = "50"
    trader.pos_side_mode = "net"
    trader.max_live_equity_usdt = 10000.0
    trader.contract_multiplier = Decimal("0.1")
    trader.min_contracts = Decimal("0.01")
    trader.position_contracts = Decimal("0")
    trader._private_write_limiter = AsyncMock()
    trader._client = AsyncMock()

    # Mock the async methods that initialize() depends on
    from src.execution.trader import PositionSnapshot as TraderPositionSnapshot

    trader.fetch_usdt_equity = AsyncMock(
        side_effect=AssertionError("must not call fetch_usdt_equity from initialize()")
    )
    trader.set_leverage = AsyncMock()
    trader.fetch_position_snapshot = AsyncMock(
        return_value=TraderPositionSnapshot(
            side=None,
            contracts=Decimal("0"),
            avg_entry_price=0.0,
            eth_qty=0.0,
            raw_pos=Decimal("0"),
        )
    )

    return trader


# ======================================================================
# Tests: initialize() uses fetch_balance() instead of fetch_usdt_equity()
# ======================================================================


class TestInitializeUsesTradingClientFetchBalance:
    @pytest.mark.asyncio
    async def test_initialize_calls_fetch_balance_once(self):
        fake = FakeTradingClient()
        trader = _make_trader(trading_client=fake)

        await trader.initialize()

        assert fake.balance_reads == 1, (
            f"Expected 1 fetch_balance() call, got {fake.balance_reads}"
        )

    @pytest.mark.asyncio
    async def test_initialize_does_not_call_fetch_usdt_equity(self):
        fake = FakeTradingClient()
        trader = _make_trader(trading_client=fake)

        # If fetch_usdt_equity is called, its AsyncMock raises AssertionError
        await trader.initialize()

        trader.fetch_usdt_equity.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_sets_account_equity_from_balance(self):
        fake = FakeTradingClient()
        fake.balance_total = Decimal("5678.90")
        trader = _make_trader(trading_client=fake)

        await trader.initialize()

        assert trader.account_equity_usdt == 5678.90, (
            f"Expected account_equity_usdt=5678.90, got {trader.account_equity_usdt}"
        )

    @pytest.mark.asyncio
    async def test_initialize_still_calls_set_leverage(self):
        fake = FakeTradingClient()
        trader = _make_trader(trading_client=fake)

        await trader.initialize()

        trader.set_leverage.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_initialize_still_calls_fetch_position_snapshot(self):
        fake = FakeTradingClient()
        trader = _make_trader(trading_client=fake)

        await trader.initialize()

        trader.fetch_position_snapshot.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_initialize_sets_position_contracts(self):
        fake = FakeTradingClient()
        trader = _make_trader(trading_client=fake)
        trader.fetch_position_snapshot = AsyncMock(
            return_value=type(trader.fetch_position_snapshot.return_value)(
                side="LONG",
                contracts=Decimal("5"),
                avg_entry_price=3000.0,
                eth_qty=0.5,
                raw_pos=Decimal("5"),
            )
        )

        from src.execution.trader import PositionSnapshot as TraderPositionSnapshot

        trader.fetch_position_snapshot = AsyncMock(
            return_value=TraderPositionSnapshot(
                side="LONG",
                contracts=Decimal("5"),
                avg_entry_price=3000.0,
                eth_qty=0.5,
                raw_pos=Decimal("5"),
            )
        )

        await trader.initialize()

        assert trader.position_contracts == Decimal("5")


# ======================================================================
# Tests: equity cap check still works with Port balance
# ======================================================================


class TestInitializeEquityCap:
    @pytest.mark.asyncio
    async def test_initialize_refuses_when_equity_exceeds_max(self):
        fake = FakeTradingClient()
        fake.balance_total = Decimal("99999.99")
        trader = _make_trader(trading_client=fake)
        trader.max_live_equity_usdt = 100.0

        with pytest.raises(RuntimeError, match="Refusing live trading"):
            await trader.initialize()


# ======================================================================
# Tests: Decimal to float behavior
# ======================================================================


class TestDecimalToFloatBehavior:
    @pytest.mark.asyncio
    async def test_decimal_total_converts_to_float_without_error(self):
        """BalanceSnapshot.total is Decimal — float(balance.total) must
        not cause logging or downstream errors in initialize()."""
        fake = FakeTradingClient()
        fake.balance_total = Decimal("1234.56")
        trader = _make_trader(trading_client=fake)

        # Must not raise
        await trader.initialize()

        assert isinstance(trader.account_equity_usdt, float)
        assert trader.account_equity_usdt == 1234.56

    @pytest.mark.asyncio
    async def test_equity_cap_check_uses_float_value(self):
        """The cap check in initialize compares against
        self.max_live_equity_usdt (a float).  float(balance.total) must
        produce a value comparable to float."""
        fake = FakeTradingClient()
        fake.balance_total = Decimal("25.00")
        trader = _make_trader(trading_client=fake)
        trader.max_live_equity_usdt = 30.0

        # Must not raise — 25.0 <= 30.0
        await trader.initialize()


# ======================================================================
# Tests: error propagation
# ======================================================================


class TestInitializeErrorPropagation:
    @pytest.mark.asyncio
    async def test_initialize_propagates_fetch_balance_error(self):
        class FailingTradingClient:
            async def fetch_balance(self):
                raise RuntimeError("balance failure")

        trader = object.__new__(Trader)
        trader.trading_client = FailingTradingClient()
        trader.fetch_usdt_equity = AsyncMock(
            side_effect=AssertionError("must not call fetch_usdt_equity")
        )

        with pytest.raises(RuntimeError, match="balance failure"):
            await trader.initialize()
