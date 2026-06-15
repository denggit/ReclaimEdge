#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_position_reads_trading_client_port.py
@Description: Tests that MarketExitManager and NearTpExecutionManager
              position reads route through TradingClientPort.fetch_position()
              instead of trader.fetch_position_snapshot().
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.execution.trading_client_port import OrderResult, PositionSnapshot

# ======================================================================
# Fake Trading Client
# ======================================================================


class FakeTradingClient:
    """A fake trading client that records fetch_position / market order calls."""

    def __init__(self):
        self.position_reads = 0
        self.market_calls: list[dict[str, Any]] = []
        self.position_sequence: list[PositionSnapshot] = []
        self.next_order_id: str | None = "market-port-1"

    async def fetch_position(self) -> PositionSnapshot:
        self.position_reads += 1
        if self.position_sequence:
            return self.position_sequence.pop(0)
        return PositionSnapshot(side=None, qty=Decimal("0"), avg_entry_price=None, raw={})

    async def place_market_order(self, *, side, qty, reduce_only, client_order_id):
        self.market_calls.append({
            "side": side,
            "qty": qty,
            "reduce_only": reduce_only,
            "client_order_id": client_order_id,
        })
        return OrderResult(
            ok=True,
            order_id=self.next_order_id,
            client_order_id=None,
            raw={},
        )


# ======================================================================
# TradeIntent builder
# ======================================================================


@dataclass
class _FakePositionSize:
    margin_usdt: float = 10.0
    notional_usdt: float = 500.0
    eth_qty: float = 0.1
    layer_index: int = 1
    layer_multiplier: float = 1.0


# ======================================================================
# FakeTrader — must NOT have working fetch_position_snapshot
# ======================================================================


def _make_market_exit_trader():
    """Build a MagicMock trader for MarketExitManager tests.
    fetch_position_snapshot must raise — position reads come from trading_client."""
    trader = MagicMock()
    trader.symbol = "ETH-USDT-SWAP"
    trader.td_mode = "isolated"
    trader.pos_side_mode = "net"
    trader.min_contracts = Decimal("0.01")
    trader.contract_multiplier = Decimal("0.1")
    trader.contract_precision = Decimal("0.01")
    trader.position_contracts = Decimal("10")
    trader.middle_runner_protective_sl_order_id = None
    trader.three_stage_post_tp1_protective_sl_order_id = None
    trader.trend_runner_sl_order_id = None
    trader.decimal_to_str = lambda v: format(Decimal(str(v)).normalize(), "f")
    trader.price_to_str = lambda v: f"{v:.2f}"

    # Position reads MUST come from trading_client now
    trader.fetch_position_snapshot = AsyncMock(
        side_effect=AssertionError("must not call trader.fetch_position_snapshot")
    )

    # Cleanup mocks
    trader.cancel_existing_reduce_only_orders = AsyncMock()
    trader.cancel_middle_runner_protective_stop = AsyncMock()
    trader.cancel_three_stage_post_tp1_protective_stop = AsyncMock()
    trader.cancel_trend_runner_protective_stop = AsyncMock()
    trader._cleanup_after_market_exit = AsyncMock()
    trader.broker_semantic_executor = None
    return trader


# ======================================================================
# MarketExitManager position read tests
# ======================================================================


class TestMarketExitManagerPositionReadThroughTradingClientPort:
    """MarketExitManager.market_exit_remaining_position_with_retries must
    read position via trading_client.fetch_position()."""

    @pytest.mark.asyncio
    async def test_fetch_position_is_called_for_initial_read(self):
        """Initial position read uses trading_client.fetch_position()."""
        from src.execution.tp_sl_market_exit_manager import MarketExitManager

        fake_tc = FakeTradingClient()
        fake_tc.position_sequence = [
            PositionSnapshot(side="LONG", qty=Decimal("10"), avg_entry_price=Decimal("3000"), raw={}),
            PositionSnapshot(side=None, qty=Decimal("0"), avg_entry_price=None, raw={}),
        ]

        trader = _make_market_exit_trader()

        with patch.object(MarketExitManager, "_broker_semantic_market_exit_enabled", return_value=False):
            manager = MarketExitManager(trader, fake_tc)
            ok, message = await manager.market_exit_remaining_position_with_retries(
                "LONG", 1, context="test-fetch-pos",
            )

        assert ok is True
        assert "market-port-1" in message
        # fetch_position was called (initial + refreshed = 2 reads)
        assert fake_tc.position_reads >= 1

        # place_market_order was called with correct qty
        assert len(fake_tc.market_calls) == 1
        call = fake_tc.market_calls[0]
        assert call["qty"] == Decimal("10")
        assert call["reduce_only"] is True

    @pytest.mark.asyncio
    async def test_trader_fetch_position_snapshot_not_called(self):
        """trader.fetch_position_snapshot must NOT be called."""
        from src.execution.tp_sl_market_exit_manager import MarketExitManager

        fake_tc = FakeTradingClient()
        fake_tc.position_sequence = [
            PositionSnapshot(side="LONG", qty=Decimal("10"), avg_entry_price=Decimal("3000"), raw={}),
            PositionSnapshot(side=None, qty=Decimal("0"), avg_entry_price=None, raw={}),
        ]

        trader = _make_market_exit_trader()

        with patch.object(MarketExitManager, "_broker_semantic_market_exit_enabled", return_value=False):
            manager = MarketExitManager(trader, fake_tc)
            await manager.market_exit_remaining_position_with_retries(
                "LONG", 1, context="test-no-trader-pos",
            )

        # trader.fetch_position_snapshot should NOT have been called
        trader.fetch_position_snapshot.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_position_case_does_not_place_order(self):
        """When position is flat, no market order is placed."""
        from src.execution.tp_sl_market_exit_manager import MarketExitManager

        fake_tc = FakeTradingClient()
        fake_tc.position_sequence = [
            PositionSnapshot(side=None, qty=Decimal("0"), avg_entry_price=None, raw={}),
        ]

        trader = _make_market_exit_trader()

        manager = MarketExitManager(trader, fake_tc)
        ok, message = await manager.market_exit_remaining_position_with_retries(
            "LONG", 1, context="test-no-pos",
        )

        assert ok is True
        assert message == "already_flat"
        assert fake_tc.position_reads == 1
        assert len(fake_tc.market_calls) == 0

    @pytest.mark.asyncio
    async def test_position_qty_used_for_order(self):
        """Market order qty comes from position.qty (not position.contracts)."""
        from src.execution.tp_sl_market_exit_manager import MarketExitManager

        fake_tc = FakeTradingClient()
        fake_tc.position_sequence = [
            PositionSnapshot(side="SHORT", qty=Decimal("5"), avg_entry_price=Decimal("3000"), raw={}),
            PositionSnapshot(side=None, qty=Decimal("0"), avg_entry_price=None, raw={}),
        ]

        trader = _make_market_exit_trader()

        with patch.object(MarketExitManager, "_broker_semantic_market_exit_enabled", return_value=False):
            manager = MarketExitManager(trader, fake_tc)
            await manager.market_exit_remaining_position_with_retries(
                "SHORT", 1, context="test-qty",
            )

        assert len(fake_tc.market_calls) == 1
        call = fake_tc.market_calls[0]
        assert call["qty"] == Decimal("5")
        assert call["side"] == "SHORT"
        assert call["reduce_only"] is True



# ======================================================================
# Missing order ID tests — must continue to pass
# ======================================================================


class TestMissingOrderIdStillEnforced:
    """When place_market_order returns order_id=None, the fail-fast must still work."""

    @pytest.mark.asyncio
    async def test_market_exit_missing_order_id(self):
        """Market exit with missing order_id still raises RuntimeError in retry loop."""
        from src.execution.tp_sl_market_exit_manager import MarketExitManager

        fake_tc = FakeTradingClient()
        fake_tc.next_order_id = None  # simulate missing ID
        fake_tc.position_sequence = [
            PositionSnapshot(side="LONG", qty=Decimal("10"), avg_entry_price=Decimal("3000"), raw={}),
        ]

        trader = _make_market_exit_trader()

        with patch.object(MarketExitManager, "_broker_semantic_market_exit_enabled", return_value=False):
            manager = MarketExitManager(trader, fake_tc)
            ok, message = await manager.market_exit_remaining_position_with_retries(
                "LONG", 1, context="test-missing-id",
            )

        assert ok is False
        assert "reduce_only_market_exit_missing_order_id" in message

