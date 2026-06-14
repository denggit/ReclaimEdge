#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_trader_market_entry_trading_client_port.py
@Description: Tests that Trader.execute_intent() routes market entry orders
              through self.trading_client.place_market_order() instead of
              direct OKX REST calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.execution.trader import LiveTradeResult, Trader
from src.execution.trading_client_port import OrderResult


# ======================================================================
# Fake Trading Client
# ======================================================================


class FakeTradingClient:
    """A fake trading client that records market order calls and returns
    controlled order IDs.  Never touches the real OKX API."""

    def __init__(self):
        self.market_calls: list[dict[str, Any]] = []
        self.next_order_id: str | None = "entry-port-1"

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
            raw={"fake": True},
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


def _make_intent(*, intent_type: str = "OPEN_LONG", side: str = "LONG", eth_qty: float = 0.1) -> Any:
    """Build a minimal TradeIntent suitable for execute_intent."""
    from src.strategies.boll_cvd_reclaim_strategy import TradeIntent

    return TradeIntent(
        intent_type=intent_type,
        side=side,
        price=3000.0,
        layer_index=1,
        tp_price=3100.0,
        reason="test",
        size=_FakePositionSize(eth_qty=eth_qty),
        fast_cvd=0.5,
        previous_fast_cvd=0.4,
        buy_ratio=0.6,
        sell_ratio=0.4,
        boll_upper=3200.0,
        boll_middle=3000.0,
        boll_lower=2800.0,
        ts_ms=1700000000000,
        avg_entry_price=3000.0,
        breakeven_price=3005.0,
        tp_mode="UPPER",
    )


# ======================================================================
# Helper: build a minimal Trader with faked dependencies
# ======================================================================


def _make_trader(*, trading_client: Any) -> Trader:
    """Create a Trader via object.__new__ and inject only what
    execute_intent needs.  The real __init__ is never called so no
    live API key checks run."""
    trader = object.__new__(Trader)
    trader.trading_client = trading_client
    trader.position_contracts = Decimal("0")
    trader.symbol = "ETH-USDT-SWAP"
    trader.td_mode = "isolated"
    trader.pos_side_mode = "net"

    # Static helpers are fine as-is
    # We keep the real eth_qty_to_contracts (relies on contract_multiplier etc.)
    trader.contract_multiplier = Decimal("0.1")
    trader.contract_precision = Decimal("0.01")
    trader.min_contracts = Decimal("0.01")

    # Mock the async dependencies
    from src.execution.trader import PositionSnapshot as TraderPositionSnapshot

    trader.fetch_position_snapshot = AsyncMock(
        return_value=TraderPositionSnapshot(
            side="LONG",
            contracts=Decimal("1"),
            avg_entry_price=3000.0,
            eth_qty=0.1,
            raw_pos=Decimal("1"),
        )
    )
    trader.replace_take_profit = AsyncMock(
        return_value=LiveTradeResult(
            ok=True,
            action="OPEN_LONG",
            order_id="entry-1",
            tp_order_id="tp-1",
            contracts="1",
            tp_price="3100.00",
            message="ok",
            entry_filled=True,
            tp_ok=True,
        )
    )

    return trader


# ======================================================================
# Tests: LONG entry
# ======================================================================


class TestLongMarketEntryRoutesThroughTradingClientPort:
    """LONG intents must call place_market_order with the correct args."""

    @pytest.mark.asyncio
    async def test_long_entry_calls_place_market_order_once(self):
        fake = FakeTradingClient()
        trader = _make_trader(trading_client=fake)
        intent = _make_intent(intent_type="OPEN_LONG", side="LONG", eth_qty=0.2)

        await trader.execute_intent(intent)

        assert len(fake.market_calls) == 1
        call = fake.market_calls[0]
        assert call["side"] == "LONG"
        assert call["reduce_only"] is False
        assert call["client_order_id"] == ""

    @pytest.mark.asyncio
    async def test_long_entry_qty_matches_computed_contracts(self):
        fake = FakeTradingClient()
        trader = _make_trader(trading_client=fake)
        intent = _make_intent(intent_type="OPEN_LONG", side="LONG", eth_qty=0.2)

        await trader.execute_intent(intent)

        # 0.2 ETH / 0.1 contract multiplier = 2 contracts
        assert fake.market_calls[0]["qty"] == Decimal("2")

    @pytest.mark.asyncio
    async def test_long_entry_does_not_call_direct_okx_request(self):
        """The market entry must not go through self.request("POST", "/api/v5/trade/order")."""
        fake = FakeTradingClient()
        trader = _make_trader(trading_client=fake)
        intent = _make_intent(intent_type="OPEN_LONG", side="LONG")

        with patch.object(trader, "request", AsyncMock()) as mock_request:
            await trader.execute_intent(intent)

        # The direct request may be called for TP/SL but NOT for the market entry order.
        # Check that no call had path "/api/v5/trade/order".
        direct_entry_calls = [
            c for c in mock_request.call_args_list
            if len(c.args) >= 2 and c.args[0] == "POST" and c.args[1] == "/api/v5/trade/order"
        ]
        assert len(direct_entry_calls) == 0, (
            f"Market entry must not call request('POST', '/api/v5/trade/order') directly; "
            f"got {len(direct_entry_calls)} call(s)"
        )

    @pytest.mark.asyncio
    async def test_long_entry_returns_valid_live_trade_result(self):
        fake = FakeTradingClient()
        trader = _make_trader(trading_client=fake)
        intent = _make_intent(intent_type="OPEN_LONG", side="LONG")

        result = await trader.execute_intent(intent)

        assert result.ok is True
        assert result.order_id == "entry-port-1"
        assert result.entry_filled is True
        assert result.tp_ok is True


# ======================================================================
# Tests: SHORT entry
# ======================================================================


class TestShortMarketEntryRoutesThroughTradingClientPort:
    """SHORT intents must call place_market_order with correct side."""

    @pytest.mark.asyncio
    async def test_short_entry_calls_place_market_order_once(self):
        fake = FakeTradingClient()
        trader = _make_trader(trading_client=fake)
        intent = _make_intent(intent_type="OPEN_SHORT", side="SHORT", eth_qty=0.15)

        await trader.execute_intent(intent)

        assert len(fake.market_calls) == 1
        call = fake.market_calls[0]
        assert call["side"] == "SHORT"
        assert call["reduce_only"] is False
        assert call["client_order_id"] == ""

    @pytest.mark.asyncio
    async def test_short_entry_qty_matches_computed_contracts(self):
        fake = FakeTradingClient()
        trader = _make_trader(trading_client=fake)
        intent = _make_intent(intent_type="OPEN_SHORT", side="SHORT", eth_qty=0.15)

        await trader.execute_intent(intent)

        # 0.15 ETH / 0.1 = 1.5 -> floor to contract precision (0.01) = 1.5
        assert fake.market_calls[0]["qty"] == Decimal("1.5")


# ======================================================================
# Tests: missing order_id → fail fast
# ======================================================================


class TestMarketEntryMissingOrderId:
    """When place_market_order returns order_id=None, a RuntimeError must be raised."""

    @pytest.mark.asyncio
    async def test_missing_order_id_raises_runtime_error(self):
        fake = FakeTradingClient()
        fake.next_order_id = None  # simulate missing ID
        trader = _make_trader(trading_client=fake)
        intent = _make_intent(intent_type="OPEN_LONG", side="LONG")

        with pytest.raises(RuntimeError, match="market_entry_order_missing_order_id"):
            await trader.execute_intent(intent)


# ======================================================================
# Tests: special intents do NOT call place_market_order
# ======================================================================


class TestSpecialIntentsDoNotRouteToPlaceMarketOrder:
    """NEAR_TP_REDUCE, MARKET_EXIT_RUNNER, and UPDATE_TP must NOT call
    place_market_order."""

    @pytest.mark.asyncio
    async def test_near_tp_reduce_does_not_call_place_market_order(self):
        fake = FakeTradingClient()
        trader = _make_trader(trading_client=fake)
        # execute_near_tp_reduce delegates to _tp_sl_manager — mock it
        trader.execute_near_tp_reduce = AsyncMock(
            return_value=LiveTradeResult(
                ok=True, action="NEAR_TP_REDUCE", order_id=None, tp_order_id=None,
                contracts="1", tp_price="0", message="mocked",
            )
        )
        intent = _make_intent(intent_type="NEAR_TP_REDUCE", side="LONG")

        await trader.execute_intent(intent)

        assert len(fake.market_calls) == 0

    @pytest.mark.asyncio
    async def test_market_exit_runner_does_not_call_place_market_order(self):
        fake = FakeTradingClient()
        trader = _make_trader(trading_client=fake)
        trader.execute_market_exit_runner = AsyncMock(
            return_value=LiveTradeResult(
                ok=True, action="MARKET_EXIT_RUNNER", order_id=None, tp_order_id=None,
                contracts="0", tp_price="0", message="mocked",
            )
        )
        intent = _make_intent(intent_type="MARKET_EXIT_RUNNER", side="LONG")

        await trader.execute_intent(intent)

        assert len(fake.market_calls) == 0

    @pytest.mark.asyncio
    async def test_update_tp_does_not_call_place_market_order(self):
        fake = FakeTradingClient()
        trader = _make_trader(trading_client=fake)
        # replace_take_profit is already mocked in _make_trader
        intent = _make_intent(intent_type="UPDATE_TP", side="LONG")

        await trader.execute_intent(intent)

        assert len(fake.market_calls) == 0
