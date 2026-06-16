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
from src.strategies.boll_cvd_reclaim_strategy import TradeIntent


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
        entry_protective_sl_price=2950.0 if side == "LONG" else 3050.0,
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
    trader.place_protective_stop_with_retries = AsyncMock(
        return_value=(True, "entry-sl-1", "protective_sl_placed")
    )
    trader.market_exit_remaining_position_with_retries = AsyncMock(
        return_value=(True, "market_exit_order_id=exit-1")
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
        """The market entry does NOT call a REST request method — Trader no longer
        exposes request()/headers() tunnels.  All trading goes through
        TradingClientPort.place_market_order()."""
        fake = FakeTradingClient()
        trader = _make_trader(trading_client=fake)
        intent = _make_intent(intent_type="OPEN_LONG", side="LONG")

        # Trader no longer has a request() method — just verify execute_intent
        # completes successfully through the TradingClientPort.
        result = await trader.execute_intent(intent)

        # The FakeTradingClient should have recorded a market order call.
        assert len(fake.market_calls) >= 1, (
            "Expected at least 1 market order call through TradingClientPort"
        )
        assert fake.market_calls[0]["side"] == "LONG"
        assert "entry_filled" in result.message or result.ok

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
    """MARKET_EXIT_RUNNER and UPDATE_TP must NOT call
    place_market_order."""

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


# ======================================================================
# Tests: entry protective SL safety
# ======================================================================


class TestEntryMissingProtectiveSlSafety:
    """When entry has no entry_protective_sl_price, immediate market exit."""

    @pytest.mark.asyncio
    async def test_missing_entry_sl_triggers_market_exit(self) -> None:
        """Intent without entry_protective_sl_price must trigger market exit."""
        fake = FakeTradingClient()
        trader = _make_trader(trading_client=fake)
        # Override: entry SL is not set
        trader.entry_protective_sl_order_id = None
        intent = _make_intent(intent_type="OPEN_LONG", side="LONG")
        # Remove the entry SL price so the safety check triggers
        intent = TradeIntent(
            intent_type="OPEN_LONG",
            side="LONG",
            price=3000.0,
            layer_index=1,
            tp_price=3100.0,
            reason="test",
            size=_FakePositionSize(eth_qty=0.1),
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
            entry_protective_sl_price=None,  # MISSING — triggers safety check
        )

        # Mock market exit to return ok
        trader.market_exit_remaining_position_with_retries = AsyncMock(
            return_value=(True, "market_exit_order_id=exit-safety")
        )

        result = await trader.execute_intent(intent)

        assert result.ok is False
        assert result.entry_filled is True
        assert result.tp_ok is False
        assert result.protective_sl_ok is False
        assert "entry_filled_but_missing_entry_protective_sl" in str(result.message)
        # Must have attempted market exit
        trader.market_exit_remaining_position_with_retries.assert_called()
        call_kwargs = trader.market_exit_remaining_position_with_retries.call_args.kwargs
        assert call_kwargs["context"] == "entry_missing_protective_sl"

    @pytest.mark.asyncio
    async def test_entry_protective_sl_failed_triggers_market_exit(self) -> None:
        """When entry SL placement fails, market exit must be called."""
        fake = FakeTradingClient()
        trader = _make_trader(trading_client=fake)
        # Override: SL placement fails
        trader.place_protective_stop_with_retries = AsyncMock(
            return_value=(False, None, "sl_placement_failed")
        )
        trader.market_exit_remaining_position_with_retries = AsyncMock(
            return_value=(True, "market_exit_order_id=exit-sl-fail")
        )

        # Intent with entry SL price (so it passes the missing check)
        intent = _make_intent(intent_type="OPEN_LONG", side="LONG", eth_qty=0.1)

        result = await trader.execute_intent(intent)

        assert result.ok is False
        assert result.entry_filled is True
        assert result.protective_sl_ok is False
        assert "entry_filled_but_entry_protective_sl_failed" in str(result.message)
        trader.market_exit_remaining_position_with_retries.assert_called()
        call_kwargs = trader.market_exit_remaining_position_with_retries.call_args.kwargs
        assert call_kwargs["context"] == "entry_protective_sl_failed"

    @pytest.mark.asyncio
    async def test_entry_sl_ok_but_tp_failed_preserves_sl_state(self) -> None:
        """When entry SL succeeds but TP fails, SL order_id/preserved to avoid naked position."""
        fake = FakeTradingClient()
        trader = _make_trader(trading_client=fake)
        # SL placement ok
        trader.place_protective_stop_with_retries = AsyncMock(
            return_value=(True, "entry-sl-ok", "protective_sl_placed")
        )
        # TP fails
        trader.replace_take_profit = AsyncMock(
            return_value=LiveTradeResult(
                ok=False,
                action="OPEN_LONG",
                order_id="entry-1",
                tp_order_id=None,
                contracts="1",
                tp_price="3100.00",
                message="tp_failed",
                entry_filled=True,
                tp_ok=False,
                protective_sl_order_id="entry-sl-ok",
                protective_sl_price="2950.00",
                protective_sl_ok=True,
            )
        )

        intent = _make_intent(intent_type="OPEN_LONG", side="LONG", eth_qty=0.1)
        result = await trader.execute_intent(intent)

        assert result.ok is False
        assert result.entry_filled is True
        assert result.tp_ok is False
        assert result.protective_sl_ok is True  # SL is still protected
        assert result.protective_sl_order_id is not None
        assert "entry_filled_but_tp_failed" in str(result.message)

    @pytest.mark.asyncio
    async def test_both_sl_and_tp_succeed_returns_full_success(self) -> None:
        """When both entry SL and TP succeed, ok=True with all state."""
        fake = FakeTradingClient()
        trader = _make_trader(trading_client=fake)
        # Both succeed (default mocks work)
        intent = _make_intent(intent_type="OPEN_LONG", side="LONG", eth_qty=0.1)
        result = await trader.execute_intent(intent)

        assert result.ok is True
        assert result.entry_filled is True
        assert result.tp_ok is True
        assert result.protective_sl_ok is True
        assert result.protective_sl_order_id is not None
