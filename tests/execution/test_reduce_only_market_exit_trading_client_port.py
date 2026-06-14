#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_reduce_only_market_exit_trading_client_port.py
@Description: Tests that reduce-only market exit / near-TP reduce orders
              route through TradingClientPort.place_market_order(reduce_only=True)
              instead of direct OKX REST calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.execution.trading_client_port import OrderResult

# ======================================================================
# Fake Trading Client
# ======================================================================


class FakeTradingClient:
    """A fake trading client that records market order calls and returns
    controlled order IDs.  Never touches the real OKX API."""

    def __init__(self):
        self.market_calls: list[dict[str, Any]] = []
        self.next_order_id: str | None = "reduce-only-market-1"

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


def _make_near_tp_reduce_intent(*, side: str = "LONG", eth_qty: float = 0.1,
                                near_tp_reduce_ratio: float = 0.5,
                                near_tp_protective_sl_price: float | None = None) -> Any:
    """Build a minimal NEAR_TP_REDUCE TradeIntent."""
    from src.strategies.boll_cvd_reclaim_strategy import TradeIntent

    return TradeIntent(
        intent_type="NEAR_TP_REDUCE",
        side=side,
        price=3050.0,
        layer_index=1,
        tp_price=3100.0,
        reason="near_tp_reduce_test",
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
        near_tp_reduce_ratio=near_tp_reduce_ratio,
        near_tp_protective_sl_price=near_tp_protective_sl_price,
    )


# ======================================================================
# Tests: MarketExitManager routes through trading_client
# ======================================================================


class TestMarketExitManagerRoutesThroughTradingClientPort:
    """MarketExitManager.market_exit_remaining_position_with_retries must
    route through trading_client.place_market_order(reduce_only=True)."""

    @pytest.mark.asyncio
    async def test_market_exit_calls_place_market_order_with_reduce_only(self):
        """Market exit non-semantic path calls place_market_order(reduce_only=True)."""
        from src.execution.tp_sl_market_exit_manager import MarketExitManager
        from src.execution.trader import PositionSnapshot

        fake_tc = FakeTradingClient()

        # Build a minimal FakeTrader
        trader = MagicMock()
        trader.symbol = "ETH-USDT-SWAP"
        trader.td_mode = "isolated"
        trader.pos_side_mode = "net"
        trader.min_contracts = Decimal("0.01")
        trader.contract_multiplier = Decimal("0.1")
        trader.contract_precision = Decimal("0.01")
        trader.position_contracts = Decimal("10")
        trader.near_tp_protective_sl_order_id = None
        trader.middle_runner_protective_sl_order_id = None
        trader.three_stage_post_tp1_protective_sl_order_id = None
        trader.trend_runner_sl_order_id = None
        trader.decimal_to_str = lambda v: format(
            Decimal(str(v)).normalize(), "f"
        )
        trader.price_to_str = lambda v: f"{v:.2f}"

        # Position snapshot: has position, then flat after order
        trader.fetch_position_snapshot = AsyncMock(side_effect=[
            PositionSnapshot("LONG", Decimal("10"), 3000.0, 1.0, Decimal("10")),
            PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0")),
        ])

        # Cleanup mocks
        trader.cancel_existing_reduce_only_orders = AsyncMock()
        trader.cancel_near_tp_protective_stop = AsyncMock()
        trader.cancel_middle_runner_protective_stop = AsyncMock()
        trader.cancel_three_stage_post_tp1_protective_stop = AsyncMock()
        trader.cancel_trend_runner_protective_stop = AsyncMock()
        trader._cleanup_after_market_exit = AsyncMock()
        trader.broker_semantic_executor = None

        # Disable semantic path
        with patch.object(MarketExitManager, "_broker_semantic_market_exit_enabled",
                          return_value=False):
            manager = MarketExitManager(trader, fake_tc)
            ok, message = await manager.market_exit_remaining_position_with_retries(
                "LONG",
                1,
                context="test-market-exit",
            )

        assert ok is True
        assert "reduce-only-market-1" in message
        assert len(fake_tc.market_calls) == 1
        call = fake_tc.market_calls[0]
        assert call["side"] == "LONG"
        assert call["qty"] == Decimal("10")
        assert call["reduce_only"] is True
        assert call["client_order_id"] == ""

    @pytest.mark.asyncio
    async def test_market_exit_no_direct_order_request(self):
        """Market exit must not call request('POST', '/api/v5/trade/order') directly."""
        from src.execution.tp_sl_market_exit_manager import MarketExitManager
        from src.execution.trader import PositionSnapshot

        fake_tc = FakeTradingClient()

        trader = MagicMock()
        trader.symbol = "ETH-USDT-SWAP"
        trader.td_mode = "isolated"
        trader.pos_side_mode = "net"
        trader.min_contracts = Decimal("0.01")
        trader.contract_multiplier = Decimal("0.1")
        trader.contract_precision = Decimal("0.01")
        trader.position_contracts = Decimal("10")
        trader.near_tp_protective_sl_order_id = None
        trader.middle_runner_protective_sl_order_id = None
        trader.three_stage_post_tp1_protective_sl_order_id = None
        trader.trend_runner_sl_order_id = None
        trader.decimal_to_str = lambda v: format(
            Decimal(str(v)).normalize(), "f"
        )
        trader.price_to_str = lambda v: f"{v:.2f}"

        trader.fetch_position_snapshot = AsyncMock(side_effect=[
            PositionSnapshot("LONG", Decimal("10"), 3000.0, 1.0, Decimal("10")),
            PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0")),
        ])
        trader.cancel_existing_reduce_only_orders = AsyncMock()
        trader.cancel_near_tp_protective_stop = AsyncMock()
        trader.cancel_middle_runner_protective_stop = AsyncMock()
        trader.cancel_three_stage_post_tp1_protective_stop = AsyncMock()
        trader.cancel_trend_runner_protective_stop = AsyncMock()
        trader._cleanup_after_market_exit = AsyncMock()
        trader.broker_semantic_executor = None

        # Mock the direct request to detect if it gets called
        mock_request = AsyncMock()
        trader.request = mock_request

        with patch.object(MarketExitManager, "_broker_semantic_market_exit_enabled",
                          return_value=False):
            manager = MarketExitManager(trader, fake_tc)
            await manager.market_exit_remaining_position_with_retries(
                "LONG",
                1,
                context="test-no-direct",
            )

        # Check that no direct POST /api/v5/trade/order was made
        direct_order_calls = [
            c for c in mock_request.call_args_list
            if len(c.args) >= 2 and c.args[0] == "POST" and c.args[1] == "/api/v5/trade/order"
        ]
        assert len(direct_order_calls) == 0, (
            f"Market exit must not call request('POST', '/api/v5/trade/order') directly; "
            f"got {len(direct_order_calls)} call(s)"
        )

    @pytest.mark.asyncio
    async def test_market_exit_short_side(self):
        """Market exit for SHORT side routes with correct side arg."""
        from src.execution.tp_sl_market_exit_manager import MarketExitManager
        from src.execution.trader import PositionSnapshot

        fake_tc = FakeTradingClient()

        trader = MagicMock()
        trader.symbol = "ETH-USDT-SWAP"
        trader.td_mode = "isolated"
        trader.pos_side_mode = "net"
        trader.min_contracts = Decimal("0.01")
        trader.contract_multiplier = Decimal("0.1")
        trader.contract_precision = Decimal("0.01")
        trader.position_contracts = Decimal("5")
        trader.near_tp_protective_sl_order_id = None
        trader.middle_runner_protective_sl_order_id = None
        trader.three_stage_post_tp1_protective_sl_order_id = None
        trader.trend_runner_sl_order_id = None
        trader.decimal_to_str = lambda v: format(
            Decimal(str(v)).normalize(), "f"
        )
        trader.price_to_str = lambda v: f"{v:.2f}"

        trader.fetch_position_snapshot = AsyncMock(side_effect=[
            PositionSnapshot("SHORT", Decimal("5"), 3000.0, 0.5, Decimal("5")),
            PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0")),
        ])
        trader.cancel_existing_reduce_only_orders = AsyncMock()
        trader.cancel_near_tp_protective_stop = AsyncMock()
        trader.cancel_middle_runner_protective_stop = AsyncMock()
        trader.cancel_three_stage_post_tp1_protective_stop = AsyncMock()
        trader.cancel_trend_runner_protective_stop = AsyncMock()
        trader._cleanup_after_market_exit = AsyncMock()
        trader.broker_semantic_executor = None
        mock_request = AsyncMock()
        trader.request = mock_request

        with patch.object(MarketExitManager, "_broker_semantic_market_exit_enabled",
                          return_value=False):
            manager = MarketExitManager(trader, fake_tc)
            await manager.market_exit_remaining_position_with_retries(
                "SHORT",
                1,
                context="test-short-exit",
            )

        assert len(fake_tc.market_calls) == 1
        call = fake_tc.market_calls[0]
        assert call["side"] == "SHORT"
        assert call["reduce_only"] is True


# ======================================================================
# Tests: MarketExitManager missing order_id → fail fast
# ======================================================================


class TestMarketExitManagerMissingOrderId:
    """When place_market_order returns order_id=None, a RuntimeError must be raised."""

    @pytest.mark.asyncio
    async def test_missing_order_id_causes_failure_return(self):
        """When place_market_order returns order_id=None, the retry loop
        catches the RuntimeError and returns (False, error_message)."""
        from src.execution.tp_sl_market_exit_manager import MarketExitManager
        from src.execution.trader import PositionSnapshot

        fake_tc = FakeTradingClient()
        fake_tc.next_order_id = None  # simulate missing ID

        trader = MagicMock()
        trader.symbol = "ETH-USDT-SWAP"
        trader.td_mode = "isolated"
        trader.pos_side_mode = "net"
        trader.min_contracts = Decimal("0.01")
        trader.contract_multiplier = Decimal("0.1")
        trader.contract_precision = Decimal("0.01")
        trader.position_contracts = Decimal("10")
        trader.near_tp_protective_sl_order_id = None
        trader.middle_runner_protective_sl_order_id = None
        trader.three_stage_post_tp1_protective_sl_order_id = None
        trader.trend_runner_sl_order_id = None
        trader.decimal_to_str = lambda v: format(
            Decimal(str(v)).normalize(), "f"
        )
        trader.price_to_str = lambda v: f"{v:.2f}"
        trader.fetch_position_snapshot = AsyncMock(return_value=PositionSnapshot(
            "LONG", Decimal("10"), 3000.0, 1.0, Decimal("10"),
        ))
        trader.broker_semantic_executor = None

        with patch.object(MarketExitManager, "_broker_semantic_market_exit_enabled",
                          return_value=False):
            manager = MarketExitManager(trader, fake_tc)
            ok, message = await manager.market_exit_remaining_position_with_retries(
                "LONG",
                1,
                context="test-missing-id",
            )

        # The RuntimeError is caught by the retry loop → returns (False, ...)
        assert ok is False
        assert "reduce_only_market_exit_missing_order_id" in message


# ======================================================================
# Tests: NearTpExecutionManager near-TP reduce routes through trading_client
# ======================================================================


class TestNearTpReduceRoutesThroughTradingClientPort:
    """NearTpExecutionManager.execute_near_tp_reduce must route through
    trading_client.place_market_order(reduce_only=True)."""

    @pytest.mark.asyncio
    async def test_near_tp_reduce_calls_place_market_order_with_reduce_only(self, monkeypatch):
        """Near-TP reduce calls place_market_order(reduce_only=True)."""
        monkeypatch.setenv("NEAR_TP_PROTECTIVE_SL_ENABLED", "false")
        from src.execution.tp_sl_near_tp_manager import NearTpExecutionManager
        from src.execution.trader import PositionSnapshot, LiveTradeResult

        fake_tc = FakeTradingClient()

        trader = MagicMock()
        trader.symbol = "ETH-USDT-SWAP"
        trader.td_mode = "isolated"
        trader.pos_side_mode = "net"
        trader.min_contracts = Decimal("0.01")
        trader.contract_multiplier = Decimal("0.1")
        trader.contract_precision = Decimal("0.01")
        trader.position_contracts = Decimal("10")
        trader.near_tp_protective_sl_order_id = None

        trader.decimal_to_str = lambda v: format(
            Decimal(str(v)).normalize(), "f"
        )
        trader.price_to_str = lambda v: f"{v:.2f}"
        trader.round_contracts_down = lambda v: v  # identity for testing

        # Position snapshot: has position, then reduced after order
        trader.fetch_position_snapshot = AsyncMock(return_value=PositionSnapshot(
            "LONG", Decimal("10"), 3000.0, 1.0, Decimal("10"),
        ))

        # replace_take_profit mock
        trader.replace_take_profit = AsyncMock(return_value=LiveTradeResult(
            ok=True,
            action="UPDATE_TP",
            order_id=None,
            tp_order_id="tp-1",
            contracts="5",
            tp_price="3100.00",
            message="ok",
            tp_ok=True,
            tp_order_ids=("tp-1",),
            protective_sl_order_id="sl-1",
            protective_sl_price="3010.00",
            protective_sl_ok=True,
        ))

        intent = _make_near_tp_reduce_intent(side="LONG", near_tp_reduce_ratio=0.5,
                                             near_tp_protective_sl_price=3010.0)

        manager = NearTpExecutionManager(
            trader=trader,
            core_tp=None,
            protective_stops=None,
            market_exit=None,
            trading_client=fake_tc,
        )

        result = await manager.execute_near_tp_reduce(intent)

        assert result.ok is True
        assert result.reduce_filled is True
        assert len(fake_tc.market_calls) == 1
        call = fake_tc.market_calls[0]
        assert call["side"] == "LONG"
        # reduce_contracts = 10 * 0.5 = 5
        assert call["qty"] == Decimal("5")
        assert call["reduce_only"] is True
        assert call["client_order_id"] == ""

    @pytest.mark.asyncio
    async def test_near_tp_reduce_no_direct_order_request(self, monkeypatch):
        """Near-TP reduce must not call request('POST', '/api/v5/trade/order') directly."""
        monkeypatch.setenv("NEAR_TP_PROTECTIVE_SL_ENABLED", "false")
        from src.execution.tp_sl_near_tp_manager import NearTpExecutionManager
        from src.execution.trader import PositionSnapshot, LiveTradeResult

        fake_tc = FakeTradingClient()

        trader = MagicMock()
        trader.symbol = "ETH-USDT-SWAP"
        trader.td_mode = "isolated"
        trader.pos_side_mode = "net"
        trader.min_contracts = Decimal("0.01")
        trader.contract_multiplier = Decimal("0.1")
        trader.contract_precision = Decimal("0.01")
        trader.position_contracts = Decimal("10")
        trader.near_tp_protective_sl_order_id = None

        trader.decimal_to_str = lambda v: format(
            Decimal(str(v)).normalize(), "f"
        )
        trader.price_to_str = lambda v: f"{v:.2f}"
        trader.round_contracts_down = lambda v: v

        trader.fetch_position_snapshot = AsyncMock(return_value=PositionSnapshot(
            "LONG", Decimal("10"), 3000.0, 1.0, Decimal("10"),
        ))
        trader.replace_take_profit = AsyncMock(return_value=LiveTradeResult(
            ok=True,
            action="UPDATE_TP",
            order_id=None,
            tp_order_id="tp-1",
            contracts="5",
            tp_price="3100.00",
            message="ok",
            tp_ok=True,
            tp_order_ids=("tp-1",),
            protective_sl_order_id="sl-1",
            protective_sl_price="3010.00",
            protective_sl_ok=True,
        ))

        mock_request = AsyncMock()
        trader.request = mock_request

        intent = _make_near_tp_reduce_intent(side="LONG", near_tp_reduce_ratio=0.5,
                                             near_tp_protective_sl_price=3010.0)

        manager = NearTpExecutionManager(
            trader=trader,
            core_tp=None,
            protective_stops=None,
            market_exit=None,
            trading_client=fake_tc,
        )

        await manager.execute_near_tp_reduce(intent)

        direct_order_calls = [
            c for c in mock_request.call_args_list
            if len(c.args) >= 2 and c.args[0] == "POST" and c.args[1] == "/api/v5/trade/order"
        ]
        assert len(direct_order_calls) == 0, (
            f"Near-TP reduce must not call request('POST', '/api/v5/trade/order') directly; "
            f"got {len(direct_order_calls)} call(s)"
        )

    @pytest.mark.asyncio
    async def test_near_tp_reduce_short_side(self, monkeypatch):
        """Near-TP reduce for SHORT side routes with correct side."""
        monkeypatch.setenv("NEAR_TP_PROTECTIVE_SL_ENABLED", "false")
        from src.execution.tp_sl_near_tp_manager import NearTpExecutionManager
        from src.execution.trader import PositionSnapshot, LiveTradeResult

        fake_tc = FakeTradingClient()

        trader = MagicMock()
        trader.symbol = "ETH-USDT-SWAP"
        trader.td_mode = "isolated"
        trader.pos_side_mode = "net"
        trader.min_contracts = Decimal("0.01")
        trader.contract_multiplier = Decimal("0.1")
        trader.contract_precision = Decimal("0.01")
        trader.position_contracts = Decimal("10")
        trader.near_tp_protective_sl_order_id = None

        trader.decimal_to_str = lambda v: format(
            Decimal(str(v)).normalize(), "f"
        )
        trader.price_to_str = lambda v: f"{v:.2f}"
        trader.round_contracts_down = lambda v: v

        trader.fetch_position_snapshot = AsyncMock(return_value=PositionSnapshot(
            "SHORT", Decimal("10"), 3000.0, 1.0, Decimal("10"),
        ))
        trader.replace_take_profit = AsyncMock(return_value=LiveTradeResult(
            ok=True,
            action="UPDATE_TP",
            order_id=None,
            tp_order_id="tp-1",
            contracts="5",
            tp_price="2900.00",
            message="ok",
            tp_ok=True,
            tp_order_ids=("tp-1",),
            protective_sl_order_id="sl-1",
            protective_sl_price="2990.00",
            protective_sl_ok=True,
        ))

        intent = _make_near_tp_reduce_intent(side="SHORT", near_tp_reduce_ratio=0.5,
                                             near_tp_protective_sl_price=2990.0)

        manager = NearTpExecutionManager(
            trader=trader,
            core_tp=None,
            protective_stops=None,
            market_exit=None,
            trading_client=fake_tc,
        )

        await manager.execute_near_tp_reduce(intent)

        assert len(fake_tc.market_calls) == 1
        call = fake_tc.market_calls[0]
        assert call["side"] == "SHORT"
        assert call["reduce_only"] is True


# ======================================================================
# Tests: NearTpExecutionManager missing order_id → fail fast
# ======================================================================


class TestNearTpReduceMissingOrderId:
    """When place_market_order returns order_id=None, a RuntimeError must be raised."""

    @pytest.mark.asyncio
    async def test_missing_order_id_raises_runtime_error(self):
        from src.execution.tp_sl_near_tp_manager import NearTpExecutionManager
        from src.execution.trader import PositionSnapshot

        fake_tc = FakeTradingClient()
        fake_tc.next_order_id = None  # simulate missing ID

        trader = MagicMock()
        trader.symbol = "ETH-USDT-SWAP"
        trader.td_mode = "isolated"
        trader.pos_side_mode = "net"
        trader.min_contracts = Decimal("0.01")
        trader.contract_multiplier = Decimal("0.1")
        trader.contract_precision = Decimal("0.01")
        trader.position_contracts = Decimal("10")
        trader.near_tp_protective_sl_order_id = None

        trader.decimal_to_str = lambda v: format(
            Decimal(str(v)).normalize(), "f"
        )
        trader.price_to_str = lambda v: f"{v:.2f}"
        trader.round_contracts_down = lambda v: v

        trader.fetch_position_snapshot = AsyncMock(return_value=PositionSnapshot(
            "LONG", Decimal("10"), 3000.0, 1.0, Decimal("10"),
        ))

        intent = _make_near_tp_reduce_intent(side="LONG", near_tp_reduce_ratio=0.5)

        manager = NearTpExecutionManager(
            trader=trader,
            core_tp=None,
            protective_stops=None,
            market_exit=None,
            trading_client=fake_tc,
        )

        with pytest.raises(RuntimeError, match="near_tp_reduce_only_market_order_missing_order_id"):
            await manager.execute_near_tp_reduce(intent)


# ======================================================================
# Tests: Sidecar is NOT migrated
# ======================================================================


class TestSidecarMigratedToTradingClientPort:
    """Sidecar entry / fixed TP are now migrated to TradingClientPort (20C-CLEAN-PORTS-08)."""

    def test_sidecar_fixed_tp_uses_place_limit_order(self):
        """SidecarManager.place_sidecar_fixed_take_profit now uses
        .place_limit_order( (migrated)."""
        from pathlib import Path

        text = Path("src/execution/tp_sl_sidecar_manager.py").read_text(encoding="utf-8")
        # The sidecar TP placement now uses TradingClientPort.place_limit_order
        assert ".place_limit_order(" in text, (
            "Sidecar fixed TP must use .place_limit_order( (migrated)"
        )
        # Verify the sidecar does NOT call place_market_order
        assert ".place_market_order(" not in text, (
            "Sidecar should NOT call place_market_order (no reduce-only market order to migrate)"
        )

    def test_trader_sidecar_market_order_uses_place_market_order(self):
        """Trader.place_sidecar_market_order now uses .place_market_order(
        via TradingClientPort — migrated."""
        from pathlib import Path

        text = Path("src/execution/trader.py").read_text(encoding="utf-8")
        sidecar_method_text = ""
        lines = text.splitlines()
        in_method = False
        for line in lines:
            if "def place_sidecar_market_order" in line:
                in_method = True
            elif in_method and line.startswith("    def "):
                in_method = False
            if in_method:
                sidecar_method_text += line + "\n"

        assert ".place_market_order(" in sidecar_method_text, (
            "place_sidecar_market_order must use .place_market_order( (migrated)"
        )
