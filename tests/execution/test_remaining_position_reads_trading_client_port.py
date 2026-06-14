#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_remaining_position_reads_trading_client_port.py
@Description: Tests that CoreTakeProfitManager and ProtectiveStopManager
              position reads route through TradingClientPort.fetch_position()
              where safe, and legacy reads are preserved where required.

              Ref: 20C-CLEAN-PORTS-09C
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.execution.trading_client_port import (
    CancelResult,
    OrderResult,
    PositionSnapshot,
)

# ======================================================================
# Fake Trading Client
# ======================================================================


class FakeTradingClient:
    """A fake trading client that records fetch_position / order calls."""

    def __init__(self):
        self.position_reads = 0
        self.limit_calls: list[dict[str, Any]] = []
        self.stop_market_calls: list[dict[str, Any]] = []
        self.cancel_calls: list[dict[str, Any]] = []
        self.position_sequence: list[PositionSnapshot] = []
        self.next_order_id: str | None = "tp-port-1"

    async def fetch_position(self) -> PositionSnapshot:
        self.position_reads += 1
        if self.position_sequence:
            return self.position_sequence.pop(0)
        return PositionSnapshot(side=None, qty=Decimal("0"), avg_entry_price=None, raw={})

    async def place_limit_order(self, *, side, qty, price, reduce_only, client_order_id):
        self.limit_calls.append({
            "side": side,
            "qty": qty,
            "price": price,
            "reduce_only": reduce_only,
            "client_order_id": client_order_id,
        })
        return OrderResult(
            ok=True,
            order_id=self.next_order_id,
            client_order_id=None,
            raw={},
        )

    async def place_stop_market_order(self, *, side, qty, trigger_price, reduce_only, client_order_id):
        self.stop_market_calls.append({
            "side": side,
            "qty": qty,
            "trigger_price": trigger_price,
            "reduce_only": reduce_only,
            "client_order_id": client_order_id,
        })
        return OrderResult(
            ok=True,
            order_id=self.next_order_id,
            client_order_id=None,
            raw={},
        )

    async def cancel_order(self, *, order_id=None, client_order_id=None):
        self.cancel_calls.append({
            "order_id": order_id,
            "client_order_id": client_order_id,
        })
        return CancelResult(ok=True, order_id=order_id, client_order_id=client_order_id, raw={})


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


def _make_tp_intent(*, side: str = "LONG", tp_plan: str = "SINGLE",
                    tp_price: float = 3100.0) -> Any:
    """Build a minimal TradeIntent for take-profit replacement."""
    from src.strategies.boll_cvd_reclaim_strategy import TradeIntent

    return TradeIntent(
        intent_type="UPDATE_TP",
        side=side,
        price=3000.0,
        layer_index=1,
        tp_price=tp_price,
        tp_plan=tp_plan,
        reason="test_tp_replace",
        size=_FakePositionSize(),
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
# FakeTrader — must NOT have working fetch_position_snapshot
# ======================================================================


def _make_core_tp_trader():
    """Build a MagicMock trader for CoreTakeProfitManager tests.
    fetch_position_snapshot must raise — position reads come from trading_client."""

    trader = MagicMock()
    trader.symbol = "ETH-USDT-SWAP"
    trader.td_mode = "isolated"
    trader.pos_side_mode = "net"
    trader.min_contracts = Decimal("0.01")
    trader.contract_multiplier = Decimal("0.1")
    trader.contract_precision = Decimal("0.01")
    trader.position_contracts = Decimal("0")
    trader.tp_order_id = None
    trader.near_tp_protective_sl_order_id = None
    trader.middle_runner_protective_sl_order_id = None
    trader.three_stage_post_tp1_protective_sl_order_id = None
    trader.trend_runner_sl_order_id = None
    trader.middle_bucket_fast_sl_order_id = None
    trader._protected_reduce_only_order_ids = set()
    trader._managed_reduce_only_order_ids = set()
    trader._allow_cancel_unmanaged_reduce_only = True

    trader.decimal_to_str = lambda v: format(Decimal(str(v)).normalize(), "f")
    trader.price_to_str = lambda v: f"{v:.2f}"
    trader.round_contracts_down = lambda v: v

    # --- Must NOT be called for position reads ---
    trader.fetch_position_snapshot = AsyncMock(
        side_effect=AssertionError("must not call trader.fetch_position_snapshot")
    )

    # --- _managed_core_contracts_from_intent (delegates to tp_sl_manager) ---
    trader._managed_core_contracts_from_intent = MagicMock(return_value=None)

    # --- TP/SL management methods (delegated to tp_sl_manager) ---
    trader._cancel_existing_take_profit_orders_for_intent = AsyncMock()
    trader._cancel_stale_runner_protective_stops_for_degrade = AsyncMock()
    trader._place_reduce_only_take_profit_orders = AsyncMock(return_value=["tp-test-mock"])

    # --- Cancel mocks ---
    trader.cancel_existing_reduce_only_orders = AsyncMock()
    trader.cancel_middle_runner_protective_stop = AsyncMock()
    trader.cancel_three_stage_post_tp1_protective_stop = AsyncMock()
    trader.cancel_trend_runner_protective_stop = AsyncMock()
    trader.cancel_near_tp_protective_stop = AsyncMock()

    # --- trader methods used internally ---
    trader.fetch_pending_algo_orders = AsyncMock(return_value=[])
    trader.verify_near_tp_protective_stop = AsyncMock(return_value=False)

    return trader


# ======================================================================
# CoreTakeProfitManager position read tests
# ======================================================================


class TestCoreTakeProfitManagerPositionReadThroughTradingClientPort:
    """CoreTakeProfitManager.replace_take_profit must read position via
    trading_client.fetch_position()."""

    @pytest.mark.asyncio
    async def test_fetch_position_is_called(self):
        """position read uses trading_client.fetch_position()."""
        from src.execution.tp_sl_core_tp_manager import CoreTakeProfitManager

        fake_tc = FakeTradingClient()
        fake_tc.position_sequence = [
            PositionSnapshot(side="LONG", qty=Decimal("10"), avg_entry_price=Decimal("3000"), raw={}),
        ]
        fake_tc.next_order_id = "tp-test-1"

        trader = _make_core_tp_trader()

        manager = CoreTakeProfitManager(
            trader=trader, protective_stops=None, trading_client=fake_tc,
        )
        intent = _make_tp_intent(side="LONG")

        result = await manager.replace_take_profit(intent)

        assert fake_tc.position_reads == 1, (
            f"fetch_position should be called once, got {fake_tc.position_reads}"
        )
        assert result.ok is True
        assert result.tp_order_id == "tp-test-mock"

    @pytest.mark.asyncio
    async def test_trader_fetch_position_snapshot_not_called(self):
        """trader.fetch_position_snapshot must NOT be called in replace_take_profit."""
        from src.execution.tp_sl_core_tp_manager import CoreTakeProfitManager

        fake_tc = FakeTradingClient()
        fake_tc.position_sequence = [
            PositionSnapshot(side="LONG", qty=Decimal("10"), avg_entry_price=Decimal("3000"), raw={}),
        ]
        fake_tc.next_order_id = "tp-test-2"

        trader = _make_core_tp_trader()

        manager = CoreTakeProfitManager(
            trader=trader, protective_stops=None, trading_client=fake_tc,
        )
        intent = _make_tp_intent(side="LONG")
        await manager.replace_take_profit(intent)

        # fetch_position_snapshot must not be called
        trader.fetch_position_snapshot.assert_not_called()

    @pytest.mark.asyncio
    async def test_position_qty_used_for_contracts_semantic(self):
        """position.qty (not .contracts) is used to derive net_contracts_for_sl."""
        from src.execution.tp_sl_core_tp_manager import CoreTakeProfitManager

        fake_tc = FakeTradingClient()
        fake_tc.position_sequence = [
            PositionSnapshot(side="SHORT", qty=Decimal("5"), avg_entry_price=Decimal("3000"), raw={}),
        ]
        fake_tc.next_order_id = "tp-test-3"

        trader = _make_core_tp_trader()

        manager = CoreTakeProfitManager(
            trader=trader, protective_stops=None, trading_client=fake_tc,
        )
        intent = _make_tp_intent(side="SHORT")

        await manager.replace_take_profit(intent)

        # position_contracts should be set to 5 (from position.qty)
        assert trader.position_contracts == Decimal("5"), (
            f"position_contracts should be 5 (from position.qty), "
            f"got {trader.position_contracts}"
        )

    @pytest.mark.asyncio
    async def test_no_position_returns_no_position_to_protect(self):
        """When position is flat/none, returns 'no position to protect'."""
        from src.execution.tp_sl_core_tp_manager import CoreTakeProfitManager

        fake_tc = FakeTradingClient()
        fake_tc.position_sequence = [
            PositionSnapshot(side=None, qty=Decimal("0"), avg_entry_price=None, raw={}),
        ]

        trader = _make_core_tp_trader()

        manager = CoreTakeProfitManager(
            trader=trader, protective_stops=None, trading_client=fake_tc,
        )
        intent = _make_tp_intent(side="LONG")

        result = await manager.replace_take_profit(intent)

        assert result.ok is False
        assert result.message == "no position to protect"
        assert fake_tc.position_reads == 1

    @pytest.mark.asyncio
    async def test_wrong_side_returns_no_position_to_protect(self):
        """When position side does not match intent side, returns 'no position to protect'."""
        from src.execution.tp_sl_core_tp_manager import CoreTakeProfitManager

        fake_tc = FakeTradingClient()
        # Position is LONG, but intent is SHORT
        fake_tc.position_sequence = [
            PositionSnapshot(side="LONG", qty=Decimal("10"), avg_entry_price=Decimal("3000"), raw={}),
        ]

        trader = _make_core_tp_trader()

        manager = CoreTakeProfitManager(
            trader=trader, protective_stops=None, trading_client=fake_tc,
        )
        intent = _make_tp_intent(side="SHORT")

        result = await manager.replace_take_profit(intent)

        assert result.ok is False
        assert result.message == "no position to protect"
        assert fake_tc.position_reads == 1


# ======================================================================
# ProtectiveStopManager — no safe position read to migrate
# ======================================================================


class TestProtectiveStopManagerNoSafePositionReadToMigrate:
    """ProtectiveStopManager has no fetch_position_snapshot() calls.
    It receives contracts as parameters, not by fetching position itself."""

    def test_protective_stop_manager_has_no_position_read_calls(self):
        """Confirm that ProtectiveStopManager source contains no
        fetch_position_snapshot() calls — nothing to migrate."""
        from pathlib import Path
        text = Path(
            "src/execution/tp_sl_protective_stop_manager.py"
        ).read_text(encoding="utf-8")

        assert "fetch_position_snapshot(" not in text, (
            "ProtectiveStopManager already has no fetch_position_snapshot calls — "
            "nothing to migrate"
        )
        assert "fetch_position(" not in text, (
            "ProtectiveStopManager already has no fetch_position calls — "
            "nothing to migrate"
        )


# ======================================================================
# B-class preserve test — no B-class call points in either file
# ======================================================================


class TestNoBClassPositionReadsInTargetFiles:
    """Neither file has B-class (eth_qty / raw_pos) position reads to preserve."""

    def test_no_eth_qty_or_raw_pos_in_core_tp_manager(self):
        """CoreTakeProfitManager after migration: verify the only position read
        is via trading_client and uses .qty (not eth_qty/raw_pos)."""
        from pathlib import Path
        text = Path(
            "src/execution/tp_sl_core_tp_manager.py"
        ).read_text(encoding="utf-8")

        # After migration, the replace_take_profit method should use
        # self.trading_client.fetch_position() and position.qty
        assert "self.trading_client.fetch_position()" in text, (
            "replace_take_profit must use self.trading_client.fetch_position()"
        )
        # No legacy PositionSnapshot eth_qty / raw_pos used
        assert "eth_qty" not in text, (
            "CoreTakeProfitManager must not reference eth_qty"
        )
        assert "raw_pos" not in text, (
            "CoreTakeProfitManager must not reference raw_pos"
        )

    def test_no_eth_qty_or_raw_pos_in_protective_stop_manager(self):
        """ProtectiveStopManager already has no eth_qty/raw_pos references."""
        from pathlib import Path
        text = Path(
            "src/execution/tp_sl_protective_stop_manager.py"
        ).read_text(encoding="utf-8")

        assert "eth_qty" not in text, (
            "ProtectiveStopManager must not reference eth_qty"
        )
        assert "raw_pos" not in text, (
            "ProtectiveStopManager must not reference raw_pos"
        )
