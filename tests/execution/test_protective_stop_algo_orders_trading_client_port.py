#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_protective_stop_algo_orders_trading_client_port.py
@Description: Verify ProtectiveStopManager.verify_near_tp_protective_stop
              uses TradingClientPort.fetch_open_algo_orders() instead of
              direct Trader.fetch_pending_algo_orders().
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from src.execution.trading_client_port import AlgoOrderSnapshot


# ======================================================================
# Fake trading client
# ======================================================================


class FakeTradingClientForProtectiveStop:
    def __init__(self) -> None:
        self.algo_order_calls = 0
        self._algo_orders: tuple[AlgoOrderSnapshot, ...] = ()

    def set_algo_orders(self, orders: tuple[AlgoOrderSnapshot, ...]) -> None:
        self._algo_orders = orders

    async def fetch_open_algo_orders(self) -> tuple[AlgoOrderSnapshot, ...]:
        self.algo_order_calls += 1
        return self._algo_orders


# ======================================================================
# Fake Trader
# ======================================================================


class FakeTraderForProtectiveStop:
    symbol = "ETH-USDT-SWAP"
    contract_precision = Decimal("0.01")

    def __init__(self) -> None:
        self.fetch_pending_algo_orders = AsyncMock(
            side_effect=AssertionError(
                "must not call trader.fetch_pending_algo_orders from verify_near_tp_protective_stop"
            )
        )

    @staticmethod
    def decimal_to_str(value: Decimal | str | int | float) -> str:
        if not isinstance(value, Decimal):
            value = Decimal(str(value))
        return format(value.normalize(), "f")

    @staticmethod
    def price_to_str(price: float) -> str:
        import math
        if not math.isfinite(price):
            raise RuntimeError(f"Invalid price: {price}")
        return f"{price:.2f}"

    def _near_tp_protective_stop_matches(self, item, algo_id, side, contracts, stop_price):
        """Legacy matcher — retained for backward compat, not called by verify."""
        raise AssertionError("legacy matcher should not be called from verify_near_tp_protective_stop")


# ======================================================================
# Tests: verify_near_tp_protective_stop uses port
# ======================================================================


class TestVerifyNearTpProtectiveStopUsesPort:
    @pytest.mark.asyncio
    async def test_calls_fetch_open_algo_orders(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        trader = FakeTraderForProtectiveStop()
        fake_client = FakeTradingClientForProtectiveStop()
        fake_client.set_algo_orders((
            AlgoOrderSnapshot(
                order_id="algo-match",
                client_order_id=None,
                side="sell",
                qty=Decimal("1.5"),
                trigger_price=Decimal("2900.00"),
                status="live",
            ),
        ))
        manager = ProtectiveStopManager(trader, fake_client)

        result = await manager.verify_near_tp_protective_stop(
            "algo-match", "LONG", Decimal("1.5"), 2900.00
        )

        assert result is True
        assert fake_client.algo_order_calls >= 1

    @pytest.mark.asyncio
    async def test_does_not_call_trader_fetch_pending_algo_orders(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        trader = FakeTraderForProtectiveStop()
        fake_client = FakeTradingClientForProtectiveStop()
        fake_client.set_algo_orders((
            AlgoOrderSnapshot(
                order_id="algo-x",
                client_order_id=None,
                side="sell",
                qty=Decimal("1.0"),
                trigger_price=Decimal("2900.00"),
            ),
        ))
        manager = ProtectiveStopManager(trader, fake_client)

        # Must not raise AssertionError
        await manager.verify_near_tp_protective_stop(
            "algo-x", "LONG", Decimal("1.0"), 2900.00
        )

        trader.fetch_pending_algo_orders.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_match_returns_false(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        trader = FakeTraderForProtectiveStop()
        fake_client = FakeTradingClientForProtectiveStop()
        fake_client.set_algo_orders((
            AlgoOrderSnapshot(
                order_id="algo-other",
                client_order_id=None,
                side="buy",
                qty=Decimal("2.0"),
                trigger_price=Decimal("3100.00"),
            ),
        ))
        manager = ProtectiveStopManager(trader, fake_client)

        result = await manager.verify_near_tp_protective_stop(
            "algo-match", "LONG", Decimal("1.5"), 2900.00
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_empty_orders_returns_false(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        trader = FakeTraderForProtectiveStop()
        fake_client = FakeTradingClientForProtectiveStop()
        fake_client.set_algo_orders(())
        manager = ProtectiveStopManager(trader, fake_client)

        result = await manager.verify_near_tp_protective_stop(
            "algo-x", "LONG", Decimal("1.0"), 2900.00
        )

        assert result is False


# ======================================================================
# Tests: DTO matcher logic
# ======================================================================


class TestSnapshotMatcher:
    def test_matches_exact(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        trader = FakeTraderForProtectiveStop()
        manager = ProtectiveStopManager(trader, FakeTradingClientForProtectiveStop())

        snap = AlgoOrderSnapshot(
            order_id="algo-001",
            client_order_id=None,
            side="sell",
            qty=Decimal("1.5"),
            trigger_price=Decimal("2900.00"),
        )

        assert manager._near_tp_protective_stop_snapshot_matches(
            snap, "algo-001", "LONG", Decimal("1.5"), 2900.00
        ) is True

    def test_wrong_algo_id_returns_false(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        trader = FakeTraderForProtectiveStop()
        manager = ProtectiveStopManager(trader, FakeTradingClientForProtectiveStop())

        snap = AlgoOrderSnapshot(
            order_id="algo-002",
            client_order_id=None,
            side="sell",
            qty=Decimal("1.5"),
            trigger_price=Decimal("2900.00"),
        )

        assert manager._near_tp_protective_stop_snapshot_matches(
            snap, "algo-001", "LONG", Decimal("1.5"), 2900.00
        ) is False

    def test_wrong_side_returns_false(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        trader = FakeTraderForProtectiveStop()
        manager = ProtectiveStopManager(trader, FakeTradingClientForProtectiveStop())

        snap = AlgoOrderSnapshot(
            order_id="algo-001",
            client_order_id=None,
            side="buy",  # SHORT close is buy, not sell
            qty=Decimal("1.5"),
            trigger_price=Decimal("2900.00"),
        )

        assert manager._near_tp_protective_stop_snapshot_matches(
            snap, "algo-001", "LONG", Decimal("1.5"), 2900.00
        ) is False

    def test_short_close_side_is_buy(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        trader = FakeTraderForProtectiveStop()
        manager = ProtectiveStopManager(trader, FakeTradingClientForProtectiveStop())

        snap = AlgoOrderSnapshot(
            order_id="algo-001",
            client_order_id=None,
            side="buy",  # SHORT close = buy
            qty=Decimal("1.5"),
            trigger_price=Decimal("3200.00"),
        )

        assert manager._near_tp_protective_stop_snapshot_matches(
            snap, "algo-001", "SHORT", Decimal("1.5"), 3200.00
        ) is True

    def test_none_qty_returns_false(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        trader = FakeTraderForProtectiveStop()
        manager = ProtectiveStopManager(trader, FakeTradingClientForProtectiveStop())

        snap = AlgoOrderSnapshot(
            order_id="algo-001",
            client_order_id=None,
            side="sell",
            qty=None,
            trigger_price=Decimal("2900.00"),
        )

        assert manager._near_tp_protective_stop_snapshot_matches(
            snap, "algo-001", "LONG", Decimal("1.5"), 2900.00
        ) is False

    def test_none_trigger_price_returns_false(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        trader = FakeTraderForProtectiveStop()
        manager = ProtectiveStopManager(trader, FakeTradingClientForProtectiveStop())

        snap = AlgoOrderSnapshot(
            order_id="algo-001",
            client_order_id=None,
            side="sell",
            qty=Decimal("1.5"),
            trigger_price=None,
        )

        assert manager._near_tp_protective_stop_snapshot_matches(
            snap, "algo-001", "LONG", Decimal("1.5"), 2900.00
        ) is False

    def test_matches_within_tolerance(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        trader = FakeTraderForProtectiveStop()
        manager = ProtectiveStopManager(trader, FakeTradingClientForProtectiveStop())

        # Very close trigger price — within tolerance
        snap = AlgoOrderSnapshot(
            order_id="algo-001",
            client_order_id=None,
            side="sell",
            qty=Decimal("1.5"),
            trigger_price=Decimal("2900.01"),
        )

        assert manager._near_tp_protective_stop_snapshot_matches(
            snap, "algo-001", "LONG", Decimal("1.5"), 2900.00
        ) is True

    def test_qty_within_tolerance(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        trader = FakeTraderForProtectiveStop()
        manager = ProtectiveStopManager(trader, FakeTradingClientForProtectiveStop())

        # Very close qty — within tolerance
        snap = AlgoOrderSnapshot(
            order_id="algo-001",
            client_order_id=None,
            side="sell",
            qty=Decimal("1.5001"),
            trigger_price=Decimal("2900.00"),
        )

        assert manager._near_tp_protective_stop_snapshot_matches(
            snap, "algo-001", "LONG", Decimal("1.5"), 2900.00
        ) is True
