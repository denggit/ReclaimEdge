#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_sidecar_order_status_trading_client_port.py
@Description: Verify SidecarTpManager.fetch_sidecar_order_status and
              Trader.fetch_sidecar_order_status use the trading client port
              instead of direct REST.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.execution.trading_client_port import OrderStatusSnapshot


# ======================================================================
# Fake trading client for sidecar status tests
# ======================================================================


class FakeTradingClientForSidecarStatus:
    def __init__(self) -> None:
        self.order_status_calls: list[dict[str, Any]] = []

    async def fetch_order_status(
        self,
        *,
        order_id: str | None = None,
        client_order_id: str | None = None,
    ) -> OrderStatusSnapshot:
        self.order_status_calls.append({"order_id": order_id, "client_order_id": client_order_id})
        return OrderStatusSnapshot(
            order_id=order_id or "ord-001",
            client_order_id=client_order_id,
            status="OPEN",
            filled_qty=Decimal("0.5"),
            avg_fill_price=Decimal("3100.00"),
            raw={"state": "live"},
        )


class FailingTradingClientForSidecarStatus:
    async def fetch_order_status(
        self,
        *,
        order_id: str | None = None,
        client_order_id: str | None = None,
    ) -> OrderStatusSnapshot:
        raise RuntimeError("fetch failed")


# ======================================================================
# Fake Trader (minimal — only what SidecarTpManager needs)
# ======================================================================


class FakeTraderForSidecarStatus:
    symbol = "ETH-USDT-SWAP"

    def __init__(self) -> None:
        self.request = AsyncMock(
            side_effect=AssertionError("must not call trader.request from fetch_sidecar_order_status")
        )


# ======================================================================
# Tests: SidecarTpManager.fetch_sidecar_order_status uses port
# ======================================================================


class TestSidecarTpManagerFetchOrderStatus:
    @pytest.mark.asyncio
    async def test_calls_trading_client_fetch_order_status(self) -> None:
        from src.execution.tp_sl_sidecar_manager import SidecarTpManager

        trader = FakeTraderForSidecarStatus()
        fake_client = FakeTradingClientForSidecarStatus()
        manager = SidecarTpManager(trader, fake_client)

        result = await manager.fetch_sidecar_order_status("ord-test-1")

        assert len(fake_client.order_status_calls) == 1
        assert fake_client.order_status_calls[0]["order_id"] == "ord-test-1"
        assert result["order_id"] == "ord-test-1"
        assert result["status"] == "OPEN"
        assert result["filled_qty"] == 0.5
        assert result["avg_fill_price"] == 3100.00

    @pytest.mark.asyncio
    async def test_does_not_call_trader_request(self) -> None:
        from src.execution.tp_sl_sidecar_manager import SidecarTpManager

        trader = FakeTraderForSidecarStatus()
        fake_client = FakeTradingClientForSidecarStatus()
        manager = SidecarTpManager(trader, fake_client)

        # Must not raise AssertionError from trader.request
        await manager.fetch_sidecar_order_status("ord-test-2")

        trader.request.assert_not_called()

    @pytest.mark.asyncio
    async def test_exception_returns_unknown_dict(self) -> None:
        from src.execution.tp_sl_sidecar_manager import SidecarTpManager

        trader = FakeTraderForSidecarStatus()
        fake_client = FailingTradingClientForSidecarStatus()
        manager = SidecarTpManager(trader, fake_client)

        result = await manager.fetch_sidecar_order_status("ord-fail")

        assert result["order_id"] == "ord-fail"
        assert result["status"] == "UNKNOWN"
        assert result["filled_qty"] is None
        assert result["avg_fill_price"] is None

    @pytest.mark.asyncio
    async def test_filled_status_maps_correctly(self) -> None:
        from src.execution.tp_sl_sidecar_manager import SidecarTpManager

        class FilledTradingClient:
            async def fetch_order_status(self, *, order_id=None, client_order_id=None):
                return OrderStatusSnapshot(
                    order_id=order_id,
                    client_order_id=None,
                    status="FILLED",
                    filled_qty=Decimal("1.0"),
                    avg_fill_price=Decimal("3200.00"),
                )

        trader = FakeTraderForSidecarStatus()
        manager = SidecarTpManager(trader, FilledTradingClient())

        result = await manager.fetch_sidecar_order_status("ord-filled")

        assert result["status"] == "FILLED"
        assert isinstance(result["filled_qty"], float)
        assert result["filled_qty"] == 1.0

    @pytest.mark.asyncio
    async def test_not_found_status_maps_correctly(self) -> None:
        from src.execution.tp_sl_sidecar_manager import SidecarTpManager

        class NotFoundTradingClient:
            async def fetch_order_status(self, *, order_id=None, client_order_id=None):
                return OrderStatusSnapshot(
                    order_id=order_id,
                    client_order_id=None,
                    status="NOT_FOUND",
                )

        trader = FakeTraderForSidecarStatus()
        manager = SidecarTpManager(trader, NotFoundTradingClient())

        result = await manager.fetch_sidecar_order_status("ord-missing")

        assert result["status"] == "NOT_FOUND"
        assert result["filled_qty"] is None
        assert result["avg_fill_price"] is None


# ======================================================================
# Tests: Trader.fetch_sidecar_order_status delegates to manager
# ======================================================================


class TestTraderFetchSidecarOrderStatus:
    @pytest.mark.asyncio
    async def test_delegates_to_tp_sl_manager(self) -> None:
        from src.execution.trader import Trader

        trader = object.__new__(Trader)

        manager_calls: list[str] = []

        class FakeManager:
            async def fetch_sidecar_order_status(self, order_id: str) -> dict[str, Any]:
                manager_calls.append(order_id)
                return {
                    "order_id": order_id,
                    "status": "FILLED",
                    "filled_qty": 1.0,
                    "avg_fill_price": 3100.0,
                }

        trader._tp_sl_manager = FakeManager()
        # Ensure trader.request is NOT called
        trader.request = AsyncMock(
            side_effect=AssertionError("must not call trader.request from fetch_sidecar_order_status")
        )

        result = await trader.fetch_sidecar_order_status("ord-delegated")

        assert manager_calls == ["ord-delegated"]
        assert result["status"] == "FILLED"
        assert result["order_id"] == "ord-delegated"
