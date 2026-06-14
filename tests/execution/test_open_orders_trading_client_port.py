#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : test_open_orders_trading_client_port.py
@Description: Tests that cancel_existing_reduce_only_orders uses
              TradingClientPort.fetch_open_orders() and
              TradingClientPort.cancel_order().
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.execution.tp_sl_execution_manager import TpSlExecutionManager
from src.execution.trading_client_port import CancelResult, OrderSnapshot


# ---------------------------------------------------------------------------
# FakeTradingClient
# ---------------------------------------------------------------------------


class FakeTradingClient:
    """Test double that records fetch_open_orders / cancel_order calls."""

    def __init__(self) -> None:
        self.open_orders: list[OrderSnapshot] = []
        self.cancel_calls: list[dict[str, str | None]] = []

    async def fetch_open_orders(self) -> list[OrderSnapshot]:
        return list(self.open_orders)

    async def cancel_order(
        self,
        *,
        order_id: str | None = None,
        client_order_id: str | None = None,
    ) -> CancelResult:
        self.cancel_calls.append(
            {"order_id": order_id, "client_order_id": client_order_id}
        )
        return CancelResult(
            ok=True,
            order_id=order_id,
            client_order_id=client_order_id,
            raw={"fake": True},
        )


# ---------------------------------------------------------------------------
# FakeTrader
# ---------------------------------------------------------------------------


class FakeTrader:
    """Minimal trader stub — fetch_broker_open_orders must NOT be called."""

    symbol = "ETH-USDT-SWAP"

    def __init__(self) -> None:
        self._protected_reduce_only_order_ids: set[str] = set()
        self._managed_reduce_only_order_ids: set[str] = set()
        self._allow_cancel_unmanaged_reduce_only: bool = True

    async def fetch_broker_open_orders(self):
        raise AssertionError("must not call trader.fetch_broker_open_orders")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tp_reduce_only(**kwargs) -> OrderSnapshot:
    defaults = {
        "order_id": "tp-1",
        "client_order_id": None,
        "side": "sell",
        "qty": Decimal("1"),
        "price": Decimal("3100"),
        "reduce_only": True,
        "raw": {},
    }
    defaults.update(kwargs)
    return OrderSnapshot(**defaults)  # type: ignore[arg-type]


def make_entry_order(**kwargs) -> OrderSnapshot:
    defaults = {
        "order_id": "entry-1",
        "client_order_id": None,
        "side": "buy",
        "qty": Decimal("1"),
        "price": Decimal("3000"),
        "reduce_only": False,
        "raw": {},
    }
    defaults.update(kwargs)
    return OrderSnapshot(**defaults)  # type: ignore[arg-type]


# ===================================================================
# Tests
# ===================================================================


@pytest.mark.asyncio
class TestReduceOnlyCancel:
    async def test_reduce_only_order_is_cancelled(self, monkeypatch) -> None:
        """A reduce_only order fetched from fetch_open_orders is cancelled."""
        monkeypatch.delenv("BROKER_SEMANTIC_REDUCE_ONLY_CANCEL_ENABLED", raising=False)

        fake_client = FakeTradingClient()
        fake_client.open_orders = [make_tp_reduce_only(order_id="tp-1")]

        trader = FakeTrader()
        manager = TpSlExecutionManager.__new__(TpSlExecutionManager)
        manager.trader = trader  # type: ignore[attr-defined]
        manager.trading_client = fake_client  # type: ignore[attr-defined]

        await manager.cancel_existing_reduce_only_orders()

        assert len(fake_client.cancel_calls) == 1
        assert fake_client.cancel_calls[0]["order_id"] == "tp-1"


@pytest.mark.asyncio
class TestNonReduceOnlySkipped:
    async def test_non_reduce_only_order_is_not_cancelled(self, monkeypatch) -> None:
        """A non-reduce_only order is skipped."""
        monkeypatch.delenv("BROKER_SEMANTIC_REDUCE_ONLY_CANCEL_ENABLED", raising=False)

        fake_client = FakeTradingClient()
        fake_client.open_orders = [make_entry_order(reduce_only=False)]

        trader = FakeTrader()
        manager = TpSlExecutionManager.__new__(TpSlExecutionManager)
        manager.trader = trader  # type: ignore[attr-defined]
        manager.trading_client = fake_client  # type: ignore[attr-defined]

        await manager.cancel_existing_reduce_only_orders()

        assert fake_client.cancel_calls == []


@pytest.mark.asyncio
class TestMultiOrderMixed:
    async def test_mixed_orders_only_cancel_reduce_only(self, monkeypatch) -> None:
        """Out of three orders, only the two reduce_only ones are cancelled."""
        monkeypatch.delenv("BROKER_SEMANTIC_REDUCE_ONLY_CANCEL_ENABLED", raising=False)

        fake_client = FakeTradingClient()
        fake_client.open_orders = [
            make_tp_reduce_only(order_id="tp-1", reduce_only=True),
            make_entry_order(order_id="entry-1", reduce_only=False),
            make_tp_reduce_only(order_id="tp-2", reduce_only=True),
        ]

        trader = FakeTrader()
        manager = TpSlExecutionManager.__new__(TpSlExecutionManager)
        manager.trader = trader  # type: ignore[attr-defined]
        manager.trading_client = fake_client  # type: ignore[attr-defined]

        await manager.cancel_existing_reduce_only_orders()

        cancelled_ids = [c["order_id"] for c in fake_client.cancel_calls]
        assert cancelled_ids == ["tp-1", "tp-2"]


@pytest.mark.asyncio
class TestTraderFetchBrokerOpenOrdersNotCalled:
    async def test_trader_fetch_broker_open_orders_not_called(self, monkeypatch) -> None:
        """fetch_broker_open_orders on the trader must NOT be called."""
        monkeypatch.delenv("BROKER_SEMANTIC_REDUCE_ONLY_CANCEL_ENABLED", raising=False)

        fake_client = FakeTradingClient()
        fake_client.open_orders = [make_tp_reduce_only(order_id="tp-1")]

        trader = FakeTrader()
        manager = TpSlExecutionManager.__new__(TpSlExecutionManager)
        manager.trader = trader  # type: ignore[attr-defined]
        manager.trading_client = fake_client  # type: ignore[attr-defined]

        await manager.cancel_existing_reduce_only_orders()

        # fetch_broker_open_orders on FakeTrader raises AssertionError if called
        assert len(fake_client.cancel_calls) == 1
