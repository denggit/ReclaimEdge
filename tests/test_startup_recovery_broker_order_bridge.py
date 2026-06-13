"""Integration tests for startup recovery broker order read bridge.

These tests verify that ``apply_main_tp_startup_recovery()`` accepts
pending orders that are BrokerOrder-like objects (not just OKX raw
dicts), correctly identifies reduce-only orders, and preserves halt /
journal semantics — without changing production code.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from src.live.startup_recovery.order_recovery import apply_main_tp_startup_recovery
from src.position_management.sidecar.model import SidecarLegStatus


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeBrokerOrder:
    """Minimal BrokerOrder-like object as returned by the broker semantic path."""

    symbol: str
    reduce_only: bool
    order_id: str


class FakeTrader:
    """Trader stub that returns pre-configured pending orders."""

    symbol = "ETH-USDT-SWAP"

    def __init__(self, pending_orders: list) -> None:
        self.pending_orders = list(pending_orders)
        self.fetch_pending_orders_called = 0
        self.tp_order_id: str | None = None

    async def fetch_pending_orders(self) -> list:
        self.fetch_pending_orders_called += 1
        return list(self.pending_orders)


class FakeJournal:
    """Journal stub that captures appended events."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def append(self, event_type: str, payload: dict, position_id: str | None = None) -> None:
        self.events.append(
            {
                "event_type": event_type,
                "payload": payload,
                "position_id": position_id,
            }
        )


# ---------------------------------------------------------------------------
# State factories
# ---------------------------------------------------------------------------


def make_execution_state() -> SimpleNamespace:
    return SimpleNamespace(
        current_position_id="pos-1",
        trading_halted=False,
        halt_reason=None,
    )


def make_startup_position() -> SimpleNamespace:
    return SimpleNamespace(has_position=True)


def make_saved_state(
    *,
    sidecar_legs: list[dict] | None = None,
    tp_order_id: str | None = None,
    tp_order_ids: list[str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        tp_order_id=tp_order_id,
        tp_order_ids=tp_order_ids or [],
        sidecar_legs=sidecar_legs or [],
    )


# ---------------------------------------------------------------------------
# Test: BrokerOrder-like reduce-only → halt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_main_tp_startup_recovery_detects_broker_order_like_reduce_only_order() -> None:
    """A BrokerOrder-like reduce-only pending order triggers halt + journal."""
    execution_state = make_execution_state()
    journal = FakeJournal()
    trader = FakeTrader(
        [
            FakeBrokerOrder(symbol="ETH-USDT-SWAP", reduce_only=True, order_id="tp-1"),
        ]
    )
    saved_state = make_saved_state()

    await apply_main_tp_startup_recovery(
        execution_state=execution_state,
        saved_state=saved_state,
        startup_position=make_startup_position(),
        trader=trader,
        journal=journal,
    )

    assert trader.fetch_pending_orders_called == 1
    assert execution_state.trading_halted is True
    assert execution_state.halt_reason == "main_tp_order_id_missing_on_startup"
    assert len(journal.events) == 1
    event = journal.events[0]
    assert event["event_type"] == "MAIN_TP_ORDER_ID_MISSING_ON_STARTUP"
    assert event["payload"]["pending_reduce_only_order_count"] == 1
    assert event["payload"]["pending_reduce_only_order_ids"] == ["tp-1"]
    assert event["payload"]["manual_intervention_required"] is True


# ---------------------------------------------------------------------------
# Test: protected sidecar TP id is excluded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_main_tp_startup_recovery_ignores_protected_sidecar_broker_order() -> None:
    """A BrokerOrder-like order whose id is a protected sidecar TP id is ignored."""
    execution_state = make_execution_state()
    journal = FakeJournal()
    trader = FakeTrader(
        [
            FakeBrokerOrder(symbol="ETH-USDT-SWAP", reduce_only=True, order_id="sidecar-tp-1"),
        ]
    )
    saved_state = make_saved_state(
        sidecar_legs=[
            {
                "status": SidecarLegStatus.OPEN.value,
                "tp_order_id": "sidecar-tp-1",
            }
        ]
    )

    await apply_main_tp_startup_recovery(
        execution_state=execution_state,
        saved_state=saved_state,
        startup_position=make_startup_position(),
        trader=trader,
        journal=journal,
    )

    assert trader.fetch_pending_orders_called == 1
    assert execution_state.trading_halted is False
    assert execution_state.halt_reason is None
    assert journal.events == []


# ---------------------------------------------------------------------------
# Test: restored tp_order_id skips pending orders entirely
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_main_tp_startup_recovery_restored_tp_id_skips_broker_order_fetch() -> None:
    """When saved_state carries a tp_order_id, the function returns early without fetching."""
    execution_state = make_execution_state()
    journal = FakeJournal()
    trader = FakeTrader(
        [
            FakeBrokerOrder(symbol="ETH-USDT-SWAP", reduce_only=True, order_id="tp-1"),
        ]
    )
    saved_state = make_saved_state(tp_order_id="restored-tp-1")

    await apply_main_tp_startup_recovery(
        execution_state=execution_state,
        saved_state=saved_state,
        startup_position=make_startup_position(),
        trader=trader,
        journal=journal,
    )

    assert trader.tp_order_id == "restored-tp-1"
    assert trader.fetch_pending_orders_called == 0
    assert execution_state.trading_halted is False
    assert journal.events == []


# ---------------------------------------------------------------------------
# Test: non-matching BrokerOrders do not trigger halt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_main_tp_startup_recovery_ignores_non_matching_broker_orders() -> None:
    """Orders with wrong symbol or non-reduce-only are silently skipped."""
    execution_state = make_execution_state()
    journal = FakeJournal()
    trader = FakeTrader(
        [
            FakeBrokerOrder(symbol="BTC-USDT-SWAP", reduce_only=True, order_id="btc-tp"),
            FakeBrokerOrder(symbol="ETH-USDT-SWAP", reduce_only=False, order_id="entry-1"),
        ]
    )
    saved_state = make_saved_state()

    await apply_main_tp_startup_recovery(
        execution_state=execution_state,
        saved_state=saved_state,
        startup_position=make_startup_position(),
        trader=trader,
        journal=journal,
    )

    assert trader.fetch_pending_orders_called == 1
    assert execution_state.trading_halted is False
    assert execution_state.halt_reason is None
    assert journal.events == []


# ---------------------------------------------------------------------------
# Source-level guard
# ---------------------------------------------------------------------------


def test_order_recovery_does_not_directly_call_broker_semantic_executor() -> None:
    """Startup recovery reads orders only through trader.fetch_pending_orders()."""
    from pathlib import Path

    text = Path("src/live/startup_recovery/order_recovery.py").read_text(encoding="utf-8")

    assert "broker_semantic_executor" not in text
    assert "BROKER_SEMANTIC_STARTUP_RECOVERY" not in text
