from __future__ import annotations

import os
from decimal import Decimal
from unittest import mock

import pytest

from src.execution.trader import PositionSnapshot
from src.live.runtime_types import ExecutionState
from src.position_management.sidecar.entry_runtime import (
    _is_okx_rate_limit_error,
    attach_sidecar_after_combined_entry,
)
from src.position_management.sidecar.planner import SidecarExecutionPlan
from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState, TradeIntent
from src.risk.simple_position_sizer import PositionSize


def _make_intent(intent_type: str = "OPEN_LONG", layer_index: int = 1) -> TradeIntent:
    return TradeIntent(
        intent_type=intent_type,  # type: ignore[arg-type]
        side="LONG",
        price=3000.0,
        layer_index=layer_index,
        tp_price=3100.0,
        reason="test",
        size=PositionSize(30.0, 1500.0, 0.5, layer_index, 1.0),
        fast_cvd=0.0,
        previous_fast_cvd=0.0,
        buy_ratio=0.0,
        sell_ratio=0.0,
        boll_upper=3100.0,
        boll_middle=3000.0,
        boll_lower=2900.0,
        ts_ms=1000 + layer_index,
        avg_entry_price=3000.0,
        breakeven_price=3003.0,
        tp_mode="MIDDLE",
    )


class FakeJournal:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict, str | None]] = []

    def append(self, event, payload, position_id=None):  # type: ignore[no-untyped-def]
        self.events.append((event, dict(payload), position_id))

    def record_entry(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        pass


class FakeStateStore:
    def __init__(self) -> None:
        self.saved: list = []

    def save(self, state) -> None:  # type: ignore[no-untyped-def]
        self.saved.append(state)

    def clear(self) -> None:
        self.saved.clear()


class FakeTrader:
    symbol = "ETH-USDT-SWAP"
    contract_multiplier = Decimal("0.1")
    contract_precision = Decimal("0.01")
    min_contracts = Decimal("0.01")
    position_contracts = Decimal("1")
    account_equity_usdt = 1000.0
    leverage = "50"

    def __init__(self) -> None:
        self.sidecar_tp_calls: list[dict] = []
        self.market_exits: list[dict] = []
        self._tp_failures: list[Exception] = []

    def decimal_to_str(self, value):
        if not isinstance(value, Decimal):
            value = Decimal(str(value))
        return format(value.normalize(), "f")

    async def place_sidecar_fixed_take_profit(self, *, side, contracts, tp_price,
                                               client_order_id=None):
        self.sidecar_tp_calls.append({
            "side": side, "contracts": str(contracts), "tp_price": tp_price,
            "client_order_id": client_order_id,
        })
        if self._tp_failures:
            raise self._tp_failures.pop(0)
        return f"tp-{len(self.sidecar_tp_calls)}"

    async def market_exit_remaining_position_with_retries(
        self, side, retry_count, *, context="generic", retry_interval_seconds=None,
    ):
        self.market_exits.append({
            "side": side, "retry_count": retry_count, "context": context,
            "retry_interval_seconds": retry_interval_seconds,
        })
        return True, "ok"

    async def fetch_position_snapshot(self) -> PositionSnapshot:
        return PositionSnapshot("LONG", Decimal("1"), 3000.0, 0.1, Decimal("1"))


def _sidecar_state() -> StrategyPositionState:
    return StrategyPositionState(
        side="LONG",
        layers=1,
        total_entry_qty=0.5,
        avg_entry_price=3000,
        sidecar_enabled_for_position=True,
        sidecar_margin_pct=0.01,
        sidecar_tp_pct=0.004,
    )


def _sidecar_plan(intent: TradeIntent) -> SidecarExecutionPlan:
    return SidecarExecutionPlan(
        enabled=True,
        side=intent.side,
        layer_index=intent.layer_index,
        core_contracts=Decimal("5"),
        sidecar_contracts=Decimal("1"),
        total_contracts=Decimal("6"),
        core_qty=0.5,
        sidecar_qty=0.1,
        total_qty=0.6,
        sidecar_margin_pct=0.01,
        layer_multiplier=1.0,
        sidecar_tp_price=3012.0,
        client_order_id="sc-clordid-001",
    )


def _execution_state() -> ExecutionState:
    return ExecutionState(
        trading_halted=False,
        halt_reason=None,
        cash_before_position=None,
        current_position_id="POS-001",
        last_order_ts_ms=None,
    )


# ── tests ────────────────────────────────────────────────────────────────


def test_is_rate_limit_error_50011() -> None:
    assert _is_okx_rate_limit_error(RuntimeError("code=50011 Rate limit reached")) is True


def test_is_rate_limit_error_text() -> None:
    assert _is_okx_rate_limit_error(RuntimeError("Rate limit reached for endpoint")) is True


def test_is_rate_limit_error_other() -> None:
    assert _is_okx_rate_limit_error(RuntimeError("Invalid order parameters")) is False


@pytest.mark.asyncio
async def test_sidecar_tp_rate_limit_retry_succeeds() -> None:
    """First call fails with 50011, second succeeds → no market exit, leg opens normally."""
    trader = FakeTrader()
    # First call fails with rate limit, second succeeds
    trader._tp_failures = [RuntimeError("50011: Rate limit reached")]

    journal = FakeJournal()
    store = FakeStateStore()
    state = _sidecar_state()
    intent_ = _make_intent()
    plan = _sidecar_plan(intent_)
    exec_state = _execution_state()

    with mock.patch.dict(os.environ, {
        "SIDECAR_TP_PLACE_RETRY_COUNT": "3",
        "SIDECAR_TP_PLACE_RETRY_INTERVAL_SECONDS": "0.01",
        "SIDECAR_TP_PLACE_RETRY_BACKOFF_MULTIPLIER": "1.0",
    }):
        ok = await attach_sidecar_after_combined_entry(
            trader=trader,
            strategy_state=state,
            execution_state=exec_state,
            intent=intent_,
            sidecar_plan=plan,
            journal=journal,
            state_store=store,
            trader_symbol="ETH-USDT-SWAP",
        )

    assert ok is True  # sidecar succeeded
    assert len(trader.sidecar_tp_calls) == 2  # initial + retry
    assert len(trader.market_exits) == 0  # no market exit needed
    assert exec_state.trading_halted is False
    assert len(state.sidecar_legs) == 1
    assert state.sidecar_legs[0]["tp_order_id"] == "tp-2"


@pytest.mark.asyncio
async def test_sidecar_tp_rate_limit_all_fail_halt_only() -> None:
    """All retries fail with 50011, HALT_ONLY → no market exit, trading halted."""
    trader = FakeTrader()
    trader._tp_failures = [
        RuntimeError("50011: Rate limit reached"),
        RuntimeError("50011: Rate limit reached"),
        RuntimeError("50011: Rate limit reached"),
    ]

    journal = FakeJournal()
    store = FakeStateStore()
    state = _sidecar_state()
    intent_ = _make_intent()
    plan = _sidecar_plan(intent_)
    exec_state = _execution_state()

    with mock.patch.dict(os.environ, {
        "SIDECAR_TP_PLACE_RETRY_COUNT": "3",
        "SIDECAR_TP_PLACE_RETRY_INTERVAL_SECONDS": "0.01",
        "SIDECAR_TP_PLACE_RETRY_BACKOFF_MULTIPLIER": "1.0",
        "SIDECAR_TP_RATE_LIMIT_FAIL_ACTION": "HALT_ONLY",
    }):
        ok = await attach_sidecar_after_combined_entry(
            trader=trader,
            strategy_state=state,
            execution_state=exec_state,
            intent=intent_,
            sidecar_plan=plan,
            journal=journal,
            state_store=store,
            trader_symbol="ETH-USDT-SWAP",
        )

    assert ok is False
    assert exec_state.trading_halted is True
    # HALT_ONLY now also arms delayed market exit
    assert exec_state.halt_reason == "sidecar_tp_place_rate_limited_delayed_market_exit_armed"
    assert len(trader.market_exits) == 0  # No immediate market exit
    assert state.sidecar_dirty is True
    assert state.delayed_market_exit_armed is True
    # Journal should contain SIDECAR_TP_PLACE_RATE_LIMITED with delayed exit armed
    rate_limited_events = [e for e in journal.events if e[0] == "SIDECAR_TP_PLACE_RATE_LIMITED"]
    assert len(rate_limited_events) == 1
    assert rate_limited_events[0][1]["fail_action"] == "HALT_ONLY"
    assert rate_limited_events[0][1].get("delayed_market_exit_armed") is True


@pytest.mark.asyncio
async def test_sidecar_tp_rate_limit_all_fail_market_exit() -> None:
    """All retries fail with 50011, MARKET_EXIT → delayed market exit armed (no immediate exit)."""
    trader = FakeTrader()
    trader._tp_failures = [
        RuntimeError("50011: Rate limit reached"),
        RuntimeError("50011: Rate limit reached"),
        RuntimeError("50011: Rate limit reached"),
    ]

    journal = FakeJournal()
    store = FakeStateStore()
    state = _sidecar_state()
    intent_ = _make_intent()
    plan = _sidecar_plan(intent_)
    exec_state = _execution_state()

    with mock.patch.dict(os.environ, {
        "SIDECAR_TP_PLACE_RETRY_COUNT": "3",
        "SIDECAR_TP_PLACE_RETRY_INTERVAL_SECONDS": "0.01",
        "SIDECAR_TP_PLACE_RETRY_BACKOFF_MULTIPLIER": "1.0",
        "SIDECAR_TP_RATE_LIMIT_FAIL_ACTION": "MARKET_EXIT",
    }):
        ok = await attach_sidecar_after_combined_entry(
            trader=trader,
            strategy_state=state,
            execution_state=exec_state,
            intent=intent_,
            sidecar_plan=plan,
            journal=journal,
            state_store=store,
            trader_symbol="ETH-USDT-SWAP",
        )

    assert ok is False
    assert exec_state.trading_halted is True
    # No immediate market exit
    assert len(trader.market_exits) == 0
    # Delayed market exit should be armed
    assert state.delayed_market_exit_armed is True
    assert "delayed_market_exit_armed" in exec_state.halt_reason or exec_state.halt_reason == "sidecar_tp_place_rate_limited_delayed_market_exit_armed"


@pytest.mark.asyncio
async def test_sidecar_tp_irrecoverable_error_market_exit() -> None:
    """Non-rate-limit error → delayed market exit armed (no immediate market exit)."""
    trader = FakeTrader()
    trader._tp_failures = [RuntimeError("Invalid order parameters")]

    journal = FakeJournal()
    store = FakeStateStore()
    state = _sidecar_state()
    intent_ = _make_intent()
    plan = _sidecar_plan(intent_)
    exec_state = _execution_state()

    with mock.patch.dict(os.environ, {
        "SIDECAR_TP_PLACE_RETRY_COUNT": "1",
    }):
        ok = await attach_sidecar_after_combined_entry(
            trader=trader,
            strategy_state=state,
            execution_state=exec_state,
            intent=intent_,
            sidecar_plan=plan,
            journal=journal,
            state_store=store,
            trader_symbol="ETH-USDT-SWAP",
        )

    assert ok is False
    # No immediate market exit
    assert len(trader.market_exits) == 0
    # Delayed market exit should be armed
    assert state.delayed_market_exit_armed is True
    assert "delayed_market_exit_armed" in exec_state.halt_reason
    # Journal should contain SIDECAR_TP_PLACE_FAILED (not RATE_LIMITED)
    failed_events = [e for e in journal.events if e[0] == "SIDECAR_TP_PLACE_FAILED"]
    assert len(failed_events) == 1
    assert failed_events[0][1]["error_type"] == "irrecoverable"
    assert failed_events[0][1].get("delayed_market_exit_armed") is True
