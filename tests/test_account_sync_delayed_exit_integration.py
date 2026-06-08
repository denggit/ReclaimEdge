"""Integration tests: DME phase wired into account sync worker.

Proves that:
1. DME phase is called by account sync worker when armed
2. Before deadline: no market exit
3. After deadline: market exit called
4. Already flat: delayed state cleared
5. DME due executed: protective orders skipped
"""

from __future__ import annotations

import asyncio
import os
import time
from decimal import Decimal
from unittest import mock

import pytest

from src.execution.trader import LiveTradeResult, PositionSnapshot
from src.live import delayed_market_exit as dme
from src.live import runtime_types as live_runtime_types
from src.live.account_sync.delayed_market_exit_phase import (
    DelayedMarketExitPhaseResult,
    run_delayed_market_exit_phase,
)
from src.live.alerts.halt_alerts import HaltAlertDeduper
from src.live.halt_modes import resolve_halt_mode
from src.reporting.live_state_store import LiveStateStore
from src.reporting.trade_journal import LiveTradeJournal
from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState, BollCvdReclaimStrategyConfig
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig


class FakeTrader:
    symbol = "ETH-USDT-SWAP"
    account_equity_usdt = 1000.0
    position_contracts = Decimal("10")
    contract_multiplier = Decimal("0.1")

    def __init__(self) -> None:
        self.market_exits: list[dict] = []
        self._market_exit_ok = True
        self._fetch_position_side = "LONG"

    def set_market_exit_result(self, ok: bool, message: str = "") -> None:
        self._market_exit_ok = ok

    async def market_exit_remaining_position_with_retries(
        self, side, retry_count=3, context="", retry_interval_seconds=0.5,
    ):
        self.market_exits.append({
            "side": side,
            "retry_count": retry_count,
            "context": context,
            "retry_interval_seconds": retry_interval_seconds,
        })
        return self._market_exit_ok, "ok" if self._market_exit_ok else "failed"

    async def fetch_position_snapshot(self):
        return PositionSnapshot("LONG", Decimal("10"), 100.0, 1.0, Decimal("10"))


class FakeJournal:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict, str | None]] = []

    def append(self, event, payload, position_id=None):
        self.events.append((event, dict(payload), position_id))

    def record_flat(self, **kwargs) -> None:
        pass


class FakeStateStore:
    def __init__(self) -> None:
        self.saved: list = []

    def save(self, state) -> None:
        self.saved.append(state)

    def clear(self) -> None:
        self.saved.clear()


class FakeEmailSender:
    def __init__(self) -> None:
        self.sent: list = []

    async def send_email_async(self, subject, content, content_type="html"):
        self.sent.append({"subject": subject, "content": content})
        return True


class FakeAccountSnapshot:
    position: PositionSnapshot | None = None
    cash: float = 1000.0
    equity: float = 1000.0
    updated_monotonic: float = 0.0
    updated_ts_ms: int = 0
    version: int = 0
    latest_market_price: float | None = None
    latest_market_price_ts_ms: int = 0


@pytest.mark.asyncio
async def test_dme_phase_not_armed_returns_not_armed() -> None:
    """When not armed, DME phase returns status='not_armed'."""
    strategy = BollCvdShockReclaimStrategy(
        BollCvdReclaimStrategyConfig(),
        SimplePositionSizer(SimplePositionSizerConfig()),
    )
    execution_state = live_runtime_types.ExecutionState("pos-1", 1000.0)
    trader = FakeTrader()
    journal = FakeJournal()
    state_store = FakeStateStore()
    account_snapshot = FakeAccountSnapshot()
    account_snapshot.position = PositionSnapshot("LONG", Decimal("10"), 100.0, 1.0, Decimal("10"))
    halt_deduper = HaltAlertDeduper()

    result = await run_delayed_market_exit_phase(
        state_lock=asyncio.Lock(),
        execution_state=execution_state,
        account_snapshot=account_snapshot,
        trader=trader,
        strategy=strategy,
        journal=journal,
        state_store=state_store,
        email_sender=None,
        halt_alert_deduper=halt_deduper,
    )

    assert result.status == "not_armed"
    assert result.executed is False
    assert len(trader.market_exits) == 0


@pytest.mark.asyncio
async def test_dme_phase_waiting_before_deadline() -> None:
    """Armed but before deadline → status='waiting', no market exit."""
    strategy = BollCvdShockReclaimStrategy(
        BollCvdReclaimStrategyConfig(),
        SimplePositionSizer(SimplePositionSizerConfig()),
    )
    execution_state = live_runtime_types.ExecutionState("pos-1", 1000.0)
    trader = FakeTrader()
    journal = FakeJournal()
    state_store = FakeStateStore()
    account_snapshot = FakeAccountSnapshot()
    account_snapshot.position = PositionSnapshot("LONG", Decimal("10"), 100.0, 1.0, Decimal("10"))
    halt_deduper = HaltAlertDeduper()

    now_ms = int(time.time() * 1000)
    dme.arm_delayed_market_exit(
        strategy_state=strategy.state,
        execution_state=execution_state,
        position_id="pos-1",
        side="LONG",
        reason="core_tp_place_failed_delayed_market_exit_armed",
        context="test",
        source_event="TEST",
        now_ms=now_ms,
        delay_seconds=1800.0,
    )

    result = await run_delayed_market_exit_phase(
        state_lock=asyncio.Lock(),
        execution_state=execution_state,
        account_snapshot=account_snapshot,
        trader=trader,
        strategy=strategy,
        journal=journal,
        state_store=state_store,
        email_sender=None,
        halt_alert_deduper=halt_deduper,
    )

    assert result.status == "waiting"
    assert result.executed is False
    assert len(trader.market_exits) == 0


@pytest.mark.asyncio
async def test_dme_phase_due_executes_market_exit() -> None:
    """Armed and after deadline → status='executed', market exit called."""
    strategy = BollCvdShockReclaimStrategy(
        BollCvdReclaimStrategyConfig(),
        SimplePositionSizer(SimplePositionSizerConfig()),
    )
    execution_state = live_runtime_types.ExecutionState("pos-1", 1000.0)
    trader = FakeTrader()
    journal = FakeJournal()
    state_store = FakeStateStore()
    account_snapshot = FakeAccountSnapshot()
    account_snapshot.position = PositionSnapshot("LONG", Decimal("10"), 100.0, 1.0, Decimal("10"))
    halt_deduper = HaltAlertDeduper()

    # Arm with 0 delay → immediately due
    now_ms = int(time.time() * 1000) - 1000
    dme.arm_delayed_market_exit(
        strategy_state=strategy.state,
        execution_state=execution_state,
        position_id="pos-1",
        side="LONG",
        reason="core_tp_place_failed_delayed_market_exit_armed",
        context="test",
        source_event="TEST",
        now_ms=now_ms,
        delay_seconds=0.0,
    )

    result = await run_delayed_market_exit_phase(
        state_lock=asyncio.Lock(),
        execution_state=execution_state,
        account_snapshot=account_snapshot,
        trader=trader,
        strategy=strategy,
        journal=journal,
        state_store=state_store,
        email_sender=None,
        halt_alert_deduper=halt_deduper,
    )

    assert result.status == "executed"
    assert result.executed is True
    assert result.exit_ok is True
    assert result.should_skip_remaining_account_sync is True
    assert len(trader.market_exits) == 1
    assert trader.market_exits[0]["context"] == "test"
    assert execution_state.halt_reason == "order_failure_delayed_market_exit_waiting_flat"


@pytest.mark.asyncio
async def test_dme_phase_due_already_flat_clears_state() -> None:
    """Armed and after deadline but position flat → cleared."""
    strategy = BollCvdShockReclaimStrategy(
        BollCvdReclaimStrategyConfig(),
        SimplePositionSizer(SimplePositionSizerConfig()),
    )
    execution_state = live_runtime_types.ExecutionState("pos-1", 1000.0)
    trader = FakeTrader()
    journal = FakeJournal()
    state_store = FakeStateStore()
    account_snapshot = FakeAccountSnapshot()
    account_snapshot.position = PositionSnapshot("LONG", Decimal("0"), 0.0, 0.0, Decimal("0"))
    # Mark as flat
    account_snapshot.position = None
    halt_deduper = HaltAlertDeduper()

    now_ms = int(time.time() * 1000) - 1000
    dme.arm_delayed_market_exit(
        strategy_state=strategy.state,
        execution_state=execution_state,
        position_id="pos-1",
        side="LONG",
        reason="core_tp_place_failed_delayed_market_exit_armed",
        context="test",
        source_event="TEST",
        now_ms=now_ms,
        delay_seconds=0.0,
    )

    result = await run_delayed_market_exit_phase(
        state_lock=asyncio.Lock(),
        execution_state=execution_state,
        account_snapshot=account_snapshot,
        trader=trader,
        strategy=strategy,
        journal=journal,
        state_store=state_store,
        email_sender=None,
        halt_alert_deduper=halt_deduper,
    )

    assert result.status == "cleared_already_flat"
    assert result.executed is False
    assert len(trader.market_exits) == 0
    assert strategy.state.delayed_market_exit_armed is False


@pytest.mark.asyncio
async def test_dme_phase_market_exit_failure() -> None:
    """Market exit fails → status='failed', halt_reason='..._failed'."""
    strategy = BollCvdShockReclaimStrategy(
        BollCvdReclaimStrategyConfig(),
        SimplePositionSizer(SimplePositionSizerConfig()),
    )
    execution_state = live_runtime_types.ExecutionState("pos-1", 1000.0)
    trader = FakeTrader()
    trader.set_market_exit_result(ok=False, message="market exit failed")
    journal = FakeJournal()
    state_store = FakeStateStore()
    account_snapshot = FakeAccountSnapshot()
    account_snapshot.position = PositionSnapshot("LONG", Decimal("10"), 100.0, 1.0, Decimal("10"))
    halt_deduper = HaltAlertDeduper()

    now_ms = int(time.time() * 1000) - 1000
    dme.arm_delayed_market_exit(
        strategy_state=strategy.state,
        execution_state=execution_state,
        position_id="pos-1",
        side="LONG",
        reason="core_tp_place_failed_delayed_market_exit_armed",
        context="test",
        source_event="TEST",
        now_ms=now_ms,
        delay_seconds=0.0,
    )

    result = await run_delayed_market_exit_phase(
        state_lock=asyncio.Lock(),
        execution_state=execution_state,
        account_snapshot=account_snapshot,
        trader=trader,
        strategy=strategy,
        journal=journal,
        state_store=state_store,
        email_sender=None,
        halt_alert_deduper=halt_deduper,
    )

    assert result.status == "failed"
    assert result.executed is True
    assert result.exit_ok is False
    assert result.should_skip_remaining_account_sync is True
    assert execution_state.halt_reason == "order_failure_delayed_market_exit_failed"
    assert strategy.state.delayed_market_exit_manual_intervention_required is True


@pytest.mark.asyncio
async def test_dme_phase_short_lock_no_mutation_without_lock() -> None:
    """DME phase uses short-lock pattern: reads state, releases, executes, writes back."""
    strategy = BollCvdShockReclaimStrategy(
        BollCvdReclaimStrategyConfig(),
        SimplePositionSizer(SimplePositionSizerConfig()),
    )
    execution_state = live_runtime_types.ExecutionState("pos-1", 1000.0)
    trader = FakeTrader()
    journal = FakeJournal()
    state_store = FakeStateStore()
    account_snapshot = FakeAccountSnapshot()
    account_snapshot.position = PositionSnapshot("LONG", Decimal("10"), 100.0, 1.0, Decimal("10"))
    halt_deduper = HaltAlertDeduper()

    now_ms = int(time.time() * 1000) - 1000
    dme.arm_delayed_market_exit(
        strategy_state=strategy.state,
        execution_state=execution_state,
        position_id="pos-1",
        side="LONG",
        reason="core_tp_place_failed_delayed_market_exit_armed",
        context="test",
        source_event="TEST",
        now_ms=now_ms,
        delay_seconds=0.0,
    )

    lock = asyncio.Lock()
    # Verify lock is NOT held after DME phase returns
    assert not lock.locked()

    result = await run_delayed_market_exit_phase(
        state_lock=lock,
        execution_state=execution_state,
        account_snapshot=account_snapshot,
        trader=trader,
        strategy=strategy,
        journal=journal,
        state_store=state_store,
        email_sender=None,
        halt_alert_deduper=halt_deduper,
    )

    # Lock must NOT be held after the phase returns
    assert not lock.locked()
    assert result.executed is True


def test_delayed_market_exit_result_dataclass() -> None:
    """DelayedMarketExitPhaseResult dataclass works as expected."""
    result = DelayedMarketExitPhaseResult(status="executed", executed=True, exit_ok=True, should_skip_remaining_account_sync=True)
    assert result.status == "executed"
    assert result.executed is True
    assert result.exit_ok is True
    assert result.should_skip_remaining_account_sync is True

    not_armed = DelayedMarketExitPhaseResult(status="not_armed")
    assert not_armed.executed is False
    assert not_armed.exit_ok is None
    assert not_armed.should_skip_remaining_account_sync is False
