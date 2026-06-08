"""Tests: sidecar entry_runtime sends halt alert email on TP failure."""

from __future__ import annotations

import os
from decimal import Decimal
from unittest import mock

import pytest

from src.execution.trader import PositionSnapshot
from src.live.runtime_types import ExecutionState
from src.position_management.sidecar.entry_runtime import attach_sidecar_after_combined_entry
from src.position_management.sidecar.planner import SidecarExecutionPlan
from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState, TradeIntent
from src.risk.simple_position_sizer import PositionSize
from src.live.alerts.halt_alerts import HaltAlertDeduper


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


class FakeStateStore:
    def __init__(self) -> None:
        self.saved: list = []

    def save(self, state) -> None:  # type: ignore[no-untyped-def]
        self.saved.append(state)


class FakeEmailSender:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_email_async(self, subject, content, content_type="html"):
        self.sent.append({"subject": subject, "content": content, "content_type": content_type})
        return True


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


@pytest.mark.asyncio
async def test_sidecar_tp_failure_sends_halt_alert() -> None:
    """When sidecar TP fails irrecoverably, halt alert email must be sent."""
    trader = FakeTrader()
    trader._tp_failures = [RuntimeError("Invalid order parameters")]

    journal = FakeJournal()
    store = FakeStateStore()
    email = FakeEmailSender()
    deduper = HaltAlertDeduper(dedup_interval_seconds=600)
    state = _sidecar_state()
    intent_ = _make_intent()
    plan = _sidecar_plan(intent_)
    exec_state = _execution_state()

    with mock.patch.dict(os.environ, {
        "SIDECAR_TP_PLACE_RETRY_COUNT": "1",
        "SIDECAR_TP_FAIL_MARKET_EXIT_RETRY_INTERVAL_SECONDS": "0.01",
    }):
        await attach_sidecar_after_combined_entry(
            trader=trader,
            strategy_state=state,
            execution_state=exec_state,
            intent=intent_,
            sidecar_plan=plan,
            journal=journal,
            state_store=store,
            trader_symbol="ETH-USDT-SWAP",
            email_sender=email,
            halt_alert_deduper=deduper,
        )

    assert exec_state.trading_halted is True
    assert len(email.sent) == 1
    sent = email.sent[0]
    assert "CRITICAL" in sent["subject"]
    assert "HALT" in sent["subject"]
    assert "sidecar_tp_place_failed" in sent["subject"]


@pytest.mark.asyncio
async def test_sidecar_tp_rate_limited_halt_alert() -> None:
    """Rate-limited sidecar TP with HALT_ONLY should send critical alert."""
    trader = FakeTrader()
    trader._tp_failures = [
        RuntimeError("50011: Rate limit reached"),
        RuntimeError("50011: Rate limit reached"),
        RuntimeError("50011: Rate limit reached"),
    ]

    journal = FakeJournal()
    store = FakeStateStore()
    email = FakeEmailSender()
    deduper = HaltAlertDeduper(dedup_interval_seconds=600)
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
        await attach_sidecar_after_combined_entry(
            trader=trader,
            strategy_state=state,
            execution_state=exec_state,
            intent=intent_,
            sidecar_plan=plan,
            journal=journal,
            state_store=store,
            trader_symbol="ETH-USDT-SWAP",
            email_sender=email,
            halt_alert_deduper=deduper,
        )

    assert exec_state.trading_halted is True
    assert exec_state.halt_reason == "sidecar_tp_place_rate_limited_delayed_market_exit_armed"
    assert len(email.sent) == 1
    assert "delayed_market_exit_armed" in str(email.sent[0]["subject"]).lower() or "sidecar_tp_place_rate_limited" in email.sent[0]["subject"]


@pytest.mark.asyncio
async def test_no_email_when_email_sender_is_none() -> None:
    """When email_sender is None, no alert is sent and no exception is raised."""
    trader = FakeTrader()
    trader._tp_failures = [RuntimeError("Invalid order parameters")]

    journal = FakeJournal()
    store = FakeStateStore()
    deduper = HaltAlertDeduper(dedup_interval_seconds=600)
    state = _sidecar_state()
    intent_ = _make_intent()
    plan = _sidecar_plan(intent_)
    exec_state = _execution_state()

    with mock.patch.dict(os.environ, {"SIDECAR_TP_PLACE_RETRY_COUNT": "1"}):
        ok = await attach_sidecar_after_combined_entry(
            trader=trader,
            strategy_state=state,
            execution_state=exec_state,
            intent=intent_,
            sidecar_plan=plan,
            journal=journal,
            state_store=store,
            trader_symbol="ETH-USDT-SWAP",
            email_sender=None,  # no email
            halt_alert_deduper=deduper,
        )

    assert ok is False
    # No crash — function completed without email
