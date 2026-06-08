"""Tests for delayed_market_exit.py — arm, clear, due, and payload helpers."""

from __future__ import annotations

import pytest

from src.live import delayed_market_exit as dme
from src.live.runtime_types import ExecutionState
from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState


def _make_strategy_state() -> StrategyPositionState:
    return StrategyPositionState()


def _make_execution_state() -> ExecutionState:
    return ExecutionState(
        current_position_id="test-pos-001",
        cash_before_position=1000.0,
    )


# ── arm_delayed_market_exit ────────────────────────────────────────────


def test_arm_sets_all_state_fields() -> None:
    state = _make_strategy_state()
    exec_state = _make_execution_state()
    now_ms = 1_700_000_000_000

    payload = dme.arm_delayed_market_exit(
        strategy_state=state,
        execution_state=exec_state,
        position_id="pos-1",
        side="LONG",
        reason="core_tp_place_failed_delayed_market_exit_armed",
        context="core_tp_place_failed",
        source_event="TEST_EVENT",
        now_ms=now_ms,
        delay_seconds=1800.0,
        error="test error",
    )

    assert state.delayed_market_exit_armed is True
    assert state.delayed_market_exit_reason == "core_tp_place_failed_delayed_market_exit_armed"
    assert state.delayed_market_exit_context == "core_tp_place_failed"
    assert state.delayed_market_exit_side == "LONG"
    assert state.delayed_market_exit_position_id == "pos-1"
    assert state.delayed_market_exit_source_event == "TEST_EVENT"
    assert state.delayed_market_exit_armed_ts_ms == now_ms
    assert state.delayed_market_exit_deadline_ts_ms == now_ms + 1_800_000
    assert state.delayed_market_exit_manual_intervention_required is True
    assert state.delayed_market_exit_last_error == "test error"

    assert exec_state.trading_halted is True
    assert exec_state.halt_reason == "core_tp_place_failed_delayed_market_exit_armed"

    assert payload["delayed_market_exit_armed"] is True
    assert payload["delay_seconds"] == 1800.0
    assert payload["countdown_seconds"] == 1800.0


def test_arm_default_delay_is_1800_seconds() -> None:
    state = _make_strategy_state()
    exec_state = _make_execution_state()
    now_ms = 1_700_000_000_000

    dme.arm_delayed_market_exit(
        strategy_state=state,
        execution_state=exec_state,
        position_id="pos-1",
        side="LONG",
        reason="test_reason",
        context="test_context",
        source_event="TEST_EVENT",
        now_ms=now_ms,
        # delay_seconds not provided → use default
    )

    assert state.delayed_market_exit_deadline_ts_ms == now_ms + 1_800_000  # 1800s default


def test_arm_custom_delay() -> None:
    state = _make_strategy_state()
    exec_state = _make_execution_state()
    now_ms = 1_700_000_000_000

    dme.arm_delayed_market_exit(
        strategy_state=state,
        execution_state=exec_state,
        position_id="pos-1",
        side="SHORT",
        reason="test_reason",
        context="test_context",
        source_event="TEST_EVENT",
        now_ms=now_ms,
        delay_seconds=600.0,  # 10 minutes
    )

    assert state.delayed_market_exit_deadline_ts_ms == now_ms + 600_000


# ── delayed_market_exit_due ────────────────────────────────────────────


def test_due_before_deadline_returns_false() -> None:
    state = _make_strategy_state()
    exec_state = _make_execution_state()
    now_ms = 1_700_000_000_000

    dme.arm_delayed_market_exit(
        strategy_state=state,
        execution_state=exec_state,
        position_id="pos-1",
        side="LONG",
        reason="test_reason",
        context="test",
        source_event="TEST",
        now_ms=now_ms,
        delay_seconds=1800.0,
    )

    # 10 minutes later — still before deadline
    assert dme.delayed_market_exit_due(state, now_ms + 600_000) is False


def test_due_after_deadline_returns_true() -> None:
    state = _make_strategy_state()
    exec_state = _make_execution_state()
    now_ms = 1_700_000_000_000

    dme.arm_delayed_market_exit(
        strategy_state=state,
        execution_state=exec_state,
        position_id="pos-1",
        side="LONG",
        reason="test_reason",
        context="test",
        source_event="TEST",
        now_ms=now_ms,
        delay_seconds=1800.0,
    )

    # 31 minutes later — after deadline
    assert dme.delayed_market_exit_due(state, now_ms + 1_860_000) is True


def test_due_exactly_at_deadline_returns_true() -> None:
    state = _make_strategy_state()
    exec_state = _make_execution_state()
    now_ms = 1_700_000_000_000

    dme.arm_delayed_market_exit(
        strategy_state=state,
        execution_state=exec_state,
        position_id="pos-1",
        side="LONG",
        reason="test_reason",
        context="test",
        source_event="TEST",
        now_ms=now_ms,
        delay_seconds=1800.0,
    )

    assert dme.delayed_market_exit_due(state, now_ms + 1_800_000) is True


def test_due_when_not_armed_returns_false() -> None:
    state = _make_strategy_state()
    assert dme.delayed_market_exit_due(state, 1_700_000_000_000) is False


def test_due_when_deadline_none_returns_false() -> None:
    state = _make_strategy_state()
    state.delayed_market_exit_armed = True
    state.delayed_market_exit_deadline_ts_ms = None
    assert dme.delayed_market_exit_due(state, 1_700_000_000_000) is False


# ── clear_delayed_market_exit ──────────────────────────────────────────


def test_clear_resets_all_fields() -> None:
    state = _make_strategy_state()
    exec_state = _make_execution_state()
    now_ms = 1_700_000_000_000

    dme.arm_delayed_market_exit(
        strategy_state=state,
        execution_state=exec_state,
        position_id="pos-1",
        side="LONG",
        reason="test_reason",
        context="test",
        source_event="TEST",
        now_ms=now_ms,
    )

    dme.clear_delayed_market_exit(state)

    assert state.delayed_market_exit_armed is False
    assert state.delayed_market_exit_reason is None
    assert state.delayed_market_exit_context is None
    assert state.delayed_market_exit_side is None
    assert state.delayed_market_exit_position_id is None
    assert state.delayed_market_exit_source_event is None
    assert state.delayed_market_exit_armed_ts_ms is None
    assert state.delayed_market_exit_deadline_ts_ms is None
    assert state.delayed_market_exit_manual_intervention_required is False
    assert state.delayed_market_exit_last_error is None


# ── delayed_market_exit_payload ────────────────────────────────────────


def test_payload_returns_current_state() -> None:
    state = _make_strategy_state()
    exec_state = _make_execution_state()
    now_ms = 1_700_000_000_000

    dme.arm_delayed_market_exit(
        strategy_state=state,
        execution_state=exec_state,
        position_id="pos-1",
        side="SHORT",
        reason="test_reason",
        context="test",
        source_event="TEST",
        now_ms=now_ms,
    )

    payload = dme.delayed_market_exit_payload(state)
    assert payload["delayed_market_exit_armed"] is True
    assert payload["delayed_market_exit_reason"] == "test_reason"
    assert payload["delayed_market_exit_side"] == "SHORT"
    assert payload["delayed_market_exit_armed_ts_ms"] == now_ms
    assert payload["delayed_market_exit_deadline_ts_ms"] == now_ms + 1_800_000


def test_payload_when_not_armed_returns_defaults() -> None:
    state = _make_strategy_state()
    payload = dme.delayed_market_exit_payload(state)
    assert payload["delayed_market_exit_armed"] is False
    assert payload["delayed_market_exit_reason"] is None
