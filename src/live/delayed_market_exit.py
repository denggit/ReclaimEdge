"""Unified delayed market exit module for ReclaimEdge live trading.

This module manages the **state** of a delayed market exit that is armed
after an order placement failure.  It does NOT:

- Place market orders.
- Read/write OKX.
- Run on the tick path.
- Scan journals.
- Make strategy decisions.

Its only job is to provide the persisted state fields and helper functions
so that the account sync phase (or equivalent low-frequency worker) can
decide whether the countdown has expired and execute the market exit.

Design rules (hard):
1. Arm after order failure → NO immediate market exit.
2. CRITICAL email sent immediately on arm.
3. Countdown (default 1800 s / 30 min) must expire before market exit.
4. State MUST be persisted (via strategy state fields stored by LiveStateStore).
5. Program restart must restore the countdown state.
6. If position is flat before countdown expires → clear delayed exit state.
7. Email failure must never affect the arm state.
8. DME execution is idempotent: at most ONE market exit per armed event.
"""

from __future__ import annotations

import os
from typing import Any


# ── Configurable defaults ──────────────────────────────────────────────

def _order_failure_market_exit_delay_seconds() -> float:
    """Default delay before automatic market exit after order placement failure.

    Hard rule: must be 1800 s (30 minutes) unless the user has explicitly set
    the ORDER_FAILURE_MARKET_EXIT_DELAY_SECONDS env var.
    """
    return float(os.getenv("ORDER_FAILURE_MARKET_EXIT_DELAY_SECONDS", "1800"))


# ── Unified delayed market exit state fields ───────────────────────────
#
# These fields are added to StrategyPositionState (see boll_cvd_reclaim_strategy.py).
# They are persisted via LiveStateStore and survive program restarts.
#
# Fields:
#   delayed_market_exit_armed: bool = False
#   delayed_market_exit_reason: str | None = None
#   delayed_market_exit_context: str | None = None
#   delayed_market_exit_side: str | None = None
#   delayed_market_exit_position_id: str | None = None
#   delayed_market_exit_source_event: str | None = None
#   delayed_market_exit_armed_ts_ms: int | None = None
#   delayed_market_exit_deadline_ts_ms: int | None = None
#   delayed_market_exit_manual_intervention_required: bool = False
#   delayed_market_exit_last_error: str | None = None
#   delayed_market_exit_status: str | None = None  # ARMED|WAITING_FLAT|FAILED|CLEARED
#   delayed_market_exit_executed_ts_ms: int | None = None
#   delayed_market_exit_exit_attempt_count: int = 0
#   delayed_market_exit_last_exit_message: str | None = None


# ── Status machine ─────────────────────────────────────────────────────
#
#   None ──(arm)──> ARMED ──(countdown)──> [DME phase checks]
#                      │
#   ARMED ──(due + exit_ok)──> WAITING_FLAT ──(flat settlement)──> CLEARED
#   ARMED ──(due + !exit_ok)──> FAILED (manual intervention)
#   ARMED ──(position already flat)──> CLEARED
#
#   WAITING_FLAT: no further market exit.  Wait for flat settlement.
#   FAILED:        no further auto market exit.  Manual intervention required.
#   CLEARED:       all DME state reset to defaults.


def arm_delayed_market_exit(
    *,
    strategy_state: Any,  # StrategyPositionState
    execution_state: Any,  # ExecutionState
    position_id: str | None,
    side: str,
    reason: str,
    context: str,
    source_event: str,
    now_ms: int,
    delay_seconds: float | None = None,
    error: str | None = None,
) -> dict:
    """Arm a delayed market exit after an order placement failure.

    Does NOT place any market order.  Sets the persisted state fields so that
    the account sync phase can execute the exit after the countdown expires.

    Returns a payload dict suitable for journaling and email construction.
    """
    if delay_seconds is None:
        delay_seconds = _order_failure_market_exit_delay_seconds()

    deadline_ts_ms = now_ms + int(delay_seconds * 1000)

    # ── Write to strategy state (persisted via LiveStateStore) ──────────
    strategy_state.delayed_market_exit_armed = True
    strategy_state.delayed_market_exit_reason = reason
    strategy_state.delayed_market_exit_context = context
    strategy_state.delayed_market_exit_side = side
    strategy_state.delayed_market_exit_position_id = position_id
    strategy_state.delayed_market_exit_source_event = source_event
    strategy_state.delayed_market_exit_armed_ts_ms = now_ms
    strategy_state.delayed_market_exit_deadline_ts_ms = deadline_ts_ms
    strategy_state.delayed_market_exit_manual_intervention_required = True
    strategy_state.delayed_market_exit_last_error = error
    # ── Idempotency fields ────────────────────────────────────────────
    strategy_state.delayed_market_exit_status = "ARMED"
    strategy_state.delayed_market_exit_executed_ts_ms = None
    strategy_state.delayed_market_exit_exit_attempt_count = 0
    strategy_state.delayed_market_exit_last_exit_message = None

    # ── Set execution state halt ────────────────────────────────────────
    execution_state.trading_halted = True
    execution_state.halt_reason = reason
    execution_state.halt_until_ts_ms = None

    payload: dict[str, Any] = {
        "delayed_market_exit_armed": True,
        "delayed_market_exit_reason": reason,
        "delayed_market_exit_context": context,
        "delayed_market_exit_side": side,
        "delayed_market_exit_position_id": position_id,
        "delayed_market_exit_source_event": source_event,
        "delayed_market_exit_armed_ts_ms": now_ms,
        "delayed_market_exit_deadline_ts_ms": deadline_ts_ms,
        "delay_seconds": delay_seconds,
        "countdown_seconds": delay_seconds,
        "manual_intervention_required": True,
        "error": error,
        "delayed_market_exit_status": "ARMED",
    }

    return payload


def clear_delayed_market_exit(strategy_state: Any) -> None:
    """Clear all delayed market exit state fields."""
    strategy_state.delayed_market_exit_armed = False
    strategy_state.delayed_market_exit_reason = None
    strategy_state.delayed_market_exit_context = None
    strategy_state.delayed_market_exit_side = None
    strategy_state.delayed_market_exit_position_id = None
    strategy_state.delayed_market_exit_source_event = None
    strategy_state.delayed_market_exit_armed_ts_ms = None
    strategy_state.delayed_market_exit_deadline_ts_ms = None
    strategy_state.delayed_market_exit_manual_intervention_required = False
    strategy_state.delayed_market_exit_last_error = None
    # ── Idempotency fields ────────────────────────────────────────────
    strategy_state.delayed_market_exit_status = None
    strategy_state.delayed_market_exit_executed_ts_ms = None
    strategy_state.delayed_market_exit_exit_attempt_count = 0
    strategy_state.delayed_market_exit_last_exit_message = None


def delayed_market_exit_due(strategy_state: Any, now_ms: int) -> bool:
    """Return True if the delayed market exit countdown has expired.

    Only returns True for status None or ARMED.  Once the status has
    transitioned to WAITING_FLAT, FAILED, or CLEARED, the exit is no
    longer due — preventing repeat executions.
    """
    if not getattr(strategy_state, "delayed_market_exit_armed", False):
        return False
    status = getattr(strategy_state, "delayed_market_exit_status", None)
    if status not in {None, "ARMED"}:
        return False
    deadline = getattr(strategy_state, "delayed_market_exit_deadline_ts_ms", None)
    if deadline is None:
        return False
    return now_ms >= deadline


def delayed_market_exit_due_from_snapshot(snapshot: dict, now_ms: int) -> bool:
    """Check due from a snapshot dict (no live state read).

    Returns True only when:
    - armed=True
    - status is None or "ARMED"
    - deadline_ts_ms is not None
    - now_ms >= deadline_ts_ms
    """
    armed = bool(snapshot.get("armed"))
    if not armed:
        return False
    status = snapshot.get("status")
    if status not in {None, "ARMED"}:
        return False
    deadline = snapshot.get("deadline_ts_ms")
    if deadline is None:
        return False
    return now_ms >= deadline


def mark_delayed_market_exit_waiting_flat(
    strategy_state: Any,
    *,
    executed_ts_ms: int,
    exit_message: str | None = None,
) -> None:
    """Transition status to WAITING_FLAT after a successful market exit.

    This prevents repeat market exit on subsequent account sync cycles.
    delayed_market_exit_armed stays True so the flat settlement phase
    can record the was-armed fact in the flat journal.
    """
    strategy_state.delayed_market_exit_status = "WAITING_FLAT"
    strategy_state.delayed_market_exit_executed_ts_ms = executed_ts_ms
    strategy_state.delayed_market_exit_exit_attempt_count = (
        int(getattr(strategy_state, "delayed_market_exit_exit_attempt_count", 0) or 0) + 1
    )
    strategy_state.delayed_market_exit_last_exit_message = exit_message


def mark_delayed_market_exit_failed(
    strategy_state: Any,
    *,
    executed_ts_ms: int,
    exit_message: str | None = None,
) -> None:
    """Transition status to FAILED after an unsuccessful market exit.

    This prevents further automatic retry.  Manual intervention is required.
    """
    strategy_state.delayed_market_exit_status = "FAILED"
    strategy_state.delayed_market_exit_executed_ts_ms = executed_ts_ms
    strategy_state.delayed_market_exit_exit_attempt_count = (
        int(getattr(strategy_state, "delayed_market_exit_exit_attempt_count", 0) or 0) + 1
    )
    strategy_state.delayed_market_exit_last_exit_message = exit_message
    strategy_state.delayed_market_exit_manual_intervention_required = True
    strategy_state.delayed_market_exit_last_error = exit_message


def delayed_market_exit_payload(strategy_state: Any) -> dict:
    """Return a dict of the current delayed market exit state for journaling/email."""
    return {
        "delayed_market_exit_armed": getattr(strategy_state, "delayed_market_exit_armed", False),
        "delayed_market_exit_reason": getattr(strategy_state, "delayed_market_exit_reason", None),
        "delayed_market_exit_context": getattr(strategy_state, "delayed_market_exit_context", None),
        "delayed_market_exit_side": getattr(strategy_state, "delayed_market_exit_side", None),
        "delayed_market_exit_position_id": getattr(strategy_state, "delayed_market_exit_position_id", None),
        "delayed_market_exit_source_event": getattr(strategy_state, "delayed_market_exit_source_event", None),
        "delayed_market_exit_armed_ts_ms": getattr(strategy_state, "delayed_market_exit_armed_ts_ms", None),
        "delayed_market_exit_deadline_ts_ms": getattr(strategy_state, "delayed_market_exit_deadline_ts_ms", None),
        "delayed_market_exit_manual_intervention_required": getattr(
            strategy_state, "delayed_market_exit_manual_intervention_required", False
        ),
        "delayed_market_exit_last_error": getattr(strategy_state, "delayed_market_exit_last_error", None),
        # ── Idempotency fields ────────────────────────────────────────
        "delayed_market_exit_status": getattr(strategy_state, "delayed_market_exit_status", None),
        "delayed_market_exit_executed_ts_ms": getattr(strategy_state, "delayed_market_exit_executed_ts_ms", None),
        "delayed_market_exit_exit_attempt_count": getattr(strategy_state, "delayed_market_exit_exit_attempt_count", 0),
        "delayed_market_exit_last_exit_message": getattr(strategy_state, "delayed_market_exit_last_exit_message", None),
    }
