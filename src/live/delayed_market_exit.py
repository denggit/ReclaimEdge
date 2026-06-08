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


def delayed_market_exit_due(strategy_state: Any, now_ms: int) -> bool:
    """Return True if the delayed market exit countdown has expired."""
    if not getattr(strategy_state, "delayed_market_exit_armed", False):
        return False
    deadline = getattr(strategy_state, "delayed_market_exit_deadline_ts_ms", None)
    if deadline is None:
        return False
    return now_ms >= deadline


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
    }
