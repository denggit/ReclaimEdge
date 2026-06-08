"""Delayed market exit execution phase for the account sync worker.

This phase is called after every account position sync.  It checks whether
a delayed market exit is armed and, if the countdown has expired, executes
the market exit with retries.

Design rules:
1. NOT on the tick path — runs in the account sync worker (low frequency).
2. State is persisted in StrategyPositionState (survives restart).
3. Background tasks are NOT the sole mechanism — this phase is the authority.
4. Email failure does not affect execution.
5. Short-lock pattern: read under lock, release, execute OKX/email, lock to write.
"""

from __future__ import annotations

import asyncio
import copy
import html
import os
import time
from dataclasses import dataclass
from typing import Any

from src.execution.trader import Trader
from src.live import delayed_market_exit as dme
from src.live import runtime_types as live_runtime_types
from src.live.alerts.halt_alerts import (
    HaltAlertDeduper,
    HaltAlertPayload,
    send_halt_alert_once,
)
from src.live.halt_modes import resolve_halt_mode
from src.reporting.live_state_store import LiveStateStore
from src.reporting.trade_journal import LiveTradeJournal
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy
from src.utils.email_sender import EmailSender
from src.utils.log import get_logger

logger = get_logger(__name__)

# Low-frequency logging: only log "still waiting" every N seconds.
_DME_WAITING_LOG_INTERVAL_SECONDS = 300  # 5 minutes
_last_dme_waiting_log_ts: float = 0.0


@dataclass(frozen=True)
class DelayedMarketExitPhaseResult:
    """Result of a delayed market exit phase run."""
    status: str  # "not_armed" | "waiting" | "cleared_already_flat" | "executed" | "failed"
    executed: bool = False
    exit_ok: bool | None = None
    should_skip_remaining_account_sync: bool = False


async def _send_delayed_market_exit_alert(
    *,
    email_sender: Any,
    halt_alert_deduper: HaltAlertDeduper | None,
    symbol: str,
    position_id: str | None,
    halt_reason: str,
    side: str | None = None,
    manual_intervention_required: bool = True,
    message: str = "",
    extra: dict | None = None,
) -> None:
    """Send a rate-limited alert email from the delayed market exit phase."""
    if email_sender is None or halt_alert_deduper is None:
        return
    try:
        await send_halt_alert_once(
            email_sender=email_sender,
            deduper=halt_alert_deduper,
            payload=HaltAlertPayload(
                symbol=symbol,
                position_id=position_id,
                halt_reason=halt_reason,
                halt_mode=resolve_halt_mode(halt_reason),
                side=side,
                layer=None,
                has_position=True,
                sidecar_dirty=False,
                manual_intervention_required=manual_intervention_required,
                message=message,
                extra=extra or {},
            ),
        )
    except Exception:
        logger.exception("DME_HALT_ALERT_EXCEPTION | halt_reason=%s", halt_reason)


async def run_delayed_market_exit_phase(
    *,
    state_lock: asyncio.Lock,
    execution_state: live_runtime_types.ExecutionState,
    account_snapshot: live_runtime_types.AccountSnapshot,
    trader: Trader,
    strategy: BollCvdShockReclaimStrategy,
    journal: LiveTradeJournal,
    state_store: LiveStateStore,
    email_sender: EmailSender | None,
    halt_alert_deduper: HaltAlertDeduper | None,
) -> DelayedMarketExitPhaseResult:
    """Check and execute delayed market exit if countdown has expired.

    Uses a short-lock pattern:
    1. Short lock to read state snapshot
    2. Release lock, execute OKX/email
    3. Short lock to write back results

    This is called from the account sync worker after the position snapshot
    has been refreshed by pre_core_position phase.
    """
    global _last_dme_waiting_log_ts

    # ── Phase 1: Short lock to read state snapshot ──────────────────────
    async with state_lock:
        armed = getattr(strategy.state, "delayed_market_exit_armed", False)
        if not armed:
            return DelayedMarketExitPhaseResult(status="not_armed")

        # Snapshot all delayed exit fields under lock
        dme_snapshot: dict[str, Any] = {
            "armed": armed,
            "reason": strategy.state.delayed_market_exit_reason,
            "context": strategy.state.delayed_market_exit_context,
            "side": strategy.state.delayed_market_exit_side,
            "position_id": strategy.state.delayed_market_exit_position_id,
            "source_event": strategy.state.delayed_market_exit_source_event,
            "armed_ts_ms": strategy.state.delayed_market_exit_armed_ts_ms,
            "deadline_ts_ms": strategy.state.delayed_market_exit_deadline_ts_ms,
            "manual_intervention_required": strategy.state.delayed_market_exit_manual_intervention_required,
            "last_error": strategy.state.delayed_market_exit_last_error,
        }
        current_halt_reason = execution_state.halt_reason
        current_position_id = execution_state.current_position_id
        cash_before_position = execution_state.cash_before_position

    now_ms = int(time.time() * 1000)

    # ── Phase 2: Check due (no lock needed) ─────────────────────────────
    if not dme.delayed_market_exit_due(strategy.state, now_ms):
        now_mono = time.monotonic()
        if now_mono - _last_dme_waiting_log_ts >= _DME_WAITING_LOG_INTERVAL_SECONDS:
            _last_dme_waiting_log_ts = now_mono
            deadline = dme_snapshot["deadline_ts_ms"]
            remaining_s = max(0, (deadline - now_ms) / 1000) if deadline else 0
            logger.info(
                "DELAYED_MARKET_EXIT_WAITING | position_id=%s side=%s reason=%s deadline_ts_ms=%s remaining_seconds=%.0f armed_ts_ms=%s",
                dme_snapshot["position_id"],
                dme_snapshot["side"],
                dme_snapshot["reason"],
                deadline,
                remaining_s,
                dme_snapshot["armed_ts_ms"],
            )
        return DelayedMarketExitPhaseResult(status="waiting")

    # ── Phase 3: Countdown expired — read position (no lock) ────────────
    side = dme_snapshot["side"]
    position_id = dme_snapshot["position_id"]
    reason = dme_snapshot["reason"]
    context = dme_snapshot["context"]
    source_event = dme_snapshot["source_event"]

    position = account_snapshot.position

    # ── Phase 3a: Already flat or wrong side ────────────────────────────
    if position is None or not position.has_position or (side and position.side != side):
        logger.warning(
            "DELAYED_MARKET_EXIT_SKIPPED_ALREADY_FLAT | position_id=%s side=%s reason=%s has_position=%s position_side=%s",
            position_id,
            side,
            reason,
            position.has_position if position else False,
            position.side if position else None,
        )

        async with state_lock:
            dme_payload = dme.delayed_market_exit_payload(strategy.state)
            dme.clear_delayed_market_exit(strategy.state)
            # Clear halt if delayed-exit related
            clearable_suffixes = (
                "_delayed_market_exit_armed",
                "_waiting_flat",
                "_delayed_market_exit_failed",
            )
            if execution_state.halt_reason and any(
                execution_state.halt_reason.endswith(p) for p in clearable_suffixes
            ):
                execution_state.trading_halted = False
                execution_state.halt_reason = None
                execution_state.halt_until_ts_ms = None
            current_position_id = execution_state.current_position_id
            cash_before_position = execution_state.cash_before_position
            state_for_save = copy.deepcopy(strategy.state)

        if hasattr(journal, "append"):
            journal.append(
                "DELAYED_MARKET_EXIT_SKIPPED_ALREADY_FLAT",
                {
                    **dme_payload,
                    "delayed_market_exit_was_armed": True,
                    "delayed_market_exit_cleared": True,
                    "reason_skipped": "already_flat_or_wrong_side",
                },
                position_id=position_id,
            )

        state_store.save(
            LiveStateStore.from_strategy_state(
                position_id=current_position_id,
                symbol=trader.symbol,
                strategy_state=state_for_save,
                cash_before_position=cash_before_position,
            )
        )

        await _send_delayed_market_exit_alert(
            email_sender=email_sender,
            halt_alert_deduper=halt_alert_deduper,
            symbol=trader.symbol,
            position_id=position_id,
            halt_reason="delayed_market_exit_waiting_flat",
            side=side,
            manual_intervention_required=False,
            message="Delayed market exit skipped: position already flat or wrong side.",
            extra={"delayed_market_exit_was_armed": True, "already_flat": True},
        )
        return DelayedMarketExitPhaseResult(status="cleared_already_flat")

    # ── Phase 4: Position exists — execute market exit (NO lock) ────────
    retry_count = int(os.getenv("ORDER_FAILURE_MARKET_EXIT_RETRY_COUNT", "3"))
    retry_interval = float(os.getenv("ORDER_FAILURE_MARKET_EXIT_RETRY_INTERVAL_SECONDS", "0.5"))

    logger.warning(
        "DELAYED_MARKET_EXIT_EXECUTING | position_id=%s side=%s reason=%s context=%s retry_count=%s",
        position_id,
        side,
        reason,
        context,
        retry_count,
    )

    exit_ok, exit_message = await trader.market_exit_remaining_position_with_retries(
        side,
        retry_count=retry_count,
        context=context or "delayed_market_exit",
        retry_interval_seconds=retry_interval,
    )

    # ── Phase 5: Short lock to write results ────────────────────────────
    if exit_ok:
        async with state_lock:
            execution_state.trading_halted = True
            execution_state.halt_reason = "order_failure_delayed_market_exit_waiting_flat"
            current_position_id = execution_state.current_position_id
            cash_before_position = execution_state.cash_before_position
            state_for_save = copy.deepcopy(strategy.state)

        if hasattr(journal, "append"):
            journal.append(
                "DELAYED_MARKET_EXIT_EXECUTED",
                {
                    **dme.delayed_market_exit_payload(strategy.state),
                    "exit_ok": True,
                    "exit_message": exit_message,
                    "trading_halted": True,
                    "halt_reason": "order_failure_delayed_market_exit_waiting_flat",
                },
                position_id=position_id,
            )

        state_store.save(
            LiveStateStore.from_strategy_state(
                position_id=current_position_id,
                symbol=trader.symbol,
                strategy_state=state_for_save,
                cash_before_position=cash_before_position,
            )
        )

        logger.warning(
            "DELAYED_MARKET_EXIT_EXECUTED | position_id=%s side=%s reason=%s exit_message=%s",
            position_id,
            side,
            reason,
            exit_message,
        )

        await _send_delayed_market_exit_alert(
            email_sender=email_sender,
            halt_alert_deduper=halt_alert_deduper,
            symbol=trader.symbol,
            position_id=position_id,
            halt_reason="order_failure_delayed_market_exit_waiting_flat",
            side=side,
            manual_intervention_required=False,
            message=f"Delayed market exit executed successfully. exit_message={exit_message}.",
            extra={"exit_ok": True, "exit_message": exit_message},
        )
        return DelayedMarketExitPhaseResult(
            status="executed",
            executed=True,
            exit_ok=True,
            should_skip_remaining_account_sync=True,
        )
    else:
        async with state_lock:
            execution_state.trading_halted = True
            execution_state.halt_reason = "order_failure_delayed_market_exit_failed"
            strategy.state.delayed_market_exit_manual_intervention_required = True
            strategy.state.delayed_market_exit_last_error = exit_message
            if reason and "sidecar" in reason:
                strategy.state.sidecar_dirty = True
            current_position_id = execution_state.current_position_id
            cash_before_position = execution_state.cash_before_position
            state_for_save = copy.deepcopy(strategy.state)

        if hasattr(journal, "append"):
            journal.append(
                "DELAYED_MARKET_EXIT_FAILED",
                {
                    **dme.delayed_market_exit_payload(strategy.state),
                    "exit_ok": False,
                    "exit_message": exit_message,
                    "trading_halted": True,
                    "halt_reason": "order_failure_delayed_market_exit_failed",
                    "manual_intervention_required": True,
                },
                position_id=position_id,
            )

        state_store.save(
            LiveStateStore.from_strategy_state(
                position_id=current_position_id,
                symbol=trader.symbol,
                strategy_state=state_for_save,
                cash_before_position=cash_before_position,
            )
        )

        logger.error(
            "DELAYED_MARKET_EXIT_FAILED | position_id=%s side=%s reason=%s exit_message=%s manual_intervention_required=true",
            position_id,
            side,
            reason,
            exit_message,
        )

        fail_subject = (
            f"[ReclaimEdge][CRITICAL] ORDER FAILURE - DELAYED MARKET EXIT FAILED "
            f"{trader.symbol} {reason}"
        )
        fail_content = (
            "<div style='font-family:Arial,Helvetica,sans-serif;line-height:1.55;'>"
            "<h2>Delayed Market Exit FAILED — Manual Intervention Required</h2>"
            f"<p><b>symbol:</b> {html.escape(trader.symbol)}</p>"
            f"<p><b>position_id:</b> {html.escape(str(position_id))}</p>"
            f"<p><b>side:</b> {html.escape(str(side))}</p>"
            f"<p><b>halt_reason:</b> {html.escape(str(reason))}</p>"
            f"<p><b>context:</b> {html.escape(str(context))}</p>"
            f"<p><b>exit_message:</b> {html.escape(exit_message)}</p>"
            "<p><b>action:</b> MANUAL INTERVENTION REQUIRED — market exit FAILED.</p>"
            "</div>"
        )
        if email_sender is not None:
            try:
                ok = await email_sender.send_email_async(fail_subject, fail_content, content_type="html")
                if not ok:
                    logger.error("Failed to send delayed market exit failed email")
            except Exception:
                logger.exception("DME_FAILED_EMAIL_EXCEPTION")

        return DelayedMarketExitPhaseResult(
            status="failed",
            executed=True,
            exit_ok=False,
            should_skip_remaining_account_sync=True,
        )
