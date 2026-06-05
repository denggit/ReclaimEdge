from __future__ import annotations

import html
from typing import Any

from src.live import time_utils as live_time_utils
from src.live.runtime_types import ExecutionState
from src.reporting.trade_journal import LiveTradeJournal
from src.risk.rolling_loss_guard import RollingLossGuard, RollingLossGuardDecision
from src.utils.email_sender import EmailSender
from src.utils.log import get_logger

logger = get_logger(__name__)


def rolling_loss_halt_reason(action: str) -> str | None:
    if action == "SOFT_HALT":
        return "rolling_loss_soft_halt"
    if action == "HARD_HALT":
        return "rolling_loss_hard_halt"
    return None


def rolling_loss_guard_subject(action: str, threshold_pct: float | None = None) -> str:
    pct = int(round((threshold_pct or 0.0) * 100))
    if action == "WARN":
        return f"Rolling loss guard warning: {pct}% realized loss reached"
    if action == "SOFT_HALT":
        return f"Rolling loss guard soft halt: {pct}% realized loss reached"
    if action == "HARD_HALT":
        return f"Rolling loss guard hard halt: {pct}% realized loss reached"
    if action == "RESUME":
        return "Rolling loss guard cooldown ended; trading resumed"
    return "Rolling loss guard update"


def build_rolling_loss_guard_email(action: str, payload: dict[str, Any]) -> tuple[str, str]:
    subject = rolling_loss_guard_subject(action, payload.get("threshold_pct"))
    halt_until = payload.get("halt_until_ts_ms")
    halt_until_text = live_time_utils.format_ts_ms(halt_until) if isinstance(halt_until, int) else "-"
    threshold = payload.get("threshold_pct")
    threshold_text = f"{float(threshold) * 100:.2f}%" if threshold is not None else "-"
    reference_flat_equity = float(payload.get("reference_flat_equity") or 0.0)
    flat_equity = float(payload.get("flat_equity") or 0.0)
    segment_retention = float(payload.get("segment_retention") or 1.0)
    cumulative_retention = float(payload.get("cumulative_retention") or 1.0)
    drawdown_pct = float(payload.get("drawdown_pct") or payload.get("loss_pct") or 0.0)
    content = f"""
<div style="font-family: Arial, Helvetica, sans-serif; line-height: 1.55; color: #222; max-width: 760px;">
  <h2>{html.escape(subject)}</h2>
  <p>This guard never force-closes an open position; this event was evaluated only after the account reached FLAT.</p>
  <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">mode</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{html.escape(str(payload.get("mode") or "flat_to_flat_drawdown"))}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">reference_flat_equity</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{reference_flat_equity:.4f}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">flat_equity</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{flat_equity:.4f}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">segment_retention</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{segment_retention * 100:.2f}%</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">cumulative_retention</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{cumulative_retention * 100:.2f}%</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">drawdown_pct</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{drawdown_pct * 100:.2f}%</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">loss_usdt</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{float(payload.get("loss_usdt") or 0.0):.4f}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">threshold</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{html.escape(threshold_text)}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">halt_until</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{html.escape(halt_until_text)}</td></tr>
  </table>
  <p><strong>Reason:</strong> {html.escape(str(payload.get("reason") or action))}</p>
</div>
""".strip()
    return subject, content


def rolling_loss_guard_payload(action: str, decision: RollingLossGuardDecision) -> dict[str, Any]:
    return {
        "action": action,
        "mode": "flat_to_flat_drawdown",
        "window_start_ts_ms": None,
        "window_end_ts_ms": None,
        "baseline_equity": decision.reference_flat_equity,
        "reference_flat_equity": decision.reference_flat_equity,
        "flat_equity": decision.flat_equity,
        "segment_retention": decision.segment_retention,
        "segment_return_pct": decision.segment_return_pct,
        "cumulative_retention": decision.cumulative_retention,
        "drawdown_pct": decision.drawdown_pct,
        "max_drawdown_pct": decision.max_drawdown_pct,
        "rolling_realized_pnl": decision.rolling_realized_pnl,
        "loss_usdt": decision.loss_usdt,
        "loss_pct": decision.loss_pct,
        "threshold_pct": decision.threshold_pct,
        "halt_hours": decision.halt_hours,
        "halt_until_ts_ms": decision.halt_until_ts_ms,
        "reason": decision.reason,
    }


def rolling_loss_guard_state_payload(action: str, guard: RollingLossGuard, reason: str) -> dict[str, Any]:
    state = guard.state
    if state is None:
        raise RuntimeError("RollingLossGuard state is not loaded")
    return {
        "action": action,
        "mode": "flat_to_flat_drawdown",
        "window_start_ts_ms": None,
        "window_end_ts_ms": None,
        "baseline_equity": state.reference_flat_equity,
        "reference_flat_equity": state.reference_flat_equity,
        "flat_equity": state.last_flat_equity,
        "segment_retention": state.last_segment_retention,
        "segment_return_pct": state.last_segment_return_pct,
        "cumulative_retention": state.cumulative_retention,
        "drawdown_pct": state.drawdown_pct,
        "max_drawdown_pct": state.max_drawdown_pct,
        "rolling_realized_pnl": 0.0,
        "loss_usdt": 0.0,
        "loss_pct": state.drawdown_pct,
        "threshold_pct": None,
        "halt_hours": None,
        "halt_until_ts_ms": state.halt_until_ts_ms,
        "reason": reason,
    }


async def record_and_notify_rolling_loss_guard(
    *,
    journal: LiveTradeJournal,
    email_sender: EmailSender | None,
    payload: dict[str, Any],
    email_enabled: bool,
) -> None:
    if (
        payload.get("critical_halt_preserved") is not None
        or payload.get("existing_halt_reason") is not None
        or payload.get("rolling_loss_halt_not_applied") is not None
    ) and hasattr(journal, "append"):
        journal.append("ROLLING_LOSS_GUARD", payload)
    else:
        journal.record_rolling_loss_guard(**payload)
    if not email_enabled or email_sender is None:
        return
    subject, content = build_rolling_loss_guard_email(str(payload["action"]), payload)
    try:
        ok = await email_sender.send_email_async(subject, content, content_type="html")
    except Exception:
        logger.exception("Failed to send rolling loss guard email | action=%s", payload["action"])
        return
    if not ok:
        logger.error("Failed to send rolling loss guard email | action=%s", payload["action"])


async def apply_rolling_loss_guard_startup_state(
    *,
    rolling_loss_guard: RollingLossGuard,
    execution_state: ExecutionState,
    has_position: bool,
    equity: float,
    now_ms: int,
    journal: LiveTradeJournal,
    email_sender: EmailSender | None,
) -> None:
    if rolling_loss_guard.state is None or not rolling_loss_guard.state.enabled:
        return
    if (
        rolling_loss_guard.state.halt_active
        and rolling_loss_guard.state.halt_until_ts_ms is not None
        and now_ms < rolling_loss_guard.state.halt_until_ts_ms
    ):
        execution_state.trading_halted = True
        execution_state.halt_reason = (
            "rolling_loss_hard_halt"
            if rolling_loss_guard.state.halt_level == "HARD"
            else "rolling_loss_soft_halt"
        )
        execution_state.halt_until_ts_ms = rolling_loss_guard.state.halt_until_ts_ms
        logger.warning(
            "ROLLING_LOSS_GUARD_RESTORED | halt_active=true halt_level=%s halt_until_ts_ms=%s",
            rolling_loss_guard.state.halt_level,
            rolling_loss_guard.state.halt_until_ts_ms,
        )
    elif rolling_loss_guard.state.halt_active and not has_position:
        rolling_loss_guard.mark_resumed(now_ms, equity)
        await record_and_notify_rolling_loss_guard(
            journal=journal,
            email_sender=email_sender,
            payload=rolling_loss_guard_state_payload(
                "RESUME",
                rolling_loss_guard,
                "startup_rolling_loss_cooldown_elapsed_and_account_flat",
            ),
            email_enabled=rolling_loss_guard.config.email_enabled,
        )
