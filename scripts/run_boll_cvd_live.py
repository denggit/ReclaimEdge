from __future__ import annotations

import asyncio
import copy
import datetime as dt
import html
import logging
import os
import sys
import time
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from src.execution.trader import PositionSnapshot, Trader  # noqa: E402
from src.indicators.cvd_tracker import CvdTracker, CvdTrackerConfig  # noqa: E402
from src.live import config_helpers as live_config_helpers  # noqa: E402
from src.live import runtime_types as live_runtime_types  # noqa: E402
from src.live import time_utils as live_time_utils  # noqa: E402
from src.monitors.boll_band_breakout_monitor import (  # noqa: E402
    BollBandBreakoutMonitor,
    BollBandBreakoutMonitorConfig,
    MarketTickEvent,
)
from src.position_management.cost_basis import calculate_remaining_breakeven_price  # noqa: E402
from src.position_management.sidecar.model import (  # noqa: E402
    SidecarLegStatus,
    sidecar_open_contracts,
    sidecar_open_qty,
    trim_sidecar_legs_for_state,
)
from src.position_management.sidecar.planner import (  # noqa: E402
    SidecarExecutionPlan,
    build_combined_entry_intent,
    sidecar_client_order_id as build_sidecar_client_order_id,
)
from src.position_management.sidecar.reconciler import (  # noqa: E402
    build_core_position_view,
    is_sidecar_dirty_missing_tp_order,
    mark_sidecar_leg_open_unprotected,
    mark_sidecar_leg_force_closed,
    mark_sidecar_leg_tp_filled,
    mark_sidecar_leg_unknown_halted,
    sidecar_leg_from_fill,
)
from src.reporting.daily_trade_reporter import DailyTradeReporter, ReportRuntimeContext  # noqa: E402
from src.reporting.journal_compactor import compact_after_weekly_summary  # noqa: E402
from src.reporting.live_state_store import LiveStateStore  # noqa: E402
from src.reporting.trade_journal import LiveTradeJournal  # noqa: E402
from src.risk.simple_position_sizer import (  # noqa: E402
    SimplePositionSizer,
    SimplePositionSizerConfig,
)
from src.risk.rolling_loss_guard import (  # noqa: E402
    ROLLING_LOSS_HALT_REASONS,
    RollingLossGuard,
    RollingLossGuardDecision,
)
from src.strategies.boll_cvd_reclaim_strategy import (  # noqa: E402
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
    StrategyPositionState,
    TradeIntent,
)
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy  # noqa: E402
from src.utils.email_sender import EmailSender  # noqa: E402
from src.utils.log import get_logger  # noqa: E402

logger = get_logger(__name__)

SPLIT_TP_PLANS = {"SPLIT_PARTIAL_FINAL", "SPLIT_50_50"}
POSITION_MANAGEMENT_INTENTS = {"UPDATE_TP", "NEAR_TP_REDUCE", "MARKET_EXIT_RUNNER"}
THREE_STAGE_RESTART_DIRTY_HALT_REASON = "three_stage_post_tp1_sl_cancel_failed_on_tp2_restart"
THREE_STAGE_RUNTIME_DIRTY_HALT_REASON = "three_stage_post_tp1_sl_dirty_state_blocked"
THREE_STAGE_CANCEL_PENDING_HALT_REASON = "three_stage_post_tp1_sl_cancel_pending_on_tp2"
DEFAULT_NET_REMAINING_FEE_BUFFER_PCT = 0.001


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
    execution_state: live_runtime_types.ExecutionState,
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


def build_report_context(
    *,
    account_snapshot: live_runtime_types.AccountSnapshot,
    execution_state: live_runtime_types.ExecutionState,
    period_start_cash: float | None = None,
    period_start_equity: float | None = None,
) -> ReportRuntimeContext:
    position = account_snapshot.position
    return ReportRuntimeContext(
        current_position_id=execution_state.current_position_id,
        current_has_position=bool(position and position.has_position),
        current_cash=account_snapshot.cash,
        current_equity=account_snapshot.equity,
        period_start_cash=period_start_cash,
        period_start_equity=period_start_equity,
    )


def build_live_failure_email(intent: TradeIntent, error: Exception, rolled_back: bool, halted: bool) -> tuple[str, str]:
    subject = f"LIVE order failed | ETH-USDT-SWAP | {intent.intent_type} | layer {intent.layer_index}"
    event_time = live_time_utils.format_ts_ms(intent.ts_ms)
    state_text = "Strategy state has been rolled back." if rolled_back else "Entry may be live. Strategy state was NOT rolled back."
    halt_text = "Trading has been halted. Please check OKX manually." if halted else "Trading is not halted."
    content = f"""
<div style="font-family: Arial, Helvetica, sans-serif; line-height: 1.55; color: #222; max-width: 760px;">
  <h2>LIVE order failed</h2>
  <p><strong>{html.escape(state_text)}</strong></p>
  <p><strong>{html.escape(halt_text)}</strong></p>
  <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">intent_type</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{html.escape(intent.intent_type)}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">side</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{html.escape(intent.side)}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">layer</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.layer_index}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">price</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.price:.4f}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">tp_price</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.tp_price:.4f}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">partial_tp_price</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{getattr(intent, 'partial_tp_price', None) or '-'}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">tp_plan</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{html.escape(getattr(intent, 'tp_plan', 'SINGLE'))}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">tp_mode</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{html.escape(intent.tp_mode)}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">avg_entry</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.avg_entry_price:.4f}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">breakeven</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.breakeven_price:.4f}</td></tr>
  </table>
  <p><strong>Reason:</strong> {html.escape(intent.reason)}</p>
  <p><strong>Error:</strong> {html.escape(str(error))}</p>
  <p><strong>Event time:</strong> {html.escape(event_time)}</p>
</div>
""".strip()
    return subject, content


def restore_strategy_from_position(strategy: BollCvdReclaimStrategy, position: PositionSnapshot, now_ms: int | None = None) -> None:
    if not position.has_position or position.side is None or position.avg_entry_price <= 0:
        return
    now_ms = int(now_ms or live_time_utils.utc_ms())
    strategy.state = StrategyPositionState(
        side=position.side,
        layers=1,
        last_entry_price=position.avg_entry_price,
        tp_price=None,
        last_order_ts_ms=now_ms,
        first_entry_ts_ms=now_ms,
        last_tp_update_ts_ms=0,
        total_entry_qty=position.eth_qty,
        total_entry_notional=position.avg_entry_price * position.eth_qty,
        avg_entry_price=position.avg_entry_price,
        sidecar_enabled_for_position=False,
    )
    logger.warning(
        "Recovered existing position into strategy state | side=%s contracts=%s eth_qty=%.6f avg_entry=%.4f first_entry_ts_ms=%s last_order_ts_ms=%s",
        position.side,
        position.contracts,
        position.eth_qty,
        position.avg_entry_price,
        now_ms,
        now_ms,
    )


def restore_strategy_from_saved_state(strategy: BollCvdReclaimStrategy, saved_state) -> None:  # type: ignore[no-untyped-def]
    tp_plan = getattr(saved_state, "tp_plan", "SINGLE")
    if tp_plan == "SPLIT_50_50":
        tp_plan = "SPLIT_PARTIAL_FINAL"
    strategy.state = StrategyPositionState(
        side=saved_state.side,
        layers=saved_state.layers,
        last_entry_price=saved_state.last_entry_price,
        tp_price=saved_state.tp_price,
        tp_order_id=getattr(saved_state, "tp_order_id", None),
        tp_order_ids=list(getattr(saved_state, "tp_order_ids", []) or []),
        partial_tp_price=getattr(saved_state, "partial_tp_price", None),
        partial_tp_ratio=getattr(saved_state, "partial_tp_ratio", 0.0),
        tp_plan=tp_plan,
        partial_tp_consumed=getattr(saved_state, "partial_tp_consumed", False),
        last_order_ts_ms=saved_state.last_order_ts_ms,
        first_entry_ts_ms=getattr(saved_state, "first_entry_ts_ms", 0),
        add_freeze_until_ts_ms=getattr(saved_state, "add_freeze_until_ts_ms", 0),
        add_freeze_penalty_count=getattr(saved_state, "add_freeze_penalty_count", 0),
        last_tp_update_ts_ms=saved_state.last_tp_update_ts_ms,
        last_tp_update_candle_ts_ms=saved_state.last_tp_update_candle_ts_ms,
        total_entry_qty=saved_state.total_entry_qty,
        total_entry_notional=saved_state.total_entry_notional,
        avg_entry_price=saved_state.avg_entry_price,
        breakeven_price=saved_state.breakeven_price,
        position_cost_entry_notional=getattr(saved_state, "position_cost_entry_notional", 0.0),
        position_cost_exit_notional=getattr(saved_state, "position_cost_exit_notional", 0.0),
        position_cost_remaining_qty=getattr(saved_state, "position_cost_remaining_qty", 0.0),
        net_remaining_breakeven_price=getattr(saved_state, "net_remaining_breakeven_price", 0.0),
        tp_mode=saved_state.tp_mode,
        near_tp_armed=getattr(saved_state, "near_tp_armed", False),
        near_tp_reduce_pending=getattr(saved_state, "near_tp_reduce_pending", False),
        near_tp_protected=getattr(saved_state, "near_tp_protected", False),
        near_tp_best_price=getattr(saved_state, "near_tp_best_price", None),
        near_tp_armed_ts_ms=getattr(saved_state, "near_tp_armed_ts_ms", 0),
        near_tp_pending_ts_ms=getattr(saved_state, "near_tp_pending_ts_ms", 0),
        near_tp_trigger_ts_ms=getattr(saved_state, "near_tp_trigger_ts_ms", 0),
        near_tp_protective_sl_price=getattr(saved_state, "near_tp_protective_sl_price", None),
        near_tp_protective_sl_order_id=getattr(saved_state, "near_tp_protective_sl_order_id", None),
        near_tp_add_disabled=getattr(saved_state, "near_tp_add_disabled", False),
        middle_runner_enabled_for_position=getattr(saved_state, "middle_runner_enabled_for_position", False),
        middle_runner_pending=getattr(saved_state, "middle_runner_pending", False),
        middle_runner_active=getattr(saved_state, "middle_runner_active", False),
        middle_runner_first_close_ratio=getattr(saved_state, "middle_runner_first_close_ratio", 0.0),
        middle_runner_keep_ratio=getattr(saved_state, "middle_runner_keep_ratio", 0.0),
        middle_runner_first_tp_price=getattr(saved_state, "middle_runner_first_tp_price", None),
        middle_runner_final_tp_price=getattr(saved_state, "middle_runner_final_tp_price", None),
        middle_runner_protective_sl_price=getattr(saved_state, "middle_runner_protective_sl_price", None),
        middle_runner_protective_sl_order_id=getattr(saved_state, "middle_runner_protective_sl_order_id", None),
        middle_runner_extension_triggered=getattr(saved_state, "middle_runner_extension_triggered", False),
        middle_runner_add_disabled=getattr(saved_state, "middle_runner_add_disabled", False),
        middle_runner_size_mismatch_protected=getattr(saved_state, "middle_runner_size_mismatch_protected", False),
        middle_runner_size_mismatch_warning_ts_ms=getattr(saved_state, "middle_runner_size_mismatch_warning_ts_ms", 0),
        middle_runner_sl_diag_last_signature=getattr(saved_state, "middle_runner_sl_diag_last_signature", None),
        three_stage_runner_enabled_for_position=getattr(saved_state, "three_stage_runner_enabled_for_position", False),
        three_stage_tp1_price=getattr(saved_state, "three_stage_tp1_price", None),
        three_stage_tp2_price=getattr(saved_state, "three_stage_tp2_price", None),
        three_stage_runner_initial_tp_price=getattr(saved_state, "three_stage_runner_initial_tp_price", None),
        three_stage_tp1_ratio=getattr(saved_state, "three_stage_tp1_ratio", 0.0),
        three_stage_tp2_ratio=getattr(saved_state, "three_stage_tp2_ratio", 0.0),
        three_stage_runner_ratio=getattr(saved_state, "three_stage_runner_ratio", 0.0),
        three_stage_tp1_consumed=getattr(saved_state, "three_stage_tp1_consumed", False),
        three_stage_tp2_consumed=getattr(saved_state, "three_stage_tp2_consumed", False),
        three_stage_post_tp1_protective_sl_price=getattr(saved_state, "three_stage_post_tp1_protective_sl_price", None),
        three_stage_post_tp1_protective_sl_order_id=getattr(saved_state, "three_stage_post_tp1_protective_sl_order_id", None),
        three_stage_post_tp1_sl_extension_triggered=getattr(saved_state, "three_stage_post_tp1_sl_extension_triggered", False),
        three_stage_post_tp1_protected=getattr(saved_state, "three_stage_post_tp1_protected", False),
        three_stage_post_tp1_sl_diag_last_signature=getattr(saved_state, "three_stage_post_tp1_sl_diag_last_signature", None),
        three_stage_pre_tp1_degrade_stage=getattr(saved_state, "three_stage_pre_tp1_degrade_stage", None),
        three_stage_pre_tp1_degraded_ts_ms=getattr(saved_state, "three_stage_pre_tp1_degraded_ts_ms", 0),
        trend_runner_active=getattr(saved_state, "trend_runner_active", False),
        trend_runner_trend_start_ts_ms=getattr(saved_state, "trend_runner_trend_start_ts_ms", 0),
        trend_runner_adjust_count=getattr(saved_state, "trend_runner_adjust_count", 0),
        trend_runner_last_update_candle_ts_ms=getattr(saved_state, "trend_runner_last_update_candle_ts_ms", 0),
        trend_runner_tp_price=getattr(saved_state, "trend_runner_tp_price", None),
        trend_runner_sl_price=getattr(saved_state, "trend_runner_sl_price", None),
        trend_runner_tp_order_id=getattr(saved_state, "trend_runner_tp_order_id", None),
        trend_runner_sl_order_id=getattr(saved_state, "trend_runner_sl_order_id", None),
        trend_runner_exit_reason=getattr(saved_state, "trend_runner_exit_reason", None),
        trend_runner_reverse_candidate=getattr(saved_state, "trend_runner_reverse_candidate", False),
        trend_runner_reverse_start_ts_ms=getattr(saved_state, "trend_runner_reverse_start_ts_ms", 0),
        trend_runner_reverse_start_price=getattr(saved_state, "trend_runner_reverse_start_price", None),
        trend_runner_reverse_extreme_price=getattr(saved_state, "trend_runner_reverse_extreme_price", None),
        trend_runner_reverse_fast_cvd_start=getattr(saved_state, "trend_runner_reverse_fast_cvd_start", 0.0),
        trend_runner_reverse_samples=getattr(saved_state, "trend_runner_reverse_samples", []) or [],
        sidecar_enabled_for_position=getattr(saved_state, "sidecar_enabled_for_position", False),
        sidecar_margin_pct=getattr(saved_state, "sidecar_margin_pct", 0.0),
        sidecar_tp_pct=getattr(saved_state, "sidecar_tp_pct", 0.0),
        sidecar_total_qty=getattr(saved_state, "sidecar_total_qty", 0.0),
        sidecar_open_qty=getattr(saved_state, "sidecar_open_qty", 0.0),
        sidecar_total_notional=getattr(saved_state, "sidecar_total_notional", 0.0),
        sidecar_realized_qty=getattr(saved_state, "sidecar_realized_qty", 0.0),
        sidecar_legs=list(getattr(saved_state, "sidecar_legs", []) or []),
        sidecar_dirty=getattr(saved_state, "sidecar_dirty", False),
        sidecar_halt_reason=getattr(saved_state, "sidecar_halt_reason", None),
        near_tp_sidecar_skip_logged=getattr(saved_state, "near_tp_sidecar_skip_logged", False),
        last_add_skip_log_reason=getattr(saved_state, "last_add_skip_log_reason", None),
        last_add_skip_log_ts_ms=getattr(saved_state, "last_add_skip_log_ts_ms", 0),
        core_contracts=getattr(saved_state, "core_contracts", None),
        core_eth_qty=getattr(saved_state, "core_eth_qty", 0.0),
        startup_force_tp_reconcile=bool(getattr(saved_state, "startup_force_tp_reconcile", False)),
    )
    logger.warning(
        "Recovered strategy state from local disk | position_id=%s side=%s layers=%s avg_entry=%.4f tp=%s partial_tp=%s tp_plan=%s partial_tp_consumed=%s",
        saved_state.position_id,
        saved_state.side,
        saved_state.layers,
        saved_state.avg_entry_price,
        saved_state.tp_price,
        getattr(saved_state, "partial_tp_price", None),
        tp_plan,
        getattr(saved_state, "partial_tp_consumed", False),
    )


def sync_strategy_cost_from_position(strategy: BollCvdReclaimStrategy, position: PositionSnapshot) -> None:
    if not position.has_position or position.side is None or position.avg_entry_price <= 0:
        return
    if strategy.state.side is None or strategy.state.side != position.side or strategy.state.layers <= 0:
        restore_strategy_from_position(strategy, position)
        return
    if getattr(strategy.state, "three_stage_runner_enabled_for_position", False):
        strategy.state.avg_entry_price = position.avg_entry_price
        strategy.state.last_entry_price = strategy.state.last_entry_price or position.avg_entry_price
        return
    strategy.state.total_entry_qty = position.eth_qty
    strategy.state.total_entry_notional = position.avg_entry_price * position.eth_qty
    strategy.state.avg_entry_price = position.avg_entry_price
    strategy.state.last_entry_price = strategy.state.last_entry_price or position.avg_entry_price


def refresh_net_remaining_breakeven(strategy_state: StrategyPositionState, fee_buffer_pct: float = DEFAULT_NET_REMAINING_FEE_BUFFER_PCT) -> None:
    if strategy_state.side not in {"LONG", "SHORT"}:
        strategy_state.net_remaining_breakeven_price = 0.0
        return
    basis = calculate_remaining_breakeven_price(
        side=strategy_state.side,
        entry_notional=float(getattr(strategy_state, "position_cost_entry_notional", 0.0) or 0.0),
        exit_notional=float(getattr(strategy_state, "position_cost_exit_notional", 0.0) or 0.0),
        remaining_qty=float(getattr(strategy_state, "position_cost_remaining_qty", 0.0) or 0.0),
        fee_buffer_pct=fee_buffer_pct,
    )
    strategy_state.net_remaining_breakeven_price = float(basis.buffered_breakeven_price or 0.0)


def record_remaining_entry_notional(
    strategy_state: StrategyPositionState,
    *,
    qty: float,
    price: float,
    fee_buffer_pct: float = DEFAULT_NET_REMAINING_FEE_BUFFER_PCT,
) -> None:
    if qty <= 0 or price <= 0:
        return
    strategy_state.position_cost_entry_notional += float(qty) * float(price)
    strategy_state.position_cost_remaining_qty += float(qty)
    refresh_net_remaining_breakeven(strategy_state, fee_buffer_pct)


def record_remaining_exit_notional(
    strategy_state: StrategyPositionState,
    *,
    qty: float,
    price: float,
    remaining_qty: float | None = None,
    fee_buffer_pct: float = DEFAULT_NET_REMAINING_FEE_BUFFER_PCT,
) -> None:
    if qty <= 0 or price <= 0:
        return
    strategy_state.position_cost_exit_notional += float(qty) * float(price)
    if remaining_qty is None:
        strategy_state.position_cost_remaining_qty = max(float(strategy_state.position_cost_remaining_qty or 0.0) - float(qty), 0.0)
    else:
        strategy_state.position_cost_remaining_qty = max(float(remaining_qty or 0.0), 0.0)
    refresh_net_remaining_breakeven(strategy_state, fee_buffer_pct)


def remaining_total_qty_from_core_position(strategy_state: StrategyPositionState, core_position: PositionSnapshot) -> float:
    return max(float(core_position.eth_qty or 0.0), 0.0) + sidecar_open_qty(list(getattr(strategy_state, "sidecar_legs", []) or []))


def record_core_position_reduction_exit(
    strategy_state: StrategyPositionState,
    core_position: PositionSnapshot,
    *,
    exit_price: float | None,
    fee_buffer_pct: float = DEFAULT_NET_REMAINING_FEE_BUFFER_PCT,
    expected_remaining_qty: float | None = None,
) -> None:
    price = float(exit_price or 0.0)
    if price <= 0:
        return
    new_remaining_qty = remaining_total_qty_from_core_position(strategy_state, core_position)
    old_remaining_qty = float(getattr(strategy_state, "position_cost_remaining_qty", 0.0) or 0.0)
    if expected_remaining_qty is not None and old_remaining_qty > expected_remaining_qty > new_remaining_qty:
        qty = old_remaining_qty - expected_remaining_qty
        remaining_qty = expected_remaining_qty
    else:
        qty = old_remaining_qty - new_remaining_qty
        remaining_qty = new_remaining_qty
    if qty <= 0:
        total_entry_qty = float(getattr(strategy_state, "total_entry_qty", 0.0) or 0.0)
        qty = max(total_entry_qty - float(core_position.eth_qty or 0.0), 0.0)
    record_remaining_exit_notional(
        strategy_state,
        qty=qty,
        price=price,
        remaining_qty=remaining_qty,
        fee_buffer_pct=fee_buffer_pct,
    )


def record_sidecar_tp_fill_exit(
    strategy_state: StrategyPositionState,
    leg: dict[str, Any],
    status: dict[str, Any],
    *,
    fee_buffer_pct: float = DEFAULT_NET_REMAINING_FEE_BUFFER_PCT,
) -> None:
    filled_qty = _coerce_positive_float(status.get("filled_qty")) or _coerce_positive_float(leg.get("qty"))
    fill_price = _coerce_positive_float(status.get("avg_fill_price")) or _coerce_positive_float(leg.get("tp_price"))
    if filled_qty is None or fill_price is None:
        return
    record_remaining_exit_notional(
        strategy_state,
        qty=filled_qty,
        price=fill_price,
        fee_buffer_pct=fee_buffer_pct,
    )


def _coerce_positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def mark_partial_tp_consumed_if_position_reduced(strategy: BollCvdReclaimStrategy, position: PositionSnapshot) -> bool:
    state = strategy.state
    original_plan = getattr(state, "tp_plan", "SINGLE")
    if original_plan not in SPLIT_TP_PLANS:
        return False
    if not position.has_position or position.side != state.side:
        return False
    total_entry_qty = float(getattr(state, "total_entry_qty", 0.0) or 0.0)
    if total_entry_qty <= 0:
        return False

    old_partial_tp_price = getattr(state, "partial_tp_price", None)
    partial_tp_ratio = float(getattr(state, "partial_tp_ratio", 0.0) or 0.0)
    reduction_ratio = 1 - (float(position.eth_qty) / total_entry_qty)
    required_ratio = max(0.05, partial_tp_ratio * 0.5)
    if reduction_ratio < required_ratio:
        return False

    record_core_position_reduction_exit(
        state,
        position,
        exit_price=old_partial_tp_price,
        fee_buffer_pct=strategy.config.breakeven_fee_buffer_pct,
    )
    state.partial_tp_consumed = True
    state.partial_tp_price = None
    state.partial_tp_ratio = 0.0
    state.tp_plan = "SINGLE"
    logger.warning(
        "SPLIT_TP_CONSUMED | side=%s original_plan=%s partial_tp_price=%s old_qty=%.8f new_qty=%.8f reduction_ratio=%.6f required_ratio=%.6f partial_ratio=%.4f",
        state.side,
        original_plan,
        old_partial_tp_price,
        total_entry_qty,
        position.eth_qty,
        reduction_ratio,
        required_ratio,
        partial_tp_ratio,
    )
    return True


def mark_middle_runner_active_if_position_reduced(strategy: BollCvdReclaimStrategy, position: PositionSnapshot) -> bool:
    state = strategy.state
    if not getattr(state, "middle_runner_pending", False):
        return False
    if getattr(state, "middle_runner_active", False):
        return False
    if not position.has_position or position.side != state.side:
        return False
    total_entry_qty = float(getattr(state, "total_entry_qty", 0.0) or 0.0)
    keep_ratio = float(getattr(state, "middle_runner_keep_ratio", 0.0) or 0.0)
    if total_entry_qty <= 0 or keep_ratio <= 0 or keep_ratio >= 1:
        logger.warning(
            "MIDDLE_RUNNER_ORDER_WARNING | reason=activation_size_unknown side=%s total_entry_qty=%.8f keep_ratio=%.6f okx_eth_qty=%.8f",
            state.side,
            total_entry_qty,
            keep_ratio,
            position.eth_qty,
        )
        return False

    expected_qty = total_entry_qty * keep_ratio
    tolerance = max(total_entry_qty * 0.03, expected_qty * 0.10, 0.000001)
    if abs(float(position.eth_qty) - expected_qty) > tolerance:
        reduction_ratio = 1 - (float(position.eth_qty) / total_entry_qty)
        if reduction_ratio > 0.05:
            state.middle_runner_add_disabled = True
            now_ms = int(time.time() * 1000)
            last_warning_ms = int(getattr(state, "middle_runner_size_mismatch_warning_ts_ms", 0) or 0)
            if last_warning_ms <= 0 or now_ms - last_warning_ms >= 60_000:
                state.middle_runner_size_mismatch_warning_ts_ms = now_ms
                logger.warning(
                    "MIDDLE_RUNNER_ORDER_WARNING | reason=partial_size_mismatch_add_disabled side=%s old_qty=%.8f new_qty=%.8f expected_qty=%.8f tolerance=%.8f reduction_ratio=%.6f keep_ratio=%.6f",
                    state.side,
                    total_entry_qty,
                    position.eth_qty,
                    expected_qty,
                    tolerance,
                    reduction_ratio,
                    keep_ratio,
                )
        return False

    record_core_position_reduction_exit(
        state,
        position,
        exit_price=getattr(state, "middle_runner_first_tp_price", None),
        fee_buffer_pct=strategy.config.breakeven_fee_buffer_pct,
    )
    logger.warning(
        "MIDDLE_RUNNER_COST_BASIS_AFTER_FIRST_CLOSE | side=%s total_entry_qty=%.8f okx_core_eth_qty=%.8f sidecar_open_qty=%.8f position_cost_entry_notional=%.4f position_cost_exit_notional=%.4f position_cost_remaining_qty=%.8f net_remaining_breakeven_price=%.4f avg_entry_price=%.4f first_tp_price=%s first_close_ratio=%.4f keep_ratio=%.4f",
        state.side,
        total_entry_qty,
        float(position.eth_qty or 0.0),
        sidecar_open_qty(list(getattr(state, "sidecar_legs", []) or [])),
        float(getattr(state, "position_cost_entry_notional", 0.0) or 0.0),
        float(getattr(state, "position_cost_exit_notional", 0.0) or 0.0),
        float(getattr(state, "position_cost_remaining_qty", 0.0) or 0.0),
        float(getattr(state, "net_remaining_breakeven_price", 0.0) or 0.0),
        float(getattr(state, "avg_entry_price", 0.0) or 0.0),
        getattr(state, "middle_runner_first_tp_price", None),
        float(getattr(state, "middle_runner_first_close_ratio", 0.0) or 0.0),
        keep_ratio,
    )
    state.middle_runner_pending = False
    state.middle_runner_active = True
    state.middle_runner_add_disabled = True
    state.partial_tp_consumed = True
    state.partial_tp_price = None
    state.partial_tp_ratio = 0.0
    state.tp_plan = "SINGLE"
    logger.warning(
        "MIDDLE_RUNNER_ACTIVATED | side=%s old_qty=%.8f new_qty=%.8f expected_qty=%.8f first_close_ratio=%.4f keep_ratio=%.4f final_tp_price=%s add_disabled=true",
        state.side,
        total_entry_qty,
        position.eth_qty,
        expected_qty,
        getattr(state, "middle_runner_first_close_ratio", 0.0),
        keep_ratio,
        getattr(state, "middle_runner_final_tp_price", None),
    )
    return True


def mark_three_stage_progress_if_position_reduced(strategy: BollCvdReclaimStrategy, position: PositionSnapshot, ts_ms: int) -> str | None:
    state = strategy.state
    if not getattr(state, "three_stage_runner_enabled_for_position", False):
        return None
    if not position.has_position or position.side != state.side:
        return None
    total_entry_qty = float(getattr(state, "total_entry_qty", 0.0) or 0.0)
    if total_entry_qty <= 0:
        return None

    remaining_ratio = float(position.eth_qty) / total_entry_qty
    tp1_ratio = float(getattr(state, "three_stage_tp1_ratio", 0.0) or 0.0)
    tp2_ratio = float(getattr(state, "three_stage_tp2_ratio", 0.0) or 0.0)
    runner_ratio = float(getattr(state, "three_stage_runner_ratio", 0.0) or 0.0)
    after_tp1_ratio = max(0.0, 1.0 - tp1_ratio)
    after_tp2_ratio = max(0.0, runner_ratio)
    tp1_tolerance = max(0.02, tp1_ratio * 0.05, 0.000001)
    tp2_tolerance = max(0.01, runner_ratio * 0.10, 0.000001)
    event: str | None = None

    if not getattr(state, "three_stage_tp1_consumed", False) and remaining_ratio <= after_tp1_ratio + tp1_tolerance:
        expected_after_tp1_qty = total_entry_qty * after_tp1_ratio + sidecar_open_qty(list(getattr(state, "sidecar_legs", []) or []))
        will_mark_tp2_now = remaining_ratio <= after_tp2_ratio + tp2_tolerance
        record_core_position_reduction_exit(
            state,
            position,
            exit_price=getattr(state, "three_stage_tp1_price", None),
            fee_buffer_pct=strategy.config.breakeven_fee_buffer_pct,
            expected_remaining_qty=expected_after_tp1_qty if will_mark_tp2_now else None,
        )
        logger.warning(
            "THREE_STAGE_COST_BASIS_AFTER_TP1 | side=%s total_entry_qty=%.8f okx_core_eth_qty=%.8f sidecar_open_qty=%.8f position_cost_entry_notional=%.4f position_cost_exit_notional=%.4f position_cost_remaining_qty=%.8f net_remaining_breakeven_price=%.4f avg_entry_price=%.4f tp1_price=%s tp1_ratio=%.4f remaining_ratio=%.6f",
            state.side,
            total_entry_qty,
            float(position.eth_qty or 0.0),
            sidecar_open_qty(list(getattr(state, "sidecar_legs", []) or [])),
            float(getattr(state, "position_cost_entry_notional", 0.0) or 0.0),
            float(getattr(state, "position_cost_exit_notional", 0.0) or 0.0),
            float(getattr(state, "position_cost_remaining_qty", 0.0) or 0.0),
            float(getattr(state, "net_remaining_breakeven_price", 0.0) or 0.0),
            float(getattr(state, "avg_entry_price", 0.0) or 0.0),
            getattr(state, "three_stage_tp1_price", None),
            tp1_ratio,
            remaining_ratio,
        )
        state.three_stage_tp1_consumed = True
        state.partial_tp_consumed = True
        event = "TP1"
        logger.warning(
            "THREE_STAGE_TP1_FILLED | side=%s old_qty=%.8f new_qty=%.8f remaining_ratio=%.6f expected_after_tp1=%.6f tp1_ratio=%.4f",
            state.side,
            total_entry_qty,
            position.eth_qty,
            remaining_ratio,
            after_tp1_ratio,
            tp1_ratio,
        )

    if (
        getattr(state, "three_stage_tp1_consumed", False)
        and not getattr(state, "three_stage_tp2_consumed", False)
        and remaining_ratio <= after_tp2_ratio + tp2_tolerance
    ):
        record_core_position_reduction_exit(
            state,
            position,
            exit_price=getattr(state, "three_stage_tp2_price", None),
            fee_buffer_pct=strategy.config.breakeven_fee_buffer_pct,
        )
        state.three_stage_tp2_consumed = True
        state.trend_runner_active = True
        state.trend_runner_trend_start_ts_ms = ts_ms
        state.trend_runner_adjust_count = 0
        state.trend_runner_last_update_candle_ts_ms = 0
        state.trend_runner_tp_price = None
        state.trend_runner_sl_price = None
        state.trend_runner_tp_order_id = None
        state.trend_runner_sl_order_id = None
        state.tp_plan = "SINGLE"
        state.partial_tp_price = None
        state.partial_tp_ratio = 0.0
        logger.warning(
            "TREND_RUNNER_ACTIVATED | side=%s old_qty=%.8f new_qty=%.8f remaining_ratio=%.6f runner_ratio=%.6f tp2_ratio=%.4f runner_tp=%s runner_sl=%s trend_start_ts_ms=%s",
            state.side,
            total_entry_qty,
            position.eth_qty,
            remaining_ratio,
            runner_ratio,
            tp2_ratio,
            getattr(state, "trend_runner_tp_price", None),
            getattr(state, "trend_runner_sl_price", None),
            ts_ms,
        )
        return "TP1_TP2" if event == "TP1" else "TP2"
    return event


def append_three_stage_progress_journal_events(journal: Any, payload: dict[str, Any]) -> None:
    event = payload.get("event")
    position_id = payload.get("position_id")
    if event in {"TP1", "TP1_TP2"}:
        journal.append("THREE_STAGE_TP1_FILLED", dict(payload), position_id=position_id)
    if event in {"TP2", "TP1_TP2"}:
        journal.append("THREE_STAGE_TP2_FILLED", dict(payload), position_id=position_id)
        journal.append("TREND_RUNNER_ACTIVATED", dict(payload), position_id=position_id)


def middle_runner_activation_boll(strategy: BollCvdReclaimStrategy):
    state = strategy.state
    middle = getattr(state, "middle_runner_first_tp_price", None)
    final = getattr(state, "middle_runner_final_tp_price", None) or getattr(state, "tp_price", None)
    if middle is None or final is None:
        return None
    width = abs(float(final) - float(middle))
    if width <= 0:
        return None
    upper = float(final) if state.side == "LONG" else float(middle) + width
    lower = float(middle) - width if state.side == "LONG" else float(final)
    return type("MiddleRunnerBoll", (), {"middle": float(middle), "upper": upper, "lower": lower})()


def three_stage_post_tp1_boll(strategy: BollCvdReclaimStrategy):
    state = strategy.state
    middle = getattr(state, "three_stage_tp1_price", None)
    outer = getattr(state, "three_stage_tp2_price", None)
    if middle is None or outer is None:
        return None
    width = abs(float(outer) - float(middle))
    if width <= 0:
        return None
    upper = float(outer) if state.side == "LONG" else float(middle) + width
    lower = float(middle) - width if state.side == "LONG" else float(outer)
    return type("ThreeStagePostTp1Boll", (), {"middle": float(middle), "upper": upper, "lower": lower})()


def three_stage_post_tp1_current_price(account_snapshot: live_runtime_types.AccountSnapshot, position: PositionSnapshot, post_tp1_boll: Any, now_ms: int) -> tuple[float, str]:
    latest_price = getattr(account_snapshot, "latest_market_price", None)
    latest_ts_ms = int(getattr(account_snapshot, "latest_market_price_ts_ms", 0) or 0)
    max_age_seconds = float(os.getenv("LATEST_MARKET_PRICE_MAX_AGE_SECONDS", "30"))
    max_age_ms = max(int(max_age_seconds * 1000), 0)
    if latest_price is not None and float(latest_price) > 0:
        age_ms = now_ms - latest_ts_ms if latest_ts_ms > 0 else max_age_ms + 1
        if latest_ts_ms > 0 and age_ms <= max_age_ms:
            return float(latest_price), "latest_market_price"
        logger.warning(
            "THREE_STAGE_POST_TP1_SL_PRICE_FALLBACK | reason=latest_market_price_stale latest_price=%.4f latest_ts_ms=%s age_ms=%s max_age_ms=%s",
            float(latest_price),
            latest_ts_ms,
            age_ms,
            max_age_ms,
        )
    if position.avg_entry_price > 0:
        logger.warning(
            "THREE_STAGE_POST_TP1_SL_PRICE_FALLBACK | reason=latest_market_price_missing fallback=position_avg_entry avg_entry=%.4f boll_middle=%.4f",
            position.avg_entry_price,
            float(getattr(post_tp1_boll, "middle", 0.0) or 0.0),
        )
        return float(position.avg_entry_price), "position_avg_entry"
    fallback = float(getattr(post_tp1_boll, "middle", 0.0) or 0.0)
    logger.warning(
        "THREE_STAGE_POST_TP1_SL_PRICE_FALLBACK | reason=latest_market_price_and_avg_entry_missing fallback=boll_middle boll_middle=%.4f",
        fallback,
    )
    return fallback, "boll_middle"


def middle_runner_size_mismatch_needs_degraded_protection(strategy: BollCvdReclaimStrategy, position: PositionSnapshot) -> bool:
    state = strategy.state
    if not getattr(state, "middle_runner_pending", False):
        return False
    if getattr(state, "middle_runner_active", False):
        return False
    if getattr(state, "middle_runner_size_mismatch_protected", False):
        return False
    if not getattr(state, "middle_runner_add_disabled", False):
        return False
    if not position.has_position or position.side != state.side:
        return False
    total_entry_qty = float(getattr(state, "total_entry_qty", 0.0) or 0.0)
    if total_entry_qty <= 0:
        return False
    reduction_ratio = 1 - (float(position.eth_qty) / total_entry_qty)
    return reduction_ratio >= 0.5


def position_log_key(position: PositionSnapshot) -> tuple[str, str, float]:
    if not position.has_position or position.side is None:
        return ("FLAT", "0", 0.0)
    return (position.side, str(position.contracts), round(position.avg_entry_price, 2))


def open_sidecar_legs_exceed_limit(state: StrategyPositionState, max_legs: int) -> bool:
    open_count = sum(
        1
        for leg in list(getattr(state, "sidecar_legs", []) or [])
        if leg.get("status") in {SidecarLegStatus.OPEN.value, SidecarLegStatus.OPEN_UNPROTECTED.value}
    )
    return open_count > max(int(max_legs), 1)


def refresh_sidecar_state_totals(state: StrategyPositionState, max_legs: int = 10) -> None:
    state.sidecar_legs = trim_sidecar_legs_for_state(list(getattr(state, "sidecar_legs", []) or []), max_legs)
    state.sidecar_open_qty = sidecar_open_qty(state.sidecar_legs)
    state.sidecar_total_qty = sum(float(leg.get("qty") or 0.0) for leg in state.sidecar_legs)
    state.sidecar_total_notional = sum(float(leg.get("qty") or 0.0) * float(leg.get("entry_price") or 0.0) for leg in state.sidecar_legs)
    state.sidecar_realized_qty = sum(
        float(leg.get("qty") or 0.0)
        for leg in state.sidecar_legs
        if leg.get("status") in {SidecarLegStatus.TP_FILLED.value, SidecarLegStatus.FORCE_CLOSED.value, SidecarLegStatus.CANCELLED.value}
    )


def apply_core_position_view_to_state(state: StrategyPositionState, core_position: PositionSnapshot) -> None:
    if core_position.has_position:
        state.core_contracts = str(core_position.contracts)
        state.core_eth_qty = float(core_position.eth_qty)
    else:
        state.core_contracts = None
        state.core_eth_qty = 0.0


def with_runtime_managed_core(intent: TradeIntent, account_position: PositionSnapshot | None) -> TradeIntent:
    if not getattr(intent, "managed_core_contracts", None) and account_position is not None and account_position.has_position:
        if intent.intent_type in {"UPDATE_TP", "NEAR_TP_REDUCE", "MARKET_EXIT_RUNNER"}:
            return replace(
                intent,
                managed_core_contracts=str(account_position.contracts),
                managed_core_eth_qty=float(account_position.eth_qty),
            )
    return intent


def sidecar_client_order_id(position_id: str | None, layer_index: int, ts_ms: int) -> str:
    return build_sidecar_client_order_id(position_id, layer_index, ts_ms)


def sidecar_position_mismatch(okx_position: PositionSnapshot, state: StrategyPositionState, tolerance_qty: float = 0.000001) -> bool:
    if not getattr(state, "sidecar_enabled_for_position", False):
        return False
    open_qty = sidecar_open_qty(list(getattr(state, "sidecar_legs", []) or []))
    if open_qty <= 0:
        return False
    if not okx_position.has_position or okx_position.side != state.side:
        return True
    if open_qty - float(okx_position.eth_qty) > tolerance_qty:
        return True
    core_position = build_core_position_view(okx_position, open_qty, sidecar_open_contracts(state.sidecar_legs))
    return abs((core_position.eth_qty + open_qty) - okx_position.eth_qty) > tolerance_qty


async def attach_sidecar_after_combined_entry(
    *,
    trader: Trader,
    strategy_state: StrategyPositionState,
    execution_state: live_runtime_types.ExecutionState,
    intent: TradeIntent,
    sidecar_plan: SidecarExecutionPlan,
    journal: LiveTradeJournal,
    state_store: LiveStateStore,
    trader_symbol: str,
    fee_buffer_pct: float = DEFAULT_NET_REMAINING_FEE_BUFFER_PCT,
) -> bool:
    if not getattr(strategy_state, "sidecar_enabled_for_position", False):
        return True
    if intent.intent_type not in {"OPEN_LONG", "OPEN_SHORT", "ADD_LONG", "ADD_SHORT"}:
        return True
    position_id = execution_state.current_position_id
    contracts = str(sidecar_plan.sidecar_contracts)
    filled_qty = float(sidecar_plan.sidecar_qty)
    record_remaining_entry_notional(
        strategy_state,
        qty=filled_qty,
        price=float(intent.price),
        fee_buffer_pct=fee_buffer_pct,
    )
    leg = sidecar_leg_from_fill(
        leg_id=f"{position_id}:SC:{intent.layer_index}:{intent.ts_ms}",
        position_id=str(position_id or ""),
        layer_index=intent.layer_index,
        side=intent.side,
        entry_price=float(intent.price),
        qty=filled_qty,
        contracts=contracts,
        margin_pct=float(sidecar_plan.sidecar_margin_pct),
        layer_multiplier=float(sidecar_plan.layer_multiplier),
        tp_pct=float(strategy_state.sidecar_tp_pct or 0.0),
        tp_order_id=None,
        ts_ms=int(intent.ts_ms),
    )
    leg["tp_price"] = float(sidecar_plan.sidecar_tp_price)
    leg["sidecar_client_order_id"] = sidecar_plan.client_order_id

    # ── Place sidecar TP BEFORE appending leg to state ─────────────────
    # The leg must never appear in strategy_state.sidecar_legs with
    # status=OPEN + tp_order_id=None.  If the pre-core reconcile runs
    # concurrently it would flag that intermediate state as dirty and
    # halt the position unnecessarily.
    try:
        tp_order_id = await trader.place_sidecar_fixed_take_profit(
            side=intent.side,
            contracts=contracts,
            tp_price=float(leg["tp_price"]),
            client_order_id=sidecar_plan.client_order_id,
        )
    except Exception as exc:
        leg = mark_sidecar_leg_open_unprotected(leg, int(intent.ts_ms), warning_recorded=True)
        strategy_state.sidecar_legs.append(leg)
        execution_state.trading_halted = True
        strategy_state.sidecar_dirty = True
        exit_ok, exit_message = await trader.market_exit_remaining_position_with_retries(
            intent.side,
            retry_count=int(os.getenv("SIDECAR_TP_FAIL_MARKET_EXIT_RETRY_COUNT", os.getenv("NEAR_TP_SL_FAIL_MARKET_EXIT_RETRY_COUNT", "3"))),
        )
        if exit_ok:
            execution_state.halt_reason = "sidecar_tp_place_failed_market_exit_waiting_flat"
            strategy_state.sidecar_halt_reason = "sidecar_tp_place_failed_market_exit_waiting_flat"
        else:
            execution_state.halt_reason = "sidecar_tp_place_failed"
            strategy_state.sidecar_halt_reason = "sidecar_tp_place_failed"
        refresh_sidecar_state_totals(strategy_state, int(os.getenv("SIDECAR_MAX_LEGS", "10")))
        state_store.save(LiveStateStore.from_strategy_state(position_id=position_id, symbol=trader_symbol, strategy_state=strategy_state, cash_before_position=execution_state.cash_before_position))
        manual_intervention_required = not exit_ok
        journal.append(
            "SIDECAR_TP_PLACE_FAILED",
            {
                **dict(leg),
                "error": str(exc),
                "market_exit_attempted": True,
                "market_exit_ok": exit_ok,
                "market_exit_message": exit_message,
                "sidecar_contracts": str(sidecar_plan.sidecar_contracts),
                "sidecar_qty": sidecar_plan.sidecar_qty,
                "core_contracts": str(sidecar_plan.core_contracts),
                "net_contracts": str(sidecar_plan.total_contracts),
                "total_contracts": str(sidecar_plan.total_contracts),
                "sidecar_status": SidecarLegStatus.OPEN_UNPROTECTED.value,
                "manual_intervention_required": manual_intervention_required,
            },
            position_id=position_id,
        )
        logger.error(
            "SIDECAR_TP_PLACE_FAILED | position_id=%s leg_id=%s error=%s market_exit_attempted=true market_exit_ok=%s manual_intervention_required=%s",
            position_id,
            leg.get("leg_id"),
            exc,
            exit_ok,
            manual_intervention_required,
        )
        return False

    # TP placed successfully → now append leg to state with tp_order_id set
    leg["tp_order_id"] = tp_order_id
    leg["updated_ts_ms"] = int(intent.ts_ms)
    strategy_state.sidecar_legs.append(leg)
    refresh_sidecar_state_totals(strategy_state, int(os.getenv("SIDECAR_MAX_LEGS", "10")))
    state_store.save(LiveStateStore.from_strategy_state(position_id=position_id, symbol=trader_symbol, strategy_state=strategy_state, cash_before_position=execution_state.cash_before_position))
    journal.append("SIDECAR_LEG_OPENED", dict(leg), position_id=position_id)
    journal.append("SIDECAR_TP_PLACED", dict(leg), position_id=position_id)
    return True


async def execute_sidecar_after_core_entry(
    *,
    trader: Trader,
    strategy_state: StrategyPositionState,
    execution_state: live_runtime_types.ExecutionState,
    intent: TradeIntent,
    journal: LiveTradeJournal,
    state_store: LiveStateStore,
    trader_symbol: str,
) -> bool:
    logger.error(
        "SIDECAR_LEGACY_AFTER_CORE_ENTRY_DISABLED | position_id=%s intent_type=%s side=%s layer=%s",
        execution_state.current_position_id,
        intent.intent_type,
        intent.side,
        intent.layer_index,
    )
    return False


async def reconcile_sidecar_orders_before_core_view(
    *,
    trader: Trader,
    strategy: BollCvdShockReclaimStrategy,
    execution_state: live_runtime_types.ExecutionState,
    journal: LiveTradeJournal,
    state_store: LiveStateStore,
    trader_symbol: str,
    ts_ms: int,
    state_lock: asyncio.Lock,
) -> live_runtime_types.SidecarPreCoreReconcileResult:
    """Reconcile sidecar TP order status BEFORE constructing core_position view.

    When Sidecar is enabled and OPEN sidecar legs exist, this must be called
    before computing core_position = OKX_net - sidecar_open_qty. Otherwise a
    sidecar TP that already filled on OKX (but not yet reflected in local state)
    would cause core_position to be understated, which can incorrectly trigger
    TP progress markers or pollute strategy average entry.

    Returns live_runtime_types.SidecarPreCoreReconcileResult:
      - queried: True if we performed any REST order status fetch for OPEN legs.
      - changed: True if any sidecar state was modified and saved.

    Sets trading_halted if unrecoverable state is detected.
    """
    # Pending orders mean core position is in flux — do not reconcile sidecar
    # orders or advance core state.  Return False / False to allow the caller
    # to fall through safely without blocking the sync cycle.
    if execution_state.pending_order_count > 0:
        return live_runtime_types.SidecarPreCoreReconcileResult(queried=False, changed=False)

    if not getattr(strategy.state, "sidecar_enabled_for_position", False):
        return live_runtime_types.SidecarPreCoreReconcileResult(queried=False, changed=False)

    # --- Phase 1: handle dirty / missing TP orders under lock (no network) ---
    dirty_changed = False
    async with state_lock:
        for index, leg in enumerate(list(strategy.state.sidecar_legs)):
            if leg.get("status") != SidecarLegStatus.OPEN.value:
                continue
            if is_sidecar_dirty_missing_tp_order(leg):
                execution_state.trading_halted = True
                execution_state.halt_reason = "sidecar_tp_order_missing_or_unknown"
                strategy.state.sidecar_dirty = True
                strategy.state.sidecar_halt_reason = "sidecar_tp_order_missing_or_unknown"
                if not leg.get("warning_recorded") and hasattr(journal, "append"):
                    journal.append("SIDECAR_TP_ORDER_MISSING_OR_UNKNOWN", dict(leg), position_id=execution_state.current_position_id)
                strategy.state.sidecar_legs[index] = mark_sidecar_leg_unknown_halted(leg, ts_ms)
                dirty_changed = True
        if dirty_changed:
            refresh_sidecar_state_totals(strategy.state, int(os.getenv("SIDECAR_MAX_LEGS", "10")))
            state_store.save(LiveStateStore.from_strategy_state(
                position_id=execution_state.current_position_id,
                symbol=trader_symbol,
                strategy_state=strategy.state,
                cash_before_position=execution_state.cash_before_position,
            ))
        # Snapshot remaining OPEN legs for network queries
        open_legs: list[tuple[int, str, str]] = []
        for index, leg in enumerate(strategy.state.sidecar_legs):
            if leg.get("status") != SidecarLegStatus.OPEN.value:
                continue
            if is_sidecar_dirty_missing_tp_order(leg):
                continue
            tp_order_id = leg.get("tp_order_id")
            if not tp_order_id:
                continue
            open_legs.append((index, str(tp_order_id), str(leg.get("leg_id", ""))))
        position_id = execution_state.current_position_id
        cash_before_position = execution_state.cash_before_position

    if not open_legs:
        return live_runtime_types.SidecarPreCoreReconcileResult(queried=False, changed=dirty_changed)

    # --- Phase 2: query order status (outside lock) ---
    leg_updates: list[tuple[int, str, dict[str, Any], str]] = []
    for index, order_id, leg_id in open_legs:
        status = await trader.fetch_sidecar_order_status(order_id)
        order_status = status.get("status")
        if order_status != "OPEN":
            leg_updates.append((index, order_status, status, leg_id))

    if not leg_updates:
        # We queried open legs but found no status changes.
        return live_runtime_types.SidecarPreCoreReconcileResult(queried=True, changed=False)

    # --- Phase 3: apply updates under lock ---
    changed = dirty_changed
    async with state_lock:
        for index, order_status, status_dict, expected_leg_id in leg_updates:
            if index >= len(strategy.state.sidecar_legs):
                continue
            leg = strategy.state.sidecar_legs[index]
            if leg.get("status") != SidecarLegStatus.OPEN.value:
                continue
            if leg.get("leg_id") != expected_leg_id:
                continue

            if order_status == "FILLED":
                record_sidecar_tp_fill_exit(
                    strategy.state,
                    leg,
                    status_dict,
                    fee_buffer_pct=getattr(getattr(strategy, "config", None), "breakeven_fee_buffer_pct", DEFAULT_NET_REMAINING_FEE_BUFFER_PCT),
                )
                strategy.state.sidecar_legs[index] = mark_sidecar_leg_tp_filled(leg, ts_ms)
                if hasattr(journal, "append"):
                    journal.append("SIDECAR_TP_FILLED", {**dict(leg), **status_dict}, position_id=position_id)
                changed = True

                # Sidecar TP reduces OKX net position → existing global SL orders
                # may now exceed current net position. Must halt for manual reconciliation.
                active_global_sl_orders: list[str] = []
                for sl_field in (
                    "near_tp_protective_sl_order_id",
                    "middle_runner_protective_sl_order_id",
                    "three_stage_post_tp1_protective_sl_order_id",
                    "trend_runner_sl_order_id",
                ):
                    # Use trader fallback (same as monitor_sidecar_orders_once)
                    # because SL orders may have been placed by startup recovery
                    # or by a previous session and only tracked on the trader.
                    sl_order_id = (
                        getattr(strategy.state, sl_field, None)
                        or getattr(trader, sl_field, None)
                    )
                    if sl_order_id:
                        active_global_sl_orders.append(f"{sl_field}={sl_order_id}")
                if active_global_sl_orders:
                    execution_state.trading_halted = True
                    execution_state.halt_reason = "sidecar_tp_filled_requires_global_sl_reconcile"
                    strategy.state.sidecar_dirty = True
                    strategy.state.sidecar_halt_reason = "sidecar_tp_filled_requires_global_sl_reconcile"
                    if hasattr(journal, "append"):
                        journal.append(
                            "SIDECAR_TP_FILLED_REQUIRES_GLOBAL_SL_RECONCILE",
                            {
                                "active_global_sl_orders": active_global_sl_orders,
                                "trading_halted": True,
                                "halt_reason": "sidecar_tp_filled_requires_global_sl_reconcile",
                                "manual_intervention_required": True,
                            },
                            position_id=position_id,
                        )
                    logger.error(
                        "SIDECAR_TP_FILLED_REQUIRES_GLOBAL_SL_RECONCILE | position_id=%s leg_id=%s active_global_sl_orders=%s trading_halted=true halt_reason=sidecar_tp_filled_requires_global_sl_reconcile manual_intervention_required=true",
                        position_id,
                        leg.get("leg_id"),
                        active_global_sl_orders,
                    )
                continue

            if order_status in {"CANCELED", "NOT_FOUND", "UNKNOWN"}:
                # Without a verified core view we cannot determine whether the
                # remaining OKX position is core-only or core+sidecar. Halt.
                execution_state.trading_halted = True
                execution_state.halt_reason = "sidecar_tp_order_missing_or_unknown"
                strategy.state.sidecar_dirty = True
                strategy.state.sidecar_halt_reason = "sidecar_tp_order_missing_or_unknown"
                if not leg.get("warning_recorded") and hasattr(journal, "append"):
                    journal.append(
                        "SIDECAR_TP_ORDER_MISSING_OR_UNKNOWN",
                        {**dict(leg), **status_dict, "manual_intervention_required": True},
                        position_id=position_id,
                    )
                strategy.state.sidecar_legs[index] = mark_sidecar_leg_unknown_halted(leg, ts_ms)
                changed = True
                logger.error(
                    "SIDECAR_TP_ORDER_MISSING_OR_UNKNOWN | position_id=%s leg_id=%s status=%s manual_intervention_required=true",
                    position_id,
                    leg.get("leg_id"),
                    order_status,
                )

        if changed:
            refresh_sidecar_state_totals(strategy.state, int(os.getenv("SIDECAR_MAX_LEGS", "10")))
            state_store.save(LiveStateStore.from_strategy_state(
                position_id=position_id,
                symbol=trader_symbol,
                strategy_state=strategy.state,
                cash_before_position=cash_before_position,
            ))

    return live_runtime_types.SidecarPreCoreReconcileResult(queried=True, changed=changed)


async def monitor_sidecar_orders_once(
    *,
    trader: Trader,
    strategy_state: StrategyPositionState,
    execution_state: live_runtime_types.ExecutionState,
    journal: LiveTradeJournal,
    state_store: LiveStateStore,
    trader_symbol: str,
    core_position: PositionSnapshot,
    position_id: str | None,
    cash_before_position: float | None,
    ts_ms: int,
    fee_buffer_pct: float = DEFAULT_NET_REMAINING_FEE_BUFFER_PCT,
) -> None:
    if not getattr(strategy_state, "sidecar_enabled_for_position", False):
        return
    changed = False
    core_active = bool(core_position.has_position)
    for index, leg in enumerate(list(strategy_state.sidecar_legs)):
        if leg.get("status") != SidecarLegStatus.OPEN.value:
            continue
        if is_sidecar_dirty_missing_tp_order(leg):
            execution_state.trading_halted = True
            execution_state.halt_reason = "sidecar_tp_order_missing_or_unknown"
            strategy_state.sidecar_dirty = True
            strategy_state.sidecar_halt_reason = "sidecar_tp_order_missing_or_unknown"
            if not leg.get("warning_recorded"):
                journal.append("SIDECAR_TP_ORDER_MISSING_OR_UNKNOWN", dict(leg), position_id=position_id)
            strategy_state.sidecar_legs[index] = mark_sidecar_leg_unknown_halted(leg, ts_ms)
            changed = True
            continue
        status = await trader.fetch_sidecar_order_status(str(leg["tp_order_id"]))
        order_status = status.get("status")
        if order_status == "OPEN":
            continue
        if order_status == "FILLED":
            record_sidecar_tp_fill_exit(
                strategy_state,
                leg,
                status,
                fee_buffer_pct=fee_buffer_pct,
            )
            strategy_state.sidecar_legs[index] = mark_sidecar_leg_tp_filled(leg, ts_ms)
            journal.append("SIDECAR_TP_FILLED", {**dict(leg), **status}, position_id=position_id)
            changed = True
            # Sidecar TP filled reduces OKX net position → existing global SL orders
            # may now exceed current net position. Must halt for manual reconciliation.
            active_global_sl_orders: list[str] = []
            for sl_field in (
                "near_tp_protective_sl_order_id",
                "middle_runner_protective_sl_order_id",
                "three_stage_post_tp1_protective_sl_order_id",
                "trend_runner_sl_order_id",
            ):
                sl_order_id = getattr(strategy_state, sl_field, None) or getattr(trader, sl_field, None)
                if sl_order_id:
                    active_global_sl_orders.append(f"{sl_field}={sl_order_id}")
            if active_global_sl_orders:
                execution_state.trading_halted = True
                execution_state.halt_reason = "sidecar_tp_filled_requires_global_sl_reconcile"
                strategy_state.sidecar_dirty = True
                strategy_state.sidecar_halt_reason = "sidecar_tp_filled_requires_global_sl_reconcile"
                journal.append(
                    "SIDECAR_TP_FILLED_REQUIRES_GLOBAL_SL_RECONCILE",
                    {
                        "active_global_sl_orders": active_global_sl_orders,
                        "trading_halted": True,
                        "halt_reason": "sidecar_tp_filled_requires_global_sl_reconcile",
                        "manual_intervention_required": True,
                    },
                    position_id=position_id,
                )
                logger.error(
                    "SIDECAR_TP_FILLED_REQUIRES_GLOBAL_SL_RECONCILE | position_id=%s leg_id=%s active_global_sl_orders=%s trading_halted=true halt_reason=sidecar_tp_filled_requires_global_sl_reconcile manual_intervention_required=true",
                    position_id,
                    leg.get("leg_id"),
                    active_global_sl_orders,
                )
            continue
        if order_status in {"CANCELED", "NOT_FOUND", "UNKNOWN"} and core_active:
            execution_state.trading_halted = True
            execution_state.halt_reason = "sidecar_tp_order_missing_or_unknown"
            strategy_state.sidecar_dirty = True
            strategy_state.sidecar_halt_reason = "sidecar_tp_order_missing_or_unknown"
            if not leg.get("warning_recorded"):
                journal.append("SIDECAR_TP_ORDER_MISSING_OR_UNKNOWN", {**dict(leg), **status, "manual_intervention_required": True}, position_id=position_id)
            strategy_state.sidecar_legs[index] = mark_sidecar_leg_unknown_halted(leg, ts_ms)
            logger.error("SIDECAR_TP_ORDER_MISSING_OR_UNKNOWN | position_id=%s leg_id=%s status=%s manual_intervention_required=true", position_id, leg.get("leg_id"), order_status)
            changed = True
    if changed:
        refresh_sidecar_state_totals(strategy_state, int(os.getenv("SIDECAR_MAX_LEGS", "10")))
        state_store.save(LiveStateStore.from_strategy_state(position_id=position_id, symbol=trader_symbol, strategy_state=strategy_state, cash_before_position=cash_before_position))


async def force_close_sidecar_after_core_flat(
    *,
    trader: Trader,
    strategy_state: StrategyPositionState,
    execution_state: live_runtime_types.ExecutionState,
    journal: LiveTradeJournal,
    state_store: LiveStateStore,
    trader_symbol: str,
    position_id: str | None,
    cash_before_position: float | None,
    ts_ms: int,
) -> bool:
    if sidecar_open_qty(strategy_state.sidecar_legs) <= 0:
        return True
    expected_sidecar_contracts = sidecar_open_contracts(strategy_state.sidecar_legs)
    okx_position = await trader.fetch_position_snapshot()
    tolerance = Decimal(str(os.getenv("SIDECAR_FORCE_CLOSE_CONTRACT_TOLERANCE", "0.01")))
    if (
        not okx_position.has_position
        or okx_position.side != strategy_state.side
        or abs(okx_position.contracts - expected_sidecar_contracts) > tolerance
    ):
        execution_state.trading_halted = True
        execution_state.halt_reason = "sidecar_force_close_position_mismatch"
        strategy_state.sidecar_dirty = True
        strategy_state.sidecar_halt_reason = "sidecar_force_close_position_mismatch"
        payload = {
            "okx_side": okx_position.side,
            "okx_contracts": str(okx_position.contracts),
            "sidecar_open_contracts": str(expected_sidecar_contracts),
            "tolerance": str(tolerance),
            "manual_intervention_required": True,
        }
        journal.append("SIDECAR_FORCE_CLOSE_POSITION_MISMATCH", payload, position_id=position_id)
        logger.error(
            "SIDECAR_FORCE_CLOSE_POSITION_MISMATCH | position_id=%s okx_side=%s okx_contracts=%s sidecar_open_contracts=%s tolerance=%s trading_halted=true manual_intervention_required=true",
            position_id,
            okx_position.side,
            okx_position.contracts,
            expected_sidecar_contracts,
            tolerance,
        )
        state_store.save(LiveStateStore.from_strategy_state(position_id=position_id, symbol=trader_symbol, strategy_state=strategy_state, cash_before_position=cash_before_position))
        return False
    try:
        for leg in strategy_state.sidecar_legs:
            if leg.get("status") == SidecarLegStatus.OPEN.value and leg.get("tp_order_id"):
                ok = await trader.cancel_sidecar_take_profit(str(leg["tp_order_id"]))
                if not ok:
                    raise RuntimeError(f"cancel_sidecar_tp_failed order_id={leg.get('tp_order_id')}")
        side = strategy_state.side
        if side is None:
            raise RuntimeError("side_missing")
        exit_ok, exit_message = await trader.market_exit_remaining_position_with_retries(
            side,
            retry_count=int(os.getenv("NEAR_TP_SL_FAIL_MARKET_EXIT_RETRY_COUNT", "3")),
        )
        if not exit_ok:
            raise RuntimeError(exit_message)
    except Exception as exc:
        execution_state.trading_halted = True
        execution_state.halt_reason = "sidecar_force_close_failed"
        strategy_state.sidecar_dirty = True
        strategy_state.sidecar_halt_reason = "sidecar_force_close_failed"
        journal.append("SIDECAR_FORCE_CLOSE_FAILED", {"error": str(exc), "manual_intervention_required": True}, position_id=position_id)
        logger.error("SIDECAR_FORCE_CLOSE_FAILED | position_id=%s error=%s manual_intervention_required=true", position_id, exc)
        state_store.save(LiveStateStore.from_strategy_state(position_id=position_id, symbol=trader_symbol, strategy_state=strategy_state, cash_before_position=cash_before_position))
        return False
    strategy_state.sidecar_legs = [
        mark_sidecar_leg_force_closed(leg, ts_ms)
        if leg.get("status") in {SidecarLegStatus.OPEN.value, SidecarLegStatus.OPEN_UNPROTECTED.value}
        else leg
        for leg in strategy_state.sidecar_legs
    ]
    refresh_sidecar_state_totals(strategy_state, int(os.getenv("SIDECAR_MAX_LEGS", "10")))
    journal.append("SIDECAR_FORCE_CLOSED_AFTER_CORE_FLAT", {"side": strategy_state.side, "reason": "core_flat"}, position_id=position_id)
    state_store.save(LiveStateStore.from_strategy_state(position_id=position_id, symbol=trader_symbol, strategy_state=strategy_state, cash_before_position=cash_before_position))
    return True


async def fetch_usdt_cash_balance(trader: Trader) -> float:
    res = await trader.request("GET", "/api/v5/account/balance?ccy=USDT")
    data = res.get("data", [])
    if not data:
        return 0.0
    for item in data[0].get("details", []):
        if item.get("ccy") == "USDT":
            return float(item.get("cashBal") or item.get("availBal") or item.get("availEq") or item.get("eq") or 0.0)
    return float(data[0].get("totalEq") or 0.0)


async def fetch_settled_flat_balance(
    trader: Trader,
    *,
    attempts: int,
    interval_seconds: float,
    stable_delta_usdt: float,
    cash_equity_max_diff_usdt: float,
) -> live_runtime_types.SettledFlatBalance:
    attempts = max(int(attempts), 1)
    previous_flat_cash: float | None = None
    last_cash: float | None = None
    last_equity: float | None = None
    last_position: PositionSnapshot | None = None
    last_attempt = 0
    for attempt in range(1, attempts + 1):
        last_attempt = attempt
        try:
            position = await trader.fetch_position_snapshot()
            cash = await fetch_usdt_cash_balance(trader)
            equity = await trader.fetch_usdt_equity()
        except Exception as exc:
            if last_cash is not None and last_equity is not None:
                return live_runtime_types.SettledFlatBalance(
                    cash=last_cash,
                    equity=last_equity,
                    attempts=last_attempt,
                    stable=False,
                    reason=f"error_after_last_balance:{type(exc).__name__}:{exc}",
                )
            raise

        last_position = position
        last_cash = cash
        last_equity = equity
        if position.has_position:
            if attempt < attempts and interval_seconds > 0:
                await asyncio.sleep(interval_seconds)
            continue

        cash_equity_stable = abs(cash - equity) <= cash_equity_max_diff_usdt
        cash_repeat_stable = previous_flat_cash is not None and abs(cash - previous_flat_cash) <= stable_delta_usdt
        if cash_equity_stable and cash_repeat_stable:
            return live_runtime_types.SettledFlatBalance(
                cash=cash,
                equity=equity,
                attempts=attempt,
                stable=True,
                reason="cash_equity_stable",
            )
        previous_flat_cash = cash
        if attempt < attempts and interval_seconds > 0:
            await asyncio.sleep(interval_seconds)

    if last_cash is None or last_equity is None:
        raise RuntimeError("flat balance settlement finished without any balance sample")
    if last_position is not None and not last_position.has_position:
        return live_runtime_types.SettledFlatBalance(
            cash=last_equity,
            equity=last_equity,
            attempts=attempts,
            stable=False,
            reason="fallback_to_equity_after_timeout",
        )
    return live_runtime_types.SettledFlatBalance(
        cash=last_cash,
        equity=last_equity,
        attempts=attempts,
        stable=False,
        reason="position_not_flat_after_timeout",
    )


def three_stage_dirty_post_tp1_sl_after_tp2(state: StrategyPositionState) -> bool:
    return bool(
        getattr(state, "trend_runner_active", False)
        and getattr(state, "three_stage_tp2_consumed", False)
        and getattr(state, "three_stage_post_tp1_protective_sl_order_id", None)
    )


def three_stage_dirty_post_tp1_payload(
    *,
    strategy: BollCvdReclaimStrategy,
    execution_state: live_runtime_types.ExecutionState,
    reason: str,
) -> dict[str, Any]:
    state = strategy.state
    return {
        "position_id": execution_state.current_position_id,
        "side": getattr(state, "side", None),
        "protective_sl_order_id": getattr(state, "three_stage_post_tp1_protective_sl_order_id", None),
        "protective_sl_price": getattr(state, "three_stage_post_tp1_protective_sl_price", None),
        "trend_runner_active": getattr(state, "trend_runner_active", False),
        "three_stage_tp2_consumed": getattr(state, "three_stage_tp2_consumed", False),
        "trading_halted": True,
        "halt_reason": execution_state.halt_reason,
        "reason": reason,
    }


def append_three_stage_dirty_post_tp1_event(
    *,
    event_name: str,
    strategy: BollCvdReclaimStrategy,
    execution_state: live_runtime_types.ExecutionState,
    journal: LiveTradeJournal,
    state_store: LiveStateStore,
    trader_symbol: str,
    reason: str,
) -> None:
    payload = three_stage_dirty_post_tp1_payload(strategy=strategy, execution_state=execution_state, reason=reason)
    if hasattr(journal, "append"):
        journal.append(event_name, payload, position_id=execution_state.current_position_id)
    state_store.save(
        LiveStateStore.from_strategy_state(
            position_id=execution_state.current_position_id,
            symbol=trader_symbol,
            strategy_state=strategy.state,
            cash_before_position=execution_state.cash_before_position,
        )
    )
    logger.error(
        "%s | position_id=%s side=%s protective_sl_order_id=%s protective_sl_price=%s trend_runner_active=%s three_stage_tp2_consumed=%s trading_halted=true halt_reason=%s manual_intervention_required=true",
        event_name,
        execution_state.current_position_id,
        payload.get("side"),
        payload.get("protective_sl_order_id"),
        payload.get("protective_sl_price"),
        payload.get("trend_runner_active"),
        payload.get("three_stage_tp2_consumed"),
        execution_state.halt_reason,
    )


def apply_three_stage_startup_safety_gate(
    *,
    strategy: BollCvdReclaimStrategy,
    execution_state: live_runtime_types.ExecutionState,
    saved_state: Any,
    startup_position: PositionSnapshot,
    journal: LiveTradeJournal,
    state_store: LiveStateStore,
    trader_symbol: str,
) -> bool:
    if not startup_position.has_position:
        return False
    if saved_state is None:
        return False
    if not three_stage_dirty_post_tp1_sl_after_tp2(strategy.state):
        return False
    execution_state.trading_halted = True
    execution_state.halt_reason = THREE_STAGE_RESTART_DIRTY_HALT_REASON
    execution_state.halt_until_ts_ms = None
    append_three_stage_dirty_post_tp1_event(
        event_name="THREE_STAGE_TP1_PROTECTIVE_SL_CANCEL_FAILED_ON_TP2_RESTORED",
        strategy=strategy,
        execution_state=execution_state,
        journal=journal,
        state_store=state_store,
        trader_symbol=trader_symbol,
        reason="restart_restored_dirty_post_tp1_sl_after_tp2_manual_intervention_required",
    )
    return True


async def apply_sidecar_startup_recovery(
    *,
    strategy: BollCvdReclaimStrategy,
    execution_state: live_runtime_types.ExecutionState,
    saved_state: Any,
    startup_position: PositionSnapshot,
    trader: Trader,
    journal: LiveTradeJournal,
    state_store: LiveStateStore,
) -> None:
    if not startup_position.has_position:
        strategy.state.sidecar_enabled_for_position = False
        strategy.state.sidecar_legs = []
        refresh_sidecar_state_totals(strategy.state, int(os.getenv("SIDECAR_MAX_LEGS", "10")))
        state_store.clear()
        return
    saved_legs = list(getattr(saved_state, "sidecar_legs", []) or []) if saved_state is not None else []
    saved_sidecar_enabled = bool(getattr(saved_state, "sidecar_enabled_for_position", False)) if saved_state is not None else False
    if saved_sidecar_enabled:
        strategy.state.sidecar_enabled_for_position = True
        strategy.state.sidecar_margin_pct = float(getattr(saved_state, "sidecar_margin_pct", strategy.state.sidecar_margin_pct) or 0.0)
        strategy.state.sidecar_tp_pct = float(getattr(saved_state, "sidecar_tp_pct", strategy.state.sidecar_tp_pct) or 0.0)
    open_legs = [
        leg
        for leg in saved_legs
        if leg.get("status") in {SidecarLegStatus.OPEN.value, SidecarLegStatus.OPEN_UNPROTECTED.value}
    ]
    if not open_legs:
        refresh_sidecar_state_totals(strategy.state, int(os.getenv("SIDECAR_MAX_LEGS", "10")))
        if saved_sidecar_enabled:
            state_store.save(
                LiveStateStore.from_strategy_state(
                    position_id=execution_state.current_position_id,
                    symbol=trader.symbol,
                    strategy_state=strategy.state,
                    cash_before_position=execution_state.cash_before_position,
                )
            )
            return
        if not saved_legs and os.getenv("SIDECAR_ENABLED", "false").strip().lower() in {"1", "true", "yes", "y", "on"} and hasattr(journal, "append"):
            strategy.state.sidecar_enabled_for_position = False
            strategy.state.sidecar_margin_pct = 0.0
            strategy.state.sidecar_tp_pct = 0.0
            journal.append(
                "SIDECAR_DISABLED_FOR_RECOVERED_POSITION",
                {
                    "side": startup_position.side,
                    "okx_eth_qty": startup_position.eth_qty,
                    "reason": "startup_position_has_no_saved_sidecar_state",
                },
                position_id=execution_state.current_position_id,
            )
        return

    changed = False
    for index, leg in enumerate(list(strategy.state.sidecar_legs)):
        if leg.get("status") == SidecarLegStatus.OPEN_UNPROTECTED.value:
            execution_state.trading_halted = True
            execution_state.halt_reason = str(getattr(strategy.state, "sidecar_halt_reason", None) or "sidecar_tp_place_failed")
            strategy.state.sidecar_dirty = True
            strategy.state.sidecar_halt_reason = execution_state.halt_reason
            continue
        if leg.get("status") != SidecarLegStatus.OPEN.value:
            continue
        order_id = leg.get("tp_order_id")
        if not order_id:
            status = {"order_id": None, "status": "UNKNOWN"}
        else:
            status = await trader.fetch_sidecar_order_status(str(order_id))
        if status.get("status") == "OPEN":
            continue
        if status.get("status") == "FILLED":
            record_sidecar_tp_fill_exit(
                strategy.state,
                leg,
                status,
                fee_buffer_pct=getattr(getattr(strategy, "config", None), "breakeven_fee_buffer_pct", DEFAULT_NET_REMAINING_FEE_BUFFER_PCT),
            )
            strategy.state.sidecar_legs[index] = mark_sidecar_leg_tp_filled(leg, live_time_utils.utc_ms())
            if hasattr(journal, "append"):
                journal.append("SIDECAR_TP_FILLED", {**dict(leg), **status, "source": "startup_recovery"}, position_id=execution_state.current_position_id)
            changed = True
            continue
        execution_state.trading_halted = True
        execution_state.halt_reason = "sidecar_startup_order_state_unknown"
        strategy.state.sidecar_dirty = True
        strategy.state.sidecar_halt_reason = "sidecar_startup_order_state_unknown"
        strategy.state.sidecar_legs[index] = mark_sidecar_leg_unknown_halted(leg, live_time_utils.utc_ms())
        if hasattr(journal, "append"):
            journal.append(
                "SIDECAR_STARTUP_ORDER_STATE_UNKNOWN",
                {**dict(leg), **status, "manual_intervention_required": True},
                position_id=execution_state.current_position_id,
            )
        logger.error(
            "SIDECAR_STARTUP_ORDER_STATE_UNKNOWN | position_id=%s leg_id=%s order_id=%s status=%s trading_halted=true manual_intervention_required=true",
            execution_state.current_position_id,
            leg.get("leg_id"),
            order_id,
            status.get("status"),
        )
        changed = True
    refresh_sidecar_state_totals(strategy.state, int(os.getenv("SIDECAR_MAX_LEGS", "10")))
    if changed or getattr(strategy.state, "sidecar_enabled_for_position", False):
        state_store.save(
            LiveStateStore.from_strategy_state(
                position_id=execution_state.current_position_id,
                symbol=trader.symbol,
                strategy_state=strategy.state,
                cash_before_position=execution_state.cash_before_position,
            )
        )


async def apply_main_tp_startup_recovery(
    *,
    execution_state: live_runtime_types.ExecutionState,
    saved_state: Any,
    startup_position: PositionSnapshot,
    trader: Trader,
    journal: LiveTradeJournal,
) -> None:
    if not startup_position.has_position:
        return
    restored_tp_order_id = getattr(saved_state, "tp_order_id", None) if saved_state is not None else None
    restored_tp_order_ids = list(getattr(saved_state, "tp_order_ids", []) or []) if saved_state is not None else []
    if not restored_tp_order_id and restored_tp_order_ids:
        restored_tp_order_id = ",".join(str(item) for item in restored_tp_order_ids if item)
    if restored_tp_order_id:
        trader.tp_order_id = str(restored_tp_order_id)
        return
    try:
        pending_orders = await trader.fetch_pending_orders()
    except Exception as exc:
        execution_state.trading_halted = True
        execution_state.halt_reason = "main_tp_order_id_missing_on_startup"
        if hasattr(journal, "append"):
            journal.append(
                "MAIN_TP_ORDER_ID_MISSING_ON_STARTUP",
                {"reason": "pending_order_check_failed", "error": str(exc), "manual_intervention_required": True},
                position_id=execution_state.current_position_id,
            )
        logger.error("MAIN_TP_ORDER_ID_MISSING_ON_STARTUP | reason=pending_order_check_failed error=%s trading_halted=true manual_intervention_required=true", exc)
        return
    protected_sidecar_tp_ids = {
        str(leg.get("tp_order_id"))
        for leg in list(getattr(saved_state, "sidecar_legs", []) or [])
        if leg.get("status") == SidecarLegStatus.OPEN.value and leg.get("tp_order_id")
    } if saved_state is not None else set()
    reduce_only_orders = [
        item
        for item in pending_orders
        if item.get("instId") == trader.symbol and str(item.get("reduceOnly", "")).lower() == "true"
        and str(item.get("ordId")) not in protected_sidecar_tp_ids
    ]
    if reduce_only_orders:
        execution_state.trading_halted = True
        execution_state.halt_reason = "main_tp_order_id_missing_on_startup"
        if hasattr(journal, "append"):
            journal.append(
                "MAIN_TP_ORDER_ID_MISSING_ON_STARTUP",
                {
                    "pending_reduce_only_order_count": len(reduce_only_orders),
                    "pending_reduce_only_order_ids": [item.get("ordId") for item in reduce_only_orders],
                    "manual_intervention_required": True,
                },
                position_id=execution_state.current_position_id,
            )
        logger.error(
            "MAIN_TP_ORDER_ID_MISSING_ON_STARTUP | pending_reduce_only_order_count=%s trading_halted=true manual_intervention_required=true",
            len(reduce_only_orders),
        )
def queue_log_level(queue_size: int) -> int | None:
    if queue_size < 500:
        return None
    if queue_size < 2000:
        return logging.INFO
    if queue_size < 8000:
        return logging.WARNING
    return logging.ERROR


def queue_oldest_command_age_seconds(queue: asyncio.Queue[live_runtime_types.TradeCommand]) -> float:
    try:
        oldest = queue._queue[0]  # type: ignore[attr-defined]
    except Exception:
        return 0.0
    return max(time.monotonic() - oldest.created_monotonic, 0.0)


async def enqueue_strategy_tick(
    event: MarketTickEvent,
    strategy_tick_queue: asyncio.Queue[MarketTickEvent],
    state_lock: asyncio.Lock,
    execution_state: live_runtime_types.ExecutionState,
) -> None:
    if event.boll is None:
        return
    try:
        strategy_tick_queue.put_nowait(event)
    except asyncio.QueueFull:
        logger.error(
            "STRATEGY_TICK_QUEUE_FULL | price=%.4f tick_ts_ms=%s queue_size=%s",
            event.tick.price,
            event.tick.ts_ms,
            strategy_tick_queue.qsize(),
        )
        async with state_lock:
            execution_state.trading_halted = True


async def enqueue_execution_command(
    command: live_runtime_types.TradeCommand,
    execution_queue: asyncio.Queue[live_runtime_types.TradeCommand],
    state_lock: asyncio.Lock,
    execution_state: live_runtime_types.ExecutionState,
) -> bool:
    async with state_lock:
        if execution_queue.full():
            logger.error(
                "EXECUTION_QUEUE_FULL | intent_type=%s side=%s tick_ts_ms=%s queue_size=%s",
                command.intent.intent_type,
                command.intent.side,
                command.tick_ts_ms,
                execution_queue.qsize(),
            )
            execution_state.trading_halted = True
            return False
        execution_state.pending_order_count += 1
        try:
            execution_queue.put_nowait(command)
        except asyncio.QueueFull:
            execution_state.pending_order_count = max(execution_state.pending_order_count - 1, 0)
            execution_state.trading_halted = True
            logger.error(
                "EXECUTION_QUEUE_FULL | intent_type=%s side=%s tick_ts_ms=%s queue_size=%s",
                command.intent.intent_type,
                command.intent.side,
                command.tick_ts_ms,
                execution_queue.qsize(),
            )
            return False
    return True


async def strategy_tick_worker(
    *,
    strategy_tick_queue: asyncio.Queue[MarketTickEvent],
    execution_queue: asyncio.Queue[live_runtime_types.TradeCommand],
    state_lock: asyncio.Lock,
    account_snapshot: live_runtime_types.AccountSnapshot,
    execution_state: live_runtime_types.ExecutionState,
    cvd: CvdTracker,
    strategy: BollCvdShockReclaimStrategy,
    heartbeat_seconds: float,
    account_stale_warn_seconds: float,
    strategy_lag_warn_seconds: float,
) -> None:
    last_heartbeat = 0.0
    last_lag_log = 0.0
    last_account_stale_log = 0.0
    latest_tick_ts_ms = 0
    while True:
        event = await strategy_tick_queue.get()
        try:
            if event.boll is None:
                continue
            latest_tick_ts_ms = max(latest_tick_ts_ms, event.tick.ts_ms)
            async with state_lock:
                account_snapshot.latest_market_price = event.tick.price
                account_snapshot.latest_market_price_ts_ms = event.tick.ts_ms
            now = time.monotonic()
            tick_lag_seconds = max(time.time() - event.tick.ts_ms / 1000, 0.0)
            queue_size = strategy_tick_queue.qsize()
            level = queue_log_level(queue_size)
            if (level is not None or tick_lag_seconds >= strategy_lag_warn_seconds) and now - last_lag_log >= 30:
                logger.log(
                    level or logging.WARNING,
                    "STRATEGY_TICK_LAG | tick_lag_seconds=%.3f strategy_queue_size=%s latest_tick_ts_ms=%s processed_tick_ts_ms=%s",
                    tick_lag_seconds,
                    queue_size,
                    latest_tick_ts_ms,
                    event.tick.ts_ms,
                )
                last_lag_log = now

            account_age_seconds = max(now - account_snapshot.updated_monotonic, 0.0) if account_snapshot.updated_monotonic > 0 else float("inf")
            if account_age_seconds >= account_stale_warn_seconds and now - last_account_stale_log >= 60:
                logger.warning(
                    "ACCOUNT_SNAPSHOT_STALE | age_seconds=%.1f threshold=%.1f",
                    account_age_seconds,
                    account_stale_warn_seconds,
                )
                last_account_stale_log = now

            cvd_snapshot = cvd.update(
                side=event.tick.side,
                size=event.tick.size,
                price=event.tick.price,
                ts_ms=event.tick.ts_ms,
            )
            if now - last_heartbeat >= heartbeat_seconds:
                last_heartbeat = now
                logger.info(
                    "MARKET_TICK_HEARTBEAT | price=%.4f tick_ts_ms=%s side=%s size=%.8f boll_lower=%.4f boll_middle=%.4f boll_upper=%.4f switch=%s fast_cvd=%.8f previous_fast_cvd=%.8f buy_ratio=%.4f sell_ratio=%.4f burst_net_move_pct=%.6f move_ratio=%.2f volume_ratio=%.2f burst_range_pct=%.6f baseline_range_pct=%.6f burst_volume=%.8f baseline_volume=%.8f up_burst=%s down_burst=%s",
                    event.tick.price,
                    event.tick.ts_ms,
                    event.tick.side,
                    event.tick.size,
                    event.boll.lower,
                    event.boll.middle,
                    event.boll.upper,
                    event.boll.alert_switch_on,
                    cvd_snapshot.fast_cvd,
                    cvd_snapshot.previous_fast_cvd,
                    cvd_snapshot.buy_ratio,
                    cvd_snapshot.sell_ratio,
                    cvd_snapshot.burst_net_move_pct,
                    cvd_snapshot.burst_move_ratio,
                    cvd_snapshot.burst_volume_ratio,
                    cvd_snapshot.burst_range_pct,
                    cvd_snapshot.baseline_range_pct,
                    cvd_snapshot.burst_volume,
                    cvd_snapshot.baseline_volume,
                    cvd_snapshot.up_burst,
                    cvd_snapshot.down_burst,
                )

            async with state_lock:
                trading_halted = execution_state.trading_halted
                halt_reason = execution_state.halt_reason
                pending_order_count = execution_state.pending_order_count
                has_position = bool(account_snapshot.position and account_snapshot.position.has_position)
            allow_position_management_only = (
                trading_halted
                and halt_reason in ROLLING_LOSS_HALT_REASONS
                and has_position
            )
            if pending_order_count > 0:
                continue
            if trading_halted and not allow_position_management_only:
                continue

            backup_state = copy.deepcopy(strategy.state)
            intents = strategy.on_tick(
                price=event.tick.price,
                ts_ms=event.tick.ts_ms,
                boll=event.boll,
                cvd=cvd_snapshot,
            )
            if allow_position_management_only:
                intents = [intent for intent in intents if intent.intent_type in POSITION_MANAGEMENT_INTENTS]
            for intent in intents:
                if getattr(strategy.state, "sidecar_enabled_for_position", False):
                    intent = with_runtime_managed_core(intent, account_snapshot.position)
                command = live_runtime_types.TradeCommand(
                    intent=intent,
                    strategy_state_snapshot=backup_state,
                    tick_ts_ms=event.tick.ts_ms,
                    created_monotonic=time.monotonic(),
                    account_snapshot_updated_ts_ms=account_snapshot.updated_ts_ms,
                    reason=intent.reason,
                )
                ok = await enqueue_execution_command(command, execution_queue, state_lock, execution_state)
                if not ok:
                    async with state_lock:
                        strategy.state = backup_state
                    break
        except Exception:
            logger.exception("Strategy tick worker failed")
        finally:
            strategy_tick_queue.task_done()


def with_entry_add_managed_core_contracts(
    *,
    intent: TradeIntent,
    strategy_state: StrategyPositionState,
    account_core_position: PositionSnapshot | None,
    trader: Trader,
) -> TradeIntent:
    if not strategy_state.sidecar_enabled_for_position:
        return intent
    if intent.intent_type not in {"OPEN_LONG", "OPEN_SHORT", "ADD_LONG", "ADD_SHORT"}:
        return intent
    if intent.managed_core_contracts:
        return intent

    current_core_contracts = Decimal("0")
    current_core_eth_qty = 0.0

    if account_core_position is not None and account_core_position.has_position and account_core_position.side == intent.side:
        current_core_contracts = account_core_position.contracts
        current_core_eth_qty = account_core_position.eth_qty

    new_core_contracts = trader.eth_qty_to_contracts(Decimal(str(intent.size.eth_qty)))
    expected_core_contracts = current_core_contracts + new_core_contracts

    return replace(
        intent,
        managed_core_contracts=str(expected_core_contracts),
        managed_core_eth_qty=current_core_eth_qty + float(intent.size.eth_qty),
    )


async def execution_worker(
    *,
    execution_queue: asyncio.Queue[live_runtime_types.TradeCommand],
    state_lock: asyncio.Lock,
    execution_state: live_runtime_types.ExecutionState,
    account_snapshot: live_runtime_types.AccountSnapshot,
    trader: Trader,
    strategy: BollCvdShockReclaimStrategy,
    journal: LiveTradeJournal,
    state_store: LiveStateStore,
    email_sender: EmailSender,
    backlog_log_seconds: float,
) -> None:
    last_backlog_log = 0.0
    while True:
        command = await execution_queue.get()
        result = None
        try:
            queue_size = execution_queue.qsize()
            level = queue_log_level(queue_size)
            now = time.monotonic()
            if level is not None and now - last_backlog_log >= backlog_log_seconds:
                logger.log(
                    level,
                    "EXECUTION_QUEUE_BACKLOG | queue_size=%s maxsize=%s oldest_command_age_seconds=%.3f",
                    queue_size,
                    execution_queue.maxsize,
                    queue_oldest_command_age_seconds(execution_queue),
                )
                last_backlog_log = now

            dirty_post_tp1_sl_blocked = False
            dirty_post_tp1_sl_should_record = False
            async with state_lock:
                if three_stage_dirty_post_tp1_sl_after_tp2(strategy.state):
                    dirty_post_tp1_sl_blocked = True
                    dirty_post_tp1_sl_should_record = not (
                        execution_state.trading_halted
                        and execution_state.halt_reason == THREE_STAGE_RUNTIME_DIRTY_HALT_REASON
                    )
                    execution_state.trading_halted = True
                    execution_state.halt_reason = THREE_STAGE_RUNTIME_DIRTY_HALT_REASON
                    execution_state.halt_until_ts_ms = None
            if dirty_post_tp1_sl_blocked:
                if dirty_post_tp1_sl_should_record:
                    append_three_stage_dirty_post_tp1_event(
                        event_name="THREE_STAGE_DIRTY_POST_TP1_SL_BLOCKED_RUNNER_UPDATE",
                        strategy=strategy,
                        execution_state=execution_state,
                        journal=journal,
                        state_store=state_store,
                        trader_symbol=trader.symbol,
                        reason="dirty_post_tp1_sl_after_tp2_blocks_runner_update_manual_intervention_required",
                    )
                logger.warning(
                    "EXECUTION_SKIPPED | reason=three_stage_dirty_post_tp1_sl intent_type=%s side=%s tick_ts_ms=%s",
                    command.intent.intent_type,
                    command.intent.side,
                    command.tick_ts_ms,
                )
                continue

            async with state_lock:
                rolling_management_allowed = (
                    execution_state.trading_halted
                    and execution_state.halt_reason in ROLLING_LOSS_HALT_REASONS
                    and command.intent.intent_type in POSITION_MANAGEMENT_INTENTS
                )
                if execution_state.trading_halted and not rolling_management_allowed:
                    logger.warning(
                        "EXECUTION_SKIPPED | reason=trading_halted intent_type=%s side=%s tick_ts_ms=%s",
                        command.intent.intent_type,
                        command.intent.side,
                        command.tick_ts_ms,
                    )
                    continue
                current_position_id = execution_state.current_position_id
                cash_before_position = execution_state.cash_before_position

            entry_cash_before = cash_before_position
            if command.intent.intent_type != "UPDATE_TP" and current_position_id is None:
                entry_cash_before = await fetch_usdt_cash_balance(trader)

            if command.intent.intent_type in {"ADD_LONG", "ADD_SHORT"} and getattr(command.strategy_state_snapshot, "tp_plan", "SINGLE") in SPLIT_TP_PLANS:
                position = await trader.fetch_position_snapshot()
                if position.has_position and position.side == command.intent.side:
                    consumed = False
                    async with state_lock:
                        current_strategy_state = copy.deepcopy(strategy.state)
                        strategy.state = copy.deepcopy(command.strategy_state_snapshot)
                        consumed = mark_partial_tp_consumed_if_position_reduced(strategy, position)
                        if consumed:
                            sync_strategy_cost_from_position(strategy, position)
                            current_position_id = execution_state.current_position_id
                            cash_before_position = execution_state.cash_before_position
                            strategy_state_for_save = copy.deepcopy(strategy.state)
                            account_snapshot.position = position
                            trader.position_contracts = position.contracts
                        else:
                            strategy.state = current_strategy_state
                    if consumed:
                        state_store.save(
                            LiveStateStore.from_strategy_state(
                                position_id=current_position_id,
                                symbol=trader.symbol,
                                strategy_state=strategy_state_for_save,
                                cash_before_position=cash_before_position,
                            )
                        )
                        logger.warning(
                            "EXECUTION_SKIPPED | reason=partial_tp_consumed_before_add stale_add_command_skipped intent_type=%s side=%s layer=%s okx_eth_qty=%.8f strategy_eth_qty=%.8f tp_plan=%s",
                            command.intent.intent_type,
                            command.intent.side,
                            command.intent.layer_index,
                            position.eth_qty,
                            strategy_state_for_save.total_entry_qty,
                            strategy_state_for_save.tp_plan,
                        )
                        continue

            entry_intent = with_entry_add_managed_core_contracts(
                intent=command.intent,
                strategy_state=strategy.state,
                account_core_position=account_snapshot.position,
                trader=trader,
            )
            if entry_intent is not command.intent:
                command = replace(command, intent=entry_intent)

            # Guard: Sidecar enabled position must never execute NEAR_TP_REDUCE
            if command.intent.intent_type == "NEAR_TP_REDUCE" and getattr(strategy.state, "sidecar_enabled_for_position", False):
                logger.error(
                    "SIDECAR_BLOCKS_NEAR_TP_REDUCE | sidecar_enabled_for_position=True; NEAR_TP_REDUCE would reduce sidecar portion of OKX net position trading_halted=true halt_reason=sidecar_blocks_near_tp_reduce",
                )
                async with state_lock:
                    execution_state.trading_halted = True
                    execution_state.halt_reason = "sidecar_blocks_near_tp_reduce"
                    strategy.state.sidecar_dirty = True
                    strategy.state.sidecar_halt_reason = "sidecar_blocks_near_tp_reduce"
                    current_position_id = execution_state.current_position_id
                    cash_before_position = execution_state.cash_before_position
                if hasattr(journal, "append"):
                    journal.append(
                        "SIDECAR_BLOCKS_NEAR_TP_REDUCE",
                        {
                            "sidecar_enabled_for_position": True,
                            "trading_halted": True,
                            "halt_reason": "sidecar_blocks_near_tp_reduce",
                            "intent_type": command.intent.intent_type,
                            "side": command.intent.side,
                            "manual_intervention_required": True,
                        },
                        position_id=current_position_id,
                    )
                state_store.save(
                    LiveStateStore.from_strategy_state(
                        position_id=current_position_id,
                        symbol=trader.symbol,
                        strategy_state=strategy.state,
                        cash_before_position=cash_before_position,
                    )
                )
                continue

            sidecar_plan: SidecarExecutionPlan | None = None
            if command.intent.intent_type in {"OPEN_LONG", "OPEN_SHORT", "ADD_LONG", "ADD_SHORT"}:
                async with state_lock:
                    if execution_state.current_position_id is None:
                        execution_state.current_position_id = journal.new_position_id(
                            trader.symbol,
                            command.intent.side,
                            command.intent.ts_ms,
                        )
                        execution_state.cash_before_position = entry_cash_before
                    current_position_id = execution_state.current_position_id
                combined_plan = build_combined_entry_intent(
                    intent=command.intent,
                    sidecar_enabled=bool(getattr(strategy.state, "sidecar_enabled_for_position", False)),
                    account_equity_usdt=float(trader.account_equity_usdt),
                    leverage=float(getattr(trader, "leverage", getattr(getattr(trader, "config", None), "leverage", 50)) or 50),
                    sidecar_margin_pct=float(getattr(strategy.state, "sidecar_margin_pct", 0.0) or 0.0),
                    sidecar_tp_pct=float(getattr(strategy.state, "sidecar_tp_pct", 0.0) or 0.0),
                    position_id=current_position_id,
                    contract_multiplier=getattr(trader, "contract_multiplier", Decimal("0.1")),
                    contract_precision=getattr(trader, "contract_precision", Decimal("0.01")),
                )
                command = replace(command, intent=combined_plan.execution_intent)
                sidecar_plan = combined_plan.sidecar_plan

            result = await trader.execute_intent(command.intent)
            if not result.ok:
                raise RuntimeError(result.message)

            if command.intent.intent_type == "UPDATE_TP":
                async with state_lock:
                    current_position_id = execution_state.current_position_id
                    cash_before_position = execution_state.cash_before_position
                    execution_state.last_order_ts_ms = command.intent.ts_ms
                    if getattr(command.intent, "middle_runner_active", False):
                        if getattr(result, "protective_sl_order_id", None):
                            strategy.state.middle_runner_protective_sl_order_id = result.protective_sl_order_id
                        if live_config_helpers._parse_optional_float(getattr(result, "protective_sl_price", "")) is not None:
                            strategy.state.middle_runner_protective_sl_price = live_config_helpers._parse_optional_float(result.protective_sl_price)
                    if getattr(command.intent, "trend_runner_active", False):
                        if getattr(result, "protective_sl_order_id", None):
                            strategy.state.trend_runner_sl_order_id = result.protective_sl_order_id
                        if live_config_helpers._parse_optional_float(getattr(result, "protective_sl_price", "")) is not None:
                            strategy.state.trend_runner_sl_price = live_config_helpers._parse_optional_float(result.protective_sl_price)
                        strategy.state.trend_runner_tp_order_id = result.tp_order_id
                    strategy.state.tp_order_id = result.tp_order_id
                    strategy.state.tp_order_ids = list(getattr(result, "tp_order_ids", ()) or [])
                    if getattr(command.intent, "three_stage_post_tp1_protective_sl_price", None) is not None and getattr(command.intent, "three_stage_tp1_consumed", False):
                        if getattr(result, "protective_sl_order_id", None):
                            strategy.state.three_stage_post_tp1_protective_sl_order_id = result.protective_sl_order_id
                        if live_config_helpers._parse_optional_float(getattr(result, "protective_sl_price", "")) is not None:
                            strategy.state.three_stage_post_tp1_protective_sl_price = live_config_helpers._parse_optional_float(result.protective_sl_price)
                        strategy.state.three_stage_post_tp1_protected = bool(getattr(result, "protective_sl_ok", False))
                    strategy_state_for_save = copy.deepcopy(strategy.state)
                    equity = account_snapshot.equity
                journal.record_tp_update(position_id=current_position_id, intent=command.intent, result=result, equity=equity)
                if (
                    (getattr(command.intent, "middle_runner_active", False) or getattr(command.intent, "middle_runner_pending", False))
                    and hasattr(journal, "append")
                ):
                    journal.append(
                        "MIDDLE_RUNNER_TP_UPDATED",
                        {
                            "side": command.intent.side,
                            "first_tp_price": getattr(command.intent, "partial_tp_price", None),
                            "final_tp_price": command.intent.tp_price,
                            "protective_sl_price": getattr(result, "protective_sl_price", "") or getattr(command.intent, "middle_runner_protective_sl_price", None),
                            "protective_sl_order_id": getattr(result, "protective_sl_order_id", None),
                            "boll_lower": command.intent.boll_lower,
                            "boll_middle": command.intent.boll_middle,
                            "boll_upper": command.intent.boll_upper,
                            "reason": command.intent.reason,
                        },
                        position_id=current_position_id,
                    )
                if getattr(command.intent, "trend_runner_active", False) and hasattr(journal, "append"):
                    journal.append(
                        "TREND_RUNNER_UPDATE",
                        {
                            "side": command.intent.side,
                            "tp_plan": "THREE_STAGE_RUNNER",
                            "runner_tp_price": getattr(command.intent, "trend_runner_tp_price", None) or command.intent.tp_price,
                            "runner_sl_price": getattr(result, "protective_sl_price", "") or getattr(command.intent, "trend_runner_sl_price", None),
                            "runner_sl_order_id": getattr(result, "protective_sl_order_id", None),
                            "trend_runner_active": True,
                            "trend_runner_adjust_count": getattr(command.intent, "trend_runner_adjust_count", 0),
                            "boll_lower": command.intent.boll_lower,
                            "boll_middle": command.intent.boll_middle,
                            "boll_upper": command.intent.boll_upper,
                            "reason": command.intent.reason,
                        },
                        position_id=current_position_id,
                    )
                if (
                    getattr(command.intent, "three_stage_post_tp1_protective_sl_price", None) is not None
                    and getattr(command.intent, "three_stage_tp1_consumed", False)
                    and not getattr(command.intent, "trend_runner_active", False)
                    and hasattr(journal, "append")
                ):
                    journal.append(
                        "THREE_STAGE_TP1_PROTECTIVE_SL_UPDATED",
                        {
                            "side": command.intent.side,
                            "contracts": result.contracts,
                            "protective_sl_price": getattr(result, "protective_sl_price", "") or getattr(command.intent, "three_stage_post_tp1_protective_sl_price", None),
                            "protective_sl_order_id": getattr(result, "protective_sl_order_id", None),
                            "old_protective_sl_order_id": getattr(command.intent, "three_stage_post_tp1_protective_sl_order_id", None),
                            "avg_entry_price": command.intent.avg_entry_price,
                            "tp1_price": getattr(command.intent, "three_stage_tp1_price", None),
                            "tp1_ratio": getattr(command.intent, "three_stage_tp1_ratio", 0.0),
                            "tp2_price": getattr(command.intent, "three_stage_tp2_price", None),
                            "tp2_ratio": getattr(command.intent, "three_stage_tp2_ratio", 0.0),
                            "runner_ratio": getattr(command.intent, "three_stage_runner_ratio", 0.0),
                            "reason": command.intent.reason,
                            "retry_config": "NEAR_TP_PROTECTIVE_SL_RETRY_COUNT/NEAR_TP_PROTECTIVE_SL_RETRY_INTERVAL_SECONDS",
                        },
                        position_id=current_position_id,
                    )
                state_store.save(LiveStateStore.from_strategy_state(position_id=current_position_id, symbol=trader.symbol, strategy_state=strategy_state_for_save, cash_before_position=cash_before_position))
                logger.warning(
                    "LIVE TP update success | side=%s layer=%s price=%.4f contracts=%s tp_price=%s tp_mode=%s tp_plan=%s partial_tp=%s avg_entry=%.4f breakeven=%.4f tp_order_id=%s",
                    command.intent.side,
                    command.intent.layer_index,
                    command.intent.price,
                    result.contracts,
                    result.tp_price,
                    command.intent.tp_mode,
                    getattr(command.intent, "tp_plan", "SINGLE"),
                    getattr(command.intent, "partial_tp_price", None),
                    command.intent.avg_entry_price,
                    command.intent.breakeven_price,
                    result.tp_order_id,
                )
            elif command.intent.intent_type == "NEAR_TP_REDUCE":
                fail_action = None
                if not getattr(result, "protective_sl_ok", False) and getattr(result, "near_tp_exit_all", False):
                    fail_action = "MARKET_EXIT"
                remaining_position: PositionSnapshot | None = None
                remaining_position_sync_error: str | None = None
                if getattr(result, "protective_sl_ok", False) and not getattr(result, "near_tp_exit_all", False):
                    try:
                        position = await trader.fetch_position_snapshot()
                        if position.has_position and position.side == command.intent.side:
                            remaining_position = position
                        else:
                            remaining_position_sync_error = f"position_absent_or_side_mismatch has_position={position.has_position} side={position.side}"
                    except Exception:
                        remaining_position_sync_error = "fetch_position_failed"
                        logger.exception("NEAR_TP_STATE_PROTECTED | failed_to_sync_remaining_position_before_save")
                async with state_lock:
                    current_position_id = execution_state.current_position_id
                    cash_before_position = execution_state.cash_before_position
                    execution_state.last_order_ts_ms = command.intent.ts_ms
                    strategy.state.tp_order_id = result.tp_order_id
                    strategy.state.tp_order_ids = list(getattr(result, "tp_order_ids", ()) or [])
                    near_tp_state_synced = False
                    if getattr(result, "near_tp_exit_all", False):
                        execution_state.trading_halted = True
                        execution_state.halt_reason = "near_tp_exit_all_waiting_flat"
                    elif getattr(result, "protective_sl_ok", False):
                        if remaining_position is not None:
                            sync_strategy_cost_from_position(strategy, remaining_position)
                            account_snapshot.position = remaining_position
                            trader.position_contracts = remaining_position.contracts
                            near_tp_state_synced = True
                        else:
                            execution_state.trading_halted = True
                            execution_state.halt_reason = "near_tp_protected_sync_failed"
                            logger.warning(
                                "NEAR_TP_STATE_PROTECTED_SYNC_FAILED | position_id=%s reason=%s trading_halted=true",
                                current_position_id,
                                remaining_position_sync_error or "unknown",
                            )
                        strategy.state.near_tp_protected = True
                        strategy.state.near_tp_reduce_pending = False
                        strategy_config = getattr(strategy, "config", None)
                        strategy.state.near_tp_add_disabled = bool(getattr(strategy_config, "near_tp_disable_add_after_reduce", True))
                        strategy.state.near_tp_protective_sl_price = getattr(command.intent, "near_tp_protective_sl_price", None) or live_config_helpers._parse_optional_float(getattr(result, "protective_sl_price", ""))
                        strategy.state.near_tp_protective_sl_order_id = getattr(result, "protective_sl_order_id", None)
                        strategy.state.tp_plan = "SINGLE"
                        strategy.state.partial_tp_price = None
                        strategy.state.partial_tp_ratio = 0.0
                        strategy.state.partial_tp_consumed = True
                        logger.warning(
                            "NEAR_TP_STATE_PROTECTED | position_id=%s protective_sl_order_id=%s protective_sl_price=%s",
                            current_position_id,
                            strategy.state.near_tp_protective_sl_order_id,
                            strategy.state.near_tp_protective_sl_price,
                        )
                    strategy_state_for_save = copy.deepcopy(strategy.state)
                    equity = account_snapshot.equity
                journal.record_near_tp_reduce(
                    position_id=current_position_id,
                    symbol=trader.symbol,
                    intent=command.intent,
                    result=result,
                    protective_sl_fail_action=fail_action,
                )
                if getattr(result, "protective_sl_ok", False) and not getattr(result, "near_tp_exit_all", False):
                    state_store.save(
                        LiveStateStore.from_strategy_state(
                            position_id=current_position_id,
                            symbol=trader.symbol,
                            strategy_state=strategy_state_for_save,
                            cash_before_position=cash_before_position,
                        )
                    )
                    if not near_tp_state_synced:
                        subject = "Near-TP protected but position sync failed"
                        content = (
                            "<div style='font-family:Arial,Helvetica,sans-serif;line-height:1.55;'>"
                            "<h2>Near-TP protected but position sync failed</h2>"
                            "<p>Reduce succeeded and protective SL was placed. Protected state was saved.</p>"
                            "<p>Trading is temporarily halted until account sync refreshes the position.</p>"
                            f"<p><b>position_id:</b> {html.escape(str(current_position_id))}</p>"
                            f"<p><b>protective_sl_order_id:</b> {html.escape(str(getattr(result, 'protective_sl_order_id', None)))}</p>"
                            f"<p><b>reason:</b> {html.escape(str(remaining_position_sync_error or 'unknown'))}</p>"
                            "</div>"
                        )
                        ok = await email_sender.send_email_async(subject, content, content_type="html")
                        if not ok:
                            logger.error("Failed to send Near-TP protected sync failure email")
                if fail_action == "MARKET_EXIT":
                    subject = "Near-TP protective SL failed; market-exited remaining position"
                    content = (
                        "<div style='font-family:Arial,Helvetica,sans-serif;line-height:1.55;'>"
                        "<h2>Near-TP protective SL failed</h2>"
                        f"<p>Remaining position was market-exited successfully. Trading is temporarily halted until account sync records FLAT.</p>"
                        f"<p><b>position_id:</b> {html.escape(str(current_position_id))}</p>"
                        f"<p><b>message:</b> {html.escape(result.message)}</p>"
                        "</div>"
                    )
                    ok = await email_sender.send_email_async(subject, content, content_type="html")
                    if not ok:
                        logger.error("Failed to send Near-TP market-exit success email")
                logger.warning(
                    "LIVE Near-TP reduce success | side=%s layer=%s price=%.4f contracts_before=%s contracts_reduced=%s contracts_after=%s tp_order_id=%s protective_sl_ok=%s protective_sl_order_id=%s near_tp_exit_all=%s equity=%.4f",
                    command.intent.side,
                    command.intent.layer_index,
                    command.intent.price,
                    result.contracts_before,
                    result.contracts_reduced,
                    result.contracts_after,
                    result.tp_order_id,
                    result.protective_sl_ok,
                    result.protective_sl_order_id,
                    result.near_tp_exit_all,
                    equity or 0.0,
                )
            elif command.intent.intent_type == "MARKET_EXIT_RUNNER":
                async with state_lock:
                    current_position_id = execution_state.current_position_id
                    execution_state.last_order_ts_ms = command.intent.ts_ms
                    execution_state.trading_halted = True
                    execution_state.halt_reason = "trend_runner_market_exit_waiting_flat"
                    strategy.state.trend_runner_exit_reason = getattr(command.intent, "trend_runner_exit_reason", None) or command.intent.reason
                    strategy_state_for_save = copy.deepcopy(strategy.state)
                    cash_before_position = execution_state.cash_before_position
                journal.record_trend_runner_market_exit(
                    position_id=current_position_id,
                    symbol=trader.symbol,
                    intent=command.intent,
                    result=result,
                )
                state_store.save(
                    LiveStateStore.from_strategy_state(
                        position_id=current_position_id,
                        symbol=trader.symbol,
                        strategy_state=strategy_state_for_save,
                        cash_before_position=cash_before_position,
                    )
                )
                logger.warning(
                    "LIVE Trend Runner market exit success | side=%s reason=%s contracts_before=%s contracts_after=%s message=%s",
                    command.intent.side,
                    command.intent.reason,
                    result.contracts_before,
                    result.contracts_after,
                    result.message,
                )
            else:
                new_position_id = None
                async with state_lock:
                    if execution_state.current_position_id is None:
                        new_position_id = journal.new_position_id(trader.symbol, command.intent.side, command.intent.ts_ms)
                        execution_state.current_position_id = new_position_id
                        execution_state.cash_before_position = entry_cash_before
                    current_position_id = execution_state.current_position_id
                    cash_before_position = execution_state.cash_before_position
                    execution_state.last_order_ts_ms = command.intent.ts_ms
                    strategy.state.tp_order_id = result.tp_order_id
                    strategy.state.tp_order_ids = list(getattr(result, "tp_order_ids", ()) or [])
                    strategy_state_for_save = copy.deepcopy(strategy.state)
                    equity = account_snapshot.equity
                journal.record_entry(
                    position_id=current_position_id or new_position_id or "",
                    intent=command.intent,
                    result=result,
                    cash_before_position=cash_before_position,
                    equity=equity,
                    extra={"symbol": trader.symbol},
                )
                sidecar_ok = True
                if sidecar_plan is not None:
                    sidecar_ok = await attach_sidecar_after_combined_entry(
                        trader=trader,
                        strategy_state=strategy.state,
                        execution_state=execution_state,
                        intent=command.intent,
                        sidecar_plan=sidecar_plan,
                        journal=journal,
                        state_store=state_store,
                        trader_symbol=trader.symbol,
                        fee_buffer_pct=strategy.config.breakeven_fee_buffer_pct,
                    )
                async with state_lock:
                    strategy_state_for_save = copy.deepcopy(strategy.state)
                if not sidecar_ok:
                    logger.error(
                        "LIVE sidecar failed after core entry | position_id=%s intent_type=%s side=%s layer=%s trading_halted=true halt_reason=%s",
                        current_position_id or new_position_id,
                        command.intent.intent_type,
                        command.intent.side,
                        command.intent.layer_index,
                        execution_state.halt_reason,
                    )
                if getattr(command.intent, "tp_plan", "SINGLE") == "MIDDLE_RUNNER" and hasattr(journal, "append"):
                    journal.append(
                        "MIDDLE_RUNNER_PLANNED",
                        {
                            "side": command.intent.side,
                            "layers": command.intent.layer_index,
                            "avg_entry_price": command.intent.avg_entry_price,
                            "first_tp_price": getattr(command.intent, "partial_tp_price", None),
                            "final_tp_price": command.intent.tp_price,
                            "first_close_ratio": getattr(command.intent, "partial_tp_ratio", 0.0),
                            "keep_ratio": getattr(command.intent, "middle_runner_keep_ratio", 0.0),
                            "boll_lower": command.intent.boll_lower,
                            "boll_middle": command.intent.boll_middle,
                            "boll_upper": command.intent.boll_upper,
                        },
                        position_id=current_position_id or new_position_id or "",
                    )
                if getattr(command.intent, "tp_plan", "SINGLE") == "THREE_STAGE_RUNNER" and hasattr(journal, "append"):
                    journal.append(
                        "THREE_STAGE_RUNNER_PLANNED",
                        {
                            "side": command.intent.side,
                            "layers": command.intent.layer_index,
                            "avg_entry_price": command.intent.avg_entry_price,
                            "tp_plan": "THREE_STAGE_RUNNER",
                            "tp1_price": getattr(command.intent, "three_stage_tp1_price", None),
                            "tp1_ratio": getattr(command.intent, "three_stage_tp1_ratio", 0.0),
                            "tp2_price": getattr(command.intent, "three_stage_tp2_price", None),
                            "tp2_ratio": getattr(command.intent, "three_stage_tp2_ratio", 0.0),
                            "runner_tp_price": getattr(command.intent, "three_stage_runner_tp_price", None),
                            "runner_sl_price": getattr(command.intent, "three_stage_runner_sl_price", None),
                            "runner_ratio": getattr(command.intent, "three_stage_runner_ratio", 0.0),
                            "runner_sl_order_id": getattr(result, "protective_sl_order_id", None),
                            "boll_lower": command.intent.boll_lower,
                            "boll_middle": command.intent.boll_middle,
                            "boll_upper": command.intent.boll_upper,
                        },
                        position_id=current_position_id or new_position_id or "",
                    )
                state_store.save(LiveStateStore.from_strategy_state(position_id=current_position_id, symbol=trader.symbol, strategy_state=strategy_state_for_save, cash_before_position=cash_before_position))
                logger.warning(
                    "LIVE entry success | intent_type=%s side=%s layer=%s price=%.4f contracts=%s tp_price=%s tp_mode=%s tp_plan=%s partial_tp=%s avg_entry=%.4f breakeven=%.4f order_id=%s tp_order_id=%s",
                    command.intent.intent_type,
                    command.intent.side,
                    command.intent.layer_index,
                    command.intent.price,
                    result.contracts,
                    result.tp_price,
                    command.intent.tp_mode,
                    getattr(command.intent, "tp_plan", "SINGLE"),
                    getattr(command.intent, "partial_tp_price", None),
                    command.intent.avg_entry_price,
                    command.intent.breakeven_price,
                    result.order_id,
                    result.tp_order_id,
                )
        except Exception as exc:
            await handle_execution_failure(
                command=command,
                result=result,
                error=exc,
                state_lock=state_lock,
                execution_state=execution_state,
                trader=trader,
                strategy=strategy,
                journal=journal,
                email_sender=email_sender,
            )
        finally:
            async with state_lock:
                execution_state.pending_order_count = max(execution_state.pending_order_count - 1, 0)
            execution_queue.task_done()


async def handle_execution_failure(
    *,
    command: live_runtime_types.TradeCommand,
    result: Any | None,
    error: Exception,
    state_lock: asyncio.Lock,
    execution_state: live_runtime_types.ExecutionState,
    trader: Trader,
    strategy: BollCvdShockReclaimStrategy,
    journal: LiveTradeJournal,
    email_sender: EmailSender,
) -> live_runtime_types.ExecutionReport:
    contracts = trader.position_contracts
    if result is None or getattr(result, "entry_filled", False) or getattr(result, "reduce_filled", False):
        try:
            position = await trader.fetch_position_snapshot()
            contracts = position.contracts
        except Exception:
            contracts = trader.position_contracts

    entry_may_be_live = bool(getattr(result, "entry_filled", False)) or bool(getattr(result, "reduce_filled", False)) or contracts > 0
    rolled_back = False
    async with state_lock:
        current_position_id = execution_state.current_position_id
        if entry_may_be_live:
            execution_state.trading_halted = True
            if str(error) == "reduce_only_order_identity_unknown":
                execution_state.halt_reason = "reduce_only_order_identity_unknown"
            elif command.intent.intent_type == "NEAR_TP_REDUCE" and getattr(result, "reduce_filled", False):
                execution_state.halt_reason = "near_tp_reduce_failure"
            else:
                execution_state.halt_reason = "execution_failure_live_position"
            trader.position_contracts = contracts
        else:
            strategy.state = copy.deepcopy(command.strategy_state_snapshot)
            rolled_back = True
        halted = execution_state.trading_halted

    if entry_may_be_live:
        logger.exception("LIVE execution failed after/possibly after entry. Trading halted; strategy state NOT rolled back.")
    else:
        logger.exception("LIVE execution failed before entry; strategy state has been rolled back")

    try:
        journal.record_error(position_id=current_position_id, intent=command.intent, error=error, rolled_back=rolled_back, halted=halted)
        if str(error) == "reduce_only_order_identity_unknown" and hasattr(journal, "append"):
            journal.append(
                "REDUCE_ONLY_ORDER_IDENTITY_UNKNOWN",
                {
                    "intent_type": command.intent.intent_type,
                    "side": command.intent.side,
                    "trading_halted": halted,
                    "manual_intervention_required": True,
                },
                position_id=current_position_id,
            )
    except Exception:
        logger.exception("Failed to write trade journal error event")

    if command.intent.intent_type == "NEAR_TP_REDUCE" and getattr(result, "reduce_filled", False) and halted:
        subject = "CRITICAL: Near-TP protective SL and market exit failed"
        content = f"""
<div style="font-family: Arial, Helvetica, sans-serif; line-height: 1.55; color: #222; max-width: 760px;">
  <h2>CRITICAL: Near-TP protective SL and market exit failed</h2>
  <p><strong>Trading has been halted. Manual OKX intervention is required.</strong></p>
  <p><strong>position_id:</strong> {html.escape(str(current_position_id))}</p>
  <p><strong>side:</strong> {html.escape(command.intent.side)}</p>
  <p><strong>contracts_after:</strong> {html.escape(str(getattr(result, 'contracts_after', contracts)))}</p>
  <p><strong>protective_sl_price:</strong> {html.escape(str(getattr(result, 'protective_sl_price', '-')))}</p>
  <p><strong>Error:</strong> {html.escape(str(error))}</p>
</div>
""".strip()
    else:
        subject, content = build_live_failure_email(command.intent, error, rolled_back=rolled_back, halted=halted)
    ok = await email_sender.send_email_async(subject, content, content_type="html")
    if not ok:
        logger.error("Failed to send live execution failure email")

    return live_runtime_types.ExecutionReport(
        command=command,
        result=result,
        ok=False,
        error=error,
        entry_may_be_live=entry_may_be_live,
        created_monotonic=command.created_monotonic,
        finished_monotonic=time.monotonic(),
    )


async def account_position_sync_worker(
    *,
    state_lock: asyncio.Lock,
    account_snapshot: live_runtime_types.AccountSnapshot,
    execution_state: live_runtime_types.ExecutionState,
    trader: Trader,
    sizer: SimplePositionSizer,
    strategy: BollCvdShockReclaimStrategy,
    journal: LiveTradeJournal,
    state_store: LiveStateStore,
    position_sync_seconds: float,
    account_sync_seconds: float,
    cash_log_min_delta_usdt: float,
    rolling_loss_guard: RollingLossGuard | None = None,
    email_sender: EmailSender | None = None,
) -> None:
    last_account_sync = 0.0
    last_logged_cash = account_snapshot.cash
    last_logged_equity = account_snapshot.equity
    last_logged_position_key = position_log_key(account_snapshot.position) if account_snapshot.position is not None else ("FLAT", "0", 0.0)
    consecutive_failures = 0
    first_failure_monotonic = 0.0
    last_failure_log = 0.0
    last_stale_log = 0.0
    last_cash_event_log = 0.0
    last_flat_detected_monotonic = 0.0
    last_sidecar_status_check = 0.0
    sync_failure_log_interval_seconds = float(os.getenv("ACCOUNT_SYNC_FAILURE_LOG_INTERVAL_SECONDS", "60"))
    sync_stale_warn_seconds = float(os.getenv("ACCOUNT_SYNC_STALE_WARN_SECONDS", "180"))
    cash_transfer_detect_enabled = os.getenv("CASH_TRANSFER_DETECT_ENABLED", "true").strip().lower() in {"1", "true", "yes", "y", "on"}
    cash_transfer_min_delta_usdt = float(os.getenv("CASH_TRANSFER_MIN_DELTA_USDT", "0.5"))
    cash_transfer_settle_seconds = float(os.getenv("CASH_TRANSFER_SETTLE_SECONDS", "120"))
    cash_transfer_after_flat_cooldown_seconds = float(os.getenv("CASH_TRANSFER_AFTER_FLAT_COOLDOWN_SECONDS", "180"))
    flat_balance_confirm_attempts = int(os.getenv("FLAT_BALANCE_CONFIRM_ATTEMPTS", "5"))
    flat_balance_confirm_interval_seconds = float(os.getenv("FLAT_BALANCE_CONFIRM_INTERVAL_SECONDS", "1.5"))
    flat_balance_stable_delta_usdt = float(os.getenv("FLAT_BALANCE_STABLE_DELTA_USDT", "0.05"))
    flat_balance_cash_equity_max_diff_usdt = float(os.getenv("FLAT_BALANCE_CASH_EQUITY_MAX_DIFF_USDT", "0.10"))
    cash_drift_min_delta_usdt = float(os.getenv("CASH_DRIFT_MIN_DELTA_USDT", "0.5"))
    cash_event_log_interval_seconds = float(os.getenv("CASH_EVENT_LOG_INTERVAL_SECONDS", "60"))
    while True:
        try:
            await asyncio.sleep(position_sync_seconds)
            now = time.monotonic()
            cash = account_snapshot.cash
            equity = account_snapshot.equity
            if now - last_account_sync >= account_sync_seconds:
                equity = await trader.fetch_usdt_equity()
                cash = await fetch_usdt_cash_balance(trader)
                last_account_sync = now

            position = await trader.fetch_position_snapshot()
            core_position = position
            current_position_key = position_log_key(core_position)
            record_flat_payload: dict[str, Any] | None = None
            pending_flat_payload: dict[str, Any] | None = None
            cash_transfer_payload: dict[str, Any] | None = None
            cash_drift_payload: dict[str, Any] | None = None
            save_state_payload: tuple[str | None, StrategyPositionState, float | None] | None = None
            middle_runner_sl_payload: dict[str, Any] | None = None
            middle_runner_activation_payload: dict[str, Any] | None = None
            three_stage_post_tp1_sl_payload: dict[str, Any] | None = None
            three_stage_post_tp1_cancel_payload: dict[str, Any] | None = None
            three_stage_event_payload: dict[str, Any] | None = None
            force_close_sidecar = False
            clear_state = False
            flat_previous_halt_reason: str | None = None
            # ── Pre-core sidecar reconciliation ──────────────────────────
            # Sidecar TP may have already filled on OKX but local state still
            # counts it as open.  If we compute core_position = OKX_net -
            # stale_sidecar_open_qty first and discover the fill later via
            # monitor_sidecar_orders_once, the stale core view can incorrectly
            # trigger TP progress markers or pollute strategy cost.
            #
            # Reconcile sidecar orders NOW so that refresh_sidecar_state_totals
            # and build_core_position_view inside the main state_lock block
            # always see up-to-date sidecar_open_qty.
            _sidecar_pre_core_result = await reconcile_sidecar_orders_before_core_view(
                trader=trader,
                strategy=strategy,
                execution_state=execution_state,
                journal=journal,
                state_store=state_store,
                trader_symbol=trader.symbol,
                ts_ms=live_time_utils.utc_ms(),
                state_lock=state_lock,
            )
            sidecar_reconciled_this_sync = _sidecar_pre_core_result.queried
            sidecar_state_changed_this_sync = _sidecar_pre_core_result.changed
            # ── End pre-core reconciliation ──────────────────────────────
            async with state_lock:
                pending_order_count = execution_state.pending_order_count
                refresh_sidecar_state_totals(strategy.state, int(os.getenv("SIDECAR_MAX_LEGS", "10")))
                if open_sidecar_legs_exceed_limit(strategy.state, int(os.getenv("SIDECAR_MAX_LEGS", "10"))):
                    execution_state.trading_halted = True
                    execution_state.halt_reason = "sidecar_open_legs_exceed_max"
                    strategy.state.sidecar_dirty = True
                    strategy.state.sidecar_halt_reason = "sidecar_open_legs_exceed_max"
                    if hasattr(journal, "append"):
                        journal.append(
                            "SIDECAR_OPEN_LEGS_EXCEED_MAX",
                            {
                                "open_leg_count": sum(
                                    1
                                    for leg in strategy.state.sidecar_legs
                                    if leg.get("status") in {SidecarLegStatus.OPEN.value, SidecarLegStatus.OPEN_UNPROTECTED.value}
                                ),
                                "sidecar_max_legs": int(os.getenv("SIDECAR_MAX_LEGS", "10")),
                                "manual_intervention_required": True,
                            },
                            position_id=execution_state.current_position_id,
                        )
                open_sidecar_qty = sidecar_open_qty(strategy.state.sidecar_legs)
                core_position = build_core_position_view(position, open_sidecar_qty, sidecar_open_contracts(strategy.state.sidecar_legs))
                apply_core_position_view_to_state(strategy.state, core_position)
                current_position_key = position_log_key(core_position)
                if sidecar_position_mismatch(position, strategy.state):
                    execution_state.trading_halted = True
                    execution_state.halt_reason = "core_sidecar_position_mismatch"
                    strategy.state.sidecar_dirty = True
                    strategy.state.sidecar_halt_reason = "core_sidecar_position_mismatch"
                    if hasattr(journal, "append"):
                        journal.append(
                            "CORE_SIDECAR_POSITION_MISMATCH",
                            {
                                "okx_eth_qty": position.eth_qty,
                                "core_eth_qty": core_position.eth_qty,
                                "sidecar_open_qty": open_sidecar_qty,
                                "manual_intervention_required": True,
                            },
                            position_id=execution_state.current_position_id,
                        )
                    logger.error(
                        "CORE_SIDECAR_POSITION_MISMATCH | position_id=%s okx_eth_qty=%.8f core_eth_qty=%.8f sidecar_open_qty=%.8f trading_halted=true manual_intervention_required=true",
                        execution_state.current_position_id,
                        position.eth_qty,
                        core_position.eth_qty,
                        open_sidecar_qty,
                    )
                force_close_sidecar = bool(
                    pending_order_count == 0
                    and not core_position.has_position
                    and open_sidecar_qty > 0
                    and getattr(strategy.state, "sidecar_enabled_for_position", False)
                    and getattr(sizer.config, "sidecar_close_when_core_flat", True)
                )
                flat_transition_detected = (
                    pending_order_count == 0
                    and not core_position.has_position
                    and not force_close_sidecar
                    and strategy.state.layers > 0
                )
                if flat_transition_detected:
                    pending_flat_payload = {
                        "position_id": execution_state.current_position_id,
                        "symbol": trader.symbol,
                        "side": strategy.state.side,
                        "cash_before_position": execution_state.cash_before_position,
                        "reason": "OKX position is flat. TP filled or manual close detected.",
                        "layers": strategy.state.layers,
                        "avg_entry_price": strategy.state.avg_entry_price,
                        "last_tp_price": strategy.state.tp_price,
                        "last_partial_tp_price": getattr(strategy.state, "partial_tp_price", None),
                        "last_tp_plan": getattr(strategy.state, "tp_plan", "SINGLE"),
                        "partial_tp_consumed": getattr(strategy.state, "partial_tp_consumed", False),
                        "near_tp_protective_sl_order_id": getattr(strategy.state, "near_tp_protective_sl_order_id", None),
                        "middle_runner_protective_sl_order_id": getattr(strategy.state, "middle_runner_protective_sl_order_id", None),
                        "three_stage_post_tp1_protective_sl_order_id": getattr(strategy.state, "three_stage_post_tp1_protective_sl_order_id", None),
                        "trend_runner_sl_order_id": getattr(strategy.state, "trend_runner_sl_order_id", None),
                        "trend_runner_exit_reason": getattr(strategy.state, "trend_runner_exit_reason", None),
                    }
                    execution_state.trading_halted = True
                    last_flat_detected_monotonic = now
                    logger.warning("POSITION_SYNC_CHANGED | flat_on_okx=true. Confirming settled balance before FLAT journal.")
                else:
                    account_snapshot.position = core_position
                    account_snapshot.cash = cash
                    account_snapshot.equity = equity
                    account_snapshot.updated_monotonic = time.monotonic()
                    account_snapshot.updated_ts_ms = live_time_utils.utc_ms()
                    account_snapshot.version += 1
                    trader.account_equity_usdt = equity
                    sizer.update_account_equity(equity)

                    cash_delta = cash - last_logged_cash
                    seconds_since_last_order = (
                        cash_transfer_settle_seconds
                        if execution_state.last_order_ts_ms == 0
                        else max((live_time_utils.utc_ms() - execution_state.last_order_ts_ms) / 1000, 0.0)
                    )
                    unsafe_reasons = []
                    if pending_order_count > 0:
                        unsafe_reasons.append("pending_order")
                    if core_position.has_position:
                        unsafe_reasons.append("has_position")
                    if strategy.state.layers != 0:
                        unsafe_reasons.append("strategy_layers")
                    if execution_state.current_position_id is not None:
                        unsafe_reasons.append("current_position_id")
                    if seconds_since_last_order < cash_transfer_settle_seconds:
                        unsafe_reasons.append("order_settle")
                    in_flat_settle_cooldown = (
                        last_flat_detected_monotonic > 0
                        and now - last_flat_detected_monotonic < cash_transfer_after_flat_cooldown_seconds
                    )
                    if in_flat_settle_cooldown:
                        unsafe_reasons.append("flat_settle_cooldown")
                    safe_for_cash_transfer = (
                        cash_transfer_detect_enabled
                        and pending_order_count == 0
                        and not core_position.has_position
                        and strategy.state.layers == 0
                        and execution_state.current_position_id is None
                        and seconds_since_last_order >= cash_transfer_settle_seconds
                        and not in_flat_settle_cooldown
                        and abs(cash_delta) >= cash_transfer_min_delta_usdt
                    )
                    if safe_for_cash_transfer:
                        direction = "DEPOSIT" if cash_delta > 0 else "WITHDRAWAL"
                        cash_transfer_payload = {
                            "direction": direction,
                            "amount": cash_delta,
                            "cash_before": last_logged_cash,
                            "cash_after": cash,
                            "equity_before": last_logged_equity,
                            "equity_after": equity,
                            "reason": "safe_flat_account_sync",
                        }
                        if now - last_cash_event_log >= cash_event_log_interval_seconds:
                            logger.warning(
                                "CASH_TRANSFER_DETECTED | direction=%s amount=%.4f cash_before=%.4f cash_after=%.4f",
                                direction,
                                cash_delta,
                                last_logged_cash,
                                cash,
                            )
                            last_cash_event_log = now
                    elif unsafe_reasons and abs(cash_delta) >= cash_drift_min_delta_usdt:
                        drift_reason = "unsafe_state:" + ",".join(unsafe_reasons)
                        cash_drift_payload = {
                            "amount": cash_delta,
                            "cash_before": last_logged_cash,
                            "cash_after": cash,
                            "equity_before": last_logged_equity,
                            "equity_after": equity,
                            "reason": drift_reason,
                        }
                        if now - last_cash_event_log >= cash_event_log_interval_seconds:
                            logger.warning(
                                "ACCOUNT_CASH_DRIFT | amount=%.4f cash_before=%.4f cash_after=%.4f reason=%s",
                                cash_delta,
                                last_logged_cash,
                                cash,
                                drift_reason,
                            )
                            last_cash_event_log = now

                    if abs(cash - last_logged_cash) >= cash_log_min_delta_usdt:
                        logger.info(
                            "CASH_SYNC_CHANGED | cash=%.4f previous=%.4f equity=%.4f layer_margin_pct=%.4f leverage=%.2f",
                            cash,
                            last_logged_cash,
                            equity,
                            sizer.config.layer_margin_pct,
                            sizer.config.leverage,
                        )
                        last_logged_cash = cash
                        last_logged_equity = equity
                    elif cash_transfer_payload is not None or cash_drift_payload is not None:
                        last_logged_cash = cash
                        last_logged_equity = equity

                if not flat_transition_detected and core_position.has_position:
                    trader.position_contracts = core_position.contracts
                    # Position reduction detection must run every account sync,
                    # even when pending orders exist (e.g. TP2 / Sidecar TP still
                    # pending after TP1 fill). The mark_* helpers are internally
                    # idempotent via consumed/active flags.
                    middle_runner_activated = mark_middle_runner_active_if_position_reduced(strategy, core_position)
                    three_stage_event = mark_three_stage_progress_if_position_reduced(strategy, core_position, live_time_utils.utc_ms())
                    mark_partial_tp_consumed_if_position_reduced(strategy, core_position)
                    sync_strategy_cost_from_position(strategy, core_position)
                    if pending_order_count > 0 and three_stage_event is not None:
                        logger.warning(
                            "THREE_STAGE_POSITION_REDUCTION_DETECTED_WITH_PENDING_ORDERS | "
                            "event=%s pending_order_count=%s side=%s old_total_eth_qty=%.8f new_core_eth_qty=%.8f core_contracts=%s net_contracts=%s sidecar_open_eth_qty=%.8f",
                            three_stage_event,
                            pending_order_count,
                            core_position.side,
                            float(getattr(strategy.state, "total_entry_qty", 0.0) or 0.0),
                            float(core_position.eth_qty or 0.0),
                            core_position.contracts,
                            position.contracts if position.has_position else 0,
                            sidecar_open_qty(list(getattr(strategy.state, "sidecar_legs", []) or [])),
                        )
                    if three_stage_event is not None:
                        if three_stage_event in {"TP1", "TP1_TP2"} and three_stage_event != "TP1_TP2":
                            config = getattr(strategy, "config", None)
                            if bool(getattr(config, "three_stage_post_tp1_protective_sl_enabled", True)):
                                post_tp1_boll = three_stage_post_tp1_boll(strategy)
                                protective_sl = None
                                current_price = None
                                price_source = "missing"
                                if post_tp1_boll is not None and core_position.side is not None:
                                    current_price, price_source = three_stage_post_tp1_current_price(account_snapshot, core_position, post_tp1_boll, live_time_utils.utc_ms())
                                    base_sl = strategy._calculate_three_stage_post_tp1_protective_sl(core_position.side, current_price, post_tp1_boll)
                                    extension_sl = strategy._apply_three_stage_post_tp1_extension_trigger(core_position.side, current_price, post_tp1_boll, base_sl)
                                    protective_sl = strategy._tighten_optional_three_stage_post_tp1_sl(core_position.side, base_sl, extension_sl)
                                strategy.state.three_stage_post_tp1_protective_sl_price = protective_sl
                                # Global protective SL must cover OKX net position (core + sidecar)
                                if not position.has_position or position.side != core_position.side or position.contracts <= 0:
                                    execution_state.trading_halted = True
                                    execution_state.halt_reason = "three_stage_post_tp1_global_sl_net_position_missing"
                                    if hasattr(journal, "append"):
                                        journal.append(
                                            "THREE_STAGE_POST_TP1_GLOBAL_SL_NET_POSITION_MISSING",
                                            {
                                                "position_id": execution_state.current_position_id,
                                                "core_side": core_position.side,
                                                "core_contracts": core_position.contracts,
                                                "net_side": position.side if position.has_position else None,
                                                "net_contracts": position.contracts if position.has_position else 0,
                                                "trading_halted": True,
                                                "halt_reason": "three_stage_post_tp1_global_sl_net_position_missing",
                                                "manual_intervention_required": True,
                                            },
                                            position_id=execution_state.current_position_id,
                                        )
                                    state_store.save(LiveStateStore.from_strategy_state(
                                        position_id=execution_state.current_position_id,
                                        symbol=trader.symbol,
                                        strategy_state=strategy.state,
                                        cash_before_position=execution_state.cash_before_position,
                                    ))
                                    logger.error(
                                        "THREE_STAGE_POST_TP1_GLOBAL_SL_NET_POSITION_MISSING | position_id=%s core_side=%s core_contracts=%s net_side=%s net_contracts=%s trading_halted=true halt_reason=three_stage_post_tp1_global_sl_net_position_missing manual_intervention_required=true",
                                        execution_state.current_position_id,
                                        core_position.side,
                                        core_position.contracts,
                                        position.side if position.has_position else None,
                                        position.contracts if position.has_position else 0,
                                    )
                                else:
                                    three_stage_post_tp1_sl_payload = {
                                        "position_id": execution_state.current_position_id,
                                        "side": core_position.side,
                                        "contracts": position.contracts,
                                        "core_contracts": core_position.contracts,
                                        "net_contracts": position.contracts,
                                        "protective_sl_price": protective_sl,
                                        "old_sl_order_id": getattr(strategy.state, "three_stage_post_tp1_protective_sl_order_id", None),
                                        "current_price": current_price,
                                        "current_price_source": price_source,
                                        "reason": "three_stage_tp1_filled",
                                    }
                        if three_stage_event in {"TP2", "TP1_TP2"}:
                            old_post_tp1_sl_order_id = getattr(strategy.state, "three_stage_post_tp1_protective_sl_order_id", None)
                            three_stage_post_tp1_cancel_payload = {
                                "position_id": execution_state.current_position_id,
                                "side": position.side,
                                "protective_sl_order_id": old_post_tp1_sl_order_id,
                                "protective_sl_price": getattr(strategy.state, "three_stage_post_tp1_protective_sl_price", None),
                                "pending_halt_applied": False,
                                "existing_halt_reason": execution_state.halt_reason if execution_state.trading_halted else None,
                            }
                            if old_post_tp1_sl_order_id:
                                if not execution_state.trading_halted:
                                    execution_state.trading_halted = True
                                    execution_state.halt_reason = THREE_STAGE_CANCEL_PENDING_HALT_REASON
                                    three_stage_post_tp1_cancel_payload["pending_halt_applied"] = True
                                else:
                                    logger.error(
                                        "THREE_STAGE_TP1_PROTECTIVE_SL_CANCEL_PENDING_ON_TP2 | position_id=%s algoId=%s existing_halt_reason=%s pending_halt_not_applied=true",
                                        execution_state.current_position_id,
                                        old_post_tp1_sl_order_id,
                                        execution_state.halt_reason,
                                    )
                        three_stage_event_payload = {
                            "event": three_stage_event,
                            "position_id": execution_state.current_position_id,
                            "side": core_position.side,
                            "layers": strategy.state.layers,
                            "avg_entry_price": strategy.state.avg_entry_price,
                            "tp_plan": "THREE_STAGE_RUNNER",
                            "tp1_price": getattr(strategy.state, "three_stage_tp1_price", None),
                            "tp1_ratio": getattr(strategy.state, "three_stage_tp1_ratio", 0.0),
                            "tp2_price": getattr(strategy.state, "three_stage_tp2_price", None),
                            "tp2_ratio": getattr(strategy.state, "three_stage_tp2_ratio", 0.0),
                            "runner_tp_price": getattr(strategy.state, "trend_runner_tp_price", None),
                            "runner_sl_price": getattr(strategy.state, "trend_runner_sl_price", None),
                            "runner_ratio": getattr(strategy.state, "three_stage_runner_ratio", 0.0),
                            "trend_runner_active": getattr(strategy.state, "trend_runner_active", False),
                            "trend_runner_adjust_count": getattr(strategy.state, "trend_runner_adjust_count", 0),
                            "trend_runner_trend_start_ts_ms": getattr(strategy.state, "trend_runner_trend_start_ts_ms", 0),
                        }
                    if middle_runner_activated:
                        config = getattr(strategy, "config", None)
                        if bool(getattr(config, "middle_runner_disable_add_after_partial", True)):
                            strategy.state.middle_runner_add_disabled = True
                        middle_runner_activation_payload = {
                            "position_id": execution_state.current_position_id,
                            "side": core_position.side,
                            "layers": strategy.state.layers,
                            "avg_entry_price": strategy.state.avg_entry_price,
                            "first_tp_price": getattr(strategy.state, "middle_runner_first_tp_price", None),
                            "final_tp_price": getattr(strategy.state, "middle_runner_final_tp_price", None),
                            "first_close_ratio": getattr(strategy.state, "middle_runner_first_close_ratio", 0.0),
                            "keep_ratio": getattr(strategy.state, "middle_runner_keep_ratio", 0.0),
                            "reason": "partial_tp_filled",
                        }
                        if bool(getattr(config, "middle_runner_protective_sl_enabled", True)):
                            runner_boll = middle_runner_activation_boll(strategy)
                            current_price = getattr(runner_boll, "middle", 0.0) if runner_boll is not None else 0.0
                            protective_sl = (
                                strategy._calculate_middle_runner_protective_sl(core_position.side, current_price, runner_boll)
                                if runner_boll is not None and core_position.side is not None
                                else None
                            )
                            strategy.state.middle_runner_protective_sl_price = protective_sl
                            # Global protective SL must cover OKX net position (core + sidecar)
                            if not position.has_position or position.side != core_position.side or position.contracts <= 0:
                                execution_state.trading_halted = True
                                execution_state.halt_reason = "middle_runner_global_sl_net_position_missing"
                                if hasattr(journal, "append"):
                                    journal.append(
                                        "MIDDLE_RUNNER_GLOBAL_SL_NET_POSITION_MISSING",
                                        {
                                            "position_id": execution_state.current_position_id,
                                            "core_side": core_position.side,
                                            "core_contracts": core_position.contracts,
                                            "net_side": position.side if position.has_position else None,
                                            "net_contracts": position.contracts if position.has_position else 0,
                                            "trading_halted": True,
                                            "halt_reason": "middle_runner_global_sl_net_position_missing",
                                            "manual_intervention_required": True,
                                        },
                                        position_id=execution_state.current_position_id,
                                    )
                                state_store.save(LiveStateStore.from_strategy_state(
                                    position_id=execution_state.current_position_id,
                                    symbol=trader.symbol,
                                    strategy_state=strategy.state,
                                    cash_before_position=execution_state.cash_before_position,
                                ))
                                logger.error(
                                    "MIDDLE_RUNNER_GLOBAL_SL_NET_POSITION_MISSING | position_id=%s core_side=%s core_contracts=%s net_side=%s net_contracts=%s trading_halted=true halt_reason=middle_runner_global_sl_net_position_missing manual_intervention_required=true",
                                    execution_state.current_position_id,
                                    core_position.side,
                                    core_position.contracts,
                                    position.side if position.has_position else None,
                                    position.contracts if position.has_position else 0,
                                )
                            else:
                                middle_runner_sl_payload = {
                                    "position_id": execution_state.current_position_id,
                                    "side": core_position.side,
                                    "contracts": position.contracts,
                                    "core_contracts": core_position.contracts,
                                    "net_contracts": position.contracts,
                                    "protective_sl_price": protective_sl,
                                    "old_sl_order_id": getattr(strategy.state, "middle_runner_protective_sl_order_id", None),
                                    "reason": "partial_tp_filled",
                                }
                    elif middle_runner_size_mismatch_needs_degraded_protection(strategy, core_position):
                        runner_boll = middle_runner_activation_boll(strategy)
                        current_price = getattr(runner_boll, "middle", 0.0) if runner_boll is not None else 0.0
                        protective_sl = (
                            strategy._calculate_middle_runner_protective_sl(core_position.side, current_price, runner_boll)
                            if runner_boll is not None and core_position.side is not None
                            else None
                        )
                        strategy.state.middle_runner_protective_sl_price = protective_sl
                        # Global protective SL must cover OKX net position (core + sidecar)
                        if not position.has_position or position.side != core_position.side or position.contracts <= 0:
                            execution_state.trading_halted = True
                            execution_state.halt_reason = "middle_runner_global_sl_net_position_missing"
                            if hasattr(journal, "append"):
                                journal.append(
                                    "MIDDLE_RUNNER_GLOBAL_SL_NET_POSITION_MISSING",
                                    {
                                        "position_id": execution_state.current_position_id,
                                        "core_side": core_position.side,
                                        "core_contracts": core_position.contracts,
                                        "net_side": position.side if position.has_position else None,
                                        "net_contracts": position.contracts if position.has_position else 0,
                                        "trading_halted": True,
                                        "halt_reason": "middle_runner_global_sl_net_position_missing",
                                        "manual_intervention_required": True,
                                    },
                                    position_id=execution_state.current_position_id,
                                )
                            state_store.save(LiveStateStore.from_strategy_state(
                                position_id=execution_state.current_position_id,
                                symbol=trader.symbol,
                                strategy_state=strategy.state,
                                cash_before_position=execution_state.cash_before_position,
                            ))
                            logger.error(
                                "MIDDLE_RUNNER_GLOBAL_SL_NET_POSITION_MISSING | position_id=%s core_side=%s core_contracts=%s net_side=%s net_contracts=%s trading_halted=true halt_reason=middle_runner_global_sl_net_position_missing manual_intervention_required=true",
                                execution_state.current_position_id,
                                core_position.side,
                                core_position.contracts,
                                position.side if position.has_position else None,
                                position.contracts if position.has_position else 0,
                            )
                        else:
                            middle_runner_sl_payload = {
                                "position_id": execution_state.current_position_id,
                                "side": core_position.side,
                                "contracts": position.contracts,
                                "core_contracts": core_position.contracts,
                                "net_contracts": position.contracts,
                                "protective_sl_price": protective_sl,
                                "old_sl_order_id": getattr(strategy.state, "middle_runner_protective_sl_order_id", None),
                                "reason": "partial_size_mismatch_degraded",
                            }
                        middle_runner_activation_payload = {
                            "position_id": execution_state.current_position_id,
                            "side": core_position.side,
                            "layers": strategy.state.layers,
                            "avg_entry_price": strategy.state.avg_entry_price,
                            "first_tp_price": getattr(strategy.state, "middle_runner_first_tp_price", None),
                            "final_tp_price": getattr(strategy.state, "middle_runner_final_tp_price", None),
                            "first_close_ratio": getattr(strategy.state, "middle_runner_first_close_ratio", 0.0),
                            "keep_ratio": getattr(strategy.state, "middle_runner_keep_ratio", 0.0),
                            "reason": "partial_size_mismatch_degraded",
                        }
                    if (
                        execution_state.trading_halted
                        and execution_state.halt_reason == "near_tp_protected_sync_failed"
                        and getattr(strategy.state, "near_tp_protected", False)
                    ):
                        execution_state.trading_halted = False
                        execution_state.halt_reason = None
                        logger.warning(
                            "NEAR_TP_PROTECTED_SYNC_RECOVERED | trading_halted=false side=%s contracts=%s avg_entry=%.4f",
                            core_position.side,
                            core_position.contracts,
                            core_position.avg_entry_price,
                        )
                    save_state_payload = (execution_state.current_position_id, copy.deepcopy(strategy.state), execution_state.cash_before_position)
                    if current_position_key != last_logged_position_key:
                        logger.info(
                            "POSITION_SYNC_CHANGED | side=%s contracts=%s avg_entry=%.4f eth_qty=%.6f strategy_layers=%s",
                            core_position.side,
                            core_position.contracts,
                            core_position.avg_entry_price,
                            core_position.eth_qty,
                            strategy.state.layers,
                        )
                        last_logged_position_key = current_position_key

            if force_close_sidecar:
                await force_close_sidecar_after_core_flat(
                    trader=trader,
                    strategy_state=strategy.state,
                    execution_state=execution_state,
                    journal=journal,
                    state_store=state_store,
                    trader_symbol=trader.symbol,
                    position_id=execution_state.current_position_id,
                    cash_before_position=execution_state.cash_before_position,
                    ts_ms=live_time_utils.utc_ms(),
                )
                continue

            sidecar_check_seconds = max(float(getattr(sizer.config, "sidecar_order_status_check_seconds", 5.0) or 5.0), 0.0)
            if (
                sidecar_check_seconds >= 0
                and now - last_sidecar_status_check >= sidecar_check_seconds
                and getattr(strategy.state, "sidecar_enabled_for_position", False)
                and not sidecar_reconciled_this_sync
                and pending_order_count == 0
            ):
                last_sidecar_status_check = now
                await monitor_sidecar_orders_once(
                    trader=trader,
                    strategy_state=strategy.state,
                    execution_state=execution_state,
                    journal=journal,
                    state_store=state_store,
                    trader_symbol=trader.symbol,
                    core_position=core_position,
                    position_id=execution_state.current_position_id,
                    cash_before_position=execution_state.cash_before_position,
                    ts_ms=live_time_utils.utc_ms(),
                    fee_buffer_pct=strategy.config.breakeven_fee_buffer_pct,
                )

            if pending_flat_payload is not None:
                try:
                    settled = await fetch_settled_flat_balance(
                        trader,
                        attempts=flat_balance_confirm_attempts,
                        interval_seconds=flat_balance_confirm_interval_seconds,
                        stable_delta_usdt=flat_balance_stable_delta_usdt,
                        cash_equity_max_diff_usdt=flat_balance_cash_equity_max_diff_usdt,
                    )
                except Exception as exc:
                    logger.exception("FLAT_BALANCE_SETTLE_FAILED | falling back to latest account equity before FLAT journal")
                    settled = live_runtime_types.SettledFlatBalance(
                        cash=equity,
                        equity=equity,
                        attempts=0,
                        stable=False,
                        reason=f"fallback_to_equity_after_error:{type(exc).__name__}:{exc}",
                    )
                logger.warning(
                    "FLAT_BALANCE_SETTLED | cash=%.4f equity=%.4f attempts=%s stable=%s reason=%s",
                    settled.cash,
                    settled.equity,
                    settled.attempts,
                    settled.stable,
                    settled.reason,
                )
                record_flat_payload = {
                    **pending_flat_payload,
                    "cash_after": settled.cash,
                    "equity_after": settled.equity,
                }
                cash = settled.cash
                equity = settled.equity
                protective_sl_order_id = pending_flat_payload.get("near_tp_protective_sl_order_id")
                if protective_sl_order_id:
                    try:
                        await trader.cancel_near_tp_protective_stop(protective_sl_order_id)
                    except Exception:
                        logger.warning("NEAR_TP_PROTECTIVE_SL_CANCEL_ON_FLAT | algoId=%s failed_unhandled", protective_sl_order_id)
                middle_runner_sl_order_id = pending_flat_payload.get("middle_runner_protective_sl_order_id")
                if middle_runner_sl_order_id:
                    try:
                        await trader.cancel_middle_runner_protective_stop(middle_runner_sl_order_id)
                    except Exception:
                        logger.warning("MIDDLE_RUNNER_CANCELLED | reason=flat_sl_cancel_failed algoId=%s", middle_runner_sl_order_id)
                three_stage_post_tp1_sl_order_id = pending_flat_payload.get("three_stage_post_tp1_protective_sl_order_id")
                if three_stage_post_tp1_sl_order_id:
                    try:
                        await trader.cancel_three_stage_post_tp1_protective_stop(three_stage_post_tp1_sl_order_id)
                    except Exception:
                        logger.warning("THREE_STAGE_TP1_PROTECTIVE_SL_CANCELLED | reason=flat_sl_cancel_failed algoId=%s", three_stage_post_tp1_sl_order_id)
                trend_runner_sl_order_id = pending_flat_payload.get("trend_runner_sl_order_id")
                if trend_runner_sl_order_id:
                    try:
                        await trader.cancel_trend_runner_protective_stop(trend_runner_sl_order_id)
                    except Exception:
                        logger.warning("TREND_RUNNER_CANCELLED | reason=flat_sl_cancel_failed algoId=%s", trend_runner_sl_order_id)
                async with state_lock:
                    flat_previous_halt_reason = execution_state.halt_reason if execution_state.trading_halted else None
                    account_snapshot.position = position
                    account_snapshot.cash = settled.cash
                    account_snapshot.equity = settled.equity
                    account_snapshot.updated_monotonic = time.monotonic()
                    account_snapshot.updated_ts_ms = live_time_utils.utc_ms()
                    account_snapshot.version += 1
                    trader.account_equity_usdt = settled.equity
                    sizer.update_account_equity(settled.equity)
                    strategy.state = StrategyPositionState()
                    trader.mark_flat()
                    flat_clearable_halt_reasons = {
                        None,
                        "near_tp_exit_all_waiting_flat",
                        "near_tp_protected_sync_failed",
                        "trend_runner_market_exit_waiting_flat",
                        "three_stage_post_tp1_sl_failed_market_exit_waiting_flat",
                        "sidecar_tp_place_failed_market_exit_waiting_flat",
                        "rolling_loss_soft_halt",
                        "rolling_loss_hard_halt",
                    }
                    preserve_critical_halt = (
                        rolling_loss_guard is not None
                        and flat_previous_halt_reason not in flat_clearable_halt_reasons
                    )
                    execution_state.trading_halted = preserve_critical_halt
                    execution_state.halt_reason = flat_previous_halt_reason if preserve_critical_halt else None
                    execution_state.halt_until_ts_ms = None
                    execution_state.current_position_id = None
                    execution_state.cash_before_position = None
                    clear_state = True
                    last_logged_cash = settled.cash
                    last_logged_equity = settled.equity
                    last_logged_position_key = current_position_key
                    logger.warning("NEAR_TP_STATE_CLEARED_ON_FLAT | protective_sl_order_id=%s", protective_sl_order_id)

            if cash_transfer_payload is not None:
                journal.record_cash_transfer(**cash_transfer_payload)
                if rolling_loss_guard is not None and rolling_loss_guard.state is not None and rolling_loss_guard.state.enabled:
                    transfer_equity_after = cash_transfer_payload.get("equity_after")
                    transfer_cash_after = cash_transfer_payload.get("cash_after")
                    new_reference = transfer_equity_after if transfer_equity_after is not None else transfer_cash_after
                    if new_reference is not None:
                        rolling_loss_guard.adjust_flat_reference_for_cash_transfer(
                            now_ms=live_time_utils.utc_ms(),
                            new_flat_equity=float(new_reference),
                            reason=str(cash_transfer_payload.get("reason") or "safe_flat_cash_transfer"),
                        )
            if cash_drift_payload is not None:
                journal.record_account_cash_drift(**cash_drift_payload)
            if three_stage_event_payload is not None and hasattr(journal, "append"):
                append_three_stage_progress_journal_events(journal, three_stage_event_payload)
            if three_stage_post_tp1_cancel_payload is not None:
                old_order_id = three_stage_post_tp1_cancel_payload.get("protective_sl_order_id")
                cancel_ok = True
                if old_order_id:
                    try:
                        cancel_ok = await trader.cancel_three_stage_post_tp1_protective_stop(old_order_id)
                    except Exception:
                        cancel_ok = False
                        logger.exception(
                            "THREE_STAGE_TP1_PROTECTIVE_SL_CANCEL_FAILED_ON_TP2 | position_id=%s algoId=%s cancel_exception=true manual_intervention_required=true",
                            three_stage_post_tp1_cancel_payload.get("position_id"),
                            old_order_id,
                        )
                if cancel_ok:
                    async with state_lock:
                        strategy.state.three_stage_post_tp1_protective_sl_order_id = None
                        strategy.state.three_stage_post_tp1_protective_sl_price = None
                        strategy.state.three_stage_post_tp1_protected = False
                        if (
                            three_stage_post_tp1_cancel_payload.get("pending_halt_applied")
                            and execution_state.trading_halted
                            and execution_state.halt_reason == THREE_STAGE_CANCEL_PENDING_HALT_REASON
                        ):
                            execution_state.trading_halted = False
                            execution_state.halt_reason = None
                        save_state_payload = (execution_state.current_position_id, copy.deepcopy(strategy.state), execution_state.cash_before_position)
                    if hasattr(journal, "append"):
                        journal.append(
                            "THREE_STAGE_TP1_PROTECTIVE_SL_CANCELLED_ON_TP2",
                            {
                                **three_stage_post_tp1_cancel_payload,
                                "cancel_ok": True,
                                "reason": "three_stage_tp2_filled",
                            },
                            position_id=three_stage_post_tp1_cancel_payload.get("position_id"),
                        )
                    logger.warning(
                        "THREE_STAGE_TP1_PROTECTIVE_SL_CANCELLED_ON_TP2 | position_id=%s algoId=%s cancel_ok=true",
                        three_stage_post_tp1_cancel_payload.get("position_id"),
                        old_order_id,
                    )
                else:
                    async with state_lock:
                        strategy.state.three_stage_post_tp1_protective_sl_order_id = old_order_id
                        strategy.state.three_stage_post_tp1_protective_sl_price = three_stage_post_tp1_cancel_payload.get("protective_sl_price")
                        strategy.state.three_stage_post_tp1_protected = True
                        execution_state.trading_halted = True
                        execution_state.halt_reason = "three_stage_post_tp1_sl_cancel_failed_on_tp2"
                        save_state_payload = (execution_state.current_position_id, copy.deepcopy(strategy.state), execution_state.cash_before_position)
                    if hasattr(journal, "append"):
                        journal.append(
                            "THREE_STAGE_TP1_PROTECTIVE_SL_CANCEL_FAILED_ON_TP2",
                            {
                                **three_stage_post_tp1_cancel_payload,
                                "critical": True,
                                "cancel_ok": False,
                                "trading_halted": True,
                                "halt_reason": "three_stage_post_tp1_sl_cancel_failed_on_tp2",
                                "reason": "manual_intervention_required_old_post_tp1_sl_may_remain_on_exchange",
                            },
                            position_id=three_stage_post_tp1_cancel_payload.get("position_id"),
                        )
                    logger.error(
                        "THREE_STAGE_TP1_PROTECTIVE_SL_CANCEL_FAILED_ON_TP2 | position_id=%s algoId=%s protective_sl_price=%s trading_halted=true halt_reason=three_stage_post_tp1_sl_cancel_failed_on_tp2 manual_intervention_required=true",
                        three_stage_post_tp1_cancel_payload.get("position_id"),
                        old_order_id,
                        three_stage_post_tp1_cancel_payload.get("protective_sl_price"),
                    )
            if three_stage_post_tp1_sl_payload is not None:
                sl_price = three_stage_post_tp1_sl_payload.get("protective_sl_price")
                sl_order_id = None
                sl_ok = False
                sl_message = "protective_sl_price_missing"
                if sl_price is not None and three_stage_post_tp1_sl_payload.get("side") is not None:
                    sl_ok, sl_order_id, sl_message = await trader.place_three_stage_post_tp1_protective_stop_with_retries(
                        three_stage_post_tp1_sl_payload["side"],
                        three_stage_post_tp1_sl_payload["contracts"],
                        float(sl_price),
                        retry_count=int(os.getenv("NEAR_TP_PROTECTIVE_SL_RETRY_COUNT", "3")),
                        retry_interval_seconds=float(os.getenv("NEAR_TP_PROTECTIVE_SL_RETRY_INTERVAL_SECONDS", "1")),
                    )
                if sl_ok:
                    old_sl_order_id = three_stage_post_tp1_sl_payload.get("old_sl_order_id")
                    if old_sl_order_id and old_sl_order_id != sl_order_id:
                        await trader.cancel_three_stage_post_tp1_protective_stop(old_sl_order_id)
                    async with state_lock:
                        strategy.state.three_stage_post_tp1_protective_sl_order_id = sl_order_id
                        strategy.state.three_stage_post_tp1_protective_sl_price = float(sl_price)
                        strategy.state.three_stage_post_tp1_protected = True
                        save_state_payload = (execution_state.current_position_id, copy.deepcopy(strategy.state), execution_state.cash_before_position)
                    if hasattr(journal, "append"):
                        journal.append(
                            "THREE_STAGE_TP1_PROTECTIVE_SL_PLACED",
                            {
                                "position_id": three_stage_post_tp1_sl_payload.get("position_id"),
                                "side": three_stage_post_tp1_sl_payload.get("side"),
                                "contracts": str(three_stage_post_tp1_sl_payload.get("contracts")),
                                "core_contracts": three_stage_post_tp1_sl_payload.get("core_contracts"),
                                "net_contracts": three_stage_post_tp1_sl_payload.get("net_contracts"),
                                "sl_contracts": str(three_stage_post_tp1_sl_payload.get("contracts")),
                                "protective_sl_price": sl_price,
                                "protective_sl_order_id": sl_order_id,
                                "current_price": three_stage_post_tp1_sl_payload.get("current_price"),
                                "current_price_source": three_stage_post_tp1_sl_payload.get("current_price_source"),
                                "avg_entry_price": getattr(strategy.state, "avg_entry_price", None),
                                "tp1_price": getattr(strategy.state, "three_stage_tp1_price", None),
                                "tp1_ratio": getattr(strategy.state, "three_stage_tp1_ratio", 0.0),
                                "tp2_price": getattr(strategy.state, "three_stage_tp2_price", None),
                                "tp2_ratio": getattr(strategy.state, "three_stage_tp2_ratio", 0.0),
                                "runner_ratio": getattr(strategy.state, "three_stage_runner_ratio", 0.0),
                                "reason": "three_stage_tp1_filled",
                                "retry_config": "NEAR_TP_PROTECTIVE_SL_RETRY_COUNT/NEAR_TP_PROTECTIVE_SL_RETRY_INTERVAL_SECONDS",
                            },
                            position_id=three_stage_post_tp1_sl_payload.get("position_id"),
                        )
                    logger.warning(
                        "THREE_STAGE_TP1_PROTECTIVE_SL_PLACED | position_id=%s side=%s core_contracts=%s net_contracts=%s sl_contracts=%s protective_sl_price=%s protective_sl_order_id=%s retry_config=near_tp",
                        three_stage_post_tp1_sl_payload.get("position_id"),
                        three_stage_post_tp1_sl_payload.get("side"),
                        three_stage_post_tp1_sl_payload.get("core_contracts"),
                        three_stage_post_tp1_sl_payload.get("net_contracts"),
                        three_stage_post_tp1_sl_payload.get("contracts"),
                        sl_price,
                        sl_order_id,
                    )
                else:
                    # ── Risk control: protective SL failed → immediate full market exit ──
                    side = three_stage_post_tp1_sl_payload.get("side")
                    exit_ok, exit_message = (False, "side_missing")
                    if side is not None:
                        exit_ok, exit_message = await trader.market_exit_remaining_position_with_retries(
                            side,
                            retry_count=int(os.getenv("NEAR_TP_SL_FAIL_MARKET_EXIT_RETRY_COUNT", "3")),
                        )
                    core_contracts = three_stage_post_tp1_sl_payload.get("core_contracts")
                    net_contracts = three_stage_post_tp1_sl_payload.get("net_contracts")
                    sl_contracts = three_stage_post_tp1_sl_payload.get("contracts")
                    manual_intervention_required = not exit_ok
                    if exit_ok:
                        halt_reason = "three_stage_post_tp1_sl_failed_market_exit_waiting_flat"
                    else:
                        halt_reason = "three_stage_post_tp1_protective_sl_failure"
                    async with state_lock:
                        execution_state.trading_halted = True
                        execution_state.halt_reason = halt_reason
                    state_store.save(
                        LiveStateStore.from_strategy_state(
                            position_id=execution_state.current_position_id,
                            symbol=trader.symbol,
                            strategy_state=strategy.state,
                            cash_before_position=execution_state.cash_before_position,
                        )
                    )
                    if hasattr(journal, "append"):
                        journal.append(
                            "THREE_STAGE_POST_TP1_PROTECTIVE_SL_FAILED",
                            {
                                "position_id": three_stage_post_tp1_sl_payload.get("position_id"),
                                "side": side,
                                "protective_sl_price": sl_price,
                                "reason": sl_message,
                                "trading_halted": True,
                                "halt_reason": halt_reason,
                                "retry_config": "NEAR_TP_PROTECTIVE_SL_RETRY_COUNT/NEAR_TP_PROTECTIVE_SL_RETRY_INTERVAL_SECONDS",
                                "market_exit_attempted": True,
                                "market_exit_ok": exit_ok,
                                "market_exit_message": exit_message,
                                "core_contracts": core_contracts,
                                "net_contracts": net_contracts,
                                "sl_contracts": str(sl_contracts) if sl_contracts is not None else None,
                                "manual_intervention_required": manual_intervention_required,
                            },
                            position_id=three_stage_post_tp1_sl_payload.get("position_id"),
                        )
                    logger.error(
                        "THREE_STAGE_POST_TP1_PROTECTIVE_SL_FAILED | position_id=%s side=%s sl_price=%s sl_message=%s market_exit_attempted=true market_exit_ok=%s market_exit_message=%s core_contracts=%s net_contracts=%s sl_contracts=%s manual_intervention_required=%s",
                        three_stage_post_tp1_sl_payload.get("position_id"),
                        side,
                        sl_price,
                        sl_message,
                        exit_ok,
                        exit_message,
                        core_contracts,
                        net_contracts,
                        sl_contracts,
                        manual_intervention_required,
                    )
            middle_runner_activation_recorded = False
            if middle_runner_sl_payload is not None:
                sl_price = middle_runner_sl_payload.get("protective_sl_price")
                sl_order_id = None
                sl_ok = False
                sl_message = "protective_sl_price_missing"
                if sl_price is not None and middle_runner_sl_payload.get("side") is not None:
                    sl_ok, sl_order_id, sl_message = await trader.place_middle_runner_protective_stop_with_retries(
                        middle_runner_sl_payload["side"],
                        middle_runner_sl_payload["contracts"],
                        float(sl_price),
                        retry_count=int(os.getenv("NEAR_TP_PROTECTIVE_SL_RETRY_COUNT", "3")),
                        retry_interval_seconds=float(os.getenv("NEAR_TP_PROTECTIVE_SL_RETRY_INTERVAL_SECONDS", "1")),
                    )
                if sl_ok:
                    old_sl_order_id = middle_runner_sl_payload.get("old_sl_order_id")
                    if old_sl_order_id and old_sl_order_id != sl_order_id:
                        await trader.cancel_middle_runner_protective_stop(old_sl_order_id)
                    async with state_lock:
                        strategy.state.middle_runner_protective_sl_order_id = sl_order_id
                        strategy.state.middle_runner_protective_sl_price = float(sl_price)
                        if middle_runner_sl_payload.get("reason") == "partial_size_mismatch_degraded":
                            strategy.state.middle_runner_size_mismatch_protected = True
                        save_state_payload = (execution_state.current_position_id, copy.deepcopy(strategy.state), execution_state.cash_before_position)
                    if hasattr(journal, "append"):
                        event_name = (
                            "MIDDLE_RUNNER_SIZE_MISMATCH_PROTECTED"
                            if middle_runner_sl_payload.get("reason") == "partial_size_mismatch_degraded"
                            else "MIDDLE_RUNNER_ACTIVATED"
                        )
                        journal.append(
                            event_name,
                            {
                                **(middle_runner_activation_payload or {}),
                                "side": middle_runner_sl_payload.get("side"),
                                "contracts": str(middle_runner_sl_payload.get("contracts")),
                                "core_contracts": middle_runner_sl_payload.get("core_contracts"),
                                "net_contracts": middle_runner_sl_payload.get("net_contracts"),
                                "sl_contracts": str(middle_runner_sl_payload.get("contracts")),
                                "protective_sl_price": sl_price,
                                "protective_sl_order_id": sl_order_id,
                                "reason": middle_runner_sl_payload.get("reason", "partial_tp_filled"),
                            },
                            position_id=middle_runner_sl_payload.get("position_id"),
                        )
                        if event_name == "MIDDLE_RUNNER_ACTIVATED":
                            middle_runner_activation_recorded = True
                    logger.warning(
                        "%s | position_id=%s side=%s core_contracts=%s net_contracts=%s sl_contracts=%s protective_sl_price=%s protective_sl_order_id=%s",
                        "MIDDLE_RUNNER_SIZE_MISMATCH_PROTECTED" if middle_runner_sl_payload.get("reason") == "partial_size_mismatch_degraded" else "MIDDLE_RUNNER_ACTIVATED",
                        middle_runner_sl_payload.get("position_id"),
                        middle_runner_sl_payload.get("side"),
                        middle_runner_sl_payload.get("core_contracts"),
                        middle_runner_sl_payload.get("net_contracts"),
                        middle_runner_sl_payload.get("contracts"),
                        sl_price,
                        sl_order_id,
                    )
                else:
                    side = middle_runner_sl_payload.get("side")
                    exit_ok, exit_message = (False, "side_missing")
                    if side is not None:
                        exit_ok, exit_message = await trader.market_exit_remaining_position_with_retries(
                            side,
                            retry_count=int(os.getenv("NEAR_TP_SL_FAIL_MARKET_EXIT_RETRY_COUNT", "3")),
                        )
                    async with state_lock:
                        execution_state.trading_halted = True
                        execution_state.halt_reason = "middle_runner_protective_sl_failure"
                    if hasattr(journal, "append"):
                        journal.append(
                            "MIDDLE_RUNNER_ORDER_WARNING",
                            {
                                "side": side,
                                "protective_sl_price": sl_price,
                                "reason": f"protective_sl_failed:{sl_message};market_exit_ok={exit_ok};{exit_message}",
                            },
                            position_id=middle_runner_sl_payload.get("position_id"),
                        )
                    logger.error(
                        "MIDDLE_RUNNER_ORDER_WARNING | reason=protective_sl_failed side=%s sl_price=%s sl_message=%s market_exit_ok=%s market_exit_message=%s",
                        side,
                        sl_price,
                        sl_message,
                        exit_ok,
                        exit_message,
                    )
            if (
                middle_runner_activation_payload is not None
                and not middle_runner_activation_recorded
                and middle_runner_sl_payload is None
                and hasattr(journal, "append")
            ):
                journal.append(
                    "MIDDLE_RUNNER_ACTIVATED",
                    {
                        **middle_runner_activation_payload,
                        "protective_sl_price": None,
                        "protective_sl_order_id": None,
                        "reason": "partial_tp_filled_protective_sl_disabled",
                    },
                    position_id=middle_runner_activation_payload.get("position_id"),
                )
            if record_flat_payload is not None:
                record_flat_payload.pop("near_tp_protective_sl_order_id", None)
                record_flat_payload.pop("middle_runner_protective_sl_order_id", None)
                record_flat_payload.pop("three_stage_post_tp1_protective_sl_order_id", None)
                record_flat_payload.pop("trend_runner_sl_order_id", None)
                journal.record_flat(**record_flat_payload)
                if rolling_loss_guard is not None and rolling_loss_guard.state is not None and rolling_loss_guard.state.enabled:
                    guard_now_ms = live_time_utils.utc_ms()
                    flat_equity = record_flat_payload.get("equity_after") or record_flat_payload.get("cash_after")
                    flat_event_id = (
                        record_flat_payload.get("position_id")
                        or (pending_flat_payload or {}).get("position_id")
                        or execution_state.current_position_id
                    )
                    decision = rolling_loss_guard.evaluate_after_flat(
                        now_ms=guard_now_ms,
                        flat_equity=float(flat_equity) if flat_equity is not None else None,
                        flat_event_id=str(flat_event_id) if flat_event_id else None,
                        has_position=False,
                    )
                    if decision.action is not None:
                        halt_reason = rolling_loss_halt_reason(decision.action)
                        critical_halt_preserved = False
                        if halt_reason is not None:
                            can_apply_rolling_halt = flat_previous_halt_reason in {
                                None,
                                "near_tp_exit_all_waiting_flat",
                                "near_tp_protected_sync_failed",
                                "trend_runner_market_exit_waiting_flat",
                                "sidecar_tp_place_failed_market_exit_waiting_flat",
                                "rolling_loss_soft_halt",
                                "rolling_loss_hard_halt",
                            }
                            critical_halt_preserved = halt_reason is not None and not can_apply_rolling_halt
                            if critical_halt_preserved:
                                logger.warning(
                                    "ROLLING_LOSS_GUARD_TRIGGERED_BUT_CRITICAL_HALT_PRESERVED | action=%s existing_halt_reason=%s loss_pct=%.6f halt_not_applied=true",
                                    decision.action,
                                    flat_previous_halt_reason,
                                    decision.loss_pct,
                                )
                            else:
                                async with state_lock:
                                    if (
                                        not execution_state.trading_halted
                                        or execution_state.halt_reason in ROLLING_LOSS_HALT_REASONS
                                        or execution_state.halt_reason is None
                                    ):
                                        execution_state.trading_halted = True
                                        execution_state.halt_reason = halt_reason
                                        execution_state.halt_until_ts_ms = decision.halt_until_ts_ms
                        payload = rolling_loss_guard_payload(decision.action, decision)
                        if critical_halt_preserved:
                            payload.update(
                                {
                                    "critical_halt_preserved": True,
                                    "existing_halt_reason": flat_previous_halt_reason,
                                    "rolling_loss_halt_not_applied": True,
                                }
                            )
                        await record_and_notify_rolling_loss_guard(
                            journal=journal,
                            email_sender=email_sender,
                            payload=payload,
                            email_enabled=rolling_loss_guard.config.email_enabled and not critical_halt_preserved,
                        )
            if clear_state:
                state_store.clear()
            if save_state_payload is not None:
                position_id, strategy_state, cash_before_position = save_state_payload
                state_store.save(LiveStateStore.from_strategy_state(position_id=position_id, symbol=trader.symbol, strategy_state=strategy_state, cash_before_position=cash_before_position))
            if rolling_loss_guard is not None and rolling_loss_guard.state is not None and rolling_loss_guard.state.enabled:
                guard_now_ms = live_time_utils.utc_ms()
                has_position_now = bool(position.has_position)
                async with state_lock:
                    halted = execution_state.trading_halted
                    halt_reason = execution_state.halt_reason
                if (
                    halted
                    and halt_reason in ROLLING_LOSS_HALT_REASONS
                    and rolling_loss_guard.should_resume(guard_now_ms, has_position_now)
                ):
                    rolling_loss_guard.mark_resumed(guard_now_ms, equity)
                    rolling_loss_guard.save()
                    async with state_lock:
                        if execution_state.halt_reason in ROLLING_LOSS_HALT_REASONS:
                            execution_state.trading_halted = False
                            execution_state.halt_reason = None
                            execution_state.halt_until_ts_ms = None
                    payload = rolling_loss_guard_state_payload(
                        "RESUME",
                        rolling_loss_guard,
                        "rolling_loss_cooldown_elapsed_and_account_flat",
                    )
                    await record_and_notify_rolling_loss_guard(
                        journal=journal,
                        email_sender=email_sender,
                        payload=payload,
                        email_enabled=rolling_loss_guard.config.email_enabled,
                    )
                    logger.warning(
                        "ROLLING_DRAWDOWN_GUARD_RESUMED | trading_halted=false reference_flat_equity=%.4f drawdown_pct=%.6f",
                        rolling_loss_guard.state.reference_flat_equity,
                        rolling_loss_guard.state.drawdown_pct,
                    )
                elif (
                    halted
                    and halt_reason in ROLLING_LOSS_HALT_REASONS
                    and rolling_loss_guard.state.halt_until_ts_ms is not None
                    and guard_now_ms >= rolling_loss_guard.state.halt_until_ts_ms
                    and has_position_now
                ):
                    logger.warning("ROLLING_LOSS_GUARD_RESUME_DELAYED | reason=position_open halt_until_ts_ms=%s", rolling_loss_guard.state.halt_until_ts_ms)
            if consecutive_failures > 0:
                logger.warning("ACCOUNT_SYNC_RECOVERED | failures=%s", consecutive_failures)
            consecutive_failures = 0
            first_failure_monotonic = 0.0
        except Exception as exc:
            now = time.monotonic()
            consecutive_failures += 1
            if first_failure_monotonic <= 0:
                first_failure_monotonic = now
            last_success_age_seconds = (
                max(now - account_snapshot.updated_monotonic, 0.0)
                if account_snapshot.updated_monotonic > 0
                else float("inf")
            )
            if now - last_failure_log >= sync_failure_log_interval_seconds:
                logger.warning(
                    "ACCOUNT_SYNC_FAILED | failures=%s error_type=%s error=%s last_success_age_seconds=%.1f",
                    consecutive_failures,
                    type(exc).__name__,
                    str(exc),
                    last_success_age_seconds,
                )
                last_failure_log = now
            if now - first_failure_monotonic >= sync_stale_warn_seconds and now - last_stale_log >= sync_failure_log_interval_seconds:
                logger.warning(
                    "ACCOUNT_SYNC_STALE | failures=%s last_success_age_seconds=%.1f risk=account_snapshot_may_be_stale",
                    consecutive_failures,
                    last_success_age_seconds,
                )
                last_stale_log = now


def expected_saved_state_remaining_qty(saved_state: Any) -> tuple[float, str]:  # type: ignore[no-untyped-def]
    """Estimate the current remaining position qty from saved state.

    total_entry_qty is the original entry quantity, which may have been
    reduced by partial TP / Three-Stage TP1 / Middle Runner first close.
    This function computes the *expected current remaining* qty using the
    best available information, prioritized:

      1. Sidecar: core_eth_qty + sidecar_open_qty(sidecar_legs)
      2. core_eth_qty (when > 0, even without sidecar)
      3. position_cost_remaining_qty
      4. Three-Stage deduction from consumed flags
      5. Middle Runner keep_ratio
      6. Fallback: total_entry_qty

    Returns (qty, source_label).
    """
    # ── 1. Sidecar: net position = core + sidecar open ──────────────────
    if bool(getattr(saved_state, "sidecar_enabled_for_position", False)):
        core_qty = float(getattr(saved_state, "core_eth_qty", 0.0) or 0.0)
        sidecar_legs = list(getattr(saved_state, "sidecar_legs", []) or [])
        sc_open = sidecar_open_qty(sidecar_legs)
        expected = core_qty + sc_open
        if expected > 0:
            return expected, "sidecar_core_plus_open"

    # ── 2. core_eth_qty (present even when sidecar is disabled) ────────
    core_qty = float(getattr(saved_state, "core_eth_qty", 0.0) or 0.0)
    if core_qty > 0:
        return core_qty, "core_eth_qty"

    # ── 3. position_cost_remaining_qty ──────────────────────────────────
    cost_remaining = float(getattr(saved_state, "position_cost_remaining_qty", 0.0) or 0.0)
    if cost_remaining > 0:
        return cost_remaining, "position_cost_remaining_qty"

    # ── 4. Three-Stage deduction ────────────────────────────────────────
    total_entry = float(getattr(saved_state, "total_entry_qty", 0.0) or 0.0)
    if (
        total_entry > 0
        and bool(getattr(saved_state, "three_stage_runner_enabled_for_position", False))
    ):
        tp2_consumed = bool(getattr(saved_state, "three_stage_tp2_consumed", False))
        tp1_consumed = bool(getattr(saved_state, "three_stage_tp1_consumed", False))
        runner_ratio = float(getattr(saved_state, "three_stage_runner_ratio", 0.0) or 0.0)
        tp2_ratio = float(getattr(saved_state, "three_stage_tp2_ratio", 0.0) or 0.0)
        if tp2_consumed and runner_ratio > 0:
            return total_entry * runner_ratio, "three_stage_runner"
        if tp1_consumed and (tp2_ratio + runner_ratio) > 0:
            return total_entry * (tp2_ratio + runner_ratio), "three_stage_after_tp1"

    # ── 5. Middle Runner active ─────────────────────────────────────────
    if total_entry > 0 and bool(getattr(saved_state, "middle_runner_active", False)):
        keep_ratio = float(getattr(saved_state, "middle_runner_keep_ratio", 0.0) or 0.0)
        if keep_ratio > 0:
            return total_entry * keep_ratio, "middle_runner_active"

    # ── 6. Fallback: total_entry_qty ────────────────────────────────────
    if total_entry > 0:
        return total_entry, "total_entry_qty"

    return 0.0, "none"


def trusted_startup_saved_state(  # type: ignore[no-untyped-def]
    saved_state: Any,
    startup_position: PositionSnapshot,
    max_avg_diff_pct: float | None = None,
    max_qty_diff_pct: float | None = None,
) -> Any:
    """Return saved_state only when it matches the current OKX position.

    A saved_state is trusted when ALL of these hold:
      1. saved_state is not None
      2. startup_position.has_position is True
      3. saved_state.side == startup_position.side
      4. saved_state.layers > 0
      5. avg_entry_price is within max_avg_diff_pct of OKX avg
      6. expected remaining qty is within max_qty_diff_pct of OKX qty

    The expected remaining qty accounts for partial exits (Three-Stage TP1,
    Middle Runner first close, etc.) via expected_saved_state_remaining_qty().

    Tolerance defaults are read from env vars with fallback values:
      STARTUP_SAVED_STATE_MAX_AVG_DIFF_PCT  → 0.003  (0.3%)
      STARTUP_SAVED_STATE_MAX_QTY_DIFF_PCT  → 0.05   (5%)
    """
    if max_avg_diff_pct is None:
        max_avg_diff_pct = float(os.getenv("STARTUP_SAVED_STATE_MAX_AVG_DIFF_PCT", "0.003"))
    if max_qty_diff_pct is None:
        max_qty_diff_pct = float(os.getenv("STARTUP_SAVED_STATE_MAX_QTY_DIFF_PCT", "0.05"))

    # ── basic identity checks ─────────────────────────────────────────
    if saved_state is None:
        return None
    if not startup_position.has_position:
        return None
    if getattr(saved_state, "side", None) != startup_position.side:
        logger.warning(
            "STARTUP_SAVED_STATE_UNTRUSTED | reason=side_mismatch saved_side=%s okx_side=%s",
            getattr(saved_state, "side", None),
            startup_position.side,
        )
        return None
    if int(getattr(saved_state, "layers", 0) or 0) <= 0:
        logger.warning(
            "STARTUP_SAVED_STATE_UNTRUSTED | reason=layers_zero_or_missing saved_layers=%s",
            getattr(saved_state, "layers", None),
        )
        return None

    # ── avg_entry check ───────────────────────────────────────────────
    saved_avg = float(getattr(saved_state, "avg_entry_price", 0.0) or 0.0)
    pos_avg = float(startup_position.avg_entry_price or 0.0)
    if saved_avg <= 0:
        logger.warning(
            "STARTUP_SAVED_STATE_UNTRUSTED | reason=avg_entry_missing_or_zero saved_avg=%s okx_avg=%.4f",
            getattr(saved_state, "avg_entry_price", None),
            pos_avg,
        )
        return None
    if pos_avg <= 0:
        logger.warning(
            "STARTUP_SAVED_STATE_UNTRUSTED | reason=okx_avg_entry_zero saved_avg=%.4f okx_avg=%.4f",
            saved_avg,
            pos_avg,
        )
        return None
    avg_diff = abs(saved_avg - pos_avg) / pos_avg
    if avg_diff > max_avg_diff_pct:
        logger.warning(
            "STARTUP_SAVED_STATE_UNTRUSTED | reason=avg_entry_mismatch saved_avg=%.4f okx_avg=%.4f diff_pct=%.6f max_diff_pct=%.6f",
            saved_avg,
            pos_avg,
            avg_diff,
            max_avg_diff_pct,
        )
        return None

    # ── size check ────────────────────────────────────────────────────
    expected_qty, qty_source = expected_saved_state_remaining_qty(saved_state)
    pos_qty = float(startup_position.eth_qty or 0.0)

    if expected_qty <= 0:
        logger.warning(
            "STARTUP_SAVED_STATE_UNTRUSTED | reason=qty_missing_or_zero expected_qty=%.8f qty_source=%s saved_total_entry_qty=%s saved_core_eth_qty=%s",
            expected_qty,
            qty_source,
            getattr(saved_state, "total_entry_qty", None),
            getattr(saved_state, "core_eth_qty", None),
        )
        return None
    if pos_qty <= 0:
        logger.warning(
            "STARTUP_SAVED_STATE_UNTRUSTED | reason=okx_qty_zero expected_qty=%.8f qty_source=%s okx_qty=%.8f",
            expected_qty,
            qty_source,
            pos_qty,
        )
        return None

    logger.info(
        "STARTUP_SAVED_STATE_QTY_EXPECTED | source=%s expected_qty=%.8f okx_qty=%.8f total_entry_qty=%s sidecar_enabled=%s",
        qty_source,
        expected_qty,
        pos_qty,
        getattr(saved_state, "total_entry_qty", None),
        bool(getattr(saved_state, "sidecar_enabled_for_position", False)),
    )

    qty_diff = abs(expected_qty - pos_qty) / pos_qty
    if qty_diff > max_qty_diff_pct:
        logger.warning(
            "STARTUP_SAVED_STATE_UNTRUSTED | reason=qty_mismatch expected_qty=%.8f okx_qty=%.8f diff_pct=%.6f max_diff_pct=%.6f qty_source=%s",
            expected_qty,
            pos_qty,
            qty_diff,
            max_qty_diff_pct,
            qty_source,
        )
        return None

    return saved_state


async def main() -> None:
    load_dotenv()
    if not live_config_helpers.live_trading_enabled():
        raise RuntimeError("LIVE_TRADING is not true. Refusing to start live runner.")

    monitor_config = BollBandBreakoutMonitorConfig.from_env()
    cvd_config = CvdTrackerConfig.from_env()
    email_sender = EmailSender()
    journal = LiveTradeJournal()
    rolling_loss_guard = RollingLossGuard.from_env()
    state_store = LiveStateStore()
    reporter = DailyTradeReporter(journal, email_sender)
    trader = Trader()
    await trader.start()
    try:
        await trader.initialize()
        sizer = SimplePositionSizer(SimplePositionSizerConfig.from_account_equity(trader.account_equity_usdt))
        strategy = BollCvdShockReclaimStrategy(BollCvdReclaimStrategyConfig.from_env(), sizer)
        startup_position = await trader.fetch_position_snapshot()
        startup_cash = await fetch_usdt_cash_balance(trader)
        rolling_loss_guard.load_or_initialize(live_time_utils.utc_ms(), trader.account_equity_usdt)
        journal.record_cash_baseline(
            source="startup",
            cash=startup_cash,
            equity=trader.account_equity_usdt,
            note="Live runner startup cash baseline.",
        )
    except Exception:
        await trader.close()
        raise
    current_position_id: str | None = None
    cash_before_position: float | None = None

    saved_state = state_store.load()
    trusted_saved_state = trusted_startup_saved_state(saved_state, startup_position)
    if startup_position.has_position:
        if trusted_saved_state is not None:
            restore_strategy_from_saved_state(strategy, trusted_saved_state)
            current_position_id = trusted_saved_state.position_id
            cash_before_position = trusted_saved_state.cash_before_position
        else:
            restore_strategy_from_position(strategy, startup_position, live_time_utils.utc_ms())
            current_position_id = journal.new_position_id(trader.symbol, startup_position.side or "UNKNOWN")
            cash_before_position = startup_cash
            journal.record_startup_recovery(
                position_id=current_position_id,
                symbol=trader.symbol,
                side=startup_position.side or "UNKNOWN",
                contracts=str(startup_position.contracts),
                eth_qty=startup_position.eth_qty,
                avg_entry=startup_position.avg_entry_price,
                cash=startup_cash,
                equity=trader.account_equity_usdt,
            )
        strategy.state.startup_force_tp_reconcile = True
        logger.warning(
            "STARTUP_FORCE_TP_RECONCILE_ARMED | position_id=%s side=%s layers=%s tp_plan=%s last_tp_update_candle_ts_ms=%s trusted_saved_state=%s",
            current_position_id,
            strategy.state.side,
            strategy.state.layers,
            getattr(strategy.state, "tp_plan", "SINGLE"),
            getattr(strategy.state, "last_tp_update_candle_ts_ms", 0),
            trusted_saved_state is not None,
        )
        state_store.save(LiveStateStore.from_strategy_state(position_id=current_position_id, symbol=trader.symbol, strategy_state=strategy.state, cash_before_position=cash_before_position))
    else:
        state_store.clear()

    cvd = CvdTracker(cvd_config)
    state_lock = asyncio.Lock()
    account_snapshot = live_runtime_types.AccountSnapshot(
        position=startup_position,
        cash=startup_cash,
        equity=trader.account_equity_usdt,
        updated_monotonic=time.monotonic(),
        updated_ts_ms=live_time_utils.utc_ms(),
        version=1,
    )
    execution_state = live_runtime_types.ExecutionState(
        current_position_id=current_position_id,
        cash_before_position=cash_before_position,
        trading_halted=False,
    )
    await apply_main_tp_startup_recovery(
        execution_state=execution_state,
        saved_state=trusted_saved_state,
        startup_position=startup_position,
        trader=trader,
        journal=journal,
    )
    await apply_sidecar_startup_recovery(
        strategy=strategy,
        execution_state=execution_state,
        saved_state=trusted_saved_state,
        startup_position=startup_position,
        trader=trader,
        journal=journal,
        state_store=state_store,
    )
    refresh_sidecar_state_totals(strategy.state, int(os.getenv("SIDECAR_MAX_LEGS", "10")))
    startup_core_position = build_core_position_view(
        startup_position,
        sidecar_open_qty(strategy.state.sidecar_legs),
        sidecar_open_contracts(strategy.state.sidecar_legs),
    )
    apply_core_position_view_to_state(strategy.state, startup_core_position)
    account_snapshot.position = startup_core_position
    if startup_position.has_position:
        state_store.save(LiveStateStore.from_strategy_state(position_id=current_position_id, symbol=trader.symbol, strategy_state=strategy.state, cash_before_position=cash_before_position))
    await apply_rolling_loss_guard_startup_state(
        rolling_loss_guard=rolling_loss_guard,
        execution_state=execution_state,
        has_position=startup_position.has_position,
        equity=trader.account_equity_usdt,
        now_ms=live_time_utils.utc_ms(),
        journal=journal,
        email_sender=email_sender,
    )
    apply_three_stage_startup_safety_gate(
        strategy=strategy,
        execution_state=execution_state,
        saved_state=trusted_saved_state,
        startup_position=startup_position,
        journal=journal,
        state_store=state_store,
        trader_symbol=trader.symbol,
    )
    strategy_tick_queue: asyncio.Queue[MarketTickEvent] = asyncio.Queue(maxsize=int(os.getenv("STRATEGY_TICK_QUEUE_MAXSIZE", "20000")))
    execution_queue: asyncio.Queue[live_runtime_types.TradeCommand] = asyncio.Queue(maxsize=int(os.getenv("EXECUTION_QUEUE_MAXSIZE", "1000")))
    position_sync_seconds = float(os.getenv("POSITION_SYNC_SECONDS", "5"))
    account_sync_seconds = float(os.getenv("ACCOUNT_SYNC_SECONDS", "60"))
    cash_log_min_delta_usdt = float(os.getenv("ACCOUNT_LOG_MIN_DELTA_USDT", "0.01"))
    market_tick_heartbeat_seconds = float(os.getenv("MARKET_TICK_HEARTBEAT_SECONDS", "60"))
    account_snapshot_stale_warn_seconds = float(os.getenv("ACCOUNT_SNAPSHOT_STALE_WARN_SECONDS", "30"))
    strategy_lag_warn_seconds = float(os.getenv("STRATEGY_TICK_LAG_WARN_SECONDS", "2"))
    execution_backlog_log_seconds = float(os.getenv("EXECUTION_QUEUE_BACKLOG_LOG_SECONDS", "30"))

    async def daily_report_loop() -> None:
        raw_time = os.getenv("DAILY_REPORT_TIME", "09:00")
        hour, minute = live_time_utils.parse_daily_report_time(raw_time)
        logger.info("Daily trade report loop started | DAILY_REPORT_TIME=%s", raw_time)
        while True:
            target = live_time_utils.next_daily_report_time(hour, minute)
            sleep_seconds = max((target - dt.datetime.now().astimezone()).total_seconds(), 1)
            await asyncio.sleep(sleep_seconds)
            try:
                context = build_report_context(
                    account_snapshot=account_snapshot,
                    execution_state=execution_state,
                )
                ok = await reporter.send_last_24h_report(context)
                if ok:
                    logger.info("Daily trade report sent successfully")
                else:
                    logger.error("Daily trade report failed")
            except Exception:
                logger.exception("Daily trade report loop failed")

    async def weekly_summary_loop() -> None:
        enabled = os.getenv("WEEKLY_SUMMARY_ENABLED", "true").strip().lower() in {"1", "true", "yes", "y", "on"}
        if not enabled:
            logger.info("Weekly overall summary loop disabled")
            return

        raw_time = os.getenv("WEEKLY_SUMMARY_TIME", "10:00")
        raw_weekday = os.getenv("WEEKLY_SUMMARY_WEEKDAY", "0")
        compact_after_success = os.getenv("WEEKLY_COMPACT_AFTER_SUCCESS", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
        hour, minute = live_time_utils.parse_weekly_report_time(raw_time)
        weekday = int(raw_weekday)
        if weekday < 0 or weekday > 6:
            raise ValueError(f"Invalid WEEKLY_SUMMARY_WEEKDAY={raw_weekday}")

        logger.info(
            "Weekly compaction config | WEEKLY_COMPACT_AFTER_SUCCESS=%s risk=enable_only_after_summary_merge_verified",
            compact_after_success,
        )
        logger.info(
            "Weekly overall summary loop started | WEEKLY_SUMMARY_WEEKDAY=%s WEEKLY_SUMMARY_TIME=%s",
            weekday,
            raw_time,
        )

        while True:
            target = live_time_utils.next_weekly_summary_time(hour, minute, weekday)
            sleep_seconds = max((target - dt.datetime.now().astimezone()).total_seconds(), 1)
            await asyncio.sleep(sleep_seconds)
            try:
                context = build_report_context(
                    account_snapshot=account_snapshot,
                    execution_state=execution_state,
                )
                ok = await reporter.send_overall_summary_report(context)
                if ok:
                    logger.info("Weekly overall summary report sent successfully")
                    if compact_after_success:
                        async with state_lock:
                            compact_position_id = execution_state.current_position_id
                        result = await asyncio.to_thread(
                            compact_after_weekly_summary,
                            journal,
                            target,
                            compact_position_id,
                        )
                        if result.archived_event_count > 0:
                            logger.warning(
                                "JOURNAL_COMPACTED | archived_event_count=%s retained_event_count=%s archive_path=%s",
                                result.archived_event_count,
                                result.retained_event_count,
                                result.archive_path,
                            )
                else:
                    logger.error("Weekly overall summary report failed")
            except Exception:
                logger.exception("Weekly overall summary report loop failed")

    async def on_market_tick(event: MarketTickEvent) -> None:
        await enqueue_strategy_tick(event, strategy_tick_queue, state_lock, execution_state)

    monitor = BollBandBreakoutMonitor(
        config=monitor_config,
        tick_handlers=[on_market_tick],
    )
    try:
        await asyncio.gather(
            account_position_sync_worker(
                state_lock=state_lock,
                account_snapshot=account_snapshot,
                execution_state=execution_state,
                trader=trader,
                sizer=sizer,
                strategy=strategy,
                journal=journal,
                state_store=state_store,
                position_sync_seconds=position_sync_seconds,
                account_sync_seconds=account_sync_seconds,
                cash_log_min_delta_usdt=cash_log_min_delta_usdt,
                rolling_loss_guard=rolling_loss_guard,
                email_sender=email_sender,
            ),
            strategy_tick_worker(
                strategy_tick_queue=strategy_tick_queue,
                execution_queue=execution_queue,
                state_lock=state_lock,
                account_snapshot=account_snapshot,
                execution_state=execution_state,
                cvd=cvd,
                strategy=strategy,
                heartbeat_seconds=market_tick_heartbeat_seconds,
                account_stale_warn_seconds=account_snapshot_stale_warn_seconds,
                strategy_lag_warn_seconds=strategy_lag_warn_seconds,
            ),
            execution_worker(
                execution_queue=execution_queue,
                state_lock=state_lock,
                execution_state=execution_state,
                account_snapshot=account_snapshot,
                trader=trader,
                strategy=strategy,
                journal=journal,
                state_store=state_store,
                email_sender=email_sender,
                backlog_log_seconds=execution_backlog_log_seconds,
            ),
            daily_report_loop(),
            weekly_summary_loop(),
            monitor.run_forever(),
        )
    finally:
        await trader.close()


if __name__ == "__main__":
    asyncio.run(main())
