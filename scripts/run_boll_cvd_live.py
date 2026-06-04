from __future__ import annotations

import asyncio
import copy
import datetime as dt
import html
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from src.execution.trader import PositionSnapshot, Trader  # noqa: E402
from src.indicators.cvd_tracker import CvdTracker, CvdTrackerConfig  # noqa: E402
from src.monitors.boll_band_breakout_monitor import (  # noqa: E402
    BollBandBreakoutMonitor,
    BollBandBreakoutMonitorConfig,
    MarketTickEvent,
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


def live_trading_enabled() -> bool:
    return os.getenv("LIVE_TRADING", "false").strip().lower() in {"1", "true", "yes", "y", "on"}


def format_ts_ms(ts_ms: int) -> str:
    return dt.datetime.fromtimestamp(ts_ms / 1000).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


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
    halt_until_text = format_ts_ms(halt_until) if isinstance(halt_until, int) else "-"
    threshold = payload.get("threshold_pct")
    threshold_text = f"{float(threshold) * 100:.2f}%" if threshold is not None else "-"
    content = f"""
<div style="font-family: Arial, Helvetica, sans-serif; line-height: 1.55; color: #222; max-width: 760px;">
  <h2>{html.escape(subject)}</h2>
  <p>This guard never force-closes an open position; this event was evaluated only after the account reached FLAT.</p>
  <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">baseline_equity</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{float(payload.get("baseline_equity") or 0.0):.4f}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">window_start</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{html.escape(format_ts_ms(int(payload.get("window_start_ts_ms") or 0)))}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">window_end</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{html.escape(format_ts_ms(int(payload.get("window_end_ts_ms") or 0)))}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">rolling_realized_pnl</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{float(payload.get("rolling_realized_pnl") or 0.0):.4f}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">loss_usdt</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{float(payload.get("loss_usdt") or 0.0):.4f}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">loss_pct</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{float(payload.get("loss_pct") or 0.0) * 100:.2f}%</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">threshold</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{html.escape(threshold_text)}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">halt_until</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{html.escape(halt_until_text)}</td></tr>
  </table>
  <p><strong>Reason:</strong> {html.escape(str(payload.get("reason") or action))}</p>
</div>
""".strip()
    return subject, content


def rolling_loss_guard_payload(action: str, decision: RollingLossGuardDecision) -> dict[str, Any]:
    state = decision.state
    return {
        "action": action,
        "window_start_ts_ms": state.window_start_ts_ms,
        "window_end_ts_ms": state.window_end_ts_ms,
        "baseline_equity": state.baseline_equity,
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
        "window_start_ts_ms": state.window_start_ts_ms,
        "window_end_ts_ms": state.window_end_ts_ms,
        "baseline_equity": state.baseline_equity,
        "rolling_realized_pnl": state.last_window_realized_pnl,
        "loss_usdt": max(0.0, -state.last_window_realized_pnl),
        "loss_pct": state.last_loss_pct,
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
        rolling_loss_guard.save()
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
    elif now_ms >= rolling_loss_guard.state.window_end_ts_ms and not has_position:
        rolling_loss_guard.reset_window(now_ms, equity)
        rolling_loss_guard.save()
        journal.record_rolling_loss_guard(
            **rolling_loss_guard_state_payload(
                "WINDOW_RESET",
                rolling_loss_guard,
                "startup_rolling_loss_window_expired_and_account_flat",
            )
        )


def parse_daily_report_time(value: str) -> tuple[int, int]:
    hour_text, minute_text = value.strip().split(":", 1)
    hour = int(hour_text)
    minute = int(minute_text)
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError(f"Invalid DAILY_REPORT_TIME={value}")
    return hour, minute


def parse_weekly_report_time(value: str) -> tuple[int, int]:
    return parse_daily_report_time(value)


def next_daily_report_time(hour: int, minute: int) -> dt.datetime:
    now = dt.datetime.now().astimezone()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += dt.timedelta(days=1)
    return target


def next_weekly_summary_time(hour: int, minute: int, weekday: int = 0) -> dt.datetime:
    now = dt.datetime.now().astimezone()
    days_ahead = weekday - now.weekday()
    target_date = now.date() + dt.timedelta(days=days_ahead)
    target = dt.datetime.combine(target_date, dt.time(hour, minute), tzinfo=now.tzinfo)
    if target <= now:
        target += dt.timedelta(days=7)
    return target


def build_report_context(
    *,
    account_snapshot: AccountSnapshot,
    execution_state: ExecutionState,
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
    event_time = format_ts_ms(intent.ts_ms)
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
    now_ms = int(now_ms or utc_ms())
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
        partial_tp_price=getattr(saved_state, "partial_tp_price", None),
        partial_tp_ratio=getattr(saved_state, "partial_tp_ratio", 0.0),
        tp_plan=tp_plan,
        partial_tp_consumed=getattr(saved_state, "partial_tp_consumed", False),
        last_order_ts_ms=saved_state.last_order_ts_ms,
        first_entry_ts_ms=getattr(saved_state, "first_entry_ts_ms", 0),
        last_tp_update_ts_ms=saved_state.last_tp_update_ts_ms,
        last_tp_update_candle_ts_ms=saved_state.last_tp_update_candle_ts_ms,
        total_entry_qty=saved_state.total_entry_qty,
        total_entry_notional=saved_state.total_entry_notional,
        avg_entry_price=saved_state.avg_entry_price,
        breakeven_price=saved_state.breakeven_price,
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


async def fetch_usdt_cash_balance(trader: Trader) -> float:
    res = await trader.request("GET", "/api/v5/account/balance?ccy=USDT")
    data = res.get("data", [])
    if not data:
        return 0.0
    for item in data[0].get("details", []):
        if item.get("ccy") == "USDT":
            return float(item.get("cashBal") or item.get("availBal") or item.get("availEq") or item.get("eq") or 0.0)
    return float(data[0].get("totalEq") or 0.0)


@dataclass(frozen=True)
class SettledFlatBalance:
    cash: float
    equity: float
    attempts: int
    stable: bool
    reason: str


async def fetch_settled_flat_balance(
    trader: Trader,
    *,
    attempts: int,
    interval_seconds: float,
    stable_delta_usdt: float,
    cash_equity_max_diff_usdt: float,
) -> SettledFlatBalance:
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
                return SettledFlatBalance(
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
            return SettledFlatBalance(
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
        return SettledFlatBalance(
            cash=last_equity,
            equity=last_equity,
            attempts=attempts,
            stable=False,
            reason="fallback_to_equity_after_timeout",
        )
    return SettledFlatBalance(
        cash=last_cash,
        equity=last_equity,
        attempts=attempts,
        stable=False,
        reason="position_not_flat_after_timeout",
    )


@dataclass
class AccountSnapshot:
    position: PositionSnapshot | None
    cash: float
    equity: float
    updated_monotonic: float
    updated_ts_ms: int
    version: int = 0


@dataclass
class ExecutionState:
    current_position_id: str | None
    cash_before_position: float | None
    trading_halted: bool = False
    last_order_ts_ms: int = 0
    pending_order_count: int = 0
    halt_reason: str | None = None
    halt_until_ts_ms: int | None = None


@dataclass(frozen=True)
class TradeCommand:
    intent: TradeIntent
    strategy_state_snapshot: StrategyPositionState
    tick_ts_ms: int
    created_monotonic: float
    account_snapshot_updated_ts_ms: int
    reason: str


@dataclass(frozen=True)
class ExecutionReport:
    command: TradeCommand
    result: Any | None
    ok: bool
    error: Exception | None
    entry_may_be_live: bool
    created_monotonic: float
    finished_monotonic: float


def utc_ms() -> int:
    return int(time.time() * 1000)


def _parse_optional_float(value: Any) -> float | None:
    try:
        if value in {None, ""}:
            return None
        return float(value)
    except Exception:
        return None


def queue_log_level(queue_size: int) -> int | None:
    if queue_size < 500:
        return None
    if queue_size < 2000:
        return logging.INFO
    if queue_size < 8000:
        return logging.WARNING
    return logging.ERROR


def queue_oldest_command_age_seconds(queue: asyncio.Queue[TradeCommand]) -> float:
    try:
        oldest = queue._queue[0]  # type: ignore[attr-defined]
    except Exception:
        return 0.0
    return max(time.monotonic() - oldest.created_monotonic, 0.0)


async def enqueue_strategy_tick(
    event: MarketTickEvent,
    strategy_tick_queue: asyncio.Queue[MarketTickEvent],
    state_lock: asyncio.Lock,
    execution_state: ExecutionState,
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
    command: TradeCommand,
    execution_queue: asyncio.Queue[TradeCommand],
    state_lock: asyncio.Lock,
    execution_state: ExecutionState,
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
    execution_queue: asyncio.Queue[TradeCommand],
    state_lock: asyncio.Lock,
    account_snapshot: AccountSnapshot,
    execution_state: ExecutionState,
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
                command = TradeCommand(
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


async def execution_worker(
    *,
    execution_queue: asyncio.Queue[TradeCommand],
    state_lock: asyncio.Lock,
    execution_state: ExecutionState,
    account_snapshot: AccountSnapshot,
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
                        if _parse_optional_float(getattr(result, "protective_sl_price", "")) is not None:
                            strategy.state.middle_runner_protective_sl_price = _parse_optional_float(result.protective_sl_price)
                    if getattr(command.intent, "trend_runner_active", False):
                        if getattr(result, "protective_sl_order_id", None):
                            strategy.state.trend_runner_sl_order_id = result.protective_sl_order_id
                        if _parse_optional_float(getattr(result, "protective_sl_price", "")) is not None:
                            strategy.state.trend_runner_sl_price = _parse_optional_float(result.protective_sl_price)
                        strategy.state.trend_runner_tp_order_id = result.tp_order_id
                    if getattr(command.intent, "three_stage_post_tp1_protective_sl_price", None) is not None and getattr(command.intent, "three_stage_tp1_consumed", False):
                        if getattr(result, "protective_sl_order_id", None):
                            strategy.state.three_stage_post_tp1_protective_sl_order_id = result.protective_sl_order_id
                        if _parse_optional_float(getattr(result, "protective_sl_price", "")) is not None:
                            strategy.state.three_stage_post_tp1_protective_sl_price = _parse_optional_float(result.protective_sl_price)
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
                        strategy.state.near_tp_protective_sl_price = getattr(command.intent, "near_tp_protective_sl_price", None) or _parse_optional_float(getattr(result, "protective_sl_price", ""))
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
    command: TradeCommand,
    result: Any | None,
    error: Exception,
    state_lock: asyncio.Lock,
    execution_state: ExecutionState,
    trader: Trader,
    strategy: BollCvdShockReclaimStrategy,
    journal: LiveTradeJournal,
    email_sender: EmailSender,
) -> ExecutionReport:
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
            if command.intent.intent_type == "NEAR_TP_REDUCE" and getattr(result, "reduce_filled", False):
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

    return ExecutionReport(
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
    account_snapshot: AccountSnapshot,
    execution_state: ExecutionState,
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
            current_position_key = position_log_key(position)
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
            clear_state = False
            flat_previous_halt_reason: str | None = None
            async with state_lock:
                pending_order_count = execution_state.pending_order_count
                flat_transition_detected = (
                    pending_order_count == 0
                    and not position.has_position
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
                    account_snapshot.position = position
                    account_snapshot.cash = cash
                    account_snapshot.equity = equity
                    account_snapshot.updated_monotonic = time.monotonic()
                    account_snapshot.updated_ts_ms = utc_ms()
                    account_snapshot.version += 1
                    trader.account_equity_usdt = equity
                    sizer.update_account_equity(equity)

                    cash_delta = cash - last_logged_cash
                    seconds_since_last_order = (
                        cash_transfer_settle_seconds
                        if execution_state.last_order_ts_ms == 0
                        else max((utc_ms() - execution_state.last_order_ts_ms) / 1000, 0.0)
                    )
                    unsafe_reasons = []
                    if pending_order_count > 0:
                        unsafe_reasons.append("pending_order")
                    if position.has_position:
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
                        and not position.has_position
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

                if not flat_transition_detected and position.has_position:
                    trader.position_contracts = position.contracts
                    if pending_order_count == 0:
                        middle_runner_activated = mark_middle_runner_active_if_position_reduced(strategy, position)
                        three_stage_event = mark_three_stage_progress_if_position_reduced(strategy, position, utc_ms())
                        mark_partial_tp_consumed_if_position_reduced(strategy, position)
                        sync_strategy_cost_from_position(strategy, position)
                        if three_stage_event is not None:
                            if three_stage_event in {"TP1", "TP1_TP2"} and three_stage_event != "TP1_TP2":
                                config = getattr(strategy, "config", None)
                                if bool(getattr(config, "three_stage_post_tp1_protective_sl_enabled", True)):
                                    post_tp1_boll = three_stage_post_tp1_boll(strategy)
                                    current_price = getattr(post_tp1_boll, "middle", 0.0) if post_tp1_boll is not None else 0.0
                                    protective_sl = (
                                        strategy._calculate_three_stage_post_tp1_protective_sl(position.side, current_price, post_tp1_boll)
                                        if post_tp1_boll is not None and position.side is not None
                                        else None
                                    )
                                    strategy.state.three_stage_post_tp1_protective_sl_price = protective_sl
                                    three_stage_post_tp1_sl_payload = {
                                        "position_id": execution_state.current_position_id,
                                        "side": position.side,
                                        "contracts": position.contracts,
                                        "protective_sl_price": protective_sl,
                                        "old_sl_order_id": getattr(strategy.state, "three_stage_post_tp1_protective_sl_order_id", None),
                                        "reason": "three_stage_tp1_filled",
                                    }
                            if three_stage_event in {"TP2", "TP1_TP2"}:
                                three_stage_post_tp1_cancel_payload = {
                                    "position_id": execution_state.current_position_id,
                                    "side": position.side,
                                    "protective_sl_order_id": getattr(strategy.state, "three_stage_post_tp1_protective_sl_order_id", None),
                                    "protective_sl_price": getattr(strategy.state, "three_stage_post_tp1_protective_sl_price", None),
                                }
                            three_stage_event_payload = {
                                "event": three_stage_event,
                                "position_id": execution_state.current_position_id,
                                "side": position.side,
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
                                "side": position.side,
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
                                    strategy._calculate_middle_runner_protective_sl(position.side, current_price, runner_boll)
                                    if runner_boll is not None and position.side is not None
                                    else None
                                )
                                strategy.state.middle_runner_protective_sl_price = protective_sl
                                middle_runner_sl_payload = {
                                    "position_id": execution_state.current_position_id,
                                    "side": position.side,
                                    "contracts": position.contracts,
                                    "protective_sl_price": protective_sl,
                                    "old_sl_order_id": getattr(strategy.state, "middle_runner_protective_sl_order_id", None),
                                    "reason": "partial_tp_filled",
                                }
                        elif middle_runner_size_mismatch_needs_degraded_protection(strategy, position):
                            runner_boll = middle_runner_activation_boll(strategy)
                            current_price = getattr(runner_boll, "middle", 0.0) if runner_boll is not None else 0.0
                            protective_sl = (
                                strategy._calculate_middle_runner_protective_sl(position.side, current_price, runner_boll)
                                if runner_boll is not None and position.side is not None
                                else None
                            )
                            strategy.state.middle_runner_protective_sl_price = protective_sl
                            middle_runner_sl_payload = {
                                "position_id": execution_state.current_position_id,
                                "side": position.side,
                                "contracts": position.contracts,
                                "protective_sl_price": protective_sl,
                                "old_sl_order_id": getattr(strategy.state, "middle_runner_protective_sl_order_id", None),
                                "reason": "partial_size_mismatch_degraded",
                            }
                            middle_runner_activation_payload = {
                                "position_id": execution_state.current_position_id,
                                "side": position.side,
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
                                position.side,
                                position.contracts,
                                position.avg_entry_price,
                            )
                        save_state_payload = (execution_state.current_position_id, copy.deepcopy(strategy.state), execution_state.cash_before_position)
                        if current_position_key != last_logged_position_key:
                            logger.info(
                                "POSITION_SYNC_CHANGED | side=%s contracts=%s avg_entry=%.4f eth_qty=%.6f strategy_layers=%s",
                                position.side,
                                position.contracts,
                                position.avg_entry_price,
                                position.eth_qty,
                                strategy.state.layers,
                            )
                            last_logged_position_key = current_position_key

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
                    settled = SettledFlatBalance(
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
                    account_snapshot.updated_ts_ms = utc_ms()
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
            if cash_drift_payload is not None:
                journal.record_account_cash_drift(**cash_drift_payload)
            if three_stage_event_payload is not None and hasattr(journal, "append"):
                append_three_stage_progress_journal_events(journal, three_stage_event_payload)
            if three_stage_post_tp1_cancel_payload is not None:
                old_order_id = three_stage_post_tp1_cancel_payload.get("protective_sl_order_id")
                cancel_ok = True
                if old_order_id:
                    cancel_ok = await trader.cancel_three_stage_post_tp1_protective_stop(old_order_id)
                async with state_lock:
                    strategy.state.three_stage_post_tp1_protective_sl_order_id = None
                    strategy.state.three_stage_post_tp1_protective_sl_price = None
                    strategy.state.three_stage_post_tp1_protected = False
                    save_state_payload = (execution_state.current_position_id, copy.deepcopy(strategy.state), execution_state.cash_before_position)
                if hasattr(journal, "append"):
                    journal.append(
                        "THREE_STAGE_TP1_PROTECTIVE_SL_CANCELLED_ON_TP2",
                        {
                            **three_stage_post_tp1_cancel_payload,
                            "cancel_ok": cancel_ok,
                            "reason": "three_stage_tp2_filled",
                        },
                        position_id=three_stage_post_tp1_cancel_payload.get("position_id"),
                    )
                logger.warning(
                    "THREE_STAGE_TP1_PROTECTIVE_SL_CANCELLED_ON_TP2 | position_id=%s algoId=%s cancel_ok=%s",
                    three_stage_post_tp1_cancel_payload.get("position_id"),
                    old_order_id,
                    cancel_ok,
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
                                "protective_sl_price": sl_price,
                                "protective_sl_order_id": sl_order_id,
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
                        "THREE_STAGE_TP1_PROTECTIVE_SL_PLACED | position_id=%s side=%s contracts=%s protective_sl_price=%s protective_sl_order_id=%s retry_config=near_tp",
                        three_stage_post_tp1_sl_payload.get("position_id"),
                        three_stage_post_tp1_sl_payload.get("side"),
                        three_stage_post_tp1_sl_payload.get("contracts"),
                        sl_price,
                        sl_order_id,
                    )
                else:
                    async with state_lock:
                        execution_state.trading_halted = True
                        execution_state.halt_reason = "three_stage_post_tp1_protective_sl_failure"
                    if hasattr(journal, "append"):
                        journal.append(
                            "THREE_STAGE_POST_TP1_PROTECTIVE_SL_FAILED",
                            {
                                "position_id": three_stage_post_tp1_sl_payload.get("position_id"),
                                "side": three_stage_post_tp1_sl_payload.get("side"),
                                "protective_sl_price": sl_price,
                                "reason": sl_message,
                                "trading_halted": True,
                                "retry_config": "NEAR_TP_PROTECTIVE_SL_RETRY_COUNT/NEAR_TP_PROTECTIVE_SL_RETRY_INTERVAL_SECONDS",
                            },
                            position_id=three_stage_post_tp1_sl_payload.get("position_id"),
                        )
                    logger.error(
                        "THREE_STAGE_POST_TP1_PROTECTIVE_SL_FAILED | position_id=%s side=%s sl_price=%s sl_message=%s trading_halted=true retry_config=near_tp",
                        three_stage_post_tp1_sl_payload.get("position_id"),
                        three_stage_post_tp1_sl_payload.get("side"),
                        sl_price,
                        sl_message,
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
                                "protective_sl_price": sl_price,
                                "protective_sl_order_id": sl_order_id,
                                "reason": middle_runner_sl_payload.get("reason", "partial_tp_filled"),
                            },
                            position_id=middle_runner_sl_payload.get("position_id"),
                        )
                        if event_name == "MIDDLE_RUNNER_ACTIVATED":
                            middle_runner_activation_recorded = True
                    logger.warning(
                        "%s | position_id=%s side=%s protective_sl_price=%s protective_sl_order_id=%s",
                        "MIDDLE_RUNNER_SIZE_MISMATCH_PROTECTED" if middle_runner_sl_payload.get("reason") == "partial_size_mismatch_degraded" else "MIDDLE_RUNNER_ACTIVATED",
                        middle_runner_sl_payload.get("position_id"),
                        middle_runner_sl_payload.get("side"),
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
                    guard_now_ms = utc_ms()
                    decision = rolling_loss_guard.evaluate_after_flat(
                        now_ms=guard_now_ms,
                        journal_events=journal.load_events(),
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
                guard_now_ms = utc_ms()
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
                    logger.warning("ROLLING_LOSS_GUARD_RESUMED | trading_halted=false baseline_equity=%.4f", equity)
                elif (
                    halted
                    and halt_reason in ROLLING_LOSS_HALT_REASONS
                    and rolling_loss_guard.state.halt_until_ts_ms is not None
                    and guard_now_ms >= rolling_loss_guard.state.halt_until_ts_ms
                    and has_position_now
                ):
                    logger.warning("ROLLING_LOSS_GUARD_RESUME_DELAYED | reason=position_open halt_until_ts_ms=%s", rolling_loss_guard.state.halt_until_ts_ms)
                elif (
                    not halted
                    and rolling_loss_guard.should_reset_expired_window(guard_now_ms, has_position_now)
                ):
                    rolling_loss_guard.reset_window(guard_now_ms, equity)
                    rolling_loss_guard.save()
                    journal.record_rolling_loss_guard(
                        **rolling_loss_guard_state_payload(
                            "WINDOW_RESET",
                            rolling_loss_guard,
                            "rolling_loss_window_expired_and_account_flat",
                        )
                    )
                    logger.info("ROLLING_LOSS_GUARD_WINDOW_RESET | baseline_equity=%.4f", equity)
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


async def main() -> None:
    load_dotenv()
    if not live_trading_enabled():
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
        rolling_loss_guard.load_or_initialize(utc_ms(), trader.account_equity_usdt)
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
    if startup_position.has_position:
        if saved_state and saved_state.side == startup_position.side and saved_state.layers > 0:
            restore_strategy_from_saved_state(strategy, saved_state)
            current_position_id = saved_state.position_id
            cash_before_position = saved_state.cash_before_position
        else:
            restore_strategy_from_position(strategy, startup_position, utc_ms())
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
        state_store.save(LiveStateStore.from_strategy_state(position_id=current_position_id, symbol=trader.symbol, strategy_state=strategy.state, cash_before_position=cash_before_position))
    else:
        state_store.clear()

    cvd = CvdTracker(cvd_config)
    state_lock = asyncio.Lock()
    account_snapshot = AccountSnapshot(
        position=startup_position,
        cash=startup_cash,
        equity=trader.account_equity_usdt,
        updated_monotonic=time.monotonic(),
        updated_ts_ms=utc_ms(),
        version=1,
    )
    execution_state = ExecutionState(
        current_position_id=current_position_id,
        cash_before_position=cash_before_position,
        trading_halted=False,
    )
    await apply_rolling_loss_guard_startup_state(
        rolling_loss_guard=rolling_loss_guard,
        execution_state=execution_state,
        has_position=startup_position.has_position,
        equity=trader.account_equity_usdt,
        now_ms=utc_ms(),
        journal=journal,
        email_sender=email_sender,
    )
    strategy_tick_queue: asyncio.Queue[MarketTickEvent] = asyncio.Queue(maxsize=int(os.getenv("STRATEGY_TICK_QUEUE_MAXSIZE", "20000")))
    execution_queue: asyncio.Queue[TradeCommand] = asyncio.Queue(maxsize=int(os.getenv("EXECUTION_QUEUE_MAXSIZE", "1000")))
    position_sync_seconds = float(os.getenv("POSITION_SYNC_SECONDS", "5"))
    account_sync_seconds = float(os.getenv("ACCOUNT_SYNC_SECONDS", "60"))
    cash_log_min_delta_usdt = float(os.getenv("ACCOUNT_LOG_MIN_DELTA_USDT", "0.01"))
    market_tick_heartbeat_seconds = float(os.getenv("MARKET_TICK_HEARTBEAT_SECONDS", "60"))
    account_snapshot_stale_warn_seconds = float(os.getenv("ACCOUNT_SNAPSHOT_STALE_WARN_SECONDS", "30"))
    strategy_lag_warn_seconds = float(os.getenv("STRATEGY_TICK_LAG_WARN_SECONDS", "2"))
    execution_backlog_log_seconds = float(os.getenv("EXECUTION_QUEUE_BACKLOG_LOG_SECONDS", "30"))

    async def daily_report_loop() -> None:
        raw_time = os.getenv("DAILY_REPORT_TIME", "09:00")
        hour, minute = parse_daily_report_time(raw_time)
        logger.info("Daily trade report loop started | DAILY_REPORT_TIME=%s", raw_time)
        while True:
            target = next_daily_report_time(hour, minute)
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
        hour, minute = parse_weekly_report_time(raw_time)
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
            target = next_weekly_summary_time(hour, minute, weekday)
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
