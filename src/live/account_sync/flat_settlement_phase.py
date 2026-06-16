from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from src.execution.trader import PositionSnapshot, Trader
from src.live import delayed_market_exit as dme
from src.live import runtime_types as live_runtime_types
from src.live import time_utils as live_time_utils
from src.live.account_sync import flat_balance as live_flat_balance
from src.live.account_sync.entry_sl_exit_classifier import classify_entry_sl_exit_for_cooldown
from src.reporting.live_state_store import LiveStateStore
from src.reporting.trade_journal import LiveTradeJournal
from src.risk import rolling_loss_live as rolling_loss_live_helpers
from src.risk.rolling_loss_guard import ROLLING_LOSS_HALT_REASONS, RollingLossGuard
from src.risk.simple_position_sizer import SimplePositionSizer
from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy
from src.utils.email_sender import EmailSender
from src.utils.log import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class AccountSyncFlatSettlementPrepareResult:
    cash: float
    equity: float
    record_flat_payload: dict[str, Any] | None
    clear_state: bool
    flat_previous_halt_reason: str | None
    last_logged_cash: float
    last_logged_equity: float
    last_logged_position_key: Any


async def prepare_account_sync_flat_settlement_phase(
        *,
        state_lock: asyncio.Lock,
        account_snapshot: live_runtime_types.AccountSnapshot,
        execution_state: live_runtime_types.ExecutionState,
        trader: Trader,
        sizer: SimplePositionSizer,
        strategy: BollCvdShockReclaimStrategy,
        rolling_loss_guard: RollingLossGuard | None,
        pending_flat_payload: dict[str, Any] | None,
        position: PositionSnapshot,
        current_position_key: Any,
        cash: float,
        equity: float,
        flat_balance_confirm_attempts: int,
        flat_balance_confirm_interval_seconds: float,
        flat_balance_stable_delta_usdt: float,
        flat_balance_cash_equity_max_diff_usdt: float,
        last_logged_cash: float,
        last_logged_equity: float,
        last_logged_position_key: Any,
) -> AccountSyncFlatSettlementPrepareResult:
    if pending_flat_payload is None:
        return AccountSyncFlatSettlementPrepareResult(
            cash=cash,
            equity=equity,
            record_flat_payload=None,
            clear_state=False,
            flat_previous_halt_reason=None,
            last_logged_cash=last_logged_cash,
            last_logged_equity=last_logged_equity,
            last_logged_position_key=last_logged_position_key,
        )

    try:
        settled = await live_flat_balance.fetch_settled_flat_balance(
            trader,
            attempts=flat_balance_confirm_attempts,
            interval_seconds=flat_balance_confirm_interval_seconds,
            stable_delta_usdt=flat_balance_stable_delta_usdt,
            cash_equity_max_diff_usdt=flat_balance_cash_equity_max_diff_usdt,
        )
    except Exception as exc:
        logger.exception(
            "FLAT_BALANCE_SETTLE_FAILED | falling back to latest account equity before FLAT journal")
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
    record_flat_payload: dict[str, Any] = {
        **pending_flat_payload,
        "cash_after": settled.cash,
        "equity_after": settled.equity,
        "delayed_market_exit_was_armed": getattr(strategy.state, "delayed_market_exit_armed", False),
        "delayed_market_exit_reason": getattr(strategy.state, "delayed_market_exit_reason", None),
        "delayed_market_exit_status": getattr(strategy.state, "delayed_market_exit_status", None),
        "delayed_market_exit_executed_ts_ms": getattr(strategy.state, "delayed_market_exit_executed_ts_ms", None),
        "delayed_market_exit_exit_attempt_count": getattr(strategy.state, "delayed_market_exit_exit_attempt_count", 0),
        "delayed_market_exit_cleared": True,
    }
    result_cash = settled.cash
    result_equity = settled.equity

    entry_sl_order_id = pending_flat_payload.get("entry_protective_sl_order_id")
    if entry_sl_order_id:
        try:
            await trader.cancel_protective_stop(entry_sl_order_id)
        except Exception:
            logger.warning("ENTRY_PROTECTIVE_SL_CANCEL_ON_FLAT | algoId=%s failed_unhandled", entry_sl_order_id)

    middle_runner_sl_order_id = pending_flat_payload.get("middle_runner_protective_sl_order_id")
    if middle_runner_sl_order_id:
        try:
            await trader.cancel_middle_runner_protective_stop(middle_runner_sl_order_id)
        except Exception:
            logger.warning("MIDDLE_RUNNER_CANCELLED | reason=flat_sl_cancel_failed algoId=%s",
                           middle_runner_sl_order_id)
    three_stage_post_tp1_sl_order_id = pending_flat_payload.get(
        "three_stage_post_tp1_protective_sl_order_id")
    if three_stage_post_tp1_sl_order_id:
        try:
            await trader.cancel_three_stage_post_tp1_protective_stop(three_stage_post_tp1_sl_order_id)
        except Exception:
            logger.warning(
                "THREE_STAGE_TP1_PROTECTIVE_SL_CANCELLED | reason=flat_sl_cancel_failed algoId=%s",
                three_stage_post_tp1_sl_order_id)
    trend_runner_sl_order_id = pending_flat_payload.get("trend_runner_sl_order_id")
    if trend_runner_sl_order_id:
        try:
            await trader.cancel_trend_runner_protective_stop(trend_runner_sl_order_id)
        except Exception:
            logger.warning("TREND_RUNNER_CANCELLED | reason=flat_sl_cancel_failed algoId=%s",
                           trend_runner_sl_order_id)
    middle_bucket_fast_sl_order_id = pending_flat_payload.get("middle_bucket_split_fast_sl_order_id")
    if middle_bucket_fast_sl_order_id:
        try:
            await trader.cancel_algo_order(middle_bucket_fast_sl_order_id)
        except Exception:
            logger.warning("MIDDLE_BUCKET_FAST_SL_CANCELLED | reason=flat_sl_cancel_failed algoId=%s",
                           middle_bucket_fast_sl_order_id)

    # ── Post-entry SL cooldown: classify exit via dedicated classifier ──
    # The decision is made here (after settled balance is visible) rather
    # than in pre_core_position so we can use actual realized PnL to
    # distinguish an entry protective SL loss from a TP fill or manual close.
    entry_sl_cooldown_candidate = bool(pending_flat_payload.get("entry_sl_cooldown_candidate", False))
    cash_before_pos = pending_flat_payload.get("cash_before_position")
    realized_delta: float | None = None
    if cash_before_pos is not None:
        try:
            realized_delta = float(settled.cash) - float(cash_before_pos)
        except (TypeError, ValueError):
            realized_delta = 0.0

    flat_side = str(pending_flat_payload.get("side") or "UNKNOWN")

    classification = classify_entry_sl_exit_for_cooldown(
        entry_sl_cooldown_candidate=entry_sl_cooldown_candidate,
        entry_protective_sl_order_id=pending_flat_payload.get("entry_protective_sl_order_id"),
        filled_order_id=pending_flat_payload.get("filled_order_id"),
        filled_algo_id=pending_flat_payload.get("filled_algo_id"),
        exit_reason=pending_flat_payload.get("exit_reason"),
        realized_delta=realized_delta,
        partial_tp_consumed=bool(pending_flat_payload.get("partial_tp_consumed", False)),
        three_stage_tp1_consumed=bool(pending_flat_payload.get("three_stage_tp1_consumed", False)),
        three_stage_tp2_consumed=bool(pending_flat_payload.get("three_stage_tp2_consumed", False)),
        trend_runner_exit_reason=pending_flat_payload.get("trend_runner_exit_reason"),
        manual_close_detected=bool(pending_flat_payload.get("manual_close_detected", False)),
        allow_loss_heuristic=bool(pending_flat_payload.get("allow_loss_heuristic", True)),
    )

    if classification.should_arm_cooldown:
        strategy.arm_post_entry_sl_cooldown(
            ts_ms=live_time_utils.utc_ms(),
            side=flat_side,
            reason=classification.reason,
        )
        if classification.confidence == "EXACT":
            logger.warning(
                "POST_ENTRY_SL_COOLDOWN_ARMED_EXACT | side=%s reason=%s confidence=%s "
                "cash_before=%.4f cash_after=%.4f realized=%.4f",
                flat_side, classification.reason, classification.confidence,
                float(cash_before_pos) if cash_before_pos else 0.0, settled.cash,
                realized_delta if realized_delta is not None else 0.0,
            )
        elif classification.confidence == "HEURISTIC":
            logger.warning(
                "POST_ENTRY_SL_COOLDOWN_ARMED_HEURISTIC | side=%s reason=%s confidence=%s "
                "cash_before=%.4f cash_after=%.4f realized=%.4f",
                flat_side, classification.reason, classification.confidence,
                float(cash_before_pos) if cash_before_pos else 0.0, settled.cash,
                realized_delta if realized_delta is not None else 0.0,
            )
    else:
        logger.info(
            "POST_ENTRY_SL_COOLDOWN_SKIPPED | side=%s reason=%s confidence=%s "
            "cash_before=%.4f cash_after=%.4f realized=%.4f",
            flat_side, classification.reason, classification.confidence,
            float(cash_before_pos) if cash_before_pos else 0.0, settled.cash,
            realized_delta if realized_delta is not None else 0.0,
        )

    async with state_lock:
        result_flat_previous_halt_reason = execution_state.halt_reason if execution_state.trading_halted else None
        # ── Preserve post-entry SL cooldown across flat state reset ──────
        saved_cooldown_until = int(getattr(strategy.state, "post_entry_sl_cooldown_until_ts_ms", 0) or 0)
        saved_cooldown_side = getattr(strategy.state, "post_entry_sl_cooldown_side", None)
        saved_cooldown_reason = getattr(strategy.state, "post_entry_sl_cooldown_reason", None)
        account_snapshot.position = position
        account_snapshot.cash = settled.cash
        account_snapshot.equity = settled.equity
        account_snapshot.updated_monotonic = time.monotonic()
        account_snapshot.updated_ts_ms = live_time_utils.utc_ms()
        account_snapshot.version += 1
        trader.account_equity_usdt = settled.equity
        sizer.update_account_equity(settled.equity)
        strategy.state = StrategyPositionState()
        # ── Restore post-entry SL cooldown onto fresh state ──────────────
        if saved_cooldown_until > 0:
            strategy.state.post_entry_sl_cooldown_until_ts_ms = saved_cooldown_until
            strategy.state.post_entry_sl_cooldown_side = saved_cooldown_side
            strategy.state.post_entry_sl_cooldown_reason = saved_cooldown_reason
        trader.mark_flat()
        flat_clearable_halt_reasons = {
            None,
            "trend_runner_market_exit_waiting_flat",
            "three_stage_post_tp1_sl_failed_market_exit_waiting_flat",
            "order_failure_delayed_market_exit_waiting_flat",
            "delayed_market_exit_waiting_flat",
            "rolling_loss_soft_halt",
            "rolling_loss_hard_halt",
            "three_stage_post_tp1_sl_failed_delayed_market_exit_armed",
            "middle_runner_sl_failed_delayed_market_exit_armed",
            "middle_bucket_fast_sl_failed_delayed_market_exit_armed",
            "middle_bucket_fast_sl_invalid_delayed_market_exit_armed",
            "core_tp_place_failed_delayed_market_exit_armed",
            # DME market exit failed: position flat clears halt.
            "order_failure_delayed_market_exit_failed",
        }
        preserve_critical_halt = (
                rolling_loss_guard is not None
                and result_flat_previous_halt_reason not in flat_clearable_halt_reasons
        )
        execution_state.trading_halted = preserve_critical_halt
        execution_state.halt_reason = result_flat_previous_halt_reason if preserve_critical_halt else None
        execution_state.halt_until_ts_ms = None
        execution_state.current_position_id = None
        execution_state.cash_before_position = None
        result_clear_state = True
        result_last_logged_cash = settled.cash
        result_last_logged_equity = settled.equity
        result_last_logged_position_key = current_position_key
    return AccountSyncFlatSettlementPrepareResult(
        cash=result_cash,
        equity=result_equity,
        record_flat_payload=record_flat_payload,
        clear_state=result_clear_state,
        flat_previous_halt_reason=result_flat_previous_halt_reason,
        last_logged_cash=result_last_logged_cash,
        last_logged_equity=result_last_logged_equity,
        last_logged_position_key=result_last_logged_position_key,
    )


async def finalize_account_sync_flat_settlement_phase(
        *,
        state_lock: asyncio.Lock,
        execution_state: live_runtime_types.ExecutionState,
        journal: LiveTradeJournal,
        email_sender: EmailSender | None,
        state_store: LiveStateStore,
        rolling_loss_guard: RollingLossGuard | None,
        record_flat_payload: dict[str, Any] | None,
        pending_flat_payload: dict[str, Any] | None,
        flat_previous_halt_reason: str | None,
        clear_state: bool,
) -> None:
    if record_flat_payload is not None:
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
                halt_reason = rolling_loss_live_helpers.rolling_loss_halt_reason(decision.action)
                critical_halt_preserved = False
                if halt_reason is not None:
                    can_apply_rolling_halt = flat_previous_halt_reason in {
                        None,
                        "trend_runner_market_exit_waiting_flat",
                        "order_failure_delayed_market_exit_waiting_flat",
                        "delayed_market_exit_waiting_flat",
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
                payload = rolling_loss_live_helpers.rolling_loss_guard_payload(decision.action, decision)
                if critical_halt_preserved:
                    payload.update(
                        {
                            "critical_halt_preserved": True,
                            "existing_halt_reason": flat_previous_halt_reason,
                            "rolling_loss_halt_not_applied": True,
                        }
                    )
                await rolling_loss_live_helpers.record_and_notify_rolling_loss_guard(
                    journal=journal,
                    email_sender=email_sender,
                    payload=payload,
                    email_enabled=rolling_loss_guard.config.email_enabled and not critical_halt_preserved,
                )

    if clear_state:
        state_store.clear()
