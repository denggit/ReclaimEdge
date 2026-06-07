from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from src.execution.trader import Trader
from src.live import runtime_types as live_runtime_types
from src.live import time_utils as live_time_utils
from src.live.account_sync import flat_settlement_phase as account_sync_flat_settlement_phase
from src.live.account_sync import pre_core_position as account_sync_pre_core_position
from src.live.account_sync import protective_orders_phase as account_sync_protective_orders_phase
from src.live.account_sync import tp_progress_phase as account_sync_tp_progress_phase
from src.position_management import core_position_view as core_position_view_helpers
from src.position_management import tp_progress as tp_progress_helpers
from src.position_management.sidecar import force_close_runtime as sidecar_force_close_runtime
from src.position_management.sidecar import monitor_runtime as sidecar_monitor_runtime
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
    last_logged_position_key = (
        core_position_view_helpers.position_log_key(account_snapshot.position)
        if account_snapshot.position is not None
        else ("FLAT", "0", 0.0)
    )
    consecutive_failures = 0
    first_failure_monotonic = 0.0
    last_failure_log = 0.0
    last_stale_log = 0.0
    last_cash_event_log = 0.0
    last_flat_detected_monotonic = 0.0
    last_sidecar_status_check = 0.0
    sync_failure_log_interval_seconds = float(os.getenv("ACCOUNT_SYNC_FAILURE_LOG_INTERVAL_SECONDS", "60"))
    sync_stale_warn_seconds = float(os.getenv("ACCOUNT_SYNC_STALE_WARN_SECONDS", "180"))
    cash_transfer_detect_enabled = os.getenv("CASH_TRANSFER_DETECT_ENABLED", "true").strip().lower() in {"1", "true",
                                                                                                         "yes", "y",
                                                                                                         "on"}
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
            pre_core_result = await account_sync_pre_core_position.run_account_sync_pre_core_position_phase(
                state_lock=state_lock,
                account_snapshot=account_snapshot,
                execution_state=execution_state,
                trader=trader,
                sizer=sizer,
                strategy=strategy,
                journal=journal,
                state_store=state_store,
                now=now,
                last_account_sync=last_account_sync,
                account_sync_seconds=account_sync_seconds,
                cash_transfer_detect_enabled=cash_transfer_detect_enabled,
                cash_transfer_min_delta_usdt=cash_transfer_min_delta_usdt,
                cash_transfer_settle_seconds=cash_transfer_settle_seconds,
                cash_transfer_after_flat_cooldown_seconds=cash_transfer_after_flat_cooldown_seconds,
                cash_drift_min_delta_usdt=cash_drift_min_delta_usdt,
                cash_event_log_interval_seconds=cash_event_log_interval_seconds,
                cash_log_min_delta_usdt=cash_log_min_delta_usdt,
                last_logged_cash=last_logged_cash,
                last_logged_equity=last_logged_equity,
                last_cash_event_log=last_cash_event_log,
                last_flat_detected_monotonic=last_flat_detected_monotonic,
            )
            cash = pre_core_result.cash
            equity = pre_core_result.equity
            position = pre_core_result.position
            core_position = pre_core_result.core_position
            current_position_key = pre_core_result.current_position_key
            pending_order_count = pre_core_result.pending_order_count
            force_close_sidecar = pre_core_result.force_close_sidecar
            pending_flat_payload = pre_core_result.pending_flat_payload
            cash_transfer_payload = pre_core_result.cash_transfer_payload
            cash_drift_payload = pre_core_result.cash_drift_payload
            sidecar_reconciled_this_sync = pre_core_result.sidecar_reconciled_this_sync
            sidecar_state_changed_this_sync = pre_core_result.sidecar_state_changed_this_sync
            last_account_sync = pre_core_result.last_account_sync
            last_logged_cash = pre_core_result.last_logged_cash
            last_logged_equity = pre_core_result.last_logged_equity
            last_cash_event_log = pre_core_result.last_cash_event_log
            last_flat_detected_monotonic = pre_core_result.last_flat_detected_monotonic
            record_flat_payload: dict[str, Any] | None = None
            save_state_payload: tuple[str | None, StrategyPositionState, float | None] | None = None
            middle_runner_sl_payload: dict[str, Any] | None = None
            middle_runner_activation_payload: dict[str, Any] | None = None
            three_stage_post_tp1_sl_payload: dict[str, Any] | None = None
            three_stage_post_tp1_cancel_payload: dict[str, Any] | None = None
            three_stage_event_payload: dict[str, Any] | None = None
            middle_bucket_split_event_payload: dict[str, Any] | None = None
            middle_bucket_split_fast_protection_payload: dict[str, Any] | None = None
            clear_state = False
            flat_previous_halt_reason: str | None = None
            if pending_flat_payload is None and core_position.has_position:
                tp_progress_result = account_sync_tp_progress_phase.run_account_sync_tp_progress_phase(
                    account_snapshot=account_snapshot,
                    execution_state=execution_state,
                    trader=trader,
                    strategy=strategy,
                    journal=journal,
                    state_store=state_store,
                    position=position,
                    core_position=core_position,
                    current_position_key=current_position_key,
                    pending_order_count=pending_order_count,
                    last_logged_position_key=last_logged_position_key,
                )
                save_state_payload = tp_progress_result.save_state_payload
                middle_runner_sl_payload = tp_progress_result.middle_runner_sl_payload
                middle_runner_activation_payload = tp_progress_result.middle_runner_activation_payload
                three_stage_post_tp1_sl_payload = tp_progress_result.three_stage_post_tp1_sl_payload
                three_stage_post_tp1_cancel_payload = tp_progress_result.three_stage_post_tp1_cancel_payload
                three_stage_event_payload = tp_progress_result.three_stage_event_payload
                middle_bucket_split_event_payload = tp_progress_result.middle_bucket_split_event_payload
                middle_bucket_split_fast_protection_payload = tp_progress_result.middle_bucket_split_fast_protection_payload
                last_logged_position_key = tp_progress_result.last_logged_position_key

            if force_close_sidecar:
                await sidecar_force_close_runtime.force_close_sidecar_after_core_flat(
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

            sidecar_check_seconds = max(float(getattr(sizer.config, "sidecar_order_status_check_seconds", 5.0) or 5.0),
                                        0.0)
            if (
                    sidecar_check_seconds >= 0
                    and now - last_sidecar_status_check >= sidecar_check_seconds
                    and getattr(strategy.state, "sidecar_enabled_for_position", False)
                    and not sidecar_reconciled_this_sync
                    and pending_order_count == 0
            ):
                last_sidecar_status_check = now
                await sidecar_monitor_runtime.monitor_sidecar_orders_once(
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

            flat_settlement_result = await account_sync_flat_settlement_phase.prepare_account_sync_flat_settlement_phase(
                state_lock=state_lock,
                account_snapshot=account_snapshot,
                execution_state=execution_state,
                trader=trader,
                sizer=sizer,
                strategy=strategy,
                rolling_loss_guard=rolling_loss_guard,
                pending_flat_payload=pending_flat_payload,
                position=position,
                current_position_key=current_position_key,
                cash=cash,
                equity=equity,
                flat_balance_confirm_attempts=flat_balance_confirm_attempts,
                flat_balance_confirm_interval_seconds=flat_balance_confirm_interval_seconds,
                flat_balance_stable_delta_usdt=flat_balance_stable_delta_usdt,
                flat_balance_cash_equity_max_diff_usdt=flat_balance_cash_equity_max_diff_usdt,
                last_logged_cash=last_logged_cash,
                last_logged_equity=last_logged_equity,
                last_logged_position_key=last_logged_position_key,
            )
            cash = flat_settlement_result.cash
            equity = flat_settlement_result.equity
            record_flat_payload = flat_settlement_result.record_flat_payload
            clear_state = flat_settlement_result.clear_state
            flat_previous_halt_reason = flat_settlement_result.flat_previous_halt_reason
            last_logged_cash = flat_settlement_result.last_logged_cash
            last_logged_equity = flat_settlement_result.last_logged_equity
            last_logged_position_key = flat_settlement_result.last_logged_position_key

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
                tp_progress_helpers.append_three_stage_progress_journal_events(journal, three_stage_event_payload)
            protective_result = await account_sync_protective_orders_phase.run_account_sync_protective_orders_phase(
                state_lock=state_lock,
                execution_state=execution_state,
                trader=trader,
                strategy=strategy,
                journal=journal,
                state_store=state_store,
                save_state_payload=save_state_payload,
                three_stage_post_tp1_cancel_payload=three_stage_post_tp1_cancel_payload,
                three_stage_post_tp1_sl_payload=three_stage_post_tp1_sl_payload,
                middle_runner_sl_payload=middle_runner_sl_payload,
                middle_runner_activation_payload=middle_runner_activation_payload,
                middle_bucket_split_event_payload=middle_bucket_split_event_payload,
                middle_bucket_split_fast_protection_payload=middle_bucket_split_fast_protection_payload,
            )
            save_state_payload = protective_result.save_state_payload
            await account_sync_flat_settlement_phase.finalize_account_sync_flat_settlement_phase(
                state_lock=state_lock,
                execution_state=execution_state,
                journal=journal,
                email_sender=email_sender,
                state_store=state_store,
                rolling_loss_guard=rolling_loss_guard,
                record_flat_payload=record_flat_payload,
                pending_flat_payload=pending_flat_payload,
                flat_previous_halt_reason=flat_previous_halt_reason,
                clear_state=clear_state,
            )
            if save_state_payload is not None:
                position_id, strategy_state, cash_before_position = save_state_payload
                state_store.save(LiveStateStore.from_strategy_state(position_id=position_id, symbol=trader.symbol,
                                                                    strategy_state=strategy_state,
                                                                    cash_before_position=cash_before_position))
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
                    payload = rolling_loss_live_helpers.rolling_loss_guard_state_payload(
                        "RESUME",
                        rolling_loss_guard,
                        "rolling_loss_cooldown_elapsed_and_account_flat",
                    )
                    await rolling_loss_live_helpers.record_and_notify_rolling_loss_guard(
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
                    logger.warning("ROLLING_LOSS_GUARD_RESUME_DELAYED | reason=position_open halt_until_ts_ms=%s",
                                   rolling_loss_guard.state.halt_until_ts_ms)
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
