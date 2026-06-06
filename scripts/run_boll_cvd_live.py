from __future__ import annotations

import asyncio
import datetime as dt
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from src.execution.trader import Trader  # noqa: E402
from src.indicators.cvd_tracker import CvdTracker, CvdTrackerConfig  # noqa: E402
from src.live import config_helpers as live_config_helpers  # noqa: E402
from src.live import queue_helpers as live_queue_helpers  # noqa: E402
from src.live import runtime_types as live_runtime_types  # noqa: E402
from src.live import time_utils as live_time_utils  # noqa: E402
from src.live.workers import execution_worker as execution_worker_module  # noqa: E402
from src.live.workers import strategy_tick_worker as strategy_tick_worker_module  # noqa: E402
from src.live.startup_recovery import basic_restore as startup_basic_restore  # noqa: E402
from src.live.startup_recovery import order_recovery as startup_order_recovery  # noqa: E402
from src.live.startup_recovery import trust_validation as startup_trust_validation  # noqa: E402
from src.live.account_sync import flat_balance as live_flat_balance  # noqa: E402
from src.live.account_sync import pre_core_position as account_sync_pre_core_position  # noqa: E402
from src.live.account_sync import protective_orders_phase as account_sync_protective_orders_phase  # noqa: E402
from src.live.account_sync import tp_progress_phase as account_sync_tp_progress_phase  # noqa: E402
from src.live.account_sync import flat_settlement_phase as account_sync_flat_settlement_phase  # noqa: E402
from src.position_management import core_position_view as core_position_view_helpers  # noqa: E402
from src.position_management import runner_live_helpers  # noqa: E402
from src.monitors.boll_band_breakout_monitor import (  # noqa: E402
    BollBandBreakoutMonitor,
    BollBandBreakoutMonitorConfig,
    MarketTickEvent,
)

from src.position_management import tp_progress as tp_progress_helpers  # noqa: E402
from src.position_management.sidecar import runtime_state as sidecar_runtime_state  # noqa: E402
from src.position_management.sidecar.model import (  # noqa: E402
    sidecar_open_contracts,
    sidecar_open_qty,
)

from src.position_management.sidecar.reconciler import build_core_position_view  # noqa: E402
from src.position_management.sidecar import force_close_runtime as sidecar_force_close_runtime  # noqa: E402
from src.position_management.sidecar import monitor_runtime as sidecar_monitor_runtime  # noqa: E402
from src.reporting import live_report_helpers as report_helpers  # noqa: E402
from src.reporting.daily_trade_reporter import DailyTradeReporter  # noqa: E402
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
)
from src.risk import rolling_loss_live as rolling_loss_live_helpers  # noqa: E402
from src.strategies.boll_cvd_reclaim_strategy import (  # noqa: E402
    BollCvdReclaimStrategyConfig,
    StrategyPositionState,
)
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy  # noqa: E402
from src.utils.email_sender import EmailSender  # noqa: E402
from src.utils.log import get_logger  # noqa: E402

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
        startup_cash = await live_flat_balance.fetch_usdt_cash_balance(trader)
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
    trusted_saved_state = startup_trust_validation.trusted_startup_saved_state(saved_state, startup_position)
    if startup_position.has_position:
        if trusted_saved_state is not None:
            startup_basic_restore.restore_strategy_from_saved_state(strategy, trusted_saved_state)
            current_position_id = trusted_saved_state.position_id
            cash_before_position = trusted_saved_state.cash_before_position
        else:
            startup_basic_restore.restore_strategy_from_position(strategy, startup_position, live_time_utils.utc_ms())
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
        state_store.save(LiveStateStore.from_strategy_state(position_id=current_position_id, symbol=trader.symbol,
                                                            strategy_state=strategy.state,
                                                            cash_before_position=cash_before_position))
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
    await startup_order_recovery.apply_main_tp_startup_recovery(
        execution_state=execution_state,
        saved_state=trusted_saved_state,
        startup_position=startup_position,
        trader=trader,
        journal=journal,
    )
    await startup_order_recovery.apply_sidecar_startup_recovery(
        strategy=strategy,
        execution_state=execution_state,
        saved_state=trusted_saved_state,
        startup_position=startup_position,
        trader=trader,
        journal=journal,
        state_store=state_store,
    )
    sidecar_runtime_state.refresh_sidecar_state_totals(strategy.state, int(os.getenv("SIDECAR_MAX_LEGS", "10")))
    startup_core_position = build_core_position_view(
        startup_position,
        sidecar_open_qty(strategy.state.sidecar_legs),
        sidecar_open_contracts(strategy.state.sidecar_legs),
    )
    core_position_view_helpers.apply_core_position_view_to_state(strategy.state, startup_core_position)
    account_snapshot.position = startup_core_position
    if startup_position.has_position:
        state_store.save(LiveStateStore.from_strategy_state(position_id=current_position_id, symbol=trader.symbol,
                                                            strategy_state=strategy.state,
                                                            cash_before_position=cash_before_position))
    await rolling_loss_live_helpers.apply_rolling_loss_guard_startup_state(
        rolling_loss_guard=rolling_loss_guard,
        execution_state=execution_state,
        has_position=startup_position.has_position,
        equity=trader.account_equity_usdt,
        now_ms=live_time_utils.utc_ms(),
        journal=journal,
        email_sender=email_sender,
    )
    runner_live_helpers.apply_three_stage_startup_safety_gate(
        strategy=strategy,
        execution_state=execution_state,
        saved_state=trusted_saved_state,
        startup_position=startup_position,
        journal=journal,
        state_store=state_store,
        trader_symbol=trader.symbol,
    )
    strategy_tick_queue: asyncio.Queue[MarketTickEvent] = asyncio.Queue(
        maxsize=int(os.getenv("STRATEGY_TICK_QUEUE_MAXSIZE", "20000")))
    execution_queue: asyncio.Queue[live_runtime_types.TradeCommand] = asyncio.Queue(
        maxsize=int(os.getenv("EXECUTION_QUEUE_MAXSIZE", "1000")))
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
                context = report_helpers.build_report_context(
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
        compact_after_success = os.getenv("WEEKLY_COMPACT_AFTER_SUCCESS", "false").strip().lower() in {"1", "true",
                                                                                                       "yes", "y", "on"}
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
                context = report_helpers.build_report_context(
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
        await live_queue_helpers.enqueue_strategy_tick(event, strategy_tick_queue, state_lock, execution_state)

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
            strategy_tick_worker_module.strategy_tick_worker(
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
            execution_worker_module.execution_worker(
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
                sidecar_skip_first_layer=sizer.config.sidecar_skip_first_layer,
            ),
            daily_report_loop(),
            weekly_summary_loop(),
            monitor.run_forever(),
        )
    finally:
        await trader.close()


if __name__ == "__main__":
    asyncio.run(main())
