from __future__ import annotations

import asyncio
import copy
import datetime as dt
import logging
import os
import sys
import time
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
from src.position_management import core_position_view as core_position_view_helpers  # noqa: E402
from src.position_management import runner_live_helpers  # noqa: E402
from src.monitors.boll_band_breakout_monitor import (  # noqa: E402
    BollBandBreakoutMonitor,
    BollBandBreakoutMonitorConfig,
    MarketTickEvent,
)
from src.position_management import cost_runtime as position_cost_runtime  # noqa: E402
from src.position_management import tp_progress as tp_progress_helpers  # noqa: E402
from src.position_management.sidecar import runtime_state as sidecar_runtime_state  # noqa: E402
from src.position_management.sidecar.model import (  # noqa: E402
    SidecarLegStatus,
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
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
    StrategyPositionState,
    TradeIntent,
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
                trader.position_contracts = core_position.contracts
                # Position reduction detection must run every account sync,
                # even when pending orders exist (e.g. TP2 / Sidecar TP still
                # pending after TP1 fill). The mark_* helpers are internally
                # idempotent via consumed/active flags.
                middle_runner_activated = tp_progress_helpers.mark_middle_runner_active_if_position_reduced(strategy, core_position)
                three_stage_event = tp_progress_helpers.mark_three_stage_progress_if_position_reduced(strategy, core_position, live_time_utils.utc_ms())
                tp_progress_helpers.mark_partial_tp_consumed_if_position_reduced(strategy, core_position)
                position_cost_runtime.sync_strategy_cost_from_position(
                    strategy,
                    core_position,
                    restore_from_position=startup_basic_restore.restore_strategy_from_position,
                )
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
                            post_tp1_boll = runner_live_helpers.three_stage_post_tp1_boll(strategy)
                            protective_sl = None
                            current_price = None
                            price_source = "missing"
                            if post_tp1_boll is not None and core_position.side is not None:
                                current_price, price_source = runner_live_helpers.three_stage_post_tp1_current_price(account_snapshot, core_position, post_tp1_boll, live_time_utils.utc_ms())
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
                                            "core_contracts": float(core_position.contracts),
                                            "net_side": position.side if position.has_position else None,
                                            "net_contracts": float(position.contracts) if position.has_position else 0,
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
                                    "contracts": float(position.contracts),
                                    "core_contracts": float(core_position.contracts),
                                    "net_contracts": float(position.contracts),
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
                                execution_state.halt_reason = runner_live_helpers.THREE_STAGE_CANCEL_PENDING_HALT_REASON
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
                        runner_boll = runner_live_helpers.middle_runner_activation_boll(strategy)
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
                                        "core_contracts": float(core_position.contracts),
                                        "net_side": position.side if position.has_position else None,
                                        "net_contracts": float(position.contracts) if position.has_position else 0,
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
                                "contracts": float(position.contracts),
                                "core_contracts": float(core_position.contracts),
                                "net_contracts": float(position.contracts),
                                "protective_sl_price": protective_sl,
                                "old_sl_order_id": getattr(strategy.state, "middle_runner_protective_sl_order_id", None),
                                "reason": "partial_tp_filled",
                            }
                elif runner_live_helpers.middle_runner_size_mismatch_needs_degraded_protection(strategy, core_position):
                    runner_boll = runner_live_helpers.middle_runner_activation_boll(strategy)
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
                                    "core_contracts": float(core_position.contracts),
                                    "net_side": position.side if position.has_position else None,
                                    "net_contracts": float(position.contracts) if position.has_position else 0,
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
                            "contracts": float(position.contracts),
                            "core_contracts": float(core_position.contracts),
                            "net_contracts": float(position.contracts),
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

            sidecar_check_seconds = max(float(getattr(sizer.config, "sidecar_order_status_check_seconds", 5.0) or 5.0), 0.0)
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

            if pending_flat_payload is not None:
                try:
                    settled = await live_flat_balance.fetch_settled_flat_balance(
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
                tp_progress_helpers.append_three_stage_progress_journal_events(journal, three_stage_event_payload)
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
                            and execution_state.halt_reason == runner_live_helpers.THREE_STAGE_CANCEL_PENDING_HALT_REASON
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
                        halt_reason = rolling_loss_live_helpers.rolling_loss_halt_reason(decision.action)
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
        state_store.save(LiveStateStore.from_strategy_state(position_id=current_position_id, symbol=trader.symbol, strategy_state=strategy.state, cash_before_position=cash_before_position))
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
