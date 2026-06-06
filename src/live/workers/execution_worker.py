from __future__ import annotations

import asyncio
import copy
import html
import time
from dataclasses import replace
from decimal import Decimal

from src.execution.trader import PositionSnapshot, Trader
from src.live import config_helpers as live_config_helpers
from src.live import queue_helpers as live_queue_helpers
from src.live import runtime_types as live_runtime_types
from src.live.account_sync import flat_balance as live_flat_balance
from src.live.startup_recovery import basic_restore as startup_basic_restore
from src.live.workers import execution_failure as execution_failure_handler
from src.live.workers import strategy_tick_worker as strategy_tick_worker_module
from src.position_management import core_position_view as core_position_view_helpers
from src.position_management import cost_runtime as position_cost_runtime
from src.position_management import runner_live_helpers
from src.position_management import tp_progress as tp_progress_helpers
from src.position_management.sidecar import entry_runtime as sidecar_entry_runtime
from src.position_management.sidecar.planner import (
    SidecarExecutionPlan,
    build_combined_entry_intent,
)
from src.reporting.live_state_store import LiveStateStore
from src.reporting.trade_journal import LiveTradeJournal
from src.risk.rolling_loss_guard import ROLLING_LOSS_HALT_REASONS
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy
from src.utils.email_sender import EmailSender
from src.utils.log import get_logger

logger = get_logger(__name__)


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
    sidecar_skip_first_layer: bool = True,
) -> None:
    last_backlog_log = 0.0
    while True:
        command = await execution_queue.get()
        result = None
        try:
            queue_size = execution_queue.qsize()
            level = live_queue_helpers.queue_log_level(queue_size)
            now = time.monotonic()
            if level is not None and now - last_backlog_log >= backlog_log_seconds:
                logger.log(
                    level,
                    "EXECUTION_QUEUE_BACKLOG | queue_size=%s maxsize=%s oldest_command_age_seconds=%.3f",
                    queue_size,
                    execution_queue.maxsize,
                    live_queue_helpers.queue_oldest_command_age_seconds(execution_queue),
                )
                last_backlog_log = now

            dirty_post_tp1_sl_blocked = False
            dirty_post_tp1_sl_should_record = False
            async with state_lock:
                if runner_live_helpers.three_stage_dirty_post_tp1_sl_after_tp2(strategy.state):
                    dirty_post_tp1_sl_blocked = True
                    dirty_post_tp1_sl_should_record = not (
                        execution_state.trading_halted
                        and execution_state.halt_reason == runner_live_helpers.THREE_STAGE_RUNTIME_DIRTY_HALT_REASON
                    )
                    execution_state.trading_halted = True
                    execution_state.halt_reason = runner_live_helpers.THREE_STAGE_RUNTIME_DIRTY_HALT_REASON
                    execution_state.halt_until_ts_ms = None
            if dirty_post_tp1_sl_blocked:
                if dirty_post_tp1_sl_should_record:
                    runner_live_helpers.append_three_stage_dirty_post_tp1_event(
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
                    and command.intent.intent_type in strategy_tick_worker_module.POSITION_MANAGEMENT_INTENTS
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
                entry_cash_before = await live_flat_balance.fetch_usdt_cash_balance(trader)

            if command.intent.intent_type in {"ADD_LONG", "ADD_SHORT"} and getattr(command.strategy_state_snapshot, "tp_plan", "SINGLE") in tp_progress_helpers.SPLIT_TP_PLANS:
                position = await trader.fetch_position_snapshot()
                if position.has_position and position.side == command.intent.side:
                    consumed = False
                    async with state_lock:
                        current_strategy_state = copy.deepcopy(strategy.state)
                        strategy.state = copy.deepcopy(command.strategy_state_snapshot)
                        consumed = tp_progress_helpers.mark_partial_tp_consumed_if_position_reduced(strategy, position)
                        if consumed:
                            position_cost_runtime.sync_strategy_cost_from_position(
                                strategy,
                                position,
                                restore_from_position=startup_basic_restore.restore_strategy_from_position,
                            )
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

            entry_intent = core_position_view_helpers.with_entry_add_managed_core_contracts(
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
                    sidecar_skip_first_layer=sidecar_skip_first_layer,
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
                            position_cost_runtime.sync_strategy_cost_from_position(
                                strategy,
                                remaining_position,
                                restore_from_position=startup_basic_restore.restore_strategy_from_position,
                            )
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
                    sidecar_ok = await sidecar_entry_runtime.attach_sidecar_after_combined_entry(
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
            await execution_failure_handler.handle_execution_failure(
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
