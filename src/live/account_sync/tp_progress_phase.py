from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from src.execution.trader import PositionSnapshot, Trader
from src.live import runtime_types as live_runtime_types
from src.live import time_utils as live_time_utils
from src.live.startup_recovery import basic_restore as startup_basic_restore
from src.position_management import cost_runtime as position_cost_runtime
from src.position_management import runner_live_helpers
from src.position_management import tp_progress as tp_progress_helpers
from src.position_management.sidecar.model import sidecar_open_qty
from src.reporting.live_state_store import LiveStateStore
from src.reporting.trade_journal import LiveTradeJournal
from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy
from src.utils.log import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class AccountSyncTpProgressResult:
    save_state_payload: tuple[str | None, StrategyPositionState, float | None] | None
    middle_runner_sl_payload: dict[str, Any] | None
    middle_runner_activation_payload: dict[str, Any] | None
    three_stage_post_tp1_sl_payload: dict[str, Any] | None
    three_stage_post_tp1_cancel_payload: dict[str, Any] | None
    three_stage_event_payload: dict[str, Any] | None
    last_logged_position_key: Any


def run_account_sync_tp_progress_phase(
    *,
    account_snapshot: live_runtime_types.AccountSnapshot,
    execution_state: live_runtime_types.ExecutionState,
    trader: Trader,
    strategy: BollCvdShockReclaimStrategy,
    journal: LiveTradeJournal,
    state_store: LiveStateStore,
    position: PositionSnapshot,
    core_position: PositionSnapshot,
    current_position_key: Any,
    pending_order_count: int,
    last_logged_position_key: Any,
) -> AccountSyncTpProgressResult:
    # Position reduction detection must run every account sync,
    # even when pending orders exist (e.g. TP2 / Sidecar TP still
    # pending after TP1 fill). The mark_* helpers are internally
    # idempotent via consumed/active flags.
    trader.position_contracts = core_position.contracts
    three_stage_post_tp1_sl_payload: dict[str, Any] | None = None
    three_stage_post_tp1_cancel_payload: dict[str, Any] | None = None
    three_stage_event_payload: dict[str, Any] | None = None
    middle_runner_sl_payload: dict[str, Any] | None = None
    middle_runner_activation_payload: dict[str, Any] | None = None

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
                    "contracts": position.contracts,
                    "core_contracts": core_position.contracts,
                    "net_contracts": position.contracts,
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

    return AccountSyncTpProgressResult(
        save_state_payload=save_state_payload,
        middle_runner_sl_payload=middle_runner_sl_payload,
        middle_runner_activation_payload=middle_runner_activation_payload,
        three_stage_post_tp1_sl_payload=three_stage_post_tp1_sl_payload,
        three_stage_post_tp1_cancel_payload=three_stage_post_tp1_cancel_payload,
        three_stage_event_payload=three_stage_event_payload,
        last_logged_position_key=last_logged_position_key,
    )
