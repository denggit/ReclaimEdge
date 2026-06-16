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
    middle_bucket_split_event_payload: dict[str, Any] | None
    middle_bucket_split_fast_protection_payload: dict[str, Any] | None
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
    # even when pending orders exist (e.g. TP2 still
    # pending after TP1 fill). The mark_* helpers are internally
    # idempotent via consumed/active flags.
    trader.position_contracts = core_position.contracts
    three_stage_post_tp1_sl_payload: dict[str, Any] | None = None
    three_stage_post_tp1_cancel_payload: dict[str, Any] | None = None
    three_stage_event_payload: dict[str, Any] | None = None
    middle_runner_sl_payload: dict[str, Any] | None = None
    middle_runner_activation_payload: dict[str, Any] | None = None

    # ── Middle Bucket Split progress MUST run first when active ──────
    # When split is active, the split progress owns fast/slow fill detection.
    # Old TP progress (mark_middle_runner_active, mark_three_stage_progress)
    # must NOT run for the same fills to avoid duplicate cost recording.
    middle_bucket_split_progress = tp_progress_helpers.mark_middle_bucket_split_progress_if_position_reduced(
        strategy, core_position)
    middle_bucket_split_event = middle_bucket_split_progress.event if middle_bucket_split_progress else None
    pre_split_tp_plan = (
        middle_bucket_split_progress.pre_split_tp_plan
        if middle_bucket_split_progress
        else getattr(strategy.state, "tp_plan", "SINGLE")
    )

    # Explicitly decide which progress path runs based on the split event
    split_owns_progress = middle_bucket_split_event in {
        "MIDDLE_BUCKET_FAST",
        "MIDDLE_BUCKET_SLOW_ONLY",
        "MIDDLE_BUCKET_FULL",
        "MIDDLE_BUCKET_SLOW",
    }
    if split_owns_progress:
        # Split owns the progress path; skip old TP progress entirely
        middle_runner_activated = False
        three_stage_event = None
    else:
        middle_runner_activated = tp_progress_helpers.mark_middle_runner_active_if_position_reduced(strategy, core_position)
        three_stage_event = tp_progress_helpers.mark_three_stage_progress_if_position_reduced(strategy, core_position,
                                                                                              live_time_utils.utc_ms())
    tp_progress_helpers.mark_partial_tp_consumed_if_position_reduced(strategy, core_position)
    position_cost_runtime.sync_strategy_cost_from_position(
        strategy,
        core_position,
        restore_from_position=startup_basic_restore.restore_strategy_from_position,
    )
    if pending_order_count > 0 and three_stage_event is not None:
        logger.warning(
            "THREE_STAGE_POSITION_REDUCTION_DETECTED_WITH_PENDING_ORDERS | "
            "event=%s pending_order_count=%s side=%s old_total_eth_qty=%.8f new_core_eth_qty=%.8f core_contracts=%s net_contracts=%s",
            three_stage_event,
            pending_order_count,
            core_position.side,
            float(getattr(strategy.state, "total_entry_qty", 0.0) or 0.0),
            float(core_position.eth_qty or 0.0),
            core_position.contracts,
            position.contracts if position.has_position else 0,
            0.0,
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
                    current_price, price_source = runner_live_helpers.three_stage_post_tp1_current_price(
                        account_snapshot, core_position, post_tp1_boll, live_time_utils.utc_ms())
                    base_sl = strategy._calculate_three_stage_post_tp1_protective_sl(core_position.side, current_price,
                                                                                     post_tp1_boll)
                    extension_sl = strategy._apply_three_stage_post_tp1_extension_trigger(core_position.side,
                                                                                          current_price, post_tp1_boll,
                                                                                          base_sl)
                    protective_sl = strategy._tighten_optional_three_stage_post_tp1_sl(core_position.side, base_sl,
                                                                                       extension_sl)
                # Global protective SL must cover OKX net position
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
                    old_sl_order_id = getattr(strategy.state, "three_stage_post_tp1_protective_sl_order_id", None)
                    old_sl_price = getattr(strategy.state, "three_stage_post_tp1_protective_sl_price", None)
                    three_stage_post_tp1_sl_payload = {
                        "position_id": execution_state.current_position_id,
                        "side": core_position.side,
                        "contracts": position.contracts,
                        "core_contracts": core_position.contracts,
                        "net_contracts": position.contracts,
                        "protective_sl_price": protective_sl,
                        "old_sl_order_id": old_sl_order_id,
                        "old_sl_price": old_sl_price,
                        "old_protected": getattr(strategy.state, "three_stage_post_tp1_protected", False),
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
            # Global protective SL must cover OKX net position
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
                old_sl_order_id = getattr(strategy.state, "middle_runner_protective_sl_order_id", None)
                old_sl_price = getattr(strategy.state, "middle_runner_protective_sl_price", None)
                middle_runner_sl_payload = {
                    "position_id": execution_state.current_position_id,
                    "side": core_position.side,
                    "contracts": position.contracts,
                    "core_contracts": core_position.contracts,
                    "net_contracts": position.contracts,
                    "protective_sl_price": protective_sl,
                    "old_sl_order_id": old_sl_order_id,
                    "old_sl_price": old_sl_price,
                    "old_protected": bool(old_sl_order_id) and old_sl_price is not None,
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
        # Global protective SL must cover OKX net position
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
            old_sl_order_id = getattr(strategy.state, "middle_runner_protective_sl_order_id", None)
            old_sl_price = getattr(strategy.state, "middle_runner_protective_sl_price", None)
            middle_runner_sl_payload = {
                "position_id": execution_state.current_position_id,
                "side": core_position.side,
                "contracts": position.contracts,
                "core_contracts": core_position.contracts,
                "net_contracts": position.contracts,
                "protective_sl_price": protective_sl,
                "old_sl_order_id": old_sl_order_id,
                "old_sl_price": old_sl_price,
                "old_protected": bool(old_sl_order_id) and old_sl_price is not None,
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

    # ── Middle Bucket Split event & journal ────────────────────────
    middle_bucket_split_event_payload: dict[str, Any] | None = None
    middle_bucket_split_fast_protection_payload: dict[str, Any] | None = None
    if middle_bucket_split_event is not None:
        middle_bucket_split_event_payload = {
            "event": middle_bucket_split_event,
            "position_id": execution_state.current_position_id,
            "side": core_position.side,
            "layers": strategy.state.layers,
            "avg_entry_price": strategy.state.avg_entry_price,
            "tp_plan": pre_split_tp_plan,
            "pre_split_tp_plan": pre_split_tp_plan,
            "middle_bucket_ratio": getattr(strategy.state, "middle_bucket_split_middle_bucket_ratio", 0.0),
            "fast_ratio_of_bucket": getattr(strategy.state, "middle_bucket_split_fast_ratio_of_bucket", 0.0),
            "slow_ratio_of_bucket": getattr(strategy.state, "middle_bucket_split_slow_ratio_of_bucket", 0.0),
            "fast_total_ratio": getattr(strategy.state, "middle_bucket_split_fast_total_ratio", 0.0),
            "slow_total_ratio": getattr(strategy.state, "middle_bucket_split_slow_total_ratio", 0.0),
            "fast_price": getattr(strategy.state, "middle_bucket_split_fast_price", None),
            "slow_price": getattr(strategy.state, "middle_bucket_split_slow_price", None),
            "effective_price": getattr(strategy.state, "middle_bucket_split_effective_price", None),
            "fast_sl_price": getattr(strategy.state, "middle_bucket_split_fast_sl_price", None),
            "fast_consumed": getattr(strategy.state, "middle_bucket_split_fast_consumed", False),
            "slow_consumed": getattr(strategy.state, "middle_bucket_split_slow_consumed", False),
            "add_disabled": getattr(strategy.state, "middle_bucket_split_add_disabled", False),
        }
        if hasattr(journal, "append"):
            tp_progress_helpers.append_middle_bucket_split_journal_events(
                journal, middle_bucket_split_event_payload)
        # Fast fill → trigger protection
        if middle_bucket_split_event == "MIDDLE_BUCKET_FAST":
            fast_sl_price = getattr(strategy.state, "middle_bucket_split_fast_sl_price", None)
            # Use actual market price, NOT position.avg_entry_price
            current_price = 0.0
            current_price_source = "missing"
            current_price_ts_ms = 0
            latest_market = getattr(account_snapshot, "latest_market_price", None)
            if latest_market is not None and float(latest_market) > 0:
                current_price = float(latest_market)
                current_price_source = "latest_market_price"
                current_price_ts_ms = live_time_utils.utc_ms()
            elif hasattr(core_position, "mark_price") and getattr(core_position, "mark_price", None) is not None:
                current_price = float(getattr(core_position, "mark_price", 0.0) or 0.0)
                current_price_source = "mark_price"
            elif hasattr(position, "avg_entry_price") and position.avg_entry_price > 0:
                current_price = float(position.avg_entry_price)
                current_price_source = "degraded_current_price_source_avg_entry"
            middle_bucket_split_fast_protection_payload = {
                "position_id": execution_state.current_position_id,
                "side": core_position.side,
                "avg_entry_price": float(strategy.state.avg_entry_price or 0.0),
                "fast_sl_price": fast_sl_price,
                "current_price": current_price,
                "current_price_source": current_price_source,
                "current_price_ts_ms": current_price_ts_ms,
                "core_contracts": core_position.contracts,
                "net_contracts": position.contracts if position.has_position else 0,
                "invalid_action": getattr(strategy.config, "middle_bucket_split_fast_sl_invalid_action", "MARKET_EXIT"),
                "enabled": bool(getattr(strategy.config, "middle_bucket_split_fast_sl_enabled", True)),
                "old_sl_order_id": getattr(strategy.state, "middle_bucket_split_fast_sl_order_id", None),
                "old_sl_price": getattr(strategy.state, "middle_bucket_split_fast_sl_price", None),
                "old_protected": getattr(strategy.state, "middle_bucket_split_fast_sl_protected", False),
            }

        # ── Full fill → generate post-TP1 / middle_runner dynamic SL ──
        if middle_bucket_split_event in {"MIDDLE_BUCKET_FULL", "MIDDLE_BUCKET_SLOW"}:
            tp_plan = pre_split_tp_plan
            old_fast_sl_order_id = getattr(strategy.state, "middle_bucket_split_fast_sl_order_id", None)
            filled_reason = (
                "middle_bucket_full_filled"
                if middle_bucket_split_event == "MIDDLE_BUCKET_FULL"
                else "middle_bucket_slow_filled"
            )

            if tp_plan == "THREE_STAGE_RUNNER" and getattr(strategy.state, "three_stage_tp1_consumed", False):
                config = getattr(strategy, "config", None)
                if bool(getattr(config, "three_stage_post_tp1_protective_sl_enabled", True)):
                    post_tp1_boll = runner_live_helpers.three_stage_post_tp1_boll(strategy)
                    protective_sl = None
                    current_price = None
                    price_source = "missing"
                    if post_tp1_boll is not None and core_position.side is not None:
                        current_price, price_source = runner_live_helpers.three_stage_post_tp1_current_price(
                            account_snapshot, core_position, post_tp1_boll, live_time_utils.utc_ms())
                        base_sl = strategy._calculate_three_stage_post_tp1_protective_sl(
                            core_position.side, current_price, post_tp1_boll)
                        extension_sl = strategy._apply_three_stage_post_tp1_extension_trigger(
                            core_position.side, current_price, post_tp1_boll, base_sl)
                        protective_sl = strategy._tighten_optional_three_stage_post_tp1_sl(
                            core_position.side, base_sl, extension_sl)
                    if not position.has_position or position.side != core_position.side or position.contracts <= 0:
                        execution_state.trading_halted = True
                        execution_state.halt_reason = "middle_bucket_slow_post_tp1_net_position_missing"
                        logger.error(
                            "MIDDLE_BUCKET_SPLIT_SLOW_THREE_STAGE_NET_MISSING | position_id=%s trading_halted=true",
                            execution_state.current_position_id,
                        )
                    else:
                        old_sl_order_id = old_fast_sl_order_id or getattr(
                            strategy.state, "three_stage_post_tp1_protective_sl_order_id", None)
                        old_sl_price = (
                            getattr(strategy.state, "middle_bucket_split_fast_sl_price", None)
                            if old_fast_sl_order_id
                            else getattr(strategy.state, "three_stage_post_tp1_protective_sl_price", None)
                        )
                        old_protected = (
                            getattr(strategy.state, "middle_bucket_split_fast_sl_protected", False)
                            if old_fast_sl_order_id
                            else getattr(strategy.state, "three_stage_post_tp1_protected", False)
                        )
                        three_stage_post_tp1_sl_payload = {
                            "position_id": execution_state.current_position_id,
                            "side": core_position.side,
                            "contracts": position.contracts,
                            "core_contracts": core_position.contracts,
                            "net_contracts": position.contracts,
                            "protective_sl_price": protective_sl,
                            "old_sl_order_id": old_sl_order_id,
                            "old_sl_price": old_sl_price,
                            "old_protected": old_protected,
                            "current_price": current_price,
                            "current_price_source": price_source,
                            "reason": filled_reason,
                        }
                # Generate event payload for full split fill (journal + log context)
                three_stage_event = "TP1"  # signal that TP1 was filled via split full path
                three_stage_event_payload = {
                    "event": "TP1",
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
                    "split_source": filled_reason,
                }

            elif tp_plan == "MIDDLE_RUNNER" and getattr(strategy.state, "middle_runner_active", False):
                config = getattr(strategy, "config", None)
                if bool(getattr(config, "middle_runner_protective_sl_enabled", True)):
                    runner_boll = runner_live_helpers.middle_runner_activation_boll(strategy)
                    current_price = getattr(runner_boll, "middle", 0.0) if runner_boll is not None else 0.0
                    protective_sl = (
                        strategy._calculate_middle_runner_protective_sl(core_position.side, current_price, runner_boll)
                        if runner_boll is not None and core_position.side is not None
                        else None
                    )
                    if not position.has_position or position.side != core_position.side or position.contracts <= 0:
                        execution_state.trading_halted = True
                        execution_state.halt_reason = "middle_bucket_slow_middle_runner_net_position_missing"
                        logger.error(
                            "MIDDLE_BUCKET_SPLIT_SLOW_MIDDLE_RUNNER_NET_MISSING | position_id=%s trading_halted=true",
                            execution_state.current_position_id,
                        )
                    else:
                        old_sl_order_id = old_fast_sl_order_id or getattr(
                            strategy.state, "middle_runner_protective_sl_order_id", None)
                        old_sl_price = (
                            getattr(strategy.state, "middle_bucket_split_fast_sl_price", None)
                            if old_fast_sl_order_id
                            else getattr(strategy.state, "middle_runner_protective_sl_price", None)
                        )
                        old_protected = (
                            getattr(strategy.state, "middle_bucket_split_fast_sl_protected", False)
                            if old_fast_sl_order_id
                            else bool(old_sl_order_id) and old_sl_price is not None
                        )
                        middle_runner_sl_payload = {
                            "position_id": execution_state.current_position_id,
                            "side": core_position.side,
                            "contracts": position.contracts,
                            "core_contracts": core_position.contracts,
                            "net_contracts": position.contracts,
                            "protective_sl_price": protective_sl,
                            "old_sl_order_id": old_sl_order_id,
                            "old_sl_price": old_sl_price,
                            "old_protected": old_protected,
                            "reason": filled_reason,
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
                        "reason": filled_reason,
                    }
    save_state_payload = (
        execution_state.current_position_id, copy.deepcopy(strategy.state), execution_state.cash_before_position)
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
        middle_bucket_split_event_payload=middle_bucket_split_event_payload,
        middle_bucket_split_fast_protection_payload=middle_bucket_split_fast_protection_payload,
        last_logged_position_key=last_logged_position_key,
    )
