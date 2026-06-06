from __future__ import annotations

import os
from typing import Any

from src.execution.trader import PositionSnapshot
from src.live.runtime_types import AccountSnapshot, ExecutionState
from src.reporting.live_state_store import LiveStateStore
from src.reporting.trade_journal import LiveTradeJournal
from src.strategies.boll_cvd_reclaim_strategy import BollCvdReclaimStrategy, StrategyPositionState
from src.utils.log import get_logger

logger = get_logger(__name__)

THREE_STAGE_RESTART_DIRTY_HALT_REASON = "three_stage_post_tp1_sl_cancel_failed_on_tp2_restart"
THREE_STAGE_RUNTIME_DIRTY_HALT_REASON = "three_stage_post_tp1_sl_dirty_state_blocked"
THREE_STAGE_CANCEL_PENDING_HALT_REASON = "three_stage_post_tp1_sl_cancel_pending_on_tp2"


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


def three_stage_post_tp1_current_price(
    account_snapshot: AccountSnapshot,
    position: PositionSnapshot,
    post_tp1_boll: Any,
    now_ms: int,
) -> tuple[float, str]:
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


def middle_runner_size_mismatch_needs_degraded_protection(
    strategy: BollCvdReclaimStrategy,
    position: PositionSnapshot,
) -> bool:
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


def three_stage_dirty_post_tp1_sl_after_tp2(state: StrategyPositionState) -> bool:
    return bool(
        getattr(state, "trend_runner_active", False)
        and getattr(state, "three_stage_tp2_consumed", False)
        and getattr(state, "three_stage_post_tp1_protective_sl_order_id", None)
    )


def three_stage_dirty_post_tp1_payload(
    *,
    strategy: BollCvdReclaimStrategy,
    execution_state: ExecutionState,
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
    execution_state: ExecutionState,
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
    execution_state: ExecutionState,
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
