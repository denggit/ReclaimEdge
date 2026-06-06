from __future__ import annotations

import os

from src.execution.trader import Trader
from src.live import runtime_types as live_runtime_types
from src.position_management import cost_runtime as position_cost_runtime
from src.position_management.sidecar import runtime_state as sidecar_runtime_state
from src.position_management.sidecar.model import SidecarLegStatus
from src.position_management.sidecar.planner import SidecarExecutionPlan
from src.position_management.sidecar.reconciler import (
    mark_sidecar_leg_open_unprotected,
    sidecar_leg_from_fill,
)
from src.reporting.live_state_store import LiveStateStore
from src.reporting.trade_journal import LiveTradeJournal
from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState, TradeIntent
from src.utils.log import get_logger

logger = get_logger(__name__)


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
        fee_buffer_pct: float = position_cost_runtime.DEFAULT_NET_REMAINING_FEE_BUFFER_PCT,
) -> bool:
    if not getattr(strategy_state, "sidecar_enabled_for_position", False):
        return True
    if intent.intent_type not in {"OPEN_LONG", "OPEN_SHORT", "ADD_LONG", "ADD_SHORT"}:
        return True
    position_id = execution_state.current_position_id
    contracts = str(sidecar_plan.sidecar_contracts)
    filled_qty = float(sidecar_plan.sidecar_qty)
    position_cost_runtime.record_remaining_entry_notional(
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
            retry_count=int(os.getenv("SIDECAR_TP_FAIL_MARKET_EXIT_RETRY_COUNT",
                                      os.getenv("NEAR_TP_SL_FAIL_MARKET_EXIT_RETRY_COUNT", "3"))),
        )
        if exit_ok:
            execution_state.halt_reason = "sidecar_tp_place_failed_market_exit_waiting_flat"
            strategy_state.sidecar_halt_reason = "sidecar_tp_place_failed_market_exit_waiting_flat"
        else:
            execution_state.halt_reason = "sidecar_tp_place_failed"
            strategy_state.sidecar_halt_reason = "sidecar_tp_place_failed"
        sidecar_runtime_state.refresh_sidecar_state_totals(strategy_state, int(os.getenv("SIDECAR_MAX_LEGS", "10")))
        state_store.save(LiveStateStore.from_strategy_state(position_id=position_id, symbol=trader_symbol,
                                                            strategy_state=strategy_state,
                                                            cash_before_position=execution_state.cash_before_position))
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
    sidecar_runtime_state.refresh_sidecar_state_totals(strategy_state, int(os.getenv("SIDECAR_MAX_LEGS", "10")))
    state_store.save(
        LiveStateStore.from_strategy_state(position_id=position_id, symbol=trader_symbol, strategy_state=strategy_state,
                                           cash_before_position=execution_state.cash_before_position))
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
