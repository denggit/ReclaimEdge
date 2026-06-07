from __future__ import annotations

import os

from src.execution.trader import PositionSnapshot, Trader
from src.live import runtime_types as live_runtime_types
from src.position_management import cost_runtime as position_cost_runtime
from src.position_management.sidecar import runtime_state as sidecar_runtime_state
from src.position_management.sidecar.fill_telemetry import (
    build_sidecar_fill_telemetry,
    log_sidecar_tp_filled,
)
from src.position_management.sidecar.model import SidecarLegStatus, sidecar_open_qty
from src.position_management.sidecar.reconciler import (
    is_sidecar_dirty_missing_tp_order,
    mark_sidecar_leg_tp_filled,
    mark_sidecar_leg_unknown_halted,
)
from src.reporting.live_state_store import LiveStateStore
from src.reporting.trade_journal import LiveTradeJournal
from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState
from src.utils.log import get_logger

logger = get_logger(__name__)


async def monitor_sidecar_orders_once(
        *,
        trader: Trader,
        strategy_state: StrategyPositionState,
        execution_state: live_runtime_types.ExecutionState,
        journal: LiveTradeJournal,
        state_store: LiveStateStore,
        trader_symbol: str,
        core_position: PositionSnapshot,
        position_id: str | None,
        cash_before_position: float | None,
        ts_ms: int,
        fee_buffer_pct: float = position_cost_runtime.DEFAULT_NET_REMAINING_FEE_BUFFER_PCT,
) -> None:
    if not getattr(strategy_state, "sidecar_enabled_for_position", False):
        return
    changed = False
    core_active = bool(core_position.has_position)
    for index, leg in enumerate(list(strategy_state.sidecar_legs)):
        if leg.get("status") != SidecarLegStatus.OPEN.value:
            continue
        if is_sidecar_dirty_missing_tp_order(leg):
            execution_state.trading_halted = True
            execution_state.halt_reason = "sidecar_tp_order_missing_or_unknown"
            strategy_state.sidecar_dirty = True
            strategy_state.sidecar_halt_reason = "sidecar_tp_order_missing_or_unknown"
            if not leg.get("warning_recorded"):
                journal.append("SIDECAR_TP_ORDER_MISSING_OR_UNKNOWN", dict(leg), position_id=position_id)
            strategy_state.sidecar_legs[index] = mark_sidecar_leg_unknown_halted(leg, ts_ms)
            changed = True
            continue
        status = await trader.fetch_sidecar_order_status(str(leg["tp_order_id"]))
        order_status = status.get("status")
        if order_status == "OPEN":
            continue
        if order_status == "FILLED":
            position_cost_runtime.record_sidecar_tp_fill_exit(
                strategy_state,
                leg,
                status,
                fee_buffer_pct=fee_buffer_pct,
            )
            strategy_state.sidecar_legs[index] = mark_sidecar_leg_tp_filled(leg, ts_ms)
            journal.append("SIDECAR_TP_FILLED", {**dict(leg), **status}, position_id=position_id)
            changed = True
            # Compute real remaining sidecar open qty after this fill
            _mt_open_qty_after = sidecar_open_qty(strategy_state.sidecar_legs)
            # Log sidecar TP fill in main log for observability
            _mt_telemetry = build_sidecar_fill_telemetry(
                source="monitor_runtime",
                leg=leg,
                status=status,
            )
            log_sidecar_tp_filled(
                logger,
                position_id=position_id,
                telemetry=_mt_telemetry,
                leg=leg,
                status=status,
                sidecar_open_qty_after=_mt_open_qty_after,
            )
            # Sidecar TP filled reduces OKX net position → existing global SL orders
            # may now exceed current net position. Must halt for manual reconciliation.
            active_global_sl_orders: list[str] = []
            for sl_field in (
                    "near_tp_protective_sl_order_id",
                    "middle_runner_protective_sl_order_id",
                    "three_stage_post_tp1_protective_sl_order_id",
                    "trend_runner_sl_order_id",
            ):
                sl_order_id = getattr(strategy_state, sl_field, None) or getattr(trader, sl_field, None)
                if sl_order_id:
                    active_global_sl_orders.append(f"{sl_field}={sl_order_id}")
            if active_global_sl_orders:
                execution_state.trading_halted = True
                execution_state.halt_reason = "sidecar_tp_filled_requires_global_sl_reconcile"
                strategy_state.sidecar_dirty = True
                strategy_state.sidecar_halt_reason = "sidecar_tp_filled_requires_global_sl_reconcile"
                journal.append(
                    "SIDECAR_TP_FILLED_REQUIRES_GLOBAL_SL_RECONCILE",
                    {
                        "active_global_sl_orders": active_global_sl_orders,
                        "trading_halted": True,
                        "halt_reason": "sidecar_tp_filled_requires_global_sl_reconcile",
                        "manual_intervention_required": True,
                    },
                    position_id=position_id,
                )
                logger.error(
                    "SIDECAR_TP_FILLED_REQUIRES_GLOBAL_SL_RECONCILE | position_id=%s leg_id=%s active_global_sl_orders=%s trading_halted=true halt_reason=sidecar_tp_filled_requires_global_sl_reconcile manual_intervention_required=true",
                    position_id,
                    leg.get("leg_id"),
                    active_global_sl_orders,
                )
            continue
        if order_status in {"CANCELED", "NOT_FOUND", "UNKNOWN"} and core_active:
            execution_state.trading_halted = True
            execution_state.halt_reason = "sidecar_tp_order_missing_or_unknown"
            strategy_state.sidecar_dirty = True
            strategy_state.sidecar_halt_reason = "sidecar_tp_order_missing_or_unknown"
            if not leg.get("warning_recorded"):
                journal.append("SIDECAR_TP_ORDER_MISSING_OR_UNKNOWN",
                               {**dict(leg), **status, "manual_intervention_required": True}, position_id=position_id)
            strategy_state.sidecar_legs[index] = mark_sidecar_leg_unknown_halted(leg, ts_ms)
            logger.error(
                "SIDECAR_TP_ORDER_MISSING_OR_UNKNOWN | position_id=%s leg_id=%s status=%s manual_intervention_required=true",
                position_id, leg.get("leg_id"), order_status)
            changed = True
    if changed:
        sidecar_runtime_state.refresh_sidecar_state_totals(strategy_state, int(os.getenv("SIDECAR_MAX_LEGS", "10")))
        state_store.save(LiveStateStore.from_strategy_state(position_id=position_id, symbol=trader_symbol,
                                                            strategy_state=strategy_state,
                                                            cash_before_position=cash_before_position))
