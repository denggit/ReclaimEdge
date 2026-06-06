from __future__ import annotations

import os
from typing import Any

from src.execution.trader import PositionSnapshot, Trader
from src.live import runtime_types as live_runtime_types
from src.live import time_utils as live_time_utils
from src.position_management import cost_runtime as position_cost_runtime
from src.position_management.sidecar import runtime_state as sidecar_runtime_state
from src.position_management.sidecar.model import SidecarLegStatus
from src.position_management.sidecar.reconciler import (
    mark_sidecar_leg_tp_filled,
    mark_sidecar_leg_unknown_halted,
)
from src.reporting.live_state_store import LiveStateStore
from src.reporting.trade_journal import LiveTradeJournal
from src.strategies.boll_cvd_reclaim_strategy import BollCvdReclaimStrategy
from src.utils.log import get_logger

logger = get_logger(__name__)


async def apply_sidecar_startup_recovery(
    *,
    strategy: BollCvdReclaimStrategy,
    execution_state: live_runtime_types.ExecutionState,
    saved_state: Any,
    startup_position: PositionSnapshot,
    trader: Trader,
    journal: LiveTradeJournal,
    state_store: LiveStateStore,
) -> None:
    if not startup_position.has_position:
        strategy.state.sidecar_enabled_for_position = False
        strategy.state.sidecar_legs = []
        sidecar_runtime_state.refresh_sidecar_state_totals(strategy.state, int(os.getenv("SIDECAR_MAX_LEGS", "10")))
        state_store.clear()
        return
    saved_legs = list(getattr(saved_state, "sidecar_legs", []) or []) if saved_state is not None else []
    saved_sidecar_enabled = bool(getattr(saved_state, "sidecar_enabled_for_position", False)) if saved_state is not None else False
    if saved_sidecar_enabled:
        strategy.state.sidecar_enabled_for_position = True
        strategy.state.sidecar_margin_pct = float(getattr(saved_state, "sidecar_margin_pct", strategy.state.sidecar_margin_pct) or 0.0)
        strategy.state.sidecar_tp_pct = float(getattr(saved_state, "sidecar_tp_pct", strategy.state.sidecar_tp_pct) or 0.0)
    open_legs = [
        leg
        for leg in saved_legs
        if leg.get("status") in {SidecarLegStatus.OPEN.value, SidecarLegStatus.OPEN_UNPROTECTED.value}
    ]
    if not open_legs:
        sidecar_runtime_state.refresh_sidecar_state_totals(strategy.state, int(os.getenv("SIDECAR_MAX_LEGS", "10")))
        if saved_sidecar_enabled:
            state_store.save(
                LiveStateStore.from_strategy_state(
                    position_id=execution_state.current_position_id,
                    symbol=trader.symbol,
                    strategy_state=strategy.state,
                    cash_before_position=execution_state.cash_before_position,
                )
            )
            return
        if (
            saved_state is None
            and os.getenv("SIDECAR_ENABLED", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
            and hasattr(journal, "append")
        ):
            strategy.state.sidecar_enabled_for_position = False
            strategy.state.sidecar_margin_pct = 0.0
            strategy.state.sidecar_tp_pct = 0.0
            journal.append(
                "SIDECAR_DISABLED_FOR_RECOVERED_POSITION",
                {
                    "side": startup_position.side,
                    "okx_eth_qty": startup_position.eth_qty,
                    "reason": "startup_position_has_no_saved_sidecar_state",
                },
                position_id=execution_state.current_position_id,
            )
        return

    changed = False
    for index, leg in enumerate(list(strategy.state.sidecar_legs)):
        if leg.get("status") == SidecarLegStatus.OPEN_UNPROTECTED.value:
            execution_state.trading_halted = True
            execution_state.halt_reason = str(getattr(strategy.state, "sidecar_halt_reason", None) or "sidecar_tp_place_failed")
            strategy.state.sidecar_dirty = True
            strategy.state.sidecar_halt_reason = execution_state.halt_reason
            continue
        if leg.get("status") != SidecarLegStatus.OPEN.value:
            continue
        order_id = leg.get("tp_order_id")
        if not order_id:
            status = {"order_id": None, "status": "UNKNOWN"}
        else:
            status = await trader.fetch_sidecar_order_status(str(order_id))
        if status.get("status") == "OPEN":
            continue
        if status.get("status") == "FILLED":
            position_cost_runtime.record_sidecar_tp_fill_exit(
                strategy.state,
                leg,
                status,
                fee_buffer_pct=getattr(
                    getattr(strategy, "config", None),
                    "breakeven_fee_buffer_pct",
                    position_cost_runtime.DEFAULT_NET_REMAINING_FEE_BUFFER_PCT,
                ),
            )
            strategy.state.sidecar_legs[index] = mark_sidecar_leg_tp_filled(leg, live_time_utils.utc_ms())
            if hasattr(journal, "append"):
                journal.append("SIDECAR_TP_FILLED", {**dict(leg), **status, "source": "startup_recovery"}, position_id=execution_state.current_position_id)
            changed = True
            continue
        execution_state.trading_halted = True
        execution_state.halt_reason = "sidecar_startup_order_state_unknown"
        strategy.state.sidecar_dirty = True
        strategy.state.sidecar_halt_reason = "sidecar_startup_order_state_unknown"
        strategy.state.sidecar_legs[index] = mark_sidecar_leg_unknown_halted(leg, live_time_utils.utc_ms())
        if hasattr(journal, "append"):
            journal.append(
                "SIDECAR_STARTUP_ORDER_STATE_UNKNOWN",
                {**dict(leg), **status, "manual_intervention_required": True},
                position_id=execution_state.current_position_id,
            )
        logger.error(
            "SIDECAR_STARTUP_ORDER_STATE_UNKNOWN | position_id=%s leg_id=%s order_id=%s status=%s trading_halted=true manual_intervention_required=true",
            execution_state.current_position_id,
            leg.get("leg_id"),
            order_id,
            status.get("status"),
        )
        changed = True
    sidecar_runtime_state.refresh_sidecar_state_totals(strategy.state, int(os.getenv("SIDECAR_MAX_LEGS", "10")))
    if changed or getattr(strategy.state, "sidecar_enabled_for_position", False):
        state_store.save(
            LiveStateStore.from_strategy_state(
                position_id=execution_state.current_position_id,
                symbol=trader.symbol,
                strategy_state=strategy.state,
                cash_before_position=execution_state.cash_before_position,
            )
        )


async def apply_main_tp_startup_recovery(
    *,
    execution_state: live_runtime_types.ExecutionState,
    saved_state: Any,
    startup_position: PositionSnapshot,
    trader: Trader,
    journal: LiveTradeJournal,
) -> None:
    if not startup_position.has_position:
        return
    restored_tp_order_id = getattr(saved_state, "tp_order_id", None) if saved_state is not None else None
    restored_tp_order_ids = list(getattr(saved_state, "tp_order_ids", []) or []) if saved_state is not None else []
    if not restored_tp_order_id and restored_tp_order_ids:
        restored_tp_order_id = ",".join(str(item) for item in restored_tp_order_ids if item)
    if restored_tp_order_id:
        trader.tp_order_id = str(restored_tp_order_id)
        return
    try:
        pending_orders = await trader.fetch_pending_orders()
    except Exception as exc:
        execution_state.trading_halted = True
        execution_state.halt_reason = "main_tp_order_id_missing_on_startup"
        if hasattr(journal, "append"):
            journal.append(
                "MAIN_TP_ORDER_ID_MISSING_ON_STARTUP",
                {"reason": "pending_order_check_failed", "error": str(exc), "manual_intervention_required": True},
                position_id=execution_state.current_position_id,
            )
        logger.error("MAIN_TP_ORDER_ID_MISSING_ON_STARTUP | reason=pending_order_check_failed error=%s trading_halted=true manual_intervention_required=true", exc)
        return
    protected_sidecar_tp_ids = {
        str(leg.get("tp_order_id"))
        for leg in list(getattr(saved_state, "sidecar_legs", []) or [])
        if leg.get("status") == SidecarLegStatus.OPEN.value and leg.get("tp_order_id")
    } if saved_state is not None else set()
    reduce_only_orders = [
        item
        for item in pending_orders
        if item.get("instId") == trader.symbol and str(item.get("reduceOnly", "")).lower() == "true"
        and str(item.get("ordId")) not in protected_sidecar_tp_ids
    ]
    if reduce_only_orders:
        execution_state.trading_halted = True
        execution_state.halt_reason = "main_tp_order_id_missing_on_startup"
        if hasattr(journal, "append"):
            journal.append(
                "MAIN_TP_ORDER_ID_MISSING_ON_STARTUP",
                {
                    "pending_reduce_only_order_count": len(reduce_only_orders),
                    "pending_reduce_only_order_ids": [item.get("ordId") for item in reduce_only_orders],
                    "manual_intervention_required": True,
                },
                position_id=execution_state.current_position_id,
            )
        logger.error(
            "MAIN_TP_ORDER_ID_MISSING_ON_STARTUP | pending_reduce_only_order_count=%s trading_halted=true manual_intervention_required=true",
            len(reduce_only_orders),
        )
