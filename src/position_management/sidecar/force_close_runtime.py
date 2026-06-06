from __future__ import annotations

import os
from decimal import Decimal

from src.execution.trader import Trader
from src.live import runtime_types as live_runtime_types
from src.position_management.sidecar import runtime_state as sidecar_runtime_state
from src.position_management.sidecar.model import (
    SidecarLegStatus,
    sidecar_open_contracts,
    sidecar_open_qty,
)
from src.position_management.sidecar.reconciler import mark_sidecar_leg_force_closed
from src.reporting.live_state_store import LiveStateStore
from src.reporting.trade_journal import LiveTradeJournal
from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState
from src.utils.log import get_logger

logger = get_logger(__name__)


async def force_close_sidecar_after_core_flat(
        *,
        trader: Trader,
        strategy_state: StrategyPositionState,
        execution_state: live_runtime_types.ExecutionState,
        journal: LiveTradeJournal,
        state_store: LiveStateStore,
        trader_symbol: str,
        position_id: str | None,
        cash_before_position: float | None,
        ts_ms: int,
) -> bool:
    if sidecar_open_qty(strategy_state.sidecar_legs) <= 0:
        return True
    expected_sidecar_contracts = sidecar_open_contracts(strategy_state.sidecar_legs)
    okx_position = await trader.fetch_position_snapshot()
    tolerance = Decimal(str(os.getenv("SIDECAR_FORCE_CLOSE_CONTRACT_TOLERANCE", "0.01")))
    if (
            not okx_position.has_position
            or okx_position.side != strategy_state.side
            or abs(okx_position.contracts - expected_sidecar_contracts) > tolerance
    ):
        execution_state.trading_halted = True
        execution_state.halt_reason = "sidecar_force_close_position_mismatch"
        strategy_state.sidecar_dirty = True
        strategy_state.sidecar_halt_reason = "sidecar_force_close_position_mismatch"
        payload = {
            "okx_side": okx_position.side,
            "okx_contracts": str(okx_position.contracts),
            "sidecar_open_contracts": str(expected_sidecar_contracts),
            "tolerance": str(tolerance),
            "manual_intervention_required": True,
        }
        journal.append("SIDECAR_FORCE_CLOSE_POSITION_MISMATCH", payload, position_id=position_id)
        logger.error(
            "SIDECAR_FORCE_CLOSE_POSITION_MISMATCH | position_id=%s okx_side=%s okx_contracts=%s sidecar_open_contracts=%s tolerance=%s trading_halted=true manual_intervention_required=true",
            position_id,
            okx_position.side,
            okx_position.contracts,
            expected_sidecar_contracts,
            tolerance,
        )
        state_store.save(LiveStateStore.from_strategy_state(position_id=position_id, symbol=trader_symbol,
                                                            strategy_state=strategy_state,
                                                            cash_before_position=cash_before_position))
        return False
    try:
        for leg in strategy_state.sidecar_legs:
            if leg.get("status") == SidecarLegStatus.OPEN.value and leg.get("tp_order_id"):
                ok = await trader.cancel_sidecar_take_profit(str(leg["tp_order_id"]))
                if not ok:
                    raise RuntimeError(f"cancel_sidecar_tp_failed order_id={leg.get('tp_order_id')}")
        side = strategy_state.side
        if side is None:
            raise RuntimeError("side_missing")
        exit_ok, exit_message = await trader.market_exit_remaining_position_with_retries(
            side,
            retry_count=int(os.getenv("NEAR_TP_SL_FAIL_MARKET_EXIT_RETRY_COUNT", "3")),
        )
        if not exit_ok:
            raise RuntimeError(exit_message)
    except Exception as exc:
        execution_state.trading_halted = True
        execution_state.halt_reason = "sidecar_force_close_failed"
        strategy_state.sidecar_dirty = True
        strategy_state.sidecar_halt_reason = "sidecar_force_close_failed"
        journal.append("SIDECAR_FORCE_CLOSE_FAILED", {"error": str(exc), "manual_intervention_required": True},
                       position_id=position_id)
        logger.error("SIDECAR_FORCE_CLOSE_FAILED | position_id=%s error=%s manual_intervention_required=true",
                     position_id, exc)
        state_store.save(LiveStateStore.from_strategy_state(position_id=position_id, symbol=trader_symbol,
                                                            strategy_state=strategy_state,
                                                            cash_before_position=cash_before_position))
        return False
    strategy_state.sidecar_legs = [
        mark_sidecar_leg_force_closed(leg, ts_ms)
        if leg.get("status") in {SidecarLegStatus.OPEN.value, SidecarLegStatus.OPEN_UNPROTECTED.value}
        else leg
        for leg in strategy_state.sidecar_legs
    ]
    sidecar_runtime_state.refresh_sidecar_state_totals(strategy_state, int(os.getenv("SIDECAR_MAX_LEGS", "10")))
    journal.append("SIDECAR_FORCE_CLOSED_AFTER_CORE_FLAT", {"side": strategy_state.side, "reason": "core_flat"},
                   position_id=position_id)
    state_store.save(
        LiveStateStore.from_strategy_state(position_id=position_id, symbol=trader_symbol, strategy_state=strategy_state,
                                           cash_before_position=cash_before_position))
    return True
