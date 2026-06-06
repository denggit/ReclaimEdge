from __future__ import annotations

from src.position_management.sidecar.model import (
    SidecarLegStatus,
    sidecar_open_qty,
    trim_sidecar_legs_for_state,
)
from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState


def open_sidecar_legs_exceed_limit(
    state: StrategyPositionState,
    max_legs: int,
) -> bool:
    open_count = sum(
        1
        for leg in list(getattr(state, "sidecar_legs", []) or [])
        if leg.get("status") in {SidecarLegStatus.OPEN.value, SidecarLegStatus.OPEN_UNPROTECTED.value}
    )
    return open_count > max(int(max_legs), 1)


def refresh_sidecar_state_totals(
    state: StrategyPositionState,
    max_legs: int = 10,
) -> None:
    state.sidecar_legs = trim_sidecar_legs_for_state(list(getattr(state, "sidecar_legs", []) or []), max_legs)
    state.sidecar_open_qty = sidecar_open_qty(state.sidecar_legs)
    state.sidecar_total_qty = sum(float(leg.get("qty") or 0.0) for leg in state.sidecar_legs)
    state.sidecar_total_notional = sum(float(leg.get("qty") or 0.0) * float(leg.get("entry_price") or 0.0) for leg in state.sidecar_legs)
    state.sidecar_realized_qty = sum(
        float(leg.get("qty") or 0.0)
        for leg in state.sidecar_legs
        if leg.get("status") in {
            SidecarLegStatus.TP_FILLED.value,
            SidecarLegStatus.FORCE_CLOSED.value,
            SidecarLegStatus.CANCELLED.value,
        }
    )
