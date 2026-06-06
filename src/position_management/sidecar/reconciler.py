from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import Any

from src.execution.trader import PositionSnapshot
from src.position_management.sidecar.model import SidecarLegStatus, calculate_sidecar_tp_price


def build_core_position_view(
        okx_position: PositionSnapshot,
        sidecar_open_qty: float,
        sidecar_open_contracts: Decimal | str | float | None = None,
) -> PositionSnapshot:
    if not okx_position.has_position:
        return okx_position
    open_qty = max(float(sidecar_open_qty or 0.0), 0.0)
    open_contracts = Decimal(str(sidecar_open_contracts)) if sidecar_open_contracts is not None else Decimal(
        str(open_qty / 0.1))
    core_eth_qty = max(float(okx_position.eth_qty) - open_qty, 0.0)
    core_contracts = max(okx_position.contracts - open_contracts, Decimal("0"))
    raw_pos = core_contracts if okx_position.side == "LONG" else -core_contracts
    if core_contracts <= 0 or core_eth_qty <= 0:
        return PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0"))
    return replace(okx_position, contracts=core_contracts, eth_qty=core_eth_qty, raw_pos=raw_pos)


def should_force_close_sidecar_after_core_flat(strategy_state: Any, okx_position: PositionSnapshot) -> bool:
    _ = okx_position
    return bool(
        getattr(strategy_state, "sidecar_open_qty", 0.0) > 0 and getattr(strategy_state, "sidecar_enabled_for_position",
                                                                         False))


def is_sidecar_dirty_missing_tp_order(leg: dict[str, Any]) -> bool:
    return bool(leg.get("status") == SidecarLegStatus.OPEN.value and not leg.get("tp_order_id"))


def sidecar_leg_from_fill(
        *,
        leg_id: str,
        position_id: str,
        layer_index: int,
        side: str,
        entry_price: float,
        qty: float,
        contracts: str,
        margin_pct: float,
        layer_multiplier: float,
        tp_pct: float,
        tp_order_id: str | None,
        ts_ms: int,
) -> dict[str, Any]:
    return {
        "leg_id": leg_id,
        "position_id": position_id,
        "layer_index": int(layer_index),
        "side": side,
        "entry_price": float(entry_price),
        "qty": float(qty),
        "contracts": str(contracts),
        "margin_pct": float(margin_pct),
        "layer_multiplier": float(layer_multiplier),
        "tp_pct": float(tp_pct),
        "tp_price": calculate_sidecar_tp_price(side, entry_price, tp_pct),  # type: ignore[arg-type]
        "tp_order_id": tp_order_id,
        "status": SidecarLegStatus.OPEN.value,
        "created_ts_ms": int(ts_ms),
        "updated_ts_ms": int(ts_ms),
    }


def mark_sidecar_leg_tp_filled(leg: dict[str, Any], ts_ms: int) -> dict[str, Any]:
    updated = dict(leg)
    updated["status"] = SidecarLegStatus.TP_FILLED.value
    updated["updated_ts_ms"] = int(ts_ms)
    return updated


def mark_sidecar_leg_force_closed(leg: dict[str, Any], ts_ms: int) -> dict[str, Any]:
    updated = dict(leg)
    updated["status"] = SidecarLegStatus.FORCE_CLOSED.value
    updated["updated_ts_ms"] = int(ts_ms)
    return updated


def mark_sidecar_leg_unknown_halted(leg: dict[str, Any], ts_ms: int, *, warning_recorded: bool = True) -> dict[
    str, Any]:
    updated = dict(leg)
    updated["status"] = SidecarLegStatus.UNKNOWN_HALTED.value
    updated["updated_ts_ms"] = int(ts_ms)
    updated["last_warning_ts_ms"] = int(ts_ms)
    updated["warning_recorded"] = warning_recorded
    return updated


def mark_sidecar_leg_open_unprotected(leg: dict[str, Any], ts_ms: int, *, warning_recorded: bool = True) -> dict[
    str, Any]:
    updated = dict(leg)
    updated["status"] = SidecarLegStatus.OPEN_UNPROTECTED.value
    updated["tp_order_id"] = None
    updated["updated_ts_ms"] = int(ts_ms)
    updated["last_warning_ts_ms"] = int(ts_ms)
    updated["warning_recorded"] = warning_recorded
    return updated
