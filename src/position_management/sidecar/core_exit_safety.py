from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from src.position_management.sidecar.model import SidecarLegStatus, sanitize_okx_client_order_id


@dataclass(frozen=True)
class SidecarCoreExitRisk:
    risky: bool
    reason: str
    risky_leg_ids: tuple[str, ...]


def open_sidecar_legs(legs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        leg for leg in legs
        if leg.get("status") in {SidecarLegStatus.OPEN.value, SidecarLegStatus.OPEN_UNPROTECTED.value}
    ]


def core_tp_is_loss_for_position(side: str, core_tp_price: float, breakeven_price: float | None) -> bool:
    if breakeven_price is None or breakeven_price <= 0:
        return False
    if side == "LONG":
        return core_tp_price <= breakeven_price
    if side == "SHORT":
        return core_tp_price >= breakeven_price
    return False


def sidecar_leg_tp_is_beyond_core_exit(side: str, core_tp_price: float, leg_tp_price: float) -> bool:
    # LONG: price rises to TP. If sidecar TP is higher than core TP, core can close first.
    if side == "LONG":
        return leg_tp_price > core_tp_price
    # SHORT: price falls to TP. If sidecar TP is lower than core TP, core can close first.
    if side == "SHORT":
        return leg_tp_price < core_tp_price
    return False


def classify_sidecar_core_final_exit_risk(
    *,
    side: str,
    core_tp_price: float,
    breakeven_price: float | None,
    sidecar_legs: list[dict[str, Any]],
) -> SidecarCoreExitRisk:
    open_legs = open_sidecar_legs(sidecar_legs)
    if not open_legs:
        return SidecarCoreExitRisk(False, "no_open_sidecar_legs", ())
    if core_tp_is_loss_for_position(side, core_tp_price, breakeven_price):
        return SidecarCoreExitRisk(
            True,
            "core_tp_loss_vs_breakeven",
            tuple(str(leg.get("leg_id") or "") for leg in open_legs),
        )
    risky_leg_ids = []
    for leg in open_legs:
        try:
            leg_tp = float(leg.get("tp_price"))
        except Exception:
            risky_leg_ids.append(str(leg.get("leg_id") or ""))
            continue
        if sidecar_leg_tp_is_beyond_core_exit(side, core_tp_price, leg_tp):
            risky_leg_ids.append(str(leg.get("leg_id") or ""))
    if risky_leg_ids:
        return SidecarCoreExitRisk(True, "sidecar_tp_beyond_core_final_exit", tuple(risky_leg_ids))
    return SidecarCoreExitRisk(False, "sidecar_tp_reaches_before_or_at_core_exit", ())


def active_sidecar_tp_order_ids(sidecar_legs: list[dict[str, Any]]) -> tuple[str, ...]:
    """Return a deduplicated, order-preserving tuple of tp_order_id strings
    from every sidecar leg whose status is OPEN or OPEN_UNPROTECTED and
    whose tp_order_id is present (non-empty, non-None).

    This helper is intentionally pure:
      - no trader access
      - no network
      - no state writes
      - no strategy dependency
    """
    ids: list[str] = []
    for leg in open_sidecar_legs(sidecar_legs):
        order_id = leg.get("tp_order_id")
        if order_id:
            ids.append(str(order_id))
    return tuple(dict.fromkeys(ids))


def sidecar_core_exit_client_order_id(
    *,
    position_id: str | None,
    leg_id: str,
    old_tp_order_id: str | None,
    ts_ms: int,
) -> str:
    raw_key = f"{position_id or ''}|{leg_id}|{old_tp_order_id or ''}|{int(ts_ms)}"
    digest = hashlib.sha1(raw_key.encode("utf-8")).hexdigest()[:18]
    raw = f"SCE{digest}"
    return sanitize_okx_client_order_id(raw)
