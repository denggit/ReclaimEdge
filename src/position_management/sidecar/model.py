from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Literal

PositionSide = Literal["LONG", "SHORT"]


class SidecarLegStatus(str, Enum):
    OPEN = "OPEN"
    TP_FILLED = "TP_FILLED"
    FORCE_CLOSED = "FORCE_CLOSED"
    CANCELLED = "CANCELLED"
    UNKNOWN_HALTED = "UNKNOWN_HALTED"


@dataclass(frozen=True)
class SidecarLeg:
    leg_id: str
    position_id: str
    layer_index: int
    side: PositionSide
    entry_price: float
    qty: float
    contracts: str
    margin_pct: float
    layer_multiplier: float
    tp_pct: float
    tp_price: float
    tp_order_id: str | None
    status: SidecarLegStatus
    created_ts_ms: int
    updated_ts_ms: int
    last_warning_ts_ms: int = 0
    warning_recorded: bool = False


def calculate_core_margin_pct(layer_margin_pct: float, sidecar_enabled: bool, sidecar_margin_pct: float) -> float:
    if not sidecar_enabled:
        return float(layer_margin_pct)
    return float(layer_margin_pct) - float(sidecar_margin_pct)


def calculate_sidecar_tp_price(side: PositionSide, entry_price: float, tp_pct: float) -> float:
    if side == "LONG":
        return float(entry_price) * (1.0 + float(tp_pct))
    return float(entry_price) * (1.0 - float(tp_pct))


def calculate_sidecar_margin(layer_margin_pct: float, sidecar_margin_pct: float, layer_multiplier: float) -> float:
    _ = layer_margin_pct
    return float(sidecar_margin_pct) * float(layer_multiplier)


def calculate_sidecar_qty(
    *,
    account_equity_usdt: float,
    price: float,
    leverage: float,
    layer_margin_pct: float,
    sidecar_margin_pct: float,
    layer_multiplier: float,
) -> float:
    if price <= 0:
        return 0.0
    margin_pct = calculate_sidecar_margin(layer_margin_pct, sidecar_margin_pct, layer_multiplier)
    notional = float(account_equity_usdt) * margin_pct * float(leverage)
    return notional / float(price)


def sidecar_open_qty(legs: list[dict[str, Any]] | list[SidecarLeg]) -> float:
    total = 0.0
    for leg in legs:
        status = _leg_value(leg, "status")
        if status == SidecarLegStatus.OPEN.value:
            total += float(_leg_value(leg, "qty") or 0.0)
    return total


def serialize_sidecar_legs(legs: list[SidecarLeg] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for leg in legs:
        if isinstance(leg, SidecarLeg):
            item = asdict(leg)
            item["status"] = leg.status.value
        else:
            item = dict(leg)
            if isinstance(item.get("status"), SidecarLegStatus):
                item["status"] = item["status"].value
        serialized.append(item)
    return serialized


def deserialize_sidecar_legs(data: Any) -> list[SidecarLeg]:
    if not isinstance(data, list):
        return []
    legs: list[SidecarLeg] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            raw_status = str(item.get("status") or SidecarLegStatus.UNKNOWN_HALTED.value)
            status = SidecarLegStatus(raw_status)
            legs.append(
                SidecarLeg(
                    leg_id=str(item["leg_id"]),
                    position_id=str(item["position_id"]),
                    layer_index=int(item["layer_index"]),
                    side=str(item["side"]),  # type: ignore[arg-type]
                    entry_price=float(item["entry_price"]),
                    qty=float(item["qty"]),
                    contracts=str(item["contracts"]),
                    margin_pct=float(item["margin_pct"]),
                    layer_multiplier=float(item["layer_multiplier"]),
                    tp_pct=float(item["tp_pct"]),
                    tp_price=float(item["tp_price"]),
                    tp_order_id=item.get("tp_order_id"),
                    status=status,
                    created_ts_ms=int(item["created_ts_ms"]),
                    updated_ts_ms=int(item["updated_ts_ms"]),
                    last_warning_ts_ms=int(item.get("last_warning_ts_ms") or 0),
                    warning_recorded=bool(item.get("warning_recorded", False)),
                )
            )
        except Exception:
            continue
    return legs


def trim_sidecar_legs_for_state(legs: list[dict[str, Any]] | list[SidecarLeg], max_legs: int) -> list[dict[str, Any]]:
    max_count = max(int(max_legs), 1)
    serialized = serialize_sidecar_legs(legs)
    open_legs = [leg for leg in serialized if leg.get("status") == SidecarLegStatus.OPEN.value]
    if len(open_legs) > max_count:
        open_legs.sort(key=lambda leg: int(leg.get("created_ts_ms") or 0))
        return open_legs
    recent_done = [leg for leg in serialized if leg.get("status") != SidecarLegStatus.OPEN.value]
    recent_done.sort(key=lambda leg: int(leg.get("updated_ts_ms") or leg.get("created_ts_ms") or 0), reverse=True)
    kept = open_legs + recent_done[: max(max_count - len(open_legs), 0)]
    kept.sort(key=lambda leg: int(leg.get("created_ts_ms") or 0))
    return kept


def _leg_value(leg: dict[str, Any] | SidecarLeg, key: str) -> Any:
    if isinstance(leg, SidecarLeg):
        value = getattr(leg, key)
        if isinstance(value, SidecarLegStatus):
            return value.value
        return value
    return leg.get(key)
