from __future__ import annotations

from dataclasses import dataclass, field
from logging import Logger
from typing import Any, Iterable


@dataclass(frozen=True)
class SidecarFillTelemetry:
    """Telemetry payload for a sidecar TP fill event.

    Designed to be built from a sidecar leg dict and order status dict,
    and then logged / merged / passed into cash-drift reasoning.
    """

    filled_count: int = 0
    filled_leg_ids: tuple[str, ...] = ()
    filled_order_ids: tuple[str, ...] = ()
    filled_qty: float = 0.0
    realized_notional: float | None = None
    source: str = ""


def empty_sidecar_fill_telemetry(source: str = "") -> SidecarFillTelemetry:
    """Return a zero-value telemetry payload for the given source."""
    return SidecarFillTelemetry(source=source)


def build_sidecar_fill_telemetry(
    source: str,
    leg: dict[str, Any],
    status: dict[str, Any],
) -> SidecarFillTelemetry:
    """Build a single-fill telemetry payload from one leg and its order status."""
    filled_qty_val = status.get("filled_qty")
    filled_qty: float = float(filled_qty_val) if filled_qty_val is not None else 0.0

    avg_fill_price_val = status.get("avg_fill_price")
    avg_fill_price: float | None = float(avg_fill_price_val) if avg_fill_price_val is not None else None

    leg_qty: float = float(leg.get("qty", 0.0))
    realized_notional: float | None = (
        round(avg_fill_price * filled_qty, 8)
        if avg_fill_price is not None and filled_qty > 0
        else None
    )

    leg_id = str(leg.get("leg_id", ""))
    order_id = str(leg.get("tp_order_id", ""))

    return SidecarFillTelemetry(
        filled_count=1,
        filled_leg_ids=(leg_id,),
        filled_order_ids=(order_id,),
        filled_qty=filled_qty if filled_qty > 0 else leg_qty,
        realized_notional=realized_notional,
        source=source,
    )


def merge_sidecar_fill_telemetry(
    items: Iterable[SidecarFillTelemetry],
) -> SidecarFillTelemetry:
    """Merge multiple telemetry items into a single summary."""
    filled_count = 0
    filled_leg_ids: list[str] = []
    filled_order_ids: list[str] = []
    filled_qty = 0.0
    realized_notional: float | None = None
    source = ""

    for t in items:
        if t.filled_count == 0:
            continue
        filled_count += t.filled_count
        filled_leg_ids.extend(t.filled_leg_ids)
        filled_order_ids.extend(t.filled_order_ids)
        filled_qty += t.filled_qty
        if t.realized_notional is not None:
            realized_notional = (
                (realized_notional or 0.0) + t.realized_notional
            )
        if t.source:
            source = t.source

    return SidecarFillTelemetry(
        filled_count=filled_count,
        filled_leg_ids=tuple(filled_leg_ids),
        filled_order_ids=tuple(filled_order_ids),
        filled_qty=filled_qty,
        realized_notional=realized_notional,
        source=source,
    )


def log_sidecar_tp_filled(
    logger: Logger,
    *,
    position_id: str | None,
    telemetry: SidecarFillTelemetry,
    leg: dict[str, Any] | None = None,
    status: dict[str, Any] | None = None,
    core_position: Any = None,
) -> None:
    """Emit a warning-level log for a sidecar TP fill event."""
    parts: list[str] = [
        f"source={telemetry.source}",
        f"position_id={position_id or '-'}",
    ]

    if telemetry.filled_leg_ids:
        parts.append(f"leg_ids={','.join(telemetry.filled_leg_ids)}")
    if telemetry.filled_order_ids:
        parts.append(f"order_ids={','.join(telemetry.filled_order_ids)}")
    parts.append(f"filled_qty={telemetry.filled_qty}")

    if status is not None:
        avg_fill_price = status.get("avg_fill_price")
        if avg_fill_price is not None:
            parts.append(f"avg_fill_price={avg_fill_price}")

    if leg is not None:
        tp_price = leg.get("tp_price")
        if tp_price is not None:
            parts.append(f"tp_price={tp_price}")
        parts.append(f"sidecar_open_qty_after={leg.get('qty', 0.0) * 0}")

    if telemetry.realized_notional is not None:
        parts.append(f"realized_notional={telemetry.realized_notional}")

    logger.warning("SIDECAR_TP_FILLED | %s", " ".join(parts))
