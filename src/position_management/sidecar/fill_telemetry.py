from __future__ import annotations

from dataclasses import dataclass
from logging import Logger
from typing import Any, Iterable

from src.position_management.sidecar.fill_normalization import (
    SidecarFillSnapshot,
    normalize_sidecar_tp_fill,
)


@dataclass(frozen=True)
class SidecarFillTelemetry:
    """Telemetry payload for a sidecar TP fill event.

    Designed to be built from a sidecar leg dict and order status dict,
    and then logged / merged / passed into cash-drift reasoning.

    ``filled_qty`` is filled_eth_qty (ETH, not contracts).
    ``filled_contracts`` is the raw OKX accFillSz (contracts).
    """

    filled_count: int = 0
    filled_leg_ids: tuple[str, ...] = ()
    filled_order_ids: tuple[str, ...] = ()
    filled_qty: float = 0.0
    filled_contracts: float = 0.0
    filled_eth_qty: float = 0.0
    filled_notional_usdt: float | None = None
    realized_notional: float | None = None
    source: str = ""


def empty_sidecar_fill_telemetry(source: str = "") -> SidecarFillTelemetry:
    """Return a zero-value telemetry payload for the given source."""
    return SidecarFillTelemetry(source=source)


def _telemetry_from_snapshot(
    source: str,
    snapshot: SidecarFillSnapshot,
) -> SidecarFillTelemetry:
    """Build a SidecarFillTelemetry from a normalized SidecarFillSnapshot."""
    return SidecarFillTelemetry(
        filled_count=1,
        filled_leg_ids=(snapshot.leg_id,),
        filled_order_ids=(snapshot.order_id,),
        filled_qty=snapshot.filled_eth_qty,
        filled_contracts=snapshot.filled_contracts,
        filled_eth_qty=snapshot.filled_eth_qty,
        filled_notional_usdt=snapshot.filled_notional_usdt,
        realized_notional=snapshot.filled_notional_usdt,
        source=source,
    )


def build_sidecar_fill_telemetry(
    source: str,
    leg: dict[str, Any],
    status: dict[str, Any],
) -> SidecarFillTelemetry:
    """Build a single-fill telemetry payload from one leg and its order status.

    Uses normalize_sidecar_tp_fill to convert OKX accFillSz (contracts) to ETH.
    """
    snapshot = normalize_sidecar_tp_fill(leg=leg, status=status)
    return _telemetry_from_snapshot(source, snapshot)


def merge_sidecar_fill_telemetry(
    items: Iterable[SidecarFillTelemetry],
) -> SidecarFillTelemetry:
    """Merge multiple telemetry items into a single summary."""
    filled_count = 0
    filled_leg_ids: list[str] = []
    filled_order_ids: list[str] = []
    filled_qty = 0.0
    filled_contracts = 0.0
    filled_eth_qty = 0.0
    filled_notional_usdt: float | None = None
    source = ""

    for t in items:
        if t.filled_count == 0:
            continue
        filled_count += t.filled_count
        filled_leg_ids.extend(t.filled_leg_ids)
        filled_order_ids.extend(t.filled_order_ids)
        filled_qty += t.filled_qty
        filled_contracts += t.filled_contracts
        filled_eth_qty += t.filled_eth_qty
        if t.filled_notional_usdt is not None:
            filled_notional_usdt = (filled_notional_usdt or 0.0) + t.filled_notional_usdt
        if t.source:
            source = t.source

    return SidecarFillTelemetry(
        filled_count=filled_count,
        filled_leg_ids=tuple(filled_leg_ids),
        filled_order_ids=tuple(filled_order_ids),
        filled_qty=filled_qty,
        filled_contracts=filled_contracts,
        filled_eth_qty=filled_eth_qty,
        filled_notional_usdt=filled_notional_usdt,
        realized_notional=filled_notional_usdt,
        source=source,
    )


def log_sidecar_tp_filled(
    logger: Logger,
    *,
    position_id: str | None,
    telemetry: SidecarFillTelemetry,
    leg: dict[str, Any] | None = None,
    status: dict[str, Any] | None = None,
    sidecar_open_qty_after: float | None = None,
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
    parts.append(f"filled_contracts={telemetry.filled_contracts}")
    parts.append(f"filled_eth_qty={telemetry.filled_eth_qty}")

    if status is not None:
        avg_fill_price = status.get("avg_fill_price")
        if avg_fill_price is not None:
            parts.append(f"avg_fill_price={avg_fill_price}")

    if leg is not None:
        tp_price = leg.get("tp_price")
        if tp_price is not None:
            parts.append(f"tp_price={tp_price}")

    if telemetry.filled_notional_usdt is not None:
        parts.append(f"filled_notional_usdt={telemetry.filled_notional_usdt}")

    parts.append(f"sidecar_open_qty_after={sidecar_open_qty_after}")

    logger.warning("SIDECAR_TP_FILLED | %s", " ".join(parts))
