from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SidecarFillSnapshot:
    """Normalized sidecar TP fill quantities from leg dict + order status.

    ``filled_contracts`` is contracts (OKX accFillSz).
    ``filled_eth_qty`` is ETH.
    """

    leg_id: str = ""
    order_id: str = ""
    filled_contracts: float = 0.0
    filled_eth_qty: float = 0.0
    avg_fill_price: float | None = None
    tp_price: float | None = None
    filled_notional_usdt: float | None = None


def _safe_float(value: Any, *, positive_only: bool = False) -> float | None:
    """Parse *value* to float, returning None on any failure."""
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if positive_only and parsed <= 0:
        return None
    return parsed


def safe_positive_float(value: Any) -> float | None:
    """Public safe parser for a positive float. Returns None on any failure or non-positive."""
    return _safe_float(value, positive_only=True)


def normalize_sidecar_tp_fill(
    *,
    leg: dict[str, Any],
    status: dict[str, Any] | None = None,
    contract_multiplier: float = 0.1,
) -> SidecarFillSnapshot:
    """Normalize sidecar TP fill quantities from a leg dict and order status.

    Rules (priority-ordered):
    1. status["filled_qty"] is treated as **contracts** (OKX accFillSz).
    2. leg["contracts"] is treated as **contracts**.
    3. leg["qty"] is treated as **ETH qty**.

    filled_eth_qty:
      - If status["filled_qty"] exists: float(status["filled_qty"]) * contract_multiplier
      - Else: leg["qty"] (ETH)

    filled_contracts:
      - If status["filled_qty"] exists: float(status["filled_qty"])
      - Else: float(leg.get("contracts"))
      - If leg["contracts"] is also missing but leg["qty"] exists:
          filled_contracts = leg_qty / contract_multiplier
        (leg["qty"] is ETH, not contracts — must convert.)
    """
    status_dict = status or {}
    leg_id = str(leg.get("leg_id", ""))
    order_id = str(leg.get("tp_order_id", ""))

    # ── filled_contracts ──
    raw_filled = _safe_float(status_dict.get("filled_qty"), positive_only=True)
    if raw_filled is not None:
        filled_contracts = raw_filled
    else:
        leg_contracts = _safe_float(leg.get("contracts"))
        if leg_contracts is not None:
            filled_contracts = leg_contracts
        else:
            leg_qty_val = _safe_float(leg.get("qty"))
            if leg_qty_val is not None and contract_multiplier > 0:
                filled_contracts = leg_qty_val / contract_multiplier
            else:
                filled_contracts = leg_qty_val or 0.0

    # ── filled_eth_qty ──
    eth_from_status = _safe_float(status_dict.get("filled_qty"), positive_only=True)
    if eth_from_status is not None:
        # status["filled_qty"] is contracts → multiply by contract_multiplier
        if contract_multiplier > 0:
            filled_eth_qty = eth_from_status * contract_multiplier
        else:
            filled_eth_qty = eth_from_status  # fallback: treat as ETH directly
    else:
        filled_eth_qty = _safe_float(leg.get("qty")) or 0.0

    # ── prices ──
    avg_fill_price = _safe_float(status_dict.get("avg_fill_price"), positive_only=True)
    tp_price = _safe_float(leg.get("tp_price"), positive_only=True)

    # ── filled_notional_usdt ──
    fill_price = avg_fill_price or tp_price
    filled_notional_usdt: float | None = None
    if fill_price is not None and filled_eth_qty > 0:
        filled_notional_usdt = round(fill_price * filled_eth_qty, 8)

    return SidecarFillSnapshot(
        leg_id=leg_id,
        order_id=order_id,
        filled_contracts=filled_contracts,
        filled_eth_qty=filled_eth_qty,
        avg_fill_price=avg_fill_price,
        tp_price=tp_price,
        filled_notional_usdt=filled_notional_usdt,
    )
