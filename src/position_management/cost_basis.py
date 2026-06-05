from __future__ import annotations

from dataclasses import dataclass

from src.position_management.sidecar.model import PositionSide
from src.utils.log import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RemainingCostBasis:
    side: PositionSide
    entry_notional: float
    exit_notional: float
    remaining_qty: float
    raw_breakeven_price: float | None
    buffered_breakeven_price: float | None


def calculate_remaining_breakeven_price(
    *,
    side: PositionSide,
    entry_notional: float,
    exit_notional: float,
    remaining_qty: float,
    fee_buffer_pct: float,
) -> RemainingCostBasis:
    safe_remaining_qty = float(remaining_qty or 0.0)
    safe_entry_notional = float(entry_notional or 0.0)
    safe_exit_notional = float(exit_notional or 0.0)
    if safe_remaining_qty <= 0:
        return RemainingCostBasis(
            side=side,
            entry_notional=safe_entry_notional,
            exit_notional=safe_exit_notional,
            remaining_qty=safe_remaining_qty,
            raw_breakeven_price=None,
            buffered_breakeven_price=None,
        )

    raw = (safe_entry_notional - safe_exit_notional) / safe_remaining_qty
    if raw <= 0:
        logger.warning(
            "REMAINING_COST_BASIS_WARNING | reason=non_positive_raw_breakeven side=%s entry_notional=%.8f exit_notional=%.8f remaining_qty=%.8f raw=%.8f",
            side,
            safe_entry_notional,
            safe_exit_notional,
            safe_remaining_qty,
            raw,
        )
        raw = max(raw, 0.0)

    fee = max(float(fee_buffer_pct or 0.0), 0.0)
    buffered = raw * (1.0 + fee) if side == "LONG" else raw * (1.0 - fee)
    return RemainingCostBasis(
        side=side,
        entry_notional=safe_entry_notional,
        exit_notional=safe_exit_notional,
        remaining_qty=safe_remaining_qty,
        raw_breakeven_price=raw,
        buffered_breakeven_price=buffered,
    )
