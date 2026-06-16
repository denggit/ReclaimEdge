"""Realized profit calculation — pure helper.

Computes realized profit from TP1 and TP2 fills for a Three-Stage Runner
position.  No exchange calls, no I/O — only arithmetic on position state.
"""

from __future__ import annotations

from typing import Literal


def calculate_three_stage_realized_profit(
    *,
    side: Literal["LONG", "SHORT"],
    avg_entry_price: float,
    total_entry_qty: float,
    tp1_price: float | None,
    tp1_ratio: float,
    tp2_price: float | None,
    tp2_ratio: float,
    tp1_consumed: bool,
    tp2_consumed: bool,
) -> float | None:
    """Calculate realized profit from TP1 and TP2 fills.

    Returns:
        Total realized profit in USDT as a positive float, or ``None``
        when insufficient data prevents a reliable calculation.
    """
    if avg_entry_price <= 0 or total_entry_qty <= 0:
        return None

    realized = 0.0
    can_calculate = False

    # ── TP1 realized profit ────────────────────────────────────────────
    if tp1_consumed and tp1_price is not None and tp1_price > 0 and tp1_ratio > 0:
        tp1_qty = total_entry_qty * tp1_ratio
        if side == "LONG":
            tp1_profit = (tp1_price - avg_entry_price) * tp1_qty
        else:
            tp1_profit = (avg_entry_price - tp1_price) * tp1_qty
        if tp1_profit > 0:
            realized += tp1_profit
            can_calculate = True

    # ── TP2 realized profit ────────────────────────────────────────────
    if tp2_consumed and tp2_price is not None and tp2_price > 0 and tp2_ratio > 0:
        tp2_qty = total_entry_qty * tp2_ratio
        if side == "LONG":
            tp2_profit = (tp2_price - avg_entry_price) * tp2_qty
        else:
            tp2_profit = (avg_entry_price - tp2_price) * tp2_qty
        if tp2_profit > 0:
            realized += tp2_profit
            can_calculate = True

    if not can_calculate:
        return None

    return max(realized, 0.0)
