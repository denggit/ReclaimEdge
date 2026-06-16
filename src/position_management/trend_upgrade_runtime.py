"""Trend Upgrade Add-on runtime helpers.

Pure helpers for applying Trend Upgrade Add-on state after successful
execution.  No exchange calls, no I/O — called from the execution command
processor inside ``state_lock``.
"""

from __future__ import annotations

from typing import Any

from src.position_management.cost_runtime import (
    record_remaining_entry_notional,
)
from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState


def apply_trend_upgrade_addon_state(
    state: StrategyPositionState,
    *,
    intent: Any,
    result: Any,
) -> None:
    """Apply Trend Upgrade Add-on state after successful execution.

    MUST only be called when ``result.ok`` is True and the intent's
    ``entry_regime`` is ``"TREND_UPGRADE_ADDON"``.

    Writes:
    - entry_regime / position_management_mode
    - trend_upgrade_active / trend_upgrade_addon_active
    - addon count, entry price, qty, risk budget, SL price
    - trend_trailing_sl_price
    """
    state.entry_regime = "TREND_UPGRADE_ADDON"
    state.position_management_mode = "TREND_UPGRADE_ADDON"
    state.trend_upgrade_active = True
    state.trend_upgrade_addon_active = True
    state.trend_upgrade_addon_count += 1
    state.trend_upgrade_addon_entry_price = float(intent.price)
    state.trend_upgrade_addon_qty = float(getattr(intent.size, "eth_qty", 0.0) or 0.0)

    # Risk budget: prefer size.risk_usdt, fallback to 0.0
    risk_budget = float(getattr(intent.size, "risk_usdt", 0.0) or 0.0)
    state.trend_upgrade_addon_risk_budget_usdt = risk_budget

    sl_price = getattr(intent, "entry_protective_sl_price", None)
    state.trend_upgrade_addon_sl_price = float(sl_price) if sl_price is not None else None
    state.trend_upgrade_last_ts_ms = int(intent.ts_ms)

    # Update trailing SL to the entry protective SL
    if sl_price is not None:
        state.trend_trailing_sl_price = float(sl_price)

    # Update entry protective SL from execution result
    sl_order_id = getattr(result, "protective_sl_order_id", None)
    if sl_order_id:
        state.entry_protective_sl_order_id = str(sl_order_id)
    state.entry_protective_sl_protected = bool(
        getattr(result, "protective_sl_ok", False)
    )


def update_core_position_cost_for_addon(
    state: StrategyPositionState,
    *,
    addon_qty: float,
    addon_notional: float,
    addon_price: float,
    fee_buffer_pct: float = 0.001,
) -> None:
    """Update core position cost fields after Trend Upgrade Add-on fill.

    Recomputes:
    - total_entry_qty
    - total_entry_notional
    - avg_entry_price
    - last_entry_price

    Also records the add-on notional via ``record_remaining_entry_notional``
    so that ``position_cost_entry_notional`` / ``position_cost_remaining_qty`` /
    ``net_remaining_breakeven_price`` stay consistent.
    """
    if addon_qty <= 0 or addon_notional <= 0 or addon_price <= 0:
        return

    old_qty = float(state.total_entry_qty or 0.0)
    old_notional = float(state.total_entry_notional or 0.0)

    new_qty = old_qty + addon_qty
    new_notional = old_notional + addon_notional
    new_avg = new_notional / new_qty if new_qty > 0 else 0.0

    state.total_entry_qty = new_qty
    state.total_entry_notional = new_notional
    state.avg_entry_price = new_avg
    state.last_entry_price = addon_price

    # Keep position cost tracking consistent
    record_remaining_entry_notional(
        state,
        qty=addon_qty,
        price=addon_price,
        fee_buffer_pct=fee_buffer_pct,
    )
