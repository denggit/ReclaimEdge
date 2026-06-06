from __future__ import annotations

from typing import Any, Callable

from src.execution.trader import PositionSnapshot
from src.position_management.cost_basis import calculate_remaining_breakeven_price
from src.position_management.sidecar.model import sidecar_open_qty
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategy,
    StrategyPositionState,
)

DEFAULT_NET_REMAINING_FEE_BUFFER_PCT = 0.001


def sync_strategy_cost_from_position(
        strategy: BollCvdReclaimStrategy,
        position: PositionSnapshot,
        *,
        restore_from_position: Callable[[BollCvdReclaimStrategy, PositionSnapshot], None] | None = None,
) -> None:
    if not position.has_position or position.side is None or position.avg_entry_price <= 0:
        return
    if strategy.state.side is None or strategy.state.side != position.side or strategy.state.layers <= 0:
        if restore_from_position is not None:
            restore_from_position(strategy, position)
        return
    if getattr(strategy.state, "three_stage_runner_enabled_for_position", False):
        strategy.state.avg_entry_price = position.avg_entry_price
        strategy.state.last_entry_price = strategy.state.last_entry_price or position.avg_entry_price
        return
    strategy.state.total_entry_qty = position.eth_qty
    strategy.state.total_entry_notional = position.avg_entry_price * position.eth_qty
    strategy.state.avg_entry_price = position.avg_entry_price
    strategy.state.last_entry_price = strategy.state.last_entry_price or position.avg_entry_price


def refresh_net_remaining_breakeven(strategy_state: StrategyPositionState,
                                    fee_buffer_pct: float = DEFAULT_NET_REMAINING_FEE_BUFFER_PCT) -> None:
    if strategy_state.side not in {"LONG", "SHORT"}:
        strategy_state.net_remaining_breakeven_price = 0.0
        return
    basis = calculate_remaining_breakeven_price(
        side=strategy_state.side,
        entry_notional=float(getattr(strategy_state, "position_cost_entry_notional", 0.0) or 0.0),
        exit_notional=float(getattr(strategy_state, "position_cost_exit_notional", 0.0) or 0.0),
        remaining_qty=float(getattr(strategy_state, "position_cost_remaining_qty", 0.0) or 0.0),
        fee_buffer_pct=fee_buffer_pct,
    )
    strategy_state.net_remaining_breakeven_price = float(basis.buffered_breakeven_price or 0.0)


def record_remaining_entry_notional(
        strategy_state: StrategyPositionState,
        *,
        qty: float,
        price: float,
        fee_buffer_pct: float = DEFAULT_NET_REMAINING_FEE_BUFFER_PCT,
) -> None:
    if qty <= 0 or price <= 0:
        return
    strategy_state.position_cost_entry_notional += float(qty) * float(price)
    strategy_state.position_cost_remaining_qty += float(qty)
    refresh_net_remaining_breakeven(strategy_state, fee_buffer_pct)


def record_remaining_exit_notional(
        strategy_state: StrategyPositionState,
        *,
        qty: float,
        price: float,
        remaining_qty: float | None = None,
        fee_buffer_pct: float = DEFAULT_NET_REMAINING_FEE_BUFFER_PCT,
) -> None:
    if qty <= 0 or price <= 0:
        return
    strategy_state.position_cost_exit_notional += float(qty) * float(price)
    if remaining_qty is None:
        strategy_state.position_cost_remaining_qty = max(
            float(strategy_state.position_cost_remaining_qty or 0.0) - float(qty), 0.0)
    else:
        strategy_state.position_cost_remaining_qty = max(float(remaining_qty or 0.0), 0.0)
    refresh_net_remaining_breakeven(strategy_state, fee_buffer_pct)


def remaining_total_qty_from_core_position(strategy_state: StrategyPositionState,
                                           core_position: PositionSnapshot) -> float:
    return max(float(core_position.eth_qty or 0.0), 0.0) + sidecar_open_qty(
        list(getattr(strategy_state, "sidecar_legs", []) or []))


def record_core_position_reduction_exit(
        strategy_state: StrategyPositionState,
        core_position: PositionSnapshot,
        *,
        exit_price: float | None,
        fee_buffer_pct: float = DEFAULT_NET_REMAINING_FEE_BUFFER_PCT,
        expected_remaining_qty: float | None = None,
) -> None:
    price = float(exit_price or 0.0)
    if price <= 0:
        return

    total_entry_qty = float(getattr(strategy_state, "total_entry_qty", 0.0) or 0.0)
    new_remaining_qty = remaining_total_qty_from_core_position(strategy_state, core_position)
    old_remaining_qty = float(getattr(strategy_state, "position_cost_remaining_qty", 0.0) or 0.0)

    if expected_remaining_qty is not None and expected_remaining_qty > new_remaining_qty:
        # Simultaneous TP1+TP2: position was reduced by both TP1 and TP2
        # in the same fill.  Compute TP1's reduction directly from
        # total_entry_qty and the caller-supplied expected_remaining_qty,
        # excluding sidecar open qty from the core reduction.
        sidecar_qty_from_state = sidecar_open_qty(
            list(getattr(strategy_state, "sidecar_legs", []) or [])
        )
        expected_core_remaining = max(expected_remaining_qty - sidecar_qty_from_state, 0.0)
        reduced_qty = max(total_entry_qty - expected_core_remaining, 0.0)
        remaining_qty = expected_remaining_qty
    else:
        # Normal single reduction (or TP2 after TP1 in a previous cycle).
        # Use tracked state for chaining correctness, but clamp to prevent
        # inflation from stale state drift.
        reduced_qty = max(old_remaining_qty - new_remaining_qty, 0.0)
        max_reduced_qty = max(
            total_entry_qty - float(core_position.eth_qty or 0.0), 0.0
        )
        if reduced_qty > max_reduced_qty:
            reduced_qty = max_reduced_qty
        # Also clamp: reduced_qty must not be less than 0
        if reduced_qty <= 0:
            reduced_qty = max_reduced_qty
        remaining_qty = new_remaining_qty

    record_remaining_exit_notional(
        strategy_state,
        qty=reduced_qty,
        price=price,
        remaining_qty=remaining_qty,
        fee_buffer_pct=fee_buffer_pct,
    )


def record_sidecar_tp_fill_exit(
        strategy_state: StrategyPositionState,
        leg: dict[str, Any],
        status: dict[str, Any],
        *,
        fee_buffer_pct: float = DEFAULT_NET_REMAINING_FEE_BUFFER_PCT,
) -> None:
    filled_qty = _coerce_positive_float(status.get("filled_qty")) or _coerce_positive_float(leg.get("qty"))
    fill_price = _coerce_positive_float(status.get("avg_fill_price")) or _coerce_positive_float(leg.get("tp_price"))
    if filled_qty is None or fill_price is None:
        return
    record_remaining_exit_notional(
        strategy_state,
        qty=filled_qty,
        price=fill_price,
        fee_buffer_pct=fee_buffer_pct,
    )


def _coerce_positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed
