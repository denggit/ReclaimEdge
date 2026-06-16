from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PositionSide = Literal["LONG", "SHORT"]


def _interpolate_to_middle(anchor: float, middle: float, ratio: float) -> float:
    """Interpolate from anchor toward middle by ratio."""
    return anchor + (middle - anchor) * ratio


@dataclass(frozen=True)
class MiddleRunnerStateValues:
    """All Middle Runner state fields as an immutable value object."""

    middle_runner_enabled_for_position: bool
    middle_runner_pending: bool
    middle_runner_active: bool
    middle_runner_first_close_ratio: float
    middle_runner_keep_ratio: float
    middle_runner_first_tp_price: float | None
    middle_runner_final_tp_price: float | None
    middle_runner_protective_sl_price: float | None
    middle_runner_protective_sl_order_id: str | None
    middle_runner_extension_triggered: bool
    middle_runner_add_disabled: bool
    middle_runner_size_mismatch_protected: bool
    middle_runner_size_mismatch_warning_ts_ms: int
    middle_runner_sl_time_tighten_candle_count: int
    middle_runner_sl_time_tighten_last_candle_ts_ms: int
    middle_runner_sl_time_tighten_log_candle_ts_ms: int


@dataclass(frozen=True)
class MiddleRunnerProtectiveSlDecision:
    """Result of calculating the protective SL for the Middle Runner."""

    protective_sl: float | None
    candidate_cost: float
    candidate_structure: float
    reason: str


@dataclass(frozen=True)
class MiddleRunnerExtensionDecision:
    """Result of evaluating the Middle Runner extension trigger."""

    protective_sl: float | None
    extension_triggered: bool
    trigger_price: float | None


def reset_middle_runner_state_values() -> MiddleRunnerStateValues:
    """Return the reset / cleared values for all Middle Runner state fields.

    Equivalent to the original _reset_middle_runner_state field clearing.
    """
    return MiddleRunnerStateValues(
        middle_runner_enabled_for_position=False,
        middle_runner_pending=False,
        middle_runner_active=False,
        middle_runner_first_close_ratio=0.0,
        middle_runner_keep_ratio=0.0,
        middle_runner_first_tp_price=None,
        middle_runner_final_tp_price=None,
        middle_runner_protective_sl_price=None,
        middle_runner_protective_sl_order_id=None,
        middle_runner_extension_triggered=False,
        middle_runner_add_disabled=False,
        middle_runner_size_mismatch_protected=False,
        middle_runner_size_mismatch_warning_ts_ms=0,
        middle_runner_sl_time_tighten_candle_count=0,
        middle_runner_sl_time_tighten_last_candle_ts_ms=0,
        middle_runner_sl_time_tighten_log_candle_ts_ms=0,
    )


def planned_middle_runner_state_values(
        *,
        first_tp_price: float | None,
        final_tp_price: float,
        configured_first_close_ratio: float,
) -> MiddleRunnerStateValues:
    """Return the planned / initial values for Middle Runner state.

    Equivalent to the original _set_middle_runner_planned field assignments.
    """
    first_close_ratio = min(max(configured_first_close_ratio, 0.1), 0.95)
    return MiddleRunnerStateValues(
        middle_runner_enabled_for_position=True,
        middle_runner_pending=True,
        middle_runner_active=False,
        middle_runner_first_close_ratio=first_close_ratio,
        middle_runner_keep_ratio=1.0 - first_close_ratio,
        middle_runner_first_tp_price=first_tp_price,
        middle_runner_final_tp_price=final_tp_price,
        middle_runner_protective_sl_price=None,
        middle_runner_protective_sl_order_id=None,
        middle_runner_extension_triggered=False,
        middle_runner_add_disabled=False,
        middle_runner_size_mismatch_protected=False,
        middle_runner_size_mismatch_warning_ts_ms=0,
        middle_runner_sl_time_tighten_candle_count=0,
        middle_runner_sl_time_tighten_last_candle_ts_ms=0,
        middle_runner_sl_time_tighten_log_candle_ts_ms=0,
    )


def tighten_middle_runner_sl(
        *,
        side: PositionSide,
        old_sl: float,
        new_sl: float,
) -> float:
    """Tighten the Middle Runner protective SL in the correct direction.

    LONG  -> max(old_sl, new_sl)  (raise SL toward entry)
    SHORT -> min(old_sl, new_sl)  (lower SL toward entry)
    """
    if side == "LONG":
        return max(old_sl, new_sl)
    return min(old_sl, new_sl)


def tighten_optional_middle_runner_sl(
        *,
        side: PositionSide,
        old_sl: float | None,
        new_sl: float | None,
) -> float | None:
    """Tighten optional Middle Runner SL.

    If new_sl is None, keep old_sl.
    If old_sl is None, adopt new_sl.
    Otherwise call tighten_middle_runner_sl.
    """
    if new_sl is None:
        return old_sl
    if old_sl is None:
        return new_sl
    return tighten_middle_runner_sl(side=side, old_sl=old_sl, new_sl=new_sl)


def calculate_middle_runner_protective_sl(
        *,
        side: PositionSide,
        current_price: float,
        avg_entry_price: float,
        net_remaining_breakeven_price: float,
        breakeven_fee_buffer_pct: float,
        boll_middle: float,
        boll_upper: float,
        boll_lower: float,
        sl_tighten_ratio: float,
) -> MiddleRunnerProtectiveSlDecision:
    """Pure calculation of the Middle Runner protective SL.

    Does NOT read state, write state, or emit logs.

    New logic (2025-06):
      - candidate_cost  = cost line (net_remaining_breakeven_price or avg_entry ± fee)
      - candidate_structure = opening-side outer band (LONG → lower, SHORT → upper)
      - protective_sl = max(cost, structure) for LONG / min(cost, structure) for SHORT
      - No interpolation toward middle; no clamp to middle.
      - sl_tighten_ratio is accepted for backward compatibility but ignored.
    """
    avg_entry = avg_entry_price
    base_breakeven = net_remaining_breakeven_price

    if current_price <= 0 or (base_breakeven <= 0 and avg_entry <= 0):
        return MiddleRunnerProtectiveSlDecision(
            protective_sl=None,
            candidate_cost=0.0,
            candidate_structure=0.0,
            reason="missing_cost_basis",
        )

    fee = breakeven_fee_buffer_pct
    # sl_tighten_ratio is no longer used — the SL is purely structural.

    if side == "LONG":
        after_partial_breakeven = base_breakeven if base_breakeven > 0 else avg_entry * (1 + fee)
        candidate_cost = after_partial_breakeven
        candidate_structure = boll_lower
        protective_sl = max(candidate_cost, candidate_structure)
        if protective_sl >= current_price:
            return MiddleRunnerProtectiveSlDecision(
                protective_sl=None,
                candidate_cost=candidate_cost,
                candidate_structure=candidate_structure,
                reason="long_sl_not_below_current",
            )
        return MiddleRunnerProtectiveSlDecision(
            protective_sl=protective_sl,
            candidate_cost=candidate_cost,
            candidate_structure=candidate_structure,
            reason="calculated",
        )

    # SHORT
    after_partial_breakeven = base_breakeven if base_breakeven > 0 else avg_entry * (1 - fee)
    candidate_cost = after_partial_breakeven
    candidate_structure = boll_upper
    protective_sl = min(candidate_cost, candidate_structure)
    if protective_sl <= current_price:
        return MiddleRunnerProtectiveSlDecision(
            protective_sl=None,
            candidate_cost=candidate_cost,
            candidate_structure=candidate_structure,
            reason="short_sl_not_above_current",
        )
    return MiddleRunnerProtectiveSlDecision(
        protective_sl=protective_sl,
        candidate_cost=candidate_cost,
        candidate_structure=candidate_structure,
        reason="calculated",
    )


def apply_middle_runner_extension_trigger(
        *,
        side: PositionSide,
        current_price: float,
        protective_sl: float | None,
        boll_middle: float,
        boll_upper: float,
        boll_lower: float,
        extension_trigger_ratio: float,
        already_triggered: bool,
) -> MiddleRunnerExtensionDecision:
    """Pure evaluation of the Middle Runner extension trigger.

    Does NOT read state, write state, or emit logs.
    Equivalent to the original _apply_middle_runner_extension_trigger
    calculation logic.

    The *already_triggered* flag is accepted but NOT used by the pure
    function — the caller uses it to decide whether to emit the
    one-shot log.
    """
    _ = already_triggered  # reserved for caller-side logging decision

    ratio = min(max(extension_trigger_ratio, 0.0), 1.0)

    if side == "LONG":
        trigger_price = boll_middle + (boll_upper - boll_middle) * ratio
        if current_price < trigger_price:
            return MiddleRunnerExtensionDecision(
                protective_sl=protective_sl,
                extension_triggered=False,
                trigger_price=trigger_price,
            )
        new_sl = boll_middle if protective_sl is None else max(protective_sl, boll_middle)
    else:
        trigger_price = boll_middle - (boll_middle - boll_lower) * ratio
        if current_price > trigger_price:
            return MiddleRunnerExtensionDecision(
                protective_sl=protective_sl,
                extension_triggered=False,
                trigger_price=trigger_price,
            )
        new_sl = boll_middle if protective_sl is None else min(protective_sl, boll_middle)

    return MiddleRunnerExtensionDecision(
        protective_sl=new_sl,
        extension_triggered=True,
        trigger_price=trigger_price,
    )
