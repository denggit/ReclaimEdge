"""Pure Three-Stage Runner state values, ratios, and post-TP1 SL helpers.

These functions do NOT import the strategy class, state, logger, or env.
All inputs are received as explicit keyword arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PositionSide = Literal["LONG", "SHORT"]


def _interpolate_to_middle(anchor: float, middle: float, ratio: float) -> float:
    """Interpolate from anchor toward middle by ratio."""
    return anchor + (middle - anchor) * ratio


# ── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ThreeStageRatios:
    """Normalized Three-Stage ratios (tp1 / tp2 / runner)."""
    tp1_ratio: float
    tp2_ratio: float
    runner_ratio: float


@dataclass(frozen=True)
class ThreeStageStateValues:
    """All Three-Stage Runner state fields as an immutable value object.

    Does NOT include Trend Runner fields.
    """
    three_stage_runner_enabled_for_position: bool
    three_stage_tp1_price: float | None
    three_stage_tp2_price: float | None
    three_stage_runner_initial_tp_price: float | None
    three_stage_tp1_ratio: float
    three_stage_tp2_ratio: float
    three_stage_runner_ratio: float
    three_stage_tp1_consumed: bool
    three_stage_tp2_consumed: bool
    three_stage_post_tp1_protective_sl_price: float | None
    three_stage_post_tp1_protective_sl_order_id: str | None
    three_stage_post_tp1_sl_extension_triggered: bool
    three_stage_post_tp1_protected: bool
    three_stage_post_tp1_sl_time_tighten_candle_count: int
    three_stage_post_tp1_sl_time_tighten_last_candle_ts_ms: int
    three_stage_post_tp1_sl_time_tighten_log_candle_ts_ms: int


@dataclass(frozen=True)
class ThreeStageDynamicTargetValues:
    """Fields updated by dynamic target update (no reset of consumed/protective)."""
    three_stage_runner_enabled_for_position: bool
    three_stage_tp1_price: float | None
    three_stage_tp2_price: float | None
    three_stage_tp1_ratio: float
    three_stage_tp2_ratio: float
    three_stage_runner_ratio: float


@dataclass(frozen=True)
class ThreeStagePostTp1ProtectiveSlDecision:
    """Result of calculating the post-TP1 protective SL."""
    protective_sl: float | None
    candidate_cost: float
    candidate_structure: float
    reason: str


@dataclass(frozen=True)
class ThreeStagePostTp1ExtensionDecision:
    """Result of evaluating the post-TP1 extension trigger."""
    protective_sl: float | None
    extension_triggered: bool
    trigger_price: float | None


# ── Ratio normalization ─────────────────────────────────────────────────────

def normalize_three_stage_ratios(
        *,
        tp1_ratio: float,
        tp2_ratio: float,
        runner_ratio: float,
) -> ThreeStageRatios:
    """Normalize the three ratios so they sum to 1.0.

    Equivalent to the original _normalized_three_stage_ratios.
    Falls back to 0.60 / 0.20 / 0.20 when total <= 0.
    """
    tp1 = max(float(tp1_ratio), 0.0)
    tp2 = max(float(tp2_ratio), 0.0)
    runner = max(float(runner_ratio), 0.0)
    total = tp1 + tp2 + runner
    if total <= 0:
        return ThreeStageRatios(tp1_ratio=0.60, tp2_ratio=0.20, runner_ratio=0.20)
    return ThreeStageRatios(
        tp1_ratio=tp1 / total,
        tp2_ratio=tp2 / total,
        runner_ratio=runner / total,
    )


# ── State value constructors ────────────────────────────────────────────────

def reset_three_stage_state_values() -> ThreeStageStateValues:
    """Return the reset / cleared values for all Three-Stage state fields.

    Equivalent to the original _reset_three_stage_runner_state Three-Stage
    field clearing, but does NOT include Trend Runner fields.
    """
    return ThreeStageStateValues(
        three_stage_runner_enabled_for_position=False,
        three_stage_tp1_price=None,
        three_stage_tp2_price=None,
        three_stage_runner_initial_tp_price=None,
        three_stage_tp1_ratio=0.0,
        three_stage_tp2_ratio=0.0,
        three_stage_runner_ratio=0.0,
        three_stage_tp1_consumed=False,
        three_stage_tp2_consumed=False,
        three_stage_post_tp1_protective_sl_price=None,
        three_stage_post_tp1_protective_sl_order_id=None,
        three_stage_post_tp1_sl_extension_triggered=False,
        three_stage_post_tp1_protected=False,
        three_stage_post_tp1_sl_time_tighten_candle_count=0,
        three_stage_post_tp1_sl_time_tighten_last_candle_ts_ms=0,
        three_stage_post_tp1_sl_time_tighten_log_candle_ts_ms=0,
    )


def planned_three_stage_state_values(
        *,
        tp1_price: float | None,
        tp2_price: float | None,
        ratios: ThreeStageRatios,
) -> ThreeStageStateValues:
    """Return the planned / initial values for Three-Stage Runner state.

    Equivalent to the original _set_three_stage_runner_planned Three-Stage
    field assignments, but does NOT include Trend Runner field clearing.
    """
    return ThreeStageStateValues(
        three_stage_runner_enabled_for_position=True,
        three_stage_tp1_price=tp1_price,
        three_stage_tp2_price=tp2_price,
        three_stage_runner_initial_tp_price=None,
        three_stage_tp1_ratio=ratios.tp1_ratio,
        three_stage_tp2_ratio=ratios.tp2_ratio,
        three_stage_runner_ratio=ratios.runner_ratio,
        three_stage_tp1_consumed=False,
        three_stage_tp2_consumed=False,
        three_stage_post_tp1_protective_sl_price=None,
        three_stage_post_tp1_protective_sl_order_id=None,
        three_stage_post_tp1_sl_extension_triggered=False,
        three_stage_post_tp1_protected=False,
        three_stage_post_tp1_sl_time_tighten_candle_count=0,
        three_stage_post_tp1_sl_time_tighten_last_candle_ts_ms=0,
        three_stage_post_tp1_sl_time_tighten_log_candle_ts_ms=0,
    )


def update_three_stage_dynamic_target_values(
        *,
        tp1_price: float | None,
        tp2_price: float | None,
        ratios: ThreeStageRatios,
) -> ThreeStageDynamicTargetValues:
    """Return dynamic target update values (no reset of consumed/protective).

    Equivalent to the original _update_three_stage_dynamic_targets_without_reset
    field assignments. Does NOT modify consumed flags, protective SL, or
    extension flags.
    """
    return ThreeStageDynamicTargetValues(
        three_stage_runner_enabled_for_position=True,
        three_stage_tp1_price=tp1_price,
        three_stage_tp2_price=tp2_price,
        three_stage_tp1_ratio=ratios.tp1_ratio,
        three_stage_tp2_ratio=ratios.tp2_ratio,
        three_stage_runner_ratio=ratios.runner_ratio,
    )


def reset_three_stage_post_tp1_sl_time_tighten_values() -> tuple[int, int, int]:
    """Return reset values for the three post-TP1 SL time-tighten fields.

    Returns (candle_count, last_candle_ts_ms, log_candle_ts_ms).
    """
    return 0, 0, 0


# ── SL tightening helpers ───────────────────────────────────────────────────

def tighten_three_stage_post_tp1_sl(
        *,
        side: PositionSide,
        old_sl: float,
        new_sl: float,
) -> float:
    """Tighten the Three-Stage post-TP1 protective SL in the correct direction.

    LONG  -> max(old_sl, new_sl)  (raise SL toward entry)
    SHORT -> min(old_sl, new_sl)  (lower SL toward entry)
    """
    if side == "LONG":
        return max(old_sl, new_sl)
    return min(old_sl, new_sl)


def tighten_optional_three_stage_post_tp1_sl(
        *,
        side: PositionSide,
        old_sl: float | None,
        new_sl: float | None,
) -> float | None:
    """Tighten optional Three-Stage post-TP1 SL.

    If new_sl is None, keep old_sl.
    If old_sl is None, adopt new_sl.
    Otherwise call tighten_three_stage_post_tp1_sl.
    """
    if new_sl is None:
        return old_sl
    if old_sl is None:
        return new_sl
    return tighten_three_stage_post_tp1_sl(side=side, old_sl=old_sl, new_sl=new_sl)


# ── Post-TP1 protective SL calculation ──────────────────────────────────────

def calculate_three_stage_post_tp1_protective_sl(
        *,
        side: PositionSide,
        current_price: float,
        avg_entry_price: float,
        net_remaining_breakeven_price: float,
        breakeven_fee_buffer_pct: float,
        tp1_price: float | None,
        tp1_ratio: float,
        boll_middle: float,
        boll_upper: float,
        boll_lower: float,
        sl_tighten_ratio: float,
) -> ThreeStagePostTp1ProtectiveSlDecision:
    """Pure calculation of the Three-Stage post-TP1 protective SL.

    Does NOT read state, write state, or emit logs.

    New logic (2025-06):
      - candidate_cost  = post-TP1 cost line (net_remaining_breakeven_price or formula fallback)
      - candidate_structure = opening-side outer band (LONG → lower, SHORT → upper)
      - protective_sl = max(cost, structure) for LONG / min(cost, structure) for SHORT
      - No interpolation toward middle; no clamp to middle.
      - sl_tighten_ratio is accepted for backward compatibility but ignored.
    """
    avg_entry = avg_entry_price
    base_breakeven = net_remaining_breakeven_price

    # Gate: current_price must be positive (always required, regardless of base_breakeven).
    if current_price <= 0:
        return ThreeStagePostTp1ProtectiveSlDecision(
            protective_sl=None,
            candidate_cost=0.0,
            candidate_structure=0.0,
            reason="missing_cost_basis",
        )

    # Gates that only matter when base_breakeven <= 0 (formula path).
    # When base_breakeven > 0, the calculation uses base_breakeven directly
    # and does not need avg_entry, tp1_price, or tp1_ratio.
    if base_breakeven <= 0:
        if avg_entry <= 0:
            return ThreeStagePostTp1ProtectiveSlDecision(
                protective_sl=None,
                candidate_cost=0.0,
                candidate_structure=0.0,
                reason="missing_cost_basis",
            )
        if tp1_price is None:
            return ThreeStagePostTp1ProtectiveSlDecision(
                protective_sl=None,
                candidate_cost=0.0,
                candidate_structure=0.0,
                reason="missing_tp1_price",
            )
        if tp1_ratio <= 0 or tp1_ratio >= 1:
            return ThreeStagePostTp1ProtectiveSlDecision(
                protective_sl=None,
                candidate_cost=0.0,
                candidate_structure=0.0,
                reason="invalid_tp1_ratio",
            )

    fee = breakeven_fee_buffer_pct
    # sl_tighten_ratio is no longer used — the SL is purely structural.

    if side == "LONG":
        if base_breakeven > 0:
            post_tp1_breakeven_buffered = base_breakeven
        else:
            post_tp1_breakeven = avg_entry - tp1_ratio * (float(tp1_price) - avg_entry) / (1 - tp1_ratio)
            post_tp1_breakeven_buffered = post_tp1_breakeven * (1 + fee)
        candidate_cost = post_tp1_breakeven_buffered
        candidate_structure = boll_lower
        protective_sl = max(candidate_cost, candidate_structure)
        if protective_sl >= current_price:
            return ThreeStagePostTp1ProtectiveSlDecision(
                protective_sl=None,
                candidate_cost=candidate_cost,
                candidate_structure=candidate_structure,
                reason="long_sl_not_below_current",
            )
        return ThreeStagePostTp1ProtectiveSlDecision(
            protective_sl=protective_sl,
            candidate_cost=candidate_cost,
            candidate_structure=candidate_structure,
            reason="calculated",
        )

    # SHORT
    if base_breakeven > 0:
        post_tp1_breakeven_buffered = base_breakeven
    else:
        post_tp1_breakeven = avg_entry + tp1_ratio * (avg_entry - float(tp1_price)) / (1 - tp1_ratio)
        post_tp1_breakeven_buffered = post_tp1_breakeven * (1 - fee)
    candidate_cost = post_tp1_breakeven_buffered
    candidate_structure = boll_upper
    protective_sl = min(candidate_cost, candidate_structure)
    if protective_sl <= current_price:
        return ThreeStagePostTp1ProtectiveSlDecision(
            protective_sl=None,
            candidate_cost=candidate_cost,
            candidate_structure=candidate_structure,
            reason="short_sl_not_above_current",
        )
    return ThreeStagePostTp1ProtectiveSlDecision(
        protective_sl=protective_sl,
        candidate_cost=candidate_cost,
        candidate_structure=candidate_structure,
        reason="calculated",
    )


# ── Post-TP1 extension trigger ──────────────────────────────────────────────

def apply_three_stage_post_tp1_extension_trigger(
        *,
        side: PositionSide,
        current_price: float,
        protective_sl: float | None,
        boll_middle: float,
        boll_upper: float,
        boll_lower: float,
        extension_trigger_ratio: float,
) -> ThreeStagePostTp1ExtensionDecision:
    """Pure evaluation of the Three-Stage post-TP1 extension trigger.

    Does NOT read state, write state, or emit logs.
    Equivalent to the original _apply_three_stage_post_tp1_extension_trigger
    calculation logic.
    """
    ratio = min(max(extension_trigger_ratio, 0.0), 1.0)

    if side == "LONG":
        trigger_price = boll_middle + (boll_upper - boll_middle) * ratio
        if current_price < trigger_price:
            return ThreeStagePostTp1ExtensionDecision(
                protective_sl=protective_sl,
                extension_triggered=False,
                trigger_price=trigger_price,
            )
        new_sl = boll_middle if protective_sl is None else max(protective_sl, boll_middle)
    else:
        trigger_price = boll_middle - (boll_middle - boll_lower) * ratio
        if current_price > trigger_price:
            return ThreeStagePostTp1ExtensionDecision(
                protective_sl=protective_sl,
                extension_triggered=False,
                trigger_price=trigger_price,
            )
        new_sl = boll_middle if protective_sl is None else min(protective_sl, boll_middle)

    return ThreeStagePostTp1ExtensionDecision(
        protective_sl=new_sl,
        extension_triggered=True,
        trigger_price=trigger_price,
    )
