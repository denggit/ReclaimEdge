"""Pure TP price selection and TP plan decision helpers.

These functions do NOT import the strategy class, state, logger, or env.
All state/config fields are received as explicit keyword arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TpMode = Literal["MIDDLE", "UPPER", "LOWER"]
TpPlan = Literal["SINGLE", "MIDDLE_RUNNER", "THREE_STAGE_RUNNER"]
PositionSide = Literal["LONG", "SHORT"]


@dataclass(frozen=True)
class TpBandSnapshot:
    """Snapshot of TP-only BOLL band fields, decoupled from BollSnapshot."""
    middle: float
    upper: float
    lower: float
    tp_middle: float | None
    tp_upper: float | None
    tp_lower: float | None
    tp_window: int | None


@dataclass(frozen=True)
class TpPriceSelection:
    price: float
    mode: TpMode


@dataclass(frozen=True)
class TpMiddleSelection:
    price: float
    source: str


@dataclass(frozen=True)
class TpOuterSelection:
    price: float
    source: str


@dataclass(frozen=True)
class TpPlanSelection:
    partial_tp_price: float | None
    partial_tp_ratio: float
    tp_plan: TpPlan


@dataclass(frozen=True)
class TpPlanUnchangedDecision:
    unchanged: bool


# ── TP_BOLL availability ────────────────────────────────────────────────

def tp_boll_available(
        *,
        tp_boll_enabled: bool,
        tp_middle: float | None,
        tp_upper: float | None,
        tp_lower: float | None,
) -> bool:
    """True when a valid TP-only BOLL snapshot is present."""
    return (
            tp_boll_enabled
            and tp_middle is not None
            and tp_upper is not None
            and tp_lower is not None
    )


# ── Middle price selection ──────────────────────────────────────────────

def select_tp_middle(
        *,
        tp_band: TpBandSnapshot,
        tp_boll_enabled: bool,
) -> TpMiddleSelection:
    """Return (middle_price, source) preferring TP_BOLL15 middle."""
    if tp_boll_available(
            tp_boll_enabled=tp_boll_enabled,
            tp_middle=tp_band.tp_middle,
            tp_upper=tp_band.tp_upper,
            tp_lower=tp_band.tp_lower,
    ):
        return TpMiddleSelection(price=float(tp_band.tp_middle), source="TP_BOLL")  # type: ignore[arg-type]
    return TpMiddleSelection(price=float(tp_band.middle), source="STRUCTURE_BOLL")


def select_tp_middle_with_profit_fallback(
        *,
        side: PositionSide,
        effective_be: float,
        min_net_profit: float,
        tp_band: TpBandSnapshot,
        tp_boll_enabled: bool,
) -> TpMiddleSelection | None:
    """Return (middle_price, source) for TP1 / first TP with profit-distance fallback.

    Unlike select_tp_middle() which is the raw low-level resolver, this
    helper enforces the min-net-profit check so that a TP1 price is never
    worse than what select_tp_price() would have accepted for SINGLE mode.

    Returns None when:
    - effective_be <= 0 (no breakeven → cannot validate profit).
    - Neither TP_BOLL15 middle nor structure BOLL20 middle meets the
      min-net-profit requirement (caller must fall back to SINGLE outer).

    LONG:  TP_BOLL15 middle first → structure BOLL20 middle if TP_BOLL15
           profit is insufficient → None if neither works.
    SHORT: TP_BOLL15 middle first → structure BOLL20 middle if TP_BOLL15
           profit is insufficient → None if neither works.
    """
    if effective_be <= 0:
        return None

    if side == "LONG":
        required = effective_be * (1 + min_net_profit)

        # 1) Try TP_BOLL15 middle
        tp_mid = select_tp_middle(tp_band=tp_band, tp_boll_enabled=tp_boll_enabled)
        if tp_mid.price >= required:
            return tp_mid

        # 2) Fallback to structure BOLL20 middle (explicit profit check)
        if tp_band.middle >= required:
            return TpMiddleSelection(price=float(tp_band.middle), source="STRUCTURE_BOLL_PROFIT_FALLBACK")

        # 3) Neither meets profit — None (caller must fall back to SINGLE outer)
        return None

    # SHORT
    required = effective_be * (1 - min_net_profit)

    tp_mid = select_tp_middle(tp_band=tp_band, tp_boll_enabled=tp_boll_enabled)
    if tp_mid.price <= required:
        return tp_mid

    if tp_band.middle <= required:
        return TpMiddleSelection(price=float(tp_band.middle), source="STRUCTURE_BOLL_PROFIT_FALLBACK")

    return None


# ── Outer price selection ───────────────────────────────────────────────

def select_tp_outer(
        *,
        side: PositionSide,
        tp_band: TpBandSnapshot,
        tp_boll_enabled: bool,
) -> TpOuterSelection:
    """Return (outer_price, source) for the given side."""
    if tp_boll_available(
            tp_boll_enabled=tp_boll_enabled,
            tp_middle=tp_band.tp_middle,
            tp_upper=tp_band.tp_upper,
            tp_lower=tp_band.tp_lower,
    ):
        if side == "LONG":
            return TpOuterSelection(price=float(tp_band.tp_upper), source="TP_BOLL")  # type: ignore[arg-type]
        return TpOuterSelection(price=float(tp_band.tp_lower), source="TP_BOLL")  # type: ignore[arg-type]
    if side == "LONG":
        return TpOuterSelection(price=float(tp_band.upper), source="STRUCTURE_BOLL")
    return TpOuterSelection(price=float(tp_band.lower), source="STRUCTURE_BOLL")


def select_tp_outer_with_profit_fallback(
        *,
        side: PositionSide,
        effective_be: float,
        min_net_profit: float,
        tp_band: TpBandSnapshot,
        tp_boll_enabled: bool,
) -> TpOuterSelection:
    """Select outer TP with profit-distance fallback.

    0) If the selected outer BOLL is on the loss side (vs effective breakeven),
       use half-min-profit fallback instead of locking in a losing TP.

    1) Try TP_BOLL15 outer first.
    2) If profit insufficient, try structure BOLL20 outer.
    3) If both insufficient, use the direction-correct farther outer
       as last resort (caller MUST log TP_OUTER_PROFIT_INSUFFICIENT_FALLBACK).

    LONG  outer valid: price >= effective_be * (1 + min_net_profit)
    SHORT outer valid: price <= effective_be * (1 - min_net_profit)
    """
    min_net_profit = abs(float(min_net_profit))
    half_min_profit = min_net_profit * 0.5

    if effective_be <= 0:
        return select_tp_outer(side=side, tp_band=tp_band, tp_boll_enabled=tp_boll_enabled)

    tp_outer = select_tp_outer(side=side, tp_band=tp_band, tp_boll_enabled=tp_boll_enabled)

    if side == "LONG":
        # If the selected outer BOLL is at or below breakeven, use half-min-profit fallback
        if tp_outer.price <= effective_be:
            return TpOuterSelection(
                price=effective_be * (1 + half_min_profit),
                source="TP_OUTER_HALF_MIN_PROFIT_FALLBACK",
            )

        required = effective_be * (1 + min_net_profit)
        if tp_outer.price >= required:
            return tp_outer

        # TP_BOLL15 outer insufficient → try structure BOLL20 outer
        structure_outer = float(tp_band.upper)
        if structure_outer >= required:
            return TpOuterSelection(price=structure_outer, source="STRUCTURE_BOLL_OUTER_PROFIT_FALLBACK")

        # Both insufficient → farther outer as last resort
        fallback = max(tp_outer.price, structure_outer)
        return TpOuterSelection(price=fallback, source="TP_OUTER_PROFIT_INSUFFICIENT_FALLBACK")

    # SHORT
    # If the selected outer BOLL is at or above breakeven, use half-min-profit fallback
    if tp_outer.price >= effective_be:
        return TpOuterSelection(
            price=effective_be * (1 - half_min_profit),
            source="TP_OUTER_HALF_MIN_PROFIT_FALLBACK",
        )

    required = effective_be * (1 - min_net_profit)
    if tp_outer.price <= required:
        return tp_outer

    # TP_BOLL15 outer insufficient → try structure BOLL20 outer
    structure_outer = float(tp_band.lower)
    if structure_outer <= required:
        return TpOuterSelection(price=structure_outer, source="STRUCTURE_BOLL_OUTER_PROFIT_FALLBACK")

    # Both insufficient → farther outer as last resort
    fallback = min(tp_outer.price, structure_outer)
    return TpOuterSelection(price=fallback, source="TP_OUTER_PROFIT_INSUFFICIENT_FALLBACK")


# ── Effective breakeven ─────────────────────────────────────────────────

def effective_breakeven_for_tp_selection(
        *,
        side: PositionSide,
        net_remaining_breakeven_price: float,
        avg_entry_price: float,
        breakeven_fee_buffer_pct: float,
) -> float:
    """Compute the effective breakeven price used for TP selection."""
    if net_remaining_breakeven_price > 0:
        return net_remaining_breakeven_price
    if avg_entry_price <= 0:
        return 0.0
    if side == "LONG":
        return avg_entry_price * (1 + breakeven_fee_buffer_pct)
    return avg_entry_price * (1 - breakeven_fee_buffer_pct)


# ── TP price selection ──────────────────────────────────────────────────

def select_tp_price(
        *,
        side: PositionSide,
        effective_be: float,
        min_net_profit: float,
        tp_band: TpBandSnapshot,
        tp_boll_enabled: bool,
) -> TpPriceSelection:
    """Select TP price preferring TP_BOLL15, with profit-distance fallback.

    For MIDDLE: TP_BOLL15 middle → BOLL20 middle → outer with profit fallback.
    Outer fallback: TP_BOLL15 outer → BOLL20 outer → farther outer as last resort.
    """
    min_net_profit = abs(float(min_net_profit))

    if effective_be <= 0:
        return TpPriceSelection(price=float(tp_band.middle), mode="MIDDLE")

    if side == "LONG":
        middle_required_price = effective_be * (1 + min_net_profit)

        # 1) Try TP_BOLL15 middle
        tp_mid = select_tp_middle(tp_band=tp_band, tp_boll_enabled=tp_boll_enabled)
        if tp_mid.price >= middle_required_price:
            return TpPriceSelection(price=tp_mid.price, mode="MIDDLE")

        # 2) Fallback to structure BOLL20 middle
        if tp_band.middle >= middle_required_price:
            return TpPriceSelection(price=float(tp_band.middle), mode="MIDDLE")

        # 3) Neither middle works — outer with profit fallback
        tp_outer = select_tp_outer_with_profit_fallback(
            side=side,
            effective_be=effective_be,
            min_net_profit=min_net_profit,
            tp_band=tp_band,
            tp_boll_enabled=tp_boll_enabled,
        )
        return TpPriceSelection(price=tp_outer.price, mode="UPPER")

    # SHORT
    middle_required_price = effective_be * (1 - min_net_profit)

    tp_mid = select_tp_middle(tp_band=tp_band, tp_boll_enabled=tp_boll_enabled)
    if tp_mid.price <= middle_required_price:
        return TpPriceSelection(price=tp_mid.price, mode="MIDDLE")

    if tp_band.middle <= middle_required_price:
        return TpPriceSelection(price=float(tp_band.middle), mode="MIDDLE")

    tp_outer = select_tp_outer_with_profit_fallback(
        side=side,
        effective_be=effective_be,
        min_net_profit=min_net_profit,
        tp_band=tp_band,
        tp_boll_enabled=tp_boll_enabled,
    )
    return TpPriceSelection(price=tp_outer.price, mode="LOWER")


# ── Plan-allowed gates ──────────────────────────────────────────────────

def three_stage_runner_plan_allowed(
        *,
        three_stage_runner_enabled: bool,
        three_stage_pre_tp1_degrade_stage: str | None,
        tp_mode: TpMode | None,
        boll_exists: bool,
        near_tp_protected: bool,
        near_tp_add_disabled: bool,
        partial_tp_consumed: bool,
        middle_runner_enabled_for_position: bool,
        middle_runner_pending: bool,
        middle_runner_active: bool,
        tp_plan: TpPlan | None,
        trend_runner_active: bool,
) -> bool:
    """Return True when Three-Stage Runner plan is allowed."""
    if not three_stage_runner_enabled:
        return False
    if three_stage_pre_tp1_degrade_stage is not None:
        return False
    if tp_mode != "MIDDLE" or not boll_exists:
        return False
    if near_tp_protected or near_tp_add_disabled:
        return False
    if partial_tp_consumed:
        return False
    if (
            middle_runner_enabled_for_position
            or middle_runner_pending
            or middle_runner_active
            or tp_plan == "MIDDLE_RUNNER"
            or trend_runner_active
    ):
        return False
    return True


def middle_runner_plan_allowed(
        *,
        middle_runner_enabled: bool,
        tp_mode: TpMode | None,
        boll_exists: bool,
        near_tp_protected: bool,
        near_tp_add_disabled: bool,
        partial_tp_consumed: bool,
        middle_runner_active: bool,
        three_stage_runner_enabled_for_position: bool,
        tp_plan: TpPlan | None,
        three_stage_tp1_consumed: bool,
        three_stage_tp2_consumed: bool,
) -> bool:
    """Return True when Middle Runner plan is allowed."""
    if not middle_runner_enabled:
        return False
    if tp_mode != "MIDDLE" or not boll_exists:
        return False
    if near_tp_protected or near_tp_add_disabled:
        return False
    if partial_tp_consumed:
        return False
    if middle_runner_active:
        return False
    if (
            three_stage_runner_enabled_for_position
            or tp_plan == "THREE_STAGE_RUNNER"
            or three_stage_tp1_consumed
            or three_stage_tp2_consumed
    ):
        return False
    return True


# ── TP plan selection ───────────────────────────────────────────────────

def select_tp_plan(
        *,
        side: PositionSide,
        final_tp: float,
        layers: int,
        tp_mode: TpMode | None,
        boll_exists: bool,
        three_stage_pre_tp1_degrade_stage: str | None,
        middle_runner_first_close_ratio: float,
        tp_middle_profit_fallback_price: float,
        three_stage_runner_plan_allowed: bool,
        three_stage_tp1_ratio: float,
        three_stage_runner_enabled: bool,
        middle_runner_plan_allowed: bool,
) -> TpPlanSelection:
    """Select the TP plan (SINGLE / MIDDLE_RUNNER / THREE_STAGE_RUNNER).

    All state/config fields are received as explicit parameters so the
    function is fully pure and testable.
    """
    if three_stage_pre_tp1_degrade_stage == "SINGLE":
        return TpPlanSelection(partial_tp_price=None, partial_tp_ratio=0.0, tp_plan="SINGLE")
    if three_stage_pre_tp1_degrade_stage == "MIDDLE_RUNNER":
        if tp_mode == "MIDDLE" and boll_exists:
            first_close_ratio = min(max(middle_runner_first_close_ratio, 0.1), 0.95)
            return TpPlanSelection(
                partial_tp_price=tp_middle_profit_fallback_price,
                partial_tp_ratio=first_close_ratio,
                tp_plan="MIDDLE_RUNNER",
            )
        return TpPlanSelection(partial_tp_price=None, partial_tp_ratio=0.0, tp_plan="SINGLE")
    if three_stage_runner_plan_allowed:
        return TpPlanSelection(
            partial_tp_price=tp_middle_profit_fallback_price,
            partial_tp_ratio=three_stage_tp1_ratio,
            tp_plan="THREE_STAGE_RUNNER",
        )
    if three_stage_runner_enabled:
        return TpPlanSelection(partial_tp_price=None, partial_tp_ratio=0.0, tp_plan="SINGLE")
    if middle_runner_plan_allowed:
        first_close_ratio = min(max(middle_runner_first_close_ratio, 0.1), 0.95)
        return TpPlanSelection(
            partial_tp_price=tp_middle_profit_fallback_price,
            partial_tp_ratio=first_close_ratio,
            tp_plan="MIDDLE_RUNNER",
        )
    # Fallback: SINGLE outer TP
    return TpPlanSelection(partial_tp_price=None, partial_tp_ratio=0.0, tp_plan="SINGLE")


# ── TP plan unchanged check ─────────────────────────────────────────────

def tp_plan_unchanged(
        *,
        current_tp_price: float | None,
        current_tp_plan: TpPlan | None,
        current_partial_tp_price: float | None,
        current_partial_tp_ratio: float,
        new_tp_price: float,
        new_partial_tp_price: float | None,
        new_partial_tp_ratio: float,
        new_tp_plan: TpPlan,
) -> TpPlanUnchangedDecision:
    """Return True when the new TP plan is unchanged vs the current one."""
    if current_tp_price is None:
        return TpPlanUnchangedDecision(unchanged=False)
    if abs(current_tp_price - new_tp_price) / new_tp_price >= 0.0001:
        return TpPlanUnchangedDecision(unchanged=False)
    if current_tp_plan != new_tp_plan:
        return TpPlanUnchangedDecision(unchanged=False)
    if abs(current_partial_tp_ratio - new_partial_tp_ratio) >= 0.0001:
        return TpPlanUnchangedDecision(unchanged=False)
    if current_partial_tp_price is None or new_partial_tp_price is None:
        unchanged = current_partial_tp_price is None and new_partial_tp_price is None
        return TpPlanUnchangedDecision(unchanged=unchanged)
    unchanged = abs(current_partial_tp_price - new_partial_tp_price) / new_partial_tp_price < 0.0001
    return TpPlanUnchangedDecision(unchanged=unchanged)
