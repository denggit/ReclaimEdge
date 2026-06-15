"""Pure Near-TP Reduce calculation helpers.

These functions do NOT import the strategy class, state, logger, or env.
All inputs are passed as explicit keyword arguments so the helpers remain
fully pure and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PositionSide = Literal["LONG", "SHORT"]
TpPlan = Literal["SINGLE", "MIDDLE_RUNNER", "THREE_STAGE_RUNNER"]


@dataclass(frozen=True)
class NearTpStateValues:
    near_tp_armed: bool
    near_tp_reduce_pending: bool
    near_tp_protected: bool
    near_tp_best_price: float | None
    near_tp_armed_ts_ms: int
    near_tp_pending_ts_ms: int
    near_tp_trigger_ts_ms: int
    near_tp_protective_sl_price: float | None
    near_tp_protective_sl_order_id: str | None
    near_tp_add_disabled: bool
    near_tp_sidecar_skip_logged: bool


@dataclass(frozen=True)
class NearTpGateDecision:
    allowed: bool
    reason: str


@dataclass(frozen=True)
class NearTpProgress:
    progress: float
    profit_pct: float
    near_by_distance: bool
    near_by_progress: bool
    min_profit_seen_ok: bool
    reduce_profit_ok: bool


@dataclass(frozen=True)
class NearTpBestPriceDecision:
    best_price: float
    changed: bool


@dataclass(frozen=True)
class NearTpGivebackDecision:
    giveback: float
    floating_profit_path: float
    threshold: float
    triggered: bool


# ── Reset state values ────────────────────────────────────────────────────

def reset_near_tp_state_values() -> NearTpStateValues:
    """Return a NearTpStateValues with all fields reset to defaults.

    Equivalent to the original _reset_near_tp_state field assignments.
    """
    return NearTpStateValues(
        near_tp_armed=False,
        near_tp_reduce_pending=False,
        near_tp_protected=False,
        near_tp_best_price=None,
        near_tp_armed_ts_ms=0,
        near_tp_pending_ts_ms=0,
        near_tp_trigger_ts_ms=0,
        near_tp_protective_sl_price=None,
        near_tp_protective_sl_order_id=None,
        near_tp_add_disabled=False,
        near_tp_sidecar_skip_logged=False,
    )


# ── Sidecar skip gating ───────────────────────────────────────────────────

def near_tp_sidecar_skip_allowed(
        *,
        sidecar_enabled_for_position: bool,
) -> NearTpGateDecision:
    """Return a gate decision for the Near-TP sidecar skip check.

    When sidecar is enabled for the position, Near-TP reduce should be
    skipped entirely (only the first time a log is emitted).
    """
    if sidecar_enabled_for_position:
        return NearTpGateDecision(allowed=False, reason="sidecar_enabled")
    return NearTpGateDecision(allowed=True, reason="ok")


# ── Plan / runner gating ──────────────────────────────────────────────────

def near_tp_plan_allowed(
        *,
        tp_plan: TpPlan,
        middle_runner_pending: bool,
        middle_runner_active: bool,
        three_stage_runner_enabled_for_position: bool,
        trend_runner_active: bool,
        partial_tp_consumed: bool,
) -> NearTpGateDecision:
    """Return a gate decision for Near-TP Reduce plan eligibility.

    Equivalent to the runner / split gating block at the top of
    _maybe_near_tp_reduce:

    - Middle Runner (pending or active) disables Near TP.
    - Three-Stage Runner enabled or Trend Runner active disables Near TP.
    """
    if tp_plan == "MIDDLE_RUNNER" or middle_runner_pending or middle_runner_active:
        return NearTpGateDecision(allowed=False, reason="middle_runner")
    if tp_plan == "THREE_STAGE_RUNNER" or three_stage_runner_enabled_for_position or trend_runner_active:
        return NearTpGateDecision(allowed=False, reason="three_stage_or_trend_runner")
    return NearTpGateDecision(allowed=True, reason="ok")


# ── Progress / profit / near-by-distance calculation ──────────────────────

def calculate_near_tp_progress(
        *,
        side: PositionSide,
        price: float,
        avg_entry_price: float,
        final_tp_price: float,
        near_tp_max_distance_usd: float,
        near_tp_min_reduce_profit_pct: float,
        near_tp_min_profit_pct: float,
        near_tp_min_progress_ratio: float,
) -> NearTpProgress | None:
    """Calculate Near-TP progress metrics.

    Returns None when the inputs are invalid (e.g. avg_entry_price <= 0,
    price <= 0, or final_tp_price on the wrong side of avg).
    """
    if avg_entry_price <= 0 or price <= 0:
        return None

    if side == "LONG":
        if final_tp_price <= avg_entry_price:
            return None
        progress = (price - avg_entry_price) / (final_tp_price - avg_entry_price)
        profit_pct = (price - avg_entry_price) / avg_entry_price
        near_by_distance = final_tp_price - price <= near_tp_max_distance_usd
    else:
        if final_tp_price >= avg_entry_price:
            return None
        progress = (avg_entry_price - price) / (avg_entry_price - final_tp_price)
        profit_pct = (avg_entry_price - price) / avg_entry_price
        near_by_distance = price - final_tp_price <= near_tp_max_distance_usd

    reduce_profit_ok = profit_pct >= near_tp_min_reduce_profit_pct
    min_profit_seen_ok = profit_pct >= near_tp_min_profit_pct
    near_by_progress = progress >= near_tp_min_progress_ratio

    return NearTpProgress(
        progress=progress,
        profit_pct=profit_pct,
        near_by_distance=near_by_distance,
        near_by_progress=near_by_progress,
        min_profit_seen_ok=min_profit_seen_ok,
        reduce_profit_ok=reduce_profit_ok,
    )


# ── Arming condition ──────────────────────────────────────────────────────

def should_arm_near_tp(*, progress: NearTpProgress) -> bool:
    """Return True when Near-TP should be armed.

    Equivalent to the original arm condition:
    (near_by_progress or near_by_distance) and min_profit_seen_ok
    """
    return (progress.near_by_progress or progress.near_by_distance) and progress.min_profit_seen_ok


# ── Best price update ─────────────────────────────────────────────────────

def update_near_tp_best_price(
        *,
        side: PositionSide,
        old_best_price: float | None,
        price: float,
) -> NearTpBestPriceDecision:
    """Compute the new best price and whether it changed.

    LONG:  best = max(old_best, price)
    SHORT: best = min(old_best, price)
    """
    old_best = old_best_price if old_best_price is not None else price
    if side == "LONG":
        best = max(old_best, price)
    else:
        best = min(old_best, price)
    changed = best != old_best
    return NearTpBestPriceDecision(best_price=best, changed=changed)


# ── Giveback / floating-profit-path / threshold ───────────────────────────

def calculate_near_tp_giveback(
        *,
        side: PositionSide,
        price: float,
        avg_entry_price: float,
        best_price: float,
        near_tp_giveback_usd: float,
        near_tp_giveback_pct: float,
        near_tp_giveback_profit_ratio: float,
) -> NearTpGivebackDecision:
    """Calculate giveback and whether it triggers the near-TP reduction.

    LONG:
      giveback = best_price - price
      floating_profit_path = best_price - avg_entry_price
    SHORT:
      giveback = price - best_price
      floating_profit_path = avg_entry_price - best_price

    threshold = max(
        near_tp_giveback_usd,
        price * near_tp_giveback_pct,
        floating_profit_path * near_tp_giveback_profit_ratio,
    )
    triggered = giveback >= threshold
    """
    if side == "LONG":
        giveback = best_price - price
        floating_profit_path = best_price - avg_entry_price
    else:
        giveback = price - best_price
        floating_profit_path = avg_entry_price - best_price

    threshold = max(
        near_tp_giveback_usd,
        price * near_tp_giveback_pct,
        floating_profit_path * near_tp_giveback_profit_ratio,
    )
    triggered = giveback >= threshold

    return NearTpGivebackDecision(
        giveback=giveback,
        floating_profit_path=floating_profit_path,
        threshold=threshold,
        triggered=triggered,
    )


# ── Protective SL ─────────────────────────────────────────────────────────

def calculate_near_tp_protective_sl(
        *,
        side: PositionSide,
        avg_entry_price: float,
        near_tp_protective_sl_profit_pct: float,
) -> float:
    """Calculate the near-TP protective stop-loss price.

    LONG:  avg_entry_price * (1 + pct)
    SHORT: avg_entry_price * (1 - pct)
    """
    if side == "LONG":
        return avg_entry_price * (1 + near_tp_protective_sl_profit_pct)
    return avg_entry_price * (1 - near_tp_protective_sl_profit_pct)


# ── Pending reduce trigger ────────────────────────────────────────────────

def near_tp_pending_can_reduce(*, reduce_profit_ok: bool) -> bool:
    """Return True when a pending Near-TP reduce should fire.

    When the position is in pending state and profit recovers to
    reduce_profit_ok, the reduce should execute.
    """
    return bool(reduce_profit_ok)
