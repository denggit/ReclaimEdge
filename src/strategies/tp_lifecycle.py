"""Helpers for TP lifecycle state transitions."""

from __future__ import annotations


def is_pre_tp1_lifecycle(state: object) -> bool:
    """Return True only while the position is still in pre-TP1 lifecycle."""
    return (
        not bool(getattr(state, "three_stage_tp1_consumed", False))
        and not bool(getattr(state, "three_stage_tp2_consumed", False))
        and not bool(getattr(state, "trend_runner_active", False))
        and not bool(getattr(state, "middle_runner_active", False))
        and not bool(getattr(state, "partial_tp_consumed", False))
    )


def recover_pre_tp1_degrade_stage_after_add(
    *,
    state: object,
    position_age_seconds: float | None,
) -> str | None:
    """Adjust the pre-TP1 degrade cap after an add-position TP replan."""
    current_stage = getattr(state, "three_stage_pre_tp1_degrade_stage", None)
    if not is_pre_tp1_lifecycle(state):
        return current_stage
    if position_age_seconds is None:
        return current_stage

    age = max(float(position_age_seconds), 0.0)
    if age < 3 * 60 * 60:
        setattr(state, "three_stage_pre_tp1_degrade_stage", None)
        if hasattr(state, "three_stage_pre_tp1_degraded_ts_ms"):
            setattr(state, "three_stage_pre_tp1_degraded_ts_ms", 0)
        return None
    if age < 6 * 60 * 60:
        setattr(state, "three_stage_pre_tp1_degrade_stage", "MIDDLE_RUNNER")
        return "MIDDLE_RUNNER"

    setattr(state, "three_stage_pre_tp1_degrade_stage", "SINGLE")
    return "SINGLE"
