"""Pre-TP1 degrade replan helper — pure function shared by EntryAddFlowCoordinator
and TpUpdateCoordinator.

This module provides a single pure function that re-computes the pre-TP1
degrade stage cap based on current position age.  It is intentionally stateless:
no logging, no I/O, no env reads, no strategy access.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreTp1DegradeReplanDecision:
    """Result of re-computing the pre-TP1 degrade stage cap.

    Attributes:
        new_stage: The new degrade stage (None, "MIDDLE_RUNNER", or "SINGLE").
        degraded_ts_ms: Timestamp when the degrade was applied (0 if None).
        age_seconds: Position age in seconds computed from first_entry_ts_ms.
        cap_applicable: Whether the Three-Stage lifecycle cap is applicable.
        reason: A human-readable reason string for the decision.
    """

    new_stage: str | None
    degraded_ts_ms: int
    age_seconds: float
    cap_applicable: bool
    reason: str


def decide_pre_tp1_degrade_stage_for_replan(
    *,
    first_entry_ts_ms: int,
    ts_ms: int,
    is_pre_tp1: bool,
    three_stage_replan_cap_applicable: bool,
    degrade_enabled: bool,
    middle_runner_after_seconds: int,
    single_after_seconds: int,
) -> PreTp1DegradeReplanDecision:
    """Re-compute the pre-TP1 degrade stage based on current position age.

    This is a pure function — no side effects, no logging, no I/O.

    Rules (all based on ``first_entry_ts_ms``, which is never reset):

    * If ``is_pre_tp1`` is False → new_stage = None (post-TP1 / runner active)
    * If ``three_stage_replan_cap_applicable`` is False → new_stage = None
    * If ``degrade_enabled`` is False → new_stage = None
    * age >= ``single_after_seconds`` → new_stage = "SINGLE"
    * age >= ``middle_runner_after_seconds`` → new_stage = "MIDDLE_RUNNER"
    * Otherwise → new_stage = None
    """
    # Compute position age
    age_seconds: float
    if first_entry_ts_ms > 0:
        age_seconds = max((ts_ms - first_entry_ts_ms) / 1000.0, 0.0)
    else:
        age_seconds = 0.0

    # Guard: pre-TP1 lifecycle only — post-TP1 / runner active must not write
    # degrade stage
    if not is_pre_tp1:
        return PreTp1DegradeReplanDecision(
            new_stage=None,
            degraded_ts_ms=0,
            age_seconds=age_seconds,
            cap_applicable=False,
            reason="not_pre_tp1_lifecycle",
        )

    # Guard: cap only applies to Three-Stage lifecycle
    if not three_stage_replan_cap_applicable:
        return PreTp1DegradeReplanDecision(
            new_stage=None,
            degraded_ts_ms=0,
            age_seconds=age_seconds,
            cap_applicable=False,
            reason="not_three_stage_lifecycle",
        )

    # Guard: degrade disabled
    if not degrade_enabled:
        return PreTp1DegradeReplanDecision(
            new_stage=None,
            degraded_ts_ms=0,
            age_seconds=age_seconds,
            cap_applicable=True,
            reason="degrade_disabled",
        )

    # Age-based cap
    if age_seconds >= single_after_seconds:
        return PreTp1DegradeReplanDecision(
            new_stage="SINGLE",
            degraded_ts_ms=ts_ms,
            age_seconds=age_seconds,
            cap_applicable=True,
            reason="age_gte_single_after",
        )

    if age_seconds >= middle_runner_after_seconds:
        return PreTp1DegradeReplanDecision(
            new_stage="MIDDLE_RUNNER",
            degraded_ts_ms=ts_ms,
            age_seconds=age_seconds,
            cap_applicable=True,
            reason="age_gte_middle_runner_after",
        )

    return PreTp1DegradeReplanDecision(
        new_stage=None,
        degraded_ts_ms=0,
        age_seconds=age_seconds,
        cap_applicable=True,
        reason="age_under_middle_runner_after",
    )
