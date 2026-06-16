"""Entry SL exit classifier for post-entry SL cooldown decisions.

Pure functions — no OKX API calls, no strategy reads.  All classification
is driven by input parameters so the logic stays testable and audit-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EntrySlExitClassification:
    """Result of classifying a flat exit for post-entry SL cooldown arming."""

    should_arm_cooldown: bool
    reason: str
    confidence: str  # EXACT | HEURISTIC | SKIPPED
    realized_delta: float | None = None


# ── Exit reasons that are explicit non-SL and must skip cooldown ──────────
_EXPLICIT_NON_SL_EXIT_REASONS: frozenset[str] = frozenset(
    {
        "take_profit",
        "manual_close",
        "market_exit_runner",
        "trend_runner_exit",
    }
)


def classify_entry_sl_exit_for_cooldown(
    *,
    entry_sl_cooldown_candidate: bool,
    entry_protective_sl_order_id: str | None,
    filled_order_id: str | None = None,
    filled_algo_id: str | None = None,
    exit_reason: str | None = None,
    realized_delta: float | None = None,
    partial_tp_consumed: bool = False,
    three_stage_tp1_consumed: bool = False,
    three_stage_tp2_consumed: bool = False,
    trend_runner_exit_reason: str | None = None,
    manual_close_detected: bool = False,
    allow_loss_heuristic: bool = True,
) -> EntrySlExitClassification:
    """Classify whether a flat position exit should arm the post-entry SL cooldown.

    Rules (in priority order)
    --------------------------
    1. **EXACT entry SL** — when we can match the fill against the known
       entry protective SL order/algo ID, or the exit_reason explicitly
       marks an entry protective SL.
    2. **Explicit skip** — partial TP consumed, three-stage TP consumed,
       trend-runner exit, manual close, or known non-SL exit reasons.
    3. **HEURISTIC fallback** — candidate + negative realized_delta, when
       allow_loss_heuristic is enabled and manual_close is not detected.
    """

    # ── Gate: not a candidate at all ──────────────────────────────────
    if not entry_sl_cooldown_candidate:
        return EntrySlExitClassification(
            should_arm_cooldown=False,
            confidence="SKIPPED",
            reason="not_candidate",
            realized_delta=realized_delta,
        )

    # ── Explicit skips: partial / three-stage TP consumed ─────────────
    if partial_tp_consumed:
        return EntrySlExitClassification(
            should_arm_cooldown=False,
            confidence="SKIPPED",
            reason="partial_tp_consumed",
            realized_delta=realized_delta,
        )
    if three_stage_tp1_consumed:
        return EntrySlExitClassification(
            should_arm_cooldown=False,
            confidence="SKIPPED",
            reason="three_stage_tp1_consumed",
            realized_delta=realized_delta,
        )
    if three_stage_tp2_consumed:
        return EntrySlExitClassification(
            should_arm_cooldown=False,
            confidence="SKIPPED",
            reason="three_stage_tp2_consumed",
            realized_delta=realized_delta,
        )

    # ── Explicit skips: trend-runner / manual close / known non-SL ────
    if trend_runner_exit_reason is not None:
        return EntrySlExitClassification(
            should_arm_cooldown=False,
            confidence="SKIPPED",
            reason="trend_runner_exit",
            realized_delta=realized_delta,
        )
    if manual_close_detected:
        return EntrySlExitClassification(
            should_arm_cooldown=False,
            confidence="SKIPPED",
            reason="manual_close_detected",
            realized_delta=realized_delta,
        )
    if exit_reason is not None and exit_reason in _EXPLICIT_NON_SL_EXIT_REASONS:
        return EntrySlExitClassification(
            should_arm_cooldown=False,
            confidence="SKIPPED",
            reason=f"non_sl_exit_reason:{exit_reason}",
            realized_delta=realized_delta,
        )

    # ── EXACT entry SL ───────────────────────────────────────────────
    if entry_protective_sl_order_id is not None:
        if (
            (filled_order_id is not None and filled_order_id == entry_protective_sl_order_id)
            or (filled_algo_id is not None and filled_algo_id == entry_protective_sl_order_id)
        ):
            return EntrySlExitClassification(
                should_arm_cooldown=True,
                confidence="EXACT",
                reason="entry_protective_sl_fill",
                realized_delta=realized_delta,
            )

    if exit_reason in {"entry_protective_sl", "entry_protective_sl_loss_flat"}:
        return EntrySlExitClassification(
            should_arm_cooldown=True,
            confidence="EXACT",
            reason="entry_protective_sl_fill",
            realized_delta=realized_delta,
        )

    # ── HEURISTIC fallback ───────────────────────────────────────────
    if realized_delta is not None and realized_delta < 0:
        if not allow_loss_heuristic:
            return EntrySlExitClassification(
                should_arm_cooldown=False,
                confidence="SKIPPED",
                reason="loss_heuristic_disabled",
                realized_delta=realized_delta,
            )
        return EntrySlExitClassification(
            should_arm_cooldown=True,
            confidence="HEURISTIC",
            reason="negative_flat_before_partial_tp",
            realized_delta=realized_delta,
        )

    # ── Non-loss flat ────────────────────────────────────────────────
    if realized_delta is not None and realized_delta >= 0:
        return EntrySlExitClassification(
            should_arm_cooldown=False,
            confidence="SKIPPED",
            reason="non_loss_flat",
            realized_delta=realized_delta,
        )

    # ── No signal at all ─────────────────────────────────────────────
    return EntrySlExitClassification(
        should_arm_cooldown=False,
        confidence="SKIPPED",
        reason="no_signal",
        realized_delta=realized_delta,
    )
