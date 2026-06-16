from __future__ import annotations

# ── Halt mode constants ─────────────────────────────────────────────────

FULL_HALT = "FULL_HALT"
"""Completely stop all on_tick processing.  No entry, no TP update, no SL management."""

ENTRY_HALT_POSITION_MANAGEMENT_ALLOWED = "ENTRY_HALT_POSITION_MANAGEMENT_ALLOWED"
"""Block new entries and adds, but allow position management (UPDATE_TP, MARKET_EXIT_RUNNER)."""

"""Halt mode classification for ReclaimEdge live trading.

Every halt_reason is mapped to a halt_mode that dictates what the system
is allowed to do while halted.  This is NOT a tick-path module — it
contains no async, no IO, no strategy logic.

Sidecar runtime has been removed. SIDECAR_DIRTY_HALT and all sidecar halt
reasons have been removed. Unknown sidecar reasons default to FULL_HALT.
"""

# ── Halt mode constants ─────────────────────────────────────────────────

FULL_HALT = "FULL_HALT"
"""Completely stop all on_tick processing.  No entry, no TP update, no SL management."""

ENTRY_HALT_POSITION_MANAGEMENT_ALLOWED = "ENTRY_HALT_POSITION_MANAGEMENT_ALLOWED"
"""Block new entries and adds, but allow position management (UPDATE_TP, MARKET_EXIT_RUNNER)."""

# SIDECAR_DIRTY_HALT removed — Sidecar runtime has been removed.
# Legacy halt reasons that are still encountered default to FULL_HALT.

# ── Intent classification ──────────────────────────────────────────────

# All position management intents (used by rolling loss halt).
POSITION_MANAGEMENT_INTENTS = frozenset({
    "UPDATE_TP",
    "MARKET_EXIT_RUNNER",
})


def allowed_intents_for_halt_mode(halt_mode: str) -> frozenset[str]:
    """Return the set of intent_type values allowed while halted under *halt_mode*.

    Returns:
        A frozenset of intent_type strings.  An empty frozenset means
        **nothing** is allowed (FULL_HALT).
    """
    if halt_mode == FULL_HALT:
        return frozenset()
    if halt_mode == ENTRY_HALT_POSITION_MANAGEMENT_ALLOWED:
        return POSITION_MANAGEMENT_INTENTS
    # Unknown mode → conservative: allow nothing.
    return frozenset()


def is_intent_allowed_during_halt(intent_type: str, halt_mode: str) -> bool:
    """Check whether *intent_type* is permitted while the system is halted."""
    return intent_type in allowed_intents_for_halt_mode(halt_mode)


# ── Halt reasons that map to each mode ──────────────────────────────────

# Rolling loss halts — stop entries but keep managing the existing position.
_ROLLING_LOSS_HALT_REASONS = frozenset({
    "rolling_loss_soft_halt",
    "rolling_loss_hard_halt",
})

# Explicitly known FULL_HALT reasons.
_FULL_HALT_REASONS = frozenset({
    # ── Flat-settlement waiting states ────────────────────────────────
    "trend_runner_market_exit_waiting_flat",
    "three_stage_post_tp1_sl_cancel_failed_on_tp2",
    "three_stage_post_tp1_sl_failed_market_exit_waiting_flat",
    "three_stage_post_tp1_protective_sl_failure",
    "middle_runner_protective_sl_failure",
    "middle_bucket_fast_sl_failed_market_exit_waiting_flat",
    "middle_bucket_fast_sl_failed_market_exit_failed",
    "middle_bucket_fast_sl_invalid_market_exit_waiting_flat",
    "middle_bucket_fast_sl_invalid_market_exit_failed",
    "middle_bucket_fast_sl_invalid_halt_only",
    "three_stage_dirty_post_tp1_sl_blocks_runner_update",
    # ── Delayed market exit waiting-flat / failed ─────────────────────
    "order_failure_delayed_market_exit_waiting_flat",
    "order_failure_delayed_market_exit_failed",
    "delayed_market_exit_waiting_flat",
    # ── Delayed market exit armed (protective SL failures) ────────────
    "three_stage_post_tp1_sl_failed_delayed_market_exit_armed",
    "middle_runner_sl_failed_delayed_market_exit_armed",
    "middle_bucket_fast_sl_failed_delayed_market_exit_armed",
    "middle_bucket_fast_sl_invalid_delayed_market_exit_armed",
    "core_tp_place_failed_delayed_market_exit_armed",
})


def resolve_halt_mode(halt_reason: str | None) -> str:
    """Map a halt_reason to its halt mode.

    Returns one of:
        FULL_HALT
        ENTRY_HALT_POSITION_MANAGEMENT_ALLOWED

    Sidecar dirty halt mode has been removed. Legacy sidecar halt
    reasons will resolve to FULL_HALT for safety.
    """
    if not halt_reason:
        return FULL_HALT

    if halt_reason in _ROLLING_LOSS_HALT_REASONS:
        return ENTRY_HALT_POSITION_MANAGEMENT_ALLOWED

    if halt_reason in _FULL_HALT_REASONS:
        return FULL_HALT

    # Unknown halt — default to FULL_HALT for safety.
    return FULL_HALT


def is_entry_blocked_by_halt(halt_mode: str) -> bool:
    """Return True if OPEN_LONG/OPEN_SHORT/ADD_LONG/ADD_SHORT must be blocked."""
    return halt_mode in {FULL_HALT, ENTRY_HALT_POSITION_MANAGEMENT_ALLOWED}


def is_sidecar_blocked_by_halt(halt_mode: str) -> bool:
    """Return True if new sidecar actions must be blocked. (Sidecar removed, always True.)"""
    return True


def allows_core_position_management(halt_mode: str) -> bool:
    """Return True if core TP/SL/runner management is permitted."""
    return halt_mode in {ENTRY_HALT_POSITION_MANAGEMENT_ALLOWED, "SIDECAR_DIRTY_HALT"}
