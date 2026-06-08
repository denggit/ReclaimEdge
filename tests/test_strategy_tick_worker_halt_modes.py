"""Tests: strategy_tick_worker halt mode intent filtering."""

from __future__ import annotations

import pytest

from src.live.halt_modes import (
    FULL_HALT,
    ENTRY_HALT_POSITION_MANAGEMENT_ALLOWED,
    SIDECAR_DIRTY_HALT,
    CORE_POSITION_MANAGEMENT_INTENTS,
    POSITION_MANAGEMENT_INTENTS,
    allowed_intents_for_halt_mode,
    allows_core_position_management,
    is_entry_blocked_by_halt,
    is_sidecar_blocked_by_halt,
    resolve_halt_mode,
)


# ── Mock intent for testing filtering ───────────────────────────────────


class FakeIntent:
    def __init__(self, intent_type: str) -> None:
        self.intent_type = intent_type


ENTRY_INTENTS = ["OPEN_LONG", "OPEN_SHORT", "ADD_LONG", "ADD_SHORT"]
PM_INTENTS = ["UPDATE_TP", "NEAR_TP_REDUCE", "MARKET_EXIT_RUNNER"]
CORE_PM_INTENTS = ["UPDATE_TP", "MARKET_EXIT_RUNNER"]


# ── halt_mode resolution tests ──────────────────────────────────────────


def test_halt_mode_full_halt_for_unknown_reason() -> None:
    assert resolve_halt_mode("some_unknown_reason") == FULL_HALT


def test_halt_mode_full_halt_for_none() -> None:
    assert resolve_halt_mode(None) == FULL_HALT


def test_halt_mode_sidecar_dirty_for_sidecar_fail() -> None:
    assert resolve_halt_mode("sidecar_tp_place_failed") == SIDECAR_DIRTY_HALT
    assert resolve_halt_mode("sidecar_tp_place_rate_limited_unprotected") == SIDECAR_DIRTY_HALT
    assert resolve_halt_mode("sidecar_dirty_unprotected") == SIDECAR_DIRTY_HALT


def test_halt_mode_waiting_flat_is_full_halt() -> None:
    """sidecar_tp_place_failed_market_exit_waiting_flat should now be FULL_HALT."""
    assert resolve_halt_mode("sidecar_tp_place_failed_market_exit_waiting_flat") == FULL_HALT
    assert resolve_halt_mode("sidecar_tp_rate_limited_market_exit_waiting_flat") == FULL_HALT


def test_halt_mode_entry_halt_for_rolling_loss() -> None:
    assert resolve_halt_mode("rolling_loss_soft_halt") == ENTRY_HALT_POSITION_MANAGEMENT_ALLOWED
    assert resolve_halt_mode("rolling_loss_hard_halt") == ENTRY_HALT_POSITION_MANAGEMENT_ALLOWED


# ── entry / sidecar blocking tests ──────────────────────────────────────


def test_entry_blocked_in_all_halt_modes() -> None:
    """New entries must be blocked in all halt modes."""
    for mode in [FULL_HALT, ENTRY_HALT_POSITION_MANAGEMENT_ALLOWED, SIDECAR_DIRTY_HALT]:
        assert is_entry_blocked_by_halt(mode) is True


def test_sidecar_blocked_in_full_and_sidecar_dirty() -> None:
    assert is_sidecar_blocked_by_halt(FULL_HALT) is True
    assert is_sidecar_blocked_by_halt(SIDECAR_DIRTY_HALT) is True


def test_sidecar_allowed_in_entry_halt() -> None:
    """In ENTRY_HALT, sidecar is NOT explicitly blocked (but new entries are)."""
    assert is_sidecar_blocked_by_halt(ENTRY_HALT_POSITION_MANAGEMENT_ALLOWED) is False


def test_core_pm_allowed_in_entry_and_sidecar_dirty() -> None:
    assert allows_core_position_management(ENTRY_HALT_POSITION_MANAGEMENT_ALLOWED) is True
    assert allows_core_position_management(SIDECAR_DIRTY_HALT) is True


def test_core_pm_blocked_in_full_halt() -> None:
    assert allows_core_position_management(FULL_HALT) is False


# ── Intent filtering simulation (using unified halt_modes helper) ────────


def _halt_filter_intents(intents: list[FakeIntent], halt_mode: str) -> list[FakeIntent]:
    """Replicate the filtering logic that strategy_tick_worker uses (now uses halt_modes helper)."""
    allowed = allowed_intents_for_halt_mode(halt_mode)
    return [i for i in intents if i.intent_type in allowed]


def test_full_halt_blocks_all_intents() -> None:
    intents = [FakeIntent(t) for t in ENTRY_INTENTS + PM_INTENTS]
    filtered = _halt_filter_intents(intents, FULL_HALT)
    assert len(filtered) == 0


def test_entry_halt_keeps_only_pm_intents() -> None:
    intents = [FakeIntent(t) for t in ["OPEN_LONG", "UPDATE_TP", "ADD_SHORT", "MARKET_EXIT_RUNNER", "NEAR_TP_REDUCE"]]
    filtered = _halt_filter_intents(intents, ENTRY_HALT_POSITION_MANAGEMENT_ALLOWED)
    # ENTRY_HALT allows UPDATE_TP, NEAR_TP_REDUCE, MARKET_EXIT_RUNNER
    types = [i.intent_type for i in filtered]
    assert "UPDATE_TP" in types
    assert "NEAR_TP_REDUCE" in types
    assert "MARKET_EXIT_RUNNER" in types
    assert "OPEN_LONG" not in types
    assert "ADD_SHORT" not in types


def test_sidecar_dirty_halt_keeps_only_core_pm_intents() -> None:
    """SIDECAR_DIRTY_HALT: only UPDATE_TP and MARKET_EXIT_RUNNER (no NEAR_TP_REDUCE)."""
    intents = [FakeIntent(t) for t in ["OPEN_LONG", "UPDATE_TP", "NEAR_TP_REDUCE", "ADD_LONG", "MARKET_EXIT_RUNNER"]]
    filtered = _halt_filter_intents(intents, SIDECAR_DIRTY_HALT)
    types = [i.intent_type for i in filtered]
    assert "UPDATE_TP" in types
    assert "MARKET_EXIT_RUNNER" in types
    assert "NEAR_TP_REDUCE" not in types
    assert "OPEN_LONG" not in types
    assert "ADD_LONG" not in types


def test_sidecar_dirty_halt_allows_core_update_tp() -> None:
    """UPDATE_TP must pass through SIDECAR_DIRTY_HALT filter."""
    intents = [FakeIntent("UPDATE_TP")]
    filtered = _halt_filter_intents(intents, SIDECAR_DIRTY_HALT)
    assert len(filtered) == 1
    assert filtered[0].intent_type == "UPDATE_TP"


def test_sidecar_dirty_halt_allows_market_exit_runner() -> None:
    """MARKET_EXIT_RUNNER must pass through SIDECAR_DIRTY_HALT filter."""
    intents = [FakeIntent("MARKET_EXIT_RUNNER")]
    filtered = _halt_filter_intents(intents, SIDECAR_DIRTY_HALT)
    assert len(filtered) == 1
    assert filtered[0].intent_type == "MARKET_EXIT_RUNNER"


def test_sidecar_dirty_halt_blocks_near_tp_reduce() -> None:
    """NEAR_TP_REDUCE must be blocked in SIDECAR_DIRTY_HALT."""
    intents = [FakeIntent("NEAR_TP_REDUCE")]
    filtered = _halt_filter_intents(intents, SIDECAR_DIRTY_HALT)
    assert len(filtered) == 0


def test_sidecar_dirty_halt_blocks_new_open() -> None:
    """OPEN_LONG must be blocked in SIDECAR_DIRTY_HALT."""
    intents = [FakeIntent("OPEN_LONG")]
    filtered = _halt_filter_intents(intents, SIDECAR_DIRTY_HALT)
    assert len(filtered) == 0


def test_sidecar_dirty_halt_blocks_add() -> None:
    """ADD_LONG must be blocked in SIDECAR_DIRTY_HALT."""
    intents = [FakeIntent("ADD_LONG")]
    filtered = _halt_filter_intents(intents, SIDECAR_DIRTY_HALT)
    assert len(filtered) == 0


# ── Delayed market exit armed reasons ──────────────────────────────────


def test_sidecar_delayed_exit_armed_is_sidecar_dirty() -> None:
    """Sidecar-related delayed exit armed reasons should be SIDECAR_DIRTY_HALT."""
    reasons = [
        "sidecar_tp_place_failed_delayed_market_exit_armed",
        "sidecar_tp_place_rate_limited_delayed_market_exit_armed",
    ]
    for r in reasons:
        assert resolve_halt_mode(r) == SIDECAR_DIRTY_HALT, r


def test_protective_sl_delayed_exit_armed_is_full_halt() -> None:
    """Protective SL delayed exit armed reasons should be FULL_HALT."""
    reasons = [
        "three_stage_post_tp1_sl_failed_delayed_market_exit_armed",
        "middle_runner_sl_failed_delayed_market_exit_armed",
        "middle_bucket_fast_sl_failed_delayed_market_exit_armed",
        "middle_bucket_fast_sl_invalid_delayed_market_exit_armed",
        "near_tp_protective_sl_failed_delayed_market_exit_armed",
        "core_tp_place_failed_delayed_market_exit_armed",
    ]
    for r in reasons:
        assert resolve_halt_mode(r) == FULL_HALT, r


# ── allowed_intents_for_halt_mode vs _halt_filter_intents consistency ──


def test_filter_uses_same_allowed_intents() -> None:
    """The filter function must return the same result as allowed_intents_for_halt_mode."""
    all_types = ENTRY_INTENTS + PM_INTENTS
    for mode in [FULL_HALT, ENTRY_HALT_POSITION_MANAGEMENT_ALLOWED, SIDECAR_DIRTY_HALT]:
        intents = [FakeIntent(t) for t in all_types]
        filtered = _halt_filter_intents(intents, mode)
        filtered_types = {i.intent_type for i in filtered}
        expected = allowed_intents_for_halt_mode(mode)
        assert filtered_types == set(expected), f"Mismatch for mode={mode}"
