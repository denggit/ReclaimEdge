"""Tests: strategy_tick_worker halt mode intent filtering."""

from __future__ import annotations

import pytest

from src.live.halt_modes import (
    FULL_HALT,
    ENTRY_HALT_POSITION_MANAGEMENT_ALLOWED,
    SIDECAR_DIRTY_HALT,
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


# ── halt_mode resolution tests ──────────────────────────────────────────


def test_halt_mode_full_halt_for_unknown_reason() -> None:
    assert resolve_halt_mode("some_unknown_reason") == FULL_HALT


def test_halt_mode_full_halt_for_none() -> None:
    assert resolve_halt_mode(None) == FULL_HALT


def test_halt_mode_sidecar_dirty_for_sidecar_fail() -> None:
    assert resolve_halt_mode("sidecar_tp_place_failed") == SIDECAR_DIRTY_HALT
    assert resolve_halt_mode("sidecar_tp_place_rate_limited_unprotected") == SIDECAR_DIRTY_HALT
    assert resolve_halt_mode("sidecar_tp_place_failed_market_exit_waiting_flat") == SIDECAR_DIRTY_HALT
    assert resolve_halt_mode("sidecar_dirty_unprotected") == SIDECAR_DIRTY_HALT


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


# ── Intent filtering simulation (from strategy_tick_worker perspective) ──


def _halt_filter_intents(intents: list[FakeIntent], halt_mode: str) -> list[FakeIntent]:
    """Replicate the filtering logic that strategy_tick_worker uses."""
    if halt_mode == FULL_HALT:
        return []
    if halt_mode in {ENTRY_HALT_POSITION_MANAGEMENT_ALLOWED, SIDECAR_DIRTY_HALT}:
        return [i for i in intents if i.intent_type in {"UPDATE_TP", "NEAR_TP_REDUCE", "MARKET_EXIT_RUNNER"}]
    return intents


def test_full_halt_blocks_all_intents() -> None:
    intents = [FakeIntent(t) for t in ENTRY_INTENTS + PM_INTENTS]
    filtered = _halt_filter_intents(intents, FULL_HALT)
    assert len(filtered) == 0


def test_entry_halt_keeps_only_pm_intents() -> None:
    intents = [FakeIntent(t) for t in ["OPEN_LONG", "UPDATE_TP", "ADD_SHORT", "MARKET_EXIT_RUNNER"]]
    filtered = _halt_filter_intents(intents, ENTRY_HALT_POSITION_MANAGEMENT_ALLOWED)
    assert [i.intent_type for i in filtered] == ["UPDATE_TP", "MARKET_EXIT_RUNNER"]


def test_sidecar_dirty_halt_keeps_only_pm_intents() -> None:
    """SIDECAR_DIRTY_HALT: core TP/SL management allowed, no entries, no sidecar."""
    intents = [FakeIntent(t) for t in ["OPEN_LONG", "UPDATE_TP", "NEAR_TP_REDUCE", "ADD_LONG"]]
    filtered = _halt_filter_intents(intents, SIDECAR_DIRTY_HALT)
    assert [i.intent_type for i in filtered] == ["UPDATE_TP", "NEAR_TP_REDUCE"]


def test_sidecar_dirty_halt_allows_core_update_tp() -> None:
    """UPDATE_TP must pass through SIDECAR_DIRTY_HALT filter."""
    intents = [FakeIntent("UPDATE_TP")]
    filtered = _halt_filter_intents(intents, SIDECAR_DIRTY_HALT)
    assert len(filtered) == 1
    assert filtered[0].intent_type == "UPDATE_TP"


def test_sidecar_dirty_halt_blocks_new_open() -> None:
    """OPEN_LONG must be blocked in SIDECAR_DIRTY_HALT."""
    intents = [FakeIntent("OPEN_LONG")]
    filtered = _halt_filter_intents(intents, SIDECAR_DIRTY_HALT)
    assert len(filtered) == 0
