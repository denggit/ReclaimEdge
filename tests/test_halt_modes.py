"""Tests for halt_modes.py — unified halt mode intent helper."""

from __future__ import annotations

import pytest

from src.live.halt_modes import (
    FULL_HALT,
    ENTRY_HALT_POSITION_MANAGEMENT_ALLOWED,
    SIDECAR_DIRTY_HALT,
    CORE_POSITION_MANAGEMENT_INTENTS,
    POSITION_MANAGEMENT_INTENTS,
    allowed_intents_for_halt_mode,
    is_intent_allowed_during_halt,
    resolve_halt_mode,
)


# ── allowed_intents_for_halt_mode ──────────────────────────────────────


def test_full_halt_allows_nothing() -> None:
    assert allowed_intents_for_halt_mode(FULL_HALT) == frozenset()


def test_entry_halt_allows_position_management() -> None:
    assert allowed_intents_for_halt_mode(ENTRY_HALT_POSITION_MANAGEMENT_ALLOWED) == POSITION_MANAGEMENT_INTENTS


def test_sidecar_dirty_halt_allows_core_pm_only() -> None:
    """SIDECAR_DIRTY_HALT allows UPDATE_TP and MARKET_EXIT_RUNNER."""
    allowed = allowed_intents_for_halt_mode(SIDECAR_DIRTY_HALT)
    assert allowed == CORE_POSITION_MANAGEMENT_INTENTS
    assert "UPDATE_TP" in allowed
    assert "MARKET_EXIT_RUNNER" in allowed


def test_sidecar_dirty_halt_does_not_allow_open_long() -> None:
    allowed = allowed_intents_for_halt_mode(SIDECAR_DIRTY_HALT)
    assert "OPEN_LONG" not in allowed


def test_sidecar_dirty_halt_does_not_allow_add_long() -> None:
    allowed = allowed_intents_for_halt_mode(SIDECAR_DIRTY_HALT)
    assert "ADD_LONG" not in allowed


def test_unknown_mode_allows_nothing() -> None:
    """Unknown halt mode should return empty set (conservative)."""
    assert allowed_intents_for_halt_mode("UNKNOWN_MODE") == frozenset()


# ── is_intent_allowed_during_halt ──────────────────────────────────────


def test_is_intent_allowed_sidecar_dirty_update_tp() -> None:
    assert is_intent_allowed_during_halt("UPDATE_TP", SIDECAR_DIRTY_HALT) is True


def test_is_intent_allowed_sidecar_dirty_market_exit_runner() -> None:
    assert is_intent_allowed_during_halt("MARKET_EXIT_RUNNER", SIDECAR_DIRTY_HALT) is True


def test_is_intent_allowed_full_halt_nothing() -> None:
    for intent_type in ["UPDATE_TP", "MARKET_EXIT_RUNNER", "OPEN_LONG", "ADD_LONG"]:
        assert is_intent_allowed_during_halt(intent_type, FULL_HALT) is False


# ── halt_reason classification ─────────────────────────────────────────


def test_waiting_flat_reasons_are_full_halt() -> None:
    """All *_waiting_flat reasons must resolve to FULL_HALT."""
    waiting_flat_reasons = [
        "sidecar_tp_place_failed_market_exit_waiting_flat",
        "sidecar_tp_rate_limited_market_exit_waiting_flat",
        "sidecar_core_exit_delayed_market_exit_waiting_flat",
        "order_failure_delayed_market_exit_waiting_flat",
        "delayed_market_exit_waiting_flat",
    ]
    for reason in waiting_flat_reasons:
        assert resolve_halt_mode(reason) == FULL_HALT, f"{reason} should be FULL_HALT"


def test_delayed_market_exit_armed_sidecar_reasons_are_sidecar_dirty() -> None:
    """Sidecar-related delayed exit armed reasons resolve to SIDECAR_DIRTY_HALT."""
    reasons = [
        "sidecar_tp_place_failed_delayed_market_exit_armed",
        "sidecar_tp_place_rate_limited_delayed_market_exit_armed",
    ]
    for reason in reasons:
        assert resolve_halt_mode(reason) == SIDECAR_DIRTY_HALT, f"{reason} should be SIDECAR_DIRTY_HALT"


def test_delayed_market_exit_armed_protective_sl_reasons_are_full_halt() -> None:
    """Non-sidecar delayed exit armed reasons resolve to FULL_HALT."""
    reasons = [
        "three_stage_post_tp1_sl_failed_delayed_market_exit_armed",
        "middle_runner_sl_failed_delayed_market_exit_armed",
        "middle_bucket_fast_sl_failed_delayed_market_exit_armed",
        "middle_bucket_fast_sl_invalid_delayed_market_exit_armed",
        "core_tp_place_failed_delayed_market_exit_armed",
    ]
    for reason in reasons:
        assert resolve_halt_mode(reason) == FULL_HALT, f"{reason} should be FULL_HALT"


def test_sidecar_tp_place_failed_is_sidecar_dirty() -> None:
    assert resolve_halt_mode("sidecar_tp_place_failed") == SIDECAR_DIRTY_HALT


def test_sidecar_tp_rate_limited_unprotected_is_sidecar_dirty() -> None:
    assert resolve_halt_mode("sidecar_tp_place_rate_limited_unprotected") == SIDECAR_DIRTY_HALT


def test_rolling_loss_halt_is_entry_management() -> None:
    assert resolve_halt_mode("rolling_loss_soft_halt") == ENTRY_HALT_POSITION_MANAGEMENT_ALLOWED
    assert resolve_halt_mode("rolling_loss_hard_halt") == ENTRY_HALT_POSITION_MANAGEMENT_ALLOWED


def test_order_failure_failed_is_full_halt() -> None:
    assert resolve_halt_mode("order_failure_delayed_market_exit_failed") == FULL_HALT
