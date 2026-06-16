"""Tests for entry_sl_exit_classifier — exhaustive coverage of all classification rules."""

from __future__ import annotations

import pytest
from src.live.account_sync.entry_sl_exit_classifier import (
    EntrySlExitClassification,
    classify_entry_sl_exit_for_cooldown,
)


def _classify(**overrides):
    """Shorthand for classify_entry_sl_exit_for_cooldown with sensible defaults."""
    defaults = dict(
        entry_sl_cooldown_candidate=True,
        entry_protective_sl_order_id="algo-12345",
        filled_order_id=None,
        filled_algo_id=None,
        exit_reason=None,
        realized_delta=None,
        partial_tp_consumed=False,
        three_stage_tp1_consumed=False,
        three_stage_tp2_consumed=False,
        trend_runner_exit_reason=None,
        manual_close_detected=False,
        allow_loss_heuristic=True,
    )
    defaults.update(overrides)
    return classify_entry_sl_exit_for_cooldown(**defaults)


class TestCandidateGate:
    """candidate=False must always return SKIPPED."""

    def test_candidate_false_skipped(self) -> None:
        r = _classify(entry_sl_cooldown_candidate=False, realized_delta=-50.0)
        assert r.should_arm_cooldown is False
        assert r.confidence == "SKIPPED"
        assert r.reason == "not_candidate"


class TestPartialTpSkips:
    """Partial TP / Three-Stage TP consumed must skip."""

    def test_partial_tp_consumed_skipped(self) -> None:
        r = _classify(partial_tp_consumed=True, realized_delta=-50.0)
        assert r.should_arm_cooldown is False
        assert r.confidence == "SKIPPED"
        assert r.reason == "partial_tp_consumed"

    def test_three_stage_tp1_consumed_skipped(self) -> None:
        r = _classify(three_stage_tp1_consumed=True, realized_delta=-50.0)
        assert r.should_arm_cooldown is False
        assert r.confidence == "SKIPPED"
        assert r.reason == "three_stage_tp1_consumed"

    def test_three_stage_tp2_consumed_skipped(self) -> None:
        r = _classify(three_stage_tp2_consumed=True, realized_delta=-50.0)
        assert r.should_arm_cooldown is False
        assert r.confidence == "SKIPPED"
        assert r.reason == "three_stage_tp2_consumed"


class TestTrendRunnerExit:
    """trend_runner_exit_reason not None must skip."""

    def test_trend_runner_exit_skipped(self) -> None:
        r = _classify(trend_runner_exit_reason="reverse_burst", realized_delta=-50.0)
        assert r.should_arm_cooldown is False
        assert r.confidence == "SKIPPED"
        assert r.reason == "trend_runner_exit"


class TestManualClose:
    """manual_close_detected=True must skip even with loss."""

    def test_manual_close_loss_skipped(self) -> None:
        r = _classify(manual_close_detected=True, realized_delta=-50.0)
        assert r.should_arm_cooldown is False
        assert r.confidence == "SKIPPED"
        assert r.reason == "manual_close_detected"


class TestExactEntrySl:
    """EXACT entry SL matching via filled_algo_id, filled_order_id, or exit_reason."""

    def test_filled_algo_id_matches_entry_sl_order_id_arms_exact(self) -> None:
        r = _classify(
            entry_protective_sl_order_id="algo-abc",
            filled_algo_id="algo-abc",
            realized_delta=-50.0,
        )
        assert r.should_arm_cooldown is True
        assert r.confidence == "EXACT"
        assert r.reason == "entry_protective_sl_fill"

    def test_filled_order_id_matches_entry_sl_order_id_arms_exact(self) -> None:
        r = _classify(
            entry_protective_sl_order_id="algo-abc",
            filled_order_id="algo-abc",
            realized_delta=-50.0,
        )
        assert r.should_arm_cooldown is True
        assert r.confidence == "EXACT"
        assert r.reason == "entry_protective_sl_fill"

    def test_exit_reason_entry_protective_sl_arms_exact(self) -> None:
        r = _classify(
            entry_protective_sl_order_id="algo-abc",
            exit_reason="entry_protective_sl",
            realized_delta=-50.0,
        )
        assert r.should_arm_cooldown is True
        assert r.confidence == "EXACT"
        assert r.reason == "entry_protective_sl_fill"

    def test_exit_reason_entry_protective_sl_loss_flat_arms_exact(self) -> None:
        r = _classify(
            entry_protective_sl_order_id="algo-abc",
            exit_reason="entry_protective_sl_loss_flat",
            realized_delta=-50.0,
        )
        assert r.should_arm_cooldown is True
        assert r.confidence == "EXACT"
        assert r.reason == "entry_protective_sl_fill"


class TestHeuristicArming:
    """HEURISTIC arming via realized_delta < 0."""

    def test_realized_delta_negative_arms_heuristic(self) -> None:
        r = _classify(realized_delta=-30.0)
        assert r.should_arm_cooldown is True
        assert r.confidence == "HEURISTIC"
        assert r.reason == "negative_flat_before_partial_tp"

    def test_realized_delta_zero_skipped(self) -> None:
        r = _classify(realized_delta=0.0)
        assert r.should_arm_cooldown is False
        assert r.confidence == "SKIPPED"
        assert r.reason == "non_loss_flat"

    def test_realized_delta_positive_skipped(self) -> None:
        r = _classify(realized_delta=25.0)
        assert r.should_arm_cooldown is False
        assert r.confidence == "SKIPPED"
        assert r.reason == "non_loss_flat"

    def test_realized_delta_negative_but_heuristic_disabled_skipped(self) -> None:
        r = _classify(realized_delta=-50.0, allow_loss_heuristic=False)
        assert r.should_arm_cooldown is False
        assert r.confidence == "SKIPPED"
        assert r.reason == "loss_heuristic_disabled"


class TestExplicitNonSlExitReasons:
    """Known non-SL exit_reason values must skip."""

    def test_take_profit_skipped(self) -> None:
        r = _classify(exit_reason="take_profit", realized_delta=-50.0)
        assert r.should_arm_cooldown is False
        assert r.confidence == "SKIPPED"
        assert "non_sl_exit_reason" in r.reason

    def test_manual_close_skipped(self) -> None:
        r = _classify(exit_reason="manual_close", realized_delta=-50.0)
        assert r.should_arm_cooldown is False
        assert r.confidence == "SKIPPED"

    def test_market_exit_runner_skipped(self) -> None:
        r = _classify(exit_reason="market_exit_runner", realized_delta=-50.0)
        assert r.should_arm_cooldown is False
        assert r.confidence == "SKIPPED"

    def test_trend_runner_exit_skipped(self) -> None:
        r = _classify(exit_reason="trend_runner_exit", realized_delta=-50.0)
        assert r.should_arm_cooldown is False
        assert r.confidence == "SKIPPED"


class TestNoSignal:
    """When no signal is available, skip."""

    def test_no_realized_delta_skipped(self) -> None:
        r = _classify(realized_delta=None)
        assert r.should_arm_cooldown is False
        assert r.confidence == "SKIPPED"
        assert r.reason == "no_signal"
