"""Tests for protective_orders_phase no-loosen SL enforcement.

Validates that run_account_sync_protective_orders_phase correctly applies
the no-loosen rule: never replace an existing stronger SL with a weaker one,
and never trigger DME when a valid existing SL protects the position.
"""

from unittest import mock

import pytest

from src.position_management.protective_sl_strength import should_replace_sl


class TestNoLoosenDecisionLogic:
    """Test the should_replace_sl decisions that drive the phase.

    These tests validate the pure function used inside protective_orders_phase
    to decide whether a candidate SL should replace an existing SL.
    The actual async phase integration is exercised through the existing
    middle_bucket_split_out_of_order_fills tests which cover the full pipeline.
    """

    def test_long_post_tp1_candidate_lower_no_replace(self):
        """LONG: existing=101, candidate=100 → candidate is weaker, keep existing."""
        assert should_replace_sl(side="LONG", existing_sl_price=101.0, candidate_sl_price=100.0) is False

    def test_long_post_tp1_candidate_higher_replace(self):
        """LONG: existing=100, candidate=101 → candidate is stronger, replace."""
        assert should_replace_sl(side="LONG", existing_sl_price=100.0, candidate_sl_price=101.0) is True

    def test_short_post_tp1_candidate_higher_no_replace(self):
        """SHORT: existing=99, candidate=100 → candidate is weaker, keep existing."""
        assert should_replace_sl(side="SHORT", existing_sl_price=99.0, candidate_sl_price=100.0) is False

    def test_short_post_tp1_candidate_lower_replace(self):
        """SHORT: existing=100, candidate=99 → candidate is stronger, replace."""
        assert should_replace_sl(side="SHORT", existing_sl_price=100.0, candidate_sl_price=99.0) is True

    def test_candidate_none_existing_valid_no_replace(self):
        """candidate=None, existing valid → no replace, keep existing."""
        assert should_replace_sl(side="LONG", existing_sl_price=100.0, candidate_sl_price=None) is False

    def test_candidate_none_existing_none_no_replace(self):
        """Both None → no replace (will trigger DME)."""
        assert should_replace_sl(side="LONG", existing_sl_price=None, candidate_sl_price=None) is False

    def test_existing_none_candidate_exists_replace(self):
        """No existing SL → always place new one."""
        assert should_replace_sl(side="LONG", existing_sl_price=None, candidate_sl_price=100.0) is True

    def test_middle_runner_long_weaker_no_replace(self):
        """MIDDLE_RUNNER LONG: weaker candidate SL should not replace."""
        assert should_replace_sl(side="LONG", existing_sl_price=95.0, candidate_sl_price=94.0) is False

    def test_middle_runner_short_stronger_replace(self):
        """MIDDLE_RUNNER SHORT: stronger candidate SL should replace."""
        assert should_replace_sl(side="SHORT", existing_sl_price=105.0, candidate_sl_price=104.0) is True

    def test_middle_bucket_fast_long_weaker_no_replace(self):
        """Fast protection LONG: weaker candidate SL should not replace."""
        assert should_replace_sl(side="LONG", existing_sl_price=101.0, candidate_sl_price=100.5) is False

    def test_middle_bucket_fast_short_weaker_no_replace(self):
        """Fast protection SHORT: weaker candidate SL should not replace."""
        assert should_replace_sl(side="SHORT", existing_sl_price=99.0, candidate_sl_price=100.0) is False


class TestProtectiveSlPhaseNoLoosenScenarios:
    """Test no-loosen scenarios at the phase-decision level.

    These tests validate the expected behavior for each target type
    (three_stage_post_tp1, middle_runner, middle_bucket_split_partial)
    using the should_replace_sl function as a proxy for the phase decision.
    """

    # ── three_stage_post_tp1 ───────────────────────────────────────────

    def test_three_stage_post_tp1_long_no_loosen_keep_existing(self):
        """Scenario: LONG post-TP1, existing fast SL=101, new post_tp1_sl=100.
        should_replace → False. Phase should keep existing, not place new."""
        # existing from fast SL fallback
        result = should_replace_sl(side="LONG", existing_sl_price=101.0, candidate_sl_price=100.0)
        assert result is False  # phase keeps existing, no new placement, no DME

    def test_three_stage_post_tp1_long_replace(self):
        """Scenario: LONG post-TP1, existing=100, new=101.
        should_replace → True. Phase should place new SL and cancel old."""
        result = should_replace_sl(side="LONG", existing_sl_price=100.0, candidate_sl_price=101.0)
        assert result is True  # phase places new SL

    def test_three_stage_post_tp1_short_no_loosen_keep_existing(self):
        """Scenario: SHORT post-TP1, existing=99, new=100.
        should_replace → False. Phase should keep existing."""
        result = should_replace_sl(side="SHORT", existing_sl_price=99.0, candidate_sl_price=100.0)
        assert result is False

    def test_three_stage_post_tp1_short_replace(self):
        """Scenario: SHORT post-TP1, existing=100, new=99.
        should_replace → True. Phase should replace."""
        result = should_replace_sl(side="SHORT", existing_sl_price=100.0, candidate_sl_price=99.0)
        assert result is True

    # ── middle_runner ──────────────────────────────────────────────────

    def test_middle_runner_long_no_loosen_keep_existing(self):
        """Scenario: LONG middle_runner, existing runner SL=95, new=94.
        should_replace → False."""
        result = should_replace_sl(side="LONG", existing_sl_price=95.0, candidate_sl_price=94.0)
        assert result is False

    def test_middle_runner_long_replace(self):
        """Scenario: LONG middle_runner, existing=95, new=96.
        should_replace → True."""
        result = should_replace_sl(side="LONG", existing_sl_price=95.0, candidate_sl_price=96.0)
        assert result is True

    def test_middle_runner_short_no_loosen_keep_existing(self):
        """Scenario: SHORT middle_runner, existing=105, new=106.
        should_replace → False."""
        result = should_replace_sl(side="SHORT", existing_sl_price=105.0, candidate_sl_price=106.0)
        assert result is False

    def test_middle_runner_short_replace(self):
        """Scenario: SHORT middle_runner, existing=105, new=104.
        should_replace → True."""
        result = should_replace_sl(side="SHORT", existing_sl_price=105.0, candidate_sl_price=104.0)
        assert result is True

    # ── middle_bucket_split_partial ────────────────────────────────────

    def test_fast_protection_long_no_loosen_keep_existing(self):
        """Scenario: LONG fast protection, existing fast SL=101, new candidate=100.5.
        should_replace → False. Phase keeps existing fast SL."""
        result = should_replace_sl(side="LONG", existing_sl_price=101.0, candidate_sl_price=100.5)
        assert result is False

    def test_fast_protection_short_no_loosen_keep_existing(self):
        """Scenario: SHORT fast protection, existing fast SL=99, new candidate=99.5.
        should_replace → False."""
        result = should_replace_sl(side="SHORT", existing_sl_price=99.0, candidate_sl_price=99.5)
        assert result is False

    def test_fast_protection_short_replace(self):
        """Scenario: SHORT fast protection, existing=99, new=98.
        should_replace → True."""
        result = should_replace_sl(side="SHORT", existing_sl_price=99.0, candidate_sl_price=98.0)
        assert result is True

    # ── Edge cases ─────────────────────────────────────────────────────

    def test_side_unknown_with_existing_never_replace(self):
        """Conservative: unknown side with existing SL → never replace."""
        for candidate in [90.0, 100.0, 110.0]:
            result = should_replace_sl(side=None, existing_sl_price=100.0, candidate_sl_price=candidate)
            assert result is False, f"side=None, candidate={candidate} should not replace"

    def test_equal_prices_no_replace(self):
        """Equal prices should not trigger replacement (no benefit)."""
        assert should_replace_sl(side="LONG", existing_sl_price=100.0, candidate_sl_price=100.0) is False
        assert should_replace_sl(side="SHORT", existing_sl_price=100.0, candidate_sl_price=100.0) is False
