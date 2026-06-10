"""Tests for middle bucket split out-of-order fill detection.

Covers: fast-first, slow-first, simultaneous fills, fast-after-slow, slow-after-fast.
"""

from unittest import mock

import pytest
from src.position_management.tp_progress import (
    mark_middle_bucket_split_progress_if_position_reduced,
)


def _make_mock_strategy(**overrides):
    """Build a mock BollCvdReclaimStrategy with state attributes."""
    strategy = mock.MagicMock()
    # Default state values for a middle bucket split active position
    defaults = {
        "middle_bucket_split_active": True,
        "side": "LONG",
        "total_entry_qty": 10.0,
        "middle_bucket_split_middle_bucket_ratio": 0.70,
        "middle_bucket_split_fast_total_ratio": 0.49,  # 0.70 * 0.70
        "middle_bucket_split_slow_total_ratio": 0.21,  # 0.70 - 0.49
        "middle_bucket_split_fast_consumed": False,
        "middle_bucket_split_slow_consumed": False,
        "middle_bucket_split_fast_price": 3000.0,
        "middle_bucket_split_slow_price": 2950.0,
        "middle_bucket_split_effective_price": 2985.0,
        "middle_bucket_split_fast_sl_price": None,
        "middle_bucket_split_fast_sl_order_id": None,
        "middle_bucket_split_fast_sl_protected": False,
        "middle_bucket_split_add_disabled": False,
        "avg_entry_price": 2800.0,
        "tp_plan": "THREE_STAGE_RUNNER",
        "three_stage_tp1_consumed": False,
        "partial_tp_consumed": False,
        "middle_runner_pending": False,
        "middle_runner_active": False,
        "middle_runner_add_disabled": False,
        "partial_tp_price": None,
        "partial_tp_ratio": 0.0,
        "last_tp_update_candle_ts_ms": 0,
    }
    defaults.update(overrides)

    for key, value in defaults.items():
        setattr(strategy.state, key, value)

    strategy.config.middle_bucket_split_fast_sl_fee_buffer_pct = 0.001
    strategy.config.breakeven_fee_buffer_pct = 0.001
    return strategy


def _make_mock_position(side="LONG", eth_qty=10.0):
    """Build a mock PositionSnapshot."""
    pos = mock.MagicMock()
    pos.has_position = True
    pos.side = side
    pos.eth_qty = eth_qty
    return pos


class TestFastFirstFill:
    """Test 1: fast first fill is still detected correctly."""

    def test_fast_first_fill_detected(self):
        """Fast fills first, slow not yet filled."""
        # after_fast = 1 - 0.49 = 0.51
        # remaining_ratio should be <= 0.51 + tolerance
        # remaining qty = 5.1 → ratio = 0.51
        strategy = _make_mock_strategy()
        position = _make_mock_position(eth_qty=5.1)

        with mock.patch(
            "src.position_management.tp_progress.position_cost_runtime.record_core_position_reduction_exit"
        ):
            event = mark_middle_bucket_split_progress_if_position_reduced(strategy, position)

        assert event == "MIDDLE_BUCKET_FAST"
        assert strategy.state.middle_bucket_split_fast_consumed is True
        assert strategy.state.middle_bucket_split_slow_consumed is False
        assert strategy.state.three_stage_tp1_consumed is False
        assert strategy.state.middle_bucket_split_fast_sl_price is not None
        assert strategy.state.middle_bucket_split_add_disabled is True

    def test_fast_first_does_not_trigger_full_tp1(self):
        """Fast fill alone does NOT mark full TP1 consumed."""
        strategy = _make_mock_strategy()
        position = _make_mock_position(eth_qty=5.1)

        with mock.patch(
            "src.position_management.tp_progress.position_cost_runtime.record_core_position_reduction_exit"
        ):
            event = mark_middle_bucket_split_progress_if_position_reduced(strategy, position)

        assert event == "MIDDLE_BUCKET_FAST"
        assert strategy.state.three_stage_tp1_consumed is False
        assert strategy.state.partial_tp_consumed is False


class TestSlowFirstFill:
    """Test 2: slow fills first (out-of-order) is correctly detected."""

    def test_slow_first_fill_detected(self):
        """Slow fills first (SHORT scenario where BOLL20 fills before BOLL15)."""
        # slow_total_ratio = 0.21
        # after_slow = 1 - 0.21 = 0.79
        # remaining ratio <= 0.79 + tolerance
        # remaining qty = 7.9 → ratio = 0.79
        strategy = _make_mock_strategy(side="SHORT")
        position = _make_mock_position(side="SHORT", eth_qty=7.9)

        with mock.patch(
            "src.position_management.tp_progress.position_cost_runtime.record_core_position_reduction_exit"
        ):
            event = mark_middle_bucket_split_progress_if_position_reduced(strategy, position)

        assert event == "MIDDLE_BUCKET_SLOW_ONLY"
        assert strategy.state.middle_bucket_split_slow_consumed is True
        assert strategy.state.middle_bucket_split_fast_consumed is False
        assert strategy.state.three_stage_tp1_consumed is False
        # Partial protective SL should be set (stored in fast_sl_price field)
        assert strategy.state.middle_bucket_split_fast_sl_price is not None
        assert strategy.state.middle_bucket_split_add_disabled is True


class TestFastAfterSlowFullTP1:
    """Test 3: fast fills after slow, completing full TP1 bucket."""

    def test_fast_after_slow_completes_full_tp1(self):
        """Slow consumed first, then fast fills to reach full bucket."""
        # full_bucket after_ratio = 1 - 0.70 = 0.30
        # remaining qty = 3.0 → ratio = 0.30
        strategy = _make_mock_strategy(
            middle_bucket_split_slow_consumed=True,
            middle_bucket_split_fast_consumed=False,
            tp_plan="THREE_STAGE_RUNNER",
        )
        position = _make_mock_position(eth_qty=3.0)

        with mock.patch(
            "src.position_management.tp_progress.position_cost_runtime.record_core_position_reduction_exit"
        ):
            event = mark_middle_bucket_split_progress_if_position_reduced(strategy, position)

        assert event == "MIDDLE_BUCKET_FULL"
        assert strategy.state.middle_bucket_split_fast_consumed is True
        assert strategy.state.middle_bucket_split_slow_consumed is True
        assert strategy.state.three_stage_tp1_consumed is True
        assert strategy.state.partial_tp_consumed is True


class TestSlowAfterFastFullTP1:
    """Test 4: slow fills after fast, completing full TP1 bucket."""

    def test_slow_after_fast_completes_full_tp1(self):
        """Fast consumed first, then slow fills to reach full bucket."""
        strategy = _make_mock_strategy(
            middle_bucket_split_fast_consumed=True,
            middle_bucket_split_slow_consumed=False,
            tp_plan="THREE_STAGE_RUNNER",
        )
        position = _make_mock_position(eth_qty=3.0)

        with mock.patch(
            "src.position_management.tp_progress.position_cost_runtime.record_core_position_reduction_exit"
        ):
            event = mark_middle_bucket_split_progress_if_position_reduced(strategy, position)

        assert event == "MIDDLE_BUCKET_FULL"
        assert strategy.state.middle_bucket_split_slow_consumed is True
        assert strategy.state.three_stage_tp1_consumed is True


class TestSimultaneousFill:
    """Test 5: both legs fill in the same account sync round."""

    def test_simultaneous_fill_detected(self):
        """Both legs fill at once — position reduced directly to after_full."""
        strategy = _make_mock_strategy(
            middle_bucket_split_fast_consumed=False,
            middle_bucket_split_slow_consumed=False,
            tp_plan="THREE_STAGE_RUNNER",
        )
        position = _make_mock_position(eth_qty=3.0)  # after_full = 0.30 * 10 = 3.0

        with mock.patch(
            "src.position_management.tp_progress.position_cost_runtime.record_core_position_reduction_exit"
        ):
            event = mark_middle_bucket_split_progress_if_position_reduced(strategy, position)

        assert event == "MIDDLE_BUCKET_FULL"
        assert strategy.state.middle_bucket_split_fast_consumed is True
        assert strategy.state.middle_bucket_split_slow_consumed is True
        assert strategy.state.three_stage_tp1_consumed is True

    def test_simultaneous_fill_uses_effective_price(self):
        """When both fill same round, effective_price is used for cost recording."""
        strategy = _make_mock_strategy(
            middle_bucket_split_effective_price=2985.0,
        )
        position = _make_mock_position(eth_qty=3.0)

        with mock.patch(
            "src.position_management.tp_progress.position_cost_runtime.record_core_position_reduction_exit"
        ) as mock_record:
            mark_middle_bucket_split_progress_if_position_reduced(strategy, position)

        # Check that exit_price was set to effective_price
        call_kwargs = mock_record.call_args.kwargs
        assert call_kwargs["exit_price"] == 2985.0


class TestMiddleRunnerFullTP1:
    """Test 6: Middle Runner mode full TP1 completion."""

    def test_middle_runner_full_tp1(self):
        """Full bucket completed in MIDDLE_RUNNER plan."""
        strategy = _make_mock_strategy(
            tp_plan="MIDDLE_RUNNER",
            middle_runner_pending=True,
            middle_runner_active=False,
            middle_runner_add_disabled=False,
        )
        position = _make_mock_position(eth_qty=3.0)

        with mock.patch(
            "src.position_management.tp_progress.position_cost_runtime.record_core_position_reduction_exit"
        ):
            event = mark_middle_bucket_split_progress_if_position_reduced(strategy, position)

        assert event == "MIDDLE_BUCKET_FULL"
        assert strategy.state.middle_runner_active is True
        assert strategy.state.middle_runner_pending is False
        assert strategy.state.tp_plan == "SINGLE"
        assert strategy.state.middle_runner_add_disabled is True


class TestNoProgress:
    """Tests for no progress cases."""

    def test_not_active_returns_none(self):
        """Split not active => None."""
        strategy = _make_mock_strategy(middle_bucket_split_active=False)
        position = _make_mock_position(eth_qty=5.0)
        event = mark_middle_bucket_split_progress_if_position_reduced(strategy, position)
        assert event is None

    def test_both_already_consumed_returns_none(self):
        """Both legs already consumed => None."""
        strategy = _make_mock_strategy(
            middle_bucket_split_fast_consumed=True,
            middle_bucket_split_slow_consumed=True,
        )
        position = _make_mock_position(eth_qty=3.0)
        event = mark_middle_bucket_split_progress_if_position_reduced(strategy, position)
        assert event is None

    def test_no_significant_reduction_returns_none(self):
        """Position not reduced enough => None."""
        strategy = _make_mock_strategy()
        position = _make_mock_position(eth_qty=9.5)  # barely reduced
        event = mark_middle_bucket_split_progress_if_position_reduced(strategy, position)
        assert event is None

    def test_no_position_returns_none(self):
        """No open position => None."""
        strategy = _make_mock_strategy()
        position = _make_mock_position(eth_qty=0.0)
        position.has_position = False
        event = mark_middle_bucket_split_progress_if_position_reduced(strategy, position)
        assert event is None

    def test_side_mismatch_returns_none(self):
        """Position side differs from state side => None."""
        strategy = _make_mock_strategy(side="LONG")
        position = _make_mock_position(side="SHORT", eth_qty=5.0)
        event = mark_middle_bucket_split_progress_if_position_reduced(strategy, position)
        assert event is None


class TestFastBeforeSlowOrdering:
    """Test that check ordering prevents fast-only from being misidentified as slow-only.

    When fast_total_ratio > slow_total_ratio, the fast-only remaining_ratio could
    also satisfy the slow threshold. The check order (full → fast → slow) ensures
    fast-only is correctly identified as FAST, not SLOW_ONLY.
    """

    def test_fast_only_not_misidentified_as_slow(self):
        """When fast_total > slow_total, fast-only is still MIDDLE_BUCKET_FAST."""
        # fast_total = 0.49, slow_total = 0.21
        # after_fast = 0.51, after_slow = 0.79
        # remaining = 0.51 should trigger FAST, not SLOW_ONLY
        strategy = _make_mock_strategy(
            middle_bucket_split_fast_total_ratio=0.49,
            middle_bucket_split_slow_total_ratio=0.21,
        )
        position = _make_mock_position(eth_qty=5.1)

        with mock.patch(
            "src.position_management.tp_progress.position_cost_runtime.record_core_position_reduction_exit"
        ):
            event = mark_middle_bucket_split_progress_if_position_reduced(strategy, position)

        assert event == "MIDDLE_BUCKET_FAST"
        assert strategy.state.middle_bucket_split_fast_consumed is True
        assert strategy.state.middle_bucket_split_slow_consumed is False


class TestRepeatSyncNoMisidentification:
    """Regression tests: mutual exclusion prevents repeat-sync misidentification.

    Before the fix, after fast fills in sync #1, sync #2 with the same
    remaining_ratio would satisfy the slow-only condition (because
    after_fast ≤ after_slow) and incorrectly mark slow_consumed=True.
    With mutual exclusion, both single-leg branches require BOTH legs
    unconsumed, so a consumed leg blocks the other single-leg branch.
    """

    def test_fast_first_repeat_sync_no_slow_misidentify(self):
        """Fast fills in sync 1; sync 2 with same qty must NOT trigger slow-only.

        This is the critical regression test for Problem 1.
        """
        strategy = _make_mock_strategy(
            middle_bucket_split_fast_total_ratio=0.49,
            middle_bucket_split_slow_total_ratio=0.21,
        )
        position = _make_mock_position(eth_qty=5.1)  # remaining_ratio = 0.51 (after_fast)

        # Sync 1: fast fills
        with mock.patch(
            "src.position_management.tp_progress.position_cost_runtime.record_core_position_reduction_exit"
        ):
            event1 = mark_middle_bucket_split_progress_if_position_reduced(strategy, position)

        assert event1 == "MIDDLE_BUCKET_FAST"
        assert strategy.state.middle_bucket_split_fast_consumed is True
        assert strategy.state.middle_bucket_split_slow_consumed is False
        assert strategy.state.three_stage_tp1_consumed is False

        # Sync 2: same remaining_ratio, slow did NOT actually fill
        event2 = mark_middle_bucket_split_progress_if_position_reduced(strategy, position)
        assert event2 is None, (
            "Repeat sync must NOT trigger SLOW_ONLY — fast is already consumed, "
            "so slow-only branch (which requires BOTH unconsumed) should be blocked"
        )
        assert strategy.state.middle_bucket_split_slow_consumed is False, (
            "slow_consumed must remain False — slow did not actually fill"
        )
        assert strategy.state.three_stage_tp1_consumed is False

    def test_slow_first_repeat_sync_no_fast_misidentify(self):
        """Slow fills first; sync 2 with same qty must NOT trigger fast-only."""
        strategy = _make_mock_strategy(
            side="SHORT",
            middle_bucket_split_fast_total_ratio=0.49,
            middle_bucket_split_slow_total_ratio=0.21,
        )
        # remaining_ratio = 0.79 (after_slow for slow_total=0.21)
        position = _make_mock_position(side="SHORT", eth_qty=7.9)

        # Sync 1: slow fills (out-of-order)
        with mock.patch(
            "src.position_management.tp_progress.position_cost_runtime.record_core_position_reduction_exit"
        ):
            event1 = mark_middle_bucket_split_progress_if_position_reduced(strategy, position)

        assert event1 == "MIDDLE_BUCKET_SLOW_ONLY"
        assert strategy.state.middle_bucket_split_slow_consumed is True
        assert strategy.state.middle_bucket_split_fast_consumed is False

        # Sync 2: same remaining_ratio, fast did NOT actually fill
        event2 = mark_middle_bucket_split_progress_if_position_reduced(strategy, position)
        assert event2 is None, "Repeat sync must NOT trigger FAST when slow already consumed"
        assert strategy.state.middle_bucket_split_fast_consumed is False


class TestWaitingOtherLeg:
    """After one leg consumed but full threshold not reached, function returns None
    and logs MIDDLE_BUCKET_SPLIT_WAITING_OTHER_LEG."""

    def test_waiting_other_leg_after_fast(self):
        """Fast consumed, remaining not yet at after_full → return None (waiting)."""
        strategy = _make_mock_strategy(
            middle_bucket_split_fast_consumed=True,
            middle_bucket_split_slow_consumed=False,
        )
        # remaining = 5.1 → ratio 0.51, not yet at after_full (0.30)
        position = _make_mock_position(eth_qty=5.1)

        event = mark_middle_bucket_split_progress_if_position_reduced(strategy, position)
        assert event is None  # waiting for slow to reach full threshold

    def test_waiting_other_leg_after_slow(self):
        """Slow consumed, remaining not yet at after_full → return None (waiting)."""
        strategy = _make_mock_strategy(
            middle_bucket_split_fast_consumed=False,
            middle_bucket_split_slow_consumed=True,
        )
        position = _make_mock_position(eth_qty=7.9)

        event = mark_middle_bucket_split_progress_if_position_reduced(strategy, position)
        assert event is None  # waiting for fast to reach full threshold
