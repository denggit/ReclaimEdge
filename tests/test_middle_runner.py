from __future__ import annotations

import unittest

from src.strategies import middle_runner as mr


class TestResetMiddleRunnerStateValues(unittest.TestCase):
    """Tests for reset_middle_runner_state_values."""

    def test_all_fields_are_reset(self) -> None:
        values = mr.reset_middle_runner_state_values()
        self.assertFalse(values.middle_runner_enabled_for_position)
        self.assertFalse(values.middle_runner_pending)
        self.assertFalse(values.middle_runner_active)
        self.assertEqual(values.middle_runner_first_close_ratio, 0.0)
        self.assertEqual(values.middle_runner_keep_ratio, 0.0)
        self.assertIsNone(values.middle_runner_first_tp_price)
        self.assertIsNone(values.middle_runner_final_tp_price)
        self.assertIsNone(values.middle_runner_protective_sl_price)
        self.assertIsNone(values.middle_runner_protective_sl_order_id)
        self.assertFalse(values.middle_runner_extension_triggered)
        self.assertFalse(values.middle_runner_add_disabled)
        self.assertFalse(values.middle_runner_size_mismatch_protected)
        self.assertEqual(values.middle_runner_size_mismatch_warning_ts_ms, 0)

    def test_sl_time_tighten_fields_are_zero(self) -> None:
        values = mr.reset_middle_runner_state_values()
        self.assertEqual(values.middle_runner_sl_time_tighten_candle_count, 0)
        self.assertEqual(values.middle_runner_sl_time_tighten_last_candle_ts_ms, 0)
        self.assertEqual(values.middle_runner_sl_time_tighten_log_candle_ts_ms, 0)


class TestPlannedMiddleRunnerStateValues(unittest.TestCase):
    """Tests for planned_middle_runner_state_values."""

    def test_first_close_ratio_clamped_lower(self) -> None:
        values = mr.planned_middle_runner_state_values(
            first_tp_price=100.0,
            final_tp_price=110.0,
            configured_first_close_ratio=0.05,
        )
        self.assertEqual(values.middle_runner_first_close_ratio, 0.1)

    def test_first_close_ratio_clamped_upper(self) -> None:
        values = mr.planned_middle_runner_state_values(
            first_tp_price=100.0,
            final_tp_price=110.0,
            configured_first_close_ratio=0.99,
        )
        self.assertEqual(values.middle_runner_first_close_ratio, 0.95)

    def test_first_close_ratio_in_range(self) -> None:
        values = mr.planned_middle_runner_state_values(
            first_tp_price=100.0,
            final_tp_price=110.0,
            configured_first_close_ratio=0.8,
        )
        self.assertEqual(values.middle_runner_first_close_ratio, 0.8)

    def test_keep_ratio_is_one_minus_first_close(self) -> None:
        values = mr.planned_middle_runner_state_values(
            first_tp_price=100.0,
            final_tp_price=110.0,
            configured_first_close_ratio=0.7,
        )
        self.assertAlmostEqual(values.middle_runner_keep_ratio, 0.3)

    def test_enabled_and_pending_flags_set(self) -> None:
        values = mr.planned_middle_runner_state_values(
            first_tp_price=100.0,
            final_tp_price=110.0,
            configured_first_close_ratio=0.8,
        )
        self.assertTrue(values.middle_runner_enabled_for_position)
        self.assertTrue(values.middle_runner_pending)
        self.assertFalse(values.middle_runner_active)

    def test_tp_prices_written(self) -> None:
        values = mr.planned_middle_runner_state_values(
            first_tp_price=101.5,
            final_tp_price=110.0,
            configured_first_close_ratio=0.8,
        )
        self.assertEqual(values.middle_runner_first_tp_price, 101.5)
        self.assertEqual(values.middle_runner_final_tp_price, 110.0)

    def test_tp_prices_none_first(self) -> None:
        values = mr.planned_middle_runner_state_values(
            first_tp_price=None,
            final_tp_price=110.0,
            configured_first_close_ratio=0.8,
        )
        self.assertIsNone(values.middle_runner_first_tp_price)
        self.assertEqual(values.middle_runner_final_tp_price, 110.0)

    def test_protective_sl_and_order_cleared(self) -> None:
        values = mr.planned_middle_runner_state_values(
            first_tp_price=100.0,
            final_tp_price=110.0,
            configured_first_close_ratio=0.8,
        )
        self.assertIsNone(values.middle_runner_protective_sl_price)
        self.assertIsNone(values.middle_runner_protective_sl_order_id)
        self.assertFalse(values.middle_runner_extension_triggered)
        self.assertFalse(values.middle_runner_add_disabled)
        self.assertFalse(values.middle_runner_size_mismatch_protected)

    def test_sl_time_tighten_reset(self) -> None:
        values = mr.planned_middle_runner_state_values(
            first_tp_price=100.0,
            final_tp_price=110.0,
            configured_first_close_ratio=0.8,
        )
        self.assertEqual(values.middle_runner_sl_time_tighten_candle_count, 0)
        self.assertEqual(values.middle_runner_sl_time_tighten_last_candle_ts_ms, 0)
        self.assertEqual(values.middle_runner_sl_time_tighten_log_candle_ts_ms, 0)


class TestTightenMiddleRunnerSl(unittest.TestCase):
    """Tests for tighten_middle_runner_sl."""

    def test_long_returns_max(self) -> None:
        result = mr.tighten_middle_runner_sl(side="LONG", old_sl=100.0, new_sl=105.0)
        self.assertEqual(result, 105.0)

    def test_long_returns_old_when_higher(self) -> None:
        result = mr.tighten_middle_runner_sl(side="LONG", old_sl=105.0, new_sl=100.0)
        self.assertEqual(result, 105.0)

    def test_long_equal(self) -> None:
        result = mr.tighten_middle_runner_sl(side="LONG", old_sl=100.0, new_sl=100.0)
        self.assertEqual(result, 100.0)

    def test_short_returns_min(self) -> None:
        result = mr.tighten_middle_runner_sl(side="SHORT", old_sl=100.0, new_sl=95.0)
        self.assertEqual(result, 95.0)

    def test_short_returns_old_when_lower(self) -> None:
        result = mr.tighten_middle_runner_sl(side="SHORT", old_sl=95.0, new_sl=100.0)
        self.assertEqual(result, 95.0)

    def test_short_equal(self) -> None:
        result = mr.tighten_middle_runner_sl(side="SHORT", old_sl=100.0, new_sl=100.0)
        self.assertEqual(result, 100.0)


class TestTightenOptionalMiddleRunnerSl(unittest.TestCase):
    """Tests for tighten_optional_middle_runner_sl."""

    def test_new_sl_none_returns_old(self) -> None:
        result = mr.tighten_optional_middle_runner_sl(side="LONG", old_sl=100.0, new_sl=None)
        self.assertEqual(result, 100.0)

    def test_old_sl_none_returns_new(self) -> None:
        result = mr.tighten_optional_middle_runner_sl(side="LONG", old_sl=None, new_sl=100.0)
        self.assertEqual(result, 100.0)

    def test_both_present_uses_tighten_long(self) -> None:
        result = mr.tighten_optional_middle_runner_sl(side="LONG", old_sl=100.0, new_sl=105.0)
        self.assertEqual(result, 105.0)

    def test_both_present_uses_tighten_short(self) -> None:
        result = mr.tighten_optional_middle_runner_sl(side="SHORT", old_sl=100.0, new_sl=95.0)
        self.assertEqual(result, 95.0)

    def test_both_none_returns_none(self) -> None:
        result = mr.tighten_optional_middle_runner_sl(side="LONG", old_sl=None, new_sl=None)
        self.assertIsNone(result)


class TestCalculateMiddleRunnerProtectiveSlLong(unittest.TestCase):
    """Tests for calculate_middle_runner_protective_sl - LONG side."""

    def _call(self, **overrides):
        defaults = dict(
            side="LONG",
            current_price=105.0,
            avg_entry_price=100.0,
            net_remaining_breakeven_price=101.0,
            breakeven_fee_buffer_pct=0.001,
            boll_middle=106.0,
            boll_upper=110.0,
            boll_lower=98.0,
            sl_tighten_ratio=0.50,
        )
        defaults.update(overrides)
        return mr.calculate_middle_runner_protective_sl(**defaults)

    def test_uses_net_breakeven_when_positive(self) -> None:
        decision = self._call(net_remaining_breakeven_price=100.5)
        self.assertEqual(decision.reason, "calculated")
        self.assertIsNotNone(decision.protective_sl)

    def test_uses_avg_entry_fallback(self) -> None:
        decision = self._call(
            net_remaining_breakeven_price=0.0,
            avg_entry_price=100.0,
        )
        self.assertEqual(decision.reason, "calculated")
        self.assertIsNotNone(decision.protective_sl)

    def test_protective_sl_below_current_price(self) -> None:
        decision = self._call(current_price=107.0)
        self.assertEqual(decision.reason, "calculated")
        self.assertIsNotNone(decision.protective_sl)
        self.assertLess(decision.protective_sl, 107.0)

    def test_sl_not_below_current_returns_none(self) -> None:
        # Set values so that protective_sl >= current_price
        decision = self._call(
            current_price=101.0,
            net_remaining_breakeven_price=101.8,
            avg_entry_price=101.5,
            boll_middle=102.0,
            boll_lower=95.0,
        )
        self.assertEqual(decision.reason, "long_sl_not_below_current")
        self.assertIsNone(decision.protective_sl)

    def test_missing_cost_basis_zero_price(self) -> None:
        decision = self._call(current_price=0)
        self.assertEqual(decision.reason, "missing_cost_basis")
        self.assertIsNone(decision.protective_sl)
        self.assertEqual(decision.candidate_cost, 0.0)
        self.assertEqual(decision.candidate_structure, 0.0)

    def test_missing_cost_basis_zero_avg_and_be(self) -> None:
        decision = self._call(
            net_remaining_breakeven_price=0.0,
            avg_entry_price=0.0,
        )
        self.assertEqual(decision.reason, "missing_cost_basis")
        self.assertIsNone(decision.protective_sl)

    def test_negative_current_price(self) -> None:
        decision = self._call(current_price=-1.0)
        self.assertEqual(decision.reason, "missing_cost_basis")
        self.assertIsNone(decision.protective_sl)

    def test_protective_sl_capped_by_boll_middle(self) -> None:
        decision = self._call(
            current_price=101.0,
            net_remaining_breakeven_price=103.0,
            avg_entry_price=102.0,
            boll_middle=102.0,
            boll_lower=95.0,
        )
        if decision.protective_sl is not None:
            self.assertLessEqual(decision.protective_sl, 102.0)

    def test_higher_sl_tighten_ratio_moves_sl_closer_to_middle(self) -> None:
        d1 = self._call(sl_tighten_ratio=0.30)
        d2 = self._call(sl_tighten_ratio=0.80)
        if d1.protective_sl is not None and d2.protective_sl is not None:
            # Higher ratio → candidate values closer to middle (higher for LONG)
            self.assertGreaterEqual(d2.candidate_cost, d1.candidate_cost)
            self.assertGreaterEqual(d2.candidate_structure, d1.candidate_structure)


class TestCalculateMiddleRunnerProtectiveSlShort(unittest.TestCase):
    """Tests for calculate_middle_runner_protective_sl - SHORT side."""

    def _call(self, **overrides):
        defaults = dict(
            side="SHORT",
            current_price=95.0,
            avg_entry_price=100.0,
            net_remaining_breakeven_price=99.0,
            breakeven_fee_buffer_pct=0.001,
            boll_middle=94.0,
            boll_upper=106.0,
            boll_lower=90.0,
            sl_tighten_ratio=0.50,
        )
        defaults.update(overrides)
        return mr.calculate_middle_runner_protective_sl(**defaults)

    def test_uses_net_breakeven_when_positive(self) -> None:
        decision = self._call(net_remaining_breakeven_price=99.5)
        self.assertEqual(decision.reason, "calculated")
        self.assertIsNotNone(decision.protective_sl)

    def test_uses_avg_entry_fallback_short(self) -> None:
        decision = self._call(
            net_remaining_breakeven_price=0.0,
            avg_entry_price=100.0,
        )
        self.assertEqual(decision.reason, "calculated")
        self.assertIsNotNone(decision.protective_sl)

    def test_protective_sl_above_current_price(self) -> None:
        decision = self._call(current_price=93.0)
        self.assertEqual(decision.reason, "calculated")
        self.assertIsNotNone(decision.protective_sl)
        self.assertGreater(decision.protective_sl, 93.0)

    def test_sl_not_above_current_returns_none(self) -> None:
        decision = self._call(
            current_price=99.0,
            net_remaining_breakeven_price=98.2,
            avg_entry_price=98.5,
            boll_middle=98.0,
            boll_upper=105.0,
        )
        self.assertEqual(decision.reason, "short_sl_not_above_current")
        self.assertIsNone(decision.protective_sl)

    def test_missing_cost_basis_zero_price(self) -> None:
        decision = self._call(current_price=0)
        self.assertEqual(decision.reason, "missing_cost_basis")
        self.assertIsNone(decision.protective_sl)

    def test_missing_cost_basis_zero_avg_and_be(self) -> None:
        decision = self._call(
            net_remaining_breakeven_price=0.0,
            avg_entry_price=0.0,
        )
        self.assertEqual(decision.reason, "missing_cost_basis")
        self.assertIsNone(decision.protective_sl)

    def test_protective_sl_floored_by_boll_middle(self) -> None:
        decision = self._call(
            current_price=97.5,
            net_remaining_breakeven_price=96.0,
            avg_entry_price=100.0,
            boll_middle=98.0,
            boll_upper=106.0,
        )
        if decision.protective_sl is not None:
            self.assertGreaterEqual(decision.protective_sl, 98.0)


class TestApplyMiddleRunnerExtensionTriggerLong(unittest.TestCase):
    """Tests for apply_middle_runner_extension_trigger - LONG side."""

    def _call(self, **overrides):
        defaults = dict(
            side="LONG",
            current_price=106.0,
            protective_sl=None,
            boll_middle=100.0,
            boll_upper=110.0,
            boll_lower=90.0,
            extension_trigger_ratio=0.6,
            already_triggered=False,
        )
        defaults.update(overrides)
        return mr.apply_middle_runner_extension_trigger(**defaults)

    def test_below_trigger_no_extension(self) -> None:
        decision = self._call(current_price=104.0)
        # trigger = 100 + (110-100)*0.6 = 106. 104 < 106 → no trigger
        self.assertFalse(decision.extension_triggered)
        self.assertIsNone(decision.protective_sl)
        self.assertEqual(decision.trigger_price, 106.0)

    def test_at_trigger_triggers_extension(self) -> None:
        decision = self._call(current_price=106.0)
        self.assertTrue(decision.extension_triggered)
        self.assertEqual(decision.trigger_price, 106.0)

    def test_above_trigger_triggers_extension(self) -> None:
        decision = self._call(current_price=108.0)
        self.assertTrue(decision.extension_triggered)

    def test_extension_sl_is_boll_middle_when_no_previous_sl(self) -> None:
        decision = self._call(current_price=108.0, protective_sl=None)
        self.assertTrue(decision.extension_triggered)
        self.assertEqual(decision.protective_sl, 100.0)

    def test_extension_sl_is_max_of_old_and_middle(self) -> None:
        decision = self._call(current_price=108.0, protective_sl=99.0)
        self.assertTrue(decision.extension_triggered)
        self.assertEqual(decision.protective_sl, 100.0)

    def test_extension_sl_keeps_old_when_higher_than_middle(self) -> None:
        decision = self._call(current_price=108.0, protective_sl=102.0)
        self.assertTrue(decision.extension_triggered)
        self.assertEqual(decision.protective_sl, 102.0)

    def test_already_triggered_does_not_affect_decision(self) -> None:
        d1 = self._call(already_triggered=False, current_price=108.0)
        d2 = self._call(already_triggered=True, current_price=108.0)
        self.assertEqual(d1.extension_triggered, d2.extension_triggered)
        self.assertEqual(d1.protective_sl, d2.protective_sl)
        self.assertEqual(d1.trigger_price, d2.trigger_price)

    def test_ratio_zero_trigger_at_middle(self) -> None:
        decision = self._call(extension_trigger_ratio=0.0, current_price=101.0)
        # trigger = 100 + (110-100)*0 = 100
        self.assertEqual(decision.trigger_price, 100.0)
        self.assertTrue(decision.extension_triggered)

    def test_ratio_one_trigger_at_upper(self) -> None:
        decision = self._call(extension_trigger_ratio=1.0, current_price=109.0)
        # trigger = 100 + (110-100)*1 = 110
        self.assertEqual(decision.trigger_price, 110.0)
        self.assertFalse(decision.extension_triggered)


class TestApplyMiddleRunnerExtensionTriggerShort(unittest.TestCase):
    """Tests for apply_middle_runner_extension_trigger - SHORT side."""

    def _call(self, **overrides):
        defaults = dict(
            side="SHORT",
            current_price=94.0,
            protective_sl=None,
            boll_middle=100.0,
            boll_upper=110.0,
            boll_lower=90.0,
            extension_trigger_ratio=0.6,
            already_triggered=False,
        )
        defaults.update(overrides)
        return mr.apply_middle_runner_extension_trigger(**defaults)

    def test_above_trigger_no_extension(self) -> None:
        decision = self._call(current_price=96.0)
        # trigger = 100 - (100-90)*0.6 = 94. 96 > 94 → no trigger
        self.assertFalse(decision.extension_triggered)
        self.assertIsNone(decision.protective_sl)

    def test_at_trigger_triggers_extension(self) -> None:
        decision = self._call(current_price=94.0)
        self.assertTrue(decision.extension_triggered)
        self.assertEqual(decision.trigger_price, 94.0)

    def test_below_trigger_triggers_extension(self) -> None:
        decision = self._call(current_price=92.0)
        self.assertTrue(decision.extension_triggered)

    def test_extension_sl_is_boll_middle_when_no_previous_sl(self) -> None:
        decision = self._call(current_price=92.0, protective_sl=None)
        self.assertTrue(decision.extension_triggered)
        self.assertEqual(decision.protective_sl, 100.0)

    def test_extension_sl_is_min_of_old_and_middle(self) -> None:
        decision = self._call(current_price=92.0, protective_sl=101.0)
        self.assertTrue(decision.extension_triggered)
        self.assertEqual(decision.protective_sl, 100.0)

    def test_extension_sl_keeps_old_when_lower_than_middle(self) -> None:
        decision = self._call(current_price=92.0, protective_sl=98.0)
        self.assertTrue(decision.extension_triggered)
        self.assertEqual(decision.protective_sl, 98.0)

    def test_ratio_zero_trigger_at_middle(self) -> None:
        decision = self._call(extension_trigger_ratio=0.0, current_price=99.0)
        self.assertEqual(decision.trigger_price, 100.0)
        self.assertTrue(decision.extension_triggered)

    def test_ratio_one_trigger_at_lower(self) -> None:
        decision = self._call(extension_trigger_ratio=1.0, current_price=91.0)
        self.assertEqual(decision.trigger_price, 90.0)
        self.assertFalse(decision.extension_triggered)


if __name__ == "__main__":
    unittest.main()
