"""Tests for the pure three_stage_runner module."""

from __future__ import annotations

import unittest

from src.strategies import three_stage_runner as tsr


class TestNormalizeThreeStageRatios(unittest.TestCase):
    """Tests for normalize_three_stage_ratios."""

    def test_normal_ratios(self) -> None:
        r = tsr.normalize_three_stage_ratios(tp1_ratio=0.6, tp2_ratio=0.2, runner_ratio=0.2)
        self.assertAlmostEqual(r.tp1_ratio, 0.6)
        self.assertAlmostEqual(r.tp2_ratio, 0.2)
        self.assertAlmostEqual(r.runner_ratio, 0.2)

    def test_ratios_sum_lt_1_are_normalized(self) -> None:
        r = tsr.normalize_three_stage_ratios(tp1_ratio=0.3, tp2_ratio=0.1, runner_ratio=0.1)
        self.assertAlmostEqual(r.tp1_ratio, 0.6)
        self.assertAlmostEqual(r.tp2_ratio, 0.2)
        self.assertAlmostEqual(r.runner_ratio, 0.2)
        self.assertAlmostEqual(r.tp1_ratio + r.tp2_ratio + r.runner_ratio, 1.0)

    def test_negative_ratios_clamped_to_zero(self) -> None:
        r = tsr.normalize_three_stage_ratios(tp1_ratio=-0.1, tp2_ratio=0.3, runner_ratio=0.3)
        self.assertAlmostEqual(r.tp1_ratio, 0.0)
        self.assertAlmostEqual(r.tp2_ratio, 0.5)
        self.assertAlmostEqual(r.runner_ratio, 0.5)

    def test_all_zero_falls_back_to_default(self) -> None:
        r = tsr.normalize_three_stage_ratios(tp1_ratio=0.0, tp2_ratio=0.0, runner_ratio=0.0)
        self.assertAlmostEqual(r.tp1_ratio, 0.60)
        self.assertAlmostEqual(r.tp2_ratio, 0.20)
        self.assertAlmostEqual(r.runner_ratio, 0.20)

    def test_all_negative_falls_back_to_default(self) -> None:
        r = tsr.normalize_three_stage_ratios(tp1_ratio=-0.5, tp2_ratio=-0.3, runner_ratio=-0.2)
        self.assertAlmostEqual(r.tp1_ratio, 0.60)
        self.assertAlmostEqual(r.tp2_ratio, 0.20)
        self.assertAlmostEqual(r.runner_ratio, 0.20)

    def test_extreme_ratios_sum_to_one(self) -> None:
        r = tsr.normalize_three_stage_ratios(tp1_ratio=1.0, tp2_ratio=1.0, runner_ratio=1.0)
        self.assertAlmostEqual(r.tp1_ratio, 1 / 3)
        self.assertAlmostEqual(r.tp2_ratio, 1 / 3)
        self.assertAlmostEqual(r.runner_ratio, 1 / 3)


class TestResetThreeStageStateValues(unittest.TestCase):
    """Tests for reset_three_stage_state_values."""

    def test_all_three_stage_fields_reset(self) -> None:
        values = tsr.reset_three_stage_state_values()
        self.assertFalse(values.three_stage_runner_enabled_for_position)
        self.assertIsNone(values.three_stage_tp1_price)
        self.assertIsNone(values.three_stage_tp2_price)
        self.assertIsNone(values.three_stage_runner_initial_tp_price)
        self.assertEqual(values.three_stage_tp1_ratio, 0.0)
        self.assertEqual(values.three_stage_tp2_ratio, 0.0)
        self.assertEqual(values.three_stage_runner_ratio, 0.0)
        self.assertFalse(values.three_stage_tp1_consumed)
        self.assertFalse(values.three_stage_tp2_consumed)
        self.assertIsNone(values.three_stage_post_tp1_protective_sl_price)
        self.assertIsNone(values.three_stage_post_tp1_protective_sl_order_id)
        self.assertFalse(values.three_stage_post_tp1_sl_extension_triggered)
        self.assertFalse(values.three_stage_post_tp1_protected)

    def test_post_tp1_sl_time_tighten_fields_are_zero(self) -> None:
        values = tsr.reset_three_stage_state_values()
        self.assertEqual(values.three_stage_post_tp1_sl_time_tighten_candle_count, 0)
        self.assertEqual(values.three_stage_post_tp1_sl_time_tighten_last_candle_ts_ms, 0)
        self.assertEqual(values.three_stage_post_tp1_sl_time_tighten_log_candle_ts_ms, 0)

    def test_dataclass_has_no_trend_runner_fields(self) -> None:
        values = tsr.reset_three_stage_state_values()
        self.assertFalse(hasattr(values, "trend_runner_active"))


class TestPlannedThreeStageStateValues(unittest.TestCase):
    """Tests for planned_three_stage_state_values."""

    def setUp(self) -> None:
        self.ratios = tsr.ThreeStageRatios(tp1_ratio=0.6, tp2_ratio=0.2, runner_ratio=0.2)

    def test_enabled_flag_set(self) -> None:
        values = tsr.planned_three_stage_state_values(
            tp1_price=100.0, tp2_price=110.0, ratios=self.ratios,
        )
        self.assertTrue(values.three_stage_runner_enabled_for_position)

    def test_tp_prices_written(self) -> None:
        values = tsr.planned_three_stage_state_values(
            tp1_price=100.0, tp2_price=110.0, ratios=self.ratios,
        )
        self.assertEqual(values.three_stage_tp1_price, 100.0)
        self.assertEqual(values.three_stage_tp2_price, 110.0)

    def test_ratios_written(self) -> None:
        values = tsr.planned_three_stage_state_values(
            tp1_price=100.0, tp2_price=110.0, ratios=self.ratios,
        )
        self.assertAlmostEqual(values.three_stage_tp1_ratio, 0.6)
        self.assertAlmostEqual(values.three_stage_tp2_ratio, 0.2)
        self.assertAlmostEqual(values.three_stage_runner_ratio, 0.2)

    def test_consumed_flags_false(self) -> None:
        values = tsr.planned_three_stage_state_values(
            tp1_price=100.0, tp2_price=110.0, ratios=self.ratios,
        )
        self.assertFalse(values.three_stage_tp1_consumed)
        self.assertFalse(values.three_stage_tp2_consumed)

    def test_protective_and_extension_cleared(self) -> None:
        values = tsr.planned_three_stage_state_values(
            tp1_price=100.0, tp2_price=110.0, ratios=self.ratios,
        )
        self.assertIsNone(values.three_stage_post_tp1_protective_sl_price)
        self.assertIsNone(values.three_stage_post_tp1_protective_sl_order_id)
        self.assertFalse(values.three_stage_post_tp1_sl_extension_triggered)
        self.assertFalse(values.three_stage_post_tp1_protected)

    def test_initial_tp_price_is_none(self) -> None:
        values = tsr.planned_three_stage_state_values(
            tp1_price=100.0, tp2_price=110.0, ratios=self.ratios,
        )
        self.assertIsNone(values.three_stage_runner_initial_tp_price)

    def test_sl_time_tighten_fields_are_zero(self) -> None:
        values = tsr.planned_three_stage_state_values(
            tp1_price=100.0, tp2_price=110.0, ratios=self.ratios,
        )
        self.assertEqual(values.three_stage_post_tp1_sl_time_tighten_candle_count, 0)
        self.assertEqual(values.three_stage_post_tp1_sl_time_tighten_last_candle_ts_ms, 0)
        self.assertEqual(values.three_stage_post_tp1_sl_time_tighten_log_candle_ts_ms, 0)


class TestUpdateThreeStageDynamicTargetValues(unittest.TestCase):
    """Tests for update_three_stage_dynamic_target_values."""

    def setUp(self) -> None:
        self.ratios = tsr.ThreeStageRatios(tp1_ratio=0.6, tp2_ratio=0.2, runner_ratio=0.2)

    def test_enabled_flag_set(self) -> None:
        values = tsr.update_three_stage_dynamic_target_values(
            tp1_price=100.0, tp2_price=110.0, ratios=self.ratios,
        )
        self.assertTrue(values.three_stage_runner_enabled_for_position)

    def test_tp_prices_and_ratios_written(self) -> None:
        values = tsr.update_three_stage_dynamic_target_values(
            tp1_price=100.0, tp2_price=110.0, ratios=self.ratios,
        )
        self.assertEqual(values.three_stage_tp1_price, 100.0)
        self.assertEqual(values.three_stage_tp2_price, 110.0)
        self.assertAlmostEqual(values.three_stage_tp1_ratio, 0.6)
        self.assertAlmostEqual(values.three_stage_tp2_ratio, 0.2)
        self.assertAlmostEqual(values.three_stage_runner_ratio, 0.2)

    def test_dataclass_has_no_consumed_or_protective_fields(self) -> None:
        values = tsr.update_three_stage_dynamic_target_values(
            tp1_price=100.0, tp2_price=110.0, ratios=self.ratios,
        )
        self.assertFalse(hasattr(values, "three_stage_tp1_consumed"))
        self.assertFalse(hasattr(values, "three_stage_tp2_consumed"))
        self.assertFalse(hasattr(values, "three_stage_post_tp1_protective_sl_price"))


class TestResetThreeStagePostTp1SlTimeTightenValues(unittest.TestCase):
    """Tests for reset_three_stage_post_tp1_sl_time_tighten_values."""

    def test_returns_zeros(self) -> None:
        count, last_ts, log_ts = tsr.reset_three_stage_post_tp1_sl_time_tighten_values()
        self.assertEqual(count, 0)
        self.assertEqual(last_ts, 0)
        self.assertEqual(log_ts, 0)


class TestTightenThreeStagePostTp1Sl(unittest.TestCase):
    """Tests for tighten_three_stage_post_tp1_sl."""

    def test_long_uses_max(self) -> None:
        result = tsr.tighten_three_stage_post_tp1_sl(side="LONG", old_sl=90.0, new_sl=95.0)
        self.assertAlmostEqual(result, 95.0)

    def test_long_keeps_higher_old(self) -> None:
        result = tsr.tighten_three_stage_post_tp1_sl(side="LONG", old_sl=95.0, new_sl=90.0)
        self.assertAlmostEqual(result, 95.0)

    def test_short_uses_min(self) -> None:
        result = tsr.tighten_three_stage_post_tp1_sl(side="SHORT", old_sl=110.0, new_sl=105.0)
        self.assertAlmostEqual(result, 105.0)

    def test_short_keeps_lower_old(self) -> None:
        result = tsr.tighten_three_stage_post_tp1_sl(side="SHORT", old_sl=105.0, new_sl=110.0)
        self.assertAlmostEqual(result, 105.0)


class TestTightenOptionalThreeStagePostTp1Sl(unittest.TestCase):
    """Tests for tighten_optional_three_stage_post_tp1_sl."""

    def test_new_none_returns_old(self) -> None:
        result = tsr.tighten_optional_three_stage_post_tp1_sl(
            side="LONG", old_sl=90.0, new_sl=None,
        )
        self.assertAlmostEqual(result, 90.0)

    def test_old_none_returns_new(self) -> None:
        result = tsr.tighten_optional_three_stage_post_tp1_sl(
            side="LONG", old_sl=None, new_sl=95.0,
        )
        self.assertAlmostEqual(result, 95.0)

    def test_both_present_uses_tighten(self) -> None:
        result = tsr.tighten_optional_three_stage_post_tp1_sl(
            side="LONG", old_sl=90.0, new_sl=95.0,
        )
        self.assertAlmostEqual(result, 95.0)

    def test_both_none_returns_none(self) -> None:
        result = tsr.tighten_optional_three_stage_post_tp1_sl(
            side="LONG", old_sl=None, new_sl=None,
        )
        self.assertIsNone(result)


class TestCalculateThreeStagePostTp1ProtectiveSlLong(unittest.TestCase):
    """Tests for calculate_three_stage_post_tp1_protective_sl (LONG side)."""

    def test_long_with_net_breakeven(self) -> None:
        decision = tsr.calculate_three_stage_post_tp1_protective_sl(
            side="LONG",
            current_price=100.0,
            avg_entry_price=95.0,
            net_remaining_breakeven_price=96.0,  # > 0 → use directly
            breakeven_fee_buffer_pct=0.001,
            tp1_price=101.0,
            tp1_ratio=0.6,
            boll_middle=100.0,
            boll_upper=110.0,
            boll_lower=90.0,
            sl_tighten_ratio=0.5,
        )
        self.assertEqual(decision.reason, "calculated")
        self.assertIsNotNone(decision.protective_sl)
        # Protective SL should be capped at boll_middle
        self.assertLessEqual(float(decision.protective_sl), 100.0)  # type: ignore[arg-type]

    def test_long_without_net_breakeven(self) -> None:
        decision = tsr.calculate_three_stage_post_tp1_protective_sl(
            side="LONG",
            current_price=100.0,
            avg_entry_price=95.0,
            net_remaining_breakeven_price=0.0,  # <= 0 → use formula
            breakeven_fee_buffer_pct=0.001,
            tp1_price=101.0,
            tp1_ratio=0.6,
            boll_middle=100.0,
            boll_upper=110.0,
            boll_lower=90.0,
            sl_tighten_ratio=0.5,
        )
        self.assertEqual(decision.reason, "calculated")
        self.assertIsNotNone(decision.protective_sl)

    def test_long_sl_not_below_current(self) -> None:
        decision = tsr.calculate_three_stage_post_tp1_protective_sl(
            side="LONG",
            current_price=98.0,
            avg_entry_price=95.0,
            net_remaining_breakeven_price=97.0,
            breakeven_fee_buffer_pct=0.001,
            tp1_price=101.0,
            tp1_ratio=0.6,
            boll_middle=99.5,
            boll_upper=110.0,
            boll_lower=97.0,
            sl_tighten_ratio=0.9,
        )
        self.assertEqual(decision.reason, "long_sl_not_below_current")
        self.assertIsNone(decision.protective_sl)


class TestCalculateThreeStagePostTp1ProtectiveSlShort(unittest.TestCase):
    """Tests for calculate_three_stage_post_tp1_protective_sl (SHORT side)."""

    def test_short_with_net_breakeven(self) -> None:
        decision = tsr.calculate_three_stage_post_tp1_protective_sl(
            side="SHORT",
            current_price=100.0,
            avg_entry_price=105.0,
            net_remaining_breakeven_price=104.0,  # > 0 → use directly
            breakeven_fee_buffer_pct=0.001,
            tp1_price=99.0,
            tp1_ratio=0.6,
            boll_middle=100.0,
            boll_upper=110.0,
            boll_lower=90.0,
            sl_tighten_ratio=0.5,
        )
        self.assertEqual(decision.reason, "calculated")
        self.assertIsNotNone(decision.protective_sl)
        self.assertGreaterEqual(float(decision.protective_sl), 100.0)  # type: ignore[arg-type]

    def test_short_without_net_breakeven(self) -> None:
        decision = tsr.calculate_three_stage_post_tp1_protective_sl(
            side="SHORT",
            current_price=100.0,
            avg_entry_price=105.0,
            net_remaining_breakeven_price=0.0,  # <= 0 → use formula
            breakeven_fee_buffer_pct=0.001,
            tp1_price=99.0,
            tp1_ratio=0.6,
            boll_middle=100.0,
            boll_upper=110.0,
            boll_lower=90.0,
            sl_tighten_ratio=0.5,
        )
        self.assertEqual(decision.reason, "calculated")
        self.assertIsNotNone(decision.protective_sl)

    def test_short_sl_not_above_current(self) -> None:
        decision = tsr.calculate_three_stage_post_tp1_protective_sl(
            side="SHORT",
            current_price=102.0,
            avg_entry_price=105.0,
            net_remaining_breakeven_price=103.0,
            breakeven_fee_buffer_pct=0.001,
            tp1_price=99.0,
            tp1_ratio=0.6,
            boll_middle=100.5,
            boll_upper=110.0,
            boll_lower=90.0,
            sl_tighten_ratio=0.9,
        )
        self.assertEqual(decision.reason, "short_sl_not_above_current")
        self.assertIsNone(decision.protective_sl)


class TestCalculateThreeStagePostTp1ProtectiveSlInvalid(unittest.TestCase):
    """Tests for invalid inputs to calculate_three_stage_post_tp1_protective_sl."""

    def test_missing_tp1_price(self) -> None:
        decision = tsr.calculate_three_stage_post_tp1_protective_sl(
            side="LONG",
            current_price=100.0,
            avg_entry_price=95.0,
            net_remaining_breakeven_price=0.0,
            breakeven_fee_buffer_pct=0.001,
            tp1_price=None,
            tp1_ratio=0.6,
            boll_middle=100.0,
            boll_upper=110.0,
            boll_lower=90.0,
            sl_tighten_ratio=0.5,
        )
        self.assertEqual(decision.reason, "missing_tp1_price")
        self.assertIsNone(decision.protective_sl)

    def test_current_price_zero(self) -> None:
        decision = tsr.calculate_three_stage_post_tp1_protective_sl(
            side="LONG",
            current_price=0.0,
            avg_entry_price=95.0,
            net_remaining_breakeven_price=0.0,
            breakeven_fee_buffer_pct=0.001,
            tp1_price=101.0,
            tp1_ratio=0.6,
            boll_middle=100.0,
            boll_upper=110.0,
            boll_lower=90.0,
            sl_tighten_ratio=0.5,
        )
        self.assertEqual(decision.reason, "missing_cost_basis")

    def test_avg_entry_zero(self) -> None:
        decision = tsr.calculate_three_stage_post_tp1_protective_sl(
            side="LONG",
            current_price=100.0,
            avg_entry_price=0.0,
            net_remaining_breakeven_price=0.0,
            breakeven_fee_buffer_pct=0.001,
            tp1_price=101.0,
            tp1_ratio=0.6,
            boll_middle=100.0,
            boll_upper=110.0,
            boll_lower=90.0,
            sl_tighten_ratio=0.5,
        )
        self.assertEqual(decision.reason, "missing_cost_basis")

    def test_tp1_ratio_zero(self) -> None:
        decision = tsr.calculate_three_stage_post_tp1_protective_sl(
            side="LONG",
            current_price=100.0,
            avg_entry_price=95.0,
            net_remaining_breakeven_price=0.0,
            breakeven_fee_buffer_pct=0.001,
            tp1_price=101.0,
            tp1_ratio=0.0,
            boll_middle=100.0,
            boll_upper=110.0,
            boll_lower=90.0,
            sl_tighten_ratio=0.5,
        )
        self.assertEqual(decision.reason, "invalid_tp1_ratio")

    def test_tp1_ratio_ge_one(self) -> None:
        decision = tsr.calculate_three_stage_post_tp1_protective_sl(
            side="LONG",
            current_price=100.0,
            avg_entry_price=95.0,
            net_remaining_breakeven_price=0.0,
            breakeven_fee_buffer_pct=0.001,
            tp1_price=101.0,
            tp1_ratio=1.0,
            boll_middle=100.0,
            boll_upper=110.0,
            boll_lower=90.0,
            sl_tighten_ratio=0.5,
        )
        self.assertEqual(decision.reason, "invalid_tp1_ratio")


class TestApplyThreeStagePostTp1ExtensionTriggerLong(unittest.TestCase):
    """Tests for apply_three_stage_post_tp1_extension_trigger (LONG)."""

    def test_long_below_trigger_no_extension(self) -> None:
        decision = tsr.apply_three_stage_post_tp1_extension_trigger(
            side="LONG",
            current_price=103.0,
            protective_sl=90.0,
            boll_middle=100.0,
            boll_upper=110.0,
            boll_lower=90.0,
            extension_trigger_ratio=0.6,
        )
        # trigger = 100 + (110-100)*0.6 = 106.0; 103 < 106 → no trigger
        self.assertFalse(decision.extension_triggered)
        self.assertAlmostEqual(float(decision.protective_sl), 90.0)  # type: ignore[arg-type]
        self.assertAlmostEqual(float(decision.trigger_price), 106.0)  # type: ignore[arg-type]

    def test_long_above_trigger_with_sl(self) -> None:
        decision = tsr.apply_three_stage_post_tp1_extension_trigger(
            side="LONG",
            current_price=108.0,
            protective_sl=90.0,
            boll_middle=100.0,
            boll_upper=110.0,
            boll_lower=90.0,
            extension_trigger_ratio=0.6,
        )
        # trigger = 106; 108 >= 106 → extension, new_sl = max(90, 100) = 100
        self.assertTrue(decision.extension_triggered)
        self.assertAlmostEqual(float(decision.protective_sl), 100.0)  # type: ignore[arg-type]

    def test_long_above_trigger_no_sl(self) -> None:
        decision = tsr.apply_three_stage_post_tp1_extension_trigger(
            side="LONG",
            current_price=108.0,
            protective_sl=None,
            boll_middle=100.0,
            boll_upper=110.0,
            boll_lower=90.0,
            extension_trigger_ratio=0.6,
        )
        self.assertTrue(decision.extension_triggered)
        self.assertAlmostEqual(float(decision.protective_sl), 100.0)  # type: ignore[arg-type]


class TestApplyThreeStagePostTp1ExtensionTriggerShort(unittest.TestCase):
    """Tests for apply_three_stage_post_tp1_extension_trigger (SHORT)."""

    def test_short_above_trigger_no_extension(self) -> None:
        decision = tsr.apply_three_stage_post_tp1_extension_trigger(
            side="SHORT",
            current_price=97.0,
            protective_sl=110.0,
            boll_middle=100.0,
            boll_upper=110.0,
            boll_lower=90.0,
            extension_trigger_ratio=0.6,
        )
        # trigger = 100 - (100-90)*0.6 = 94.0; 97 > 94 → no trigger
        self.assertFalse(decision.extension_triggered)
        self.assertAlmostEqual(float(decision.protective_sl), 110.0)  # type: ignore[arg-type]
        self.assertAlmostEqual(float(decision.trigger_price), 94.0)  # type: ignore[arg-type]

    def test_short_below_trigger_with_sl(self) -> None:
        decision = tsr.apply_three_stage_post_tp1_extension_trigger(
            side="SHORT",
            current_price=93.0,
            protective_sl=110.0,
            boll_middle=100.0,
            boll_upper=110.0,
            boll_lower=90.0,
            extension_trigger_ratio=0.6,
        )
        # trigger = 94; 93 <= 94 → extension, new_sl = min(110, 100) = 100
        self.assertTrue(decision.extension_triggered)
        self.assertAlmostEqual(float(decision.protective_sl), 100.0)  # type: ignore[arg-type]

    def test_short_below_trigger_no_sl(self) -> None:
        decision = tsr.apply_three_stage_post_tp1_extension_trigger(
            side="SHORT",
            current_price=93.0,
            protective_sl=None,
            boll_middle=100.0,
            boll_upper=110.0,
            boll_lower=90.0,
            extension_trigger_ratio=0.6,
        )
        self.assertTrue(decision.extension_triggered)
        self.assertAlmostEqual(float(decision.protective_sl), 100.0)  # type: ignore[arg-type]

    def test_extension_trigger_ratio_clamped(self) -> None:
        decision = tsr.apply_three_stage_post_tp1_extension_trigger(
            side="SHORT",
            current_price=93.0,
            protective_sl=110.0,
            boll_middle=100.0,
            boll_upper=110.0,
            boll_lower=90.0,
            extension_trigger_ratio=1.5,
        )
        # ratio clamped to 1.0, trigger = 100 - (100-90)*1.0 = 90.0; 93 > 90 → no trigger
        self.assertFalse(decision.extension_triggered)
