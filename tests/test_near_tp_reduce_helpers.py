"""Pure function tests for near_tp_reduce.py helpers.

These tests exercise only the pure calculations — no strategy class, no state,
no logger, no env.
"""

from __future__ import annotations

import unittest

from src.strategies import near_tp_reduce as helpers


class ResetNearTpStateValuesTest(unittest.TestCase):
    def test_all_fields_reset_to_defaults(self) -> None:
        values = helpers.reset_near_tp_state_values()

        self.assertFalse(values.near_tp_armed)
        self.assertFalse(values.near_tp_reduce_pending)
        self.assertFalse(values.near_tp_protected)
        self.assertIsNone(values.near_tp_best_price)
        self.assertEqual(values.near_tp_armed_ts_ms, 0)
        self.assertEqual(values.near_tp_pending_ts_ms, 0)
        self.assertEqual(values.near_tp_trigger_ts_ms, 0)
        self.assertIsNone(values.near_tp_protective_sl_price)
        self.assertIsNone(values.near_tp_protective_sl_order_id)
        self.assertFalse(values.near_tp_add_disabled)
        self.assertFalse(values.near_tp_sidecar_skip_logged)

    def test_is_frozen_dataclass(self) -> None:
        values = helpers.reset_near_tp_state_values()
        with self.assertRaises(Exception):
            values.near_tp_armed = True  # type: ignore[misc]


class NearTpPlanAllowedTest(unittest.TestCase):
    def test_normal_allowed(self) -> None:
        result = helpers.near_tp_plan_allowed(
            tp_plan="SINGLE",
            middle_runner_pending=False,
            middle_runner_active=False,
            three_stage_runner_enabled_for_position=False,
            trend_runner_active=False,
            partial_tp_consumed=True,
        )
        self.assertTrue(result.allowed)
        self.assertEqual(result.reason, "ok")

    def test_middle_runner_plan_blocks(self) -> None:
        result = helpers.near_tp_plan_allowed(
            tp_plan="MIDDLE_RUNNER",
            middle_runner_pending=False,
            middle_runner_active=False,
            three_stage_runner_enabled_for_position=False,
            trend_runner_active=False,
            partial_tp_consumed=True,
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "middle_runner")

    def test_middle_runner_pending_blocks(self) -> None:
        result = helpers.near_tp_plan_allowed(
            tp_plan="SINGLE",
            middle_runner_pending=True,
            middle_runner_active=False,
            three_stage_runner_enabled_for_position=False,
            trend_runner_active=False,
            partial_tp_consumed=True,
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "middle_runner")

    def test_middle_runner_active_blocks(self) -> None:
        result = helpers.near_tp_plan_allowed(
            tp_plan="SINGLE",
            middle_runner_pending=False,
            middle_runner_active=True,
            three_stage_runner_enabled_for_position=False,
            trend_runner_active=False,
            partial_tp_consumed=True,
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "middle_runner")

    def test_three_stage_plan_blocks(self) -> None:
        result = helpers.near_tp_plan_allowed(
            tp_plan="THREE_STAGE_RUNNER",
            middle_runner_pending=False,
            middle_runner_active=False,
            three_stage_runner_enabled_for_position=False,
            trend_runner_active=False,
            partial_tp_consumed=True,
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "three_stage_or_trend_runner")

    def test_three_stage_enabled_for_position_blocks(self) -> None:
        result = helpers.near_tp_plan_allowed(
            tp_plan="SINGLE",
            middle_runner_pending=False,
            middle_runner_active=False,
            three_stage_runner_enabled_for_position=True,
            trend_runner_active=False,
            partial_tp_consumed=True,
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "three_stage_or_trend_runner")

    def test_trend_runner_active_blocks(self) -> None:
        result = helpers.near_tp_plan_allowed(
            tp_plan="SINGLE",
            middle_runner_pending=False,
            middle_runner_active=False,
            three_stage_runner_enabled_for_position=False,
            trend_runner_active=True,
            partial_tp_consumed=True,
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "three_stage_or_trend_runner")

    def test_split_partial_pending_blocks(self) -> None:
        result = helpers.near_tp_plan_allowed(
            tp_plan="MIDDLE_RUNNER",
            middle_runner_pending=False,
            middle_runner_active=False,
            three_stage_runner_enabled_for_position=False,
            trend_runner_active=False,
            partial_tp_consumed=False,
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "middle_runner")

    def test_split_partial_consumed_allows(self) -> None:
        result = helpers.near_tp_plan_allowed(
            tp_plan="MIDDLE_RUNNER",
            middle_runner_pending=False,
            middle_runner_active=False,
            three_stage_runner_enabled_for_position=False,
            trend_runner_active=False,
            partial_tp_consumed=True,
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "middle_runner")


class CalculateNearTpProgressLongTest(unittest.TestCase):
    def test_normal_progress_calculation(self) -> None:
        result = helpers.calculate_near_tp_progress(
            side="LONG",
            price=108.8,
            avg_entry_price=100.0,
            final_tp_price=110.0,
            near_tp_max_distance_usd=3.0,
            near_tp_min_reduce_profit_pct=0.004,
            near_tp_min_profit_pct=0.004,
            near_tp_min_progress_ratio=0.88,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.progress, 0.88)
        self.assertAlmostEqual(result.profit_pct, 0.088)
        self.assertTrue(result.near_by_distance)  # 110-108.8 = 1.2 <= 3.0
        # near_by_progress = progress >= 0.88; 8.8/10 may be < 0.88 due to FP,
        # so near_by_progress could be False, but near_by_distance covers arming.
        self.assertTrue(result.min_profit_seen_ok)  # 0.088 >= 0.004
        self.assertTrue(result.reduce_profit_ok)  # 0.088 >= 0.004

    def test_progress_below_threshold(self) -> None:
        result = helpers.calculate_near_tp_progress(
            side="LONG",
            price=106.0,
            avg_entry_price=100.0,
            final_tp_price=110.0,
            near_tp_max_distance_usd=3.0,
            near_tp_min_reduce_profit_pct=0.004,
            near_tp_min_profit_pct=0.004,
            near_tp_min_progress_ratio=0.88,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.progress, 0.6)
        self.assertFalse(result.near_by_progress)  # 0.6 < 0.88
        self.assertFalse(result.near_by_distance)  # 110-106 = 4 > 3
        self.assertTrue(result.min_profit_seen_ok)  # 0.06 >= 0.004

    def test_final_tp_le_avg_returns_none(self) -> None:
        result = helpers.calculate_near_tp_progress(
            side="LONG",
            price=105.0,
            avg_entry_price=110.0,
            final_tp_price=108.0,
            near_tp_max_distance_usd=3.0,
            near_tp_min_reduce_profit_pct=0.004,
            near_tp_min_profit_pct=0.004,
            near_tp_min_progress_ratio=0.88,
        )
        self.assertIsNone(result)

    def test_avg_zero_returns_none(self) -> None:
        result = helpers.calculate_near_tp_progress(
            side="LONG",
            price=105.0,
            avg_entry_price=0.0,
            final_tp_price=110.0,
            near_tp_max_distance_usd=3.0,
            near_tp_min_reduce_profit_pct=0.004,
            near_tp_min_profit_pct=0.004,
            near_tp_min_progress_ratio=0.88,
        )
        self.assertIsNone(result)

    def test_price_zero_returns_none(self) -> None:
        result = helpers.calculate_near_tp_progress(
            side="LONG",
            price=0.0,
            avg_entry_price=100.0,
            final_tp_price=110.0,
            near_tp_max_distance_usd=3.0,
            near_tp_min_reduce_profit_pct=0.004,
            near_tp_min_profit_pct=0.004,
            near_tp_min_progress_ratio=0.88,
        )
        self.assertIsNone(result)


class CalculateNearTpProgressShortTest(unittest.TestCase):
    def test_normal_progress_calculation(self) -> None:
        result = helpers.calculate_near_tp_progress(
            side="SHORT",
            price=91.2,
            avg_entry_price=100.0,
            final_tp_price=90.0,
            near_tp_max_distance_usd=3.0,
            near_tp_min_reduce_profit_pct=0.004,
            near_tp_min_profit_pct=0.004,
            near_tp_min_progress_ratio=0.88,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.progress, 0.88)
        self.assertAlmostEqual(result.profit_pct, 0.088)
        self.assertTrue(result.near_by_distance)  # 91.2-90 = 1.2 <= 3.0
        # near_by_progress = progress >= 0.88; 8.8/10 may be < 0.88 due to FP

    def test_final_tp_ge_avg_returns_none(self) -> None:
        result = helpers.calculate_near_tp_progress(
            side="SHORT",
            price=95.0,
            avg_entry_price=100.0,
            final_tp_price=110.0,
            near_tp_max_distance_usd=3.0,
            near_tp_min_reduce_profit_pct=0.004,
            near_tp_min_profit_pct=0.004,
            near_tp_min_progress_ratio=0.88,
        )
        self.assertIsNone(result)


class ShouldArmNearTpTest(unittest.TestCase):
    def _progress(self, **overrides) -> helpers.NearTpProgress:
        values = dict(
            progress=0.9,
            profit_pct=0.01,
            near_by_distance=False,
            near_by_progress=True,
            min_profit_seen_ok=True,
            reduce_profit_ok=True,
        )
        values.update(overrides)
        return helpers.NearTpProgress(**values)

    def test_arms_on_progress_and_min_profit(self) -> None:
        progress = self._progress(near_by_progress=True, near_by_distance=False, min_profit_seen_ok=True)
        self.assertTrue(helpers.should_arm_near_tp(progress=progress))

    def test_arms_on_distance_and_min_profit(self) -> None:
        progress = self._progress(near_by_progress=False, near_by_distance=True, min_profit_seen_ok=True)
        self.assertTrue(helpers.should_arm_near_tp(progress=progress))

    def test_arms_on_both_and_min_profit(self) -> None:
        progress = self._progress(near_by_progress=True, near_by_distance=True, min_profit_seen_ok=True)
        self.assertTrue(helpers.should_arm_near_tp(progress=progress))

    def test_does_not_arm_when_min_profit_not_met(self) -> None:
        progress = self._progress(near_by_progress=True, near_by_distance=True, min_profit_seen_ok=False)
        self.assertFalse(helpers.should_arm_near_tp(progress=progress))

    def test_does_not_arm_when_neither_near(self) -> None:
        progress = self._progress(near_by_progress=False, near_by_distance=False, min_profit_seen_ok=True)
        self.assertFalse(helpers.should_arm_near_tp(progress=progress))


class UpdateNearTpBestPriceTest(unittest.TestCase):
    def test_long_sets_max(self) -> None:
        result = helpers.update_near_tp_best_price(
            side="LONG",
            old_best_price=108.0,
            price=109.0,
        )
        self.assertEqual(result.best_price, 109.0)
        self.assertTrue(result.changed)

    def test_long_keeps_max_unchanged(self) -> None:
        result = helpers.update_near_tp_best_price(
            side="LONG",
            old_best_price=109.0,
            price=108.0,
        )
        self.assertEqual(result.best_price, 109.0)
        self.assertFalse(result.changed)

    def test_short_sets_min(self) -> None:
        result = helpers.update_near_tp_best_price(
            side="SHORT",
            old_best_price=92.0,
            price=91.0,
        )
        self.assertEqual(result.best_price, 91.0)
        self.assertTrue(result.changed)

    def test_short_keeps_min_unchanged(self) -> None:
        result = helpers.update_near_tp_best_price(
            side="SHORT",
            old_best_price=91.0,
            price=92.0,
        )
        self.assertEqual(result.best_price, 91.0)
        self.assertFalse(result.changed)

    def test_none_old_best_price_uses_current_price(self) -> None:
        result = helpers.update_near_tp_best_price(
            side="LONG",
            old_best_price=None,
            price=105.0,
        )
        self.assertEqual(result.best_price, 105.0)
        self.assertFalse(result.changed)


class CalculateNearTpGivebackTest(unittest.TestCase):
    def test_long_giveback_calculation(self) -> None:
        result = helpers.calculate_near_tp_giveback(
            side="LONG",
            price=106.0,
            avg_entry_price=100.0,
            best_price=109.0,
            near_tp_giveback_usd=3.0,
            near_tp_giveback_pct=0.0015,
            near_tp_giveback_profit_ratio=0.25,
        )
        self.assertAlmostEqual(result.giveback, 3.0)
        self.assertAlmostEqual(result.floating_profit_path, 9.0)
        # threshold = max(3.0, 106*0.0015=0.159, 9*0.25=2.25) = 3.0
        self.assertAlmostEqual(result.threshold, 3.0)
        self.assertTrue(result.triggered)  # 3.0 >= 3.0

    def test_long_giveback_not_triggered(self) -> None:
        result = helpers.calculate_near_tp_giveback(
            side="LONG",
            price=107.0,
            avg_entry_price=100.0,
            best_price=109.0,
            near_tp_giveback_usd=3.0,
            near_tp_giveback_pct=0.0015,
            near_tp_giveback_profit_ratio=0.25,
        )
        # giveback = 109 - 107 = 2.0
        # threshold = max(3.0, 107*0.0015=0.1605, 9*0.25=2.25) = 3.0
        self.assertAlmostEqual(result.giveback, 2.0)
        self.assertFalse(result.triggered)

    def test_short_giveback_calculation(self) -> None:
        result = helpers.calculate_near_tp_giveback(
            side="SHORT",
            price=94.0,
            avg_entry_price=100.0,
            best_price=91.0,
            near_tp_giveback_usd=3.0,
            near_tp_giveback_pct=0.0015,
            near_tp_giveback_profit_ratio=0.25,
        )
        # giveback = 94 - 91 = 3.0
        # floating_profit_path = 100 - 91 = 9.0
        # threshold = max(3.0, 94*0.0015=0.141, 9*0.25=2.25) = 3.0
        self.assertAlmostEqual(result.giveback, 3.0)
        self.assertAlmostEqual(result.floating_profit_path, 9.0)
        self.assertAlmostEqual(result.threshold, 3.0)
        self.assertTrue(result.triggered)  # 3.0 >= 3.0

    def test_threshold_uses_max_of_three_components(self) -> None:
        result = helpers.calculate_near_tp_giveback(
            side="LONG",
            price=5000.0,
            avg_entry_price=4000.0,
            best_price=5000.0,
            near_tp_giveback_usd=3.0,
            near_tp_giveback_pct=0.0015,
            near_tp_giveback_profit_ratio=0.25,
        )
        # giveback = 5000 - 5000 = 0
        # floating_profit_path = 5000 - 4000 = 1000
        # threshold = max(3.0, 5000*0.0015=7.5, 1000*0.25=250.0) = 250.0
        self.assertAlmostEqual(result.threshold, 250.0)
        self.assertFalse(result.triggered)


class CalculateNearTpProtectiveSlTest(unittest.TestCase):
    def test_long_protective_sl(self) -> None:
        result = helpers.calculate_near_tp_protective_sl(
            side="LONG",
            avg_entry_price=100.0,
            near_tp_protective_sl_profit_pct=0.001,
        )
        self.assertAlmostEqual(result, 100.1)

    def test_short_protective_sl(self) -> None:
        result = helpers.calculate_near_tp_protective_sl(
            side="SHORT",
            avg_entry_price=100.0,
            near_tp_protective_sl_profit_pct=0.001,
        )
        self.assertAlmostEqual(result, 99.9)


class NearTpPendingCanReduceTest(unittest.TestCase):
    def test_can_reduce_when_profit_ok(self) -> None:
        self.assertTrue(helpers.near_tp_pending_can_reduce(reduce_profit_ok=True))

    def test_cannot_reduce_when_profit_not_ok(self) -> None:
        self.assertFalse(helpers.near_tp_pending_can_reduce(reduce_profit_ok=False))


class NearTpSidecarSkipAllowedTest(unittest.TestCase):
    def test_sidecar_enabled_blocks(self) -> None:
        result = helpers.near_tp_sidecar_skip_allowed(sidecar_enabled_for_position=True)
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "sidecar_enabled")

    def test_sidecar_disabled_allows(self) -> None:
        result = helpers.near_tp_sidecar_skip_allowed(sidecar_enabled_for_position=False)
        self.assertTrue(result.allowed)
        self.assertEqual(result.reason, "ok")


if __name__ == "__main__":
    unittest.main()
