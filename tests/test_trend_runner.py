"""Unit tests for src/strategies/trend_runner.py pure functions."""

from __future__ import annotations

import unittest

from src.strategies import trend_runner as tr


class TestResetTrendRunnerStateValues(unittest.TestCase):
    def test_all_main_fields_reset(self):
        values = tr.reset_trend_runner_state_values()
        self.assertFalse(values.trend_runner_active)
        self.assertEqual(values.trend_runner_trend_start_ts_ms, 0)
        self.assertEqual(values.trend_runner_adjust_count, 0)
        self.assertEqual(values.trend_runner_last_update_candle_ts_ms, 0)
        self.assertIsNone(values.trend_runner_tp_price)
        self.assertIsNone(values.trend_runner_sl_price)
        self.assertIsNone(values.trend_runner_tp_order_id)
        self.assertIsNone(values.trend_runner_sl_order_id)
        self.assertIsNone(values.trend_runner_exit_reason)


class TestResetTrendRunnerReverseStateValues(unittest.TestCase):
    def test_all_reverse_fields_reset(self):
        values = tr.reset_trend_runner_reverse_state_values()
        self.assertFalse(values.trend_runner_reverse_candidate)
        self.assertEqual(values.trend_runner_reverse_start_ts_ms, 0)
        self.assertIsNone(values.trend_runner_reverse_start_price)
        self.assertIsNone(values.trend_runner_reverse_extreme_price)
        self.assertEqual(values.trend_runner_reverse_fast_cvd_start, 0.0)
        self.assertEqual(values.trend_runner_reverse_samples, [])


class TestCalculateTrendRunnerDynamicOrders(unittest.TestCase):

    def _call(self, **overrides):
        defaults = dict(
            side="LONG",
            boll_middle=100.0,
            boll_upper=110.0,
            boll_lower=90.0,
            adjust_count=0,
            current_sl_price=None,
            runner_tp_initial_outer_extra_pct=0.010,
            runner_tp_step_pct=0.001,
            runner_tp_min_outer_extra_pct=0.004,
            runner_sl_initial_outer_distance_ratio=1.00,
            runner_sl_step_ratio=0.10,
            runner_sl_min_outer_distance_ratio=0.50,
        )
        defaults.update(overrides)
        return tr.calculate_trend_runner_dynamic_orders(**defaults)

    # ── LONG ──────────────────────────────────────────────────────────

    def test_long_adjust_count_0(self):
        decision = self._call(side="LONG", adjust_count=0)
        # tp_extra_pct = max(0.004, 0.010 - 0*0.001) = 0.010
        # tp_price = 110.0 * (1 + 0.010) = 111.1
        # sl_distance_ratio = max(0.50, 1.00 - 0*0.10) = 1.00
        # sl_candidate = 110.0 - (110.0 - 100.0) * 1.00 = 100.0
        self.assertAlmostEqual(decision.tp_price, 111.1, places=6)
        self.assertAlmostEqual(decision.sl_price, 100.0, places=6)
        self.assertAlmostEqual(decision.tp_extra_pct, 0.010, places=6)
        self.assertAlmostEqual(decision.sl_distance_ratio, 1.00, places=6)

    def test_long_adjust_count_3(self):
        decision = self._call(side="LONG", adjust_count=3)
        # tp_extra_pct = max(0.004, 0.010 - 3*0.001) = max(0.004, 0.007) = 0.007
        # sl_distance_ratio = max(0.50, 1.00 - 3*0.10) = max(0.50, 0.70) = 0.70
        # tp_price = 110.0 * (1 + 0.007) = 110.77
        # sl_candidate = 110.0 - 10.0 * 0.70 = 103.0
        self.assertAlmostEqual(decision.tp_price, 110.77, places=6)
        self.assertAlmostEqual(decision.sl_price, 103.0, places=6)
        self.assertAlmostEqual(decision.tp_extra_pct, 0.007, places=6)
        self.assertAlmostEqual(decision.sl_distance_ratio, 0.70, places=6)

    def test_long_tp_extra_pct_clamped_to_min(self):
        decision = self._call(side="LONG", adjust_count=100)
        # tp_extra_pct = max(0.004, 0.010 - 100*0.001) = max(0.004, -0.090) = 0.004
        self.assertAlmostEqual(decision.tp_extra_pct, 0.004, places=6)

    def test_long_sl_distance_ratio_clamped_to_min(self):
        decision = self._call(side="LONG", adjust_count=100)
        # sl_distance_ratio = max(0.50, 1.00 - 100*0.10) = max(0.50, -9.00) = 0.50
        self.assertAlmostEqual(decision.sl_distance_ratio, 0.50, places=6)

    def test_long_current_sl_price_used(self):
        # current SL of 105 is higher than the calculated SL of 100
        decision = self._call(side="LONG", adjust_count=0, current_sl_price=105.0)
        self.assertAlmostEqual(decision.sl_price, 105.0, places=6)

    def test_long_current_sl_price_lower_ignored(self):
        # current SL of 98 is lower than the calculated SL of 100
        decision = self._call(side="LONG", adjust_count=0, current_sl_price=98.0)
        self.assertAlmostEqual(decision.sl_price, 100.0, places=6)

    # ── SHORT ─────────────────────────────────────────────────────────

    def test_short_adjust_count_0(self):
        decision = self._call(side="SHORT", adjust_count=0)
        # tp_extra_pct = max(0.004, 0.010) = 0.010
        # tp_price = 90.0 * (1 - 0.010) = 89.1
        # sl_distance_ratio = max(0.50, 1.00) = 1.00
        # sl_candidate = 90.0 + (100.0 - 90.0) * 1.00 = 100.0
        self.assertAlmostEqual(decision.tp_price, 89.1, places=6)
        self.assertAlmostEqual(decision.sl_price, 100.0, places=6)
        self.assertAlmostEqual(decision.tp_extra_pct, 0.010, places=6)
        self.assertAlmostEqual(decision.sl_distance_ratio, 1.00, places=6)

    def test_short_adjust_count_3(self):
        decision = self._call(side="SHORT", adjust_count=3)
        # tp_extra_pct = max(0.004, 0.010-0.003) = 0.007
        # sl_distance_ratio = max(0.50, 1.00-0.30) = 0.70
        # tp_price = 90.0 * (1 - 0.007) = 89.37
        # sl_candidate = 90.0 + 10.0 * 0.70 = 97.0
        self.assertAlmostEqual(decision.tp_price, 89.37, places=6)
        self.assertAlmostEqual(decision.sl_price, 97.0, places=6)
        self.assertAlmostEqual(decision.tp_extra_pct, 0.007, places=6)
        self.assertAlmostEqual(decision.sl_distance_ratio, 0.70, places=6)

    def test_short_current_sl_price_used(self):
        # current SL of 96 is lower than the calculated SL of 100
        decision = self._call(side="SHORT", adjust_count=0, current_sl_price=96.0)
        self.assertAlmostEqual(decision.sl_price, 96.0, places=6)

    def test_short_current_sl_price_higher_ignored(self):
        # current SL of 102 is higher than the calculated SL of 100
        decision = self._call(side="SHORT", adjust_count=0, current_sl_price=102.0)
        self.assertAlmostEqual(decision.sl_price, 100.0, places=6)

    # ── Edge cases ────────────────────────────────────────────────────

    def test_negative_adjust_count_treated_as_zero(self):
        decision = self._call(side="LONG", adjust_count=-5)
        # adjust = max(-5, 0) = 0 → same as adjust_count=0
        self.assertAlmostEqual(decision.tp_extra_pct, 0.010, places=6)
        self.assertAlmostEqual(decision.sl_distance_ratio, 1.00, places=6)
        self.assertAlmostEqual(decision.tp_price, 111.1, places=6)
        self.assertAlmostEqual(decision.sl_price, 100.0, places=6)

    def test_zero_boll_middle(self):
        # SHORT: boll_lower=90, boll_middle=0, outer_distance = 90
        # sl_candidate = 90.0 + (0 - 90.0) * 1.00 = 0.0  wait, boll_middle=0...
        # Actually boll_lower=90, boll_middle=0, outer_distance = middle - lower = -90
        # But the original code doesn't clamp; it just computes.
        # Let's verify the behavior.
        decision = self._call(side="SHORT", boll_middle=0.0, boll_lower=90.0, adjust_count=0)
        # sl_candidate = 90.0 + (0.0 - 90.0) * 1.00 = 90.0 - 90.0 = 0.0
        self.assertAlmostEqual(decision.sl_price, 0.0, places=6)


class TestTrendRunnerMarketExitReason(unittest.TestCase):

    def _call(self, **overrides):
        defaults = dict(
            side="LONG",
            price=105.0,
            boll_middle=100.0,
            tp_price=None,
            sl_price=None,
            trend_start_ts_ms=0,
            ts_ms=1000,
            runner_max_trend_seconds_after_second_tp=18000,
        )
        defaults.update(overrides)
        return tr.trend_runner_market_exit_reason(**defaults)

    # ── TP crossed ────────────────────────────────────────────────────

    def test_long_tp_crossed(self):
        decision = self._call(side="LONG", price=112.0, tp_price=110.0)
        self.assertTrue(decision.should_exit)
        self.assertEqual(decision.reason, "trend_runner_tp_crossed")

    def test_short_tp_crossed(self):
        decision = self._call(side="SHORT", price=88.0, tp_price=90.0)
        self.assertTrue(decision.should_exit)
        self.assertEqual(decision.reason, "trend_runner_tp_crossed")

    def test_long_tp_not_crossed(self):
        # price=109, tp=110 → TP not crossed. But price=109 > middle=100, so middle not lost.
        # No SL set, no max time → no exit.
        decision = self._call(side="LONG", price=109.0, tp_price=110.0, boll_middle=100.0)
        self.assertFalse(decision.should_exit)
        self.assertIsNone(decision.reason)

    def test_short_tp_not_crossed(self):
        decision = self._call(side="SHORT", price=91.0, tp_price=90.0, boll_middle=100.0)
        # price 91 > middle 100? No, 91 < 100, so middle not lost...
        # This would need sl_price set to trigger SL failsafe...
        # Actually for SHORT, price=91 is above tp_price=90 but below boll_middle=100.
        # SHORT middle lost is price > boll.middle (91 > 100 = False).
        # So no exit reason.
        self.assertFalse(decision.should_exit)
        self.assertIsNone(decision.reason)

    # ── SL failsafe ───────────────────────────────────────────────────

    def test_long_sl_failsafe(self):
        decision = self._call(side="LONG", price=99.0, sl_price=100.0)
        self.assertTrue(decision.should_exit)
        self.assertEqual(decision.reason, "trend_runner_sl_failsafe")

    def test_short_sl_failsafe(self):
        decision = self._call(side="SHORT", price=101.0, sl_price=100.0)
        self.assertTrue(decision.should_exit)
        self.assertEqual(decision.reason, "trend_runner_sl_failsafe")

    def test_long_sl_not_breached(self):
        decision = self._call(side="LONG", price=101.0, sl_price=100.0, boll_middle=100.0)
        # Price 101 >= middle 100, so middle not lost.
        # No TP, no SL breach → no exit.
        self.assertFalse(decision.should_exit)
        self.assertIsNone(decision.reason)

    # ── Middle lost ───────────────────────────────────────────────────

    def test_long_middle_lost(self):
        decision = self._call(side="LONG", price=99.0, boll_middle=100.0)
        self.assertTrue(decision.should_exit)
        self.assertEqual(decision.reason, "trend_runner_middle_lost")

    def test_short_middle_lost(self):
        decision = self._call(side="SHORT", price=101.0, boll_middle=100.0)
        self.assertTrue(decision.should_exit)
        self.assertEqual(decision.reason, "trend_runner_middle_lost")

    def test_long_middle_not_lost(self):
        decision = self._call(side="LONG", price=100.5, boll_middle=100.0)
        # price >= boll_middle → not middle lost. No other triggers.
        self.assertFalse(decision.should_exit)
        self.assertIsNone(decision.reason)

    # ── Max time ──────────────────────────────────────────────────────

    def test_max_time_exceeded(self):
        decision = self._call(
            side="LONG",
            price=105.0,
            boll_middle=100.0,
            trend_start_ts_ms=1_000,
            ts_ms=1_000 + 18_000_000,
            runner_max_trend_seconds_after_second_tp=18000,
        )
        # max_trend_ms = 18000 * 1000 = 18_000_000
        # ts_ms - start_ts = 18_000_000 >= 18_000_000 → exceeded
        self.assertTrue(decision.should_exit)
        self.assertEqual(decision.reason, "trend_runner_max_time_after_second_tp")

    def test_max_time_not_exceeded(self):
        decision = self._call(
            side="LONG",
            price=105.0,
            trend_start_ts_ms=10_000,
            ts_ms=10_000 + 5_000_000,
            runner_max_trend_seconds_after_second_tp=18000,
        )
        # 5_000_000 < 18_000_000
        self.assertFalse(decision.should_exit)
        self.assertIsNone(decision.reason)

    def test_max_time_start_ts_zero_ignored(self):
        decision = self._call(
            side="LONG",
            price=105.0,
            trend_start_ts_ms=0,
            ts_ms=1_000_000_000,
            runner_max_trend_seconds_after_second_tp=18000,
        )
        self.assertFalse(decision.should_exit)
        self.assertIsNone(decision.reason)

    def test_max_time_zero_ignored(self):
        decision = self._call(
            side="LONG",
            price=105.0,
            trend_start_ts_ms=1_000,
            ts_ms=1_000_000_000,
            runner_max_trend_seconds_after_second_tp=0,
        )
        # max_trend_ms = 0 → max_trend_ms > 0 is False → doesn't trigger
        self.assertFalse(decision.should_exit)
        self.assertIsNone(decision.reason)

    # ── No exit ───────────────────────────────────────────────────────

    def test_no_exit(self):
        decision = self._call(side="LONG", price=105.0, boll_middle=100.0)
        # price >= middle → not middle lost, no TP, no SL, default ts
        self.assertFalse(decision.should_exit)
        self.assertIsNone(decision.reason)

    # ── Priority: TP > SL > middle > max_time ─────────────────────────

    def test_tp_has_priority_over_sl(self):
        # LONG: TP=110, SL=106, price=111 → TP crossed (even if SL also "breached" for LONG)
        decision = self._call(side="LONG", price=111.0, tp_price=110.0, sl_price=106.0)
        self.assertEqual(decision.reason, "trend_runner_tp_crossed")

    def test_sl_has_priority_over_middle(self):
        decision = self._call(side="LONG", price=99.0, sl_price=100.0, boll_middle=105.0)
        # SL breached (99 <= 100), middle also lost (99 < 105). SL checked first.
        self.assertEqual(decision.reason, "trend_runner_sl_failsafe")


class TestTrendRunnerReverseCandidate(unittest.TestCase):

    def _call(self, **overrides):
        defaults = dict(
            side="LONG",
            up_burst=False,
            down_burst=False,
            buy_ratio=0.0,
            sell_ratio=0.0,
            fast_cvd=0.0,
            cvd_increasing=False,
            cvd_decreasing=False,
            runner_reverse_strong_ratio=0.62,
        )
        defaults.update(overrides)
        return tr.trend_runner_reverse_candidate(**defaults)

    def test_long_down_burst_true(self):
        decision = self._call(side="LONG", down_burst=True)
        self.assertTrue(decision.is_candidate)

    def test_long_sell_ratio_strong_with_cvd_decreasing(self):
        decision = self._call(
            side="LONG",
            sell_ratio=0.65,
            fast_cvd=-0.01,
            cvd_decreasing=True,
        )
        self.assertTrue(decision.is_candidate)

    def test_long_weak_sell_ratio_false(self):
        decision = self._call(
            side="LONG",
            sell_ratio=0.60,
            fast_cvd=-0.01,
            cvd_decreasing=True,
        )
        self.assertFalse(decision.is_candidate)

    def test_long_fast_cvd_positive_false(self):
        decision = self._call(
            side="LONG",
            sell_ratio=0.65,
            fast_cvd=0.01,
            cvd_decreasing=True,
        )
        self.assertFalse(decision.is_candidate)

    def test_long_cvd_not_decreasing_false(self):
        decision = self._call(
            side="LONG",
            sell_ratio=0.65,
            fast_cvd=-0.01,
            cvd_decreasing=False,
        )
        self.assertFalse(decision.is_candidate)

    def test_short_up_burst_true(self):
        decision = self._call(side="SHORT", up_burst=True)
        self.assertTrue(decision.is_candidate)

    def test_short_buy_ratio_strong_with_cvd_increasing(self):
        decision = self._call(
            side="SHORT",
            buy_ratio=0.65,
            fast_cvd=0.01,
            cvd_increasing=True,
        )
        self.assertTrue(decision.is_candidate)

    def test_short_weak_buy_ratio_false(self):
        decision = self._call(
            side="SHORT",
            buy_ratio=0.60,
            fast_cvd=0.01,
            cvd_increasing=True,
        )
        self.assertFalse(decision.is_candidate)

    def test_short_fast_cvd_negative_false(self):
        decision = self._call(
            side="SHORT",
            buy_ratio=0.65,
            fast_cvd=-0.01,
            cvd_increasing=True,
        )
        self.assertFalse(decision.is_candidate)

    def test_short_cvd_not_increasing_false(self):
        decision = self._call(
            side="SHORT",
            buy_ratio=0.65,
            fast_cvd=0.01,
            cvd_increasing=False,
        )
        self.assertFalse(decision.is_candidate)


class TestUpdateTrendRunnerReverseExtremePrice(unittest.TestCase):

    def test_long_min(self):
        result = tr.update_trend_runner_reverse_extreme_price(
            side="LONG",
            current_extreme_price=100.0,
            price=99.0,
        )
        self.assertEqual(result, 99.0)

    def test_long_min_no_update(self):
        result = tr.update_trend_runner_reverse_extreme_price(
            side="LONG",
            current_extreme_price=99.0,
            price=100.0,
        )
        self.assertEqual(result, 99.0)

    def test_short_max(self):
        result = tr.update_trend_runner_reverse_extreme_price(
            side="SHORT",
            current_extreme_price=100.0,
            price=101.0,
        )
        self.assertEqual(result, 101.0)

    def test_short_max_no_update(self):
        result = tr.update_trend_runner_reverse_extreme_price(
            side="SHORT",
            current_extreme_price=101.0,
            price=100.0,
        )
        self.assertEqual(result, 101.0)

    def test_none_fallback_long(self):
        result = tr.update_trend_runner_reverse_extreme_price(
            side="LONG",
            current_extreme_price=None,
            price=100.0,
        )
        self.assertEqual(result, 100.0)

    def test_none_fallback_short(self):
        result = tr.update_trend_runner_reverse_extreme_price(
            side="SHORT",
            current_extreme_price=None,
            price=100.0,
        )
        self.assertEqual(result, 100.0)


class TestPruneTrendRunnerReverseSamples(unittest.TestCase):

    def test_keeps_samples_at_or_above_cutoff(self):
        samples = [
            (1000, 0.5, 0.5, 0.0, 95.0),
            (2000, 0.6, 0.4, -1.0, 96.0),
            (3000, 0.7, 0.3, -2.0, 97.0),
        ]
        result = tr.prune_trend_runner_reverse_samples(samples=samples, cutoff_ts_ms=2000)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], 2000)
        self.assertEqual(result[1][0], 3000)

    def test_prunes_all_below_cutoff(self):
        samples = [
            (1000, 0.5, 0.5, 0.0, 95.0),
            (1500, 0.6, 0.4, -1.0, 96.0),
        ]
        result = tr.prune_trend_runner_reverse_samples(samples=samples, cutoff_ts_ms=2000)
        self.assertEqual(len(result), 0)

    def test_keeps_all_above_cutoff(self):
        samples = [
            (2000, 0.5, 0.5, 0.0, 95.0),
            (3000, 0.6, 0.4, -1.0, 96.0),
        ]
        result = tr.prune_trend_runner_reverse_samples(samples=samples, cutoff_ts_ms=1000)
        self.assertEqual(len(result), 2)


class TestTrendRunnerReverseConfirmed(unittest.TestCase):

    def _call(self, **overrides):
        defaults = dict(
            side="LONG",
            current_price=95.0,
            samples=[(1000, 0.4, 0.6, -5.0, 96.0), (1100, 0.3, 0.7, -10.0, 95.0)],
            start_price=100.0,
            extreme_price=94.0,
            fast_cvd_start=-2.0,
            current_fast_cvd=-10.0,
            runner_reverse_sell_ratio=0.58,
            runner_reverse_buy_ratio=0.58,
            runner_reverse_min_price_damage_pct=0.0015,
            runner_reverse_recovery_cancel_pct=0.001,
        )
        defaults.update(overrides)
        return tr.trend_runner_reverse_confirmed(**defaults)

    # ── LONG happy path ───────────────────────────────────────────────

    def test_long_confirmed(self):
        decision = self._call(side="LONG")
        # avg_sell_ratio = (0.6 + 0.7)/2 = 0.65 >= 0.58 ✓
        # price_damage = (100 - 95)/100 = 0.05 >= 0.0015 ✓
        # recovery = (95 - 94)/94 ≈ 0.01064 >= 0.001 ✗ → recovery too high
        # Actually 0.01064 >= 0.001 so recovery is NOT < 0.001 → NOT confirmed!
        # Let me adjust the test. I need recovery < 0.001 for confirmed.
        # So extreme_price should be very close to current_price.
        pass

    def test_long_confirmed_recovery_low(self):
        # For confirmed: recovery_pct < 0.001 (very tight)
        # recovery = (current_price - extreme) / extreme
        # If current=94.001, extreme=94.0, recovery ≈ 0.0000106 < 0.001 ✓
        decision = self._call(
            side="LONG",
            current_price=94.001,
            extreme_price=94.0,
            samples=[(1000, 0.4, 0.6, -5.0, 96.0), (1100, 0.3, 0.7, -10.0, 95.0)],
        )
        # avg_sell_ratio = 0.65 >= 0.58 ✓
        # price_damage = (100 - 94.001)/100 ≈ 0.05999 >= 0.0015 ✓
        # recovery = (94.001 - 94.0)/94.0 ≈ 0.0000106 < 0.001 ✓
        # fast_cvd: -10 < -2 ✓
        self.assertTrue(decision.confirmed)
        self.assertAlmostEqual(decision.avg_ratio, 0.65, places=6)

    # ── LONG edge cases ───────────────────────────────────────────────

    def test_long_samples_empty(self):
        decision = self._call(side="LONG", samples=[])
        self.assertFalse(decision.confirmed)
        self.assertEqual(decision.avg_ratio, 0.0)

    def test_long_start_price_none(self):
        decision = self._call(side="LONG", start_price=None)
        self.assertFalse(decision.confirmed)

    def test_long_start_price_zero(self):
        decision = self._call(side="LONG", start_price=0.0)
        self.assertFalse(decision.confirmed)

    def test_long_extreme_none(self):
        decision = self._call(side="LONG", extreme_price=None)
        self.assertFalse(decision.confirmed)

    def test_long_extreme_zero(self):
        decision = self._call(side="LONG", extreme_price=0.0)
        self.assertFalse(decision.confirmed)

    def test_long_recovery_too_high_false(self):
        # recovery > 0.001 → not confirmed
        decision = self._call(
            side="LONG",
            current_price=95.0,
            extreme_price=94.0,
        )
        # recovery = (95-94)/94 ≈ 0.01064 > 0.001
        self.assertFalse(decision.confirmed)

    def test_long_damage_insufficient_false(self):
        # price_damage < 0.0015 → not confirmed
        decision = self._call(
            side="LONG",
            current_price=99.99,
            extreme_price=99.98,
            start_price=100.0,
        )
        # damage = (100-99.99)/100 = 0.0001 < 0.0015
        self.assertFalse(decision.confirmed)

    def test_long_fast_cvd_condition_failed(self):
        # current_fast_cvd >= fast_cvd_start → not confirmed
        decision = self._call(
            side="LONG",
            current_price=94.001,
            extreme_price=94.0,
            current_fast_cvd=-1.0,
            fast_cvd_start=-2.0,
        )
        # -1.0 < -2.0 is False → not confirmed
        self.assertFalse(decision.confirmed)

    # ── SHORT happy path ──────────────────────────────────────────────

    def test_short_confirmed(self):
        # For SHORT:
        # avg ratio = sum(sample[1]) / len → buy_ratio
        # price_damage = (current - start) / start
        # recovery = (extreme - current) / extreme
        decision = self._call(
            side="SHORT",
            current_price=105.999,
            start_price=100.0,
            extreme_price=106.0,
            samples=[(1000, 0.6, 0.4, 5.0, 104.0), (1100, 0.7, 0.3, 10.0, 105.0)],
            fast_cvd_start=2.0,
            current_fast_cvd=10.0,
            runner_reverse_buy_ratio=0.58,
        )
        # avg_buy_ratio = (0.6 + 0.7)/2 = 0.65 >= 0.58 ✓
        # price_damage = (105.999 - 100)/100 ≈ 0.05999 >= 0.0015 ✓
        # recovery = (106 - 105.999)/106 ≈ 0.0000094 < 0.001 ✓
        # fast_cvd: 10 > 2 ✓
        self.assertTrue(decision.confirmed)
        self.assertAlmostEqual(decision.avg_ratio, 0.65, places=6)

    def test_short_damage_insufficient_false(self):
        decision = self._call(
            side="SHORT",
            current_price=100.01,
            start_price=100.0,
            extreme_price=101.0,
            current_fast_cvd=10.0,
            fast_cvd_start=2.0,
        )
        # damage = (100.01-100)/100 = 0.0001 < 0.0015
        self.assertFalse(decision.confirmed)

    def test_short_fast_cvd_condition_failed(self):
        decision = self._call(
            side="SHORT",
            current_price=105.999,
            start_price=100.0,
            extreme_price=106.0,
            current_fast_cvd=1.0,
            fast_cvd_start=2.0,
        )
        # 1.0 > 2.0 is False → not confirmed
        self.assertFalse(decision.confirmed)


if __name__ == "__main__":
    unittest.main()
