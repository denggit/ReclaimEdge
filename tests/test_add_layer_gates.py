from __future__ import annotations

import unittest

from src.indicators.cvd_tracker import CvdSnapshot
from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
    StrategyPositionState,
)

NOW_MS = 2_000_000


def boll() -> BollSnapshot:
    return BollSnapshot("ETH-USDT-SWAP", 1_000, 100.0, 110.0, 120.0, 90.0, 0.1, 0.1, True, True)


def cvd() -> CvdSnapshot:
    return CvdSnapshot(
        ts_ms=1_000,
        price=100.0,
        side="buy",
        size=1.0,
        signed_delta=1.0,
        total_cvd=1.0,
        fast_cvd=1.0,
        previous_fast_cvd=0.0,
        buy_volume=1.0,
        sell_volume=0.0,
        buy_ratio=1.0,
        sell_ratio=0.0,
        cross_positive=True,
        cross_negative=False,
        cvd_increasing=True,
        cvd_decreasing=False,
        no_new_low=True,
        no_new_high=True,
        window_low=99.0,
        window_high=101.0,
        burst_net_move_pct=0.0,
        burst_range_pct=0.0,
        baseline_range_pct=0.0,
        burst_move_ratio=0.0,
        burst_volume=0.0,
        baseline_volume=0.0,
        burst_volume_ratio=0.0,
        up_burst=False,
        down_burst=False,
    )


def strategy(**config_overrides) -> BollCvdReclaimStrategy:
    values = dict(max_layers=12)
    values.update(config_overrides)
    return BollCvdReclaimStrategy(
        BollCvdReclaimStrategyConfig(**values),
        SimplePositionSizer(SimplePositionSizerConfig()),
    )


def long_state(**overrides) -> StrategyPositionState:
    values = dict(
        side="LONG",
        layers=2,
        last_entry_price=100.0,
        last_order_ts_ms=0,
        total_entry_qty=1.0,
        total_entry_notional=100.0,
        avg_entry_price=100.0,
    )
    values.update(overrides)
    return StrategyPositionState(**values)


def short_state(**overrides) -> StrategyPositionState:
    values = dict(
        side="SHORT",
        layers=2,
        last_entry_price=100.0,
        last_order_ts_ms=0,
        total_entry_qty=1.0,
        total_entry_notional=100.0,
        avg_entry_price=100.0,
    )
    values.update(overrides)
    return StrategyPositionState(**values)


class AddLayerGateTest(unittest.TestCase):
    def test_linear_add_gap_target_layer_2_uses_base(self) -> None:
        strat = strategy()
        self.assertAlmostEqual(strat._add_layer_gap_pct_for_target_layer(2), 0.003)

    def test_linear_add_gap_target_layer_3_uses_base_plus_step(self) -> None:
        strat = strategy()
        self.assertAlmostEqual(strat._add_layer_gap_pct_for_target_layer(3), 0.004)

    def test_linear_add_gap_target_layer_4_uses_base_plus_2_step(self) -> None:
        strat = strategy()
        self.assertAlmostEqual(strat._add_layer_gap_pct_for_target_layer(4), 0.005)

    def test_linear_add_gap_target_layer_8_uses_base_plus_6_step(self) -> None:
        strat = strategy()
        self.assertAlmostEqual(strat._add_layer_gap_pct_for_target_layer(8), 0.009)

    def test_linear_add_gap_target_layer_10_uses_base_plus_8_step(self) -> None:
        strat = strategy()
        self.assertAlmostEqual(strat._add_layer_gap_pct_for_target_layer(10), 0.011)

    def test_linear_add_gap_target_layer_20_uses_base_plus_18_step(self) -> None:
        strat = strategy()
        self.assertAlmostEqual(strat._add_layer_gap_pct_for_target_layer(20), 0.021)

    def test_long_add_l4_blocked_at_99_60_when_gap_is_0_005(self) -> None:
        strat = strategy()
        strat.state = long_state(layers=3)
        # L4 gap = 0.003 + (4-2)*0.001 = 0.005, required_price = 100 * 0.995 = 99.50
        gap_ok, gap_pct, required_price = strat._add_gap_passed("LONG", 99.60, 4)
        blocked = strat._maybe_open_or_add_long(99.60, NOW_MS, boll(), cvd())
        self.assertFalse(gap_ok)
        self.assertAlmostEqual(gap_pct, 0.005)
        self.assertAlmostEqual(required_price, 99.50)
        self.assertIsNone(blocked)

    def test_long_add_l4_allowed_at_99_50_when_gap_is_0_005(self) -> None:
        strat = strategy()
        strat.state = long_state(layers=3)
        gap_ok, gap_pct, required_price = strat._add_gap_passed("LONG", 99.50, 4)
        allowed = strat._maybe_open_or_add_long(99.50, NOW_MS, boll(), cvd())
        self.assertTrue(gap_ok)
        self.assertAlmostEqual(gap_pct, 0.005)
        self.assertAlmostEqual(required_price, 99.50)
        self.assertIsNotNone(allowed)
        self.assertEqual(allowed.intent_type, "ADD_LONG")
        self.assertEqual(allowed.layer_index, 4)

    def test_short_add_l4_blocked_at_100_40_when_gap_is_0_005(self) -> None:
        strat = strategy()
        strat.state = short_state(layers=3)
        # L4 gap = 0.005, required_price = 100 * 1.005 = 100.50
        gap_ok, gap_pct, required_price = strat._add_gap_passed("SHORT", 100.40, 4)
        blocked = strat._maybe_open_or_add_short(100.40, NOW_MS, boll(), cvd())
        self.assertFalse(gap_ok)
        self.assertAlmostEqual(gap_pct, 0.005)
        self.assertAlmostEqual(required_price, 100.50)
        self.assertIsNone(blocked)

    def test_short_add_l4_allowed_at_100_50_when_gap_is_0_005(self) -> None:
        strat = strategy()
        strat.state = short_state(layers=3)
        gap_ok, gap_pct, required_price = strat._add_gap_passed("SHORT", 100.50, 4)
        allowed = strat._maybe_open_or_add_short(100.50, NOW_MS, boll(), cvd())
        self.assertTrue(gap_ok)
        self.assertAlmostEqual(gap_pct, 0.005)
        self.assertAlmostEqual(required_price, 100.50)
        self.assertIsNotNone(allowed)
        self.assertEqual(allowed.intent_type, "ADD_SHORT")
        self.assertEqual(allowed.layer_index, 4)

    def test_first_add_block_prevents_add_within_30_minutes(self) -> None:
        strat = strategy()
        strat.state = long_state(layers=1, last_order_ts_ms=NOW_MS - 29 * 60 * 1000)

        result = strat._maybe_open_or_add_long(99.0, NOW_MS, boll(), cvd())

        self.assertIsNone(result)
        self.assertEqual(strat.state.layers, 1)

    def test_first_add_block_cannot_be_bypassed_even_when_adverse_gap_is_1pct(self) -> None:
        long_strat = strategy()
        long_strat.state = long_state(layers=1, last_order_ts_ms=NOW_MS - 29 * 60 * 1000)

        long_result = long_strat._maybe_open_or_add_long(99.0, NOW_MS, boll(), cvd())

        self.assertIsNone(long_result)
        self.assertEqual(long_strat.state.layers, 1)

        short_strat = strategy()
        short_strat.state = short_state(layers=1, last_order_ts_ms=NOW_MS - 29 * 60 * 1000)

        short_result = short_strat._maybe_open_or_add_short(101.0, NOW_MS, boll(), cvd())

        self.assertIsNone(short_result)
        self.assertEqual(short_strat.state.layers, 1)

    def test_first_add_allowed_after_30_minutes(self) -> None:
        strat = strategy()
        strat.state = long_state(layers=1, last_order_ts_ms=NOW_MS - 31 * 60 * 1000)

        result = strat._maybe_open_or_add_long(99.70, NOW_MS, boll(), cvd())

        self.assertIsNotNone(result)
        self.assertEqual(result.intent_type, "ADD_LONG")
        self.assertEqual(result.layer_index, 2)

    def test_add_interval_blocks_second_add_before_10_minutes(self) -> None:
        strat = strategy()
        strat.state = long_state(layers=2, last_order_ts_ms=NOW_MS - 5 * 60 * 1000)

        result = strat._maybe_open_or_add_long(99.70, NOW_MS, boll(), cvd())

        self.assertIsNone(result)
        self.assertEqual(strat.state.layers, 2)

    def test_add_interval_target_layer_2_blocks_when_gap_below_dynamic_bypass(self) -> None:
        strat = strategy()
        strat.state = long_state(layers=2, last_order_ts_ms=NOW_MS - 5 * 60 * 1000)

        gap_ok, gap_pct, _ = strat._add_gap_passed("LONG", 99.50, 2)
        passed, reason = strat._add_timing_passed("LONG", 99.50, NOW_MS, 2)

        self.assertTrue(gap_ok)
        self.assertAlmostEqual(gap_pct, 0.003)
        self.assertFalse(passed)
        self.assertEqual(reason, "add_interval")

    def test_add_interval_target_layer_2_bypassed_when_gap_reaches_dynamic_bypass(self) -> None:
        strat = strategy()
        strat.state = long_state(layers=2, last_order_ts_ms=NOW_MS - 5 * 60 * 1000)

        # bypass_gap = 0.003 * 2 = 0.006, adverse = (100 - 99.39) / 100 = 0.0061 >= 0.006
        gap_ok, gap_pct, _ = strat._add_gap_passed("LONG", 99.39, 2)
        passed, reason = strat._add_timing_passed("LONG", 99.39, NOW_MS, 2)

        self.assertTrue(gap_ok)
        self.assertAlmostEqual(gap_pct, 0.003)
        self.assertTrue(passed)
        self.assertEqual(reason, "ok")

    def test_add_interval_target_layer_4_blocks_when_gap_below_dynamic_bypass(self) -> None:
        strat = strategy()
        strat.state = long_state(layers=3, last_order_ts_ms=NOW_MS - 5 * 60 * 1000)

        # L4 gap = 0.005, bypass_gap = 0.005 * 2 = 0.010
        # price=99.10, adverse = 0.009 < 0.010 → blocked
        gap_ok, gap_pct, _ = strat._add_gap_passed("LONG", 99.10, 4)
        passed, reason = strat._add_timing_passed("LONG", 99.10, NOW_MS, 4)

        self.assertTrue(gap_ok)
        self.assertAlmostEqual(gap_pct, 0.005)
        self.assertFalse(passed)
        self.assertEqual(reason, "add_interval")

    def test_add_interval_target_layer_4_bypassed_when_gap_reaches_dynamic_bypass(self) -> None:
        strat = strategy()
        strat.state = long_state(layers=3, last_order_ts_ms=NOW_MS - 5 * 60 * 1000)

        # L4 gap = 0.005, bypass_gap = 0.005 * 2 = 0.010
        # price=98.99, adverse = 0.0101 >= 0.010 → bypassed
        gap_ok, gap_pct, _ = strat._add_gap_passed("LONG", 98.99, 4)
        passed, reason = strat._add_timing_passed("LONG", 98.99, NOW_MS, 4)

        self.assertTrue(gap_ok)
        self.assertAlmostEqual(gap_pct, 0.005)
        self.assertTrue(passed)
        self.assertEqual(reason, "ok")

    def test_add_interval_bypass_gap_uses_double_target_layer_gap(self) -> None:
        strat = strategy()

        self.assertAlmostEqual(strat._add_min_interval_bypass_gap_pct_for_target_layer(2), 0.006)
        self.assertAlmostEqual(strat._add_min_interval_bypass_gap_pct_for_target_layer(4), 0.010)
        self.assertAlmostEqual(strat._add_min_interval_bypass_gap_pct_for_target_layer(8), 0.018)
        self.assertAlmostEqual(strat._add_min_interval_bypass_gap_pct_for_target_layer(11), 0.024)

    def test_add_interval_bypassed_when_gap_reaches_target_layer_gap_times_2(self) -> None:
        strat = strategy()
        strat.state = long_state(layers=2, last_order_ts_ms=NOW_MS - 5 * 60 * 1000)

        # target_layer=3, linear gap = 0.003 + (3-2)*0.001 = 0.004
        # bypass_gap = 0.004 * 2 = 0.008
        # adverse = (100 - 99.19) / 100 = 0.0081 >= 0.008 → bypassed
        result = strat._maybe_open_or_add_long(99.19, NOW_MS, boll(), cvd())

        self.assertIsNotNone(result)
        self.assertEqual(result.intent_type, "ADD_LONG")
        self.assertEqual(result.layer_index, 3)

    def test_legacy_interval_bypass_config_does_not_control_dynamic_bypass(self) -> None:
        strat = strategy(add_min_interval_bypass_gap_pct=0.003)
        strat.state = long_state(layers=10, last_order_ts_ms=NOW_MS - 5 * 60 * 1000)

        # L11 gap = 0.003 + (11-2)*0.001 = 0.012, bypass = 0.012 * 2 = 0.024
        # price=99.20, adverse = 0.008 < 0.024 → blocked
        blocked = strat._maybe_open_or_add_long(99.20, NOW_MS, boll(), cvd())

        self.assertIsNone(blocked)
        self.assertEqual(strat.state.layers, 10)

        strat.state = long_state(layers=10, last_order_ts_ms=NOW_MS - 5 * 60 * 1000)
        allowed = strat._maybe_open_or_add_long(97.59, NOW_MS, boll(), cvd())

        self.assertIsNotNone(allowed)
        self.assertEqual(allowed.intent_type, "ADD_LONG")
        self.assertEqual(allowed.layer_index, 11)

    def test_long_add_blocked_when_avg_improvement_below_0_12_pct(self) -> None:
        strat = strategy()
        strat.state = long_state(
            layers=1,
            total_entry_qty=100.0,
            total_entry_notional=10_000.0,
            avg_entry_price=100.0,
        )

        passed, improvement_pct, projected_avg = strat._add_avg_improvement_passed("LONG", 99.70, 2)
        result = strat._maybe_open_or_add_long(99.70, NOW_MS, boll(), cvd())

        self.assertFalse(passed)
        self.assertLess(improvement_pct, 0.0012)
        self.assertGreater(projected_avg, 99.88)
        self.assertIsNone(result)
        self.assertEqual(strat.state.layers, 1)

    def test_long_add_allowed_when_avg_improvement_meets_0_12_pct(self) -> None:
        strat = strategy()
        strat.state = long_state(
            layers=1,
            total_entry_qty=100.0,
            total_entry_notional=10_000.0,
            avg_entry_price=100.0,
        )

        passed, improvement_pct, _ = strat._add_avg_improvement_passed("LONG", 99.0, 2)
        result = strat._maybe_open_or_add_long(99.0, NOW_MS, boll(), cvd())

        self.assertTrue(passed)
        self.assertGreaterEqual(improvement_pct, 0.0012)
        self.assertIsNotNone(result)
        self.assertEqual(result.intent_type, "ADD_LONG")
        self.assertEqual(result.layer_index, 2)
        self.assertIn("补仓后均价改善", result.reason)

    def test_short_add_blocked_when_avg_improvement_below_0_12_pct(self) -> None:
        strat = strategy()
        strat.state = short_state(
            layers=1,
            total_entry_qty=100.0,
            total_entry_notional=10_000.0,
            avg_entry_price=100.0,
        )

        passed, improvement_pct, projected_avg = strat._add_avg_improvement_passed("SHORT", 100.30, 2)
        result = strat._maybe_open_or_add_short(100.30, NOW_MS, boll(), cvd())

        self.assertFalse(passed)
        self.assertLess(improvement_pct, 0.0012)
        self.assertLess(projected_avg, 100.12)
        self.assertIsNone(result)
        self.assertEqual(strat.state.layers, 1)

    def test_short_add_allowed_when_avg_improvement_meets_0_12_pct(self) -> None:
        strat = strategy()
        strat.state = short_state(
            layers=1,
            total_entry_qty=100.0,
            total_entry_notional=10_000.0,
            avg_entry_price=100.0,
        )

        passed, improvement_pct, _ = strat._add_avg_improvement_passed("SHORT", 101.0, 2)
        result = strat._maybe_open_or_add_short(101.0, NOW_MS, boll(), cvd())

        self.assertTrue(passed)
        self.assertGreaterEqual(improvement_pct, 0.0012)
        self.assertIsNotNone(result)
        self.assertEqual(result.intent_type, "ADD_SHORT")
        self.assertEqual(result.layer_index, 2)
        self.assertIn("补仓后均价改善", result.reason)

    def test_open_long_not_affected_by_first_add_block_or_avg_improvement_gate(self) -> None:
        strat = strategy(add_min_avg_improvement_pct=1.0)
        strat.state.last_order_ts_ms = NOW_MS

        result = strat._maybe_open_or_add_long(99.70, NOW_MS, boll(), cvd())

        self.assertIsNotNone(result)
        self.assertEqual(result.intent_type, "OPEN_LONG")
        self.assertEqual(result.layer_index, 1)

    def test_near_tp_add_disabled_still_blocks_add_before_new_gates(self) -> None:
        strat = strategy()
        strat.state = long_state(
            layers=2,
            last_order_ts_ms=NOW_MS - 60 * 60 * 1000,
            total_entry_qty=100.0,
            total_entry_notional=10_000.0,
            near_tp_add_disabled=True,
        )

        with self.assertLogs("src.strategies.boll_cvd_reclaim_strategy", level="INFO") as logs:
            result = strat._maybe_open_or_add_long(99.70, NOW_MS, boll(), cvd())

        self.assertIsNone(result)
        self.assertEqual(strat.state.layers, 2)
        self.assertIn("reason=near_tp_protected", "\n".join(logs.output))
        self.assertNotIn("reason=avg_improvement", "\n".join(logs.output))

    def test_middle_runner_active_blocks_add_long_but_not_open(self) -> None:
        open_strat = strategy()
        open_strat.state.middle_runner_active = True
        opened = open_strat._maybe_open_or_add_long(99.70, NOW_MS, boll(), cvd())
        self.assertIsNotNone(opened)
        self.assertEqual(opened.intent_type, "OPEN_LONG")

        strat = strategy()
        strat.state = long_state(
            layers=2,
            last_order_ts_ms=NOW_MS - 60 * 60 * 1000,
            total_entry_qty=100.0,
            total_entry_notional=10_000.0,
            middle_runner_active=True,
            middle_runner_add_disabled=True,
        )
        with self.assertLogs("src.strategies.boll_cvd_reclaim_strategy", level="INFO") as logs:
            result = strat._maybe_open_or_add_long(99.70, NOW_MS, boll(), cvd())
        self.assertIsNone(result)
        self.assertIn("reason=middle_runner_active", "\n".join(logs.output))

    def test_middle_runner_active_blocks_add_short_but_not_open(self) -> None:
        open_strat = strategy()
        open_strat.state.middle_runner_active = True
        opened = open_strat._maybe_open_or_add_short(100.30, NOW_MS, boll(), cvd())
        self.assertIsNotNone(opened)
        self.assertEqual(opened.intent_type, "OPEN_SHORT")

        strat = strategy()
        strat.state = short_state(
            layers=2,
            last_order_ts_ms=NOW_MS - 60 * 60 * 1000,
            total_entry_qty=100.0,
            total_entry_notional=10_000.0,
            middle_runner_active=True,
            middle_runner_add_disabled=True,
        )
        with self.assertLogs("src.strategies.boll_cvd_reclaim_strategy", level="INFO") as logs:
            result = strat._maybe_open_or_add_short(100.30, NOW_MS, boll(), cvd())
        self.assertIsNone(result)
        self.assertIn("reason=middle_runner_active", "\n".join(logs.output))


class AddLayerGatesPureTest(unittest.TestCase):
    """Direct tests for src.strategies.add_layer_gates pure functions."""

    def test_linear_add_layer_gap_vs_target_layer(self) -> None:
        from src.strategies.add_layer_gates import add_layer_gap_pct_for_target_layer

        cases = [
            (2, 0.003),
            (3, 0.004),
            (4, 0.005),
            (5, 0.006),
            (6, 0.007),
            (7, 0.008),
            (8, 0.009),
            (9, 0.010),
            (10, 0.011),
            (20, 0.021),
        ]
        for target_layer, expected in cases:
            with self.subTest(target_layer=target_layer):
                result = add_layer_gap_pct_for_target_layer(
                    target_layer=target_layer,
                    add_gap_mode="linear",
                    add_gap_base_pct=0.003,
                    add_gap_step_pct=0.001,
                )
                self.assertAlmostEqual(result, expected)

    def test_add_layer_gap_step_pct_zero_all_layers_equal_base(self) -> None:
        from src.strategies.add_layer_gates import add_layer_gap_pct_for_target_layer

        for target_layer in (2, 5, 10, 20):
            with self.subTest(target_layer=target_layer):
                result = add_layer_gap_pct_for_target_layer(
                    target_layer=target_layer,
                    add_gap_mode="linear",
                    add_gap_base_pct=0.005,
                    add_gap_step_pct=0.0,
                )
                self.assertAlmostEqual(result, 0.005)

    def test_unsupported_add_gap_mode_raises_value_error(self) -> None:
        from src.strategies.add_layer_gates import add_layer_gap_pct_for_target_layer

        for bad_mode in ("", "segmented", "fixed"):
            with self.subTest(mode=bad_mode):
                with self.assertRaises(ValueError):
                    add_layer_gap_pct_for_target_layer(
                        target_layer=2,
                        add_gap_mode=bad_mode,
                        add_gap_base_pct=0.003,
                        add_gap_step_pct=0.001,
                    )

    def test_space_trimmed_mode_linear_passes(self) -> None:
        from src.strategies.add_layer_gates import add_layer_gap_pct_for_target_layer

        # " linear " should be trimmed and lowered to "linear"
        result = add_layer_gap_pct_for_target_layer(
            target_layer=2,
            add_gap_mode=" linear ",
            add_gap_base_pct=0.003,
            add_gap_step_pct=0.001,
        )
        self.assertAlmostEqual(result, 0.003)

    def test_check_add_gap_long_passed_linear(self) -> None:
        from src.strategies.add_layer_gates import check_add_gap

        decision = check_add_gap(
            side="LONG",
            price=99.50,
            last_entry_price=100.0,
            target_layer=4,
            add_gap_mode="linear",
            add_gap_base_pct=0.003,
            add_gap_step_pct=0.001,
        )
        self.assertTrue(decision.ok)
        # L4 gap = 0.003 + (4-2)*0.001 = 0.005, required_price = 100 * 0.995 = 99.50
        self.assertAlmostEqual(decision.gap_pct, 0.005)
        self.assertAlmostEqual(decision.required_price, 99.50)

    def test_check_add_gap_long_blocked_linear(self) -> None:
        from src.strategies.add_layer_gates import check_add_gap

        decision = check_add_gap(
            side="LONG",
            price=99.60,
            last_entry_price=100.0,
            target_layer=4,
            add_gap_mode="linear",
            add_gap_base_pct=0.003,
            add_gap_step_pct=0.001,
        )
        self.assertFalse(decision.ok)
        self.assertAlmostEqual(decision.gap_pct, 0.005)
        # required_price = 100 * 0.995 = 99.50, price 99.60 > 99.50 → blocked
        self.assertAlmostEqual(decision.required_price, 99.50)

    def test_check_add_gap_short_passed_linear(self) -> None:
        from src.strategies.add_layer_gates import check_add_gap

        decision = check_add_gap(
            side="SHORT",
            price=100.50,
            last_entry_price=100.0,
            target_layer=4,
            add_gap_mode="linear",
            add_gap_base_pct=0.003,
            add_gap_step_pct=0.001,
        )
        self.assertTrue(decision.ok)
        # L4 gap = 0.005, required_price = 100 * 1.005 = 100.50
        self.assertAlmostEqual(decision.gap_pct, 0.005)
        self.assertAlmostEqual(decision.required_price, 100.50)

    def test_check_add_gap_short_blocked_linear(self) -> None:
        from src.strategies.add_layer_gates import check_add_gap

        decision = check_add_gap(
            side="SHORT",
            price=100.40,
            last_entry_price=100.0,
            target_layer=4,
            add_gap_mode="linear",
            add_gap_base_pct=0.003,
            add_gap_step_pct=0.001,
        )
        self.assertFalse(decision.ok)
        self.assertAlmostEqual(decision.gap_pct, 0.005)
        self.assertAlmostEqual(decision.required_price, 100.50)

    def test_check_add_gap_missing_last_entry(self) -> None:
        from src.strategies.add_layer_gates import check_add_gap

        for bad_price in (None, 0.0, -1.0):
            with self.subTest(last_entry_price=bad_price):
                decision = check_add_gap(
                    side="LONG",
                    price=99.0,
                    last_entry_price=bad_price,
                    target_layer=2,
                    add_gap_mode="linear",
                    add_gap_base_pct=0.003,
                    add_gap_step_pct=0.001,
                )
                self.assertFalse(decision.ok)
                self.assertAlmostEqual(decision.gap_pct, 0.003)
                self.assertAlmostEqual(decision.required_price, 0.0)

    def test_check_base_add_timing_missing_last_entry(self) -> None:
        from src.strategies.add_layer_gates import check_base_add_timing

        decision = check_base_add_timing(
            side="LONG",
            price=99.0,
            ts_ms=1_000_000,
            target_layer=2,
            layers=1,
            last_entry_price=None,
            last_order_ts_ms=0,
            first_add_block_seconds=1800,
            add_min_interval_seconds=600,
            add_gap_mode="linear",
            add_gap_base_pct=0.003,
            add_gap_step_pct=0.001,
        )
        self.assertFalse(decision.ok)
        self.assertEqual(decision.reason, "missing_last_entry")

    def test_check_base_add_timing_first_add_blocked(self) -> None:
        from src.strategies.add_layer_gates import check_base_add_timing

        decision = check_base_add_timing(
            side="LONG",
            price=99.0,
            ts_ms=1_000_000,
            target_layer=2,
            layers=1,
            last_entry_price=100.0,
            last_order_ts_ms=900_000,  # 100s ago, < 1800s
            first_add_block_seconds=1800,
            add_min_interval_seconds=600,
            add_gap_mode="linear",
            add_gap_base_pct=0.003,
            add_gap_step_pct=0.001,
        )
        self.assertFalse(decision.ok)
        self.assertEqual(decision.reason, "first_add_block")

    def test_check_base_add_timing_first_add_allowed(self) -> None:
        from src.strategies.add_layer_gates import check_base_add_timing

        decision = check_base_add_timing(
            side="LONG",
            price=99.0,
            ts_ms=3_000_000,
            target_layer=2,
            layers=1,
            last_entry_price=100.0,
            last_order_ts_ms=1_000_000,  # 2000s ago, > 1800s
            first_add_block_seconds=1800,
            add_min_interval_seconds=600,
            add_gap_mode="linear",
            add_gap_base_pct=0.003,
            add_gap_step_pct=0.001,
        )
        self.assertTrue(decision.ok)
        self.assertEqual(decision.reason, "ok")

    def test_check_base_add_timing_add_interval_blocked(self) -> None:
        from src.strategies.add_layer_gates import check_base_add_timing

        decision = check_base_add_timing(
            side="LONG",
            price=99.70,  # 0.3% adverse gap
            ts_ms=1_500_000,
            target_layer=2,
            layers=2,
            last_entry_price=100.0,
            last_order_ts_ms=1_000_000,  # 500s ago < 600s
            first_add_block_seconds=1800,
            add_min_interval_seconds=600,
            add_gap_mode="linear",
            add_gap_base_pct=0.003,
            add_gap_step_pct=0.001,
        )
        self.assertFalse(decision.ok)
        self.assertEqual(decision.reason, "add_interval")

    def test_check_base_add_timing_add_interval_bypassed(self) -> None:
        from src.strategies.add_layer_gates import check_base_add_timing

        # adverse_gap = (100 - 99.39) / 100 = 0.0061 >= bypass_gap = 0.006
        decision = check_base_add_timing(
            side="LONG",
            price=99.39,
            ts_ms=1_500_000,
            target_layer=2,
            layers=2,
            last_entry_price=100.0,
            last_order_ts_ms=1_000_000,  # 500s ago < 600s
            first_add_block_seconds=1800,
            add_min_interval_seconds=600,
            add_gap_mode="linear",
            add_gap_base_pct=0.003,
            add_gap_step_pct=0.001,
        )
        self.assertTrue(decision.ok)
        self.assertEqual(decision.reason, "ok")

    def test_check_base_add_timing_elapsed_passes_interval(self) -> None:
        from src.strategies.add_layer_gates import check_base_add_timing

        decision = check_base_add_timing(
            side="LONG",
            price=99.70,
            ts_ms=2_000_000,
            target_layer=2,
            layers=2,
            last_entry_price=100.0,
            last_order_ts_ms=1_000_000,  # 1000s ago > 600s
            first_add_block_seconds=1800,
            add_min_interval_seconds=600,
            add_gap_mode="linear",
            add_gap_base_pct=0.003,
            add_gap_step_pct=0.001,
        )
        self.assertTrue(decision.ok)
        self.assertEqual(decision.reason, "ok")

    def test_check_add_avg_improvement_long_passed(self) -> None:
        from src.strategies.add_layer_gates import check_add_avg_improvement

        decision = check_add_avg_improvement(
            side="LONG",
            price=99.0,  # cheaper buy lowers avg
            required_improvement_pct=0.0012,
            old_qty=1.0,
            old_notional=100.0,
            old_avg=100.0,
            add_qty=1.0,
        )
        self.assertTrue(decision.ok)
        self.assertGreaterEqual(decision.improvement_pct, 0.0012)
        self.assertLess(decision.projected_avg, 100.0)

    def test_check_add_avg_improvement_long_blocked(self) -> None:
        from src.strategies.add_layer_gates import check_add_avg_improvement

        decision = check_add_avg_improvement(
            side="LONG",
            price=99.70,  # small improvement
            required_improvement_pct=0.0012,
            old_qty=100.0,
            old_notional=10_000.0,
            old_avg=100.0,
            add_qty=0.1,
        )
        self.assertFalse(decision.ok)
        self.assertLess(decision.improvement_pct, 0.0012)

    def test_check_add_avg_improvement_short_passed(self) -> None:
        from src.strategies.add_layer_gates import check_add_avg_improvement

        decision = check_add_avg_improvement(
            side="SHORT",
            price=101.0,  # higher short entry raises avg
            required_improvement_pct=0.0012,
            old_qty=1.0,
            old_notional=100.0,
            old_avg=100.0,
            add_qty=1.0,
        )
        self.assertTrue(decision.ok)
        self.assertGreaterEqual(decision.improvement_pct, 0.0012)
        self.assertGreater(decision.projected_avg, 100.0)

    def test_check_add_avg_improvement_short_blocked(self) -> None:
        from src.strategies.add_layer_gates import check_add_avg_improvement

        decision = check_add_avg_improvement(
            side="SHORT",
            price=100.30,  # small improvement
            required_improvement_pct=0.0012,
            old_qty=100.0,
            old_notional=10_000.0,
            old_avg=100.0,
            add_qty=0.1,
        )
        self.assertFalse(decision.ok)
        self.assertLess(decision.improvement_pct, 0.0012)

    def test_check_add_avg_improvement_zero_required(self) -> None:
        from src.strategies.add_layer_gates import check_add_avg_improvement

        decision = check_add_avg_improvement(
            side="LONG",
            price=100.0,
            required_improvement_pct=0.0,
            old_qty=1.0,
            old_notional=100.0,
            old_avg=100.0,
            add_qty=0.0,
        )
        self.assertTrue(decision.ok)
        self.assertAlmostEqual(decision.improvement_pct, 0.0)
        self.assertAlmostEqual(decision.projected_avg, 100.0)

    def test_check_add_avg_improvement_zero_old_qty(self) -> None:
        from src.strategies.add_layer_gates import check_add_avg_improvement

        decision = check_add_avg_improvement(
            side="LONG",
            price=99.0,
            required_improvement_pct=0.0012,
            old_qty=0.0,
            old_notional=10_000.0,
            old_avg=100.0,
            add_qty=0.1,
        )
        self.assertFalse(decision.ok)
        self.assertAlmostEqual(decision.improvement_pct, 0.0)
        self.assertAlmostEqual(decision.projected_avg, 100.0)

    def test_check_add_avg_improvement_zero_old_notional(self) -> None:
        from src.strategies.add_layer_gates import check_add_avg_improvement

        decision = check_add_avg_improvement(
            side="LONG",
            price=99.0,
            required_improvement_pct=0.0012,
            old_qty=100.0,
            old_notional=0.0,
            old_avg=100.0,
            add_qty=0.1,
        )
        self.assertFalse(decision.ok)
        self.assertAlmostEqual(decision.improvement_pct, 0.0)
        self.assertAlmostEqual(decision.projected_avg, 100.0)

    def test_check_add_avg_improvement_zero_old_avg(self) -> None:
        from src.strategies.add_layer_gates import check_add_avg_improvement

        decision = check_add_avg_improvement(
            side="LONG",
            price=99.0,
            required_improvement_pct=0.0012,
            old_qty=100.0,
            old_notional=10_000.0,
            old_avg=0.0,
            add_qty=0.1,
        )
        self.assertFalse(decision.ok)
        self.assertAlmostEqual(decision.improvement_pct, 0.0)
        self.assertAlmostEqual(decision.projected_avg, 0.0)

    def test_check_add_avg_improvement_zero_add_qty(self) -> None:
        from src.strategies.add_layer_gates import check_add_avg_improvement

        decision = check_add_avg_improvement(
            side="LONG",
            price=99.0,
            required_improvement_pct=0.0012,
            old_qty=100.0,
            old_notional=10_000.0,
            old_avg=100.0,
            add_qty=0.0,
        )
        self.assertFalse(decision.ok)
        self.assertAlmostEqual(decision.improvement_pct, 0.0)
        self.assertAlmostEqual(decision.projected_avg, 100.0)

    def test_add_elapsed_seconds(self) -> None:
        from src.strategies.add_layer_gates import add_elapsed_seconds

        self.assertAlmostEqual(add_elapsed_seconds(ts_ms=1_500_000, last_order_ts_ms=1_000_000), 500.0)
        self.assertAlmostEqual(add_elapsed_seconds(ts_ms=500_000, last_order_ts_ms=1_000_000), 0.0)

    def test_adverse_gap_pct_long(self) -> None:
        from src.strategies.add_layer_gates import adverse_gap_pct

        self.assertAlmostEqual(adverse_gap_pct(side="LONG", price=99.0, last_entry_price=100.0), 0.01)
        self.assertAlmostEqual(adverse_gap_pct(side="LONG", price=101.0, last_entry_price=100.0), -0.01)

    def test_adverse_gap_pct_short(self) -> None:
        from src.strategies.add_layer_gates import adverse_gap_pct

        self.assertAlmostEqual(adverse_gap_pct(side="SHORT", price=101.0, last_entry_price=100.0), 0.01)
        self.assertAlmostEqual(adverse_gap_pct(side="SHORT", price=99.0, last_entry_price=100.0), -0.01)

    def test_adverse_gap_pct_missing_last_entry(self) -> None:
        from src.strategies.add_layer_gates import adverse_gap_pct

        self.assertAlmostEqual(adverse_gap_pct(side="LONG", price=99.0, last_entry_price=None), 0.0)
        self.assertAlmostEqual(adverse_gap_pct(side="LONG", price=99.0, last_entry_price=0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
