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
    """Linear gap gate tests using strategy wrapper."""

    def test_linear_gap_L2_uses_0_3_pct(self) -> None:
        strat = strategy()
        self.assertAlmostEqual(strat._add_layer_gap_pct_for_target_layer(2), 0.003)

    def test_linear_gap_L3_uses_0_4_pct(self) -> None:
        strat = strategy()
        self.assertAlmostEqual(strat._add_layer_gap_pct_for_target_layer(3), 0.004)

    def test_linear_gap_L5_uses_0_6_pct(self) -> None:
        strat = strategy()
        self.assertAlmostEqual(strat._add_layer_gap_pct_for_target_layer(5), 0.006)

    def test_linear_gap_L7_uses_0_8_pct(self) -> None:
        strat = strategy()
        self.assertAlmostEqual(strat._add_layer_gap_pct_for_target_layer(7), 0.008)

    def test_linear_gap_L11_uses_1_2_pct(self) -> None:
        strat = strategy()
        self.assertAlmostEqual(strat._add_layer_gap_pct_for_target_layer(11), 0.012)

    def test_linear_gap_L20_uses_2_1_pct(self) -> None:
        strat = strategy()
        self.assertAlmostEqual(strat._add_layer_gap_pct_for_target_layer(20), 0.021)

    def test_linear_gap_L1_uses_base(self) -> None:
        strat = strategy()
        self.assertAlmostEqual(strat._add_layer_gap_pct_for_target_layer(1), 0.003)

    def test_long_add_L2_allowed_at_0_3pct_gap(self) -> None:
        """L2 gap=0.3%, last_entry=100, required=99.70 → allowed at 99.70."""
        strat = strategy()
        strat.state = long_state(layers=1)
        gap_ok, gap_pct, required_price = strat._add_gap_passed("LONG", 99.70, 2)
        result = strat._maybe_open_or_add_long(99.70, NOW_MS, boll(), cvd())

        self.assertTrue(gap_ok)
        self.assertAlmostEqual(gap_pct, 0.003)
        self.assertAlmostEqual(required_price, 99.70)
        self.assertIsNotNone(result)
        self.assertEqual(result.intent_type, "ADD_LONG")
        self.assertEqual(result.layer_index, 2)
        self.assertIn("0.30%", result.reason)

    def test_long_add_L3_blocked_at_0_3pct_allowed_at_0_4pct(self) -> None:
        """L3 gap=0.4%, last_entry=100, required=99.60."""
        strat = strategy()
        strat.state = long_state(layers=2)
        gap_ok, gap_pct, required_price = strat._add_gap_passed("LONG", 99.70, 3)
        blocked = strat._maybe_open_or_add_long(99.70, NOW_MS, boll(), cvd())

        self.assertFalse(gap_ok)
        self.assertAlmostEqual(gap_pct, 0.004)
        self.assertAlmostEqual(required_price, 99.60)
        self.assertIsNone(blocked)

        strat.state = long_state(layers=2)
        allowed = strat._maybe_open_or_add_long(99.60, NOW_MS, boll(), cvd())

        self.assertIsNotNone(allowed)
        self.assertEqual(allowed.intent_type, "ADD_LONG")
        self.assertEqual(allowed.layer_index, 3)
        self.assertIn("0.40%", allowed.reason)

    def test_long_add_L5_blocked_at_0_5pct_allowed_at_0_6pct(self) -> None:
        """L5 gap=0.6%, last_entry=100, required=99.40."""
        strat = strategy()
        strat.state = long_state(layers=4)

        gap_ok, gap_pct, required_price = strat._add_gap_passed("LONG", 99.50, 5)
        blocked = strat._maybe_open_or_add_long(99.50, NOW_MS, boll(), cvd())

        self.assertFalse(gap_ok)
        self.assertAlmostEqual(gap_pct, 0.006)
        self.assertAlmostEqual(required_price, 99.40)
        self.assertIsNone(blocked)

        strat.state = long_state(layers=4)
        allowed = strat._maybe_open_or_add_long(99.40, NOW_MS, boll(), cvd())

        self.assertIsNotNone(allowed)
        self.assertEqual(allowed.intent_type, "ADD_LONG")
        self.assertEqual(allowed.layer_index, 5)
        self.assertIn("0.60%", allowed.reason)

    def test_long_add_L7_allowed_at_0_8pct(self) -> None:
        """L7 gap=0.8%, last_entry=100, required=99.20."""
        strat = strategy()
        strat.state = long_state(layers=6)

        gap_ok, gap_pct, required_price = strat._add_gap_passed("LONG", 99.30, 7)
        blocked = strat._maybe_open_or_add_long(99.30, NOW_MS, boll(), cvd())

        self.assertFalse(gap_ok)
        self.assertAlmostEqual(gap_pct, 0.008)
        self.assertAlmostEqual(required_price, 99.20)
        self.assertIsNone(blocked)

        strat.state = long_state(layers=6)
        allowed = strat._maybe_open_or_add_long(99.20, NOW_MS, boll(), cvd())

        self.assertIsNotNone(allowed)
        self.assertEqual(allowed.intent_type, "ADD_LONG")
        self.assertEqual(allowed.layer_index, 7)
        self.assertIn("0.80%", allowed.reason)

    def test_short_add_linear_symmetric(self) -> None:
        strat = strategy()

        cases = [
            (1, 2, 100.20, 100.30, 0.003),
            (2, 3, 100.30, 100.40, 0.004),
            (4, 5, 100.50, 100.60, 0.006),
            (6, 7, 100.70, 100.80, 0.008),
            (10, 11, 101.10, 101.20, 0.012),
        ]
        for layers, target_layer, blocked_price, allowed_price, expected_gap in cases:
            with self.subTest(target_layer=target_layer):
                strat.state = short_state(layers=layers)
                gap_ok, gap_pct, required_price = strat._add_gap_passed("SHORT", blocked_price, target_layer)
                blocked = strat._maybe_open_or_add_short(blocked_price, NOW_MS, boll(), cvd())

                self.assertFalse(gap_ok)
                self.assertAlmostEqual(gap_pct, expected_gap)
                self.assertAlmostEqual(required_price, allowed_price)
                self.assertIsNone(blocked)

                strat.state = short_state(layers=layers)
                allowed = strat._maybe_open_or_add_short(allowed_price, NOW_MS, boll(), cvd())

                self.assertIsNotNone(allowed)
                self.assertEqual(allowed.intent_type, "ADD_SHORT")
                self.assertEqual(allowed.layer_index, target_layer)
                self.assertIn(f"{expected_gap * 100:.2f}%", allowed.reason)

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

        short_strat = strategy()
        short_strat.state = short_state(layers=1, last_order_ts_ms=NOW_MS - 29 * 60 * 1000)
        short_result = short_strat._maybe_open_or_add_short(101.0, NOW_MS, boll(), cvd())
        self.assertIsNone(short_result)

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

    def test_add_interval_L2_blocks_when_gap_below_dynamic_bypass(self) -> None:
        strat = strategy()
        strat.state = long_state(layers=2, last_order_ts_ms=NOW_MS - 5 * 60 * 1000)
        gap_ok, gap_pct, _ = strat._add_gap_passed("LONG", 99.50, 2)
        passed, reason = strat._add_timing_passed("LONG", 99.50, NOW_MS, 2)
        self.assertTrue(gap_ok)
        self.assertAlmostEqual(gap_pct, 0.003)
        self.assertFalse(passed)
        self.assertEqual(reason, "add_interval")

    def test_add_interval_L2_bypassed_when_gap_reaches_dynamic_bypass(self) -> None:
        strat = strategy()
        strat.state = long_state(layers=2, last_order_ts_ms=NOW_MS - 5 * 60 * 1000)
        # L2 bypass = 0.003 * 2 = 0.006. price=99.39 → gap=0.0061 >= 0.006
        gap_ok, gap_pct, _ = strat._add_gap_passed("LONG", 99.39, 2)
        passed, reason = strat._add_timing_passed("LONG", 99.39, NOW_MS, 2)
        self.assertTrue(gap_ok)
        self.assertAlmostEqual(gap_pct, 0.003)
        self.assertTrue(passed)
        self.assertEqual(reason, "ok")

    def test_add_interval_L5_blocked_when_gap_below_dynamic_bypass(self) -> None:
        """L5 bypass = 0.006 * 2 = 0.012. price=99.0 → gap=1% < 1.2% → blocked."""
        strat = strategy()
        strat.state = long_state(layers=4, last_order_ts_ms=NOW_MS - 5 * 60 * 1000)
        gap_ok, gap_pct, _ = strat._add_gap_passed("LONG", 99.0, 5)
        passed, reason = strat._add_timing_passed("LONG", 99.0, NOW_MS, 5)
        self.assertTrue(gap_ok)
        self.assertAlmostEqual(gap_pct, 0.006)
        self.assertFalse(passed)
        self.assertEqual(reason, "add_interval")

    def test_add_interval_L5_bypassed_when_gap_reaches_dynamic_bypass(self) -> None:
        """L5 bypass = 0.012. price=98.79 → gap=0.0121 >= 0.012 → allowed."""
        strat = strategy()
        strat.state = long_state(layers=4, last_order_ts_ms=NOW_MS - 5 * 60 * 1000)
        gap_ok, gap_pct, _ = strat._add_gap_passed("LONG", 98.79, 5)
        passed, reason = strat._add_timing_passed("LONG", 98.79, NOW_MS, 5)
        self.assertTrue(gap_ok)
        self.assertAlmostEqual(gap_pct, 0.006)
        self.assertTrue(passed)
        self.assertEqual(reason, "ok")

    def test_add_interval_bypass_uses_double_linear_gap(self) -> None:
        strat = strategy()
        self.assertAlmostEqual(strat._add_min_interval_bypass_gap_pct_for_target_layer(2), 0.006)
        self.assertAlmostEqual(strat._add_min_interval_bypass_gap_pct_for_target_layer(3), 0.008)
        self.assertAlmostEqual(strat._add_min_interval_bypass_gap_pct_for_target_layer(7), 0.016)
        self.assertAlmostEqual(strat._add_min_interval_bypass_gap_pct_for_target_layer(11), 0.024)

    def test_add_interval_bypassed_when_gap_reaches_linear_gap_times_2(self) -> None:
        strat = strategy()
        strat.state = long_state(layers=2, last_order_ts_ms=NOW_MS - 5 * 60 * 1000)
        # L3 bypass = 0.004 * 2 = 0.008. price=99.19 → gap=0.0081 >= 0.008
        result = strat._maybe_open_or_add_long(99.19, NOW_MS, boll(), cvd())
        self.assertIsNotNone(result)
        self.assertEqual(result.intent_type, "ADD_LONG")
        self.assertEqual(result.layer_index, 3)

    def test_legacy_interval_bypass_config_does_not_control_dynamic_bypass(self) -> None:
        strat = strategy(add_min_interval_bypass_gap_pct=0.003)
        strat.state = long_state(layers=10, last_order_ts_ms=NOW_MS - 5 * 60 * 1000)
        # L11 linear gap = 0.012, bypass = 0.024. price=99.20 → gap=0.008 < 0.024 → blocked
        blocked = strat._maybe_open_or_add_long(99.20, NOW_MS, boll(), cvd())
        self.assertIsNone(blocked)

        strat.state = long_state(layers=10, last_order_ts_ms=NOW_MS - 5 * 60 * 1000)
        # price=97.59 → gap=0.0241 >= 0.024 → allowed
        allowed = strat._maybe_open_or_add_long(97.59, NOW_MS, boll(), cvd())
        self.assertIsNotNone(allowed)
        self.assertEqual(allowed.intent_type, "ADD_LONG")
        self.assertEqual(allowed.layer_index, 11)

    def test_long_add_blocked_when_avg_improvement_below_0_12_pct(self) -> None:
        strat = strategy()
        strat.state = long_state(
            layers=1, total_entry_qty=100.0, total_entry_notional=10_000.0, avg_entry_price=100.0,
        )
        passed, improvement_pct, projected_avg = strat._add_avg_improvement_passed("LONG", 99.70, 2)
        result = strat._maybe_open_or_add_long(99.70, NOW_MS, boll(), cvd())
        self.assertFalse(passed)
        self.assertLess(improvement_pct, 0.0012)
        self.assertGreater(projected_avg, 99.88)
        self.assertIsNone(result)

    def test_long_add_allowed_when_avg_improvement_meets_0_12_pct(self) -> None:
        strat = strategy()
        strat.state = long_state(
            layers=1, total_entry_qty=100.0, total_entry_notional=10_000.0, avg_entry_price=100.0,
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
            layers=1, total_entry_qty=100.0, total_entry_notional=10_000.0, avg_entry_price=100.0,
        )
        passed, improvement_pct, projected_avg = strat._add_avg_improvement_passed("SHORT", 100.30, 2)
        result = strat._maybe_open_or_add_short(100.30, NOW_MS, boll(), cvd())
        self.assertFalse(passed)
        self.assertLess(improvement_pct, 0.0012)
        self.assertLess(projected_avg, 100.12)
        self.assertIsNone(result)

    def test_short_add_allowed_when_avg_improvement_meets_0_12_pct(self) -> None:
        strat = strategy()
        strat.state = short_state(
            layers=1, total_entry_qty=100.0, total_entry_notional=10_000.0, avg_entry_price=100.0,
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
            layers=2, last_order_ts_ms=NOW_MS - 60 * 60 * 1000,
            total_entry_qty=100.0, total_entry_notional=10_000.0, near_tp_add_disabled=True,
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
            layers=2, last_order_ts_ms=NOW_MS - 60 * 60 * 1000,
            total_entry_qty=100.0, total_entry_notional=10_000.0,
            middle_runner_active=True, middle_runner_add_disabled=True,
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
            layers=2, last_order_ts_ms=NOW_MS - 60 * 60 * 1000,
            total_entry_qty=100.0, total_entry_notional=10_000.0,
            middle_runner_active=True, middle_runner_add_disabled=True,
        )
        with self.assertLogs("src.strategies.boll_cvd_reclaim_strategy", level="INFO") as logs:
            result = strat._maybe_open_or_add_short(100.30, NOW_MS, boll(), cvd())
        self.assertIsNone(result)
        self.assertIn("reason=middle_runner_active", "\n".join(logs.output))


class AddLayerGatesPureTest(unittest.TestCase):
    """Direct tests for add_layer_gates linear gap functions."""

    def test_linear_gap_L2(self) -> None:
        from src.strategies.add_layer_gates import add_layer_gap_pct_for_target_layer
        result = add_layer_gap_pct_for_target_layer(
            target_layer=2, add_gap_mode="linear", add_gap_base_pct=0.003, add_gap_step_pct=0.001,
        )
        self.assertAlmostEqual(result, 0.003)

    def test_linear_gap_L3(self) -> None:
        from src.strategies.add_layer_gates import add_layer_gap_pct_for_target_layer
        result = add_layer_gap_pct_for_target_layer(
            target_layer=3, add_gap_mode="linear", add_gap_base_pct=0.003, add_gap_step_pct=0.001,
        )
        self.assertAlmostEqual(result, 0.004)

    def test_linear_gap_L4(self) -> None:
        from src.strategies.add_layer_gates import add_layer_gap_pct_for_target_layer
        result = add_layer_gap_pct_for_target_layer(
            target_layer=4, add_gap_mode="linear", add_gap_base_pct=0.003, add_gap_step_pct=0.001,
        )
        self.assertAlmostEqual(result, 0.005)

    def test_linear_gap_L5(self) -> None:
        from src.strategies.add_layer_gates import add_layer_gap_pct_for_target_layer
        result = add_layer_gap_pct_for_target_layer(
            target_layer=5, add_gap_mode="linear", add_gap_base_pct=0.003, add_gap_step_pct=0.001,
        )
        self.assertAlmostEqual(result, 0.006)

    def test_linear_gap_L6(self) -> None:
        from src.strategies.add_layer_gates import add_layer_gap_pct_for_target_layer
        result = add_layer_gap_pct_for_target_layer(
            target_layer=6, add_gap_mode="linear", add_gap_base_pct=0.003, add_gap_step_pct=0.001,
        )
        self.assertAlmostEqual(result, 0.007)

    def test_linear_gap_L7(self) -> None:
        from src.strategies.add_layer_gates import add_layer_gap_pct_for_target_layer
        result = add_layer_gap_pct_for_target_layer(
            target_layer=7, add_gap_mode="linear", add_gap_base_pct=0.003, add_gap_step_pct=0.001,
        )
        self.assertAlmostEqual(result, 0.008)

    def test_linear_gap_L11(self) -> None:
        from src.strategies.add_layer_gates import add_layer_gap_pct_for_target_layer
        result = add_layer_gap_pct_for_target_layer(
            target_layer=11, add_gap_mode="linear", add_gap_base_pct=0.003, add_gap_step_pct=0.001,
        )
        self.assertAlmostEqual(result, 0.012)

    def test_linear_gap_L20(self) -> None:
        from src.strategies.add_layer_gates import add_layer_gap_pct_for_target_layer
        result = add_layer_gap_pct_for_target_layer(
            target_layer=20, add_gap_mode="linear", add_gap_base_pct=0.003, add_gap_step_pct=0.001,
        )
        self.assertAlmostEqual(result, 0.021)

    def test_linear_gap_L1_uses_base(self) -> None:
        from src.strategies.add_layer_gates import add_layer_gap_pct_for_target_layer
        result = add_layer_gap_pct_for_target_layer(
            target_layer=1, add_gap_mode="linear", add_gap_base_pct=0.003, add_gap_step_pct=0.001,
        )
        self.assertAlmostEqual(result, 0.003)

    def test_unsupported_mode_raises(self) -> None:
        from src.strategies.add_layer_gates import add_layer_gap_pct_for_target_layer
        with self.assertRaises(RuntimeError) as ctx:
            add_layer_gap_pct_for_target_layer(
                target_layer=2, add_gap_mode="tier", add_gap_base_pct=0.003, add_gap_step_pct=0.001,
            )
        self.assertIn("tier", str(ctx.exception))

    def test_bypass_L2_is_double_base(self) -> None:
        from src.strategies.add_layer_gates import add_min_interval_bypass_gap_pct_for_target_layer
        result = add_min_interval_bypass_gap_pct_for_target_layer(
            target_layer=2, add_gap_mode="linear", add_gap_base_pct=0.003, add_gap_step_pct=0.001,
        )
        self.assertAlmostEqual(result, 0.006)

    def test_bypass_L3_is_double_linear(self) -> None:
        from src.strategies.add_layer_gates import add_min_interval_bypass_gap_pct_for_target_layer
        result = add_min_interval_bypass_gap_pct_for_target_layer(
            target_layer=3, add_gap_mode="linear", add_gap_base_pct=0.003, add_gap_step_pct=0.001,
        )
        self.assertAlmostEqual(result, 0.008)

    def test_bypass_L7_is_double_linear(self) -> None:
        from src.strategies.add_layer_gates import add_min_interval_bypass_gap_pct_for_target_layer
        result = add_min_interval_bypass_gap_pct_for_target_layer(
            target_layer=7, add_gap_mode="linear", add_gap_base_pct=0.003, add_gap_step_pct=0.001,
        )
        self.assertAlmostEqual(result, 0.016)

    def test_bypass_L11_is_double_linear(self) -> None:
        from src.strategies.add_layer_gates import add_min_interval_bypass_gap_pct_for_target_layer
        result = add_min_interval_bypass_gap_pct_for_target_layer(
            target_layer=11, add_gap_mode="linear", add_gap_base_pct=0.003, add_gap_step_pct=0.001,
        )
        self.assertAlmostEqual(result, 0.024)

    def test_check_add_gap_long_passed(self) -> None:
        from src.strategies.add_layer_gates import check_add_gap
        decision = check_add_gap(
            side="LONG", price=99.70, last_entry_price=100.0, target_layer=2,
            add_gap_mode="linear", add_gap_base_pct=0.003, add_gap_step_pct=0.001,
        )
        self.assertTrue(decision.ok)
        self.assertAlmostEqual(decision.gap_pct, 0.003)
        self.assertAlmostEqual(decision.required_price, 99.70)

    def test_check_add_gap_long_blocked(self) -> None:
        from src.strategies.add_layer_gates import check_add_gap
        decision = check_add_gap(
            side="LONG", price=99.80, last_entry_price=100.0, target_layer=2,
            add_gap_mode="linear", add_gap_base_pct=0.003, add_gap_step_pct=0.001,
        )
        self.assertFalse(decision.ok)
        self.assertAlmostEqual(decision.gap_pct, 0.003)
        self.assertAlmostEqual(decision.required_price, 99.70)

    def test_check_add_gap_short_passed(self) -> None:
        from src.strategies.add_layer_gates import check_add_gap
        decision = check_add_gap(
            side="SHORT", price=100.30, last_entry_price=100.0, target_layer=2,
            add_gap_mode="linear", add_gap_base_pct=0.003, add_gap_step_pct=0.001,
        )
        self.assertTrue(decision.ok)
        self.assertAlmostEqual(decision.gap_pct, 0.003)
        self.assertAlmostEqual(decision.required_price, 100.30)

    def test_check_add_gap_short_blocked(self) -> None:
        from src.strategies.add_layer_gates import check_add_gap
        decision = check_add_gap(
            side="SHORT", price=100.20, last_entry_price=100.0, target_layer=2,
            add_gap_mode="linear", add_gap_base_pct=0.003, add_gap_step_pct=0.001,
        )
        self.assertFalse(decision.ok)
        self.assertAlmostEqual(decision.gap_pct, 0.003)
        self.assertAlmostEqual(decision.required_price, 100.30)

    def test_check_add_gap_missing_last_entry(self) -> None:
        from src.strategies.add_layer_gates import check_add_gap
        for bad_price in (None, 0.0, -1.0):
            with self.subTest(last_entry_price=bad_price):
                decision = check_add_gap(
                    side="LONG", price=99.0, last_entry_price=bad_price, target_layer=2,
                    add_gap_mode="linear", add_gap_base_pct=0.003, add_gap_step_pct=0.001,
                )
                self.assertFalse(decision.ok)
                self.assertAlmostEqual(decision.gap_pct, 0.003)
                self.assertAlmostEqual(decision.required_price, 0.0)

    def test_check_base_add_timing_missing_last_entry(self) -> None:
        from src.strategies.add_layer_gates import check_base_add_timing
        decision = check_base_add_timing(
            side="LONG", price=99.0, ts_ms=1_000_000, target_layer=2, layers=1,
            last_entry_price=None, last_order_ts_ms=0,
            first_add_block_seconds=1800, add_min_interval_seconds=600,
            add_gap_mode="linear", add_gap_base_pct=0.003, add_gap_step_pct=0.001,
        )
        self.assertFalse(decision.ok)
        self.assertEqual(decision.reason, "missing_last_entry")

    def test_check_base_add_timing_first_add_blocked(self) -> None:
        from src.strategies.add_layer_gates import check_base_add_timing
        decision = check_base_add_timing(
            side="LONG", price=99.0, ts_ms=1_000_000, target_layer=2, layers=1,
            last_entry_price=100.0, last_order_ts_ms=900_000,
            first_add_block_seconds=1800, add_min_interval_seconds=600,
            add_gap_mode="linear", add_gap_base_pct=0.003, add_gap_step_pct=0.001,
        )
        self.assertFalse(decision.ok)
        self.assertEqual(decision.reason, "first_add_block")

    def test_check_base_add_timing_first_add_allowed(self) -> None:
        from src.strategies.add_layer_gates import check_base_add_timing
        decision = check_base_add_timing(
            side="LONG", price=99.0, ts_ms=3_000_000, target_layer=2, layers=1,
            last_entry_price=100.0, last_order_ts_ms=1_000_000,
            first_add_block_seconds=1800, add_min_interval_seconds=600,
            add_gap_mode="linear", add_gap_base_pct=0.003, add_gap_step_pct=0.001,
        )
        self.assertTrue(decision.ok)
        self.assertEqual(decision.reason, "ok")

    def test_check_base_add_timing_add_interval_blocked(self) -> None:
        from src.strategies.add_layer_gates import check_base_add_timing
        decision = check_base_add_timing(
            side="LONG", price=99.70, ts_ms=1_500_000, target_layer=2, layers=2,
            last_entry_price=100.0, last_order_ts_ms=1_000_000,
            first_add_block_seconds=1800, add_min_interval_seconds=600,
            add_gap_mode="linear", add_gap_base_pct=0.003, add_gap_step_pct=0.001,
        )
        self.assertFalse(decision.ok)
        self.assertEqual(decision.reason, "add_interval")

    def test_check_base_add_timing_add_interval_bypassed(self) -> None:
        from src.strategies.add_layer_gates import check_base_add_timing
        decision = check_base_add_timing(
            side="LONG", price=99.39, ts_ms=1_500_000, target_layer=2, layers=2,
            last_entry_price=100.0, last_order_ts_ms=1_000_000,
            first_add_block_seconds=1800, add_min_interval_seconds=600,
            add_gap_mode="linear", add_gap_base_pct=0.003, add_gap_step_pct=0.001,
        )
        self.assertTrue(decision.ok)
        self.assertEqual(decision.reason, "ok")

    def test_check_base_add_timing_elapsed_passes_interval(self) -> None:
        from src.strategies.add_layer_gates import check_base_add_timing
        decision = check_base_add_timing(
            side="LONG", price=99.70, ts_ms=2_000_000, target_layer=2, layers=2,
            last_entry_price=100.0, last_order_ts_ms=1_000_000,
            first_add_block_seconds=1800, add_min_interval_seconds=600,
            add_gap_mode="linear", add_gap_base_pct=0.003, add_gap_step_pct=0.001,
        )
        self.assertTrue(decision.ok)
        self.assertEqual(decision.reason, "ok")

    def test_check_add_avg_improvement_long_passed(self) -> None:
        from src.strategies.add_layer_gates import check_add_avg_improvement
        decision = check_add_avg_improvement(
            side="LONG", price=99.0, required_improvement_pct=0.0012,
            old_qty=1.0, old_notional=100.0, old_avg=100.0, add_qty=1.0,
        )
        self.assertTrue(decision.ok)
        self.assertGreaterEqual(decision.improvement_pct, 0.0012)
        self.assertLess(decision.projected_avg, 100.0)

    def test_check_add_avg_improvement_long_blocked(self) -> None:
        from src.strategies.add_layer_gates import check_add_avg_improvement
        decision = check_add_avg_improvement(
            side="LONG", price=99.70, required_improvement_pct=0.0012,
            old_qty=100.0, old_notional=10_000.0, old_avg=100.0, add_qty=0.1,
        )
        self.assertFalse(decision.ok)
        self.assertLess(decision.improvement_pct, 0.0012)

    def test_check_add_avg_improvement_short_passed(self) -> None:
        from src.strategies.add_layer_gates import check_add_avg_improvement
        decision = check_add_avg_improvement(
            side="SHORT", price=101.0, required_improvement_pct=0.0012,
            old_qty=1.0, old_notional=100.0, old_avg=100.0, add_qty=1.0,
        )
        self.assertTrue(decision.ok)
        self.assertGreaterEqual(decision.improvement_pct, 0.0012)
        self.assertGreater(decision.projected_avg, 100.0)

    def test_check_add_avg_improvement_short_blocked(self) -> None:
        from src.strategies.add_layer_gates import check_add_avg_improvement
        decision = check_add_avg_improvement(
            side="SHORT", price=100.30, required_improvement_pct=0.0012,
            old_qty=100.0, old_notional=10_000.0, old_avg=100.0, add_qty=0.1,
        )
        self.assertFalse(decision.ok)
        self.assertLess(decision.improvement_pct, 0.0012)

    def test_check_add_avg_improvement_zero_required(self) -> None:
        from src.strategies.add_layer_gates import check_add_avg_improvement
        decision = check_add_avg_improvement(
            side="LONG", price=100.0, required_improvement_pct=0.0,
            old_qty=1.0, old_notional=100.0, old_avg=100.0, add_qty=0.0,
        )
        self.assertTrue(decision.ok)
        self.assertAlmostEqual(decision.improvement_pct, 0.0)
        self.assertAlmostEqual(decision.projected_avg, 100.0)

    def test_check_add_avg_improvement_zero_old_qty(self) -> None:
        from src.strategies.add_layer_gates import check_add_avg_improvement
        decision = check_add_avg_improvement(
            side="LONG", price=99.0, required_improvement_pct=0.0012,
            old_qty=0.0, old_notional=10_000.0, old_avg=100.0, add_qty=0.1,
        )
        self.assertFalse(decision.ok)
        self.assertAlmostEqual(decision.improvement_pct, 0.0)
        self.assertAlmostEqual(decision.projected_avg, 100.0)

    def test_check_add_avg_improvement_zero_old_notional(self) -> None:
        from src.strategies.add_layer_gates import check_add_avg_improvement
        decision = check_add_avg_improvement(
            side="LONG", price=99.0, required_improvement_pct=0.0012,
            old_qty=100.0, old_notional=0.0, old_avg=100.0, add_qty=0.1,
        )
        self.assertFalse(decision.ok)
        self.assertAlmostEqual(decision.improvement_pct, 0.0)
        self.assertAlmostEqual(decision.projected_avg, 100.0)

    def test_check_add_avg_improvement_zero_old_avg(self) -> None:
        from src.strategies.add_layer_gates import check_add_avg_improvement
        decision = check_add_avg_improvement(
            side="LONG", price=99.0, required_improvement_pct=0.0012,
            old_qty=100.0, old_notional=10_000.0, old_avg=0.0, add_qty=0.1,
        )
        self.assertFalse(decision.ok)
        self.assertAlmostEqual(decision.improvement_pct, 0.0)
        self.assertAlmostEqual(decision.projected_avg, 0.0)

    def test_check_add_avg_improvement_zero_add_qty(self) -> None:
        from src.strategies.add_layer_gates import check_add_avg_improvement
        decision = check_add_avg_improvement(
            side="LONG", price=99.0, required_improvement_pct=0.0012,
            old_qty=100.0, old_notional=10_000.0, old_avg=100.0, add_qty=0.0,
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


class ConfigValidationTest(unittest.TestCase):
    """Tests for BollCvdReclaimStrategyConfig validation."""

    def test_unsupported_gap_mode_raises(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            BollCvdReclaimStrategyConfig(add_gap_mode="tier")
        self.assertIn("tier", str(ctx.exception))

    def test_zero_base_pct_raises(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            BollCvdReclaimStrategyConfig(add_gap_base_pct=0)
        self.assertIn("ADD_GAP_BASE_PCT", str(ctx.exception))

    def test_negative_step_pct_raises(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            BollCvdReclaimStrategyConfig(add_gap_step_pct=-0.001)
        self.assertIn("ADD_GAP_STEP_PCT", str(ctx.exception))

    def test_valid_linear_config_does_not_raise(self) -> None:
        cfg = BollCvdReclaimStrategyConfig(
            add_gap_mode="linear", add_gap_base_pct=0.003, add_gap_step_pct=0.001,
        )
        self.assertEqual(cfg.add_gap_mode, "linear")
        self.assertAlmostEqual(cfg.add_gap_base_pct, 0.003)
        self.assertAlmostEqual(cfg.add_gap_step_pct, 0.001)


if __name__ == "__main__":
    unittest.main()
