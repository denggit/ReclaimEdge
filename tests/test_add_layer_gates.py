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
    def test_add_gap_target_layer_2_to_6_uses_0_3_pct(self) -> None:
        strat = strategy()

        for target_layer in range(2, 7):
            with self.subTest(target_layer=target_layer):
                self.assertAlmostEqual(strat._add_layer_gap_pct_for_target_layer(target_layer), 0.003)

        strat.state = long_state(layers=5)
        gap_ok, gap_pct, required_price = strat._add_gap_passed("LONG", 99.70, 6)
        result = strat._maybe_open_or_add_long(99.70, NOW_MS, boll(), cvd())

        self.assertTrue(gap_ok)
        self.assertAlmostEqual(gap_pct, 0.003)
        self.assertAlmostEqual(required_price, 99.70)
        self.assertIsNotNone(result)
        self.assertEqual(result.intent_type, "ADD_LONG")
        self.assertEqual(result.layer_index, 6)
        self.assertIn("0.30%", result.reason)

    def test_add_gap_target_layer_7_to_8_uses_0_4_pct(self) -> None:
        strat = strategy()

        for target_layer in range(7, 9):
            with self.subTest(target_layer=target_layer):
                self.assertAlmostEqual(strat._add_layer_gap_pct_for_target_layer(target_layer), 0.004)

        strat.state = long_state(layers=6)
        gap_ok, gap_pct, required_price = strat._add_gap_passed("LONG", 99.70, 7)
        blocked = strat._maybe_open_or_add_long(99.70, NOW_MS, boll(), cvd())

        self.assertFalse(gap_ok)
        self.assertAlmostEqual(gap_pct, 0.004)
        self.assertAlmostEqual(required_price, 99.60)
        self.assertIsNone(blocked)

        strat.state = long_state(layers=6)
        allowed = strat._maybe_open_or_add_long(99.60, NOW_MS, boll(), cvd())

        self.assertIsNotNone(allowed)
        self.assertEqual(allowed.intent_type, "ADD_LONG")
        self.assertEqual(allowed.layer_index, 7)
        self.assertIn("0.40%", allowed.reason)

    def test_add_gap_target_layer_9_to_10_uses_0_6_pct(self) -> None:
        strat = strategy()

        for target_layer in range(9, 11):
            with self.subTest(target_layer=target_layer):
                self.assertAlmostEqual(strat._add_layer_gap_pct_for_target_layer(target_layer), 0.006)

        strat.state = long_state(layers=8)
        gap_ok, gap_pct, required_price = strat._add_gap_passed("LONG", 99.50, 9)
        blocked = strat._maybe_open_or_add_long(99.50, NOW_MS, boll(), cvd())

        self.assertFalse(gap_ok)
        self.assertAlmostEqual(gap_pct, 0.006)
        self.assertAlmostEqual(required_price, 99.40)
        self.assertIsNone(blocked)

        strat.state = long_state(layers=8)
        allowed = strat._maybe_open_or_add_long(99.40, NOW_MS, boll(), cvd())

        self.assertIsNotNone(allowed)
        self.assertEqual(allowed.intent_type, "ADD_LONG")
        self.assertEqual(allowed.layer_index, 9)
        self.assertIn("0.60%", allowed.reason)

    def test_add_gap_target_layer_11_plus_uses_0_8_pct(self) -> None:
        strat = strategy()

        for target_layer in (11, 12, 20):
            with self.subTest(target_layer=target_layer):
                self.assertAlmostEqual(strat._add_layer_gap_pct_for_target_layer(target_layer), 0.008)

        strat.state = long_state(layers=10)

        gap_ok, gap_pct, required_price = strat._add_gap_passed("LONG", 99.30, 11)
        blocked = strat._maybe_open_or_add_long(99.30, NOW_MS, boll(), cvd())

        self.assertFalse(gap_ok)
        self.assertAlmostEqual(gap_pct, 0.008)
        self.assertAlmostEqual(required_price, 99.20)
        self.assertIsNone(blocked)

        strat.state = long_state(layers=10)
        allowed = strat._maybe_open_or_add_long(99.20, NOW_MS, boll(), cvd())

        self.assertIsNotNone(allowed)
        self.assertEqual(allowed.intent_type, "ADD_LONG")
        self.assertEqual(allowed.layer_index, 11)
        self.assertIn("0.80%", allowed.reason)

    def test_short_add_gap_tiers_are_symmetric(self) -> None:
        strat = strategy()

        cases = [
            (1, 2, 100.20, 100.30, 0.003),
            (6, 7, 100.30, 100.40, 0.004),
            (8, 9, 100.50, 100.60, 0.006),
            (10, 11, 100.70, 100.80, 0.008),
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

        gap_ok, gap_pct, _ = strat._add_gap_passed("LONG", 99.39, 2)
        passed, reason = strat._add_timing_passed("LONG", 99.39, NOW_MS, 2)

        self.assertTrue(gap_ok)
        self.assertAlmostEqual(gap_pct, 0.003)
        self.assertTrue(passed)
        self.assertEqual(reason, "ok")

    def test_add_interval_target_layer_9_blocks_when_gap_below_dynamic_bypass(self) -> None:
        strat = strategy()
        strat.state = long_state(layers=8, last_order_ts_ms=NOW_MS - 5 * 60 * 1000)

        gap_ok, gap_pct, _ = strat._add_gap_passed("LONG", 99.30, 9)
        passed, reason = strat._add_timing_passed("LONG", 99.30, NOW_MS, 9)

        self.assertTrue(gap_ok)
        self.assertAlmostEqual(gap_pct, 0.006)
        self.assertFalse(passed)
        self.assertEqual(reason, "add_interval")

    def test_add_interval_target_layer_9_bypassed_when_gap_reaches_dynamic_bypass(self) -> None:
        strat = strategy()
        strat.state = long_state(layers=8, last_order_ts_ms=NOW_MS - 5 * 60 * 1000)

        gap_ok, gap_pct, _ = strat._add_gap_passed("LONG", 98.79, 9)
        passed, reason = strat._add_timing_passed("LONG", 98.79, NOW_MS, 9)

        self.assertTrue(gap_ok)
        self.assertAlmostEqual(gap_pct, 0.006)
        self.assertTrue(passed)
        self.assertEqual(reason, "ok")

    def test_add_interval_bypass_gap_uses_double_target_layer_gap(self) -> None:
        strat = strategy()

        self.assertAlmostEqual(strat._add_min_interval_bypass_gap_pct_for_target_layer(2), 0.006)
        self.assertAlmostEqual(strat._add_min_interval_bypass_gap_pct_for_target_layer(7), 0.008)
        self.assertAlmostEqual(strat._add_min_interval_bypass_gap_pct_for_target_layer(9), 0.012)
        self.assertAlmostEqual(strat._add_min_interval_bypass_gap_pct_for_target_layer(11), 0.016)

    def test_add_interval_bypassed_when_gap_reaches_target_layer_gap_times_2(self) -> None:
        strat = strategy()
        strat.state = long_state(layers=2, last_order_ts_ms=NOW_MS - 5 * 60 * 1000)

        result = strat._maybe_open_or_add_long(99.39, NOW_MS, boll(), cvd())

        self.assertIsNotNone(result)
        self.assertEqual(result.intent_type, "ADD_LONG")
        self.assertEqual(result.layer_index, 3)

    def test_legacy_interval_bypass_config_does_not_control_dynamic_bypass(self) -> None:
        strat = strategy(add_min_interval_bypass_gap_pct=0.003)
        strat.state = long_state(layers=10, last_order_ts_ms=NOW_MS - 5 * 60 * 1000)

        blocked = strat._maybe_open_or_add_long(99.20, NOW_MS, boll(), cvd())

        self.assertIsNone(blocked)
        self.assertEqual(strat.state.layers, 10)

        strat.state = long_state(layers=10, last_order_ts_ms=NOW_MS - 5 * 60 * 1000)
        allowed = strat._maybe_open_or_add_long(98.39, NOW_MS, boll(), cvd())

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

    def test_add_layer_gap_pct_layer_2_to_6_uses_base(self) -> None:
        from src.strategies.add_layer_gates import add_layer_gap_pct_for_target_layer

        for target_layer in range(2, 7):
            with self.subTest(target_layer=target_layer):
                result = add_layer_gap_pct_for_target_layer(
                    target_layer=target_layer,
                    add_layer_gap_pct=0.003,
                    add_layer_gap_pct_layer_7_8=0.004,
                    add_layer_gap_pct_layer_9_10=0.006,
                    add_layer_gap_pct_layer_11_plus=0.008,
                )
                self.assertAlmostEqual(result, 0.003)

    def test_add_layer_gap_pct_layer_7_to_8_uses_layer_7_8(self) -> None:
        from src.strategies.add_layer_gates import add_layer_gap_pct_for_target_layer

        for target_layer in range(7, 9):
            with self.subTest(target_layer=target_layer):
                result = add_layer_gap_pct_for_target_layer(
                    target_layer=target_layer,
                    add_layer_gap_pct=0.003,
                    add_layer_gap_pct_layer_7_8=0.004,
                    add_layer_gap_pct_layer_9_10=0.006,
                    add_layer_gap_pct_layer_11_plus=0.008,
                )
                self.assertAlmostEqual(result, 0.004)

    def test_add_layer_gap_pct_layer_9_to_10_uses_layer_9_10(self) -> None:
        from src.strategies.add_layer_gates import add_layer_gap_pct_for_target_layer

        for target_layer in range(9, 11):
            with self.subTest(target_layer=target_layer):
                result = add_layer_gap_pct_for_target_layer(
                    target_layer=target_layer,
                    add_layer_gap_pct=0.003,
                    add_layer_gap_pct_layer_7_8=0.004,
                    add_layer_gap_pct_layer_9_10=0.006,
                    add_layer_gap_pct_layer_11_plus=0.008,
                )
                self.assertAlmostEqual(result, 0.006)

    def test_add_layer_gap_pct_layer_11_plus_uses_layer_11_plus(self) -> None:
        from src.strategies.add_layer_gates import add_layer_gap_pct_for_target_layer

        for target_layer in (11, 12, 15, 20):
            with self.subTest(target_layer=target_layer):
                result = add_layer_gap_pct_for_target_layer(
                    target_layer=target_layer,
                    add_layer_gap_pct=0.003,
                    add_layer_gap_pct_layer_7_8=0.004,
                    add_layer_gap_pct_layer_9_10=0.006,
                    add_layer_gap_pct_layer_11_plus=0.008,
                )
                self.assertAlmostEqual(result, 0.008)

    def test_check_add_gap_long_passed(self) -> None:
        from src.strategies.add_layer_gates import check_add_gap

        decision = check_add_gap(
            side="LONG",
            price=99.70,
            last_entry_price=100.0,
            target_layer=2,
            add_layer_gap_pct=0.003,
            add_layer_gap_pct_layer_7_8=0.004,
            add_layer_gap_pct_layer_9_10=0.006,
            add_layer_gap_pct_layer_11_plus=0.008,
        )
        self.assertTrue(decision.ok)
        self.assertAlmostEqual(decision.gap_pct, 0.003)
        self.assertAlmostEqual(decision.required_price, 99.70)

    def test_check_add_gap_long_blocked(self) -> None:
        from src.strategies.add_layer_gates import check_add_gap

        decision = check_add_gap(
            side="LONG",
            price=99.80,
            last_entry_price=100.0,
            target_layer=2,
            add_layer_gap_pct=0.003,
            add_layer_gap_pct_layer_7_8=0.004,
            add_layer_gap_pct_layer_9_10=0.006,
            add_layer_gap_pct_layer_11_plus=0.008,
        )
        self.assertFalse(decision.ok)
        self.assertAlmostEqual(decision.gap_pct, 0.003)
        self.assertAlmostEqual(decision.required_price, 99.70)

    def test_check_add_gap_short_passed(self) -> None:
        from src.strategies.add_layer_gates import check_add_gap

        decision = check_add_gap(
            side="SHORT",
            price=100.30,
            last_entry_price=100.0,
            target_layer=2,
            add_layer_gap_pct=0.003,
            add_layer_gap_pct_layer_7_8=0.004,
            add_layer_gap_pct_layer_9_10=0.006,
            add_layer_gap_pct_layer_11_plus=0.008,
        )
        self.assertTrue(decision.ok)
        self.assertAlmostEqual(decision.gap_pct, 0.003)
        self.assertAlmostEqual(decision.required_price, 100.30)

    def test_check_add_gap_short_blocked(self) -> None:
        from src.strategies.add_layer_gates import check_add_gap

        decision = check_add_gap(
            side="SHORT",
            price=100.20,
            last_entry_price=100.0,
            target_layer=2,
            add_layer_gap_pct=0.003,
            add_layer_gap_pct_layer_7_8=0.004,
            add_layer_gap_pct_layer_9_10=0.006,
            add_layer_gap_pct_layer_11_plus=0.008,
        )
        self.assertFalse(decision.ok)
        self.assertAlmostEqual(decision.gap_pct, 0.003)
        self.assertAlmostEqual(decision.required_price, 100.30)

    def test_check_add_gap_missing_last_entry(self) -> None:
        from src.strategies.add_layer_gates import check_add_gap

        for bad_price in (None, 0.0, -1.0):
            with self.subTest(last_entry_price=bad_price):
                decision = check_add_gap(
                    side="LONG",
                    price=99.0,
                    last_entry_price=bad_price,
                    target_layer=2,
                    add_layer_gap_pct=0.003,
                    add_layer_gap_pct_layer_7_8=0.004,
                    add_layer_gap_pct_layer_9_10=0.006,
                    add_layer_gap_pct_layer_11_plus=0.008,
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
            add_layer_gap_pct=0.003,
            add_layer_gap_pct_layer_7_8=0.004,
            add_layer_gap_pct_layer_9_10=0.006,
            add_layer_gap_pct_layer_11_plus=0.008,
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
            add_layer_gap_pct=0.003,
            add_layer_gap_pct_layer_7_8=0.004,
            add_layer_gap_pct_layer_9_10=0.006,
            add_layer_gap_pct_layer_11_plus=0.008,
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
            add_layer_gap_pct=0.003,
            add_layer_gap_pct_layer_7_8=0.004,
            add_layer_gap_pct_layer_9_10=0.006,
            add_layer_gap_pct_layer_11_plus=0.008,
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
            add_layer_gap_pct=0.003,
            add_layer_gap_pct_layer_7_8=0.004,
            add_layer_gap_pct_layer_9_10=0.006,
            add_layer_gap_pct_layer_11_plus=0.008,
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
            add_layer_gap_pct=0.003,
            add_layer_gap_pct_layer_7_8=0.004,
            add_layer_gap_pct_layer_9_10=0.006,
            add_layer_gap_pct_layer_11_plus=0.008,
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
            add_layer_gap_pct=0.003,
            add_layer_gap_pct_layer_7_8=0.004,
            add_layer_gap_pct_layer_9_10=0.006,
            add_layer_gap_pct_layer_11_plus=0.008,
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
