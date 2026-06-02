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
    def test_add_gap_target_layer_2_to_8_uses_0_3_pct(self) -> None:
        strat = strategy()
        strat.state = long_state(layers=7)

        gap_ok, gap_pct, required_price = strat._add_gap_passed("LONG", 99.70, 8)
        result = strat._maybe_open_or_add_long(99.70, NOW_MS, boll(), cvd())

        self.assertTrue(gap_ok)
        self.assertAlmostEqual(gap_pct, 0.003)
        self.assertAlmostEqual(required_price, 99.70)
        self.assertIsNotNone(result)
        self.assertEqual(result.intent_type, "ADD_LONG")
        self.assertEqual(result.layer_index, 8)
        self.assertIn("0.30%", result.reason)

    def test_add_gap_target_layer_9_uses_0_4_pct(self) -> None:
        strat = strategy()
        strat.state = long_state(layers=8)

        gap_ok, gap_pct, required_price = strat._add_gap_passed("LONG", 99.70, 9)
        blocked = strat._maybe_open_or_add_long(99.70, NOW_MS, boll(), cvd())

        self.assertFalse(gap_ok)
        self.assertAlmostEqual(gap_pct, 0.004)
        self.assertAlmostEqual(required_price, 99.60)
        self.assertIsNone(blocked)

        strat.state = long_state(layers=8)
        allowed = strat._maybe_open_or_add_long(99.60, NOW_MS, boll(), cvd())

        self.assertIsNotNone(allowed)
        self.assertEqual(allowed.intent_type, "ADD_LONG")
        self.assertEqual(allowed.layer_index, 9)
        self.assertIn("0.40%", allowed.reason)

    def test_add_gap_target_layer_11_uses_0_5_pct(self) -> None:
        strat = strategy()
        strat.state = long_state(layers=10)

        gap_ok, gap_pct, required_price = strat._add_gap_passed("LONG", 99.60, 11)
        blocked = strat._maybe_open_or_add_long(99.60, NOW_MS, boll(), cvd())

        self.assertFalse(gap_ok)
        self.assertAlmostEqual(gap_pct, 0.005)
        self.assertAlmostEqual(required_price, 99.50)
        self.assertIsNone(blocked)

        strat.state = long_state(layers=10)
        allowed = strat._maybe_open_or_add_long(99.50, NOW_MS, boll(), cvd())

        self.assertIsNotNone(allowed)
        self.assertEqual(allowed.intent_type, "ADD_LONG")
        self.assertEqual(allowed.layer_index, 11)
        self.assertIn("0.50%", allowed.reason)

    def test_short_add_gap_tiers_are_symmetric(self) -> None:
        strat = strategy()
        strat.state = short_state(layers=8)

        gap_ok, gap_pct, required_price = strat._add_gap_passed("SHORT", 100.30, 9)
        blocked = strat._maybe_open_or_add_short(100.30, NOW_MS, boll(), cvd())

        self.assertFalse(gap_ok)
        self.assertAlmostEqual(gap_pct, 0.004)
        self.assertAlmostEqual(required_price, 100.40)
        self.assertIsNone(blocked)

        strat.state = short_state(layers=8)
        allowed = strat._maybe_open_or_add_short(100.40, NOW_MS, boll(), cvd())

        self.assertIsNotNone(allowed)
        self.assertEqual(allowed.intent_type, "ADD_SHORT")
        self.assertEqual(allowed.layer_index, 9)
        self.assertIn("0.40%", allowed.reason)

    def test_first_add_block_prevents_add_within_30_minutes(self) -> None:
        strat = strategy()
        strat.state = long_state(layers=1, last_order_ts_ms=NOW_MS - 29 * 60 * 1000)

        result = strat._maybe_open_or_add_long(99.0, NOW_MS, boll(), cvd())

        self.assertIsNone(result)
        self.assertEqual(strat.state.layers, 1)

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

    def test_add_interval_bypassed_when_gap_ge_0_5_pct(self) -> None:
        strat = strategy()
        strat.state = long_state(layers=2, last_order_ts_ms=NOW_MS - 5 * 60 * 1000)

        result = strat._maybe_open_or_add_long(99.50, NOW_MS, boll(), cvd())

        self.assertIsNotNone(result)
        self.assertEqual(result.intent_type, "ADD_LONG")
        self.assertEqual(result.layer_index, 3)

    def test_interval_bypass_does_not_skip_tier_gap(self) -> None:
        strat = strategy(add_min_interval_bypass_gap_pct=0.003)
        strat.state = long_state(layers=10, last_order_ts_ms=NOW_MS - 5 * 60 * 1000)

        blocked = strat._maybe_open_or_add_long(99.60, NOW_MS, boll(), cvd())

        self.assertIsNone(blocked)
        self.assertEqual(strat.state.layers, 10)

        strat.state = long_state(layers=10, last_order_ts_ms=NOW_MS - 5 * 60 * 1000)
        allowed = strat._maybe_open_or_add_long(99.50, NOW_MS, boll(), cvd())

        self.assertIsNotNone(allowed)
        self.assertEqual(allowed.intent_type, "ADD_LONG")
        self.assertEqual(allowed.layer_index, 11)

    def test_open_long_not_affected_by_first_add_block(self) -> None:
        strat = strategy()

        result = strat._maybe_open_or_add_long(99.70, NOW_MS, boll(), cvd())

        self.assertIsNotNone(result)
        self.assertEqual(result.intent_type, "OPEN_LONG")
        self.assertEqual(result.layer_index, 1)

    def test_near_tp_add_disabled_still_blocks_add_before_new_gates(self) -> None:
        strat = strategy()
        strat.state = long_state(
            layers=2,
            last_order_ts_ms=NOW_MS - 60 * 60 * 1000,
            near_tp_add_disabled=True,
        )

        result = strat._maybe_open_or_add_long(99.50, NOW_MS, boll(), cvd())

        self.assertIsNone(result)
        self.assertEqual(strat.state.layers, 2)


if __name__ == "__main__":
    unittest.main()
