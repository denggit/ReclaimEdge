from __future__ import annotations

import unittest
import importlib.util
import sys
import types
from unittest.mock import patch

from src.indicators.cvd_tracker import CvdSnapshot

if importlib.util.find_spec("aiohttp") is None:
    sys.modules.setdefault("aiohttp", types.ModuleType("aiohttp"))

from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import BollCvdReclaimStrategyConfig
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy


def boll_snapshot() -> BollSnapshot:
    return BollSnapshot(
        inst_id="ETH-USDT-SWAP",
        candle_ts_ms=0,
        close=105.0,
        middle=105.0,
        upper=110.0,
        lower=100.0,
        upper_distance_pct=0.05,
        lower_distance_pct=0.05,
        alert_switch_on=True,
        live_mode=True,
    )


def cvd_snapshot(*, up_burst: bool = False, down_burst: bool = False) -> CvdSnapshot:
    return CvdSnapshot(
        ts_ms=0,
        price=105.0,
        side="unknown",
        size=0.0,
        signed_delta=0.0,
        total_cvd=0.0,
        fast_cvd=0.0,
        previous_fast_cvd=0.0,
        buy_volume=0.0,
        sell_volume=0.0,
        buy_ratio=0.0,
        sell_ratio=0.0,
        cross_positive=False,
        cross_negative=False,
        cvd_increasing=False,
        cvd_decreasing=False,
        no_new_low=False,
        no_new_high=False,
        window_low=105.0,
        window_high=105.0,
        burst_net_move_pct=0.0,
        burst_range_pct=0.01,
        baseline_range_pct=0.001,
        burst_move_ratio=10.0,
        burst_volume=10.0,
        baseline_volume=1.0,
        burst_volume_ratio=10.0,
        up_burst=up_burst,
        down_burst=down_burst,
    )


def strategy(**overrides) -> BollCvdShockReclaimStrategy:
    values = dict(min_outside_pct=0.001)
    values.update(overrides)
    config = BollCvdReclaimStrategyConfig(**values)
    sizer = SimplePositionSizer(SimplePositionSizerConfig())
    return BollCvdShockReclaimStrategy(config, sizer)


class BollCvdShockReclaimStrategyTest(unittest.TestCase):
    def test_lower_shock_sets_deep_enough_when_extreme_reaches_min_outside_pct(self) -> None:
        strat = strategy()

        with self.assertLogs("src.strategies.boll_cvd_reclaim_strategy", level="INFO") as logs:
            strat.on_tick(99.9, 10_000, boll_snapshot(), cvd_snapshot(down_burst=True))

        self.assertTrue(strat.state.lower_deep_enough)
        self.assertIn("LOWER_DEEP_ENOUGH", "\n".join(logs.output))

    def test_upper_shock_sets_deep_enough_when_extreme_reaches_min_outside_pct(self) -> None:
        strat = strategy()

        with self.assertLogs("src.strategies.boll_cvd_reclaim_strategy", level="INFO") as logs:
            strat.on_tick(110.11, 10_000, boll_snapshot(), cvd_snapshot(up_burst=True))

        self.assertTrue(strat.state.upper_deep_enough)
        self.assertIn("UPPER_DEEP_ENOUGH", "\n".join(logs.output))

    def test_shock_does_not_set_deep_enough_before_min_outside_pct(self) -> None:
        lower_strat = strategy()
        upper_strat = strategy()

        lower_strat.on_tick(99.95, 10_000, boll_snapshot(), cvd_snapshot(down_burst=True))
        upper_strat.on_tick(110.05, 10_000, boll_snapshot(), cvd_snapshot(up_burst=True))

        self.assertFalse(lower_strat.state.lower_deep_enough)
        self.assertFalse(upper_strat.state.upper_deep_enough)

    def test_lower_outside_no_burst_logs_at_low_frequency_without_arming(self) -> None:
        strat = strategy()
        strat.outside_no_burst_log_interval_seconds = 2

        with self.assertLogs("src.strategies.boll_cvd_shock_reclaim_strategy", level="INFO") as logs:
            with patch("src.strategies.boll_cvd_shock_reclaim_strategy.time.monotonic", side_effect=[100.0, 101.0, 102.1]):
                first = strat.on_tick(99.9, 10_000, boll_snapshot(), cvd_snapshot(down_burst=False))
                second = strat.on_tick(99.8, 9_000, boll_snapshot(), cvd_snapshot(down_burst=False))
                third = strat.on_tick(99.7, 8_000, boll_snapshot(), cvd_snapshot(down_burst=False))

        output = "\n".join(logs.output)
        self.assertNotIn("LOWER_ARMED", output)
        self.assertEqual(output.count("LOWER_OUTSIDE_NO_BURST"), 2)
        self.assertFalse(strat.state.lower_armed)
        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(third, [])

    def test_first_entry_starts_add_freeze_chain(self) -> None:
        strat = strategy(first_add_block_seconds=2700, add_min_interval_seconds=1800)
        ts_ms = 100_000

        strat._open_position("LONG", "OPEN_LONG", 100.0, ts_ms, boll_snapshot(), cvd_snapshot(), "test")

        self.assertEqual(strat.state.first_entry_ts_ms, ts_ms)
        self.assertEqual(strat.state.add_freeze_until_ts_ms, ts_ms + 2_700_000)
        self.assertEqual(strat.state.add_freeze_penalty_count, 0)

    def test_first_add_in_freeze_requires_first_bypass_multiplier(self) -> None:
        strat = strategy(first_add_block_seconds=2700, add_layer_gap_pct=0.003)
        strat.state.layers = 1
        strat.state.last_entry_price = 100.0
        strat.state.last_order_ts_ms = 100_000
        strat.state.first_entry_ts_ms = 100_000
        strat.state.add_freeze_until_ts_ms = 100_000 + 2_700_000
        strat.first_add_block_bypass_multiplier = 5.0

        ok, reason = strat._add_timing_passed("LONG", 98.6, 200_000, 2)

        self.assertFalse(ok)
        self.assertEqual(reason, "add_freeze")

    def test_first_add_in_freeze_bypasses_at_5x_and_extends_freeze(self) -> None:
        strat = strategy(first_add_block_seconds=2700, add_min_interval_seconds=1800, add_layer_gap_pct=0.003)
        ts_ms = 100_000
        strat._open_position("LONG", "OPEN_LONG", 100.0, ts_ms, boll_snapshot(), cvd_snapshot(), "test")
        old_freeze_until = strat.state.add_freeze_until_ts_ms
        add_ts_ms = old_freeze_until - 300_000
        ok, reason = strat._add_timing_passed("LONG", 98.5, add_ts_ms, 2)
        self.assertTrue(ok)
        self.assertEqual(reason, "first_add_block_bypassed")

        strat._open_position("LONG", "ADD_LONG", 98.5, add_ts_ms, boll_snapshot(), cvd_snapshot(), "add")

        self.assertEqual(strat.state.add_freeze_until_ts_ms, old_freeze_until + 1_800_000)
        self.assertEqual(strat.state.add_freeze_penalty_count, 1)
        self.assertEqual(strat.state.first_entry_ts_ms, ts_ms)

    def test_second_add_same_freeze_requires_3x(self) -> None:
        strat = strategy(add_layer_gap_pct=0.003, add_min_interval_bypass_multiplier=2.0)
        strat.state.layers = 2
        strat.state.last_entry_price = 100.0
        strat.state.add_freeze_until_ts_ms = 2_000_000
        strat.state.add_freeze_penalty_count = 1

        too_small, reason = strat._add_timing_passed("LONG", 99.2, 1_000_000, 3)
        enough, enough_reason = strat._add_timing_passed("LONG", 99.1, 1_000_000, 3)

        self.assertFalse(too_small)
        self.assertEqual(reason, "add_freeze")
        self.assertTrue(enough)
        self.assertEqual(enough_reason, "add_freeze_bypassed")

    def test_add_freeze_skipped_log_is_throttled_by_time_and_key(self) -> None:
        strat = strategy(add_layer_gap_pct=0.003, add_min_interval_bypass_multiplier=2.0)
        strat.state.layers = 2
        strat.state.last_entry_price = 100.0
        strat.state.add_freeze_until_ts_ms = 2_000_000
        strat.state.add_freeze_penalty_count = 1

        ok, reason = strat._add_timing_passed("LONG", 99.2, 1_000_000, 3)
        self.assertFalse(ok)
        self.assertEqual(reason, "add_freeze")

        with patch("src.strategies.boll_cvd_shock_reclaim_strategy.logger.info") as log_info:
            strat._log_add_timing_skipped("LONG", "add_freeze", 99.2, 1_000_000, 3)
            strat._log_add_timing_skipped("LONG", "add_freeze", 99.1, 1_001_000, 3)
            self.assertEqual(log_info.call_count, 1)

            strat._log_add_timing_skipped("LONG", "add_freeze", 99.0, 1_030_000, 3)
            self.assertEqual(log_info.call_count, 2)

            strat._log_add_timing_skipped("LONG", "add_freeze", 98.9, 1_031_000, 4)
            self.assertEqual(log_info.call_count, 3)

            strat.state.add_freeze_penalty_count = 2
            strat._log_add_timing_skipped("LONG", "add_freeze", 98.8, 1_032_000, 4)
            self.assertEqual(log_info.call_count, 4)

    def test_penalty_increments_to_4x_after_second_freeze_add(self) -> None:
        strat = strategy(add_layer_gap_pct=0.003, add_min_interval_seconds=1800, add_min_interval_bypass_multiplier=2.0)
        strat.state.side = "LONG"
        strat.state.layers = 2
        strat.state.last_entry_price = 100.0
        strat.state.total_entry_qty = 1.0
        strat.state.total_entry_notional = 100.0
        strat.state.avg_entry_price = 100.0
        strat.state.add_freeze_until_ts_ms = 2_000_000
        strat.state.add_freeze_penalty_count = 1

        strat._open_position("LONG", "ADD_LONG", 99.1, 1_000_000, boll_snapshot(), cvd_snapshot(), "add")

        self.assertEqual(strat.state.add_freeze_penalty_count, 2)
        self.assertAlmostEqual(strat._active_add_freeze_bypass_multiplier(), 4.0)

    def test_freeze_expiry_resets_penalty(self) -> None:
        strat = strategy(add_layer_gap_pct=0.003)
        strat.state.layers = 2
        strat.state.last_entry_price = 100.0
        strat.state.last_order_ts_ms = 900_000
        strat.state.add_freeze_until_ts_ms = 1_000_000
        strat.state.add_freeze_penalty_count = 2

        ok, reason = strat._add_timing_passed("LONG", 99.39, 1_000_001, 3)

        self.assertTrue(ok)
        self.assertEqual(reason, "ok")
        self.assertEqual(strat.state.add_freeze_until_ts_ms, 0)
        self.assertEqual(strat.state.add_freeze_penalty_count, 0)

    def test_add_after_freeze_inactive_starts_new_interval_freeze(self) -> None:
        strat = strategy(add_min_interval_seconds=1800)
        strat.state.side = "LONG"
        strat.state.layers = 2
        strat.state.last_entry_price = 100.0
        strat.state.total_entry_qty = 1.0
        strat.state.total_entry_notional = 100.0
        strat.state.avg_entry_price = 100.0
        strat.state.add_freeze_until_ts_ms = 0
        strat.state.add_freeze_penalty_count = 0

        strat._open_position("LONG", "ADD_LONG", 99.0, 3_000_000, boll_snapshot(), cvd_snapshot(), "add")

        self.assertEqual(strat.state.add_freeze_until_ts_ms, 3_000_000 + 1_800_000)
        self.assertEqual(strat.state.add_freeze_penalty_count, 0)
        self.assertAlmostEqual(strat._active_add_freeze_bypass_multiplier(), 2.0)


if __name__ == "__main__":
    unittest.main()
