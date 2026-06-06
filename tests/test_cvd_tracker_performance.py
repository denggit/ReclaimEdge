from __future__ import annotations

import builtins
import inspect
import os
import random
import time
import unittest
from unittest.mock import patch

from src.indicators.cvd_tracker import CvdSnapshot, CvdTracker, CvdTrackerConfig, Event, RangeSample


def perf_config() -> CvdTrackerConfig:
    return CvdTrackerConfig(
        fast_window_seconds=60,
        price_stall_seconds=2,
        burst_window_seconds=3,
        burst_baseline_seconds=60,
        burst_min_move_ratio=2.5,
        burst_min_volume_ratio=2.0,
    )


class CvdTrackerPerformanceTest(unittest.TestCase):
    def test_hot_path_dataclasses_use_slots(self) -> None:
        event = Event(
            ts_ms=1,
            price=100.0,
            signed_delta=1.0,
            buy_volume=1.0,
            sell_volume=0.0,
            volume=1.0,
        )
        sample = RangeSample(ts_ms=1, range_pct=0.01)
        snapshot = CvdSnapshot(
            ts_ms=1,
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
            no_new_low=False,
            no_new_high=False,
            window_low=100.0,
            window_high=100.0,
            burst_net_move_pct=0.0,
            burst_range_pct=0.0,
            baseline_range_pct=0.0,
            burst_move_ratio=0.0,
            burst_volume=1.0,
            baseline_volume=0.0,
            burst_volume_ratio=0.0,
            up_burst=False,
            down_burst=False,
        )

        self.assertFalse(hasattr(event, "__dict__"))
        self.assertFalse(hasattr(sample, "__dict__"))
        self.assertFalse(hasattr(snapshot, "__dict__"))

    def test_update_stats_logs_periodic_summary(self) -> None:
        with patch.dict(
                os.environ,
                {
                    "CVD_UPDATE_STATS_INTERVAL_SECONDS": "0.001",
                    "CVD_UPDATE_SLOW_LOG_MS": "0.0001",
                },
        ):
            tracker = CvdTracker(perf_config())

        with self.assertLogs("src.indicators.cvd_tracker", level="INFO") as logs:
            tracker.update("buy", 1.0, 100.0, 1_000)
            time.sleep(0.002)
            tracker.update("sell", 1.0, 99.9, 1_001)

        output = "\n".join(logs.output)
        self.assertIn("CVD_UPDATE_STATS", output)
        self.assertNotIn("CVD_UPDATE_SLOW", output)
        self.assertIn("avg_ms=", output)
        self.assertIn("p95_ms=", output)
        self.assertIn("p99_ms=", output)
        self.assertIn("max_ms=", output)
        self.assertIn("slow_count=", output)

    def test_ordered_update_path_handles_20k_ticks_quickly(self) -> None:
        tracker = CvdTracker(perf_config())
        rng = random.Random(7)
        price = 100.0
        snapshot: CvdSnapshot | None = None

        started = time.perf_counter()
        for index in range(20_000):
            price *= 1 + rng.uniform(-0.00003, 0.00003)
            side = "buy" if rng.random() >= 0.5 else "sell"
            size = rng.uniform(0.01, 3.0)
            snapshot = tracker.update(side, size, price, 1_000_000 + index * 3)
        elapsed = time.perf_counter() - started

        self.assertIsNotNone(snapshot)
        self.assertGreater(snapshot.window_high, 0)
        self.assertGreaterEqual(snapshot.window_high, snapshot.window_low)
        self.assertGreaterEqual(snapshot.buy_ratio, 0)
        self.assertLessEqual(snapshot.buy_ratio, 1)
        self.assertLess(elapsed, 5.0)

    def test_high_volume_down_burst_detected(self) -> None:
        tracker = CvdTracker(perf_config())
        ts = 2_000_000

        for index in range(600):
            phase_price = 100.02 if index % 2 == 0 else 99.98
            tracker.update("buy", 1.0, phase_price, ts + index * 100)

        snapshot = None
        for offset_ms, price in (
                (60_000, 100.00),
                (60_200, 99.86),
                (60_400, 99.70),
                (60_600, 99.52),
                (60_800, 99.38),
                (61_000, 99.29),
        ):
            snapshot = tracker.update("sell", 80.0, price, ts + offset_ms)

        self.assertIsNotNone(snapshot)
        self.assertTrue(snapshot.down_burst)
        self.assertGreaterEqual(snapshot.burst_move_ratio, tracker.config.burst_min_move_ratio)
        self.assertGreaterEqual(snapshot.burst_volume_ratio, tracker.config.burst_min_volume_ratio)
        self.assertLess(snapshot.burst_net_move_pct, 0)

    def test_monotonic_price_windows_track_low_high_and_no_new_low(self) -> None:
        tracker = CvdTracker(
            CvdTrackerConfig(
                fast_window_seconds=5,
                price_stall_seconds=2,
                price_stall_tolerance_pct=0.0005,
                burst_window_seconds=3,
                burst_baseline_seconds=60,
            )
        )
        base = 3_000_000
        tracker.update("sell", 1.0, 100.0, base)
        tracker.update("sell", 1.0, 99.0, base + 500)
        snapshot = tracker.update("buy", 1.0, 99.08, base + 1_000)

        self.assertTrue(snapshot.no_new_low)
        self.assertEqual(snapshot.window_low, 99.0)
        self.assertEqual(snapshot.window_high, 100.0)

        snapshot = tracker.update("buy", 1.0, 100.5, base + 2_600)
        self.assertEqual(snapshot.window_low, 99.08)
        self.assertEqual(snapshot.window_high, 100.5)

    def test_out_of_order_default_drop_does_not_sort_or_mutate_windows(self) -> None:
        with patch.dict(os.environ, {"CVD_OUT_OF_ORDER_POLICY": "drop_for_realtime"}):
            tracker = CvdTracker(perf_config())
        tracker.update("buy", 1.0, 100.0, 1_000)
        tracker.update("buy", 1.0, 100.1, 1_001)
        fast_len = len(tracker._fast_events)
        burst_len = len(tracker._burst_events)
        baseline_recent_len = len(tracker._recent_for_baseline)
        total_cvd = tracker._total_cvd

        with patch.object(builtins, "sorted", side_effect=AssertionError("sorted fallback forbidden")):
            with self.assertLogs("src.indicators.cvd_tracker", level="WARNING") as logs:
                snapshot = tracker.update("sell", 1.0, 99.9, 999)

        self.assertIn("CVD_TICK_OUT_OF_ORDER", "\n".join(logs.output))
        self.assertEqual(len(tracker._fast_events), fast_len)
        self.assertEqual(len(tracker._burst_events), burst_len)
        self.assertEqual(len(tracker._recent_for_baseline), baseline_recent_len)
        self.assertEqual(tracker._total_cvd, total_cvd)
        self.assertEqual(snapshot.ts_ms, 999)

    def test_update_hot_path_has_no_sorted_or_legacy_nested_scan_calls(self) -> None:
        source = inspect.getsource(CvdTracker.update)
        self.assertNotIn("sorted(", source)
        self.assertNotIn("_events_since", source)
        self.assertNotIn("_baseline_avg_range_pct", source)
        self.assertNotIn("_burst_stats", source)
        self.assertFalse(hasattr(CvdTracker, "_events_since"))
        self.assertFalse(hasattr(CvdTracker, "_baseline_avg_range_pct"))


if __name__ == "__main__":
    unittest.main()
