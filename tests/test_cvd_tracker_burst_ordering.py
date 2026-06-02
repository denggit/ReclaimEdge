from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.indicators.cvd_tracker import CvdTracker, CvdTrackerConfig


def config() -> CvdTrackerConfig:
    return CvdTrackerConfig(
        fast_window_seconds=5,
        price_stall_seconds=2,
        burst_window_seconds=3,
        burst_baseline_seconds=60,
        burst_min_move_ratio=2.5,
        burst_min_volume_ratio=2.0,
    )


def feed_baseline(tracker: CvdTracker, start_ts: int) -> None:
    for offset_ms in range(0, 60_000, 1_000):
        price = 100.02 if (offset_ms // 1_000) % 2 == 0 else 99.98
        tracker.update("buy", 1.0, price, start_ts + offset_ms)


def feed_down_burst(tracker: CvdTracker, start_ts: int, *, include_last: bool = True):
    snapshot = None
    ticks = [
        (60_000, 100.0),
        (61_000, 99.80),
        (62_000, 99.50),
        (63_000, 99.29),
    ]
    if not include_last:
        ticks = ticks[:-1]
    for offset_ms, price in ticks:
        snapshot = tracker.update("sell", 10.0, price, start_ts + offset_ms)
    return snapshot


class CvdTrackerBurstOrderingTest(unittest.TestCase):
    def test_ordered_fast_down_move_triggers_down_burst(self) -> None:
        tracker = CvdTracker(config())
        start_ts = 1_000_000
        feed_baseline(tracker, start_ts)

        snapshot = feed_down_burst(tracker, start_ts)

        self.assertIsNotNone(snapshot)
        self.assertTrue(snapshot.down_burst)
        self.assertGreaterEqual(snapshot.burst_move_ratio, 2.5)
        self.assertGreaterEqual(snapshot.burst_volume_ratio, 2.0)
        self.assertLess(snapshot.burst_net_move_pct, 0)

    def test_out_of_order_tick_is_dropped_for_realtime(self) -> None:
        tracker = CvdTracker(config())
        start_ts = 2_000_000
        feed_baseline(tracker, start_ts)
        tracker.update("sell", 10.0, 100.0, start_ts + 60_000)
        tracker.update("sell", 10.0, 99.80, start_ts + 61_000)
        tracker.update("sell", 10.0, 99.50, start_ts + 62_000)
        latest = tracker.update("sell", 10.0, 99.00, start_ts + 64_000)
        burst_len = len(tracker._burst_events)
        fast_len = len(tracker._fast_events)
        total_cvd = tracker._total_cvd

        with self.assertLogs("src.indicators.cvd_tracker", level="WARNING") as logs:
            snapshot = tracker.update("sell", 10.0, 99.29, start_ts + 63_000)

        self.assertIn("CVD_TICK_OUT_OF_ORDER", "\n".join(logs.output))
        self.assertEqual(len(tracker._burst_events), burst_len)
        self.assertEqual(len(tracker._fast_events), fast_len)
        self.assertEqual(tracker._total_cvd, total_cvd)
        self.assertEqual(snapshot.fast_cvd, latest.fast_cvd)
        self.assertFalse(snapshot.down_burst)

    def test_out_of_order_log_throttle_uses_monotonic_not_tick_timestamp(self) -> None:
        with patch.dict(os.environ, {"CVD_UPDATE_SLOW_LOG_MS": "0", "CVD_UPDATE_STATS_INTERVAL_SECONDS": "0"}):
            tracker = CvdTracker(config())
        tracker.update("buy", 1.0, 100.0, 10_000)

        with patch("src.indicators.cvd_tracker.time.monotonic", side_effect=[100.0, 101.0, 106.1]):
            with self.assertLogs("src.indicators.cvd_tracker", level="WARNING") as logs:
                tracker.update("sell", 1.0, 99.9, 9_900)
                tracker.update("sell", 1.0, 99.8, 9_800)
                tracker.update("sell", 1.0, 99.7, 9_700)

        output = "\n".join(logs.output)
        self.assertEqual(output.count("CVD_TICK_OUT_OF_ORDER"), 2)


if __name__ == "__main__":
    unittest.main()
