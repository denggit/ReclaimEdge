from __future__ import annotations

import unittest
import importlib.util
import sys
import types

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


def strategy() -> BollCvdShockReclaimStrategy:
    config = BollCvdReclaimStrategyConfig(min_outside_pct=0.001)
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


if __name__ == "__main__":
    unittest.main()
