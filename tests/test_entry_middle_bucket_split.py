"""Tests for middle bucket split on Three-Stage and Middle Runner initial entry.

Verifies:
- Three-Stage: OPEN_SHORT / OPEN_LONG with SPLIT, UNSPLIT_SLOW_MIDDLE, FALLBACK/DISABLED.
- Middle Runner: OPEN_SHORT / OPEN_LONG with SPLIT, UNSPLIT_SLOW_MIDDLE.
- Sub-leg too small: intent still carries split fields; execution layer handles
  the subleg-too-small disable (verified by existing execution-layer tests).
"""

from __future__ import annotations

import unittest

from src.indicators.cvd_tracker import CvdSnapshot
from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
)
from src.strategies.entry_add_flow_coordinator import EntryAddFlowCoordinator


# ── reusable helpers ────────────────────────────────────────────────────

def _boll(
    middle: float = 2000.0,
    upper: float = 2100.0,
    lower: float = 1900.0,
    tp_middle: float | None = None,
    candle_ts_ms: int = 1000,
) -> BollSnapshot:
    return BollSnapshot(
        inst_id="ETH-USDT-SWAP",
        candle_ts_ms=candle_ts_ms,
        close=middle,
        middle=middle,
        upper=upper,
        lower=lower,
        upper_distance_pct=0.05,
        lower_distance_pct=0.05,
        alert_switch_on=True,
        live_mode=True,
        tp_middle=tp_middle,
    )


def _cvd(buy_ratio: float = 0.6, sell_ratio: float = 0.4) -> CvdSnapshot:
    return CvdSnapshot(
        ts_ms=1000,
        price=2000.0,
        side="buy",
        size=1.0,
        signed_delta=0.1,
        total_cvd=0.5,
        fast_cvd=0.1,
        previous_fast_cvd=0.09,
        buy_volume=60.0,
        sell_volume=40.0,
        buy_ratio=buy_ratio,
        sell_ratio=sell_ratio,
        cross_positive=True,
        cross_negative=False,
        cvd_increasing=True,
        cvd_decreasing=False,
        no_new_low=True,
        no_new_high=False,
        window_low=1990.0,
        window_high=2010.0,
        burst_net_move_pct=0.0,
        burst_range_pct=0.01,
        baseline_range_pct=0.001,
        burst_move_ratio=10.0,
        burst_volume=10.0,
        baseline_volume=1.0,
        burst_volume_ratio=10.0,
        up_burst=False,
        down_burst=False,
    )


def _sizer() -> SimplePositionSizer:
    cfg = SimplePositionSizerConfig()
    return SimplePositionSizer(cfg)


def _coordinator(strategy: BollCvdReclaimStrategy) -> EntryAddFlowCoordinator:
    return EntryAddFlowCoordinator(strategy)


# ── Three-Stage initial entry with middle bucket split: SHORT ──────────

class TestThreeStageEntrySplitShort(unittest.TestCase):
    """OPEN_SHORT with Three-Stage + middle bucket split enabled."""

    def _make_strategy(self) -> BollCvdReclaimStrategy:
        config = BollCvdReclaimStrategyConfig(
            three_stage_runner_enabled=True,
            middle_bucket_split_enabled=True,
            middle_bucket_split_fast_ratio=0.70,
            tp_boll_enabled=True,
            tp_min_net_profit_pct=0.002,
            split_tp_enabled=False,
        )
        sizer = _sizer()
        return BollCvdReclaimStrategy(config, sizer)

    def test_short_entry_split_both_middles_ok(self):
        """SHORT entry with BOLL15 and BOLL20 both satisfying profit → SPLIT."""
        strategy = self._make_strategy()
        # SHORT entry at 2100; BOLL15 tp_middle=2080, BOLL20 middle=2090
        # Effective breakeven ~2100; required = 2100 * 0.998 = 2095.8
        # Both 2080 and 2090 are < 2095.8 → both OK for SHORT
        boll = _boll(middle=2090.0, upper=2150.0, lower=2050.0, tp_middle=2080.0)
        cvd = _cvd()
        coord = _coordinator(strategy)

        intent = coord.open_position("SHORT", "OPEN_SHORT", 2100.0, 1000, boll, cvd, "base")

        # Intent should still be THREE_STAGE_RUNNER
        self.assertEqual(intent.tp_plan, "THREE_STAGE_RUNNER")
        # Middle bucket split should be active
        self.assertTrue(intent.middle_bucket_split_active)
        # Fast price = BOLL15 tp_middle
        self.assertEqual(intent.middle_bucket_split_fast_price, 2080.0)
        # Slow price = BOLL20 middle
        self.assertEqual(intent.middle_bucket_split_slow_price, 2090.0)
        # Effective price = weighted average: 2080 * 0.70 + 2090 * 0.30 = 2083.0
        expected_effective = 2080.0 * 0.70 + 2090.0 * 0.30
        self.assertAlmostEqual(
            intent.middle_bucket_split_effective_price, expected_effective, places=4,
        )
        # three_stage_tp1_price should equal the effective price
        self.assertAlmostEqual(
            intent.three_stage_tp1_price, expected_effective, places=4,
        )
        # state should also reflect split
        self.assertTrue(strategy.state.middle_bucket_split_active)
        self.assertEqual(strategy.state.middle_bucket_split_fast_price, 2080.0)
        self.assertEqual(strategy.state.middle_bucket_split_slow_price, 2090.0)

    def test_short_entry_split_logs_middle_bucket_split_selected(self):
        """SPLIT on entry should log MIDDLE_BUCKET_SPLIT_SELECTED."""
        strategy = self._make_strategy()
        boll = _boll(middle=2090.0, upper=2150.0, lower=2050.0, tp_middle=2080.0)
        cvd = _cvd()
        coord = _coordinator(strategy)

        with self.assertLogs("src.strategies.middle_bucket_split_apply", level="WARNING") as logs:
            coord.open_position("SHORT", "OPEN_SHORT", 2100.0, 1000, boll, cvd, "base")
        output = "\n".join(logs.output)
        self.assertIn("MIDDLE_BUCKET_SPLIT_SELECTED", output)
        self.assertIn("plan=THREE_STAGE_RUNNER", output)
        self.assertIn("side=SHORT", output)


# ── Three-Stage initial entry with middle bucket split: LONG ───────────

class TestThreeStageEntrySplitLong(unittest.TestCase):
    """OPEN_LONG with Three-Stage + middle bucket split enabled."""

    def _make_strategy(self) -> BollCvdReclaimStrategy:
        config = BollCvdReclaimStrategyConfig(
            three_stage_runner_enabled=True,
            middle_bucket_split_enabled=True,
            middle_bucket_split_fast_ratio=0.70,
            tp_boll_enabled=True,
            tp_min_net_profit_pct=0.002,
            split_tp_enabled=False,
        )
        sizer = _sizer()
        return BollCvdReclaimStrategy(config, sizer)

    def test_long_entry_split_both_middles_ok(self):
        """LONG entry with BOLL15 and BOLL20 both satisfying profit → SPLIT."""
        strategy = self._make_strategy()
        # LONG entry at 1900; BOLL15 tp_middle=1920, BOLL20 middle=1910
        # Effective breakeven ~1900; required = 1900 * 1.002 = 1903.8
        # Both 1920 and 1910 are >= 1903.8 → both OK for LONG
        boll = _boll(middle=1910.0, upper=2000.0, lower=1800.0, tp_middle=1920.0)
        cvd = _cvd()
        coord = _coordinator(strategy)

        intent = coord.open_position("LONG", "OPEN_LONG", 1900.0, 1000, boll, cvd, "base")

        self.assertEqual(intent.tp_plan, "THREE_STAGE_RUNNER")
        self.assertTrue(intent.middle_bucket_split_active)
        self.assertEqual(intent.middle_bucket_split_fast_price, 1920.0)
        self.assertEqual(intent.middle_bucket_split_slow_price, 1910.0)
        # Effective price = 1920 * 0.70 + 1910 * 0.30 = 1917.0
        expected_effective = 1920.0 * 0.70 + 1910.0 * 0.30
        self.assertAlmostEqual(
            intent.middle_bucket_split_effective_price, expected_effective, places=4,
        )
        self.assertAlmostEqual(
            intent.three_stage_tp1_price, expected_effective, places=4,
        )
        self.assertTrue(strategy.state.middle_bucket_split_active)
        self.assertEqual(strategy.state.middle_bucket_split_fast_price, 1920.0)
        self.assertEqual(strategy.state.middle_bucket_split_slow_price, 1910.0)


# ── UNSPLIT_SLOW_MIDDLE ────────────────────────────────────────────────

class TestThreeStageEntryUnsplitSlowMiddle(unittest.TestCase):
    """Fast middle insufficient but slow middle OK → UNSPLIT_SLOW_MIDDLE."""

    def _make_strategy(self) -> BollCvdReclaimStrategy:
        config = BollCvdReclaimStrategyConfig(
            three_stage_runner_enabled=True,
            middle_bucket_split_enabled=True,
            middle_bucket_split_fast_ratio=0.70,
            tp_boll_enabled=True,
            tp_min_net_profit_pct=0.002,
            split_tp_enabled=False,
        )
        sizer = _sizer()
        return BollCvdReclaimStrategy(config, sizer)

    def test_long_entry_unsplit_slow_middle(self):
        """LONG: BOLL15 tp_middle too close to entry, BOLL20 OK → UNSPLIT."""
        strategy = self._make_strategy()
        # LONG entry at 1900; required = 1900 * 1.002 = 1903.8
        # BOLL15 tp_middle=1901 < 1903.8 → insufficient
        # BOLL20 middle=1910 >= 1903.8 → sufficient
        boll = _boll(middle=1910.0, upper=2000.0, lower=1800.0, tp_middle=1901.0)
        cvd = _cvd()
        coord = _coordinator(strategy)

        intent = coord.open_position("LONG", "OPEN_LONG", 1900.0, 1000, boll, cvd, "base")

        self.assertEqual(intent.tp_plan, "THREE_STAGE_RUNNER")
        # Split should NOT be active (UNSPLIT_SLOW_MIDDLE)
        self.assertFalse(intent.middle_bucket_split_active)
        # tp1_price should be BOLL20 middle
        self.assertAlmostEqual(intent.three_stage_tp1_price, 1910.0, places=4)
        # partial_tp_price should be BOLL20 middle
        self.assertAlmostEqual(intent.partial_tp_price, 1910.0, places=4)
        # State should reflect no split
        self.assertFalse(strategy.state.middle_bucket_split_active)

    def test_long_entry_unsplit_logs_skipped(self):
        """UNSPLIT on entry should log MIDDLE_BUCKET_SPLIT_SKIPPED."""
        strategy = self._make_strategy()
        boll = _boll(middle=1910.0, upper=2000.0, lower=1800.0, tp_middle=1901.0)
        cvd = _cvd()
        coord = _coordinator(strategy)

        with self.assertLogs("src.strategies.middle_bucket_split_apply", level="WARNING") as logs:
            coord.open_position("LONG", "OPEN_LONG", 1900.0, 1000, boll, cvd, "base")
        output = "\n".join(logs.output)
        self.assertIn("MIDDLE_BUCKET_SPLIT_SKIPPED", output)
        self.assertIn("plan=THREE_STAGE_RUNNER", output)

    def test_short_entry_unsplit_slow_middle(self):
        """SHORT: BOLL15 tp_middle too close to entry, BOLL20 OK → UNSPLIT."""
        strategy = self._make_strategy()
        # SHORT entry at 2100; required = 2100 * 0.998 = 2095.8
        # BOLL15 tp_middle=2097 > 2095.8 → insufficient (not far enough below)
        # BOLL20 middle=2080 < 2095.8 → sufficient
        boll = _boll(middle=2080.0, upper=2150.0, lower=2050.0, tp_middle=2097.0)
        cvd = _cvd()
        coord = _coordinator(strategy)

        intent = coord.open_position("SHORT", "OPEN_SHORT", 2100.0, 1000, boll, cvd, "base")

        self.assertEqual(intent.tp_plan, "THREE_STAGE_RUNNER")
        self.assertFalse(intent.middle_bucket_split_active)
        self.assertAlmostEqual(intent.three_stage_tp1_price, 2080.0, places=4)


# ── FALLBACK_OUTER / DISABLED ──────────────────────────────────────────

class TestThreeStageEntrySplitFallbackDisabled(unittest.TestCase):
    """FALLBACK_OUTER or DISABLED: split not applied, existing behaviour preserved."""

    def _make_strategy(self, **overrides) -> BollCvdReclaimStrategy:
        vals = dict(
            three_stage_runner_enabled=True,
            middle_bucket_split_enabled=True,
            middle_bucket_split_fast_ratio=0.70,
            tp_boll_enabled=True,
            tp_min_net_profit_pct=0.002,
            split_tp_enabled=False,
        )
        vals.update(overrides)
        config = BollCvdReclaimStrategyConfig(**vals)
        sizer = _sizer()
        return BollCvdReclaimStrategy(config, sizer)

    def test_fallback_outer_both_insufficient(self):
        """Both BOLL15 and BOLL20 insufficient → FALLBACK_OUTER.
        Entry should still succeed with THREE_STAGE_RUNNER (or fallback plan)."""
        strategy = self._make_strategy()
        # LONG entry at 1900; required = 1903.8
        # Both BOLL15=1901 and BOLL20=1902 are < required → FALLBACK_OUTER
        boll = _boll(middle=1902.0, upper=2000.0, lower=1800.0, tp_middle=1901.0)
        cvd = _cvd()
        coord = _coordinator(strategy)

        intent = coord.open_position("LONG", "OPEN_LONG", 1900.0, 1000, boll, cvd, "base")

        # Should still succeed — split just wasn't applied
        self.assertIsNotNone(intent)
        self.assertFalse(intent.middle_bucket_split_active)
        self.assertFalse(strategy.state.middle_bucket_split_active)

    def test_disabled_when_config_false(self):
        """When middle_bucket_split_enabled=False, entry should work without split."""
        strategy = self._make_strategy(middle_bucket_split_enabled=False)
        boll = _boll(middle=1910.0, upper=2000.0, lower=1800.0, tp_middle=1920.0)
        cvd = _cvd()
        coord = _coordinator(strategy)

        intent = coord.open_position("LONG", "OPEN_LONG", 1900.0, 1000, boll, cvd, "base")

        self.assertIsNotNone(intent)
        self.assertFalse(intent.middle_bucket_split_active)
        self.assertFalse(strategy.state.middle_bucket_split_active)


# ── Sub-leg too small: intent carries split fields ─────────────────────

class TestThreeStageEntrySplitSublegTooSmall(unittest.TestCase):
    """When contracts are small, the intent still carries split fields.
    The execution layer handles the subleg-too-small disable downstream.
    """

    def _make_strategy(self) -> BollCvdReclaimStrategy:
        config = BollCvdReclaimStrategyConfig(
            three_stage_runner_enabled=True,
            middle_bucket_split_enabled=True,
            middle_bucket_split_fast_ratio=0.70,
            tp_boll_enabled=True,
            tp_min_net_profit_pct=0.002,
            split_tp_enabled=False,
        )
        sizer = _sizer()
        return BollCvdReclaimStrategy(config, sizer)

    def test_split_active_on_intent_even_with_small_position(self):
        """Even with small contracts, the entry path sets split state.
        The execution layer will handle subleg-too-small downstream.
        """
        strategy = self._make_strategy()
        boll = _boll(middle=1910.0, upper=2000.0, lower=1800.0, tp_middle=1920.0)
        cvd = _cvd()
        coord = _coordinator(strategy)

        intent = coord.open_position("LONG", "OPEN_LONG", 1900.0, 1000, boll, cvd, "base")

        self.assertEqual(intent.tp_plan, "THREE_STAGE_RUNNER")
        # Split state is populated — execution layer decides whether to use it
        self.assertTrue(intent.middle_bucket_split_active)
        self.assertIsNotNone(intent.middle_bucket_split_fast_price)
        self.assertIsNotNone(intent.middle_bucket_split_slow_price)
        self.assertIsNotNone(intent.middle_bucket_split_effective_price)
        # Fast and slow ratio fields are populated
        self.assertGreater(intent.middle_bucket_split_fast_total_ratio, 0.0)
        self.assertGreater(intent.middle_bucket_split_slow_total_ratio, 0.0)


# ── Middle Runner initial entry with middle bucket split: SHORT ────────

class TestMiddleRunnerEntrySplitShort(unittest.TestCase):
    """OPEN_SHORT with Middle Runner + middle bucket split enabled."""

    def _make_strategy(self) -> BollCvdReclaimStrategy:
        config = BollCvdReclaimStrategyConfig(
            three_stage_runner_enabled=False,
            middle_runner_enabled=True,
            middle_bucket_split_enabled=True,
            middle_bucket_split_fast_ratio=0.70,
            tp_boll_enabled=True,
            tp_min_net_profit_pct=0.002,
            split_tp_enabled=False,
        )
        sizer = _sizer()
        return BollCvdReclaimStrategy(config, sizer)

    def test_short_entry_split_both_middles_ok(self):
        """SHORT entry with BOLL15 and BOLL20 both satisfying profit → SPLIT."""
        strategy = self._make_strategy()
        # SHORT entry at 2100; BOLL15 tp_middle=2080, BOLL20 middle=2090
        # Effective breakeven ~2100; required = 2100 * 0.998 = 2095.8
        # Both 2080 and 2090 are < 2095.8 → both OK for SHORT
        boll = _boll(middle=2090.0, upper=2150.0, lower=2050.0, tp_middle=2080.0)
        cvd = _cvd()
        coord = _coordinator(strategy)

        intent = coord.open_position("SHORT", "OPEN_SHORT", 2100.0, 1000, boll, cvd, "base")

        self.assertEqual(intent.tp_plan, "MIDDLE_RUNNER")
        self.assertTrue(intent.middle_bucket_split_active)
        self.assertEqual(intent.middle_bucket_split_fast_price, 2080.0)
        self.assertEqual(intent.middle_bucket_split_slow_price, 2090.0)
        # Effective price = 2080 * 0.70 + 2090 * 0.30 = 2083.0
        expected_effective = 2080.0 * 0.70 + 2090.0 * 0.30
        self.assertAlmostEqual(
            intent.middle_bucket_split_effective_price, expected_effective, places=4,
        )
        # middle_runner_first_tp_price should equal effective price
        self.assertAlmostEqual(
            intent.middle_runner_first_tp_price, expected_effective, places=4,
        )
        # middle_runner_final_tp_price should be set (outer TP)
        self.assertIsNotNone(intent.middle_runner_final_tp_price)
        # State should reflect split
        self.assertTrue(strategy.state.middle_bucket_split_active)
        self.assertEqual(strategy.state.middle_bucket_split_fast_price, 2080.0)
        self.assertEqual(strategy.state.middle_bucket_split_slow_price, 2090.0)

    def test_short_entry_split_logs_middle_bucket_split_selected(self):
        """SPLIT on entry should log MIDDLE_BUCKET_SPLIT_SELECTED for MIDDLE_RUNNER."""
        strategy = self._make_strategy()
        boll = _boll(middle=2090.0, upper=2150.0, lower=2050.0, tp_middle=2080.0)
        cvd = _cvd()
        coord = _coordinator(strategy)

        with self.assertLogs("src.strategies.middle_bucket_split_apply", level="WARNING") as logs:
            coord.open_position("SHORT", "OPEN_SHORT", 2100.0, 1000, boll, cvd, "base")
        output = "\n".join(logs.output)
        self.assertIn("MIDDLE_BUCKET_SPLIT_SELECTED", output)
        self.assertIn("plan=MIDDLE_RUNNER", output)
        self.assertIn("side=SHORT", output)


# ── Middle Runner initial entry with middle bucket split: LONG ─────────

class TestMiddleRunnerEntrySplitLong(unittest.TestCase):
    """OPEN_LONG with Middle Runner + middle bucket split enabled."""

    def _make_strategy(self) -> BollCvdReclaimStrategy:
        config = BollCvdReclaimStrategyConfig(
            three_stage_runner_enabled=False,
            middle_runner_enabled=True,
            middle_bucket_split_enabled=True,
            middle_bucket_split_fast_ratio=0.70,
            tp_boll_enabled=True,
            tp_min_net_profit_pct=0.002,
            split_tp_enabled=False,
        )
        sizer = _sizer()
        return BollCvdReclaimStrategy(config, sizer)

    def test_long_entry_split_both_middles_ok(self):
        """LONG entry with BOLL15 and BOLL20 both satisfying profit → SPLIT."""
        strategy = self._make_strategy()
        # LONG entry at 1900; BOLL15 tp_middle=1920, BOLL20 middle=1910
        # Effective breakeven ~1900; required = 1900 * 1.002 = 1903.8
        # Both 1920 and 1910 are >= 1903.8 → both OK for LONG
        boll = _boll(middle=1910.0, upper=2000.0, lower=1800.0, tp_middle=1920.0)
        cvd = _cvd()
        coord = _coordinator(strategy)

        intent = coord.open_position("LONG", "OPEN_LONG", 1900.0, 1000, boll, cvd, "base")

        self.assertEqual(intent.tp_plan, "MIDDLE_RUNNER")
        self.assertTrue(intent.middle_bucket_split_active)
        self.assertEqual(intent.middle_bucket_split_fast_price, 1920.0)
        self.assertEqual(intent.middle_bucket_split_slow_price, 1910.0)
        expected_effective = 1920.0 * 0.70 + 1910.0 * 0.30
        self.assertAlmostEqual(
            intent.middle_bucket_split_effective_price, expected_effective, places=4,
        )
        self.assertAlmostEqual(
            intent.middle_runner_first_tp_price, expected_effective, places=4,
        )
        self.assertIsNotNone(intent.middle_runner_final_tp_price)
        self.assertTrue(strategy.state.middle_bucket_split_active)
        self.assertEqual(strategy.state.middle_bucket_split_fast_price, 1920.0)
        self.assertEqual(strategy.state.middle_bucket_split_slow_price, 1910.0)


# ── Middle Runner UNSPLIT_SLOW_MIDDLE ──────────────────────────────────

class TestMiddleRunnerEntryUnsplitSlowMiddle(unittest.TestCase):
    """Middle Runner: fast middle insufficient but slow middle OK → UNSPLIT_SLOW_MIDDLE."""

    def _make_strategy(self) -> BollCvdReclaimStrategy:
        config = BollCvdReclaimStrategyConfig(
            three_stage_runner_enabled=False,
            middle_runner_enabled=True,
            middle_bucket_split_enabled=True,
            middle_bucket_split_fast_ratio=0.70,
            tp_boll_enabled=True,
            tp_min_net_profit_pct=0.002,
            split_tp_enabled=False,
        )
        sizer = _sizer()
        return BollCvdReclaimStrategy(config, sizer)

    def test_long_entry_unsplit_slow_middle(self):
        """LONG: BOLL15 tp_middle too close to entry, BOLL20 OK → UNSPLIT."""
        strategy = self._make_strategy()
        # LONG entry at 1900; required = 1900 * 1.002 = 1903.8
        # BOLL15 tp_middle=1901 < 1903.8 → insufficient
        # BOLL20 middle=1910 >= 1903.8 → sufficient
        boll = _boll(middle=1910.0, upper=2000.0, lower=1800.0, tp_middle=1901.0)
        cvd = _cvd()
        coord = _coordinator(strategy)

        intent = coord.open_position("LONG", "OPEN_LONG", 1900.0, 1000, boll, cvd, "base")

        self.assertEqual(intent.tp_plan, "MIDDLE_RUNNER")
        self.assertFalse(intent.middle_bucket_split_active)
        # middle_runner_first_tp_price should be BOLL20 middle
        self.assertAlmostEqual(intent.middle_runner_first_tp_price, 1910.0, places=4)
        # partial_tp_price should be BOLL20 middle
        self.assertAlmostEqual(intent.partial_tp_price, 1910.0, places=4)
        self.assertFalse(strategy.state.middle_bucket_split_active)

    def test_short_entry_unsplit_slow_middle(self):
        """SHORT: BOLL15 tp_middle too close to entry, BOLL20 OK → UNSPLIT."""
        strategy = self._make_strategy()
        # SHORT entry at 2100; required = 2100 * 0.998 = 2095.8
        # BOLL15 tp_middle=2097 > 2095.8 → insufficient
        # BOLL20 middle=2080 < 2095.8 → sufficient
        boll = _boll(middle=2080.0, upper=2150.0, lower=2050.0, tp_middle=2097.0)
        cvd = _cvd()
        coord = _coordinator(strategy)

        intent = coord.open_position("SHORT", "OPEN_SHORT", 2100.0, 1000, boll, cvd, "base")

        self.assertEqual(intent.tp_plan, "MIDDLE_RUNNER")
        self.assertFalse(intent.middle_bucket_split_active)
        self.assertAlmostEqual(intent.middle_runner_first_tp_price, 2080.0, places=4)

    def test_entry_unsplit_logs_skipped(self):
        """UNSPLIT on entry should log MIDDLE_BUCKET_SPLIT_SKIPPED for MIDDLE_RUNNER."""
        strategy = self._make_strategy()
        boll = _boll(middle=1910.0, upper=2000.0, lower=1800.0, tp_middle=1901.0)
        cvd = _cvd()
        coord = _coordinator(strategy)

        with self.assertLogs("src.strategies.middle_bucket_split_apply", level="WARNING") as logs:
            coord.open_position("LONG", "OPEN_LONG", 1900.0, 1000, boll, cvd, "base")
        output = "\n".join(logs.output)
        self.assertIn("MIDDLE_BUCKET_SPLIT_SKIPPED", output)
        self.assertIn("plan=MIDDLE_RUNNER", output)


if __name__ == "__main__":
    unittest.main()
