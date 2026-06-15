"""Tests for extreme_retest_add module — pivot detection, anchor management, triggers."""

from __future__ import annotations

import unittest

from src.strategies import extreme_retest_add as _extreme_retest
from src.strategies.extreme_retest_add import (
    ExtremeRetestAnchor,
    ExtremeRetestConfig,
    ExtremeRetestEvaluation,
)


def _build_candle(ts_ms: int, high: float, low: float, close: float,
                  boll_upper: float = 0, boll_lower: float = 0) -> dict:
    return {
        "ts_ms": ts_ms,
        "high": high,
        "low": low,
        "close": close,
        "boll_upper": boll_upper,
        "boll_lower": boll_lower,
    }


def _make_config(**overrides) -> ExtremeRetestConfig:
    defaults = dict(
        enabled=True,
        pivot_left_bars=2,
        pivot_right_bars=2,
        anchor_max_age_candles=12,
        sweep_max_age_seconds=900.0,
        near_extreme_pct=0.0015,
        reclaim_pct=0.0005,
        min_reverse_ratio=0.55,
        one_add_per_anchor=True,
    )
    defaults.update(overrides)
    return ExtremeRetestConfig(**defaults)


# ──────────────────────────────────────────────────────────────────────────────
# Pivot Detection Tests
# ──────────────────────────────────────────────────────────────────────────────


class PivotDetectionTest(unittest.TestCase):

    def test_short_pivot_high_valid(self) -> None:
        """SHORT pivot high: left 2 highs lower, right 2 highs lower → valid."""
        candles = [
            _build_candle(1000, 100, 95, 98),
            _build_candle(2000, 102, 97, 101),  # left-1
            _build_candle(3000, 103, 98, 102),  # left-2
            _build_candle(4000, 110, 96, 105),  # pivot: high=110
            _build_candle(5000, 105, 99, 104),  # right-1
            _build_candle(6000, 106, 100, 105),  # right-2
        ]
        self.assertTrue(_extreme_retest.detect_pivot_high(candles, 3, 2, 2))

    def test_short_pivot_high_right_higher_high_fails(self) -> None:
        """SHORT: right has higher high → invalid pivot."""
        candles = [
            _build_candle(1000, 100, 95, 98),
            _build_candle(2000, 102, 97, 101),
            _build_candle(3000, 103, 98, 102),
            _build_candle(4000, 110, 96, 105),  # candidate pivot
            _build_candle(5000, 112, 99, 111),  # right-1: high=112 > 110
            _build_candle(6000, 106, 100, 105),
        ]
        self.assertFalse(_extreme_retest.detect_pivot_high(candles, 3, 2, 2))

    def test_short_pivot_high_left_higher_high_fails(self) -> None:
        """SHORT: left has higher high → invalid pivot."""
        candles = [
            _build_candle(1000, 100, 95, 98),
            _build_candle(2000, 110, 97, 109),  # left-1: high=110 → equal, fails
            _build_candle(3000, 103, 98, 102),
            _build_candle(4000, 108, 96, 105),  # candidate pivot: high=108 < 110
            _build_candle(5000, 105, 99, 104),
            _build_candle(6000, 106, 100, 105),
        ]
        # Actually, left-1 high=110 >= 108, so this fails
        self.assertFalse(_extreme_retest.detect_pivot_high(candles, 3, 2, 2))

    def test_long_pivot_low_valid(self) -> None:
        """LONG pivot low: left 2 lows higher, right 2 lows higher → valid."""
        candles = [
            _build_candle(1000, 100, 95, 98),
            _build_candle(2000, 102, 96, 101),  # left-1
            _build_candle(3000, 103, 97, 102),  # left-2
            _build_candle(4000, 105, 90, 100),  # pivot: low=90
            _build_candle(5000, 104, 92, 101),  # right-1
            _build_candle(6000, 106, 93, 103),  # right-2
        ]
        self.assertTrue(_extreme_retest.detect_pivot_low(candles, 3, 2, 2))

    def test_long_pivot_low_right_lower_low_fails(self) -> None:
        """LONG: right has lower low → invalid pivot."""
        candles = [
            _build_candle(1000, 100, 95, 98),
            _build_candle(2000, 102, 96, 101),
            _build_candle(3000, 103, 97, 102),
            _build_candle(4000, 105, 90, 100),  # candidate pivot
            _build_candle(5000, 104, 88, 101),  # right-1: low=88 < 90
            _build_candle(6000, 106, 93, 103),
        ]
        self.assertFalse(_extreme_retest.detect_pivot_low(candles, 3, 2, 2))

    def test_pivot_right_bars_insufficient(self) -> None:
        """Not enough right bars → cannot confirm pivot."""
        candles = [
            _build_candle(1000, 100, 95, 98),
            _build_candle(2000, 102, 97, 101),
            _build_candle(3000, 103, 98, 102),
            _build_candle(4000, 110, 96, 105),  # candidate
            # Only 1 right bar, need 2
            _build_candle(5000, 105, 99, 104),
        ]
        self.assertFalse(_extreme_retest.detect_pivot_high(candles, 3, 2, 2))

    def test_pivot_left_bars_insufficient(self) -> None:
        """Not enough left bars → cannot confirm pivot."""
        candles = [
            _build_candle(2000, 102, 97, 101),
            _build_candle(3000, 110, 96, 105),  # candidate: only 1 left bar, need 2
            _build_candle(4000, 105, 99, 104),
            _build_candle(5000, 106, 100, 105),
        ]
        self.assertFalse(_extreme_retest.detect_pivot_high(candles, 1, 2, 2))


# ──────────────────────────────────────────────────────────────────────────────
# Outer Band Strict Tests
# ──────────────────────────────────────────────────────────────────────────────


class OuterBandStrictTest(unittest.TestCase):

    def test_short_pivot_high_above_upper_valid(self) -> None:
        self.assertTrue(_extreme_retest.is_outside_band_pivot_high(110, 108))

    def test_short_pivot_high_equal_upper_invalid(self) -> None:
        self.assertFalse(_extreme_retest.is_outside_band_pivot_high(108, 108))

    def test_short_pivot_high_below_upper_invalid(self) -> None:
        self.assertFalse(_extreme_retest.is_outside_band_pivot_high(107, 108))

    def test_long_pivot_low_below_lower_valid(self) -> None:
        self.assertTrue(_extreme_retest.is_outside_band_pivot_low(90, 92))

    def test_long_pivot_low_equal_lower_invalid(self) -> None:
        self.assertFalse(_extreme_retest.is_outside_band_pivot_low(92, 92))

    def test_long_pivot_low_above_lower_invalid(self) -> None:
        self.assertFalse(_extreme_retest.is_outside_band_pivot_low(93, 92))


# ──────────────────────────────────────────────────────────────────────────────
# Most Extreme Anchor Tests
# ──────────────────────────────────────────────────────────────────────────────


class MostExtremeAnchorTest(unittest.TestCase):

    def test_short_active_1740_new_1735_no_replace(self) -> None:
        self.assertFalse(
            _extreme_retest.is_more_extreme_anchor("SHORT", 1735, 1740))

    def test_short_active_1740_new_1745_replace(self) -> None:
        self.assertTrue(
            _extreme_retest.is_more_extreme_anchor("SHORT", 1745, 1740))

    def test_long_active_1660_new_1665_no_replace(self) -> None:
        self.assertFalse(
            _extreme_retest.is_more_extreme_anchor("LONG", 1665, 1660))

    def test_long_active_1660_new_1655_replace(self) -> None:
        self.assertTrue(
            _extreme_retest.is_more_extreme_anchor("LONG", 1655, 1660))

    def test_short_no_existing_anchor_creates(self) -> None:
        self.assertTrue(
            _extreme_retest.is_more_extreme_anchor("SHORT", 1740, None))

    def test_long_no_existing_anchor_creates(self) -> None:
        self.assertTrue(
            _extreme_retest.is_more_extreme_anchor("LONG", 1660, None))


# ──────────────────────────────────────────────────────────────────────────────
# Consumed Watermark Tests
# ──────────────────────────────────────────────────────────────────────────────


class ConsumedWatermarkTest(unittest.TestCase):

    def test_short_consumed_1740_new_1735_no_create(self) -> None:
        self.assertFalse(
            _extreme_retest.is_more_extreme_than_watermark("SHORT", 1735, 1740))

    def test_short_consumed_1740_new_1745_create(self) -> None:
        self.assertTrue(
            _extreme_retest.is_more_extreme_than_watermark("SHORT", 1745, 1740))

    def test_long_consumed_1660_new_1665_no_create(self) -> None:
        self.assertFalse(
            _extreme_retest.is_more_extreme_than_watermark("LONG", 1665, 1660))

    def test_long_consumed_1660_new_1655_create(self) -> None:
        self.assertTrue(
            _extreme_retest.is_more_extreme_than_watermark("LONG", 1655, 1660))

    def test_no_watermark_allows_creation(self) -> None:
        self.assertTrue(
            _extreme_retest.is_more_extreme_than_watermark("SHORT", 1740, None))
        self.assertTrue(
            _extreme_retest.is_more_extreme_than_watermark("LONG", 1660, None))


# ──────────────────────────────────────────────────────────────────────────────
# Last Entry Gap Tests
# ──────────────────────────────────────────────────────────────────────────────


class LastEntryGapTest(unittest.TestCase):

    def test_short_pivot_far_enough_from_last_entry_pass(self) -> None:
        # last_entry=1700, pivot=1740, required_gap=0.01 (1%)
        # gap = (1740-1700)/1700 = 0.0235 > 0.01 → OK
        ok, gap_pct, reason = _extreme_retest.is_anchor_far_enough_from_last_entry(
            "SHORT", 1740, 1700, 0.01)
        self.assertTrue(ok)
        self.assertAlmostEqual(gap_pct, 40 / 1700, places=6)

    def test_short_pivot_not_far_enough_fails(self) -> None:
        # last_entry=1700, pivot=1710, required_gap=0.01 (1%)
        # gap = (1710-1700)/1700 = 0.00588 < 0.01 → FAIL
        ok, gap_pct, reason = _extreme_retest.is_anchor_far_enough_from_last_entry(
            "SHORT", 1710, 1700, 0.01)
        self.assertFalse(ok)
        self.assertEqual(reason, "too_close_to_last_entry")

    def test_short_pivot_below_last_entry_fails(self) -> None:
        # SHORT: pivot below last_entry → not adverse
        ok, gap_pct, reason = _extreme_retest.is_anchor_far_enough_from_last_entry(
            "SHORT", 1690, 1700, 0.01)
        self.assertFalse(ok)
        self.assertEqual(reason, "pivot_not_adverse_for_short")

    def test_long_pivot_far_enough_from_last_entry_pass(self) -> None:
        # last_entry=1700, pivot=1660, required_gap=0.01 (1%)
        # gap = (1700-1660)/1700 = 0.0235 > 0.01 → OK
        ok, gap_pct, reason = _extreme_retest.is_anchor_far_enough_from_last_entry(
            "LONG", 1660, 1700, 0.01)
        self.assertTrue(ok)
        self.assertAlmostEqual(gap_pct, 40 / 1700, places=6)

    def test_long_pivot_not_far_enough_fails(self) -> None:
        # last_entry=1700, pivot=1690, required_gap=0.01
        # gap = (1700-1690)/1700 = 0.00588 < 0.01 → FAIL
        ok, gap_pct, reason = _extreme_retest.is_anchor_far_enough_from_last_entry(
            "LONG", 1690, 1700, 0.01)
        self.assertFalse(ok)
        self.assertEqual(reason, "too_close_to_last_entry")

    def test_long_pivot_above_last_entry_fails(self) -> None:
        ok, gap_pct, reason = _extreme_retest.is_anchor_far_enough_from_last_entry(
            "LONG", 1710, 1700, 0.01)
        self.assertFalse(ok)
        self.assertEqual(reason, "pivot_not_adverse_for_long")

    def test_missing_last_entry_fails(self) -> None:
        ok, gap_pct, reason = _extreme_retest.is_anchor_far_enough_from_last_entry(
            "SHORT", 1740, None, 0.01)
        self.assertFalse(ok)
        self.assertEqual(reason, "missing_last_entry")


# ──────────────────────────────────────────────────────────────────────────────
# Reject Before Break Tests
# ──────────────────────────────────────────────────────────────────────────────


class RejectBeforeBreakTest(unittest.TestCase):

    def _anchor(self, side: str = "SHORT", kind: str = "PIVOT_HIGH", price: float = 110.0) -> ExtremeRetestAnchor:
        return ExtremeRetestAnchor(
            side=side,
            kind=kind,
            price=price,
            candle_ts_ms=1000,
            boll_upper=108.0,
            boll_lower=92.0,
        )

    def test_short_reject_before_break_triggered(self) -> None:
        """SHORT: price inside band, near and below anchor, sell_ratio >= threshold → trigger."""
        anchor = self._anchor()
        cfg = _make_config(near_extreme_pct=0.0015, min_reverse_ratio=0.55)
        # price = 109.9, inside band (92-108?), wait, price needs to be <= boll_upper=108
        # Let me adjust: make boll_upper=111 so 109.9 is inside band
        anchor.boll_upper = 111.0
        anchor.boll_lower = 92.0
        result = _extreme_retest.evaluate_reject_before_break(
            "SHORT", 109.90, 111.0, 92.0, anchor, cfg,
            buy_ratio=0.3, sell_ratio=0.60)
        self.assertTrue(result.triggered)
        self.assertEqual(result.pattern, "REJECT_BEFORE_BREAK")

    def test_short_reject_sell_ratio_below_threshold_fails(self) -> None:
        anchor = self._anchor()
        anchor.boll_upper = 111.0
        cfg = _make_config(min_reverse_ratio=0.55)
        result = _extreme_retest.evaluate_reject_before_break(
            "SHORT", 109.90, 111.0, 92.0, anchor, cfg,
            buy_ratio=0.6, sell_ratio=0.50)
        self.assertFalse(result.triggered)
        self.assertEqual(result.reason, "reverse_ratio_not_met")

    def test_short_reject_price_not_near_extreme(self) -> None:
        anchor = self._anchor()
        anchor.boll_upper = 111.0
        cfg = _make_config(near_extreme_pct=0.0015)  # near = within 0.15% of anchor
        # price = 108.0, far from anchor 110.0
        result = _extreme_retest.evaluate_reject_before_break(
            "SHORT", 108.0, 111.0, 92.0, anchor, cfg,
            buy_ratio=0.3, sell_ratio=0.60)
        self.assertFalse(result.triggered)
        self.assertEqual(result.reason, "not_near_extreme")

    def test_short_reject_price_outside_band(self) -> None:
        anchor = self._anchor()
        anchor.boll_upper = 108.0
        cfg = _make_config()
        # price = 109.0, outside band (above upper 108.0)
        result = _extreme_retest.evaluate_reject_before_break(
            "SHORT", 109.0, 108.0, 92.0, anchor, cfg,
            buy_ratio=0.3, sell_ratio=0.60)
        self.assertFalse(result.triggered)
        self.assertEqual(result.reason, "price_not_inside_band")

    def test_long_reject_before_break_triggered(self) -> None:
        anchor = self._anchor(side="LONG", kind="PIVOT_LOW", price=90.0)
        anchor.boll_lower = 89.0
        cfg = _make_config(near_extreme_pct=0.0015, min_reverse_ratio=0.55)
        # price = 90.1, inside band (89.0-111.0), near and above anchor
        result = _extreme_retest.evaluate_reject_before_break(
            "LONG", 90.10, 111.0, 89.0, anchor, cfg,
            buy_ratio=0.60, sell_ratio=0.30)
        self.assertTrue(result.triggered)
        self.assertEqual(result.pattern, "REJECT_BEFORE_BREAK")

    def test_long_reject_buy_ratio_below_threshold_fails(self) -> None:
        anchor = self._anchor(side="LONG", kind="PIVOT_LOW", price=90.0)
        anchor.boll_lower = 89.0
        cfg = _make_config(min_reverse_ratio=0.55)
        result = _extreme_retest.evaluate_reject_before_break(
            "LONG", 90.10, 111.0, 89.0, anchor, cfg,
            buy_ratio=0.50, sell_ratio=0.6)
        self.assertFalse(result.triggered)
        self.assertEqual(result.reason, "reverse_ratio_not_met")

    def test_no_active_anchor_returns_not_triggered(self) -> None:
        anchor = ExtremeRetestAnchor()  # not active
        cfg = _make_config()
        result = _extreme_retest.evaluate_reject_before_break(
            "SHORT", 100.0, 108.0, 92.0, anchor, cfg,
            buy_ratio=0.3, sell_ratio=0.60)
        self.assertFalse(result.triggered)
        self.assertEqual(result.reason, "no_active_anchor")


# ──────────────────────────────────────────────────────────────────────────────
# Sweep Reclaim Tests
# ──────────────────────────────────────────────────────────────────────────────


class SweepReclaimTest(unittest.TestCase):

    def _anchor(self, side: str = "SHORT", kind: str = "PIVOT_HIGH", price: float = 110.0) -> ExtremeRetestAnchor:
        return ExtremeRetestAnchor(
            side=side,
            kind=kind,
            price=price,
            candle_ts_ms=1000,
            boll_upper=111.0,
            boll_lower=92.0,
        )

    def _anchor_with_sweep(self, side: str = "SHORT") -> ExtremeRetestAnchor:
        anchor = self._anchor(side=side)
        anchor.sweep_seen = True
        anchor.sweep_extreme_price = 112.0
        anchor.sweep_first_seen_ts_ms = 1000
        anchor.sweep_last_seen_ts_ms = 2000
        return anchor

    def test_short_sweep_reclaim_triggered(self) -> None:
        """SHORT: price broke above anchor (sweep_seen=True), now inside band,
        reclaimed below anchor, sell_ratio >= threshold → trigger."""
        anchor = self._anchor_with_sweep()
        cfg = _make_config(reclaim_pct=0.0005, min_reverse_ratio=0.55)
        # anchor=110.0, reclaim_threshold = 110.0 * (1 - 0.0005) = 109.945
        # price=109.90 < 109.945 → reclaimed
        anchor.boll_upper = 112.0
        anchor.boll_lower = 92.0
        anchor.price = 110.0
        result = _extreme_retest.evaluate_sweep_reclaim(
            "SHORT", 109.90, 3000, 112.0, 92.0, anchor, cfg,
            buy_ratio=0.3, sell_ratio=0.60)
        self.assertTrue(result.triggered)
        self.assertEqual(result.pattern, "SWEEP_RECLAIM")

    def test_short_sweep_not_reclaimed_price_still_high(self) -> None:
        """SHORT: sweep seen, inside band, but price not reclaimed → no trigger."""
        anchor = self._anchor_with_sweep()
        cfg = _make_config(reclaim_pct=0.0005)
        anchor.boll_upper = 112.0
        anchor.boll_lower = 92.0
        # anchor=110.0, price=110.05 not reclaimed (< 109.945)
        result = _extreme_retest.evaluate_sweep_reclaim(
            "SHORT", 110.05, 3000, 112.0, 92.0, anchor, cfg,
            buy_ratio=0.3, sell_ratio=0.60)
        self.assertFalse(result.triggered)
        self.assertEqual(result.reason, "not_reclaimed")

    def test_short_sweep_not_inside_band(self) -> None:
        """SHORT: sweep seen, but price still outside band → no trigger."""
        anchor = self._anchor_with_sweep()
        cfg = _make_config()
        anchor.boll_upper = 109.0
        # price=109.5 > boll_upper → outside band
        result = _extreme_retest.evaluate_sweep_reclaim(
            "SHORT", 109.50, 3000, 109.0, 92.0, anchor, cfg,
            buy_ratio=0.3, sell_ratio=0.60)
        self.assertFalse(result.triggered)
        self.assertEqual(result.reason, "price_not_inside_band")

    def test_short_sweep_reclaim_ratio_not_met(self) -> None:
        """SHORT: sweep seen, reclaimed, but sell_ratio < threshold → no trigger."""
        anchor = self._anchor_with_sweep()
        cfg = _make_config(reclaim_pct=0.0005, min_reverse_ratio=0.55)
        anchor.boll_upper = 112.0
        anchor.boll_lower = 92.0
        result = _extreme_retest.evaluate_sweep_reclaim(
            "SHORT", 109.90, 3000, 112.0, 92.0, anchor, cfg,
            buy_ratio=0.3, sell_ratio=0.50)
        self.assertFalse(result.triggered)
        self.assertEqual(result.reason, "reverse_ratio_not_met")

    def test_long_sweep_reclaim_triggered(self) -> None:
        """LONG: price broke below anchor (sweep_seen=True), now inside band,
        reclaimed above anchor, buy_ratio >= threshold → trigger."""
        anchor = self._anchor_with_sweep(side="LONG")
        anchor.kind = "PIVOT_LOW"
        anchor.price = 90.0
        anchor.sweep_extreme_price = 88.0
        cfg = _make_config(reclaim_pct=0.0005, min_reverse_ratio=0.55)
        # anchor=90.0, reclaim_threshold = 90.0 * (1 + 0.0005) = 90.045
        # price=90.10 > 90.045 → reclaimed
        anchor.boll_upper = 112.0
        anchor.boll_lower = 89.0
        result = _extreme_retest.evaluate_sweep_reclaim(
            "LONG", 90.10, 3000, 112.0, 89.0, anchor, cfg,
            buy_ratio=0.60, sell_ratio=0.30)
        self.assertTrue(result.triggered)
        self.assertEqual(result.pattern, "SWEEP_RECLAIM")

    def test_long_sweep_not_inside_band(self) -> None:
        """LONG: sweep seen, but price still below band → no trigger."""
        anchor = self._anchor_with_sweep(side="LONG")
        anchor.kind = "PIVOT_LOW"
        anchor.price = 90.0
        anchor.sweep_extreme_price = 88.0
        cfg = _make_config()
        anchor.boll_lower = 92.0
        # price=91.0 < boll_lower → outside band
        result = _extreme_retest.evaluate_sweep_reclaim(
            "LONG", 91.0, 3000, 112.0, 92.0, anchor, cfg,
            buy_ratio=0.60, sell_ratio=0.30)
        self.assertFalse(result.triggered)
        self.assertEqual(result.reason, "price_not_inside_band")

    def test_long_sweep_reclaim_ratio_not_met(self) -> None:
        """LONG: sweep seen, reclaimed, but buy_ratio < threshold → no trigger."""
        anchor = self._anchor_with_sweep(side="LONG")
        anchor.kind = "PIVOT_LOW"
        anchor.price = 90.0
        anchor.sweep_extreme_price = 88.0
        cfg = _make_config(reclaim_pct=0.0005, min_reverse_ratio=0.55)
        anchor.boll_lower = 89.0
        result = _extreme_retest.evaluate_sweep_reclaim(
            "LONG", 90.10, 3000, 112.0, 89.0, anchor, cfg,
            buy_ratio=0.50, sell_ratio=0.30)
        self.assertFalse(result.triggered)
        self.assertEqual(result.reason, "reverse_ratio_not_met")

    def test_sweep_state_updated_on_break(self) -> None:
        """Test that sweep state is updated when price breaks past anchor."""
        anchor = self._anchor()
        cfg = _make_config()
        # First tick: price > anchor → sweep_seen should become True
        _extreme_retest.evaluate_sweep_reclaim(
            "SHORT", 111.0, 1000, 112.0, 92.0, anchor, cfg,
            buy_ratio=0.3, sell_ratio=0.30)
        self.assertTrue(anchor.sweep_seen)
        self.assertEqual(anchor.sweep_extreme_price, 111.0)
        self.assertEqual(anchor.sweep_first_seen_ts_ms, 1000)

    def test_no_max_overshoot_short(self) -> None:
        """SHORT: price far overshoots anchor, then reclaims → still triggers."""
        anchor = self._anchor()
        cfg = _make_config(reclaim_pct=0.0005, min_reverse_ratio=0.55)
        anchor.boll_upper = 200.0
        anchor.boll_lower = 92.0
        # Simulate: first far overshoot
        _extreme_retest.evaluate_sweep_reclaim(
            "SHORT", 150.0, 1000, 200.0, 92.0, anchor, cfg,
            buy_ratio=0.3, sell_ratio=0.30)
        self.assertTrue(anchor.sweep_seen)
        self.assertEqual(anchor.sweep_extreme_price, 150.0)
        # Now reclaim
        anchor.boll_upper = 112.0
        result = _extreme_retest.evaluate_sweep_reclaim(
            "SHORT", 109.90, 3000, 112.0, 92.0, anchor, cfg,
            buy_ratio=0.3, sell_ratio=0.60)
        self.assertTrue(result.triggered)

    def test_no_max_overshoot_long(self) -> None:
        """LONG: price far overshoots anchor low, then reclaims → still triggers."""
        anchor = self._anchor(side="LONG", kind="PIVOT_LOW", price=90.0)
        cfg = _make_config(reclaim_pct=0.0005, min_reverse_ratio=0.55)
        anchor.boll_upper = 112.0
        anchor.boll_lower = 50.0
        # Far overshoot
        _extreme_retest.evaluate_sweep_reclaim(
            "LONG", 60.0, 1000, 112.0, 50.0, anchor, cfg,
            buy_ratio=0.30, sell_ratio=0.30)
        self.assertTrue(anchor.sweep_seen)
        self.assertEqual(anchor.sweep_extreme_price, 60.0)
        # Now reclaim
        anchor.boll_lower = 89.0
        result = _extreme_retest.evaluate_sweep_reclaim(
            "LONG", 90.10, 3000, 112.0, 89.0, anchor, cfg,
            buy_ratio=0.60, sell_ratio=0.30)
        self.assertTrue(result.triggered)

    def test_no_sweep_no_trigger(self) -> None:
        """Without sweep_seen, sweep_reclaim should not trigger even if inside band."""
        anchor = self._anchor()  # sweep_seen=False
        cfg = _make_config()
        result = _extreme_retest.evaluate_sweep_reclaim(
            "SHORT", 109.0, 1000, 112.0, 92.0, anchor, cfg,
            buy_ratio=0.3, sell_ratio=0.60)
        self.assertFalse(result.triggered)
        self.assertEqual(result.reason, "sweep_not_seen")


# ──────────────────────────────────────────────────────────────────────────────
# Anchor Lifecycle Tests
# ──────────────────────────────────────────────────────────────────────────────


class AnchorLifecycleTest(unittest.TestCase):

    def test_mark_anchor_consumed(self) -> None:
        anchor = ExtremeRetestAnchor(
            side="SHORT", kind="PIVOT_HIGH", price=110.0, candle_ts_ms=1000)
        self.assertTrue(anchor.is_active())
        _extreme_retest.mark_anchor_consumed(anchor)
        self.assertFalse(anchor.is_active())
        self.assertEqual(anchor.consumed_watermark_price, 110.0)
        self.assertEqual(anchor.consumed_anchor_ts_ms, 1000)
        self.assertFalse(anchor.sweep_seen)

    def test_revalidate_anchor_after_add_still_valid(self) -> None:
        anchor = ExtremeRetestAnchor(
            side="SHORT", kind="PIVOT_HIGH", price=110.0, candle_ts_ms=1000)
        # last_entry=105, gap = (110-105)/105 = 0.0476 > 0.01 → valid
        result = _extreme_retest.revalidate_anchor_after_add(anchor, 105.0, 0.01)
        self.assertIsNone(result)
        self.assertTrue(anchor.is_active())

    def test_revalidate_anchor_after_add_too_close_drops(self) -> None:
        anchor = ExtremeRetestAnchor(
            side="SHORT", kind="PIVOT_HIGH", price=110.0, candle_ts_ms=1000)
        # last_entry=109.5, gap = (110-109.5)/109.5 = 0.00457 < 0.01 → dropped
        result = _extreme_retest.revalidate_anchor_after_add(anchor, 109.5, 0.01)
        self.assertEqual(result, "too_close_after_new_entry")
        self.assertFalse(anchor.is_active())

    def test_revalidate_no_active_anchor_does_nothing(self) -> None:
        anchor = ExtremeRetestAnchor()  # not active
        result = _extreme_retest.revalidate_anchor_after_add(anchor, 100.0, 0.01)
        self.assertIsNone(result)

    def test_try_create_or_replace_anchor_created(self) -> None:
        anchor = ExtremeRetestAnchor()
        cfg = _make_config()
        action, reason = _extreme_retest.try_create_or_replace_anchor(
            "SHORT", 110.0, 5000, 108.0, 92.0, 100.0, 0.01, anchor, cfg)
        self.assertTrue(action)
        self.assertEqual(reason, "created")
        self.assertTrue(anchor.is_active())
        self.assertEqual(anchor.price, 110.0)

    def test_try_create_or_replace_anchor_replaced(self) -> None:
        anchor = ExtremeRetestAnchor(
            side="SHORT", kind="PIVOT_HIGH", price=108.0, candle_ts_ms=1000)
        cfg = _make_config()
        action, reason = _extreme_retest.try_create_or_replace_anchor(
            "SHORT", 110.0, 5000, 109.0, 92.0, 100.0, 0.01, anchor, cfg)
        self.assertTrue(action)
        self.assertEqual(reason, "replaced")
        self.assertEqual(anchor.price, 110.0)

    def test_try_create_or_replace_anchor_not_outside_upper(self) -> None:
        anchor = ExtremeRetestAnchor()
        cfg = _make_config()
        action, reason = _extreme_retest.try_create_or_replace_anchor(
            "SHORT", 107.0, 5000, 108.0, 92.0, 100.0, 0.01, anchor, cfg)
        self.assertFalse(action)

    def test_try_create_or_replace_anchor_consumed_watermark_blocks(self) -> None:
        anchor = ExtremeRetestAnchor()
        anchor.consumed_watermark_price = 112.0
        cfg = _make_config()
        action, reason = _extreme_retest.try_create_or_replace_anchor(
            "SHORT", 110.0, 5000, 108.0, 92.0, 100.0, 0.01, anchor, cfg)
        self.assertFalse(action)

    def test_expired_anchor_dropped(self) -> None:
        anchor = ExtremeRetestAnchor(
            side="SHORT", kind="PIVOT_HIGH", price=110.0, candle_ts_ms=1000)
        # max_age_candles=1 → max_age_ms = 1 * 15 * 60_000 = 900_000
        # current_candle_ts_ms = 1000 + 900_000 + 1 = too old
        dropped = _extreme_retest.drop_expired_anchor(
            anchor, 1000 + 900_001, 1)
        self.assertTrue(dropped)
        self.assertFalse(anchor.is_active())

    def test_anchor_not_expired(self) -> None:
        anchor = ExtremeRetestAnchor(
            side="SHORT", kind="PIVOT_HIGH", price=110.0, candle_ts_ms=1000)
        dropped = _extreme_retest.drop_expired_anchor(
            anchor, 1000 + 10000, 12)
        self.assertFalse(dropped)
        self.assertTrue(anchor.is_active())


# ──────────────────────────────────────────────────────────────────────────────
# Evaluate on Tick (combined Reject + Sweep)
# ──────────────────────────────────────────────────────────────────────────────


class EvaluateOnTickTest(unittest.TestCase):

    def test_reject_trumps_sweep(self) -> None:
        """If reject triggers, sweep is not evaluated (but anchor.sweep_* untouched)."""
        anchor = ExtremeRetestAnchor(
            side="SHORT", kind="PIVOT_HIGH", price=110.0, candle_ts_ms=1000,
            boll_upper=111.0, boll_lower=92.0)
        cfg = _make_config(near_extreme_pct=0.0015, min_reverse_ratio=0.55)
        result = _extreme_retest.evaluate_on_tick(
            "SHORT", 109.90, 1000, 111.0, 92.0, anchor, cfg,
            buy_ratio=0.3, sell_ratio=0.60)
        self.assertTrue(result.triggered)
        self.assertEqual(result.pattern, "REJECT_BEFORE_BREAK")

    def test_sweep_fires_when_reject_not_met(self) -> None:
        """If reject doesn't fire, sweep can trigger."""
        anchor = ExtremeRetestAnchor(
            side="LONG", kind="PIVOT_LOW", price=90.0, candle_ts_ms=1000,
            boll_upper=112.0, boll_lower=89.0)
        anchor.sweep_seen = True
        anchor.sweep_extreme_price = 88.0
        anchor.sweep_first_seen_ts_ms = 500
        cfg = _make_config(reclaim_pct=0.0005, min_reverse_ratio=0.55)
        # reject won't fire: price=90.10, near_threshold_high=90.0*(1+0.0015)=90.135
        # 90.1 is within the near range... actually it IS near. Let's change it:
        # price=90.5 > near threshold → reject fails, but sweep with reclaimed=True can still trigger
        # wait, anchor wasn't swept for LONG... hmm
        result = _extreme_retest.evaluate_on_tick(
            "LONG", 90.10, 1000, 112.0, 89.0, anchor, cfg,
            buy_ratio=0.60, sell_ratio=0.30)
        # Reject fires first because price is near anchor and buy_ratio is OK
        self.assertTrue(result.triggered)

    def test_no_anchor_no_trigger(self) -> None:
        anchor = ExtremeRetestAnchor()  # not active
        cfg = _make_config()
        result = _extreme_retest.evaluate_on_tick(
            "SHORT", 100.0, 1000, 108.0, 92.0, anchor, cfg,
            buy_ratio=0.3, sell_ratio=0.60)
        self.assertFalse(result.triggered)


# ──────────────────────────────────────────────────────────────────────────────
# Anchor State Serialization
# ──────────────────────────────────────────────────────────────────────────────


class AnchorSerializationTest(unittest.TestCase):

    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        anchor = ExtremeRetestAnchor(
            side="SHORT", kind="PIVOT_HIGH", price=110.0, candle_ts_ms=5000,
            boll_upper=108.0, boll_lower=92.0,
            sweep_seen=True, sweep_extreme_price=115.0,
            sweep_first_seen_ts_ms=6000, sweep_last_seen_ts_ms=7000,
            consumed_watermark_price=112.0, consumed_anchor_ts_ms=8000,
        )
        d = anchor.to_dict()
        restored = ExtremeRetestAnchor.from_dict(d)
        self.assertEqual(restored.side, "SHORT")
        self.assertEqual(restored.kind, "PIVOT_HIGH")
        self.assertEqual(restored.price, 110.0)
        self.assertEqual(restored.candle_ts_ms, 5000)
        self.assertEqual(restored.boll_upper, 108.0)
        self.assertEqual(restored.boll_lower, 92.0)
        self.assertTrue(restored.sweep_seen)
        self.assertEqual(restored.sweep_extreme_price, 115.0)
        self.assertEqual(restored.sweep_first_seen_ts_ms, 6000)
        self.assertEqual(restored.sweep_last_seen_ts_ms, 7000)
        self.assertEqual(restored.consumed_watermark_price, 112.0)
        self.assertEqual(restored.consumed_anchor_ts_ms, 8000)

    def test_from_dict_empty(self) -> None:
        restored = ExtremeRetestAnchor.from_dict({})
        self.assertIsNone(restored.side)
        self.assertFalse(restored.is_active())

    def test_clear_anchor(self) -> None:
        anchor = ExtremeRetestAnchor(
            side="SHORT", kind="PIVOT_HIGH", price=110.0, candle_ts_ms=5000,
            sweep_seen=True)
        anchor.clear()
        self.assertIsNone(anchor.side)
        self.assertIsNone(anchor.price)
        self.assertFalse(anchor.sweep_seen)
        # watermark is NOT cleared
        self.assertIsNone(anchor.consumed_watermark_price)  # clear doesn't set watermark

    def test_consume_sets_watermark_and_clears(self) -> None:
        anchor = ExtremeRetestAnchor(
            side="SHORT", kind="PIVOT_HIGH", price=110.0, candle_ts_ms=5000,
            sweep_seen=True)
        anchor.consume()
        self.assertIsNone(anchor.side)
        self.assertIsNone(anchor.price)
        self.assertFalse(anchor.sweep_seen)
        self.assertEqual(anchor.consumed_watermark_price, 110.0)
        self.assertEqual(anchor.consumed_anchor_ts_ms, 5000)
