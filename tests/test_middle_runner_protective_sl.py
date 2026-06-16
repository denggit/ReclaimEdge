"""Tests for the new relaxed Middle Runner protective SL logic.

Verifies:
- No interpolation toward middle.
- No clamp to middle.
- LONG: protective_sl = max(cost_line, boll_lower).
- SHORT: protective_sl = min(cost_line, boll_upper).
- sl_tighten_ratio is ignored.
- Invalid current_price returns the correct reason.
- Missing cost basis returns missing_cost_basis.
"""

from __future__ import annotations

import unittest

from src.strategies.middle_runner import (
    calculate_middle_runner_protective_sl,
)


class TestMiddleRunnerProtectiveSlLongNew(unittest.TestCase):
    """LONG side — new relaxed logic."""

    def _call(self, **overrides):
        defaults = dict(
            side="LONG",
            current_price=120.0,
            avg_entry_price=100.0,
            net_remaining_breakeven_price=101.0,
            breakeven_fee_buffer_pct=0.001,
            boll_middle=105.0,
            boll_upper=115.0,
            boll_lower=95.0,
            sl_tighten_ratio=0.50,
        )
        defaults.update(overrides)
        return calculate_middle_runner_protective_sl(**defaults)

    # ── 1. cost_line > boll_lower → protective_sl = cost_line ─────────
    def test_cost_above_lower_uses_cost(self) -> None:
        decision = self._call(
            net_remaining_breakeven_price=101.0,
            boll_lower=95.0,
        )
        self.assertEqual(decision.reason, "calculated")
        self.assertEqual(decision.protective_sl, 101.0)
        self.assertEqual(decision.candidate_cost, 101.0)
        self.assertEqual(decision.candidate_structure, 95.0)

    # ── 2. boll_lower > cost_line → protective_sl = boll_lower ────────
    def test_lower_above_cost_uses_lower(self) -> None:
        decision = self._call(
            net_remaining_breakeven_price=90.0,
            boll_lower=98.0,
        )
        self.assertEqual(decision.reason, "calculated")
        self.assertEqual(decision.protective_sl, 98.0)
        self.assertEqual(decision.candidate_cost, 90.0)
        self.assertEqual(decision.candidate_structure, 98.0)

    # ── 3. no interpolation to middle ─────────────────────────────────
    def test_no_interpolation_to_middle(self) -> None:
        """candidate_cost and candidate_structure are raw values,
        NOT interpolated toward middle."""
        decision = self._call(
            net_remaining_breakeven_price=101.0,
            boll_middle=105.0,
            boll_lower=95.0,
            sl_tighten_ratio=0.80,  # was used for interpolation; now ignored
        )
        self.assertEqual(decision.candidate_cost, 101.0)
        self.assertEqual(decision.candidate_structure, 95.0)
        # With old logic (ratio=0.80): cost = 101 + (105-101)*0.80 = 104.2
        # structure = 95 + (105-95)*0.80 = 103.0
        # Neither should appear — they should be raw.
        self.assertNotAlmostEqual(decision.candidate_cost, 104.2, places=4)
        self.assertNotAlmostEqual(decision.candidate_structure, 103.0, places=4)

    # ── 4. no clamp to middle ─────────────────────────────────────────
    def test_no_clamp_to_middle(self) -> None:
        """protective_sl can exceed boll_middle (LONG)."""
        decision = self._call(
            net_remaining_breakeven_price=107.0,
            boll_middle=105.0,
            boll_lower=95.0,
            current_price=110.0,
        )
        self.assertEqual(decision.reason, "calculated")
        # protective_sl = max(107, 95) = 107, which is above middle=105
        # Old logic would clamp to 105.
        self.assertEqual(decision.protective_sl, 107.0)
        self.assertGreater(float(decision.protective_sl), 105.0)

    # ── 5. sl_tighten_ratio does not change protective_sl ──────────────
    def test_sl_tighten_ratio_ignored(self) -> None:
        d1 = self._call(sl_tighten_ratio=0.30)
        d2 = self._call(sl_tighten_ratio=0.80)
        self.assertEqual(d1.protective_sl, d2.protective_sl)
        self.assertEqual(d1.candidate_cost, d2.candidate_cost)
        self.assertEqual(d1.candidate_structure, d2.candidate_structure)

    # ── 6. current_price invalid → long_sl_not_below_current ──────────
    def test_sl_not_below_current(self) -> None:
        decision = self._call(
            current_price=100.0,
            net_remaining_breakeven_price=101.0,
            boll_lower=95.0,
        )
        self.assertEqual(decision.reason, "long_sl_not_below_current")
        self.assertIsNone(decision.protective_sl)

    # ── 7. missing cost basis ─────────────────────────────────────────
    def test_missing_cost_basis(self) -> None:
        decision = self._call(
            current_price=0.0,
            net_remaining_breakeven_price=0.0,
            avg_entry_price=0.0,
        )
        self.assertEqual(decision.reason, "missing_cost_basis")
        self.assertIsNone(decision.protective_sl)

    # ── 8. fallback to avg_entry when net_remaining_breakeven <= 0 ────
    def test_fallback_to_avg_entry(self) -> None:
        decision = self._call(
            net_remaining_breakeven_price=0.0,
            avg_entry_price=100.0,
            boll_lower=95.0,
            current_price=105.0,
        )
        self.assertEqual(decision.reason, "calculated")
        # cost_line = 100 * (1 + 0.001) = 100.1
        expected_cost = 100.0 * (1 + 0.001)
        self.assertAlmostEqual(decision.candidate_cost, expected_cost, places=4)
        # protective_sl = max(100.1, 95.0) = 100.1
        self.assertAlmostEqual(float(decision.protective_sl), expected_cost, places=4)


class TestMiddleRunnerProtectiveSlShortNew(unittest.TestCase):
    """SHORT side — new relaxed logic."""

    def _call(self, **overrides):
        defaults = dict(
            side="SHORT",
            current_price=80.0,
            avg_entry_price=100.0,
            net_remaining_breakeven_price=99.0,
            breakeven_fee_buffer_pct=0.001,
            boll_middle=95.0,
            boll_upper=105.0,
            boll_lower=85.0,
            sl_tighten_ratio=0.50,
        )
        defaults.update(overrides)
        return calculate_middle_runner_protective_sl(**defaults)

    # ── 1. cost_line < boll_upper → protective_sl = cost_line ─────────
    def test_cost_below_upper_uses_cost(self) -> None:
        decision = self._call(
            net_remaining_breakeven_price=99.0,
            boll_upper=105.0,
        )
        self.assertEqual(decision.reason, "calculated")
        self.assertEqual(decision.protective_sl, 99.0)
        self.assertEqual(decision.candidate_cost, 99.0)
        self.assertEqual(decision.candidate_structure, 105.0)

    # ── 2. boll_upper < cost_line → protective_sl = boll_upper ────────
    def test_upper_below_cost_uses_upper(self) -> None:
        decision = self._call(
            net_remaining_breakeven_price=110.0,
            boll_upper=105.0,
        )
        self.assertEqual(decision.reason, "calculated")
        self.assertEqual(decision.protective_sl, 105.0)
        self.assertEqual(decision.candidate_cost, 110.0)
        self.assertEqual(decision.candidate_structure, 105.0)

    # ── 3. no interpolation to middle ─────────────────────────────────
    def test_no_interpolation_to_middle(self) -> None:
        decision = self._call(
            net_remaining_breakeven_price=99.0,
            boll_middle=95.0,
            boll_upper=105.0,
            sl_tighten_ratio=0.80,
        )
        self.assertEqual(decision.candidate_cost, 99.0)
        self.assertEqual(decision.candidate_structure, 105.0)
        # Old logic would interpolate toward middle: cost = 99 + (95-99)*0.80 = 95.8
        # structure = 105 + (95-105)*0.80 = 97.0
        self.assertNotAlmostEqual(decision.candidate_cost, 95.8, places=4)
        self.assertNotAlmostEqual(decision.candidate_structure, 97.0, places=4)

    # ── 4. no clamp to middle ─────────────────────────────────────────
    def test_no_clamp_to_middle(self) -> None:
        """protective_sl can go below boll_middle (SHORT)."""
        decision = self._call(
            net_remaining_breakeven_price=93.0,
            boll_middle=95.0,
            boll_upper=105.0,
            current_price=80.0,
        )
        self.assertEqual(decision.reason, "calculated")
        # protective_sl = min(93, 105) = 93, which is below middle=95
        # Old logic would clamp to 95.
        self.assertEqual(decision.protective_sl, 93.0)
        self.assertLess(float(decision.protective_sl), 95.0)

    # ── 5. sl_tighten_ratio does not change protective_sl ──────────────
    def test_sl_tighten_ratio_ignored(self) -> None:
        d1 = self._call(sl_tighten_ratio=0.30)
        d2 = self._call(sl_tighten_ratio=0.80)
        self.assertEqual(d1.protective_sl, d2.protective_sl)
        self.assertEqual(d1.candidate_cost, d2.candidate_cost)
        self.assertEqual(d1.candidate_structure, d2.candidate_structure)

    # ── 6. current_price invalid → short_sl_not_above_current ─────────
    def test_sl_not_above_current(self) -> None:
        decision = self._call(
            current_price=100.0,
            net_remaining_breakeven_price=99.0,
            boll_upper=105.0,
        )
        self.assertEqual(decision.reason, "short_sl_not_above_current")
        self.assertIsNone(decision.protective_sl)

    # ── 7. missing cost basis ─────────────────────────────────────────
    def test_missing_cost_basis(self) -> None:
        decision = self._call(
            current_price=0.0,
            net_remaining_breakeven_price=0.0,
            avg_entry_price=0.0,
        )
        self.assertEqual(decision.reason, "missing_cost_basis")
        self.assertIsNone(decision.protective_sl)


if __name__ == "__main__":
    unittest.main()
