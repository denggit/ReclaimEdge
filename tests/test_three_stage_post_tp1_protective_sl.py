"""Tests for the new relaxed Three-Stage post-TP1 protective SL logic.

Verifies:
- No interpolation toward middle.
- No clamp to middle.
- LONG: protective_sl = max(cost_line, boll_lower).
- SHORT: protective_sl = min(cost_line, boll_upper).
- net_remaining_breakeven_price > 0 → used directly as candidate_cost.
- net_remaining_breakeven_price <= 0 → fallback formula preserved.
- sl_tighten_ratio is ignored.
- Invalid inputs (missing tp1_price, invalid tp1_ratio, zero avg_entry)
  still return the correct reason.
- Invalid current_price returns long_sl_not_below_current / short_sl_not_above_current.
"""

from __future__ import annotations

import unittest

from src.strategies.three_stage_runner import (
    calculate_three_stage_post_tp1_protective_sl,
)


class TestThreeStagePostTp1ProtectiveSlLongNew(unittest.TestCase):
    """LONG side — new relaxed logic."""

    def _call(self, **overrides):
        defaults = dict(
            side="LONG",
            current_price=120.0,
            avg_entry_price=100.0,
            net_remaining_breakeven_price=101.0,  # > 0 → used directly
            breakeven_fee_buffer_pct=0.001,
            tp1_price=105.0,
            tp1_ratio=0.6,
            boll_middle=110.0,
            boll_upper=120.0,
            boll_lower=90.0,
            sl_tighten_ratio=0.50,
        )
        defaults.update(overrides)
        return calculate_three_stage_post_tp1_protective_sl(**defaults)

    # ── 1. net_remaining_breakeven > 0 → used directly as candidate_cost
    def test_net_breakeven_used_directly(self) -> None:
        decision = self._call(net_remaining_breakeven_price=102.0)
        self.assertEqual(decision.reason, "calculated")
        self.assertEqual(decision.candidate_cost, 102.0)
        self.assertEqual(decision.candidate_structure, 90.0)
        # protective_sl = max(102, 90) = 102
        self.assertEqual(decision.protective_sl, 102.0)

    # ── 2. candidate_structure = boll_lower ────────────────────────────
    def test_candidate_structure_is_boll_lower(self) -> None:
        decision = self._call(boll_lower=88.0)
        self.assertEqual(decision.candidate_structure, 88.0)

    # ── 3. protective_sl = max(candidate_cost, boll_lower) ─────────────
    def test_max_of_cost_and_lower(self) -> None:
        # cost < lower → use lower
        decision = self._call(
            net_remaining_breakeven_price=85.0,
            boll_lower=92.0,
        )
        self.assertEqual(decision.reason, "calculated")
        self.assertEqual(decision.protective_sl, 92.0)

    # ── 4. no interpolation to middle ─────────────────────────────────
    def test_no_interpolation_to_middle(self) -> None:
        decision = self._call(
            net_remaining_breakeven_price=101.0,
            boll_middle=110.0,
            boll_lower=90.0,
            sl_tighten_ratio=0.80,
        )
        self.assertEqual(decision.candidate_cost, 101.0)
        self.assertEqual(decision.candidate_structure, 90.0)
        # Old logic with ratio=0.80: cost = 101 + (110-101)*0.80 = 108.2
        # structure = 90 + (110-90)*0.80 = 106.0
        self.assertNotAlmostEqual(decision.candidate_cost, 108.2, places=2)
        self.assertNotAlmostEqual(decision.candidate_structure, 106.0, places=2)

    # ── 5. no clamp to middle ─────────────────────────────────────────
    def test_no_clamp_to_middle(self) -> None:
        decision = self._call(
            net_remaining_breakeven_price=112.0,
            boll_middle=110.0,
            boll_lower=90.0,
            current_price=115.0,
        )
        self.assertEqual(decision.reason, "calculated")
        # protective_sl = max(112, 90) = 112, which is above middle=110
        # Old logic would clamp to 110.
        self.assertEqual(decision.protective_sl, 112.0)
        self.assertGreater(float(decision.protective_sl), 110.0)

    # ── 6. sl_tighten_ratio ignored ────────────────────────────────────
    def test_sl_tighten_ratio_ignored(self) -> None:
        d1 = self._call(sl_tighten_ratio=0.30)
        d2 = self._call(sl_tighten_ratio=0.80)
        self.assertEqual(d1.protective_sl, d2.protective_sl)
        self.assertEqual(d1.candidate_cost, d2.candidate_cost)
        self.assertEqual(d1.candidate_structure, d2.candidate_structure)

    # ── 7. fallback formula when net_remaining_breakeven <= 0 ───────────
    def test_fallback_formula_when_no_net_breakeven(self) -> None:
        decision = self._call(
            net_remaining_breakeven_price=0.0,
            avg_entry_price=100.0,
            tp1_price=105.0,
            tp1_ratio=0.6,
            boll_lower=90.0,
        )
        self.assertEqual(decision.reason, "calculated")
        # post_tp1_breakeven = 100 - 0.6*(105-100)/(1-0.6) = 100 - 7.5 = 92.5
        # buffered = 92.5 * 1.001 = 92.5925
        # candidate_cost = 92.5925, candidate_structure = 90.0
        # protective_sl = max(92.5925, 90.0) = 92.5925
        expected_be = 100.0 - 0.6 * (105.0 - 100.0) / (1 - 0.6)  # = 92.5
        expected_buffered = expected_be * (1 + 0.001)  # = 92.5925
        self.assertAlmostEqual(decision.candidate_cost, expected_buffered, places=4)

    # ── 8. sl_not_below_current ────────────────────────────────────────
    def test_sl_not_below_current(self) -> None:
        decision = self._call(
            current_price=100.0,
            net_remaining_breakeven_price=101.0,
            boll_lower=95.0,
        )
        self.assertEqual(decision.reason, "long_sl_not_below_current")
        self.assertIsNone(decision.protective_sl)

    # ── 9. missing_tp1_price ───────────────────────────────────────────
    def test_missing_tp1_price(self) -> None:
        decision = self._call(
            net_remaining_breakeven_price=0.0,
            tp1_price=None,
        )
        self.assertEqual(decision.reason, "missing_tp1_price")
        self.assertIsNone(decision.protective_sl)

    # ── 10. invalid_tp1_ratio ─────────────────────────────────────────
    def test_invalid_tp1_ratio_zero(self) -> None:
        decision = self._call(
            net_remaining_breakeven_price=0.0,
            tp1_ratio=0.0,
        )
        self.assertEqual(decision.reason, "invalid_tp1_ratio")

    def test_invalid_tp1_ratio_one(self) -> None:
        decision = self._call(
            net_remaining_breakeven_price=0.0,
            tp1_ratio=1.0,
        )
        self.assertEqual(decision.reason, "invalid_tp1_ratio")

    # ── 11. missing_cost_basis ────────────────────────────────────────
    def test_missing_cost_basis_zero_price(self) -> None:
        decision = self._call(current_price=0.0)
        self.assertEqual(decision.reason, "missing_cost_basis")
        self.assertIsNone(decision.protective_sl)


class TestThreeStagePostTp1ProtectiveSlShortNew(unittest.TestCase):
    """SHORT side — new relaxed logic."""

    def _call(self, **overrides):
        defaults = dict(
            side="SHORT",
            current_price=80.0,
            avg_entry_price=100.0,
            net_remaining_breakeven_price=99.0,  # > 0 → used directly
            breakeven_fee_buffer_pct=0.001,
            tp1_price=95.0,
            tp1_ratio=0.6,
            boll_middle=90.0,
            boll_upper=110.0,
            boll_lower=80.0,
            sl_tighten_ratio=0.50,
        )
        defaults.update(overrides)
        return calculate_three_stage_post_tp1_protective_sl(**defaults)

    # ── 1. net_remaining_breakeven > 0 → used directly as candidate_cost
    def test_net_breakeven_used_directly(self) -> None:
        decision = self._call(net_remaining_breakeven_price=98.0)
        self.assertEqual(decision.reason, "calculated")
        self.assertEqual(decision.candidate_cost, 98.0)
        self.assertEqual(decision.candidate_structure, 110.0)
        # protective_sl = min(98, 110) = 98
        self.assertEqual(decision.protective_sl, 98.0)

    # ── 2. candidate_structure = boll_upper ────────────────────────────
    def test_candidate_structure_is_boll_upper(self) -> None:
        decision = self._call(boll_upper=108.0)
        self.assertEqual(decision.candidate_structure, 108.0)

    # ── 3. protective_sl = min(candidate_cost, boll_upper) ─────────────
    def test_min_of_cost_and_upper(self) -> None:
        # cost > upper → use upper
        decision = self._call(
            net_remaining_breakeven_price=115.0,
            boll_upper=110.0,
        )
        self.assertEqual(decision.reason, "calculated")
        self.assertEqual(decision.protective_sl, 110.0)

    # ── 4. no interpolation to middle ─────────────────────────────────
    def test_no_interpolation_to_middle(self) -> None:
        decision = self._call(
            net_remaining_breakeven_price=99.0,
            boll_middle=90.0,
            boll_upper=110.0,
            sl_tighten_ratio=0.80,
        )
        self.assertEqual(decision.candidate_cost, 99.0)
        self.assertEqual(decision.candidate_structure, 110.0)
        # Old: cost = 99 + (90-99)*0.80 = 91.8; structure = 110 + (90-110)*0.80 = 94.0
        self.assertNotAlmostEqual(decision.candidate_cost, 91.8, places=2)
        self.assertNotAlmostEqual(decision.candidate_structure, 94.0, places=2)

    # ── 5. no clamp to middle ─────────────────────────────────────────
    def test_no_clamp_to_middle(self) -> None:
        decision = self._call(
            net_remaining_breakeven_price=88.0,
            boll_middle=90.0,
            boll_upper=110.0,
            current_price=75.0,
        )
        self.assertEqual(decision.reason, "calculated")
        # protective_sl = min(88, 110) = 88, which is below middle=90
        # Old logic would clamp to 90.
        self.assertEqual(decision.protective_sl, 88.0)
        self.assertLess(float(decision.protective_sl), 90.0)

    # ── 6. sl_tighten_ratio ignored ────────────────────────────────────
    def test_sl_tighten_ratio_ignored(self) -> None:
        d1 = self._call(sl_tighten_ratio=0.30)
        d2 = self._call(sl_tighten_ratio=0.80)
        self.assertEqual(d1.protective_sl, d2.protective_sl)

    # ── 7. fallback formula when net_remaining_breakeven <= 0 ───────────
    def test_fallback_formula_when_no_net_breakeven(self) -> None:
        decision = self._call(
            net_remaining_breakeven_price=0.0,
            avg_entry_price=100.0,
            tp1_price=95.0,
            tp1_ratio=0.6,
            boll_upper=110.0,
        )
        self.assertEqual(decision.reason, "calculated")
        # post_tp1_breakeven = 100 + 0.6*(100-95)/(1-0.6) = 100 + 7.5 = 107.5
        # buffered = 107.5 * (1 - 0.001) = 107.3925
        # candidate_cost = 107.3925
        expected_be = 100.0 + 0.6 * (100.0 - 95.0) / (1 - 0.6)  # = 107.5
        expected_buffered = expected_be * (1 - 0.001)  # = 107.3925
        self.assertAlmostEqual(decision.candidate_cost, expected_buffered, places=4)

    # ── 8. sl_not_above_current ────────────────────────────────────────
    def test_sl_not_above_current(self) -> None:
        decision = self._call(
            current_price=100.0,
            net_remaining_breakeven_price=99.0,
            boll_upper=105.0,
        )
        self.assertEqual(decision.reason, "short_sl_not_above_current")
        self.assertIsNone(decision.protective_sl)

    # ── 9. missing_tp1_price ───────────────────────────────────────────
    def test_missing_tp1_price(self) -> None:
        decision = self._call(
            net_remaining_breakeven_price=0.0,
            tp1_price=None,
        )
        self.assertEqual(decision.reason, "missing_tp1_price")
        self.assertIsNone(decision.protective_sl)


if __name__ == "__main__":
    unittest.main()
