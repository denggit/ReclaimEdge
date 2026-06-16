"""Tests for trend middle trailing SL pure functions."""

import pytest
from src.strategies.trend_middle_trailing_sl import (
    calculate_trend_middle_sl,
    is_trend_sl_tightened,
    tighten_trend_sl,
)


# ── calculate_trend_middle_sl ──────────────────────────────────────────


class TestCalculateTrendMiddleSL:
    def test_long(self):
        sl = calculate_trend_middle_sl(boll_middle=3000.0, buffer_pct=0.001, side="LONG")
        # 3000 * (1 - 0.001) = 3000 * 0.999 = 2997
        assert sl == 2997.0

    def test_short(self):
        sl = calculate_trend_middle_sl(boll_middle=3000.0, buffer_pct=0.001, side="SHORT")
        # 3000 * (1 + 0.001) = 3000 * 1.001 = 3003
        assert sl == pytest.approx(3003.0)

    def test_zero_buffer(self):
        sl = calculate_trend_middle_sl(boll_middle=3000.0, buffer_pct=0.0, side="LONG")
        assert sl == 3000.0

    def test_large_buffer(self):
        sl = calculate_trend_middle_sl(boll_middle=3000.0, buffer_pct=0.05, side="SHORT")
        assert sl == 3150.0

    def test_raises_on_negative_middle(self):
        with pytest.raises(ValueError):
            calculate_trend_middle_sl(boll_middle=0.0, buffer_pct=0.001, side="LONG")

    def test_raises_on_negative_buffer(self):
        with pytest.raises(ValueError):
            calculate_trend_middle_sl(boll_middle=3000.0, buffer_pct=-0.001, side="LONG")


# ── tighten_trend_sl ───────────────────────────────────────────────────


class TestTightenTrendSL:
    """Trend SL only tightens, never loosens.

    LONG:  higher SL = tighter  (closer to entry price)
    SHORT: lower  SL = tighter  (closer to entry price)
    """

    # ── LONG ───────────────────────────────────────────────────────────

    def test_long_tighten(self):
        # old_sl = 2900, candidate = 2950 → tighter (higher)
        result = tighten_trend_sl(
            old_sl=2900.0, candidate_sl=2950.0, current_price=3000.0, side="LONG",
        )
        assert result == 2950.0

    def test_long_loosen_rejected(self):
        # old_sl = 2950, candidate = 2900 → would loosen
        result = tighten_trend_sl(
            old_sl=2950.0, candidate_sl=2900.0, current_price=3000.0, side="LONG",
        )
        assert result is None

    def test_long_unchanged(self):
        result = tighten_trend_sl(
            old_sl=2950.0, candidate_sl=2950.0, current_price=3000.0, side="LONG",
        )
        assert result is None

    def test_long_no_old_sl(self):
        result = tighten_trend_sl(
            old_sl=None, candidate_sl=2950.0, current_price=3000.0, side="LONG",
        )
        assert result == 2950.0

    def test_long_invalid_candidate_at_price(self):
        # candidate equals current price → invalid
        result = tighten_trend_sl(
            old_sl=2900.0, candidate_sl=3000.0, current_price=3000.0, side="LONG",
        )
        assert result is None

    def test_long_invalid_candidate_above_price(self):
        # candidate above current price → invalid for LONG
        result = tighten_trend_sl(
            old_sl=2900.0, candidate_sl=3100.0, current_price=3000.0, side="LONG",
        )
        assert result is None

    # ── SHORT ──────────────────────────────────────────────────────────

    def test_short_tighten(self):
        # old_sl = 3100, candidate = 3050 → tighter (lower)
        result = tighten_trend_sl(
            old_sl=3100.0, candidate_sl=3050.0, current_price=3000.0, side="SHORT",
        )
        assert result == 3050.0

    def test_short_loosen_rejected(self):
        # old_sl = 3050, candidate = 3100 → would loosen
        result = tighten_trend_sl(
            old_sl=3050.0, candidate_sl=3100.0, current_price=3000.0, side="SHORT",
        )
        assert result is None

    def test_short_unchanged(self):
        result = tighten_trend_sl(
            old_sl=3050.0, candidate_sl=3050.0, current_price=3000.0, side="SHORT",
        )
        assert result is None

    def test_short_no_old_sl(self):
        result = tighten_trend_sl(
            old_sl=None, candidate_sl=3050.0, current_price=3000.0, side="SHORT",
        )
        assert result == 3050.0

    def test_short_invalid_candidate_at_price(self):
        result = tighten_trend_sl(
            old_sl=3100.0, candidate_sl=3000.0, current_price=3000.0, side="SHORT",
        )
        assert result is None

    def test_short_invalid_candidate_below_price(self):
        result = tighten_trend_sl(
            old_sl=3100.0, candidate_sl=2900.0, current_price=3000.0, side="SHORT",
        )
        assert result is None


# ── is_trend_sl_tightened ──────────────────────────────────────────────


class TestIsTrendSLTightened:
    def test_long_tightened(self):
        assert is_trend_sl_tightened(old_sl=2900.0, new_sl=2950.0, side="LONG") is True

    def test_long_not_tightened(self):
        assert is_trend_sl_tightened(old_sl=2950.0, new_sl=2900.0, side="LONG") is False

    def test_short_tightened(self):
        assert is_trend_sl_tightened(old_sl=3100.0, new_sl=3050.0, side="SHORT") is True

    def test_short_not_tightened(self):
        assert is_trend_sl_tightened(old_sl=3050.0, new_sl=3100.0, side="SHORT") is False

    def test_none_to_float_is_tightened(self):
        assert is_trend_sl_tightened(old_sl=None, new_sl=3000.0, side="LONG") is True

    def test_float_to_none_not_tightened(self):
        assert is_trend_sl_tightened(old_sl=3000.0, new_sl=None, side="LONG") is False

    def test_both_none_not_tightened(self):
        assert is_trend_sl_tightened(old_sl=None, new_sl=None, side="LONG") is False

    def test_unchanged_not_tightened(self):
        assert is_trend_sl_tightened(old_sl=3000.0, new_sl=3000.0, side="LONG") is False
