"""Tests for protective_sl_strength.py — SL strength comparison helpers."""

import pytest
from src.position_management.protective_sl_strength import (
    should_replace_sl,
    stronger_sl_price,
)


class TestStrongerSlPrice:
    """Tests for stronger_sl_price()."""

    def test_long_candidate_higher_is_stronger(self):
        """LONG: higher SL is stronger."""
        result = stronger_sl_price(side="LONG", existing_sl_price=100.0, candidate_sl_price=101.0)
        assert result == 101.0

    def test_long_existing_higher_is_stronger(self):
        """LONG: existing higher stays."""
        result = stronger_sl_price(side="LONG", existing_sl_price=101.0, candidate_sl_price=100.0)
        assert result == 101.0

    def test_long_equal_returns_existing(self):
        """LONG: equal prices, existing wins (no unnecessary replace)."""
        result = stronger_sl_price(side="LONG", existing_sl_price=100.0, candidate_sl_price=100.0)
        assert result == 100.0

    def test_short_candidate_lower_is_stronger(self):
        """SHORT: lower SL is stronger."""
        result = stronger_sl_price(side="SHORT", existing_sl_price=100.0, candidate_sl_price=99.0)
        assert result == 99.0

    def test_short_existing_lower_is_stronger(self):
        """SHORT: existing lower stays."""
        result = stronger_sl_price(side="SHORT", existing_sl_price=99.0, candidate_sl_price=100.0)
        assert result == 99.0

    def test_short_equal_returns_existing(self):
        """SHORT: equal prices, existing wins."""
        result = stronger_sl_price(side="SHORT", existing_sl_price=100.0, candidate_sl_price=100.0)
        assert result == 100.0

    def test_existing_none_returns_candidate(self):
        """When existing is None, return candidate."""
        result = stronger_sl_price(side="LONG", existing_sl_price=None, candidate_sl_price=100.0)
        assert result == 100.0

    def test_candidate_none_returns_existing(self):
        """When candidate is None, return existing."""
        result = stronger_sl_price(side="LONG", existing_sl_price=100.0, candidate_sl_price=None)
        assert result == 100.0

    def test_both_none_returns_none(self):
        """Both None returns None."""
        result = stronger_sl_price(side="LONG", existing_sl_price=None, candidate_sl_price=None)
        assert result is None

    def test_side_unknown_existing_exists(self):
        """Side unknown with existing: prefer existing."""
        result = stronger_sl_price(side=None, existing_sl_price=100.0, candidate_sl_price=101.0)
        assert result == 100.0

    def test_side_unknown_no_existing(self):
        """Side unknown without existing: return candidate."""
        result = stronger_sl_price(side=None, existing_sl_price=None, candidate_sl_price=101.0)
        assert result == 101.0


class TestShouldReplaceSl:
    """Tests for should_replace_sl()."""

    def test_long_candidate_higher_replace_true(self):
        """LONG: candidate=101 > existing=100 => replace."""
        assert should_replace_sl(side="LONG", existing_sl_price=100.0, candidate_sl_price=101.0) is True

    def test_long_candidate_lower_replace_false(self):
        """LONG: candidate=100 < existing=101 => no replace."""
        assert should_replace_sl(side="LONG", existing_sl_price=101.0, candidate_sl_price=100.0) is False

    def test_long_candidate_equal_replace_false(self):
        """LONG: candidate==existing => no replace (not strictly stronger)."""
        assert should_replace_sl(side="LONG", existing_sl_price=100.0, candidate_sl_price=100.0) is False

    def test_short_candidate_lower_replace_true(self):
        """SHORT: candidate=99 < existing=100 => replace."""
        assert should_replace_sl(side="SHORT", existing_sl_price=100.0, candidate_sl_price=99.0) is True

    def test_short_candidate_higher_replace_false(self):
        """SHORT: candidate=100 > existing=99 => no replace."""
        assert should_replace_sl(side="SHORT", existing_sl_price=99.0, candidate_sl_price=100.0) is False

    def test_short_candidate_equal_replace_false(self):
        """SHORT: candidate==existing => no replace."""
        assert should_replace_sl(side="SHORT", existing_sl_price=100.0, candidate_sl_price=100.0) is False

    def test_existing_none_candidate_exists_replace_true(self):
        """No existing SL => always replace."""
        assert should_replace_sl(side="LONG", existing_sl_price=None, candidate_sl_price=100.0) is True

    def test_candidate_none_replace_false(self):
        """No candidate => never replace."""
        assert should_replace_sl(side="LONG", existing_sl_price=100.0, candidate_sl_price=None) is False

    def test_candidate_none_existing_none_replace_false(self):
        """Both None => no replace."""
        assert should_replace_sl(side="LONG", existing_sl_price=None, candidate_sl_price=None) is False

    def test_side_unknown_existing_exists_replace_false(self):
        """Side unknown with existing SL => no replace (conservative)."""
        assert should_replace_sl(side=None, existing_sl_price=100.0, candidate_sl_price=101.0) is False

    def test_side_unknown_no_existing_replace_true(self):
        """Side unknown without existing => replace (no existing to protect)."""
        assert should_replace_sl(side=None, existing_sl_price=None, candidate_sl_price=101.0) is True
