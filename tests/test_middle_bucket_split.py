"""Tests for middle_bucket_split.py — pure calculation module."""

from src.strategies.middle_bucket_split import (
    MiddleBucketSplitDecision,
    build_middle_bucket_split_decision,
    calculate_fast_protective_sl,
    is_stop_valid_for_current_price,
)


class TestBuildMiddleBucketSplitDecision:
    """Tests for build_middle_bucket_split_decision()."""

    def test_disabled_returns_disabled(self):
        decision = build_middle_bucket_split_decision(
            enabled=False,
            side="LONG",
            middle_bucket_ratio=0.70,
            fast_ratio_of_bucket=0.70,
            fast_middle_price=1650.0,
            slow_middle_price=1640.0,
            effective_breakeven=1600.0,
            min_net_profit_pct=0.002,
        )
        assert decision.enabled is False
        assert decision.reason == "disabled"
        assert decision.action == "DISABLED"

    def test_long_fast_and_slow_both_sufficient(self):
        """LONG: fast=1650, slow=1640, breakeven=1600, min_profit=0.002 → required=1603.2.
        Both fast and slow >= required → split_enabled."""
        decision = build_middle_bucket_split_decision(
            enabled=True,
            side="LONG",
            middle_bucket_ratio=0.70,
            fast_ratio_of_bucket=0.70,
            fast_middle_price=1650.0,
            slow_middle_price=1640.0,
            effective_breakeven=1600.0,
            min_net_profit_pct=0.002,
        )
        assert decision.enabled is True
        assert decision.reason == "split_enabled"
        assert decision.action == "SPLIT"
        # fast_total = 0.70 * 0.70 = 0.49
        assert abs(decision.fast_total_ratio - 0.49) < 0.0001
        # slow_total = 0.70 * 0.30 = 0.21
        assert abs(decision.slow_total_ratio - 0.21) < 0.0001
        # effective_price = 1650 * 0.70 + 1640 * 0.30 = 1155 + 492 = 1647
        expected_effective = 1650.0 * 0.70 + 1640.0 * 0.30
        assert abs(float(decision.effective_price or 0.0) - expected_effective) < 0.01

    def test_long_fast_insufficient_slow_ok(self):
        """LONG: breakeven=1600, required=1603.2. fast=1601 (<required), slow=1640 (>=required)."""
        decision = build_middle_bucket_split_decision(
            enabled=True,
            side="LONG",
            middle_bucket_ratio=0.70,
            fast_ratio_of_bucket=0.70,
            fast_middle_price=1601.0,
            slow_middle_price=1640.0,
            effective_breakeven=1600.0,
            min_net_profit_pct=0.002,
        )
        assert decision.enabled is False
        assert decision.reason == "fast_middle_profit_insufficient_slow_middle_ok"
        assert decision.action == "UNSPLIT_SLOW_MIDDLE"

    def test_long_both_insufficient(self):
        """LONG: breakeven=1600, required=1603.2. fast=1601, slow=1602 → both insufficient."""
        decision = build_middle_bucket_split_decision(
            enabled=True,
            side="LONG",
            middle_bucket_ratio=0.70,
            fast_ratio_of_bucket=0.70,
            fast_middle_price=1601.0,
            slow_middle_price=1602.0,
            effective_breakeven=1600.0,
            min_net_profit_pct=0.002,
        )
        assert decision.enabled is False
        assert decision.reason == "middle_profit_insufficient"
        assert decision.action == "FALLBACK_OUTER"

    def test_short_fast_and_slow_both_sufficient(self):
        """SHORT: breakeven=1600, min_profit=0.002 → required=1596.8.
        fast=1580, slow=1590 → both <= required → split_enabled."""
        decision = build_middle_bucket_split_decision(
            enabled=True,
            side="SHORT",
            middle_bucket_ratio=0.70,
            fast_ratio_of_bucket=0.70,
            fast_middle_price=1580.0,
            slow_middle_price=1590.0,
            effective_breakeven=1600.0,
            min_net_profit_pct=0.002,
        )
        assert decision.enabled is True
        assert decision.reason == "split_enabled"
        assert decision.action == "SPLIT"
        # fast_total = 0.70 * 0.70 = 0.49
        assert abs(decision.fast_total_ratio - 0.49) < 0.0001
        # slow_total = 0.70 * 0.30 = 0.21
        assert abs(decision.slow_total_ratio - 0.21) < 0.0001
        expected_effective = 1580.0 * 0.70 + 1590.0 * 0.30
        assert abs(float(decision.effective_price or 0.0) - expected_effective) < 0.01

    def test_short_fast_insufficient_slow_ok(self):
        """SHORT: breakeven=1600, required=1596.8. fast=1598 (>required), slow=1590 (<=required)."""
        decision = build_middle_bucket_split_decision(
            enabled=True,
            side="SHORT",
            middle_bucket_ratio=0.70,
            fast_ratio_of_bucket=0.70,
            fast_middle_price=1598.0,
            slow_middle_price=1590.0,
            effective_breakeven=1600.0,
            min_net_profit_pct=0.002,
        )
        assert decision.enabled is False
        assert decision.reason == "fast_middle_profit_insufficient_slow_middle_ok"
        assert decision.action == "UNSPLIT_SLOW_MIDDLE"

    def test_short_both_insufficient(self):
        """SHORT: breakeven=1600, required=1596.8. fast=1598, slow=1599 → both insufficient."""
        decision = build_middle_bucket_split_decision(
            enabled=True,
            side="SHORT",
            middle_bucket_ratio=0.70,
            fast_ratio_of_bucket=0.70,
            fast_middle_price=1598.0,
            slow_middle_price=1599.0,
            effective_breakeven=1600.0,
            min_net_profit_pct=0.002,
        )
        assert decision.enabled is False
        assert decision.reason == "middle_profit_insufficient"
        assert decision.action == "FALLBACK_OUTER"

    def test_invalid_middle_bucket_ratio(self):
        decision = build_middle_bucket_split_decision(
            enabled=True,
            side="LONG",
            middle_bucket_ratio=0.0,
            fast_ratio_of_bucket=0.70,
            fast_middle_price=1650.0,
            slow_middle_price=1640.0,
            effective_breakeven=1600.0,
            min_net_profit_pct=0.002,
        )
        assert decision.enabled is False
        assert decision.reason == "invalid_middle_bucket_ratio"
        assert decision.action == "INVALID"

    def test_invalid_fast_ratio(self):
        decision = build_middle_bucket_split_decision(
            enabled=True,
            side="LONG",
            middle_bucket_ratio=0.70,
            fast_ratio_of_bucket=1.0,
            fast_middle_price=1650.0,
            slow_middle_price=1640.0,
            effective_breakeven=1600.0,
            min_net_profit_pct=0.002,
        )
        assert decision.enabled is False
        assert decision.reason == "invalid_fast_ratio"
        assert decision.action == "INVALID"

    def test_fast_middle_missing(self):
        decision = build_middle_bucket_split_decision(
            enabled=True,
            side="LONG",
            middle_bucket_ratio=0.70,
            fast_ratio_of_bucket=0.70,
            fast_middle_price=None,
            slow_middle_price=1640.0,
            effective_breakeven=1600.0,
            min_net_profit_pct=0.002,
        )
        assert decision.enabled is False
        assert decision.reason == "fast_middle_missing"
        assert decision.action == "INVALID"

    def test_slow_middle_missing(self):
        decision = build_middle_bucket_split_decision(
            enabled=True,
            side="LONG",
            middle_bucket_ratio=0.70,
            fast_ratio_of_bucket=0.70,
            fast_middle_price=1650.0,
            slow_middle_price=None,
            effective_breakeven=1600.0,
            min_net_profit_pct=0.002,
        )
        assert decision.enabled is False
        assert decision.reason == "slow_middle_missing"
        assert decision.action == "INVALID"

    def test_invalid_effective_breakeven(self):
        decision = build_middle_bucket_split_decision(
            enabled=True,
            side="LONG",
            middle_bucket_ratio=0.70,
            fast_ratio_of_bucket=0.70,
            fast_middle_price=1650.0,
            slow_middle_price=1640.0,
            effective_breakeven=0.0,
            min_net_profit_pct=0.002,
        )
        assert decision.enabled is False
        assert decision.reason == "invalid_effective_breakeven"
        assert decision.action == "INVALID"

    def test_ratios_are_formula_computed_not_hardcoded(self):
        """Verify that fast/slow ratios are computed from inputs, not hardcoded.

        Using middle_bucket_ratio=0.60, fast_ratio_of_bucket=0.80:
          fast_total = 0.60 * 0.80 = 0.48
          slow_total = 0.60 * 0.20 = 0.12
        """
        decision = build_middle_bucket_split_decision(
            enabled=True,
            side="LONG",
            middle_bucket_ratio=0.60,
            fast_ratio_of_bucket=0.80,
            fast_middle_price=1650.0,
            slow_middle_price=1640.0,
            effective_breakeven=1600.0,
            min_net_profit_pct=0.002,
        )
        assert decision.enabled is True
        assert decision.action == "SPLIT"
        expected_fast = 0.60 * 0.80
        expected_slow = 0.60 * 0.20
        assert abs(decision.fast_total_ratio - expected_fast) < 0.0001
        assert abs(decision.slow_total_ratio - expected_slow) < 0.0001

    def test_slow_middle_profit_insufficient(self):
        """LONG: fast OK, slow not OK (rare case)."""
        decision = build_middle_bucket_split_decision(
            enabled=True,
            side="LONG",
            middle_bucket_ratio=0.70,
            fast_ratio_of_bucket=0.70,
            fast_middle_price=1650.0,
            slow_middle_price=1601.0,
            effective_breakeven=1600.0,
            min_net_profit_pct=0.002,
        )
        assert decision.enabled is False
        assert decision.reason == "slow_middle_profit_insufficient"
        assert decision.action == "FALLBACK_OUTER"


class TestCalculateFastProtectiveSl:
    """Tests for calculate_fast_protective_sl()."""

    def test_long_fast_sl(self):
        sl = calculate_fast_protective_sl(
            side="LONG",
            avg_entry_price=1600.0,
            fee_buffer_pct=0.001,
        )
        assert sl is not None
        assert abs(sl - 1601.6) < 0.01

    def test_short_fast_sl(self):
        sl = calculate_fast_protective_sl(
            side="SHORT",
            avg_entry_price=1600.0,
            fee_buffer_pct=0.001,
        )
        assert sl is not None
        assert abs(sl - 1598.4) < 0.01

    def test_invalid_avg_entry_price(self):
        sl = calculate_fast_protective_sl(
            side="LONG",
            avg_entry_price=0.0,
            fee_buffer_pct=0.001,
        )
        assert sl is None

    def test_negative_avg_entry_price(self):
        sl = calculate_fast_protective_sl(
            side="LONG",
            avg_entry_price=-100.0,
            fee_buffer_pct=0.001,
        )
        assert sl is None


class TestIsStopValidForCurrentPrice:
    """Tests for is_stop_valid_for_current_price()."""

    def test_long_sl_valid(self):
        """LONG: stop=1601.6 < current=1620 → valid."""
        assert is_stop_valid_for_current_price(
            side="LONG",
            stop_price=1601.6,
            current_price=1620.0,
        ) is True

    def test_long_sl_invalid(self):
        """LONG: stop=1601.6 >= current=1600 → invalid."""
        assert is_stop_valid_for_current_price(
            side="LONG",
            stop_price=1601.6,
            current_price=1600.0,
        ) is False

    def test_short_sl_valid(self):
        """SHORT: stop=1598.4 > current=1580 → valid."""
        assert is_stop_valid_for_current_price(
            side="SHORT",
            stop_price=1598.4,
            current_price=1580.0,
        ) is True

    def test_short_sl_invalid(self):
        """SHORT: stop=1598.4 <= current=1600 → invalid."""
        assert is_stop_valid_for_current_price(
            side="SHORT",
            stop_price=1598.4,
            current_price=1600.0,
        ) is False

    def test_none_stop_price(self):
        assert is_stop_valid_for_current_price(
            side="LONG",
            stop_price=None,
            current_price=1620.0,
        ) is False

    def test_zero_current_price(self):
        assert is_stop_valid_for_current_price(
            side="LONG",
            stop_price=1600.0,
            current_price=0.0,
        ) is False
