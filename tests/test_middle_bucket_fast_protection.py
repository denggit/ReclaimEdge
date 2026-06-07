"""Tests for middle_bucket_fast_protection.py."""

from src.position_management.middle_bucket_fast_protection import (
    FastProtectionDecision,
    build_fast_protection_decision,
)


class TestBuildFastProtectionDecision:
    """Tests for build_fast_protection_decision()."""

    def test_disabled_returns_noop(self):
        decision = build_fast_protection_decision(
            side="LONG",
            avg_entry_price=1600.0,
            current_price=1620.0,
            fee_buffer_pct=0.001,
            invalid_action="MARKET_EXIT",
            enabled=False,
        )
        assert decision.action == "NOOP"
        assert decision.reason == "disabled"

    def test_long_sl_valid_place_sl(self):
        """LONG: avg=1600, fee=0.001 → sl=1601.6. current=1620 → sl < current → valid."""
        decision = build_fast_protection_decision(
            side="LONG",
            avg_entry_price=1600.0,
            current_price=1620.0,
            fee_buffer_pct=0.001,
            invalid_action="MARKET_EXIT",
            enabled=True,
        )
        assert decision.action == "PLACE_SL"
        assert decision.sl_price is not None
        assert abs(float(decision.sl_price) - 1601.6) < 0.01
        assert decision.reason == "sl_valid"

    def test_long_sl_invalid_market_exit(self):
        """LONG: avg=1600, sl=1601.6. current=1600 → sl >= current → invalid → MARKET_EXIT."""
        decision = build_fast_protection_decision(
            side="LONG",
            avg_entry_price=1600.0,
            current_price=1600.0,
            fee_buffer_pct=0.001,
            invalid_action="MARKET_EXIT",
            enabled=True,
        )
        assert decision.action == "MARKET_EXIT"
        assert decision.reason == "sl_invalid_market_exit"

    def test_short_sl_valid_place_sl(self):
        """SHORT: avg=1600, sl=1598.4. current=1580 → sl > current → valid."""
        decision = build_fast_protection_decision(
            side="SHORT",
            avg_entry_price=1600.0,
            current_price=1580.0,
            fee_buffer_pct=0.001,
            invalid_action="MARKET_EXIT",
            enabled=True,
        )
        assert decision.action == "PLACE_SL"
        assert decision.sl_price is not None
        assert abs(float(decision.sl_price) - 1598.4) < 0.01

    def test_short_sl_invalid_market_exit(self):
        """SHORT: avg=1600, sl=1598.4. current=1600 → sl <= current → invalid → MARKET_EXIT."""
        decision = build_fast_protection_decision(
            side="SHORT",
            avg_entry_price=1600.0,
            current_price=1600.0,
            fee_buffer_pct=0.001,
            invalid_action="MARKET_EXIT",
            enabled=True,
        )
        assert decision.action == "MARKET_EXIT"
        assert decision.reason == "sl_invalid_market_exit"

    def test_missing_avg_entry_price(self):
        decision = build_fast_protection_decision(
            side="LONG",
            avg_entry_price=0.0,
            current_price=1620.0,
            fee_buffer_pct=0.001,
            invalid_action="MARKET_EXIT",
            enabled=True,
        )
        assert decision.action == "MARKET_EXIT"
        assert decision.reason == "missing_price_or_cost_basis"

    def test_missing_current_price(self):
        decision = build_fast_protection_decision(
            side="LONG",
            avg_entry_price=1600.0,
            current_price=0.0,
            fee_buffer_pct=0.001,
            invalid_action="MARKET_EXIT",
            enabled=True,
        )
        assert decision.action == "MARKET_EXIT"
        assert decision.reason == "missing_price_or_cost_basis"

    def test_invalid_action_halt_only(self):
        """LONG: sl invalid. invalid_action=HALT_ONLY → HALT_ONLY."""
        decision = build_fast_protection_decision(
            side="LONG",
            avg_entry_price=1600.0,
            current_price=1600.0,
            fee_buffer_pct=0.001,
            invalid_action="HALT_ONLY",
            enabled=True,
        )
        assert decision.action == "HALT_ONLY"

    def test_invalid_action_keep_position(self):
        """LONG: sl invalid. invalid_action=KEEP_POSITION → KEEP_POSITION."""
        decision = build_fast_protection_decision(
            side="LONG",
            avg_entry_price=1600.0,
            current_price=1600.0,
            fee_buffer_pct=0.001,
            invalid_action="KEEP_POSITION",
            enabled=True,
        )
        assert decision.action == "KEEP_POSITION"

    def test_unknown_invalid_action_defaults_to_market_exit(self):
        decision = build_fast_protection_decision(
            side="LONG",
            avg_entry_price=1600.0,
            current_price=1600.0,
            fee_buffer_pct=0.001,
            invalid_action="UNKNOWN_ACTION",
            enabled=True,
        )
        assert decision.action == "MARKET_EXIT"
