from __future__ import annotations

from src.strategies.regime.router import RegimeRouter, RouterInput
from src.strategies.regime.types import (
    RegimeDecisionType,
    TrendState,
)


def _input(**kwargs) -> RouterInput:
    defaults = dict(
        trend_state=TrendState.NO_TREND,
        trend_confirmed=False,
        trend_confirmed_direction=None,
        trend_candidate_active=False,
        trend_candidate_direction=None,
        trend_failed=False,
        trend_failure_reason=None,
        trend_blocks_mean_reversion=False,
        mr_long_allowed=False,
        mr_short_allowed=False,
        cooldown_side=None,
        cooldown_until_ts_ms=0,
        cooldown_scope="SIDE",
        ts_ms=10000,
    )
    defaults.update(kwargs)
    return RouterInput(**defaults)


router = RegimeRouter()


# ── Tests ─────────────────────────────────────────────────────────────


class TestConflictDetection:
    """Test 1: TREND_LONG + MEAN_REVERSION_SHORT same tick → CONFLICT_NO_TRADE."""

    def test_trend_long_vs_mr_short_conflict(self):
        result = router.route(_input(
            trend_state=TrendState.TREND_UP_CONFIRMED,
            trend_confirmed=True,
            trend_confirmed_direction="LONG",
            mr_short_allowed=True,
            ts_ms=10000,
        ))
        assert result.decision_type == RegimeDecisionType.CONFLICT_NO_TRADE
        assert "regime_conflict" in result.reason

    def test_trend_short_vs_mr_long_conflict(self):
        result = router.route(_input(
            trend_state=TrendState.TREND_DOWN_CONFIRMED,
            trend_confirmed=True,
            trend_confirmed_direction="SHORT",
            mr_long_allowed=True,
            ts_ms=10000,
        ))
        assert result.decision_type == RegimeDecisionType.CONFLICT_NO_TRADE
        assert "regime_conflict" in result.reason


class TestTrendConfirmedOutput:
    """Test 2: TREND_CONFIRMED_UP → TREND_LONG."""

    def test_trend_up_confirmed_outputs_long(self):
        result = router.route(_input(
            trend_state=TrendState.TREND_UP_CONFIRMED,
            trend_confirmed=True,
            trend_confirmed_direction="LONG",
            ts_ms=10000,
        ))
        assert result.decision_type == RegimeDecisionType.TREND_LONG
        assert result.side == "LONG"

    def test_trend_down_confirmed_outputs_short(self):
        result = router.route(_input(
            trend_state=TrendState.TREND_DOWN_CONFIRMED,
            trend_confirmed=True,
            trend_confirmed_direction="SHORT",
            ts_ms=10000,
        ))
        assert result.decision_type == RegimeDecisionType.TREND_SHORT
        assert result.side == "SHORT"


class TestTrendCandidateActiveNoTrade:
    """Test 3: trend blocks mean-reversion → NO_TRADE."""

    def test_candidate_active_no_trade(self):
        result = router.route(_input(
            trend_state=TrendState.TREND_UP_CANDIDATE,
            trend_candidate_active=True,
            trend_candidate_direction="LONG",
            trend_failed=False,
            trend_blocks_mean_reversion=True,
            mr_short_allowed=True,  # MR is also possible, but candidate blocks it
            ts_ms=10000,
        ))
        assert result.decision_type == RegimeDecisionType.NO_TRADE
        assert "waiting_confirmation" in result.reason

    def test_no_block_when_trend_not_blocking(self):
        """When trend_blocks_mean_reversion=False, MR can proceed."""
        result = router.route(_input(
            trend_state=TrendState.TREND_FAILED,
            trend_failed=True,
            trend_blocks_mean_reversion=False,
            mr_short_allowed=True,
            ts_ms=10000,
        ))
        assert result.decision_type == RegimeDecisionType.MEAN_REVERSION_SHORT


class TestTrendFailedAllowsMeanReversion:
    """Test 4: trend failed + MR short valid → MEAN_REVERSION_SHORT."""

    def test_trend_failed_mr_short(self):
        result = router.route(_input(
            trend_state=TrendState.TREND_FAILED,
            trend_failed=True,
            trend_failure_reason="cvd_diverges_from_price",
            mr_short_allowed=True,
            ts_ms=10000,
        ))
        assert result.decision_type == RegimeDecisionType.MEAN_REVERSION_SHORT
        assert result.side == "SHORT"

    def test_trend_failed_mr_long(self):
        result = router.route(_input(
            trend_state=TrendState.TREND_FAILED,
            trend_failed=True,
            trend_failure_reason="cvd_diverges_from_price",
            mr_long_allowed=True,
            ts_ms=10000,
        ))
        assert result.decision_type == RegimeDecisionType.MEAN_REVERSION_LONG
        assert result.side == "LONG"


class TestCooldownSide:
    """Test 5: cooldown SIDE LONG → blocks LONG decisions, allows SHORT."""

    def test_side_cooldown_blocks_long(self):
        result = router.route(_input(
            trend_state=TrendState.TREND_UP_CONFIRMED,
            trend_confirmed=True,
            trend_confirmed_direction="LONG",
            cooldown_side="LONG",
            cooldown_until_ts_ms=20000,  # still active at ts=10000
            cooldown_scope="SIDE",
            ts_ms=10000,
        ))
        assert result.decision_type == RegimeDecisionType.NO_TRADE
        assert "cooldown_side" in result.reason

    def test_side_cooldown_allows_opposite(self):
        result = router.route(_input(
            trend_state=TrendState.TREND_DOWN_CONFIRMED,
            trend_confirmed=True,
            trend_confirmed_direction="SHORT",
            cooldown_side="LONG",
            cooldown_until_ts_ms=20000,
            cooldown_scope="SIDE",
            ts_ms=10000,
        ))
        # SHORT is not blocked by LONG cooldown
        assert result.decision_type == RegimeDecisionType.TREND_SHORT

    def test_side_cooldown_blocks_mean_reversion_same_side(self):
        result = router.route(_input(
            trend_state=TrendState.TREND_FAILED,
            trend_failed=True,
            mr_long_allowed=True,
            cooldown_side="LONG",
            cooldown_until_ts_ms=20000,
            cooldown_scope="SIDE",
            ts_ms=10000,
        ))
        assert result.decision_type == RegimeDecisionType.NO_TRADE
        assert "cooldown_side" in result.reason and "LONG" in result.reason

    def test_side_cooldown_expired_allows(self):
        result = router.route(_input(
            trend_state=TrendState.TREND_UP_CONFIRMED,
            trend_confirmed=True,
            trend_confirmed_direction="LONG",
            cooldown_side="LONG",
            cooldown_until_ts_ms=5000,  # expired at ts=10000
            cooldown_scope="SIDE",
            ts_ms=10000,
        ))
        assert result.decision_type == RegimeDecisionType.TREND_LONG


class TestCooldownGlobal:
    """Test 6: cooldown GLOBAL → blocks both sides."""

    def test_global_cooldown_blocks_all(self):
        result = router.route(_input(
            trend_state=TrendState.TREND_UP_CONFIRMED,
            trend_confirmed=True,
            trend_confirmed_direction="LONG",
            cooldown_side="LONG",
            cooldown_until_ts_ms=20000,
            cooldown_scope="GLOBAL",
            ts_ms=10000,
        ))
        assert result.decision_type == RegimeDecisionType.NO_TRADE
        assert "cooldown_global" in result.reason

    def test_global_cooldown_blocks_mr(self):
        result = router.route(_input(
            trend_state=TrendState.TREND_FAILED,
            trend_failed=True,
            mr_short_allowed=True,
            cooldown_side="SHORT",
            cooldown_until_ts_ms=20000,
            cooldown_scope="GLOBAL",
            ts_ms=10000,
        ))
        assert result.decision_type == RegimeDecisionType.NO_TRADE
        assert "cooldown_global" in result.reason


class TestNoCandidate:
    """Test 7: no candidate at all → NO_TRADE."""

    def test_no_candidate_outputs_no_trade(self):
        result = router.route(_input(ts_ms=10000))
        assert result.decision_type == RegimeDecisionType.NO_TRADE
        assert result.reason == "no_candidate"


class TestMeanReversionOnly:
    """Test: only mean-reversion allowed, no trend → outputs MEAN_REVERSION."""

    def test_mr_only_short(self):
        result = router.route(_input(
            mr_short_allowed=True,
            ts_ms=10000,
        ))
        assert result.decision_type == RegimeDecisionType.MEAN_REVERSION_SHORT

    def test_mr_only_long(self):
        result = router.route(_input(
            mr_long_allowed=True,
            ts_ms=10000,
        ))
        assert result.decision_type == RegimeDecisionType.MEAN_REVERSION_LONG


# ── New tests ────────────────────────────────────────────────────────────


class TestCooldownBeforeConflict:
    """SIDE cooldown filters BEFORE conflict detection."""

    def test_side_cooldown_long_filters_trend_before_conflict(self):
        """LONG cooldown → TREND_LONG filtered out → MEAN_REVERSION_SHORT survives alone."""
        result = router.route(_input(
            trend_state=TrendState.TREND_UP_CONFIRMED,
            trend_confirmed=True,
            trend_confirmed_direction="LONG",
            mr_short_allowed=True,
            cooldown_side="LONG",
            cooldown_until_ts_ms=20000,
            cooldown_scope="SIDE",
            ts_ms=10000,
        ))
        # LONG cooldown filters TREND_LONG; MEAN_REVERSION_SHORT is opposite side → survives
        # No conflict because only SHORT remains
        assert result.decision_type == RegimeDecisionType.MEAN_REVERSION_SHORT
        assert result.side == "SHORT"

    def test_side_cooldown_short_filters_trend_before_conflict(self):
        """SHORT cooldown → TREND_SHORT filtered out → MEAN_REVERSION_LONG survives alone."""
        result = router.route(_input(
            trend_state=TrendState.TREND_DOWN_CONFIRMED,
            trend_confirmed=True,
            trend_confirmed_direction="SHORT",
            mr_long_allowed=True,
            cooldown_side="SHORT",
            cooldown_until_ts_ms=20000,
            cooldown_scope="SIDE",
            ts_ms=10000,
        ))
        assert result.decision_type == RegimeDecisionType.MEAN_REVERSION_LONG
        assert result.side == "LONG"

    def test_side_cooldown_blocks_all_returns_no_trade(self):
        """No decisions survive cooldown → NO_TRADE."""
        result = router.route(_input(
            trend_state=TrendState.TREND_UP_CONFIRMED,
            trend_confirmed=True,
            trend_confirmed_direction="LONG",
            mr_long_allowed=True,
            cooldown_side="LONG",
            cooldown_until_ts_ms=20000,
            cooldown_scope="SIDE",
            ts_ms=10000,
        ))
        assert result.decision_type == RegimeDecisionType.NO_TRADE
        assert "cooldown_side" in result.reason


class TestTrendBlocksMeanReversionInRouter:
    """trend_blocks_mean_reversion=True blocks MR decisions in router."""

    def test_trend_blocks_prevents_mr(self):
        result = router.route(_input(
            trend_state=TrendState.TREND_UP_CANDIDATE,
            trend_candidate_active=True,
            trend_candidate_direction="LONG",
            trend_blocks_mean_reversion=True,
            mr_short_allowed=True,
            ts_ms=10000,
        ))
        assert result.decision_type == RegimeDecisionType.NO_TRADE
        assert "waiting_confirmation" in result.reason

    def test_trend_not_blocking_allows_mr(self):
        result = router.route(_input(
            trend_state=TrendState.NO_TREND,
            trend_blocks_mean_reversion=False,
            mr_short_allowed=True,
            ts_ms=10000,
        ))
        assert result.decision_type == RegimeDecisionType.MEAN_REVERSION_SHORT
