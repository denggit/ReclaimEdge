"""Tests for TrendBreakoutAssessor and TrendBreakoutDecision."""

import pytest
from src.strategies.regime.types import BandSnapshot, TrendState
from src.strategies.trend_breakout import TrendBreakoutAssessor, TrendBreakoutDecision


# ── helpers ────────────────────────────────────────────────────────────

def _band(
    upper: float = 3100.0,
    middle: float = 3000.0,
    lower: float = 2900.0,
    candle_ts_ms: int = 1000000,
) -> BandSnapshot:
    return BandSnapshot(
        upper=upper, middle=middle, lower=lower,
        candle_ts_ms=candle_ts_ms, source="closed_or_frozen",
    )


def _compressed_bands(
    count: int = 20,
    base_upper: float = 3005.0,
    base_middle: float = 3000.0,
    base_lower: float = 2995.0,
    start_ts_ms: int = 900000,
    step_ms: int = 60000,
) -> list[BandSnapshot]:
    """Produce a sequence of narrow (compressed) bands."""
    bands: list[BandSnapshot] = []
    for i in range(count):
        bands.append(BandSnapshot(
            upper=base_upper,
            middle=base_middle,
            lower=base_lower,
            candle_ts_ms=start_ts_ms + i * step_ms,
            source="closed_or_frozen",
        ))
    return bands


def _make_assessor(**overrides) -> TrendBreakoutAssessor:
    kwargs = dict(
        compression_valid_after_seconds=7200,
        confirm_min_seconds=60,
        confirm_max_seconds=180,
        range_expansion_ratio_min=3.0,
        volume_expansion_ratio_min=3.0,
        outside_occupancy_min_ratio=0.70,
        min_new_extreme_count=2,
        max_inside_reclaim_seconds=3,
        cvd_min_buy_ratio=0.58,
        cvd_min_sell_ratio=0.58,
        cvd_max_pullback_ratio=0.45,
    )
    kwargs.update(overrides)
    return TrendBreakoutAssessor(**kwargs)


# ── TrendBreakoutDecision ──────────────────────────────────────────────

class TestTrendBreakoutDecision:
    def test_construct_no_breakout(self):
        d = TrendBreakoutDecision(
            is_trend_breakout=False,
            direction=None,
            reason="no_breakout",
        )
        assert d.is_trend_breakout is False
        assert d.direction is None
        assert d.trend_state == TrendState.NO_TREND

    def test_construct_confirmed_long(self):
        d = TrendBreakoutDecision(
            is_trend_breakout=True,
            direction="LONG",
            reason="trend_confirmed",
            trend_state=TrendState.TREND_UP_CONFIRMED,
            confidence=0.9,
            blocks_mean_reversion=True,
        )
        assert d.is_trend_breakout is True
        assert d.direction == "LONG"
        assert d.confidence == 0.9
        assert d.blocks_mean_reversion is True


# ── TrendBreakoutAssessor ──────────────────────────────────────────────

class TestTrendBreakoutAssessor:
    """Tests for TrendBreakoutAssessor — compression + trend detection."""

    def test_assess_no_compression_no_bands(self):
        """Without band history, no compression episode exists → no breakout
        detected, but the TrendDetector reports no_recent_compression."""
        assessor = _make_assessor()
        decision = assessor.assess(
            price=3200.0, ts_ms=1000000,
            boll_upper=3100.0, boll_middle=3000.0, boll_lower=2900.0,
            fast_cvd=0.001, buy_ratio=0.60, sell_ratio=0.40,
            range_expansion_passed=True, volume_expansion_passed=True,
            sustained_volume_passed=True, outside_occupancy_passed=True,
        )
        assert decision.is_trend_breakout is False
        assert "compression" in decision.reason.lower()
        assert decision.trend_state == TrendState.NO_TREND

    def test_assess_price_inside_band_no_breakout(self):
        """Price inside band → no breakout direction detected."""
        assessor = _make_assessor()
        assessor.feed_band(_band())
        decision = assessor.assess(
            price=3000.0, ts_ms=1000000,
            boll_upper=3100.0, boll_middle=3000.0, boll_lower=2900.0,
            fast_cvd=0.001, buy_ratio=0.60, sell_ratio=0.40,
            range_expansion_passed=True, volume_expansion_passed=True,
            sustained_volume_passed=True, outside_occupancy_passed=True,
        )
        assert decision.is_trend_breakout is False
        assert decision.reason == "no_breakout"

    def test_assess_price_above_band_detects_up_breakout(self):
        """Price above upper band → UP breakout but no compression → no trend."""
        assessor = _make_assessor()
        # Feed one band (not enough for compression episode)
        assessor.feed_band(_band(upper=3100.0, middle=3000.0, lower=2900.0))
        decision = assessor.assess(
            price=3200.0, ts_ms=1000000,
            boll_upper=3100.0, boll_middle=3000.0, boll_lower=2900.0,
            fast_cvd=0.001, buy_ratio=0.70, sell_ratio=0.30,
            range_expansion_passed=True, volume_expansion_passed=True,
            sustained_volume_passed=True, outside_occupancy_passed=True,
        )
        # No compression → trend detector returns NO_TREND
        assert decision.is_trend_breakout is False
        assert decision.trend_state == TrendState.NO_TREND
        assert "no_recent_compression" in decision.reason

    def test_assess_with_compression_and_expansion_has_cvd_confirming(self):
        """With valid compression + expansion + strong UP CVD → trend candidate."""
        assessor = _make_assessor(confirm_min_seconds=60)
        # Feed compressed bands to build a compression episode
        for band in _compressed_bands(count=20):
            assessor.feed_band(band)

        # First call: detect breakout and set anchor
        _first = assessor.assess(
            price=3200.0, ts_ms=1000000,
            boll_upper=3150.0, boll_middle=3000.0, boll_lower=2850.0,
            fast_cvd=0.001, buy_ratio=0.70, sell_ratio=0.30,
            episode_buy_volume=1000.0, episode_sell_volume=500.0,
            episode_cvd_max=0.001, episode_cvd_min=0.0005,
            range_expansion_passed=True, volume_expansion_passed=True,
            sustained_volume_passed=True, outside_occupancy_passed=True,
        )
        # First call: breakout detected but anchor CVD == current CVD → delta=0
        # So CVD won't confirm (delta must be > 0 for UP)
        # This is correct — confirmation needs time for CVD to accumulate

        # Second call: CVD has accumulated since anchor
        decision = assessor.assess(
            price=3250.0, ts_ms=1100000,  # 1000s later
            boll_upper=3150.0, boll_middle=3000.0, boll_lower=2850.0,
            fast_cvd=0.003,  # CVD increased from anchor (0.001 → 0.003)
            buy_ratio=0.70, sell_ratio=0.30,
            episode_buy_volume=2000.0, episode_sell_volume=500.0,
            episode_cvd_max=0.003, episode_cvd_min=0.0005,
            range_expansion_passed=True, volume_expansion_passed=True,
            sustained_volume_passed=True, outside_occupancy_passed=True,
            new_extreme_count=2,
        )
        # With compression + CVD confirming + expansions → candidate or confirmed
        # (at 1000s, we're well past confirm_min_seconds=60s, so it could be confirmed)
        assert decision.trend_state in (
            TrendState.TREND_UP_CANDIDATE,
            TrendState.TREND_UP_CONFIRMED,
        )

    def test_assess_price_below_band_detects_down_breakout(self):
        """Price below lower band with CVD confirming → DOWN candidate."""
        assessor = _make_assessor(confirm_min_seconds=60)
        for band in _compressed_bands(count=20):
            assessor.feed_band(band)

        # First call: detect breakout
        _first = assessor.assess(
            price=2800.0, ts_ms=1000000,
            boll_upper=3100.0, boll_middle=3000.0, boll_lower=2900.0,
            fast_cvd=-0.001, buy_ratio=0.30, sell_ratio=0.70,
            episode_buy_volume=500.0, episode_sell_volume=1000.0,
            episode_cvd_max=-0.0005, episode_cvd_min=-0.001,
            range_expansion_passed=True, volume_expansion_passed=True,
            sustained_volume_passed=True, outside_occupancy_passed=True,
        )

        # Second call: CVD has further declined since anchor
        decision = assessor.assess(
            price=2780.0, ts_ms=1100000,
            boll_upper=3100.0, boll_middle=3000.0, boll_lower=2900.0,
            fast_cvd=-0.003,  # CVD decreased from anchor (-0.001 → -0.003)
            buy_ratio=0.30, sell_ratio=0.70,
            episode_buy_volume=500.0, episode_sell_volume=2000.0,
            episode_cvd_max=-0.0005, episode_cvd_min=-0.003,
            range_expansion_passed=True, volume_expansion_passed=True,
            sustained_volume_passed=True, outside_occupancy_passed=True,
            new_extreme_count=2,
        )
        assert decision.trend_state in (
            TrendState.TREND_DOWN_CANDIDATE,
            TrendState.TREND_DOWN_CONFIRMED,
        )

    def test_reset_clears_state(self):
        assessor = _make_assessor()
        for band in _compressed_bands(count=20):
            assessor.feed_band(band)
        assessor.reset()
        # After reset, no band history → no compression
        decision = assessor.assess(
            price=3200.0, ts_ms=2000000,
            boll_upper=3100.0, boll_middle=3000.0, boll_lower=2900.0,
            fast_cvd=0.001, buy_ratio=0.70, sell_ratio=0.30,
        )
        assert decision.trend_state == TrendState.NO_TREND

    def test_trend_detector_property(self):
        assessor = _make_assessor()
        assert assessor.trend_detector is not None
        assert assessor.trend_detector.state == TrendState.NO_TREND

    def test_compression_detector_property(self):
        assessor = _make_assessor()
        assert assessor.compression_detector is not None

    def test_feed_band_limits_history(self):
        """Feeding more than 96 bands keeps only the most recent 96."""
        assessor = _make_assessor()
        for i in range(200):
            assessor.feed_band(BandSnapshot(
                upper=3005.0, middle=3000.0, lower=2995.0,
                candle_ts_ms=i * 60000, source="closed_or_frozen",
            ))
        # Should not crash — internal ring buffer capped at 96
        decision = assessor.assess(
            price=3200.0, ts_ms=20000000,
            boll_upper=3100.0, boll_middle=3000.0, boll_lower=2900.0,
            fast_cvd=0.002, buy_ratio=0.70, sell_ratio=0.30,
            episode_buy_volume=2000.0, episode_sell_volume=500.0,
            episode_cvd_max=0.003, episode_cvd_min=0.001,
            range_expansion_passed=True, volume_expansion_passed=True,
            sustained_volume_passed=True, outside_occupancy_passed=True,
            new_extreme_count=2,
        )
        # Should not crash; result depends on compression detection
        assert decision is not None
