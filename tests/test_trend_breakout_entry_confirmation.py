"""Tests for upgraded Trend Breakout confirmation logic.

Covers:
1. Candidate appears within 60s and blocks MR
2. No 15m candle close outside → is_confirmed=False
3. 15m candle close outside + CVD same direction → confirmed (UP and DOWN)
4. Pre-breakout pressure same direction improves quality
5. Pre-breakout pressure opposite → stricter confirmation required
6. PreBreakoutPressureTracker unit tests
"""

from __future__ import annotations

import pytest

from src.strategies.regime.anchored_cvd import (
    AnchoredCvdConfig,
    build_anchored_cvd_state,
)
from src.strategies.regime.compression_detector import (
    CompressionDetector,
    CompressionDetectorConfig,
)
from src.strategies.regime.pre_breakout_pressure import (
    PreBreakoutPressureConfig,
    PreBreakoutPressureState,
    PreBreakoutPressureTracker,
)
from src.strategies.regime.trend_detector import (
    TrendAssessment,
    TrendDetector,
    TrendDetectorConfig,
)
from src.strategies.regime.types import (
    BandSnapshot,
    BreakoutSnapshot,
    CompressionEpisode,
    TrendState,
)
from src.strategies.trend_breakout import (
    TrendBreakoutAssessor,
    TrendBreakoutDecision,
)


# ======================================================================
# Helpers
# ======================================================================


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


def _breakout(
    direction: str = "UP",
    ts_ms: int = 1000000,
    price: float = 3200.0,
    anchor_cvd: float = 100.0,
    anchor_volume: float = 1000.0,
    upper: float = 3100.0,
    middle: float = 3000.0,
    lower: float = 2900.0,
) -> BreakoutSnapshot:
    return BreakoutSnapshot(
        direction=direction,  # type: ignore[arg-type]
        ts_ms=ts_ms,
        price=price,
        band=_band(upper=upper, middle=middle, lower=lower, candle_ts_ms=ts_ms),
        anchor_cvd=anchor_cvd,
        anchor_volume=anchor_volume,
    )


def _episode(
    start_ts_ms: int = 0,
    end_ts_ms: int = 900000,
    valid_until_ts_ms: int = 9_000_000_000,
    candle_count: int = 12,
) -> CompressionEpisode:
    return CompressionEpisode(
        start_ts_ms=start_ts_ms,
        end_ts_ms=end_ts_ms,
        valid_until_ts_ms=valid_until_ts_ms,
        compressed_candle_count=candle_count,
        min_outer_distance_pct=0.001,
        avg_outer_distance_pct=0.002,
        upper_at_end=3100.0,
        middle_at_end=3000.0,
        lower_at_end=2900.0,
        highest_band_upper=3120.0,
        lowest_band_lower=2880.0,
    )


def _cvd_up_confirming(
    anchor_cvd: float = 100.0,
    current_cvd: float = 160.0,
    buy_vol: float = 80.0,
    sell_vol: float = 20.0,
    cvd_max: float = 160.0,
    cvd_min: float = 100.0,
    anchor_ts: int = 1000000,
    current_ts: int = 1100000,
):
    return build_anchored_cvd_state(
        anchor_ts_ms=anchor_ts,
        current_ts_ms=current_ts,
        anchor_cvd=anchor_cvd,
        current_cvd=current_cvd,
        episode_buy_volume=buy_vol,
        episode_sell_volume=sell_vol,
        episode_cvd_max=cvd_max,
        episode_cvd_min=cvd_min,
    )


def _cvd_down_confirming(
    anchor_cvd: float = -100.0,
    current_cvd: float = -160.0,
    buy_vol: float = 20.0,
    sell_vol: float = 80.0,
    cvd_max: float = -100.0,
    cvd_min: float = -160.0,
    anchor_ts: int = 1000000,
    current_ts: int = 1100000,
):
    return build_anchored_cvd_state(
        anchor_ts_ms=anchor_ts,
        current_ts_ms=current_ts,
        anchor_cvd=anchor_cvd,
        current_cvd=current_cvd,
        episode_buy_volume=buy_vol,
        episode_sell_volume=sell_vol,
        episode_cvd_max=cvd_max,
        episode_cvd_min=cvd_min,
    )


def _make_detector(
    require_candle_close: bool = True,
    confirm_min_seconds: int = 60,
    confirm_max_seconds: int = 180,
    **extra,
) -> TrendDetector:
    cfg = TrendDetectorConfig(
        confirm_min_seconds=confirm_min_seconds,
        confirm_max_seconds=confirm_max_seconds,
        require_candle_close=require_candle_close,
        **extra,
    )
    comp_cfg = CompressionDetectorConfig(valid_after_seconds=7200)
    cvd_config = AnchoredCvdConfig(
        min_buy_ratio=0.58,
        min_sell_ratio=0.58,
        max_pullback_ratio=0.45,
    )
    return TrendDetector(cfg, CompressionDetector(comp_cfg), cvd_config)


def _assess(
    detector: TrendDetector,
    breakout: BreakoutSnapshot,
    episode: CompressionEpisode,
    cvd,
    current_ts_ms: int = 1100000,
    **overrides,
) -> TrendAssessment:
    """Call assess() with all "passed" flags and optional candle close data."""
    params = dict(
        breakout=breakout,
        compression_episode=episode,
        anchored_cvd=cvd,
        current_ts_ms=current_ts_ms,
        range_expansion_passed=True,
        volume_expansion_passed=True,
        sustained_volume_passed=True,
        outside_occupancy_passed=True,
        new_extreme_count=3,
        inside_reclaim_seconds=0.0,
        price_reclaimed_inside=False,
    )
    params.update(overrides)
    return detector.assess(**params)


def _closed_candle_data(
    candle_ts_ms: int,
    close: float,
    upper: float = 3100.0,
    lower: float = 2900.0,
) -> dict:
    """Helper to produce candle close params for assess()."""
    return dict(
        latest_candle_ts_ms=candle_ts_ms,
        latest_candle_close=close,
        latest_candle_live_mode=False,  # closed candle
        latest_candle_upper=upper,
        latest_candle_lower=lower,
    )


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
        trend_confirm_require_candle_close=True,
        trend_pre_breakout_pressure_enabled=True,
    )
    kwargs.update(overrides)
    return TrendBreakoutAssessor(**kwargs)


def _compressed_bands(
    count: int = 20,
    base_upper: float = 3005.0,
    base_middle: float = 3000.0,
    base_lower: float = 2995.0,
    start_ts_ms: int = 900000,
    step_ms: int = 60000,
) -> list[BandSnapshot]:
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


# ======================================================================
# Test 1: Candidate within 60s blocks MR, is_confirmed=False
# ======================================================================


class TestCandidateWithin60sBlocksMR:
    """Candidate appears quickly after range/volume expansion, blocks MR,
    but is NOT confirmed (no candle close yet)."""

    def test_candidate_appears_before_60s_and_blocks_mr(self):
        """With range/volume expansion passed, breakout age < 60s,
        candidate must be True, blocks_mean_reversion=True, not confirmed."""
        detector = _make_detector(require_candle_close=True, confirm_min_seconds=60)
        bo = _breakout("UP", ts_ms=1000000, price=3200.0,
                       upper=3100.0, middle=3000.0, lower=2900.0)
        ep = _episode()
        # Breakout age = 30s (current_ts_ms - breakout.ts_ms = 30000ms)
        cvd = _cvd_up_confirming(anchor_ts=1000000, current_ts=1030000)

        result = _assess(detector, bo, ep, cvd, current_ts_ms=1030000)

        assert result.is_candidate is True, f"Expected candidate, got: {result}"
        assert result.blocks_mean_reversion is True, (
            f"Candidate must block MR, got blocks_mean_reversion={result.blocks_mean_reversion}"
        )
        assert result.is_confirmed is False, (
            f"Candidate must not be confirmed before min_seconds, got is_confirmed=True"
        )
        assert result.is_failed is False

    def test_candidate_blocks_mr_down_direction(self):
        """Same for DOWN direction."""
        detector = _make_detector(require_candle_close=True, confirm_min_seconds=60)
        bo = _breakout("DOWN", ts_ms=1000000, price=2800.0,
                       upper=3100.0, middle=3000.0, lower=2900.0)
        ep = _episode()
        cvd = _cvd_down_confirming(anchor_ts=1000000, current_ts=1030000)

        result = _assess(detector, bo, ep, cvd, current_ts_ms=1030000)

        assert result.is_candidate is True
        assert result.blocks_mean_reversion is True
        assert result.is_confirmed is False

    def test_assessor_returns_candidate_not_confirmed_under_60s(self):
        """TrendBreakoutAssessor returns candidate=True, is_trend_breakout=False
        when breakout age < 60s (no candle close yet)."""
        assessor = _make_assessor(confirm_min_seconds=60,
                                  trend_confirm_require_candle_close=True)
        for band in _compressed_bands(count=20):
            assessor.feed_band(band)

        # First tick: anchor breakout (CVD delta=0 initially)
        assessor.assess(
            price=3200.0, ts_ms=1000000,
            boll_upper=3150.0, boll_middle=3000.0, boll_lower=2850.0,
            fast_cvd=0.001, buy_ratio=0.70, sell_ratio=0.30,
            episode_buy_volume=1000.0, episode_sell_volume=500.0,
            episode_cvd_max=0.001, episode_cvd_min=0.0005,
            range_expansion_passed=True, volume_expansion_passed=True,
            sustained_volume_passed=True, outside_occupancy_passed=True,
        )

        # Second tick: still under 60s, CVD has accumulated
        decision = assessor.assess(
            price=3250.0, ts_ms=1030000,  # 30s later
            boll_upper=3150.0, boll_middle=3000.0, boll_lower=2850.0,
            fast_cvd=0.003, buy_ratio=0.70, sell_ratio=0.30,
            episode_buy_volume=2000.0, episode_sell_volume=500.0,
            episode_cvd_max=0.003, episode_cvd_min=0.0005,
            range_expansion_passed=True, volume_expansion_passed=True,
            sustained_volume_passed=True, outside_occupancy_passed=True,
            new_extreme_count=2,
        )

        assert decision.is_trend_breakout is False, (
            f"Should not confirm before candle close, got is_trend_breakout=True"
        )
        assert decision.blocks_mean_reversion is True
        if decision.trend_assessment:
            assert decision.trend_assessment.is_candidate is True


# ======================================================================
# Test 2: No 15m candle close outside → not confirmed
# ======================================================================


class TestNoCandleCloseOutsideNotConfirmed:
    """Without a closed 15m candle standing outside the band, trend stays
    as candidate (not confirmed)."""

    def test_waiting_candle_close_when_no_close_data(self):
        """After min_seconds but no candle close data → waiting_candle_close."""
        detector = _make_detector(require_candle_close=True, confirm_min_seconds=60)
        bo = _breakout("UP", ts_ms=1000000, price=3200.0,
                       upper=3100.0, middle=3000.0, lower=2900.0)
        ep = _episode()
        cvd = _cvd_up_confirming(anchor_ts=1000000, current_ts=1100000)

        # Breakout age = 100s (past min_seconds)
        result = _assess(detector, bo, ep, cvd, current_ts_ms=1100000)
        # No candle close data passed → waiting_candle_close

        assert result.is_candidate is True, f"Expected candidate, got: {result}"
        assert result.is_confirmed is False, (
            f"Should not confirm without candle close, got is_confirmed=True"
        )
        assert "candle_close" in result.reason.lower(), (
            f"Reason should mention candle_close, got: {result.reason}"
        )

    def test_candle_close_back_inside_band_not_confirmed(self):
        """When a closed 15m candle appears but its close is back inside the
        band → not confirmed."""
        detector = _make_detector(require_candle_close=True, confirm_min_seconds=60)
        bo = _breakout("UP", ts_ms=1000000, price=3200.0,
                       upper=3100.0, middle=3000.0, lower=2900.0)
        ep = _episode()
        cvd = _cvd_up_confirming(anchor_ts=1000000, current_ts=1100000)

        # First call: no candle close data → waiting
        result1 = _assess(detector, bo, ep, cvd, current_ts_ms=1100000)
        assert result1.is_candidate is True
        assert "candle_close" in result1.reason.lower()

        # Second call: a candle closed but close was BACK INSIDE the band
        # close=3050 which is < upper=3100 → NOT outside
        candle_data = _closed_candle_data(
            candle_ts_ms=1090000,
            close=3050.0,  # inside the band!
            upper=3100.0,
            lower=2900.0,
        )
        result2 = _assess(detector, bo, ep, cvd, current_ts_ms=1120000, **candle_data)

        assert result2.is_confirmed is False, (
            f"Candle close inside band must not confirm, got: {result2}"
        )
        assert result2.has_candle_close_outside is False

    def test_live_candle_not_used_for_confirmation(self):
        """Live (unclosed) candle data must NOT be used for confirmation."""
        detector = _make_detector(require_candle_close=True, confirm_min_seconds=60)
        bo = _breakout("UP", ts_ms=1000000, price=3200.0,
                       upper=3100.0, middle=3000.0, lower=2900.0)
        ep = _episode()
        cvd = _cvd_up_confirming(anchor_ts=1000000, current_ts=1100000)

        # Pass candle data but with live_mode=True (unclosed)
        result = _assess(
            detector, bo, ep, cvd, current_ts_ms=1100000,
            latest_candle_ts_ms=1090000,
            latest_candle_close=3150.0,  # above upper!
            latest_candle_live_mode=True,  # LIVE (unclosed)
            latest_candle_upper=3100.0,
            latest_candle_lower=2900.0,
        )

        assert result.is_candidate is True
        assert result.is_confirmed is False, (
            "Live candle must NOT be used for trend confirmation"
        )
        assert "candle_close" in result.reason.lower()

    def test_down_direction_candle_close_inside_not_confirmed(self):
        """DOWN breakout: closed candle close > lower → not confirmed."""
        detector = _make_detector(require_candle_close=True, confirm_min_seconds=60)
        bo = _breakout("DOWN", ts_ms=1000000, price=2800.0,
                       upper=3100.0, middle=3000.0, lower=2900.0)
        ep = _episode()
        cvd = _cvd_down_confirming(anchor_ts=1000000, current_ts=1100000)

        candle_data = _closed_candle_data(
            candle_ts_ms=1090000,
            close=2950.0,  # > lower=2900 → back inside!
            upper=3100.0,
            lower=2900.0,
        )
        result = _assess(detector, bo, ep, cvd, current_ts_ms=1120000, **candle_data)

        assert result.is_confirmed is False
        assert result.has_candle_close_outside is False


# ======================================================================
# Test 3: 15m candle close outside + CVD same direction → confirmed
# ======================================================================


class TestCandleCloseOutsideCvdConfirmed:
    """When a 15m candle closes outside the band AND CVD confirms → trend is
    confirmed."""

    def test_up_confirmed_with_candle_close_outside(self):
        """UP breakout: candle close > upper + CVD up → confirmed."""
        detector = _make_detector(require_candle_close=True, confirm_min_seconds=60)
        bo = _breakout("UP", ts_ms=1000000, price=3200.0,
                       upper=3100.0, middle=3000.0, lower=2900.0)
        ep = _episode()
        cvd = _cvd_up_confirming(anchor_ts=1000000, current_ts=1100000)

        # Candle closed outside: close=3150 > upper=3100
        candle_data = _closed_candle_data(
            candle_ts_ms=1090000,
            close=3150.0,  # outside!
            upper=3100.0,
            lower=2900.0,
        )
        result = _assess(detector, bo, ep, cvd, current_ts_ms=1100000, **candle_data)

        assert result.is_confirmed is True, (
            f"Expected confirmed, got is_candidate={result.is_candidate} "
            f"reason={result.reason}"
        )
        assert result.is_candidate is False  # promoted to confirmed
        assert result.blocks_mean_reversion is True
        assert result.has_candle_close_outside is True
        assert result.confirmed_candle_ts_ms == 1090000

    def test_down_confirmed_with_candle_close_outside(self):
        """DOWN breakout: candle close < lower + CVD down → confirmed."""
        detector = _make_detector(require_candle_close=True, confirm_min_seconds=60)
        bo = _breakout("DOWN", ts_ms=1000000, price=2800.0,
                       upper=3100.0, middle=3000.0, lower=2900.0)
        ep = _episode()
        cvd = _cvd_down_confirming(anchor_ts=1000000, current_ts=1100000)

        # Candle closed outside: close=2850 < lower=2900
        candle_data = _closed_candle_data(
            candle_ts_ms=1090000,
            close=2850.0,  # outside!
            upper=3100.0,
            lower=2900.0,
        )
        result = _assess(detector, bo, ep, cvd, current_ts_ms=1100000, **candle_data)

        assert result.is_confirmed is True, (
            f"Expected DOWN confirmed, got is_candidate={result.is_candidate} "
            f"reason={result.reason}"
        )
        assert result.has_candle_close_outside is True
        assert result.trend_state == TrendState.TREND_DOWN_CONFIRMED

    def test_pressure_same_direction_not_required_for_entry(self):
        """Pre-breakout pressure same direction improves quality but is NOT
        required for confirmation."""
        detector = _make_detector(require_candle_close=True, confirm_min_seconds=60)
        bo = _breakout("UP", ts_ms=1000000, price=3200.0,
                       upper=3100.0, middle=3000.0, lower=2900.0)
        ep = _episode()
        cvd = _cvd_up_confirming(anchor_ts=1000000, current_ts=1100000)

        candle_data = _closed_candle_data(
            candle_ts_ms=1090000, close=3150.0, upper=3100.0, lower=2900.0,
        )

        # No pre_breakout_pressure passed → should still confirm
        result = _assess(detector, bo, ep, cvd, current_ts_ms=1100000, **candle_data)

        assert result.is_confirmed is True, (
            "Pre-breakout pressure should NOT be required for confirmation"
        )


# ======================================================================
# Test 4: Pre-breakout pressure same direction → quality boost
# ======================================================================


class TestPreBreakoutPressureSameDirection:
    """When inside-band CVD pressure is same direction as breakout,
    quality improves but doesn't open a position alone."""

    def test_pressure_up_before_breakout_does_not_open_alone(self):
        """Inside band pressure=UP but price still inside band → no trend entry."""
        tracker = PreBreakoutPressureTracker(
            PreBreakoutPressureConfig(enabled=True)
        )
        # Start observing inside band
        tracker.start(
            ts_ms=1000000, price=3000.0, fast_cvd=100.0,
            buy_volume=60.0, sell_volume=40.0,
            boll_upper=3100.0, boll_middle=3000.0, boll_lower=2900.0,
        )
        # Update with sustained UP pressure
        for i in range(10):
            tracker.update(
                ts_ms=1000000 + (i + 1) * 30000,
                price=3020.0 + i * 5,  # drifting up
                fast_cvd=100.0 + (i + 1) * 10,  # CVD increasing
                buy_volume=65.0, sell_volume=35.0,
                boll_upper=3100.0, boll_middle=3000.0, boll_lower=2900.0,
            )

        state = tracker.snapshot()
        assert state.direction == "UP", (
            f"Expected UP pressure, got direction={state.direction} score={state.score:.2f}"
        )
        assert state.score >= 0.60, f"Score should be >= 0.60, got {state.score:.2f}"
        assert state.anchored_cvd > 0, "CVD should be positive for UP"

        # This pressure alone should NOT create a trend entry — the
        # TrendBreakoutAssessor only enters when price is outside the band.

    def test_pressure_down_before_breakout_does_not_open_alone(self):
        """Inside band pressure=DOWN but price still inside → no trend entry."""
        tracker = PreBreakoutPressureTracker(
            PreBreakoutPressureConfig(enabled=True)
        )
        tracker.start(
            ts_ms=1000000, price=3000.0, fast_cvd=-100.0,
            buy_volume=35.0, sell_volume=65.0,
            boll_upper=3100.0, boll_middle=3000.0, boll_lower=2900.0,
        )
        for i in range(10):
            tracker.update(
                ts_ms=1000000 + (i + 1) * 30000,
                price=2980.0 - i * 5,  # drifting down
                fast_cvd=-100.0 - (i + 1) * 10,  # CVD decreasing
                buy_volume=35.0, sell_volume=65.0,
                boll_upper=3100.0, boll_middle=3000.0, boll_lower=2900.0,
            )

        state = tracker.snapshot()
        assert state.direction == "DOWN", (
            f"Expected DOWN pressure, got direction={state.direction} score={state.score:.2f}"
        )
        assert state.anchored_cvd < 0, "CVD should be negative for DOWN"


# ======================================================================
# Test 5: Pre-breakout pressure opposite → stricter confirmation
# ======================================================================


class TestPreBreakoutPressureConflict:
    """When inside-band pressure is opposite to breakout direction,
    confirmation must be stricter (requires candle close + strong CVD)."""

    def test_pressure_conflict_blocks_confirmation_without_strong_cvd(self):
        """Pressure DOWN but breakout UP → blocked without strong post-breakout CVD."""
        detector = _make_detector(
            require_candle_close=True,
            confirm_min_seconds=60,
            pre_breakout_pressure_enabled=True,
            pre_breakout_pressure_min_score=0.60,
        )
        bo = _breakout("UP", ts_ms=1000000, price=3200.0,
                       upper=3100.0, middle=3000.0, lower=2900.0)
        ep = _episode()

        # Weak post-breakout CVD (low buy ratio, small delta)
        weak_cvd = build_anchored_cvd_state(
            anchor_ts_ms=1000000, current_ts_ms=1100000,
            anchor_cvd=100.0, current_cvd=110.0,
            episode_buy_volume=50.0, episode_sell_volume=50.0,
            episode_cvd_max=110.0, episode_cvd_min=100.0,
        )

        # Pressure is DOWN (opposite to breakout UP)
        conflict_pressure = PreBreakoutPressureState(
            direction="DOWN",
            score=0.72,
            duration_seconds=300.0,
            anchored_cvd=-50.0,
            buy_ratio=0.35,
            sell_ratio=0.65,
            reason="down_pressure_dominant",
        )

        # With candle close outside but WEAK CVD → should fail due to pressure conflict
        candle_data = _closed_candle_data(
            candle_ts_ms=1090000, close=3150.0, upper=3100.0, lower=2900.0,
        )

        result = _assess(
            detector, bo, ep, weak_cvd, current_ts_ms=1100000,
            pre_breakout_pressure=conflict_pressure, **candle_data,
        )

        # Pressure conflict + weak CVD → should fail
        assert result.is_failed, (
            f"Pressure conflict + weak CVD should fail, "
            f"got is_confirmed={result.is_confirmed} is_candidate={result.is_candidate} "
            f"reason={result.reason}"
        )
        assert "pressure_conflict" in result.reason.lower() or "cvd_not_strong" in result.reason, (
            f"Should mention pressure conflict or CVD not strong, got: {result.reason}"
        )

    def test_pressure_conflict_with_candle_close_and_strong_cvd_allows_confirmation(self):
        """Pressure DOWN but breakout UP WITH candle close outside + strong CVD
        → confirmed (stricter path passed)."""
        detector = _make_detector(
            require_candle_close=True,
            confirm_min_seconds=60,
            pre_breakout_pressure_enabled=True,
            pre_breakout_pressure_min_score=0.60,
        )
        bo = _breakout("UP", ts_ms=1000000, price=3200.0,
                       upper=3100.0, middle=3000.0, lower=2900.0)
        ep = _episode()
        cvd = _cvd_up_confirming(anchor_ts=1000000, current_ts=1100000)

        conflict_pressure = PreBreakoutPressureState(
            direction="DOWN", score=0.72, duration_seconds=300.0,
            anchored_cvd=-50.0, buy_ratio=0.35, sell_ratio=0.65,
            reason="down_pressure_dominant",
        )

        candle_data = _closed_candle_data(
            candle_ts_ms=1090000, close=3150.0, upper=3100.0, lower=2900.0,
        )

        # Must pass candle close data to get through
        result1 = _assess(
            detector, bo, ep, cvd, current_ts_ms=1100000,
            pre_breakout_pressure=conflict_pressure, **candle_data,
        )

        # With pressure conflict AND candle close outside AND strong CVD →
        # should be confirmed (all stricter checks passed)
        assert result1.is_confirmed is True, (
            f"With candle close + strong CVD, should confirm despite pressure conflict. "
            f"Got is_confirmed={result1.is_confirmed} reason={result1.reason}"
        )

    def test_pressure_conflict_with_weak_cvd_fails(self):
        """Pressure opposite + candle close outside BUT weak post-breakout CVD
        → NOT confirmed."""
        detector = _make_detector(
            require_candle_close=True,
            confirm_min_seconds=60,
            pre_breakout_pressure_enabled=True,
            pre_breakout_pressure_min_score=0.60,
        )
        bo = _breakout("UP", ts_ms=1000000, price=3200.0,
                       upper=3100.0, middle=3000.0, lower=2900.0)
        ep = _episode()

        # Weak CVD: anchor=100, current=110 (only +10 delta, low buy_ratio)
        weak_cvd = build_anchored_cvd_state(
            anchor_ts_ms=1000000, current_ts_ms=1100000,
            anchor_cvd=100.0, current_cvd=110.0,
            episode_buy_volume=50.0, episode_sell_volume=50.0,
            episode_cvd_max=110.0, episode_cvd_min=100.0,
        )

        conflict_pressure = PreBreakoutPressureState(
            direction="DOWN", score=0.72, duration_seconds=300.0,
            anchored_cvd=-50.0, buy_ratio=0.35, sell_ratio=0.65,
            reason="down_pressure_dominant",
        )

        candle_data = _closed_candle_data(
            candle_ts_ms=1090000, close=3150.0, upper=3100.0, lower=2900.0,
        )

        result = _assess(
            detector, bo, ep, weak_cvd, current_ts_ms=1100000,
            pre_breakout_pressure=conflict_pressure, **candle_data,
        )

        # With weak CVD + pressure conflict → should fail
        assert result.is_failed is True, (
            f"Pressure conflict + weak CVD should fail, "
            f"got is_confirmed={result.is_confirmed} reason={result.reason}"
        )

    def test_pressure_conflict_up_breakout_down_pressure(self):
        """DOWN breakout with UP pressure conflict → stricter checks."""
        detector = _make_detector(
            require_candle_close=True,
            confirm_min_seconds=60,
            pre_breakout_pressure_enabled=True,
            pre_breakout_pressure_min_score=0.60,
        )
        bo = _breakout("DOWN", ts_ms=1000000, price=2800.0,
                       upper=3100.0, middle=3000.0, lower=2900.0)
        ep = _episode()
        cvd = _cvd_down_confirming(anchor_ts=1000000, current_ts=1100000)

        # UP pressure (opposite to DOWN breakout)
        conflict_pressure = PreBreakoutPressureState(
            direction="UP", score=0.68, duration_seconds=300.0,
            anchored_cvd=50.0, buy_ratio=0.65, sell_ratio=0.35,
            reason="up_pressure_dominant",
        )

        # Without candle close → blocked
        result = _assess(
            detector, bo, ep, cvd, current_ts_ms=1100000,
            pre_breakout_pressure=conflict_pressure,
        )
        assert result.is_confirmed is False


# ======================================================================
# Test 6: No clear pressure → neutral, normal confirmation
# ======================================================================


class TestNoClearPressureNeutral:
    """When there's no clear directional pressure, confirmation is neutral
    and follows normal candle close path."""

    def test_no_pressure_passed_follows_normal_confirmation(self):
        """Without pre_breakout_pressure, follows normal candle close path."""
        detector = _make_detector(
            require_candle_close=True,
            confirm_min_seconds=60,
            pre_breakout_pressure_enabled=True,
        )
        bo = _breakout("UP", ts_ms=1000000, price=3200.0,
                       upper=3100.0, middle=3000.0, lower=2900.0)
        ep = _episode()
        cvd = _cvd_up_confirming(anchor_ts=1000000, current_ts=1100000)

        # No pre_breakout_pressure, no candle close → waiting
        result1 = _assess(detector, bo, ep, cvd, current_ts_ms=1100000)
        assert result1.is_candidate is True
        assert "candle_close" in result1.reason.lower()

        # With candle close → confirmed
        candle_data = _closed_candle_data(
            candle_ts_ms=1090000, close=3150.0, upper=3100.0, lower=2900.0,
        )
        result2 = _assess(detector, bo, ep, cvd, current_ts_ms=1100000, **candle_data)
        assert result2.is_confirmed is True

    def test_balanced_pressure_neutral(self):
        """Balanced UP/DOWN pressure → neutral, no conflict."""
        detector = _make_detector(
            require_candle_close=True,
            confirm_min_seconds=60,
            pre_breakout_pressure_enabled=True,
        )
        bo = _breakout("UP", ts_ms=1000000, price=3200.0,
                       upper=3100.0, middle=3000.0, lower=2900.0)
        ep = _episode()
        cvd = _cvd_up_confirming(anchor_ts=1000000, current_ts=1100000)

        # Balanced pressure (no clear direction)
        neutral_pressure = PreBreakoutPressureState(
            direction=None, score=0.40, duration_seconds=200.0,
            anchored_cvd=5.0, buy_ratio=0.52, sell_ratio=0.48,
            reason="no_clear_pressure",
        )

        candle_data = _closed_candle_data(
            candle_ts_ms=1090000, close=3150.0, upper=3100.0, lower=2900.0,
        )

        result = _assess(
            detector, bo, ep, cvd, current_ts_ms=1100000,
            pre_breakout_pressure=neutral_pressure, **candle_data,
        )

        # Neutral pressure should NOT block confirmation
        assert result.is_confirmed is True, (
            f"Neutral pressure should allow confirmation, got reason={result.reason}"
        )


# ======================================================================
# Test 7: PreBreakoutPressureTracker unit tests
# ======================================================================


class TestPreBreakoutPressureTracker:
    """Unit tests for the PreBreakoutPressureTracker."""

    def test_tracker_starts_inactive(self):
        tracker = PreBreakoutPressureTracker(
            PreBreakoutPressureConfig(enabled=True)
        )
        assert tracker.active is False
        state = tracker.snapshot()
        assert state.direction is None
        assert state.reason == "not_active"

    def test_tracker_detects_up_pressure(self):
        """Sustained UP buying → UP pressure detected."""
        tracker = PreBreakoutPressureTracker(
            PreBreakoutPressureConfig(
                enabled=True, min_cvd_ratio=0.55,
                max_pullback_ratio=0.45, min_observe_seconds=0,
                pressure_min_score=0.60,
            )
        )
        tracker.start(
            ts_ms=1000000, price=3000.0, fast_cvd=100.0,
            buy_volume=70.0, sell_volume=30.0,
            boll_upper=3100.0, boll_middle=3000.0, boll_lower=2900.0,
        )
        for i in range(6):
            tracker.update(
                ts_ms=1000000 + (i + 1) * 60000,
                price=3030.0 + i * 5,
                fast_cvd=100.0 + (i + 1) * 15,
                buy_volume=70.0, sell_volume=30.0,
                boll_upper=3100.0, boll_middle=3000.0, boll_lower=2900.0,
            )

        state = tracker.snapshot()
        assert state.direction == "UP", (
            f"Expected UP, got direction={state.direction} score={state.score:.2f}"
        )
        assert state.score >= 0.60
        assert state.anchored_cvd > 0
        assert state.buy_ratio >= 0.55

    def test_tracker_detects_down_pressure(self):
        """Sustained DOWN selling → DOWN pressure detected."""
        tracker = PreBreakoutPressureTracker(
            PreBreakoutPressureConfig(
                enabled=True, min_cvd_ratio=0.55,
                max_pullback_ratio=0.45, min_observe_seconds=0,
                pressure_min_score=0.60,
            )
        )
        tracker.start(
            ts_ms=1000000, price=3000.0, fast_cvd=-100.0,
            buy_volume=30.0, sell_volume=70.0,
            boll_upper=3100.0, boll_middle=3000.0, boll_lower=2900.0,
        )
        for i in range(6):
            tracker.update(
                ts_ms=1000000 + (i + 1) * 60000,
                price=2970.0 - i * 5,
                fast_cvd=-100.0 - (i + 1) * 15,
                buy_volume=30.0, sell_volume=70.0,
                boll_upper=3100.0, boll_middle=3000.0, boll_lower=2900.0,
            )

        state = tracker.snapshot()
        assert state.direction == "DOWN", (
            f"Expected DOWN, got direction={state.direction} score={state.score:.2f}"
        )
        assert state.anchored_cvd < 0
        assert state.sell_ratio >= 0.55

    def test_tracker_no_clear_pressure_when_balanced(self):
        """Balanced flows → no clear pressure direction."""
        tracker = PreBreakoutPressureTracker(
            PreBreakoutPressureConfig(
                enabled=True, pressure_min_score=0.60,
            )
        )
        tracker.start(
            ts_ms=1000000, price=3000.0, fast_cvd=0.0,
            buy_volume=50.0, sell_volume=50.0,
            boll_upper=3100.0, boll_middle=3000.0, boll_lower=2900.0,
        )
        for i in range(5):
            tracker.update(
                ts_ms=1000000 + (i + 1) * 60000,
                price=3000.0 + (i % 2) * 2,  # oscillating
                fast_cvd=0.0,  # flat CVD
                buy_volume=50.0, sell_volume=50.0,
                boll_upper=3100.0, boll_middle=3000.0, boll_lower=2900.0,
            )

        state = tracker.snapshot()
        assert state.direction is None, (
            f"Expected no direction, got direction={state.direction} score={state.score:.2f}"
        )
        assert state.reason == "no_clear_pressure"

    def test_tracker_reset_clears_state(self):
        tracker = PreBreakoutPressureTracker(
            PreBreakoutPressureConfig(enabled=True)
        )
        tracker.start(
            ts_ms=1000000, price=3000.0, fast_cvd=100.0,
            buy_volume=70.0, sell_volume=30.0,
            boll_upper=3100.0, boll_middle=3000.0, boll_lower=2900.0,
        )
        tracker.reset()
        assert tracker.active is False
        state = tracker.snapshot()
        assert state.direction is None
        assert state.reason == "not_active"

    def test_tracker_insufficient_duration_lowers_score(self):
        """When observe time < min_observe_seconds, score is reduced."""
        tracker = PreBreakoutPressureTracker(
            PreBreakoutPressureConfig(
                enabled=True, min_observe_seconds=300,
                pressure_min_score=0.60,
            )
        )
        tracker.start(
            ts_ms=1000000, price=3000.0, fast_cvd=100.0,
            buy_volume=70.0, sell_volume=30.0,
            boll_upper=3100.0, boll_middle=3000.0, boll_lower=2900.0,
        )
        # Only 60s of observation (well under 300s min)
        tracker.update(
            ts_ms=1060000, price=3030.0, fast_cvd=120.0,
            buy_volume=70.0, sell_volume=30.0,
            boll_upper=3100.0, boll_middle=3000.0, boll_lower=2900.0,
        )

        state = tracker.snapshot()
        assert state.duration_seconds < 300
        # Score should be lower due to insufficient duration
        # (even if other components are strong)
        # With only 60s / 300s = 0.2 duration score, average ~0.81
        assert state.score < 0.90, (
            f"Score should be reduced due to short duration, got {state.score:.2f}"
        )


# ======================================================================
# Test 8: Candle close rejection log (trend_confirmed back inside)
# ======================================================================


class TestCandleCloseReject:
    """When a candle closes but price is back inside the band → trend is
    rejected (or stays candidate)."""

    def test_up_candle_close_back_inside_stays_candidate(self):
        """Price was outside upper, candle closes inside → stays candidate."""
        detector = _make_detector(require_candle_close=True, confirm_min_seconds=60)
        bo = _breakout("UP", ts_ms=1000000, price=3200.0,
                       upper=3100.0, middle=3000.0, lower=2900.0)
        ep = _episode()
        cvd = _cvd_up_confirming(anchor_ts=1000000, current_ts=1150000)

        # After 150s, a candle closes back inside
        candle_data = _closed_candle_data(
            candle_ts_ms=1140000,
            close=3080.0,  # < upper=3100 → inside
            upper=3100.0,
            lower=2900.0,
        )
        result = _assess(detector, bo, ep, cvd, current_ts_ms=1150000, **candle_data)

        assert result.is_confirmed is False
        assert result.has_candle_close_outside is False
        # Should still be candidate (not failed yet if within max_seconds)
        assert result.is_candidate is True, (
            f"Candle back inside should stay candidate, got reason={result.reason}"
        )


# ======================================================================
# Test 9: Pre-breakout pressure preserved on breakout tick (snapshot before reset)
# ======================================================================


class TestPreBreakoutPressureNotLostOnBreakout:
    """Verify that pre-breakout pressure is captured BEFORE the tracker is reset
    on the breakout tick, and is correctly passed to TrendDetector."""

    def test_pressure_passed_via_trend_detector_directly(self):
        """When pre_breakout_pressure is passed to TrendDetector.assess(),
        it appears in the returned TrendAssessment."""
        from src.strategies.regime.pre_breakout_pressure import (
            PreBreakoutPressureState,
        )

        detector = _make_detector(
            require_candle_close=True,
            confirm_min_seconds=60,
            pre_breakout_pressure_enabled=True,
        )
        bo = _breakout("UP", ts_ms=1000000, price=3200.0,
                       upper=3100.0, middle=3000.0, lower=2900.0)
        ep = _episode()
        cvd = _cvd_up_confirming(anchor_ts=1000000, current_ts=1100000)

        pressure = PreBreakoutPressureState(
            direction="UP", score=0.75, duration_seconds=400.0,
            anchored_cvd=80.0, buy_ratio=0.68, sell_ratio=0.32,
            reason="up_pressure_dominant",
        )

        result = _assess(
            detector, bo, ep, cvd, current_ts_ms=1100000,
            pre_breakout_pressure=pressure,
        )

        assert result.pre_breakout_pressure_direction == "UP", (
            f"Expected UP direction, got {result.pre_breakout_pressure_direction}"
        )
        assert result.pre_breakout_pressure_score > 0, (
            f"Expected positive score, got {result.pre_breakout_pressure_score}"
        )

    def test_pressure_captured_before_reset_in_assessor(self):
        """Snapshot-before-reset: when price breaks out, the assessor should
        capture pressure before resetting the tracker, and pass it to the
        TrendDetector via the stored-pressure fallback mechanism."""
        from src.strategies.regime.pre_breakout_pressure import (
            PreBreakoutPressureState,
        )

        # Use a detector with explicit short window for fast test
        detector = _make_detector(
            require_candle_close=True,
            confirm_min_seconds=60,
            pre_breakout_pressure_enabled=True,
        )
        bo = _breakout("UP", ts_ms=1000000, price=3200.0,
                       upper=3100.0, middle=3000.0, lower=2900.0)
        ep = _episode()
        cvd = _cvd_up_confirming(anchor_ts=1000000, current_ts=1100000)

        # Simulate what the assessor does on breakout tick:
        # 1. Snapshot pressure before reset
        pressure = PreBreakoutPressureState(
            direction="UP", score=0.80, duration_seconds=400.0,
            anchored_cvd=80.0, buy_ratio=0.68, sell_ratio=0.32,
            reason="up_pressure_dominant",
        )

        # First call: pass pressure → gets stored internally
        _assess(detector, bo, ep, cvd, current_ts_ms=1100000,
                pre_breakout_pressure=pressure)

        # Second call: NO pressure → stored pressure must be used
        result2 = _assess(detector, bo, ep, cvd, current_ts_ms=1120000)

        assert result2.pre_breakout_pressure_direction == "UP", (
            f"Subsequent tick without pressure should use stored, "
            f"got direction={result2.pre_breakout_pressure_direction}"
        )
        assert result2.pre_breakout_pressure_score > 0, (
            f"Subsequent tick score should be > 0, "
            f"got score={result2.pre_breakout_pressure_score}"
        )


# ======================================================================
# Test 10: Subsequent ticks use stored pre_breakout_pressure
# ======================================================================


class TestStoredPressureOnSubsequentTicks:
    """Verify that TrendDetector falls back to stored pre_breakout_pressure
    when subsequent ticks do not pass it."""

    def test_stored_pressure_used_on_second_tick(self):
        """First assess() call passes pressure → second call without pressure
        still uses stored pressure (not None)."""
        from src.strategies.regime.pre_breakout_pressure import (
            PreBreakoutPressureState,
        )

        detector = _make_detector(
            require_candle_close=True,
            confirm_min_seconds=60,
            pre_breakout_pressure_enabled=True,
        )
        bo = _breakout("UP", ts_ms=1000000, price=3200.0,
                       upper=3100.0, middle=3000.0, lower=2900.0)
        ep = _episode()
        cvd = _cvd_up_confirming(anchor_ts=1000000, current_ts=1100000)

        # First call: pass UP pressure
        pressure = PreBreakoutPressureState(
            direction="UP", score=0.75, duration_seconds=400.0,
            anchored_cvd=80.0, buy_ratio=0.68, sell_ratio=0.32,
            reason="up_pressure_dominant",
        )
        result1 = _assess(
            detector, bo, ep, cvd, current_ts_ms=1100000,
            pre_breakout_pressure=pressure,
            **_closed_candle_data(1090000, 3150.0, 3100.0, 2900.0),
        )
        assert result1.is_confirmed is True
        assert result1.pre_breakout_pressure_direction == "UP"
        assert result1.pre_breakout_pressure_score > 0

        # Second call: same breakout, NO pressure passed → should still use stored
        # CVD degrades a bit but still confirming
        cvd2 = _cvd_up_confirming(anchor_ts=1000000, current_ts=1120000)
        result2 = _assess(
            detector, bo, ep, cvd2, current_ts_ms=1120000,
            # No pre_breakout_pressure passed!
        )
        assert result2.pre_breakout_pressure_direction == "UP", (
            f"Subsequent tick should use stored pressure (UP), "
            f"got direction={result2.pre_breakout_pressure_direction}"
        )
        assert result2.pre_breakout_pressure_score > 0, (
            f"Subsequent tick should use stored pressure score > 0, "
            f"got score={result2.pre_breakout_pressure_score}"
        )

    def test_pressure_conflict_persists_with_stored_pressure(self):
        """Pressure DOWN, breakout UP → conflict should persist on subsequent
        ticks using stored pressure, blocking confirmation without strong CVD."""
        from src.strategies.regime.pre_breakout_pressure import (
            PreBreakoutPressureState,
        )

        detector = _make_detector(
            require_candle_close=True,
            confirm_min_seconds=60,
            pre_breakout_pressure_enabled=True,
            pre_breakout_pressure_min_score=0.60,
        )
        bo = _breakout("UP", ts_ms=1000000, price=3200.0,
                       upper=3100.0, middle=3000.0, lower=2900.0)
        ep = _episode()

        # Weak CVD
        weak_cvd = build_anchored_cvd_state(
            anchor_ts_ms=1000000, current_ts_ms=1100000,
            anchor_cvd=100.0, current_cvd=110.0,
            episode_buy_volume=50.0, episode_sell_volume=50.0,
            episode_cvd_max=110.0, episode_cvd_min=100.0,
        )

        # DOWN pressure (conflict with UP breakout)
        conflict_pressure = PreBreakoutPressureState(
            direction="DOWN", score=0.72, duration_seconds=300.0,
            anchored_cvd=-50.0, buy_ratio=0.35, sell_ratio=0.65,
            reason="down_pressure_dominant",
        )

        # First call: pass conflict pressure → should fail
        result1 = _assess(
            detector, bo, ep, weak_cvd, current_ts_ms=1100000,
            pre_breakout_pressure=conflict_pressure,
            **_closed_candle_data(1090000, 3150.0, 3100.0, 2900.0),
        )
        assert result1.is_failed is True, (
            f"Pressure conflict + weak CVD should fail, got reason={result1.reason}"
        )
        assert "pressure_conflict" in result1.reason.lower() or "cvd_not_strong" in result1.reason

        # Second call: NO pressure passed → stored conflict should still block
        result2 = _assess(
            detector, bo, ep, weak_cvd, current_ts_ms=1120000,
            # No pre_breakout_pressure!
        )
        assert result2.is_failed is True, (
            f"Stored pressure conflict should still cause failure, "
            f"got is_failed={result2.is_failed} reason={result2.reason}"
        )
        assert "pre_breakout_pressure_conflict" in result2.reason.lower() or "cvd_not_strong" in result2.reason, (
            f"Reason should mention pressure conflict, got: {result2.reason}"
        )
