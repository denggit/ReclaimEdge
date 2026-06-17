from __future__ import annotations

from src.strategies.regime.anchored_cvd import (
    AnchoredCvdConfig,
    build_anchored_cvd_state,
)
from src.strategies.regime.compression_detector import (
    CompressionDetector,
    CompressionDetectorConfig,
)
from src.strategies.regime.trend_detector import (
    TrendDetector,
    TrendDetectorConfig,
)
from src.strategies.regime.types import (
    BandSnapshot,
    BreakoutSnapshot,
    CompressionEpisode,
    TrendState,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _band_snapshot(upper: float = 2100.0, middle: float = 2000.0, lower: float = 1900.0, candle_ts_ms: int = 1000) -> BandSnapshot:
    return BandSnapshot(upper=upper, middle=middle, lower=lower, candle_ts_ms=candle_ts_ms)


def _breakout(direction: str = "UP", ts_ms: int = 10000, price: float = 2110.0,
               anchor_cvd: float = 100.0, anchor_volume: float = 1000.0,
               upper: float = 2100.0, middle: float = 2000.0, lower: float = 1900.0) -> BreakoutSnapshot:
    return BreakoutSnapshot(
        direction=direction,  # type: ignore[arg-type]
        ts_ms=ts_ms,
        price=price,
        band=_band_snapshot(upper=upper, middle=middle, lower=lower, candle_ts_ms=ts_ms),
        anchor_cvd=anchor_cvd,
        anchor_volume=anchor_volume,
    )


def _episode(
    start_ts_ms: int = 0,
    end_ts_ms: int = 9000,
    valid_until_ts_ms: int = 9_000_000_000,  # far future
    candle_count: int = 12,
) -> CompressionEpisode:
    return CompressionEpisode(
        start_ts_ms=start_ts_ms,
        end_ts_ms=end_ts_ms,
        valid_until_ts_ms=valid_until_ts_ms,
        compressed_candle_count=candle_count,
        min_outer_distance_pct=0.001,
        avg_outer_distance_pct=0.002,
        upper_at_end=2100.0,
        middle_at_end=2000.0,
        lower_at_end=1900.0,
        highest_band_upper=2100.0,
        lowest_band_lower=1900.0,
    )


def _cvd_up_confirming(anchor_cvd: float = 100.0, current_cvd: float = 160.0,
                       buy_vol: float = 80.0, sell_vol: float = 20.0,
                       cvd_max: float = 160.0, cvd_min: float = 100.0,
                       anchor_ts: int = 10000, current_ts: int = 20000):
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
    trend_cfg: dict | None = None,
    comp_cfg: dict | None = None,
    cvd_cfg: dict | None = None,
) -> TrendDetector:
    # Default require_candle_close=False for backward compat with existing tests.
    # Use explicit confirm window (60/180) — new defaults are 900/1200.
    default_cfg = {"require_candle_close": False, "confirm_min_seconds": 60, "confirm_max_seconds": 180}
    default_cfg.update(trend_cfg or {})
    tcfg = TrendDetectorConfig(**default_cfg)
    ccfg = CompressionDetectorConfig(**(comp_cfg or {}))
    cvd_cfg_ = AnchoredCvdConfig(**(cvd_cfg or {}))
    return TrendDetector(tcfg, CompressionDetector(ccfg), cvd_cfg_)


def _assess_pass(detector: TrendDetector, breakout: BreakoutSnapshot,
                 episode: CompressionEpisode, cvd, current_ts_ms: int = 20000,
                 **overrides):
    """Call assess() with all "passed" flags by default."""
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


# ── Tests ─────────────────────────────────────────────────────────────


class TestTrendUpConfirmed:
    """Test 1: all conditions met → TREND_UP_CONFIRMED."""

    def test_full_conditions_trend_up_confirmed(self):
        detector = _make_detector()
        bo = _breakout("UP", ts_ms=10000)
        ep = _episode()
        cvd = _cvd_up_confirming()

        result = _assess_pass(detector, bo, ep, cvd, current_ts_ms=bo.ts_ms + 70_000)
        assert result.is_confirmed is True
        assert result.is_candidate is False
        assert result.is_failed is False
        assert result.blocks_mean_reversion is True
        assert result.trend_state == TrendState.TREND_UP_CONFIRMED


class TestTrendCandidateWaiting:
    """Test 2: conditions met but min_seconds not reached → TREND_UP_CANDIDATE."""

    def test_min_seconds_not_reached_stays_expanding(self):
        detector = _make_detector()
        bo = _breakout("UP", ts_ms=10000)
        ep = _episode()
        cvd = _cvd_up_confirming(anchor_ts=10000, current_ts=10010)

        # Only 10ms after breakout — well under 60s
        # Expansion passed → immediately becomes candidate, blocks MR
        result = _assess_pass(detector, bo, ep, cvd, current_ts_ms=10010)
        assert result.is_confirmed is False
        assert result.is_candidate is True
        assert result.blocks_mean_reversion is True
        assert result.trend_state == TrendState.TREND_UP_CANDIDATE
        assert result.reason == "trend_candidate_waiting_min_seconds"

    def test_min_seconds_reached_but_occupancy_not_passed_stays_candidate(self):
        detector = _make_detector()
        bo = _breakout("UP", ts_ms=10000)
        ep = _episode()
        cvd = _cvd_up_confirming()

        # 70s after breakout, but outside occupancy fails
        result = _assess_pass(detector, bo, ep, cvd, current_ts_ms=bo.ts_ms + 70_000,
                              outside_occupancy_passed=False)
        assert result.is_failed is True
        assert result.blocks_mean_reversion is False
        assert result.reason == "outside_occupancy_insufficient"


class TestTrendFailedInsideReclaim:
    """Test 3: price reclaimed inside too long → TREND_FAILED."""

    def test_inside_reclaim_too_long(self):
        detector = _make_detector({"max_inside_reclaim_seconds": 3})
        bo = _breakout("UP", ts_ms=10000)
        ep = _episode()
        cvd = _cvd_up_confirming()

        result = _assess_pass(detector, bo, ep, cvd, current_ts_ms=bo.ts_ms + 70_000,
                              price_reclaimed_inside=True, inside_reclaim_seconds=5.0)
        assert result.is_failed is True
        assert result.blocks_mean_reversion is False
        assert result.reason == "inside_reclaim_too_long"


class TestTrendFailedCvdDiverges:
    """Test 4: anchored CVD diverges against breakout → TREND_FAILED."""

    def test_cvd_diverges_fails_trend(self):
        detector = _make_detector()
        bo = _breakout("UP", ts_ms=10000, anchor_cvd=100.0)
        ep = _episode()
        # CVD going down while price goes up → diverges
        cvd = build_anchored_cvd_state(
            anchor_ts_ms=10000, current_ts_ms=bo.ts_ms + 70_000,
            anchor_cvd=100.0, current_cvd=90.0,
            episode_buy_volume=30.0, episode_sell_volume=70.0,
            episode_cvd_max=100.0, episode_cvd_min=90.0,
        )

        result = _assess_pass(detector, bo, ep, cvd, current_ts_ms=bo.ts_ms + 70_000)
        assert result.is_failed is True
        assert result.blocks_mean_reversion is False
        assert "cvd_diverges" in result.reason


class TestCompressionExpiredNoTrend:
    """Test 5: compression expired → behavior depends on active candidate."""

    def test_compression_expired(self):
        """Compression expired while trend candidate was active → TREND_FAILED."""
        detector = _make_detector()
        bo = _breakout("UP", ts_ms=10000)
        # First, establish an active candidate
        ep_active = _episode(valid_until_ts_ms=9_000_000_000)
        cvd = _cvd_up_confirming(anchor_ts=10000, current_ts=10010)
        # This sets state to TREND_UP_CANDIDATE
        _assess_pass(detector, bo, ep_active, cvd, current_ts_ms=10010)

        # Now call again with an expired episode — candidate was active → fail
        ep_expired = _episode(valid_until_ts_ms=5000)
        result = _assess_pass(detector, bo, ep_expired, cvd, current_ts_ms=bo.ts_ms + 70_000)
        assert result.is_failed is True
        assert result.blocks_mean_reversion is False
        assert result.reason == "compression_expired"

    def test_compression_expired_no_candidate(self):
        """Compression expired but no active candidate → NO_TREND, not failed."""
        detector = _make_detector()
        bo = _breakout("UP", ts_ms=10000)
        # Episode already expired (valid_until in the past), no prior candidate
        ep = _episode(valid_until_ts_ms=5000)
        cvd = _cvd_up_confirming()

        result = _assess_pass(detector, bo, ep, cvd, current_ts_ms=bo.ts_ms + 70_000)
        assert result.is_failed is False
        assert result.blocks_mean_reversion is False
        assert result.trend_state == TrendState.NO_TREND
        assert result.reason == "compression_expired_no_candidate"


class TestCompressionEndedButStillValid:
    """Test 6: compression ended 60 min ago, within valid_after_seconds → trend allowed."""

    def test_compression_ended_60m_ago_still_allowed(self):
        # Create detector with 7200s valid_after
        detector = _make_detector(comp_cfg={"valid_after_seconds": 7200})
        bo = _breakout("UP", ts_ms=100000)
        ep = _episode(
            start_ts_ms=0,
            end_ts_ms=10000,  # compression ended 90s before breakout
            valid_until_ts_ms=10000 + 7200 * 1000,  # valid for 2 hours
        )
        cvd = _cvd_up_confirming(anchor_ts=100000, current_ts=100000 + 70_000)

        result = _assess_pass(detector, bo, ep, cvd, current_ts_ms=100000 + 70_000)
        assert result.is_confirmed is True
        assert result.blocks_mean_reversion is True
        assert result.trend_state == TrendState.TREND_UP_CONFIRMED


class TestNoCompressionEpisodeNoTrend:
    """Test: no compression episode → no trend."""

    def test_no_episode(self):
        detector = _make_detector()
        bo = _breakout("UP", ts_ms=10000)
        cvd = _cvd_up_confirming()

        result = _assess_pass(detector, bo, None, cvd, current_ts_ms=bo.ts_ms + 70_000)
        assert result.is_failed is False
        assert result.is_candidate is False
        assert result.is_confirmed is False
        assert result.blocks_mean_reversion is False
        assert result.trend_state == TrendState.NO_TREND
        assert result.reason == "no_recent_compression"


class TestMaxConfirmSecondsExceeded:
    """Test: confirm_max_seconds exceeded → failed."""

    def test_max_seconds_exceeded(self):
        detector = _make_detector({"confirm_max_seconds": 180})
        bo = _breakout("UP", ts_ms=10000)
        ep = _episode()
        cvd = _cvd_up_confirming()

        # 200s after breakout → exceeds 180s max
        result = _assess_pass(detector, bo, ep, cvd, current_ts_ms=bo.ts_ms + 200_000)
        assert result.is_failed is True
        assert result.blocks_mean_reversion is False
        assert result.reason == "confirm_max_seconds_exceeded"


# ── New tests: boundary conditions ──────────────────────────────────────


class TestTrendCandidateBlocksMeanReversion:
    """Trend candidate in min_seconds window → blocks_mean_reversion=True."""

    def test_up_candidate_blocks_mr_before_confirm(self):
        detector = _make_detector()
        bo = _breakout("UP", ts_ms=10000)
        ep = _episode()
        cvd = _cvd_up_confirming(anchor_ts=10000, current_ts=10030)

        # 30s after breakout: expansion passed, CVD confirms, price still outside
        result = _assess_pass(detector, bo, ep, cvd, current_ts_ms=bo.ts_ms + 30_000,
                              new_extreme_count=1)
        assert result.is_candidate is True
        assert result.is_confirmed is False
        assert result.is_failed is False
        assert result.blocks_mean_reversion is True
        assert result.trend_state == TrendState.TREND_UP_CANDIDATE
        assert result.reason == "trend_candidate_waiting_min_seconds"

    def test_down_candidate_blocks_mr_before_confirm(self):
        detector = _make_detector()
        bo = _breakout("DOWN", ts_ms=10000, price=1890.0,
                       upper=2100.0, middle=2000.0, lower=1900.0)
        ep = _episode()
        # CVD confirming DOWN: sell dominant, CVD decreasing
        cvd = build_anchored_cvd_state(
            anchor_ts_ms=10000, current_ts_ms=10030,
            anchor_cvd=200.0, current_cvd=140.0,
            episode_buy_volume=20.0, episode_sell_volume=80.0,
            episode_cvd_max=200.0, episode_cvd_min=140.0,
        )

        result = _assess_pass(detector, bo, ep, cvd, current_ts_ms=bo.ts_ms + 30_000,
                              new_extreme_count=1)
        assert result.is_candidate is True
        assert result.is_confirmed is False
        assert result.is_failed is False
        assert result.blocks_mean_reversion is True
        assert result.trend_state == TrendState.TREND_DOWN_CANDIDATE
        assert result.reason == "trend_candidate_waiting_min_seconds"


class TestFastReclaimBeforeMinSeconds:
    """Fast reclaim + CVD divergence releases MR before 60 seconds."""

    def test_up_fast_reclaim_releases_mr(self):
        detector = _make_detector({"max_inside_reclaim_seconds": 3})
        bo = _breakout("UP", ts_ms=10000, anchor_cvd=100.0)
        ep = _episode()
        # CVD diverging: price went up but CVD going down
        cvd = build_anchored_cvd_state(
            anchor_ts_ms=10000, current_ts_ms=10010,
            anchor_cvd=100.0, current_cvd=85.0,
            episode_buy_volume=30.0, episode_sell_volume=70.0,
            episode_cvd_max=100.0, episode_cvd_min=85.0,
        )

        # 10s after breakout, inside reclaim > 3s, CVD diverges
        result = _assess_pass(detector, bo, ep, cvd, current_ts_ms=bo.ts_ms + 10_000,
                              price_reclaimed_inside=True, inside_reclaim_seconds=5.0,
                              new_extreme_count=1)
        assert result.is_failed is True
        assert result.blocks_mean_reversion is False
        assert result.reason == "fast_reclaim_with_cvd_divergence"

    def test_down_fast_reclaim_releases_mr(self):
        detector = _make_detector({"max_inside_reclaim_seconds": 3})
        bo = _breakout("DOWN", ts_ms=10000, price=1890.0,
                       upper=2100.0, middle=2000.0, lower=1900.0, anchor_cvd=200.0)
        ep = _episode()
        # CVD diverging: price went down but CVD going up
        cvd = build_anchored_cvd_state(
            anchor_ts_ms=10000, current_ts_ms=10010,
            anchor_cvd=200.0, current_cvd=250.0,
            episode_buy_volume=70.0, episode_sell_volume=30.0,
            episode_cvd_max=250.0, episode_cvd_min=200.0,
        )

        result = _assess_pass(detector, bo, ep, cvd, current_ts_ms=bo.ts_ms + 10_000,
                              price_reclaimed_inside=True, inside_reclaim_seconds=5.0,
                              new_extreme_count=1)
        assert result.is_failed is True
        assert result.blocks_mean_reversion is False
        assert result.reason == "fast_reclaim_with_cvd_divergence"


class TestTrendCandidateNotConfirmedBeforeMinSeconds:
    """Trend candidate active but not confirmed before min_seconds."""

    def test_not_enough_time_not_confirmed(self):
        detector = _make_detector()
        bo = _breakout("UP", ts_ms=10000)
        ep = _episode()
        cvd = _cvd_up_confirming(anchor_ts=10000, current_ts=10030)

        # 30s: all conditions good, but not enough time → candidate, not confirmed
        result = _assess_pass(detector, bo, ep, cvd, current_ts_ms=bo.ts_ms + 30_000)
        assert result.is_candidate is True
        assert result.is_confirmed is False
        assert result.is_failed is False
        assert result.blocks_mean_reversion is True
        assert result.reason == "trend_candidate_waiting_min_seconds"


class TestTrendConfirmedAfterMinSeconds:
    """60s+ and all conditions → confirmed trend."""

    def test_up_confirmed_after_min_seconds(self):
        detector = _make_detector()
        bo = _breakout("UP", ts_ms=10000)
        ep = _episode()
        cvd = _cvd_up_confirming(anchor_ts=10000, current_ts=bo.ts_ms + 70_000)

        result = _assess_pass(detector, bo, ep, cvd, current_ts_ms=bo.ts_ms + 70_000)
        assert result.is_candidate is False
        assert result.is_confirmed is True
        assert result.is_failed is False
        assert result.blocks_mean_reversion is True
        assert result.trend_state == TrendState.TREND_UP_CONFIRMED
        assert result.reason == "trend_confirmed"

    def test_down_confirmed_after_min_seconds(self):
        detector = _make_detector()
        bo = _breakout("DOWN", ts_ms=10000, price=1890.0,
                       upper=2100.0, middle=2000.0, lower=1900.0)
        ep = _episode()
        cvd = build_anchored_cvd_state(
            anchor_ts_ms=10000, current_ts_ms=bo.ts_ms + 70_000,
            anchor_cvd=200.0, current_cvd=140.0,
            episode_buy_volume=20.0, episode_sell_volume=80.0,
            episode_cvd_max=200.0, episode_cvd_min=140.0,
        )

        result = _assess_pass(detector, bo, ep, cvd, current_ts_ms=bo.ts_ms + 70_000)
        assert result.is_candidate is False
        assert result.is_confirmed is True
        assert result.is_failed is False
        assert result.blocks_mean_reversion is True
        assert result.trend_state == TrendState.TREND_DOWN_CONFIRMED
        assert result.reason == "trend_confirmed"


class TestNoCompressionIsNotTrendFailed:
    """no_recent_compression is NO_TREND, not TREND_FAILED."""

    def test_no_compression_not_failed(self):
        detector = _make_detector()
        bo = _breakout("UP", ts_ms=10000)
        cvd = _cvd_up_confirming()

        result = _assess_pass(detector, bo, None, cvd, current_ts_ms=bo.ts_ms + 70_000)
        assert result.is_failed is False
        assert result.blocks_mean_reversion is False
        assert result.trend_state == TrendState.NO_TREND
        assert result.reason == "no_recent_compression"


# ── New tests: stored pre-breakout pressure ────────────────────────────


class TestStoredPreBreakoutPressure:
    """Subsequent ticks without pre_breakout_pressure fall back to stored value."""

    def test_stored_pressure_used_when_not_passed(self):
        """First tick stores pressure → subsequent tick without pressure uses stored."""
        from src.strategies.regime.pre_breakout_pressure import (
            PreBreakoutPressureState,
        )

        detector = _make_detector({"require_candle_close": False,
                                   "confirm_min_seconds": 60, "confirm_max_seconds": 180})
        bo = _breakout("UP", ts_ms=10000)
        ep = _episode()
        cvd = _cvd_up_confirming(anchor_ts=10000, current_ts=80000)

        # First call: pass UP pressure → gets stored internally
        pressure = PreBreakoutPressureState(
            direction="UP", score=0.80, duration_seconds=400.0,
            anchored_cvd=80.0, buy_ratio=0.68, sell_ratio=0.32,
            reason="up_pressure_dominant",
        )
        result1 = _assess_pass(
            detector, bo, ep, cvd, current_ts_ms=80000,
            pre_breakout_pressure=pressure,
        )
        assert result1.is_confirmed is True
        assert result1.pre_breakout_pressure_direction == "UP"
        assert result1.pre_breakout_pressure_score == 0.80

        # Second call: NO pressure → stored pressure should be used
        result2 = _assess_pass(detector, bo, ep, cvd, current_ts_ms=85000)
        assert result2.pre_breakout_pressure_direction == "UP", (
            f"Should fall back to stored pressure direction=UP, "
            f"got {result2.pre_breakout_pressure_direction}"
        )
        assert result2.pre_breakout_pressure_score > 0, (
            f"Should fall back to stored pressure score > 0, "
            f"got {result2.pre_breakout_pressure_score}"
        )

    def test_current_pressure_takes_priority_over_stored(self):
        """When current tick passes new pressure, it overrides stored."""
        from src.strategies.regime.pre_breakout_pressure import (
            PreBreakoutPressureState,
        )

        detector = _make_detector({"require_candle_close": False,
                                   "confirm_min_seconds": 60, "confirm_max_seconds": 180})
        bo = _breakout("UP", ts_ms=10000)
        ep = _episode()
        cvd = _cvd_up_confirming(anchor_ts=10000, current_ts=80000)

        # Store UP pressure
        up_pressure = PreBreakoutPressureState(
            direction="UP", score=0.80, duration_seconds=400.0,
            anchored_cvd=80.0, buy_ratio=0.68, sell_ratio=0.32,
            reason="up_pressure_dominant",
        )
        _assess_pass(
            detector, bo, ep, cvd, current_ts_ms=80000,
            pre_breakout_pressure=up_pressure,
        )

        # Now pass new pressure (neutral) — should override stored UP
        neutral = PreBreakoutPressureState(
            direction=None, score=0.30, duration_seconds=100.0,
            anchored_cvd=5.0, buy_ratio=0.52, sell_ratio=0.48,
            reason="no_clear_pressure",
        )
        result = _assess_pass(
            detector, bo, ep, cvd, current_ts_ms=85000,
            pre_breakout_pressure=neutral,
        )
        # Should use the newly-passed neutral, not stored UP
        # (neutral pressure has direction=None → not stored)
        assert result.pre_breakout_pressure_direction is None, (
            f"Current neutral pressure should override stored UP, "
            f"got direction={result.pre_breakout_pressure_direction}"
        )


# ── Default config tests ───────────────────────────────────────────────


class TestTrendDetectorConfigDefaults:
    """Verify TrendDetectorConfig defaults are 900/1200 for candle close compat."""

    def test_default_confirm_window_is_900_1200(self):
        cfg = TrendDetectorConfig()
        assert cfg.confirm_min_seconds == 900, (
            f"Default confirm_min_seconds should be 900, got {cfg.confirm_min_seconds}"
        )
        assert cfg.confirm_max_seconds == 1200, (
            f"Default confirm_max_seconds should be 1200, got {cfg.confirm_max_seconds}"
        )
        assert cfg.require_candle_close is True

