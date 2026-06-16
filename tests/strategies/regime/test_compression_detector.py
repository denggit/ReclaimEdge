from __future__ import annotations

from src.strategies.regime.compression_detector import (
    CompressionDetector,
    CompressionDetectorConfig,
)
from src.strategies.regime.types import BandSnapshot


def _band(candle_ts_ms: int, middle: float, outer_distance_pct: float) -> BandSnapshot:
    """Create a BandSnapshot with upper = middle * (1 + outer_distance_pct)."""
    upper = middle * (1.0 + outer_distance_pct)
    lower = middle * (1.0 - outer_distance_pct)
    return BandSnapshot(
        upper=upper,
        middle=middle,
        lower=lower,
        candle_ts_ms=candle_ts_ms,
    )


def _bands(middle: float, outer_distance_pct: float, count: int, start_ts_ms: int = 0, step_ms: int = 900_000) -> list[BandSnapshot]:
    """Generate *count* identical bands spaced *step_ms* apart."""
    result = []
    for i in range(count):
        result.append(_band(start_ts_ms + i * step_ms, middle, outer_distance_pct))
    return result


# ── Test helpers ──────────────────────────────────────────────────────


def make_config(**kwargs) -> CompressionDetectorConfig:
    defaults = dict(
        lookback_candles=96,
        min_candles=8,
        percentile=0.20,
        max_outer_distance_pct=0.005,
        valid_after_seconds=7200,
    )
    defaults.update(kwargs)
    return CompressionDetectorConfig(**defaults)


# ── Tests ─────────────────────────────────────────────────────────────


class TestCompressionEpisodeDetection:
    """Test 1: Consecutive low-percentile + absolute below 0.5% → CompressionEpisode."""

    def test_consecutive_low_distance_generates_episode(self):
        config = make_config(min_candles=8, lookback_candles=96)
        detector = CompressionDetector(config)

        # Create 96 lookback candles with varying (but low) distances
        bands = _bands(middle=2000.0, outer_distance_pct=0.004, count=96)
        # Then 8 more with same low distance
        bands += _bands(middle=2000.0, outer_distance_pct=0.003, count=8, start_ts_ms=96 * 900_000)

        episode = detector.detect(bands, current_ts_ms=bands[-1].candle_ts_ms + 1000)
        assert episode is not None
        assert episode.compressed_candle_count >= 8


class TestHighDistanceNoEpisode:
    """Test 2: Low percentile but outer_distance_pct > 0.5% → no episode."""

    def test_high_absolute_distance_blocks_episode(self):
        config = make_config(min_candles=8, max_outer_distance_pct=0.005, lookback_candles=96)
        detector = CompressionDetector(config)

        # Background: low distances
        bands = _bands(middle=2000.0, outer_distance_pct=0.003, count=96)
        # Then 8 candles with distance > 0.5%
        bands += _bands(middle=2000.0, outer_distance_pct=0.006, count=8, start_ts_ms=96 * 900_000)

        episode = detector.detect(bands, current_ts_ms=bands[-1].candle_ts_ms + 1000)
        # The last 8 candles fail the absolute check, so most recent episode
        # is the background one, but those are at the tail of lookback and
        # may or may not form an episode. The key assertion: the HIGH-distance
        # candles don't form a NEW episode.
        # detector.detect scans from the end, so it should find the earlier one
        # or nothing.
        # If episode is found, it should NOT include the high-distance candles.
        if episode is not None:
            assert episode.end_ts_ms < 96 * 900_000  # before high-distance candles


class TestCompressionMemoryAfterExpansion:
    """Test 3: Compression ended 1 hour ago, bands widened, still valid."""

    def test_compression_valid_after_bands_widen(self):
        config = make_config(min_candles=8, valid_after_seconds=7200, lookback_candles=96)
        detector = CompressionDetector(config)

        # Compression episode: bands 1–20
        bands = _bands(middle=2000.0, outer_distance_pct=0.002, count=20)
        episode_ts = bands[-1].candle_ts_ms

        # Detect the compression
        episode = detector.detect(bands, current_ts_ms=episode_ts + 1000)
        assert episode is not None

        # Now 1 hour later (3600s), bands have widened
        widened_bands = _bands(middle=2000.0, outer_distance_pct=0.008, count=4,
                               start_ts_ms=episode_ts + 900_000, step_ms=900_000)
        all_bands = bands + widened_bands
        current_ts = widened_bands[-1].candle_ts_ms + 1000

        # detect() re-scans the full history and re-detects the original
        # compression (it still exists in the old bands).  That's correct
        # — the episode memory is refreshed.  The key assertion: the
        # memory is still valid and the episode did NOT originate in the
        # widened region (its end_ts_ms is before the widened candles).
        re_detected = detector.detect(all_bands, current_ts_ms=current_ts)
        assert re_detected is not None  # old episode re-detected
        assert re_detected.end_ts_ms < widened_bands[0].candle_ts_ms  # NOT from widened region
        assert detector.is_recent_compression_valid(current_ts) is True

        valid = detector.get_valid_episode(current_ts)
        assert valid is not None
        assert valid.compressed_candle_count >= 8
        # The last candle in the full history is NOT compressed (0.008 > 0.005)
        # but the compression episode memory is still valid


class TestCompressionExpires:
    """Test 4: Compression episode past valid_after_seconds → expired."""

    def test_compression_expires_after_valid_window(self):
        config = make_config(min_candles=8, valid_after_seconds=3600, lookback_candles=96)
        detector = CompressionDetector(config)

        bands = _bands(middle=2000.0, outer_distance_pct=0.002, count=20)
        episode_ts = bands[-1].candle_ts_ms

        episode = detector.detect(bands, current_ts_ms=episode_ts + 1000)
        assert episode is not None

        # Move time forward past valid_after_seconds
        future_ts = episode.valid_until_ts_ms + 1000
        detector.detect([], current_ts_ms=future_ts)  # triggers expiry check
        assert detector.is_recent_compression_valid(future_ts) is False
        assert detector.get_valid_episode(future_ts) is None


class TestBreakoutNotRequireLastCandleCompressed:
    """Test 5: Breakout does NOT require the last candle to still be compressed."""

    def test_breakout_allowed_after_compression_ended(self):
        config = make_config(min_candles=8, valid_after_seconds=7200, lookback_candles=96)
        detector = CompressionDetector(config)

        # Build compression
        bands = _bands(middle=2000.0, outer_distance_pct=0.002, count=20)
        episode_ts = bands[-1].candle_ts_ms
        detector.detect(bands, current_ts_ms=episode_ts + 1000)

        # 30 minutes later: bands widened, price breaks out
        breakout_ts = episode_ts + 30 * 60_000 + 900_000
        widened = _bands(middle=2000.0, outer_distance_pct=0.010, count=2,
                         start_ts_ms=episode_ts + 900_000, step_ms=900_000)
        all_bands = bands + widened
        current_ts = widened[-1].candle_ts_ms + 1000

        detector.detect(all_bands, current_ts_ms=current_ts)
        # Compression memory should still be valid (only 30 min + 1 candle)
        assert detector.is_recent_compression_valid(current_ts) is True
        # The last candle is NOT compressed (0.010 > 0.005)
        # but the episode is still valid → breakout can be post-compression
