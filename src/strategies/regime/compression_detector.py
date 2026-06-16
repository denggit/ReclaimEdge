from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import List, Optional

from src.strategies.regime.types import BandSnapshot, CompressionEpisode


@dataclass(frozen=True)
class CompressionDetectorConfig:
    lookback_candles: int = 96
    min_candles: int = 8
    percentile: float = 0.20
    max_outer_distance_pct: float = 0.005
    valid_after_seconds: int = 7200

    def __post_init__(self) -> None:
        if self.lookback_candles < self.min_candles:
            raise ValueError(
                f"lookback_candles ({self.lookback_candles}) must be >= "
                f"min_candles ({self.min_candles})"
            )
        if not (0 < self.percentile < 1):
            raise ValueError(
                f"percentile ({self.percentile}) must be in (0, 1)"
            )
        if self.max_outer_distance_pct <= 0:
            raise ValueError(
                f"max_outer_distance_pct ({self.max_outer_distance_pct}) must be > 0"
            )
        if self.valid_after_seconds < 0:
            raise ValueError(
                f"valid_after_seconds ({self.valid_after_seconds}) must be >= 0"
            )


class CompressionDetector:
    """Detects low-volatility compression episodes from BOLL band history.

    Maintains a *memory* of the most recent valid CompressionEpisode.  Once a
    compression episode ends, it remains valid for ``valid_after_seconds`` so
    that a breakout occurring well after the bands have already widened can
    still be treated as "post-compression breakout".
    """

    def __init__(self, config: CompressionDetectorConfig) -> None:
        self._config = config
        self._latest_episode: Optional[CompressionEpisode] = None

    @property
    def latest_episode(self) -> Optional[CompressionEpisode]:
        return self._latest_episode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        bands: List[BandSnapshot],
        current_ts_ms: int,
    ) -> Optional[CompressionEpisode]:
        """Scan *bands* (oldest → newest) and update latest compression memory.

        Returns a *new* episode when one is detected, otherwise ``None``.
        Use :meth:`get_valid_episode` to retrieve the memory (possibly stale
        but still within its validity window).
        """
        episode = self._scan(bands)
        if episode is not None:
            self._latest_episode = episode
        # Expire the memory if it's past valid_until_ts_ms
        if (
            self._latest_episode is not None
            and current_ts_ms > self._latest_episode.valid_until_ts_ms
        ):
            self._latest_episode = None
        return episode

    def get_valid_episode(self, current_ts_ms: int) -> Optional[CompressionEpisode]:
        """Return the latest compression episode if it is still valid."""
        if self._latest_episode is None:
            return None
        if current_ts_ms > self._latest_episode.valid_until_ts_ms:
            self._latest_episode = None
            return None
        return self._latest_episode

    def is_recent_compression_valid(self, current_ts_ms: int) -> bool:
        """Check whether a recent compression episode is still valid."""
        return self.get_valid_episode(current_ts_ms) is not None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _scan(
        self, bands: List[BandSnapshot]
    ) -> Optional[CompressionEpisode]:
        """Find the *most recent* complete compression episode in *bands*.

        An episode is a contiguous run of at least ``min_candles`` candles
        where EACH candle satisfies:

        1. ``outer_distance_pct <= max_outer_distance_pct``
        2. ``outer_distance_pct <= lookback_percentile``

        The lookback percentile is computed over the preceding
        ``lookback_candles`` bands.
        """
        if len(bands) < self._config.min_candles:
            return None

        distances = [_outer_distance_pct(b) for b in bands]

        # Walk backwards to find the most recent episode
        run_start: Optional[int] = None
        for i in range(len(bands) - 1, -1, -1):
            if self._is_compressed(distances, i):
                if run_start is None:
                    run_start = i  # candidate end of a backwards run
            else:
                if run_start is not None:
                    # We've found a complete run from (i+1) to run_start
                    start_idx = i + 1
                    end_idx = run_start
                    candle_count = end_idx - start_idx + 1
                    if candle_count >= self._config.min_candles:
                        return self._build_episode(bands, start_idx, end_idx)
                    run_start = None

        # Check if the run extends to the beginning
        if run_start is not None:
            start_idx = 0
            end_idx = run_start
            candle_count = end_idx - start_idx + 1
            if candle_count >= self._config.min_candles:
                return self._build_episode(bands, start_idx, end_idx)

        return None

    def _is_compressed(
        self, distances: List[float], index: int
    ) -> bool:
        d = distances[index]
        # Absolute check
        if d > self._config.max_outer_distance_pct:
            return False
        # Percentile check: need lookback_candles before this candle
        lookback = self._config.lookback_candles
        lookback_start = max(0, index - lookback)
        lookback_window = distances[lookback_start:index]
        if len(lookback_window) == 0:
            # Not enough history for percentile — fall back to absolute check
            return True
        threshold = _percentile(lookback_window, self._config.percentile)
        return d <= threshold

    def _build_episode(
        self,
        bands: List[BandSnapshot],
        start_idx: int,
        end_idx: int,
    ) -> CompressionEpisode:
        episode_bands = bands[start_idx : end_idx + 1]
        distances = [_outer_distance_pct(b) for b in episode_bands]
        candle_count = len(episode_bands)

        highs = [b.upper for b in episode_bands]  # proxy: no candle high in BandSnapshot
        lows = [b.lower for b in episode_bands]   # proxy: no candle low in BandSnapshot
        # Use upper as price proxy for highest_high_during_compression
        # and lower as proxy for lowest_low_during_compression
        highest_high = max(b.upper for b in episode_bands)
        lowest_low = min(b.lower for b in episode_bands)

        end_band = episode_bands[-1]
        valid_until_ts_ms = (
            end_band.candle_ts_ms + self._config.valid_after_seconds * 1000
        )

        return CompressionEpisode(
            start_ts_ms=episode_bands[0].candle_ts_ms,
            end_ts_ms=end_band.candle_ts_ms,
            valid_until_ts_ms=valid_until_ts_ms,
            compressed_candle_count=candle_count,
            min_outer_distance_pct=min(distances),
            avg_outer_distance_pct=mean(distances),
            upper_at_end=end_band.upper,
            middle_at_end=end_band.middle,
            lower_at_end=end_band.lower,
            highest_high=highest_high,
            lowest_low=lowest_low,
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _outer_distance_pct(band: BandSnapshot) -> float:
    """Distance from upper to middle as fraction of middle."""
    if band.middle == 0:
        return 0.0
    return abs(band.upper - band.middle) / abs(band.middle)


def _percentile(values: List[float], p: float) -> float:
    """Compute the p-th percentile using linear interpolation.

    Uses the same method as ``numpy.percentile`` with ``method='linear'``.
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    # Index position (0-based)
    k = (n - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[int(f)] * (c - k)
    d1 = sorted_vals[int(c)] * (k - f)
    return d0 + d1
