from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.strategies.regime.anchored_cvd import (
    AnchoredCvdConfig,
    is_cvd_confirming_trend,
    is_cvd_diverging_from_price,
)
from src.strategies.regime.compression_detector import (
    CompressionDetector,
    CompressionDetectorConfig,
)
from src.strategies.regime.types import (
    AnchoredCvdState,
    BreakoutSnapshot,
    CompressionEpisode,
    TrendState,
)


@dataclass(frozen=True)
class TrendDetectorConfig:
    confirm_min_seconds: int = 60
    confirm_max_seconds: int = 180
    range_expansion_ratio_min: float = 3.0
    volume_expansion_ratio_min: float = 3.0
    outside_occupancy_min_ratio: float = 0.70
    min_new_extreme_count: int = 2
    max_inside_reclaim_seconds: int = 3

    def __post_init__(self) -> None:
        if self.confirm_min_seconds <= 0:
            raise ValueError(
                f"confirm_min_seconds ({self.confirm_min_seconds}) must be > 0"
            )
        if self.confirm_max_seconds < self.confirm_min_seconds:
            raise ValueError(
                f"confirm_max_seconds ({self.confirm_max_seconds}) must be >= "
                f"confirm_min_seconds ({self.confirm_min_seconds})"
            )
        if self.range_expansion_ratio_min <= 0:
            raise ValueError(
                f"range_expansion_ratio_min ({self.range_expansion_ratio_min}) must be > 0"
            )
        if self.volume_expansion_ratio_min <= 0:
            raise ValueError(
                f"volume_expansion_ratio_min ({self.volume_expansion_ratio_min}) must be > 0"
            )
        if not (0 < self.outside_occupancy_min_ratio <= 1):
            raise ValueError(
                f"outside_occupancy_min_ratio ({self.outside_occupancy_min_ratio}) "
                f"must be in (0, 1]"
            )
        if self.min_new_extreme_count < 0:
            raise ValueError(
                f"min_new_extreme_count ({self.min_new_extreme_count}) must be >= 0"
            )
        if self.max_inside_reclaim_seconds < 0:
            raise ValueError(
                f"max_inside_reclaim_seconds ({self.max_inside_reclaim_seconds}) must be >= 0"
            )


@dataclass(frozen=True)
class TrendAssessment:
    """Result of a single trend evaluation tick."""
    trend_state: TrendState
    is_candidate: bool
    is_confirmed: bool
    is_failed: bool
    reason: str


class TrendDetector:
    """Stateful detector that tracks trend evolution from compression through
    candidate → confirmed → failed.

    Uses event-anchored cumulative CVD (not 5-second CVD) for core confirmation.
    """

    def __init__(
        self,
        config: TrendDetectorConfig,
        compression_detector: CompressionDetector,
        cvd_config: AnchoredCvdConfig,
    ) -> None:
        self._config = config
        self._compression_detector = compression_detector
        self._cvd_config = cvd_config
        self._state: TrendState = TrendState.NO_TREND
        self._breakout: Optional[BreakoutSnapshot] = None
        self._episode: Optional[CompressionEpisode] = None
        self._failure_reason: Optional[str] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> TrendState:
        return self._state

    @property
    def failure_reason(self) -> Optional[str]:
        return self._failure_reason

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(
        self,
        *,
        breakout: BreakoutSnapshot,
        compression_episode: Optional[CompressionEpisode],
        anchored_cvd: AnchoredCvdState,
        current_ts_ms: int,
        range_expansion_passed: bool,
        volume_expansion_passed: bool,
        sustained_volume_passed: bool,
        outside_occupancy_passed: bool,
        new_extreme_count: int,
        inside_reclaim_seconds: float,
        price_reclaimed_inside: bool,
    ) -> TrendAssessment:
        """Evaluate trend state for one tick.

        Returns a :class:`TrendAssessment` with the current state and flags.
        This is a pure-logic method — no side effects besides updating
        internal state.
        """
        self._breakout = breakout
        self._episode = compression_episode

        # ── Guard: no recent compression → no trend ──────────────────
        if compression_episode is None:
            return self._set_failed("no_recent_compression")

        if current_ts_ms > compression_episode.valid_until_ts_ms:
            return self._set_failed("compression_expired")

        # ── Determine breakout duration ──────────────────────────────
        breakout_duration_sec = (current_ts_ms - breakout.ts_ms) / 1000.0

        # ── Check trend-candidate prerequisites ──────────────────────
        if not range_expansion_passed:
            # Insufficient range expansion yet — still expanding
            self._state = TrendState.POST_COMPRESSION_EXPANDING
            return TrendAssessment(
                trend_state=self._state,
                is_candidate=False,
                is_confirmed=False,
                is_failed=False,
                reason="range_expansion_not_met",
            )

        if not volume_expansion_passed:
            self._state = TrendState.POST_COMPRESSION_EXPANDING
            return TrendAssessment(
                trend_state=self._state,
                is_candidate=False,
                is_confirmed=False,
                is_failed=False,
                reason="volume_expansion_not_met",
            )

        # ── Check trend-failure conditions ───────────────────────────
        # 1. Confirm_max_seconds exceeded
        if breakout_duration_sec > self._config.confirm_max_seconds:
            return self._set_failed("confirm_max_seconds_exceeded")

        # 2. Price reclaimed inside too long
        if price_reclaimed_inside and inside_reclaim_seconds > self._config.max_inside_reclaim_seconds:
            return self._set_failed("inside_reclaim_too_long")

        # 3. Outside occupancy too low (must have enough data)
        if breakout_duration_sec >= self._config.confirm_min_seconds and not outside_occupancy_passed:
            return self._set_failed("outside_occupancy_insufficient")

        # 4. Anchored CVD diverges
        direction = breakout.direction
        price_new_extreme = new_extreme_count >= self._config.min_new_extreme_count
        if price_new_extreme and is_cvd_diverging_from_price(
            direction, anchored_cvd, True, self._cvd_config
        ):
            return self._set_failed("cvd_diverges_from_price")

        # ── Set candidate direction ──────────────────────────────────
        if breakout_duration_sec >= self._config.confirm_min_seconds:
            if direction == "UP":
                self._state = TrendState.TREND_UP_CANDIDATE
            else:
                self._state = TrendState.TREND_DOWN_CANDIDATE
        else:
            # Still within min_seconds window — just expanding
            self._state = TrendState.POST_COMPRESSION_EXPANDING
            return TrendAssessment(
                trend_state=self._state,
                is_candidate=False,
                is_confirmed=False,
                is_failed=False,
                reason="waiting_confirm_min_seconds",
            )

        # ── Check confirmed conditions ───────────────────────────────
        if self._is_confirmed(
            breakout, anchored_cvd, outside_occupancy_passed,
            sustained_volume_passed, new_extreme_count,
        ):
            if direction == "UP":
                self._state = TrendState.TREND_UP_CONFIRMED
            else:
                self._state = TrendState.TREND_DOWN_CONFIRMED
            return TrendAssessment(
                trend_state=self._state,
                is_candidate=False,  # promoted to confirmed
                is_confirmed=True,
                is_failed=False,
                reason="trend_confirmed",
            )

        # Still a candidate
        return TrendAssessment(
            trend_state=self._state,
            is_candidate=True,
            is_confirmed=False,
            is_failed=False,
            reason="trend_candidate_waiting_confirmation",
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _is_confirmed(
        self,
        breakout: BreakoutSnapshot,
        anchored_cvd: AnchoredCvdState,
        outside_occupancy_passed: bool,
        sustained_volume_passed: bool,
        new_extreme_count: int,
    ) -> bool:
        """All conditions must pass for trend to be confirmed."""
        if not outside_occupancy_passed:
            return False
        if not sustained_volume_passed:
            return False
        if new_extreme_count < self._config.min_new_extreme_count:
            return False
        # Event-anchored cumulative CVD confirms direction
        if not is_cvd_confirming_trend(
            breakout.direction, anchored_cvd, self._cvd_config
        ):
            return False
        return True

    def _set_failed(self, reason: str) -> TrendAssessment:
        self._state = TrendState.TREND_FAILED
        self._failure_reason = reason
        return TrendAssessment(
            trend_state=self._state,
            is_candidate=False,
            is_confirmed=False,
            is_failed=True,
            reason=reason,
        )

    def reset(self) -> None:
        """Reset detector to initial state."""
        self._state = TrendState.NO_TREND
        self._breakout = None
        self._episode = None
        self._failure_reason = None
