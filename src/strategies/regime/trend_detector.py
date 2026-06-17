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
from src.strategies.regime.pre_breakout_pressure import (
    PreBreakoutPressureState,
)
from src.strategies.regime.types import (
    AnchoredCvdState,
    BreakoutSnapshot,
    CompressionEpisode,
    TrendState,
)


@dataclass(frozen=True)
class TrendDetectorConfig:
    confirm_min_seconds: int = 900
    confirm_max_seconds: int = 1200
    range_expansion_ratio_min: float = 3.0
    volume_expansion_ratio_min: float = 3.0
    outside_occupancy_min_ratio: float = 0.70
    min_new_extreme_count: int = 2
    max_inside_reclaim_seconds: int = 3
    # ── Candle Close Confirmation ──────────────────────────────────────
    require_candle_close: bool = True
    # ── Pre-Breakout Pressure ──────────────────────────────────────────
    pre_breakout_pressure_enabled: bool = True
    pre_breakout_pressure_min_score: float = 0.60

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
    blocks_mean_reversion: bool
    reason: str
    # ── Candle Close Confirmation ──────────────────────────────────────
    has_candle_close_outside: bool = False
    confirmed_candle_ts_ms: int = 0
    # ── Pre-Breakout Pressure ──────────────────────────────────────────
    pre_breakout_pressure_direction: str | None = None
    pre_breakout_pressure_score: float = 0.0


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
        # ── Candle Close Confirmation state ──────────────────────────────
        self._last_closed_candle_ts_ms: int = 0
        self._closed_candle_close: float = 0.0
        self._closed_candle_upper: float = 0.0
        self._closed_candle_lower: float = 0.0
        self._confirmed_candle_ts_ms: int = 0
        self._has_candle_close_outside: bool = False
        # ── Pre-Breakout Pressure state ──────────────────────────────────
        self._pre_breakout_pressure: PreBreakoutPressureState | None = None

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
        # ── Candle Close Confirmation params ────────────────────────────
        latest_candle_ts_ms: int = 0,
        latest_candle_close: float = 0.0,
        latest_candle_live_mode: bool = True,
        latest_candle_upper: float = 0.0,
        latest_candle_lower: float = 0.0,
        # ── Pre-Breakout Pressure ───────────────────────────────────────
        pre_breakout_pressure: PreBreakoutPressureState | None = None,
    ) -> TrendAssessment:
        """Evaluate trend state for one tick.

        Returns a :class:`TrendAssessment` with the current state and flags.
        This is a pure-logic method — no side effects besides updating
        internal state.
        """
        self._breakout = breakout
        self._episode = compression_episode
        direction = breakout.direction

        # ── Guard: no recent compression → no trend (NOT failed) ──────
        if compression_episode is None:
            self._state = TrendState.NO_TREND
            self._failure_reason = None
            self._reset_candle_close_state()
            return TrendAssessment(
                trend_state=self._state,
                is_candidate=False,
                is_confirmed=False,
                is_failed=False,
                blocks_mean_reversion=False,
                reason="no_recent_compression",
            )

        # ── Guard: compression expired ─────────────────────────────────
        if current_ts_ms > compression_episode.valid_until_ts_ms:
            if self._has_active_candidate():
                return self._set_failed("compression_expired")
            self._state = TrendState.NO_TREND
            self._failure_reason = None
            return TrendAssessment(
                trend_state=self._state,
                is_candidate=False,
                is_confirmed=False,
                is_failed=False,
                blocks_mean_reversion=False,
                reason="compression_expired_no_candidate",
            )

        # ── Determine breakout duration ──────────────────────────────
        breakout_duration_sec = (current_ts_ms - breakout.ts_ms) / 1000.0

        # ── Prerequisites: range and volume expansion ────────────────
        if not range_expansion_passed:
            self._state = TrendState.POST_COMPRESSION_EXPANDING
            return TrendAssessment(
                trend_state=self._state,
                is_candidate=False,
                is_confirmed=False,
                is_failed=False,
                blocks_mean_reversion=False,
                reason="range_expansion_not_met",
            )

        if not volume_expansion_passed:
            self._state = TrendState.POST_COMPRESSION_EXPANDING
            return TrendAssessment(
                trend_state=self._state,
                is_candidate=False,
                is_confirmed=False,
                is_failed=False,
                blocks_mean_reversion=False,
                reason="volume_expansion_not_met",
            )

        # ── Expansion passed → immediately become candidate ───────────
        if direction == "UP":
            self._state = TrendState.TREND_UP_CANDIDATE
        else:
            self._state = TrendState.TREND_DOWN_CANDIDATE

        # ── Track candle close events ──────────────────────────────────
        self._track_candle_close(
            candle_ts_ms=latest_candle_ts_ms,
            candle_close=latest_candle_close,
            live_mode=latest_candle_live_mode,
            candle_upper=latest_candle_upper,
            candle_lower=latest_candle_lower,
            breakout_direction=direction,
        )

        # ── Store pre-breakout pressure ────────────────────────────────
        if pre_breakout_pressure is not None:
            self._pre_breakout_pressure = pre_breakout_pressure

        # Use current pressure if provided, otherwise fall back to stored.
        effective_pressure = pre_breakout_pressure or self._pre_breakout_pressure

        # ── Failure check 1: confirm_max_seconds exceeded ─────────────
        if breakout_duration_sec > self._config.confirm_max_seconds:
            return self._set_failed("confirm_max_seconds_exceeded")

        # ── Failure check 2: inside reclaim too long (fast-fail) ──────
        if price_reclaimed_inside and inside_reclaim_seconds > self._config.max_inside_reclaim_seconds:
            cvd_diverges = is_cvd_diverging_from_price(
                direction, anchored_cvd, True, self._cvd_config
            )
            cvd_confirms = is_cvd_confirming_trend(
                direction, anchored_cvd, self._cvd_config
            )
            if cvd_diverges or not cvd_confirms:
                return self._set_failed("fast_reclaim_with_cvd_divergence")
            return self._set_failed("inside_reclaim_too_long")

        # ── Failure check 3: outside occupancy too low ────────────────
        if breakout_duration_sec >= self._config.confirm_min_seconds and not outside_occupancy_passed:
            return self._set_failed("outside_occupancy_insufficient")

        # ── Failure check 4: anchored CVD diverges ────────────────────
        price_new_extreme = new_extreme_count >= self._config.min_new_extreme_count
        if price_new_extreme and is_cvd_diverging_from_price(
            direction, anchored_cvd, True, self._cvd_config
        ):
            return self._set_failed("cvd_diverges_from_price")

        # ── Candidate is active but waiting for min_seconds ────────────
        if breakout_duration_sec < self._config.confirm_min_seconds:
            return TrendAssessment(
                trend_state=self._state,
                is_candidate=True,
                is_confirmed=False,
                is_failed=False,
                blocks_mean_reversion=True,
                reason="trend_candidate_waiting_min_seconds",
            )

        # ── Candle close check (if enabled) ────────────────────────────
        if self._config.require_candle_close and not self._has_candle_close_outside:
            return TrendAssessment(
                trend_state=self._state,
                is_candidate=True,
                is_confirmed=False,
                is_failed=False,
                blocks_mean_reversion=True,
                reason="trend_waiting_candle_close",
                has_candle_close_outside=False,
                confirmed_candle_ts_ms=self._confirmed_candle_ts_ms,
                pre_breakout_pressure_direction=(
                    effective_pressure.direction if effective_pressure else None
                ),
                pre_breakout_pressure_score=(
                    effective_pressure.score if effective_pressure else 0.0
                ),
            )

        # ── Pre-breakout pressure conflict check ────────────────────────
        # If pressure is opposite to breakout direction → stricter checks
        pressure_conflict = self._is_pressure_in_conflict(direction, effective_pressure)
        if pressure_conflict:
            # Must have candle close outside + strong post-breakout CVD
            if not self._has_candle_close_outside:
                return TrendAssessment(
                    trend_state=self._state,
                    is_candidate=True,
                    is_confirmed=False,
                    is_failed=False,
                    blocks_mean_reversion=True,
                    reason="pre_breakout_pressure_conflict_waiting_candle_close",
                    pre_breakout_pressure_direction=(
                        effective_pressure.direction if effective_pressure else None
                    ),
                    pre_breakout_pressure_score=(
                        effective_pressure.score if effective_pressure else 0.0
                    ),
                )
            if not anchored_cvd or not is_cvd_confirming_trend(
                direction, anchored_cvd, self._cvd_config
            ):
                return self._set_failed("pre_breakout_pressure_conflict_cvd_not_strong")

        # ── Check all confirmed conditions ─────────────────────────────
        if self._is_confirmed(
            breakout, anchored_cvd, outside_occupancy_passed,
            sustained_volume_passed, new_extreme_count,
        ):
            if direction == "UP":
                self._state = TrendState.TREND_UP_CONFIRMED
            else:
                self._state = TrendState.TREND_DOWN_CONFIRMED
            pressure_dir = (
                effective_pressure.direction if effective_pressure else None
            )
            pressure_score = (
                effective_pressure.score if effective_pressure else 0.0
            )
            return TrendAssessment(
                trend_state=self._state,
                is_candidate=False,  # promoted to confirmed
                is_confirmed=True,
                is_failed=False,
                blocks_mean_reversion=True,
                reason="trend_confirmed",
                has_candle_close_outside=self._has_candle_close_outside,
                confirmed_candle_ts_ms=self._confirmed_candle_ts_ms,
                pre_breakout_pressure_direction=pressure_dir,
                pre_breakout_pressure_score=pressure_score,
            )

        # Still a candidate — still viable, blocks MR
        pressure_dir = (
            effective_pressure.direction if effective_pressure else None
        )
        pressure_score = (
            effective_pressure.score if effective_pressure else 0.0
        )
        return TrendAssessment(
            trend_state=self._state,
            is_candidate=True,
            is_confirmed=False,
            is_failed=False,
            blocks_mean_reversion=True,
            reason="trend_candidate_waiting_confirmation",
            has_candle_close_outside=self._has_candle_close_outside,
            confirmed_candle_ts_ms=self._confirmed_candle_ts_ms,
            pre_breakout_pressure_direction=pressure_dir,
            pre_breakout_pressure_score=pressure_score,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _track_candle_close(
        self,
        *,
        candle_ts_ms: int,
        candle_close: float,
        live_mode: bool,
        candle_upper: float,
        candle_lower: float,
        breakout_direction: str,
    ) -> None:
        """Detect when a new 15m candle has closed and check if the breakout
        direction is confirmed by the closed candle's price position."""
        # Only process when we have a valid candle timestamp
        if candle_ts_ms <= 0:
            return
        # Only track closed candles (not live/unclosed)
        if live_mode:
            return
        # Only process NEW closed candles (different from last tracked)
        if candle_ts_ms == self._last_closed_candle_ts_ms:
            return

        self._last_closed_candle_ts_ms = candle_ts_ms
        self._closed_candle_close = candle_close
        self._closed_candle_upper = candle_upper
        self._closed_candle_lower = candle_lower

        # Check if the breakout direction is confirmed by this closed candle
        if breakout_direction == "UP":
            outside = candle_close > candle_upper
        else:
            outside = candle_close < candle_lower

        if outside:
            self._has_candle_close_outside = True
            self._confirmed_candle_ts_ms = candle_ts_ms
        else:
            # Previous close was outside but this one is back inside → reject
            if self._has_candle_close_outside:
                self._has_candle_close_outside = False
                self._confirmed_candle_ts_ms = 0

    def _reset_candle_close_state(self) -> None:
        """Clear candle close tracking state."""
        self._last_closed_candle_ts_ms = 0
        self._closed_candle_close = 0.0
        self._closed_candle_upper = 0.0
        self._closed_candle_lower = 0.0
        self._confirmed_candle_ts_ms = 0
        self._has_candle_close_outside = False

    def _is_pressure_in_conflict(
        self,
        breakout_direction: str,
        pressure: PreBreakoutPressureState | None,
    ) -> bool:
        """Check if pre-breakout pressure conflicts with breakout direction.

        Returns True when the pressure has a clear direction opposite to the
        breakout AND the pressure score is at or above the minimum.
        """
        if not self._config.pre_breakout_pressure_enabled:
            return False
        if pressure is None or pressure.direction is None:
            return False
        if pressure.score < self._config.pre_breakout_pressure_min_score:
            return False
        return pressure.direction != breakout_direction

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
        # If candle close is required, it must have been observed
        if self._config.require_candle_close and not self._has_candle_close_outside:
            return False
        # Event-anchored cumulative CVD confirms direction
        if not is_cvd_confirming_trend(
            breakout.direction, anchored_cvd, self._cvd_config
        ):
            return False
        return True

    def _has_active_candidate(self) -> bool:
        """Check whether there is already an active trend candidate."""
        return self._state in {
            TrendState.TREND_UP_CANDIDATE,
            TrendState.TREND_DOWN_CANDIDATE,
        }

    def _set_failed(self, reason: str) -> TrendAssessment:
        self._state = TrendState.TREND_FAILED
        self._failure_reason = reason
        return TrendAssessment(
            trend_state=self._state,
            is_candidate=False,
            is_confirmed=False,
            is_failed=True,
            blocks_mean_reversion=False,
            reason=reason,
        )

    def reset(self) -> None:
        """Reset detector to initial state."""
        self._state = TrendState.NO_TREND
        self._breakout = None
        self._episode = None
        self._failure_reason = None
        self._reset_candle_close_state()
        self._pre_breakout_pressure = None
