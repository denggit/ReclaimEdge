"""Trend Breakout Assessor — coordinates compression detection, trend detection,
and anchored CVD analysis to produce a TrendBreakoutDecision per tick.

Pure logic layer.  No exchange calls, no I/O, no execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

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
    AnchoredCvdState,
    BandSnapshot,
    BreakoutSnapshot,
    CompressionEpisode,
    TrendState,
)

# ── Maximum number of band snapshots to retain for compression detection ──
_MAX_BAND_HISTORY = 96


@dataclass(frozen=True)
class TrendBreakoutDecision:
    """Result of a full trend breakout assessment tick."""

    is_trend_breakout: bool
    direction: str | None  # "LONG" | "SHORT" | None
    reason: str
    trend_state: TrendState = TrendState.NO_TREND
    trend_assessment: TrendAssessment | None = None
    breakout_price: float | None = None
    breakout_ts_ms: int | None = None
    confidence: float = 0.0
    blocks_mean_reversion: bool = False


class TrendBreakoutAssessor:
    """Coordinates compression detection, trend detection, and CVD analysis
    to produce a single :class:`TrendBreakoutDecision` per tick.

    Maintains its own ``CompressionDetector``, ``TrendDetector`` and
    band-history ring buffer.  Configuration parameters are passed at
    construction and come from the strategy-level env vars.
    """

    def __init__(
        self,
        *,
        compression_valid_after_seconds: int = 7200,
        confirm_min_seconds: int = 60,
        confirm_max_seconds: int = 180,
        range_expansion_ratio_min: float = 3.0,
        volume_expansion_ratio_min: float = 3.0,
        outside_occupancy_min_ratio: float = 0.70,
        min_new_extreme_count: int = 2,
        max_inside_reclaim_seconds: int = 3,
        cvd_min_buy_ratio: float = 0.58,
        cvd_min_sell_ratio: float = 0.58,
        cvd_max_pullback_ratio: float = 0.45,
        # ── Candle Close Confirmation ──────────────────────────────────
        trend_confirm_require_candle_close: bool = True,
        # ── Pre-Breakout Pressure ──────────────────────────────────────
        trend_pre_breakout_pressure_enabled: bool = True,
        trend_pre_breakout_min_cvd_ratio: float = 0.55,
        trend_pre_breakout_max_pullback_ratio: float = 0.45,
        trend_pre_breakout_min_observe_seconds: int = 300,
        trend_pre_breakout_pressure_min_score: float = 0.60,
    ) -> None:
        self._compression_config = CompressionDetectorConfig(
            valid_after_seconds=compression_valid_after_seconds,
        )
        self._compression_detector = CompressionDetector(self._compression_config)
        self._trend_config = TrendDetectorConfig(
            confirm_min_seconds=confirm_min_seconds,
            confirm_max_seconds=confirm_max_seconds,
            range_expansion_ratio_min=range_expansion_ratio_min,
            volume_expansion_ratio_min=volume_expansion_ratio_min,
            outside_occupancy_min_ratio=outside_occupancy_min_ratio,
            min_new_extreme_count=min_new_extreme_count,
            max_inside_reclaim_seconds=max_inside_reclaim_seconds,
            require_candle_close=trend_confirm_require_candle_close,
            pre_breakout_pressure_enabled=trend_pre_breakout_pressure_enabled,
            pre_breakout_pressure_min_score=trend_pre_breakout_pressure_min_score,
        )
        self._cvd_config = AnchoredCvdConfig(
            min_buy_ratio=cvd_min_buy_ratio,
            min_sell_ratio=cvd_min_sell_ratio,
            max_pullback_ratio=cvd_max_pullback_ratio,
        )
        self._trend_detector = TrendDetector(
            self._trend_config,
            self._compression_detector,
            self._cvd_config,
        )

        # ── Pre-Breakout Pressure Tracker ─────────────────────────────────
        self._pressure_config = PreBreakoutPressureConfig(
            enabled=trend_pre_breakout_pressure_enabled,
            min_cvd_ratio=trend_pre_breakout_min_cvd_ratio,
            max_pullback_ratio=trend_pre_breakout_max_pullback_ratio,
            min_observe_seconds=trend_pre_breakout_min_observe_seconds,
            pressure_min_score=trend_pre_breakout_pressure_min_score,
        )
        self._pressure_tracker = PreBreakoutPressureTracker(self._pressure_config)

        # ── Band history ring buffer ───────────────────────────────────────
        self._band_history: list[BandSnapshot] = []

        # ── Per-tick mutable state ──────────────────────────────────────────
        self._latest_breakout: BreakoutSnapshot | None = None
        self._latest_cvd_state: AnchoredCvdState | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def trend_detector(self) -> TrendDetector:
        return self._trend_detector

    @property
    def compression_detector(self) -> CompressionDetector:
        return self._compression_detector

    # ------------------------------------------------------------------
    # Band history
    # ------------------------------------------------------------------

    def feed_band(self, band: BandSnapshot) -> None:
        """Push one band snapshot into the ring buffer for compression detection."""
        self._band_history.append(band)
        # Keep at most _MAX_BAND_HISTORY entries
        if len(self._band_history) > _MAX_BAND_HISTORY:
            self._band_history = self._band_history[-_MAX_BAND_HISTORY:]

    def detect_compression(self, ts_ms: int) -> CompressionEpisode | None:
        """Run compression detection on the current band history ring buffer.

        Used by live startup warmup to check whether the pre-fed historical
        bands already form a valid compression episode.  Returns the latest
        episode if one is detected and still valid, or ``None``.
        """
        return self._compression_detector.detect(self._band_history, ts_ms)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(
        self,
        *,
        price: float,
        ts_ms: int,
        boll_upper: float,
        boll_middle: float,
        boll_lower: float,
        fast_cvd: float,
        buy_ratio: float,
        sell_ratio: float,
        episode_buy_volume: float = 0.0,
        episode_sell_volume: float = 0.0,
        episode_cvd_max: float = 0.0,
        episode_cvd_min: float = 0.0,
        # ── Pre-computed expansion / occupancy booleans ──────────────────
        range_expansion_passed: bool = False,
        volume_expansion_passed: bool = False,
        sustained_volume_passed: bool = False,
        outside_occupancy_passed: bool = False,
        new_extreme_count: int = 0,
        inside_reclaim_seconds: float = 0.0,
        price_reclaimed_inside: bool = False,
        # ── Candle Close Confirmation params ────────────────────────────
        latest_candle_ts_ms: int = 0,
        latest_candle_close: float = 0.0,
        latest_candle_live_mode: bool = True,
    ) -> TrendBreakoutDecision:
        """Run a full trend breakout assessment for one tick.

        This is a pure-logic coordinator — it does **not** call any
        exchange or execution APIs.  All data must be passed in.
        """
        # 1. Feed bands to compression detector
        if self._band_history:
            self._compression_detector.detect(self._band_history, ts_ms)
        compression_episode = self._compression_detector.get_valid_episode(ts_ms)

        # 2. Detect breakout direction — persist anchor across ticks
        current_direction = self._classify_direction(price, boll_upper, boll_lower)
        if current_direction is None:
            # Price back inside band — clear breakout.
            # Update pressure tracker while price is inside band.
            if compression_episode is not None and self._pressure_config.enabled:
                if not self._pressure_tracker.active:
                    self._pressure_tracker.start(
                        ts_ms=ts_ms,
                        price=price,
                        fast_cvd=fast_cvd,
                        buy_volume=episode_buy_volume,
                        sell_volume=episode_sell_volume,
                        boll_upper=boll_upper,
                        boll_middle=boll_middle,
                        boll_lower=boll_lower,
                    )
                else:
                    self._pressure_tracker.update(
                        ts_ms=ts_ms,
                        price=price,
                        fast_cvd=fast_cvd,
                        buy_volume=episode_buy_volume,
                        sell_volume=episode_sell_volume,
                        boll_upper=boll_upper,
                        boll_middle=boll_middle,
                        boll_lower=boll_lower,
                    )
            breakout = None
        elif self._latest_breakout is None or self._latest_breakout.direction != current_direction:
            # New breakout or direction change — anchor at current CVD.
            # Stop pressure tracking since price is now outside the band.
            self._pressure_tracker.reset()
            breakout = BreakoutSnapshot(
                direction=current_direction,
                ts_ms=ts_ms,
                price=price,
                band=BandSnapshot(
                    upper=boll_upper, middle=boll_middle, lower=boll_lower,
                    candle_ts_ms=ts_ms, source="live",
                ),
                anchor_cvd=fast_cvd,
                anchor_volume=0.0,
            )
            self._latest_breakout = breakout
        else:
            # Same direction, ongoing — reuse existing anchor
            breakout = self._latest_breakout

        # 3. Capture pre-breakout pressure snapshot (before breakout cleared it)
        pre_breakout_pressure: PreBreakoutPressureState | None = None
        if self._pressure_tracker.active and current_direction is not None:
            # Price just moved outside — snapshot the pressure before resetting
            pre_breakout_pressure = self._pressure_tracker.snapshot()
        elif self._pressure_config.enabled and current_direction is None and compression_episode is not None:
            # Still inside band — snapshot current pressure state
            pre_breakout_pressure = self._pressure_tracker.snapshot()

        # 4. If no breakout direction, feed trend detector for stale-candidate
        #    tracking but return early.
        if breakout is None:
            trend_state = self._trend_detector.state
            if trend_state in (
                TrendState.TREND_UP_CANDIDATE,
                TrendState.TREND_DOWN_CANDIDATE,
                TrendState.TREND_UP_CONFIRMED,
                TrendState.TREND_DOWN_CONFIRMED,
            ):
                # Determine the actual trend direction from current state
                if trend_state in (TrendState.TREND_UP_CANDIDATE, TrendState.TREND_UP_CONFIRMED):
                    candidate_direction = "UP"
                else:
                    candidate_direction = "DOWN"

                if self._latest_breakout is not None:
                    _existing = self._latest_breakout
                else:
                    _existing = self._make_breakout(
                        direction=candidate_direction,
                        ts_ms=ts_ms,
                        price=price,
                        boll_upper=boll_upper,
                        boll_middle=boll_middle,
                        boll_lower=boll_lower,
                        anchor_cvd=fast_cvd,
                    )
                self._trend_detector.assess(
                    breakout=_existing,
                    compression_episode=compression_episode,
                    anchored_cvd=self._latest_cvd_state or self._dummy_cvd(ts_ms, fast_cvd),
                    current_ts_ms=ts_ms,
                    range_expansion_passed=range_expansion_passed,
                    volume_expansion_passed=volume_expansion_passed,
                    sustained_volume_passed=sustained_volume_passed,
                    outside_occupancy_passed=outside_occupancy_passed,
                    new_extreme_count=new_extreme_count,
                    inside_reclaim_seconds=inside_reclaim_seconds,
                    price_reclaimed_inside=True,
                    latest_candle_ts_ms=latest_candle_ts_ms,
                    latest_candle_close=latest_candle_close,
                    latest_candle_live_mode=latest_candle_live_mode,
                    latest_candle_upper=boll_upper,
                    latest_candle_lower=boll_lower,
                )
            return TrendBreakoutDecision(
                is_trend_breakout=False,
                direction=None,
                reason="no_breakout",
                trend_state=self._trend_detector.state,
            )

        # 5. Build anchored CVD state
        anchored_cvd = build_anchored_cvd_state(
            anchor_ts_ms=breakout.ts_ms,
            current_ts_ms=ts_ms,
            anchor_cvd=breakout.anchor_cvd,
            current_cvd=fast_cvd,
            episode_buy_volume=episode_buy_volume,
            episode_sell_volume=episode_sell_volume,
            episode_cvd_max=episode_cvd_max,
            episode_cvd_min=episode_cvd_min,
        )
        self._latest_cvd_state = anchored_cvd

        # 6. Run trend detector with candle close + pre-breakout pressure
        assessment = self._trend_detector.assess(
            breakout=breakout,
            compression_episode=compression_episode,
            anchored_cvd=anchored_cvd,
            current_ts_ms=ts_ms,
            range_expansion_passed=range_expansion_passed,
            volume_expansion_passed=volume_expansion_passed,
            sustained_volume_passed=sustained_volume_passed,
            outside_occupancy_passed=outside_occupancy_passed,
            new_extreme_count=new_extreme_count,
            inside_reclaim_seconds=inside_reclaim_seconds,
            price_reclaimed_inside=price_reclaimed_inside,
            latest_candle_ts_ms=latest_candle_ts_ms,
            latest_candle_close=latest_candle_close,
            latest_candle_live_mode=latest_candle_live_mode,
            latest_candle_upper=boll_upper,
            latest_candle_lower=boll_lower,
            pre_breakout_pressure=pre_breakout_pressure,
        )

        # 7. Produce decision
        if assessment.is_confirmed:
            direction = "LONG" if breakout.direction == "UP" else "SHORT"
            return TrendBreakoutDecision(
                is_trend_breakout=True,
                direction=direction,
                reason=f"trend_confirmed: {assessment.reason}",
                trend_state=assessment.trend_state,
                trend_assessment=assessment,
                breakout_price=breakout.price,
                breakout_ts_ms=breakout.ts_ms,
                confidence=0.9,
                blocks_mean_reversion=assessment.blocks_mean_reversion,
            )

        if assessment.is_candidate:
            direction = "LONG" if breakout.direction == "UP" else "SHORT"
            return TrendBreakoutDecision(
                is_trend_breakout=False,
                direction=direction,
                reason=f"trend_candidate: {assessment.reason}",
                trend_state=assessment.trend_state,
                trend_assessment=assessment,
                breakout_price=breakout.price,
                breakout_ts_ms=breakout.ts_ms,
                blocks_mean_reversion=assessment.blocks_mean_reversion,
            )

        if assessment.is_failed:
            return TrendBreakoutDecision(
                is_trend_breakout=False,
                direction=None,
                reason=f"trend_failed: {assessment.reason}",
                trend_state=assessment.trend_state,
                trend_assessment=assessment,
                blocks_mean_reversion=assessment.blocks_mean_reversion,
            )

        # No trend
        return TrendBreakoutDecision(
            is_trend_breakout=False,
            direction=None,
            reason=assessment.reason,
            trend_state=assessment.trend_state,
            trend_assessment=assessment,
            blocks_mean_reversion=assessment.blocks_mean_reversion,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_direction(
        price: float,
        boll_upper: float,
        boll_lower: float,
    ) -> str | None:
        """Return "UP" if price > upper band, "DOWN" if < lower, else None."""
        if price > boll_upper:
            return "UP"
        if price < boll_lower:
            return "DOWN"
        return None

    @staticmethod
    def _make_breakout(
        *,
        direction: str,
        ts_ms: int,
        price: float,
        boll_upper: float,
        boll_middle: float,
        boll_lower: float,
        anchor_cvd: float,
    ) -> BreakoutSnapshot:
        return BreakoutSnapshot(
            direction=direction,
            ts_ms=ts_ms,
            price=price,
            band=BandSnapshot(
                upper=boll_upper,
                middle=boll_middle,
                lower=boll_lower,
                candle_ts_ms=ts_ms,
                source="live",
            ),
            anchor_cvd=anchor_cvd,
            anchor_volume=0.0,
        )

    @staticmethod
    def _dummy_cvd(ts_ms: int, fast_cvd: float) -> AnchoredCvdState:
        return AnchoredCvdState(
            anchor_ts_ms=ts_ms,
            current_ts_ms=ts_ms,
            anchor_cvd=fast_cvd,
            current_cvd=fast_cvd,
            episode_cvd_delta=0.0,
            episode_cvd_max=fast_cvd,
            episode_cvd_min=fast_cvd,
            episode_cvd_drawdown_ratio=0.0,
            episode_buy_volume=0.0,
            episode_sell_volume=0.0,
            episode_total_volume=0.0,
            episode_buy_ratio=0.0,
            episode_sell_ratio=0.0,
            cvd_slope=0.0,
        )

    def reset(self) -> None:
        """Reset all internal state (detectors + band history)."""
        self._compression_detector = CompressionDetector(self._compression_config)
        self._trend_detector = TrendDetector(
            self._trend_config,
            self._compression_detector,
            self._cvd_config,
        )
        self._pressure_tracker = PreBreakoutPressureTracker(self._pressure_config)
        self._band_history = []
        self._latest_breakout = None
        self._latest_cvd_state = None
