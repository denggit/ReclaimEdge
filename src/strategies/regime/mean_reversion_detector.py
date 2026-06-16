from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from src.strategies.regime.anchored_cvd import (
    AnchoredCvdConfig,
    is_cvd_diverging_from_price,
)
from src.strategies.regime.types import (
    AnchoredCvdState,
    BreakoutDirection,
    BreakoutSnapshot,
    RegimeSide,
    TrendState,
)


@dataclass(frozen=True)
class MeanReversionDetectorConfig:
    """Configuration for mean-reversion gate.

    This module only acts as a router gate — it does NOT replace
    the existing BollCvdReclaimStrategy entry logic.
    """
    # Whether to require CVD divergence for mean-reversion
    require_cvd_divergence: bool = True

    def __post_init__(self) -> None:
        pass


@dataclass(frozen=True)
class MeanReversionGate:
    """Output of the mean-reversion detector for a single evaluation."""
    side: RegimeSide | None
    allowed: bool
    reason: str


class MeanReversionDetector:
    """Gate that decides whether mean-reversion is allowed given the current
    trend state.

    Rules (UP breakout / upper band):
        - TREND_UP_CANDIDATE (not failed) → block MEAN_REVERSION_SHORT
        - TREND_UP_CONFIRMED             → block MEAN_REVERSION_SHORT
        - TREND_FAILED + inside reclaim + CVD divergence → allow MEAN_REVERSION_SHORT

    Rules (DOWN breakout / lower band):
        - TREND_DOWN_CANDIDATE (not failed) → block MEAN_REVERSION_LONG
        - TREND_DOWN_CONFIRMED               → block MEAN_REVERSION_LONG
        - TREND_FAILED + inside reclaim + CVD divergence → allow MEAN_REVERSION_LONG

    When there is NO trend candidate at all:
        - Allow mean-reversion gate to pass (ordinary reclaim entry).
    """

    def __init__(
        self,
        config: MeanReversionDetectorConfig,
        cvd_config: AnchoredCvdConfig,
    ) -> None:
        self._config = config
        self._cvd_config = cvd_config

    def evaluate(
        self,
        *,
        breakout: BreakoutSnapshot,
        anchored_cvd: AnchoredCvdState,
        trend_state: TrendState,
        trend_failed: bool,
        trend_failure_reason: Optional[str],
        price_reclaimed_inside: bool,
        price_new_extreme: bool,
    ) -> MeanReversionGate:
        """Decide whether mean-reversion is allowed at this tick.

        Returns a :class:`MeanReversionGate` with the allowed side and reason.
        """
        direction = breakout.direction

        # ── Case 1: Trend is confirmed — block opposite mean-reversion ──
        if trend_state == TrendState.TREND_UP_CONFIRMED:
            return MeanReversionGate(
                side="SHORT",
                allowed=False,
                reason="trend_up_confirmed_blocks_mean_reversion_short",
            )
        if trend_state == TrendState.TREND_DOWN_CONFIRMED:
            return MeanReversionGate(
                side="LONG",
                allowed=False,
                reason="trend_down_confirmed_blocks_mean_reversion_long",
            )

        # ── Case 2: Trend candidate active and NOT failed — block ──
        if trend_state in (TrendState.TREND_UP_CANDIDATE,) and not trend_failed:
            return MeanReversionGate(
                side="SHORT",
                allowed=False,
                reason="trend_up_candidate_active_blocks_mean_reversion_short",
            )
        if trend_state in (TrendState.TREND_DOWN_CANDIDATE,) and not trend_failed:
            return MeanReversionGate(
                side="LONG",
                allowed=False,
                reason="trend_down_candidate_active_blocks_mean_reversion_long",
            )

        # ── Case 3: Trend failed — allow mean-reversion if conditions met ──
        if trend_failed:
            if direction == "UP":
                side: RegimeSide = "SHORT"
                # Require price reclaimed inside AND CVD divergence
                if not price_reclaimed_inside:
                    return MeanReversionGate(
                        side=side,
                        allowed=False,
                        reason="trend_failed_but_price_not_reclaimed",
                    )
                if self._config.require_cvd_divergence:
                    cvd_diverges = is_cvd_diverging_from_price(
                        "UP", anchored_cvd, price_new_extreme, self._cvd_config
                    )
                    if not cvd_diverges:
                        return MeanReversionGate(
                            side=side,
                            allowed=False,
                            reason="trend_failed_but_cvd_not_diverging",
                        )
                return MeanReversionGate(
                    side=side,
                    allowed=True,
                    reason="trend_failed_mean_reversion_short_allowed",
                )
            else:  # DOWN
                side = "LONG"
                if not price_reclaimed_inside:
                    return MeanReversionGate(
                        side=side,
                        allowed=False,
                        reason="trend_failed_but_price_not_reclaimed",
                    )
                if self._config.require_cvd_divergence:
                    cvd_diverges = is_cvd_diverging_from_price(
                        "DOWN", anchored_cvd, price_new_extreme, self._cvd_config
                    )
                    if not cvd_diverges:
                        return MeanReversionGate(
                            side=side,
                            allowed=False,
                            reason="trend_failed_but_cvd_not_diverging",
                        )
                return MeanReversionGate(
                    side=side,
                    allowed=True,
                    reason="trend_failed_mean_reversion_long_allowed",
                )

        # ── Case 4: No trend candidate — ordinary mean-reversion gate ──
        if trend_state in (TrendState.NO_TREND, TrendState.COMPRESSION_ACTIVE,
                           TrendState.POST_COMPRESSION_EXPANDING):
            if direction == "UP":
                return MeanReversionGate(
                    side="SHORT",
                    allowed=True,
                    reason="no_trend_candidate_mean_reversion_short_allowed",
                )
            else:
                return MeanReversionGate(
                    side="LONG",
                    allowed=True,
                    reason="no_trend_candidate_mean_reversion_long_allowed",
                )

        # ── Fallback: no decision ──
        return MeanReversionGate(
            side=None,
            allowed=False,
            reason="no_mean_reversion_decision",
        )
