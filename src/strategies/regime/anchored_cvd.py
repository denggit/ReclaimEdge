from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from src.strategies.regime.types import AnchoredCvdState, BreakoutDirection


@dataclass(frozen=True)
class AnchoredCvdConfig:
    min_buy_ratio: float = 0.58
    min_sell_ratio: float = 0.58
    max_pullback_ratio: float = 0.45

    def __post_init__(self) -> None:
        if not (0 < self.min_buy_ratio <= 1):
            raise ValueError(
                f"min_buy_ratio ({self.min_buy_ratio}) must be in (0, 1]"
            )
        if not (0 < self.min_sell_ratio <= 1):
            raise ValueError(
                f"min_sell_ratio ({self.min_sell_ratio}) must be in (0, 1]"
            )
        if not (0 <= self.max_pullback_ratio <= 1):
            raise ValueError(
                f"max_pullback_ratio ({self.max_pullback_ratio}) must be in [0, 1]"
            )


def build_anchored_cvd_state(
    *,
    anchor_ts_ms: int,
    current_ts_ms: int,
    anchor_cvd: float,
    current_cvd: float,
    episode_buy_volume: float,
    episode_sell_volume: float,
    episode_cvd_max: float,
    episode_cvd_min: float,
) -> AnchoredCvdState:
    """Construct an :class:`AnchoredCvdState` from raw episode aggregates.

    All values are expected to be cumulative from the anchor point
    (time-of-breakout) through the current tick.
    """
    episode_cvd_delta = current_cvd - anchor_cvd
    episode_total_volume = episode_buy_volume + episode_sell_volume

    if episode_total_volume > 0:
        episode_buy_ratio = episode_buy_volume / episode_total_volume
        episode_sell_ratio = episode_sell_volume / episode_total_volume
    else:
        episode_buy_ratio = 0.0
        episode_sell_ratio = 0.0

    # CVD pullback from its episode extreme
    delta_abs = max(abs(episode_cvd_delta), 1e-12)
    if episode_cvd_delta >= 0:
        # UP trend: pullback is from episode_cvd_max toward current
        episode_cvd_drawdown_ratio = (
            max(0.0, episode_cvd_max - current_cvd) / delta_abs
        )
    else:
        # DOWN trend: pullback is from episode_cvd_min toward current
        episode_cvd_drawdown_ratio = (
            max(0.0, current_cvd - episode_cvd_min) / delta_abs
        )

    # CVD slope (delta per second)
    elapsed_sec = (current_ts_ms - anchor_ts_ms) / 1000.0
    if elapsed_sec > 0:
        cvd_slope = episode_cvd_delta / elapsed_sec
    else:
        cvd_slope = 0.0

    return AnchoredCvdState(
        anchor_ts_ms=anchor_ts_ms,
        current_ts_ms=current_ts_ms,
        anchor_cvd=anchor_cvd,
        current_cvd=current_cvd,
        episode_cvd_delta=episode_cvd_delta,
        episode_cvd_max=episode_cvd_max,
        episode_cvd_min=episode_cvd_min,
        episode_cvd_drawdown_ratio=episode_cvd_drawdown_ratio,
        episode_buy_volume=episode_buy_volume,
        episode_sell_volume=episode_sell_volume,
        episode_total_volume=episode_total_volume,
        episode_buy_ratio=episode_buy_ratio,
        episode_sell_ratio=episode_sell_ratio,
        cvd_slope=cvd_slope,
    )


def is_cvd_confirming_trend(
    direction: BreakoutDirection,
    state: AnchoredCvdState,
    config: AnchoredCvdConfig,
) -> bool:
    """Return True when event-anchored cumulative CVD confirms the trend.

    UP breakout:
        - episode_cvd_delta > 0
        - episode_buy_ratio >= min_buy_ratio
        - episode_cvd_drawdown_ratio <= max_pullback_ratio
        - cvd_slope >= 0

    DOWN breakout:
        - episode_cvd_delta < 0
        - episode_sell_ratio >= min_sell_ratio
        - episode_cvd_drawdown_ratio <= max_pullback_ratio
        - cvd_slope <= 0
    """
    if direction == "UP":
        if state.episode_cvd_delta <= 0:
            return False
        if state.episode_buy_ratio < config.min_buy_ratio:
            return False
        if state.episode_cvd_drawdown_ratio > config.max_pullback_ratio:
            return False
        if state.cvd_slope < 0:
            return False
        return True
    else:  # DOWN
        if state.episode_cvd_delta >= 0:
            return False
        if state.episode_sell_ratio < config.min_sell_ratio:
            return False
        if state.episode_cvd_drawdown_ratio > config.max_pullback_ratio:
            return False
        if state.cvd_slope > 0:
            return False
        return True


def is_cvd_diverging_from_price(
    direction: BreakoutDirection,
    state: AnchoredCvdState,
    price_new_extreme: bool,
    config: AnchoredCvdConfig,
) -> bool:
    """Return True when price makes a new extreme but anchored CVD diverges.

    This is the signal that mean-reversion may be more appropriate than trend.

    UP breakout + price_new_high:
        - episode_cvd_delta <= 0 (CVD not increasing), OR
        - episode_cvd_drawdown_ratio > max_pullback_ratio (CVD pulling back too much)

    DOWN breakout + price_new_low:
        - episode_cvd_delta >= 0 (CVD not decreasing), OR
        - episode_cvd_drawdown_ratio > max_pullback_ratio (CVD pulling back too much)
    """
    if not price_new_extreme:
        return False

    if direction == "UP":
        # Price new high but CVD is flat/declining or pulling back deeply
        if state.episode_cvd_delta <= 0:
            return True
        if state.episode_cvd_drawdown_ratio > config.max_pullback_ratio:
            return True
        return False
    else:  # DOWN
        # Price new low but CVD is flat/increasing or pulling back deeply
        if state.episode_cvd_delta >= 0:
            return True
        if state.episode_cvd_drawdown_ratio > config.max_pullback_ratio:
            return True
        return False


def get_cvd_direction_label(state: AnchoredCvdState) -> str:
    """Return a human-readable direction label for the CVD delta."""
    if state.episode_cvd_delta > 0:
        return "increasing"
    elif state.episode_cvd_delta < 0:
        return "decreasing"
    return "flat"
