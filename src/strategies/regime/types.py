from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


RegimeSide = Literal["LONG", "SHORT"]
BreakoutDirection = Literal["UP", "DOWN"]


class RegimeDecisionType(str, Enum):
    NO_TRADE = "NO_TRADE"
    MEAN_REVERSION_LONG = "MEAN_REVERSION_LONG"
    MEAN_REVERSION_SHORT = "MEAN_REVERSION_SHORT"
    TREND_LONG = "TREND_LONG"
    TREND_SHORT = "TREND_SHORT"
    TREND_UPGRADE_LONG = "TREND_UPGRADE_LONG"
    TREND_UPGRADE_SHORT = "TREND_UPGRADE_SHORT"
    CONFLICT_NO_TRADE = "CONFLICT_NO_TRADE"


class TrendState(str, Enum):
    NO_TREND = "NO_TREND"
    COMPRESSION_ACTIVE = "COMPRESSION_ACTIVE"
    POST_COMPRESSION_EXPANDING = "POST_COMPRESSION_EXPANDING"
    TREND_UP_CANDIDATE = "TREND_UP_CANDIDATE"
    TREND_DOWN_CANDIDATE = "TREND_DOWN_CANDIDATE"
    TREND_UP_CONFIRMED = "TREND_UP_CONFIRMED"
    TREND_DOWN_CONFIRMED = "TREND_DOWN_CONFIRMED"
    TREND_FAILED = "TREND_FAILED"


@dataclass(frozen=True)
class BandSnapshot:
    upper: float
    middle: float
    lower: float
    candle_ts_ms: int
    source: str = "closed_or_frozen"


@dataclass(frozen=True)
class CompressionEpisode:
    start_ts_ms: int
    end_ts_ms: int
    valid_until_ts_ms: int
    compressed_candle_count: int
    min_outer_distance_pct: float
    avg_outer_distance_pct: float
    upper_at_end: float
    middle_at_end: float
    lower_at_end: float
    highest_high: float
    lowest_low: float


@dataclass(frozen=True)
class BreakoutSnapshot:
    direction: BreakoutDirection
    ts_ms: int
    price: float
    band: BandSnapshot
    anchor_cvd: float
    anchor_volume: float


@dataclass(frozen=True)
class AnchoredCvdState:
    anchor_ts_ms: int
    current_ts_ms: int
    anchor_cvd: float
    current_cvd: float
    episode_cvd_delta: float
    episode_cvd_max: float
    episode_cvd_min: float
    episode_cvd_drawdown_ratio: float
    episode_buy_volume: float
    episode_sell_volume: float
    episode_total_volume: float
    episode_buy_ratio: float
    episode_sell_ratio: float
    cvd_slope: float


@dataclass(frozen=True)
class RegimeDecision:
    decision_type: RegimeDecisionType
    side: RegimeSide | None
    reason: str
    confidence: float = 0.0
    trend_state: TrendState = TrendState.NO_TREND
