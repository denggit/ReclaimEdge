"""Pre-Breakout Directional Pressure Tracker.

Observes directional CVD pressure inside the BOLL band after compression is
detected but before breakout.  The pressure direction and score influence
trend confirmation quality without being a hard gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class PreBreakoutPressureState:
    """Snapshot of directional pressure observed inside the band."""

    direction: str | None  # "UP" | "DOWN" | None
    score: float  # 0.0 – 1.0, higher = stronger pressure
    duration_seconds: float
    anchored_cvd: float
    buy_ratio: float
    sell_ratio: float
    reason: str = ""


@dataclass
class PreBreakoutPressureConfig:
    """Configuration for the pre-breakout pressure tracker."""

    enabled: bool = True
    min_cvd_ratio: float = 0.55
    max_pullback_ratio: float = 0.45
    min_observe_seconds: int = 300
    pressure_min_score: float = 0.60


class PreBreakoutPressureTracker:
    """Stateful tracker that observes directional CVD pressure inside the BOLL
    band after compression is valid but before a breakout occurs.

    Tracks anchored CVD, buy/sell volume, and price drift toward the band
    edges to compute directional pressure scores for UP and DOWN.

    Usage::

        tracker = PreBreakoutPressureTracker(config)
        tracker.start(ts_ms, price, fast_cvd, buy_volume, sell_volume, upper, middle, lower)
        # ... each tick inside band:
        tracker.update(ts_ms, price, fast_cvd, buy_volume, sell_volume, upper, middle, lower)
        state = tracker.snapshot()
    """

    def __init__(self, config: PreBreakoutPressureConfig) -> None:
        self._config = config
        self._active: bool = False
        # ── Anchor values (set at start) ────────────────────────────────
        self._anchor_ts_ms: int = 0
        self._anchor_price: float = 0.0
        self._anchor_cvd: float = 0.0
        # ── Accumulated values ───────────────────────────────────────────
        self._latest_cvd: float = 0.0
        self._total_buy_volume: float = 0.0
        self._total_sell_volume: float = 0.0
        self._min_cvd: float = 0.0
        self._max_cvd: float = 0.0
        self._latest_price: float = 0.0
        self._latest_upper: float = 0.0
        self._latest_middle: float = 0.0
        self._latest_lower: float = 0.0
        self._latest_ts_ms: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active(self) -> bool:
        """Whether the tracker is currently observing pressure."""
        return self._active

    @property
    def config(self) -> PreBreakoutPressureConfig:
        return self._config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(
        self,
        ts_ms: int,
        price: float,
        fast_cvd: float,
        buy_volume: float,
        sell_volume: float,
        boll_upper: float,
        boll_middle: float,
        boll_lower: float,
    ) -> None:
        """Begin observing directional pressure from this anchor point."""
        self._active = True
        self._anchor_ts_ms = ts_ms
        self._anchor_price = price
        self._anchor_cvd = fast_cvd
        self._latest_cvd = fast_cvd
        self._total_buy_volume = buy_volume
        self._total_sell_volume = sell_volume
        self._min_cvd = fast_cvd
        self._max_cvd = fast_cvd
        self._latest_price = price
        self._latest_upper = boll_upper
        self._latest_middle = boll_middle
        self._latest_lower = boll_lower
        self._latest_ts_ms = ts_ms

    def update(
        self,
        ts_ms: int,
        price: float,
        fast_cvd: float,
        buy_volume: float,
        sell_volume: float,
        boll_upper: float,
        boll_middle: float,
        boll_lower: float,
    ) -> None:
        """Accumulate one tick of observation."""
        if not self._active:
            return
        self._latest_cvd = fast_cvd
        self._total_buy_volume += buy_volume
        self._total_sell_volume += sell_volume
        if fast_cvd > self._max_cvd:
            self._max_cvd = fast_cvd
        if fast_cvd < self._min_cvd:
            self._min_cvd = fast_cvd
        self._latest_price = price
        self._latest_upper = boll_upper
        self._latest_middle = boll_middle
        self._latest_lower = boll_lower
        self._latest_ts_ms = ts_ms

    def snapshot(self) -> PreBreakoutPressureState:
        """Compute the current directional pressure assessment."""
        if not self._active:
            return PreBreakoutPressureState(
                direction=None,
                score=0.0,
                duration_seconds=0.0,
                anchored_cvd=0.0,
                buy_ratio=0.0,
                sell_ratio=0.0,
                reason="not_active",
            )

        duration_sec = (self._latest_ts_ms - self._anchor_ts_ms) / 1000.0
        anchored_cvd = self._latest_cvd - self._anchor_cvd

        total_volume = self._total_buy_volume + self._total_sell_volume
        if total_volume > 0:
            buy_ratio = self._total_buy_volume / total_volume
            sell_ratio = self._total_sell_volume / total_volume
        else:
            buy_ratio = 0.0
            sell_ratio = 0.0

        # ── Compute price drift toward each band edge ───────────────────
        band_range = self._latest_upper - self._latest_lower
        if band_range > 0:
            price_drift_to_upper = (
                self._latest_price - self._anchor_price
            ) / band_range
            price_drift_to_lower = (
                self._anchor_price - self._latest_price
            ) / band_range
        else:
            price_drift_to_upper = 0.0
            price_drift_to_lower = 0.0

        # ── CVD pullback ratio ──────────────────────────────────────────
        cvd_range = self._max_cvd - self._min_cvd
        if cvd_range > 0:
            if anchored_cvd >= 0:
                cvd_pullback_ratio = (self._max_cvd - self._latest_cvd) / cvd_range if cvd_range > 0 else 0.0
            else:
                cvd_pullback_ratio = (self._latest_cvd - self._min_cvd) / cvd_range if cvd_range > 0 else 0.0
        else:
            cvd_pullback_ratio = 0.0

        # ── Compute UP pressure score ───────────────────────────────────
        up_score = self._compute_direction_score(
            anchored_cvd=anchored_cvd,
            buy_ratio=buy_ratio,
            price_drift=price_drift_to_upper,
            cvd_pullback_ratio=cvd_pullback_ratio,
            duration_sec=duration_sec,
            direction="UP",
        )

        # ── Compute DOWN pressure score ─────────────────────────────────
        down_score = self._compute_direction_score(
            anchored_cvd=-anchored_cvd,  # mirror for DOWN
            buy_ratio=sell_ratio,  # sell_ratio is the buy side for DOWN
            price_drift=price_drift_to_lower,
            cvd_pullback_ratio=cvd_pullback_ratio,
            duration_sec=duration_sec,
            direction="DOWN",
        )

        # ── Determine dominant direction ────────────────────────────────
        min_score = self._config.pressure_min_score
        if up_score >= min_score and up_score > down_score:
            direction = "UP"
            score = up_score
            reason = "up_pressure_dominant"
        elif down_score >= min_score and down_score > up_score:
            direction = "DOWN"
            score = down_score
            reason = "down_pressure_dominant"
        else:
            direction = None
            score = max(up_score, down_score)
            if score < min_score:
                reason = "no_clear_pressure"
            else:
                reason = "pressure_balanced"

        return PreBreakoutPressureState(
            direction=direction,
            score=score,
            duration_seconds=duration_sec,
            anchored_cvd=anchored_cvd,
            buy_ratio=buy_ratio,
            sell_ratio=sell_ratio,
            reason=reason,
        )

    def reset(self) -> None:
        """Stop observing and clear all accumulated data."""
        self._active = False
        self._anchor_ts_ms = 0
        self._anchor_price = 0.0
        self._anchor_cvd = 0.0
        self._latest_cvd = 0.0
        self._total_buy_volume = 0.0
        self._total_sell_volume = 0.0
        self._min_cvd = 0.0
        self._max_cvd = 0.0
        self._latest_price = 0.0
        self._latest_upper = 0.0
        self._latest_middle = 0.0
        self._latest_lower = 0.0
        self._latest_ts_ms = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_direction_score(
        self,
        *,
        anchored_cvd: float,
        buy_ratio: float,
        price_drift: float,
        cvd_pullback_ratio: float,
        duration_sec: float,
        direction: str,
    ) -> float:
        """Compute a 0.0–1.0 score for a given direction.

        Components (equally weighted):
        - anchored_cvd > 0 (direction confirmed by CVD)
        - buy_ratio >= min_cvd_ratio (buying pressure for UP, selling for DOWN)
        - price drift toward target band edge
        - CVD pullback not too large
        - duration >= min_observe_seconds
        """
        cfg = self._config
        components: list[float] = []

        # 1. CVD confirms direction (0 or 1)
        if direction == "UP":
            cvd_ok = 1.0 if anchored_cvd > 0 else 0.0
        else:
            cvd_ok = 1.0 if anchored_cvd < 0 else 0.0
        components.append(cvd_ok)

        # 2. Buy/sell ratio sufficient (0 to 1)
        if buy_ratio >= cfg.min_cvd_ratio:
            ratio_score = min(1.0, (buy_ratio - cfg.min_cvd_ratio) / (1.0 - cfg.min_cvd_ratio) + 0.5)
        else:
            ratio_score = max(0.0, buy_ratio / cfg.min_cvd_ratio * 0.5)
        components.append(ratio_score)

        # 3. Price drift toward target edge (0 to 1)
        drift_score = max(0.0, min(1.0, price_drift * 10.0))
        components.append(drift_score)

        # 4. CVD pullback not too large (0 to 1)
        if cvd_pullback_ratio <= cfg.max_pullback_ratio:
            pullback_score = 1.0 - (cvd_pullback_ratio / max(cfg.max_pullback_ratio, 0.01)) * 0.5
        else:
            pullback_score = max(0.0, 0.5 - (cvd_pullback_ratio - cfg.max_pullback_ratio) * 2.0)
        components.append(max(0.0, pullback_score))

        # 5. Duration sufficient (0 to 1)
        if duration_sec >= cfg.min_observe_seconds:
            dur_score = 1.0
        else:
            dur_score = max(0.0, duration_sec / max(cfg.min_observe_seconds, 1))
        components.append(dur_score)

        # Equal weight average
        return sum(components) / len(components)
