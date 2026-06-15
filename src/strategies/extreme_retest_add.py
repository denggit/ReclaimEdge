"""EXTREME_RETEST_ADD — 15m outer-band extreme pivot retest add trigger.

This module provides pure-logic helpers for detecting 15m closed-candle
outer-band extreme pivots, maintaining an active anchor, and evaluating
Reject Before Break and Sweep Reclaim add triggers.

All state is scalar — no tick scanning, no pandas, no file IO.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from src.strategies.add_layer_gates import PositionSide, adverse_gap_pct
from src.utils.log import get_logger

logger = get_logger(__name__)

ExtremeRetestPattern = Literal["REJECT_BEFORE_BREAK", "SWEEP_RECLAIM"]
ExtremeRetestAnchorKind = Literal["PIVOT_HIGH", "PIVOT_LOW"]


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExtremeRetestConfig:
    """Immutable extreme retest configuration loaded from env."""

    enabled: bool = False

    pivot_left_bars: int = 2
    pivot_right_bars: int = 2
    anchor_max_age_candles: int = 12
    sweep_max_age_seconds: float = 900.0

    near_extreme_pct: float = 0.0015
    reclaim_pct: float = 0.0005
    min_reverse_ratio: float = 0.55

    one_add_per_anchor: bool = True

    @classmethod
    def from_env(cls) -> "ExtremeRetestConfig":
        return cls(
            enabled=_env_bool("EXTREME_RETEST_ADD_ENABLED", False),
            pivot_left_bars=int(os.getenv("EXTREME_RETEST_PIVOT_LEFT_BARS", "2")),
            pivot_right_bars=int(os.getenv("EXTREME_RETEST_PIVOT_RIGHT_BARS", "2")),
            anchor_max_age_candles=int(os.getenv("EXTREME_RETEST_ANCHOR_MAX_AGE_CANDLES", "12")),
            sweep_max_age_seconds=float(os.getenv("EXTREME_RETEST_SWEEP_MAX_AGE_SECONDS", "900")),
            near_extreme_pct=float(os.getenv("EXTREME_RETEST_NEAR_EXTREME_PCT", "0.0015")),
            reclaim_pct=float(os.getenv("EXTREME_RETEST_RECLAIM_PCT", "0.0005")),
            min_reverse_ratio=float(os.getenv("EXTREME_RETEST_MIN_REVERSE_RATIO", "0.55")),
            one_add_per_anchor=_env_bool("EXTREME_RETEST_ONE_ADD_PER_ANCHOR", True),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Anchor
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ExtremeRetestAnchor:
    """Active extreme retest anchor owned by the strategy state."""

    side: PositionSide | None = None
    kind: ExtremeRetestAnchorKind | None = None
    price: float | None = None
    candle_ts_ms: int | None = None
    boll_upper: float | None = None
    boll_lower: float | None = None

    # Sweep state
    sweep_seen: bool = False
    sweep_extreme_price: float | None = None
    sweep_first_seen_ts_ms: int | None = None
    sweep_last_seen_ts_ms: int | None = None

    # Consumed watermark
    consumed_watermark_price: float | None = None
    consumed_anchor_ts_ms: int | None = None

    def is_active(self) -> bool:
        return self.side is not None and self.price is not None

    def is_consumed(self) -> bool:
        return self.consumed_watermark_price is not None

    def clear(self) -> None:
        self.side = None
        self.kind = None
        self.price = None
        self.candle_ts_ms = None
        self.boll_upper = None
        self.boll_lower = None
        self.sweep_seen = False
        self.sweep_extreme_price = None
        self.sweep_first_seen_ts_ms = None
        self.sweep_last_seen_ts_ms = None

    def clear_sweep(self) -> None:
        self.sweep_seen = False
        self.sweep_extreme_price = None
        self.sweep_first_seen_ts_ms = None
        self.sweep_last_seen_ts_ms = None

    def consume(self) -> None:
        """Mark anchor consumed: record watermark, clear anchor + sweep."""
        if self.price is not None:
            self.consumed_watermark_price = self.price
        if self.candle_ts_ms is not None:
            self.consumed_anchor_ts_ms = self.candle_ts_ms
        self.clear()

    def to_dict(self) -> dict:
        return {
            "extreme_retest_anchor_side": self.side,
            "extreme_retest_anchor_kind": self.kind,
            "extreme_retest_anchor_price": self.price,
            "extreme_retest_anchor_candle_ts_ms": self.candle_ts_ms,
            "extreme_retest_anchor_boll_upper": self.boll_upper,
            "extreme_retest_anchor_boll_lower": self.boll_lower,
            "extreme_retest_sweep_seen": self.sweep_seen,
            "extreme_retest_sweep_extreme_price": self.sweep_extreme_price,
            "extreme_retest_sweep_first_seen_ts_ms": self.sweep_first_seen_ts_ms,
            "extreme_retest_sweep_last_seen_ts_ms": self.sweep_last_seen_ts_ms,
            "extreme_retest_consumed_watermark_price": self.consumed_watermark_price,
            "extreme_retest_consumed_anchor_ts_ms": self.consumed_anchor_ts_ms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExtremeRetestAnchor":
        return cls(
            side=d.get("extreme_retest_anchor_side"),
            kind=d.get("extreme_retest_anchor_kind"),
            price=d.get("extreme_retest_anchor_price"),
            candle_ts_ms=d.get("extreme_retest_anchor_candle_ts_ms"),
            boll_upper=d.get("extreme_retest_anchor_boll_upper"),
            boll_lower=d.get("extreme_retest_anchor_boll_lower"),
            sweep_seen=bool(d.get("extreme_retest_sweep_seen", False)),
            sweep_extreme_price=d.get("extreme_retest_sweep_extreme_price"),
            sweep_first_seen_ts_ms=d.get("extreme_retest_sweep_first_seen_ts_ms"),
            sweep_last_seen_ts_ms=d.get("extreme_retest_sweep_last_seen_ts_ms"),
            consumed_watermark_price=d.get("extreme_retest_consumed_watermark_price"),
            consumed_anchor_ts_ms=d.get("extreme_retest_consumed_anchor_ts_ms"),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation result
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExtremeRetestEvaluation:
    triggered: bool
    decision: str  # "reject_before_break" | "sweep_reclaim" | "none"
    reason: str  # human-readable skip reason when not triggered
    pattern: ExtremeRetestPattern | None = None
    anchor_price: float | None = None
    anchor_kind: ExtremeRetestAnchorKind | None = None
    sweep_seen: bool = False
    sweep_extreme_price: float | None = None
    inside_band: bool = False
    near_extreme: bool = False
    reclaimed: bool = False
    buy_ratio: float = 0.0
    sell_ratio: float = 0.0
    reverse_ratio_ok: bool = False

    @classmethod
    def not_triggered(cls, reason: str, **kwargs) -> "ExtremeRetestEvaluation":
        return cls(triggered=False, decision="none", reason=reason, **kwargs)

    @classmethod
    def reject_before_break(
        cls, anchor_price: float, anchor_kind: ExtremeRetestAnchorKind, **kwargs
    ) -> "ExtremeRetestEvaluation":
        return cls(
            triggered=True,
            decision="reject_before_break",
            reason="ok",
            pattern="REJECT_BEFORE_BREAK",
            anchor_price=anchor_price,
            anchor_kind=anchor_kind,
            **kwargs,
        )

    @classmethod
    def sweep_reclaim(
        cls, anchor_price: float, anchor_kind: ExtremeRetestAnchorKind, **kwargs
    ) -> "ExtremeRetestEvaluation":
        return cls(
            triggered=True,
            decision="sweep_reclaim",
            reason="ok",
            pattern="SWEEP_RECLAIM",
            anchor_price=anchor_price,
            anchor_kind=anchor_kind,
            **kwargs,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Candle helpers (no pandas, no dataframe)
# ──────────────────────────────────────────────────────────────────────────────


def _candle_high(c: dict) -> float:
    return float(c.get("high", c.get("h", 0)) or 0)


def _candle_low(c: dict) -> float:
    return float(c.get("low", c.get("l", 0)) or 0)


# ──────────────────────────────────────────────────────────────────────────────
# Pivot detection (closed candles only)
# ──────────────────────────────────────────────────────────────────────────────


def detect_pivot_high(
    candles: list[dict],  # list of closed 15m candles, index 0 = oldest
    pivot_idx: int,
    left_bars: int = 2,
    right_bars: int = 2,
) -> bool:
    """Check if candle[pivot_idx].high is a local pivot high.

    A pivot high requires:
    1. left N candles have lower highs
    2. right N candles have lower highs
    3. at least left_bars+right_bars+1 candles total
    """
    n = len(candles)
    if pivot_idx - left_bars < 0 or pivot_idx + right_bars >= n:
        return False

    pivot_high = _candle_high(candles[pivot_idx])
    if pivot_high <= 0:
        return False

    for i in range(pivot_idx - left_bars, pivot_idx):
        if _candle_high(candles[i]) >= pivot_high:
            return False

    for i in range(pivot_idx + 1, pivot_idx + right_bars + 1):
        if _candle_high(candles[i]) >= pivot_high:
            return False

    return True


def detect_pivot_low(
    candles: list[dict],
    pivot_idx: int,
    left_bars: int = 2,
    right_bars: int = 2,
) -> bool:
    """Check if candle[pivot_idx].low is a local pivot low.

    A pivot low requires:
    1. left N candles have higher lows
    2. right N candles have higher lows
    3. at least left_bars+right_bars+1 candles total
    """
    n = len(candles)
    if pivot_idx - left_bars < 0 or pivot_idx + right_bars >= n:
        return False

    pivot_low = _candle_low(candles[pivot_idx])
    if pivot_low <= 0:
        return False

    for i in range(pivot_idx - left_bars, pivot_idx):
        if _candle_low(candles[i]) <= pivot_low:
            return False

    for i in range(pivot_idx + 1, pivot_idx + right_bars + 1):
        if _candle_low(candles[i]) <= pivot_low:
            return False

    return True


# ──────────────────────────────────────────────────────────────────────────────
# Outer-band strict checks
# ──────────────────────────────────────────────────────────────────────────────


def is_outside_band_pivot_high(pivot_high: float, boll_upper: float) -> bool:
    """Pivot high must be STRICTLY above the upper band."""
    return pivot_high > boll_upper


def is_outside_band_pivot_low(pivot_low: float, boll_lower: float) -> bool:
    """Pivot low must be STRICTLY below the lower band."""
    return pivot_low < boll_lower


# ──────────────────────────────────────────────────────────────────────────────
# Anchor extremity comparison
# ──────────────────────────────────────────────────────────────────────────────


def is_more_extreme_anchor(
    side: PositionSide,
    candidate_price: float,
    existing_price: float | None,
) -> bool:
    """Return True if candidate is more extreme than the existing anchor."""
    if existing_price is None:
        return True
    if side == "SHORT":
        return candidate_price > existing_price
    return candidate_price < existing_price


def is_more_extreme_than_watermark(
    side: PositionSide,
    candidate_price: float,
    consumed_watermark_price: float | None,
) -> bool:
    """Return True if candidate is more extreme than the consumed watermark."""
    if consumed_watermark_price is None:
        return True
    if side == "SHORT":
        return candidate_price > consumed_watermark_price
    return candidate_price < consumed_watermark_price


# ──────────────────────────────────────────────────────────────────────────────
# Gap to last entry
# ──────────────────────────────────────────────────────────────────────────────


def is_anchor_far_enough_from_last_entry(
    side: PositionSide,
    anchor_price: float,
    last_entry_price: float | None,
    effective_required_gap_pct: float,
) -> tuple[bool, float, str]:
    """Check if the anchor is far enough from last_entry_price.

    Returns (ok, anchor_adverse_gap_pct, reason).
    """
    if last_entry_price is None or last_entry_price <= 0:
        return False, 0.0, "missing_last_entry"

    anchor_adverse_gap_pct = adverse_gap_pct(
        side=side, price=anchor_price, last_entry_price=last_entry_price
    )
    # For SHORT: pivot_high > last_entry, so adverse_gap = (pivot_high - last_entry) / last_entry
    # For LONG:  pivot_low < last_entry, so adverse_gap = (last_entry - pivot_low) / last_entry
    # In both cases adverse_gap_pct is positive when the anchor is in the right direction

    if side == "SHORT" and anchor_price <= last_entry_price:
        return False, anchor_adverse_gap_pct, "pivot_not_adverse_for_short"

    if side == "LONG" and anchor_price >= last_entry_price:
        return False, anchor_adverse_gap_pct, "pivot_not_adverse_for_long"

    if anchor_adverse_gap_pct < effective_required_gap_pct:
        return False, anchor_adverse_gap_pct, "too_close_to_last_entry"

    return True, anchor_adverse_gap_pct, "ok"


# ──────────────────────────────────────────────────────────────────────────────
# Reject Before Break evaluation
# ──────────────────────────────────────────────────────────────────────────────


def evaluate_reject_before_break(
    side: PositionSide,
    price: float,
    boll_upper: float,
    boll_lower: float,
    anchor: ExtremeRetestAnchor,
    config: ExtremeRetestConfig,
    buy_ratio: float,
    sell_ratio: float,
) -> ExtremeRetestEvaluation:
    """Evaluate Reject Before Break for an active anchor.

    SHORT: price is near and below anchor; anchor is PIVOT_HIGH
    LONG:  price is near and above anchor; anchor is PIVOT_LOW
    """
    if not anchor.is_active() or anchor.price is None:
        return ExtremeRetestEvaluation.not_triggered("no_active_anchor")

    # Must be inside band
    inside_band = (boll_lower <= price <= boll_upper)
    if not inside_band:
        return ExtremeRetestEvaluation.not_triggered(
            "price_not_inside_band",
            inside_band=False,
            buy_ratio=buy_ratio,
            sell_ratio=sell_ratio,
        )

    if side == "SHORT":
        if anchor.kind != "PIVOT_HIGH":
            return ExtremeRetestEvaluation.not_triggered(
                "anchor_kind_mismatch",
                inside_band=True,
                anchor_kind=anchor.kind,
                buy_ratio=buy_ratio,
                sell_ratio=sell_ratio,
            )
        # Price must be near anchor and at or below it
        near_threshold_low = anchor.price * (1.0 - config.near_extreme_pct)
        near_extreme = (near_threshold_low <= price <= anchor.price)
        if not near_extreme:
            return ExtremeRetestEvaluation.not_triggered(
                "not_near_extreme",
                inside_band=True,
                anchor_kind=anchor.kind,
                near_extreme=False,
                buy_ratio=buy_ratio,
                sell_ratio=sell_ratio,
                anchor_price=anchor.price,
            )
        reverse_ratio_ok = sell_ratio >= config.min_reverse_ratio
        if not reverse_ratio_ok:
            return ExtremeRetestEvaluation.not_triggered(
                "reverse_ratio_not_met",
                inside_band=True,
                anchor_kind=anchor.kind,
                near_extreme=True,
                buy_ratio=buy_ratio,
                sell_ratio=sell_ratio,
                reverse_ratio_ok=False,
                anchor_price=anchor.price,
            )
        return ExtremeRetestEvaluation.reject_before_break(
            anchor_price=anchor.price,
            anchor_kind=anchor.kind,
            inside_band=True,
            near_extreme=True,
            buy_ratio=buy_ratio,
            sell_ratio=sell_ratio,
            reverse_ratio_ok=True,
        )

    # LONG
    if anchor.kind != "PIVOT_LOW":
        return ExtremeRetestEvaluation.not_triggered(
            "anchor_kind_mismatch",
            inside_band=True,
            anchor_kind=anchor.kind,
            buy_ratio=buy_ratio,
            sell_ratio=sell_ratio,
        )
    # Price must be near anchor and at or above it
    near_threshold_high = anchor.price * (1.0 + config.near_extreme_pct)
    near_extreme = (anchor.price <= price <= near_threshold_high)
    if not near_extreme:
        return ExtremeRetestEvaluation.not_triggered(
            "not_near_extreme",
            inside_band=True,
            anchor_kind=anchor.kind,
            near_extreme=False,
            buy_ratio=buy_ratio,
            sell_ratio=sell_ratio,
            anchor_price=anchor.price,
        )
    reverse_ratio_ok = buy_ratio >= config.min_reverse_ratio
    if not reverse_ratio_ok:
        return ExtremeRetestEvaluation.not_triggered(
            "reverse_ratio_not_met",
            inside_band=True,
            anchor_kind=anchor.kind,
            near_extreme=True,
            buy_ratio=buy_ratio,
            sell_ratio=sell_ratio,
            reverse_ratio_ok=False,
            anchor_price=anchor.price,
        )
    return ExtremeRetestEvaluation.reject_before_break(
        anchor_price=anchor.price,
        anchor_kind=anchor.kind,
        inside_band=True,
        near_extreme=True,
        buy_ratio=buy_ratio,
        sell_ratio=sell_ratio,
        reverse_ratio_ok=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Sweep Reclaim evaluation
# ──────────────────────────────────────────────────────────────────────────────


def _update_sweep_state(
    side: PositionSide,
    price: float,
    anchor_price: float,
    ts_ms: int,
    anchor: ExtremeRetestAnchor,
    config: ExtremeRetestConfig,
) -> None:
    """Update sweep state if price breaks past the anchor."""
    # Expire old sweep if too old
    if anchor.sweep_seen and anchor.sweep_first_seen_ts_ms is not None:
        sweep_age_s = (ts_ms - anchor.sweep_first_seen_ts_ms) / 1000.0
        if sweep_age_s > config.sweep_max_age_seconds:
            anchor.clear_sweep()

    if side == "SHORT":
        if anchor.kind == "PIVOT_HIGH" and price > anchor_price:
            if not anchor.sweep_seen:
                anchor.sweep_seen = True
                anchor.sweep_first_seen_ts_ms = ts_ms
                anchor.sweep_last_seen_ts_ms = ts_ms
                anchor.sweep_extreme_price = price
            else:
                anchor.sweep_last_seen_ts_ms = ts_ms
                current_extreme = anchor.sweep_extreme_price or price
                anchor.sweep_extreme_price = max(current_extreme, price)
    else:  # LONG
        if anchor.kind == "PIVOT_LOW" and price < anchor_price:
            if not anchor.sweep_seen:
                anchor.sweep_seen = True
                anchor.sweep_first_seen_ts_ms = ts_ms
                anchor.sweep_last_seen_ts_ms = ts_ms
                anchor.sweep_extreme_price = price
            else:
                anchor.sweep_last_seen_ts_ms = ts_ms
                current_extreme = anchor.sweep_extreme_price or price
                anchor.sweep_extreme_price = min(current_extreme, price)


def evaluate_sweep_reclaim(
    side: PositionSide,
    price: float,
    ts_ms: int,
    boll_upper: float,
    boll_lower: float,
    anchor: ExtremeRetestAnchor,
    config: ExtremeRetestConfig,
    buy_ratio: float,
    sell_ratio: float,
) -> ExtremeRetestEvaluation:
    """Evaluate Sweep Reclaim for an active anchor.

    Phase 1: record sweep if price breaks past anchor.
    Phase 2: if sweep_seen, price is inside band, price reclaimed,
             and CVD reverse ratio meets threshold → trigger.
    """
    if not anchor.is_active() or anchor.price is None:
        return ExtremeRetestEvaluation.not_triggered("no_active_anchor")

    # Update sweep state
    _update_sweep_state(side, price, anchor.price, ts_ms, anchor, config)

    inside_band = (boll_lower <= price <= boll_upper)

    if not anchor.sweep_seen:
        return ExtremeRetestEvaluation.not_triggered(
            "sweep_not_seen",
            inside_band=inside_band,
            sweep_seen=False,
            anchor_price=anchor.price,
            anchor_kind=anchor.kind,
        )

    # Sweep has been seen — now check reclaim
    if not inside_band:
        return ExtremeRetestEvaluation.not_triggered(
            "price_not_inside_band",
            inside_band=False,
            sweep_seen=True,
            sweep_extreme_price=anchor.sweep_extreme_price,
            anchor_price=anchor.price,
            anchor_kind=anchor.kind,
        )

    if side == "SHORT":
        reclaim_threshold = anchor.price * (1.0 - config.reclaim_pct)
        reclaimed = (price <= reclaim_threshold)
        reverse_ratio_ok = sell_ratio >= config.min_reverse_ratio

        if not reclaimed:
            return ExtremeRetestEvaluation.not_triggered(
                "not_reclaimed",
                inside_band=True,
                sweep_seen=True,
                sweep_extreme_price=anchor.sweep_extreme_price,
                reclaimed=False,
                anchor_price=anchor.price,
                anchor_kind=anchor.kind,
                buy_ratio=buy_ratio,
                sell_ratio=sell_ratio,
            )
        if not reverse_ratio_ok:
            return ExtremeRetestEvaluation.not_triggered(
                "reverse_ratio_not_met",
                inside_band=True,
                sweep_seen=True,
                sweep_extreme_price=anchor.sweep_extreme_price,
                reclaimed=True,
                reverse_ratio_ok=False,
                anchor_price=anchor.price,
                anchor_kind=anchor.kind,
                buy_ratio=buy_ratio,
                sell_ratio=sell_ratio,
            )
        return ExtremeRetestEvaluation.sweep_reclaim(
            anchor_price=anchor.price,
            anchor_kind=anchor.kind,
            inside_band=True,
            sweep_seen=True,
            sweep_extreme_price=anchor.sweep_extreme_price,
            reclaimed=True,
            buy_ratio=buy_ratio,
            sell_ratio=sell_ratio,
            reverse_ratio_ok=True,
        )

    else:  # LONG
        reclaim_threshold = anchor.price * (1.0 + config.reclaim_pct)
        reclaimed = (price >= reclaim_threshold)
        reverse_ratio_ok = buy_ratio >= config.min_reverse_ratio

        if not reclaimed:
            return ExtremeRetestEvaluation.not_triggered(
                "not_reclaimed",
                inside_band=True,
                sweep_seen=True,
                sweep_extreme_price=anchor.sweep_extreme_price,
                reclaimed=False,
                anchor_price=anchor.price,
                anchor_kind=anchor.kind,
                buy_ratio=buy_ratio,
                sell_ratio=sell_ratio,
            )
        if not reverse_ratio_ok:
            return ExtremeRetestEvaluation.not_triggered(
                "reverse_ratio_not_met",
                inside_band=True,
                sweep_seen=True,
                sweep_extreme_price=anchor.sweep_extreme_price,
                reclaimed=True,
                reverse_ratio_ok=False,
                anchor_price=anchor.price,
                anchor_kind=anchor.kind,
                buy_ratio=buy_ratio,
                sell_ratio=sell_ratio,
            )
        return ExtremeRetestEvaluation.sweep_reclaim(
            anchor_price=anchor.price,
            anchor_kind=anchor.kind,
            inside_band=True,
            sweep_seen=True,
            sweep_extreme_price=anchor.sweep_extreme_price,
            reclaimed=True,
            buy_ratio=buy_ratio,
            sell_ratio=sell_ratio,
            reverse_ratio_ok=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Anchor lifecycle helpers
# ──────────────────────────────────────────────────────────────────────────────


def mark_anchor_consumed(anchor: ExtremeRetestAnchor) -> None:
    """Consume the anchor after a successful ADD intent."""
    price = anchor.price
    ts = anchor.candle_ts_ms
    anchor.consume()
    logger.info(
        "EXTREME_RETEST_ANCHOR_CONSUMED | anchor_price=%s anchor_candle_ts_ms=%s",
        price,
        ts,
    )


def revalidate_anchor_after_add(
    anchor: ExtremeRetestAnchor,
    last_entry_price: float | None,
    effective_required_gap_pct: float,
) -> str | None:
    """Revalidate active anchor after any ADD changes last_entry_price.

    Returns reason string if anchor was dropped, None if still valid.
    """
    if not anchor.is_active() or anchor.price is None or anchor.side is None:
        return None

    ok, gap_pct, reason = is_anchor_far_enough_from_last_entry(
        side=anchor.side,
        anchor_price=anchor.price,
        last_entry_price=last_entry_price,
        effective_required_gap_pct=effective_required_gap_pct,
    )

    if not ok:
        price = anchor.price
        side = anchor.side
        kind = anchor.kind
        anchor.clear()
        logger.info(
            "EXTREME_RETEST_ANCHOR_DROPPED | reason=too_close_after_new_entry "
            "anchor_price=%s last_entry_price=%s anchor_adverse_gap_pct=%.4f%% "
            "effective_required_gap_pct=%.4f%% side=%s kind=%s",
            price,
            last_entry_price,
            gap_pct * 100,
            effective_required_gap_pct * 100,
            side,
            kind,
        )
        return "too_close_after_new_entry"

    return None


# ──────────────────────────────────────────────────────────────────────────────
# Build / reject anchor (for use in strategy)
# ──────────────────────────────────────────────────────────────────────────────


def try_create_or_replace_anchor(
    side: PositionSide,
    candidate_price: float,
    candle_ts_ms: int,
    boll_upper: float,
    boll_lower: float,
    last_entry_price: float | None,
    effective_required_gap_pct: float,
    anchor: ExtremeRetestAnchor,
    config: ExtremeRetestConfig,
) -> tuple[bool, str]:
    """Try to create or replace the active anchor.

    Returns (action_taken, reason):
    - "created"
    - "replaced"
    - "ignored"
    - "rejected"
    """
    kind: ExtremeRetestAnchorKind = "PIVOT_HIGH" if side == "SHORT" else "PIVOT_LOW"

    # Check strict outer-band condition
    if side == "SHORT":
        if not is_outside_band_pivot_high(candidate_price, boll_upper):
            logger.info(
                "EXTREME_RETEST_ANCHOR_REJECTED | reason=not_outside_upper "
                "side=%s candidate_price=%s boll_upper=%s",
                side,
                candidate_price,
                boll_upper,
            )
            return False, "rejected"
    else:
        if not is_outside_band_pivot_low(candidate_price, boll_lower):
            logger.info(
                "EXTREME_RETEST_ANCHOR_REJECTED | reason=not_outside_lower "
                "side=%s candidate_price=%s boll_lower=%s",
                side,
                candidate_price,
                boll_lower,
            )
            return False, "rejected"

    # Check consumed watermark
    if anchor.is_consumed() and not is_more_extreme_than_watermark(
        side, candidate_price, anchor.consumed_watermark_price
    ):
        logger.info(
            "EXTREME_RETEST_ANCHOR_REJECTED | reason=not_more_extreme_than_consumed "
            "side=%s candidate_price=%s consumed_watermark_price=%s",
            side,
            candidate_price,
            anchor.consumed_watermark_price,
        )
        return False, "rejected"

    # Check if more extreme than existing active anchor
    if anchor.is_active() and anchor.side == side:
        if not is_more_extreme_anchor(side, candidate_price, anchor.price):
            logger.info(
                "EXTREME_RETEST_ANCHOR_REJECTED | reason=not_more_extreme "
                "side=%s candidate_price=%s active_anchor_price=%s",
                side,
                candidate_price,
                anchor.price,
            )
            return False, "ignored"
        # More extreme — will replace
        old_price = anchor.price
        old_sweep = anchor.sweep_seen

    # Check last_entry gap
    ok, gap_pct, gap_reason = is_anchor_far_enough_from_last_entry(
        side=side,
        anchor_price=candidate_price,
        last_entry_price=last_entry_price,
        effective_required_gap_pct=effective_required_gap_pct,
    )
    if not ok:
        logger.info(
            "EXTREME_RETEST_ANCHOR_REJECTED | reason=%s "
            "side=%s candidate_price=%s last_entry_price=%s "
            "anchor_adverse_gap_pct=%.4f%% effective_required_gap_pct=%.4f%% "
            "boll_upper=%s boll_lower=%s consumed_watermark_price=%s",
            gap_reason,
            side,
            candidate_price,
            last_entry_price,
            gap_pct * 100,
            effective_required_gap_pct * 100,
            boll_upper,
            boll_lower,
            anchor.consumed_watermark_price,
        )
        return False, "rejected"

    # Create or replace
    if anchor.is_active() and anchor.side == side:
        old_price = anchor.price
        old_sweep_seen = anchor.sweep_seen
        old_consumed = anchor.is_consumed()
        anchor.side = side
        anchor.kind = kind
        anchor.price = candidate_price
        anchor.candle_ts_ms = candle_ts_ms
        anchor.boll_upper = boll_upper
        anchor.boll_lower = boll_lower
        anchor.clear_sweep()
        outside_distance_pct = (
            (candidate_price - boll_upper) / boll_upper
            if side == "SHORT"
            else (boll_lower - candidate_price) / boll_lower
        )
        logger.info(
            "EXTREME_RETEST_ANCHOR_REPLACED | side=%s old_anchor_price=%s "
            "new_anchor_price=%s reason=MORE_EXTREME_PIVOT "
            "old_sweep_seen=%s old_used=%s",
            side,
            old_price,
            candidate_price,
            old_sweep_seen,
            old_consumed,
        )
        logger.info(
            "EXTREME_RETEST_ANCHOR_UPDATED | side=%s kind=%s anchor_price=%s "
            "anchor_candle_ts_ms=%s boll_upper=%s boll_lower=%s "
            "outside_band_distance_pct=%.4f%% last_entry_price=%s "
            "anchor_adverse_gap_pct=%.4f%% effective_required_gap_pct=%.4f%% "
            "pivot_left_bars=%s pivot_right_bars=%s",
            side,
            kind,
            candidate_price,
            candle_ts_ms,
            boll_upper,
            boll_lower,
            outside_distance_pct * 100,
            last_entry_price,
            gap_pct * 100,
            effective_required_gap_pct * 100,
            config.pivot_left_bars,
            config.pivot_right_bars,
        )
        return True, "replaced"

    # Fresh create
    anchor.side = side
    anchor.kind = kind
    anchor.price = candidate_price
    anchor.candle_ts_ms = candle_ts_ms
    anchor.boll_upper = boll_upper
    anchor.boll_lower = boll_lower
    anchor.clear_sweep()
    outside_distance_pct = (
        (candidate_price - boll_upper) / boll_upper
        if side == "SHORT"
        else (boll_lower - candidate_price) / boll_lower
    )
    logger.info(
        "EXTREME_RETEST_ANCHOR_UPDATED | side=%s kind=%s anchor_price=%s "
        "anchor_candle_ts_ms=%s boll_upper=%s boll_lower=%s "
        "outside_band_distance_pct=%.4f%% last_entry_price=%s "
        "anchor_adverse_gap_pct=%.4f%% effective_required_gap_pct=%.4f%% "
        "pivot_left_bars=%s pivot_right_bars=%s",
        side,
        kind,
        candidate_price,
        candle_ts_ms,
        boll_upper,
        boll_lower,
        outside_distance_pct * 100,
        last_entry_price,
        gap_pct * 100,
        effective_required_gap_pct * 100,
        config.pivot_left_bars,
        config.pivot_right_bars,
    )
    return True, "created"


# ──────────────────────────────────────────────────────────────────────────────
# Anchor age check
# ──────────────────────────────────────────────────────────────────────────────


def is_anchor_expired(
    anchor: ExtremeRetestAnchor,
    current_candle_ts_ms: int,
    max_age_candles: int,
) -> bool:
    """Check if the anchor is too old (based on candle count, not wall time)."""
    if not anchor.is_active() or anchor.candle_ts_ms is None:
        return False
    # Approximate: if the candle timestamp difference exceeds max_age_candles * 15min
    max_age_ms = max_age_candles * 15 * 60_000
    return current_candle_ts_ms - anchor.candle_ts_ms > max_age_ms


def drop_expired_anchor(
    anchor: ExtremeRetestAnchor,
    current_candle_ts_ms: int,
    max_age_candles: int,
) -> bool:
    """Drop anchor if expired. Returns True if dropped."""
    if is_anchor_expired(anchor, current_candle_ts_ms, max_age_candles):
        logger.info(
            "EXTREME_RETEST_ANCHOR_DROPPED | reason=expired "
            "anchor_price=%s anchor_candle_ts_ms=%s current_candle_ts_ms=%s",
            anchor.price,
            anchor.candle_ts_ms,
            current_candle_ts_ms,
        )
        anchor.clear()
        return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Closed-candle rebuilding (for startup restore when state is untrusted)
# ──────────────────────────────────────────────────────────────────────────────


def rebuild_anchor_from_closed_candles(
    side: PositionSide,
    candles: list[dict],  # most recent closed candles, index 0 = oldest
    boll_upper: float,
    boll_lower: float,
    last_entry_price: float | None,
    effective_required_gap_pct: float,
    consumed_watermark_price: float | None,
    config: ExtremeRetestConfig,
) -> ExtremeRetestAnchor | None:
    """Try to rebuild an anchor from closed 15m candle history.

    Only used during startup restore when saved state is not trusted.
    Each candidate pivot is validated against its OWN candle's boll_upper/boll_lower.
    """
    anchor = ExtremeRetestAnchor()
    pivot_left = config.pivot_left_bars
    pivot_right = config.pivot_right_bars

    best_price: float | None = None
    best_idx: int = -1
    best_upper: float = 0.0
    best_lower: float = 0.0

    # Scan from newest to oldest (index len-1 down to pivot_left)
    for i in range(len(candles) - 1 - pivot_right, pivot_left - 1, -1):
        candidate = candles[i]
        # Use THIS candidate's own boll band
        candidate_upper = float(candidate.get("boll_upper") or 0)
        candidate_lower = float(candidate.get("boll_lower") or 0)

        if side == "SHORT":
            if not detect_pivot_high(candles, i, pivot_left, pivot_right):
                continue
            pivot_price = _candle_high(candles[i])
            if not is_outside_band_pivot_high(pivot_price, candidate_upper):
                continue
            if best_price is None or pivot_price > best_price:
                best_price = pivot_price
                best_idx = i
                best_upper = candidate_upper
                best_lower = candidate_lower
        else:
            if not detect_pivot_low(candles, i, pivot_left, pivot_right):
                continue
            pivot_price = _candle_low(candles[i])
            if not is_outside_band_pivot_low(pivot_price, candidate_lower):
                continue
            if best_price is None or pivot_price < best_price:
                best_price = pivot_price
                best_idx = i
                best_upper = candidate_upper
                best_lower = candidate_lower

    if best_price is None or best_idx < 0:
        logger.info(
            "EXTREME_RETEST_STATE_DROPPED | reason=untrusted_saved_state_or_no_valid_anchor "
            "side=%s",
            side,
        )
        return None

    # Check consumed watermark
    if not is_more_extreme_than_watermark(side, best_price, consumed_watermark_price):
        logger.info(
            "EXTREME_RETEST_STATE_DROPPED | reason=untrusted_saved_state_or_no_valid_anchor "
            "side=%s candidate_price=%s consumed_watermark=%s",
            side,
            best_price,
            consumed_watermark_price,
        )
        return None

    # Check last_entry gap
    ok, gap_pct, gap_reason = is_anchor_far_enough_from_last_entry(
        side=side,
        anchor_price=best_price,
        last_entry_price=last_entry_price,
        effective_required_gap_pct=effective_required_gap_pct,
    )
    if not ok:
        logger.info(
            "EXTREME_RETEST_STATE_DROPPED | reason=untrusted_saved_state_or_no_valid_anchor "
            "side=%s candidate_price=%s reason_detail=%s",
            side,
            best_price,
            gap_reason,
        )
        return None

    candle = candles[best_idx]
    candle_ts = int(candle.get("ts_ms", candle.get("ts", 0)) or 0)
    kind: ExtremeRetestAnchorKind = "PIVOT_HIGH" if side == "SHORT" else "PIVOT_LOW"

    anchor.side = side
    anchor.kind = kind
    anchor.price = best_price
    anchor.candle_ts_ms = candle_ts
    # Store the CANDIDATE's own boll band, not the latest candle's
    anchor.boll_upper = best_upper
    anchor.boll_lower = best_lower
    anchor.consumed_watermark_price = consumed_watermark_price

    logger.info(
        "EXTREME_RETEST_STATE_REBUILT_FROM_CANDLES | side=%s kind=%s "
        "anchor_price=%s anchor_candle_ts_ms=%s "
        "anchor_boll_upper=%s anchor_boll_lower=%s",
        side,
        kind,
        best_price,
        candle_ts,
        best_upper,
        best_lower,
    )
    return anchor


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation on tick (orchestrates Reject + Sweep evaluation)
# ──────────────────────────────────────────────────────────────────────────────


def evaluate_on_tick(
    side: PositionSide,
    price: float,
    ts_ms: int,
    boll_upper: float,
    boll_lower: float,
    anchor: ExtremeRetestAnchor,
    config: ExtremeRetestConfig,
    buy_ratio: float,
    sell_ratio: float,
) -> ExtremeRetestEvaluation:
    """Evaluate both Reject Before Break and Sweep Reclaim on a tick.

    This is the main entry point for the tick path.
    """
    # First try reject_before_break (this does not mutate anchor)
    reject_eval = evaluate_reject_before_break(
        side=side,
        price=price,
        boll_upper=boll_upper,
        boll_lower=boll_lower,
        anchor=anchor,
        config=config,
        buy_ratio=buy_ratio,
        sell_ratio=sell_ratio,
    )
    if reject_eval.triggered:
        return reject_eval

    # Then try sweep_reclaim (this may mutate anchor.sweep_*)
    sweep_eval = evaluate_sweep_reclaim(
        side=side,
        price=price,
        ts_ms=ts_ms,
        boll_upper=boll_upper,
        boll_lower=boll_lower,
        anchor=anchor,
        config=config,
        buy_ratio=buy_ratio,
        sell_ratio=sell_ratio,
    )
    return sweep_eval


# ──────────────────────────────────────────────────────────────────────────────
# Utils
# ──────────────────────────────────────────────────────────────────────────────


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
