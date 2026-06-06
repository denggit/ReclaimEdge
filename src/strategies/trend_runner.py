"""Pure Trend Runner state values, reset helpers, dynamic TP/SL calculation,
market-exit reason judgement, reverse burst candidate / confirmation / pruning,
and reverse extreme price update.

These functions do NOT import the strategy class, state, logger, or env.
All inputs are received as explicit keyword arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PositionSide = Literal["LONG", "SHORT"]


# ── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrendRunnerStateValues:
    """All Trend Runner main state fields as an immutable value object."""

    trend_runner_active: bool
    trend_runner_trend_start_ts_ms: int
    trend_runner_adjust_count: int
    trend_runner_last_update_candle_ts_ms: int
    trend_runner_tp_price: float | None
    trend_runner_sl_price: float | None
    trend_runner_tp_order_id: str | None
    trend_runner_sl_order_id: str | None
    trend_runner_exit_reason: str | None


@dataclass(frozen=True)
class TrendRunnerReverseStateValues:
    """All Trend Runner reverse-burst state fields as an immutable value object."""

    trend_runner_reverse_candidate: bool
    trend_runner_reverse_start_ts_ms: int
    trend_runner_reverse_start_price: float | None
    trend_runner_reverse_extreme_price: float | None
    trend_runner_reverse_fast_cvd_start: float
    trend_runner_reverse_samples: list


@dataclass(frozen=True)
class TrendRunnerDynamicOrders:
    """Result of calculating Trend Runner dynamic TP/SL orders."""

    tp_price: float
    sl_price: float
    tp_extra_pct: float
    sl_distance_ratio: float


@dataclass(frozen=True)
class TrendRunnerExitDecision:
    """Result of evaluating whether Trend Runner should market-exit."""

    should_exit: bool
    reason: str | None


@dataclass(frozen=True)
class TrendRunnerReverseCandidateDecision:
    """Result of checking whether a reverse-burst candidate is present."""

    is_candidate: bool


@dataclass(frozen=True)
class TrendRunnerReverseConfirmDecision:
    """Result of confirming a Trend Runner reverse burst."""

    confirmed: bool
    avg_ratio: float
    price_damage_pct: float
    recovery_pct: float


@dataclass(frozen=True)
class TrendRunnerReverseProgressDecision:
    """Result of progressing Trend Runner reverse burst sampling / confirmation."""

    reason: str | None
    should_start_candidate: bool
    should_reset: bool
    new_extreme_price: float | None
    pruned_samples: list
    elapsed_ms: int


# ── Reset helpers ───────────────────────────────────────────────────────────

def reset_trend_runner_state_values() -> TrendRunnerStateValues:
    """Return the reset / initial Trend Runner main state values."""
    return TrendRunnerStateValues(
        trend_runner_active=False,
        trend_runner_trend_start_ts_ms=0,
        trend_runner_adjust_count=0,
        trend_runner_last_update_candle_ts_ms=0,
        trend_runner_tp_price=None,
        trend_runner_sl_price=None,
        trend_runner_tp_order_id=None,
        trend_runner_sl_order_id=None,
        trend_runner_exit_reason=None,
    )


def reset_trend_runner_reverse_state_values() -> TrendRunnerReverseStateValues:
    """Return the reset / initial Trend Runner reverse-burst state values."""
    return TrendRunnerReverseStateValues(
        trend_runner_reverse_candidate=False,
        trend_runner_reverse_start_ts_ms=0,
        trend_runner_reverse_start_price=None,
        trend_runner_reverse_extreme_price=None,
        trend_runner_reverse_fast_cvd_start=0.0,
        trend_runner_reverse_samples=[],
    )


# ── Dynamic TP/SL calculation ───────────────────────────────────────────────

def calculate_trend_runner_dynamic_orders(
    *,
    side: PositionSide,
    boll_middle: float,
    boll_upper: float,
    boll_lower: float,
    adjust_count: int,
    current_sl_price: float | None,
    runner_tp_initial_outer_extra_pct: float,
    runner_tp_step_pct: float,
    runner_tp_min_outer_extra_pct: float,
    runner_sl_initial_outer_distance_ratio: float,
    runner_sl_step_ratio: float,
    runner_sl_min_outer_distance_ratio: float,
) -> TrendRunnerDynamicOrders:
    """Calculate Trend Runner dynamic TP/SL prices.

    Equivalent to the original ``_calculate_trend_runner_dynamic_orders``.
    """
    adjust = max(adjust_count, 0)
    tp_extra_pct = max(
        runner_tp_min_outer_extra_pct,
        runner_tp_initial_outer_extra_pct - adjust * runner_tp_step_pct,
    )
    sl_distance_ratio = max(
        runner_sl_min_outer_distance_ratio,
        runner_sl_initial_outer_distance_ratio - adjust * runner_sl_step_ratio,
    )

    if side == "LONG":
        tp_price = boll_upper * (1.0 + tp_extra_pct)
        sl_candidate = boll_upper - (boll_upper - boll_middle) * sl_distance_ratio
        if current_sl_price is not None:
            sl_price = max(current_sl_price, sl_candidate)
        else:
            sl_price = sl_candidate
    else:  # SHORT
        tp_price = boll_lower * (1.0 - tp_extra_pct)
        sl_candidate = boll_lower + (boll_middle - boll_lower) * sl_distance_ratio
        if current_sl_price is not None:
            sl_price = min(current_sl_price, sl_candidate)
        else:
            sl_price = sl_candidate

    return TrendRunnerDynamicOrders(
        tp_price=tp_price,
        sl_price=sl_price,
        tp_extra_pct=tp_extra_pct,
        sl_distance_ratio=sl_distance_ratio,
    )


# ── Market exit reason ──────────────────────────────────────────────────────

def trend_runner_market_exit_reason(
    *,
    side: PositionSide,
    price: float,
    boll_middle: float,
    tp_price: float | None,
    sl_price: float | None,
    trend_start_ts_ms: int,
    ts_ms: int,
    runner_max_trend_seconds_after_second_tp: int,
) -> TrendRunnerExitDecision:
    """Evaluate static Trend Runner market-exit conditions (no reverse burst).

    Checks (in priority order):
    1. TP crossed
    2. SL failsafe
    3. Middle band lost
    4. Max trend time exceeded after second TP

    Reverse-burst is NOT evaluated here; the caller handles it separately.
    """
    # 1. TP crossed
    if tp_price is not None:
        if side == "LONG" and price >= tp_price:
            return TrendRunnerExitDecision(should_exit=True, reason="trend_runner_tp_crossed")
        if side == "SHORT" and price <= tp_price:
            return TrendRunnerExitDecision(should_exit=True, reason="trend_runner_tp_crossed")

    # 2. SL failsafe
    if sl_price is not None:
        if side == "LONG" and price <= sl_price:
            return TrendRunnerExitDecision(should_exit=True, reason="trend_runner_sl_failsafe")
        if side == "SHORT" and price >= sl_price:
            return TrendRunnerExitDecision(should_exit=True, reason="trend_runner_sl_failsafe")

    # 3. Middle band lost
    if side == "LONG" and price < boll_middle:
        return TrendRunnerExitDecision(should_exit=True, reason="trend_runner_middle_lost")
    if side == "SHORT" and price > boll_middle:
        return TrendRunnerExitDecision(should_exit=True, reason="trend_runner_middle_lost")

    # 4. Max trend time exceeded
    start_ts = int(trend_start_ts_ms or 0)
    max_trend_ms = int(runner_max_trend_seconds_after_second_tp * 1000)
    if start_ts > 0 and max_trend_ms > 0 and ts_ms - start_ts >= max_trend_ms:
        return TrendRunnerExitDecision(should_exit=True, reason="trend_runner_max_time_after_second_tp")

    return TrendRunnerExitDecision(should_exit=False, reason=None)


# ── Reverse burst candidate ─────────────────────────────────────────────────

def trend_runner_reverse_candidate(
    *,
    side: PositionSide,
    up_burst: bool,
    down_burst: bool,
    buy_ratio: float,
    sell_ratio: float,
    fast_cvd: float,
    cvd_increasing: bool,
    cvd_decreasing: bool,
    runner_reverse_strong_ratio: float,
) -> TrendRunnerReverseCandidateDecision:
    """Check whether a Trend Runner reverse-burst candidate condition is met.

    Equivalent to the original ``_trend_runner_reverse_candidate``.
    """
    if side == "LONG":
        is_candidate = bool(
            down_burst
            or (
                sell_ratio >= runner_reverse_strong_ratio
                and fast_cvd < 0
                and cvd_decreasing
            )
        )
    else:  # SHORT
        is_candidate = bool(
            up_burst
            or (
                buy_ratio >= runner_reverse_strong_ratio
                and fast_cvd > 0
                and cvd_increasing
            )
        )
    return TrendRunnerReverseCandidateDecision(is_candidate=is_candidate)


# ── Reverse extreme price ───────────────────────────────────────────────────

def update_trend_runner_reverse_extreme_price(
    *,
    side: PositionSide,
    current_extreme_price: float | None,
    price: float,
) -> float:
    """Update the Trend Runner reverse extreme price for the current tick.

    Equivalent to the original in-line extreme price update logic.
    """
    if side == "LONG":
        return min(current_extreme_price if current_extreme_price is not None else price, price)
    else:  # SHORT
        return max(current_extreme_price if current_extreme_price is not None else price, price)


# ── Reverse sample pruning ──────────────────────────────────────────────────

def prune_trend_runner_reverse_samples(
    *,
    samples: list,
    cutoff_ts_ms: int,
) -> list:
    """Prune Trend Runner reverse samples to those within the confirm window.

    Equivalent to the original list comprehension filter.
    """
    return [sample for sample in samples if sample[0] >= cutoff_ts_ms]


# ── Reverse burst confirmation ──────────────────────────────────────────────

def trend_runner_reverse_confirmed(
    *,
    side: PositionSide,
    current_price: float,
    samples: list,
    start_price: float | None,
    extreme_price: float | None,
    fast_cvd_start: float,
    current_fast_cvd: float,
    runner_reverse_sell_ratio: float,
    runner_reverse_buy_ratio: float,
    runner_reverse_min_price_damage_pct: float,
    runner_reverse_recovery_cancel_pct: float,
) -> TrendRunnerReverseConfirmDecision:
    """Confirm whether the Trend Runner reverse burst should trigger.

    Equivalent to the original ``_trend_runner_reverse_confirmed``.
    """
    if not samples:
        return TrendRunnerReverseConfirmDecision(
            confirmed=False, avg_ratio=0.0, price_damage_pct=0.0, recovery_pct=0.0,
        )

    if start_price is None or start_price <= 0 or extreme_price is None or extreme_price <= 0:
        return TrendRunnerReverseConfirmDecision(
            confirmed=False, avg_ratio=0.0, price_damage_pct=0.0, recovery_pct=0.0,
        )

    if side == "LONG":
        avg_ratio = sum(float(sample[2]) for sample in samples) / len(samples)
        price_damage_pct = (start_price - current_price) / start_price
        recovery_pct = (current_price - extreme_price) / extreme_price
        confirmed = (
            avg_ratio >= runner_reverse_sell_ratio
            and current_fast_cvd < fast_cvd_start
            and price_damage_pct >= runner_reverse_min_price_damage_pct
            and recovery_pct < runner_reverse_recovery_cancel_pct
        )
    else:  # SHORT
        avg_ratio = sum(float(sample[1]) for sample in samples) / len(samples)
        price_damage_pct = (current_price - start_price) / start_price
        recovery_pct = (extreme_price - current_price) / extreme_price
        confirmed = (
            avg_ratio >= runner_reverse_buy_ratio
            and current_fast_cvd > fast_cvd_start
            and price_damage_pct >= runner_reverse_min_price_damage_pct
            and recovery_pct < runner_reverse_recovery_cancel_pct
        )

    return TrendRunnerReverseConfirmDecision(
        confirmed=confirmed,
        avg_ratio=avg_ratio,
        price_damage_pct=price_damage_pct,
        recovery_pct=recovery_pct,
    )
