"""Middle Bucket Split — pure calculation module.

Splits the middle bucket (TP1 / first-close) into two legs:
  - fast leg:  placed at BOLL15 middle (tp_middle)
  - slow leg:  placed at BOLL20 middle (structure middle)

This module does NO I/O, NO state access, NO OKX calls, NO logging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MiddleBucketSplitDecision:
    """Result of evaluating whether the middle bucket can be split.

    The ``action`` field is the canonical driver for upstream control flow.
    Callers MUST branch on ``action``, not on ``reason`` strings.
    """

    enabled: bool
    reason: str
    action: Literal[
        "DISABLED",
        "SPLIT",
        "UNSPLIT_SLOW_MIDDLE",
        "FALLBACK_OUTER",
        "INVALID",
    ]
    middle_bucket_ratio: float
    fast_ratio_of_bucket: float
    slow_ratio_of_bucket: float
    fast_total_ratio: float
    slow_total_ratio: float
    fast_price: float | None
    slow_price: float | None
    effective_price: float | None
    required_price: float | None


def build_middle_bucket_split_decision(
    *,
    enabled: bool,
    side: Literal["LONG", "SHORT"],
    middle_bucket_ratio: float,
    fast_ratio_of_bucket: float,
    fast_middle_price: float | None,
    slow_middle_price: float | None,
    effective_breakeven: float,
    min_net_profit_pct: float,
) -> MiddleBucketSplitDecision:
    """Evaluate whether the middle bucket can be split into fast/slow legs.

    Returns a decision with enabled=True only when both fast and slow middle
    prices satisfy the minimum net-profit requirement relative to the
    effective breakeven price.
    """

    def _disabled(
        reason: str,
        action: Literal["DISABLED", "INVALID"] = "INVALID",
    ) -> MiddleBucketSplitDecision:
        return MiddleBucketSplitDecision(
            enabled=False,
            reason=reason,
            action=action,
            middle_bucket_ratio=middle_bucket_ratio,
            fast_ratio_of_bucket=fast_ratio_of_bucket,
            slow_ratio_of_bucket=1.0 - fast_ratio_of_bucket,
            fast_total_ratio=0.0,
            slow_total_ratio=0.0,
            fast_price=fast_middle_price,
            slow_price=slow_middle_price,
            effective_price=None,
            required_price=None,
        )

    if not enabled:
        return _disabled("disabled", action="DISABLED")

    if middle_bucket_ratio <= 0.0 or middle_bucket_ratio >= 1.0:
        return _disabled("invalid_middle_bucket_ratio")

    if fast_ratio_of_bucket <= 0.0 or fast_ratio_of_bucket >= 1.0:
        return _disabled("invalid_fast_ratio")

    if fast_middle_price is None:
        return _disabled("fast_middle_missing")

    if slow_middle_price is None:
        return _disabled("slow_middle_missing")

    if effective_breakeven <= 0.0:
        return _disabled("invalid_effective_breakeven")

    min_profit = abs(float(min_net_profit_pct))

    if side == "LONG":
        required_price = effective_breakeven * (1.0 + min_profit)
        fast_ok = fast_middle_price >= required_price
        slow_ok = slow_middle_price >= required_price
    else:
        required_price = effective_breakeven * (1.0 - min_profit)
        fast_ok = fast_middle_price <= required_price
        slow_ok = slow_middle_price <= required_price

    slow_ratio_of_bucket = 1.0 - fast_ratio_of_bucket
    fast_total_ratio = middle_bucket_ratio * fast_ratio_of_bucket
    slow_total_ratio = middle_bucket_ratio * slow_ratio_of_bucket

    if fast_ok and slow_ok:
        effective_price = (
            fast_middle_price * fast_ratio_of_bucket
            + slow_middle_price * slow_ratio_of_bucket
        )
        return MiddleBucketSplitDecision(
            enabled=True,
            reason="split_enabled",
            action="SPLIT",
            middle_bucket_ratio=middle_bucket_ratio,
            fast_ratio_of_bucket=fast_ratio_of_bucket,
            slow_ratio_of_bucket=slow_ratio_of_bucket,
            fast_total_ratio=fast_total_ratio,
            slow_total_ratio=slow_total_ratio,
            fast_price=fast_middle_price,
            slow_price=slow_middle_price,
            effective_price=effective_price,
            required_price=required_price,
        )

    if not fast_ok and slow_ok:
        return MiddleBucketSplitDecision(
            enabled=False,
            reason="fast_middle_profit_insufficient_slow_middle_ok",
            action="UNSPLIT_SLOW_MIDDLE",
            middle_bucket_ratio=middle_bucket_ratio,
            fast_ratio_of_bucket=fast_ratio_of_bucket,
            slow_ratio_of_bucket=slow_ratio_of_bucket,
            fast_total_ratio=0.0,
            slow_total_ratio=0.0,
            fast_price=fast_middle_price,
            slow_price=slow_middle_price,
            effective_price=None,
            required_price=required_price,
        )

    if not fast_ok and not slow_ok:
        return MiddleBucketSplitDecision(
            enabled=False,
            reason="middle_profit_insufficient",
            action="FALLBACK_OUTER",
            middle_bucket_ratio=middle_bucket_ratio,
            fast_ratio_of_bucket=fast_ratio_of_bucket,
            slow_ratio_of_bucket=slow_ratio_of_bucket,
            fast_total_ratio=0.0,
            slow_total_ratio=0.0,
            fast_price=fast_middle_price,
            slow_price=slow_middle_price,
            effective_price=None,
            required_price=required_price,
        )

    # fast_ok=True but slow_ok=False (theoretically rare)
    return MiddleBucketSplitDecision(
        enabled=False,
        reason="slow_middle_profit_insufficient",
        action="FALLBACK_OUTER",
        middle_bucket_ratio=middle_bucket_ratio,
        fast_ratio_of_bucket=fast_ratio_of_bucket,
        slow_ratio_of_bucket=slow_ratio_of_bucket,
        fast_total_ratio=0.0,
        slow_total_ratio=0.0,
        fast_price=fast_middle_price,
        slow_price=slow_middle_price,
        effective_price=None,
        required_price=required_price,
    )


def calculate_fast_protective_sl(
    *,
    side: Literal["LONG", "SHORT"],
    avg_entry_price: float,
    fee_buffer_pct: float,
) -> float | None:
    """Calculate the breakeven-protection SL price for the fast leg.

    Uses the ORIGINAL avg_entry_price (not affected by partial fills).
    Returns None if avg_entry_price is invalid.
    """
    if avg_entry_price <= 0.0:
        return None
    if side == "LONG":
        return avg_entry_price * (1.0 + float(fee_buffer_pct))
    return avg_entry_price * (1.0 - float(fee_buffer_pct))


def is_stop_valid_for_current_price(
    *,
    side: Literal["LONG", "SHORT"],
    stop_price: float | None,
    current_price: float,
) -> bool:
    """Check whether a stop price is still valid relative to current price.

    LONG:  stop must be BELOW current price to be valid.
    SHORT: stop must be ABOVE current price to be valid.
    """
    if stop_price is None or current_price <= 0.0:
        return False
    if side == "LONG":
        return float(stop_price) < float(current_price)
    return float(stop_price) > float(current_price)
