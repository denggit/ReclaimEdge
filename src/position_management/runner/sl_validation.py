from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class RunnerSlValidationResult:
    valid: bool
    reason: str


def validate_runner_protective_sl_price(
    *,
    side: str,
    current_price: Decimal,
    new_sl_price: Decimal,
    tick_size: Decimal | None = None,
) -> RunnerSlValidationResult:
    _ = tick_size
    if current_price <= 0:
        return RunnerSlValidationResult(True, "missing_current_price_skip_validation")
    normalized_side = str(side or "").upper()
    if normalized_side == "LONG":
        if new_sl_price < current_price:
            return RunnerSlValidationResult(True, "ok")
        return RunnerSlValidationResult(False, "long_sl_not_below_last_price")
    if normalized_side == "SHORT":
        if new_sl_price > current_price:
            return RunnerSlValidationResult(True, "ok")
        return RunnerSlValidationResult(False, "short_sl_not_above_last_price")
    return RunnerSlValidationResult(False, "unknown_side")
