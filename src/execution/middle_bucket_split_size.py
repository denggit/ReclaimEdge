"""Middle Bucket Split size pre-check — pure calculation module.

Pre-checks whether split sub-leg contract sizes meet the minimum-contract
requirement BEFORE order construction, so the execution layer can disable
the split and propagate the result back to the strategy state layer.

This module does NO I/O, NO OKX calls, NO state access, NO logging.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from src.execution.order_specs import round_contracts_down


@dataclass(frozen=True)
class MiddleBucketSplitSizeCheck:
    ok: bool
    reason: str
    tp1_total_contracts: Decimal
    fast_contracts: Decimal
    slow_contracts: Decimal
    min_contracts: Decimal


def check_three_stage_middle_bucket_split_size(
    *,
    position_contracts: Decimal,
    min_contracts: Decimal,
    contract_precision: Decimal,
    three_stage_tp1_ratio: Decimal,
    fast_ratio_of_bucket: Decimal,
) -> MiddleBucketSplitSizeCheck:
    """Pre-check Three-Stage middle bucket split sub-leg sizes.

    Uses EXACTLY the same rounding rules as
    :func:`src.execution.order_specs.build_take_profit_order_specs`.

    Returns ``ok=True`` only when both fast_contracts and slow_contracts
    are >= min_contracts.
    """
    _rnd = lambda c: round_contracts_down(contracts=c, contract_precision=contract_precision)

    if three_stage_tp1_ratio <= 0 or fast_ratio_of_bucket <= 0 or fast_ratio_of_bucket >= 1:
        return MiddleBucketSplitSizeCheck(
            ok=False,
            reason="invalid_ratios",
            tp1_total_contracts=Decimal("0"),
            fast_contracts=Decimal("0"),
            slow_contracts=Decimal("0"),
            min_contracts=min_contracts,
        )

    tp1_total = _rnd(position_contracts * three_stage_tp1_ratio)
    fast_contracts = _rnd(tp1_total * fast_ratio_of_bucket)
    slow_contracts = tp1_total - fast_contracts

    ok = fast_contracts >= min_contracts and slow_contracts >= min_contracts
    reason = "ok" if ok else "subleg_too_small"

    return MiddleBucketSplitSizeCheck(
        ok=ok,
        reason=reason,
        tp1_total_contracts=tp1_total,
        fast_contracts=fast_contracts,
        slow_contracts=slow_contracts,
        min_contracts=min_contracts,
    )


def check_middle_runner_bucket_split_size(
    *,
    position_contracts: Decimal,
    min_contracts: Decimal,
    contract_precision: Decimal,
    partial_tp_ratio: Decimal,
    fast_ratio_of_bucket: Decimal,
) -> MiddleBucketSplitSizeCheck:
    """Pre-check Middle Runner middle bucket split sub-leg sizes.

    Uses EXACTLY the same rounding rules as
    :func:`src.execution.order_specs.build_take_profit_order_specs`.

    Returns ``ok=True`` only when both fast_contracts and slow_contracts
    are >= min_contracts.
    """
    _rnd = lambda c: round_contracts_down(contracts=c, contract_precision=contract_precision)

    if partial_tp_ratio <= 0 or fast_ratio_of_bucket <= 0 or fast_ratio_of_bucket >= 1:
        return MiddleBucketSplitSizeCheck(
            ok=False,
            reason="invalid_ratios",
            tp1_total_contracts=Decimal("0"),
            fast_contracts=Decimal("0"),
            slow_contracts=Decimal("0"),
            min_contracts=min_contracts,
        )

    partial_total = _rnd(position_contracts * partial_tp_ratio)
    fast_contracts = _rnd(partial_total * fast_ratio_of_bucket)
    slow_contracts = partial_total - fast_contracts

    ok = fast_contracts >= min_contracts and slow_contracts >= min_contracts
    reason = "ok" if ok else "subleg_too_small"

    return MiddleBucketSplitSizeCheck(
        ok=ok,
        reason=reason,
        tp1_total_contracts=partial_total,
        fast_contracts=fast_contracts,
        slow_contracts=slow_contracts,
        min_contracts=min_contracts,
    )
