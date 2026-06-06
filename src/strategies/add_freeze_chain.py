from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.strategies.add_layer_gates import PositionSide, add_elapsed_seconds, adverse_gap_pct


@dataclass(frozen=True)
class ShockAddTimingDecision:
    ok: bool
    reason: str
    adverse_gap_pct: float = 0.0
    required_gap_pct: float = 0.0
    multiplier: float = 0.0
    first_elapsed_seconds: float = 0.0
    freeze_remaining_seconds: float = 0.0


@dataclass(frozen=True)
class AddFreezeStartDecision:
    enabled: bool
    freeze_until_ts_ms: int
    penalty_count: int


@dataclass(frozen=True)
class AddFreezeExtensionDecision:
    changed: bool
    freeze_until_ts_ms: int
    penalty_count: int
    extension_seconds: int


AddFreezeSkipLogKey = tuple[str, int, int, int, float]


def add_freeze_active(
    *,
    add_freeze_chain_enabled: bool,
    add_freeze_until_ts_ms: int,
    ts_ms: int,
) -> bool:
    return bool(
        add_freeze_chain_enabled
        and int(add_freeze_until_ts_ms or 0) > ts_ms
    )


def add_freeze_remaining_seconds(
    *,
    add_freeze_until_ts_ms: int,
    ts_ms: int,
) -> float:
    until = int(add_freeze_until_ts_ms or 0)
    return max((until - ts_ms) / 1000, 0.0)


def should_reset_add_freeze_if_expired(
    *,
    add_freeze_until_ts_ms: int,
    ts_ms: int,
) -> bool:
    return int(add_freeze_until_ts_ms or 0) <= ts_ms


def active_add_freeze_bypass_multiplier(
    *,
    layers: int,
    penalty_count: int,
    first_add_block_bypass_multiplier: float,
    add_min_interval_bypass_multiplier: float,
) -> float:
    penalty = int(penalty_count or 0)
    if layers == 1 and penalty <= 0:
        return first_add_block_bypass_multiplier
    return float(add_min_interval_bypass_multiplier) + penalty


def first_entry_elapsed_seconds(
    *,
    ts_ms: int,
    first_entry_ts_ms: int,
    last_order_ts_ms: int,
) -> float:
    first = int(first_entry_ts_ms or 0)
    if first <= 0:
        first = int(last_order_ts_ms or 0)
    return max((ts_ms - first) / 1000, 0.0)


def first_add_block_required_gap_pct(
    *,
    target_layer_gap_pct: float,
    first_add_block_bypass_multiplier: float,
) -> float:
    return target_layer_gap_pct * first_add_block_bypass_multiplier


def check_shock_add_timing(
    *,
    side: PositionSide,
    price: float,
    ts_ms: int,
    target_layer: int,
    layers: int,
    last_entry_price: float | None,
    last_order_ts_ms: int,
    first_entry_ts_ms: int,
    add_freeze_chain_enabled: bool,
    add_freeze_until_ts_ms: int,
    add_freeze_penalty_count: int,
    first_add_block_seconds: int,
    add_min_interval_seconds: int,
    add_min_interval_bypass_multiplier: float,
    first_add_block_bypass_multiplier: float,
    target_layer_gap_pct: float,
) -> ShockAddTimingDecision:
    if last_entry_price is None or last_entry_price <= 0:
        return ShockAddTimingDecision(ok=False, reason="missing_last_entry")

    if not add_freeze_chain_enabled:
        first_elapsed = first_entry_elapsed_seconds(
            ts_ms=ts_ms,
            first_entry_ts_ms=first_entry_ts_ms,
            last_order_ts_ms=last_order_ts_ms,
        )
        if layers >= 1 and first_elapsed < first_add_block_seconds:
            adverse = adverse_gap_pct(side=side, price=price, last_entry_price=last_entry_price)
            required = target_layer_gap_pct * first_add_block_bypass_multiplier
            if adverse < required:
                return ShockAddTimingDecision(
                    ok=False, reason="first_add_block",
                    adverse_gap_pct=adverse,
                    required_gap_pct=required,
                    multiplier=first_add_block_bypass_multiplier,
                    first_elapsed_seconds=first_elapsed,
                )
            return ShockAddTimingDecision(
                ok=True, reason="first_add_block_bypassed",
                adverse_gap_pct=adverse,
                required_gap_pct=required,
                multiplier=first_add_block_bypass_multiplier,
                first_elapsed_seconds=first_elapsed,
            )
        if layers == 1:
            return ShockAddTimingDecision(ok=True, reason="ok")

    # Caller is expected to have already reset expired freeze before calling
    # this function when add_freeze_chain_enabled=True.
    active = add_freeze_active(
        add_freeze_chain_enabled=add_freeze_chain_enabled,
        add_freeze_until_ts_ms=add_freeze_until_ts_ms,
        ts_ms=ts_ms,
    )
    if active:
        adverse = adverse_gap_pct(side=side, price=price, last_entry_price=last_entry_price)
        multiplier = active_add_freeze_bypass_multiplier(
            layers=layers,
            penalty_count=add_freeze_penalty_count,
            first_add_block_bypass_multiplier=first_add_block_bypass_multiplier,
            add_min_interval_bypass_multiplier=add_min_interval_bypass_multiplier,
        )
        required = target_layer_gap_pct * multiplier
        freeze_remaining = add_freeze_remaining_seconds(
            add_freeze_until_ts_ms=add_freeze_until_ts_ms, ts_ms=ts_ms,
        )
        if adverse < required:
            return ShockAddTimingDecision(
                ok=False, reason="add_freeze",
                adverse_gap_pct=adverse,
                required_gap_pct=required,
                multiplier=multiplier,
                freeze_remaining_seconds=freeze_remaining,
            )
        if layers == 1 and int(add_freeze_penalty_count or 0) <= 0:
            return ShockAddTimingDecision(
                ok=True, reason="first_add_block_bypassed",
                adverse_gap_pct=adverse,
                required_gap_pct=required,
                multiplier=multiplier,
                freeze_remaining_seconds=freeze_remaining,
            )
        return ShockAddTimingDecision(
            ok=True, reason="add_freeze_bypassed",
            adverse_gap_pct=adverse,
            required_gap_pct=required,
            multiplier=multiplier,
            freeze_remaining_seconds=freeze_remaining,
        )

    if layers == 1:
        return ShockAddTimingDecision(ok=True, reason="ok")

    if layers >= 2:
        elapsed_seconds = add_elapsed_seconds(ts_ms=ts_ms, last_order_ts_ms=last_order_ts_ms)
        adverse = adverse_gap_pct(side=side, price=price, last_entry_price=last_entry_price)
        bypass_gap = target_layer_gap_pct * add_min_interval_bypass_multiplier
        if elapsed_seconds < add_min_interval_seconds and adverse < bypass_gap:
            return ShockAddTimingDecision(
                ok=False, reason="add_interval",
                adverse_gap_pct=adverse,
                required_gap_pct=bypass_gap,
                multiplier=add_min_interval_bypass_multiplier,
                first_elapsed_seconds=elapsed_seconds,
            )

    return ShockAddTimingDecision(ok=True, reason="ok")


def start_add_freeze_after_first_entry(
    *,
    ts_ms: int,
    add_freeze_chain_enabled: bool,
    first_add_block_seconds: int,
) -> AddFreezeStartDecision:
    if not add_freeze_chain_enabled:
        return AddFreezeStartDecision(enabled=False, freeze_until_ts_ms=0, penalty_count=0)
    return AddFreezeStartDecision(
        enabled=True,
        freeze_until_ts_ms=ts_ms + int(first_add_block_seconds * 1000),
        penalty_count=0,
    )


def extend_add_freeze_after_successful_add(
    *,
    ts_ms: int,
    add_freeze_chain_enabled: bool,
    add_min_interval_seconds: int,
    add_freeze_until_ts_ms: int,
    add_freeze_penalty_count: int,
    was_active_freeze: bool,
) -> AddFreezeExtensionDecision:
    if not add_freeze_chain_enabled:
        return AddFreezeExtensionDecision(
            changed=False, freeze_until_ts_ms=0, penalty_count=0, extension_seconds=0,
        )
    extension_ms = int(add_min_interval_seconds * 1000)
    if extension_ms <= 0:
        return AddFreezeExtensionDecision(
            changed=False, freeze_until_ts_ms=0, penalty_count=0, extension_seconds=0,
        )
    if was_active_freeze:
        base_until = max(int(add_freeze_until_ts_ms or 0), ts_ms)
        freeze_until_ts_ms = base_until + extension_ms
        penalty_count = int(add_freeze_penalty_count or 0) + 1
    else:
        freeze_until_ts_ms = ts_ms + extension_ms
        penalty_count = 0
    return AddFreezeExtensionDecision(
        changed=True,
        freeze_until_ts_ms=freeze_until_ts_ms,
        penalty_count=penalty_count,
        extension_seconds=add_min_interval_seconds,
    )


def add_freeze_skip_log_key(
    *,
    side: PositionSide,
    layers: int,
    target_layer: int,
    penalty_count: int,
    multiplier: float,
) -> AddFreezeSkipLogKey:
    return (
        side,
        int(layers),
        int(target_layer),
        int(penalty_count or 0),
        round(float(multiplier), 6),
    )


def should_emit_add_freeze_skip_log(
    *,
    last_key: AddFreezeSkipLogKey | None,
    current_key: AddFreezeSkipLogKey,
    last_ts_ms: int,
    ts_ms: int,
    interval_ms: int,
) -> bool:
    if last_key == current_key and ts_ms - last_ts_ms < interval_ms:
        return False
    return True
