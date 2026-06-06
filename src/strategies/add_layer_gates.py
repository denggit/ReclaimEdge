from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PositionSide = Literal["LONG", "SHORT"]


@dataclass(frozen=True)
class AddTimingDecision:
    ok: bool
    reason: str


@dataclass(frozen=True)
class AddGapDecision:
    ok: bool
    gap_pct: float
    required_price: float


@dataclass(frozen=True)
class AddAvgImprovementDecision:
    ok: bool
    improvement_pct: float
    projected_avg: float


def add_layer_gap_pct_for_target_layer(
        *,
        target_layer: int,
        add_layer_gap_pct: float,
        add_layer_gap_pct_layer_7_8: float,
        add_layer_gap_pct_layer_9_10: float,
        add_layer_gap_pct_layer_11_plus: float,
) -> float:
    if target_layer >= 11:
        return add_layer_gap_pct_layer_11_plus
    if target_layer >= 9:
        return add_layer_gap_pct_layer_9_10
    if target_layer >= 7:
        return add_layer_gap_pct_layer_7_8
    return add_layer_gap_pct


def add_min_interval_bypass_gap_pct_for_target_layer(
        *,
        target_layer: int,
        add_layer_gap_pct: float,
        add_layer_gap_pct_layer_7_8: float,
        add_layer_gap_pct_layer_9_10: float,
        add_layer_gap_pct_layer_11_plus: float,
) -> float:
    return add_layer_gap_pct_for_target_layer(
        target_layer=target_layer,
        add_layer_gap_pct=add_layer_gap_pct,
        add_layer_gap_pct_layer_7_8=add_layer_gap_pct_layer_7_8,
        add_layer_gap_pct_layer_9_10=add_layer_gap_pct_layer_9_10,
        add_layer_gap_pct_layer_11_plus=add_layer_gap_pct_layer_11_plus,
    ) * 2


def add_elapsed_seconds(*, ts_ms: int, last_order_ts_ms: int) -> float:
    return max((ts_ms - last_order_ts_ms) / 1000, 0.0)


def adverse_gap_pct(*, side: PositionSide, price: float, last_entry_price: float | None) -> float:
    if last_entry_price is None or last_entry_price <= 0:
        return 0.0
    if side == "LONG":
        return (last_entry_price - price) / last_entry_price
    return (price - last_entry_price) / last_entry_price


def check_add_gap(
        *,
        side: PositionSide,
        price: float,
        last_entry_price: float | None,
        target_layer: int,
        add_layer_gap_pct: float,
        add_layer_gap_pct_layer_7_8: float,
        add_layer_gap_pct_layer_9_10: float,
        add_layer_gap_pct_layer_11_plus: float,
) -> AddGapDecision:
    gap_pct = add_layer_gap_pct_for_target_layer(
        target_layer=target_layer,
        add_layer_gap_pct=add_layer_gap_pct,
        add_layer_gap_pct_layer_7_8=add_layer_gap_pct_layer_7_8,
        add_layer_gap_pct_layer_9_10=add_layer_gap_pct_layer_9_10,
        add_layer_gap_pct_layer_11_plus=add_layer_gap_pct_layer_11_plus,
    )
    if last_entry_price is None or last_entry_price <= 0:
        return AddGapDecision(False, gap_pct, 0.0)

    if side == "LONG":
        required_price = last_entry_price * (1 - gap_pct)
        return AddGapDecision(price <= required_price, gap_pct, required_price)

    required_price = last_entry_price * (1 + gap_pct)
    return AddGapDecision(price >= required_price, gap_pct, required_price)


def check_base_add_timing(
        *,
        side: PositionSide,
        price: float,
        ts_ms: int,
        target_layer: int,
        layers: int,
        last_entry_price: float | None,
        last_order_ts_ms: int,
        first_add_block_seconds: int,
        add_min_interval_seconds: int,
        add_layer_gap_pct: float,
        add_layer_gap_pct_layer_7_8: float,
        add_layer_gap_pct_layer_9_10: float,
        add_layer_gap_pct_layer_11_plus: float,
) -> AddTimingDecision:
    if last_entry_price is None or last_entry_price <= 0:
        return AddTimingDecision(False, "missing_last_entry")

    elapsed_seconds = add_elapsed_seconds(ts_ms=ts_ms, last_order_ts_ms=last_order_ts_ms)
    if layers == 1:
        if elapsed_seconds < first_add_block_seconds:
            return AddTimingDecision(False, "first_add_block")
        return AddTimingDecision(True, "ok")

    if layers >= 2:
        adverse_gap_pct_val = adverse_gap_pct(side=side, price=price, last_entry_price=last_entry_price)
        bypass_gap_pct = add_min_interval_bypass_gap_pct_for_target_layer(
            target_layer=target_layer,
            add_layer_gap_pct=add_layer_gap_pct,
            add_layer_gap_pct_layer_7_8=add_layer_gap_pct_layer_7_8,
            add_layer_gap_pct_layer_9_10=add_layer_gap_pct_layer_9_10,
            add_layer_gap_pct_layer_11_plus=add_layer_gap_pct_layer_11_plus,
        )
        if elapsed_seconds < add_min_interval_seconds and adverse_gap_pct_val < bypass_gap_pct:
            return AddTimingDecision(False, "add_interval")

    return AddTimingDecision(True, "ok")


def check_add_avg_improvement(
        *,
        side: PositionSide,
        price: float,
        required_improvement_pct: float,
        old_qty: float,
        old_notional: float,
        old_avg: float,
        add_qty: float,
) -> AddAvgImprovementDecision:
    if required_improvement_pct <= 0:
        return AddAvgImprovementDecision(True, 0.0, old_avg)
    if old_qty <= 0 or old_notional <= 0 or old_avg <= 0 or add_qty <= 0:
        return AddAvgImprovementDecision(False, 0.0, old_avg)

    projected_qty = old_qty + add_qty
    projected_notional = old_notional + price * add_qty
    projected_avg = projected_notional / projected_qty
    if side == "LONG":
        improvement_pct = (old_avg - projected_avg) / old_avg
    else:
        improvement_pct = (projected_avg - old_avg) / old_avg
    return AddAvgImprovementDecision(improvement_pct >= required_improvement_pct, improvement_pct, projected_avg)
