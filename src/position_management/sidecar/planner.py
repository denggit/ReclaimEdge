from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_DOWN

from src.position_management.sidecar.model import (
    PositionSide,
    calculate_sidecar_margin,
    calculate_sidecar_qty,
    calculate_sidecar_tp_price,
    sanitize_okx_client_order_id,
)
from src.risk.simple_position_sizer import PositionSize
from src.strategies.boll_cvd_reclaim_strategy import TradeIntent


@dataclass(frozen=True)
class SidecarExecutionPlan:
    enabled: bool
    side: PositionSide
    layer_index: int
    core_qty: float
    core_contracts: Decimal
    sidecar_qty: float
    sidecar_contracts: Decimal
    total_qty: float
    total_contracts: Decimal
    sidecar_tp_price: float
    sidecar_margin_pct: float
    layer_multiplier: float
    client_order_id: str | None


@dataclass(frozen=True)
class CombinedEntryIntentPlan:
    execution_intent: TradeIntent
    sidecar_plan: SidecarExecutionPlan | None


def build_sidecar_execution_plan(
    *,
    enabled: bool,
    side: PositionSide,
    layer_index: int,
    core_qty: float,
    entry_price: float,
    account_equity_usdt: float,
    leverage: float,
    layer_margin_pct: float,
    sidecar_margin_pct: float,
    sidecar_tp_pct: float,
    layer_multiplier: float,
    position_id: str | None,
    ts_ms: int,
    contract_multiplier: Decimal | str | float = Decimal("0.1"),
    contract_precision: Decimal | str | float = Decimal("0.01"),
) -> SidecarExecutionPlan | None:
    if not enabled or sidecar_margin_pct <= 0 or entry_price <= 0:
        return None
    multiplier = Decimal(str(contract_multiplier))
    precision = Decimal(str(contract_precision))
    core_contracts = _qty_to_contracts(core_qty, multiplier, precision)
    sidecar_qty = calculate_sidecar_qty(
        account_equity_usdt=account_equity_usdt,
        price=entry_price,
        leverage=leverage,
        layer_margin_pct=layer_margin_pct,
        sidecar_margin_pct=sidecar_margin_pct,
        layer_multiplier=layer_multiplier,
    )
    sidecar_contracts = _qty_to_contracts(sidecar_qty, multiplier, precision)
    if core_contracts <= 0 or sidecar_contracts <= 0:
        return None
    total_contracts = core_contracts + sidecar_contracts
    total_qty = float(total_contracts * multiplier)
    return SidecarExecutionPlan(
        enabled=True,
        side=side,
        layer_index=int(layer_index),
        core_qty=float(core_contracts * multiplier),
        core_contracts=core_contracts,
        sidecar_qty=float(sidecar_contracts * multiplier),
        sidecar_contracts=sidecar_contracts,
        total_qty=total_qty,
        total_contracts=total_contracts,
        sidecar_tp_price=calculate_sidecar_tp_price(side, entry_price, sidecar_tp_pct),
        sidecar_margin_pct=calculate_sidecar_margin(layer_margin_pct, sidecar_margin_pct, layer_multiplier),
        layer_multiplier=float(layer_multiplier),
        client_order_id=sidecar_client_order_id(position_id, layer_index, ts_ms),
    )


def build_combined_entry_intent(
    *,
    intent: TradeIntent,
    sidecar_enabled: bool,
    account_equity_usdt: float,
    leverage: float,
    sidecar_margin_pct: float,
    sidecar_tp_pct: float,
    position_id: str | None,
    contract_multiplier: Decimal | str | float = Decimal("0.1"),
    contract_precision: Decimal | str | float = Decimal("0.01"),
) -> CombinedEntryIntentPlan:
    if intent.intent_type not in {"OPEN_LONG", "OPEN_SHORT", "ADD_LONG", "ADD_SHORT"}:
        return CombinedEntryIntentPlan(execution_intent=intent, sidecar_plan=None)
    sidecar_plan = build_sidecar_execution_plan(
        enabled=sidecar_enabled,
        side=intent.side,
        layer_index=intent.layer_index,
        core_qty=float(intent.size.eth_qty),
        entry_price=float(intent.price),
        account_equity_usdt=account_equity_usdt,
        leverage=leverage,
        layer_margin_pct=float(getattr(intent.size, "margin_usdt", 0.0) or 0.0),
        sidecar_margin_pct=sidecar_margin_pct,
        sidecar_tp_pct=sidecar_tp_pct,
        layer_multiplier=float(intent.size.layer_multiplier or 1.0),
        position_id=position_id,
        ts_ms=int(intent.ts_ms),
        contract_multiplier=contract_multiplier,
        contract_precision=contract_precision,
    )
    if sidecar_plan is None:
        return CombinedEntryIntentPlan(execution_intent=intent, sidecar_plan=None)
    total_size = replace(
        intent.size,
        eth_qty=sidecar_plan.total_qty,
        notional_usdt=sidecar_plan.total_qty * float(intent.price),
    )
    execution_intent = replace(
        intent,
        size=total_size,
        managed_core_contracts=str(sidecar_plan.core_contracts),
        managed_core_eth_qty=sidecar_plan.core_qty,
    )
    return CombinedEntryIntentPlan(execution_intent=execution_intent, sidecar_plan=sidecar_plan)


def sidecar_client_order_id(position_id: str | None, layer_index: int, ts_ms: int) -> str:
    digest = hashlib.sha1(str(position_id or "unknown").encode("utf-8")).hexdigest()[:10]
    raw = f"SC{digest}L{int(layer_index)}T{int(ts_ms) % 100000}"
    return sanitize_okx_client_order_id(raw)


def _qty_to_contracts(qty: float, contract_multiplier: Decimal, contract_precision: Decimal) -> Decimal:
    if qty <= 0 or contract_multiplier <= 0:
        return Decimal("0")
    contracts = (Decimal(str(qty)) + Decimal("1e-12")) / contract_multiplier
    return contracts.quantize(contract_precision, rounding=ROUND_DOWN)
