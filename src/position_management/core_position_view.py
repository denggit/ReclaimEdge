from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

from src.execution.trader import PositionSnapshot, Trader
from src.position_management.sidecar.model import sidecar_open_contracts, sidecar_open_qty
from src.position_management.sidecar.reconciler import build_core_position_view
from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState, TradeIntent

POSITION_MANAGEMENT_INTENTS = {"UPDATE_TP", "NEAR_TP_REDUCE", "MARKET_EXIT_RUNNER"}
ENTRY_ADD_INTENTS = {"OPEN_LONG", "OPEN_SHORT", "ADD_LONG", "ADD_SHORT"}


def position_log_key(position: PositionSnapshot) -> tuple[str, str, float]:
    if not position.has_position or position.side is None:
        return ("FLAT", "0", 0.0)
    return (position.side, str(position.contracts), round(position.avg_entry_price, 2))


def apply_core_position_view_to_state(state: StrategyPositionState, core_position: PositionSnapshot) -> None:
    if core_position.has_position:
        state.core_contracts = str(core_position.contracts)
        state.core_eth_qty = float(core_position.eth_qty)
    else:
        state.core_contracts = None
        state.core_eth_qty = 0.0


def with_runtime_managed_core(intent: TradeIntent, account_position: PositionSnapshot | None) -> TradeIntent:
    if not getattr(intent, "managed_core_contracts",
                   None) and account_position is not None and account_position.has_position:
        if intent.intent_type in POSITION_MANAGEMENT_INTENTS:
            return replace(
                intent,
                managed_core_contracts=str(account_position.contracts),
                managed_core_eth_qty=float(account_position.eth_qty),
            )
    return intent


def with_entry_add_managed_core_contracts(
        *,
        intent: TradeIntent,
        strategy_state: StrategyPositionState,
        account_core_position: PositionSnapshot | None,
        trader: Trader,
) -> TradeIntent:
    if not strategy_state.sidecar_enabled_for_position:
        return intent
    if intent.intent_type not in ENTRY_ADD_INTENTS:
        return intent
    if intent.managed_core_contracts:
        return intent

    current_core_contracts = Decimal("0")
    current_core_eth_qty = 0.0

    if account_core_position is not None and account_core_position.has_position and account_core_position.side == intent.side:
        current_core_contracts = account_core_position.contracts
        current_core_eth_qty = account_core_position.eth_qty

    new_core_contracts = trader.eth_qty_to_contracts(Decimal(str(intent.size.eth_qty)))
    expected_core_contracts = current_core_contracts + new_core_contracts

    return replace(
        intent,
        managed_core_contracts=str(expected_core_contracts),
        managed_core_eth_qty=current_core_eth_qty + float(intent.size.eth_qty),
    )


def sidecar_position_mismatch(
        okx_position: PositionSnapshot,
        state: StrategyPositionState,
        tolerance_qty: float = 0.000001,
) -> bool:
    if not getattr(state, "sidecar_enabled_for_position", False):
        return False
    open_qty = sidecar_open_qty(list(getattr(state, "sidecar_legs", []) or []))
    if open_qty <= 0:
        return False
    if not okx_position.has_position or okx_position.side != state.side:
        return True
    if open_qty - float(okx_position.eth_qty) > tolerance_qty:
        return True
    core_position = build_core_position_view(okx_position, open_qty, sidecar_open_contracts(state.sidecar_legs))
    return abs((core_position.eth_qty + open_qty) - okx_position.eth_qty) > tolerance_qty
