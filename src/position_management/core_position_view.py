from __future__ import annotations

from dataclasses import replace

from src.execution.trader import PositionSnapshot
from src.strategies.boll_cvd_reclaim_strategy import TradeIntent

POSITION_MANAGEMENT_INTENTS = {"UPDATE_TP", "MARKET_EXIT_RUNNER"}


def position_log_key(position: PositionSnapshot) -> tuple[str, str, float]:
    if not position.has_position or position.side is None:
        return ("FLAT", "0", 0.0)
    return (position.side, str(position.contracts), round(position.avg_entry_price, 2))


def apply_core_position_view_to_state(state, core_position: PositionSnapshot) -> None:
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
