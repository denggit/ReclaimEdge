"""Middle Bucket Fast Protection — post-fast-TP breakeven protection.

After the BOLL15 fast leg fills, this module decides what protection action
to take on the remaining core position:

  - PLACE_SL:  place a conditional stop-loss at avg_entry_price + fee_buffer
  - MARKET_EXIT:  immediately market-close the remaining core
  - HALT_ONLY:  halt live trading (manual intervention required)
  - KEEP_POSITION:  leave the position unprotected (marked as risky)
  - NOOP:  no action needed (feature disabled)

This module does NO I/O, NO state access, NO OKX calls, NO logging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.strategies.middle_bucket_split import (
    calculate_fast_protective_sl,
    is_stop_valid_for_current_price,
)

PositionSide = Literal["LONG", "SHORT"]
ProtectionAction = Literal["PLACE_SL", "MARKET_EXIT", "HALT_ONLY", "KEEP_POSITION", "NOOP"]


@dataclass(frozen=True)
class FastProtectionDecision:
    """Result of evaluating post-fast-TP protection for remaining core."""
    action: ProtectionAction
    sl_price: float | None
    reason: str


def build_fast_protection_decision(
    *,
    side: PositionSide,
    avg_entry_price: float,
    current_price: float,
    fee_buffer_pct: float,
    invalid_action: str,
    enabled: bool,
) -> FastProtectionDecision:
    """Decide what to do after the fast TP leg has filled.

    Returns a FastProtectionDecision with the recommended action and
    optional SL price.
    """
    if not enabled:
        return FastProtectionDecision(action="NOOP", sl_price=None, reason="disabled")

    if avg_entry_price <= 0.0 or current_price <= 0.0:
        return FastProtectionDecision(
            action="MARKET_EXIT",
            sl_price=None,
            reason="missing_price_or_cost_basis",
        )

    fast_sl = calculate_fast_protective_sl(
        side=side,
        avg_entry_price=avg_entry_price,
        fee_buffer_pct=fee_buffer_pct,
    )

    if fast_sl is None:
        return FastProtectionDecision(
            action="MARKET_EXIT",
            sl_price=None,
            reason="failed_to_calculate_sl",
        )

    if is_stop_valid_for_current_price(
        side=side,
        stop_price=fast_sl,
        current_price=current_price,
    ):
        return FastProtectionDecision(
            action="PLACE_SL",
            sl_price=fast_sl,
            reason="sl_valid",
        )

    # SL is invalid — apply the configured invalid_action
    action_map: dict[str, ProtectionAction] = {
        "MARKET_EXIT": "MARKET_EXIT",
        "HALT_ONLY": "HALT_ONLY",
        "KEEP_POSITION": "KEEP_POSITION",
    }
    action = action_map.get(invalid_action.upper(), "MARKET_EXIT")
    return FastProtectionDecision(
        action=action,
        sl_price=fast_sl,
        reason=f"sl_invalid_{action.lower()}",
    )
