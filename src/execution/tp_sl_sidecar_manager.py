"""SidecarTpManager — stub retained for execution facade compatibility.

Sidecar runtime has been removed.  All sidecar TP/SL methods are no-ops
that return safe defaults.  This file is kept only so that
TpSlExecutionManager does not break on import.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any


class SidecarTpManager:
    """Stub: all sidecar operations are no-ops. Sidecar runtime has been removed."""

    def __init__(self, trader: Any, trading_client: Any) -> None:
        self._trader = trader
        self._tcp = trading_client

    async def place_sidecar_fixed_take_profit(
        self,
        *,
        side: str,
        contracts: Decimal | str,
        tp_price: float,
        client_order_id: str | None = None,
    ) -> str:
        """No-op stub — sidecar runtime removed."""
        return ""

    async def cancel_sidecar_take_profit(self, order_id: str | None) -> bool:
        """No-op stub — sidecar runtime removed."""
        return True

    async def place_sidecar_market_close(
        self,
        *,
        side: str,
        contracts: Decimal | str,
    ) -> dict[str, Any]:
        """No-op stub — sidecar runtime removed."""
        return {"order_id": "", "status": "filled"}

    async def fetch_sidecar_order_status(self, order_id: str) -> dict[str, Any]:
        """No-op stub — sidecar runtime removed."""
        return {"status": "FILLED"}
