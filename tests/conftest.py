from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT_STR = str(_PROJECT_ROOT)

if _PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT_STR)


# ======================================================================
# Shared test helpers for faking OkxPrivateClient in Trader tests
# ======================================================================


class FakeOkxClient:
    """Fake OkxPrivateClient for use in tests that create Trader via __new__.

    Provides a ``request`` method that returns canned responses.
    When no canned responses are available, attempts to use the trader's
    monkeypatched methods as a fallback for position/balance queries.
    Does NOT create any real network connections.
    """

    def __init__(self, trader: object | None = None) -> None:
        self.request_calls: list[tuple[str, str, object]] = []
        self.request_responses: list[dict] = []
        self._trader = trader
        self._in_fallback: bool = False  # recursion guard

    async def request(
        self, method: str, endpoint: str, payload: object = None
    ) -> dict:
        self.request_calls.append((method, endpoint, payload))
        if self.request_responses:
            return self.request_responses.pop(0)

        # Fallback: try to get data from the trader's monkeypatched methods
        # Only when NOT already in a fallback (recursion guard).
        if self._trader is not None and not self._in_fallback:
            # Position queries
            if "/api/v5/account/positions" in endpoint:
                self._in_fallback = True
                try:
                    pos = await self._trader.fetch_position_snapshot()
                    if pos is not None and getattr(pos, "has_position", False):
                        side = getattr(pos, "side", None)
                        raw_pos = getattr(pos, "raw_pos", getattr(pos, "contracts", 0))
                        qty = abs(raw_pos) if hasattr(raw_pos, '__abs__') else raw_pos
                        # OKX uses positive pos for LONG, negative for SHORT
                        pos_str = str(-qty) if side == "SHORT" else str(qty)
                        return {
                            "code": "0",
                            "msg": "",
                            "data": [{
                                "instId": getattr(self._trader, "symbol", "ETH-USDT-SWAP"),
                                "pos": pos_str,
                                "avgPx": str(getattr(pos, "avg_entry_price", 0.0)),
                            }],
                        }
                    return {"code": "0", "msg": "", "data": []}
                except Exception:
                    return {"code": "0", "msg": "", "data": []}
                finally:
                    self._in_fallback = False

            # Balance queries
            if "/api/v5/account/balance" in endpoint:
                self._in_fallback = True
                try:
                    equity = await self._trader.fetch_usdt_equity()
                    return {
                        "code": "0",
                        "msg": "",
                        "data": [{"totalEq": str(equity), "details": [
                            {"ccy": "USDT", "eq": str(equity)}]}],
                    }
                except Exception:
                    return {"code": "0", "msg": "", "data": [{"totalEq": "0"}]}
                finally:
                    self._in_fallback = False

            # Algo orders queries
            if "/api/v5/trade/orders-algo-pending" in endpoint:
                self._in_fallback = True
                try:
                    raw_orders = await self._trader.fetch_pending_algo_orders()
                    return {"code": "0", "msg": "", "data": raw_orders}
                except Exception:
                    return {"code": "0", "msg": "", "data": []}
                finally:
                    self._in_fallback = False

        # Default empty success response
        return {"code": "0", "msg": "", "data": []}

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass
