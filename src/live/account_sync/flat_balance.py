from __future__ import annotations

import asyncio

from src.execution.trader import PositionSnapshot, Trader
from src.live.runtime_types import SettledFlatBalance
from src.utils.log import get_logger

logger = get_logger(__name__)


async def fetch_usdt_cash_balance(trader: Trader) -> float:
    """Fetch USDT *cash* balance (cashBal / availBal) from OKX.

    Tries, in order:
    1. A ``request()`` callable on the trader (backward-compat for test fakes).
    2. The private REST client wired into the bound trading client.
    3. ``trader.fetch_usdt_equity()`` as a final fallback.
    """
    # 1. Backward compat: test FakeTraders often expose request()
    req = getattr(trader, "request", None)
    if callable(req):
        res = await req("GET", "/api/v5/account/balance?ccy=USDT")
        data = res.get("data", [])
        if data:
            for item in data[0].get("details", []):
                if item.get("ccy") == "USDT":
                    return float(
                        item.get("cashBal")
                        or item.get("availBal")
                        or item.get("availEq")
                        or item.get("eq")
                        or 0.0
                    )
            return float(data[0].get("totalEq") or 0.0)
        return 0.0

    # 2. Production path: use trading client's private REST client
    trading_client = getattr(trader, "trading_client", None)
    if trading_client is not None:
        private_client = getattr(trading_client, "_client", None)
        if private_client is not None:
            res = await private_client.request(
                "GET", "/api/v5/account/balance?ccy=USDT"
            )
            data = res.get("data", [])
            if not data:
                return 0.0
            for item in data[0].get("details", []):
                if item.get("ccy") == "USDT":
                    return float(
                        item.get("cashBal")
                        or item.get("availBal")
                        or item.get("availEq")
                        or item.get("eq")
                        or 0.0
                    )
            return float(data[0].get("totalEq") or 0.0)

    # 3. Final fallback
    return await trader.fetch_usdt_equity()


async def fetch_settled_flat_balance(
        trader: Trader,
        *,
        attempts: int,
        interval_seconds: float,
        stable_delta_usdt: float,
        cash_equity_max_diff_usdt: float,
) -> SettledFlatBalance:
    attempts = max(int(attempts), 1)
    previous_flat_cash: float | None = None
    last_cash: float | None = None
    last_equity: float | None = None
    last_position: PositionSnapshot | None = None
    last_attempt = 0
    for attempt in range(1, attempts + 1):
        last_attempt = attempt
        try:
            position = await trader.fetch_position_snapshot()
            cash = await fetch_usdt_cash_balance(trader)
            equity = await trader.fetch_usdt_equity()
        except Exception as exc:
            if last_cash is not None and last_equity is not None:
                return SettledFlatBalance(
                    cash=last_cash,
                    equity=last_equity,
                    attempts=last_attempt,
                    stable=False,
                    reason=f"error_after_last_balance:{type(exc).__name__}:{exc}",
                )
            raise

        last_position = position
        last_cash = cash
        last_equity = equity
        if position.has_position:
            if attempt < attempts and interval_seconds > 0:
                await asyncio.sleep(interval_seconds)
            continue

        cash_equity_stable = abs(cash - equity) <= cash_equity_max_diff_usdt
        cash_repeat_stable = previous_flat_cash is not None and abs(cash - previous_flat_cash) <= stable_delta_usdt
        if cash_equity_stable and cash_repeat_stable:
            return SettledFlatBalance(
                cash=cash,
                equity=equity,
                attempts=attempt,
                stable=True,
                reason="cash_equity_stable",
            )
        previous_flat_cash = cash
        if attempt < attempts and interval_seconds > 0:
            await asyncio.sleep(interval_seconds)

    if last_cash is None or last_equity is None:
        raise RuntimeError("flat balance settlement finished without any balance sample")
    if last_position is not None and not last_position.has_position:
        return SettledFlatBalance(
            cash=last_equity,
            equity=last_equity,
            attempts=attempts,
            stable=False,
            reason="fallback_to_equity_after_timeout",
        )
    return SettledFlatBalance(
        cash=last_cash,
        equity=last_equity,
        attempts=attempts,
        stable=False,
        reason="position_not_flat_after_timeout",
    )
