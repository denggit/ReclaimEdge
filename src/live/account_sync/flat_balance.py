from __future__ import annotations

import asyncio

from src.execution.trader import PositionSnapshot, Trader
from src.live.runtime_types import SettledFlatBalance
from src.utils.log import get_logger

logger = get_logger(__name__)


async def fetch_usdt_cash_balance(trader: Trader) -> float:
    """Fetch USDT *cash* balance through the bound TradingClientPort.

    Prefers ``trading_client.fetch_balance().available`` (cashBal /
    availBal).  Falls back to ``trader.fetch_usdt_equity()`` when no
    trading client is bound (e.g. in legacy test harnesses).
    """
    trading_client = getattr(trader, "trading_client", None)
    if trading_client is not None:
        balance = await trading_client.fetch_balance()
        if balance.available is not None:
            return float(balance.available)
        raw_cash = balance.raw.get("cash_balance_usdt")
        if raw_cash is not None:
            return float(raw_cash)
        return float(balance.total)

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
