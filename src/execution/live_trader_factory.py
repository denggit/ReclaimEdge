from __future__ import annotations

import os
from collections.abc import Mapping

from src.execution.live_trader_protocol import LiveTraderProtocol
from src.execution.trader import Trader


def create_live_trader(
    env: Mapping[str, str] | None = None,
) -> LiveTraderProtocol:
    """Create and return a live trader for the configured exchange.

    Parameters
    ----------
    env:
        Optional mapping of environment variables.  When ``None`` (the
        default) the real ``os.environ`` is used.

    Returns
    -------
    LiveTraderProtocol
        A live trader instance:
        - ``EXCHANGE=okx`` (or unset) → ``Trader`` (OKX).
        - ``EXCHANGE=binance`` → ``BinanceLiveTrader``.

    Raises
    ------
    ValueError
        When ``EXCHANGE`` is set to an unsupported value.
    """
    values = os.environ if env is None else env
    exchange = values.get("EXCHANGE", "okx").strip().lower() or "okx"

    if exchange == "okx":
        return Trader()

    if exchange == "binance":
        from src.execution.binance_live_trader import BinanceLiveTrader

        return BinanceLiveTrader(env=values)

    raise ValueError(
        f"Unsupported exchange: {exchange!r}. Supported values are 'okx' and 'binance'."
    )
