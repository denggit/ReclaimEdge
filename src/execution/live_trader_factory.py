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
        - ``EXCHANGE=binance`` → raises ``RuntimeError`` (blocked by build).

    Raises
    ------
    RuntimeError
        When ``EXCHANGE=binance`` — Binance live trading is not wired yet.
    ValueError
        When ``EXCHANGE`` is set to an unsupported value.
    """
    values = os.environ if env is None else env
    exchange = values.get("EXCHANGE", "okx").strip().lower() or "okx"

    if exchange == "okx":
        return Trader()

    if exchange == "binance":
        from src.live.binance_live_preflight import (
            build_binance_live_preflight_report,
            format_binance_live_blocked_message,
        )

        report = build_binance_live_preflight_report(
            values,
            orders_globally_enabled=False,
        )
        raise RuntimeError(format_binance_live_blocked_message(report))

    raise ValueError(
        f"Unsupported exchange: {exchange!r}. Supported values are 'okx' and 'binance'."
    )
