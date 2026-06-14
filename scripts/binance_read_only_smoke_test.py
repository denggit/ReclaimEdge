#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : binance_read_only_smoke_test.py
@Description: Binance signed read-only smoke test — only reads position and
              open orders, NEVER places / cancels orders or changes settings.

WARNING: This script READS Binance USD-M Futures private account data
(position and open orders).  It does NOT place orders, cancel orders,
change leverage, or modify any account setting.

This script requires an explicit environment variable to confirm intent.

Prerequisites (account-level, must be configured BEFORE running):
    MARGIN_MODE   = isolated
    POSITION_MODE = net    (Binance account must be in One-way Mode)

Unified config env:
    EXCHANGE=binance
    TRADE_ASSET=ETH
    QUOTE_ASSET=USDT
    MARKET_TYPE=PERPETUAL
    MARGIN_MODE=isolated
    POSITION_MODE=net
    LEVERAGE=20
    KLINE_INTERVAL=15m

Usage::

    EXCHANGE=binance                                    \\
    EXCHANGE_API_KEY=...                                \\
    EXCHANGE_API_SECRET=...                             \\
    BINANCE_READ_ONLY_SMOKE_CONFIRM=I_UNDERSTAND_THIS_READS_BINANCE_PRIVATE_ACCOUNT \\
    python scripts/binance_read_only_smoke_test.py

No strategy imports.  No execution imports.  No live runtime imports.
No order placement.  No order cancellation.  No leverage changes.
"""

from __future__ import annotations

import asyncio
import os
import sys

import aiohttp

from src.exchanges.binance.aiohttp_transport import AiohttpBinanceTransport
from src.exchanges.binance.client import BinanceBrokerClient
from src.exchanges.models import ExchangeName
from src.exchanges.runtime_config import (
    ExchangeRuntimeConfig,
    load_unified_runtime_config,
    SUPPORTED_CANONICAL_SYMBOL,
    SUPPORTED_KLINE_INTERVAL,
    SUPPORTED_MARGIN_MODE,
    SUPPORTED_POSITION_MODE,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANONICAL_SYMBOL: str = "ETH-USDT-PERP"
BINANCE_SYMBOL: str = "ETHUSDT"

READ_ONLY_CONFIRM_ENV: str = "BINANCE_READ_ONLY_SMOKE_CONFIRM"
READ_ONLY_CONFIRM_VALUE: str = "I_UNDERSTAND_THIS_READS_BINANCE_PRIVATE_ACCOUNT"


# ---------------------------------------------------------------------------
# Safety gates (no network)
# ---------------------------------------------------------------------------


def require_read_only_confirmation() -> None:
    """Raise ``SystemExit`` unless the read-only confirm env var is set correctly.

    This gate is intentionally distinct from the live-order smoke test
    confirmation.  Even though this script only reads, it accesses private
    account data, so an explicit opt-in is required.
    """
    value = os.environ.get(READ_ONLY_CONFIRM_ENV, "")
    if value != READ_ONLY_CONFIRM_VALUE:
        print(
            "ERROR: This script reads Binance USD-M Futures private account data.\n"
            f"Set {READ_ONLY_CONFIRM_ENV}={READ_ONLY_CONFIRM_VALUE} to continue.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print("[read-only-smoke] read-only confirmation OK")


def validate_unified_config_for_binance_read_only(
    rt: ExchangeRuntimeConfig,
) -> str:
    """Validate the unified runtime config for read-only Binance smoke use.

    Returns the validated Binance raw symbol (ETHUSDT).

    Raises ``SystemExit`` on any validation failure.
    """
    if rt.exchange != ExchangeName.BINANCE:
        print(
            f"ERROR: EXCHANGE must be 'binance', got {rt.exchange.value!r}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if rt.canonical_symbol != SUPPORTED_CANONICAL_SYMBOL:
        print(
            f"ERROR: canonical_symbol must be {SUPPORTED_CANONICAL_SYMBOL!r}, "
            f"got {rt.canonical_symbol!r}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if rt.binance_symbol != BINANCE_SYMBOL:
        print(
            f"ERROR: binance_symbol must be {BINANCE_SYMBOL!r}, "
            f"got {rt.binance_symbol!r}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if rt.position_mode != SUPPORTED_POSITION_MODE:
        print(
            f"ERROR: POSITION_MODE must be {SUPPORTED_POSITION_MODE!r}, "
            f"got {rt.position_mode!r}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if rt.margin_mode != SUPPORTED_MARGIN_MODE:
        print(
            f"ERROR: MARGIN_MODE must be {SUPPORTED_MARGIN_MODE!r}, "
            f"got {rt.margin_mode!r}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if rt.kline_interval != SUPPORTED_KLINE_INTERVAL:
        print(
            f"ERROR: KLINE_INTERVAL must be {SUPPORTED_KLINE_INTERVAL!r}, "
            f"got {rt.kline_interval!r}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print("[read-only-smoke] unified config validated OK")
    return rt.binance_symbol


# ---------------------------------------------------------------------------
# Credential loading
# ---------------------------------------------------------------------------


def load_binance_read_only_credentials() -> tuple[str, str]:
    """Return (api_key, api_secret) from environment, or exit.

    Does NOT print the actual credential values — only confirms presence.
    """
    api_key = os.environ.get("EXCHANGE_API_KEY", "").strip()
    api_secret = os.environ.get("EXCHANGE_API_SECRET", "").strip()
    if not api_key or not api_secret:
        print(
            "ERROR: EXCHANGE_API_KEY / EXCHANGE_API_SECRET must be set",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print("[read-only-smoke] API credentials present")
    return api_key, api_secret


# ---------------------------------------------------------------------------
# Read-only runner (testable with a fake client)
# ---------------------------------------------------------------------------


async def run_read_only_smoke(
    *,
    client: BinanceBrokerClient,
    symbol: str,
) -> None:
    """Execute the read-only smoke sequence using *client*.

    Only calls ``fetch_position`` and ``fetch_open_orders`` — no writes
    of any kind.
    """
    # --- Position ---
    position = await client.fetch_position(symbol)
    has_position = position is not None
    print(
        f"[read-only-smoke] fetch_position OK | has_position={has_position}"
    )

    if position is not None:
        print(
            f"  symbol={position.symbol}"
            f"  side={position.position_side.value}"
            f"  contracts={position.quantity}"
            f"  avg_entry={position.average_entry_price}"
            f"  unrealized_pnl={position.unrealized_pnl}"
        )

    # --- Open orders ---
    open_orders = await client.fetch_open_orders(symbol)
    print(
        f"[read-only-smoke] fetch_open_orders OK | count={len(open_orders)}"
    )

    for o in open_orders:
        print(
            f"  order_id={o.order_id}"
            f"  client_order_id={o.client_order_id}"
            f"  side={o.side.value}"
            f"  type={o.order_type.value}"
            f"  price={o.price}"
            f"  quantity={o.quantity}"
            f"  reduce_only={o.reduce_only}"
        )

    print("[read-only-smoke] done | no orders were placed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    """Entry point — validate, connect, fetch, and exit."""
    require_read_only_confirmation()

    rt = load_unified_runtime_config(os.environ)
    symbol = validate_unified_config_for_binance_read_only(rt)

    api_key, api_secret = load_binance_read_only_credentials()

    async with aiohttp.ClientSession() as session:
        transport = AiohttpBinanceTransport(session=session)
        client = BinanceBrokerClient(
            api_key=api_key,
            api_secret=api_secret,
            transport=transport,
        )
        await run_read_only_smoke(client=client, symbol=symbol)


if __name__ == "__main__":
    asyncio.run(main())
