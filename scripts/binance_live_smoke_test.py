#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : binance_live_smoke_test.py
@Description: Binance ETHUSDT live smoke test — places REAL USD-M Futures orders.

WARNING: This script places REAL orders on Binance USD-M Futures.
It requires an explicit environment variable to confirm intent.

This script reads the unified runtime config (load_unified_runtime_config)
and validates that all platform-agnostic parameters match the supported
values.  It does NOT read any OKX-specific legacy env vars.

Prerequisites (account-level, must be configured BEFORE running):
    MARGIN_MODE   = isolated
    POSITION_MODE = net    (Binance account must be in One-way Mode, NOT Hedge Mode)
    SYMBOL        = ETHUSDT (canonical: ETH-USDT-PERP)

Unified config env (shared with OKX):
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
    BINANCE_LIVE_SMOKE_TEST_CONFIRM=I_UNDERSTAND_THIS_PLACES_REAL_ORDERS \\
    python scripts/binance_live_smoke_test.py

Trade flow (sequential, fail-fast with cleanup on error):
    0. preflight — unified config, position mode, exchangeInfo, mark price, balance, qty
    1. open ETHUSDT LONG (MARKET)
    2. place TP (LIMIT SELL)
    3. place SL (STOP_MARKET SELL)
    4. fetch open orders → confirm TP + SL visible
    5. cancel TP / SL
    6. market close (MARKET SELL)
    7. check position == 0
    8. final cleanup (best-effort cancel + close any residual)

No strategy imports.  No CVD.  No live main loop.
Does NOT read OKX_INST_ID, OKX_BAR, OKX_TD_MODE, or OKX_POS_SIDE_MODE.
KLINE_INTERVAL is consumed via unified config only.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import dataclass
from decimal import ROUND_UP, Decimal, InvalidOperation
from typing import Any

import aiohttp

from src.exchanges.binance.aiohttp_transport import AiohttpBinanceTransport
from src.exchanges.binance.client import BinanceBrokerClient
from src.exchanges.binance.signing import (
    BINANCE_USDM_BASE_URL,
    build_signed_request,
)
from src.exchanges.binance.transport import BinanceTransportResponse
from src.exchanges.runtime_config import (
    ExchangeRuntimeConfig,
    load_unified_runtime_config,
    SUPPORTED_CANONICAL_SYMBOL,
    SUPPORTED_KLINE_INTERVAL,
    SUPPORTED_MARGIN_MODE,
    SUPPORTED_POSITION_MODE,
)
from src.exchanges.models import (
    BrokerCancelResult,
    BrokerOrder,
    BrokerOrderRequest,
    BrokerOrderResult,
    BrokerOrderSide,
    BrokerOrderType,
    BrokerPosition,
    BrokerPositionSide,
    BrokerQuantityUnit,
    ExchangeName,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANONICAL_SYMBOL: str = "ETH-USDT-PERP"
BINANCE_SYMBOL: str = "ETHUSDT"
CLIENT_ORDER_ID_PREFIX: str = "RE_SMOKE_"

CONFIRM_ENV: str = "BINANCE_LIVE_SMOKE_TEST_CONFIRM"
CONFIRM_VALUE: str = "I_UNDERSTAND_THIS_PLACES_REAL_ORDERS"

ENV_MAX_NOTIONAL: str = "BINANCE_LIVE_SMOKE_TEST_MAX_NOTIONAL_USDT"
ENV_TP_PCT: str = "BINANCE_LIVE_SMOKE_TEST_TP_PCT"
ENV_SL_PCT: str = "BINANCE_LIVE_SMOKE_TEST_SL_PCT"

DEFAULT_MAX_NOTIONAL: Decimal = Decimal("6")
DEFAULT_TP_PCT: Decimal = Decimal("0.006")
DEFAULT_SL_PCT: Decimal = Decimal("0.006")

EXCHANGE_INFO_PATH: str = "/fapi/v1/exchangeInfo"
TICKER_PRICE_PATH: str = "/fapi/v1/ticker/price"
BALANCE_PATH: str = "/fapi/v2/balance"
# ---------------------------------------------------------------------------
# Preflight state (populated before any order is placed)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExchangeInfoFilters:
    """Parsed LOT_SIZE / MIN_NOTIONAL filters for ETHUSDT."""

    min_qty: Decimal
    step_size: Decimal
    min_notional: Decimal


@dataclass(frozen=True)
class Preflight:
    """Everything the script needs before it places the first order."""

    api_key: str
    api_secret: str
    mark_price: Decimal
    available_usdt_balance: Decimal
    max_notional: Decimal
    tp_pct: Decimal
    sl_pct: Decimal
    filters: ExchangeInfoFilters
    calculated_quantity: Decimal
    calculated_notional: Decimal


# ---------------------------------------------------------------------------
# Safety gates (no network)
# ---------------------------------------------------------------------------


def require_live_confirmation() -> None:
    """Raise ``SystemExit`` unless the live-confirm env var is set correctly."""
    value = os.environ.get(CONFIRM_ENV, "")
    if value != CONFIRM_VALUE:
        print(
            "ERROR: This script places REAL Binance USD-M Futures orders.\n"
            f"Set {CONFIRM_ENV}={CONFIRM_VALUE} to continue.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print("[preflight] live confirmation OK")


def validate_unified_config_for_binance(rt: ExchangeRuntimeConfig) -> str:
    """Validate the unified runtime config for Binance smoke test use.

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

    if rt.binance_symbol != "ETHUSDT":
        print(
            f"ERROR: binance_symbol must be 'ETHUSDT', got {rt.binance_symbol!r}",
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

    print("[preflight] unified config validated OK")
    return rt.binance_symbol


def _load_api_credential(key: str) -> str:
    value = os.environ.get(key, "").strip()
    if not value:
        print(f"ERROR: {key} is not set or empty", file=sys.stderr)
        raise SystemExit(1)
    return value


def load_binance_credentials() -> tuple[str, str]:
    """Return (api_key, api_secret) from environment, or exit."""
    api_key = _load_api_credential("EXCHANGE_API_KEY")
    api_secret = _load_api_credential("EXCHANGE_API_SECRET")
    print("[preflight] API credentials present")
    return api_key, api_secret


async def require_one_way_position_mode(api_key: str, api_secret: str) -> None:
    """Raise ``SystemExit`` unless the account is in One-way / Net Position Mode.

    Calls GET /fapi/v1/positionSide/dual (signed).  dualSidePosition=False
    means One-way Mode is active.  dualSidePosition=True (Hedge Mode) is
    rejected because the script operates in One-way mode.
    """
    signed = build_signed_request(
        method="GET",
        path="/fapi/v1/positionSide/dual",
        params={},
        api_key=api_key,
        api_secret=api_secret,
        base_url=BINANCE_USDM_BASE_URL,
    )
    transport = AiohttpBinanceTransport()
    response: BinanceTransportResponse = await transport.send(signed)

    if response.status_code >= 400:
        payload = response.payload if isinstance(response.payload, dict) else {}
        print(
            f"ERROR: positionSide/dual request failed (HTTP {response.status_code}): {payload}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    dual_side = response.payload.get("dualSidePosition") if isinstance(response.payload, dict) else None
    if dual_side is not False:
        print(
            "ERROR: Binance account must be in One-way / Net Position Mode "
            "(dualSidePosition=False).  "
            "Do NOT use Hedge Mode for this script.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print("[preflight] position mode = one-way/net OK")


# ---------------------------------------------------------------------------
# Public market data helpers
# ---------------------------------------------------------------------------


async def _public_get(path: str, params: dict[str, str] | None = None) -> Any:
    """Perform a public (unsigned) GET against Binance USD-M Futures."""
    url = f"{BINANCE_USDM_BASE_URL}{path}"
    timeout = aiohttp.ClientTimeout(total=10.0)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, params=params) as resp:
            data = await resp.json()
            if resp.status >= 400:
                msg = data if isinstance(data, dict) else {"message": str(data)}
                raise RuntimeError(f"Public GET {path} failed (HTTP {resp.status}): {msg}")
            return data


async def require_isolated_margin(
    api_key: str,
    api_secret: str,
) -> None:
    """Raise ``SystemExit`` if ETHUSDT margin type is not 'isolated'.

    Calls GET /fapi/v2/positionRisk?symbol=ETHUSDT (signed).  Even when
    positionAmt is 0 the response includes ``marginType``.
    """
    signed = build_signed_request(
        method="GET",
        path="/fapi/v2/positionRisk",
        params={"symbol": BINANCE_SYMBOL},
        api_key=api_key,
        api_secret=api_secret,
        base_url=BINANCE_USDM_BASE_URL,
    )
    transport = AiohttpBinanceTransport()
    response: BinanceTransportResponse = await transport.send(signed)

    if response.status_code >= 400:
        payload = response.payload if isinstance(response.payload, dict) else {}
        print(
            f"ERROR: positionRisk request failed (HTTP {response.status_code}): {payload}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if not isinstance(response.payload, list) or len(response.payload) == 0:
        print("ERROR: positionRisk returned no entries for ETHUSDT", file=sys.stderr)
        raise SystemExit(1)

    margin_type = str(response.payload[0].get("marginType", "")).lower()
    if margin_type != "isolated":
        print(
            f"ERROR: ETHUSDT margin type is {margin_type!r}, expected 'isolated'.  "
            "Set MARGIN_MODE=isolated before running this script.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print("[preflight] margin mode = isolated OK")


async def fetch_exchange_info_filters() -> ExchangeInfoFilters:
    """Fetch exchangeInfo and extract LOT_SIZE / MIN_NOTIONAL for ETHUSDT."""
    data = await _public_get(EXCHANGE_INFO_PATH)
    symbols = data.get("symbols", [])
    eth_symbol: dict[str, Any] | None = None
    for s in symbols:
        if s.get("symbol") == BINANCE_SYMBOL:
            eth_symbol = s
            break

    if eth_symbol is None:
        print(f"ERROR: {BINANCE_SYMBOL} not found in exchangeInfo", file=sys.stderr)
        raise SystemExit(1)

    contract_type = str(eth_symbol.get("contractType", ""))
    status = str(eth_symbol.get("status", ""))
    if contract_type != "PERPETUAL":
        print(
            f"ERROR: {BINANCE_SYMBOL} contractType is {contract_type!r}, expected PERPETUAL",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if status != "TRADING":
        print(
            f"ERROR: {BINANCE_SYMBOL} status is {status!r}, expected TRADING",
            file=sys.stderr,
        )
        raise SystemExit(1)

    filters = eth_symbol.get("filters", [])
    lot_size: dict[str, Any] | None = None
    min_notional_filter: dict[str, Any] | None = None
    for f in filters:
        if f.get("filterType") == "LOT_SIZE":
            lot_size = f
        elif f.get("filterType") in ("MIN_NOTIONAL", "NOTIONAL"):
            min_notional_filter = f

    if lot_size is None:
        print("ERROR: LOT_SIZE filter not found for ETHUSDT", file=sys.stderr)
        raise SystemExit(1)

    try:
        min_qty = Decimal(str(lot_size.get("minQty", "0")))
        step_size = Decimal(str(lot_size.get("stepSize", "0")))
    except (InvalidOperation, ValueError) as exc:
        print(f"ERROR: cannot parse LOT_SIZE filter: {exc}", file=sys.stderr)
        raise SystemExit(1)

    if min_qty <= 0 or step_size <= 0:
        print(
            f"ERROR: invalid LOT_SIZE — minQty={min_qty}, stepSize={step_size}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    parsed_min_notional: Decimal
    if min_notional_filter is not None:
        try:
            parsed_min_notional = Decimal(str(min_notional_filter.get("notional", "0")))
        except (InvalidOperation, ValueError):
            parsed_min_notional = Decimal("5")
    else:
        parsed_min_notional = Decimal("5")

    result = ExchangeInfoFilters(
        min_qty=min_qty,
        step_size=step_size,
        min_notional=parsed_min_notional,
    )
    print(
        f"[preflight] ETHUSDT filters — minQty={result.min_qty}, "
        f"stepSize={result.step_size}, minNotional={result.min_notional}"
    )
    return result


async def fetch_mark_price() -> Decimal:
    """Fetch current ETHUSDT mark price via ticker/price."""
    data = await _public_get(TICKER_PRICE_PATH, params={"symbol": BINANCE_SYMBOL})
    price_str = data.get("price")
    if price_str is None:
        print(f"ERROR: no price field in ticker response: {data}", file=sys.stderr)
        raise SystemExit(1)
    try:
        price = Decimal(str(price_str))
    except (InvalidOperation, ValueError) as exc:
        print(f"ERROR: cannot parse ticker price: {exc}", file=sys.stderr)
        raise SystemExit(1)
    if price <= 0:
        print(f"ERROR: mark price is not positive: {price}", file=sys.stderr)
        raise SystemExit(1)
    print(f"[preflight] ETHUSDT mark price = {price}")
    return price


# ---------------------------------------------------------------------------
# Balance (signed)
# ---------------------------------------------------------------------------


async def fetch_account_balance(
    api_key: str,
    api_secret: str,
) -> Decimal:
    """Fetch USDT balance from Binance USD-M Futures account.

    Uses a signed GET request via a one-shot AiohttpBinanceTransport.
    """
    signed = build_signed_request(
        method="GET",
        path=BALANCE_PATH,
        params={},
        api_key=api_key,
        api_secret=api_secret,
        base_url=BINANCE_USDM_BASE_URL,
    )
    transport = AiohttpBinanceTransport()
    response: BinanceTransportResponse = await transport.send(signed)

    if response.status_code >= 400:
        payload = response.payload if isinstance(response.payload, dict) else {}
        print(
            f"ERROR: balance request failed (HTTP {response.status_code}): {payload}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if not isinstance(response.payload, list):
        print(
            f"ERROR: unexpected balance response format: {type(response.payload)}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    for asset in response.payload:
        if asset.get("asset") == "USDT":
            try:
                available = Decimal(str(asset.get("availableBalance", "0")))
            except (InvalidOperation, ValueError) as exc:
                print(f"ERROR: cannot parse USDT balance: {exc}", file=sys.stderr)
                raise SystemExit(1)
            print(f"[preflight] available USDT balance = {available}")
            return available

    print("ERROR: USDT balance not found in account", file=sys.stderr)
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Quantity calculation
# ---------------------------------------------------------------------------


def _round_up_to_step(quantity: Decimal, step_size: Decimal) -> Decimal:
    """Round *quantity* up to the nearest *step_size* multiple."""
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    remainder = quantity % step_size
    if remainder == 0:
        return quantity
    return (quantity // step_size + 1) * step_size


def calculate_safe_quantity(
    *,
    mark_price: Decimal,
    max_notional: Decimal,
    filters: ExchangeInfoFilters,
) -> tuple[Decimal, Decimal]:
    """Return (quantity, notional) that passes Binance LOT_SIZE / MIN_NOTIONAL.

    Raises ``SystemExit`` when the account has insufficient notional.
    """
    raw_quantity = max_notional / mark_price
    quantity = _round_up_to_step(raw_quantity, filters.step_size)
    actual_notional = quantity * mark_price

    # Tighten to step size if raw was slightly below
    if actual_notional < filters.min_notional:
        # Bump quantity by one more step
        quantity = _round_up_to_step(
            filters.min_notional / mark_price, filters.step_size
        )
        actual_notional = quantity * mark_price

    if quantity < filters.min_qty:
        print(
            f"ERROR: calculated quantity {quantity} < minQty {filters.min_qty}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if actual_notional < filters.min_notional:
        print(
            f"ERROR: notional {actual_notional} < minNotional {filters.min_notional}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(
        f"[preflight] calculated quantity = {quantity} ETH "
        f"(notional ≈ {actual_notional} USDT)"
    )
    return quantity, actual_notional


_cid_counter: int = 0


def _generate_client_order_id(label: str) -> str:
    """Return a unique clientOrderId with the smoke test prefix."""
    global _cid_counter
    _cid_counter += 1
    ts = time.monotonic_ns()
    return f"{CLIENT_ORDER_ID_PREFIX}{label}_{ts}_{_cid_counter}"


# ---------------------------------------------------------------------------
# Helper: build a BrokerOrderRequest for Binance
# ---------------------------------------------------------------------------


def _make_order_request(
    *,
    side: BrokerOrderSide,
    order_type: BrokerOrderType,
    quantity: Decimal,
    price: Decimal | None = None,
    trigger_price: Decimal | None = None,
    reduce_only: bool = False,
    client_order_id: str | None = None,
) -> BrokerOrderRequest:
    return BrokerOrderRequest(
        exchange=ExchangeName.BINANCE,
        symbol=BINANCE_SYMBOL,
        side=side,
        position_side=BrokerPositionSide.NET,
        order_type=order_type,
        quantity=quantity,
        quantity_unit=BrokerQuantityUnit.BASE_ASSET,
        price=price,
        trigger_price=trigger_price,
        reduce_only=reduce_only,
        client_order_id=client_order_id,
    )


# ---------------------------------------------------------------------------
# Order operations (all use the injected BinanceBrokerClient)
# ---------------------------------------------------------------------------


async def open_long(
    client: BinanceBrokerClient,
    quantity: Decimal,
    client_order_id: str,
) -> BrokerOrderResult:
    """Open a LONG position via MARKET BUY."""
    print(f"[open] MARKET BUY {quantity} ETHUSDT (cid={client_order_id})")
    request = _make_order_request(
        side=BrokerOrderSide.BUY,
        order_type=BrokerOrderType.MARKET,
        quantity=quantity,
        client_order_id=client_order_id,
    )
    result = await client.place_order(request)
    if not result.ok:
        raise RuntimeError(f"Open LONG failed: {result.message}")
    print(f"[open] filled — orderId={result.order_id}, cid={result.client_order_id}")
    return result


async def fetch_long_position(client: BinanceBrokerClient) -> BrokerPosition | None:
    """Return the ETHUSDT position if quantity > 0, or None.

    In One-way / net mode the position side may be BOTH / NET / LONG.
    We only check that quantity is positive.
    """
    pos = await client.fetch_position(BINANCE_SYMBOL)
    if pos is not None and pos.quantity <= 0:
        return None
    return pos


async def place_tp(
    client: BinanceBrokerClient,
    quantity: Decimal,
    tp_price: Decimal,
    client_order_id: str,
) -> BrokerOrderResult:
    """Place a LIMIT SELL TP order."""
    print(f"[tp] LIMIT SELL {quantity} @ {tp_price} (cid={client_order_id})")
    request = _make_order_request(
        side=BrokerOrderSide.SELL,
        order_type=BrokerOrderType.LIMIT,
        quantity=quantity,
        price=tp_price,
        reduce_only=True,
        client_order_id=client_order_id,
    )
    result = await client.place_order(request)
    if not result.ok:
        raise RuntimeError(f"Place TP failed: {result.message}")
    print(f"[tp] placed — orderId={result.order_id}, cid={result.client_order_id}")
    return result


async def place_sl(
    client: BinanceBrokerClient,
    quantity: Decimal,
    sl_price: Decimal,
    client_order_id: str,
) -> BrokerOrderResult:
    """Place a STOP_MARKET SELL SL order."""
    print(f"[sl] STOP_MARKET SELL {quantity} @ trigger {sl_price} (cid={client_order_id})")
    request = _make_order_request(
        side=BrokerOrderSide.SELL,
        order_type=BrokerOrderType.STOP_MARKET,
        quantity=quantity,
        trigger_price=sl_price,
        reduce_only=True,
        client_order_id=client_order_id,
    )
    result = await client.place_order(request)
    if not result.ok:
        raise RuntimeError(f"Place SL failed: {result.message}")
    print(f"[sl] placed — orderId={result.order_id}, cid={result.client_order_id}")
    return result


async def fetch_open_orders(client: BinanceBrokerClient) -> list[BrokerOrder]:
    """Return current open orders for ETHUSDT."""
    orders = await client.fetch_open_orders(BINANCE_SYMBOL)
    print(f"[fetch_orders] {len(orders)} open order(s)")
    for o in orders:
        print(
            f"  orderId={o.order_id} cid={o.client_order_id} "
            f"type={o.order_type.value} side={o.side.value} "
            f"qty={o.quantity} price={o.price} trigger={o.trigger_price}"
        )
    return list(orders)


async def cancel_order_by_id(
    client: BinanceBrokerClient,
    order_id: str,
) -> BrokerCancelResult:
    """Cancel a single order by ID."""
    result = await client.cancel_order(BINANCE_SYMBOL, order_id)
    if not result.ok:
        print(
            f"[cancel] WARNING: cancel orderId={order_id} failed: {result.message}",
            file=sys.stderr,
        )
    else:
        print(f"[cancel] cancelled orderId={order_id}")
    return result


async def cancel_smoke_orders(
    client: BinanceBrokerClient,
) -> int:
    """Cancel all open orders whose clientOrderId starts with RE_SMOKE_."""
    orders = await fetch_open_orders(client)
    cancelled = 0
    for o in orders:
        cid = o.client_order_id or ""
        if not cid.startswith(CLIENT_ORDER_ID_PREFIX):
            continue
        if o.order_id is None:
            continue
        await cancel_order_by_id(client, o.order_id)
        cancelled += 1
    return cancelled


async def close_long_position(
    client: BinanceBrokerClient,
    quantity: Decimal,
    client_order_id: str,
) -> BrokerOrderResult:
    """Market-close the LONG position."""
    print(f"[close] MARKET SELL {quantity} ETHUSDT (cid={client_order_id})")
    request = _make_order_request(
        side=BrokerOrderSide.SELL,
        order_type=BrokerOrderType.MARKET,
        quantity=quantity,
        reduce_only=True,
        client_order_id=client_order_id,
    )
    result = await client.place_order(request)
    if not result.ok:
        raise RuntimeError(f"Market close failed: {result.message}")
    print(f"[close] filled — orderId={result.order_id}, cid={result.client_order_id}")
    return result


# ---------------------------------------------------------------------------
# Cleanup (best-effort, must not raise)
# ---------------------------------------------------------------------------


async def cleanup(
    client: BinanceBrokerClient,
) -> None:
    """Best-effort cleanup: cancel smoke orders, then close any residual LONG."""
    print("\n[cleanup] starting best-effort cleanup...")
    try:
        cancelled = await cancel_smoke_orders(client)
        print(f"[cleanup] cancelled {cancelled} smoke order(s)")
    except Exception as exc:
        print(f"[cleanup] WARNING: cancel step raised: {exc}", file=sys.stderr)

    await asyncio.sleep(0.5)

    try:
        pos = await fetch_long_position(client)
    except Exception as exc:
        print(f"[cleanup] WARNING: fetch position raised: {exc}", file=sys.stderr)
        pos = None

    if pos is not None and pos.quantity > 0:
        try:
            print(f"[cleanup] residual LONG position qty={pos.quantity}, attempting close...")
            cid = _generate_client_order_id("cleanup_close")
            await close_long_position(client, pos.quantity, cid)
            await asyncio.sleep(1)
            pos_after = await fetch_long_position(client)
            if pos_after is not None and pos_after.quantity > 0:
                print(
                    f"[cleanup] WARNING: position still open after cleanup close "
                    f"(qty={pos_after.quantity})",
                    file=sys.stderr,
                )
            else:
                print("[cleanup] residual position closed")
        except Exception as exc:
            print(f"[cleanup] WARNING: close step raised: {exc}", file=sys.stderr)
    else:
        print("[cleanup] no residual LONG position")

    # Final status
    try:
        final_pos = await fetch_long_position(client)
        if final_pos is not None and final_pos.quantity > 0:
            print(
                f"[cleanup] FINAL: LONG position still open qty={final_pos.quantity}",
            )
        else:
            print("[cleanup] FINAL: position = 0 ✓")
    except Exception as exc:
        print(f"[cleanup] WARNING: final position check raised: {exc}", file=sys.stderr)

    print("[cleanup] done")


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------


async def _run_smoke_test(
    client: BinanceBrokerClient,
    preflight: Preflight,
) -> bool:
    """Execute the full smoke test sequence.  Returns True on success."""
    opened_cid = _generate_client_order_id("open")
    tp_cid = _generate_client_order_id("tp")
    sl_cid = _generate_client_order_id("sl")
    close_cid = _generate_client_order_id("close")

    # ------------------------------------------------------------------
    # 1. Open LONG
    # ------------------------------------------------------------------
    open_result = await open_long(client, preflight.calculated_quantity, opened_cid)
    print("[smoke] step 1/7: open LONG ✓")

    await asyncio.sleep(1.5)

    # ------------------------------------------------------------------
    # 2. Fetch position
    # ------------------------------------------------------------------
    pos = await fetch_long_position(client)
    if pos is None or pos.quantity <= 0:
        print("ERROR: no LONG position after open order", file=sys.stderr)
        return False
    if pos.average_entry_price is None or pos.average_entry_price <= 0:
        print("ERROR: position has no valid entry price", file=sys.stderr)
        return False
    entry_price = pos.average_entry_price
    opened_qty = pos.quantity
    print(
        f"[smoke] step 2/7: position qty={opened_qty}, entryPrice={entry_price} ✓"
    )

    # ------------------------------------------------------------------
    # 3. Place TP
    # ------------------------------------------------------------------
    tp_price = entry_price * (1 + preflight.tp_pct)
    tp_price = (tp_price * 100).quantize(Decimal("1")) / 100  # round to 2 dp
    tp_result = await place_tp(client, opened_qty, tp_price, tp_cid)
    tp_order_id = tp_result.order_id
    print(f"[smoke] step 3/7: TP placed @ {tp_price} ✓")

    # ------------------------------------------------------------------
    # 4. Place SL
    # ------------------------------------------------------------------
    sl_price = entry_price * (1 - preflight.sl_pct)
    sl_price = (sl_price * 100).quantize(Decimal("1")) / 100  # round to 2 dp
    sl_result = await place_sl(client, opened_qty, sl_price, sl_cid)
    sl_order_id = sl_result.order_id
    print(f"[smoke] step 4/7: SL placed @ trigger {sl_price} ✓")

    # ------------------------------------------------------------------
    # 5. Fetch open orders
    # ------------------------------------------------------------------
    open_orders = await fetch_open_orders(client)
    tp_found = any(
        (o.order_id == tp_order_id) or (o.client_order_id == tp_cid)
        for o in open_orders
    )
    sl_found = any(
        (o.order_id == sl_order_id) or (o.client_order_id == sl_cid)
        for o in open_orders
    )
    if not tp_found:
        print("[smoke] WARNING: TP order not found in open orders", file=sys.stderr)
    if not sl_found:
        print("[smoke] WARNING: SL order not found in open orders", file=sys.stderr)
    if tp_found and sl_found:
        print("[smoke] step 5/7: open orders confirmed ✓")
    else:
        print("[smoke] step 5/7: open orders partially visible (continuing)")

    # ------------------------------------------------------------------
    # 6. Cancel TP / SL
    # ------------------------------------------------------------------
    cancelled_count = 0
    if tp_order_id is not None:
        await cancel_order_by_id(client, tp_order_id)
        cancelled_count += 1
    if sl_order_id is not None:
        await cancel_order_by_id(client, sl_order_id)
        cancelled_count += 1

    # Verify cancellation
    orders_after_cancel = await fetch_open_orders(client)
    residual_smoke = [
        o for o in orders_after_cancel
        if (o.client_order_id or "").startswith(CLIENT_ORDER_ID_PREFIX)
    ]
    if residual_smoke:
        print(
            f"[smoke] WARNING: {len(residual_smoke)} smoke order(s) still open after cancel",
            file=sys.stderr,
        )
    else:
        print("[smoke] step 6/7: cancel TP/SL ✓")

    # ------------------------------------------------------------------
    # 7. Market close
    # ------------------------------------------------------------------
    pos_before_close = await fetch_long_position(client)
    if pos_before_close is None or pos_before_close.quantity <= 0:
        print("[smoke] step 7/7: position already closed ✓")
        return True

    close_qty = pos_before_close.quantity
    await close_long_position(client, close_qty, close_cid)
    await asyncio.sleep(1.5)

    pos_after = await fetch_long_position(client)
    if pos_after is not None and pos_after.quantity > 0:
        print(
            f"[smoke] step 7/7: position close INCOMPLETE — "
            f"residual qty={pos_after.quantity}",
            file=sys.stderr,
        )
        return False

    print("[smoke] step 7/7: market close ✓ position = 0")
    return True


async def main() -> int:
    """Entry point — runs the smoke test, always attempts cleanup."""
    # --- safety gates (no network) ---
    require_live_confirmation()
    api_key, api_secret = load_binance_credentials()

    # --- unified runtime config ---
    rt = load_unified_runtime_config()
    binance_symbol = validate_unified_config_for_binance(rt)

    # --- preflight (network) ---
    print("[preflight] checking position mode...")
    await require_one_way_position_mode(api_key, api_secret)

    print("[preflight] checking margin mode...")
    await require_isolated_margin(api_key, api_secret)

    print("[preflight] fetching exchangeInfo...")
    filters = await fetch_exchange_info_filters()

    print("[preflight] fetching mark price...")
    mark_price = await fetch_mark_price()

    print("[preflight] fetching account balance...")
    available_balance = await fetch_account_balance(api_key, api_secret)

    # --- configuration ---
    max_notional_str = os.environ.get(ENV_MAX_NOTIONAL, str(DEFAULT_MAX_NOTIONAL))
    tp_pct_str = os.environ.get(ENV_TP_PCT, str(DEFAULT_TP_PCT))
    sl_pct_str = os.environ.get(ENV_SL_PCT, str(DEFAULT_SL_PCT))

    try:
        max_notional = Decimal(max_notional_str)
        tp_pct = Decimal(tp_pct_str)
        sl_pct = Decimal(sl_pct_str)
    except (InvalidOperation, ValueError) as exc:
        print(f"ERROR: cannot parse env config: {exc}", file=sys.stderr)
        return 1

    if max_notional <= 0:
        print(f"ERROR: MAX_NOTIONAL must be positive, got {max_notional}", file=sys.stderr)
        return 1

    # --- quantity calculation ---
    calculated_qty, calculated_notional = calculate_safe_quantity(
        mark_price=mark_price,
        max_notional=max_notional,
        filters=filters,
    )

    # --- balance check ---
    if available_balance < calculated_notional:
        print(
            f"ERROR: insufficient USDT balance — "
            f"need ≈ {calculated_notional}, have {available_balance}",
            file=sys.stderr,
        )
        return 1

    preflight = Preflight(
        api_key=api_key,
        api_secret=api_secret,
        mark_price=mark_price,
        available_usdt_balance=available_balance,
        max_notional=max_notional,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        filters=filters,
        calculated_quantity=calculated_qty,
        calculated_notional=calculated_notional,
    )

    print(
        "\n================================================================"
        "\n  BINANCE LIVE SMOKE TEST"
        "\n================================================================"
        f"\n  Symbol:       {binance_symbol} ({rt.canonical_symbol})"
        f"\n  Side:         LONG"
        f"\n  Quantity:     {calculated_qty} ETH"
        f"\n  Notional:     ≈ {calculated_notional} USDT"
        f"\n  Mark Price:   {mark_price}"
        f"\n  TP PCT:       {tp_pct}"
        f"\n  SL PCT:       {sl_pct}"
        "\n================================================================\n"
    )

    # --- build client ---
    transport = AiohttpBinanceTransport()
    client = BinanceBrokerClient(
        api_key=api_key,
        api_secret=api_secret,
        transport=transport,
    )

    success = False
    try:
        success = await _run_smoke_test(client, preflight)
    except Exception as exc:
        print(f"\n[smoke] ERROR during test: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        success = False
    finally:
        await cleanup(client)

    if success:
        print("\n[smoke] ALL STEPS PASSED ✓")
        return 0
    else:
        print("\n[smoke] TEST FAILED — check output above", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
