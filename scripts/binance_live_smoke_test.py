#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/13
@File       : binance_live_smoke_test.py
@Description: Binance ETHUSDT live smoke test — places REAL USD-M Futures orders.

WARNING: This script places REAL orders on Binance USD-M Futures.
It requires TWO explicit confirmations to run:
  1. LIVE_SMOKE_TEST_CONFIRM — acknowledges real order placement
  2. LIVE_CONFIRMATION — live preflight confirmation

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

Preflight env (from binance_live_preflight):
    LIVE_ENABLED=true
    LIVE_ALLOW_ORDERS=true
    LIVE_CONFIRMATION=I_UNDERSTAND_EXCHANGE_LIVE_TRADING
    LIVE_MAX_ORDER_NOTIONAL_USDT=<value <= 25>
    LIVE_MAX_POSITION_NOTIONAL_USDT=<value <= 30>
    LIVE_LEVERAGE=<value <= 20>

Usage::

    EXCHANGE=binance                                    \\
    EXCHANGE_API_KEY=...                                \\
    EXCHANGE_API_SECRET=...                             \\
    LIVE_SMOKE_TEST_CONFIRM=I_UNDERSTAND_THIS_PLACES_REAL_EXCHANGE_ORDERS \\
    LIVE_ENABLED=true                                   \\
    LIVE_ALLOW_ORDERS=true                              \\
    LIVE_CONFIRMATION=I_UNDERSTAND_EXCHANGE_LIVE_TRADING \\
    LIVE_MAX_ORDER_NOTIONAL_USDT=6                      \\
    LIVE_MAX_POSITION_NOTIONAL_USDT=6                   \\
    LIVE_LEVERAGE=20                                    \\
    python scripts/binance_live_smoke_test.py

Trade flow (sequential, fail-fast with cleanup on error):
    0. preflight — both confirmations, unified config, preflight guard,
       notional caps, position mode, leverage (conditional),
       exchangeInfo, mark price, balance, qty
    1. open ETHUSDT LONG (MARKET)
    2. place TP (LIMIT SELL)
    3. place SL (STOP_MARKET SELL) via Algo Order API
    4. fetch open orders → confirm TP + SL visible
    5. cancel TP / SL
    6. market close (MARKET SELL)
    7. check position == 0
    8. final cleanup (best-effort cancel + close any residual)

Safety gates:
    - Double confirmation: smoke + preflight
    - Max order notional hard cap at 25 USDT
    - Max position notional hard cap at 30 USDT
    - Default no leverage change without explicit opt-in
    - No existing position check before placing orders
    - Only RE_SMOKE_ prefixed old orders are cleaned

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
from src.exchanges.binance.live_preflight import (
    BINANCE_LIVE_HARD_MAX_LEVERAGE,
    build_binance_live_preflight_report,
    format_binance_live_blocked_message,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANONICAL_SYMBOL: str = "ETH-USDT-PERP"
BINANCE_SYMBOL: str = "ETHUSDT"
CLIENT_ORDER_ID_PREFIX: str = "RE_SMOKE_"

# Generic env var names (primary)
CONFIRM_ENV: str = "LIVE_SMOKE_TEST_CONFIRM"
CONFIRM_VALUE: str = "I_UNDERSTAND_THIS_PLACES_REAL_EXCHANGE_ORDERS"

ALLOW_SET_LEVERAGE_ENV: str = "LIVE_SMOKE_TEST_ALLOW_SET_LEVERAGE"
ALLOW_SET_LEVERAGE_VALUE: str = "I_UNDERSTAND_THIS_CHANGES_EXCHANGE_LEVERAGE"

ENV_MAX_NOTIONAL: str = "LIVE_SMOKE_TEST_MAX_NOTIONAL_USDT"
ENV_TP_PCT: str = "LIVE_SMOKE_TEST_TP_PCT"
ENV_SL_PCT: str = "LIVE_SMOKE_TEST_SL_PCT"

DEFAULT_MAX_NOTIONAL: Decimal = Decimal("20")
DEFAULT_TP_PCT: Decimal = Decimal("0.006")
DEFAULT_SL_PCT: Decimal = Decimal("0.006")

ENV_MARGIN_BUFFER_MULTIPLIER: str = "LIVE_SMOKE_TEST_MARGIN_BUFFER_MULTIPLIER"
DEFAULT_MARGIN_BUFFER_MULTIPLIER: Decimal = Decimal("3")

# Backward-compatible BINANCE_ aliases
_CONFIRM_ENV_ALIAS: str = "BINANCE_LIVE_SMOKE_TEST_CONFIRM"
_ALLOW_SET_LEVERAGE_ENV_ALIAS: str = "BINANCE_LIVE_SMOKE_TEST_ALLOW_SET_LEVERAGE"
_ENV_MAX_NOTIONAL_ALIAS: str = "BINANCE_LIVE_SMOKE_TEST_MAX_NOTIONAL_USDT"
_ENV_TP_PCT_ALIAS: str = "BINANCE_LIVE_SMOKE_TEST_TP_PCT"
_ENV_SL_PCT_ALIAS: str = "BINANCE_LIVE_SMOKE_TEST_SL_PCT"
_ENV_MARGIN_BUFFER_MULTIPLIER_ALIAS: str = "BINANCE_LIVE_SMOKE_TEST_MARGIN_BUFFER_MULTIPLIER"

EXCHANGE_INFO_PATH: str = "/fapi/v1/exchangeInfo"
TICKER_PRICE_PATH: str = "/fapi/v1/ticker/price"
BALANCE_PATH: str = "/fapi/v2/balance"
CHANGE_LEVERAGE_PATH: str = "/fapi/v1/leverage"
ALGO_ORDER_PATH: str = "/fapi/v1/algoOrder"
ALGO_OPEN_ORDERS_PATH: str = "/fapi/v1/openAlgoOrders"
POSITION_RISK_PATH: str = "/fapi/v2/positionRisk"

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
    leverage: int
    margin_buffer_multiplier: Decimal
    estimated_initial_margin: Decimal
    required_margin_with_buffer: Decimal


# ---------------------------------------------------------------------------
# Safety gates (no network)
# ---------------------------------------------------------------------------


def _resolve_env(primary: str, alias: str, *, description: str = "") -> str:
    """Read *primary* env var, falling back to *alias*.  Exit on conflict.

    When both *primary* and *alias* are set to different non-empty values
    the function prints an error and raises ``SystemExit(1)``.
    """
    p_val = os.environ.get(primary, "").strip()
    a_val = os.environ.get(alias, "").strip()
    if p_val and a_val and p_val != a_val:
        label = f" ({description})" if description else ""
        print(
            f"ERROR: both {primary}={p_val!r} and "
            f"{alias}={a_val!r} are set with conflicting values{label}. "
            f"Use only {primary}.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return p_val if p_val else a_val


def require_live_confirmation() -> None:
    """Raise ``SystemExit`` unless the live-confirm env var is set correctly."""
    value = _resolve_env(CONFIRM_ENV, _CONFIRM_ENV_ALIAS, description="smoke test confirmation")
    if value != CONFIRM_VALUE:
        print(
            "ERROR: This script places REAL Binance USD-M Futures orders.\n"
            f"Set {CONFIRM_ENV}={CONFIRM_VALUE} to continue.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print("[preflight] live confirmation OK")


def require_binance_live_preflight_for_smoke() -> None:
    """Run the Binance live preflight guard with orders globally enabled.

    This requires the user to set LIVE_ENABLED, LIVE_ALLOW_ORDERS,
    LIVE_CONFIRMATION, LIVE_MAX_ORDER_NOTIONAL_USDT,
    LIVE_MAX_POSITION_NOTIONAL_USDT, and LIVE_LEVERAGE
    (or their backward-compatible BINANCE_* aliases).
    """
    report = build_binance_live_preflight_report(
        os.environ,
        orders_globally_enabled=True,
    )
    if not report.ok:
        print(format_binance_live_blocked_message(report), file=sys.stderr)
        raise SystemExit(1)
    print("[preflight] Binance live preflight OK")


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


def _read_positive_decimal_env(
    name: str, default: Decimal, *, alias: str | None = None,
) -> Decimal:
    """Read a positive Decimal from environment, falling back to *default*.

    When *alias* is provided the function reads *name* first (primary)
    and falls back to *alias* (backward-compatible).  If both are set
    to different values the function raises ``SystemExit``.

    Raises ``SystemExit`` when the value is present but invalid (non-numeric,
    zero, or negative).
    """
    if alias is not None:
        raw = _resolve_env(name, alias, description=f"env var {name}")
    else:
        raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = Decimal(raw)
    except (InvalidOperation, ValueError):
        print(
            f"ERROR: {name} must be a valid decimal, got {raw!r}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if value <= 0:
        print(
            f"ERROR: {name} must be positive, got {value}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return value


# ---------------------------------------------------------------------------
# Notional cap enforcement
# ---------------------------------------------------------------------------


def require_requested_notional_cap(
    *,
    smoke_max_notional: Decimal,
    preflight_max_order_notional: Decimal,
    preflight_max_position_notional: Decimal,
) -> None:
    """Ensure the smoke max notional does not exceed user-configured limits.

    Checks:
    1. smoke_max_notional <= preflight_max_order_notional (user-set order cap)
    2. smoke_max_notional <= preflight_max_position_notional (user-set position cap)
    """
    if smoke_max_notional > preflight_max_order_notional:
        print(
            f"ERROR: max notional {smoke_max_notional} exceeds "
            f"preflight max order notional {preflight_max_order_notional}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if smoke_max_notional > preflight_max_position_notional:
        print(
            f"ERROR: max notional {smoke_max_notional} exceeds "
            f"preflight max position notional {preflight_max_position_notional}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(
        f"[preflight] notional cap OK | "
        f"smoke_max={smoke_max_notional} <= "
        f"order_cap={preflight_max_order_notional} | "
        f"position_cap={preflight_max_position_notional}"
    )


def require_calculated_notional_cap(
    *,
    calculated_notional: Decimal,
    preflight_max_order_notional: Decimal,
    preflight_max_position_notional: Decimal,
) -> None:
    """Ensure the calculated notional (after step rounding) does not exceed user-configured caps.

    Checks:
    1. calculated_notional <= preflight_max_order_notional (user-set order cap)
    2. calculated_notional <= preflight_max_position_notional (user-set position cap)
    """
    if calculated_notional > preflight_max_order_notional:
        print(
            f"ERROR: calculated notional {calculated_notional} exceeds "
            f"preflight max order notional {preflight_max_order_notional}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if calculated_notional > preflight_max_position_notional:
        print(
            f"ERROR: calculated notional {calculated_notional} exceeds "
            f"preflight max position notional {preflight_max_position_notional}",
            file=sys.stderr,
        )
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Leverage control
# ---------------------------------------------------------------------------


def allow_set_leverage() -> bool:
    """Return True only when the user explicitly opts in to leverage changes."""
    value = _resolve_env(
        ALLOW_SET_LEVERAGE_ENV, _ALLOW_SET_LEVERAGE_ENV_ALIAS,
        description="allow set leverage",
    )
    return value == ALLOW_SET_LEVERAGE_VALUE


async def require_existing_leverage(
    api_key: str,
    api_secret: str,
    expected_leverage: int,
) -> None:
    """Check that the current ETHUSDT leverage matches *expected_leverage*.

    Calls GET /fapi/v2/positionRisk?symbol=ETHUSDT (signed) and reads the
    ``leverage`` field.  Exits if the leverage does not match.
    """
    signed = build_signed_request(
        method="GET",
        path=POSITION_RISK_PATH,
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

    current_leverage_raw = response.payload[0].get("leverage")
    try:
        current_leverage = int(current_leverage_raw) if current_leverage_raw is not None else None
    except (ValueError, TypeError):
        print(
            f"ERROR: cannot parse leverage from positionRisk: {current_leverage_raw}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if current_leverage != expected_leverage:
        print(
            f"ERROR: Current ETHUSDT leverage is {current_leverage}, "
            f"expected {expected_leverage}.\n"
            f"Set Binance manually, or explicitly set "
            f"{ALLOW_SET_LEVERAGE_ENV}={ALLOW_SET_LEVERAGE_VALUE}.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"[preflight] existing leverage = {current_leverage}x OK")


# ---------------------------------------------------------------------------
# Existing position guard
# ---------------------------------------------------------------------------


async def require_no_existing_position(client: BinanceBrokerClient) -> None:
    """Raise ``SystemExit`` if ETHUSDT position already exists.

    This prevents the smoke test from accidentally touching a non-smoke position.
    """
    pos = await client.fetch_position(BINANCE_SYMBOL)
    if pos is not None and pos.quantity != 0:
        print(
            f"ERROR: existing ETHUSDT position detected before smoke test: "
            f"qty={pos.quantity}. "
            "Refusing to run tiny order smoke to avoid touching non-smoke position.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print("[preflight] no existing ETHUSDT position OK")


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
        path=POSITION_RISK_PATH,
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


async def set_initial_leverage(
    api_key: str,
    api_secret: str,
    leverage: int,
) -> None:
    """Set ETHUSDT initial leverage on Binance USD-M Futures.

    Calls POST /fapi/v1/leverage with ``symbol=ETHUSDT`` and the requested
    *leverage*.  Raises ``SystemExit`` on HTTP errors or if the response
    leverage does not match the request.
    """
    signed = build_signed_request(
        method="POST",
        path=CHANGE_LEVERAGE_PATH,
        params={"symbol": BINANCE_SYMBOL, "leverage": leverage},
        api_key=api_key,
        api_secret=api_secret,
        base_url=BINANCE_USDM_BASE_URL,
    )
    transport = AiohttpBinanceTransport()
    response: BinanceTransportResponse = await transport.send(signed)

    if response.status_code >= 400:
        payload = response.payload if isinstance(response.payload, dict) else {}
        print(
            f"ERROR: set leverage failed (HTTP {response.status_code}): {payload}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    if isinstance(response.payload, dict):
        resp_leverage = response.payload.get("leverage")
        if resp_leverage is not None and int(resp_leverage) != leverage:
            print(
                f"ERROR: leverage mismatch — requested {leverage}x, "
                f"got {resp_leverage}x",
                file=sys.stderr,
            )
            raise SystemExit(1)

    print(f"[preflight] leverage = {leverage}x OK")


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

    if actual_notional > max_notional:
        print(
            f"[preflight] WARNING: calculated notional {actual_notional} "
            f"is above requested target {max_notional} due to "
            f"Binance minQty={filters.min_qty} "
            f"minNotional={filters.min_notional} "
            f"stepSize={filters.step_size}",
            file=sys.stderr,
        )

    print(
        f"[preflight] calculated quantity = {quantity} ETH "
        f"(notional ≈ {actual_notional} USDT)"
    )
    return quantity, actual_notional


def calculate_required_margin_with_buffer(
    *,
    notional: Decimal,
    leverage: int,
    buffer_multiplier: Decimal,
) -> tuple[Decimal, Decimal]:
    """Return ``(estimated_initial_margin, required_margin_with_buffer)``.

    ``estimated_initial_margin = notional / leverage``
    ``required_margin_with_buffer = estimated_initial_margin * buffer_multiplier``

    Raises ``ValueError`` on invalid inputs.
    """
    if leverage <= 0:
        raise ValueError("leverage must be positive")
    if buffer_multiplier <= 0:
        raise ValueError("buffer_multiplier must be positive")
    if notional <= 0:
        raise ValueError("notional must be positive")

    estimated = notional / Decimal(leverage)
    required = estimated * buffer_multiplier
    return estimated, required


_cid_counter: int = 0


def _generate_client_order_id(label: str) -> str:
    """Return a unique clientOrderId with the smoke test prefix, length <= 36."""
    global _cid_counter
    _cid_counter += 1

    short_labels: dict[str, str] = {
        "open": "op",
        "tp": "tp",
        "sl": "sl",
        "close": "cl",
        "cleanup_close": "cc",
    }
    short_label = short_labels.get(label, label[:4])
    suffix = time.monotonic_ns() % 1_000_000_000
    cid = f"{CLIENT_ORDER_ID_PREFIX}{short_label}_{suffix}_{_cid_counter}"
    return cid[:36]


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
# Algo Order helpers (use direct signed requests — NOT BinanceBrokerClient)
# ---------------------------------------------------------------------------


async def place_stop_loss_algo_order(
    *,
    api_key: str,
    api_secret: str,
    quantity: Decimal,
    sl_price: Decimal,
    client_order_id: str,
) -> BrokerOrderResult:
    """Place a STOP_MARKET SELL algo order via Binance Algo Order API.

    Uses ``POST /fapi/v1/algoOrder`` with ``algoType=CONDITIONAL``.
    This is the dedicated endpoint for conditional orders (STOP_MARKET,
    TAKE_PROFIT_MARKET, etc.) — the regular ``POST /fapi/v1/order``
    rejects these types with error -4120.

    Raises ``RuntimeError`` on HTTP or business-level errors.
    """
    params: dict[str, Any] = {
        "algoType": "CONDITIONAL",
        "symbol": BINANCE_SYMBOL,
        "side": "SELL",
        "type": "STOP_MARKET",
        "quantity": str(quantity),
        "triggerPrice": str(sl_price),
        "reduceOnly": "true",
        "workingType": "MARK_PRICE",
        "clientAlgoId": client_order_id,
    }

    signed = build_signed_request(
        method="POST",
        path=ALGO_ORDER_PATH,
        params=params,
        api_key=api_key,
        api_secret=api_secret,
        base_url=BINANCE_USDM_BASE_URL,
    )

    transport = AiohttpBinanceTransport()
    response: BinanceTransportResponse = await transport.send(signed)

    if response.status_code >= 400:
        payload = response.payload if isinstance(response.payload, dict) else {}
        raise RuntimeError(
            f"Algo SL order HTTP {response.status_code}: {payload}"
        )

    if isinstance(response.payload, dict):
        code = response.payload.get("code")
        if isinstance(code, int) and code < 0:
            msg = response.payload.get("msg", "Unknown error")
            raise RuntimeError(f"Algo SL order rejected: [{code}] {msg}")

    payload = response.payload if isinstance(response.payload, dict) else {}
    algo_id = payload.get("algoId")
    cid = payload.get("clientAlgoId") or client_order_id

    print(
        f"[sl-algo] placed — algoId={algo_id}, clientAlgoId={cid}"
    )
    return BrokerOrderResult(
        exchange=ExchangeName.BINANCE,
        symbol=BINANCE_SYMBOL,
        ok=True,
        order_id=str(algo_id) if algo_id is not None else cid,
        client_order_id=cid,
        raw=response.payload,
    )


async def cancel_algo_order_by_client_id(
    *,
    api_key: str,
    api_secret: str,
    client_order_id: str,
) -> BrokerCancelResult:
    """Cancel a single algo order by its ``clientAlgoId``.

    Uses ``DELETE /fapi/v1/algoOrder`` with ``clientAlgoId`` parameter.
    """
    signed = build_signed_request(
        method="DELETE",
        path=ALGO_ORDER_PATH,
        params={
            "symbol": BINANCE_SYMBOL,
            "clientAlgoId": client_order_id,
        },
        api_key=api_key,
        api_secret=api_secret,
        base_url=BINANCE_USDM_BASE_URL,
    )

    transport = AiohttpBinanceTransport()
    response: BinanceTransportResponse = await transport.send(signed)

    if response.status_code >= 400:
        payload = response.payload if isinstance(response.payload, dict) else {}
        print(
            f"[cancel-algo] WARNING: cancel clientAlgoId={client_order_id} "
            f"failed (HTTP {response.status_code}): {payload}",
            file=sys.stderr,
        )
        return BrokerCancelResult(
            exchange=ExchangeName.BINANCE,
            symbol=BINANCE_SYMBOL,
            ok=False,
            order_id=None,
            client_order_id=client_order_id,
            raw=str(payload),
        )

    print(f"[cancel-algo] cancelled clientAlgoId={client_order_id}")
    return BrokerCancelResult(
        exchange=ExchangeName.BINANCE,
        symbol=BINANCE_SYMBOL,
        ok=True,
        order_id=None,
        client_order_id=client_order_id,
        raw=response.payload,
    )


async def fetch_algo_open_orders(
    *,
    api_key: str,
    api_secret: str,
) -> list[dict[str, Any]]:
    """Fetch open algo orders via ``GET /fapi/v1/openAlgoOrders``.

    Returns the raw list of order dicts (may be empty).
    """
    signed = build_signed_request(
        method="GET",
        path=ALGO_OPEN_ORDERS_PATH,
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
            f"[algo-orders] WARNING: fetch failed (HTTP {response.status_code}): {payload}",
            file=sys.stderr,
        )
        return []

    if not isinstance(response.payload, list):
        return []

    orders: list[dict[str, Any]] = response.payload
    print(f"[algo-orders] {len(orders)} open algo order(s)")
    for o in orders:
        print(
            f"  algoId={o.get('algoId')} clientAlgoId={o.get('clientAlgoId')} "
            f"type={o.get('orderType')} side={o.get('side')} "
            f"qty={o.get('quantity')} trigger={o.get('triggerPrice')}"
        )
    return orders


async def cancel_smoke_algo_orders(
    *,
    api_key: str,
    api_secret: str,
) -> int:
    """Cancel all open algo orders whose clientAlgoId starts with RE_SMOKE_."""
    orders = await fetch_algo_open_orders(api_key=api_key, api_secret=api_secret)
    cancelled = 0
    for o in orders:
        cid = str(o.get("clientAlgoId") or "")
        if not cid.startswith(CLIENT_ORDER_ID_PREFIX):
            continue
        await cancel_algo_order_by_client_id(
            api_key=api_key,
            api_secret=api_secret,
            client_order_id=cid,
        )
        cancelled += 1
    return cancelled


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
    *,
    api_key: str,
    api_secret: str,
    quantity: Decimal,
    sl_price: Decimal,
    client_order_id: str,
) -> BrokerOrderResult:
    """Place a STOP_MARKET SELL SL order via Binance Algo Order API.

    This no longer uses ``BinanceBrokerClient.place_order()`` because
    the regular ``POST /fapi/v1/order`` endpoint rejects conditional
    order types (STOP_MARKET) with error -4120 since 2025-12-09.
    """
    print(f"[sl] STOP_MARKET SELL {quantity} @ trigger {sl_price} (cid={client_order_id})")
    return await place_stop_loss_algo_order(
        api_key=api_key,
        api_secret=api_secret,
        quantity=quantity,
        sl_price=sl_price,
        client_order_id=client_order_id,
    )


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
    client_order_id: str | None = None,
) -> BrokerOrderResult:
    """Market-close the LONG position.

    When *client_order_id* is None the request is sent without a clientOrderId
    — this is used as a fallback when the primary close-with-cid fails.
    """
    if client_order_id:
        print(f"[close] MARKET SELL {quantity} ETHUSDT (cid={client_order_id})")
    else:
        print(f"[close] MARKET SELL {quantity} ETHUSDT (no cid)")
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
    *,
    api_key: str | None = None,
    api_secret: str | None = None,
) -> None:
    """Best-effort cleanup: cancel smoke orders (regular + algo), then close any residual LONG.

    cleanup is only called after preflight confirmed there was no initial
    position, so any residual LONG is assumed to be smoke-opened and safe
    to close.

    When the primary close (with clientOrderId) fails the function falls back
    to a close without clientOrderId — this is the safety net for when the
    clientOrderId is too long or otherwise rejected by the exchange.
    """
    print("\n[cleanup] starting best-effort cleanup...")

    # --- Cancel regular smoke orders ---
    try:
        cancelled = await cancel_smoke_orders(client)
        print(f"[cleanup] cancelled {cancelled} regular smoke order(s)")
    except Exception as exc:
        print(f"[cleanup] WARNING: cancel regular orders raised: {exc}", file=sys.stderr)

    # --- Cancel algo smoke orders ---
    if api_key and api_secret:
        try:
            algo_cancelled = await cancel_smoke_algo_orders(
                api_key=api_key,
                api_secret=api_secret,
            )
            print(f"[cleanup] cancelled {algo_cancelled} algo smoke order(s)")
        except Exception as exc:
            print(
                f"[cleanup] WARNING: cancel algo orders raised: {exc}",
                file=sys.stderr,
            )
    else:
        print(
            "[cleanup] WARNING: no api_key/api_secret provided — cannot cancel algo orders",
            file=sys.stderr,
        )

    await asyncio.sleep(0.5)

    # --- Close residual position ---
    try:
        pos = await fetch_long_position(client)
    except Exception as exc:
        print(f"[cleanup] WARNING: fetch position raised: {exc}", file=sys.stderr)
        pos = None

    if pos is not None and pos.quantity > 0:
        close_succeeded = False

        # --- Primary close with short clientOrderId ---
        try:
            cid = _generate_client_order_id("cleanup_close")
            print(
                f"[cleanup] residual LONG position qty={pos.quantity}, "
                f"attempting close (cid={cid})..."
            )
            await close_long_position(client, pos.quantity, cid)
            await asyncio.sleep(1)
            pos_after = await fetch_long_position(client)
            if pos_after is not None and pos_after.quantity > 0:
                print(
                    f"[cleanup] WARNING: position still open after primary close "
                    f"(qty={pos_after.quantity})",
                    file=sys.stderr,
                )
            else:
                print("[cleanup] residual position closed (primary)")
                close_succeeded = True
        except Exception as exc:
            print(
                f"[cleanup] WARNING: primary close raised: {exc}",
                file=sys.stderr,
            )

        # --- Fallback close without clientOrderId ---
        if not close_succeeded:
            try:
                print(
                    "[cleanup] attempting fallback close without clientOrderId..."
                )
                await close_long_position(client, pos.quantity, client_order_id=None)
                await asyncio.sleep(1)
                pos_after = await fetch_long_position(client)
                if pos_after is not None and pos_after.quantity > 0:
                    print(
                        f"[cleanup] WARNING: position still open after fallback close "
                        f"(qty={pos_after.quantity})",
                        file=sys.stderr,
                    )
                else:
                    print("[cleanup] residual position closed (fallback)")
            except Exception as exc:
                print(
                    f"[cleanup] WARNING: fallback close raised: {exc}",
                    file=sys.stderr,
                )
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
    # 4. Place SL (via Algo Order API)
    # ------------------------------------------------------------------
    sl_price = entry_price * (1 - preflight.sl_pct)
    sl_price = (sl_price * 100).quantize(Decimal("1")) / 100  # round to 2 dp
    sl_result = await place_sl(
        api_key=preflight.api_key,
        api_secret=preflight.api_secret,
        quantity=opened_qty,
        sl_price=sl_price,
        client_order_id=sl_cid,
    )
    sl_algo_id = sl_result.order_id
    print(f"[smoke] step 4/7: SL algo placed @ trigger {sl_price} ✓")

    # ------------------------------------------------------------------
    # 5. Fetch open orders (regular)
    # ------------------------------------------------------------------
    open_orders = await fetch_open_orders(client)
    tp_found = any(
        (o.order_id == tp_order_id) or (o.client_order_id == tp_cid)
        for o in open_orders
    )
    if not tp_found:
        print("[smoke] WARNING: TP order not found in regular open orders", file=sys.stderr)

    # SL is an algo order — check via algo open orders endpoint
    algo_orders = await fetch_algo_open_orders(
        api_key=preflight.api_key,
        api_secret=preflight.api_secret,
    )
    sl_found = any(
        str(o.get("clientAlgoId", "")) == sl_cid
        for o in algo_orders
    )
    if not sl_found:
        print("[smoke] WARNING: SL algo order not found in algo open orders", file=sys.stderr)

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
    if sl_cid is not None:
        await cancel_algo_order_by_client_id(
            api_key=preflight.api_key,
            api_secret=preflight.api_secret,
            client_order_id=sl_cid,
        )
        cancelled_count += 1

    # Verify cancellation (regular orders only — algo orders checked separately)
    orders_after_cancel = await fetch_open_orders(client)
    residual_smoke = [
        o for o in orders_after_cancel
        if (o.client_order_id or "").startswith(CLIENT_ORDER_ID_PREFIX)
    ]
    # Also check residual algo orders
    try:
        algo_after_cancel = await fetch_algo_open_orders(
            api_key=preflight.api_key,
            api_secret=preflight.api_secret,
        )
        residual_algo_smoke = [
            o for o in algo_after_cancel
            if str(o.get("clientAlgoId", "")).startswith(CLIENT_ORDER_ID_PREFIX)
        ]
    except Exception:
        residual_algo_smoke = []
    if residual_smoke or residual_algo_smoke:
        print(
            f"[smoke] WARNING: {len(residual_smoke)} regular + "
            f"{len(residual_algo_smoke)} algo smoke order(s) still open after cancel",
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
    require_binance_live_preflight_for_smoke()
    api_key, api_secret = load_binance_credentials()

    # --- unified runtime config ---
    rt = load_unified_runtime_config()
    binance_symbol = validate_unified_config_for_binance(rt)

    # --- configuration (no network) ---
    max_notional = _read_positive_decimal_env(
        ENV_MAX_NOTIONAL, DEFAULT_MAX_NOTIONAL, alias=_ENV_MAX_NOTIONAL_ALIAS,
    )
    tp_pct = _read_positive_decimal_env(
        ENV_TP_PCT, DEFAULT_TP_PCT, alias=_ENV_TP_PCT_ALIAS,
    )
    sl_pct = _read_positive_decimal_env(
        ENV_SL_PCT, DEFAULT_SL_PCT, alias=_ENV_SL_PCT_ALIAS,
    )
    margin_buffer_multiplier = _read_positive_decimal_env(
        ENV_MARGIN_BUFFER_MULTIPLIER, DEFAULT_MARGIN_BUFFER_MULTIPLIER,
        alias=_ENV_MARGIN_BUFFER_MULTIPLIER_ALIAS,
    )

    # --- notional cap enforcement (no network) ---
    from src.exchanges.binance.live_preflight import load_binance_live_preflight_config
    preflight_cfg = load_binance_live_preflight_config(os.environ)
    preflight_max_order = preflight_cfg.max_order_notional_usdt
    if preflight_max_order is None:
        print(
            "ERROR: LIVE_MAX_ORDER_NOTIONAL_USDT must be configured for smoke test",
            file=sys.stderr,
        )
        raise SystemExit(1)
    preflight_max_position = preflight_cfg.max_position_notional_usdt
    if preflight_max_position is None:
        print(
            "ERROR: LIVE_MAX_POSITION_NOTIONAL_USDT must be configured for smoke test",
            file=sys.stderr,
        )
        raise SystemExit(1)
    require_requested_notional_cap(
        smoke_max_notional=max_notional,
        preflight_max_order_notional=preflight_max_order,
        preflight_max_position_notional=preflight_max_position,
    )

    # --- preflight (network) ---
    print("[preflight] checking position mode...")
    await require_one_way_position_mode(api_key, api_secret)

    print("[preflight] checking margin mode...")
    await require_isolated_margin(api_key, api_secret)

    # --- leverage (conditional) ---
    if allow_set_leverage():
        print(f"[preflight] setting initial leverage to {rt.leverage}x...")
        await set_initial_leverage(api_key, api_secret, rt.leverage)
    else:
        print("[preflight] set leverage skipped; validating existing leverage...")
        await require_existing_leverage(api_key, api_secret, rt.leverage)

    print("[preflight] fetching exchangeInfo...")
    filters = await fetch_exchange_info_filters()

    print("[preflight] fetching mark price...")
    mark_price = await fetch_mark_price()

    print("[preflight] fetching account balance...")
    available_balance = await fetch_account_balance(api_key, api_secret)

    # --- quantity calculation ---
    calculated_qty, calculated_notional = calculate_safe_quantity(
        mark_price=mark_price,
        max_notional=max_notional,
        filters=filters,
    )

    # --- calculated notional cap ---
    require_calculated_notional_cap(
        calculated_notional=calculated_notional,
        preflight_max_order_notional=preflight_max_order,
        preflight_max_position_notional=preflight_max_position,
    )

    # --- leverage-aware margin check ---
    estimated_margin, required_margin = calculate_required_margin_with_buffer(
        notional=calculated_notional,
        leverage=rt.leverage,
        buffer_multiplier=margin_buffer_multiplier,
    )

    if available_balance < required_margin:
        print(
            f"ERROR: insufficient USDT margin balance — "
            f"need ≈ {required_margin} "
            f"(notional={calculated_notional}, leverage={rt.leverage}x, "
            f"buffer={margin_buffer_multiplier}x), "
            f"have {available_balance}",
            file=sys.stderr,
        )
        return 1

    print(
        f"[preflight] margin check OK — "
        f"notional≈{calculated_notional}, leverage={rt.leverage}x, "
        f"estimated_margin≈{estimated_margin}, buffer={margin_buffer_multiplier}x, "
        f"required≈{required_margin}, available={available_balance}"
    )

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
        leverage=rt.leverage,
        margin_buffer_multiplier=margin_buffer_multiplier,
        estimated_initial_margin=estimated_margin,
        required_margin_with_buffer=required_margin,
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
        f"\n  Leverage:     {rt.leverage}x"
        f"\n  Est. Margin:  ≈ {estimated_margin} USDT"
        f"\n  Req. Margin:  ≈ {required_margin} USDT (buffer={margin_buffer_multiplier}x)"
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

    # --- pre-trade gates (require client) ---
    await require_no_existing_position(client)

    print("[preflight] cleaning old RE_SMOKE_ orders...")
    regular_cancelled = await cancel_smoke_orders(client)
    algo_cancelled = await cancel_smoke_algo_orders(
        api_key=api_key, api_secret=api_secret,
    )
    print(
        f"[preflight] old smoke cleanup OK | "
        f"regular={regular_cancelled} algo={algo_cancelled}"
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
        await cleanup(
            client,
            api_key=api_key,
            api_secret=api_secret,
        )

    if success:
        print("\n[smoke] ALL STEPS PASSED ✓")
        return 0
    else:
        print("\n[smoke] TEST FAILED — check output above", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
