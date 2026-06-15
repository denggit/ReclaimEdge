#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/15
@File       : binance_small_live_launch_check.py
@Description: Binance small live launch checklist — final pre-launch safety gate.

This script is a **read-only** check run BEFORE the main live trading launcher.
It validates env, preflight, notional caps, sidecar, local state, and trader
sizing.  Any dangerous configuration is blocked.

Usage::

    PYTHONPATH=. python scripts/binance_small_live_launch_check.py
    PYTHONPATH=. python scripts/binance_small_live_launch_check.py --json
    PYTHONPATH=. python scripts/binance_small_live_launch_check.py --allow-sidecar

Rules enforced
--------------
1. Does NOT place / cancel / TP any order.
2. Does NOT start a websocket.
3. Does NOT fetch balance / position / configure instrument.
4. Does NOT call the main live trading launcher.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.exchanges.binance.live_preflight import (
    build_binance_live_preflight_report,
    load_binance_live_preflight_config,
)
from src.exchanges.runtime_adapter_factory import create_exchange_runtime_adapters
from src.exchanges.runtime_config import load_unified_runtime_config
from src.reporting.live_state_store import DEFAULT_STATE_PATH, LiveStateStore

# ---------------------------------------------------------------------------
# Default hard caps for small live
# ---------------------------------------------------------------------------

SMALL_LIVE_MAX_ALLOWED_ORDER_NOTIONAL_USDT: int = 20
SMALL_LIVE_MAX_ALLOWED_POSITION_NOTIONAL_USDT: int = 50

_TRUTHY: frozenset[str] = frozenset({"1", "true", "yes", "y", "on"})


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalStateStatus:
    """Result of loading and inspecting the local live_state.json."""

    status: str  # "absent" | "flat" | "has_position"
    has_open_position: bool
    startup_force_tp_reconcile: bool
    details: dict[str, Any]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the small live launch checklist."""
    parser = argparse.ArgumentParser(
        description="Binance small live launch checklist — read-only safety gate",
    )
    parser.add_argument(
        "--state-path",
        default=str(DEFAULT_STATE_PATH),
        help="Path to live_state.json (default: data/trade_journal/live_state.json)",
    )
    parser.add_argument(
        "--max-allowed-order-notional",
        type=int,
        default=SMALL_LIVE_MAX_ALLOWED_ORDER_NOTIONAL_USDT,
        help=(
            "Max allowed order notional USDT "
            f"(default: {SMALL_LIVE_MAX_ALLOWED_ORDER_NOTIONAL_USDT})"
        ),
    )
    parser.add_argument(
        "--max-allowed-position-notional",
        type=int,
        default=SMALL_LIVE_MAX_ALLOWED_POSITION_NOTIONAL_USDT,
        help=(
            "Max allowed position notional USDT "
            f"(default: {SMALL_LIVE_MAX_ALLOWED_POSITION_NOTIONAL_USDT})"
        ),
    )
    parser.add_argument(
        "--allow-sidecar",
        action="store_true",
        help="Allow SIDECAR_ENABLED=true (WARNING_SIDECAR_ENABLED will still be emitted)",
    )
    parser.add_argument(
        "--allow-existing-local-position",
        action="store_true",
        help="Allow existing local position from a previous session",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of human-readable text",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Local state
# ---------------------------------------------------------------------------


def load_local_state_status(path: Path) -> LocalStateStatus:
    """Load live_state.json and classify its status.

    Returns
    -------
    LocalStateStatus
        - ``status="absent"`` — file does not exist.
        - ``status="flat"`` — file exists but no open position.
        - ``status="has_position"`` — file exists with an open position.
    """
    store = LiveStateStore(path)
    state = store.load()

    if state is None:
        return LocalStateStatus(
            status="absent",
            has_open_position=False,
            startup_force_tp_reconcile=False,
            details={},
        )

    side = state.side
    layers = state.layers
    core_eth_qty = state.core_eth_qty
    position_cost_remaining_qty = state.position_cost_remaining_qty

    has_position: bool = (
        side is not None
        and side in ("LONG", "SHORT")
        and (layers > 0 or core_eth_qty > 0 or position_cost_remaining_qty > 0)
    )

    details: dict[str, Any] = {
        "side": side,
        "layers": layers,
        "core_eth_qty": core_eth_qty,
        "position_cost_remaining_qty": position_cost_remaining_qty,
    }

    if has_position:
        return LocalStateStatus(
            status="has_position",
            has_open_position=True,
            startup_force_tp_reconcile=bool(state.startup_force_tp_reconcile),
            details=details,
        )

    return LocalStateStatus(
        status="flat",
        has_open_position=False,
        startup_force_tp_reconcile=False,
        details=details,
    )


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------


def check_small_live_caps(
    env: Mapping[str, str],
    args: argparse.Namespace,
) -> list[str]:
    """Check that live notional caps do not exceed small-live limits.

    Reads the resolved env values (LIVE_* primary with BINANCE_* fallback)
    and compares them against the CLI-overridable hard caps.
    """
    preflight_config = load_binance_live_preflight_config(env)
    blocking: list[str] = []

    max_order = preflight_config.max_order_notional_usdt
    max_position = preflight_config.max_position_notional_usdt

    max_allowed_order = Decimal(str(args.max_allowed_order_notional))
    max_allowed_position = Decimal(str(args.max_allowed_position_notional))

    if max_order is not None and max_order > max_allowed_order:
        blocking.append("LIVE_MAX_ORDER_NOTIONAL_TOO_HIGH")

    if max_position is not None and max_position > max_allowed_position:
        blocking.append("LIVE_MAX_POSITION_NOTIONAL_TOO_HIGH")

    return blocking


def check_sidecar(
    env: Mapping[str, str],
    allow_sidecar: bool,
) -> tuple[list[str], list[str]]:
    """Check SIDECAR_ENABLED.

    Returns
    -------
    (blocking_reasons, warnings)
    """
    blocking: list[str] = []
    warnings: list[str] = []

    sidecar_raw: str = env.get("SIDECAR_ENABLED", "").strip().lower()
    sidecar_enabled: bool = sidecar_raw in _TRUTHY

    if sidecar_enabled:
        if not allow_sidecar:
            blocking.append("SIDECAR_ENABLED_FOR_FIRST_BINANCE_LIVE")
        else:
            warnings.append("WARNING_SIDECAR_ENABLED")

    return blocking, warnings


def check_trader_sizing(trader: Any) -> tuple[list[str], dict[str, str]]:
    """Check that Trader sizing attributes match Binance expectations.

    Returns
    -------
    (blocking_reasons, runtime_dict)
    """
    blocking: list[str] = []
    runtime: dict[str, str] = {}

    multiplier = trader.contract_multiplier
    precision = trader.contract_precision
    min_contracts = trader.min_contracts

    try:
        qty_check = trader.eth_qty_to_contracts(Decimal("0.05"))
        qty_check_str = str(qty_check)
    except Exception:
        qty_check_str = "ERROR"

    runtime["contract_multiplier"] = str(multiplier)
    runtime["contract_precision"] = str(precision)
    runtime["min_contracts"] = str(min_contracts)
    runtime["qty_check_0_05_eth"] = qty_check_str

    sizing_valid: bool = (
        multiplier == Decimal("1")
        and precision == Decimal("0.001")
        and min_contracts == Decimal("0.001")
        and qty_check_str == "0.05"
    )

    if not sizing_valid:
        blocking.append("BINANCE_TRADER_SIZING_INVALID")

    return blocking, runtime


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _emit_blocked(
    args: argparse.Namespace,
    exchange: str,
    symbol: str,
    blocking_reasons: list[str],
    warnings: list[str],
    *,
    preflight_ok: bool,
) -> None:
    """Emit blocked output (text or JSON) and return."""
    if args.json:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "exchange": exchange,
                    "symbol": symbol,
                    "blocking_reasons": blocking_reasons,
                    "warnings": warnings,
                }
            )
        )
    else:
        print("BINANCE_SMALL_LIVE_LAUNCH_BLOCKED")
        print(f"exchange={exchange}")
        print(f"symbol={symbol}")
        print(f"blocking_reasons={blocking_reasons}")
        for w in warnings:
            print(w)


def _emit_preflight_blocked(
    args: argparse.Namespace,
    exchange: str,
    symbol: str,
    blocking_reasons: list[str],
) -> None:
    """Emit preflight-blocked output (text or JSON)."""
    if args.json:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "exchange": exchange,
                    "symbol": symbol,
                    "blocking_reasons": blocking_reasons,
                    "warnings": [],
                }
            )
        )
    else:
        print("BINANCE_SMALL_LIVE_PREFLIGHT_BLOCKED")
        print(f"exchange={exchange}")
        print(f"symbol={symbol}")
        print(f"blocking_reasons={blocking_reasons}")


def _emit_ready(
    args: argparse.Namespace,
    config: Any,
    preflight_config: Any,
    sidecar_enabled: bool,
    local_state: LocalStateStatus,
    cap_blocking: list[str],
    sidecar_blocking: list[str],
    sizing_blocking: list[str],
    sizing_runtime: dict[str, str],
    warnings: list[str],
) -> None:
    """Emit ready output (text or JSON)."""
    exchange: str = config.exchange.value
    symbol: str = config.binance_symbol

    # local_state_status for display
    if local_state.status in ("absent", "flat"):
        display_state: str = "flat_or_absent"
    else:
        display_state = local_state.status

    checks: dict[str, bool] = {
        "small_live_caps_ok": len(cap_blocking) == 0,
        "sidecar_ok": len(sidecar_blocking) == 0,
        "local_state_ok": True,  # already passed
        "trader_sizing_ok": len(sizing_blocking) == 0,
    }

    side_effects: dict[str, bool] = {
        "orders_executed": False,
        "websocket_started": False,
    }

    if args.json:
        print(
            json.dumps(
                {
                    "status": "ready",
                    "exchange": exchange,
                    "symbol": symbol,
                    "preflight_ok": True,
                    "checks": checks,
                    "runtime": sizing_runtime,
                    "side_effects": side_effects,
                    "warnings": warnings,
                }
            )
        )
    else:
        print("BINANCE_SMALL_LIVE_LAUNCH_READY")
        print(f"exchange={exchange}")
        print(f"symbol={symbol}")
        print(f"trade_asset={config.trade_asset}")
        print(f"quote_asset={config.quote_asset}")
        print(f"market_type={config.market_type}")
        print(f"margin_mode={config.margin_mode}")
        print(f"position_mode={config.position_mode}")
        print(f"leverage={config.leverage}")
        print(f"live_enabled={str(preflight_config.live_enabled).lower()}")
        print(f"live_allow_orders={str(preflight_config.allow_orders).lower()}")
        print(f"max_order_notional_usdt={preflight_config.max_order_notional_usdt}")
        print(f"max_position_notional_usdt={preflight_config.max_position_notional_usdt}")
        print(f"sidecar_enabled={str(sidecar_enabled).lower()}")
        print(f"contract_multiplier={sizing_runtime['contract_multiplier']}")
        print(f"contract_precision={sizing_runtime['contract_precision']}")
        print(f"min_contracts={sizing_runtime['min_contracts']}")
        print(f"qty_check_0_05_eth={sizing_runtime['qty_check_0_05_eth']}")
        print(f"local_state_status={display_state}")
        print("orders_executed=false")
        print("websocket_started=false")
        for w in warnings:
            print(w)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the Binance small live launch checklist.

    Returns
    -------
    int
        0 — ready to launch.
        1 — config error (e.g. invalid TRADE_ASSET).
        2 — blocked (dangerous config).
        3 — wrong exchange (not binance).
    """
    args = parse_args(argv)
    env: Mapping[str, str] = dict(os.environ)
    warnings: list[str] = []
    blocking_reasons: list[str] = []

    # ── 1. Load unified runtime config ──────────────────────────────────
    try:
        config = load_unified_runtime_config(env)
    except ValueError as exc:
        if args.json:
            print(
                json.dumps(
                    {
                        "status": "config_error",
                        "exchange": env.get("EXCHANGE", "okx").strip().lower(),
                        "error": f"BINANCE_SMALL_LIVE_CONFIG_ERROR: {exc}",
                    }
                )
            )
        else:
            print("BINANCE_SMALL_LIVE_CONFIG_ERROR")
            print(f"error={exc}")
        return 1

    exchange: str = config.exchange.value
    symbol: str = config.binance_symbol

    # ── 2. Exchange must be binance ─────────────────────────────────────
    if not config.is_binance:
        if args.json:
            print(
                json.dumps(
                    {
                        "status": "wrong_exchange",
                        "exchange": exchange,
                        "error": "BINANCE_SMALL_LIVE_WRONG_EXCHANGE",
                    }
                )
            )
        else:
            print("BINANCE_SMALL_LIVE_WRONG_EXCHANGE")
            print(f"exchange={exchange}")
        return 3

    # ── 3. Preflight check (reuse existing) ─────────────────────────────
    preflight = build_binance_live_preflight_report(
        env, orders_globally_enabled=True
    )
    if not preflight.ok:
        _emit_preflight_blocked(
            args, exchange, symbol, list(preflight.blocking_reasons)
        )
        return 2

    preflight_config = preflight.config

    # ── 4. Small live notional caps ─────────────────────────────────────
    cap_blocking: list[str] = check_small_live_caps(env, args)
    blocking_reasons.extend(cap_blocking)

    # ── 5. Sidecar check ────────────────────────────────────────────────
    sidecar_blocking, sidecar_warnings = check_sidecar(env, args.allow_sidecar)
    blocking_reasons.extend(sidecar_blocking)
    warnings.extend(sidecar_warnings)

    # ── 6. Local state check ────────────────────────────────────────────
    state_path = Path(args.state_path)
    local_state = load_local_state_status(state_path)

    if local_state.has_open_position:
        if not args.allow_existing_local_position:
            blocking_reasons.append("LOCAL_STATE_HAS_OPEN_POSITION")
        elif not local_state.startup_force_tp_reconcile:
            blocking_reasons.append(
                "EXISTING_POSITION_REQUIRES_STARTUP_FORCE_TP_RECONCILE"
            )

    # If anything blocked so far, stop before creating adapters
    if blocking_reasons:
        _emit_blocked(
            args, exchange, symbol, blocking_reasons, warnings, preflight_ok=True
        )
        return 2

    # ── 7. Create runtime adapters (no network) ─────────────────────────
    try:
        adapters = create_exchange_runtime_adapters(config, env)
    except Exception as exc:
        blocking_reasons.append(f"RUNTIME_ADAPTER_CREATION_FAILED: {exc}")
        _emit_blocked(
            args, exchange, symbol, blocking_reasons, warnings, preflight_ok=True
        )
        return 2

    trader = adapters.trader
    if trader is None:
        blocking_reasons.append("TRADER_IS_NONE")
        _emit_blocked(
            args, exchange, symbol, blocking_reasons, warnings, preflight_ok=True
        )
        return 2

    # ── 8. Trader sizing check ─────────────────────────────────────────
    sizing_blocking, sizing_runtime = check_trader_sizing(trader)
    if sizing_blocking:
        blocking_reasons.extend(sizing_blocking)
        _emit_blocked(
            args, exchange, symbol, blocking_reasons, warnings, preflight_ok=True
        )
        return 2

    # ── 9. All checks passed ────────────────────────────────────────────
    sidecar_raw: str = env.get("SIDECAR_ENABLED", "").strip().lower()
    sidecar_enabled: bool = sidecar_raw in _TRUTHY

    _emit_ready(
        args,
        config,
        preflight_config,
        sidecar_enabled,
        local_state,
        cap_blocking,
        sidecar_blocking,
        sizing_blocking,
        sizing_runtime,
        warnings,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
