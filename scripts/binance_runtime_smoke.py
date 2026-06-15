#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/15
@File       : binance_runtime_smoke.py
@Description: Binance runtime smoke / live preflight safety check.

This script is the LAST safety gate before launching a real Binance runtime
on a server.  It MUST be run and pass before any live trading process starts.

What this script does:
  1. Loads the unified runtime config and confirms EXCHANGE=binance.
  2. Runs the Binance live preflight (no network).
  3. If blocked: prints blocked reasons and exits.
  4. If ready: creates runtime adapters, validates types and sizing,
     prints a ready report, and exits.

What this script NEVER does:
  - Place / cancel orders
  - Place / cancel TP / SL
  - Start websocket streams
  - Fetch balance / position
  - Configure instrument / leverage

Usage::

    # Default — blocked unless all live gates are satisfied
    PYTHONPATH=. python scripts/binance_runtime_smoke.py

    # Verify the script correctly blocks when gates are unsatisfied
    PYTHONPATH=. python scripts/binance_runtime_smoke.py --expect-blocked

    # Verify runtime adapters are created when all gates pass
    EXCHANGE=binance \\
    EXCHANGE_API_KEY=test-key \\
    EXCHANGE_API_SECRET=test-secret \\
    LIVE_ENABLED=true \\
    LIVE_ALLOW_ORDERS=true \\
    LIVE_CONFIRMATION=I_UNDERSTAND_EXCHANGE_LIVE_TRADING \\
    LIVE_MAX_ORDER_NOTIONAL_USDT=25 \\
    LIVE_MAX_POSITION_NOTIONAL_USDT=30 \\
    LIVE_LEVERAGE=20 \\
    PYTHONPATH=. python scripts/binance_runtime_smoke.py --expect-ready

    # JSON output for automation
    PYTHONPATH=. python scripts/binance_runtime_smoke.py --json --expect-ready
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from decimal import Decimal

from src.data_feed.binance.market_data_client import BinanceMarketDataClient
from src.exchanges.binance.live_preflight import (
    build_binance_live_preflight_report,
    BinanceLivePreflightReport,
)
from src.exchanges.binance.trading_client import BinanceTradingClient
from src.exchanges.models import ExchangeName
from src.exchanges.runtime_adapter_factory import create_exchange_runtime_adapters
from src.exchanges.runtime_config import load_unified_runtime_config
from src.execution.trader import Trader


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Binance runtime smoke / live preflight safety check.",
    )
    parser.add_argument(
        "--expect-blocked",
        action="store_true",
        default=False,
        help="Expect the preflight to be blocked.  Exit 0 when blocked "
        "(instead of 2), so this can be used to verify the safety gate.",
    )
    parser.add_argument(
        "--expect-ready",
        action="store_true",
        default=False,
        help="Expect the preflight to pass and runtime adapters to be created.  "
        "Use with a full live-gate env to verify adapter wiring.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        dest="json_output",
        help="Output a JSON summary instead of human-readable text.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------


def _format_blocked_text(report: BinanceLivePreflightReport) -> str:
    """Format a human-readable blocked message."""
    lines: list[str] = [
        "BINANCE_RUNTIME_SMOKE_BLOCKED",
        f"exchange={report.config.exchange}",
        "ok=false",
        "blocking_reasons=[",
    ]
    for reason in report.blocking_reasons:
        lines.append(f"  {reason}")
    lines.append("]")
    return "\n".join(lines)


def _format_ready_text(
    symbol: str,
    market_data_client_type: str,
    trading_client_type: str,
    trader_type: str,
    contract_multiplier: str,
    contract_precision: str,
    min_contracts: str,
    qty_check: str,
    sources: dict[str, str] | None = None,
    warnings: tuple[str, ...] = (),
) -> str:
    """Format a human-readable ready message."""
    lines: list[str] = [
        "BINANCE_RUNTIME_SMOKE_READY",
        "exchange=binance",
        f"symbol={symbol}",
        f"market_data_client={market_data_client_type}",
        f"trading_client={trading_client_type}",
        f"trader={trader_type}",
        f"trader_contract_multiplier={contract_multiplier}",
        f"trader_contract_precision={contract_precision}",
        f"trader_min_contracts={min_contracts}",
        f"qty_check_0_05_eth={qty_check}",
        "orders_executed=false",
        "websocket_started=false",
    ]
    if sources:
        lines.append(f"live_enabled_source={sources.get('live_enabled', '')}")
        lines.append(f"live_allow_orders_source={sources.get('allow_orders', '')}")
        lines.append(f"max_order_notional_source={sources.get('max_order_notional', '')}")
        lines.append(f"max_position_notional_source={sources.get('max_position_notional', '')}")
        lines.append(f"live_leverage_source={sources.get('leverage', '')}")
    for w in warnings:
        lines.append(w)
    return "\n".join(lines)


def _build_blocked_json(report: BinanceLivePreflightReport, symbol: str) -> dict:
    """Build JSON dict for blocked status."""
    return {
        "status": "blocked",
        "exchange": "binance",
        "symbol": symbol,
        "preflight_ok": False,
        "blocking_reasons": list(report.blocking_reasons),
    }


def _build_ready_json(
    symbol: str,
    market_data_client_type: str,
    trading_client_type: str,
    trader_type: str,
    contract_multiplier: str,
    contract_precision: str,
    min_contracts: str,
    qty_check: str,
    sources: dict[str, str] | None = None,
    warnings: tuple[str, ...] = (),
) -> dict:
    """Build JSON dict for ready status."""
    result: dict = {
        "status": "ready",
        "exchange": "binance",
        "symbol": symbol,
        "preflight_ok": True,
        "blocking_reasons": [],
        "adapters": {
            "market_data_client": market_data_client_type,
            "trading_client": trading_client_type,
            "trader": trader_type,
        },
        "trader_sizing": {
            "contract_multiplier": contract_multiplier,
            "contract_precision": contract_precision,
            "min_contracts": min_contracts,
            "qty_check_0_05_eth": qty_check,
        },
        "side_effects": {
            "orders_executed": False,
            "websocket_started": False,
        },
    }
    if sources:
        result["sources"] = sources
    if warnings:
        result["warnings"] = list(warnings)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the Binance runtime smoke preflight check.

    Returns an exit code (0 = pass, 1 = config error, 2 = blocked,
    3 = wrong exchange, 4 = expectation failed / invalid args).
    """
    args = parse_args(argv)
    env: Mapping[str, str] = dict(os.environ)

    # --- 0. Validate expectation flags (mutually exclusive) ---
    if args.expect_blocked and args.expect_ready:
        msg = "BINANCE_RUNTIME_SMOKE_INVALID_EXPECTATION_FLAGS"
        if args.json_output:
            print(json.dumps({
                "status": "invalid_args",
                "error": msg,
            }))
        else:
            print(msg)
        return 4

    # --- 1. Load unified runtime config (with error handling) ---
    try:
        config = load_unified_runtime_config(env)
    except ValueError as exc:
        raw_msg = str(exc)
        exchange_value: str = env.get("EXCHANGE", "unknown")
        if "Unsupported EXCHANGE" in raw_msg:
            full_msg = (
                f"BINANCE_RUNTIME_SMOKE_WRONG_EXCHANGE: "
                f"expected EXCHANGE=binance, got {exchange_value}"
            )
            if args.json_output:
                print(json.dumps({
                    "status": "wrong_exchange",
                    "exchange": exchange_value,
                    "error": full_msg,
                }))
            else:
                print(full_msg)
            return 3
        else:
            full_msg = f"BINANCE_RUNTIME_SMOKE_CONFIG_ERROR: {raw_msg}"
            if args.json_output:
                print(json.dumps({
                    "status": "config_error",
                    "exchange": "binance",
                    "error": full_msg,
                }))
            else:
                print(full_msg)
            return 1

    # --- 2. Exchange check ---
    if config.exchange != ExchangeName.BINANCE:
        msg = (
            f"BINANCE_RUNTIME_SMOKE_WRONG_EXCHANGE: "
            f"expected EXCHANGE=binance, got {config.exchange.value}"
        )
        if args.json_output:
            print(json.dumps({
                "status": "wrong_exchange",
                "exchange": config.exchange.value,
                "error": msg,
            }))
        else:
            print(msg)
        return 3

    symbol: str = config.binance_symbol

    # --- 3. Run preflight ---
    report = build_binance_live_preflight_report(
        env,
        orders_globally_enabled=True,
    )

    if not report.ok:
        # Blocked path
        if args.expect_ready:
            if args.json_output:
                data = _build_blocked_json(report, symbol)
                data["expected_ready"] = True
                data["error"] = (
                    "BINANCE_RUNTIME_SMOKE_EXPECTED_READY_BUT_BLOCKED"
                )
                print(json.dumps(data))
            else:
                print(_format_blocked_text(report))
                print(
                    "BINANCE_RUNTIME_SMOKE_EXPECTED_READY_BUT_BLOCKED"
                )
            return 2

        if args.json_output:
            print(json.dumps(_build_blocked_json(report, symbol)))
        else:
            print(_format_blocked_text(report))

        if args.expect_blocked:
            return 0
        return 2

    # --- 4. Preflight OK — check for --expect-blocked before adapters ---
    if args.expect_blocked:
        msg = "BINANCE_RUNTIME_SMOKE_EXPECTED_BLOCKED_BUT_READY"
        if args.json_output:
            print(json.dumps({
                "status": "expectation_failed",
                "exchange": "binance",
                "symbol": symbol,
                "preflight_ok": True,
                "error": msg,
            }))
        else:
            print(msg)
        return 4

    # --- 5. Create runtime adapters (no network) ---
    adapters = create_exchange_runtime_adapters(config=config, env=env)

    # --- 6. Validate adapter types ---
    market_data_client = adapters.market_data_client
    trading_client = adapters.trading_client
    trader = adapters.trader

    assert isinstance(market_data_client, BinanceMarketDataClient), (
        f"Expected BinanceMarketDataClient, got {type(market_data_client).__name__}"
    )
    assert isinstance(trading_client, BinanceTradingClient), (
        f"Expected BinanceTradingClient, got {type(trading_client).__name__}"
    )
    assert isinstance(trader, Trader), (
        f"Expected Trader, got {type(trader).__name__}"
    )

    market_data_client_type: str = type(market_data_client).__name__
    trading_client_type: str = type(trading_client).__name__
    trader_type: str = type(trader).__name__

    # --- 7. Validate trader sizing ---
    contract_multiplier: str = str(trader.contract_multiplier)
    contract_precision: str = str(trader.contract_precision)
    min_contracts: str = str(trader.min_contracts)

    # qty_check: 0.05 ETH → contracts for Binance (multiplier 1 = identity)
    qty_contracts: Decimal = trader.eth_qty_to_contracts(Decimal("0.05"))
    qty_check: str = str(qty_contracts)

    # --- 8. Output ---
    # Build sources from preflight config
    pf_cfg = report.config
    sources: dict[str, str] = {
        "live_enabled": pf_cfg.live_enabled_source,
        "allow_orders": pf_cfg.allow_orders_source,
        "confirmation": pf_cfg.confirmation_source,
        "max_order_notional": pf_cfg.max_order_notional_source,
        "max_position_notional": pf_cfg.max_position_notional_source,
        "leverage": pf_cfg.leverage_source,
    }
    pf_warnings: tuple[str, ...] = report.warnings

    if args.json_output:
        print(json.dumps(_build_ready_json(
            symbol=symbol,
            market_data_client_type=market_data_client_type,
            trading_client_type=trading_client_type,
            trader_type=trader_type,
            contract_multiplier=contract_multiplier,
            contract_precision=contract_precision,
            min_contracts=min_contracts,
            qty_check=qty_check,
            sources=sources,
            warnings=pf_warnings,
        )))
    else:
        print(_format_ready_text(
            symbol=symbol,
            market_data_client_type=market_data_client_type,
            trading_client_type=trading_client_type,
            trader_type=trader_type,
            contract_multiplier=contract_multiplier,
            contract_precision=contract_precision,
            min_contracts=min_contracts,
            qty_check=qty_check,
            sources=sources,
            warnings=pf_warnings,
        ))

    return 0


if __name__ == "__main__":
    sys.exit(main())
