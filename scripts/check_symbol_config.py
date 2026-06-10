#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI entry-point for dry-run symbol config check (F05).

Usage::

    python scripts/check_symbol_config.py --inst-id BTC-USDT-SWAP
    python scripts/check_symbol_config.py --inst-id BTC-USDT-SWAP --json
    python scripts/check_symbol_config.py --inst-id ETH-USDT-SWAP
    python scripts/check_symbol_config.py \\
        --symbol-config-dir config/symbols \\
        --inst-id BTC-USDT-SWAP

This script is a **safe config-check / dry-run preview** only.  It does
**not** start a worker, create a ``Trader``, connect to OKX, place
orders, send email, or write runtime state.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

# Ensure the project root is on sys.path so that ``config.*`` imports work
# when the script is invoked directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run symbol TOML config check — no worker, no OKX, no orders.",
    )
    parser.add_argument(
        "--symbol-config-dir",
        default="config/symbols",
        help="Directory containing per-symbol TOML files (default: config/symbols)",
    )
    parser.add_argument(
        "--inst-id",
        required=True,
        help="Instrument ID to check, e.g. BTC-USDT-SWAP",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output a JSON summary instead of human-readable text",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Late import — keep top-level imports minimal so that the script
    # source guard test only sees safe imports.
    # ------------------------------------------------------------------
    from config.symbol_config_check import check_symbol_config

    try:
        result = check_symbol_config(
            symbol_config_dir=Path(args.symbol_config_dir),
            inst_id=args.inst_id,
        )
    except Exception as exc:
        print(
            f"CONFIG_CHECK_FAILED | inst_id={args.inst_id} "
            f"error={type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    # -- human-readable output -------------------------------------------
    if not args.json_output:
        print("CONFIG_CHECK_OK")
        print(f"  inst_id:          {result.inst_id}")
        print(f"  enabled:          {result.enabled}")
        print(f"  live_trading:     {result.live_trading}")
        print(f"  contract_value:   {result.contract_value}")
        print(f"  min_contracts:    {result.min_contracts}")
        print(f"  contract_precision: {result.contract_precision}")
        print(f"  price_precision:  {result.price_precision}")
        print(f"  sidecar_enabled:  {result.sidecar_enabled}")
        print(f"  middle_bucket_split_enabled: {result.middle_bucket_split_enabled}")
        print(f"  trader_preview.inst_id: {result.mapped.trader_preview.inst_id}")
        print(f"  trader_preview.contract_value: {result.mapped.trader_preview.contract_value}")
        print(f"  safe_for_config_check_only: {result.safe_for_config_check_only}")
        print()
        print(
            "This check does not start a worker, does not create a Trader, "
            "and does not connect to OKX."
        )
        return 0

    # -- JSON output -----------------------------------------------------
    json.dump(result.to_summary_dict(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
