#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-symbol live startup preflight CLI (G09d).

Usage::

    python scripts/check_multi_symbol_live_startup.py

This script runs a **read-only** safety check that verifies ETH/BTC
multi-worker live configuration is internally consistent.

It does **not**:
* Start a Trader or worker.
* Connect to OKX (public or private).
* Place orders.
* Modify config files or runtime state.
* Send email.

Exit code 0 → preflight passed (ok=True).
Exit code 1 → preflight failed (errors found).
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> int:
    # Late imports keep the script-startup footprint minimal.
    from config.env_loader import load_env_config
    from src.live.startup_checks.multi_symbol_live_preflight import (
        run_multi_symbol_live_preflight,
    )

    # Load .env into os.environ so that the preflight sees real config.
    load_env_config()

    result = run_multi_symbol_live_preflight(
        env=None,  # read os.environ
        strict_requested_symbols=True,
    )

    # ── Header ──────────────────────────────────────────────────────────────
    print()
    print("RECLAIM_MULTI_SYMBOL_LIVE_PREFLIGHT")
    print(f"requested: {','.join(result.requested_symbols) or '(none)'}")
    print(f"enabled:   {','.join(result.enabled_symbols) or '(none)'}")

    if result.skipped_disabled_symbols:
        print(f"skipped disabled: {','.join(result.skipped_disabled_symbols)}")

    print("workers:")
    for wr in result.worker_results:
        print(
            f"  * {wr.symbol} "
            f"mode={wr.worker_mode} "
            f"metadata_ok={wr.metadata_ok} "
            f"market_settings_ok={wr.market_settings_ok} "
            f"heartbeat={wr.heartbeat_path}"
        )

    if result.warnings:
        print("warnings:")
        for w in result.warnings:
            print(f"  * {w}")

    if result.errors:
        print("errors:")
        for e in result.errors:
            print(f"  * {e}")

    if result.ok:
        print()
        print("PREFLIGHT OK")
        return 0
    else:
        print()
        print("PREFLIGHT FAILED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
