#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for ``scripts/check_symbol_config.py`` — dry-run config-check CLI (F05)."""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

import pytest

from scripts.check_symbol_config import main

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT_PATH = _PROJECT_ROOT / "scripts" / "check_symbol_config.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_main(argv: list[str]) -> tuple[int, str, str]:
    """Run ``main(argv)``, capture stdout/stderr, return (rc, stdout, stderr)."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        rc = main(argv)
        return rc, sys.stdout.getvalue(), sys.stderr.getvalue()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


# ---------------------------------------------------------------------------
# 1. BTC text output succeeds
# ---------------------------------------------------------------------------


def test_main_btc_text_succeeds() -> None:
    """--inst-id BTC-USDT-SWAP returns 0 and prints expected text markers."""
    rc, stdout, stderr = _run_main(["--inst-id", "BTC-USDT-SWAP"])
    assert rc == 0, f"Expected rc=0, got {rc}; stderr={stderr}"
    assert stderr == ""
    assert "CONFIG_CHECK_OK" in stdout
    assert "BTC-USDT-SWAP" in stdout
    assert ("enabled:          False" in stdout
            or "enabled: false" in stdout
            or "enabled=False" in stdout
            or "enabled=false" in stdout)
    assert ("live_trading:     False" in stdout
            or "live_trading: false" in stdout
            or "live_trading=False" in stdout
            or "live_trading=false" in stdout)
    assert "does not start" in stdout
    assert "does not create" in stdout


# ---------------------------------------------------------------------------
# 2. BTC JSON output succeeds
# ---------------------------------------------------------------------------


def test_main_btc_json_succeeds() -> None:
    """--inst-id BTC-USDT-SWAP --json returns 0 and valid JSON summary."""
    rc, stdout, stderr = _run_main(["--inst-id", "BTC-USDT-SWAP", "--json"])
    assert rc == 0, f"Expected rc=0, got {rc}; stderr={stderr}"
    assert stderr == ""
    data = json.loads(stdout)
    assert data["inst_id"] == "BTC-USDT-SWAP"
    assert data["enabled"] is False
    assert data["live_trading"] is False
    assert data["contract_value"] == "0.01"
    assert data["price_precision"] == "0.1"
    assert data["safe_for_config_check_only"] is True
    tp = data["trader_preview"]
    assert tp["inst_id"] == "BTC-USDT-SWAP"
    assert tp["contract_value"] == "0.01"
    assert tp["live_trading"] is False


# ---------------------------------------------------------------------------
# 3. ETH text succeeds
# ---------------------------------------------------------------------------


def test_main_eth_text_succeeds() -> None:
    """--inst-id ETH-USDT-SWAP returns 0."""
    rc, stdout, stderr = _run_main(["--inst-id", "ETH-USDT-SWAP"])
    assert rc == 0, f"Expected rc=0, got {rc}; stderr={stderr}"
    assert "CONFIG_CHECK_OK" in stdout


# ---------------------------------------------------------------------------
# 4. Unsupported symbol returns 1
# ---------------------------------------------------------------------------


def test_main_unsupported_symbol_returns_1() -> None:
    """--inst-id SOL-USDT-SWAP returns 1 and prints CONFIG_CHECK_FAILED."""
    rc, stdout, stderr = _run_main(["--inst-id", "SOL-USDT-SWAP"])
    assert rc == 1, f"Expected rc=1, got {rc}"
    assert "CONFIG_CHECK_FAILED" in stderr


# ---------------------------------------------------------------------------
# 5. Missing --inst-id fails
# ---------------------------------------------------------------------------


def test_main_missing_inst_id_fails() -> None:
    """Calling main without --inst-id should raise SystemExit (argparse error)."""
    with pytest.raises(SystemExit):
        main([])


# ---------------------------------------------------------------------------
# 6. Script source guard — no live/trader/network imports
# ---------------------------------------------------------------------------


def test_script_source_guard_forbidden() -> None:
    """scripts/check_symbol_config.py must NOT contain any live/trader/network tokens."""
    import io
    import tokenize

    source_bytes = _SCRIPT_PATH.read_bytes()

    forbidden = {
        "Trader",
        "SymbolWorkerApp",
        "run_symbol_worker",
        "ReclaimSupervisor",
        "LIVE_TRADING",
        "OKX_API_KEY",
        "load_dotenv",
        "asyncio",
        "requests",
        "httpx",
        "websocket",
        "send_email",
    }

    names_in_code: set[str] = set()
    try:
        for tok in tokenize.tokenize(io.BytesIO(source_bytes).readline):
            if tok.type == tokenize.NAME:
                names_in_code.add(tok.string)
    except tokenize.TokenError:
        pass

    overlap = forbidden & names_in_code
    assert not overlap, (
        f"scripts/check_symbol_config.py must NOT contain any of {sorted(overlap)} "
        f"as code identifiers — this is a config-check only script"
    )


def test_script_source_guard_required() -> None:
    """scripts/check_symbol_config.py must contain expected safe tokens."""
    source = _SCRIPT_PATH.read_text(encoding="utf-8")
    required = [
        "argparse",
        "check_symbol_config",
        "--inst-id",
        "--symbol-config-dir",
    ]
    for token in required:
        assert token in source, (
            f"scripts/check_symbol_config.py must contain '{token}'"
        )
