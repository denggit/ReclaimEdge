#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression tests ensuring ``scripts/run_boll_cvd_live.py`` calls
``build_live_symbol_runtime_configs`` exactly once, after
``trader.initialize()``, and passes ``account_equity_usdt`` (A07 fix).

These tests use AST / source inspection — they never import or instantiate
live runtime objects, Trader, or asyncio workers.
"""

from __future__ import annotations

import ast
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LIVE_SCRIPT = _PROJECT_ROOT / "scripts" / "run_boll_cvd_live.py"


def _source() -> str:
    return _LIVE_SCRIPT.read_text()


def _ast() -> ast.Module:
    return ast.parse(_source())


# ---------------------------------------------------------------------------
# 1. test_live_entry_calls_symbol_bootstrap_once
# ---------------------------------------------------------------------------


def test_live_entry_calls_symbol_bootstrap_once() -> None:
    """``build_live_symbol_runtime_configs`` must be called exactly once."""
    tree = _ast()
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "build_live_symbol_runtime_configs":
                count += 1
    assert count == 1, (
        f"Expected exactly 1 call to build_live_symbol_runtime_configs, "
        f"found {count}"
    )


# ---------------------------------------------------------------------------
# 2. test_live_entry_passes_trader_account_equity_to_bootstrap
# ---------------------------------------------------------------------------


def test_live_entry_passes_trader_account_equity_to_bootstrap() -> None:
    """The single call to ``build_live_symbol_runtime_configs`` must pass
    ``account_equity_usdt=trader.account_equity_usdt`` so the legacy
    ``.env`` path uses ``from_account_equity()`` instead of ``from_env()``."""
    tree = _ast()
    found_call = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "build_live_symbol_runtime_configs":
                found_call = True
                kw_names = {kw.arg for kw in node.keywords}
                assert "account_equity_usdt" in kw_names, (
                    "build_live_symbol_runtime_configs must receive "
                    "account_equity_usdt= to use from_account_equity() "
                    "on the legacy .env path"
                )
    assert found_call, "build_live_symbol_runtime_configs call not found"


# ---------------------------------------------------------------------------
# 3. test_live_entry_does_not_use_replace_for_account_equity
# ---------------------------------------------------------------------------


def test_live_entry_does_not_use_replace_for_account_equity() -> None:
    """Account equity must be applied via ``account_equity_usdt`` parameter
    to ``build_live_symbol_runtime_configs`` — NOT via
    ``dataclasses.replace``.

    The source must not contain ``from dataclasses import replace`` nor
    ``dry_run_equity_usdt=trader.account_equity_usdt``.
    """
    source = _source()
    assert "from dataclasses import replace" not in source, (
        "dataclasses.replace must not be imported — account equity "
        "should be passed to build_live_symbol_runtime_configs()"
    )
    assert "dry_run_equity_usdt=trader.account_equity_usdt" not in source, (
        "dry_run_equity_usdt must not be set via replace — pass "
        "account_equity_usdt to build_live_symbol_runtime_configs() instead"
    )


# ---------------------------------------------------------------------------
# 4. test_live_entry_bootstrap_occurs_after_trader_initialize
# ---------------------------------------------------------------------------


def test_live_entry_bootstrap_occurs_after_trader_initialize() -> None:
    """``build_live_symbol_runtime_configs`` must be called AFTER
    ``await trader.initialize()`` so that ``trader.account_equity_usdt``
    is available and the legacy ``.env`` path does not read
    ``DRY_RUN_EQUITY_USDT``."""
    source = _source()
    init_idx = source.index("await trader.initialize()")
    bootstrap_idx = source.index("build_live_symbol_runtime_configs(")
    assert init_idx < bootstrap_idx, (
        f"build_live_symbol_runtime_configs must be called after "
        f"trader.initialize() — found initialize at {init_idx}, "
        f"bootstrap at {bootstrap_idx}"
    )
