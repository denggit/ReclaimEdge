#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression tests ensuring ``scripts/run_boll_cvd_live.py`` calls
``build_live_symbol_runtime_configs`` exactly once (A07 fix).

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
# 2. test_live_entry_does_not_pass_account_equity_to_bootstrap
# ---------------------------------------------------------------------------


def test_live_entry_does_not_pass_account_equity_to_bootstrap() -> None:
    """The single call to ``build_live_symbol_runtime_configs`` must not pass
    ``account_equity_usdt`` — equity should be applied via
    ``dataclasses.replace`` instead."""
    tree = _ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "build_live_symbol_runtime_configs":
                for kw in node.keywords:
                    assert kw.arg != "account_equity_usdt", (
                        "build_live_symbol_runtime_configs must not receive "
                        "account_equity_usdt — use dataclasses.replace on "
                        "position_sizer_config instead"
                    )


# ---------------------------------------------------------------------------
# 3. test_live_entry_uses_dataclasses_replace_for_account_equity
# ---------------------------------------------------------------------------


def test_live_entry_uses_dataclasses_replace_for_account_equity() -> None:
    """Account equity must be applied via ``dataclasses.replace`` on
    ``position_sizer_config``, not via a second bootstrap call."""
    source = _source()
    assert "replace(" in source, (
        "Expected dataclasses.replace(…) call in the live entrypoint"
    )
    assert "dry_run_equity_usdt=trader.account_equity_usdt" in source, (
        "Expected replace(position_sizer_config, dry_run_equity_usdt=trader.account_equity_usdt)"
    )
