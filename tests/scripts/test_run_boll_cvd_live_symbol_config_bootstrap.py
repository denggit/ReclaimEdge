#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C04 regression tests ensuring ``SymbolWorkerApp.run()`` calls
``build_live_symbol_runtime_configs`` exactly once, after
``trader.initialize()``, and passes ``account_equity_usdt`` (A07 fix).
The thin live entry is only responsible for the LIVE_TRADING gate.

These tests use AST / source inspection — they never import or instantiate
live runtime objects, Trader, or asyncio workers.
"""

from __future__ import annotations

import ast
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LIVE_SCRIPT = _PROJECT_ROOT / "scripts" / "run_boll_cvd_live.py"
_APP_MODULE = _PROJECT_ROOT / "src" / "live" / "symbol_worker_app.py"


def _live_source() -> str:
    return _LIVE_SCRIPT.read_text()


def _app_source() -> str:
    return _APP_MODULE.read_text()


def _app_ast() -> ast.Module:
    return ast.parse(_app_source())


# ---------------------------------------------------------------------------
# 1. test_app_calls_symbol_bootstrap_once
# ---------------------------------------------------------------------------


def test_app_calls_symbol_bootstrap_once() -> None:
    """``build_live_symbol_runtime_configs`` must be called exactly once
    in SymbolWorkerApp.run()."""
    tree = _app_ast()
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
# 2. test_app_passes_trader_account_equity_to_bootstrap
# ---------------------------------------------------------------------------


def test_app_passes_trader_account_equity_to_bootstrap() -> None:
    """The single call to ``build_live_symbol_runtime_configs`` must pass
    ``account_equity_usdt=trader.account_equity_usdt`` so the legacy
    ``.env`` path uses ``from_account_equity()`` instead of ``from_env()``."""
    tree = _app_ast()
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
# 3. test_app_does_not_use_replace_for_account_equity
# ---------------------------------------------------------------------------


def test_app_does_not_use_replace_for_account_equity() -> None:
    """Account equity must be applied via ``account_equity_usdt`` parameter
    to ``build_live_symbol_runtime_configs`` — NOT via
    ``dataclasses.replace``.

    The source must not contain ``from dataclasses import replace`` nor
    ``dry_run_equity_usdt=trader.account_equity_usdt``.
    """
    source = _app_source()
    assert "from dataclasses import replace" not in source, (
        "dataclasses.replace must not be imported — account equity "
        "should be passed to build_live_symbol_runtime_configs()"
    )
    assert "dry_run_equity_usdt=trader.account_equity_usdt" not in source, (
        "dry_run_equity_usdt must not be set via replace — pass "
        "account_equity_usdt to build_live_symbol_runtime_configs() instead"
    )


# ---------------------------------------------------------------------------
# 4. test_app_bootstrap_occurs_after_trader_initialize
# ---------------------------------------------------------------------------


def test_app_bootstrap_occurs_after_trader_initialize() -> None:
    """``build_live_symbol_runtime_configs`` must be called AFTER
    ``await trader.initialize()`` so that ``trader.account_equity_usdt``
    is available and the legacy ``.env`` path does not read
    ``DRY_RUN_EQUITY_USDT``."""
    source = _app_source()
    init_idx = source.index("await trader.initialize()")
    bootstrap_idx = source.index("build_live_symbol_runtime_configs(")
    assert init_idx < bootstrap_idx, (
        f"build_live_symbol_runtime_configs must be called after "
        f"trader.initialize() — found initialize at {init_idx}, "
        f"bootstrap at {bootstrap_idx}"
    )


# ---------------------------------------------------------------------------
# 5. test_app_has_trader_toml_consistency_guard
# ---------------------------------------------------------------------------


def test_app_has_trader_toml_consistency_guard() -> None:
    """The SymbolWorkerApp must define ``_assert_trader_matches_symbol_config``
    and reference the error message ``TOML/env trader config mismatch``."""
    source = _app_source()
    assert "_assert_trader_matches_symbol_config" in source, (
        "SymbolWorkerApp must define _assert_trader_matches_symbol_config"
    )
    assert "TOML/env trader config mismatch" in source, (
        "SymbolWorkerApp must raise on TOML/env mismatch"
    )


# ---------------------------------------------------------------------------
# 6. test_app_consistency_guard_called_after_bootstrap_before_strategy_creation
# ---------------------------------------------------------------------------


def test_app_consistency_guard_called_after_bootstrap_before_strategy_creation() -> None:
    """The consistency guard must be called after the bootstrap call and
    before strategy objects are created (via factory as of C02)."""
    source = _app_source()

    bootstrap_idx = source.index("runtime_configs = build_live_symbol_runtime_configs(")
    # The call site (not the ``def`` line) — passes ``trader`` as first arg.
    guard_idx = source.index("_assert_trader_matches_symbol_config(trader,")
    strategy_idx = source.index("factory.create_strategy_objects(")

    assert bootstrap_idx < guard_idx < strategy_idx, (
        f"Order violation: bootstrap={bootstrap_idx}, "
        f"guard={guard_idx}, strategy={strategy_idx}"
    )


# ---------------------------------------------------------------------------
# 7. test_live_entry_does_not_use_symbol_live_trading_as_gate
# ---------------------------------------------------------------------------


def test_live_entry_does_not_use_symbol_live_trading_as_gate() -> None:
    """``symbol_config.symbol.live_trading`` must NOT appear in the live
    entrypoint source.  ``LIVE_TRADING`` is still controlled by the
    existing ``.env`` gate — A08 must not change that semantic."""
    source = _live_source()
    assert "symbol_config.symbol.live_trading" not in source, (
        "symbol_config.symbol.live_trading must not be used as a live "
        "trading gate — LIVE_TRADING is still controlled by the .env gate"
    )


# ---------------------------------------------------------------------------
# 8. test_app_sidecar_max_legs_refresh_uses_mapped_config_not_env
# ---------------------------------------------------------------------------


def test_app_sidecar_max_legs_refresh_uses_mapped_config_not_env() -> None:
    """Startup sidecar state refresh must use the mapped
    ``position_sizer_config.sidecar_max_legs`` — NOT a direct
    ``os.getenv("SIDECAR_MAX_LEGS")`` read."""
    source = _app_source()

    # Must NOT contain direct env reads for SIDECAR_MAX_LEGS.
    assert 'os.getenv("SIDECAR_MAX_LEGS"' not in source, (
        "SIDECAR_MAX_LEGS must not be read directly from env"
    )
    assert "os.getenv('SIDECAR_MAX_LEGS'" not in source, (
        "SIDECAR_MAX_LEGS must not be read directly from env"
    )

    # Must still call refresh_sidecar_state_totals.
    assert "refresh_sidecar_state_totals" in source, (
        "refresh_sidecar_state_totals must still be called"
    )

    # The max legs argument must come from the mapped config.
    assert "position_sizer_config.sidecar_max_legs" in source, (
        "sidecar max legs must come from position_sizer_config.sidecar_max_legs"
    )


# ---------------------------------------------------------------------------
# 9. test_live_entry_only_has_live_trading_gate_no_bootstrap
# ---------------------------------------------------------------------------


def test_live_entry_only_has_live_trading_gate_no_bootstrap() -> None:
    """C04 thin live entry must NOT contain symbol config bootstrap —
    that lives in SymbolWorkerApp.run()."""
    source = _live_source()

    assert "live_trading_enabled" in source, (
        "C04 live entry must keep live_trading_enabled gate"
    )
    assert "build_live_symbol_runtime_configs" not in source, (
        "C04 live entry must NOT contain build_live_symbol_runtime_configs"
    )
