#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C04 source guard — ensures ``scripts/run_boll_cvd_live.py`` is a thin
entry that calls ``SymbolWorkerApp``, and that the full runtime body lives
in ``src/live/symbol_worker_app.py``.

These tests use source inspection — they never import or instantiate
live runtime objects, Trader, or asyncio workers.
"""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LIVE_SCRIPT = _PROJECT_ROOT / "scripts" / "run_boll_cvd_live.py"
_APP_MODULE = _PROJECT_ROOT / "src" / "live" / "symbol_worker_app.py"


def _live_source() -> str:
    return _LIVE_SCRIPT.read_text()


# ============================================================================
# 1. test_live_entry_calls_symbol_worker_app
# ============================================================================


def test_live_entry_calls_symbol_worker_app() -> None:
    """C04 live entry must import SymbolWorkerApp, call from_env, and run."""
    source = _live_source()

    assert "from src.live.symbol_worker_app import SymbolWorkerApp" in source, (
        "C04 must import SymbolWorkerApp in run_boll_cvd_live.py"
    )
    assert "SymbolWorkerApp.from_env()" in source, (
        "C04 must call SymbolWorkerApp.from_env() in run_boll_cvd_live.py"
    )
    assert "await app.run()" in source, (
        "C04 must call await app.run() in run_boll_cvd_live.py"
    )


# ============================================================================
# 2. test_live_entry_keeps_dotenv_and_live_gate
# ============================================================================


def test_live_entry_keeps_dotenv_and_live_gate() -> None:
    """C04 live entry must keep load_dotenv and the LIVE_TRADING gate."""
    source = _live_source()

    assert "load_dotenv()" in source, (
        "C04 live entry must call load_dotenv()"
    )
    assert "live_config_helpers.live_trading_enabled()" in source, (
        "C04 live entry must call live_trading_enabled()"
    )
    assert "LIVE_TRADING is not true. Refusing to start live runner." in source, (
        "C04 live entry must keep the LIVE_TRADING gate error message"
    )


# ============================================================================
# 3. test_live_entry_is_thin
# ============================================================================


def test_live_entry_is_thin() -> None:
    """C04 live entry must NOT contain the old main body — all runtime
    logic lives in SymbolWorkerApp.run()."""
    source = _live_source()

    forbidden = [
        "build_live_symbol_runtime_configs(",
        "_assert_trader_matches_symbol_config",
        "handoff_legacy_runtime_files(",
        "factory.create_email_sender(",
        "factory.create_trader(",
        "factory.create_runtime_paths(",
        "factory.create_persistence(",
        "factory.create_strategy_objects(",
        "factory.create_cvd_tracker(",
        "factory.create_queues(",
        "factory.create_monitor(",
        "asyncio.gather(",
        "monitor.run_forever(",
        "DailyTradeReporter",
        "LiveTradeJournal",
        "LiveStateStore",
        "RollingLossGuard",
        "BollCvdShockReclaimStrategy",
        "BollBandBreakoutMonitor",
        "account_position_sync_worker",
        "execution_worker",
        "strategy_tick_worker",
    ]
    for token in forbidden:
        assert token not in source, (
            f"C04 live entry must not contain {token!r} — "
            f"this belongs in SymbolWorkerApp.run()"
        )


# ============================================================================
# 4. test_symbol_worker_app_module_exists_and_is_wired
# ============================================================================


def test_symbol_worker_app_module_exists_and_is_wired() -> None:
    """SymbolWorkerApp module exists on disk and IS wired into
    run_boll_cvd_live.py."""
    assert _APP_MODULE.exists(), (
        "src/live/symbol_worker_app.py must exist"
    )
    live_source = _live_source()
    assert "SymbolWorkerApp.from_env()" in live_source, (
        "C04 must reference SymbolWorkerApp.from_env() in run_boll_cvd_live.py"
    )
