#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D06 source guard — ensures ``scripts/run_reclaim_supervisor.py`` is a thin
entry that calls ``ReclaimSupervisor``, installs signal handlers, and that it
does NOT introduce child processes, trading objects, BTC, or CLI symbol parameters.

These tests use source inspection — they never import or instantiate
live runtime objects, Trader, or asyncio workers.
"""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_NEW_ENTRY = _PROJECT_ROOT / "scripts" / "run_reclaim_supervisor.py"
_LIVE_SCRIPT = _PROJECT_ROOT / "scripts" / "run_boll_cvd_live.py"
_SYMBOL_WORKER_SCRIPT = _PROJECT_ROOT / "scripts" / "run_symbol_worker.py"


def _new_entry_source() -> str:
    return _NEW_ENTRY.read_text(encoding="utf-8")


def _live_source() -> str:
    return _LIVE_SCRIPT.read_text(encoding="utf-8")


def _symbol_worker_source() -> str:
    return _SYMBOL_WORKER_SCRIPT.read_text(encoding="utf-8")


# ============================================================================
# 1. test_run_reclaim_supervisor_exists
# ============================================================================


def test_run_reclaim_supervisor_exists() -> None:
    """D02 must create scripts/run_reclaim_supervisor.py."""
    assert _NEW_ENTRY.exists(), (
        "D02 must create scripts/run_reclaim_supervisor.py"
    )


# ============================================================================
# 2. test_run_reclaim_supervisor_is_thin_entry
# ============================================================================


def test_run_reclaim_supervisor_is_thin_entry() -> None:
    """D02 thin entry must contain load_dotenv, live_trading_enabled,
    ReclaimSupervisor.from_env, await supervisor.run_forever, and asyncio.run(main)."""
    source = _new_entry_source()

    required = [
        "load_dotenv()",
        "live_config_helpers.live_trading_enabled()",
        "ReclaimSupervisor.from_env()",
        "install_supervisor_signal_handlers(supervisor)",
        "await supervisor.run_forever()",
        "asyncio.run(main())",
    ]
    for token in required:
        assert token in source, (
            f"D02 run_reclaim_supervisor.py must contain {token!r}"
        )


# ============================================================================
# 3. test_run_reclaim_supervisor_live_gate_error_message
# ============================================================================


def test_run_reclaim_supervisor_live_gate_error_message() -> None:
    """D02 run_reclaim_supervisor.py must use the correct LIVE_TRADING gate
    error message."""
    source = _new_entry_source()

    assert "LIVE_TRADING is not true. Refusing to start reclaim supervisor." in source, (
        "D02 run_reclaim_supervisor.py must use the reclaim supervisor LIVE_TRADING error message"
    )


# ============================================================================
# 4. test_run_reclaim_supervisor_no_symbol_worker_or_child_start
# ============================================================================


def test_run_reclaim_supervisor_no_symbol_worker_or_child_start() -> None:
    """D02 run_reclaim_supervisor.py must NOT import SymbolWorkerApp,
    run_symbol_worker, subprocess, multiprocessing, or any child process
    machinery."""
    source = _new_entry_source()

    forbidden = [
        "SymbolWorkerApp",
        "run_symbol_worker",
        "scripts.run_symbol_worker",
        "subprocess",
        "multiprocessing",
        "Popen(",
        "Process(",
        "ChildProcess",
        "ChildProcessSpec",
        "HeartbeatMonitor",
        "HeartbeatStatus",
        "RECLAIM_SYMBOLS",
        "BTC",
        "workers",
    ]
    for token in forbidden:
        assert token not in source, (
            f"D02 run_reclaim_supervisor.py must NOT contain {token!r}"
        )


# ============================================================================
# 5. test_run_reclaim_supervisor_no_trading_runtime_objects
# ============================================================================


def test_run_reclaim_supervisor_no_trading_runtime_objects() -> None:
    """D02 run_reclaim_supervisor.py must NOT import or create any trading
    runtime objects."""
    source = _new_entry_source()

    forbidden = [
        "Trader",
        "SymbolWorkerFactory",
        "RuntimePaths",
        "HeartbeatWriter",
        "LiveTradeJournal",
        "LiveStateStore",
        "RollingLossGuard",
        "BollCvdShockReclaimStrategy",
        "BollBandBreakoutMonitor",
        "account_position_sync_worker",
        "strategy_tick_worker",
        "execution_worker",
        "asyncio.gather(",
        "monitor.run_forever(",
        "handoff_legacy_runtime_files(",
        "build_live_symbol_runtime_configs(",
    ]
    for token in forbidden:
        assert token not in source, (
            f"D02 run_reclaim_supervisor.py must NOT contain {token!r}"
        )


# ============================================================================
# 6. test_run_reclaim_supervisor_no_multi_symbol_or_btc
# ============================================================================


def test_run_reclaim_supervisor_no_multi_symbol_or_btc() -> None:
    """D02 run_reclaim_supervisor.py must NOT introduce BTC, RECLAIM_SYMBOLS,
    argparse, or CLI symbol parameters."""
    source = _new_entry_source()

    forbidden = [
        "RECLAIM_SYMBOLS",
        "BTC-USDT-SWAP",
        "BTC",
        "argparse",
        "--symbol",
        "inst_id",
    ]
    for token in forbidden:
        assert token not in source, (
            f"D02 run_reclaim_supervisor.py must NOT contain {token!r}"
        )


# ============================================================================
# 7. test_existing_entries_not_modified_or_cross_imported
# ============================================================================


def test_existing_entries_not_modified_or_cross_imported() -> None:
    """D02 must NOT modify or cross-import the existing entries
    run_boll_cvd_live.py and run_symbol_worker.py."""
    live_source = _live_source()
    symbol_worker_source = _symbol_worker_source()

    # run_boll_cvd_live.py must still be the existing live entry.
    assert "SymbolWorkerApp.from_env()" in live_source, (
        "Existing live entry must still call SymbolWorkerApp.from_env()"
    )
    assert "await app.run()" in live_source, (
        "Existing live entry must still call await app.run()"
    )
    assert "LIVE_TRADING is not true. Refusing to start live runner." in live_source, (
        "Existing live entry must keep the live runner error message"
    )

    # run_symbol_worker.py must still be the future child entrypoint.
    assert "SymbolWorkerApp.from_env()" in symbol_worker_source, (
        "run_symbol_worker.py must still call SymbolWorkerApp.from_env()"
    )
    assert "await app.run()" in symbol_worker_source, (
        "run_symbol_worker.py must still call await app.run()"
    )
    assert "LIVE_TRADING is not true. Refusing to start symbol worker." in symbol_worker_source, (
        "run_symbol_worker.py must keep the symbol worker error message"
    )

    # Neither must reference ReclaimSupervisor or run_reclaim_supervisor.
    for source, label in [(live_source, "run_boll_cvd_live.py"), (symbol_worker_source, "run_symbol_worker.py")]:
        for token in ["run_reclaim_supervisor", "ReclaimSupervisor"]:
            assert token not in source, (
                f"{label} must NOT contain {token!r}"
            )
