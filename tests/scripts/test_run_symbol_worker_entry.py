#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D01 source guard — ensures ``scripts/run_symbol_worker.py`` is a thin
entry that calls ``SymbolWorkerApp``, and that it does NOT introduce
supervisor, multiprocessing, BTC, or CLI symbol parameters.

These tests use source inspection — they never import or instantiate
live runtime objects, Trader, or asyncio workers.
"""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_NEW_ENTRY = _PROJECT_ROOT / "scripts" / "run_symbol_worker.py"
_LIVE_SCRIPT = _PROJECT_ROOT / "scripts" / "run_boll_cvd_live.py"


def _new_entry_source() -> str:
    return _NEW_ENTRY.read_text(encoding="utf-8")


def _live_source() -> str:
    return _LIVE_SCRIPT.read_text(encoding="utf-8")


# ============================================================================
# 1. test_run_symbol_worker_exists
# ============================================================================


def test_run_symbol_worker_exists() -> None:
    """D01 must create scripts/run_symbol_worker.py."""
    assert _NEW_ENTRY.exists(), (
        "D01 must create scripts/run_symbol_worker.py"
    )


# ============================================================================
# 2. test_run_symbol_worker_is_thin_entry
# ============================================================================


def test_run_symbol_worker_is_thin_entry() -> None:
    """D01 thin entry must contain load_dotenv, live_trading_enabled,
    SymbolWorkerApp.from_env, await app.run, and asyncio.run(main)."""
    source = _new_entry_source()

    required = [
        "load_dotenv()",
        "live_config_helpers.live_trading_enabled()",
        "SymbolWorkerApp.from_env()",
        "await app.run()",
        "asyncio.run(main())",
    ]
    for token in required:
        assert token in source, (
            f"D01 run_symbol_worker.py must contain {token!r}"
        )


# ============================================================================
# 3. test_run_symbol_worker_live_gate_error_message
# ============================================================================


def test_run_symbol_worker_live_gate_error_message() -> None:
    """D01 run_symbol_worker.py must use the correct LIVE_TRADING gate
    error message."""
    source = _new_entry_source()

    assert "LIVE_TRADING is not true. Refusing to start symbol worker." in source, (
        "D01 run_symbol_worker.py must use the symbol worker LIVE_TRADING error message"
    )


# ============================================================================
# 4. test_run_symbol_worker_no_direct_runtime_objects
# ============================================================================


def test_run_symbol_worker_no_direct_runtime_objects() -> None:
    """D01 run_symbol_worker.py must NOT directly create runtime objects
    like Trader, Strategy, Factory, workers, or call asyncio.gather /
    monitor.run_forever."""
    source = _new_entry_source()

    forbidden = [
        "Trader",
        "SymbolWorkerFactory",
        "RuntimePaths",
        "HeartbeatWriter",
        "HeartbeatWriterConfig",
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
            f"D01 run_symbol_worker.py must NOT contain {token!r}"
        )


# ============================================================================
# 5. test_run_symbol_worker_no_supervisor_or_multiprocess
# ============================================================================


def test_run_symbol_worker_no_supervisor_or_multiprocess() -> None:
    """D01 run_symbol_worker.py must NOT introduce supervisor, subprocess,
    or multiprocessing."""
    source = _new_entry_source()

    forbidden = [
        "ReclaimSupervisor",
        "run_reclaim_supervisor",
        "subprocess",
        "multiprocessing",
        "Process(",
        "Popen(",
        "HeartbeatMonitor",
        "ChildProcess",
    ]
    for token in forbidden:
        assert token not in source, (
            f"D01 run_symbol_worker.py must NOT contain {token!r}"
        )


# ============================================================================
# 6. test_run_symbol_worker_no_multi_symbol_or_btc
# ============================================================================


def test_run_symbol_worker_no_multi_symbol_or_btc() -> None:
    """D01 run_symbol_worker.py must NOT introduce BTC, RECLAIM_SYMBOLS,
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
            f"D01 run_symbol_worker.py must NOT contain {token!r}"
        )


# ============================================================================
# 7. test_run_boll_cvd_live_remains_existing_entry
# ============================================================================


def test_run_boll_cvd_live_remains_existing_entry() -> None:
    """run_boll_cvd_live.py must remain the existing live entry — it still
    calls SymbolWorkerApp.from_env, await app.run, and uses the live runner
    error message."""
    source = _live_source()

    assert "SymbolWorkerApp.from_env()" in source, (
        "Existing live entry must still call SymbolWorkerApp.from_env()"
    )
    assert "await app.run()" in source, (
        "Existing live entry must still call await app.run()"
    )
    assert "LIVE_TRADING is not true. Refusing to start live runner." in source, (
        "Existing live entry must keep the live runner error message"
    )


# ============================================================================
# 8. test_run_boll_cvd_live_does_not_import_run_symbol_worker
# ============================================================================


def test_run_boll_cvd_live_does_not_import_run_symbol_worker() -> None:
    """run_boll_cvd_live.py must NOT reference run_symbol_worker."""
    source = _live_source()

    forbidden = [
        "run_symbol_worker",
        "scripts.run_symbol_worker",
    ]
    for token in forbidden:
        assert token not in source, (
            f"run_boll_cvd_live.py must NOT contain {token!r}"
        )
