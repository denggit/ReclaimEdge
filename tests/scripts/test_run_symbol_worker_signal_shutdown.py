#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D06b source guard — ensures ``scripts/run_symbol_worker.py`` installs
signal handlers via ``WorkerShutdownController`` and that it does NOT
introduce supervisor, multiprocessing, BTC, trading modules, or order
mutation symbols.
"""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_NEW_ENTRY = _PROJECT_ROOT / "scripts" / "run_symbol_worker.py"


def _new_entry_source() -> str:
    return _NEW_ENTRY.read_text(encoding="utf-8")


# ============================================================================
# 1. Must contain shutdown-related imports and wiring
# ============================================================================


def test_run_symbol_worker_contains_shutdown_controller() -> None:
    """D06b run_symbol_worker.py must create WorkerShutdownController and
    install signal handlers."""
    source = _new_entry_source()

    required = [
        "WorkerShutdownController",
        "install_symbol_worker_signal_handlers",
        "shutdown_controller",
        "SymbolWorkerApp.from_env(shutdown_controller=shutdown_controller)",
    ]
    for token in required:
        assert token in source, (
            f"D06b run_symbol_worker.py must contain {token!r}"
        )


# ============================================================================
# 2. Must NOT contain supervisor / child process / heartbeat monitor
# ============================================================================


def test_run_symbol_worker_no_supervisor_or_child() -> None:
    """D06b run_symbol_worker.py must NOT introduce supervisor or child
    process symbols."""
    source = _new_entry_source()

    forbidden = [
        "ReclaimSupervisor",
        "ChildProcess",
        "HeartbeatMonitor",
        "run_reclaim_supervisor",
        "subprocess",
        "multiprocessing",
    ]
    for token in forbidden:
        assert token not in source, (
            f"D06b run_symbol_worker.py must NOT contain {token!r}"
        )


# ============================================================================
# 3. Must NOT contain Trader / Strategy / trading modules
# ============================================================================


def test_run_symbol_worker_no_trading_modules() -> None:
    """D06b run_symbol_worker.py must NOT import Trader, Strategy, or any
    trading / order / cancel modules."""
    source = _new_entry_source()

    forbidden = [
        "Trader",
        "BollCvd",
        "account_position_sync_worker",
        "execution_worker",
        "cancel",
        "close_position",
        "market_close",
        "place_market_order",
    ]
    for token in forbidden:
        assert token not in source, (
            f"D06b run_symbol_worker.py must NOT contain {token!r}"
        )


# ============================================================================
# 4. Must NOT contain BTC or RECLAIM_SYMBOLS
# ============================================================================


def test_run_symbol_worker_no_btc_or_multi_symbol() -> None:
    """D06b run_symbol_worker.py must NOT introduce BTC or multi-symbol
    configuration."""
    source = _new_entry_source()

    forbidden = [
        "RECLAIM_SYMBOLS",
        "BTC",
    ]
    for token in forbidden:
        assert token not in source, (
            f"D06b run_symbol_worker.py must NOT contain {token!r}"
        )


# ============================================================================
# 5. LIVE_TRADING gate and thin entry preserved
# ============================================================================


def test_run_symbol_worker_preserves_live_gate() -> None:
    """D06b run_symbol_worker.py must keep the LIVE_TRADING gate and
    load_dotenv."""
    source = _new_entry_source()

    required = [
        "load_dotenv()",
        "live_config_helpers.live_trading_enabled()",
        "LIVE_TRADING is not true. Refusing to start symbol worker.",
        "SymbolWorkerApp.from_env",
        "await app.run()",
        "asyncio.run(main())",
    ]
    for token in required:
        assert token in source, (
            f"D06b run_symbol_worker.py must contain {token!r}"
        )
