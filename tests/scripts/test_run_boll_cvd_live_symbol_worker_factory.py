#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C02 source guard — ensures ``scripts/run_boll_cvd_live.py`` uses
``SymbolWorkerFactory`` for object creation and does not directly
construct core runtime objects.

These tests use source inspection — they never import or instantiate
live runtime objects, Trader, or asyncio workers.
"""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LIVE_SCRIPT = _PROJECT_ROOT / "scripts" / "run_boll_cvd_live.py"
_FACTORY_MODULE = _PROJECT_ROOT / "src" / "live" / "symbol_worker_factory.py"


def _source() -> str:
    return _LIVE_SCRIPT.read_text()


def _factory_source() -> str:
    return _FACTORY_MODULE.read_text()


# ============================================================================
# 1. test_live_entry_constructs_factory
# ============================================================================


def test_live_entry_constructs_factory() -> None:
    """The live entry must import and construct SymbolWorkerFactory."""
    source = _source()
    assert "SymbolWorkerFactory" in source, (
        "C02 must import SymbolWorkerFactory in run_boll_cvd_live.py"
    )
    assert "factory = SymbolWorkerFactory()" in source, (
        "C02 must construct SymbolWorkerFactory() in run_boll_cvd_live.py"
    )


# ============================================================================
# 2. test_live_entry_uses_factory_for_core_objects
# ============================================================================


def test_live_entry_uses_factory_for_core_objects() -> None:
    """The live entry must use factory methods for all core object creation."""
    source = _source()

    required = [
        "factory.create_email_sender(",
        "factory.create_trader(",
        "factory.create_runtime_paths(",
        "factory.create_persistence(",
        "factory.create_strategy_objects(",
        "factory.create_cvd_tracker(",
        "factory.create_queues(",
        "factory.create_monitor(",
    ]
    for token in required:
        assert token in source, (
            f"C02 live entry must use {token}"
        )


# ============================================================================
# 3. test_live_entry_no_direct_core_constructors
# ============================================================================


def test_live_entry_no_direct_core_constructors() -> None:
    """The live entry must NOT directly construct core runtime objects —
    those should be created by the factory."""
    source = _source()

    forbidden = [
        "email_sender = EmailSender()",
        "trader = Trader()",
        "runtime_paths = RuntimePaths(",
        "journal = LiveTradeJournal.from_runtime_paths(",
        "state_store = LiveStateStore.from_runtime_paths(",
        "rolling_loss_guard = RollingLossGuard.from_runtime_paths(",
        "reporter = DailyTradeReporter(",
        "sizer = SimplePositionSizer(",
        "strategy = BollCvdShockReclaimStrategy(",
        "cvd = CvdTracker(",
        "monitor = BollBandBreakoutMonitor(",
    ]
    for token in forbidden:
        assert token not in source, (
            f"C02 live entry must not directly construct: {token!r}"
        )


# ============================================================================
# 4. test_handoff_still_in_live_entry_not_factory
# ============================================================================


def test_handoff_still_in_live_entry_not_factory() -> None:
    """handoff_legacy_runtime_files must stay in the live entry, not move
    to the factory."""
    source = _source()
    factory_source = _factory_source()

    assert "handoff_legacy_runtime_files(" in source, (
        "handoff_legacy_runtime_files must remain in run_boll_cvd_live.py"
    )
    assert "handoff_legacy_runtime_files(" not in factory_source, (
        "handoff_legacy_runtime_files must NOT be in symbol_worker_factory.py"
    )


# ============================================================================
# 5. test_asyncio_gather_still_in_live_entry_not_factory
# ============================================================================


def test_asyncio_gather_still_in_live_entry_not_factory() -> None:
    """asyncio.gather must stay in the live entry, not move to the factory."""
    source = _source()
    factory_source = _factory_source()

    assert "asyncio.gather(" in source, (
        "asyncio.gather must remain in run_boll_cvd_live.py"
    )
    assert "asyncio.gather(" not in factory_source, (
        "asyncio.gather must NOT be in symbol_worker_factory.py"
    )


# ============================================================================
# 6. test_report_loops_still_in_live_entry
# ============================================================================


def test_report_loops_still_in_live_entry() -> None:
    """daily_report_loop and weekly_summary_loop must stay in the live entry."""
    source = _source()

    assert "async def daily_report_loop" in source, (
        "daily_report_loop must remain in run_boll_cvd_live.py"
    )
    assert "async def weekly_summary_loop" in source, (
        "weekly_summary_loop must remain in run_boll_cvd_live.py"
    )


# ============================================================================
# 7. test_factory_not_imported_in_workers
# ============================================================================


def test_factory_not_imported_in_workers() -> None:
    """Workers must not reference SymbolWorkerFactory."""
    worker_files = [
        "src/live/workers/strategy_tick_worker.py",
        "src/live/workers/execution_worker.py",
        "src/live/workers/execution_command_processor.py",
        "src/live/workers/account_position_sync_worker.py",
    ]

    for rel_path in worker_files:
        full_path = _PROJECT_ROOT / rel_path
        if not full_path.exists():
            continue
        source = full_path.read_text(encoding="utf-8")
        assert "SymbolWorkerFactory" not in source, (
            f"{rel_path} must not import or reference SymbolWorkerFactory"
        )
