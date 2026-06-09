#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C04 source guard — ensures ``SymbolWorkerApp`` (not the thin live entry)
uses ``SymbolWorkerFactory`` for object creation and does not directly
construct core runtime objects.

These tests use source inspection — they never import or instantiate
live runtime objects, Trader, or asyncio workers.
"""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LIVE_SCRIPT = _PROJECT_ROOT / "scripts" / "run_boll_cvd_live.py"
_APP_MODULE = _PROJECT_ROOT / "src" / "live" / "symbol_worker_app.py"
_FACTORY_MODULE = _PROJECT_ROOT / "src" / "live" / "symbol_worker_factory.py"


def _live_source() -> str:
    return _LIVE_SCRIPT.read_text()


def _app_source() -> str:
    return _APP_MODULE.read_text()


def _factory_source() -> str:
    return _FACTORY_MODULE.read_text()


# ============================================================================
# 1. test_app_uses_factory_for_core_objects
# ============================================================================


def test_app_uses_factory_for_core_objects() -> None:
    """SymbolWorkerApp.run() must use factory methods for all core object creation."""
    source = _app_source()

    required = [
        "self.factory.create_email_sender(",
        "self.factory.create_trader(",
        "self.factory.create_runtime_paths(",
        "self.factory.create_persistence(",
        "self.factory.create_strategy_objects(",
        "self.factory.create_cvd_tracker(",
        "self.factory.create_queues(",
        "self.factory.create_monitor(",
    ]
    for token in required:
        assert token in source, (
            f"C04 SymbolWorkerApp.run() must use {token}"
        )


# ============================================================================
# 2. test_live_entry_no_direct_core_constructors
# ============================================================================


def test_live_entry_no_direct_core_constructors() -> None:
    """The thin live entry must NOT directly construct core runtime objects —
    those should be created by the factory inside SymbolWorkerApp.run()."""
    source = _live_source()

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
        # Also forbid the pre-C04 direct factory construction pattern
        "factory = SymbolWorkerFactory()",
    ]
    for token in forbidden:
        assert token not in source, (
            f"C04 live entry must not directly construct: {token!r}"
        )


# ============================================================================
# 3. test_app_no_direct_core_constructors
# ============================================================================


def test_app_no_direct_core_constructors() -> None:
    """SymbolWorkerApp.run() must NOT directly construct core runtime objects —
    those should be created by the factory."""
    source = _app_source()

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
            f"C04 SymbolWorkerApp must not directly construct: {token!r}"
        )


# ============================================================================
# 4. test_handoff_still_in_app_not_factory
# ============================================================================


def test_handoff_still_in_app_not_factory() -> None:
    """handoff_legacy_runtime_files must stay in SymbolWorkerApp.run(),
    not move to the factory or the thin live entry."""
    app_source = _app_source()
    factory_source = _factory_source()
    live_source = _live_source()

    assert "handoff_legacy_runtime_files(" in app_source, (
        "C04 handoff_legacy_runtime_files must remain in SymbolWorkerApp.run()"
    )
    assert "handoff_legacy_runtime_files(" not in factory_source, (
        "handoff_legacy_runtime_files must NOT be in symbol_worker_factory.py"
    )
    assert "handoff_legacy_runtime_files(" not in live_source, (
        "C04 handoff_legacy_runtime_files must NOT be in thin live entry"
    )


# ============================================================================
# 5. test_asyncio_gather_in_app_not_factory
# ============================================================================


def test_asyncio_gather_in_app_not_factory() -> None:
    """asyncio.gather must stay in SymbolWorkerApp.run(), not move to the
    factory or the thin live entry."""
    app_source = _app_source()
    factory_source = _factory_source()
    live_source = _live_source()

    assert "asyncio.gather(" in app_source, (
        "C04 asyncio.gather must remain in SymbolWorkerApp.run()"
    )
    assert "asyncio.gather(" not in factory_source, (
        "asyncio.gather must NOT be in symbol_worker_factory.py"
    )
    assert "asyncio.gather(" not in live_source, (
        "C04 asyncio.gather must NOT be in thin live entry"
    )


# ============================================================================
# 6. test_report_loops_in_app_not_live_entry
# ============================================================================


def test_report_loops_in_app_not_live_entry() -> None:
    """daily_report_loop and weekly_summary_loop must be in
    SymbolWorkerApp.run(), not in the thin live entry."""
    app_source = _app_source()
    live_source = _live_source()

    assert "async def daily_report_loop" in app_source, (
        "C04 daily_report_loop must remain in SymbolWorkerApp.run()"
    )
    assert "async def weekly_summary_loop" in app_source, (
        "C04 weekly_summary_loop must remain in SymbolWorkerApp.run()"
    )
    assert "async def daily_report_loop" not in live_source, (
        "C04 daily_report_loop must NOT be in thin live entry"
    )
    assert "async def weekly_summary_loop" not in live_source, (
        "C04 weekly_summary_loop must NOT be in thin live entry"
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


# ============================================================================
# 8. C06 heartbeat factory source guards
# ============================================================================


def test_app_source_contains_factory_create_heartbeat_writer() -> None:
    """SymbolWorkerApp source must use self.factory.create_heartbeat_writer."""
    source = _app_source()
    assert "self.factory.create_heartbeat_writer(" in source, (
        "C06 SymbolWorkerApp must call self.factory.create_heartbeat_writer"
    )


def test_factory_source_contains_heartbeat_constructors() -> None:
    """Factory source must construct HeartbeatWriter with HeartbeatWriterConfig."""
    source = _factory_source()
    assert "HeartbeatWriter(" in source, (
        "Factory must construct HeartbeatWriter"
    )
    assert "HeartbeatWriterConfig(" in source, (
        "Factory must construct HeartbeatWriterConfig"
    )


def test_factory_source_does_not_start_heartbeat() -> None:
    """Factory must NOT start or run the heartbeat writer — only construct it."""
    source = _factory_source()
    forbidden = [
        "run_until_cancelled(",
        "write_once(",
        "asyncio.create_task",
    ]
    for token in forbidden:
        assert token not in source, (
            f"Factory must not contain {token!r}"
        )
