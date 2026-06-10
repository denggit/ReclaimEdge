#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C04 unit tests for ``src.live.symbol_worker_app``.

These tests use source inspection primarily — they do NOT start a real
Trader, OKX connection, or websocket.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pytest

from src.live.live_app_config import (
    DailyReportConfig,
    LiveAppConfig,
    LiveHeartbeatConfig,
    WeeklySummaryConfig,
)
from src.live.symbol_worker_app import SymbolWorkerApp
from src.live.symbol_worker_factory import SymbolWorkerFactory


# ---------------------------------------------------------------------------
# Logging isolation — prevent test fake-errors from contaminating
# the production logs/app.log.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True, scope="module")
def _isolate_live_log_file(tmp_path_factory: Any) -> Any:
    """Redirect all file-based logging to a temp directory so tests never
    contaminate the production logs/app.log.

    ``src.utils.log.setup_logging()`` runs at module-import time and
    attaches a ``TimedRotatingFileHandler`` pointed at the real
    ``logs/app.log``.  This fixture runs once *after* import, resets
    the log subsystem, and re-routes file output to a temp directory.
    """
    from src.utils import log as log_module

    test_log_dir = tmp_path_factory.mktemp("logs")
    os.environ["LOG_DIR"] = str(test_log_dir)

    # Reset the module-level guard so setup_logging() runs again.
    log_module._setup_done = False

    # Stop any running async queue listener.
    log_module._stop_queue_listener()

    # Remove every handler the root logger currently holds (the
    # import-time handler that points at the real logs/app.log).
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    # Re-run setup with the patched environment.
    log_module.setup_logging(None)

    yield

    # Teardown: stop the listener so temp files can be cleaned up.
    log_module._stop_queue_listener()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_APP_MODULE = _PROJECT_ROOT / "src" / "live" / "symbol_worker_app.py"
_FACTORY_MODULE = _PROJECT_ROOT / "src" / "live" / "symbol_worker_factory.py"


def _app_source() -> str:
    return _APP_MODULE.read_text()


def _factory_source() -> str:
    return _FACTORY_MODULE.read_text()


def _make_app_config() -> LiveAppConfig:
    return LiveAppConfig(
        strategy_tick_queue_maxsize=100,
        execution_queue_maxsize=10,
        position_sync_seconds=5.0,
        account_sync_seconds=60.0,
        cash_log_min_delta_usdt=0.01,
        market_tick_heartbeat_seconds=60.0,
        account_snapshot_stale_warn_seconds=30.0,
        strategy_tick_lag_warn_seconds=2.0,
        execution_queue_backlog_log_seconds=30.0,
        daily_report=DailyReportConfig(raw_time="09:00", hour=9, minute=0),
        weekly_summary=WeeklySummaryConfig(
            enabled=True,
            raw_time="10:00",
            raw_weekday="0",
            weekday=0,
            hour=10,
            minute=0,
            compact_after_success=False,
        ),
        heartbeat=LiveHeartbeatConfig(
            enabled=True, interval_seconds=10.0, stale_after_seconds=30.0
        ),
    )


# ============================================================================
# 1. test_symbol_worker_app_exists
# ============================================================================


def test_symbol_worker_app_exists() -> None:
    """SymbolWorkerApp must be importable."""
    assert SymbolWorkerApp is not None


# ============================================================================
# 2. test_symbol_worker_app_from_env_uses_live_app_config_and_factory
# ============================================================================


def test_symbol_worker_app_from_env_uses_live_app_config_and_factory() -> None:
    """from_env must use LiveAppConfig.from_env() and handle the factory
    parameter."""
    source = _app_source()
    assert "LiveAppConfig.from_env()" in source, (
        "from_env must call LiveAppConfig.from_env()"
    )
    assert "factory or SymbolWorkerFactory()" in source, (
        "from_env must use 'factory or SymbolWorkerFactory()'"
    )


# ============================================================================
# 3. test_symbol_worker_app_is_frozen_dataclass
# ============================================================================


def test_symbol_worker_app_is_frozen_dataclass() -> None:
    """SymbolWorkerApp must be a frozen=True dataclass — mutation must
    raise an exception."""
    app = SymbolWorkerApp(
        app_config=_make_app_config(),
        factory=SymbolWorkerFactory(),
    )
    with pytest.raises(Exception):
        app.factory = SymbolWorkerFactory()  # type: ignore[misc]


# ============================================================================
# 4. test_symbol_worker_app_run_has_expected_runtime_order
# ============================================================================


def test_symbol_worker_app_run_has_expected_runtime_order() -> None:
    """Verify the runtime call order inside SymbolWorkerApp.run()."""
    source = _app_source()

    ordered = [
        "factory.create_email_sender(",
        "factory.create_trader(",
        "await trader.start()",
        "await trader.initialize()",
        "build_live_symbol_runtime_configs(",
        "_assert_trader_matches_symbol_config(trader,",
        "factory.create_runtime_paths(",
        "handoff_legacy_runtime_files(",
        "factory.create_heartbeat_writer(",
        "factory.create_persistence(",
        "factory.create_strategy_objects(",
        "await trader.fetch_position_snapshot()",
        "fetch_usdt_cash_balance",
        "rolling_loss_guard.load_or_initialize(",
        "journal.record_cash_baseline(",
        "state_store.load()",
        "trusted_startup_saved_state(",
        "factory.create_cvd_tracker(",
        "apply_main_tp_startup_recovery(",
        "apply_sidecar_startup_recovery(",
        "refresh_sidecar_state_totals",
        "apply_rolling_loss_guard_startup_state(",
        "apply_three_stage_startup_safety_gate(",
        "factory.create_queues(",
        "async def daily_report_loop",
        "async def weekly_summary_loop",
        "factory.create_monitor(",
        # D06b: tasks created via asyncio.ensure_future in order:
        # account → strategy → execution → daily → weekly → heartbeat → monitor
        "account_position_sync_worker_module.account_position_sync_worker(",
        "strategy_tick_worker_module.strategy_tick_worker(",
        "execution_worker_module.execution_worker(",
        "heartbeat_writer.run_until_cancelled(",
        "monitor.run_forever()",
        "await trader.close()",
    ]

    prev_idx = 0
    for token in ordered:
        if token == "await trader.close()":
            # This token appears twice (except block + finally block).
            # The ordering check must use the *last* occurrence — the
            # one in the finally block at the end of run().
            idx = source.rfind(token)
        else:
            idx = source.find(token)
        assert idx >= 0, f"token {token!r} not found in SymbolWorkerApp source"
        assert idx >= prev_idx, (
            f"token {token!r} at {idx} is before previous token at {prev_idx}"
        )
        prev_idx = idx


# ============================================================================
# 5. test_symbol_worker_app_does_not_load_dotenv_or_live_trading_gate
# ============================================================================


def test_symbol_worker_app_does_not_load_dotenv_or_live_trading_gate() -> None:
    """SymbolWorkerApp.run() must NOT contain load_dotenv or the
    LIVE_TRADING gate — those belong to the entry script."""
    source = _app_source()

    forbidden = [
        "load_dotenv",
        "live_trading_enabled",
        "LIVE_TRADING is not true",
    ]
    for token in forbidden:
        assert token not in source, (
            f"SymbolWorkerApp must not contain {token!r}"
        )


# ============================================================================
# 6. test_symbol_worker_app_handoff_not_hidden_in_factory
# ============================================================================


def test_symbol_worker_app_handoff_not_hidden_in_factory() -> None:
    """handoff_legacy_runtime_files must be called in SymbolWorkerApp, NOT
    in the factory."""
    app_source = _app_source()
    factory_source = _factory_source()

    assert "handoff_legacy_runtime_files(" in app_source, (
        "SymbolWorkerApp must call handoff_legacy_runtime_files"
    )
    assert "handoff_legacy_runtime_files(" not in factory_source, (
        "factory must NOT call handoff_legacy_runtime_files"
    )


# ============================================================================
# 7. test_symbol_worker_app_keeps_report_loops_inside_app_for_c03
# ============================================================================


def test_symbol_worker_app_keeps_report_loops_inside_app() -> None:
    """daily_report_loop and weekly_summary_loop must be defined inside
    SymbolWorkerApp.run()."""
    source = _app_source()

    assert "async def daily_report_loop" in source, (
        "SymbolWorkerApp must define daily_report_loop"
    )
    assert "async def weekly_summary_loop" in source, (
        "SymbolWorkerApp must define weekly_summary_loop"
    )


# ============================================================================
# 8. test_symbol_worker_app_no_btc_or_supervisor
# ============================================================================


def test_symbol_worker_app_no_btc_or_supervisor() -> None:
    """SymbolWorkerApp must NOT contain any BTC, subprocess, or supervisor
    references.  Heartbeat references ARE allowed as of C06."""
    source = _app_source()

    forbidden = [
        "BTC-USDT-SWAP",
        "subprocess",
        "multiprocessing",
        "ReclaimSupervisor",
        "run_reclaim_supervisor",
        "run_symbol_worker",
    ]
    for token in forbidden:
        assert token not in source, (
            f"SymbolWorkerApp must not contain {token!r}"
        )


# ============================================================================
# 9. test_assert_trader_matches_symbol_config_present
# ============================================================================


def test_assert_trader_matches_symbol_config_present() -> None:
    """The _assert_trader_matches_symbol_config helper must be present in
    SymbolWorkerApp with the expected error message and checks."""
    source = _app_source()

    assert "def _assert_trader_matches_symbol_config" in source, (
        "SymbolWorkerApp must define _assert_trader_matches_symbol_config"
    )
    assert "TOML/env trader config mismatch" in source, (
        "SymbolWorkerApp must contain the 'TOML/env trader config mismatch' error"
    )
    assert "pos_side_mode" in source, (
        "SymbolWorkerApp must check pos_side_mode"
    )
    assert "leverage" in source, (
        "SymbolWorkerApp must check leverage"
    )


# ============================================================================
# 10. test_symbol_worker_app_heartbeat_order
# ============================================================================


def test_symbol_worker_app_heartbeat_order() -> None:
    """Verify heartbeat writer is created before asyncio.gather and runs
    between weekly_summary_loop and monitor.run_forever."""
    source = _app_source()

    assert "factory.create_heartbeat_writer(" in source, (
        "SymbolWorkerApp must call factory.create_heartbeat_writer"
    )
    assert "heartbeat_writer.run_until_cancelled(" in source, (
        "SymbolWorkerApp must call heartbeat_writer.run_until_cancelled"
    )

    create_heartbeat_writer_idx = source.find("factory.create_heartbeat_writer(")
    asyncio_gather_idx = source.find("asyncio.gather(")
    account_worker_idx = source.find("account_position_sync_worker_module.account_position_sync_worker(")
    strategy_worker_idx = source.find("strategy_tick_worker_module.strategy_tick_worker(")
    execution_worker_idx = source.find("execution_worker_module.execution_worker(")
    weekly_summary_idx = source.find("weekly_summary_loop()")
    heartbeat_idx = source.find("heartbeat_writer.run_until_cancelled(")
    monitor_idx = source.find("monitor.run_forever()")

    assert create_heartbeat_writer_idx > 0
    assert heartbeat_idx > 0
    assert create_heartbeat_writer_idx < asyncio_gather_idx, (
        "heartbeat writer must be created before asyncio.gather"
    )
    assert account_worker_idx < strategy_worker_idx < execution_worker_idx, (
        "core workers must be in order: account → strategy → execution"
    )
    assert heartbeat_idx > weekly_summary_idx, (
        "heartbeat_writer.run_until_cancelled must be after weekly_summary_loop"
    )
    assert heartbeat_idx < monitor_idx, (
        "heartbeat_writer.run_until_cancelled must be before monitor.run_forever"
    )


# ============================================================================
# 11. test_symbol_worker_app_allows_task_creation_for_shutdown
# ============================================================================


def test_symbol_worker_app_uses_named_task_references_for_shutdown() -> None:
    """SymbolWorkerApp must use named task references (account_task,
    execution_task, heartbeat_task, etc.) so the D06b two-stage shutdown
    can classify tasks into critical_drain_tasks and producer_or_aux_tasks."""
    source = _app_source()

    required = [
        "account_task",
        "execution_task",
        "critical_drain_tasks",
        "producer_or_aux_tasks",
    ]
    for token in required:
        assert token in source, (
            f"SymbolWorkerApp must contain named reference {token!r}"
        )


# ============================================================================
# E05h: source guard — required event imports and emitter
# ============================================================================


class TestE05hRequiredImports:
    def test_contains_worker_event_emitter_import(self) -> None:
        source = _app_source()
        assert "WorkerEventEmitter" in source, (
            "symbol_worker_app.py must import WorkerEventEmitter"
        )
        assert "JsonlOutbox" in source, (
            "symbol_worker_app.py must import JsonlOutbox"
        )

    def test_contains_event_constants(self) -> None:
        source = _app_source()
        required_constants = [
            "WORKER_STARTED",
            "WORKER_STARTUP_RECOVERY_COMPLETED",
            "WORKER_STARTUP_RECOVERY_FAILED",
            "WORKER_STOPPING",
            "WORKER_STOPPED",
            "WORKER_HEARTBEAT_WRITE_FAILED",
            "WORKER_DRAIN_STARTED",
            "WORKER_DRAIN_COMPLETED",
            "WORKER_DRAIN_TIMEOUT",
        ]
        for token in required_constants:
            assert token in source, (
                f"symbol_worker_app.py must import {token}"
            )

    def test_contains_emit_helper(self) -> None:
        source = _app_source()
        assert "_emit_worker_event_best_effort" in source, (
            "symbol_worker_app.py must define _emit_worker_event_best_effort"
        )

    def test_contains_worker_event_emitter_variable(self) -> None:
        source = _app_source()
        assert "worker_event_emitter: WorkerEventEmitter | None = None" in source, (
            "symbol_worker_app.py must initialise worker_event_emitter"
        )


class TestE05hForbiddenTokens:
    def test_no_supervisor_imports(self) -> None:
        source = _app_source()
        forbidden = [
            "SupervisorEventPipeline",
            "ChildEventReader",
            "AlertDeduper",
            "AlertPolicy",
            "SupervisorEmailPublisher",
            "send_email_async(",
            "process_once(",
        ]
        for token in forbidden:
            assert token not in source, (
                f"symbol_worker_app.py must not contain {token!r}"
            )

    def test_no_external_http_libs(self) -> None:
        source = _app_source()
        forbidden = [
            "import requests",
            "import httpx",
            "import websocket",
            "from okx",
            "RECLAIM_SYMBOLS",
            "BTC-USDT-SWAP",
        ]
        for token in forbidden:
            assert token not in source, (
                f"symbol_worker_app.py must not contain {token!r}"
            )
