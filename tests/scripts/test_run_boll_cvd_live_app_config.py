#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C01 source guard — ensures ``scripts/run_boll_cvd_live.py`` uses
``LiveAppConfig`` and no longer reads app-level env keys directly.

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


def _FACTORY_SOURCE() -> str:
    return _FACTORY_MODULE.read_text()


# ============================================================================
# 1. test_live_entry_constructs_live_app_config
# ============================================================================


def test_live_entry_constructs_live_app_config() -> None:
    """The live entry must construct LiveAppConfig via from_env()."""
    source = _source()
    assert "LiveAppConfig.from_env()" in source, (
        "C01 must construct LiveAppConfig.from_env() in run_boll_cvd_live.py"
    )


# ============================================================================
# 2. test_live_entry_imports_live_app_config
# ============================================================================


def test_live_entry_imports_live_app_config() -> None:
    """The live entry must import LiveAppConfig."""
    source = _source()
    assert "from src.live.live_app_config import LiveAppConfig" in source, (
        "C01 must import LiveAppConfig in run_boll_cvd_live.py"
    )


# ============================================================================
# 3. test_live_entry_no_direct_app_level_env_reads
# ============================================================================

_C01_ENV_KEYS = [
    'os.getenv("STRATEGY_TICK_QUEUE_MAXSIZE"',
    'os.getenv("EXECUTION_QUEUE_MAXSIZE"',
    'os.getenv("POSITION_SYNC_SECONDS"',
    'os.getenv("ACCOUNT_SYNC_SECONDS"',
    'os.getenv("ACCOUNT_LOG_MIN_DELTA_USDT"',
    'os.getenv("MARKET_TICK_HEARTBEAT_SECONDS"',
    'os.getenv("ACCOUNT_SNAPSHOT_STALE_WARN_SECONDS"',
    'os.getenv("STRATEGY_TICK_LAG_WARN_SECONDS"',
    'os.getenv("EXECUTION_QUEUE_BACKLOG_LOG_SECONDS"',
    'os.getenv("DAILY_REPORT_TIME"',
    'os.getenv("WEEKLY_SUMMARY_ENABLED"',
    'os.getenv("WEEKLY_SUMMARY_TIME"',
    'os.getenv("WEEKLY_SUMMARY_WEEKDAY"',
    'os.getenv("WEEKLY_COMPACT_AFTER_SUCCESS"',
]


def test_live_entry_no_direct_app_level_env_reads() -> None:
    """The live entry must NOT read any C01 app-level env keys directly."""
    source = _source()
    for key in _C01_ENV_KEYS:
        assert key not in source, (
            f"C01 must remove direct {key} from run_boll_cvd_live.py — "
            f"use LiveAppConfig instead"
        )


# ============================================================================
# 4. test_live_entry_uses_app_config_for_queue_sizes
# ============================================================================


def test_live_entry_uses_app_config_for_queue_sizes() -> None:
    """Queue maxsize values must come from app_config — as of C02 this
    flows through factory.create_queues(app_config)."""
    source = _source()
    assert "factory.create_queues(app_config)" in source, (
        "C02 must use factory.create_queues(app_config)"
    )
    assert "app_config.strategy_tick_queue_maxsize" in source or (
        "app_config.strategy_tick_queue_maxsize" in _FACTORY_SOURCE()
    ), (
        "C01/C02 app_config.strategy_tick_queue_maxsize must be used "
        "in either live entry or factory"
    )
    assert "app_config.execution_queue_maxsize" in source or (
        "app_config.execution_queue_maxsize" in _FACTORY_SOURCE()
    ), (
        "C01/C02 app_config.execution_queue_maxsize must be used "
        "in either live entry or factory"
    )


# ============================================================================
# 5. test_live_entry_uses_app_config_for_sync_and_heartbeat
# ============================================================================


def test_live_entry_uses_app_config_for_sync_and_heartbeat() -> None:
    """Sync, heartbeat, and log timing values must come from app_config."""
    source = _source()
    assert "app_config.position_sync_seconds" in source, (
        "C01 must use app_config.position_sync_seconds"
    )
    assert "app_config.account_sync_seconds" in source, (
        "C01 must use app_config.account_sync_seconds"
    )
    assert "app_config.cash_log_min_delta_usdt" in source, (
        "C01 must use app_config.cash_log_min_delta_usdt"
    )
    assert "app_config.market_tick_heartbeat_seconds" in source, (
        "C01 must use app_config.market_tick_heartbeat_seconds"
    )
    assert "app_config.account_snapshot_stale_warn_seconds" in source, (
        "C01 must use app_config.account_snapshot_stale_warn_seconds"
    )
    assert "app_config.strategy_tick_lag_warn_seconds" in source, (
        "C01 must use app_config.strategy_tick_lag_warn_seconds"
    )
    assert "app_config.execution_queue_backlog_log_seconds" in source, (
        "C01 must use app_config.execution_queue_backlog_log_seconds"
    )


# ============================================================================
# 6. test_live_entry_uses_app_config_for_report_schedules
# ============================================================================


def test_live_entry_uses_app_config_for_report_schedules() -> None:
    """Daily report and weekly summary config must come from app_config."""
    source = _source()
    assert "app_config.daily_report" in source, (
        "C01 must use app_config.daily_report"
    )
    assert "app_config.weekly_summary" in source, (
        "C01 must use app_config.weekly_summary"
    )


# ============================================================================
# 7. test_live_app_config_not_imported_in_tick_workers
# ============================================================================


def test_live_app_config_not_imported_in_tick_workers() -> None:
    """LiveAppConfig must NOT appear in tick / execution / sync worker files."""
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
        assert "LiveAppConfig" not in source, (
            f"{rel_path} must not import or reference LiveAppConfig"
        )
