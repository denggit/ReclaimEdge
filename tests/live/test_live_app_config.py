from __future__ import annotations

import os

import pytest

from src.live.live_app_config import LiveAppConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_C01_ENV_KEYS = [
    "STRATEGY_TICK_QUEUE_MAXSIZE",
    "EXECUTION_QUEUE_MAXSIZE",
    "POSITION_SYNC_SECONDS",
    "ACCOUNT_SYNC_SECONDS",
    "ACCOUNT_LOG_MIN_DELTA_USDT",
    "MARKET_TICK_HEARTBEAT_SECONDS",
    "ACCOUNT_SNAPSHOT_STALE_WARN_SECONDS",
    "STRATEGY_TICK_LAG_WARN_SECONDS",
    "EXECUTION_QUEUE_BACKLOG_LOG_SECONDS",
    "DAILY_REPORT_TIME",
    "WEEKLY_SUMMARY_ENABLED",
    "WEEKLY_SUMMARY_TIME",
    "WEEKLY_SUMMARY_WEEKDAY",
    "WEEKLY_COMPACT_AFTER_SUCCESS",
]


def _clear_c01_env() -> None:
    for key in _C01_ENV_KEYS:
        os.environ.pop(key, None)


# ---------------------------------------------------------------------------
# 1. test_from_env_defaults
# ---------------------------------------------------------------------------


def test_from_env_defaults() -> None:
    """When no C01 env vars are set, all defaults must match the original
    ``run_boll_cvd_live.py`` hard-coded values."""
    _clear_c01_env()

    cfg = LiveAppConfig.from_env()

    assert cfg.strategy_tick_queue_maxsize == 20000
    assert cfg.execution_queue_maxsize == 1000
    assert cfg.position_sync_seconds == 5.0
    assert cfg.account_sync_seconds == 60.0
    assert cfg.cash_log_min_delta_usdt == 0.01
    assert cfg.market_tick_heartbeat_seconds == 60.0
    assert cfg.account_snapshot_stale_warn_seconds == 30.0
    assert cfg.strategy_tick_lag_warn_seconds == 2.0
    assert cfg.execution_queue_backlog_log_seconds == 30.0

    assert cfg.daily_report.raw_time == "09:00"
    assert cfg.daily_report.hour == 9
    assert cfg.daily_report.minute == 0

    assert cfg.weekly_summary.enabled is True
    assert cfg.weekly_summary.raw_time == "10:00"
    assert cfg.weekly_summary.raw_weekday == "0"
    assert cfg.weekly_summary.weekday == 0
    assert cfg.weekly_summary.hour == 10
    assert cfg.weekly_summary.minute == 0
    assert cfg.weekly_summary.compact_after_success is False


# ---------------------------------------------------------------------------
# 2. test_from_env_overrides
# ---------------------------------------------------------------------------


def test_from_env_overrides() -> None:
    """All C01 env vars set to non-default values must be correctly parsed."""
    os.environ["STRATEGY_TICK_QUEUE_MAXSIZE"] = "30000"
    os.environ["EXECUTION_QUEUE_MAXSIZE"] = "2000"
    os.environ["POSITION_SYNC_SECONDS"] = "10"
    os.environ["ACCOUNT_SYNC_SECONDS"] = "120"
    os.environ["ACCOUNT_LOG_MIN_DELTA_USDT"] = "0.05"
    os.environ["MARKET_TICK_HEARTBEAT_SECONDS"] = "120"
    os.environ["ACCOUNT_SNAPSHOT_STALE_WARN_SECONDS"] = "60"
    os.environ["STRATEGY_TICK_LAG_WARN_SECONDS"] = "5"
    os.environ["EXECUTION_QUEUE_BACKLOG_LOG_SECONDS"] = "60"
    os.environ["DAILY_REPORT_TIME"] = "08:30"
    os.environ["WEEKLY_SUMMARY_ENABLED"] = "false"
    os.environ["WEEKLY_SUMMARY_TIME"] = "12:00"
    os.environ["WEEKLY_SUMMARY_WEEKDAY"] = "3"
    os.environ["WEEKLY_COMPACT_AFTER_SUCCESS"] = "true"

    try:
        cfg = LiveAppConfig.from_env()

        assert cfg.strategy_tick_queue_maxsize == 30000
        assert cfg.execution_queue_maxsize == 2000
        assert cfg.position_sync_seconds == 10.0
        assert cfg.account_sync_seconds == 120.0
        assert cfg.cash_log_min_delta_usdt == 0.05
        assert cfg.market_tick_heartbeat_seconds == 120.0
        assert cfg.account_snapshot_stale_warn_seconds == 60.0
        assert cfg.strategy_tick_lag_warn_seconds == 5.0
        assert cfg.execution_queue_backlog_log_seconds == 60.0

        assert cfg.daily_report.raw_time == "08:30"
        assert cfg.daily_report.hour == 8
        assert cfg.daily_report.minute == 30

        assert cfg.weekly_summary.enabled is False
        assert cfg.weekly_summary.raw_time == "12:00"
        assert cfg.weekly_summary.raw_weekday == "3"
        assert cfg.weekly_summary.weekday == 3
        assert cfg.weekly_summary.hour == 12
        assert cfg.weekly_summary.minute == 0
        assert cfg.weekly_summary.compact_after_success is True
    finally:
        _clear_c01_env()


# ---------------------------------------------------------------------------
# 3. test_weekly_summary_disabled
# ---------------------------------------------------------------------------


def test_weekly_summary_disabled() -> None:
    """WEEKLY_SUMMARY_ENABLED=false must yield enabled=False."""
    _clear_c01_env()
    os.environ["WEEKLY_SUMMARY_ENABLED"] = "false"
    try:
        cfg = LiveAppConfig.from_env()
        assert cfg.weekly_summary.enabled is False
    finally:
        _clear_c01_env()


# ---------------------------------------------------------------------------
# 4. test_weekly_compact_after_success_true_values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("true_value", ["1", "true", "yes", "y", "on"])
def test_weekly_compact_after_success_true_values(true_value: str) -> None:
    """WEEKLY_COMPACT_AFTER_SUCCESS must parse all accepted true strings."""
    _clear_c01_env()
    os.environ["WEEKLY_COMPACT_AFTER_SUCCESS"] = true_value
    try:
        cfg = LiveAppConfig.from_env()
        assert cfg.weekly_summary.compact_after_success is True
    finally:
        _clear_c01_env()


# ---------------------------------------------------------------------------
# 5. test_invalid_weekday_raises
# ---------------------------------------------------------------------------


def test_invalid_weekday_raises() -> None:
    """WEEKLY_SUMMARY_WEEKDAY=7 must raise ValueError with the expected message."""
    _clear_c01_env()
    os.environ["WEEKLY_SUMMARY_WEEKDAY"] = "7"
    try:
        with pytest.raises(ValueError, match="Invalid WEEKLY_SUMMARY_WEEKDAY=7"):
            LiveAppConfig.from_env()
    finally:
        _clear_c01_env()


# ---------------------------------------------------------------------------
# 6. test_invalid_daily_report_time_raises
# ---------------------------------------------------------------------------


def test_invalid_daily_report_time_raises() -> None:
    """DAILY_REPORT_TIME=bad must raise ValueError (via parse_daily_report_time)."""
    _clear_c01_env()
    os.environ["DAILY_REPORT_TIME"] = "bad"
    try:
        with pytest.raises(ValueError):
            LiveAppConfig.from_env()
    finally:
        _clear_c01_env()


# ---------------------------------------------------------------------------
# 7. test_config_is_frozen
# ---------------------------------------------------------------------------


def test_config_is_frozen() -> None:
    """LiveAppConfig is a frozen dataclass — setting an attribute must raise."""
    _clear_c01_env()
    cfg = LiveAppConfig.from_env()

    with pytest.raises(Exception):
        cfg.strategy_tick_queue_maxsize = 99999  # type: ignore[misc]

    with pytest.raises(Exception):
        cfg.daily_report.hour = 99  # type: ignore[misc]

    with pytest.raises(Exception):
        cfg.weekly_summary.enabled = False  # type: ignore[misc]
