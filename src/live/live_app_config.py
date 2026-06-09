from __future__ import annotations

import os
from dataclasses import dataclass

from src.live import time_utils as live_time_utils

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE_VALUES


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    return float(raw)


@dataclass(frozen=True)
class DailyReportConfig:
    raw_time: str
    hour: int
    minute: int


@dataclass(frozen=True)
class WeeklySummaryConfig:
    enabled: bool
    raw_time: str
    raw_weekday: str
    weekday: int
    hour: int
    minute: int
    compact_after_success: bool


@dataclass(frozen=True)
class LiveAppConfig:
    strategy_tick_queue_maxsize: int
    execution_queue_maxsize: int
    position_sync_seconds: float
    account_sync_seconds: float
    cash_log_min_delta_usdt: float
    market_tick_heartbeat_seconds: float
    account_snapshot_stale_warn_seconds: float
    strategy_tick_lag_warn_seconds: float
    execution_queue_backlog_log_seconds: float
    daily_report: DailyReportConfig
    weekly_summary: WeeklySummaryConfig

    @classmethod
    def from_env(cls) -> "LiveAppConfig":
        daily_raw_time = os.getenv("DAILY_REPORT_TIME", "09:00")
        daily_hour, daily_minute = live_time_utils.parse_daily_report_time(daily_raw_time)

        weekly_enabled = _env_bool("WEEKLY_SUMMARY_ENABLED", True)
        weekly_raw_time = os.getenv("WEEKLY_SUMMARY_TIME", "10:00")
        weekly_raw_weekday = os.getenv("WEEKLY_SUMMARY_WEEKDAY", "0")
        weekly_hour, weekly_minute = live_time_utils.parse_weekly_report_time(weekly_raw_time)
        weekly_weekday = int(weekly_raw_weekday)
        if weekly_weekday < 0 or weekly_weekday > 6:
            raise ValueError(f"Invalid WEEKLY_SUMMARY_WEEKDAY={weekly_raw_weekday}")
        compact_after_success = _env_bool("WEEKLY_COMPACT_AFTER_SUCCESS", False)

        return cls(
            strategy_tick_queue_maxsize=_env_int("STRATEGY_TICK_QUEUE_MAXSIZE", 20000),
            execution_queue_maxsize=_env_int("EXECUTION_QUEUE_MAXSIZE", 1000),
            position_sync_seconds=_env_float("POSITION_SYNC_SECONDS", 5.0),
            account_sync_seconds=_env_float("ACCOUNT_SYNC_SECONDS", 60.0),
            cash_log_min_delta_usdt=_env_float("ACCOUNT_LOG_MIN_DELTA_USDT", 0.01),
            market_tick_heartbeat_seconds=_env_float("MARKET_TICK_HEARTBEAT_SECONDS", 60.0),
            account_snapshot_stale_warn_seconds=_env_float("ACCOUNT_SNAPSHOT_STALE_WARN_SECONDS", 30.0),
            strategy_tick_lag_warn_seconds=_env_float("STRATEGY_TICK_LAG_WARN_SECONDS", 2.0),
            execution_queue_backlog_log_seconds=_env_float("EXECUTION_QUEUE_BACKLOG_LOG_SECONDS", 30.0),
            daily_report=DailyReportConfig(
                raw_time=daily_raw_time,
                hour=daily_hour,
                minute=daily_minute,
            ),
            weekly_summary=WeeklySummaryConfig(
                enabled=weekly_enabled,
                raw_time=weekly_raw_time,
                raw_weekday=weekly_raw_weekday,
                weekday=weekly_weekday,
                hour=weekly_hour,
                minute=weekly_minute,
                compact_after_success=compact_after_success,
            ),
        )
