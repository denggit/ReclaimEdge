from __future__ import annotations

import datetime as dt
import os
import time
from zoneinfo import ZoneInfo


def format_ts_ms(ts_ms: int) -> str:
    return dt.datetime.fromtimestamp(ts_ms / 1000).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def parse_daily_report_time(value: str) -> tuple[int, int]:
    hour_text, minute_text = value.strip().split(":", 1)
    hour = int(hour_text)
    minute = int(minute_text)
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError(f"Invalid DAILY_REPORT_TIME={value}")
    return hour, minute


def parse_weekly_report_time(value: str) -> tuple[int, int]:
    return parse_daily_report_time(value)


def live_report_timezone() -> ZoneInfo:
    name = os.getenv("LIVE_REPORT_TIMEZONE", "Asia/Singapore").strip() or "Asia/Singapore"
    try:
        return ZoneInfo(name)
    except Exception:
        return ZoneInfo("Asia/Singapore")


def next_daily_report_time(hour: int, minute: int) -> dt.datetime:
    tz = live_report_timezone()
    now = dt.datetime.now(tz)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += dt.timedelta(days=1)
    return target


def next_weekly_summary_time(hour: int, minute: int, weekday: int = 0) -> dt.datetime:
    tz = live_report_timezone()
    now = dt.datetime.now(tz)
    days_ahead = weekday - now.weekday()
    target_date = now.date() + dt.timedelta(days=days_ahead)
    target = dt.datetime.combine(target_date, dt.time(hour, minute), tzinfo=tz)
    if target <= now:
        target += dt.timedelta(days=7)
    return target


def utc_ms() -> int:
    return int(time.time() * 1000)
