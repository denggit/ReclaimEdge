#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日志模块，提供按日期分割的日志文件功能。

默认行为：
- 写入文件，不输出到 console；
- 每天一个日志文件；
- 默认保留最近 7 天日志，过期文件在启动和日期切换时由 TimedRotatingFileHandler 清理。

可用环境变量：
- LOG_LEVEL=INFO
- LOG_DIR=logs
- LOG_FILE_NAME=app.log
- LOG_TO_CONSOLE=false
- LOG_TO_FILE=true
- LOG_RETENTION_DAYS=7
- LOG_BACKUP_COUNT=7
"""
from __future__ import annotations

import datetime as dt
import logging
import logging.handlers
import os
import re
import sys
from pathlib import Path

_setup_done = False


def _bool_env(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return bool(default)


def _int_env(name: str, default: int, minimum: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        value = default
    else:
        try:
            value = int(str(raw).strip())
        except ValueError:
            value = default
    if minimum is not None:
        value = max(value, minimum)
    return value


def _get_log_level_from_env(default_level: int = logging.INFO) -> int:
    log_level_str = os.environ.get("LOG_LEVEL", "").upper()
    if not log_level_str:
        return default_level

    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "WARN": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
        "FATAL": logging.CRITICAL,
    }
    return level_map.get(log_level_str, default_level)


def _cleanup_expired_daily_logs(log_dir: str, log_file_name: str, retention_days: int) -> None:
    """Best-effort cleanup for daily rotated files older than retention_days."""
    if retention_days <= 0:
        return
    directory = Path(log_dir)
    if not directory.exists():
        return

    cutoff_date = dt.date.today() - dt.timedelta(days=retention_days)
    pattern = re.compile(rf"^{re.escape(log_file_name)}\.(\d{{4}}-\d{{2}}-\d{{2}})(?:\..*)?$")
    for path in directory.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if not match:
            continue
        try:
            file_date = dt.datetime.strptime(match.group(1), "%Y-%m-%d").date()
        except ValueError:
            continue
        if file_date < cutoff_date:
            try:
                path.unlink()
            except OSError:
                # 日志清理失败不应该影响交易主程序启动。
                pass


def setup_logging(log_level: int | None = None, log_dir: str = "logs") -> None:
    """
    配置根日志记录器。

    Args:
        log_level: 日志级别，如果为 None 则从 LOG_LEVEL 读取，默认为 INFO。
        log_dir: 日志目录，默认 logs，可被 LOG_DIR 覆盖。
    """
    global _setup_done
    if _setup_done:
        return

    if log_level is None:
        log_level = _get_log_level_from_env(logging.INFO)

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    log_to_console = _bool_env("LOG_TO_CONSOLE", False)
    log_to_file = _bool_env("LOG_TO_FILE", True)
    effective_log_dir = os.environ.get("LOG_DIR", log_dir)
    log_file_name = os.environ.get("LOG_FILE_NAME", "app.log")
    retention_days = _int_env("LOG_RETENTION_DAYS", 7, minimum=1)
    backup_count = _int_env("LOG_BACKUP_COUNT", retention_days, minimum=1)
    if not log_to_console and not log_to_file:
        log_to_file = True

    root_logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if log_to_file:
        os.makedirs(effective_log_dir, exist_ok=True)
        _cleanup_expired_daily_logs(effective_log_dir, log_file_name, retention_days)
        log_file = os.path.join(effective_log_dir, log_file_name)
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=log_file,
            when="midnight",
            interval=1,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.suffix = "%Y-%m-%d"
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    numba_log_level_str = os.environ.get("NUMBA_LOG_LEVEL", "").upper()
    if numba_log_level_str:
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "WARN": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        numba_log_level = level_map.get(numba_log_level_str, logging.WARNING)
    else:
        numba_log_level = logging.WARNING if log_level <= logging.WARNING else log_level

    logging.getLogger("numba").setLevel(numba_log_level)
    for module_name in ["numba.core.ssa", "numba.core.byteflow", "numba.core.interpreter"]:
        logging.getLogger(module_name).setLevel(logging.WARNING)

    root_logger.info(
        "日志系统初始化完成 | log_dir=%s log_file=%s console=%s file=%s level=%s retention_days=%s backup_count=%s",
        os.path.abspath(effective_log_dir),
        log_file_name,
        log_to_console,
        log_to_file,
        logging.getLevelName(log_level),
        retention_days,
        backup_count,
    )
    root_logger.debug("Numba日志级别: %s", numba_log_level)

    _setup_done = True


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的 logger。"""
    setup_logging(None)
    return logging.getLogger(name)


setup_logging(None)
logger = get_logger(__name__)
