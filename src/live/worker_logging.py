from __future__ import annotations

import os
import re
from pathlib import Path

_SYMBOL_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_symbol_for_log_dir(symbol: str) -> str:
    text = str(symbol or "").strip()
    if not text:
        return "UNKNOWN"
    return _SYMBOL_SAFE_RE.sub("_", text)


def configure_symbol_worker_logging_env(
    *,
    symbol: str,
    base_log_dir: str | None = None,
    file_name: str = "worker.log",
    force: bool = False,
) -> Path:
    safe_symbol = sanitize_symbol_for_log_dir(symbol)
    root = Path(base_log_dir or os.getenv("WORKER_LOG_BASE_DIR", "logs/workers"))
    target_dir = root / safe_symbol
    target_dir.mkdir(parents=True, exist_ok=True)

    if force or not os.getenv("LOG_DIR"):
        os.environ["LOG_DIR"] = str(target_dir)
    if force or not os.getenv("LOG_FILE_NAME"):
        os.environ["LOG_FILE_NAME"] = file_name

    return target_dir
