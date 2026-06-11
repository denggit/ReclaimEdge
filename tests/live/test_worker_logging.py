from __future__ import annotations

import os
from pathlib import Path

from src.live.worker_logging import (
    configure_symbol_worker_logging_env,
    sanitize_symbol_for_log_dir,
)


def test_sanitize_symbol_for_log_dir_keeps_safe_symbol() -> None:
    assert sanitize_symbol_for_log_dir("ETH-USDT-SWAP") == "ETH-USDT-SWAP"


def test_sanitize_symbol_for_log_dir_replaces_unsafe_chars() -> None:
    assert sanitize_symbol_for_log_dir("BTC/USDT:SWAP") == "BTC_USDT_SWAP"


def test_sanitize_symbol_for_log_dir_empty_symbol_is_unknown() -> None:
    assert sanitize_symbol_for_log_dir("") == "UNKNOWN"


def test_configure_symbol_worker_logging_env_creates_forced_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("LOG_DIR", raising=False)
    monkeypatch.delenv("LOG_FILE_NAME", raising=False)

    target_dir = configure_symbol_worker_logging_env(
        symbol="ETH-USDT-SWAP",
        base_log_dir=str(tmp_path),
        force=True,
    )

    assert target_dir == tmp_path / "ETH-USDT-SWAP"
    assert target_dir.is_dir()
    assert os.environ["LOG_DIR"] == str(target_dir)
    assert os.environ["LOG_FILE_NAME"] == "worker.log"


def test_configure_symbol_worker_logging_env_does_not_override_without_force(
    tmp_path: Path,
    monkeypatch,
) -> None:
    existing_dir = tmp_path / "existing"
    existing_file_name = "existing.log"
    monkeypatch.setenv("LOG_DIR", str(existing_dir))
    monkeypatch.setenv("LOG_FILE_NAME", existing_file_name)

    target_dir = configure_symbol_worker_logging_env(
        symbol="ETH-USDT-SWAP",
        base_log_dir=str(tmp_path),
        force=False,
    )

    assert target_dir == tmp_path / "ETH-USDT-SWAP"
    assert target_dir.is_dir()
    assert os.environ["LOG_DIR"] == str(existing_dir)
    assert os.environ["LOG_FILE_NAME"] == existing_file_name


def test_configure_symbol_worker_logging_env_overrides_with_force(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "existing"))
    monkeypatch.setenv("LOG_FILE_NAME", "existing.log")

    target_dir = configure_symbol_worker_logging_env(
        symbol="ETH-USDT-SWAP",
        base_log_dir=str(tmp_path),
        force=True,
    )

    assert target_dir == tmp_path / "ETH-USDT-SWAP"
    assert target_dir.is_dir()
    assert os.environ["LOG_DIR"] == str(target_dir)
    assert os.environ["LOG_FILE_NAME"] == "worker.log"
