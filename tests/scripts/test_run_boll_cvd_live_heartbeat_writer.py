#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C06 source guard — ensures the heartbeat writer IS wired into
SymbolWorkerApp.run() but NOT into the thin live entry or tick-path
workers.  The live entry remains thin, and workers do not write heartbeats.
"""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LIVE_SCRIPT = _PROJECT_ROOT / "scripts" / "run_boll_cvd_live.py"
_APP_MODULE = _PROJECT_ROOT / "src" / "live" / "symbol_worker_app.py"
_HEARTBEAT_WRITER_MODULE = _PROJECT_ROOT / "src" / "live" / "heartbeat_writer.py"


# ============================================================================
# 1. Live entry guard
# ============================================================================


def test_live_entry_does_not_reference_heartbeat_writer() -> None:
    """run_boll_cvd_live.py must NOT reference HeartbeatWriter or heartbeat —
    the thin entry stays thin."""
    source = _LIVE_SCRIPT.read_text(encoding="utf-8")

    assert "HeartbeatWriter" not in source, (
        "run_boll_cvd_live.py must not reference HeartbeatWriter"
    )
    assert "HeartbeatWriterConfig" not in source, (
        "run_boll_cvd_live.py must not reference HeartbeatWriterConfig"
    )
    assert "heartbeat" not in source, (
        "run_boll_cvd_live.py must not reference heartbeat"
    )


# ============================================================================
# 2. SymbolWorkerApp guard
# ============================================================================


def test_symbol_worker_app_starts_heartbeat_in_c06() -> None:
    """C06 MUST wire HeartbeatWriter into SymbolWorkerApp.run()."""
    source = _APP_MODULE.read_text(encoding="utf-8")

    assert "factory.create_heartbeat_writer(" in source, (
        "C06 SymbolWorkerApp must call factory.create_heartbeat_writer"
    )
    assert "heartbeat_writer.run_until_cancelled(" in source, (
        "C06 SymbolWorkerApp must call heartbeat_writer.run_until_cancelled"
    )


# ============================================================================
# 3. Workers guard
# ============================================================================


def test_workers_do_not_reference_heartbeat_writer() -> None:
    """C05 must NOT wire HeartbeatWriter into tick-path workers."""
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

        forbidden = [
            "HeartbeatWriter",
            "heartbeat_file",
            "runtime/heartbeats",
        ]
        for token in forbidden:
            assert token not in source, (
                f"{rel_path} must not reference {token!r}"
            )


# ============================================================================
# 4. RuntimePaths guard — heartbeat_file and heartbeats_dir must still exist
# ============================================================================


def test_runtime_paths_heartbeat_file_still_exists() -> None:
    """RuntimePaths.heartbeat_file and heartbeats_dir must still be defined."""
    rp_source = (
        _PROJECT_ROOT / "src" / "live" / "runtime_paths.py"
    ).read_text(encoding="utf-8")

    assert "heartbeat_file" in rp_source, (
        "RuntimePaths must still define heartbeat_file"
    )
    assert "heartbeats_dir" in rp_source, (
        "RuntimePaths must still define heartbeats_dir"
    )


# ============================================================================
# 5. No asyncio.create_task for heartbeat (SymbolWorkerApp + heartbeat_writer)
# ============================================================================


def test_no_asyncio_create_task_for_heartbeat() -> None:
    """SymbolWorkerApp must NOT use asyncio.create_task to start the heartbeat.
    The D06b two-stage shutdown uses heartbeat_task as a named reference for
    task classification (critical vs producer/aux) — that is legitimate."""
    app_source = _APP_MODULE.read_text(encoding="utf-8")

    assert "asyncio.create_task(" not in app_source, (
        "SymbolWorkerApp must not use asyncio.create_task for heartbeat"
    )

    hb_source = _HEARTBEAT_WRITER_MODULE.read_text(encoding="utf-8")
    assert "asyncio.create_task(" not in hb_source, (
        "heartbeat_writer.py must not use asyncio.create_task"
    )


# ============================================================================
# 6. C06b — degrade logic stays in heartbeat_writer.py, NOT SymbolWorkerApp
# ============================================================================


def test_heartbeat_writer_degrades_failures_without_symbol_worker_app_catch() -> None:
    """SymbolWorkerApp must NOT contain heartbeat degrade logic —
    that belongs in heartbeat_writer.py."""
    source = _APP_MODULE.read_text(encoding="utf-8")

    assert "HEARTBEAT_WRITE_FAILED" not in source, (
        "SymbolWorkerApp must not contain HEARTBEAT_WRITE_FAILED"
    )
    assert "consecutive_failures" not in source, (
        "SymbolWorkerApp must not reference consecutive_failures"
    )
    assert "last_error" not in source, (
        "SymbolWorkerApp must not reference last_error"
    )


# ============================================================================
# 7. C06b — heartbeat_writer.py contains failure degrade logic
# ============================================================================


def test_heartbeat_writer_contains_failure_degrade_logic() -> None:
    """heartbeat_writer.py must contain the C06b failure degrade logic."""
    source = _HEARTBEAT_WRITER_MODULE.read_text(encoding="utf-8")

    required = [
        "HEARTBEAT_WRITE_FAILED",
        "consecutive_failures",
        "last_error",
        "failure_log_interval_seconds",
        "except asyncio.CancelledError",
        "except Exception as exc",
        "logger.warning(",
    ]
    for token in required:
        assert token in source, (
            f"heartbeat_writer.py must contain {token!r}"
        )

    assert "logger.exception(" not in source, (
        "heartbeat_writer.py must NOT use logger.exception"
    )
