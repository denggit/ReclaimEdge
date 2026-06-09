#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""C05 source guard — ensures the heartbeat writer is NOT wired into the
live runtime yet.  C05 only adds the writer component and its tests; C06
will be responsible for integration into SymbolWorkerApp.
"""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LIVE_SCRIPT = _PROJECT_ROOT / "scripts" / "run_boll_cvd_live.py"
_APP_MODULE = _PROJECT_ROOT / "src" / "live" / "symbol_worker_app.py"


# ============================================================================
# 1. Live entry guard
# ============================================================================


def test_live_entry_does_not_reference_heartbeat_writer_in_c05() -> None:
    """C05 must NOT wire HeartbeatWriter into run_boll_cvd_live.py."""
    source = _LIVE_SCRIPT.read_text(encoding="utf-8")

    assert "HeartbeatWriter" not in source, (
        "C05 must not reference HeartbeatWriter in run_boll_cvd_live.py"
    )
    assert "HeartbeatWriterConfig" not in source, (
        "C05 must not reference HeartbeatWriterConfig in run_boll_cvd_live.py"
    )
    assert "heartbeat" not in source, (
        "C05 must not reference heartbeat in run_boll_cvd_live.py"
    )


# ============================================================================
# 2. SymbolWorkerApp guard
# ============================================================================


def test_symbol_worker_app_does_not_start_heartbeat_in_c05() -> None:
    """C05 must NOT wire HeartbeatWriter into SymbolWorkerApp.run()."""
    source = _APP_MODULE.read_text(encoding="utf-8")

    forbidden = [
        "HeartbeatWriter(",
        "HeartbeatWriterConfig(",
        "run_until_cancelled(",
        "heartbeat_file",
        "heartbeat_task",
        "create_task",
    ]
    for token in forbidden:
        assert token not in source, (
            f"C05 SymbolWorkerApp must not contain {token!r}"
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
