#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D04 source guard tests — verifies HeartbeatMonitor is NOT wired into
ReclaimSupervisor, run scripts, or ChildProcess.  D04 only adds the module;
wiring happens in D05+.
"""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ============================================================================
# 1. test_reclaim_supervisor_does_not_import_heartbeat_monitor_yet
# ============================================================================


def test_reclaim_supervisor_does_not_import_heartbeat_monitor_yet() -> None:
    source = _read(_PROJECT_ROOT / "src" / "live" / "supervisor" / "reclaim_supervisor.py")

    forbidden = [
        "HeartbeatMonitor",
        "HeartbeatStatus",
        "read_status",
        "heartbeat_file",
        "heartbeats_dir",
        "ChildProcess",
        "run_symbol_worker",
        "RECLAIM_SYMBOLS",
        "BTC",
    ]
    for token in forbidden:
        assert token not in source, (
            f"reclaim_supervisor.py must NOT contain {token!r}"
        )


# ============================================================================
# 2. test_run_reclaim_supervisor_does_not_import_heartbeat_monitor_yet
# ============================================================================


def test_run_reclaim_supervisor_does_not_import_heartbeat_monitor_yet() -> None:
    source = _read(_PROJECT_ROOT / "scripts" / "run_reclaim_supervisor.py")

    forbidden = [
        "HeartbeatMonitor",
        "HeartbeatStatus",
        "heartbeat_file",
        "heartbeats_dir",
        "ChildProcess",
        "run_symbol_worker",
        "RECLAIM_SYMBOLS",
        "BTC",
    ]
    for token in forbidden:
        assert token not in source, (
            f"run_reclaim_supervisor.py must NOT contain {token!r}"
        )


# ============================================================================
# 3. test_child_process_does_not_import_heartbeat_monitor
# ============================================================================


def test_child_process_does_not_import_heartbeat_monitor() -> None:
    source = _read(_PROJECT_ROOT / "src" / "live" / "supervisor" / "child_process.py")

    forbidden = [
        "HeartbeatMonitor",
        "HeartbeatStatus",
        "heartbeat_file",
        "heartbeats_dir",
    ]
    for token in forbidden:
        assert token not in source, (
            f"child_process.py must NOT contain {token!r}"
        )


# ============================================================================
# 4. test_existing_entries_not_changed
# ============================================================================


def test_existing_entries_not_changed() -> None:
    # run_boll_cvd_live.py: still uses SymbolWorkerApp.from_env and app.run
    boll_source = _read(_PROJECT_ROOT / "scripts" / "run_boll_cvd_live.py")
    assert "SymbolWorkerApp.from_env" in boll_source or "SymbolWorkerApp.from_env(" in boll_source, (
        "run_boll_cvd_live.py must still use SymbolWorkerApp.from_env"
    )
    assert "app.run" in boll_source or "app.run(" in boll_source, (
        "run_boll_cvd_live.py must still call app.run"
    )
    for token in ["ReclaimSupervisor", "ChildProcess", "HeartbeatMonitor"]:
        assert token not in boll_source, (
            f"run_boll_cvd_live.py must NOT contain {token!r}"
        )

    # run_symbol_worker.py: still uses SymbolWorkerApp.from_env and app.run
    worker_source = _read(_PROJECT_ROOT / "scripts" / "run_symbol_worker.py")
    assert "SymbolWorkerApp.from_env" in worker_source or "SymbolWorkerApp.from_env(" in worker_source, (
        "run_symbol_worker.py must still use SymbolWorkerApp.from_env"
    )
    assert "app.run" in worker_source or "app.run(" in worker_source, (
        "run_symbol_worker.py must still call app.run"
    )
    for token in ["ReclaimSupervisor", "ChildProcess", "HeartbeatMonitor"]:
        assert token not in worker_source, (
            f"run_symbol_worker.py must NOT contain {token!r}"
        )


# ============================================================================
# 5. test_workers_do_not_import_heartbeat_monitor
# ============================================================================


def test_workers_do_not_import_heartbeat_monitor() -> None:
    workers_dir = _PROJECT_ROOT / "src" / "live" / "workers"
    for worker_path in sorted(workers_dir.glob("*.py")):
        if worker_path.name == "__init__.py":
            continue
        source = _read(worker_path)
        forbidden = [
            "HeartbeatMonitor",
            "heartbeat_monitor",
            "read_status",
        ]
        for token in forbidden:
            assert token not in source, (
                f"{worker_path.name} must NOT contain {token!r}"
            )
