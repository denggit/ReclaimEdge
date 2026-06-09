#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D03 source guard tests — verifies ChildProcess is NOT wired into
ReclaimSupervisor, run scripts, or workers.  D03 only adds the abstraction;
wiring happens in D05+.
"""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ============================================================================
# 1. test_reclaim_supervisor_does_not_import_child_process_yet
# ============================================================================


def test_reclaim_supervisor_does_not_import_child_process_yet() -> None:
    source = _read(_PROJECT_ROOT / "src" / "live" / "supervisor" / "reclaim_supervisor.py")

    forbidden = [
        "ChildProcess",
        "ChildProcessSpec",
        "create_subprocess_exec",
        "run_symbol_worker",
        "Popen(",
        "Process(",
        "multiprocessing",
        "RECLAIM_SYMBOLS",
        "BTC",
    ]
    for token in forbidden:
        assert token not in source, (
            f"reclaim_supervisor.py must NOT contain {token!r}"
        )


# ============================================================================
# 2. test_run_reclaim_supervisor_does_not_import_child_process_yet
# ============================================================================


def test_run_reclaim_supervisor_does_not_import_child_process_yet() -> None:
    source = _read(_PROJECT_ROOT / "scripts" / "run_reclaim_supervisor.py")

    forbidden = [
        "ChildProcess",
        "ChildProcessSpec",
        "run_symbol_worker",
        "subprocess",
        "multiprocessing",
        "RECLAIM_SYMBOLS",
        "BTC",
    ]
    for token in forbidden:
        assert token not in source, (
            f"run_reclaim_supervisor.py must NOT contain {token!r}"
        )


# ============================================================================
# 3. test_existing_entries_not_changed
# ============================================================================


def test_run_boll_cvd_live_still_uses_symbol_worker_app() -> None:
    source = _read(_PROJECT_ROOT / "scripts" / "run_boll_cvd_live.py")

    assert "SymbolWorkerApp" in source, (
        "run_boll_cvd_live.py must still use SymbolWorkerApp"
    )
    assert "app.run" in source or "app.run(" in source, (
        "run_boll_cvd_live.py must still call app.run"
    )
    assert "ReclaimSupervisor" not in source, (
        "run_boll_cvd_live.py must NOT import ReclaimSupervisor"
    )
    assert "ChildProcess" not in source, (
        "run_boll_cvd_live.py must NOT import ChildProcess"
    )


def test_run_symbol_worker_still_uses_symbol_worker_app() -> None:
    source = _read(_PROJECT_ROOT / "scripts" / "run_symbol_worker.py")

    assert "SymbolWorkerApp" in source, (
        "run_symbol_worker.py must still use SymbolWorkerApp"
    )
    assert "app.run" in source or "app.run(" in source, (
        "run_symbol_worker.py must still call app.run"
    )
    assert "ReclaimSupervisor" not in source, (
        "run_symbol_worker.py must NOT import ReclaimSupervisor"
    )
    assert "ChildProcess" not in source, (
        "run_symbol_worker.py must NOT import ChildProcess"
    )


# ============================================================================
# 4. test_child_process_module_not_used_by_workers
# ============================================================================


def test_child_process_module_not_used_by_workers() -> None:
    workers_dir = _PROJECT_ROOT / "src" / "live" / "workers"
    for worker_path in sorted(workers_dir.glob("*.py")):
        if worker_path.name == "__init__.py":
            continue
        source = _read(worker_path)
        forbidden = [
            "ChildProcess",
            "create_subprocess_exec",
            "run_symbol_worker",
        ]
        for token in forbidden:
            assert token not in source, (
                f"{worker_path.name} must NOT contain {token!r}"
            )
