#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D07 source guard tests — verifies ChildProcess and HeartbeatMonitor ARE
wired into ReclaimSupervisor (ETH-only, no BTC, no multi-symbol, no restart),
signal handlers are installed from the entry, and the entry script
and workers remain untouched.
"""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ============================================================================
# 1. test_reclaim_supervisor_imports_child_process
# ============================================================================


def test_reclaim_supervisor_imports_child_process() -> None:
    source = _read(_PROJECT_ROOT / "src" / "live" / "supervisor" / "reclaim_supervisor.py")

    required = [
        "ChildProcess",
        "ChildProcessSpec",
        "build_child_spec",
        "create_child_process",
        "scripts/run_symbol_worker.py",
        # D07b allows restart_policy in supervisor.
        "RestartPolicy",
        "RestartPolicyConfig",
    ]
    for token in required:
        assert token in source, (
            f"D07 reclaim_supervisor.py must contain {token!r}"
        )


# ============================================================================
# 2. test_reclaim_supervisor_imports_heartbeat_monitor
# ============================================================================


def test_reclaim_supervisor_imports_heartbeat_monitor() -> None:
    source = _read(_PROJECT_ROOT / "src" / "live" / "supervisor" / "reclaim_supervisor.py")

    # D07 now wires HeartbeatMonitor into ReclaimSupervisor.
    required = [
        "HeartbeatMonitor",
        "HeartbeatMonitorConfig",
        "HeartbeatStatus",
        "check_heartbeat_once",
        "maybe_check_heartbeat",
    ]
    for token in required:
        assert token in source, (
            f"D07 reclaim_supervisor.py must contain {token!r}"
        )


# ============================================================================
# 3. test_reclaim_supervisor_eth_only_no_multi_symbol
# ============================================================================


def test_reclaim_supervisor_eth_only_no_multi_symbol() -> None:
    source = _read(_PROJECT_ROOT / "src" / "live" / "supervisor" / "reclaim_supervisor.py")

    assert "ETH-USDT-SWAP" in source, (
        "D07 reclaim_supervisor.py must contain ETH-USDT-SWAP (single child)"
    )

    forbidden = [
        "RECLAIM_SYMBOLS",
        "BTC-USDT-SWAP",
        "argparse",
        "--symbol",
    ]
    for token in forbidden:
        assert token not in source, (
            f"D07 reclaim_supervisor.py must NOT contain {token!r}"
        )

    # BTC token should not appear anywhere.
    assert "BTC" not in source, (
        "D07 reclaim_supervisor.py must NOT contain BTC"
    )


# ============================================================================
# 4. test_run_reclaim_supervisor_entry_still_not_wired_directly_to_child_process
# ============================================================================


def test_run_reclaim_supervisor_entry_still_not_wired_directly_to_child_process() -> None:
    source = _read(_PROJECT_ROOT / "scripts" / "run_reclaim_supervisor.py")

    # D06 allows install_supervisor_signal_handlers in the entry.
    assert "install_supervisor_signal_handlers" in source, (
        "D07 run_reclaim_supervisor.py must install signal handlers"
    )

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
            f"D07 run_reclaim_supervisor.py must NOT contain {token!r}"
        )


# ============================================================================
# 5. test_existing_entries_not_changed
# ============================================================================


def test_run_boll_cvd_live_still_uses_symbol_worker_app() -> None:
    source = _read(_PROJECT_ROOT / "scripts" / "run_boll_cvd_live.py")

    assert "SymbolWorkerApp" in source, (
        "run_boll_cvd_live.py must still use SymbolWorkerApp"
    )
    assert "app.run" in source or "app.run(" in source, (
        "run_boll_cvd_live.py must still call app.run"
    )
    for token in ["ReclaimSupervisor", "ChildProcess", "HeartbeatMonitor"]:
        assert token not in source, (
            f"run_boll_cvd_live.py must NOT contain {token!r}"
        )


def test_run_symbol_worker_still_uses_symbol_worker_app() -> None:
    source = _read(_PROJECT_ROOT / "scripts" / "run_symbol_worker.py")

    assert "SymbolWorkerApp" in source, (
        "run_symbol_worker.py must still use SymbolWorkerApp"
    )
    assert "app.run" in source or "app.run(" in source, (
        "run_symbol_worker.py must still call app.run"
    )
    for token in ["ReclaimSupervisor", "ChildProcess", "HeartbeatMonitor"]:
        assert token not in source, (
            f"run_symbol_worker.py must NOT contain {token!r}"
        )


# ============================================================================
# 6. test_workers_do_not_import_child_process
# ============================================================================


def test_workers_do_not_import_child_process() -> None:
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
