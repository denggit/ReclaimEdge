#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D07 source guard — verifies graceful shutdown wiring: SupervisorShutdownResult,
shutdown() method, signal handlers install, and that heartbeat detection is now
wired into ReclaimSupervisor.  Forbids restart, BTC, email, multi-symbol.
"""

from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ============================================================================
# 1. SupervisorShutdownResult wired
# ============================================================================


def test_supervisor_shutdown_result_exists_in_reclaim_supervisor() -> None:
    source = _read(_PROJECT_ROOT / "src" / "live" / "supervisor" / "reclaim_supervisor.py")

    required = [
        "class SupervisorShutdownResult",
        "child_name:",
        "child_started:",
        "child_running_before_shutdown:",
        "child_pid:",
        "child_returncode_before_shutdown:",
        "terminate_attempted:",
        "terminate_error:",
        "def ok(self)",
    ]
    for token in required:
        assert token in source, (
            f"D07 reclaim_supervisor.py must contain SupervisorShutdownResult with {token!r}"
        )


# ============================================================================
# 2. shutdown method exists and is idempotent
# ============================================================================


def test_shutdown_method_exists_and_is_idempotent() -> None:
    source = _read(_PROJECT_ROOT / "src" / "live" / "supervisor" / "reclaim_supervisor.py")

    assert "async def shutdown(self)" in source, (
        "D07 reclaim_supervisor.py must have async shutdown method"
    )
    assert "self._shutdown_result is not None" in source, (
        "D07 shutdown must be idempotent via _shutdown_result check"
    )
    assert "self._shutdown_started = True" in source, (
        "D07 shutdown must set _shutdown_started"
    )


# ============================================================================
# 3. request_stop logs
# ============================================================================


def test_request_stop_logs() -> None:
    source = _read(_PROJECT_ROOT / "src" / "live" / "supervisor" / "reclaim_supervisor.py")

    assert "RECLAIM_SUPERVISOR_STOP_REQUESTED" in source, (
        "D07 request_stop must log RECLAIM_SUPERVISOR_STOP_REQUESTED"
    )


# ============================================================================
# 4. run_forever uses shutdown in finally
# ============================================================================


def test_run_forever_uses_shutdown_in_finally() -> None:
    source = _read(_PROJECT_ROOT / "src" / "live" / "supervisor" / "reclaim_supervisor.py")

    assert "result = await self.shutdown()" in source, (
        "D07 run_forever finally must call await self.shutdown()"
    )
    assert "RECLAIM_SUPERVISOR_STOPPED" in source, (
        "D07 run_forever finally must log RECLAIM_SUPERVISOR_STOPPED with result fields"
    )


# ============================================================================
# 5. signal_handlers module exists
# ============================================================================


def test_signal_handlers_module_exists() -> None:
    path = _PROJECT_ROOT / "src" / "live" / "supervisor" / "signal_handlers.py"
    assert path.exists(), "D07 must keep src/live/supervisor/signal_handlers.py"


# ============================================================================
# 6. signal_handlers exports correct names
# ============================================================================


def test_signal_handlers_exports_correct_names() -> None:
    source = _read(_PROJECT_ROOT / "src" / "live" / "supervisor" / "signal_handlers.py")

    assert "class SignalHandlerInstallResult" in source, (
        "D07 signal_handlers.py must define SignalHandlerInstallResult"
    )
    assert "def install_supervisor_signal_handlers" in source, (
        "D07 signal_handlers.py must define install_supervisor_signal_handlers"
    )
    assert "add_signal_handler" in source, (
        "D07 signal_handlers.py must use add_signal_handler"
    )


# ============================================================================
# 7. supervisor init exports new names
# ============================================================================


def test_supervisor_init_exports_shutdown_and_signal_names() -> None:
    source = _read(_PROJECT_ROOT / "src" / "live" / "supervisor" / "__init__.py")

    assert "SupervisorShutdownResult" in source, (
        "D07 __init__.py must export SupervisorShutdownResult"
    )
    assert "SignalHandlerInstallResult" in source, (
        "D07 __init__.py must export SignalHandlerInstallResult"
    )
    assert "install_supervisor_signal_handlers" in source, (
        "D07 __init__.py must export install_supervisor_signal_handlers"
    )


# ============================================================================
# 8. entry installs signal handlers
# ============================================================================


def test_entry_installs_signal_handlers() -> None:
    source = _read(_PROJECT_ROOT / "scripts" / "run_reclaim_supervisor.py")

    assert "install_supervisor_signal_handlers(supervisor)" in source, (
        "D07 run_reclaim_supervisor.py must call install_supervisor_signal_handlers"
    )


# ============================================================================
# 9. heartbeat monitoring now wired, but no restart / BTC / email
# ============================================================================


def test_supervisor_has_heartbeat_but_no_restart_btc_email() -> None:
    source = _read(_PROJECT_ROOT / "src" / "live" / "supervisor" / "reclaim_supervisor.py")

    # D07 wires heartbeat detection — these must be present.
    allowed = [
        "HeartbeatMonitor",
        "HeartbeatStatus",
        "HeartbeatMonitorConfig",
        "RuntimePaths",
        "check_heartbeat_once",
        "maybe_check_heartbeat",
    ]
    for token in allowed:
        assert token in source, (
            f"D07 reclaim_supervisor.py must contain {token!r}"
        )

    # Forbidden: restart, BTC, multi-symbol, trading modules, email.
    forbidden = [
        "restart",
        "relaunch",
        "RECLAIM_SYMBOLS",
        "BTC-USDT-SWAP",
        "SymbolWorkerApp",
        "Trader",
        "EmailSender",
        "send_email",
    ]
    for token in forbidden:
        assert token not in source, (
            f"D07 reclaim_supervisor.py must NOT contain {token!r}"
        )

    # BTC token must not appear anywhere.
    assert "BTC" not in source, (
        "D07 reclaim_supervisor.py must NOT contain BTC"
    )


# ============================================================================
# 10. entry does not directly import child/heartbeat machinery
# ============================================================================


def test_entry_no_direct_child_or_heartbeat() -> None:
    source = _read(_PROJECT_ROOT / "scripts" / "run_reclaim_supervisor.py")

    forbidden = [
        "ChildProcess",
        "ChildProcessSpec",
        "HeartbeatMonitor",
        "HeartbeatStatus",
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
# 11. existing entries unchanged
# ============================================================================


def test_run_boll_cvd_live_unchanged() -> None:
    source = _read(_PROJECT_ROOT / "scripts" / "run_boll_cvd_live.py")

    assert "SymbolWorkerApp" in source, (
        "run_boll_cvd_live.py must still use SymbolWorkerApp"
    )
    for token in ["ReclaimSupervisor", "SupervisorShutdownResult", "SignalHandlerInstallResult", "install_supervisor_signal_handlers"]:
        assert token not in source, (
            f"run_boll_cvd_live.py must NOT contain {token!r}"
        )


def test_run_symbol_worker_unchanged() -> None:
    source = _read(_PROJECT_ROOT / "scripts" / "run_symbol_worker.py")

    assert "SymbolWorkerApp" in source, (
        "run_symbol_worker.py must still use SymbolWorkerApp"
    )
    for token in ["ReclaimSupervisor", "SupervisorShutdownResult", "SignalHandlerInstallResult", "install_supervisor_signal_handlers"]:
        assert token not in source, (
            f"run_symbol_worker.py must NOT contain {token!r}"
        )
