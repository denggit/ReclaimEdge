#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D06 unit tests for signal_handlers — validates signal handler installation,
unsupported platform graceful no-op, custom signals, and source constraints.
"""

from __future__ import annotations

import signal
from pathlib import Path

from src.live.supervisor.reclaim_supervisor import ReclaimSupervisor
from src.live.supervisor.signal_handlers import (
    SignalHandlerInstallResult,
    install_supervisor_signal_handlers,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SIGNAL_HANDLERS_SOURCE_PATH = (
    _PROJECT_ROOT / "src" / "live" / "supervisor" / "signal_handlers.py"
)


def _signal_handlers_source() -> str:
    return _SIGNAL_HANDLERS_SOURCE_PATH.read_text(encoding="utf-8")


# ============================================================================
# FakeLoop — simulates an asyncio event loop for signal handler tests.
# ============================================================================


class FakeLoop:
    def __init__(self, fail: bool = False) -> None:
        self.handlers: dict[str, object] = {}
        self.fail = fail

    def add_signal_handler(self, sig: signal.Signals, callback: object) -> None:
        if self.fail:
            raise NotImplementedError("unsupported")
        self.handlers[sig.name] = callback


# ============================================================================
# 1. test_install_signal_handlers_registers_sigint_sigterm
# ============================================================================


def test_install_signal_handlers_registers_sigint_sigterm() -> None:
    supervisor = ReclaimSupervisor()
    loop = FakeLoop()
    result = install_supervisor_signal_handlers(supervisor, loop=loop)
    assert result.registered == ("SIGINT", "SIGTERM")
    assert result.unsupported == ()
    assert "SIGINT" in loop.handlers
    assert "SIGTERM" in loop.handlers
    # Invoke the SIGINT handler — it should call supervisor.request_stop.
    loop.handlers["SIGINT"]()
    assert supervisor.stop_requested is True


# ============================================================================
# 2. test_install_signal_handlers_unsupported_returns_unsupported
# ============================================================================


def test_install_signal_handlers_unsupported_returns_unsupported() -> None:
    supervisor = ReclaimSupervisor()
    loop = FakeLoop(fail=True)
    result = install_supervisor_signal_handlers(supervisor, loop=loop)
    assert "SIGINT" in result.unsupported
    assert "SIGTERM" in result.unsupported
    assert result.registered == ()
    assert result.ok is False


# ============================================================================
# 3. test_install_signal_handlers_custom_signals
# ============================================================================


def test_install_signal_handlers_custom_signals() -> None:
    supervisor = ReclaimSupervisor()
    loop = FakeLoop()
    result = install_supervisor_signal_handlers(
        supervisor, loop=loop, signals=(signal.SIGTERM,)
    )
    assert result.registered == ("SIGTERM",)
    assert result.unsupported == ()
    assert "SIGTERM" in loop.handlers
    assert "SIGINT" not in loop.handlers


# ============================================================================
# 4. test_signal_handler_result_ok_property
# ============================================================================


def test_signal_handler_result_ok_property() -> None:
    ok_result = SignalHandlerInstallResult(
        registered=("SIGINT", "SIGTERM"),
        unsupported=(),
    )
    assert ok_result.ok is True

    empty_result = SignalHandlerInstallResult(
        registered=(),
        unsupported=("SIGINT", "SIGTERM"),
    )
    assert empty_result.ok is False


# ============================================================================
# 5. test_signal_handlers_source_has_no_child_heartbeat_trading_side_effects
# ============================================================================


def test_signal_handlers_source_has_no_child_heartbeat_trading_side_effects() -> None:
    source = _signal_handlers_source()

    forbidden = [
        "ChildProcess",
        "HeartbeatMonitor",
        "HeartbeatStatus",
        "SymbolWorkerApp",
        "Trader",
        "RuntimePaths",
        "RECLAIM_SYMBOLS",
        "BTC",
        "run_symbol_worker",
        "subprocess",
        "multiprocessing",
        "okx",
        "requests",
        "httpx",
        "websocket",
        "load_dotenv",
        "os.getenv",
        "send_email",
        "EmailSender",
    ]
    for token in forbidden:
        assert token not in source, (
            f"D06 signal_handlers.py must NOT contain {token!r}"
        )
