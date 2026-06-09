#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D06b unit tests for ``src.live.worker_shutdown``.

These tests verify the ``WorkerShutdownController`` and the signal handler
installer.  They do NOT connect to OKX, start a trader, or run the live
event loop.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.live.worker_shutdown import (
    WorkerShutdownController,
    WorkerShutdownInstallResult,
    install_symbol_worker_signal_handlers,
)


# ============================================================================
# WorkerShutdownController
# ============================================================================


class TestWorkerShutdownController:
    def test_initial_state(self) -> None:
        controller = WorkerShutdownController()
        assert controller.requested is False
        assert controller.reason is None

    def test_request_shutdown_sets_event_and_reason(self) -> None:
        controller = WorkerShutdownController()
        controller.request_shutdown("SIGTERM")
        assert controller.requested is True
        assert controller.reason == "SIGTERM"
        assert controller.event.is_set()

    def test_request_shutdown_is_idempotent(self) -> None:
        controller = WorkerShutdownController()
        controller.request_shutdown("SIGTERM")
        # Second call must not overwrite the original reason
        controller.request_shutdown("SIGINT")
        assert controller.requested is True
        assert controller.reason == "SIGTERM"

    def test_default_reason_is_signal(self) -> None:
        controller = WorkerShutdownController()
        controller.request_shutdown()
        assert controller.reason == "signal"


# ============================================================================
# WorkerShutdownInstallResult
# ============================================================================


class TestWorkerShutdownInstallResult:
    def test_ok_true_when_registered(self) -> None:
        result = WorkerShutdownInstallResult(
            registered=("SIGTERM",),
            unsupported=(),
        )
        assert result.ok is True

    def test_ok_false_when_no_registered(self) -> None:
        result = WorkerShutdownInstallResult(
            registered=(),
            unsupported=("SIGTERM",),
        )
        assert result.ok is False


# ============================================================================
# install_symbol_worker_signal_handlers
# ============================================================================


class TestInstallSymbolWorkerSignalHandlers:
    def test_registers_default_signals(self) -> None:
        controller = WorkerShutdownController()

        async def _run() -> WorkerShutdownInstallResult:
            return install_symbol_worker_signal_handlers(controller)

        result = asyncio.run(_run())
        assert result.ok is True
        assert "SIGINT" in result.registered
        assert "SIGTERM" in result.registered
        assert len(result.unsupported) == 0

    def test_registers_custom_signals(self) -> None:
        import signal

        controller = WorkerShutdownController()

        async def _run() -> WorkerShutdownInstallResult:
            return install_symbol_worker_signal_handlers(
                controller,
                signals=(signal.SIGTERM,),
            )

        result = asyncio.run(_run())
        assert result.ok is True
        assert "SIGTERM" in result.registered
        assert "SIGINT" not in result.registered

    def test_handles_unsupported_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When add_signal_handler raises, the signal is recorded as unsupported."""
        import signal

        class _UnsupportedLoop:
            def add_signal_handler(self, sig, callback, *args):
                raise NotImplementedError("no signal handlers")

        controller = WorkerShutdownController()
        result = install_symbol_worker_signal_handlers(
            controller,
            loop=_UnsupportedLoop(),  # type: ignore[arg-type]
            signals=(signal.SIGTERM,),
        )
        assert result.ok is False
        assert len(result.registered) == 0
        assert "SIGTERM" in result.unsupported

    def test_signal_triggers_shutdown(self) -> None:
        """End-to-end: installing a handler and sending the signal triggers
        the controller."""
        import os
        import signal

        async def _run() -> bool:
            controller = WorkerShutdownController()
            install_symbol_worker_signal_handlers(controller)
            # Send SIGINT to our own process
            os.kill(os.getpid(), signal.SIGINT)
            # Give the event loop a tick to process the signal
            await asyncio.sleep(0.01)
            return controller.requested

        requested = asyncio.run(_run())
        assert requested is True


# ============================================================================
# Source guard — worker_shutdown.py
# ============================================================================


class TestWorkerShutdownSourceGuard:
    def test_no_trading_modules(self) -> None:
        """worker_shutdown.py must not import Trader, Strategy, or any
        trading / order / cancel / OKX modules."""
        source = Path("src/live/worker_shutdown.py").read_text(encoding="utf-8")

        forbidden = [
            "Trader",
            "Strategy",
            "ChildProcess",
            "HeartbeatMonitor",
            "BollCvd",
            "SymbolWorkerApp",
            "order",
            "cancel",
            "close",
            "okx",
            "httpx",
            "requests",
            "websocket",
            "EmailSender",
        ]
        for token in forbidden:
            assert token not in source, (
                f"worker_shutdown.py must not reference {token!r}"
            )

    def test_allows_signal_and_event(self) -> None:
        """worker_shutdown.py must contain signal and asyncio.Event imports."""
        source = Path("src/live/worker_shutdown.py").read_text(encoding="utf-8")

        allowed = [
            "signal",
            "asyncio.Event",
            "WorkerShutdownController",
            "install_symbol_worker_signal_handlers",
        ]
        for token in allowed:
            assert token in source, (
                f"worker_shutdown.py must contain {token!r}"
            )
