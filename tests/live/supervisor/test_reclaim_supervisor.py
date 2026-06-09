#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D02 unit tests for ReclaimSupervisor empty shell — validates config,
from_env, request_stop, and the idle loop behaviour without importing
any trading or child-process modules.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.live.supervisor.reclaim_supervisor import (
    ReclaimSupervisor,
    ReclaimSupervisorConfig,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SUPERVISOR_SOURCE_PATH = (
    _PROJECT_ROOT / "src" / "live" / "supervisor" / "reclaim_supervisor.py"
)


def _supervisor_source() -> str:
    return _SUPERVISOR_SOURCE_PATH.read_text(encoding="utf-8")


# ============================================================================
# 1. test_default_config
# ============================================================================


def test_default_config() -> None:
    config = ReclaimSupervisorConfig()
    assert config.poll_interval_seconds == 5.0


# ============================================================================
# 2. test_invalid_poll_interval_raises
# ============================================================================


def test_invalid_poll_interval_raises() -> None:
    with pytest.raises(ValueError, match="poll_interval_seconds must be > 0"):
        ReclaimSupervisorConfig(poll_interval_seconds=0)


# ============================================================================
# 3. test_from_env_returns_empty_shell
# ============================================================================


def test_from_env_returns_empty_shell() -> None:
    supervisor = ReclaimSupervisor.from_env()
    assert isinstance(supervisor, ReclaimSupervisor)
    assert supervisor.config.poll_interval_seconds == 5.0
    assert supervisor.started_at_ms is None
    assert supervisor.stop_requested is False


# ============================================================================
# 4. test_request_stop
# ============================================================================


def test_request_stop() -> None:
    supervisor = ReclaimSupervisor()
    assert not supervisor.stop_requested
    supervisor.request_stop()
    assert supervisor.stop_requested


# ============================================================================
# 5. test_run_forever_sets_started_at_and_stops_on_request
# ============================================================================


@pytest.mark.asyncio
async def test_run_forever_sets_started_at_and_stops_on_request() -> None:
    config = ReclaimSupervisorConfig(poll_interval_seconds=0.01)
    supervisor = ReclaimSupervisor(config=config)
    task = asyncio.create_task(supervisor.run_forever())
    await asyncio.sleep(0.02)
    assert supervisor.started_at_ms is not None
    supervisor.request_stop()
    await asyncio.wait_for(task, timeout=1.0)
    assert supervisor.stop_requested is True


# ============================================================================
# 6. test_run_forever_propagates_cancelled_error
# ============================================================================


@pytest.mark.asyncio
async def test_run_forever_propagates_cancelled_error() -> None:
    config = ReclaimSupervisorConfig(poll_interval_seconds=0.01)
    supervisor = ReclaimSupervisor(config=config)
    task = asyncio.create_task(supervisor.run_forever())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# ============================================================================
# 7. test_supervisor_source_has_no_child_process_or_trading_side_effects
# ============================================================================


def test_supervisor_source_has_no_child_process_or_trading_side_effects() -> None:
    source = _supervisor_source()

    forbidden = [
        "subprocess",
        "multiprocessing",
        "Popen(",
        "Process(",
        "run_symbol_worker",
        "SymbolWorkerApp",
        "SymbolWorkerFactory",
        "Trader",
        "RuntimePaths",
        "HeartbeatWriter",
        "HeartbeatMonitor",
        "ChildProcess",
        "RECLAIM_SYMBOLS",
        "BTC",
        "account_position_sync_worker",
        "strategy_tick_worker",
        "execution_worker",
        "BollCvd",
        "okx",
        "requests",
        "httpx",
        "websocket",
        "load_dotenv",
        "os.getenv",
    ]
    for token in forbidden:
        assert token not in source, (
            f"reclaim_supervisor.py must NOT contain {token!r}"
        )


# ============================================================================
# 8. test_supervisor_loop_logs_are_not_spammy
# ============================================================================


def test_supervisor_loop_logs_are_not_spammy() -> None:
    source = _supervisor_source()

    assert "RECLAIM_SUPERVISOR_STARTED" in source, (
        "reclaim_supervisor.py must log RECLAIM_SUPERVISOR_STARTED"
    )
    assert "RECLAIM_SUPERVISOR_STOPPED" in source, (
        "reclaim_supervisor.py must log RECLAIM_SUPERVISOR_STOPPED"
    )
    assert "while not self._stop_requested" in source, (
        "reclaim_supervisor.py must have an idle while loop"
    )

    # The while loop body must NOT contain logger.info — no per-tick spam.
    lines = source.splitlines()
    inside_while = False
    for line in lines:
        stripped = line.strip()
        if "while not self._stop_requested" in stripped:
            inside_while = True
            continue
        if inside_while:
            if stripped.startswith("except ") or stripped.startswith("finally "):
                break
            assert "logger." not in stripped, (
                f"reclaim_supervisor.py while loop must not log inside the loop — found: {stripped!r}"
            )
