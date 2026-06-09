#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D05 unit tests for ReclaimSupervisor — validates config, from_env,
request_stop, child process wiring for single ETH child, and the idle loop
behaviour.  D05 does NOT wire heartbeat, multi-symbol, or restart logic.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from src.live.supervisor.child_process import ChildProcessSnapshot, ChildProcessSpec
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
# Fake child process for tests that must not launch real subprocesses.
# ============================================================================


class FakeChildProcess:
    """Duck-type compatible with ChildProcess for unit-testing run_forever."""

    def __init__(self, spec: ChildProcessSpec) -> None:
        self.spec = spec
        self.started = False
        self.terminated = False
        self._pid = 1234

    @property
    def pid(self) -> int:
        return self._pid

    async def start(self) -> ChildProcessSnapshot:
        self.started = True
        return ChildProcessSnapshot(
            name=self.spec.name,
            pid=self._pid,
            returncode=None,
            running=True,
            started=True,
        )

    async def terminate(self) -> ChildProcessSnapshot:
        self.terminated = True
        return ChildProcessSnapshot(
            name=self.spec.name,
            pid=self._pid,
            returncode=0,
            running=False,
            started=True,
        )

    def snapshot(self) -> ChildProcessSnapshot:
        return ChildProcessSnapshot(
            name=self.spec.name,
            pid=self._pid,
            returncode=None if self.started and not self.terminated else 0,
            running=self.started and not self.terminated,
            started=self.started,
        )


class FailingFakeChildProcess(FakeChildProcess):
    """A fake child whose start() raises."""

    async def start(self) -> ChildProcessSnapshot:
        self.started = True
        raise RuntimeError("boom")


# ============================================================================
# 1. test_default_config
# ============================================================================


def test_default_config() -> None:
    config = ReclaimSupervisorConfig()
    assert config.poll_interval_seconds == 5.0
    assert config.child_name == "ETH-USDT-SWAP"
    assert config.worker_script == Path("scripts/run_symbol_worker.py")
    assert config.child_terminate_timeout_seconds == 10.0
    assert config.child_kill_timeout_seconds == 5.0


# ============================================================================
# 2. test_invalid_poll_interval_raises
# ============================================================================


def test_invalid_poll_interval_raises() -> None:
    with pytest.raises(ValueError, match="poll_interval_seconds must be > 0"):
        ReclaimSupervisorConfig(poll_interval_seconds=0)


# ============================================================================
# 3. test_invalid_child_name_raises
# ============================================================================


def test_invalid_child_name_raises() -> None:
    with pytest.raises(ValueError, match="child_name"):
        ReclaimSupervisorConfig(child_name=" ")


# ============================================================================
# 4. test_invalid_child_timeouts_raise
# ============================================================================


def test_invalid_child_timeouts_raise() -> None:
    with pytest.raises(ValueError, match="child_terminate_timeout_seconds must be > 0"):
        ReclaimSupervisorConfig(child_terminate_timeout_seconds=0)

    with pytest.raises(ValueError, match="child_kill_timeout_seconds must be > 0"):
        ReclaimSupervisorConfig(child_kill_timeout_seconds=0)


# ============================================================================
# 5. test_from_env_returns_single_child_supervisor
# ============================================================================


def test_from_env_returns_single_child_supervisor() -> None:
    supervisor = ReclaimSupervisor.from_env()
    assert isinstance(supervisor, ReclaimSupervisor)
    assert supervisor.config.child_name == "ETH-USDT-SWAP"
    assert supervisor.child is None
    assert supervisor.child_snapshot is None
    assert supervisor.started_at_ms is None
    assert supervisor.stop_requested is False


# ============================================================================
# 6. test_build_child_spec
# ============================================================================


def test_build_child_spec(tmp_path: Path) -> None:
    config = ReclaimSupervisorConfig(
        project_root=tmp_path,
        worker_script=Path("scripts/run_symbol_worker.py"),
    )
    supervisor = ReclaimSupervisor(config=config)
    spec = supervisor.build_child_spec()
    assert spec.name == "ETH-USDT-SWAP"
    assert spec.argv == (sys.executable, str(tmp_path / "scripts" / "run_symbol_worker.py"))
    assert spec.cwd == tmp_path
    assert spec.env is None
    assert spec.terminate_timeout_seconds == 10.0
    assert spec.kill_timeout_seconds == 5.0


# ============================================================================
# 7. test_create_child_process_returns_child_process
# ============================================================================


def test_create_child_process_returns_child_process() -> None:
    supervisor = ReclaimSupervisor()
    child = supervisor.create_child_process()
    from src.live.supervisor.child_process import ChildProcess

    assert isinstance(child, ChildProcess)
    assert child.spec.name == "ETH-USDT-SWAP"


# ============================================================================
# 8. test_request_stop
# ============================================================================


def test_request_stop() -> None:
    supervisor = ReclaimSupervisor()
    assert not supervisor.stop_requested
    supervisor.request_stop()
    assert supervisor.stop_requested


# ============================================================================
# 9. test_run_forever_starts_child_and_stops_on_request
# ============================================================================


@pytest.mark.asyncio
async def test_run_forever_starts_child_and_stops_on_request() -> None:
    config = ReclaimSupervisorConfig(poll_interval_seconds=0.01)
    supervisor = ReclaimSupervisor(config=config)
    fake_child = FakeChildProcess(supervisor.build_child_spec())
    supervisor.create_child_process = lambda: fake_child  # type: ignore[method-assign]

    task = asyncio.create_task(supervisor.run_forever())
    await asyncio.sleep(0.02)
    assert supervisor.started_at_ms is not None
    assert fake_child.started is True
    assert supervisor.child is fake_child
    supervisor.request_stop()
    await asyncio.wait_for(task, timeout=1.0)
    assert fake_child.terminated is True


# ============================================================================
# 10. test_run_forever_cancel_terminates_child_and_propagates_cancelled_error
# ============================================================================


@pytest.mark.asyncio
async def test_run_forever_cancel_terminates_child_and_propagates_cancelled_error() -> None:
    config = ReclaimSupervisorConfig(poll_interval_seconds=0.01)
    supervisor = ReclaimSupervisor(config=config)
    fake_child = FakeChildProcess(supervisor.build_child_spec())
    supervisor.create_child_process = lambda: fake_child  # type: ignore[method-assign]

    task = asyncio.create_task(supervisor.run_forever())
    await asyncio.sleep(0.02)
    assert fake_child.started is True
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fake_child.terminated is True


# ============================================================================
# 11. test_run_forever_start_failure_propagates
# ============================================================================


@pytest.mark.asyncio
async def test_run_forever_start_failure_propagates() -> None:
    config = ReclaimSupervisorConfig(poll_interval_seconds=0.01)
    supervisor = ReclaimSupervisor(config=config)
    fake_child = FailingFakeChildProcess(supervisor.build_child_spec())
    supervisor.create_child_process = lambda: fake_child  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="boom"):
        await supervisor.run_forever()
    # started_at_ms is set before start is awaited.
    assert supervisor.started_at_ms is not None
    # _child was assigned before start, so finally attempted terminate.
    assert fake_child.terminated is True


# ============================================================================
# 12. test_supervisor_source_now_uses_child_process_but_no_heartbeat_or_multi_symbol
# ============================================================================


def test_supervisor_source_now_uses_child_process_but_no_heartbeat_or_multi_symbol() -> None:
    source = _supervisor_source()

    # D05 now wires ChildProcess — these must be present.
    assert "ChildProcess" in source, (
        "D05 reclaim_supervisor.py must import ChildProcess"
    )
    assert "ChildProcessSpec" in source, (
        "D05 reclaim_supervisor.py must import ChildProcessSpec"
    )
    assert "scripts/run_symbol_worker.py" in source, (
        "D05 reclaim_supervisor.py must reference scripts/run_symbol_worker.py"
    )

    # D05 must NOT wire heartbeat, multi-symbol, or trading modules.
    forbidden = [
        "HeartbeatMonitor",
        "HeartbeatStatus",
        "read_status",
        "RECLAIM_SYMBOLS",
        "BTC-USDT-SWAP",
        "BTC",
        "multiprocessing",
        "subprocess.Popen",
        "Popen(",
        "SymbolWorkerApp",
        "Trader",
        "RuntimePaths",
        "account_position_sync_worker",
        "strategy_tick_worker",
        "execution_worker",
        "BollCvd",
        "okx",
        "requests",
        "httpx",
        "websocket",
        "os.getenv",
    ]
    for token in forbidden:
        assert token not in source, (
            f"D05 reclaim_supervisor.py must NOT contain {token!r}"
        )


# ============================================================================
# 13. test_supervisor_loop_logs_are_not_spammy
# ============================================================================


def test_supervisor_loop_logs_are_not_spammy() -> None:
    source = _supervisor_source()

    assert "RECLAIM_SUPERVISOR_STARTED" in source, (
        "reclaim_supervisor.py must log RECLAIM_SUPERVISOR_STARTED"
    )
    assert "RECLAIM_SUPERVISOR_CHILD_STARTED" in source, (
        "reclaim_supervisor.py must log RECLAIM_SUPERVISOR_CHILD_STARTED"
    )
    assert "RECLAIM_SUPERVISOR_STOPPING" in source, (
        "reclaim_supervisor.py must log RECLAIM_SUPERVISOR_STOPPING"
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
