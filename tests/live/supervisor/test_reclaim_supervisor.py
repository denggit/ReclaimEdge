#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D06 unit tests for ReclaimSupervisor — validates config, from_env,
request_stop, graceful shutdown (shutdown idempotency, terminate behaviour,
error handling, CancelledError propagation), child process wiring for single
ETH child, and the idle loop behaviour.
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
    SupervisorShutdownResult,
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
    """Duck-type compatible with ChildProcess for unit-testing run_forever
    and shutdown.  Supports configurable running / returncode / terminate
    error behaviour."""

    def __init__(
        self,
        spec: ChildProcessSpec,
        *,
        returncode: int | None = None,
        running: bool = True,
        terminate_error: Exception | None = None,
        terminate_cancelled: bool = False,
    ) -> None:
        self.spec = spec
        self.started = False
        self.terminated = False
        self._pid = 1234
        self._returncode = returncode
        self._running = running
        self._terminate_error = terminate_error
        self._terminate_cancelled = terminate_cancelled
        self.terminate_calls = 0

    @property
    def pid(self) -> int | None:
        return self._pid if self.started else None

    async def start(self) -> ChildProcessSnapshot:
        self.started = True
        return self.snapshot()

    async def terminate(self) -> ChildProcessSnapshot:
        self.terminate_calls += 1
        if self._terminate_cancelled:
            raise asyncio.CancelledError()
        if self._terminate_error is not None:
            raise self._terminate_error
        self.terminated = True
        return self.snapshot()

    def snapshot(self) -> ChildProcessSnapshot:
        return ChildProcessSnapshot(
            name=self.spec.name,
            pid=self.pid,
            returncode=self._returncode,
            running=self._running if self.started else False,
            started=self.started,
        )


class FailingFakeChildProcess(FakeChildProcess):
    """A fake child whose start() raises after marking itself started."""

    def __init__(self, spec: ChildProcessSpec) -> None:
        super().__init__(spec, running=True)
        # start() will set started=True before raising — simulating a child
        # that was created but start itself fails.

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
    assert supervisor.shutdown_started is False
    assert supervisor.shutdown_result is None


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
    # Idempotent: second call does not change state.
    supervisor.request_stop()
    assert supervisor.stop_requested


# ============================================================================
# 9. test_shutdown_result_ok_property
# ============================================================================


def test_shutdown_result_ok_property() -> None:
    ok_result = SupervisorShutdownResult(
        child_name="test",
        child_started=True,
        child_running_before_shutdown=True,
        child_pid=123,
        child_returncode_before_shutdown=None,
        terminate_attempted=True,
        terminate_error=None,
    )
    assert ok_result.ok is True

    error_result = SupervisorShutdownResult(
        child_name="test",
        child_started=True,
        child_running_before_shutdown=True,
        child_pid=123,
        child_returncode_before_shutdown=None,
        terminate_attempted=True,
        terminate_error="RuntimeError: terminate failed",
    )
    assert error_result.ok is False


# ============================================================================
# 10. test_shutdown_before_child_created
# ============================================================================


@pytest.mark.asyncio
async def test_shutdown_before_child_created() -> None:
    supervisor = ReclaimSupervisor()
    result = await supervisor.shutdown()
    assert result.child_started is False
    assert result.terminate_attempted is False
    assert result.ok is True
    assert supervisor.stop_requested is True
    assert supervisor.shutdown_started is True
    assert supervisor.shutdown_result is result


# ============================================================================
# 11. test_shutdown_is_idempotent
# ============================================================================


@pytest.mark.asyncio
async def test_shutdown_is_idempotent() -> None:
    supervisor = ReclaimSupervisor()
    fake_child = FakeChildProcess(supervisor.build_child_spec(), running=True)
    supervisor._child = fake_child  # type: ignore[assignment]
    fake_child.started = True

    await supervisor.shutdown()
    first_result = supervisor.shutdown_result
    assert fake_child.terminate_calls == 1

    await supervisor.shutdown()
    second_result = supervisor.shutdown_result
    assert fake_child.terminate_calls == 1
    assert second_result is first_result


# ============================================================================
# 12. test_shutdown_running_child_terminates
# ============================================================================


@pytest.mark.asyncio
async def test_shutdown_running_child_terminates() -> None:
    supervisor = ReclaimSupervisor()
    fake_child = FakeChildProcess(supervisor.build_child_spec(), running=True)
    supervisor._child = fake_child  # type: ignore[assignment]
    fake_child.started = True

    result = await supervisor.shutdown()
    assert result.terminate_attempted is True
    assert fake_child.terminated is True
    assert result.child_running_before_shutdown is True


# ============================================================================
# 13. test_shutdown_exited_child_does_not_terminate
# ============================================================================


@pytest.mark.asyncio
async def test_shutdown_exited_child_does_not_terminate() -> None:
    supervisor = ReclaimSupervisor()
    fake_child = FakeChildProcess(
        supervisor.build_child_spec(), running=False, returncode=0
    )
    supervisor._child = fake_child  # type: ignore[assignment]
    fake_child.started = True

    result = await supervisor.shutdown()
    assert result.terminate_attempted is False
    assert fake_child.terminated is False
    assert fake_child.terminate_calls == 0


# ============================================================================
# 14. test_shutdown_terminate_failure_records_error
# ============================================================================


@pytest.mark.asyncio
async def test_shutdown_terminate_failure_records_error() -> None:
    supervisor = ReclaimSupervisor()
    fake_child = FakeChildProcess(
        supervisor.build_child_spec(),
        running=True,
        terminate_error=RuntimeError("terminate failed"),
    )
    supervisor._child = fake_child  # type: ignore[assignment]
    fake_child.started = True

    result = await supervisor.shutdown()
    assert result.ok is False
    assert "RuntimeError: terminate failed" in (result.terminate_error or "")


# ============================================================================
# 15. test_shutdown_cancelled_error_propagates
# ============================================================================


@pytest.mark.asyncio
async def test_shutdown_cancelled_error_propagates() -> None:
    supervisor = ReclaimSupervisor()
    fake_child = FakeChildProcess(
        supervisor.build_child_spec(), running=True, terminate_cancelled=True
    )
    supervisor._child = fake_child  # type: ignore[assignment]
    fake_child.started = True

    with pytest.raises(asyncio.CancelledError):
        await supervisor.shutdown()


# ============================================================================
# 16. test_run_forever_starts_child_and_stops_on_request
# ============================================================================


@pytest.mark.asyncio
async def test_run_forever_starts_child_and_stops_on_request() -> None:
    config = ReclaimSupervisorConfig(poll_interval_seconds=0.01)
    supervisor = ReclaimSupervisor(config=config)
    fake_child = FakeChildProcess(supervisor.build_child_spec(), running=True)
    supervisor.create_child_process = lambda: fake_child  # type: ignore[method-assign]

    task = asyncio.create_task(supervisor.run_forever())
    await asyncio.sleep(0.02)
    assert supervisor.started_at_ms is not None
    assert fake_child.started is True
    assert supervisor.child is fake_child
    supervisor.request_stop()
    await asyncio.wait_for(task, timeout=1.0)
    assert fake_child.terminated is True
    assert supervisor.shutdown_result is not None
    assert supervisor.shutdown_result.terminate_attempted is True


# ============================================================================
# 17. test_run_forever_cancel_terminates_child_and_propagates_cancelled_error
# ============================================================================


@pytest.mark.asyncio
async def test_run_forever_cancel_terminates_child_and_propagates_cancelled_error() -> None:
    config = ReclaimSupervisorConfig(poll_interval_seconds=0.01)
    supervisor = ReclaimSupervisor(config=config)
    fake_child = FakeChildProcess(supervisor.build_child_spec(), running=True)
    supervisor.create_child_process = lambda: fake_child  # type: ignore[method-assign]

    task = asyncio.create_task(supervisor.run_forever())
    await asyncio.sleep(0.02)
    assert fake_child.started is True
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fake_child.terminated is True
    assert supervisor.shutdown_result is not None


# ============================================================================
# 18. test_run_forever_start_failure_propagates
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
    # _child was assigned before start, so shutdown was called and terminated.
    assert fake_child.terminated is True
    assert supervisor.shutdown_result is not None


# ============================================================================
# 19. test_run_forever_shutdown_terminate_failure_does_not_mask_cancel
# ============================================================================


@pytest.mark.asyncio
async def test_run_forever_shutdown_terminate_failure_does_not_mask_cancel() -> None:
    config = ReclaimSupervisorConfig(poll_interval_seconds=0.01)
    supervisor = ReclaimSupervisor(config=config)
    fake_child = FakeChildProcess(
        supervisor.build_child_spec(),
        running=True,
        terminate_error=RuntimeError("terminate oops"),
    )
    supervisor.create_child_process = lambda: fake_child  # type: ignore[method-assign]

    task = asyncio.create_task(supervisor.run_forever())
    await asyncio.sleep(0.02)
    assert fake_child.started is True
    task.cancel()
    # CancelledError must propagate, not RuntimeError.
    with pytest.raises(asyncio.CancelledError):
        await task
    assert supervisor.shutdown_result is not None
    assert "RuntimeError: terminate oops" in (supervisor.shutdown_result.terminate_error or "")


# ============================================================================
# 20. test_supervisor_source_no_heartbeat_or_restart
# ============================================================================


def test_supervisor_source_no_heartbeat_or_restart() -> None:
    source = _supervisor_source()

    # D05/D06 now wires ChildProcess — these must be present.
    assert "ChildProcess" in source, (
        "D06 reclaim_supervisor.py must import ChildProcess"
    )
    assert "ChildProcessSpec" in source, (
        "D06 reclaim_supervisor.py must import ChildProcessSpec"
    )
    assert "scripts/run_symbol_worker.py" in source, (
        "D06 reclaim_supervisor.py must reference scripts/run_symbol_worker.py"
    )

    # D06 must NOT wire heartbeat, multi-symbol, restart, or trading modules.
    forbidden = [
        "HeartbeatMonitor",
        "HeartbeatStatus",
        "read_status",
        "restart",
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
            f"D06 reclaim_supervisor.py must NOT contain {token!r}"
        )


# ============================================================================
# 21. test_supervisor_loop_logs_are_not_spammy
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
