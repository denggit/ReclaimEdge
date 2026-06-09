#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D07 unit tests for ReclaimSupervisor — validates config, child exit
detection, heartbeat staleness detection, startup grace, health events,
and integration of the new loop with graceful shutdown.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import pytest

from src.live.supervisor.child_process import ChildProcessSpec, ChildProcessSnapshot
from src.live.supervisor.heartbeat_monitor import HeartbeatMonitor, HeartbeatMonitorConfig, HeartbeatStatus
from src.live.supervisor.reclaim_supervisor import (
    ReclaimSupervisor,
    ReclaimSupervisorConfig,
    SupervisorHealthEvent,
    SupervisorShutdownResult,
)
from src.live.supervisor.restart_policy import RestartPolicyConfig

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SUPERVISOR_SOURCE_PATH = (
    _PROJECT_ROOT / "src" / "live" / "supervisor" / "reclaim_supervisor.py"
)


def _supervisor_source() -> str:
    return _SUPERVISOR_SOURCE_PATH.read_text(encoding="utf-8")


# ============================================================================
# Helpers
# ============================================================================


def _write_heartbeat(path: Path, updated_at_ms: int, *, pid: int = 123, status: str = "running") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "updated_at_ms": updated_at_ms,
        "pid": pid,
        "status": status,
        "sequence": 1,
        "stale_after_seconds": 30.0,
    }))


def _write_invalid_heartbeat(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not valid json {")


def _fake_heartbeat_status(
    *,
    symbol: str = "ETH-USDT-SWAP",
    path: Path = Path("/fake/heartbeat.json"),
    status: str = "fresh",
    fresh: bool = True,
    missing: bool = False,
    stale: bool = False,
    invalid: bool = False,
    age_seconds: float = 1.0,
    error: str | None = None,
) -> HeartbeatStatus:
    return HeartbeatStatus(
        symbol=symbol,
        path=path,
        status=status,
        fresh=fresh,
        missing=missing,
        stale=stale,
        invalid=invalid,
        age_seconds=age_seconds,
        sequence=1,
        pid=999,
        worker_status="running",
        updated_at_ms=100_000,
        stale_after_seconds=30.0,
        error=error,
    )


# ============================================================================
# Fake / Spy implementations for testing
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

    async def start(self) -> ChildProcessSnapshot:
        self.started = True
        raise RuntimeError("boom")


class ExitingAfterStartFake(FakeChildProcess):
    """Fake child that starts normally but then reports exited on subsequent snapshots."""

    def __init__(self, spec: ChildProcessSpec, returncode: int = 1) -> None:
        super().__init__(spec, running=True)
        self._exit_returncode = returncode
        self._should_exit = False

    async def start(self) -> ChildProcessSnapshot:
        result = await super().start()
        self._should_exit = True
        return result

    def snapshot(self) -> ChildProcessSnapshot:
        if self._should_exit:
            return ChildProcessSnapshot(
                name=self.spec.name,
                pid=self.pid,
                returncode=self._exit_returncode,
                running=False,
                started=True,
            )
        return super().snapshot()


class CountingHeartbeatMonitor:
    """Spy heartbeat monitor that counts read_status calls."""

    def __init__(self, status: HeartbeatStatus | None = None) -> None:
        self._read_count = 0
        self._status = status or _fake_heartbeat_status()

    @property
    def read_count(self) -> int:
        return self._read_count

    def read_status(self, *, symbol: str, path: str | Path) -> HeartbeatStatus:
        self._read_count += 1
        return self._status


class FailingHeartbeatMonitor:
    """Always returns a bad heartbeat status after N reads (or immediately)."""

    def __init__(self, status: HeartbeatStatus) -> None:
        self._status = status
        self._read_count = 0

    @property
    def read_count(self) -> int:
        return self._read_count

    def read_status(self, *, symbol: str, path: str | Path) -> HeartbeatStatus:
        self._read_count += 1
        return self._status


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
    assert config.runtime_dir == Path("runtime")
    assert config.heartbeat_check_enabled is True
    assert config.heartbeat_check_interval_seconds == 5.0
    assert config.heartbeat_default_stale_after_seconds == 30.0
    assert config.heartbeat_startup_grace_seconds == 20.0
    assert config.stop_on_child_exit is True
    assert config.stop_on_bad_heartbeat is True
    assert config.restart_on_child_exit is True
    assert config.restart_on_bad_heartbeat is True
    assert config.bad_heartbeat_restart_terminate_timeout_seconds == 10.0
    assert config.restart_policy.enabled is True
    assert config.restart_policy.max_restarts == 3


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
# 4b. test_invalid_heartbeat_config_raises
# ============================================================================


def test_invalid_heartbeat_config_raises() -> None:
    with pytest.raises(ValueError, match="heartbeat_check_interval_seconds must be > 0"):
        ReclaimSupervisorConfig(heartbeat_check_interval_seconds=0)

    with pytest.raises(ValueError, match="heartbeat_default_stale_after_seconds must be > 0"):
        ReclaimSupervisorConfig(heartbeat_default_stale_after_seconds=0)

    with pytest.raises(ValueError, match="heartbeat_startup_grace_seconds must be > 0"):
        ReclaimSupervisorConfig(heartbeat_startup_grace_seconds=0)


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
# D07 new tests — runtime paths
# ============================================================================


def test_runtime_paths_and_heartbeat_path(tmp_path: Path) -> None:
    config = ReclaimSupervisorConfig(project_root=tmp_path, runtime_dir=Path("runtime"))
    supervisor = ReclaimSupervisor(config=config)
    expected = tmp_path / "runtime" / "heartbeats" / "ETH-USDT-SWAP.heartbeat.json"
    assert supervisor.heartbeat_path == expected


def test_runtime_paths_absolute_runtime_dir(tmp_path: Path) -> None:
    abs_dir = tmp_path / "custom_rt"
    config = ReclaimSupervisorConfig(project_root=tmp_path, runtime_dir=abs_dir)
    supervisor = ReclaimSupervisor(config=config)
    expected = abs_dir / "heartbeats" / "ETH-USDT-SWAP.heartbeat.json"
    assert supervisor.heartbeat_path == expected


# ============================================================================
# D07 new tests — health events
# ============================================================================


def test_health_events_initially_empty() -> None:
    supervisor = ReclaimSupervisor()
    assert supervisor.health_events == ()


# ============================================================================
# D07 new tests — child exit detection
# ============================================================================


@pytest.mark.asyncio
async def test_check_child_exit_once_running_no_event() -> None:
    supervisor = ReclaimSupervisor()
    fake_child = FakeChildProcess(supervisor.build_child_spec(), running=True)
    supervisor._child = fake_child  # type: ignore[assignment]
    fake_child.started = True

    result = await supervisor.check_child_exit_once()
    assert result is False
    assert not supervisor.stop_requested
    assert len(supervisor.health_events) == 0


@pytest.mark.asyncio
async def test_check_child_exit_once_exited_records_event_and_requests_stop() -> None:
    """With restart disabled, child exit records event and requests stop (D07 behaviour preserved)."""
    config = ReclaimSupervisorConfig(
        restart_policy=RestartPolicyConfig(enabled=False),
        stop_on_child_exit=True,
    )
    supervisor = ReclaimSupervisor(config=config)
    fake_child = FakeChildProcess(supervisor.build_child_spec(), running=False, returncode=1)
    supervisor._child = fake_child  # type: ignore[assignment]
    fake_child.started = True

    result = await supervisor.check_child_exit_once()
    assert result is True
    assert supervisor.stop_requested is True
    assert len(supervisor.health_events) >= 1
    assert supervisor.health_events[0].event_type == "CHILD_EXITED"


@pytest.mark.asyncio
async def test_check_child_exit_once_exited_no_stop_when_disabled() -> None:
    config = ReclaimSupervisorConfig(
        restart_policy=RestartPolicyConfig(enabled=False),
        stop_on_child_exit=False,
    )
    supervisor = ReclaimSupervisor(config=config)
    fake_child = FakeChildProcess(supervisor.build_child_spec(), running=False, returncode=1)
    supervisor._child = fake_child  # type: ignore[assignment]
    fake_child.started = True

    result = await supervisor.check_child_exit_once()
    assert result is True
    assert supervisor.stop_requested is False


@pytest.mark.asyncio
async def test_check_child_exit_once_no_child_returns_none() -> None:
    supervisor = ReclaimSupervisor()
    result = await supervisor.check_child_exit_once()
    assert result is False


# ============================================================================
# D07 new tests — heartbeat detection
# ============================================================================


@pytest.mark.asyncio
async def test_check_heartbeat_once_fresh_no_event(tmp_path: Path) -> None:
    config = ReclaimSupervisorConfig(project_root=tmp_path, runtime_dir=tmp_path / "runtime")
    supervisor = ReclaimSupervisor(config=config)
    hb_path = supervisor.heartbeat_path
    now_ms = int(time.time() * 1000)
    _write_heartbeat(hb_path, now_ms)

    status = await supervisor.check_heartbeat_once(now_monotonic=100.0)
    assert status is not None
    assert status.fresh is True
    assert len(supervisor.health_events) == 0
    assert supervisor.stop_requested is False


@pytest.mark.asyncio
async def test_check_heartbeat_disabled_returns_none() -> None:
    config = ReclaimSupervisorConfig(heartbeat_check_enabled=False)
    supervisor = ReclaimSupervisor(config=config)
    assert await supervisor.check_heartbeat_once() is None


@pytest.mark.asyncio
async def test_check_heartbeat_missing_inside_startup_grace_no_stop(tmp_path: Path) -> None:
    config = ReclaimSupervisorConfig(
        heartbeat_startup_grace_seconds=20.0,
        project_root=tmp_path,
        runtime_dir=tmp_path / "runtime",
    )
    supervisor = ReclaimSupervisor(config=config)
    supervisor._child_started_monotonic = 100.0

    # heartbeat path does not exist → missing
    status = await supervisor.check_heartbeat_once(now_monotonic=110.0)
    assert status is not None
    assert status.missing is True
    assert len(supervisor.health_events) == 0
    assert supervisor.stop_requested is False


@pytest.mark.asyncio
async def test_check_heartbeat_missing_after_grace_records_event_and_requests_stop(tmp_path: Path) -> None:
    """With restart disabled, bad heartbeat after grace records event and requests stop."""
    config = ReclaimSupervisorConfig(
        restart_policy=RestartPolicyConfig(enabled=False),
        restart_on_bad_heartbeat=False,
        heartbeat_startup_grace_seconds=20.0,
        project_root=tmp_path,
        runtime_dir=tmp_path / "runtime",
    )
    supervisor = ReclaimSupervisor(config=config)
    supervisor._child_started_monotonic = 100.0

    # now is 130 = 30s after start, grace is 20s → grace expired
    status = await supervisor.check_heartbeat_once(now_monotonic=130.0)
    assert status is not None
    assert status.missing is True
    assert len(supervisor.health_events) == 1
    assert supervisor.health_events[0].event_type == "HEARTBEAT_MISSING"
    assert supervisor.stop_requested is True


@pytest.mark.asyncio
async def test_check_heartbeat_stale_after_grace_records_event(tmp_path: Path) -> None:
    """With restart disabled, stale heartbeat after grace records event and requests stop."""
    config = ReclaimSupervisorConfig(
        restart_policy=RestartPolicyConfig(enabled=False),
        restart_on_bad_heartbeat=False,
        heartbeat_startup_grace_seconds=20.0,
        heartbeat_default_stale_after_seconds=30.0,
        project_root=tmp_path,
        runtime_dir=tmp_path / "runtime",
    )
    supervisor = ReclaimSupervisor(config=config)
    supervisor._child_started_monotonic = 100.0

    hb_path = supervisor.heartbeat_path
    # Write heartbeat with old timestamp (120s ago)
    old_ms = int(time.time() * 1000) - 120_000
    _write_heartbeat(hb_path, old_ms)

    status = await supervisor.check_heartbeat_once(now_monotonic=130.0)
    assert status is not None
    assert status.stale is True
    assert status.fresh is False
    assert len(supervisor.health_events) == 1
    assert supervisor.health_events[0].event_type == "HEARTBEAT_STALE"
    assert supervisor.stop_requested is True


@pytest.mark.asyncio
async def test_check_heartbeat_invalid_after_grace_records_event(tmp_path: Path) -> None:
    """With restart disabled, invalid heartbeat after grace records event and requests stop."""
    config = ReclaimSupervisorConfig(
        restart_policy=RestartPolicyConfig(enabled=False),
        restart_on_bad_heartbeat=False,
        heartbeat_startup_grace_seconds=20.0,
        project_root=tmp_path,
        runtime_dir=tmp_path / "runtime",
    )
    supervisor = ReclaimSupervisor(config=config)
    supervisor._child_started_monotonic = 100.0

    hb_path = supervisor.heartbeat_path
    _write_invalid_heartbeat(hb_path)

    status = await supervisor.check_heartbeat_once(now_monotonic=130.0)
    assert status is not None
    assert status.invalid is True
    assert len(supervisor.health_events) == 1
    assert supervisor.health_events[0].event_type == "HEARTBEAT_INVALID"
    assert supervisor.stop_requested is True


@pytest.mark.asyncio
async def test_check_heartbeat_bad_no_stop_when_disabled(tmp_path: Path) -> None:
    config = ReclaimSupervisorConfig(
        restart_policy=RestartPolicyConfig(enabled=False),
        restart_on_bad_heartbeat=False,
        stop_on_bad_heartbeat=False,
        heartbeat_startup_grace_seconds=20.0,
        project_root=tmp_path,
        runtime_dir=tmp_path / "runtime",
    )
    supervisor = ReclaimSupervisor(config=config)
    supervisor._child_started_monotonic = 100.0

    status = await supervisor.check_heartbeat_once(now_monotonic=130.0)
    assert status is not None
    assert status.missing is True
    assert len(supervisor.health_events) == 1
    assert supervisor.health_events[0].event_type == "HEARTBEAT_MISSING"
    assert supervisor.stop_requested is False


@pytest.mark.asyncio
async def test_maybe_check_heartbeat_respects_interval(tmp_path: Path) -> None:
    config = ReclaimSupervisorConfig(
        heartbeat_check_interval_seconds=5.0,
        project_root=tmp_path,
        runtime_dir=tmp_path / "runtime",
    )
    supervisor = ReclaimSupervisor(config=config)

    # Write a fresh heartbeat so read_status produces a fresh result.
    hb_path = supervisor.heartbeat_path
    now_ms = int(time.time() * 1000)
    _write_heartbeat(hb_path, now_ms)

    # First call at monotonic 100.0 — should read and cache.
    status1 = await supervisor.maybe_check_heartbeat(now_monotonic=100.0)
    assert status1 is not None
    assert status1.fresh is True
    assert supervisor._last_heartbeat_check_monotonic == 100.0
    assert supervisor._last_heartbeat_status is status1

    # Overwrite the heartbeat with stale data.
    old_ms = now_ms - 120_000
    _write_heartbeat(hb_path, old_ms)

    # Second call at monotonic 102.0 — within interval, should return cached fresh status.
    status2 = await supervisor.maybe_check_heartbeat(now_monotonic=102.0)
    assert status2 is status1  # same cached object
    assert status2.fresh is True
    assert supervisor._last_heartbeat_check_monotonic == 100.0  # unchanged

    # Third call at monotonic 106.0 — beyond interval, reads stale file.
    status3 = await supervisor.maybe_check_heartbeat(now_monotonic=106.0)
    assert status3 is not None
    assert status3.fresh is False
    assert supervisor._last_heartbeat_check_monotonic == 106.0


@pytest.mark.asyncio
async def test_maybe_check_heartbeat_disabled_returns_none() -> None:
    config = ReclaimSupervisorConfig(heartbeat_check_enabled=False)
    supervisor = ReclaimSupervisor(config=config)
    assert await supervisor.maybe_check_heartbeat(now_monotonic=100.0) is None


def test_last_heartbeat_status_property() -> None:
    supervisor = ReclaimSupervisor()
    assert supervisor.last_heartbeat_status is None
    supervisor._last_heartbeat_status = _fake_heartbeat_status(status="stale", fresh=False, stale=True)
    assert supervisor.last_heartbeat_status is not None
    assert supervisor.last_heartbeat_status.status == "stale"


def test_heartbeat_in_startup_grace_no_child_started() -> None:
    supervisor = ReclaimSupervisor()
    assert supervisor._heartbeat_in_startup_grace(now_monotonic=100.0) is False


# ============================================================================
# D07 integration tests — run_forever with child exit
# ============================================================================


@pytest.mark.asyncio
async def test_run_forever_child_exit_stops_loop_without_heartbeat_check() -> None:
    """With restart disabled, child exit triggers stop (D07 behaviour preserved)."""
    config = ReclaimSupervisorConfig(
        poll_interval_seconds=0.01,
        restart_policy=RestartPolicyConfig(enabled=False),
    )
    supervisor = ReclaimSupervisor(config=config)
    fake_child = ExitingAfterStartFake(supervisor.build_child_spec(), returncode=7)

    # Use a spy heartbeat monitor to verify it is NOT called after child exit.
    spy_monitor = CountingHeartbeatMonitor()
    supervisor._heartbeat_monitor = spy_monitor  # type: ignore[assignment]
    supervisor.create_child_process = lambda: fake_child  # type: ignore[method-assign]

    task = asyncio.create_task(supervisor.run_forever())
    await asyncio.wait_for(task, timeout=1.0)

    # Verify child exited event was recorded.
    assert len(supervisor.health_events) >= 1
    exit_events = [e for e in supervisor.health_events if e.event_type == "CHILD_EXITED"]
    assert len(exit_events) == 1
    assert exit_events[0].returncode == 7
    assert supervisor.stop_requested is True

    # Shutdown should have seen an exited child (no terminate needed).
    assert supervisor.shutdown_result is not None
    assert supervisor.shutdown_result.child_running_before_shutdown is False
    assert supervisor.shutdown_result.terminate_attempted is False


@pytest.mark.asyncio
async def test_run_forever_bad_heartbeat_stops_loop_and_shutdowns_child() -> None:
    """With restart disabled, bad heartbeat triggers stop (D07 behaviour preserved)."""
    config = ReclaimSupervisorConfig(
        poll_interval_seconds=0.01,
        heartbeat_startup_grace_seconds=0.01,  # very short grace for test
        heartbeat_check_interval_seconds=0.01,  # check every loop iteration
        restart_policy=RestartPolicyConfig(enabled=False),
    )
    supervisor = ReclaimSupervisor(config=config)
    fake_child = FakeChildProcess(supervisor.build_child_spec(), running=True)

    # Heartbeat monitor always returns stale after first read.
    stale_status = _fake_heartbeat_status(status="stale", fresh=False, stale=True, age_seconds=99.0)
    supervisor._heartbeat_monitor = FailingHeartbeatMonitor(stale_status)  # type: ignore[assignment]
    supervisor.create_child_process = lambda: fake_child  # type: ignore[method-assign]

    task = asyncio.create_task(supervisor.run_forever())
    await asyncio.wait_for(task, timeout=2.0)

    # After grace, heartbeat bad should trigger stop.
    assert supervisor.stop_requested is True
    assert fake_child.terminated is True

    # Health event should contain heartbeat stale.
    hb_events = [e for e in supervisor.health_events if e.event_type == "HEARTBEAT_STALE"]
    assert len(hb_events) >= 1
    assert hb_events[0].heartbeat_status == "stale"
    assert hb_events[0].heartbeat_age_seconds == 99.0

    # Child should have been terminated via shutdown.
    assert supervisor.shutdown_result is not None
    assert supervisor.shutdown_result.terminate_attempted is True


@pytest.mark.asyncio
async def test_run_forever_fresh_heartbeat_keeps_running_until_request_stop() -> None:
    config = ReclaimSupervisorConfig(poll_interval_seconds=0.01)
    supervisor = ReclaimSupervisor(config=config)
    fake_child = FakeChildProcess(supervisor.build_child_spec(), running=True)

    # Heartbeat monitor always returns fresh.
    fresh_status = _fake_heartbeat_status(status="fresh", fresh=True)
    supervisor._heartbeat_monitor = CountingHeartbeatMonitor(fresh_status)  # type: ignore[assignment]
    supervisor.create_child_process = lambda: fake_child  # type: ignore[method-assign]

    task = asyncio.create_task(supervisor.run_forever())
    await asyncio.sleep(0.05)
    assert fake_child.started is True
    assert not supervisor.stop_requested
    # No health events for fresh heartbeat.
    assert len(supervisor.health_events) == 0

    supervisor.request_stop()
    await asyncio.wait_for(task, timeout=1.0)
    assert fake_child.terminated is True
    assert supervisor.shutdown_result is not None
    assert supervisor.shutdown_result.terminate_attempted is True


# ============================================================================
# D07b — restart policy tests
# ============================================================================


@pytest.mark.asyncio
async def test_child_exit_restarts_child_when_policy_allows() -> None:
    """Child exit with restart policy allowed → new child started, events recorded."""
    config = ReclaimSupervisorConfig(
        restart_policy=RestartPolicyConfig(max_restarts=3, cooldown_seconds=0, window_seconds=600.0),
        restart_on_child_exit=True,
    )
    supervisor = ReclaimSupervisor(config=config)

    # Create a fake child that simulates first child exited.
    fake_exited = FakeChildProcess(supervisor.build_child_spec(), running=False, returncode=1)
    fake_exited.started = True
    supervisor._child = fake_exited  # type: ignore[assignment]

    # Mock create_child_process to return a fresh FakeChildProcess.
    new_child = FakeChildProcess(supervisor.build_child_spec(), running=True)
    supervisor.create_child_process = lambda: new_child  # type: ignore[method-assign]

    result = await supervisor.check_child_exit_once()
    assert result is True
    assert not supervisor.stop_requested
    assert supervisor.restart_policy.restart_count_in_window == 1
    assert new_child.started is True

    events = supervisor.health_events
    event_types = [e.event_type for e in events]
    assert "CHILD_EXITED" in event_types
    assert "CHILD_RESTART_REQUESTED" in event_types
    assert "CHILD_RESTARTED" in event_types


@pytest.mark.asyncio
async def test_child_exit_max_restarts_exceeded_requests_stop() -> None:
    """When max restarts exceeded, child exit suppresses restart and requests stop."""
    config = ReclaimSupervisorConfig(
        restart_policy=RestartPolicyConfig(max_restarts=1, cooldown_seconds=0, window_seconds=600.0),
        restart_on_child_exit=True,
        stop_on_child_exit=True,
    )
    supervisor = ReclaimSupervisor(config=config)
    # Use current monotonic so the entry is NOT pruned by the evaluate inside
    # check_child_exit_once → _restart_child_after_exit_once.
    supervisor._restart_policy.record_restart(now_monotonic=time.monotonic())

    fake_exited = FakeChildProcess(supervisor.build_child_spec(), running=False, returncode=1)
    fake_exited.started = True
    supervisor._child = fake_exited  # type: ignore[assignment]

    # Safety: mock create_child_process so a real subprocess is never started.
    supervisor.create_child_process = lambda: FakeChildProcess(supervisor.build_child_spec(), running=True)  # type: ignore[method-assign]

    result = await supervisor.check_child_exit_once()
    assert result is True
    assert supervisor.stop_requested is True

    event_types = [e.event_type for e in supervisor.health_events]
    assert "CHILD_EXITED" in event_types
    assert "CHILD_RESTART_SUPPRESSED" in event_types


@pytest.mark.asyncio
async def test_child_exit_restart_disabled_falls_back_to_stop() -> None:
    """When restart_on_child_exit=False and stop_on_child_exit=True, exit stops."""
    config = ReclaimSupervisorConfig(
        restart_policy=RestartPolicyConfig(enabled=False),
        restart_on_child_exit=False,
        stop_on_child_exit=True,
    )
    supervisor = ReclaimSupervisor(config=config)

    fake_exited = FakeChildProcess(supervisor.build_child_spec(), running=False, returncode=1)
    fake_exited.started = True
    supervisor._child = fake_exited  # type: ignore[assignment]

    result = await supervisor.check_child_exit_once()
    assert result is True
    assert supervisor.stop_requested is True


@pytest.mark.asyncio
async def test_bad_heartbeat_restarts_child_after_grace() -> None:
    """Bad heartbeat with restart allowed → old child terminated, new child started."""
    config = ReclaimSupervisorConfig(
        restart_policy=RestartPolicyConfig(max_restarts=3, cooldown_seconds=0, window_seconds=600.0),
        restart_on_bad_heartbeat=True,
        heartbeat_startup_grace_seconds=0.01,  # minimal grace for test
    )
    supervisor = ReclaimSupervisor(config=config)

    base = time.monotonic()
    old_child = FakeChildProcess(supervisor.build_child_spec(), running=True)
    old_child.started = True
    supervisor._child = old_child  # type: ignore[assignment]
    supervisor._child_started_monotonic = base - 30.0  # well past grace

    stale_status = _fake_heartbeat_status(status="stale", fresh=False, stale=True, age_seconds=99.0)
    supervisor._heartbeat_monitor = FailingHeartbeatMonitor(stale_status)  # type: ignore[assignment]

    new_child = FakeChildProcess(supervisor.build_child_spec(), running=True)
    supervisor.create_child_process = lambda: new_child  # type: ignore[method-assign]

    status = await supervisor.check_heartbeat_once(now_monotonic=base)
    assert status is not None
    assert supervisor.restart_policy.restart_count_in_window == 1
    assert old_child.terminated is True
    assert new_child.started is True
    assert not supervisor.stop_requested

    event_types = [e.event_type for e in supervisor.health_events]
    assert "HEARTBEAT_STALE" in event_types
    assert "CHILD_TERMINATE_FOR_RESTART_REQUESTED" in event_types
    assert "CHILD_TERMINATED_FOR_RESTART" in event_types
    assert "CHILD_RESTARTED" in event_types


@pytest.mark.asyncio
async def test_bad_heartbeat_restart_max_exceeded_requests_stop() -> None:
    """When max restarts exceeded, bad heartbeat suppresses restart and requests stop."""
    config = ReclaimSupervisorConfig(
        restart_policy=RestartPolicyConfig(max_restarts=1, cooldown_seconds=0, window_seconds=600.0),
        restart_on_bad_heartbeat=True,
        stop_on_bad_heartbeat=True,
        heartbeat_startup_grace_seconds=0.01,  # minimal grace for test
    )
    supervisor = ReclaimSupervisor(config=config)
    base = time.monotonic()
    # Use consistent timestamps so entry is within window during evaluation.
    supervisor._restart_policy.record_restart(now_monotonic=base - 1.0)

    old_child = FakeChildProcess(supervisor.build_child_spec(), running=True)
    old_child.started = True
    supervisor._child = old_child  # type: ignore[assignment]
    supervisor._child_started_monotonic = base - 30.0  # well past grace

    stale_status = _fake_heartbeat_status(status="stale", fresh=False, stale=True, age_seconds=99.0)
    supervisor._heartbeat_monitor = FailingHeartbeatMonitor(stale_status)  # type: ignore[assignment]
    # Safety: mock create_child_process so a real subprocess is never started.
    supervisor.create_child_process = lambda: FakeChildProcess(supervisor.build_child_spec(), running=True)  # type: ignore[method-assign]

    status = await supervisor.check_heartbeat_once(now_monotonic=base)
    assert status is not None
    assert supervisor.stop_requested is True

    event_types = [e.event_type for e in supervisor.health_events]
    assert "CHILD_RESTART_SUPPRESSED" in event_types


@pytest.mark.asyncio
async def test_run_forever_child_exit_restart_skips_heartbeat_same_loop() -> None:
    """After restart in a loop iteration, heartbeat is not checked in the same iteration."""
    config = ReclaimSupervisorConfig(
        poll_interval_seconds=0.01,
        heartbeat_check_interval_seconds=0.01,
        restart_policy=RestartPolicyConfig(max_restarts=3, cooldown_seconds=0, window_seconds=600.0),
        restart_on_child_exit=True,
    )
    supervisor = ReclaimSupervisor(config=config)

    fake_child = ExitingAfterStartFake(supervisor.build_child_spec(), returncode=7)
    new_child = FakeChildProcess(supervisor.build_child_spec(), running=True)
    child_factory = [fake_child, new_child]
    call_count = [0]

    def factory() -> FakeChildProcess:
        result = child_factory[call_count[0]]
        call_count[0] += 1
        return result

    supervisor.create_child_process = factory  # type: ignore[method-assign]

    # Spy heartbeat monitor — should NOT be called during restart iteration.
    spy_monitor = CountingHeartbeatMonitor()
    supervisor._heartbeat_monitor = spy_monitor  # type: ignore[assignment]

    task = asyncio.create_task(supervisor.run_forever())
    # Let the loop run for a few iterations — it should restart and keep running.
    await asyncio.sleep(0.08)
    supervisor.request_stop()
    await asyncio.wait_for(task, timeout=1.0)

    # The restarted child should have been started.
    assert new_child.started is True
    # Should have a CHILD_RESTARTED event.
    event_types = [e.event_type for e in supervisor.health_events]
    assert "CHILD_RESTARTED" in event_types


@pytest.mark.asyncio
async def test_restart_policy_property_accessible() -> None:
    """RestartPolicy is accessible and properly initialized."""
    supervisor = ReclaimSupervisor()
    rp = supervisor.restart_policy
    assert rp.config.enabled is True
    assert rp.config.max_restarts == 3
    assert rp.restart_count_in_window == 0


def test_config_invalid_bad_heartbeat_restart_timeout_raises() -> None:
    with pytest.raises(ValueError, match="bad_heartbeat_restart_terminate_timeout_seconds must be > 0"):
        ReclaimSupervisorConfig(bad_heartbeat_restart_terminate_timeout_seconds=0)


# ============================================================================
# D07 source guard — check imports and forbidden tokens
# ============================================================================


def test_source_allows_heartbeat_monitor_and_child_process_and_restart() -> None:
    source = _supervisor_source()

    # D07 must wire HeartbeatMonitor.  D07b adds restart_policy.
    allowed = [
        "HeartbeatMonitor",
        "HeartbeatStatus",
        "HeartbeatMonitorConfig",
        "RuntimePaths",
        "ChildProcess",
        "ChildProcessSpec",
        "ChildProcessSnapshot",
        "check_child_exit_once",
        "check_heartbeat_once",
        "maybe_check_heartbeat",
        "_heartbeat_in_startup_grace",
        "SupervisorHealthEvent",
        "scripts/run_symbol_worker.py",
        "RestartPolicy",
        "RestartPolicyConfig",
        "restart_policy",
        "restart_on_child_exit",
        "restart_on_bad_heartbeat",
        "_restart_child_after_exit_once",
        "_restart_child_after_bad_heartbeat_once",
    ]
    for token in allowed:
        assert token in source, (
            f"D07b reclaim_supervisor.py must contain {token!r}"
        )


def test_source_no_btc_or_email_or_trading() -> None:
    source = _supervisor_source()

    # D07b allows "restart" tokens — intentionally removed from this list.
    forbidden = [
        "RECLAIM_SYMBOLS",
        "BTC-USDT-SWAP",
        "EmailSender",
        "send_email",
        "SymbolWorkerApp",
        "Trader",
        "account_position_sync_worker",
        "strategy_tick_worker",
        "execution_worker",
        "BollCvd",
        "okx",
        "requests",
        "httpx",
        "websocket",
        "multiprocessing",
        "subprocess.Popen",
        "Popen(",
    ]
    for token in forbidden:
        assert token not in source, (
            f"D07b reclaim_supervisor.py must NOT contain {token!r}"
        )

    # BTC symbol should not appear outside of child_name default.
    assert "BTC" not in source, (
        "D07b reclaim_supervisor.py must NOT contain BTC"
    )


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
    # logger.error and logger.debug are allowed for exit/heartbeat/restart events.
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
            assert "logger.info" not in stripped, (
                f"reclaim_supervisor.py while loop must not log info inside the loop — found: {stripped!r}"
            )
