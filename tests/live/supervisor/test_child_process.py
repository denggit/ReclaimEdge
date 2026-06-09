#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D03 unit tests for ChildProcess lifecycle abstraction.

Uses a FakeProcess and monkey-patched ``asyncio.create_subprocess_exec``
to avoid launching real subprocesses or depending on live trading environment.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from unittest import mock

import pytest

from src.live.supervisor.child_process import (
    ChildProcess,
    ChildProcessSnapshot,
    ChildProcessSpec,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CHILD_PROCESS_SOURCE_PATH = (
    _PROJECT_ROOT / "src" / "live" / "supervisor" / "child_process.py"
)


def _child_process_source() -> str:
    return _CHILD_PROCESS_SOURCE_PATH.read_text(encoding="utf-8")


# ============================================================================
# FakeProcess
# ============================================================================


class FakeProcess:
    """Lightweight fake for ``asyncio.subprocess.Process`` used in unit tests.

    Allows precise control over pid, returncode, wait delay, and whether
    terminate / kill have been called.
    """

    def __init__(
        self,
        *,
        pid: int = 1234,
        returncode: int | None = None,
        wait_delay: float = 0.0,
    ):
        self.pid = pid
        self._returncode = returncode
        self.terminated = False
        self.killed = False
        self._wait_delay = wait_delay
        self._wait_call_count = 0

    @property
    def returncode(self) -> int | None:
        return self._returncode

    @returncode.setter
    def returncode(self, value: int | None) -> None:
        self._returncode = value

    async def wait(self) -> int:
        self._wait_call_count += 1
        if self._wait_delay:
            await asyncio.sleep(self._wait_delay)
        if self._returncode is None:
            self._returncode = 0
        return self._returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def eth_spec(tmp_path: Path) -> ChildProcessSpec:
    return ChildProcessSpec(
        name="eth",
        argv=("python", "scripts/run_symbol_worker.py"),
        cwd=tmp_path,
        env={"A": "B"},
    )


# ============================================================================
# 1. test_spec_validates_name
# ============================================================================


def test_spec_validates_name() -> None:
    with pytest.raises(ValueError, match="name"):
        ChildProcessSpec(name="", argv=("python",))


def test_spec_validates_name_whitespace() -> None:
    with pytest.raises(ValueError, match="name"):
        ChildProcessSpec(name="   ", argv=("python",))


# ============================================================================
# 2. test_spec_validates_argv
# ============================================================================


def test_spec_validates_argv() -> None:
    with pytest.raises(ValueError, match="argv"):
        ChildProcessSpec(name="eth", argv=())


# ============================================================================
# 3. test_spec_coerces_argv_to_tuple_and_cwd_to_path
# ============================================================================


def test_spec_coerces_argv_to_tuple_and_cwd_to_path() -> None:
    spec = ChildProcessSpec(
        name="eth",
        argv=["python", "scripts/run_symbol_worker.py"],
        cwd=".",
    )
    assert isinstance(spec.argv, tuple)
    assert isinstance(spec.cwd, Path)


# ============================================================================
# 4. test_spec_validates_timeouts
# ============================================================================


def test_spec_validates_terminate_timeout() -> None:
    with pytest.raises(ValueError, match="terminate_timeout_seconds must be > 0"):
        ChildProcessSpec(name="eth", argv=("python",), terminate_timeout_seconds=0)


def test_spec_validates_kill_timeout() -> None:
    with pytest.raises(ValueError, match="kill_timeout_seconds must be > 0"):
        ChildProcessSpec(name="eth", argv=("python",), kill_timeout_seconds=0)


# ============================================================================
# 5. test_snapshot_before_start
# ============================================================================


def test_snapshot_before_start(eth_spec: ChildProcessSpec) -> None:
    child = ChildProcess(eth_spec)
    snapshot = child.snapshot()
    assert snapshot.started is False
    assert snapshot.running is False
    assert snapshot.pid is None
    assert snapshot.returncode is None
    assert snapshot.name == "eth"


# ============================================================================
# 6. test_start_invokes_create_subprocess_exec
# ============================================================================


@pytest.mark.asyncio
async def test_start_invokes_create_subprocess_exec(tmp_path: Path) -> None:
    """Verify that start() calls asyncio.create_subprocess_exec with the
    expected argv, cwd, env, stdout=DEVNULL, stderr=DEVNULL."""

    captured_kwargs: dict = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured_kwargs["args"] = args
        captured_kwargs["kwargs"] = kwargs
        return FakeProcess(pid=999)

    spec = ChildProcessSpec(
        name="eth",
        argv=("python", "scripts/run_symbol_worker.py"),
        cwd=tmp_path,
        env={"A": "B"},
    )
    child = ChildProcess(spec)

    with mock.patch(
        "src.live.supervisor.child_process.asyncio.create_subprocess_exec",
        side_effect=fake_create_subprocess_exec,
    ):
        snapshot = await child.start()

    # Verify captured args
    assert captured_kwargs["args"] == ("python", "scripts/run_symbol_worker.py")
    kwargs = captured_kwargs["kwargs"]
    assert kwargs["cwd"] == str(tmp_path)
    assert kwargs["stdout"] is asyncio.subprocess.DEVNULL
    assert kwargs["stderr"] is asyncio.subprocess.DEVNULL

    # Verify env contains os.environ + A=B
    assert kwargs["env"] is not None
    assert "A" in kwargs["env"]
    assert kwargs["env"]["A"] == "B"
    # os.environ keys should be present
    for key in os.environ:
        assert key in kwargs["env"]

    # Verify snapshot
    assert snapshot.started is True
    assert snapshot.running is True
    assert snapshot.pid == 999
    assert snapshot.name == "eth"


# ============================================================================
# 7. test_start_when_running_raises
# ============================================================================


@pytest.mark.asyncio
async def test_start_when_running_raises() -> None:
    spec = ChildProcessSpec(name="eth", argv=("python",))
    child = ChildProcess(spec)

    async def fake_create_subprocess_exec(*args, **kwargs):
        return FakeProcess(pid=999, returncode=None)

    with mock.patch(
        "src.live.supervisor.child_process.asyncio.create_subprocess_exec",
        side_effect=fake_create_subprocess_exec,
    ):
        await child.start()

    with pytest.raises(RuntimeError, match="already running"):
        await child.start()


# ============================================================================
# 8. test_wait_before_start_returns_none
# ============================================================================


@pytest.mark.asyncio
async def test_wait_before_start_returns_none(eth_spec: ChildProcessSpec) -> None:
    child = ChildProcess(eth_spec)
    assert await child.wait() is None


# ============================================================================
# 9. test_wait_returns_returncode
# ============================================================================


@pytest.mark.asyncio
async def test_wait_returns_returncode() -> None:
    spec = ChildProcessSpec(name="eth", argv=("python",))
    child = ChildProcess(spec)
    fake = FakeProcess(pid=999, returncode=None)  # wait sets returncode to 0

    async def fake_create_subprocess_exec(*args, **kwargs):
        return fake

    with mock.patch(
        "src.live.supervisor.child_process.asyncio.create_subprocess_exec",
        side_effect=fake_create_subprocess_exec,
    ):
        await child.start()

    assert await child.wait() == 0
    assert child.returncode == 0


# ============================================================================
# 10. test_terminate_before_start_noop
# ============================================================================


@pytest.mark.asyncio
async def test_terminate_before_start_noop(eth_spec: ChildProcessSpec) -> None:
    child = ChildProcess(eth_spec)
    snapshot = await child.terminate()
    assert snapshot.started is False
    assert snapshot.running is False


# ============================================================================
# 11. test_terminate_running_process
# ============================================================================


@pytest.mark.asyncio
async def test_terminate_running_process() -> None:
    spec = ChildProcessSpec(name="eth", argv=("python",))
    child = ChildProcess(spec)
    fake = FakeProcess(pid=999, returncode=None)

    async def fake_create_subprocess_exec(*args, **kwargs):
        return fake

    with mock.patch(
        "src.live.supervisor.child_process.asyncio.create_subprocess_exec",
        side_effect=fake_create_subprocess_exec,
    ):
        await child.start()

    snapshot = await child.terminate()
    assert fake.terminated is True
    assert fake.killed is False
    assert snapshot.running is False
    assert snapshot.returncode == 0  # FakeProcess.wait sets returncode=0


# ============================================================================
# 12. test_terminate_timeout_falls_back_to_kill
# ============================================================================


@pytest.mark.asyncio
async def test_terminate_timeout_falls_back_to_kill() -> None:
    """When wait after terminate does not complete within the timeout,
    terminate falls back to kill."""

    spec = ChildProcessSpec(
        name="eth",
        argv=("python",),
        terminate_timeout_seconds=0.01,
    )
    child = ChildProcess(spec)

    class StubbornProcess(FakeProcess):
        """Process whose wait blocks until killed (simulates SIGTERM ignored)."""

        _killed = False

        async def wait(self) -> int:
            # If not yet killed, sleep long enough to trigger timeout.
            if not self._killed:
                await asyncio.sleep(10.0)
            # After kill, return -9.
            self.returncode = -9
            return -9

        def terminate(self) -> None:
            self.terminated = True
            # Does NOT set killed — simulate SIGTERM being ignored.

        def kill(self) -> None:
            self.killed = True
            self._killed = True

    stubborn = StubbornProcess(pid=999, returncode=None)

    async def fake_create_subprocess_exec(*args, **kwargs):
        return stubborn

    with mock.patch(
        "src.live.supervisor.child_process.asyncio.create_subprocess_exec",
        side_effect=fake_create_subprocess_exec,
    ):
        await child.start()

    snapshot = await child.terminate()
    assert stubborn.terminated is True
    assert stubborn.killed is True
    assert snapshot.returncode == -9
    assert snapshot.running is False


# ============================================================================
# 13. test_kill_running_process
# ============================================================================


@pytest.mark.asyncio
async def test_kill_running_process() -> None:
    spec = ChildProcessSpec(name="eth", argv=("python",))
    child = ChildProcess(spec)
    fake = FakeProcess(pid=999, returncode=None)

    async def fake_create_subprocess_exec(*args, **kwargs):
        return fake

    with mock.patch(
        "src.live.supervisor.child_process.asyncio.create_subprocess_exec",
        side_effect=fake_create_subprocess_exec,
    ):
        await child.start()

    snapshot = await child.kill()
    assert fake.killed is True
    assert snapshot.running is False
    # FakeProcess.wait sets returncode=0 for kill too (since returncode was None)


# ============================================================================
# 14. test_kill_timeout_raises
# ============================================================================


@pytest.mark.asyncio
async def test_kill_timeout_raises() -> None:
    spec = ChildProcessSpec(
        name="eth",
        argv=("python",),
        kill_timeout_seconds=0.01,
    )
    child = ChildProcess(spec)

    class UnkillableProcess(FakeProcess):
        async def wait(self) -> int:
            await asyncio.sleep(10.0)  # Never returns in time
            return -9

        def kill(self) -> None:
            self.killed = True

    unkillable = UnkillableProcess(pid=999, returncode=None)

    async def fake_create_subprocess_exec(*args, **kwargs):
        return unkillable

    with mock.patch(
        "src.live.supervisor.child_process.asyncio.create_subprocess_exec",
        side_effect=fake_create_subprocess_exec,
    ):
        await child.start()

    with pytest.raises(asyncio.TimeoutError):
        await child.kill()


# ============================================================================
# 15. test_source_has_no_trading_or_symbol_specific_side_effects
# ============================================================================


def test_source_has_no_trading_or_symbol_specific_side_effects() -> None:
    source = _child_process_source()

    forbidden = [
        "SymbolWorkerApp",
        "Trader",
        "RuntimePaths",
        "HeartbeatWriter",
        "HeartbeatMonitor",
        "RECLAIM_SYMBOLS",
        "BTC",
        "ETH-USDT-SWAP",
        "run_symbol_worker.py",
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
            f"child_process.py must NOT contain {token!r}"
        )


# ============================================================================
# 16. test_source_uses_asyncio_subprocess_exec_only_in_child_process_module
# ============================================================================


def test_source_uses_asyncio_subprocess_exec_only_in_child_process_module() -> None:
    source = _child_process_source()

    assert "asyncio.create_subprocess_exec" in source, (
        "child_process.py must use asyncio.create_subprocess_exec"
    )
    assert "multiprocessing" not in source, (
        "child_process.py must NOT use multiprocessing"
    )
    assert "subprocess.Popen" not in source, (
        "child_process.py must NOT use subprocess.Popen"
    )
    assert "Popen(" not in source, (
        "child_process.py must NOT contain Popen("
    )
    # "Process(" only counts when it's not our own ChildProcess — a bare
    # "Process(" without "Child" prefix would be suspicious.
    lines = source.splitlines()
    for line in lines:
        if "Process(" in line and "ChildProcess" not in line and "asyncio.subprocess" not in line:
            assert False, f"child_process.py suspicious Process( call: {line.strip()!r}"
