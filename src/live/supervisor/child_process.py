from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from src.utils.log import get_logger

logger = get_logger(__name__)


# ============================================================================
# ChildProcessSpec
# ============================================================================


@dataclass(frozen=True)
class ChildProcessSpec:
    """Immutable specification for a child process managed by the supervisor."""

    name: str
    argv: tuple[str, ...]
    cwd: Path | None = None
    env: Mapping[str, str] | None = None
    terminate_timeout_seconds: float = 10.0
    kill_timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("child process name must not be empty")
        if not self.argv:
            raise ValueError("child process argv must not be empty")
        if self.terminate_timeout_seconds <= 0:
            raise ValueError("terminate_timeout_seconds must be > 0")
        if self.kill_timeout_seconds <= 0:
            raise ValueError("kill_timeout_seconds must be > 0")
        object.__setattr__(self, "argv", tuple(str(x) for x in self.argv))
        if self.cwd is not None:
            object.__setattr__(self, "cwd", Path(self.cwd))


# ============================================================================
# ChildProcessSnapshot
# ============================================================================


@dataclass(frozen=True)
class ChildProcessSnapshot:
    """Point-in-time snapshot of a child process state."""

    name: str
    pid: int | None
    returncode: int | None
    running: bool
    started: bool


# ============================================================================
# ChildProcess
# ============================================================================


class ChildProcess:
    """Generic async child process lifecycle abstraction.

    Wraps ``asyncio.create_subprocess_exec`` with start / wait / terminate / kill
    and a lightweight snapshot API.  This module is symbol-agnostic — it knows
    nothing about specific trading symbols, reclaim symbol lookup, or any
    trading domain.
    """

    def __init__(self, spec: ChildProcessSpec) -> None:
        self._spec = spec
        self._process: asyncio.subprocess.Process | None = None

    # ── properties ──────────────────────────────────────────────────────

    @property
    def spec(self) -> ChildProcessSpec:
        return self._spec

    @property
    def process(self) -> asyncio.subprocess.Process | None:
        return self._process

    @property
    def started(self) -> bool:
        return self._process is not None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    @property
    def returncode(self) -> int | None:
        return self._process.returncode if self._process is not None else None

    # ── snapshot ─────────────────────────────────────────────────────────

    def snapshot(self) -> ChildProcessSnapshot:
        return ChildProcessSnapshot(
            name=self._spec.name,
            pid=self.pid,
            returncode=self.returncode,
            running=self.running,
            started=self.started,
        )

    # ── start ────────────────────────────────────────────────────────────

    async def start(self) -> ChildProcessSnapshot:
        """Launch the child process via ``asyncio.create_subprocess_exec``.

        Merges ``spec.env`` on top of ``os.environ`` when provided.
        stdout / stderr are sent to ``DEVNULL``.
        """
        if self.running:
            raise RuntimeError(f"child process already running: {self._spec.name}")
        if self._process is not None and self._process.returncode is None:
            raise RuntimeError(f"child process already running: {self._spec.name}")

        env = None
        if self._spec.env is not None:
            env = dict(os.environ)
            env.update(dict(self._spec.env))

        self._process = await asyncio.create_subprocess_exec(
            *self._spec.argv,
            cwd=str(self._spec.cwd) if self._spec.cwd is not None else None,
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        logger.info(
            "CHILD_PROCESS_STARTED | name=%s pid=%s argv=%s",
            self._spec.name,
            self.pid,
            " ".join(self._spec.argv),
        )
        return self.snapshot()

    # ── wait ─────────────────────────────────────────────────────────────

    async def wait(self) -> int | None:
        """Wait for the child process to exit.  Returns None if never started."""
        if self._process is None:
            return None
        return await self._process.wait()

    # ── terminate ────────────────────────────────────────────────────────

    async def terminate(self) -> ChildProcessSnapshot:
        """Send SIGTERM and wait up to ``terminate_timeout_seconds``.

        Falls back to :meth:`kill` on timeout.
        """
        if self._process is None:
            return self.snapshot()
        if self._process.returncode is not None:
            return self.snapshot()

        self._process.terminate()
        try:
            await asyncio.wait_for(
                self._process.wait(),
                timeout=self._spec.terminate_timeout_seconds,
            )
            logger.info(
                "CHILD_PROCESS_TERMINATED | name=%s pid=%s returncode=%s",
                self._spec.name,
                self.pid,
                self.returncode,
            )
            return self.snapshot()
        except asyncio.TimeoutError:
            return await self.kill()

    # ── kill ─────────────────────────────────────────────────────────────

    async def kill(self) -> ChildProcessSnapshot:
        """Send SIGKILL and wait up to ``kill_timeout_seconds``.

        Raises ``asyncio.TimeoutError`` if the process does not exit in time.
        """
        if self._process is None:
            return self.snapshot()
        if self._process.returncode is not None:
            return self.snapshot()

        self._process.kill()
        try:
            await asyncio.wait_for(
                self._process.wait(),
                timeout=self._spec.kill_timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.error(
                "CHILD_PROCESS_KILL_TIMEOUT | name=%s pid=%s",
                self._spec.name,
                self.pid,
            )
            raise

        logger.warning(
            "CHILD_PROCESS_KILLED | name=%s pid=%s returncode=%s",
            self._spec.name,
            self.pid,
            self.returncode,
        )
        return self.snapshot()
