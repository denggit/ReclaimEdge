from __future__ import annotations

import asyncio
import fcntl
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.log import get_logger

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name, "").strip().lower()
    if not val:
        return default
    return val in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name, "").strip()
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        _logger.warning("Invalid float for %s=%r, using default %.3f", name, val, default)
        return default


# ---------------------------------------------------------------------------
# LocalPrivateWriteRateLimiter
# ---------------------------------------------------------------------------


class LocalPrivateWriteRateLimiter:
    """Single-process private-write rate limiter using asyncio.Lock + monotonic time."""

    def __init__(self, *, enabled: bool = True, min_interval_seconds: float = 0.6) -> None:
        self._enabled = enabled
        self._min_interval = min_interval_seconds
        self._lock = asyncio.Lock()
        self._last_write_time: float = 0.0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def min_interval_seconds(self) -> float:
        return self._min_interval

    async def acquire(self) -> None:
        if not self._enabled:
            return
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_write_time
            if elapsed < self._min_interval:
                wait = self._min_interval - elapsed
                _logger.debug(
                    "LOCAL_PRIVATE_WRITE_LIMITER | waiting=%.3fs min_interval=%.3fs",
                    wait,
                    self._min_interval,
                )
                await asyncio.sleep(wait)
                self._last_write_time = time.monotonic()
            else:
                self._last_write_time = now


# ---------------------------------------------------------------------------
# SharedPrivateWriteLimiter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SharedPrivateWriteLimiterConfig:
    enabled: bool = True
    min_interval_seconds: float = 0.6
    shared_dir: Path = Path("runtime/private_write_limiter")
    lock_timeout_seconds: float = 2.0
    fallback_local: bool = True


class SharedPrivateWriteLimiterTimeout(TimeoutError):
    """Raised when the shared file lock cannot be acquired within the timeout."""


class SharedPrivateWriteLimiter:
    """Cross-process private-write rate limiter using fcntl.flock + wall-clock time.

    Multiple worker processes (e.g. ETH + BTC) share one lock file and one
    state file under *shared_dir*.  The lock serialises all private writes
    across processes; the state JSON carries the last-write wall-clock
    timestamp so that the interval is enforced globally.
    """

    _LOCK_FILENAME = "private_write.lock"
    _STATE_FILENAME = "private_write_state.json"
    _STATE_VERSION = 1

    def __init__(self, config: SharedPrivateWriteLimiterConfig) -> None:
        self._config = config
        self._lock_path = config.shared_dir / self._LOCK_FILENAME
        self._state_path = config.shared_dir / self._STATE_FILENAME

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    async def acquire(self) -> None:
        """Acquire the shared write slot (async-safe, runs blocking I/O in a thread)."""
        await asyncio.to_thread(self._acquire_blocking)

    # ------------------------------------------------------------------
    # blocking implementation (runs inside asyncio.to_thread)
    # ------------------------------------------------------------------

    def _acquire_blocking(self) -> None:
        config = self._config
        config.shared_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. acquire exclusive file lock with timeout ---
        lock_fd = self._open_lock_file()
        try:
            self._flock_with_timeout(lock_fd, config.lock_timeout_seconds)
        except SharedPrivateWriteLimiterTimeout:
            lock_fd.close()
            raise

        try:
            # --- 2. read last-write wall-clock time ---
            last_write = self._read_state()

            # --- 3. sleep remaining interval while holding the lock ---
            now = time.time()
            elapsed = now - last_write
            if elapsed < config.min_interval_seconds:
                wait = config.min_interval_seconds - elapsed
                _logger.debug(
                    "SHARED_PRIVATE_WRITE_LIMITER | waiting=%.3fs min_interval=%.3fs last_write=%.3f",
                    wait,
                    config.min_interval_seconds,
                    last_write,
                )
                time.sleep(wait)
                now = time.time()

            # --- 4. write new timestamp ---
            self._write_state(now)
        finally:
            # --- 5. release lock ---
            self._release_lock(lock_fd)
            lock_fd.close()

    # ------------------------------------------------------------------
    # lock helpers
    # ------------------------------------------------------------------

    def _open_lock_file(self) -> Any:
        """Open (or create) the lock file and return the file descriptor."""
        return open(self._lock_path, "a")

    @staticmethod
    def _flock_with_timeout(lock_fd: Any, timeout: float) -> None:
        """Acquire an exclusive fcntl.flock, retrying every 20 ms until *timeout*."""
        deadline = time.monotonic()
        # We must ensure the loop runs at least once so that when timeout=0
        # we still attempt a non-blocking lock.
        first = True
        while True:
            try:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return  # acquired
            except (BlockingIOError, OSError):
                if first:
                    first = False
                elapsed = time.monotonic() - deadline
                if elapsed >= timeout:
                    raise SharedPrivateWriteLimiterTimeout(
                        f"Shared private write lock not acquired within {timeout:.3f}s"
                    )
                time.sleep(0.02)

    @staticmethod
    def _release_lock(lock_fd: Any) -> None:
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
        except Exception:
            _logger.warning("Failed to release shared private write lock", exc_info=True)

    # ------------------------------------------------------------------
    # state file helpers
    # ------------------------------------------------------------------

    def _read_state(self) -> float:
        """Return the last_write_wall_time from the state file, or 0.0 on any error."""
        try:
            raw = self._state_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            ts = float(data.get("last_write_wall_time", 0))
            return ts
        except FileNotFoundError:
            return 0.0
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            _logger.warning(
                "Corrupted shared private write state file, resetting to 0.  error=%s",
                exc,
            )
            return 0.0

    def _write_state(self, timestamp: float) -> None:
        """Write the state JSON with the given *timestamp* and fsync."""
        data: dict[str, object] = {
            "version": self._STATE_VERSION,
            "last_write_wall_time": timestamp,
        }
        tmp_path = self._state_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        # fsync via os
        with open(tmp_path, "rb") as f:
            os.fsync(f.fileno())
        tmp_path.rename(self._state_path)


# ---------------------------------------------------------------------------
# PrivateWriteRateLimiter  (façade)
# ---------------------------------------------------------------------------


class PrivateWriteRateLimiter:
    """Façade that selects local or shared private-write limiter based on env.

    Env vars (all optional):
        OKX_PRIVATE_WRITE_RATE_LIMIT_ENABLED  (default true)
        OKX_PRIVATE_WRITE_MIN_INTERVAL_SECONDS (default 0.6)
        PRIVATE_WRITE_MIN_INTERVAL_SECONDS     (legacy fallback)
        OKX_PRIVATE_WRITE_SHARED_ENABLED       (default true)
        OKX_PRIVATE_WRITE_SHARED_DIR           (default runtime/private_write_limiter)
        OKX_PRIVATE_WRITE_SHARED_LOCK_TIMEOUT_SECONDS (default 2.0)
        OKX_PRIVATE_WRITE_SHARED_FALLBACK_LOCAL (default true)
    """

    def __init__(self) -> None:
        # --- rate-limit master switch ---
        self._enabled = _env_bool("OKX_PRIVATE_WRITE_RATE_LIMIT_ENABLED", True)

        # --- min interval: new env takes priority, legacy env as fallback ---
        min_interval = _env_float("OKX_PRIVATE_WRITE_MIN_INTERVAL_SECONDS", -1.0)
        if min_interval < 0:
            min_interval = _env_float("PRIVATE_WRITE_MIN_INTERVAL_SECONDS", 0.6)

        # --- shared vs local ---
        shared_enabled = _env_bool("OKX_PRIVATE_WRITE_SHARED_ENABLED", True)
        self._shared_enabled = shared_enabled

        shared_dir = Path(
            os.getenv("OKX_PRIVATE_WRITE_SHARED_DIR", "runtime/private_write_limiter").strip()
        )
        lock_timeout = _env_float("OKX_PRIVATE_WRITE_SHARED_LOCK_TIMEOUT_SECONDS", 2.0)
        self._fallback_local = _env_bool("OKX_PRIVATE_WRITE_SHARED_FALLBACK_LOCAL", True)

        # --- build sub-limiters ---
        self._local = LocalPrivateWriteRateLimiter(
            enabled=self._enabled,
            min_interval_seconds=min_interval,
        )

        if self._enabled and shared_enabled:
            shared_config = SharedPrivateWriteLimiterConfig(
                enabled=True,
                min_interval_seconds=min_interval,
                shared_dir=shared_dir,
                lock_timeout_seconds=lock_timeout,
                fallback_local=self._fallback_local,
            )
            self._shared: SharedPrivateWriteLimiter | None = SharedPrivateWriteLimiter(shared_config)
        else:
            self._shared = None

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def shared_enabled(self) -> bool:
        return self._shared_enabled and self._shared is not None

    # ------------------------------------------------------------------
    # acquire
    # ------------------------------------------------------------------

    async def acquire(self) -> None:
        """Wait until the rate limiter allows the next private write."""
        if not self._enabled:
            return

        if self._shared is None:
            await self._local.acquire()
            return

        try:
            await self._shared.acquire()
        except SharedPrivateWriteLimiterTimeout:
            _logger.warning(
                "SharedPrivateWriteLimiter timed out after %.3fs. "
                "fallback_local=%s",
                self._shared._config.lock_timeout_seconds,
                self._fallback_local,
            )
            if self._fallback_local:
                await self._local.acquire()
                return
            raise
        except Exception:
            _logger.exception("SharedPrivateWriteLimiter unexpected error")
            if self._fallback_local:
                await self._local.acquire()
                return
            raise
