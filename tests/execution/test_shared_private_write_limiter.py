from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
import pytest

from src.execution.private_write_limiter import (
    LocalPrivateWriteRateLimiter,
    PrivateWriteRateLimiter,
    SharedPrivateWriteLimiter,
    SharedPrivateWriteLimiterConfig,
    SharedPrivateWriteLimiterTimeout,
)


# ===========================================================================
# 1. local limiter sequential delay
# ===========================================================================


@pytest.mark.asyncio
async def test_local_limiter_sequential_delay() -> None:
    """Two consecutive acquires observe the min interval."""
    limiter = LocalPrivateWriteRateLimiter(enabled=True, min_interval_seconds=0.05)

    t0 = time.monotonic()
    await limiter.acquire()
    await limiter.acquire()
    elapsed = time.monotonic() - t0

    # The second acquire should have waited at least ~min_interval.
    assert elapsed >= 0.03, f"Expected >= 0.03s delay, got {elapsed:.4f}s"


@pytest.mark.asyncio
async def test_local_limiter_first_acquire_no_delay() -> None:
    """The very first acquire returns immediately (no prior write)."""
    limiter = LocalPrivateWriteRateLimiter(enabled=True, min_interval_seconds=0.5)

    t0 = time.monotonic()
    await limiter.acquire()
    elapsed = time.monotonic() - t0

    assert elapsed < 0.1, f"First acquire should be instant, got {elapsed:.4f}s"


# ===========================================================================
# 2. disabled limiter returns immediately
# ===========================================================================


@pytest.mark.asyncio
async def test_disabled_limiter_returns_immediately(monkeypatch) -> None:
    """OKX_PRIVATE_WRITE_RATE_LIMIT_ENABLED=false → acquire is a no-op."""
    monkeypatch.setenv("OKX_PRIVATE_WRITE_RATE_LIMIT_ENABLED", "false")
    monkeypatch.delenv("OKX_PRIVATE_WRITE_SHARED_ENABLED", raising=False)
    monkeypatch.delenv("OKX_PRIVATE_WRITE_MIN_INTERVAL_SECONDS", raising=False)
    monkeypatch.setenv("PRIVATE_WRITE_MIN_INTERVAL_SECONDS", "0.05")

    limiter = PrivateWriteRateLimiter()
    assert limiter.enabled is False

    t0 = time.monotonic()
    await limiter.acquire()
    await limiter.acquire()
    await limiter.acquire()
    elapsed = time.monotonic() - t0

    # Disabled → no waiting at all.
    assert elapsed < 0.05, f"Disabled limiter should not wait, got {elapsed:.4f}s"


# ===========================================================================
# 3. env fallback reads PRIVATE_WRITE_MIN_INTERVAL_SECONDS
# ===========================================================================


def test_env_fallback_reads_legacy_min_interval(monkeypatch) -> None:
    """Only PRIVATE_WRITE_MIN_INTERVAL_SECONDS set → min_interval picks it up."""
    # Remove the new-style env if present; set only legacy.
    monkeypatch.delenv("OKX_PRIVATE_WRITE_MIN_INTERVAL_SECONDS", raising=False)
    monkeypatch.setenv("PRIVATE_WRITE_MIN_INTERVAL_SECONDS", "0.7")
    monkeypatch.delenv("OKX_PRIVATE_WRITE_SHARED_ENABLED", raising=False)

    limiter = PrivateWriteRateLimiter()
    assert limiter._local._min_interval == 0.7


def test_env_priority_new_over_legacy(monkeypatch) -> None:
    """OKX_PRIVATE_WRITE_MIN_INTERVAL_SECONDS takes priority."""
    monkeypatch.setenv("OKX_PRIVATE_WRITE_MIN_INTERVAL_SECONDS", "0.9")
    monkeypatch.setenv("PRIVATE_WRITE_MIN_INTERVAL_SECONDS", "0.3")
    monkeypatch.delenv("OKX_PRIVATE_WRITE_SHARED_ENABLED", raising=False)

    limiter = PrivateWriteRateLimiter()
    assert limiter._local._min_interval == 0.9


# ===========================================================================
# 4. shared limiter creates files
# ===========================================================================


@pytest.mark.asyncio
async def test_shared_limiter_creates_files(tmp_path: Path) -> None:
    """After acquire, lock file and state JSON exist with correct content."""
    shared_dir = tmp_path / "limiter"
    config = SharedPrivateWriteLimiterConfig(
        enabled=True,
        min_interval_seconds=0.01,
        shared_dir=shared_dir,
        lock_timeout_seconds=2.0,
    )
    limiter = SharedPrivateWriteLimiter(config)

    await limiter.acquire()

    lock_file = shared_dir / "private_write.lock"
    state_file = shared_dir / "private_write_state.json"

    assert lock_file.exists(), f"Lock file missing: {lock_file}"
    assert state_file.exists(), f"State file missing: {state_file}"

    state = json.loads(state_file.read_text())
    assert state["version"] == 1
    assert state["last_write_wall_time"] > 0


# ===========================================================================
# 5. shared limiter enforces interval across two instances
# ===========================================================================


@pytest.mark.asyncio
async def test_shared_limiter_cross_instance_serial(tmp_path: Path) -> None:
    """Two SharedPrivateWriteLimiter instances sharing the same dir are serialised."""
    shared_dir = tmp_path / "limiter"
    config = SharedPrivateWriteLimiterConfig(
        enabled=True,
        min_interval_seconds=0.05,
        shared_dir=shared_dir,
        lock_timeout_seconds=2.0,
    )
    limiter1 = SharedPrivateWriteLimiter(config)
    limiter2 = SharedPrivateWriteLimiter(config)

    t0 = time.monotonic()
    await limiter1.acquire()
    await limiter2.acquire()
    elapsed = time.monotonic() - t0

    # limiter2 must have waited ~0.05s after limiter1 wrote its timestamp.
    assert elapsed >= 0.03, f"Expected >= 0.03s cross-instance delay, got {elapsed:.4f}s"


# ===========================================================================
# 6. corrupted state recovers
# ===========================================================================


@pytest.mark.asyncio
async def test_corrupted_state_recovers(tmp_path: Path) -> None:
    """A malformed state JSON does not prevent acquire; it is overwritten."""
    shared_dir = tmp_path / "limiter"
    shared_dir.mkdir(parents=True, exist_ok=True)
    state_file = shared_dir / "private_write_state.json"
    state_file.write_text("{bad")

    config = SharedPrivateWriteLimiterConfig(
        enabled=True,
        min_interval_seconds=0.01,
        shared_dir=shared_dir,
        lock_timeout_seconds=2.0,
    )
    limiter = SharedPrivateWriteLimiter(config)

    # Must not raise.
    await limiter.acquire()

    # State file is now valid JSON.
    raw = state_file.read_text()
    state = json.loads(raw)
    assert state["version"] == 1
    assert state["last_write_wall_time"] > 0


# ===========================================================================
# 7. shared timeout → fallback local
# ===========================================================================


@pytest.mark.asyncio
async def test_shared_timeout_fallback_local(monkeypatch, tmp_path: Path) -> None:
    """When shared limiter times out and fallback_local=true, acquire succeeds."""
    shared_dir = tmp_path / "limiter"

    monkeypatch.setenv("OKX_PRIVATE_WRITE_RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("OKX_PRIVATE_WRITE_MIN_INTERVAL_SECONDS", "0.05")
    monkeypatch.setenv("OKX_PRIVATE_WRITE_SHARED_ENABLED", "true")
    monkeypatch.setenv("OKX_PRIVATE_WRITE_SHARED_DIR", str(shared_dir))
    monkeypatch.setenv("OKX_PRIVATE_WRITE_SHARED_LOCK_TIMEOUT_SECONDS", "0.1")
    monkeypatch.setenv("OKX_PRIVATE_WRITE_SHARED_FALLBACK_LOCAL", "true")
    monkeypatch.delenv("PRIVATE_WRITE_MIN_INTERVAL_SECONDS", raising=False)

    limiter = PrivateWriteRateLimiter()
    assert limiter.shared_enabled is True
    assert limiter._fallback_local is True

    # Make the shared limiter always time out.
    async def fake_shared_acquire() -> None:
        raise SharedPrivateWriteLimiterTimeout("injected timeout")

    limiter._shared.acquire = fake_shared_acquire  # type: ignore[method-assign]

    # Must NOT raise — fallback to local.
    await limiter.acquire()


# ===========================================================================
# 8. shared timeout without fallback raises
# ===========================================================================


@pytest.mark.asyncio
async def test_shared_timeout_no_fallback_raises(monkeypatch, tmp_path: Path) -> None:
    """When shared limiter times out and fallback_local=false, acquire raises."""
    shared_dir = tmp_path / "limiter"

    monkeypatch.setenv("OKX_PRIVATE_WRITE_RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("OKX_PRIVATE_WRITE_MIN_INTERVAL_SECONDS", "0.05")
    monkeypatch.setenv("OKX_PRIVATE_WRITE_SHARED_ENABLED", "true")
    monkeypatch.setenv("OKX_PRIVATE_WRITE_SHARED_DIR", str(shared_dir))
    monkeypatch.setenv("OKX_PRIVATE_WRITE_SHARED_LOCK_TIMEOUT_SECONDS", "0.1")
    monkeypatch.setenv("OKX_PRIVATE_WRITE_SHARED_FALLBACK_LOCAL", "false")
    monkeypatch.delenv("PRIVATE_WRITE_MIN_INTERVAL_SECONDS", raising=False)

    limiter = PrivateWriteRateLimiter()
    assert limiter._fallback_local is False

    # Make the shared limiter always time out.
    async def fake_shared_acquire() -> None:
        raise SharedPrivateWriteLimiterTimeout("injected timeout")

    limiter._shared.acquire = fake_shared_acquire  # type: ignore[method-assign]

    with pytest.raises(SharedPrivateWriteLimiterTimeout, match="injected timeout"):
        await limiter.acquire()


# ===========================================================================
# 9. Trader.request only limits POST
# ===========================================================================


@pytest.mark.asyncio
async def test_trader_request_only_limits_post() -> None:
    """GET does NOT call limiter.acquire; POST calls it exactly once."""
    from src.execution.trader import Trader

    # Bypass Trader.__init__ (needs live OKX env) and inject fakes manually.
    trader = Trader.__new__(Trader)
    trader.base_url = "https://www.okx.test"
    trader.api_key = "key"
    trader.secret_key = "secret"
    trader.passphrase = "pass"
    trader._timeout_seconds = 5.0

    # Tracking state.
    acquire_calls: list[str] = []
    client_get_calls: list[str] = []
    client_post_calls: list[str] = []

    # Fake OKX client with per-method routing so we can assert behaviour.
    class FakeOkxClient:
        async def request(self, method: str, endpoint: str, payload=None):
            if method.upper() == "GET":
                client_get_calls.append(endpoint)
            else:
                client_post_calls.append(endpoint)
            return {"code": "0", "data": []}

        def headers(self, method, endpoint, body):
            return {}

        async def start(self):
            pass

        async def close(self):
            pass

    # Fake limiter.
    class FakeLimiter:
        async def acquire(self):
            acquire_calls.append("acquire")

    trader._client = FakeOkxClient()
    trader._private_write_limiter = FakeLimiter()

    # GET → limiter NOT called.
    await trader.request("GET", "/api/v5/account/balance?ccy=USDT")
    assert len(acquire_calls) == 0, f"GET should not call acquire, got {acquire_calls}"
    assert len(client_get_calls) == 1
    assert len(client_post_calls) == 0

    acquire_calls.clear()
    client_get_calls.clear()

    # POST → limiter called once, then client.request.
    await trader.request("POST", "/api/v5/trade/order", {"instId": "ETH-USDT-SWAP"})
    assert len(acquire_calls) == 1, f"POST should call acquire once, got {acquire_calls}"
    assert len(client_post_calls) == 1
    assert len(client_get_calls) == 0


# ===========================================================================
# 10. shared_enabled=false uses local limiter
# ===========================================================================


@pytest.mark.asyncio
async def test_shared_disabled_uses_local(monkeypatch) -> None:
    """OKX_PRIVATE_WRITE_SHARED_ENABLED=false → shared is None, local is used."""
    monkeypatch.setenv("OKX_PRIVATE_WRITE_RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("OKX_PRIVATE_WRITE_MIN_INTERVAL_SECONDS", "0.05")
    monkeypatch.setenv("OKX_PRIVATE_WRITE_SHARED_ENABLED", "false")
    monkeypatch.delenv("PRIVATE_WRITE_MIN_INTERVAL_SECONDS", raising=False)

    limiter = PrivateWriteRateLimiter()
    assert limiter.shared_enabled is False
    assert limiter._shared is None
    assert limiter._local.enabled is True

    # acquire should work via local limiter.
    t0 = time.monotonic()
    await limiter.acquire()
    await limiter.acquire()
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.03, f"Local limiter should enforce interval, got {elapsed:.4f}s"


# ===========================================================================
# 11. shared limiter unexpected error falls back to local
# ===========================================================================


@pytest.mark.asyncio
async def test_shared_unexpected_error_fallback_local(monkeypatch, tmp_path: Path) -> None:
    """Non-timeout exceptions in shared limiter also fall back to local."""
    shared_dir = tmp_path / "limiter"

    monkeypatch.setenv("OKX_PRIVATE_WRITE_RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("OKX_PRIVATE_WRITE_MIN_INTERVAL_SECONDS", "0.05")
    monkeypatch.setenv("OKX_PRIVATE_WRITE_SHARED_ENABLED", "true")
    monkeypatch.setenv("OKX_PRIVATE_WRITE_SHARED_DIR", str(shared_dir))
    monkeypatch.setenv("OKX_PRIVATE_WRITE_SHARED_FALLBACK_LOCAL", "true")
    monkeypatch.delenv("PRIVATE_WRITE_MIN_INTERVAL_SECONDS", raising=False)

    limiter = PrivateWriteRateLimiter()

    async def fake_shared_acquire() -> None:
        raise OSError("disk full")

    limiter._shared.acquire = fake_shared_acquire  # type: ignore[method-assign]

    # Must NOT raise — fallback to local.
    await limiter.acquire()


# ===========================================================================
# 12. rate limit disabled but shared disabled → no-op
# ===========================================================================


@pytest.mark.asyncio
async def test_rate_limit_disabled_noop(monkeypatch) -> None:
    """When the master switch is off, acquire is always a no-op."""
    monkeypatch.setenv("OKX_PRIVATE_WRITE_RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("OKX_PRIVATE_WRITE_SHARED_ENABLED", "true")
    monkeypatch.setenv("OKX_PRIVATE_WRITE_MIN_INTERVAL_SECONDS", "0.6")
    monkeypatch.delenv("PRIVATE_WRITE_MIN_INTERVAL_SECONDS", raising=False)

    limiter = PrivateWriteRateLimiter()
    assert limiter.enabled is False
    assert limiter.shared_enabled is False  # shared not constructed when disabled

    t0 = time.monotonic()
    for _ in range(10):
        await limiter.acquire()
    elapsed = time.monotonic() - t0
    assert elapsed < 0.05
