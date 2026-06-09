from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from src.live.heartbeat_writer import (
    HeartbeatWriteResult,
    HeartbeatWriter,
    HeartbeatWriterConfig,
)
from src.live.runtime_paths import RuntimePaths


# ============================================================================
# HeartbeatWriterConfig
# ============================================================================


class TestHeartbeatWriterConfigDefaults:
    def test_default_config_disabled(self) -> None:
        config = HeartbeatWriterConfig()
        assert config.enabled is False
        assert config.interval_seconds == 10.0
        assert config.stale_after_seconds == 30.0
        assert config.failure_log_interval_seconds == 60.0

    def test_invalid_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="interval_seconds must be > 0"):
            HeartbeatWriterConfig(interval_seconds=0)

    def test_invalid_stale_after_raises(self) -> None:
        with pytest.raises(ValueError, match="stale_after_seconds must be > 0"):
            HeartbeatWriterConfig(stale_after_seconds=0)

    def test_invalid_failure_log_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="failure_log_interval_seconds must be > 0"):
            HeartbeatWriterConfig(failure_log_interval_seconds=0)


# ============================================================================
# HeartbeatWriter — disabled
# ============================================================================


class TestHeartbeatWriterDisabled:
    def test_disabled_write_does_not_create_file(self, tmp_path: Path) -> None:
        runtime_paths = RuntimePaths(tmp_path / "runtime", "ETH-USDT-SWAP")
        writer = HeartbeatWriter(runtime_paths=runtime_paths)
        result = writer.write_once()
        assert result.wrote is False
        assert result.reason == "disabled"
        assert not runtime_paths.heartbeat_file.exists()
        assert not runtime_paths.heartbeats_dir.exists()


# ============================================================================
# HeartbeatWriter — enabled
# ============================================================================


class TestHeartbeatWriterEnabled:
    def test_enabled_write_creates_heartbeat_file(self, tmp_path: Path) -> None:
        config = HeartbeatWriterConfig(enabled=True)
        runtime_paths = RuntimePaths(tmp_path / "runtime", "ETH-USDT-SWAP")
        writer = HeartbeatWriter(
            runtime_paths=runtime_paths,
            config=config,
            clock_ms=lambda: 123456789,
            pid_provider=lambda: 999,
        )
        result = writer.write_once(status="running")
        assert result.wrote is True
        assert result.path == runtime_paths.heartbeat_file
        assert result.sequence == 1

        assert runtime_paths.heartbeat_file.exists()
        payload = json.loads(runtime_paths.heartbeat_file.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 1
        assert payload["inst_id"] == "ETH-USDT-SWAP"
        assert payload["symbol_slug"] == "ETH-USDT-SWAP"
        assert payload["pid"] == 999
        assert payload["status"] == "running"
        assert payload["sequence"] == 1
        assert payload["started_at_ms"] == 123456789
        assert payload["updated_at_ms"] == 123456789
        assert payload["stale_after_seconds"] == 30.0

    def test_write_once_increments_sequence_and_replaces_existing_file(
        self, tmp_path: Path
    ) -> None:
        tick = 0

        def clock() -> int:
            nonlocal tick
            tick += 1
            return tick

        config = HeartbeatWriterConfig(enabled=True)
        runtime_paths = RuntimePaths(tmp_path / "runtime", "ETH-USDT-SWAP")
        writer = HeartbeatWriter(
            runtime_paths=runtime_paths,
            config=config,
            clock_ms=clock,
            pid_provider=lambda: 42,
        )

        r1 = writer.write_once(status="running")
        assert r1.wrote is True
        assert r1.sequence == 1

        r2 = writer.write_once(status="stopping")
        assert r2.wrote is True
        assert r2.sequence == 2

        payload = json.loads(runtime_paths.heartbeat_file.read_text(encoding="utf-8"))
        assert payload["sequence"] == 2
        assert payload["status"] == "stopping"

        # No residual tmp files
        tmp_files = list(runtime_paths.heartbeats_dir.glob("*.tmp"))
        assert len(tmp_files) == 0


# ============================================================================
# build_payload
# ============================================================================


class TestBuildPayload:
    def test_build_payload_does_not_write_file(self, tmp_path: Path) -> None:
        runtime_paths = RuntimePaths(tmp_path / "runtime", "ETH-USDT-SWAP")
        writer = HeartbeatWriter(
            runtime_paths=runtime_paths,
            clock_ms=lambda: 500,
            pid_provider=lambda: 7,
        )

        payload = writer.build_payload(status="running")
        assert not runtime_paths.heartbeat_file.exists()

        # Payload contains the NEXT sequence but writer._sequence is unchanged
        assert payload["sequence"] == 1
        assert payload["pid"] == 7
        assert payload["status"] == "running"
        assert payload["started_at_ms"] == 500
        assert payload["updated_at_ms"] == 500


# ============================================================================
# run_until_cancelled — disabled
# ============================================================================


class TestRunUntilCancelledDisabled:
    @pytest.mark.asyncio
    async def test_run_until_cancelled_disabled_returns_without_writing(
        self, tmp_path: Path
    ) -> None:
        runtime_paths = RuntimePaths(tmp_path / "runtime", "ETH-USDT-SWAP")
        writer = HeartbeatWriter(runtime_paths=runtime_paths)
        await writer.run_until_cancelled()
        assert not runtime_paths.heartbeat_file.exists()


# ============================================================================
# HeartbeatWriter — write_status_once (D06b)
# ============================================================================


class TestHeartbeatWriterWriteStatusOnce:
    def test_write_status_once_stopping(self, tmp_path: Path) -> None:
        """write_status_once('stopping') must write a heartbeat with status=stopping."""
        config = HeartbeatWriterConfig(enabled=True)
        runtime_paths = RuntimePaths(tmp_path / "runtime", "ETH-USDT-SWAP")
        writer = HeartbeatWriter(
            runtime_paths=runtime_paths,
            config=config,
            clock_ms=lambda: 123456789,
            pid_provider=lambda: 42,
        )
        writer.write_status_once("stopping")
        payload = json.loads(runtime_paths.heartbeat_file.read_text(encoding="utf-8"))
        assert payload["status"] == "stopping"
        assert payload["sequence"] == 1

    def test_write_status_once_stopped(self, tmp_path: Path) -> None:
        """write_status_once('stopped') must write a heartbeat with status=stopped."""
        config = HeartbeatWriterConfig(enabled=True)
        runtime_paths = RuntimePaths(tmp_path / "runtime", "ETH-USDT-SWAP")
        writer = HeartbeatWriter(
            runtime_paths=runtime_paths,
            config=config,
            clock_ms=lambda: 123456789,
            pid_provider=lambda: 42,
        )
        writer.write_status_once("stopped")
        payload = json.loads(runtime_paths.heartbeat_file.read_text(encoding="utf-8"))
        assert payload["status"] == "stopped"

    def test_write_status_once_disabled_noop(self, tmp_path: Path) -> None:
        """write_status_once must be a no-op when the writer is disabled."""
        runtime_paths = RuntimePaths(tmp_path / "runtime", "ETH-USDT-SWAP")
        writer = HeartbeatWriter(runtime_paths=runtime_paths)
        writer.write_status_once("stopping")
        assert not runtime_paths.heartbeat_file.exists()

    def test_write_status_once_failure_does_not_raise(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """write_status_once must not raise on failure — it degrades silently."""
        config = HeartbeatWriterConfig(enabled=True)
        runtime_paths = RuntimePaths(tmp_path / "runtime", "ETH-USDT-SWAP")
        writer = HeartbeatWriter(
            runtime_paths=runtime_paths,
            config=config,
        )

        def _fail_write(*args: object, **kwargs: object) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(writer, "write_once", _fail_write)

        # Must not raise
        writer.write_status_once("stopping")


# ============================================================================
# HeartbeatWriter — failure degrade (C06b)
# ============================================================================


class TestHeartbeatWriterFailureDegrade:
    def test_write_once_failure_still_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """write_once must NOT catch exceptions — it lets them propagate."""
        config = HeartbeatWriterConfig(enabled=True)
        runtime_paths = RuntimePaths(tmp_path / "runtime", "ETH-USDT-SWAP")
        writer = HeartbeatWriter(
            runtime_paths=runtime_paths,
            config=config,
        )

        def _fail_replace(src: str, dst: str) -> None:
            raise OSError("replace failed")

        monkeypatch.setattr(os, "replace", _fail_replace)

        with pytest.raises(OSError, match="replace failed"):
            writer.write_once()

        # write_once does NOT record failure — it just raises.
        assert writer.consecutive_failures == 0
        assert writer.last_error is None

    def test_write_once_success_records_success(
        self, tmp_path: Path
    ) -> None:
        """After a successful write_once, consecutive_failures resets to 0."""
        config = HeartbeatWriterConfig(enabled=True)
        runtime_paths = RuntimePaths(tmp_path / "runtime", "ETH-USDT-SWAP")
        writer = HeartbeatWriter(
            runtime_paths=runtime_paths,
            config=config,
            clock_ms=lambda: 123456789,
            pid_provider=lambda: 42,
        )
        # Simulate a prior failure via _record_failure to verify reset
        writer._record_failure(OSError("prior"))
        assert writer.consecutive_failures == 1
        assert writer.last_error is not None

        result = writer.write_once(status="running")
        assert result.wrote is True
        assert writer.consecutive_failures == 0
        assert writer.last_error is None

    @pytest.mark.asyncio
    async def test_run_until_cancelled_degrades_write_failures(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_until_cancelled catches write failures and keeps looping."""
        runtime_paths = RuntimePaths(tmp_path / "runtime", "ETH-USDT-SWAP")
        config = HeartbeatWriterConfig(
            enabled=True,
            interval_seconds=0.01,
            failure_log_interval_seconds=999.0,
        )
        writer = HeartbeatWriter(
            runtime_paths=runtime_paths,
            config=config,
        )

        call_count = 0

        async def _fake_run(coro: object) -> None:
            pass

        def _fail_first_two(status: str = "running") -> object:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise OSError("disk full")
            # On third call, return a fake success result
            return HeartbeatWriteResult(
                wrote=True,
                path=writer.path,
                sequence=call_count,
            )

        monkeypatch.setattr(writer, "write_once", _fail_first_two)

        task = asyncio.ensure_future(writer.run_until_cancelled())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert call_count >= 2
        assert writer.consecutive_failures >= 2
        assert "OSError: disk full" in (writer.last_error or "")

    @pytest.mark.asyncio
    async def test_run_until_cancelled_resets_failure_count_after_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After a successful write, consecutive_failures resets to 0."""
        runtime_paths = RuntimePaths(tmp_path / "runtime", "ETH-USDT-SWAP")
        config = HeartbeatWriterConfig(
            enabled=True,
            interval_seconds=0.01,
            failure_log_interval_seconds=999.0,
        )
        writer = HeartbeatWriter(
            runtime_paths=runtime_paths,
            config=config,
            clock_ms=lambda: 123456789,
            pid_provider=lambda: 42,
        )

        call_count = 0

        def _fail_then_succeed(status: str = "running") -> object:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("disk full")
            # Let the real write_once run for success
            return HeartbeatWriter.write_once(writer, status=status)

        monkeypatch.setattr(writer, "write_once", _fail_then_succeed)

        task = asyncio.ensure_future(writer.run_until_cancelled())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert call_count >= 2
        assert writer.consecutive_failures == 0
        assert writer.last_error is None
        assert runtime_paths.heartbeat_file.exists()

    @pytest.mark.asyncio
    async def test_run_until_cancelled_does_not_swallow_cancelled_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CancelledError must propagate, NOT be caught by except Exception."""
        runtime_paths = RuntimePaths(tmp_path / "runtime", "ETH-USDT-SWAP")
        config = HeartbeatWriterConfig(enabled=True, interval_seconds=0.01)
        writer = HeartbeatWriter(
            runtime_paths=runtime_paths,
            config=config,
        )

        # Monkeypatch write_once to be a no-op so it doesn't fail
        def _noop(status: str = "running") -> HeartbeatWriteResult:
            return HeartbeatWriteResult(
                wrote=False, path=writer.path, reason="noop"
            )

        monkeypatch.setattr(writer, "write_once", _noop)

        task = asyncio.ensure_future(writer.run_until_cancelled())
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_run_until_cancelled_logs_failures_throttled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Failure warnings are throttled: only one log per throttle window."""
        runtime_paths = RuntimePaths(tmp_path / "runtime", "ETH-USDT-SWAP")
        config = HeartbeatWriterConfig(
            enabled=True,
            interval_seconds=0.01,
            failure_log_interval_seconds=999.0,
        )
        writer = HeartbeatWriter(
            runtime_paths=runtime_paths,
            config=config,
        )

        def _always_fail(status: str = "running") -> None:
            raise OSError("disk full")

        monkeypatch.setattr(writer, "write_once", _always_fail)

        task = asyncio.ensure_future(writer.run_until_cancelled())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        occurrences = [
            record.message
            for record in caplog.records
            if "HEARTBEAT_WRITE_FAILED" in record.message
        ]
        assert len(occurrences) == 1, (
            f"Expected exactly 1 HEARTBEAT_WRITE_FAILED log, got {len(occurrences)}"
        )


# ============================================================================
# HeartbeatWriter — schema
# ============================================================================


class TestHeartbeatWriterSchema:
    def test_heartbeat_payload_schema_unchanged(self, tmp_path: Path) -> None:
        """The heartbeat JSON schema must not include internal config fields."""
        config = HeartbeatWriterConfig(
            enabled=True,
            failure_log_interval_seconds=123.0,
        )
        runtime_paths = RuntimePaths(tmp_path / "runtime", "ETH-USDT-SWAP")
        writer = HeartbeatWriter(
            runtime_paths=runtime_paths,
            config=config,
            clock_ms=lambda: 123456789,
            pid_provider=lambda: 42,
        )
        writer.write_once(status="running")

        payload = json.loads(runtime_paths.heartbeat_file.read_text(encoding="utf-8"))
        expected_keys = {
            "schema_version",
            "inst_id",
            "symbol_slug",
            "pid",
            "status",
            "sequence",
            "started_at_ms",
            "updated_at_ms",
            "stale_after_seconds",
        }
        assert set(payload.keys()) == expected_keys, (
            f"heartbeat JSON keys {set(payload.keys())} != {expected_keys}"
        )
        # Explicitly ensure internal config fields are NOT in the payload
        assert "failure_log_interval_seconds" not in payload


# ============================================================================
# Source guard — no trading side effects
# ============================================================================


class TestHeartbeatWriterSourceGuard:
    def test_writer_source_has_no_trading_side_effects(self) -> None:
        source = Path("src/live/heartbeat_writer.py").read_text(encoding="utf-8")

        forbidden = [
            "Trader",
            "BollCvd",
            "Strategy",
            "SymbolWorkerApp",
            "SymbolWorkerFactory",
            "src.live.workers",
            "EmailSender",
            "requests",
            "httpx",
            "websocket",
            "okx",
            "load_dotenv",
            "os.getenv",
            "asyncio.create_task",
        ]
        for token in forbidden:
            assert token not in source, (
                f"heartbeat_writer.py must not reference {token!r}"
            )

    def test_writer_source_allows_degrade_tokens(self) -> None:
        """C06b failure degrade tokens must be present in heartbeat_writer.py."""
        source = Path("src/live/heartbeat_writer.py").read_text(encoding="utf-8")

        allowed = [
            "logger.warning",
            "time.monotonic",
            "asyncio.CancelledError",
        ]
        for token in allowed:
            assert token in source, (
                f"heartbeat_writer.py must contain {token!r}"
            )
        assert "logger.exception(" not in source, (
            "heartbeat_writer.py must NOT use logger.exception"
        )
