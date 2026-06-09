from __future__ import annotations

import json
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

    def test_invalid_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="interval_seconds must be > 0"):
            HeartbeatWriterConfig(interval_seconds=0)

    def test_invalid_stale_after_raises(self) -> None:
        with pytest.raises(ValueError, match="stale_after_seconds must be > 0"):
            HeartbeatWriterConfig(stale_after_seconds=0)


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
