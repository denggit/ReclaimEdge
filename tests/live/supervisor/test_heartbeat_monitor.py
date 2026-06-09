#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D04 unit tests for HeartbeatMonitor — covers config, status types,
edge cases, and source guard checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.live.supervisor.heartbeat_monitor import (
    HeartbeatMonitor,
    HeartbeatMonitorConfig,
    HeartbeatStatus,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_HEARTBEAT_MONITOR_SOURCE_PATH = (
    _PROJECT_ROOT / "src" / "live" / "supervisor" / "heartbeat_monitor.py"
)


def _heartbeat_monitor_source() -> str:
    return _HEARTBEAT_MONITOR_SOURCE_PATH.read_text(encoding="utf-8")


# ============================================================================
# 1. test_default_config
# ============================================================================


def test_default_config() -> None:
    config = HeartbeatMonitorConfig()
    assert config.default_stale_after_seconds == 30.0


# ============================================================================
# 2. test_invalid_default_stale_after_raises
# ============================================================================


def test_invalid_default_stale_after_raises() -> None:
    with pytest.raises(ValueError, match="default_stale_after_seconds must be > 0"):
        HeartbeatMonitorConfig(default_stale_after_seconds=0)


# ============================================================================
# 3. test_missing_file_status
# ============================================================================


def test_missing_file_status(tmp_path: Path) -> None:
    monitor = HeartbeatMonitor(clock_ms=lambda: 100_000)
    status = monitor.read_status(symbol="ETH-USDT-SWAP", path=tmp_path / "missing.json")
    assert status.status == "missing"
    assert status.missing is True
    assert status.fresh is False
    assert status.stale is False
    assert status.invalid is False
    assert status.ok is False
    assert status.error == "heartbeat file missing"


# ============================================================================
# 4. test_invalid_json_status
# ============================================================================


def test_invalid_json_status(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not-json")
    monitor = HeartbeatMonitor(clock_ms=lambda: 100_000)
    status = monitor.read_status(symbol="ETH-USDT-SWAP", path=path)
    assert status.status == "invalid"
    assert status.invalid is True
    assert "JSONDecodeError" in status.error


# ============================================================================
# 5. test_non_object_json_status
# ============================================================================


def test_non_object_json_status(tmp_path: Path) -> None:
    path = tmp_path / "array.json"
    path.write_text("[]")
    monitor = HeartbeatMonitor(clock_ms=lambda: 100_000)
    status = monitor.read_status(symbol="ETH-USDT-SWAP", path=path)
    assert status.status == "invalid"
    assert status.invalid is True
    assert "JSON object" in status.error


# ============================================================================
# 6. test_invalid_missing_updated_at_ms
# ============================================================================


def test_invalid_missing_updated_at_ms() -> None:
    monitor = HeartbeatMonitor(clock_ms=lambda: 100_000)
    status = monitor.evaluate_payload(
        symbol="ETH-USDT-SWAP",
        path="/tmp/test.json",
        payload={},
    )
    assert status.status == "invalid"
    assert status.invalid is True
    assert status.error is not None
    assert "updated_at_ms" in status.error


# ============================================================================
# 7. test_invalid_bad_updated_at_ms
# ============================================================================


def test_invalid_bad_updated_at_ms() -> None:
    monitor = HeartbeatMonitor(clock_ms=lambda: 100_000)
    status = monitor.evaluate_payload(
        symbol="ETH-USDT-SWAP",
        path="/tmp/test.json",
        payload={"updated_at_ms": "abc"},
    )
    assert status.status == "invalid"
    assert status.invalid is True


# ============================================================================
# 8. test_fresh_payload
# ============================================================================


def test_fresh_payload() -> None:
    monitor = HeartbeatMonitor(clock_ms=lambda: 100_000)
    payload = {
        "schema_version": 1,
        "inst_id": "ETH-USDT-SWAP",
        "symbol_slug": "ETH-USDT-SWAP",
        "pid": 123,
        "status": "running",
        "sequence": 5,
        "started_at_ms": 80_000,
        "updated_at_ms": 95_000,
        "stale_after_seconds": 30.0,
    }
    status = monitor.evaluate_payload(
        symbol="ETH-USDT-SWAP",
        path="/tmp/test.json",
        payload=payload,
    )
    assert status.status == "fresh"
    assert status.fresh is True
    assert status.ok is True
    assert status.age_seconds == 5.0
    assert status.sequence == 5
    assert status.pid == 123
    assert status.worker_status == "running"


# ============================================================================
# 9. test_stale_payload
# ============================================================================


def test_stale_payload() -> None:
    monitor = HeartbeatMonitor(clock_ms=lambda: 100_000)
    payload = {
        "updated_at_ms": 60_000,
        "stale_after_seconds": 30.0,
    }
    status = monitor.evaluate_payload(
        symbol="ETH-USDT-SWAP",
        path="/tmp/test.json",
        payload=payload,
    )
    assert status.status == "stale"
    assert status.stale is True
    assert status.fresh is False
    assert status.ok is False
    assert status.age_seconds == 40.0


# ============================================================================
# 10. test_stale_threshold_is_strictly_greater_than
# ============================================================================


def test_stale_threshold_is_strictly_greater_than() -> None:
    monitor = HeartbeatMonitor(clock_ms=lambda: 100_000)
    payload = {
        "updated_at_ms": 70_000,
        "stale_after_seconds": 30.0,
    }
    status = monitor.evaluate_payload(
        symbol="ETH-USDT-SWAP",
        path="/tmp/test.json",
        payload=payload,
    )
    assert status.status == "fresh"
    assert status.fresh is True
    assert status.stale is False
    assert status.age_seconds == 30.0


# ============================================================================
# 11. test_future_updated_at_age_clamped_to_zero
# ============================================================================


def test_future_updated_at_age_clamped_to_zero() -> None:
    monitor = HeartbeatMonitor(clock_ms=lambda: 100_000)
    payload = {
        "updated_at_ms": 110_000,
        "stale_after_seconds": 30.0,
    }
    status = monitor.evaluate_payload(
        symbol="ETH-USDT-SWAP",
        path="/tmp/test.json",
        payload=payload,
    )
    assert status.age_seconds == 0.0
    assert status.fresh is True


# ============================================================================
# 12. test_invalid_stale_after_uses_default
# ============================================================================


def test_invalid_stale_after_uses_default() -> None:
    config = HeartbeatMonitorConfig(default_stale_after_seconds=45.0)
    monitor = HeartbeatMonitor(config=config, clock_ms=lambda: 100_000)
    payload = {
        "updated_at_ms": 95_000,
        "stale_after_seconds": "bad",
    }
    status = monitor.evaluate_payload(
        symbol="ETH-USDT-SWAP",
        path="/tmp/test.json",
        payload=payload,
    )
    assert status.stale_after_seconds == 45.0


# ============================================================================
# 13. test_negative_stale_after_uses_default
# ============================================================================


def test_negative_stale_after_uses_default() -> None:
    config = HeartbeatMonitorConfig(default_stale_after_seconds=45.0)
    monitor = HeartbeatMonitor(config=config, clock_ms=lambda: 100_000)
    payload = {
        "updated_at_ms": 95_000,
        "stale_after_seconds": -1,
    }
    status = monitor.evaluate_payload(
        symbol="ETH-USDT-SWAP",
        path="/tmp/test.json",
        payload=payload,
    )
    assert status.stale_after_seconds == 45.0


# ============================================================================
# 14. test_sequence_pid_coercion
# ============================================================================


def test_sequence_pid_coercion() -> None:
    monitor = HeartbeatMonitor(clock_ms=lambda: 100_000)
    payload = {
        "updated_at_ms": 95_000,
        "sequence": "7",
        "pid": "123",
    }
    status = monitor.evaluate_payload(
        symbol="ETH-USDT-SWAP",
        path="/tmp/test.json",
        payload=payload,
    )
    assert status.sequence == 7
    assert status.pid == 123


# ============================================================================
# 15. test_bool_sequence_pid_rejected
# ============================================================================


def test_bool_sequence_pid_rejected() -> None:
    monitor = HeartbeatMonitor(clock_ms=lambda: 100_000)
    payload = {
        "updated_at_ms": 95_000,
        "sequence": True,
        "pid": False,
    }
    status = monitor.evaluate_payload(
        symbol="ETH-USDT-SWAP",
        path="/tmp/test.json",
        payload=payload,
    )
    assert status.sequence is None
    assert status.pid is None
    # updated_at_ms is valid, so status should be fresh (age 5 with clock 100k)
    assert status.status == "fresh"


# ============================================================================
# 16. test_read_status_from_file_fresh
# ============================================================================


def test_read_status_from_file_fresh(tmp_path: Path) -> None:
    path = tmp_path / "heartbeat.json"
    payload = {
        "schema_version": 1,
        "inst_id": "ETH-USDT-SWAP",
        "symbol_slug": "ETH-USDT-SWAP",
        "pid": 123,
        "status": "running",
        "sequence": 5,
        "started_at_ms": 80_000,
        "updated_at_ms": 95_000,
        "stale_after_seconds": 30.0,
    }
    path.write_text(json.dumps(payload))
    monitor = HeartbeatMonitor(clock_ms=lambda: 100_000)
    status = monitor.read_status(symbol="ETH-USDT-SWAP", path=path)
    assert status.status == "fresh"
    assert status.fresh is True
    assert status.ok is True


# ============================================================================
# 17. test_heartbeat_status_ok_property
# ============================================================================


def test_heartbeat_status_ok_property() -> None:
    # fresh -> ok is True
    fresh_status = HeartbeatStatus(
        symbol="X",
        path=Path("/tmp/x.json"),
        status="fresh",
        fresh=True,
        missing=False,
        stale=False,
        invalid=False,
        age_seconds=1.0,
        sequence=1,
        pid=100,
        worker_status="running",
        updated_at_ms=1000,
        stale_after_seconds=30.0,
    )
    assert fresh_status.ok is True

    # missing -> ok is False
    missing_status = HeartbeatStatus(
        symbol="X",
        path=Path("/tmp/x.json"),
        status="missing",
        fresh=False,
        missing=True,
        stale=False,
        invalid=False,
        age_seconds=None,
        sequence=None,
        pid=None,
        worker_status=None,
        updated_at_ms=None,
        stale_after_seconds=30.0,
        error="heartbeat file missing",
    )
    assert missing_status.ok is False

    # stale -> ok is False
    stale_status = HeartbeatStatus(
        symbol="X",
        path=Path("/tmp/x.json"),
        status="stale",
        fresh=False,
        missing=False,
        stale=True,
        invalid=False,
        age_seconds=40.0,
        sequence=1,
        pid=100,
        worker_status="running",
        updated_at_ms=60_000,
        stale_after_seconds=30.0,
    )
    assert stale_status.ok is False

    # invalid -> ok is False
    invalid_status = HeartbeatStatus(
        symbol="X",
        path=Path("/tmp/x.json"),
        status="invalid",
        fresh=False,
        missing=False,
        stale=False,
        invalid=True,
        age_seconds=None,
        sequence=None,
        pid=None,
        worker_status=None,
        updated_at_ms=None,
        stale_after_seconds=30.0,
        error="bad payload",
    )
    assert invalid_status.ok is False


# ============================================================================
# 18. test_source_has_no_trading_child_or_network_side_effects
# ============================================================================


def test_source_has_no_trading_child_or_network_side_effects() -> None:
    source = _heartbeat_monitor_source()

    forbidden = [
        "HeartbeatWriter",
        "ChildProcess",
        "ReclaimSupervisor",
        "SymbolWorkerApp",
        "Trader",
        "RuntimePaths",
        "RECLAIM_SYMBOLS",
        "BTC",
        "ETH-USDT-SWAP",
        "run_symbol_worker",
        "account_position_sync_worker",
        "strategy_tick_worker",
        "execution_worker",
        "BollCvd",
        "okx",
        "requests",
        "httpx",
        "websocket",
        "subprocess",
        "multiprocessing",
        "os.getenv",
        "load_dotenv",
        "send_email",
        "EmailSender",
        ".write_text(",
        "open(",
    ]
    for token in forbidden:
        assert token not in source, (
            f"heartbeat_monitor.py must NOT contain {token!r}"
        )
