from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.live.outbox.jsonl_outbox import JsonlOutbox
from src.live.outbox.worker_event_emitter import (
    WORKER_DRAIN_COMPLETED,
    WORKER_DRAIN_STARTED,
    WORKER_DRAIN_TIMEOUT,
    WORKER_EVENT_SEVERITIES,
    WORKER_EVENT_TYPES,
    WORKER_HEARTBEAT_WRITE_FAILED,
    WORKER_STARTED,
    WORKER_STARTUP_RECOVERY_COMPLETED,
    WORKER_STARTUP_RECOVERY_FAILED,
    WORKER_STOPPED,
    WORKER_STOPPING,
    WORKER_TRADING_HALTED,
    WorkerEvent,
    WorkerEventEmitter,
)


# ============================================================================
# 1. emit writes JSONL
# ============================================================================


class TestEmitWritesJsonl:
    def test_emit_writes_one_jsonl_line(self, tmp_path: Path) -> None:
        outbox = JsonlOutbox(
            tmp_path / "events" / "worker_events_ETH-USDT-SWAP.jsonl"
        )
        emitter = WorkerEventEmitter(symbol="ETH-USDT-SWAP", outbox=outbox)
        emitter.emit("WORKER_STARTED", {"pid": 123}, ts_ms=1000)

        assert outbox.path.exists()

        text = outbox.path.read_text(encoding="utf-8")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        assert len(lines) == 1

        obj = json.loads(lines[0])
        assert obj["event_type"] == "WORKER_STARTED"
        assert obj["ts_ms"] == 1000
        assert obj["payload"] == {
            "symbol": "ETH-USDT-SWAP",
            "severity": "INFO",
            "data": {"pid": 123},
        }


# ============================================================================
# 2. emit returns WorkerEvent
# ============================================================================


class TestEmitReturnsWorkerEvent:
    def test_emit_returns_correct_worker_event(self, tmp_path: Path) -> None:
        outbox = JsonlOutbox(tmp_path / "events" / "out.jsonl")
        emitter = WorkerEventEmitter(symbol="ETH-USDT-SWAP", outbox=outbox)
        event = emitter.emit("WORKER_STARTED", {"pid": 123}, ts_ms=1000)

        assert isinstance(event, WorkerEvent)
        assert event.ts_ms == 1000
        assert event.event_type == "WORKER_STARTED"
        assert event.symbol == "ETH-USDT-SWAP"
        assert event.severity == "INFO"
        assert event.payload == {"pid": 123}


# ============================================================================
# 3. symbol auto-written into payload
# ============================================================================


class TestSymbolAutoWrittenIntoPayload:
    def test_symbol_in_jsonl_payload(self, tmp_path: Path) -> None:
        outbox = JsonlOutbox(tmp_path / "events" / "out.jsonl")
        emitter = WorkerEventEmitter(symbol="ETH-USDT-SWAP", outbox=outbox)
        emitter.emit("WORKER_STARTED", {"pid": 1}, ts_ms=1)

        events = outbox.read_events()
        assert len(events) == 1
        assert events[0].payload["symbol"] == "ETH-USDT-SWAP"


# ============================================================================
# 4. severity defaults to INFO
# ============================================================================


class TestSeverityDefaultsToInfo:
    def test_severity_default_info_in_return(self, tmp_path: Path) -> None:
        outbox = JsonlOutbox(tmp_path / "events" / "out.jsonl")
        emitter = WorkerEventEmitter(symbol="ETH-USDT-SWAP", outbox=outbox)
        event = emitter.emit("WORKER_STOPPED", ts_ms=1)
        assert event.severity == "INFO"

    def test_severity_default_info_in_jsonl(self, tmp_path: Path) -> None:
        outbox = JsonlOutbox(tmp_path / "events" / "out.jsonl")
        emitter = WorkerEventEmitter(symbol="ETH-USDT-SWAP", outbox=outbox)
        emitter.emit("WORKER_STOPPED", ts_ms=1)

        events = outbox.read_events()
        assert events[0].payload["severity"] == "INFO"


# ============================================================================
# 5. severity supports WARNING / ERROR / CRITICAL
# ============================================================================


class TestSeveritySupportsAllLevels:
    @pytest.mark.parametrize("level", ["WARNING", "ERROR", "CRITICAL"])
    def test_severity_level(self, tmp_path: Path, level: str) -> None:
        outbox = JsonlOutbox(tmp_path / "events" / "out.jsonl")
        emitter = WorkerEventEmitter(symbol="ETH-USDT-SWAP", outbox=outbox)
        event = emitter.emit("WORKER_STARTED", ts_ms=1, severity=level)
        assert event.severity == level

        events = outbox.read_events()
        assert events[0].payload["severity"] == level

    @pytest.mark.parametrize(
        "input_level, expected",
        [("info", "INFO"), ("warning", "WARNING"), ("error", "ERROR"), ("critical", "CRITICAL")],
    )
    def test_severity_normalizes_lowercase(
        self, tmp_path: Path, input_level: str, expected: str
    ) -> None:
        outbox = JsonlOutbox(tmp_path / "events" / "out.jsonl")
        emitter = WorkerEventEmitter(symbol="ETH-USDT-SWAP", outbox=outbox)
        event = emitter.emit("WORKER_STARTED", ts_ms=1, severity=input_level)
        assert event.severity == expected

    def test_severity_rejects_invalid(self, tmp_path: Path) -> None:
        outbox = JsonlOutbox(tmp_path / "events" / "out.jsonl")
        emitter = WorkerEventEmitter(symbol="ETH-USDT-SWAP", outbox=outbox)
        with pytest.raises(ValueError):
            emitter.emit("WORKER_STARTED", ts_ms=1, severity="DEBUG")


# ============================================================================
# 6. payload defaults to {}
# ============================================================================


class TestPayloadDefaultsToEmptyDict:
    def test_payload_default_empty_in_return(self, tmp_path: Path) -> None:
        outbox = JsonlOutbox(tmp_path / "events" / "out.jsonl")
        emitter = WorkerEventEmitter(symbol="ETH-USDT-SWAP", outbox=outbox)
        event = emitter.emit("WORKER_STOPPED", ts_ms=1)
        assert event.payload == {}

    def test_payload_default_empty_in_jsonl_data(self, tmp_path: Path) -> None:
        outbox = JsonlOutbox(tmp_path / "events" / "out.jsonl")
        emitter = WorkerEventEmitter(symbol="ETH-USDT-SWAP", outbox=outbox)
        emitter.emit("WORKER_STOPPED", ts_ms=1)

        events = outbox.read_events()
        assert events[0].payload["data"] == {}


# ============================================================================
# 7. event_type empty raises ValueError
# ============================================================================


class TestEventTypeEmptyRaises:
    def test_event_type_empty_string_raises(self, tmp_path: Path) -> None:
        outbox = JsonlOutbox(tmp_path / "events" / "out.jsonl")
        emitter = WorkerEventEmitter(symbol="ETH-USDT-SWAP", outbox=outbox)
        with pytest.raises(ValueError):
            emitter.emit("", ts_ms=1)

    def test_event_type_whitespace_raises(self, tmp_path: Path) -> None:
        outbox = JsonlOutbox(tmp_path / "events" / "out.jsonl")
        emitter = WorkerEventEmitter(symbol="ETH-USDT-SWAP", outbox=outbox)
        with pytest.raises(ValueError):
            emitter.emit("   ", ts_ms=1)

    def test_event_type_none_raises(self, tmp_path: Path) -> None:
        outbox = JsonlOutbox(tmp_path / "events" / "out.jsonl")
        emitter = WorkerEventEmitter(symbol="ETH-USDT-SWAP", outbox=outbox)
        with pytest.raises(ValueError):
            emitter.emit(None, ts_ms=1)  # type: ignore[arg-type]


# ============================================================================
# 8. symbol empty raises ValueError
# ============================================================================


class TestSymbolEmptyRaises:
    def test_symbol_empty_string_raises(self, tmp_path: Path) -> None:
        outbox = JsonlOutbox(tmp_path / "events" / "out.jsonl")
        with pytest.raises(ValueError):
            WorkerEventEmitter(symbol="", outbox=outbox)

    def test_symbol_whitespace_raises(self, tmp_path: Path) -> None:
        outbox = JsonlOutbox(tmp_path / "events" / "out.jsonl")
        with pytest.raises(ValueError):
            WorkerEventEmitter(symbol="   ", outbox=outbox)

    def test_symbol_none_raises(self, tmp_path: Path) -> None:
        outbox = JsonlOutbox(tmp_path / "events" / "out.jsonl")
        with pytest.raises(ValueError):
            WorkerEventEmitter(symbol=None, outbox=outbox)  # type: ignore[arg-type]

    def test_symbol_is_stripped(self, tmp_path: Path) -> None:
        outbox = JsonlOutbox(tmp_path / "events" / "out.jsonl")
        emitter = WorkerEventEmitter(symbol="  ETH-USDT-SWAP  ", outbox=outbox)
        assert emitter.symbol == "ETH-USDT-SWAP"


# ============================================================================
# 9. payload must be dict
# ============================================================================


class TestPayloadMustBeDict:
    def test_payload_list_raises(self, tmp_path: Path) -> None:
        outbox = JsonlOutbox(tmp_path / "events" / "out.jsonl")
        emitter = WorkerEventEmitter(symbol="ETH-USDT-SWAP", outbox=outbox)
        with pytest.raises(ValueError):
            emitter.emit("WORKER_STARTED", [1, 2, 3], ts_ms=1)  # type: ignore[arg-type]

    def test_payload_string_raises(self, tmp_path: Path) -> None:
        outbox = JsonlOutbox(tmp_path / "events" / "out.jsonl")
        emitter = WorkerEventEmitter(symbol="ETH-USDT-SWAP", outbox=outbox)
        with pytest.raises(ValueError):
            emitter.emit("WORKER_STARTED", "not_a_dict", ts_ms=1)  # type: ignore[arg-type]

    def test_payload_none_is_ok(self, tmp_path: Path) -> None:
        outbox = JsonlOutbox(tmp_path / "events" / "out.jsonl")
        emitter = WorkerEventEmitter(symbol="ETH-USDT-SWAP", outbox=outbox)
        event = emitter.emit("WORKER_STARTED", None, ts_ms=1)
        assert event.payload == {}


# ============================================================================
# 10. payload shallow copy
# ============================================================================


class TestPayloadShallowCopy:
    def test_mutating_original_does_not_affect_event(self, tmp_path: Path) -> None:
        outbox = JsonlOutbox(tmp_path / "events" / "out.jsonl")
        emitter = WorkerEventEmitter(symbol="ETH-USDT-SWAP", outbox=outbox)

        payload = {"pid": 123}
        event = emitter.emit("WORKER_STARTED", payload, ts_ms=1)
        payload["pid"] = 456

        assert event.payload == {"pid": 123}

    def test_mutating_original_does_not_affect_jsonl(self, tmp_path: Path) -> None:
        outbox = JsonlOutbox(tmp_path / "events" / "out.jsonl")
        emitter = WorkerEventEmitter(symbol="ETH-USDT-SWAP", outbox=outbox)

        payload = {"pid": 123}
        emitter.emit("WORKER_STARTED", payload, ts_ms=1)
        payload["pid"] = 456

        events = outbox.read_events()
        assert events[0].payload["data"] == {"pid": 123}


# ============================================================================
# 11. read_events can read back emitter events
# ============================================================================


class TestReadEventsRoundTrip:
    def test_read_events_round_trip(self, tmp_path: Path) -> None:
        outbox = JsonlOutbox(tmp_path / "events" / "out.jsonl")
        emitter = WorkerEventEmitter(symbol="ETH-USDT-SWAP", outbox=outbox)
        emitter.emit("WORKER_STARTED", {"pid": 123}, ts_ms=1000)

        events = outbox.read_events()
        assert len(events) == 1
        assert events[0].event_type == "WORKER_STARTED"
        assert events[0].payload["symbol"] == "ETH-USDT-SWAP"
        assert events[0].payload["severity"] == "INFO"
        assert events[0].payload["data"] == {"pid": 123}


# ============================================================================
# 12. constants exist
# ============================================================================


class TestConstants:
    def test_worker_event_types_contains_all_standard_events(self) -> None:
        required = {
            WORKER_STARTED,
            WORKER_STOPPING,
            WORKER_STOPPED,
            WORKER_STARTUP_RECOVERY_COMPLETED,
            WORKER_STARTUP_RECOVERY_FAILED,
            WORKER_TRADING_HALTED,
            WORKER_HEARTBEAT_WRITE_FAILED,
            WORKER_DRAIN_STARTED,
            WORKER_DRAIN_COMPLETED,
            WORKER_DRAIN_TIMEOUT,
        }
        assert required.issubset(WORKER_EVENT_TYPES)

    def test_worker_event_severities_is_correct(self) -> None:
        assert WORKER_EVENT_SEVERITIES == {"INFO", "WARNING", "ERROR", "CRITICAL"}

    def test_worker_event_severities_is_frozenset(self) -> None:
        assert isinstance(WORKER_EVENT_SEVERITIES, frozenset)


# ============================================================================
# 13. source guard
# ============================================================================


class TestWorkerEventEmitterSourceGuard:
    def test_no_forbidden_imports(self) -> None:
        source_path = (
            Path(__file__).parents[3]
            / "src"
            / "live"
            / "outbox"
            / "worker_event_emitter.py"
        )
        source = source_path.read_text()

        forbidden = [
            "Trader",
            "Strategy",
            "requests",
            "httpx",
            "websocket",
            "okx",
            "EmailSender",
            "os.getenv",
            "load_dotenv",
            "asyncio",
            "supervisor",
            "workers",
            "SymbolWorkerApp",
            "SymbolWorkerFactory",
        ]
        for token in forbidden:
            assert token not in source, (
                f"worker_event_emitter.py must not import/use {token}"
            )


# ============================================================================
# Extra: multiple emits, symbol strip, WorkerEvent frozen
# ============================================================================


class TestMultipleEmits:
    def test_multiple_emits_appends_lines(self, tmp_path: Path) -> None:
        outbox = JsonlOutbox(tmp_path / "events" / "out.jsonl")
        emitter = WorkerEventEmitter(symbol="ETH-USDT-SWAP", outbox=outbox)
        emitter.emit("WORKER_STARTED", {"pid": 1}, ts_ms=1)
        emitter.emit("WORKER_STOPPING", {}, ts_ms=2, severity="WARNING")
        emitter.emit("WORKER_STOPPED", ts_ms=3, severity="ERROR")

        events = outbox.read_events()
        assert len(events) == 3
        assert events[0].event_type == "WORKER_STARTED"
        assert events[1].event_type == "WORKER_STOPPING"
        assert events[2].event_type == "WORKER_STOPPED"
        assert events[0].payload["severity"] == "INFO"
        assert events[1].payload["severity"] == "WARNING"
        assert events[2].payload["severity"] == "ERROR"


class TestWorkerEventIsFrozen:
    def test_worker_event_is_frozen(self) -> None:
        event = WorkerEvent(
            ts_ms=1,
            event_type="WORKER_STARTED",
            symbol="ETH-USDT-SWAP",
            severity="INFO",
            payload={},
        )
        with pytest.raises(Exception):
            event.ts_ms = 999  # type: ignore[misc]
