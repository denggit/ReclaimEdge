from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.live.outbox.jsonl_outbox import JsonlOutbox, JsonlOutboxEvent


class TestAppendCreatesParentAndReturnsEvent:
    def test_append_creates_parent_and_returns_event(self, tmp_path: Path) -> None:
        outbox = JsonlOutbox(tmp_path / "events" / "worker.jsonl")
        event = outbox.append("CHILD_STARTED", {"pid": 123}, ts_ms=1000)

        assert isinstance(event, JsonlOutboxEvent)
        assert event.event_type == "CHILD_STARTED"
        assert event.ts_ms == 1000
        assert event.payload == {"pid": 123}

        assert outbox.path.exists()

        lines = outbox.path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        obj = json.loads(lines[0])
        assert obj["event_type"] == "CHILD_STARTED"
        assert obj["ts_ms"] == 1000
        assert obj["payload"] == {"pid": 123}


class TestAppendMultipleEventsOnePerLine:
    def test_append_multiple_events_one_per_line(self, tmp_path: Path) -> None:
        outbox = JsonlOutbox(tmp_path / "multi.jsonl")
        outbox.append("A", {"x": 1}, ts_ms=1)
        outbox.append("B", {"y": 2}, ts_ms=2)

        text = outbox.path.read_text(encoding="utf-8")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        assert len(lines) == 2

        events = outbox.read_events()
        assert len(events) == 2
        assert events[0].event_type == "A"
        assert events[1].event_type == "B"


class TestAppendRejectsEmptyEventType:
    def test_append_rejects_empty_event_type(self, tmp_path: Path) -> None:
        outbox = JsonlOutbox(tmp_path / "bad.jsonl")
        with pytest.raises(ValueError):
            outbox.append("", ts_ms=1)


class TestPayloadDefaultsToEmptyDict:
    def test_payload_defaults_to_empty_dict(self, tmp_path: Path) -> None:
        outbox = JsonlOutbox(tmp_path / "ping.jsonl")
        event = outbox.append("PING", ts_ms=1)
        assert event.payload == {}


class TestReadEventsMissingFileReturnsEmptyList:
    def test_read_events_missing_file_returns_empty_list(self, tmp_path: Path) -> None:
        outbox = JsonlOutbox(tmp_path / "does_not_exist.jsonl")
        assert outbox.read_events() == []


class TestReadEventsRaisesOnInvalidJson:
    def test_read_events_raises_on_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{bad}\n", encoding="utf-8")

        outbox = JsonlOutbox(path)
        with pytest.raises(json.JSONDecodeError):
            outbox.read_events()


class TestReadEventsRaisesOnMissingEventType:
    def test_read_events_raises_on_missing_event_type(self, tmp_path: Path) -> None:
        path = tmp_path / "no_type.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"ts_ms":1,"payload":{}}\n', encoding="utf-8")

        outbox = JsonlOutbox(path)
        with pytest.raises(ValueError, match="missing event_type"):
            outbox.read_events()


class TestReadEventsRaisesOnNonObjectPayload:
    def test_read_events_raises_on_non_object_payload(self, tmp_path: Path) -> None:
        path = tmp_path / "bad_payload.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"event_type":"BAD_PAYLOAD","ts_ms":1,"payload":[1,2,3]}\n', encoding="utf-8")

        outbox = JsonlOutbox(path)
        with pytest.raises(ValueError, match="payload must be an object"):
            outbox.read_events()


class TestJsonlOutboxSourceHasNoRuntimeSideEffectImports:
    def test_jsonl_outbox_source_has_no_runtime_side_effect_imports(self) -> None:
        source_path = (
            Path(__file__).parents[3] / "src" / "live" / "outbox" / "jsonl_outbox.py"
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
        ]
        for token in forbidden:
            assert token not in source, f"jsonl_outbox.py must not import/use {token}"
