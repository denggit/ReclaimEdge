from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.live.outbox.atomic_json import read_json_or_none, write_json_atomic
from src.live.supervisor.child_event_reader import (
    ChildEvent,
    ChildEventReadError,
    ChildEventReader,
    ChildEventReadResult,
    _now_ms,
)


# ============================================================================
# Helpers
# ============================================================================

_CHILD_EVENT_READER_SOURCE = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "src"
    / "live"
    / "supervisor"
    / "child_event_reader.py"
)


def _write_outbox_lines(path: Path, *lines: str) -> None:
    """Write JSONL lines to an outbox file, appending newlines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line)
            if not line.endswith("\n"):
                fh.write("\n")
            fh.flush()


def _write_outbox_raw(path: Path, content: str) -> None:
    """Write raw content to an outbox file (exactly as given)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _append_outbox_raw(path: Path, content: str) -> None:
    """Append raw content to an outbox file."""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(content)


def _make_event(
    ts_ms: int = 1,
    event_type: str = "WORKER_STARTED",
    payload: dict | None = None,
) -> str:
    """Return a single JSONL line (with trailing newline)."""
    obj = {
        "ts_ms": ts_ms,
        "event_type": event_type,
        "payload": payload or {"symbol": "ETH-USDT-SWAP", "severity": "INFO", "data": {}},
    }
    return json.dumps(obj, sort_keys=True) + "\n"


# ============================================================================
# 1. missing outbox returns empty result
# ============================================================================


class TestMissingOutboxReturnsEmpty:
    def test_missing_outbox_returns_empty(self, tmp_path: Path) -> None:
        reader = ChildEventReader(
            outbox_path=tmp_path / "missing.jsonl",
            cursor_path=tmp_path / "cursor.json",
        )
        result = reader.read_new_events()

        assert result.events == []
        assert result.errors == []
        assert result.cursor_offset == 0
        assert result.reached_eof is True
        assert result.truncated_or_rotated is False
        assert result.bytes_read == 0

        # Must not create the outbox file.
        assert not (tmp_path / "missing.jsonl").exists()


# ============================================================================
# 2. reads new events from offset 0
# ============================================================================


class TestReadsNewEventsFromOffset0:
    def test_reads_two_events(self, tmp_path: Path) -> None:
        outbox = tmp_path / "worker_events_ETH-USDT-SWAP.jsonl"
        _write_outbox_lines(
            outbox,
            _make_event(1, "WORKER_STARTED", {"symbol": "ETH-USDT-SWAP", "severity": "INFO", "data": {"pid": 1}}),
            _make_event(2, "WORKER_STOPPED", {"symbol": "ETH-USDT-SWAP", "severity": "INFO", "data": {}}),
        )

        reader = ChildEventReader(
            outbox_path=outbox,
            cursor_path=tmp_path / "cursor.json",
        )
        result = reader.read_new_events()

        assert len(result.events) == 2
        assert result.errors == []
        assert result.cursor_offset > 0

        # Cursor file should exist.
        cursor_path = tmp_path / "cursor.json"
        assert cursor_path.exists()
        cursor_data = read_json_or_none(cursor_path)
        assert cursor_data is not None
        assert cursor_data["offset"] == result.cursor_offset
        assert cursor_data["path"] == str(outbox)

        # Verify first event.
        e0 = result.events[0]
        assert e0.ts_ms == 1
        assert e0.event_type == "WORKER_STARTED"
        assert e0.source_path == str(outbox)

        # Verify second event.
        e1 = result.events[1]
        assert e1.ts_ms == 2
        assert e1.event_type == "WORKER_STOPPED"


# ============================================================================
# 3. second read only returns appended events
# ============================================================================


class TestIncrementalRead:
    def test_second_read_only_returns_new(self, tmp_path: Path) -> None:
        outbox = tmp_path / "events.jsonl"
        cursor = tmp_path / "cursor.json"

        # Write first event.
        _write_outbox_lines(outbox, _make_event(1, "WORKER_STARTED"))

        reader = ChildEventReader(outbox_path=outbox, cursor_path=cursor)
        result1 = reader.read_new_events()
        assert len(result1.events) == 1
        assert result1.events[0].ts_ms == 1

        # Append second event.
        _write_outbox_lines(outbox, _make_event(2, "WORKER_STOPPED"))

        result2 = reader.read_new_events()
        assert len(result2.events) == 1
        assert result2.events[0].ts_ms == 2
        assert result2.events[0].event_type == "WORKER_STOPPED"


# ============================================================================
# 4. cursor survives new reader instance
# ============================================================================


class TestCursorSurvivesNewReader:
    def test_cursor_persists_across_instances(self, tmp_path: Path) -> None:
        outbox = tmp_path / "events.jsonl"
        cursor = tmp_path / "cursor.json"

        _write_outbox_lines(outbox, _make_event(1, "WORKER_STARTED"))

        reader1 = ChildEventReader(outbox_path=outbox, cursor_path=cursor)
        result1 = reader1.read_new_events()
        assert len(result1.events) == 1

        # New reader instance, same cursor.
        reader2 = ChildEventReader(outbox_path=outbox, cursor_path=cursor)
        result2 = reader2.read_new_events()
        assert len(result2.events) == 0
        assert result2.reached_eof is True

        # Append new event.
        _write_outbox_lines(outbox, _make_event(2, "WORKER_STOPPED"))

        result3 = reader2.read_new_events()
        assert len(result3.events) == 1
        assert result3.events[0].ts_ms == 2


# ============================================================================
# 5. bad JSON does not raise
# ============================================================================


class TestBadJson:
    def test_bad_json_returns_error_not_raise(self, tmp_path: Path) -> None:
        outbox = tmp_path / "events.jsonl"
        cursor = tmp_path / "cursor.json"

        # Write a bad JSON line followed by a newline.
        _write_outbox_raw(outbox, "{bad json\n")

        reader = ChildEventReader(outbox_path=outbox, cursor_path=cursor)
        result = reader.read_new_events()

        assert result.events == []
        assert len(result.errors) == 1
        assert result.errors[0].error_type == "BAD_JSON"
        # Cursor should advance past the bad line.
        assert result.cursor_offset > 0


# ============================================================================
# 6. invalid event object (not a dict)
# ============================================================================


class TestInvalidEventObject:
    def test_array_returns_invalid_event_object(self, tmp_path: Path) -> None:
        outbox = tmp_path / "events.jsonl"
        cursor = tmp_path / "cursor.json"

        _write_outbox_raw(outbox, "[]\n")

        reader = ChildEventReader(outbox_path=outbox, cursor_path=cursor)
        result = reader.read_new_events()

        assert result.events == []
        assert len(result.errors) == 1
        assert result.errors[0].error_type == "INVALID_EVENT_OBJECT"
        assert result.cursor_offset > 0

    def test_string_returns_invalid_event_object(self, tmp_path: Path) -> None:
        outbox = tmp_path / "events.jsonl"
        cursor = tmp_path / "cursor.json"

        _write_outbox_raw(outbox, '"hello"\n')

        reader = ChildEventReader(outbox_path=outbox, cursor_path=cursor)
        result = reader.read_new_events()

        assert result.events == []
        assert len(result.errors) == 1
        assert result.errors[0].error_type == "INVALID_EVENT_OBJECT"


# ============================================================================
# 7. missing / invalid ts_ms
# ============================================================================


class TestInvalidTsMs:
    def test_missing_ts_ms(self, tmp_path: Path) -> None:
        outbox = tmp_path / "events.jsonl"
        cursor = tmp_path / "cursor.json"

        obj = {"event_type": "WORKER_STARTED", "payload": {}}
        _write_outbox_raw(outbox, json.dumps(obj, sort_keys=True) + "\n")

        reader = ChildEventReader(outbox_path=outbox, cursor_path=cursor)
        result = reader.read_new_events()

        assert result.events == []
        assert len(result.errors) == 1
        assert result.errors[0].error_type == "INVALID_TS_MS"

    def test_string_ts_ms(self, tmp_path: Path) -> None:
        outbox = tmp_path / "events.jsonl"
        cursor = tmp_path / "cursor.json"

        obj = {"ts_ms": "123", "event_type": "WORKER_STARTED", "payload": {}}
        _write_outbox_raw(outbox, json.dumps(obj, sort_keys=True) + "\n")

        reader = ChildEventReader(outbox_path=outbox, cursor_path=cursor)
        result = reader.read_new_events()

        assert result.events == []
        assert len(result.errors) == 1
        assert result.errors[0].error_type == "INVALID_TS_MS"


# ============================================================================
# 8. missing / invalid event_type
# ============================================================================


class TestInvalidEventType:
    def test_missing_event_type(self, tmp_path: Path) -> None:
        outbox = tmp_path / "events.jsonl"
        cursor = tmp_path / "cursor.json"

        obj = {"ts_ms": 1, "payload": {}}
        _write_outbox_raw(outbox, json.dumps(obj, sort_keys=True) + "\n")

        reader = ChildEventReader(outbox_path=outbox, cursor_path=cursor)
        result = reader.read_new_events()

        assert result.events == []
        assert len(result.errors) == 1
        assert result.errors[0].error_type == "INVALID_EVENT_TYPE"

    def test_empty_event_type(self, tmp_path: Path) -> None:
        outbox = tmp_path / "events.jsonl"
        cursor = tmp_path / "cursor.json"

        obj = {"ts_ms": 1, "event_type": "", "payload": {}}
        _write_outbox_raw(outbox, json.dumps(obj, sort_keys=True) + "\n")

        reader = ChildEventReader(outbox_path=outbox, cursor_path=cursor)
        result = reader.read_new_events()

        assert result.events == []
        assert len(result.errors) == 1
        assert result.errors[0].error_type == "INVALID_EVENT_TYPE"

    def test_int_event_type(self, tmp_path: Path) -> None:
        outbox = tmp_path / "events.jsonl"
        cursor = tmp_path / "cursor.json"

        obj = {"ts_ms": 1, "event_type": 123, "payload": {}}
        _write_outbox_raw(outbox, json.dumps(obj, sort_keys=True) + "\n")

        reader = ChildEventReader(outbox_path=outbox, cursor_path=cursor)
        result = reader.read_new_events()

        assert result.events == []
        assert len(result.errors) == 1
        assert result.errors[0].error_type == "INVALID_EVENT_TYPE"


# ============================================================================
# 9. missing / invalid payload
# ============================================================================


class TestInvalidPayload:
    def test_missing_payload(self, tmp_path: Path) -> None:
        outbox = tmp_path / "events.jsonl"
        cursor = tmp_path / "cursor.json"

        obj = {"ts_ms": 1, "event_type": "WORKER_STARTED"}
        _write_outbox_raw(outbox, json.dumps(obj, sort_keys=True) + "\n")

        reader = ChildEventReader(outbox_path=outbox, cursor_path=cursor)
        result = reader.read_new_events()

        assert result.events == []
        assert len(result.errors) == 1
        assert result.errors[0].error_type == "INVALID_PAYLOAD"

    def test_array_payload(self, tmp_path: Path) -> None:
        outbox = tmp_path / "events.jsonl"
        cursor = tmp_path / "cursor.json"

        obj = {"ts_ms": 1, "event_type": "WORKER_STARTED", "payload": []}
        _write_outbox_raw(outbox, json.dumps(obj, sort_keys=True) + "\n")

        reader = ChildEventReader(outbox_path=outbox, cursor_path=cursor)
        result = reader.read_new_events()

        assert result.events == []
        assert len(result.errors) == 1
        assert result.errors[0].error_type == "INVALID_PAYLOAD"


# ============================================================================
# 10. incomplete trailing line is not processed
# ============================================================================


class TestIncompleteTrailingLine:
    def test_trailing_line_not_processed(self, tmp_path: Path) -> None:
        outbox = tmp_path / "events.jsonl"
        cursor = tmp_path / "cursor.json"

        # Write one complete event + partial line without newline.
        _write_outbox_raw(outbox, _make_event(1, "WORKER_STARTED") + '{"ts_ms":2,"event_type":"WORKER_STOPPED","pa')

        reader = ChildEventReader(outbox_path=outbox, cursor_path=cursor)
        result1 = reader.read_new_events()

        # Only the complete event should be returned.
        assert len(result1.events) == 1
        assert result1.events[0].ts_ms == 1

        # Cursor should NOT have advanced past the partial line.
        outbox_size_after_first = outbox.stat().st_size
        assert result1.cursor_offset < outbox_size_after_first

        # Complete the partial line.
        _append_outbox_raw(outbox, 'yload":{"symbol":"ETH","severity":"INFO","data":{}}}\n')

        result2 = reader.read_new_events()
        assert len(result2.events) == 1
        assert result2.events[0].ts_ms == 2
        assert result2.events[0].event_type == "WORKER_STOPPED"


# ============================================================================
# 11. no newline and short partial line does not advance cursor
# ============================================================================


class TestNoNewlineShortPartial:
    def test_no_newline_short_partial(self, tmp_path: Path) -> None:
        outbox = tmp_path / "events.jsonl"
        cursor = tmp_path / "cursor.json"

        # Write only a partial line without newline, shorter than max_bytes_per_read.
        _write_outbox_raw(outbox, '{"ts_ms":1,"event_type":"WORKER_STARTED","pa')

        reader = ChildEventReader(
            outbox_path=outbox,
            cursor_path=cursor,
            max_bytes_per_read=1024,
        )
        result = reader.read_new_events()

        assert result.events == []
        assert result.errors == []
        assert result.cursor_offset == 0


# ============================================================================
# 12. line too long with newline returns error and advances
# ============================================================================


class TestLineTooLongWithNewline:
    def test_line_too_long_with_newline(self, tmp_path: Path) -> None:
        outbox = tmp_path / "events.jsonl"
        cursor = tmp_path / "cursor.json"

        # Create a line longer than max_line_bytes with a valid JSON structure.
        long_payload = "x" * 100
        obj = {
            "ts_ms": 1,
            "event_type": "WORKER_STARTED",
            "payload": {"data": long_payload},
        }
        line = json.dumps(obj, sort_keys=True) + "\n"

        reader = ChildEventReader(
            outbox_path=outbox,
            cursor_path=cursor,
            max_line_bytes=32,
        )
        _write_outbox_raw(outbox, line)

        result = reader.read_new_events()

        assert result.events == []
        assert len(result.errors) == 1
        assert result.errors[0].error_type == "LINE_TOO_LONG"
        assert result.cursor_offset > 0


# ============================================================================
# 13. no newline but chunk reaches max_bytes_per_read
# ============================================================================


class TestNoNewlineMaxBytes:
    def test_no_newline_max_bytes_creates_error(self, tmp_path: Path) -> None:
        outbox = tmp_path / "events.jsonl"
        cursor = tmp_path / "cursor.json"

        # Write 64 bytes of data with no newline.
        data = "A" * 64
        _write_outbox_raw(outbox, data)

        reader = ChildEventReader(
            outbox_path=outbox,
            cursor_path=cursor,
            max_bytes_per_read=32,
            max_line_bytes=64 * 1024,
        )
        result = reader.read_new_events()

        assert len(result.errors) == 1
        assert result.errors[0].error_type == "LINE_TOO_LONG"
        # Cursor should advance by max_bytes_per_read to avoid being stuck.
        assert result.cursor_offset == 32


# ============================================================================
# 14. max_bytes_per_read limits memory
# ============================================================================


class TestMaxBytesPerRead:
    def test_max_bytes_per_read_limits_processing(self, tmp_path: Path) -> None:
        outbox = tmp_path / "events.jsonl"
        cursor = tmp_path / "cursor.json"

        # Write several events whose total size exceeds max_bytes_per_read.
        # Each event is ~150 bytes.
        for i in range(10):
            _write_outbox_lines(outbox, _make_event(i, f"EVENT_{i}"))

        reader = ChildEventReader(
            outbox_path=outbox,
            cursor_path=cursor,
            max_bytes_per_read=512,
        )
        result1 = reader.read_new_events()

        # Should have read some but not all events.
        assert len(result1.events) > 0
        assert len(result1.events) < 10
        assert result1.bytes_read <= 512
        assert result1.reached_eof is False

        # Second read should continue.
        result2 = reader.read_new_events()
        assert len(result2.events) > 0

        # Total should cover all events (keep reading until EOF).
        total = len(result1.events)
        total += len(result2.events)
        remaining = 10 - total
        if remaining > 0:
            result3 = reader.read_new_events()
            total += len(result3.events)
        assert total == 10


# ============================================================================
# 15. truncated / rotated outbox resets cursor
# ============================================================================


class TestTruncatedOutbox:
    def test_truncated_outbox_resets_cursor(self, tmp_path: Path) -> None:
        outbox = tmp_path / "events.jsonl"
        cursor = tmp_path / "cursor.json"

        # Write two events and read them.
        _write_outbox_lines(
            outbox,
            _make_event(1, "WORKER_STARTED"),
            _make_event(2, "WORKER_STOPPED"),
        )

        reader = ChildEventReader(outbox_path=outbox, cursor_path=cursor)
        result1 = reader.read_new_events()
        assert len(result1.events) == 2
        assert result1.truncated_or_rotated is False

        # Now overwrite the outbox with a shorter file (simulating truncate/rotate).
        _write_outbox_raw(outbox, _make_event(3, "WORKER_STARTED"))

        result2 = reader.read_new_events()
        assert result2.truncated_or_rotated is True
        assert len(result2.events) == 1
        assert result2.events[0].ts_ms == 3
        assert result2.cursor_offset > 0

        # Cursor should have been updated.
        cursor_data = read_json_or_none(cursor)
        assert cursor_data is not None
        assert cursor_data["offset"] == result2.cursor_offset


# ============================================================================
# 16. cursor path mismatch resets cursor
# ============================================================================


class TestCursorPathMismatch:
    def test_path_mismatch_resets_offset(self, tmp_path: Path) -> None:
        outbox = tmp_path / "events.jsonl"
        cursor = tmp_path / "cursor.json"

        # Write a cursor pointing to a different file.
        write_json_atomic(
            cursor,
            {
                "path": "some/other/file.jsonl",
                "offset": 999,
                "updated_ts_ms": 1,
            },
        )

        _write_outbox_lines(outbox, _make_event(1, "WORKER_STARTED"))

        reader = ChildEventReader(outbox_path=outbox, cursor_path=cursor)
        result = reader.read_new_events()

        # Should read from offset 0 despite cursor saying 999.
        assert len(result.events) == 1
        assert result.events[0].ts_ms == 1


# ============================================================================
# 17. invalid cursor offset resets cursor
# ============================================================================


class TestInvalidCursorOffset:
    def test_string_offset_resets(self, tmp_path: Path) -> None:
        outbox = tmp_path / "events.jsonl"
        cursor = tmp_path / "cursor.json"

        write_json_atomic(
            cursor,
            {
                "path": str(outbox),
                "offset": "bad",
                "updated_ts_ms": 1,
            },
        )

        _write_outbox_lines(outbox, _make_event(1, "WORKER_STARTED"))

        reader = ChildEventReader(outbox_path=outbox, cursor_path=cursor)
        result = reader.read_new_events()

        assert len(result.events) == 1

    def test_negative_offset_resets(self, tmp_path: Path) -> None:
        outbox = tmp_path / "events.jsonl"
        cursor = tmp_path / "cursor.json"

        write_json_atomic(
            cursor,
            {
                "path": str(outbox),
                "offset": -1,
                "updated_ts_ms": 1,
            },
        )

        _write_outbox_lines(outbox, _make_event(1, "WORKER_STARTED"))

        reader = ChildEventReader(outbox_path=outbox, cursor_path=cursor)
        result = reader.read_new_events()

        assert len(result.events) == 1


# ============================================================================
# 18. source guard
# ============================================================================


class TestChildEventReaderSourceGuard:
    def test_no_forbidden_imports(self) -> None:
        source = _CHILD_EVENT_READER_SOURCE.read_text(encoding="utf-8")

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
            "SymbolWorkerApp",
            "SymbolWorkerFactory",
            "WorkerEventEmitter",
            "JsonlOutbox",
            "src.live.workers",
            "src.trader",
            "src.strategies",
        ]
        for token in forbidden:
            assert token not in source, (
                f"child_event_reader.py must not import/use {token}"
            )


# ============================================================================
# 19. no full-file read guard
# ============================================================================


class TestNoFullFileRead:
    def test_no_full_file_read_methods(self) -> None:
        source = _CHILD_EVENT_READER_SOURCE.read_text(encoding="utf-8")

        forbidden_patterns = [
            ".read_text(",
            ".read_bytes(",
            ".readlines(",
            "list(",
        ]
        for pattern in forbidden_patterns:
            assert pattern not in source, (
                f"child_event_reader.py must not use full-file read: {pattern}"
            )


# ============================================================================
# 20. cursor uses atomic json
# ============================================================================


class TestCursorUsesAtomicJson:
    def test_uses_read_json_or_none(self) -> None:
        source = _CHILD_EVENT_READER_SOURCE.read_text(encoding="utf-8")
        assert "read_json_or_none" in source, (
            "child_event_reader.py must use read_json_or_none for cursor reads"
        )

    def test_uses_write_json_atomic(self) -> None:
        source = _CHILD_EVENT_READER_SOURCE.read_text(encoding="utf-8")
        assert "write_json_atomic" in source, (
            "child_event_reader.py must use write_json_atomic for cursor writes"
        )


# ============================================================================
# Edge-case tests
# ============================================================================


class TestBlankLines:
    def test_blank_lines_are_skipped_and_cursor_advances(self, tmp_path: Path) -> None:
        outbox = tmp_path / "events.jsonl"
        cursor = tmp_path / "cursor.json"

        _write_outbox_raw(outbox, "\n" + _make_event(1, "WORKER_STARTED") + "\n\n" + _make_event(2, "WORKER_STOPPED") + "\n")

        reader = ChildEventReader(outbox_path=outbox, cursor_path=cursor)
        result = reader.read_new_events()

        assert len(result.events) == 2
        assert result.errors == []
        assert result.events[0].ts_ms == 1
        assert result.events[1].ts_ms == 2


class TestEmptyOutbox:
    def test_empty_outbox_returns_no_events(self, tmp_path: Path) -> None:
        outbox = tmp_path / "events.jsonl"
        cursor = tmp_path / "cursor.json"

        # Create an empty outbox file.
        _write_outbox_raw(outbox, "")

        reader = ChildEventReader(outbox_path=outbox, cursor_path=cursor)
        result = reader.read_new_events()

        assert result.events == []
        assert result.errors == []
        assert result.reached_eof is True


class TestMultipleErrorsInOneRead:
    def test_multiple_bad_lines_in_one_read(self, tmp_path: Path) -> None:
        outbox = tmp_path / "events.jsonl"
        cursor = tmp_path / "cursor.json"

        _write_outbox_raw(outbox, "{bad1\n" + _make_event(1, "WORKER_STARTED") + "{bad2\n")

        reader = ChildEventReader(outbox_path=outbox, cursor_path=cursor)
        result = reader.read_new_events()

        assert len(result.events) == 1
        assert len(result.errors) == 2
        assert result.errors[0].error_type == "BAD_JSON"
        assert result.errors[1].error_type == "BAD_JSON"


class TestRawPreviewTruncation:
    def test_raw_preview_is_truncated(self, tmp_path: Path) -> None:
        outbox = tmp_path / "events.jsonl"
        cursor = tmp_path / "cursor.json"

        # Write a bad line > 256 chars.
        long_bad = "x" * 300 + "\n"
        _write_outbox_raw(outbox, long_bad)

        reader = ChildEventReader(outbox_path=outbox, cursor_path=cursor)
        result = reader.read_new_events()

        assert len(result.errors) == 1
        assert result.errors[0].error_type == "BAD_JSON"
        # raw_preview must be at most 256 chars.
        assert result.errors[0].raw_preview is not None
        assert len(result.errors[0].raw_preview) <= 256


class TestChildEventFields:
    def test_child_event_has_all_fields(self, tmp_path: Path) -> None:
        outbox = tmp_path / "events.jsonl"
        cursor = tmp_path / "cursor.json"

        _write_outbox_lines(outbox, _make_event(1, "WORKER_STARTED"))

        reader = ChildEventReader(outbox_path=outbox, cursor_path=cursor)
        result = reader.read_new_events()

        assert len(result.events) == 1
        e = result.events[0]
        assert isinstance(e, ChildEvent)
        assert isinstance(e.ts_ms, int)
        assert isinstance(e.event_type, str)
        assert isinstance(e.payload, dict)
        assert e.source_path == str(outbox)
        assert e.offset_start is not None
        assert e.offset_end is not None


class TestChildEventReadErrorFields:
    def test_read_error_has_all_fields(self, tmp_path: Path) -> None:
        outbox = tmp_path / "events.jsonl"
        cursor = tmp_path / "cursor.json"

        _write_outbox_raw(outbox, "{bad\n")

        reader = ChildEventReader(outbox_path=outbox, cursor_path=cursor)
        result = reader.read_new_events()

        assert len(result.errors) == 1
        err = result.errors[0]
        assert isinstance(err, ChildEventReadError)
        assert isinstance(err.ts_ms, int)
        assert isinstance(err.error_type, str)
        assert isinstance(err.message, str)
        assert err.source_path == str(outbox)
        assert err.raw_preview is not None


class Test_now_ms:
    def test_now_ms_returns_int(self) -> None:
        now = _now_ms()
        assert isinstance(now, int)
        assert now > 0
        # Should be within 1 second of actual time.
        assert abs(now - int(time.time() * 1000)) < 2000


class TestCursorWrittenOnAdvance:
    def test_cursor_not_written_when_no_advance(self, tmp_path: Path) -> None:
        outbox = tmp_path / "events.jsonl"
        cursor = tmp_path / "cursor.json"

        # Partial line only — cursor should not be created/updated.
        _write_outbox_raw(outbox, '{"ts_ms":1,"event_type":"WORKER_STARTED","pa')

        reader = ChildEventReader(outbox_path=outbox, cursor_path=cursor)
        reader.read_new_events()

        # Cursor should not exist since we never advanced.
        assert not cursor.exists()


class TestCursorInvalidDict:
    def test_cursor_not_a_dict_resets(self, tmp_path: Path) -> None:
        outbox = tmp_path / "events.jsonl"
        cursor = tmp_path / "cursor.json"

        # Write a cursor that is an array, not a dict.
        write_json_atomic(cursor, [1, 2, 3])

        _write_outbox_lines(outbox, _make_event(1, "WORKER_STARTED"))

        reader = ChildEventReader(outbox_path=outbox, cursor_path=cursor)
        result = reader.read_new_events()

        assert len(result.events) == 1
