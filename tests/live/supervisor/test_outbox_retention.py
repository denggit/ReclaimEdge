from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.live.outbox.jsonl_outbox import JsonlOutbox
from src.live.supervisor.outbox_retention import (
    OutboxRetentionResult,
    WorkerEventOutboxRetention,
)


def _fill_outbox(
    outbox: JsonlOutbox,
    num_events: int,
    *,
    payload_size: int = 0,
) -> int:
    """Append events and return total byte size of the outbox file."""
    extra = "x" * payload_size if payload_size > 0 else ""
    for i in range(num_events):
        outbox.append("TEST_EVENT", {"idx": i, "extra": extra})
    return outbox.path.stat().st_size


def _read_cursor(cursor_path: Path) -> dict | None:
    if not cursor_path.exists():
        return None
    return json.loads(cursor_path.read_text(encoding="utf-8"))


# ============================================================================
# 1. below threshold — no rotate
# ============================================================================


class TestBelowThresholdNoRotate:
    def test_below_threshold(self, tmp_path: Path) -> None:
        outbox_path = tmp_path / "events" / "worker_events_TEST.jsonl"
        cursor_path = tmp_path / "state" / "worker_event_cursor_TEST.json"

        outbox = JsonlOutbox(outbox_path)
        _fill_outbox(outbox, 2)

        size = outbox_path.stat().st_size

        retention = WorkerEventOutboxRetention(
            outbox_path=outbox_path,
            cursor_path=cursor_path,
            max_bytes=10 * 1024 * 1024,
        )

        result = retention.rotate_if_processed(
            cursor_offset=size,
            reached_eof=True,
        )

        assert result.rotated is False
        assert result.reason == "below_threshold"
        assert outbox_path.exists()
        assert not cursor_path.exists() or _read_cursor(cursor_path) is None


# ============================================================================
# 2. not EOF — no rotate
# ============================================================================


class TestNotEofNoRotate:
    def test_not_eof(self, tmp_path: Path) -> None:
        outbox_path = tmp_path / "events" / "worker_events_TEST.jsonl"
        cursor_path = tmp_path / "state" / "worker_event_cursor_TEST.json"

        outbox = JsonlOutbox(outbox_path)
        _fill_outbox(outbox, 100)

        retention = WorkerEventOutboxRetention(
            outbox_path=outbox_path,
            cursor_path=cursor_path,
            max_bytes=100,
        )

        result = retention.rotate_if_processed(
            cursor_offset=0,
            reached_eof=False,
        )

        assert result.rotated is False
        assert result.reason == "not_eof"
        assert outbox_path.exists()


# ============================================================================
# 3. cursor before EOF — no rotate
# ============================================================================


class TestCursorBeforeEofNoRotate:
    def test_cursor_before_eof(self, tmp_path: Path) -> None:
        outbox_path = tmp_path / "events" / "worker_events_TEST.jsonl"
        cursor_path = tmp_path / "state" / "worker_event_cursor_TEST.json"

        outbox = JsonlOutbox(outbox_path)
        _fill_outbox(outbox, 100)
        size = outbox_path.stat().st_size

        retention = WorkerEventOutboxRetention(
            outbox_path=outbox_path,
            cursor_path=cursor_path,
            max_bytes=100,
        )

        result = retention.rotate_if_processed(
            cursor_offset=size - 1,
            reached_eof=True,
        )

        assert result.rotated is False
        assert result.reason == "cursor_before_eof"
        assert outbox_path.exists()


# ============================================================================
# 4. rotate when fully processed
# ============================================================================


class TestRotateWhenProcessed:
    def test_rotate(self, tmp_path: Path) -> None:
        outbox_path = tmp_path / "events" / "worker_events_TEST.jsonl"
        cursor_path = tmp_path / "state" / "worker_event_cursor_TEST.json"
        archive_dir = tmp_path / "events" / "archive"

        outbox = JsonlOutbox(outbox_path)
        _fill_outbox(outbox, 100)

        # Read original content for comparison.
        original_content = outbox_path.read_bytes()

        size = outbox_path.stat().st_size
        now_ms = int(time.time() * 1000)

        retention = WorkerEventOutboxRetention(
            outbox_path=outbox_path,
            cursor_path=cursor_path,
            max_bytes=100,
        )

        result = retention.rotate_if_processed(
            cursor_offset=size,
            reached_eof=True,
            now_ms=now_ms,
        )

        assert result.rotated is True
        assert result.reason == "rotated"
        assert result.size_bytes == size
        assert result.cursor_offset == 0
        assert result.archive_path is not None

        # Original outbox must be gone.
        assert not outbox_path.exists()

        # Archive exists with original content.
        archive_path = Path(result.archive_path)
        assert archive_path.exists()
        assert archive_path.read_bytes() == original_content

        # Cursor was reset to 0.
        cursor_data = _read_cursor(cursor_path)
        assert cursor_data is not None
        assert cursor_data["offset"] == 0
        assert cursor_data["path"] == str(outbox_path)
        assert cursor_data["updated_ts_ms"] == now_ms
        assert cursor_data["rotated_from_size_bytes"] == size
        assert cursor_data["archive_path"] == str(archive_path)


# ============================================================================
# 5. new data after read — skip rotate
# ============================================================================


class TestNewDataAfterReadSkip:
    def test_cursor_before_eof_handled(self, tmp_path: Path) -> None:
        outbox_path = tmp_path / "events" / "worker_events_TEST.jsonl"
        cursor_path = tmp_path / "state" / "worker_event_cursor_TEST.json"

        outbox = JsonlOutbox(outbox_path)
        _fill_outbox(outbox, 100)
        size = outbox_path.stat().st_size

        retention = WorkerEventOutboxRetention(
            outbox_path=outbox_path,
            cursor_path=cursor_path,
            max_bytes=100,
        )

        # cursor_offset < size → skip.
        result = retention.rotate_if_processed(
            cursor_offset=size - 10,
            reached_eof=True,
        )

        assert result.rotated is False
        assert result.reason == "cursor_before_eof"


# ============================================================================
# 6. keep_archives cleanup
# ============================================================================


class TestKeepArchivesCleanup:
    def test_keep_archives_cleanup(self, tmp_path: Path) -> None:
        outbox_path = tmp_path / "events" / "worker_events_TEST.jsonl"
        cursor_path = tmp_path / "state" / "worker_event_cursor_TEST.json"
        archive_dir = tmp_path / "events" / "archive"

        now_ms_base = int(time.time() * 1000)

        retention = WorkerEventOutboxRetention(
            outbox_path=outbox_path,
            cursor_path=cursor_path,
            archive_dir=archive_dir,
            max_bytes=100,
            keep_archives=1,
        )

        # Rotate multiple times.
        for i in range(3):
            outbox = JsonlOutbox(outbox_path)
            _fill_outbox(outbox, 50)
            size = outbox_path.stat().st_size
            result = retention.rotate_if_processed(
                cursor_offset=size,
                reached_eof=True,
                now_ms=now_ms_base + i * 1000,
            )
            assert result.rotated is True, f"rotate {i} failed: {result.reason}"

        # Only 1 archive should remain.
        archives = sorted(archive_dir.glob("worker_events_TEST.*.jsonl"))
        assert len(archives) == 1, f"Expected 1 archive, got {len(archives)}: {archives}"

    def test_archives_deleted_count(self, tmp_path: Path) -> None:
        outbox_path = tmp_path / "events" / "worker_events_TEST.jsonl"
        cursor_path = tmp_path / "state" / "worker_event_cursor_TEST.json"
        archive_dir = tmp_path / "events" / "archive"

        now_ms_base = int(time.time() * 1000)

        retention = WorkerEventOutboxRetention(
            outbox_path=outbox_path,
            cursor_path=cursor_path,
            archive_dir=archive_dir,
            max_bytes=100,
            keep_archives=1,
        )

        # First rotate creates 1 archive (no cleanup yet).
        outbox = JsonlOutbox(outbox_path)
        _fill_outbox(outbox, 50)
        size = outbox_path.stat().st_size
        result1 = retention.rotate_if_processed(
            cursor_offset=size,
            reached_eof=True,
            now_ms=now_ms_base,
        )
        assert result1.archives_deleted == 0

        # Second rotate deletes 1 old archive.
        outbox2 = JsonlOutbox(outbox_path)
        _fill_outbox(outbox2, 50)
        size2 = outbox_path.stat().st_size
        result2 = retention.rotate_if_processed(
            cursor_offset=size2,
            reached_eof=True,
            now_ms=now_ms_base + 1000,
        )
        assert result2.rotated is True
        # The previous archive should now be deleted.
        assert result2.archives_deleted >= 1


# ============================================================================
# 7. invalid params
# ============================================================================


class TestInvalidParams:
    def test_max_bytes_true_raises(self) -> None:
        with pytest.raises(ValueError):
            WorkerEventOutboxRetention(
                outbox_path=Path("outbox.jsonl"),
                cursor_path=Path("cursor.json"),
                max_bytes=True,
            )

    def test_max_bytes_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            WorkerEventOutboxRetention(
                outbox_path=Path("outbox.jsonl"),
                cursor_path=Path("cursor.json"),
                max_bytes=0,
            )

    def test_max_bytes_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            WorkerEventOutboxRetention(
                outbox_path=Path("outbox.jsonl"),
                cursor_path=Path("cursor.json"),
                max_bytes=-1,
            )

    def test_keep_archives_true_raises(self) -> None:
        with pytest.raises(ValueError):
            WorkerEventOutboxRetention(
                outbox_path=Path("outbox.jsonl"),
                cursor_path=Path("cursor.json"),
                keep_archives=True,
            )

    def test_keep_archives_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            WorkerEventOutboxRetention(
                outbox_path=Path("outbox.jsonl"),
                cursor_path=Path("cursor.json"),
                keep_archives=-1,
            )

    def test_cursor_offset_negative_raises(self, tmp_path: Path) -> None:
        outbox_path = tmp_path / "events" / "worker_events_TEST.jsonl"
        cursor_path = tmp_path / "state" / "worker_event_cursor_TEST.json"

        retention = WorkerEventOutboxRetention(
            outbox_path=outbox_path,
            cursor_path=cursor_path,
        )

        with pytest.raises(ValueError):
            retention.rotate_if_processed(
                cursor_offset=-1,
                reached_eof=True,
            )

    def test_cursor_offset_bool_raises(self, tmp_path: Path) -> None:
        outbox_path = tmp_path / "events" / "worker_events_TEST.jsonl"
        cursor_path = tmp_path / "state" / "worker_event_cursor_TEST.json"

        retention = WorkerEventOutboxRetention(
            outbox_path=outbox_path,
            cursor_path=cursor_path,
        )

        with pytest.raises(ValueError):
            retention.rotate_if_processed(
                cursor_offset=True,
                reached_eof=True,
            )

    def test_reached_eof_not_bool_raises(self, tmp_path: Path) -> None:
        outbox_path = tmp_path / "events" / "worker_events_TEST.jsonl"
        cursor_path = tmp_path / "state" / "worker_event_cursor_TEST.json"

        retention = WorkerEventOutboxRetention(
            outbox_path=outbox_path,
            cursor_path=cursor_path,
        )

        with pytest.raises(ValueError):
            retention.rotate_if_processed(
                cursor_offset=0,
                reached_eof=1,
            )


# ============================================================================
# 8. retention failure returns reason, does not raise
# ============================================================================


class TestRetentionFailureNoRaise:
    def test_rename_error_returns_reason(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        outbox_path = tmp_path / "events" / "worker_events_TEST.jsonl"
        cursor_path = tmp_path / "state" / "worker_event_cursor_TEST.json"

        outbox = JsonlOutbox(outbox_path)
        _fill_outbox(outbox, 100)
        size = outbox_path.stat().st_size

        retention = WorkerEventOutboxRetention(
            outbox_path=outbox_path,
            cursor_path=cursor_path,
            max_bytes=100,
        )

        def _fail_rename(*args, **kwargs):
            raise OSError("simulated rename failure")

        monkeypatch.setattr(Path, "rename", _fail_rename)

        result = retention.rotate_if_processed(
            cursor_offset=size,
            reached_eof=True,
        )

        # Must not raise.
        assert result.rotated is False
        assert result.reason.startswith("error:")
        assert "OSError" in result.reason


# ============================================================================
# 9. outbox missing
# ============================================================================


class TestOutboxMissing:
    def test_outbox_missing(self, tmp_path: Path) -> None:
        outbox_path = tmp_path / "events" / "worker_events_MISSING.jsonl"
        cursor_path = tmp_path / "state" / "worker_event_cursor_TEST.json"

        retention = WorkerEventOutboxRetention(
            outbox_path=outbox_path,
            cursor_path=cursor_path,
        )

        result = retention.rotate_if_processed(
            cursor_offset=0,
            reached_eof=True,
        )

        assert result.rotated is False
        assert result.reason == "outbox_missing"


# ============================================================================
# 10. rotate creates archive_dir if missing
# ============================================================================


class TestCreatesArchiveDir:
    def test_creates_archive_dir(self, tmp_path: Path) -> None:
        outbox_path = tmp_path / "events" / "worker_events_TEST.jsonl"
        cursor_path = tmp_path / "state" / "worker_event_cursor_TEST.json"
        archive_dir = tmp_path / "nested" / "archive"

        assert not archive_dir.exists()

        outbox = JsonlOutbox(outbox_path)
        _fill_outbox(outbox, 100)
        size = outbox_path.stat().st_size

        retention = WorkerEventOutboxRetention(
            outbox_path=outbox_path,
            cursor_path=cursor_path,
            archive_dir=archive_dir,
            max_bytes=100,
        )

        result = retention.rotate_if_processed(
            cursor_offset=size,
            reached_eof=True,
        )

        assert result.rotated is True
        assert archive_dir.exists()


# ============================================================================
# 11. keep_archives=0
# ============================================================================


class TestKeepArchivesZero:
    def test_keep_archives_zero_deletes_archive(self, tmp_path: Path) -> None:
        outbox_path = tmp_path / "events" / "worker_events_TEST.jsonl"
        cursor_path = tmp_path / "state" / "worker_event_cursor_TEST.json"
        archive_dir = tmp_path / "events" / "archive"

        outbox = JsonlOutbox(outbox_path)
        _fill_outbox(outbox, 100)
        size = outbox_path.stat().st_size

        retention = WorkerEventOutboxRetention(
            outbox_path=outbox_path,
            cursor_path=cursor_path,
            archive_dir=archive_dir,
            max_bytes=100,
            keep_archives=0,
        )

        result = retention.rotate_if_processed(
            cursor_offset=size,
            reached_eof=True,
        )

        assert result.rotated is True
        # After rotation, the archive should be deleted since keep_archives=0.
        archives = sorted(archive_dir.glob("worker_events_TEST.*.jsonl"))
        assert len(archives) == 0


# ============================================================================
# source guard
# ============================================================================


class TestOutboxRetentionSourceGuard:
    def test_no_forbidden_imports(self) -> None:
        source_path = (
            Path(__file__).parents[3]
            / "src"
            / "live"
            / "supervisor"
            / "outbox_retention.py"
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
            "src.live.workers",
            "src.trader",
            "src.strategies",
            "src.live.symbol_worker_app",
            "SymbolWorkerApp",
        ]
        for token in forbidden:
            assert token not in source, (
                f"Forbidden import/usage '{token}' found in outbox_retention.py"
            )

    def test_contains_required_tokens(self) -> None:
        source_path = (
            Path(__file__).parents[3]
            / "src"
            / "live"
            / "supervisor"
            / "outbox_retention.py"
        )
        source = source_path.read_text()

        required = [
            "WorkerEventOutboxRetention",
            "OutboxRetentionResult",
            "rotate_if_processed",
            "fcntl.flock",
            "write_json_atomic",
            "cursor_offset",
            "reached_eof",
            "max_bytes",
            "keep_archives",
        ]
        for token in required:
            assert token in source, (
                f"Required token '{token}' not found in outbox_retention.py"
            )
