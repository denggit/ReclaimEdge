from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.live.outbox.atomic_json import read_json_or_none, write_json_atomic

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChildEvent:
    """A single validated event read from a child worker JSONL outbox."""

    ts_ms: int
    event_type: str
    payload: dict[str, Any]
    source_path: str
    line_no: int | None = None
    offset_start: int | None = None
    offset_end: int | None = None


@dataclass(frozen=True)
class ChildEventReadError:
    """An error encountered while reading a child worker JSONL outbox.

    These are returned as part of :class:`ChildEventReadResult` — never
    raised.  The supervisor can decide to alert on them later (E05+).
    """

    ts_ms: int
    error_type: str
    message: str
    source_path: str
    offset_start: int | None = None
    offset_end: int | None = None
    raw_preview: str | None = None


@dataclass(frozen=True)
class ChildEventReadResult:
    """Result of a single :meth:`ChildEventReader.read_new_events` call."""

    events: list[ChildEvent] = field(default_factory=list)
    errors: list[ChildEventReadError] = field(default_factory=list)
    cursor_offset: int = 0
    reached_eof: bool = False
    truncated_or_rotated: bool = False
    bytes_read: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_ms() -> int:
    return int(time.time() * 1000)


def _make_raw_preview(raw: bytes, *, max_chars: int = 256) -> str:
    """Return a safe string preview of raw bytes, decoded with replacement."""
    text = raw.decode("utf-8", errors="replace")
    return text[:max_chars]


# ---------------------------------------------------------------------------
# ChildEventReader
# ---------------------------------------------------------------------------


class ChildEventReader:
    """Incrementally reads child-worker lifecycle events from a JSONL outbox.

    Uses a cursor file to remember the last-read byte offset so repeated
    calls to :meth:`read_new_events` only return events appended since the
    previous call.

    This is a **supervisor control-plane** tool.  It must never be used
    inside the tick / trading path.

    Parameters
    ----------
    outbox_path : Path
        Path to the JSONL outbox written by a child worker.
    cursor_path : Path
        Path to the cursor state file (atomic JSON read/write).
    max_bytes_per_read : int
        Maximum bytes to read per call (default 1 MiB).
    max_line_bytes : int
        Maximum bytes per single JSONL line (default 64 KiB).
    """

    def __init__(
        self,
        *,
        outbox_path: Path,
        cursor_path: Path,
        max_bytes_per_read: int = 1024 * 1024,
        max_line_bytes: int = 64 * 1024,
    ) -> None:
        # Strict type checks — bool is an int subclass, so isinstance(True, int)
        # would silently pass.  A negative or zero value would cause fh.read(-1)
        # which reads the entire file → memory risk.
        if type(max_bytes_per_read) is not int or max_bytes_per_read <= 0:
            raise ValueError(
                f"max_bytes_per_read must be a positive int, "
                f"got {type(max_bytes_per_read).__name__}={max_bytes_per_read!r}"
            )
        if type(max_line_bytes) is not int or max_line_bytes <= 0:
            raise ValueError(
                f"max_line_bytes must be a positive int, "
                f"got {type(max_line_bytes).__name__}={max_line_bytes!r}"
            )

        self._outbox_path = outbox_path
        self._cursor_path = cursor_path
        self._max_bytes_per_read = max_bytes_per_read
        self._max_line_bytes = max_line_bytes

    # ------------------------------------------------------------------
    # read_new_events
    # ------------------------------------------------------------------

    def read_new_events(self) -> ChildEventReadResult:
        """Read events appended since the last call.

        Returns
        -------
        ChildEventReadResult
        """
        outbox = self._outbox_path

        # -- outbox missing -------------------------------------------------
        if not outbox.exists():
            return ChildEventReadResult(
                events=[],
                errors=[],
                cursor_offset=0,
                reached_eof=True,
                truncated_or_rotated=False,
                bytes_read=0,
            )

        # -- load cursor ----------------------------------------------------
        cursor_data = read_json_or_none(self._cursor_path)
        offset = self._resolve_offset(cursor_data)
        outbox_size = outbox.stat().st_size

        truncated_or_rotated = False
        if offset > outbox_size:
            # File was truncated or rotated — start from beginning.
            offset = 0
            truncated_or_rotated = True

        # -- read chunk -----------------------------------------------------
        events: list[ChildEvent] = []
        errors: list[ChildEventReadError] = []

        with outbox.open("rb") as fh:
            fh.seek(offset)
            chunk = fh.read(self._max_bytes_per_read)

        bytes_read = len(chunk)

        if bytes_read == 0:
            # If the file was truncated/rotated, we must persist the reset
            # offset=0 so a subsequent reader instance doesn't pick up a
            # stale cursor pointing past EOF.
            if truncated_or_rotated:
                self._write_cursor(0)
            return ChildEventReadResult(
                events=[],
                errors=[],
                cursor_offset=offset,
                reached_eof=True,
                truncated_or_rotated=truncated_or_rotated,
                bytes_read=0,
            )

        # -- split into complete lines + trailing ---------------------------
        lines_data, trailing, has_newline_at_end = _split_chunk(
            chunk, offset, self._max_bytes_per_read
        )

        # Process complete lines.
        for line_bytes, line_offset_start in lines_data:
            next_offset = line_offset_start + len(line_bytes)
            # Blank line: advance cursor silently.
            if line_bytes.strip() == b"":
                offset = next_offset
                continue

            # Check per-line length limit.
            if len(line_bytes) > self._max_line_bytes:
                errors.append(
                    ChildEventReadError(
                        ts_ms=_now_ms(),
                        error_type="LINE_TOO_LONG",
                        message=(
                            f"Line at offset {line_offset_start} is "
                            f"{len(line_bytes)} bytes (max {self._max_line_bytes})"
                        ),
                        source_path=str(outbox),
                        offset_start=line_offset_start,
                        offset_end=next_offset,
                        raw_preview=_make_raw_preview(line_bytes),
                    )
                )
                offset = next_offset
                continue

            # Parse + validate.
            event_or_error = _parse_line(
                line_bytes, line_offset_start, next_offset, str(outbox)
            )
            if isinstance(event_or_error, ChildEvent):
                events.append(event_or_error)
            else:
                errors.append(event_or_error)

            offset = next_offset

        # Handle trailing data.
        if trailing:
            # Trailing data without newline.
            # If the read didn't fill max_bytes_per_read, it's an
            # incomplete partial line — do NOT advance cursor past it.
            # If the read DID fill max_bytes_per_read and there's no
            # newline at all, treat as LINE_TOO_LONG.
            if not has_newline_at_end and bytes_read >= self._max_bytes_per_read:
                trailing_start = offset
                trailing_end = trailing_start + len(trailing)
                errors.append(
                    ChildEventReadError(
                        ts_ms=_now_ms(),
                        error_type="LINE_TOO_LONG",
                        message=(
                            f"No newline in {bytes_read}-byte chunk "
                            f"starting at offset {trailing_start}; "
                            f"advancing cursor to skip bad data"
                        ),
                        source_path=str(outbox),
                        offset_start=trailing_start,
                        offset_end=trailing_end,
                        raw_preview=_make_raw_preview(trailing),
                    )
                )
                offset = trailing_end
            # else: incomplete trailing line — do NOT advance cursor.

        # -- write cursor if we advanced -----------------------------------
        if offset != self._resolve_offset(cursor_data) or truncated_or_rotated:
            self._write_cursor(offset)

        return ChildEventReadResult(
            events=events,
            errors=errors,
            cursor_offset=offset,
            reached_eof=(offset >= outbox_size),
            truncated_or_rotated=truncated_or_rotated,
            bytes_read=bytes_read,
        )

    # ------------------------------------------------------------------
    # Cursor helpers
    # ------------------------------------------------------------------

    def _resolve_offset(self, cursor_data: Any) -> int:
        """Resolve cursor data to a valid byte offset.

        Returns 0 if the cursor is missing, has a path mismatch, or has an
        invalid offset.
        """
        if cursor_data is None:
            return 0

        if not isinstance(cursor_data, dict):
            return 0

        # Path mismatch → reset.
        cursor_path_val = cursor_data.get("path")
        if cursor_path_val != str(self._outbox_path):
            return 0

        # Offset validation.
        offset = cursor_data.get("offset")
        if not isinstance(offset, int) or offset < 0:
            return 0

        return offset

    def _write_cursor(self, offset: int) -> None:
        """Atomically write the cursor state."""
        write_json_atomic(
            self._cursor_path,
            {
                "path": str(self._outbox_path),
                "offset": offset,
                "updated_ts_ms": _now_ms(),
            },
            indent=2,
            sort_keys=True,
        )


# ---------------------------------------------------------------------------
# Internal: chunk splitting
# ---------------------------------------------------------------------------


def _split_chunk(
    chunk: bytes,
    chunk_offset: int,
    max_bytes_per_read: int,
) -> tuple[list[tuple[bytes, int]], bytes, bool]:
    """Split a raw chunk into (line_bytes, absolute_offset) pairs plus trailing.

    Returns
    -------
    (lines, trailing, has_newline_at_end)
        ``lines`` is a list of ``(line_bytes, absolute_offset)``.
        ``trailing`` is the bytes after the last ``\\n`` (empty if chunk
        ends with ``\\n``).
        ``has_newline_at_end`` is True if any ``\\n`` was found in the chunk.
    """
    lines: list[tuple[bytes, int]] = []
    pos = 0
    chunk_len = len(chunk)

    while True:
        nl_idx = chunk.find(b"\n", pos)
        if nl_idx == -1:
            break
        line_end = nl_idx + 1  # include newline
        line_bytes = chunk[pos:line_end]
        lines.append((line_bytes, chunk_offset + pos))
        pos = line_end

    trailing = chunk[pos:]
    has_newline_at_end = len(lines) > 0
    return lines, trailing, has_newline_at_end


# ---------------------------------------------------------------------------
# Internal: line parsing + validation
# ---------------------------------------------------------------------------


def _parse_line(
    line_bytes: bytes,
    offset_start: int,
    offset_end: int,
    source_path: str,
) -> ChildEvent | ChildEventReadError:
    """Parse a single complete line (including trailing newline) from the outbox.

    Returns a :class:`ChildEvent` on success, or a
    :class:`ChildEventReadError` on any failure.
    """
    # Strip trailing newline for JSON parsing.
    stripped = line_bytes.rstrip(b"\r\n")

    # -- JSON parse ---------------------------------------------------------
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return ChildEventReadError(
            ts_ms=_now_ms(),
            error_type="BAD_JSON",
            message=f"JSON decode error: {exc}",
            source_path=source_path,
            offset_start=offset_start,
            offset_end=offset_end,
            raw_preview=_make_raw_preview(stripped),
        )

    # -- Must be a dict -----------------------------------------------------
    if not isinstance(obj, dict):
        return ChildEventReadError(
            ts_ms=_now_ms(),
            error_type="INVALID_EVENT_OBJECT",
            message=f"Expected JSON object, got {type(obj).__name__}",
            source_path=source_path,
            offset_start=offset_start,
            offset_end=offset_end,
            raw_preview=_make_raw_preview(stripped),
        )

    # -- ts_ms --------------------------------------------------------------
    ts_ms = obj.get("ts_ms")
    if not isinstance(ts_ms, int):
        return ChildEventReadError(
            ts_ms=_now_ms(),
            error_type="INVALID_TS_MS",
            message=f"ts_ms must be int, got {type(ts_ms).__name__}: {ts_ms!r}",
            source_path=source_path,
            offset_start=offset_start,
            offset_end=offset_end,
            raw_preview=_make_raw_preview(stripped),
        )

    # -- event_type ---------------------------------------------------------
    event_type = obj.get("event_type")
    if not isinstance(event_type, str) or not event_type.strip():
        return ChildEventReadError(
            ts_ms=_now_ms(),
            error_type="INVALID_EVENT_TYPE",
            message=f"event_type must be non-empty str, got {type(event_type).__name__}: {event_type!r}",
            source_path=source_path,
            offset_start=offset_start,
            offset_end=offset_end,
            raw_preview=_make_raw_preview(stripped),
        )
    event_type = event_type.strip()

    # -- payload ------------------------------------------------------------
    payload = obj.get("payload")
    if not isinstance(payload, dict):
        return ChildEventReadError(
            ts_ms=_now_ms(),
            error_type="INVALID_PAYLOAD",
            message=f"payload must be dict, got {type(payload).__name__}: {payload!r}",
            source_path=source_path,
            offset_start=offset_start,
            offset_end=offset_end,
            raw_preview=_make_raw_preview(stripped),
        )

    return ChildEvent(
        ts_ms=ts_ms,
        event_type=event_type,
        payload=payload,
        source_path=source_path,
        offset_start=offset_start,
        offset_end=offset_end,
    )
