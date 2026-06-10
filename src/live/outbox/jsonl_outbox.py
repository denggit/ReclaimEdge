from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JsonlOutboxEvent:
    ts_ms: int
    event_type: str
    payload: dict[str, Any]

    def to_json_object(self) -> dict[str, Any]:
        return {
            "ts_ms": self.ts_ms,
            "event_type": self.event_type,
            "payload": self.payload,
        }


def _lock_path_for(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".lock")


class JsonlOutbox:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        ts_ms: int | None = None,
    ) -> JsonlOutboxEvent:
        if not event_type:
            raise ValueError("event_type must be non-empty")
        if ts_ms is None:
            ts_ms = int(time.time() * 1000)
        event = JsonlOutboxEvent(
            ts_ms=int(ts_ms),
            event_type=str(event_type),
            payload=dict(payload or {}),
        )
        self.append_event(event)
        return event

    def append_event(self, event: JsonlOutboxEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event.to_json_object(), ensure_ascii=False, sort_keys=True)
        lock_path = _lock_path_for(self.path)
        with lock_path.open("a", encoding="utf-8") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
                    fh.write("\n")
                    fh.flush()
                    os.fsync(fh.fileno())
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)

    def read_events(self) -> list[JsonlOutboxEvent]:
        if not self.path.exists():
            return []
        events: list[JsonlOutboxEvent] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                obj = json.loads(stripped)
                events.append(_event_from_json_object(obj, line_no=line_no))
        return events


def _event_from_json_object(obj: dict[str, Any], *, line_no: int) -> JsonlOutboxEvent:
    if not isinstance(obj, dict):
        raise ValueError(f"jsonl event line {line_no} must be an object")
    event_type = obj.get("event_type")
    if not event_type:
        raise ValueError(f"jsonl event line {line_no} missing event_type")
    payload = obj.get("payload", {})
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError(f"jsonl event line {line_no} payload must be an object")
    return JsonlOutboxEvent(
        ts_ms=int(obj.get("ts_ms", 0)),
        event_type=str(event_type),
        payload=payload,
    )
