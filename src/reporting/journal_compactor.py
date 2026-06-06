from __future__ import annotations

import gzip
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.reporting.daily_trade_reporter import _to_float
from src.reporting.trade_journal import JournalEvent, LiveTradeJournal, group_position_events


@dataclass(frozen=True)
class JournalCompactionResult:
    archived_event_count: int
    retained_event_count: int
    archive_path: str | None
    summary_path: str | None


def compact_after_weekly_summary(
        journal: LiveTradeJournal,
        snapshot_until: datetime,
        current_position_id: str | None,
) -> JournalCompactionResult:
    """Archive closed-position events after a successful weekly summary.

    Accounting safety rule:
    archived events may only leave live_trade_events.jsonl after a matching
    SUMMARY_SNAPSHOT is durably written. If summary writing fails after the main
    journal was compacted, restore the original journal before re-raising.
    """
    events = journal.load_events()
    archive_ids = _closed_position_event_ids_to_archive(events, snapshot_until, current_position_id)
    if not archive_ids:
        return JournalCompactionResult(
            archived_event_count=0,
            retained_event_count=len(events),
            archive_path=None,
            summary_path=None,
        )

    archived_events = [event for event in events if event.event_id in archive_ids]
    retained_events = [event for event in events if event.event_id not in archive_ids]
    archive_path = _next_archive_path(journal.path.parent / "archive", snapshot_until)
    summary_payload = _build_summary_payload(archived_events, snapshot_until, str(archive_path))

    archive_tmp_path = archive_path.with_suffix(archive_path.suffix + ".tmp")
    journal_tmp_path = journal.path.with_suffix(journal.path.suffix + ".tmp")
    journal_was_replaced = False
    try:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(archive_tmp_path, "wt", encoding="utf-8") as f:
            for event in archived_events:
                f.write(json.dumps(asdict(event), ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(archive_tmp_path, archive_path)

        _write_events_atomic(journal.path, retained_events, tmp_path=journal_tmp_path)
        journal_was_replaced = True

        try:
            journal.record_summary_snapshot(summary_payload)
        except Exception:
            # The archive may already exist, but the summary is the accounting
            # link that keeps future overall reports correct. If it cannot be
            # written, restore the full original journal so no history is lost.
            _write_events_atomic(journal.path, events, tmp_path=journal_tmp_path)
            raise
    except Exception:
        if archive_tmp_path.exists():
            archive_tmp_path.unlink()
        if journal_tmp_path.exists():
            journal_tmp_path.unlink()
        if journal_was_replaced and journal.load_events() != events:
            try:
                _write_events_atomic(journal.path, events, tmp_path=journal_tmp_path)
            except Exception:
                # Preserve the original exception; caller must inspect files if
                # this extremely rare rollback also fails.
                pass
        raise

    journal.record_journal_compacted(
        archived_event_count=len(archived_events),
        retained_event_count=len(retained_events),
        archive_path=str(archive_path),
        summary_path=str(journal.summary_path),
        snapshot_until=snapshot_until.isoformat(),
    )
    return JournalCompactionResult(
        archived_event_count=len(archived_events),
        retained_event_count=len(retained_events),
        archive_path=str(archive_path),
        summary_path=str(journal.summary_path),
    )


def _write_events_atomic(path: Path, events: list[JournalEvent], *, tmp_path: Path) -> None:
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(asdict(event), ensure_ascii=False, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _closed_position_event_ids_to_archive(
        events: list[JournalEvent],
        snapshot_until: datetime,
        current_position_id: str | None,
) -> set[str]:
    grouped = group_position_events(event for event in events if event.position_id is not None)
    archive_ids: set[str] = set()
    for position_id, items in grouped.items():
        if position_id == current_position_id:
            continue
        if not any(event.event_type == "FLAT" for event in items):
            continue
        try:
            last_ts = datetime.fromisoformat(items[-1].ts_iso)
        except Exception:
            continue
        if last_ts >= snapshot_until:
            continue
        archive_ids.update(event.event_id for event in items)
    return archive_ids


def _build_summary_payload(events: list[JournalEvent], snapshot_until: datetime, archive_path: str) -> dict[str, Any]:
    grouped = group_position_events(events)
    closed_count = 0
    win_count = 0
    loss_count = 0
    breakeven_count = 0
    known_closed_pnl = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    best_win: float | None = None
    worst_loss: float | None = None

    for position_id, items in grouped.items():
        if position_id == "UNKNOWN":
            continue
        flat_events = [event for event in items if event.event_type == "FLAT"]
        if not flat_events:
            continue
        closed_count += 1
        pnl = _to_float(flat_events[-1].payload.get("realized_pnl_usdt_est"))
        if pnl is None:
            continue
        known_closed_pnl += pnl
        if pnl > 0:
            win_count += 1
            gross_profit += pnl
            best_win = pnl if best_win is None else max(best_win, pnl)
        elif pnl < 0:
            loss_count += 1
            gross_loss += abs(pnl)
            worst_loss = pnl if worst_loss is None else min(worst_loss, pnl)
        else:
            breakeven_count += 1

    return {
        "archived_position_count": len([key for key in grouped if key != "UNKNOWN"]),
        "archived_event_count": len(events),
        "closed_count": closed_count,
        "win_count": win_count,
        "loss_count": loss_count,
        "breakeven_count": breakeven_count,
        "known_closed_pnl": known_closed_pnl,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "best_win": best_win,
        "worst_loss": worst_loss,
        "snapshot_until": snapshot_until.isoformat(),
        "archive_path": archive_path,
    }


def _next_archive_path(archive_dir: Path, snapshot_until: datetime) -> Path:
    year, week, _ = snapshot_until.isocalendar()
    base = archive_dir / f"live_trade_events_{year}-{week:02d}.jsonl.gz"
    if not base.exists():
        return base
    index = 1
    while True:
        candidate = archive_dir / f"live_trade_events_{year}-{week:02d}_{index}.jsonl.gz"
        if not candidate.exists():
            return candidate
        index += 1
