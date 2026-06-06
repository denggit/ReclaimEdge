from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from src.reporting.journal_compactor import compact_after_weekly_summary
from src.reporting.trade_journal import JournalEvent, LiveTradeJournal


def event(event_id: str, event_type: str, position_id: str | None, ts: str,
          payload: dict | None = None) -> JournalEvent:
    return JournalEvent(
        event_id=event_id,
        event_type=event_type,
        ts_iso=ts,
        position_id=position_id,
        payload=payload or {},
    )


class JournalCompactorTest(unittest.TestCase):
    def test_compactor_archives_closed_positions_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = LiveTradeJournal(Path(tmp) / "live_trade_events.jsonl")
            old = "2026-01-01T00:00:00+00:00"
            newer = "2026-01-01T01:00:00+00:00"
            for item in [
                event("closed-entry", "ENTRY", "pos_closed", old, {"cash_before_position": 100.0}),
                event("closed-flat", "FLAT", "pos_closed", newer, {"realized_pnl_usdt_est": 10.0, "cash_after": 110.0}),
                event("open-entry", "ENTRY", "pos_open", old),
                event("incomplete-entry", "ENTRY", "pos_incomplete", old),
                event("cash-transfer", "CASH_TRANSFER", None, old, {"amount": 50.0}),
                event("error", "ERROR", None, old, {"error": "x"}),
            ]:
                journal.append_event(item)

            result = compact_after_weekly_summary(
                journal,
                datetime(2026, 1, 2, tzinfo=timezone.utc),
                current_position_id="pos_open",
            )

            self.assertEqual(result.archived_event_count, 2)
            self.assertIsNotNone(result.archive_path)
            with gzip.open(result.archive_path, "rt", encoding="utf-8") as f:  # type: ignore[arg-type]
                archived = [json.loads(line) for line in f if line.strip()]
            self.assertEqual({item["event_id"] for item in archived}, {"closed-entry", "closed-flat"})

            live_events = journal.load_events()
            live_ids = {item.event_id for item in live_events}
            self.assertNotIn("closed-entry", live_ids)
            self.assertNotIn("closed-flat", live_ids)
            self.assertIn("open-entry", live_ids)
            self.assertIn("incomplete-entry", live_ids)
            self.assertIn("cash-transfer", live_ids)
            self.assertIn("error", live_ids)
            self.assertIn("JOURNAL_COMPACTED", [item.event_type for item in live_events])

            summary_events = journal.load_summary_events()
            self.assertEqual(len(summary_events), 1)
            self.assertEqual(summary_events[0].event_type, "SUMMARY_SNAPSHOT")
            self.assertEqual(summary_events[0].payload["closed_count"], 1)
            self.assertEqual(summary_events[0].payload["known_closed_pnl"], 10.0)

    def test_compactor_atomic_failure_does_not_corrupt_journal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = LiveTradeJournal(Path(tmp) / "live_trade_events.jsonl")
            for item in [
                event("closed-entry", "ENTRY", "pos_closed", "2026-01-01T00:00:00+00:00"),
                event("closed-flat", "FLAT", "pos_closed", "2026-01-01T01:00:00+00:00", {"realized_pnl_usdt_est": 1.0}),
            ]:
                journal.append_event(item)
            original = journal.path.read_text(encoding="utf-8")

            with patch("src.reporting.journal_compactor.gzip.open", side_effect=OSError("archive failed")):
                with self.assertRaises(OSError):
                    compact_after_weekly_summary(
                        journal,
                        datetime(2026, 1, 2, tzinfo=timezone.utc),
                        current_position_id=None,
                    )

            self.assertEqual(journal.path.read_text(encoding="utf-8"), original)
            self.assertFalse(journal.path.with_suffix(journal.path.suffix + ".tmp").exists())
            self.assertEqual([item.event_id for item in journal.load_events()], ["closed-entry", "closed-flat"])


if __name__ == "__main__":
    unittest.main()
