from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.reporting.trade_journal import JournalEvent, LiveTradeJournal


class LiveTradeJournalRecoveryTest(unittest.TestCase):
    def make_journal(self) -> LiveTradeJournal:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        return LiveTradeJournal(Path(tmpdir.name) / "live_trade_events.jsonl")

    def append_event(self, journal: LiveTradeJournal, event_type: str, position_id: str, payload: dict) -> None:
        journal.append_event(
            JournalEvent(
                event_id=f"{position_id}-{event_type}",
                event_type=event_type,
                ts_iso=datetime(2026, 6, 4, 1, 0, tzinfo=timezone.utc).isoformat(),
                position_id=position_id,
                payload=payload,
            )
        )

    def test_startup_position_id_reuses_matching_unclosed_position(self) -> None:
        journal = self.make_journal()
        position_id = "ETH-USDT-SWAP:LONG:old-open"
        journal.append_event(
            JournalEvent(
                event_id="entry-1",
                event_type="ENTRY",
                ts_iso=datetime(2026, 6, 4, 1, 0, tzinfo=timezone.utc).isoformat(),
                position_id=position_id,
                payload={"symbol": "ETH-USDT-SWAP", "side": "LONG", "cash_before_position": 100.0},
            )
        )

        match = journal.find_latest_unclosed_position("ETH-USDT-SWAP", "LONG")

        self.assertIsNotNone(match)
        self.assertEqual(match.position_id, position_id)
        self.assertEqual(match.cash_before_position, 100.0)
        self.assertEqual(journal.new_position_id("ETH-USDT-SWAP", "LONG"), position_id)

    def test_normal_trade_position_id_still_creates_new_id_with_ts_ms(self) -> None:
        journal = self.make_journal()
        position_id = "ETH-USDT-SWAP:LONG:old-open"
        journal.append_event(
            JournalEvent(
                event_id="entry-1",
                event_type="ENTRY",
                ts_iso=datetime(2026, 6, 4, 1, 0, tzinfo=timezone.utc).isoformat(),
                position_id=position_id,
                payload={"symbol": "ETH-USDT-SWAP", "side": "LONG", "cash_before_position": 100.0},
            )
        )

        new_id = journal.new_position_id("ETH-USDT-SWAP", "LONG", ts_ms=123456)

        self.assertNotEqual(new_id, position_id)
        self.assertTrue(new_id.startswith("ETH-USDT-SWAP:LONG:123456:"))

    def test_closed_position_is_not_reused(self) -> None:
        journal = self.make_journal()
        position_id = "ETH-USDT-SWAP:LONG:closed"
        self.append_event(journal, "ENTRY", position_id,
                          {"symbol": "ETH-USDT-SWAP", "side": "LONG", "cash_before_position": 100.0})
        self.append_event(journal, "FLAT", position_id,
                          {"symbol": "ETH-USDT-SWAP", "side": "LONG", "cash_after": 101.0})

        match = journal.find_latest_unclosed_position("ETH-USDT-SWAP", "LONG")

        self.assertIsNone(match)


if __name__ == "__main__":
    unittest.main()
