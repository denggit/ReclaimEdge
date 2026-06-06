from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.reporting.trade_journal import LiveTradeJournal


class AccountCashLedgerTest(unittest.TestCase):
    def test_cash_transfer_event_recording(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = LiveTradeJournal(Path(tmp) / "live_trade_events.jsonl")
            journal.record_cash_baseline(source="startup", cash=100.0, equity=101.0, note="boot")
            journal.record_cash_transfer(
                direction="DEPOSIT",
                amount=50.0,
                cash_before=100.0,
                cash_after=150.0,
                equity_before=101.0,
                equity_after=151.0,
                reason="safe_flat_account_sync",
            )
            journal.record_account_cash_drift(
                amount=-2.0,
                cash_before=150.0,
                cash_after=148.0,
                equity_before=151.0,
                equity_after=149.0,
                reason="unsafe_state:has_position",
            )

            events = journal.load_events()

        self.assertEqual([event.event_type for event in events],
                         ["CASH_BASELINE", "CASH_TRANSFER", "ACCOUNT_CASH_DRIFT"])
        self.assertEqual(events[0].payload["cash"], 100.0)
        self.assertEqual(events[1].payload["direction"], "DEPOSIT")
        self.assertEqual(events[1].payload["amount"], 50.0)
        self.assertEqual(events[2].payload["reason"], "unsafe_state:has_position")

    def test_startup_baseline_is_not_duplicated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = LiveTradeJournal(Path(tmp) / "live_trade_events.jsonl")
            journal.record_cash_baseline(source="startup", cash=100.0, equity=100.0)
            journal.record_cash_baseline(source="startup", cash=200.0, equity=200.0)
            journal.record_cash_baseline(source="manual", cash=300.0, equity=300.0)
            events = journal.load_events()

        self.assertEqual([event.payload["cash"] for event in events], [100.0, 300.0])


if __name__ == "__main__":
    unittest.main()
