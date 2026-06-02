from __future__ import annotations

import tempfile
import unittest
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from src.reporting.daily_trade_reporter import DailyTradeReporter, ReportRuntimeContext
from src.reporting.journal_compactor import compact_after_weekly_summary
from src.reporting.trade_journal import JournalEvent, LiveTradeJournal


def event(event_id: str, event_type: str, position_id: str | None, ts: str, payload: dict | None = None) -> JournalEvent:
    return JournalEvent(
        event_id=event_id,
        event_type=event_type,
        ts_iso=ts,
        position_id=position_id,
        payload=payload or {},
    )


class FakeEmailSender:
    async def send_email_async(self, *args, **kwargs) -> bool:  # type: ignore[no-untyped-def]
        return True


class OverallArchivedSummaryTest(unittest.TestCase):
    def test_overall_summary_includes_archived_summary_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = LiveTradeJournal(Path(tmp) / "live_trade_events.jsonl")
            reporter = DailyTradeReporter(journal=journal, email_sender=FakeEmailSender())  # type: ignore[arg-type]
            journal.append_event(
                event(
                    "summary-1",
                    "SUMMARY_SNAPSHOT",
                    None,
                    "2026-01-01T00:00:00+00:00",
                    {
                        "closed_count": 2,
                        "win_count": 1,
                        "loss_count": 1,
                        "breakeven_count": 0,
                        "known_closed_pnl": 8.0,
                        "gross_profit": 10.0,
                        "gross_loss": 2.0,
                        "best_win": 10.0,
                        "worst_loss": -2.0,
                        "archived_event_count": 10,
                        "archived_position_count": 2,
                    },
                ),
                path=journal.summary_path,
            )
            journal.append_event(
                event(
                    "active-entry",
                    "ENTRY",
                    "pos-active",
                    "2026-01-02T00:00:00+00:00",
                    {"cash_before_position": 108.0},
                )
            )
            journal.append_event(
                event(
                    "active-flat",
                    "FLAT",
                    "pos-active",
                    "2026-01-02T01:00:00+00:00",
                    {"realized_pnl_usdt_est": 3.0, "cash_before_position": 108.0, "cash_after": 111.0},
                )
            )

            subject, content = reporter.build_overall_summary_report(
                journal.load_events(),
                context=ReportRuntimeContext(period_start_cash=100.0, current_cash=111.0),
                archived=reporter.load_archived_summary_stats(),
            )

            self.assertIn("closed=3", subject)
            self.assertIn("win_rate=66.67%", subject)
            self.assertIn("pnl=11.0000U", subject)
            self.assertIn("<b>已归档平仓笔数</b><br>2", content)
            self.assertIn("<b>活跃账本平仓笔数</b><br>1", content)
            self.assertIn("<b>总已记录平仓笔数</b><br>3", content)
            self.assertIn("<b>已归档事件数</b><br>10", content)
            self.assertIn("<b>已归档仓位数</b><br>2", content)
            self.assertIn("<b>已记录平仓盈亏</b><br>11.0000 USDT", content)
            self.assertIn("<b>盈利/亏损/打平</b><br>2 / 1 / 0", content)
            self.assertIn("<b>Profit Factor</b><br>6.50", content)
            self.assertIn("<b>最大单笔盈利</b><br>10.0000 USDT", content)
            self.assertIn("<b>最大单笔亏损</b><br>-2.0000 USDT", content)
            self.assertIn("<b>未知/不完整汇总盈亏</b><br>0.0000 USDT", content)

    def test_overall_summary_after_compaction_does_not_lose_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = LiveTradeJournal(Path(tmp) / "live_trade_events.jsonl")
            reporter = DailyTradeReporter(journal=journal, email_sender=FakeEmailSender())  # type: ignore[arg-type]
            for item in [
                event("pos1-entry", "ENTRY", "pos1", "2026-01-01T00:00:00+00:00", {"cash_before_position": 100.0}),
                event("pos1-flat", "FLAT", "pos1", "2026-01-01T01:00:00+00:00", {"realized_pnl_usdt_est": 10.0, "cash_after": 110.0}),
                event("pos2-entry", "ENTRY", "pos2", "2026-01-01T02:00:00+00:00", {"cash_before_position": 110.0}),
                event("pos2-flat", "FLAT", "pos2", "2026-01-01T03:00:00+00:00", {"realized_pnl_usdt_est": -3.0, "cash_after": 107.0}),
            ]:
                journal.append_event(item)

            compact_after_weekly_summary(
                journal,
                datetime(2026, 1, 2, tzinfo=timezone.utc),
                current_position_id=None,
            )

            live_ids = {item.event_id for item in journal.load_events()}
            self.assertNotIn("pos1-entry", live_ids)
            self.assertNotIn("pos1-flat", live_ids)
            self.assertNotIn("pos2-entry", live_ids)
            self.assertNotIn("pos2-flat", live_ids)

            journal.append_event(event("pos3-entry", "ENTRY", "pos3", "2026-01-02T02:00:00+00:00", {"cash_before_position": 107.0}))
            journal.append_event(
                event(
                    "pos3-flat",
                    "FLAT",
                    "pos3",
                    "2026-01-02T03:00:00+00:00",
                    {"realized_pnl_usdt_est": 2.0, "cash_before_position": 107.0, "cash_after": 109.0},
                )
            )

            subject, content = reporter.build_overall_summary_report(
                journal.load_events(),
                context=ReportRuntimeContext(period_start_cash=100.0, current_cash=109.0),
                archived=reporter.load_archived_summary_stats(),
            )

            self.assertIn("closed=3", subject)
            self.assertIn("pnl=9.0000U", subject)
            self.assertIn("<b>已记录平仓盈亏</b><br>9.0000 USDT", content)
            self.assertIn("<b>总已记录平仓笔数</b><br>3", content)

    def test_compactor_does_not_write_summary_if_journal_replace_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = LiveTradeJournal(Path(tmp) / "live_trade_events.jsonl")
            for item in [
                event("closed-entry", "ENTRY", "pos-closed", "2026-01-01T00:00:00+00:00"),
                event("closed-flat", "FLAT", "pos-closed", "2026-01-01T01:00:00+00:00", {"realized_pnl_usdt_est": 4.0}),
            ]:
                journal.append_event(item)
            original = journal.path.read_text(encoding="utf-8")
            original_replace = os.replace

            def fail_journal_replace(src: str | Path, dst: str | Path) -> None:
                if Path(dst) == journal.path:
                    raise OSError("journal replace failed")
                original_replace(src, dst)

            with patch("src.reporting.journal_compactor.os.replace", side_effect=fail_journal_replace):
                with self.assertRaises(OSError):
                    compact_after_weekly_summary(
                        journal,
                        datetime(2026, 1, 2, tzinfo=timezone.utc),
                        current_position_id=None,
                    )

            self.assertEqual(journal.path.read_text(encoding="utf-8"), original)
            self.assertEqual(journal.load_summary_events(), [])
            self.assertEqual([item.event_id for item in journal.load_events()], ["closed-entry", "closed-flat"])

    def test_weekly_compaction_default_disabled(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "scripts" / "run_boll_cvd_live.py").read_text(encoding="utf-8")
        self.assertIn('os.getenv("WEEKLY_COMPACT_AFTER_SUCCESS", "false")', source)
        self.assertNotIn('os.getenv("WEEKLY_COMPACT_AFTER_SUCCESS", "true")', source)


if __name__ == "__main__":
    unittest.main()
