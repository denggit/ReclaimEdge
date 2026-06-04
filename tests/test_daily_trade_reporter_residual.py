from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.reporting.daily_trade_reporter import DailyReportWindow, DailyTradeReporter, ReportRuntimeContext
from src.reporting.trade_journal import JournalEvent


def event(event_type: str, payload: dict, position_id: str | None = None, ts: str = "2026-01-01T00:00:00+00:00") -> JournalEvent:
    return JournalEvent(
        event_id=f"{event_type}-{position_id or 'none'}-{len(payload)}",
        event_type=event_type,
        ts_iso=ts,
        position_id=position_id,
        payload=payload,
    )


class FakeEmailSender:
    async def send_email_async(self, *args, **kwargs) -> bool:  # type: ignore[no-untyped-def]
        return True


class DailyTradeReporterResidualTest(unittest.TestCase):
    def reporter(self) -> DailyTradeReporter:
        return DailyTradeReporter(journal=None, email_sender=FakeEmailSender())  # type: ignore[arg-type]

    def test_report_math_excludes_cash_transfer(self) -> None:
        events = [
            event("CASH_TRANSFER", {"direction": "DEPOSIT", "amount": 50.0}),
            event("FLAT", {"realized_pnl_usdt_est": 10.0}, "pos1"),
        ]
        math = DailyTradeReporter.calculate_pnl_math(
            events,
            known_closed_pnl=10.0,
            context=ReportRuntimeContext(current_cash=165.0, period_start_cash=100.0),
        )

        self.assertEqual(math.net_transfer, 50.0)
        self.assertEqual(math.strategy_total_pnl, 15.0)
        self.assertEqual(math.residual_pnl, 5.0)
        self.assertEqual(math.total_pnl, 15.0)

    def test_report_math_with_withdrawal(self) -> None:
        events = [
            event("CASH_TRANSFER", {"direction": "WITHDRAWAL", "amount": -20.0}),
            event("FLAT", {"realized_pnl_usdt_est": 5.0}, "pos1"),
        ]
        math = DailyTradeReporter.calculate_pnl_math(
            events,
            known_closed_pnl=5.0,
            context=ReportRuntimeContext(current_cash=90.0, period_start_cash=100.0),
        )

        self.assertEqual(math.net_transfer, -20.0)
        self.assertEqual(math.strategy_total_pnl, 10.0)
        self.assertEqual(math.residual_pnl, 5.0)

    def test_incomplete_positions_are_bucketed_not_duplicated(self) -> None:
        reporter = self.reporter()
        events = [
            event("ENTRY", {"side": "LONG", "price": 100.0, "layer_index": 1, "reason": "a"}, "pos1"),
            event("ENTRY", {"side": "SHORT", "price": 101.0, "layer_index": 1, "reason": "b"}, "pos2"),
        ]
        _, content = reporter.build_report(
            events,
            DailyReportWindow(
                start=datetime(2026, 1, 1, tzinfo=timezone.utc),
                end=datetime(2026, 1, 2, tzinfo=timezone.utc),
            ),
            context=ReportRuntimeContext(current_has_position=False, current_cash=110.0, period_start_cash=100.0),
        )

        self.assertNotIn("INFERRED", content)
        self.assertNotIn("pos1", content)
        self.assertNotIn("pos2", content)
        self.assertIn("不完整记录数</b><br>2", content)
        self.assertEqual(content.count("未知/不完整记录汇总"), 1)
        self.assertIn("10.0000", content)

    def test_current_open_position_is_shown_not_bucketed(self) -> None:
        reporter = self.reporter()
        events = [
            event("ENTRY", {"side": "LONG", "price": 100.0, "layer_index": 1, "reason": "open"}, "pos1"),
        ]
        _, content = reporter.build_overall_summary_report(
            events,
            context=ReportRuntimeContext(
                current_position_id="pos1",
                current_has_position=True,
                current_cash=100.0,
                period_start_cash=100.0,
            ),
        )

        self.assertIn("当前未平仓位</b><br>1", content)
        self.assertIn("不完整记录数</b><br>0", content)
        self.assertIn("无不完整记录", content)

    def test_closed_position_row_prefers_trend_runner_exit_reason(self) -> None:
        reporter = self.reporter()
        events = [
            event(
                "ENTRY",
                {"side": "LONG", "price": 100.0, "layer_index": 1, "reason": "entry", "size_margin_usdt": 1.0},
                "pos1",
            ),
            event(
                "FLAT",
                {
                    "flat_reason": "OKX position is flat. TP filled or manual close detected.",
                    "trend_runner_exit_reason": "trend_runner_max_time_after_second_tp",
                    "realized_pnl_usdt_est": 1.0,
                    "layers": 1,
                    "avg_entry_price": 100.0,
                    "last_tp_price": 111.0,
                },
                "pos1",
                ts="2026-01-01T01:00:00+00:00",
            ),
        ]

        _, content = reporter.build_report(
            events,
            DailyReportWindow(
                start=datetime(2026, 1, 1, tzinfo=timezone.utc),
                end=datetime(2026, 1, 2, tzinfo=timezone.utc),
            ),
            context=ReportRuntimeContext(current_has_position=False),
        )

        self.assertIn("trend_runner_max_time_after_second_tp", content)


if __name__ == "__main__":
    unittest.main()
