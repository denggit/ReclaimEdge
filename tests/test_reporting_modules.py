from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.reporting.daily_trade_reporter import (
    DailyReportWindow,
    DailyTradeReporter,
    ReportRuntimeContext,
)
from src.reporting.report_models import (
    ResidualPnlBucket,
    _to_float,
    fmt,
    fmt_pct,
    short_ts,
)
from src.reporting.report_pnl_math import (
    build_residual_bucket,
    calculate_pnl_math,
    net_cash_transfer,
)
from src.reporting.report_cash_events import (
    cash_drift_events,
    cash_drift_reason_label,
    cash_transfer_events,
    render_cash_events_section_html,
)
from src.reporting.report_html_sections import (
    metric_card,
    residual_bucket_html,
)
from src.reporting.report_summary_stats import (
    load_archived_summary_stats_from_events,
    max_drawdown,
    max_non_none,
    min_non_none,
    to_int,
)
from src.reporting.trade_journal import JournalEvent


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def event(event_type: str, payload: dict, position_id: str | None = None,
          ts: str = "2026-01-01T00:00:00+00:00") -> JournalEvent:
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


# ---------------------------------------------------------------------------
# report_models — formatting helpers
# ---------------------------------------------------------------------------


class FormatHelpersTest(unittest.TestCase):
    def test_fmt_float(self) -> None:
        self.assertEqual(fmt(3.14159, 2), "3.14")

    def test_fmt_none(self) -> None:
        self.assertEqual(fmt(None), "-")

    def test_fmt_pct(self) -> None:
        self.assertEqual(fmt_pct(66.666, 2), "66.67%")

    def test_fmt_pct_none(self) -> None:
        self.assertEqual(fmt_pct(None), "-")

    def test_short_ts(self) -> None:
        # short_ts uses astimezone() which converts to local time.
        # We just verify it produces a plausible formatted string.
        result = short_ts("2026-01-15T08:30:00+00:00")
        self.assertRegex(result, r"\d{2}-\d{2} \d{2}:\d{2}:\d{2}")

    def test_to_float_valid(self) -> None:
        self.assertEqual(_to_float("3.14"), 3.14)

    def test_to_float_none(self) -> None:
        self.assertIsNone(_to_float(None))

    def test_to_float_invalid(self) -> None:
        self.assertIsNone(_to_float("not_a_number"))


# ---------------------------------------------------------------------------
# report_pnl_math — net_cash_transfer
# ---------------------------------------------------------------------------


class NetCashTransferTest(unittest.TestCase):
    def test_only_cash_transfer_counted(self) -> None:
        events = [
            event("CASH_TRANSFER", {"amount": -100.0}),
            event("ACCOUNT_CASH_DRIFT", {"amount": -50.0}),
        ]
        self.assertEqual(net_cash_transfer(events), -100.0)

    def test_multiple_cash_transfers_summed(self) -> None:
        events = [
            event("CASH_TRANSFER", {"amount": 100.0}),
            event("CASH_TRANSFER", {"amount": -30.0}),
        ]
        self.assertEqual(net_cash_transfer(events), 70.0)

    def test_no_transfers_returns_zero(self) -> None:
        events = [
            event("ENTRY", {"side": "LONG"}, "pos1"),
            event("ACCOUNT_CASH_DRIFT", {"amount": -10.0}),
        ]
        self.assertEqual(net_cash_transfer(events), 0.0)


# ---------------------------------------------------------------------------
# report_pnl_math — calculate_pnl_math
# ---------------------------------------------------------------------------


class CalculatePnlMathTest(unittest.TestCase):
    def test_has_position_uses_equity(self) -> None:
        events: list[JournalEvent] = []
        math = calculate_pnl_math(
            events,
            known_closed_pnl=0.0,
            context=ReportRuntimeContext(
                period_start_cash=614.7787,
                current_cash=561.9095,
                current_equity=675.1616,
                current_has_position=True,
            ),
        )
        self.assertEqual(math.current_account_value_source, "equity")
        self.assertAlmostEqual(math.current_account_value, 675.1616, places=4)
        # strategy_total_pnl = 675.1616 - 614.7787 = 60.3829
        self.assertAlmostEqual(math.strategy_total_pnl, 675.1616 - 614.7787, places=4)

    def test_flat_uses_cash(self) -> None:
        events: list[JournalEvent] = []
        math = calculate_pnl_math(
            events,
            known_closed_pnl=0.0,
            context=ReportRuntimeContext(
                period_start_cash=100.0,
                current_cash=110.0,
                current_equity=115.0,
                current_has_position=False,
            ),
        )
        self.assertEqual(math.current_account_value_source, "cash")
        self.assertAlmostEqual(math.current_account_value, 110.0)
        self.assertAlmostEqual(math.strategy_total_pnl, 10.0)

    def test_has_position_equity_none_falls_back_to_cash(self) -> None:
        events: list[JournalEvent] = []
        math = calculate_pnl_math(
            events,
            known_closed_pnl=0.0,
            context=ReportRuntimeContext(
                period_start_cash=100.0,
                current_cash=90.0,
                current_equity=None,
                current_has_position=True,
            ),
        )
        self.assertEqual(math.current_account_value_source, "cash")

    def test_period_start_equity_used_when_available(self) -> None:
        events: list[JournalEvent] = []
        math = calculate_pnl_math(
            events,
            known_closed_pnl=0.0,
            context=ReportRuntimeContext(
                period_start_cash=614.7787,
                period_start_equity=610.0000,
                current_cash=561.9095,
                current_equity=675.1616,
                current_has_position=True,
            ),
        )
        self.assertEqual(math.current_account_value_source, "equity")
        self.assertAlmostEqual(math.period_start_value, 610.0000)
        self.assertAlmostEqual(math.strategy_total_pnl, 675.1616 - 610.0000, places=4)

    def test_no_context_returns_none_values(self) -> None:
        events: list[JournalEvent] = []
        math = calculate_pnl_math(events, known_closed_pnl=5.0, context=None)
        self.assertIsNone(math.strategy_total_pnl)
        self.assertIsNone(math.residual_pnl)
        self.assertEqual(math.current_account_value_source, "cash")


# ---------------------------------------------------------------------------
# report_pnl_math — build_residual_bucket
# ---------------------------------------------------------------------------


class BuildResidualBucketTest(unittest.TestCase):
    def test_equity_bucket_has_value_source_fields(self) -> None:
        events: list[JournalEvent] = []
        bucket = build_residual_bucket(
            events,
            incomplete_count=2,
            known_closed_pnl=0.0,
            context=ReportRuntimeContext(
                period_start_cash=614.7787,
                current_cash=561.9095,
                current_equity=675.1616,
                current_has_position=True,
            ),
        )
        self.assertEqual(bucket.incomplete_count, 2)
        self.assertEqual(bucket.current_account_value_source, "equity")
        self.assertAlmostEqual(bucket.current_account_value, 675.1616, places=4)
        self.assertAlmostEqual(bucket.period_start_value, 614.7787, places=4)

    def test_flat_bucket_has_cash_source(self) -> None:
        events: list[JournalEvent] = []
        bucket = build_residual_bucket(
            events,
            incomplete_count=0,
            known_closed_pnl=0.0,
            context=ReportRuntimeContext(
                period_start_cash=100.0,
                current_cash=110.0,
                current_has_position=False,
            ),
        )
        self.assertEqual(bucket.current_account_value_source, "cash")
        self.assertAlmostEqual(bucket.current_account_value, 110.0, places=4)


# ---------------------------------------------------------------------------
# report_html_sections — residual_bucket_html
# ---------------------------------------------------------------------------


class ResidualBucketHtmlTest(unittest.TestCase):
    def test_equity_bucket_html_includes_equity_annotation(self) -> None:
        bucket = ResidualPnlBucket(
            incomplete_count=2,
            pnl=60.3829,
            cash_start=614.7787,
            cash_end=561.9095,
            net_transfer=0.0,
            strategy_total_pnl=60.3829,
            known_closed_pnl=0.0,
            formula="current_equity - period_start_value (有仓, 优先使用 equity)",
            note="incomplete records are bucketed (有仓, 收益估算基于 equity)",
            period_start_value=614.7787,
            period_start_value_source="cash",
            current_account_value=675.1616,
            current_account_value_source="equity",
        )
        html_result = residual_bucket_html(bucket)
        self.assertIn("equity", html_result.lower())
        # Check column headers use "账户价值" instead of "现金"
        self.assertIn("起始账户价值", html_result)
        self.assertIn("当前账户价值", html_result)
        # Should not say "当前现金" as the main column header
        self.assertNotIn("当前现金</th>", html_result)

    def test_cash_bucket_html_includes_cash_annotation(self) -> None:
        bucket = ResidualPnlBucket(
            incomplete_count=1,  # need >0 so full table is rendered
            pnl=10.0,  # non-zero so it's not skipped
            cash_start=100.0,
            cash_end=110.0,
            net_transfer=0.0,
            strategy_total_pnl=10.0,
            known_closed_pnl=0.0,
            formula="current_cash - period_start_cash",
            note="incomplete records are bucketed",
            period_start_value=100.0,
            period_start_value_source="cash",
            current_account_value=110.0,
            current_account_value_source="cash",
        )
        html_result = residual_bucket_html(bucket)
        self.assertIn("cash", html_result.lower())

    def test_empty_bucket_html_returns_no_records(self) -> None:
        bucket = ResidualPnlBucket(
            incomplete_count=0,
            pnl=0.0,
            cash_start=100.0,
            cash_end=100.0,
            net_transfer=0.0,
            strategy_total_pnl=0.0,
            known_closed_pnl=0.0,
            formula="-",
            note="no incomplete records",
            period_start_value=100.0,
            period_start_value_source="cash",
            current_account_value=100.0,
            current_account_value_source="cash",
        )
        html_result = residual_bucket_html(bucket)
        self.assertIn("无不完整记录", html_result)


# ---------------------------------------------------------------------------
# report_cash_events — cash events section HTML
# ---------------------------------------------------------------------------


class CashEventsHtmlTest(unittest.TestCase):
    def test_cash_drift_negative_does_not_show_withdrawal(self) -> None:
        events = [
            event(
                "ACCOUNT_CASH_DRIFT",
                {
                    "amount": -50.0,
                    "cash_before": 614.7787,
                    "cash_after": 561.9095,
                    "reason": "unsafe_state:has_position,strategy_layers,current_position_id,order_settle",
                },
            ),
        ]
        html_result = render_cash_events_section_html(events, net_cash_transfer)
        self.assertIn("持仓中现金漂移", html_result)
        self.assertIn("保证金", html_result)
        self.assertIn("非转账", html_result)
        self.assertNotIn("取钱", html_result)
        self.assertNotIn("提现", html_result)

    def test_cash_transfer_still_shows_real_transfer(self) -> None:
        events = [
            event(
                "CASH_TRANSFER",
                {
                    "direction": "WITHDRAWAL",
                    "amount": -100.0,
                    "cash_before": 200.0,
                    "cash_after": 100.0,
                    "reason": "safe_flat_account_sync",
                },
            ),
        ]
        html_result = render_cash_events_section_html(events, net_cash_transfer)
        self.assertIn("真实转出", html_result)

    def test_no_events_shows_empty_message(self) -> None:
        events: list[JournalEvent] = []
        html_result = render_cash_events_section_html(events, net_cash_transfer)
        self.assertIn("本周期无现金变动事件", html_result)


# ---------------------------------------------------------------------------
# report_cash_events — event filter helpers
# ---------------------------------------------------------------------------


class CashEventFiltersTest(unittest.TestCase):
    def test_cash_drift_events_filter(self) -> None:
        events = [
            event("CASH_TRANSFER", {"amount": 100.0}),
            event("ACCOUNT_CASH_DRIFT", {"amount": -50.0}),
            event("ENTRY", {"side": "LONG"}, "pos1"),
        ]
        drifts = cash_drift_events(events)
        self.assertEqual(len(drifts), 1)
        self.assertEqual(drifts[0].event_type, "ACCOUNT_CASH_DRIFT")

    def test_cash_transfer_events_filter(self) -> None:
        events = [
            event("CASH_TRANSFER", {"amount": 100.0}),
            event("ACCOUNT_CASH_DRIFT", {"amount": -10.0}),
        ]
        transfers = cash_transfer_events(events)
        self.assertEqual(len(transfers), 1)
        self.assertEqual(transfers[0].event_type, "CASH_TRANSFER")


# ---------------------------------------------------------------------------
# report_summary_stats — numeric helpers
# ---------------------------------------------------------------------------


class NumericHelpersTest(unittest.TestCase):
    def test_to_int_from_float(self) -> None:
        self.assertEqual(to_int(3.7), 3)

    def test_to_int_from_none(self) -> None:
        self.assertEqual(to_int(None), 0)

    def test_max_non_none(self) -> None:
        self.assertEqual(max_non_none(None, 5.0), 5.0)
        self.assertEqual(max_non_none(3.0, 5.0), 5.0)
        self.assertEqual(max_non_none(7.0, None), 7.0)
        self.assertIsNone(max_non_none(None, None))

    def test_min_non_none(self) -> None:
        self.assertEqual(min_non_none(None, -2.0), -2.0)
        self.assertEqual(min_non_none(3.0, 5.0), 3.0)
        self.assertEqual(min_non_none(-7.0, None), -7.0)
        self.assertIsNone(min_non_none(None, None))

    def test_max_drawdown(self) -> None:
        equity = [100.0, 110.0, 95.0, 105.0]
        dd_usdt, dd_pct = max_drawdown(equity)
        self.assertAlmostEqual(dd_usdt, 15.0, places=4)  # 110 - 95
        self.assertAlmostEqual(dd_pct, 13.6363, places=3)  # 15/110*100

    def test_max_drawdown_insufficient_points(self) -> None:
        self.assertEqual(max_drawdown([100.0]), (None, None))
        self.assertEqual(max_drawdown([]), (None, None))


# ---------------------------------------------------------------------------
# report_summary_stats — load_archived_summary_stats_from_events
# ---------------------------------------------------------------------------


class LoadArchivedSummaryStatsTest(unittest.TestCase):
    def test_aggregates_multiple_snapshots(self) -> None:
        summary_events = [
            JournalEvent(
                event_id="snap-1",
                event_type="SUMMARY_SNAPSHOT",
                ts_iso="2026-01-01T00:00:00+00:00",
                position_id=None,
                payload={
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
            JournalEvent(
                event_id="snap-2",
                event_type="SUMMARY_SNAPSHOT",
                ts_iso="2026-01-02T00:00:00+00:00",
                position_id=None,
                payload={
                    "closed_count": 3,
                    "win_count": 2,
                    "loss_count": 1,
                    "breakeven_count": 0,
                    "known_closed_pnl": 5.0,
                    "gross_profit": 7.0,
                    "gross_loss": 2.0,
                    "best_win": 5.0,
                    "worst_loss": -2.0,
                    "archived_event_count": 5,
                    "archived_position_count": 1,
                },
            ),
        ]
        stats = load_archived_summary_stats_from_events(summary_events)
        self.assertEqual(stats.closed_count, 5)  # 2 + 3
        self.assertEqual(stats.win_count, 3)  # 1 + 2
        self.assertEqual(stats.known_closed_pnl, 13.0)  # 8 + 5
        self.assertAlmostEqual(stats.best_win, 10.0)
        self.assertAlmostEqual(stats.worst_loss, -2.0)
        self.assertEqual(stats.archived_event_count, 15)  # 10 + 5
        self.assertEqual(stats.archived_position_count, 3)  # 2 + 1


# ---------------------------------------------------------------------------
# report_html_sections — metric_card
# ---------------------------------------------------------------------------


class MetricCardTest(unittest.TestCase):
    def test_metric_card_renders_html(self) -> None:
        card = metric_card("测试标题", "测试值")
        self.assertIn("测试标题", card)
        self.assertIn("测试值", card)
        self.assertIn("background:#f6f8fa", card)

    def test_metric_card_escapes_html(self) -> None:
        card = metric_card("<script>", "val")
        self.assertNotIn("<script>", card)
        self.assertIn("&lt;script&gt;", card)


# ---------------------------------------------------------------------------
# overall summary report — metric card label updates
# ---------------------------------------------------------------------------


class OverallSummaryMetricLabelsTest(unittest.TestCase):
    def reporter(self) -> DailyTradeReporter:
        return DailyTradeReporter(journal=None, email_sender=FakeEmailSender())  # type: ignore[arg-type]

    def test_overall_summary_uses_account_value_labels(self) -> None:
        reporter = self.reporter()
        events = [
            event("ENTRY", {"side": "LONG", "price": 100.0, "layer_index": 1, "reason": "open"}, "pos1",
                  ts="2026-01-01T00:00:00+00:00"),
            event("FLAT", {"realized_pnl_usdt_est": 10.0, "cash_before_position": 100.0, "cash_after": 110.0}, "pos1",
                  ts="2026-01-01T01:00:00+00:00"),
        ]
        _, content = reporter.build_overall_summary_report(
            events,
            context=ReportRuntimeContext(period_start_cash=100.0, current_cash=110.0),
        )
        # Updated labels
        self.assertIn("初始账户值", content)
        self.assertIn("最新账户值", content)
        # value source is shown for flat (cash) context
        self.assertIn("USDT (cash)", content)

    def test_overall_summary_uses_equity_for_current_account_value_when_position_open(self) -> None:
        """有仓时顶部"最新账户值"卡片应使用 current_equity 并显示 source=equity。"""
        reporter = self.reporter()
        events = [
            event("ENTRY", {"side": "LONG", "price": 100.0, "layer_index": 1, "reason": "open"}, "pos1",
                  ts="2026-01-01T00:00:00+00:00"),
        ]
        context = ReportRuntimeContext(
            period_start_cash=614.7787,
            current_cash=561.9095,
            current_equity=675.1616,
            current_has_position=True,
            current_position_id="pos1",
        )
        _, content = reporter.build_overall_summary_report(events, context=context)
        self.assertIn("最新账户值", content)
        self.assertIn("675.1616", content)
        self.assertIn("equity", content)
        # 不应把 current_cash 当作最新账户值卡片主值
        self.assertNotIn("561.9095 USDT (equity)", content)

    def test_overall_summary_uses_cash_for_current_account_value_when_flat(self) -> None:
        """空仓时顶部"最新账户值"卡片应使用 current_cash 并显示 source=cash。"""
        reporter = self.reporter()
        events = [
            event("ENTRY", {"side": "LONG", "price": 100.0, "layer_index": 1, "reason": "open"}, "pos1",
                  ts="2026-01-01T00:00:00+00:00"),
            event("FLAT", {"realized_pnl_usdt_est": 10.0, "cash_before_position": 100.0, "cash_after": 110.0}, "pos1",
                  ts="2026-01-01T01:00:00+00:00"),
        ]
        context = ReportRuntimeContext(
            period_start_cash=100.0,
            current_cash=110.0,
            current_equity=999.0,
            current_has_position=False,
        )
        _, content = reporter.build_overall_summary_report(events, context=context)
        self.assertIn("最新账户值", content)
        self.assertIn("110.0000", content)
        self.assertIn("cash", content)


if __name__ == "__main__":
    unittest.main()
