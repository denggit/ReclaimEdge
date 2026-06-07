from __future__ import annotations

import unittest
from datetime import datetime, timezone

from src.reporting.daily_trade_reporter import DailyReportWindow, DailyTradeReporter, ReportRuntimeContext
from src.reporting.trade_journal import JournalEvent


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


class DailyTradeReporterCashDriftTest(unittest.TestCase):
    def reporter(self) -> DailyTradeReporter:
        return DailyTradeReporter(journal=None, email_sender=FakeEmailSender())  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # 1. ACCOUNT_CASH_DRIFT 不计入 net transfer
    # ------------------------------------------------------------------
    def test_net_transfer_excludes_cash_drift(self) -> None:
        events = [
            event("CASH_TRANSFER", {"direction": "WITHDRAWAL", "amount": -100.0}, ts="2026-01-01T01:00:00+00:00"),
            event(
                "ACCOUNT_CASH_DRIFT",
                {
                    "amount": -50.0,
                    "cash_before": 614.7787,
                    "cash_after": 561.9095,
                    "reason": "unsafe_state:has_position,strategy_layers,current_position_id,order_settle",
                },
                ts="2026-01-01T02:00:00+00:00",
            ),
        ]
        net = DailyTradeReporter._net_cash_transfer(events)
        self.assertEqual(net, -100.0,
                         "ACCOUNT_CASH_DRIFT must NOT be counted in net transfer; only CASH_TRANSFER contributes")

    # ------------------------------------------------------------------
    # 2. 有仓时用 current_equity 计算策略估算收益
    # ------------------------------------------------------------------
    def test_pnl_math_uses_equity_when_has_position(self) -> None:
        """When holding a position, strategy_total_pnl should be based on equity, not available cash."""
        events: list[JournalEvent] = []
        math = DailyTradeReporter.calculate_pnl_math(
            events,
            known_closed_pnl=0.0,
            context=ReportRuntimeContext(
                period_start_cash=614.7787,
                current_cash=561.9095,
                current_equity=675.1616,
                current_has_position=True,
            ),
        )

        # With equity: 675.1616 - 614.7787 - 0 = 60.3829
        self.assertEqual(math.current_account_value_source, "equity")
        self.assertAlmostEqual(math.current_account_value, 675.1616, places=4)
        self.assertAlmostEqual(math.strategy_total_pnl, 675.1616 - 614.7787, places=4)
        # Without the fix, the old code would have computed: 561.9095 - 614.7787 = -52.8692
        # which incorrectly looks like a loss/withdrawal
        self.assertGreater(math.strategy_total_pnl, 0,
                           "Equity-based PnL should be positive, not negative like cash-based PnL would be")

    def test_pnl_math_still_uses_equity_when_has_position_with_period_start_equity(self) -> None:
        """When both period_start_equity is available and position is held, use equity for both sides."""
        events: list[JournalEvent] = []
        math = DailyTradeReporter.calculate_pnl_math(
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
        self.assertAlmostEqual(math.current_account_value, 675.1616)
        self.assertAlmostEqual(math.strategy_total_pnl, 675.1616 - 610.0000, places=4)

    # ------------------------------------------------------------------
    # 3. 空仓时仍用 current_cash
    # ------------------------------------------------------------------
    def test_pnl_math_uses_cash_when_flat(self) -> None:
        """When flat (no position), strategy_total_pnl should use current_cash, not equity."""
        events: list[JournalEvent] = []
        math = DailyTradeReporter.calculate_pnl_math(
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

    def test_pnl_math_uses_cash_when_equity_none_even_if_has_position(self) -> None:
        """When equity is None, fall back to cash even if has_position is True."""
        events: list[JournalEvent] = []
        math = DailyTradeReporter.calculate_pnl_math(
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
        self.assertAlmostEqual(math.current_account_value, 90.0)
        self.assertAlmostEqual(math.strategy_total_pnl, -10.0)

    # ------------------------------------------------------------------
    # 4. 报告 HTML 包含 cash drift 说明
    # ------------------------------------------------------------------
    def test_report_html_includes_cash_drift_section(self) -> None:
        reporter = self.reporter()
        events = [
            event(
                "ACCOUNT_CASH_DRIFT",
                {
                    "amount": -52.8692,
                    "cash_before": 614.7787,
                    "cash_after": 561.9095,
                    "reason": "unsafe_state:has_position,strategy_layers,current_position_id,order_settle",
                },
                ts="2026-01-01T01:00:00+00:00",
            ),
            event(
                "ENTRY",
                {"side": "LONG", "price": 100.0, "layer_index": 1, "reason": "test"},
                "pos1",
                ts="2026-01-01T00:30:00+00:00",
            ),
        ]
        _, content = reporter.build_report(
            events,
            DailyReportWindow(
                start=datetime(2026, 1, 1, tzinfo=timezone.utc),
                end=datetime(2026, 1, 2, tzinfo=timezone.utc),
            ),
            context=ReportRuntimeContext(
                current_has_position=True,
                current_cash=561.9095,
                current_equity=675.1616,
                period_start_cash=614.7787,
            ),
        )

        # Should include cash drift section
        self.assertIn("持仓中现金漂移", content)
        self.assertIn("保证金", content)
        self.assertIn("非转账", content)
        self.assertIn("可用现金减少", content)

        # Should NOT include withdrawal-related terms for cash drift
        self.assertNotIn("取钱", content)
        self.assertNotIn("提现", content)

    def test_report_html_cash_transfer_still_shows_real_transfer(self) -> None:
        """CASH_TRANSFER events should still be displayed as real transfers."""
        reporter = self.reporter()
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
                ts="2026-01-01T01:00:00+00:00",
            ),
        ]
        _, content = reporter.build_report(
            events,
            DailyReportWindow(
                start=datetime(2026, 1, 1, tzinfo=timezone.utc),
                end=datetime(2026, 1, 2, tzinfo=timezone.utc),
            ),
            context=ReportRuntimeContext(
                current_has_position=False,
                current_cash=100.0,
                period_start_cash=200.0,
            ),
        )

        # CASH_TRANSFER should show as real transfer
        self.assertIn("真实转出", content)
        self.assertIn("📥 真实转出", content)

    def test_report_html_mixed_cash_transfer_and_drift(self) -> None:
        """When both CASH_TRANSFER and ACCOUNT_CASH_DRIFT exist, both sections should appear."""
        reporter = self.reporter()
        events = [
            event(
                "CASH_TRANSFER",
                {
                    "direction": "DEPOSIT",
                    "amount": 500.0,
                    "cash_before": 500.0,
                    "cash_after": 1000.0,
                    "reason": "safe_flat_account_sync",
                },
                ts="2026-01-01T01:00:00+00:00",
            ),
            event(
                "ACCOUNT_CASH_DRIFT",
                {
                    "amount": -52.8692,
                    "cash_before": 614.7787,
                    "cash_after": 561.9095,
                    "reason": "unsafe_state:has_position,strategy_layers,current_position_id,order_settle",
                },
                ts="2026-01-01T02:00:00+00:00",
            ),
            event(
                "ENTRY",
                {"side": "SHORT", "price": 1608.75, "layer_index": 3, "reason": "add"},
                "pos2",
                ts="2026-01-01T01:30:00+00:00",
            ),
        ]
        _, content = reporter.build_report(
            events,
            DailyReportWindow(
                start=datetime(2026, 1, 1, tzinfo=timezone.utc),
                end=datetime(2026, 1, 2, tzinfo=timezone.utc),
            ),
            context=ReportRuntimeContext(
                current_has_position=True,
                current_cash=561.9095,
                current_equity=675.1616,
                period_start_cash=614.7787,
            ),
        )

        # Both sections should appear
        self.assertIn("真实转入", content)
        self.assertIn("持仓中现金漂移", content)
        self.assertIn("保证金", content)

        # Net transfer in residual bucket should only count CASH_TRANSFER
        self.assertIn("净转入/转出", content)

    # ------------------------------------------------------------------
    # 5. Cash drift reason label helper
    # ------------------------------------------------------------------
    def test_cash_drift_reason_label_has_position(self) -> None:
        label = DailyTradeReporter._cash_drift_reason_label(
            "unsafe_state:has_position,strategy_layers,current_position_id,order_settle"
        )
        self.assertIn("持仓/补仓/订单结算", label)
        self.assertIn("非转账", label)
        self.assertNotIn("取钱", label)

    def test_cash_drift_reason_label_flat_cooldown(self) -> None:
        label = DailyTradeReporter._cash_drift_reason_label("flat_settle_cooldown")
        self.assertIn("冷却期", label)

    def test_cash_drift_reason_label_unknown(self) -> None:
        label = DailyTradeReporter._cash_drift_reason_label("")
        self.assertEqual(label, "未知原因")

    # ------------------------------------------------------------------
    # 6. Cash drift/transfer event filter helpers
    # ------------------------------------------------------------------
    def test_cash_drift_events_filter(self) -> None:
        events = [
            event("CASH_TRANSFER", {"amount": 100.0}),
            event("ACCOUNT_CASH_DRIFT", {"amount": -50.0, "reason": "unsafe_state:has_position"}),
            event("ACCOUNT_CASH_DRIFT", {"amount": 10.0, "reason": "flat_settle_cooldown"}),
            event("ENTRY", {"side": "LONG"}, "pos1"),
        ]
        drifts = DailyTradeReporter._cash_drift_events(events)
        self.assertEqual(len(drifts), 2)

    def test_cash_transfer_events_filter(self) -> None:
        events = [
            event("CASH_TRANSFER", {"amount": 100.0}),
            event("CASH_TRANSFER", {"amount": -50.0}),
            event("ACCOUNT_CASH_DRIFT", {"amount": -10.0, "reason": "unsafe_state:has_position"}),
        ]
        transfers = DailyTradeReporter._cash_transfer_events(events)
        self.assertEqual(len(transfers), 2)

    # ------------------------------------------------------------------
    # 7. Overall summary also includes cash drift section
    # ------------------------------------------------------------------
    def test_overall_summary_includes_cash_events_section(self) -> None:
        reporter = self.reporter()
        events = [
            event(
                "ACCOUNT_CASH_DRIFT",
                {
                    "amount": -30.0,
                    "cash_before": 500.0,
                    "cash_after": 470.0,
                    "reason": "unsafe_state:has_position,order_settle",
                },
                ts="2026-01-01T01:00:00+00:00",
            ),
            event(
                "ENTRY",
                {"side": "LONG", "price": 100.0, "layer_index": 1},
                "pos1",
                ts="2026-01-01T00:30:00+00:00",
            ),
            event(
                "FLAT",
                {"realized_pnl_usdt_est": 5.0, "cash_before_position": 500.0, "cash_after": 505.0},
                "pos1",
                ts="2026-01-01T02:00:00+00:00",
            ),
        ]
        _, content = reporter.build_overall_summary_report(
            events,
            context=ReportRuntimeContext(
                current_has_position=False,
                current_cash=505.0,
                period_start_cash=500.0,
            ),
        )

        self.assertIn("账户现金变动说明", content)
        self.assertIn("持仓中现金漂移", content)

    # ------------------------------------------------------------------
    # 8. Residual bucket shows equity-based formula note when position exists
    # ------------------------------------------------------------------
    def test_residual_bucket_shows_equity_source_when_has_position(self) -> None:
        reporter = self.reporter()
        events = [
            event(
                "ENTRY",
                {"side": "LONG", "price": 100.0, "layer_index": 1, "reason": "open"},
                "pos-open",
                ts="2026-01-01T00:30:00+00:00",
            ),
        ]
        _, content = reporter.build_report(
            events,
            DailyReportWindow(
                start=datetime(2026, 1, 1, tzinfo=timezone.utc),
                end=datetime(2026, 1, 2, tzinfo=timezone.utc),
            ),
            context=ReportRuntimeContext(
                current_position_id="pos-open",
                current_has_position=True,
                current_cash=561.9095,
                current_equity=675.1616,
                period_start_cash=614.7787,
            ),
        )

        # Should mention equity in the formula/residual note
        self.assertIn("equity", content.lower())


if __name__ == "__main__":
    unittest.main()
