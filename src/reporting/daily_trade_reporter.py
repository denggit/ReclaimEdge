from __future__ import annotations

import asyncio
import html
from datetime import datetime, timedelta, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Re-export public types so existing import paths continue to work
# ---------------------------------------------------------------------------
from src.reporting.report_models import (
    ArchivedSummaryStats,
    DailyReportWindow,
    ReportPnlMath,
    ReportRuntimeContext,
    ResidualPnlBucket,
    _to_float,
    fmt,
    fmt_pct,
    short_ts,
)
from src.reporting.report_pnl_math import (
    build_residual_bucket,
    calculate_pnl_math,
    cash_before_or_from_event,
    cash_from_event,
    infer_cash_at_or_before,
    infer_first_cash,
    net_cash_transfer,
)
from src.reporting.report_cash_events import (
    cash_drift_events,
    cash_drift_reason_label,
    cash_transfer_events,
    render_cash_events_section_html,
)
from src.reporting.report_html_sections import (
    closed_position_row,
    metric_card,
    open_position_row,
    render_daily_html,
    residual_bucket_html,
)
from src.reporting.report_summary_stats import (
    load_archived_summary_stats_from_events,
    max_drawdown,
    max_non_none,
    min_non_none,
    to_int,
)
from src.reporting.trade_journal import JournalEvent, LiveTradeJournal, group_position_events
from src.utils.email_sender import EmailSender


class DailyTradeReporter:
    """Build and send HTML trade reports from LiveTradeJournal.

    Report building reads JSONL files and renders HTML. Those operations are run
    in a worker thread so scheduled reports never block the asyncio event loop
    that handles live tick processing.
    """

    def __init__(self, journal: LiveTradeJournal, email_sender: EmailSender) -> None:
        self.journal = journal
        self.email_sender = email_sender

    # ------------------------------------------------------------------
    # public async entry points (unchanged signatures)
    # ------------------------------------------------------------------

    async def send_last_24h_report(self, context: ReportRuntimeContext | None = None) -> bool:
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=24)
        subject, content = await asyncio.to_thread(self._build_last_24h_report_sync, start, end, context)
        return await self.email_sender.send_email_async(subject, content, content_type="html")

    async def send_overall_summary_report(self, context: ReportRuntimeContext | None = None) -> bool:
        subject, content = await asyncio.to_thread(self._build_overall_summary_report_sync, context)
        return await self.email_sender.send_email_async(subject, content, content_type="html")

    # ------------------------------------------------------------------
    # sync report builders (delegated to worker thread)
    # ------------------------------------------------------------------

    def _build_last_24h_report_sync(
        self, start: datetime, end: datetime, context: ReportRuntimeContext | None
    ) -> tuple[str, str]:
        all_events = self.journal.load_events()
        events = self._filter_events(all_events, start=start, end=end)
        context = self._with_period_start_cash(context, all_events, start)
        return self.build_report(events, DailyReportWindow(start=start, end=end), context=context)

    def _build_overall_summary_report_sync(self, context: ReportRuntimeContext | None) -> tuple[str, str]:
        events = self.journal.load_events()
        archived = self.load_archived_summary_stats()
        context = self._with_overall_start_cash(context, events)
        return self.build_overall_summary_report(events, context=context, archived=archived)

    # ------------------------------------------------------------------
    # archived summary stats (thin wrapper)
    # ------------------------------------------------------------------

    def load_archived_summary_stats(self) -> ArchivedSummaryStats:
        return load_archived_summary_stats_from_events(self.journal.load_summary_events())

    # ------------------------------------------------------------------
    # report builders (core logic)
    # ------------------------------------------------------------------

    def build_report(
        self,
        events: list[JournalEvent],
        window: DailyReportWindow,
        context: ReportRuntimeContext | None = None,
    ) -> tuple[str, str]:
        grouped = group_position_events(events)
        finished_positions = []
        open_positions = []
        incomplete_count = 0
        known_closed_pnl = 0.0
        total_closed = 0
        win_count = 0

        for position_id, items in grouped.items():
            if position_id == "UNKNOWN":
                continue
            entry_events = [e for e in items if e.event_type == "ENTRY"]
            tp_events = [e for e in items if e.event_type == "TP_UPDATE"]
            flat_events = [e for e in items if e.event_type == "FLAT"]
            recovery_events = [e for e in items if e.event_type == "STARTUP_RECOVERY"]
            last_flat = flat_events[-1] if flat_events else None
            first_entry = entry_events[0] if entry_events else recovery_events[0] if recovery_events else items[0]
            last_event = items[-1]

            if last_flat is None:
                if self._is_current_open_position(position_id, context):
                    open_positions.append((position_id, first_entry, entry_events, tp_events, last_event))
                else:
                    incomplete_count += 1
                continue

            pnl = last_flat.payload.get("realized_pnl_usdt_est")
            if pnl is not None:
                pnl_float = float(pnl)
                known_closed_pnl += pnl_float
                win_count += 1 if pnl_float > 0 else 0
            total_closed += 1
            finished_positions.append((position_id, first_entry, entry_events, tp_events, last_flat))

        residual_bucket = self._build_residual_bucket(events, incomplete_count, known_closed_pnl, context)
        total_pnl = residual_bucket.strategy_total_pnl if residual_bucket.strategy_total_pnl is not None else known_closed_pnl
        subject = f"📊 ReclaimEdge 日报 | 最近24小时 | closed={total_closed} pnl={total_pnl:.4f}U"
        cash_html = self._cash_events_section_html(events)
        html_body = render_daily_html(
            window=window,
            events=events,
            finished_positions=finished_positions,
            open_positions=open_positions,
            total_pnl=total_pnl,
            known_closed_pnl=known_closed_pnl,
            total_closed=total_closed,
            win_count=win_count,
            residual_bucket=residual_bucket,
            cash_events_section_html_str=cash_html,
        )
        return subject, html_body

    def build_overall_summary_report(
        self,
        events: list[JournalEvent],
        context: ReportRuntimeContext | None = None,
        archived: ArchivedSummaryStats | None = None,
    ) -> tuple[str, str]:
        archived = archived or ArchivedSummaryStats()
        events_sorted = sorted(events, key=lambda item: item.ts_iso)
        grouped = group_position_events(events_sorted)

        active_closed_count = 0
        open_count = 0
        incomplete_count = 0
        active_win_count = 0
        active_loss_count = 0
        active_breakeven_count = 0
        entry_count = 0
        add_count = 0
        tp_update_count = 0
        three_stage_runner_count = 0
        error_count = 0
        active_known_closed_pnl = 0.0
        active_gross_profit = 0.0
        active_gross_loss = 0.0
        active_best_win: float | None = None
        active_worst_loss: float | None = None
        first_cash: float | None = None
        latest_cash: float | None = None
        equity_points: list[float] = []

        for event in events_sorted:
            if event.event_type == "ENTRY":
                entry_count += 1
                if str(event.payload.get("intent_type", "")).startswith("ADD_"):
                    add_count += 1
                if event.payload.get("tp_plan") == "THREE_STAGE_RUNNER":
                    three_stage_runner_count += 1
            elif event.event_type == "TP_UPDATE":
                tp_update_count += 1
            elif event.event_type == "ERROR":
                error_count += 1

        for position_id, items in grouped.items():
            if position_id == "UNKNOWN":
                continue
            flat_events = [e for e in items if e.event_type == "FLAT"]
            flat = flat_events[-1] if flat_events else None

            if flat is None:
                if self._is_current_open_position(position_id, context):
                    open_count += 1
                else:
                    incomplete_count += 1
                continue

            active_closed_count += 1
            pnl = _to_float(flat.payload.get("realized_pnl_usdt_est"))
            cash_before = _to_float(flat.payload.get("cash_before_position"))
            cash_after = _to_float(flat.payload.get("cash_after"))

            if first_cash is None and cash_before is not None:
                first_cash = cash_before
                equity_points.append(cash_before)
            if cash_after is not None:
                latest_cash = cash_after
                equity_points.append(cash_after)

            if pnl is None:
                continue
            active_known_closed_pnl += pnl
            if pnl > 0:
                active_win_count += 1
                active_gross_profit += pnl
                active_best_win = pnl if active_best_win is None else max(active_best_win, pnl)
            elif pnl < 0:
                active_loss_count += 1
                active_gross_loss += abs(pnl)
                active_worst_loss = pnl if active_worst_loss is None else min(active_worst_loss, pnl)
            else:
                active_breakeven_count += 1

        closed_count = archived.closed_count + active_closed_count
        win_count = archived.win_count + active_win_count
        loss_count = archived.loss_count + active_loss_count
        breakeven_count = archived.breakeven_count + active_breakeven_count
        known_closed_pnl = archived.known_closed_pnl + active_known_closed_pnl
        gross_profit = archived.gross_profit + active_gross_profit
        gross_loss = archived.gross_loss + active_gross_loss
        best_win = max_non_none(archived.best_win, active_best_win)
        worst_loss = min_non_none(archived.worst_loss, active_worst_loss)

        residual_bucket = self._build_residual_bucket(events_sorted, incomplete_count, known_closed_pnl, context)
        total_pnl = residual_bucket.strategy_total_pnl if residual_bucket.strategy_total_pnl is not None else known_closed_pnl
        if residual_bucket.cash_start is not None:
            first_cash = residual_bucket.cash_start if first_cash is None else first_cash
        if residual_bucket.cash_end is not None:
            latest_cash = residual_bucket.cash_end
            equity_points.append(residual_bucket.cash_end)

        # --- account value display: prefer equity-based values when available ---
        display_start_value = (
            residual_bucket.period_start_value
            if residual_bucket.period_start_value is not None
            else first_cash
        )
        display_current_value = (
            residual_bucket.current_account_value
            if residual_bucket.current_account_value is not None
            else latest_cash
        )
        period_value_source = getattr(residual_bucket, "period_start_value_source", None) or "cash"
        current_value_source = residual_bucket.current_account_value_source or "cash"

        win_rate = win_count / closed_count * 100 if closed_count else None
        avg_pnl = known_closed_pnl / closed_count if closed_count else None
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
        total_return_pct = (
            (display_current_value - display_start_value) / display_start_value * 100
            if display_start_value and display_current_value is not None
            else None
        )
        max_drawdown_usdt, max_drawdown_pct = max_drawdown(equity_points)
        first_ts = short_ts(events_sorted[0].ts_iso) if events_sorted else "-"
        last_ts = short_ts(events_sorted[-1].ts_iso) if events_sorted else "-"

        subject = f"📈 ReclaimEdge 周总结 | overall | closed={closed_count} win_rate={fmt_pct(win_rate)} pnl={total_pnl:.4f}U"
        content = f"""
<div style="font-family:Arial,Helvetica,sans-serif;line-height:1.5;color:#222;max-width:980px;">
  <h2>📈 ReclaimEdge 策略整体总结</h2>
  <p><b>统计范围：</b>{html.escape(first_ts)} ~ {html.escape(last_ts)}</p>

  <h3>账户收益</h3>
  <div style="display:flex;gap:12px;flex-wrap:wrap;margin:12px 0;">
    {metric_card('初始账户值', fmt(display_start_value, 4) + f' USDT ({period_value_source})')}
    {metric_card('最新账户值', fmt(display_current_value, 4) + f' USDT ({current_value_source})')}
    {metric_card('已记录平仓盈亏', fmt(known_closed_pnl, 4) + ' USDT')}
    {metric_card('未知/不完整汇总盈亏', fmt(residual_bucket.pnl, 4) + ' USDT')}
    {metric_card('累计估算盈亏', fmt(total_pnl, 4) + ' USDT')}
    {metric_card('累计收益率', fmt_pct(total_return_pct))}
    {metric_card('最大回撤', fmt(max_drawdown_usdt, 4) + ' USDT / ' + fmt_pct(max_drawdown_pct))}
  </div>

  <h3>策略表现</h3>
  <div style="display:flex;gap:12px;flex-wrap:wrap;margin:12px 0;">
    {metric_card('已归档平仓笔数', str(archived.closed_count))}
    {metric_card('活跃账本平仓笔数', str(active_closed_count))}
    {metric_card('总已记录平仓笔数', str(closed_count))}
    {metric_card('已归档事件数', str(archived.archived_event_count))}
    {metric_card('已归档仓位数', str(archived.archived_position_count))}
    {metric_card('当前未平仓位', str(open_count))}
    {metric_card('不完整记录数', str(incomplete_count))}
    {metric_card('胜率', fmt_pct(win_rate))}
    {metric_card('盈利/亏损/打平', f'{win_count} / {loss_count} / {breakeven_count}')}
    {metric_card('平均每笔已记录盈亏', fmt(avg_pnl, 4) + ' USDT')}
    {metric_card('Profit Factor', fmt(profit_factor, 2))}
    {metric_card('最大单笔盈利', fmt(best_win, 4) + ' USDT')}
    {metric_card('最大单笔亏损', fmt(worst_loss, 4) + ' USDT')}
  </div>

  <h3>未知/不完整记录汇总</h3>
  {self._residual_bucket_html(residual_bucket)}

  <h3>💰 账户现金变动说明</h3>
  {self._cash_events_section_html(events_sorted)}

  <h3>程序事件</h3>
  <div style="display:flex;gap:12px;flex-wrap:wrap;margin:12px 0;">
    {metric_card('总事件数', str(len(events_sorted)))}
    {metric_card('Entry 事件数', str(entry_count))}
    {metric_card('ADD 事件数', str(add_count))}
    {metric_card('TP 更新数', str(tp_update_count))}
    {metric_card('Three-Stage Runner 次数', str(three_stage_runner_count))}
    {metric_card('错误事件数', str(error_count))}
  </div>

  <p style="color:#777;font-size:12px;margin-top:16px;">
    说明：本报告基于 live_trade_summary.jsonl 的已归档 SUMMARY_SNAPSHOT 与 live_trade_events.jsonl 的活跃账本合并生成。缺少 FLAT 的历史记录不会逐条展示；如果当前总资金和起始资金可用，会把它们聚合为一个未知来源盈亏桶，避免重复累计不完整记录。当前有持仓时优先使用 equity 作为账户价值。
  </p>
</div>
""".strip()
        return subject, content

    # ------------------------------------------------------------------
    # thin wrappers — delegate to new modules, preserve backward compat
    # ------------------------------------------------------------------

    def _build_residual_bucket(
        self,
        events: list[JournalEvent],
        incomplete_count: int,
        known_closed_pnl: float,
        context: ReportRuntimeContext | None,
    ) -> ResidualPnlBucket:
        return build_residual_bucket(events, incomplete_count, known_closed_pnl, context)

    @staticmethod
    def calculate_pnl_math(
        events: list[JournalEvent],
        known_closed_pnl: float,
        context: ReportRuntimeContext | None,
    ) -> ReportPnlMath:
        return calculate_pnl_math(events, known_closed_pnl, context)

    @staticmethod
    def _net_cash_transfer(events: list[JournalEvent]) -> float:
        return net_cash_transfer(events)

    def _cash_events_section_html(self, events: list[JournalEvent]) -> str:
        return render_cash_events_section_html(events, net_cash_transfer)

    @staticmethod
    def _cash_drift_events(events: list[JournalEvent]) -> list[JournalEvent]:
        return cash_drift_events(events)

    @staticmethod
    def _cash_transfer_events(events: list[JournalEvent]) -> list[JournalEvent]:
        return cash_transfer_events(events)

    @staticmethod
    def _cash_drift_reason_label(reason: str) -> str:
        return cash_drift_reason_label(reason)

    def _residual_bucket_html(self, bucket: ResidualPnlBucket) -> str:
        return residual_bucket_html(bucket)

    @staticmethod
    def _with_period_start_cash(
        context: ReportRuntimeContext | None,
        all_events: list[JournalEvent],
        start: datetime,
    ) -> ReportRuntimeContext | None:
        if context is not None and context.period_start_cash is not None:
            return context
        inferred = infer_cash_at_or_before(all_events, start)
        if context is None:
            return ReportRuntimeContext(period_start_cash=inferred)
        return ReportRuntimeContext(
            current_position_id=context.current_position_id,
            current_has_position=context.current_has_position,
            current_cash=context.current_cash,
            current_equity=context.current_equity,
            period_start_cash=inferred,
            period_start_equity=context.period_start_equity,
        )

    @staticmethod
    def _with_overall_start_cash(
        context: ReportRuntimeContext | None, events: list[JournalEvent]
    ) -> ReportRuntimeContext | None:
        if context is not None and context.period_start_cash is not None:
            return context
        inferred = infer_first_cash(events)
        if context is None:
            return ReportRuntimeContext(period_start_cash=inferred)
        return ReportRuntimeContext(
            current_position_id=context.current_position_id,
            current_has_position=context.current_has_position,
            current_cash=context.current_cash,
            current_equity=context.current_equity,
            period_start_cash=inferred,
            period_start_equity=context.period_start_equity,
        )

    @staticmethod
    def _filter_events(events: list[JournalEvent], start: datetime, end: datetime) -> list[JournalEvent]:
        filtered: list[JournalEvent] = []
        for event in events:
            try:
                ts = datetime.fromisoformat(event.ts_iso)
            except Exception:
                continue
            if start <= ts < end:
                filtered.append(event)
        return filtered

    @staticmethod
    def _infer_cash_at_or_before(events: list[JournalEvent], target: datetime) -> float | None:
        return infer_cash_at_or_before(events, target)

    @staticmethod
    def _infer_first_cash(events: list[JournalEvent]) -> float | None:
        return infer_first_cash(events)

    @staticmethod
    def _cash_from_event(event: JournalEvent) -> float | None:
        return cash_from_event(event)

    @staticmethod
    def _cash_before_or_from_event(event: JournalEvent) -> float | None:
        return cash_before_or_from_event(event)

    @staticmethod
    def _is_current_open_position(position_id: str, context: ReportRuntimeContext | None) -> bool:
        if context is None or not context.current_has_position:
            return False
        return context.current_position_id == position_id

    @staticmethod
    def _metric_card(title: str, value: str) -> str:
        return metric_card(title, value)

    @staticmethod
    def _to_int(value: Any) -> int:
        return to_int(value)

    @staticmethod
    def _max_non_none(left: float | None, right: float | None) -> float | None:
        return max_non_none(left, right)

    @staticmethod
    def _min_non_none(left: float | None, right: float | None) -> float | None:
        return min_non_none(left, right)

    @staticmethod
    def _max_drawdown(equity_points: list[float]) -> tuple[float | None, float | None]:
        return max_drawdown(equity_points)

    # ------------------------------------------------------------------
    # HTML row helpers (thin wrappers)
    # ------------------------------------------------------------------

    def _closed_position_row(
        self,
        position_id: str,
        first_entry: JournalEvent,
        entries: list[JournalEvent],
        tp_events: list[JournalEvent],
        flat: JournalEvent,
    ) -> str:
        return closed_position_row(position_id, first_entry, entries, tp_events, flat)

    def _open_position_row(
        self,
        position_id: str,
        first_entry: JournalEvent,
        entries: list[JournalEvent],
        tp_events: list[JournalEvent],
        last_event: JournalEvent,
    ) -> str:
        return open_position_row(position_id, first_entry, entries, tp_events, last_event)

    def _render_html(
        self,
        *,
        window: DailyReportWindow,
        events: list[JournalEvent],
        finished_positions: list[tuple[str, JournalEvent, list[JournalEvent], list[JournalEvent], JournalEvent]],
        open_positions: list[tuple[str, JournalEvent, list[JournalEvent], list[JournalEvent], JournalEvent]],
        total_pnl: float,
        known_closed_pnl: float,
        total_closed: int,
        win_count: int,
        residual_bucket: ResidualPnlBucket,
    ) -> str:
        cash_html = self._cash_events_section_html(events)
        return render_daily_html(
            window=window,
            events=events,
            finished_positions=finished_positions,
            open_positions=open_positions,
            total_pnl=total_pnl,
            known_closed_pnl=known_closed_pnl,
            total_closed=total_closed,
            win_count=win_count,
            residual_bucket=residual_bucket,
            cash_events_section_html_str=cash_html,
        )
