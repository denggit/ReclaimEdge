from __future__ import annotations

import asyncio
import html
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from src.reporting.trade_journal import JournalEvent, LiveTradeJournal, group_position_events
from src.utils.email_sender import EmailSender


@dataclass(frozen=True)
class DailyReportWindow:
    start: datetime
    end: datetime


@dataclass(frozen=True)
class ReportRuntimeContext:
    """Runtime account state used only for report classification.

    Reports must not mutate strategy state or journal files. The context lets us
    distinguish a real open position from stale incomplete records and lets us
    aggregate stale records into one residual PnL bucket.
    """

    current_position_id: str | None = None
    current_has_position: bool = False
    current_cash: float | None = None
    current_equity: float | None = None
    period_start_cash: float | None = None
    period_start_equity: float | None = None


@dataclass(frozen=True)
class ResidualPnlBucket:
    incomplete_count: int
    pnl: float | None
    cash_start: float | None
    cash_end: float | None
    net_transfer: float
    strategy_total_pnl: float | None
    known_closed_pnl: float
    formula: str
    note: str


@dataclass(frozen=True)
class ReportPnlMath:
    period_start_cash: float | None
    current_cash: float | None
    net_transfer: float
    known_closed_pnl: float
    strategy_total_pnl: float | None
    residual_pnl: float | None
    total_pnl: float | None


@dataclass(frozen=True)
class ArchivedSummaryStats:
    closed_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    breakeven_count: int = 0
    known_closed_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    best_win: float | None = None
    worst_loss: float | None = None
    archived_event_count: int = 0
    archived_position_count: int = 0


def fmt(value: Any, digits: int = 4, default: str = "-") -> str:
    try:
        if value is None:
            return default
        return f"{float(value):.{digits}f}"
    except Exception:
        return default


def fmt_pct(value: Any, digits: int = 2, default: str = "-") -> str:
    try:
        if value is None:
            return default
        return f"{float(value):.{digits}f}%"
    except Exception:
        return default


def short_ts(ts_iso: str) -> str:
    try:
        return datetime.fromisoformat(ts_iso).astimezone().strftime("%m-%d %H:%M:%S")
    except Exception:
        return ts_iso


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


class DailyTradeReporter:
    """Build and send HTML trade reports from LiveTradeJournal.

    Report building reads JSONL files and renders HTML. Those operations are run
    in a worker thread so scheduled reports never block the asyncio event loop
    that handles live tick processing.
    """

    def __init__(self, journal: LiveTradeJournal, email_sender: EmailSender) -> None:
        self.journal = journal
        self.email_sender = email_sender

    async def send_last_24h_report(self, context: ReportRuntimeContext | None = None) -> bool:
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=24)
        subject, content = await asyncio.to_thread(self._build_last_24h_report_sync, start, end, context)
        return await self.email_sender.send_email_async(subject, content, content_type="html")

    async def send_overall_summary_report(self, context: ReportRuntimeContext | None = None) -> bool:
        subject, content = await asyncio.to_thread(self._build_overall_summary_report_sync, context)
        return await self.email_sender.send_email_async(subject, content, content_type="html")

    def _build_last_24h_report_sync(self, start: datetime, end: datetime, context: ReportRuntimeContext | None) -> \
            tuple[str, str]:
        # Load all events once in a worker thread. The report window only displays
        # recent records, but all history can provide a best-effort starting cash
        # when context.period_start_cash was not supplied by the live runner.
        all_events = self.journal.load_events()
        events = self._filter_events(all_events, start=start, end=end)
        context = self._with_period_start_cash(context, all_events, start)
        return self.build_report(events, DailyReportWindow(start=start, end=end), context=context)

    def _build_overall_summary_report_sync(self, context: ReportRuntimeContext | None) -> tuple[str, str]:
        events = self.journal.load_events()
        archived = self.load_archived_summary_stats()
        context = self._with_overall_start_cash(context, events)
        return self.build_overall_summary_report(events, context=context, archived=archived)

    def load_archived_summary_stats(self) -> ArchivedSummaryStats:
        summary_events = self.journal.load_summary_events()
        closed_count = 0
        win_count = 0
        loss_count = 0
        breakeven_count = 0
        known_closed_pnl = 0.0
        gross_profit = 0.0
        gross_loss = 0.0
        best_win: float | None = None
        worst_loss: float | None = None
        archived_event_count = 0
        archived_position_count = 0

        for event in summary_events:
            if event.event_type != "SUMMARY_SNAPSHOT":
                continue
            payload = event.payload
            closed_count += self._to_int(payload.get("closed_count"))
            win_count += self._to_int(payload.get("win_count"))
            loss_count += self._to_int(payload.get("loss_count"))
            breakeven_count += self._to_int(payload.get("breakeven_count"))
            known_closed_pnl += _to_float(payload.get("known_closed_pnl")) or 0.0
            gross_profit += _to_float(payload.get("gross_profit")) or 0.0
            gross_loss += _to_float(payload.get("gross_loss")) or 0.0
            best_win = self._max_non_none(best_win, _to_float(payload.get("best_win")))
            worst_loss = self._min_non_none(worst_loss, _to_float(payload.get("worst_loss")))
            archived_event_count += self._to_int(payload.get("archived_event_count"))
            archived_position_count += self._to_int(payload.get("archived_position_count"))

        return ArchivedSummaryStats(
            closed_count=closed_count,
            win_count=win_count,
            loss_count=loss_count,
            breakeven_count=breakeven_count,
            known_closed_pnl=known_closed_pnl,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            best_win=best_win,
            worst_loss=worst_loss,
            archived_event_count=archived_event_count,
            archived_position_count=archived_position_count,
        )

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
        html_body = self._render_html(
            window=window,
            events=events,
            finished_positions=finished_positions,
            open_positions=open_positions,
            total_pnl=total_pnl,
            known_closed_pnl=known_closed_pnl,
            total_closed=total_closed,
            win_count=win_count,
            residual_bucket=residual_bucket,
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
        split_tp_count = 0
        three_stage_runner_count = 0
        near_tp_reduce_count = 0
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
                if event.payload.get("tp_plan") == "SPLIT_PARTIAL_FINAL":
                    split_tp_count += 1
                if event.payload.get("tp_plan") == "THREE_STAGE_RUNNER":
                    three_stage_runner_count += 1
            elif event.event_type == "TP_UPDATE":
                tp_update_count += 1
            elif event.event_type == "NEAR_TP_REDUCE":
                near_tp_reduce_count += 1
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
        best_win = self._max_non_none(archived.best_win, active_best_win)
        worst_loss = self._min_non_none(archived.worst_loss, active_worst_loss)

        residual_bucket = self._build_residual_bucket(events_sorted, incomplete_count, known_closed_pnl, context)
        total_pnl = residual_bucket.strategy_total_pnl if residual_bucket.strategy_total_pnl is not None else known_closed_pnl
        if residual_bucket.cash_start is not None:
            first_cash = residual_bucket.cash_start if first_cash is None else first_cash
        if residual_bucket.cash_end is not None:
            latest_cash = residual_bucket.cash_end
            equity_points.append(residual_bucket.cash_end)

        win_rate = win_count / closed_count * 100 if closed_count else None
        avg_pnl = known_closed_pnl / closed_count if closed_count else None
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
        total_return_pct = (
                                   latest_cash - first_cash) / first_cash * 100 if first_cash and latest_cash is not None else None
        max_drawdown_usdt, max_drawdown_pct = self._max_drawdown(equity_points)
        first_ts = short_ts(events_sorted[0].ts_iso) if events_sorted else "-"
        last_ts = short_ts(events_sorted[-1].ts_iso) if events_sorted else "-"

        subject = f"📈 ReclaimEdge 周总结 | overall | closed={closed_count} win_rate={fmt_pct(win_rate)} pnl={total_pnl:.4f}U"
        content = f"""
<div style="font-family:Arial,Helvetica,sans-serif;line-height:1.5;color:#222;max-width:980px;">
  <h2>📈 ReclaimEdge 策略整体总结</h2>
  <p><b>统计范围：</b>{html.escape(first_ts)} ~ {html.escape(last_ts)}</p>

  <h3>账户收益</h3>
  <div style="display:flex;gap:12px;flex-wrap:wrap;margin:12px 0;">
    {self._metric_card('初始现金', fmt(first_cash, 4) + ' USDT')}
    {self._metric_card('最新现金', fmt(latest_cash, 4) + ' USDT')}
    {self._metric_card('已记录平仓盈亏', fmt(known_closed_pnl, 4) + ' USDT')}
    {self._metric_card('未知/不完整汇总盈亏', fmt(residual_bucket.pnl, 4) + ' USDT')}
    {self._metric_card('累计估算盈亏', fmt(total_pnl, 4) + ' USDT')}
    {self._metric_card('累计收益率', fmt_pct(total_return_pct))}
    {self._metric_card('最大回撤', fmt(max_drawdown_usdt, 4) + ' USDT / ' + fmt_pct(max_drawdown_pct))}
  </div>

  <h3>策略表现</h3>
  <div style="display:flex;gap:12px;flex-wrap:wrap;margin:12px 0;">
    {self._metric_card('已归档平仓笔数', str(archived.closed_count))}
    {self._metric_card('活跃账本平仓笔数', str(active_closed_count))}
    {self._metric_card('总已记录平仓笔数', str(closed_count))}
    {self._metric_card('已归档事件数', str(archived.archived_event_count))}
    {self._metric_card('已归档仓位数', str(archived.archived_position_count))}
    {self._metric_card('当前未平仓位', str(open_count))}
    {self._metric_card('不完整记录数', str(incomplete_count))}
    {self._metric_card('胜率', fmt_pct(win_rate))}
    {self._metric_card('盈利/亏损/打平', f'{win_count} / {loss_count} / {breakeven_count}')}
    {self._metric_card('平均每笔已记录盈亏', fmt(avg_pnl, 4) + ' USDT')}
    {self._metric_card('Profit Factor', fmt(profit_factor, 2))}
    {self._metric_card('最大单笔盈利', fmt(best_win, 4) + ' USDT')}
    {self._metric_card('最大单笔亏损', fmt(worst_loss, 4) + ' USDT')}
  </div>

  <h3>未知/不完整记录汇总</h3>
  {self._residual_bucket_html(residual_bucket)}

  <h3>程序事件</h3>
  <div style="display:flex;gap:12px;flex-wrap:wrap;margin:12px 0;">
    {self._metric_card('总事件数', str(len(events_sorted)))}
    {self._metric_card('Entry 事件数', str(entry_count))}
    {self._metric_card('ADD 事件数', str(add_count))}
    {self._metric_card('TP 更新数', str(tp_update_count))}
    {self._metric_card('Split TP 次数', str(split_tp_count))}
    {self._metric_card('Three-Stage Runner 次数', str(three_stage_runner_count))}
    {self._metric_card('Near-TP 减仓数', str(near_tp_reduce_count))}
    {self._metric_card('错误事件数', str(error_count))}
  </div>

  <p style="color:#777;font-size:12px;margin-top:16px;">
    说明：本报告基于 live_trade_summary.jsonl 的已归档 SUMMARY_SNAPSHOT 与 live_trade_events.jsonl 的活跃账本合并生成。缺少 FLAT 的历史记录不会逐条展示；如果当前总资金和起始资金可用，会把它们聚合为一个未知来源盈亏桶，避免重复累计不完整记录。
  </p>
</div>
""".strip()
        return subject, content

    def _build_residual_bucket(
            self,
            events: list[JournalEvent],
            incomplete_count: int,
            known_closed_pnl: float,
            context: ReportRuntimeContext | None,
    ) -> ResidualPnlBucket:
        math = self.calculate_pnl_math(events, known_closed_pnl, context)
        formula = "current_cash - period_start_cash - net_transfer - known_closed_pnl"
        if context is None or context.current_cash is None or context.period_start_cash is None:
            return ResidualPnlBucket(
                incomplete_count,
                None,
                context.period_start_cash if context else None,
                context.current_cash if context else None,
                math.net_transfer,
                None,
                known_closed_pnl,
                formula,
                "missing cash context; incomplete records hidden but not valued",
            )
        if incomplete_count <= 0 and math.residual_pnl == 0:
            note = "no incomplete records"
        elif incomplete_count <= 0:
            note = "no incomplete records; residual shows cash-based unaccounted strategy PnL"
        else:
            note = "incomplete records are bucketed, not displayed per position"
        return ResidualPnlBucket(
            incomplete_count,
            math.residual_pnl,
            context.period_start_cash,
            context.current_cash,
            math.net_transfer,
            math.strategy_total_pnl,
            known_closed_pnl,
            formula,
            note,
        )

    @staticmethod
    def calculate_pnl_math(
            events: list[JournalEvent],
            known_closed_pnl: float,
            context: ReportRuntimeContext | None,
    ) -> ReportPnlMath:
        net_transfer = DailyTradeReporter._net_cash_transfer(events)
        period_start_cash = context.period_start_cash if context else None
        current_cash = context.current_cash if context else None
        strategy_total_pnl = None
        residual_pnl = None
        total_pnl = None
        if current_cash is not None and period_start_cash is not None:
            strategy_total_pnl = current_cash - period_start_cash - net_transfer
            residual_pnl = strategy_total_pnl - known_closed_pnl
            total_pnl = strategy_total_pnl
        return ReportPnlMath(
            period_start_cash=period_start_cash,
            current_cash=current_cash,
            net_transfer=net_transfer,
            known_closed_pnl=known_closed_pnl,
            strategy_total_pnl=strategy_total_pnl,
            residual_pnl=residual_pnl,
            total_pnl=total_pnl,
        )

    @staticmethod
    def _net_cash_transfer(events: list[JournalEvent]) -> float:
        total = 0.0
        for event in events:
            if event.event_type != "CASH_TRANSFER":
                continue
            amount = _to_float(event.payload.get("amount"))
            if amount is not None:
                total += amount
        return total

    def _residual_bucket_html(self, bucket: ResidualPnlBucket) -> str:
        if bucket.incomplete_count <= 0 and bucket.pnl in {None, 0}:
            return "<p style='color:#777;'>无不完整记录。</p>"
        return f"""
<table style="width:100%;border-collapse:collapse;font-size:13px;">
  <tr style="background:#fef3c7;">
    <th style="padding:8px;border:1px solid #ddd;">记录数</th>
    <th style="padding:8px;border:1px solid #ddd;">起始现金</th>
    <th style="padding:8px;border:1px solid #ddd;">当前现金</th>
    <th style="padding:8px;border:1px solid #ddd;">净转入/转出</th>
    <th style="padding:8px;border:1px solid #ddd;">策略估算总收益</th>
    <th style="padding:8px;border:1px solid #ddd;">已记录平仓盈亏</th>
    <th style="padding:8px;border:1px solid #ddd;">未知汇总盈亏</th>
    <th style="padding:8px;border:1px solid #ddd;">公式</th>
    <th style="padding:8px;border:1px solid #ddd;">说明</th>
  </tr>
  <tr>
    <td style="padding:8px;border:1px solid #ddd;text-align:center;">{bucket.incomplete_count}</td>
    <td style="padding:8px;border:1px solid #ddd;text-align:right;">{fmt(bucket.cash_start, 4)}</td>
    <td style="padding:8px;border:1px solid #ddd;text-align:right;">{fmt(bucket.cash_end, 4)}</td>
    <td style="padding:8px;border:1px solid #ddd;text-align:right;">{fmt(bucket.net_transfer, 4)}</td>
    <td style="padding:8px;border:1px solid #ddd;text-align:right;">{fmt(bucket.strategy_total_pnl, 4)}</td>
    <td style="padding:8px;border:1px solid #ddd;text-align:right;">{fmt(bucket.known_closed_pnl, 4)}</td>
    <td style="padding:8px;border:1px solid #ddd;text-align:right;font-weight:700;">{fmt(bucket.pnl, 4)}</td>
    <td style="padding:8px;border:1px solid #ddd;">{html.escape(bucket.formula)}</td>
    <td style="padding:8px;border:1px solid #ddd;">{html.escape(bucket.note)}</td>
  </tr>
</table>
""".strip()

    @staticmethod
    def _with_period_start_cash(
            context: ReportRuntimeContext | None,
            all_events: list[JournalEvent],
            start: datetime,
    ) -> ReportRuntimeContext | None:
        if context is not None and context.period_start_cash is not None:
            return context
        inferred = DailyTradeReporter._infer_cash_at_or_before(all_events, start)
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
    def _with_overall_start_cash(context: ReportRuntimeContext | None,
                                 events: list[JournalEvent]) -> ReportRuntimeContext | None:
        if context is not None and context.period_start_cash is not None:
            return context
        inferred = DailyTradeReporter._infer_first_cash(events)
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
        best_ts: datetime | None = None
        best_cash: float | None = None
        for event in events:
            try:
                ts = datetime.fromisoformat(event.ts_iso)
            except Exception:
                continue
            if ts > target:
                continue
            cash = DailyTradeReporter._cash_from_event(event)
            if cash is None:
                continue
            if best_ts is None or ts >= best_ts:
                best_ts = ts
                best_cash = cash
        return best_cash

    @staticmethod
    def _infer_first_cash(events: list[JournalEvent]) -> float | None:
        best_ts: datetime | None = None
        best_cash: float | None = None
        for event in events:
            try:
                ts = datetime.fromisoformat(event.ts_iso)
            except Exception:
                continue
            cash = DailyTradeReporter._cash_before_or_from_event(event)
            if cash is None:
                continue
            if best_ts is None or ts < best_ts:
                best_ts = ts
                best_cash = cash
        return best_cash

    @staticmethod
    def _cash_from_event(event: JournalEvent) -> float | None:
        if event.event_type == "CASH_BASELINE":
            return _to_float(event.payload.get("cash"))
        if event.event_type == "CASH_TRANSFER":
            return _to_float(event.payload.get("cash_after"))
        if event.event_type == "FLAT":
            return _to_float(event.payload.get("cash_after"))
        if event.event_type == "ENTRY":
            return _to_float(event.payload.get("cash_before_position"))
        if event.event_type == "STARTUP_RECOVERY":
            return _to_float(event.payload.get("cash"))
        for key in ("cash_after", "cash_before_position", "cash"):
            value = _to_float(event.payload.get(key))
            if value is not None:
                return value
        return None

    @staticmethod
    def _cash_before_or_from_event(event: JournalEvent) -> float | None:
        if event.event_type == "CASH_BASELINE":
            return _to_float(event.payload.get("cash"))
        for key in ("cash_before_position", "cash", "cash_after"):
            value = _to_float(event.payload.get(key))
            if value is not None:
                return value
        return None

    @staticmethod
    def _is_current_open_position(position_id: str, context: ReportRuntimeContext | None) -> bool:
        if context is None or not context.current_has_position:
            return False
        return context.current_position_id == position_id

    @staticmethod
    def _metric_card(title: str, value: str) -> str:
        return f"<div style='padding:10px 14px;background:#f6f8fa;border-radius:8px;min-width:150px;'><b>{html.escape(title)}</b><br>{html.escape(value)}</div>"

    @staticmethod
    def _to_int(value: Any) -> int:
        number = _to_float(value)
        return int(number) if number is not None else 0

    @staticmethod
    def _max_non_none(left: float | None, right: float | None) -> float | None:
        if left is None:
            return right
        if right is None:
            return left
        return max(left, right)

    @staticmethod
    def _min_non_none(left: float | None, right: float | None) -> float | None:
        if left is None:
            return right
        if right is None:
            return left
        return min(left, right)

    @staticmethod
    def _max_drawdown(equity_points: list[float]) -> tuple[float | None, float | None]:
        if len(equity_points) < 2:
            return None, None
        peak = equity_points[0]
        max_dd = 0.0
        max_dd_pct = 0.0
        for equity in equity_points:
            if equity > peak:
                peak = equity
            drawdown = peak - equity
            drawdown_pct = drawdown / peak * 100 if peak > 0 else 0.0
            if drawdown > max_dd:
                max_dd = drawdown
                max_dd_pct = drawdown_pct
        return max_dd, max_dd_pct

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
        win_rate = win_count / total_closed * 100 if total_closed else None
        rows = []
        for position_id, first_entry, entries, tp_events, flat in finished_positions:
            rows.append(self._closed_position_row(position_id, first_entry, entries, tp_events, flat))
        if not rows:
            rows.append(
                "<tr><td colspan='12' style='padding:10px;text-align:center;color:#777;'>最近24小时没有完整平仓交易</td></tr>")

        open_rows = []
        for position_id, first_entry, entries, tp_events, last_event in open_positions:
            open_rows.append(self._open_position_row(position_id, first_entry, entries, tp_events, last_event))
        if not open_rows:
            open_rows.append(
                "<tr><td colspan='8' style='padding:10px;text-align:center;color:#777;'>当前无未平仓位</td></tr>")

        return f"""
<div style="font-family:Arial,Helvetica,sans-serif;line-height:1.5;color:#222;max-width:1180px;">
  <h2>📊 ReclaimEdge 实盘日报</h2>
  <p><b>统计窗口：</b>{html.escape(window.start.astimezone().strftime('%Y-%m-%d %H:%M'))} ~ {html.escape(window.end.astimezone().strftime('%Y-%m-%d %H:%M'))}</p>

  <div style="display:flex;gap:12px;flex-wrap:wrap;margin:12px 0;">
    <div style="padding:10px 14px;background:#f6f8fa;border-radius:8px;"><b>事件数</b><br>{len(events)}</div>
    <div style="padding:10px 14px;background:#f6f8fa;border-radius:8px;"><b>已记录平仓笔数</b><br>{total_closed}</div>
    <div style="padding:10px 14px;background:#f6f8fa;border-radius:8px;"><b>不完整记录数</b><br>{residual_bucket.incomplete_count}</div>
    <div style="padding:10px 14px;background:#f6f8fa;border-radius:8px;"><b>胜率</b><br>{fmt_pct(win_rate)}</div>
    <div style="padding:10px 14px;background:#f6f8fa;border-radius:8px;"><b>已记录平仓盈亏</b><br>{fmt(known_closed_pnl, 4)} USDT</div>
    <div style="padding:10px 14px;background:#f6f8fa;border-radius:8px;"><b>未知/不完整汇总盈亏</b><br>{fmt(residual_bucket.pnl, 4)} USDT</div>
    <div style="padding:10px 14px;background:#f6f8fa;border-radius:8px;"><b>估算总盈亏</b><br>{fmt(total_pnl, 4)} USDT</div>
  </div>

  <h3>✅ 已记录平仓交易</h3>
  <table style="width:100%;border-collapse:collapse;font-size:13px;">
    <tr style="background:#f0f3f6;">
      <th style="padding:8px;border:1px solid #ddd;">时间</th>
      <th style="padding:8px;border:1px solid #ddd;">方向</th>
      <th style="padding:8px;border:1px solid #ddd;">层数</th>
      <th style="padding:8px;border:1px solid #ddd;">开/补仓价格</th>
      <th style="padding:8px;border:1px solid #ddd;">加仓条件</th>
      <th style="padding:8px;border:1px solid #ddd;">均价</th>
      <th style="padding:8px;border:1px solid #ddd;">最后TP</th>
      <th style="padding:8px;border:1px solid #ddd;">平仓原因</th>
      <th style="padding:8px;border:1px solid #ddd;">盈亏U</th>
      <th style="padding:8px;border:1px solid #ddd;">盈亏%</th>
      <th style="padding:8px;border:1px solid #ddd;">BOLL条件</th>
      <th style="padding:8px;border:1px solid #ddd;">CVD条件</th>
    </tr>
    {''.join(rows)}
  </table>

  <h3>🟡 当前未平仓位</h3>
  <table style="width:100%;border-collapse:collapse;font-size:13px;">
    <tr style="background:#fff4ce;">
      <th style="padding:8px;border:1px solid #ddd;">时间</th>
      <th style="padding:8px;border:1px solid #ddd;">方向</th>
      <th style="padding:8px;border:1px solid #ddd;">层数</th>
      <th style="padding:8px;border:1px solid #ddd;">价格</th>
      <th style="padding:8px;border:1px solid #ddd;">当前TP</th>
      <th style="padding:8px;border:1px solid #ddd;">原因</th>
      <th style="padding:8px;border:1px solid #ddd;">状态</th>
      <th style="padding:8px;border:1px solid #ddd;">position_id</th>
    </tr>
    {''.join(open_rows)}
  </table>

  <h3>🧩 未知/不完整记录汇总</h3>
  {self._residual_bucket_html(residual_bucket)}

  <p style="color:#777;font-size:12px;margin-top:16px;">说明：缺少 FLAT 的历史记录不会逐条展示；如果有现金上下文，则统一汇总为 residual PnL。公式：当前现金 - 周期起始现金 - 净转入/转出 - 已记录平仓盈亏。</p>
</div>
""".strip()

    def _closed_position_row(self, position_id: str, first_entry: JournalEvent, entries: list[JournalEvent],
                             tp_events: list[JournalEvent], flat: JournalEvent) -> str:
        side = first_entry.payload.get("side")
        prices = "<br>".join(
            f"L{e.payload.get('layer_index')}: {fmt(e.payload.get('price'), 2)} / {fmt(e.payload.get('size_margin_usdt'), 3)}U ×{fmt(e.payload.get('layer_multiplier'), 2)}"
            for e in entries
        ) or "接管已有仓位"
        reasons = "<br>".join(html.escape(str(e.payload.get("reason", "-"))) for e in entries) or html.escape(
            str(first_entry.payload.get("note", "-")))
        last_entry = entries[-1] if entries else first_entry
        last_tp = tp_events[-1].payload.get("tp_price") if tp_events else flat.payload.get("last_tp_price")
        boll = f"M:{fmt(last_entry.payload.get('boll_middle'), 2)}<br>U:{fmt(last_entry.payload.get('boll_upper'), 2)}<br>L:{fmt(last_entry.payload.get('boll_lower'), 2)}"
        cvd = f"fast:{fmt(last_entry.payload.get('fast_cvd'), 4)}<br>buy:{fmt_pct((last_entry.payload.get('buy_ratio') or 0) * 100)}<br>sell:{fmt_pct((last_entry.payload.get('sell_ratio') or 0) * 100)}"
        exit_reason = flat.payload.get("trend_runner_exit_reason") or flat.payload.get("flat_reason") or "-"
        return f"""
<tr>
  <td style="padding:8px;border:1px solid #ddd;">{short_ts(first_entry.ts_iso)}<br>→ {short_ts(flat.ts_iso)}</td>
  <td style="padding:8px;border:1px solid #ddd;">{html.escape(str(side or '-'))}</td>
  <td style="padding:8px;border:1px solid #ddd;text-align:center;">{flat.payload.get('layers') or len(entries)}</td>
  <td style="padding:8px;border:1px solid #ddd;">{prices}</td>
  <td style="padding:8px;border:1px solid #ddd;">{reasons}</td>
  <td style="padding:8px;border:1px solid #ddd;text-align:right;">{fmt(flat.payload.get('avg_entry_price'), 2)}</td>
  <td style="padding:8px;border:1px solid #ddd;text-align:right;">{fmt(last_tp, 2)}</td>
  <td style="padding:8px;border:1px solid #ddd;">{html.escape(str(exit_reason))}</td>
  <td style="padding:8px;border:1px solid #ddd;text-align:right;">{fmt(flat.payload.get('realized_pnl_usdt_est'), 4)}</td>
  <td style="padding:8px;border:1px solid #ddd;text-align:right;">{fmt_pct(flat.payload.get('realized_pnl_pct_est'))}</td>
  <td style="padding:8px;border:1px solid #ddd;">{boll}</td>
  <td style="padding:8px;border:1px solid #ddd;">{cvd}</td>
</tr>
"""

    def _open_position_row(self, position_id: str, first_entry: JournalEvent, entries: list[JournalEvent],
                           tp_events: list[JournalEvent], last_event: JournalEvent) -> str:
        latest = entries[-1] if entries else last_event
        last_tp = tp_events[-1].payload.get("tp_price") if tp_events else latest.payload.get("tp_price")
        reason = latest.payload.get("reason") or latest.payload.get("note") or "-"
        return f"""
<tr>
  <td style="padding:8px;border:1px solid #ddd;">{short_ts(first_entry.ts_iso)}</td>
  <td style="padding:8px;border:1px solid #ddd;">{html.escape(str(latest.payload.get('side') or '-'))}</td>
  <td style="padding:8px;border:1px solid #ddd;text-align:center;">{latest.payload.get('layer_index') or len(entries) or '-'}</td>
  <td style="padding:8px;border:1px solid #ddd;text-align:right;">{fmt(latest.payload.get('price') or latest.payload.get('avg_entry'), 2)}</td>
  <td style="padding:8px;border:1px solid #ddd;text-align:right;">{fmt(last_tp, 2)}</td>
  <td style="padding:8px;border:1px solid #ddd;">{html.escape(str(reason))}</td>
  <td style="padding:8px;border:1px solid #ddd;">{html.escape(last_event.event_type)}</td>
  <td style="padding:8px;border:1px solid #ddd;font-size:11px;color:#777;">{html.escape(position_id)}</td>
</tr>
"""
