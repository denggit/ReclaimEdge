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

    async def send_last_24h_report(self) -> bool:
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=24)
        subject, content = await asyncio.to_thread(self._build_last_24h_report_sync, start, end)
        return await self.email_sender.send_email_async(subject, content, content_type="html")

    async def send_overall_summary_report(self) -> bool:
        subject, content = await asyncio.to_thread(self._build_overall_summary_report_sync)
        return await self.email_sender.send_email_async(subject, content, content_type="html")

    def _build_last_24h_report_sync(self, start: datetime, end: datetime) -> tuple[str, str]:
        events = self.journal.load_events(start=start, end=end)
        return self.build_report(events, DailyReportWindow(start=start, end=end))

    def _build_overall_summary_report_sync(self) -> tuple[str, str]:
        events = self.journal.load_events()
        return self.build_overall_summary_report(events)

    def build_report(self, events: list[JournalEvent], window: DailyReportWindow) -> tuple[str, str]:
        grouped = group_position_events(events)
        finished_positions = []
        open_or_unknown_positions = []
        total_pnl = 0.0
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
            if last_flat:
                pnl = last_flat.payload.get("realized_pnl_usdt_est")
                if pnl is not None:
                    total_pnl += float(pnl)
                    win_count += 1 if float(pnl) > 0 else 0
                total_closed += 1
                finished_positions.append((position_id, first_entry, entry_events, tp_events, last_flat))
            else:
                open_or_unknown_positions.append((position_id, first_entry, entry_events, tp_events, last_event))

        subject = f"📊 ReclaimEdge 日报 | 最近24小时 | closed={total_closed} pnl={total_pnl:.4f}U"
        html_body = self._render_html(
            window=window,
            events=events,
            finished_positions=finished_positions,
            open_or_unknown_positions=open_or_unknown_positions,
            total_pnl=total_pnl,
            total_closed=total_closed,
            win_count=win_count,
        )
        return subject, html_body

    def build_overall_summary_report(self, events: list[JournalEvent]) -> tuple[str, str]:
        events_sorted = sorted(events, key=lambda item: item.ts_iso)
        grouped = group_position_events(events_sorted)

        closed_count = 0
        open_count = 0
        win_count = 0
        loss_count = 0
        breakeven_count = 0
        entry_count = 0
        tp_update_count = 0
        error_count = 0
        total_pnl = 0.0
        gross_profit = 0.0
        gross_loss = 0.0
        best_win: float | None = None
        worst_loss: float | None = None
        first_cash: float | None = None
        latest_cash: float | None = None
        equity_points: list[float] = []

        for event in events_sorted:
            if event.event_type == "ENTRY":
                entry_count += 1
            elif event.event_type == "TP_UPDATE":
                tp_update_count += 1
            elif event.event_type == "ERROR":
                error_count += 1

        for position_id, items in grouped.items():
            if position_id == "UNKNOWN":
                continue
            flat_events = [e for e in items if e.event_type == "FLAT"]
            if not flat_events:
                open_count += 1
                continue

            closed_count += 1
            flat = flat_events[-1]
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
            total_pnl += pnl
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

        win_rate = win_count / closed_count * 100 if closed_count else None
        avg_pnl = total_pnl / closed_count if closed_count else None
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
        total_return_pct = (latest_cash - first_cash) / first_cash * 100 if first_cash and latest_cash is not None else None
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
    {self._metric_card('累计实现盈亏', fmt(total_pnl, 4) + ' USDT')}
    {self._metric_card('累计收益率', fmt_pct(total_return_pct))}
    {self._metric_card('最大回撤', fmt(max_drawdown_usdt, 4) + ' USDT / ' + fmt_pct(max_drawdown_pct))}
  </div>

  <h3>策略表现</h3>
  <div style="display:flex;gap:12px;flex-wrap:wrap;margin:12px 0;">
    {self._metric_card('已平仓笔数', str(closed_count))}
    {self._metric_card('未平/不完整仓位', str(open_count))}
    {self._metric_card('胜率', fmt_pct(win_rate))}
    {self._metric_card('盈利/亏损/打平', f'{win_count} / {loss_count} / {breakeven_count}')}
    {self._metric_card('平均每笔盈亏', fmt(avg_pnl, 4) + ' USDT')}
    {self._metric_card('Profit Factor', fmt(profit_factor, 2))}
    {self._metric_card('最大单笔盈利', fmt(best_win, 4) + ' USDT')}
    {self._metric_card('最大单笔亏损', fmt(worst_loss, 4) + ' USDT')}
  </div>

  <h3>程序事件</h3>
  <div style="display:flex;gap:12px;flex-wrap:wrap;margin:12px 0;">
    {self._metric_card('总事件数', str(len(events_sorted)))}
    {self._metric_card('Entry 事件数', str(entry_count))}
    {self._metric_card('TP 更新数', str(tp_update_count))}
    {self._metric_card('错误事件数', str(error_count))}
  </div>

  <p style="color:#777;font-size:12px;margin-top:16px;">
    说明：本报告基于 live_trade_events.jsonl 生成。收益和回撤使用程序记录的 cash_before_position / cash_after 估算；若程序中途接管已有仓位，早期资金曲线可能不完整。
  </p>
</div>
""".strip()
        return subject, content

    @staticmethod
    def _metric_card(title: str, value: str) -> str:
        return f"<div style='padding:10px 14px;background:#f6f8fa;border-radius:8px;min-width:150px;'><b>{html.escape(title)}</b><br>{html.escape(value)}</div>"

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
        open_or_unknown_positions: list[tuple[str, JournalEvent, list[JournalEvent], list[JournalEvent], JournalEvent]],
        total_pnl: float,
        total_closed: int,
        win_count: int,
    ) -> str:
        win_rate = win_count / total_closed * 100 if total_closed else None
        rows = []
        for position_id, first_entry, entries, tp_events, flat in finished_positions:
            rows.append(self._closed_position_row(position_id, first_entry, entries, tp_events, flat))
        if not rows:
            rows.append("<tr><td colspan='12' style='padding:10px;text-align:center;color:#777;'>最近24小时没有完整平仓交易</td></tr>")

        open_rows = []
        for position_id, first_entry, entries, tp_events, last_event in open_or_unknown_positions:
            open_rows.append(self._open_position_row(position_id, first_entry, entries, tp_events, last_event))
        if not open_rows:
            open_rows.append("<tr><td colspan='8' style='padding:10px;text-align:center;color:#777;'>无未平/不完整仓位记录</td></tr>")

        return f"""
<div style="font-family:Arial,Helvetica,sans-serif;line-height:1.5;color:#222;max-width:1180px;">
  <h2>📊 ReclaimEdge 实盘日报</h2>
  <p><b>统计窗口：</b>{html.escape(window.start.astimezone().strftime('%Y-%m-%d %H:%M'))} ~ {html.escape(window.end.astimezone().strftime('%Y-%m-%d %H:%M'))}</p>

  <div style="display:flex;gap:12px;flex-wrap:wrap;margin:12px 0;">
    <div style="padding:10px 14px;background:#f6f8fa;border-radius:8px;"><b>事件数</b><br>{len(events)}</div>
    <div style="padding:10px 14px;background:#f6f8fa;border-radius:8px;"><b>已平仓笔数</b><br>{total_closed}</div>
    <div style="padding:10px 14px;background:#f6f8fa;border-radius:8px;"><b>胜率</b><br>{fmt_pct(win_rate)}</div>
    <div style="padding:10px 14px;background:#f6f8fa;border-radius:8px;"><b>估算实现盈亏</b><br>{fmt(total_pnl, 4)} USDT</div>
  </div>

  <h3>✅ 已平仓交易</h3>
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

  <h3>🟡 未平/不完整记录</h3>
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

  <p style="color:#777;font-size:12px;margin-top:16px;">说明：盈亏基于程序记录的开仓前现金余额和平仓后现金余额估算。若程序中途首次接管已有仓位，早期入场条件可能不完整。</p>
</div>
""".strip()

    def _closed_position_row(self, position_id: str, first_entry: JournalEvent, entries: list[JournalEvent], tp_events: list[JournalEvent], flat: JournalEvent) -> str:
        side = first_entry.payload.get("side")
        prices = "<br>".join(
            f"L{e.payload.get('layer_index')}: {fmt(e.payload.get('price'), 2)} / {fmt(e.payload.get('size_margin_usdt'), 3)}U ×{fmt(e.payload.get('layer_multiplier'), 2)}"
            for e in entries
        ) or "接管已有仓位"
        reasons = "<br>".join(html.escape(str(e.payload.get("reason", "-"))) for e in entries) or html.escape(str(first_entry.payload.get("note", "-")))
        last_entry = entries[-1] if entries else first_entry
        last_tp = tp_events[-1].payload.get("tp_price") if tp_events else flat.payload.get("last_tp_price")
        boll = f"M:{fmt(last_entry.payload.get('boll_middle'), 2)}<br>U:{fmt(last_entry.payload.get('boll_upper'), 2)}<br>L:{fmt(last_entry.payload.get('boll_lower'), 2)}"
        cvd = f"fast:{fmt(last_entry.payload.get('fast_cvd'), 4)}<br>buy:{fmt_pct((last_entry.payload.get('buy_ratio') or 0) * 100)}<br>sell:{fmt_pct((last_entry.payload.get('sell_ratio') or 0) * 100)}"
        return f"""
<tr>
  <td style="padding:8px;border:1px solid #ddd;">{short_ts(first_entry.ts_iso)}<br>→ {short_ts(flat.ts_iso)}</td>
  <td style="padding:8px;border:1px solid #ddd;">{html.escape(str(side or '-'))}</td>
  <td style="padding:8px;border:1px solid #ddd;text-align:center;">{flat.payload.get('layers') or len(entries)}</td>
  <td style="padding:8px;border:1px solid #ddd;">{prices}</td>
  <td style="padding:8px;border:1px solid #ddd;">{reasons}</td>
  <td style="padding:8px;border:1px solid #ddd;text-align:right;">{fmt(flat.payload.get('avg_entry_price'), 2)}</td>
  <td style="padding:8px;border:1px solid #ddd;text-align:right;">{fmt(last_tp, 2)}</td>
  <td style="padding:8px;border:1px solid #ddd;">{html.escape(str(flat.payload.get('flat_reason') or '-'))}</td>
  <td style="padding:8px;border:1px solid #ddd;text-align:right;">{fmt(flat.payload.get('realized_pnl_usdt_est'), 4)}</td>
  <td style="padding:8px;border:1px solid #ddd;text-align:right;">{fmt_pct(flat.payload.get('realized_pnl_pct_est'))}</td>
  <td style="padding:8px;border:1px solid #ddd;">{boll}</td>
  <td style="padding:8px;border:1px solid #ddd;">{cvd}</td>
</tr>
"""

    def _open_position_row(self, position_id: str, first_entry: JournalEvent, entries: list[JournalEvent], tp_events: list[JournalEvent], last_event: JournalEvent) -> str:
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
