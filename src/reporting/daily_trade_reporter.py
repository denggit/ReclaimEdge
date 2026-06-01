from __future__ import annotations

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


class DailyTradeReporter:
    """Build and send a daily HTML trade report from LiveTradeJournal."""

    def __init__(self, journal: LiveTradeJournal, email_sender: EmailSender) -> None:
        self.journal = journal
        self.email_sender = email_sender

    async def send_last_24h_report(self) -> bool:
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=24)
        events = self.journal.load_events(start=start, end=end)
        subject, content = self.build_report(events, DailyReportWindow(start=start, end=end))
        return await self.email_sender.send_email_async(subject, content, content_type="html")

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
