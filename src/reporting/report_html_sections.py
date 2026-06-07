from __future__ import annotations

import html

from src.reporting.report_models import (
    DailyReportWindow,
    ResidualPnlBucket,
    _to_float,
    fmt,
    fmt_pct,
    short_ts,
)
from src.reporting.trade_journal import JournalEvent


# ---------------------------------------------------------------------------
# metric card helper
# ---------------------------------------------------------------------------


def metric_card(title: str, value: str) -> str:
    return (
        f"<div style='padding:10px 14px;background:#f6f8fa;border-radius:8px;min-width:150px;'>"
        f"<b>{html.escape(title)}</b><br>{html.escape(value)}</div>"
    )


# ---------------------------------------------------------------------------
# residual bucket HTML
# ---------------------------------------------------------------------------


def residual_bucket_html(bucket: ResidualPnlBucket) -> str:
    if bucket.incomplete_count <= 0 and bucket.pnl in {None, 0}:
        return "<p style='color:#777;'>无不完整记录。</p>"

    note_text = html.escape(bucket.note)
    net_transfer_note = "（仅统计空仓安全状态下确认的真实转账；持仓中的 cash drift 不计入）"

    # --- build value cells with source annotations ---
    start_display = fmt(bucket.period_start_value if bucket.period_start_value is not None else bucket.cash_start, 4)
    if bucket.period_start_value is not None and bucket.cash_start is not None:
        # if period_start_value differs from cash_start (equity used), show hint
        if abs(bucket.period_start_value - bucket.cash_start) > 0.001:
            start_display += " (equity)"

    current_display = fmt(bucket.current_account_value if bucket.current_account_value is not None else bucket.cash_end, 4)
    current_source = bucket.current_account_value_source or "cash"
    if bucket.current_account_value is not None:
        current_display += f" ({current_source})"

    return f"""
<table style="width:100%;border-collapse:collapse;font-size:13px;">
  <tr style="background:#fef3c7;">
    <th style="padding:8px;border:1px solid #ddd;">记录数</th>
    <th style="padding:8px;border:1px solid #ddd;">起始账户价值</th>
    <th style="padding:8px;border:1px solid #ddd;">当前账户价值</th>
    <th style="padding:8px;border:1px solid #ddd;">净转入/转出</th>
    <th style="padding:8px;border:1px solid #ddd;">策略估算总收益</th>
    <th style="padding:8px;border:1px solid #ddd;">已记录平仓盈亏</th>
    <th style="padding:8px;border:1px solid #ddd;">未知汇总盈亏</th>
    <th style="padding:8px;border:1px solid #ddd;">公式</th>
    <th style="padding:8px;border:1px solid #ddd;">说明</th>
  </tr>
  <tr>
    <td style="padding:8px;border:1px solid #ddd;text-align:center;">{bucket.incomplete_count}</td>
    <td style="padding:8px;border:1px solid #ddd;text-align:right;">{start_display}</td>
    <td style="padding:8px;border:1px solid #ddd;text-align:right;">{current_display}</td>
    <td style="padding:8px;border:1px solid #ddd;text-align:right;">{fmt(bucket.net_transfer, 4)}<br><span style='color:#888;font-size:10px;'>{net_transfer_note}</span></td>
    <td style="padding:8px;border:1px solid #ddd;text-align:right;">{fmt(bucket.strategy_total_pnl, 4)}</td>
    <td style="padding:8px;border:1px solid #ddd;text-align:right;">{fmt(bucket.known_closed_pnl, 4)}</td>
    <td style="padding:8px;border:1px solid #ddd;text-align:right;font-weight:700;">{fmt(bucket.pnl, 4)}</td>
    <td style="padding:8px;border:1px solid #ddd;">{html.escape(bucket.formula)}</td>
    <td style="padding:8px;border:1px solid #ddd;">{note_text}</td>
  </tr>
</table>
""".strip()


# ---------------------------------------------------------------------------
# closed / open position row helpers
# ---------------------------------------------------------------------------


def closed_position_row(
    position_id: str,
    first_entry: JournalEvent,
    entries: list[JournalEvent],
    tp_events: list[JournalEvent],
    flat: JournalEvent,
) -> str:
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


def open_position_row(
    position_id: str,
    first_entry: JournalEvent,
    entries: list[JournalEvent],
    tp_events: list[JournalEvent],
    last_event: JournalEvent,
) -> str:
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


# ---------------------------------------------------------------------------
# daily report HTML renderer
# ---------------------------------------------------------------------------


def render_daily_html(
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
    cash_events_section_html_str: str,
) -> str:
    win_rate = win_count / total_closed * 100 if total_closed else None

    rows: list[str] = []
    for position_id, first_entry, entries, tp_events, flat in finished_positions:
        rows.append(closed_position_row(position_id, first_entry, entries, tp_events, flat))
    if not rows:
        rows.append(
            "<tr><td colspan='12' style='padding:10px;text-align:center;color:#777;'>最近24小时没有完整平仓交易</td></tr>")

    open_rows: list[str] = []
    for position_id, first_entry, entries, tp_events, last_event in open_positions:
        open_rows.append(open_position_row(position_id, first_entry, entries, tp_events, last_event))
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
  {residual_bucket_html(residual_bucket)}

  <h3>💰 账户现金变动说明</h3>
  {cash_events_section_html_str}

  <p style="color:#777;font-size:12px;margin-top:16px;">说明：缺少 FLAT 的历史记录不会逐条展示；如果有现金上下文，则统一汇总为 residual PnL。公式：当前账户价值 - 周期起始价值 - 净转入/转出 - 已记录平仓盈亏。当前有持仓时优先使用 equity 作为账户价值。</p>
</div>
""".strip()
