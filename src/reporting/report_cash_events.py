from __future__ import annotations

import html

from src.reporting.report_models import _to_float, fmt, short_ts
from src.reporting.trade_journal import JournalEvent


# ---------------------------------------------------------------------------
# event filter helpers
# ---------------------------------------------------------------------------


def cash_drift_events(events: list[JournalEvent]) -> list[JournalEvent]:
    return [event for event in events if event.event_type == "ACCOUNT_CASH_DRIFT"]


def cash_transfer_events(events: list[JournalEvent]) -> list[JournalEvent]:
    return [event for event in events if event.event_type == "CASH_TRANSFER"]


# ---------------------------------------------------------------------------
# cash drift reason label
# ---------------------------------------------------------------------------


def cash_drift_reason_label(reason: str) -> str:
    """Return a human-readable Chinese label for ACCOUNT_CASH_DRIFT reason strings."""
    if not reason:
        return "未知原因"
    keywords = ["has_position", "strategy_layers", "current_position_id", "order_settle"]
    if any(kw in reason for kw in keywords):
        return "持仓/补仓/订单结算期间的可用现金变化，非转账行为"
    if "flat_settle_cooldown" in reason:
        return "平仓结算冷却期内的可用现金变化"
    return "可用现金漂移（非转账）"


# ---------------------------------------------------------------------------
# HTML section rendering for cash events
# ---------------------------------------------------------------------------


def render_cash_events_section_html(
    events: list[JournalEvent],
    net_cash_transfer_fn,
) -> str:
    """Render a cash events explanation section for reports.

    *net_cash_transfer_fn* must be a callable that accepts ``list[JournalEvent]``
    and returns a ``float`` (the net transfer amount counting only CASH_TRANSFER).
    This avoids a circular dependency on report_pnl_math.
    """
    transfer_events = cash_transfer_events(events)
    drift_events = cash_drift_events(events)

    if not transfer_events and not drift_events:
        return "<p style='color:#777;'>本周期无现金变动事件。</p>"

    rows: list[str] = []

    for event in transfer_events:
        direction = event.payload.get("direction", "-")
        amount = _to_float(event.payload.get("amount"))
        cash_before = _to_float(event.payload.get("cash_before"))
        cash_after = _to_float(event.payload.get("cash_after"))
        reason = event.payload.get("reason", "-")
        label = "真实转入" if direction == "DEPOSIT" else "真实转出"
        rows.append(
            f"<tr style='background:#e8f5e9;'>"
            f"<td style='padding:8px;border:1px solid #ddd;'>{short_ts(event.ts_iso)}</td>"
            f"<td style='padding:8px;border:1px solid #ddd;'>📥 {label}</td>"
            f"<td style='padding:8px;border:1px solid #ddd;text-align:right;'>{fmt(amount, 4)} USDT</td>"
            f"<td style='padding:8px;border:1px solid #ddd;text-align:right;'>{fmt(cash_before, 4)} → {fmt(cash_after, 4)}</td>"
            f"<td style='padding:8px;border:1px solid #ddd;'>{html.escape(str(reason))}</td>"
            f"</tr>"
        )

    for event in drift_events:
        amount = _to_float(event.payload.get("amount"))
        cash_before = _to_float(event.payload.get("cash_before"))
        cash_after = _to_float(event.payload.get("cash_after"))
        reason = event.payload.get("reason", "-")
        label = cash_drift_reason_label(reason)
        direction_text = "可用现金减少（保证金占用增加/结算漂移）" if (amount or 0) < 0 else "可用现金增加（保证金释放/结算漂移）"
        rows.append(
            f"<tr style='background:#fff8e1;'>"
            f"<td style='padding:8px;border:1px solid #ddd;'>{short_ts(event.ts_iso)}</td>"
            f"<td style='padding:8px;border:1px solid #ddd;'>⚡ 持仓中现金漂移</td>"
            f"<td style='padding:8px;border:1px solid #ddd;text-align:right;'>{fmt(amount, 4)} USDT</td>"
            f"<td style='padding:8px;border:1px solid #ddd;text-align:right;'>{fmt(cash_before, 4)} → {fmt(cash_after, 4)}</td>"
            f"<td style='padding:8px;border:1px solid #ddd;'>{html.escape(direction_text)}<br><span style='color:#888;font-size:11px;'>{html.escape(label)}</span></td>"
            f"</tr>"
        )

    net_transfer = net_cash_transfer_fn(events)
    return f"""
<table style="width:100%;border-collapse:collapse;font-size:13px;">
  <tr style="background:#f0f3f6;">
    <th style="padding:8px;border:1px solid #ddd;">时间</th>
    <th style="padding:8px;border:1px solid #ddd;">类型</th>
    <th style="padding:8px;border:1px solid #ddd;">金额</th>
    <th style="padding:8px;border:1px solid #ddd;">现金变化</th>
    <th style="padding:8px;border:1px solid #ddd;">说明</th>
  </tr>
  {''.join(rows)}
</table>
<p style='color:#777;font-size:11px;margin-top:4px;'>
  📥 真实转入/转出 = 仅统计空仓安全状态下确认的真实转账，共 {len(transfer_events)} 笔，净额 {fmt(net_transfer, 4)} USDT。<br>
  ⚡ 持仓中现金漂移 = 保证金占用/订单结算/sidecar-core 仓位同步导致的可用现金变化，非转账行为，不计入净转入/转出，共 {len(drift_events)} 笔。
</p>
""".strip()
