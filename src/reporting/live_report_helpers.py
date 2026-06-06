from __future__ import annotations

import html

from src.live import time_utils as live_time_utils
from src.live.runtime_types import AccountSnapshot, ExecutionState
from src.reporting.daily_trade_reporter import ReportRuntimeContext
from src.strategies.boll_cvd_reclaim_strategy import TradeIntent


def build_report_context(
        *,
        account_snapshot: AccountSnapshot,
        execution_state: ExecutionState,
        period_start_cash: float | None = None,
        period_start_equity: float | None = None,
) -> ReportRuntimeContext:
    position = account_snapshot.position
    return ReportRuntimeContext(
        current_position_id=execution_state.current_position_id,
        current_has_position=bool(position and position.has_position),
        current_cash=account_snapshot.cash,
        current_equity=account_snapshot.equity,
        period_start_cash=period_start_cash,
        period_start_equity=period_start_equity,
    )


def build_live_failure_email(intent: TradeIntent, error: Exception, rolled_back: bool, halted: bool) -> tuple[str, str]:
    subject = f"LIVE order failed | ETH-USDT-SWAP | {intent.intent_type} | layer {intent.layer_index}"
    event_time = live_time_utils.format_ts_ms(intent.ts_ms)
    state_text = "Strategy state has been rolled back." if rolled_back else "Entry may be live. Strategy state was NOT rolled back."
    halt_text = "Trading has been halted. Please check OKX manually." if halted else "Trading is not halted."
    content = f"""
<div style="font-family: Arial, Helvetica, sans-serif; line-height: 1.55; color: #222; max-width: 760px;">
  <h2>LIVE order failed</h2>
  <p><strong>{html.escape(state_text)}</strong></p>
  <p><strong>{html.escape(halt_text)}</strong></p>
  <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">intent_type</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{html.escape(intent.intent_type)}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">side</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{html.escape(intent.side)}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">layer</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.layer_index}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">price</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.price:.4f}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">tp_price</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.tp_price:.4f}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">partial_tp_price</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{getattr(intent, 'partial_tp_price', None) or '-'}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">tp_plan</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{html.escape(getattr(intent, 'tp_plan', 'SINGLE'))}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">tp_mode</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{html.escape(intent.tp_mode)}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">avg_entry</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.avg_entry_price:.4f}</td></tr>
    <tr><td style="padding: 8px; border-bottom: 1px solid #eee;">breakeven</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: right;">{intent.breakeven_price:.4f}</td></tr>
  </table>
  <p><strong>Reason:</strong> {html.escape(intent.reason)}</p>
  <p><strong>Error:</strong> {html.escape(str(error))}</p>
  <p><strong>Event time:</strong> {html.escape(event_time)}</p>
</div>
""".strip()
    return subject, content
