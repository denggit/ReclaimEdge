# -*- coding: utf-8 -*-
"""Email alerts for portfolio allocator live failures."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

from src.utils.log import get_logger

if TYPE_CHECKING:
    from src.utils.email_sender import EmailSender

logger = get_logger(__name__)


def _esc(value: object) -> str:
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


async def send_allocator_commit_failed_alert(
    *,
    email_sender: "EmailSender | None",
    symbol: str | None,
    position_id: str | None,
    action: str | None,
    live_action: str | None,
    order_id: str | None,
    entry_filled: bool,
    ok: bool,
    error_type: str,
    error: str,
) -> bool:
    """Send a best-effort ledger commit failure alert."""
    if email_sender is None:
        return False

    symbol_text = _esc(symbol or "UNKNOWN")
    subject = f"🚨 ReclaimEdge Portfolio Ledger Commit Failed | {symbol_text}"
    body = f"""
<html>
  <body>
    <h2>ReclaimEdge Portfolio Ledger Commit Failed</h2>
    <table>
      <tr><th align="left">symbol</th><td>{symbol_text}</td></tr>
      <tr><th align="left">position_id</th><td>{_esc(position_id)}</td></tr>
      <tr><th align="left">allocator action</th><td>{_esc(action)}</td></tr>
      <tr><th align="left">live_result.action</th><td>{_esc(live_action)}</td></tr>
      <tr><th align="left">order_id</th><td>{_esc(order_id)}</td></tr>
      <tr><th align="left">entry_filled</th><td>{_esc(entry_filled)}</td></tr>
      <tr><th align="left">ok</th><td>{_esc(ok)}</td></tr>
      <tr><th align="left">error_type</th><td>{_esc(error_type)}</td></tr>
      <tr><th align="left">error</th><td>{_esc(error)}</td></tr>
    </table>
    <p>
      The order may already be filled. No market close was triggered.
      TP/SL/order management remains untouched. Manual ledger/account
      reconciliation is recommended.
    </p>
  </body>
</html>
""".strip()

    try:
        return bool(
            await email_sender.send_email_async(
                subject,
                body,
                content_type="html",
            )
        )
    except Exception:
        logger.exception(
            "PORTFOLIO_ALLOCATOR_COMMIT_FAILED_ALERT_SEND_FAILED | symbol=%s",
            symbol,
        )
        return False
