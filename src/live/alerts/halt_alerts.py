"""Rate-limited critical email alerts for trading halts.

This module ONLY sends emails.  It does no trading, never touches OKX,
never modifies strategy state, and never scans the journal.

All email failures are logged at ERROR level and NEVER raise.
"""

from __future__ import annotations

import html
import time
from dataclasses import dataclass, field
from typing import Any

from src.live.halt_modes import (
    FULL_HALT,
    allows_core_position_management,
    resolve_halt_mode,
)
from src.utils.log import get_logger

logger = get_logger(__name__)

# ── Deduper ────────────────────────────────────────────────────────────

_DEFAULT_DEDUP_INTERVAL_SECONDS = 600.0  # 10 minutes


class HaltAlertDeduper:
    """Rate-limit halt alert emails so the same halt doesn't spam the inbox.

    Key = symbol + position_id + halt_reason + halt_mode.
    Each key is allowed one email per dedup_interval_seconds.
    """

    def __init__(self, dedup_interval_seconds: float = _DEFAULT_DEDUP_INTERVAL_SECONDS) -> None:
        self._interval = max(1.0, dedup_interval_seconds)
        self._last_sent: dict[str, float] = {}

    def should_send(self, key: str, now_monotonic: float | None = None) -> bool:
        """Return True if an alert for *key* can be sent now."""
        now = now_monotonic if now_monotonic is not None else time.monotonic()
        last = self._last_sent.get(key)
        if last is not None and (now - last) < self._interval:
            return False
        self._last_sent[key] = now
        return True


# ── Payload ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HaltAlertPayload:
    symbol: str
    position_id: str | None
    halt_reason: str | None
    halt_mode: str
    side: str | None = None
    layer: int | None = None
    has_position: bool = False
    manual_intervention_required: bool = False
    message: str = ""
    ts_ms: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ── Core send function ──────────────────────────────────────────────────


async def send_halt_alert_once(
    *,
    email_sender: Any,  # EmailSender — duck-typed to avoid import cycle
    payload: HaltAlertPayload,
    deduper: HaltAlertDeduper,
) -> bool:
    """Send a critical halt alert email, rate-limited by deduper.

    Returns True if an email was sent (or attempted), False if suppressed
    by the deduper.  Email failures are logged and never propagate.
    """
    dedup_key = _dedup_key(payload)
    if not deduper.should_send(dedup_key):
        logger.debug("HALT_ALERT_SUPPRESSED | key=%s", dedup_key)
        return False

    subject = _build_subject(payload)
    content = _build_html(payload)

    try:
        ok = await email_sender.send_email_async(subject, content, content_type="html")
        if ok:
            logger.warning(
                "HALT_ALERT_SENT | symbol=%s position_id=%s halt_reason=%s halt_mode=%s manual_intervention=%s",
                payload.symbol,
                payload.position_id,
                payload.halt_reason,
                payload.halt_mode,
                payload.manual_intervention_required,
            )
        else:
            logger.error(
                "HALT_ALERT_SEND_FAILED | symbol=%s position_id=%s halt_reason=%s halt_mode=%s",
                payload.symbol,
                payload.position_id,
                payload.halt_reason,
                payload.halt_mode,
            )
        return ok
    except Exception as exc:
        logger.error(
            "HALT_ALERT_EXCEPTION | symbol=%s position_id=%s halt_reason=%s error=%s",
            payload.symbol,
            payload.position_id,
            payload.halt_reason,
            exc,
        )
        return False


# ── Internal helpers ────────────────────────────────────────────────────


def _dedup_key(payload: HaltAlertPayload) -> str:
    return "|".join([
        payload.symbol or "",
        payload.position_id or "",
        payload.halt_reason or "",
        payload.halt_mode or "",
    ])


def _build_subject(payload: HaltAlertPayload) -> str:
    return (
        f"[ReclaimEdge][CRITICAL] HALT {payload.symbol} "
        f"{payload.halt_reason or 'UNKNOWN'}"
    )


def _build_html(payload: HaltAlertPayload) -> str:
    core_pm = "YES" if allows_core_position_management(payload.halt_mode) else "NO"
    mode_label = {
        FULL_HALT: "FULL_HALT — all trading frozen",
        "ENTRY_HALT_POSITION_MANAGEMENT_ALLOWED": "ENTRY_HALT — position management only",
    }.get(payload.halt_mode, payload.halt_mode)

    # Build extra rows
    extra_rows = ""
    if payload.extra:
        for k, v in payload.extra.items():
            extra_rows += f"<tr><td><b>{html.escape(str(k))}</b></td><td>{html.escape(str(v))}</td></tr>\n"

    return f"""<div style="font-family:Arial,Helvetica,sans-serif;line-height:1.55;">
<h2>🚨 ReclaimEdge — CRITICAL HALT</h2>

<table style="border-collapse:collapse;width:100%%;max-width:600px;">
<tr><td style="padding:4px 8px;width:200px;"><b>symbol</b></td><td>{html.escape(str(payload.symbol))}</td></tr>
<tr><td><b>position_id</b></td><td>{html.escape(str(payload.position_id))}</td></tr>
<tr><td><b>halt_reason</b></td><td style="color:#c00;font-weight:bold;">{html.escape(str(payload.halt_reason))}</td></tr>
<tr><td><b>halt_mode</b></td><td>{html.escape(mode_label)}</td></tr>
<tr><td><b>side</b></td><td>{html.escape(str(payload.side))}</td></tr>
<tr><td><b>layer</b></td><td>{html.escape(str(payload.layer))}</td></tr>
<tr><td><b>has_position</b></td><td>{payload.has_position}</td></tr>
<tr><td><b>manual_intervention_required</b></td><td style="color:#c00;font-weight:bold;">{payload.manual_intervention_required}</td></tr>
<tr><td><b>core_position_management_allowed</b></td><td>{core_pm}</td></tr>
<tr><td><b>message</b></td><td>{html.escape(payload.message)}</td></tr>
{extra_rows}
</table>

<h3>🔧 Suggested Manual Actions</h3>
<ol>
<li>Check current OKX position (contracts, side, avg entry).</li>
<li>Check active reduce-only / algo orders on OKX.</li>
<li>Manually market-close the remaining position if risk tolerance is exceeded.</li>
</ol>

<hr>
<p style="color:#666;font-size:12px;">
This email was sent by ReclaimEdge halt alert system.  Same halt is suppressed for 10 minutes.
</p>
</div>"""
