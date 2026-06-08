"""Tests for halt alert email module: dedup logic, payload, send flow."""

from __future__ import annotations

import time
from unittest import mock

import pytest

from src.live.alerts.halt_alerts import (
    HaltAlertDeduper,
    HaltAlertPayload,
    _build_subject,
    _dedup_key,
    send_halt_alert_once,
)
from src.live.halt_modes import (
    FULL_HALT,
    SIDECAR_DIRTY_HALT,
    resolve_halt_mode,
)


class FakeEmailSender:
    def __init__(self, fail: bool = False) -> None:
        self.sent: list[dict] = []
        self._fail = fail

    async def send_email_async(self, subject, content, content_type="html"):
        if self._fail:
            return False
        self.sent.append({"subject": subject, "content": content, "content_type": content_type})
        return True


def _payload(**overrides) -> HaltAlertPayload:
    kw = {
        "symbol": "ETH-USDT-SWAP",
        "position_id": "POS-001",
        "halt_reason": "sidecar_tp_place_failed",
        "halt_mode": resolve_halt_mode("sidecar_tp_place_failed"),
        "side": "LONG",
        "layer": 2,
        "has_position": True,
        "sidecar_dirty": True,
        "manual_intervention_required": True,
        "message": "Sidecar TP failed after core entry.",
    }
    kw.update(overrides)
    return HaltAlertPayload(**kw)


# ── Dedup tests ─────────────────────────────────────────────────────────


def test_dedup_key_includes_halt_reason_and_mode() -> None:
    p = _payload()
    key = _dedup_key(p)
    assert "ETH-USDT-SWAP" in key
    assert "POS-001" in key
    assert "sidecar_tp_place_failed" in key
    assert SIDECAR_DIRTY_HALT in key


def test_deduper_allows_first_send() -> None:
    deduper = HaltAlertDeduper(dedup_interval_seconds=600)
    assert deduper.should_send("key-A", now_monotonic=100.0) is True


def test_deduper_suppresses_within_window() -> None:
    deduper = HaltAlertDeduper(dedup_interval_seconds=600)
    assert deduper.should_send("key-A", now_monotonic=100.0) is True
    assert deduper.should_send("key-A", now_monotonic=200.0) is False  # 100s < 600s


def test_deduper_allows_after_window() -> None:
    deduper = HaltAlertDeduper(dedup_interval_seconds=600)
    assert deduper.should_send("key-A", now_monotonic=100.0) is True
    assert deduper.should_send("key-A", now_monotonic=750.0) is True  # >600s passed


def test_deduper_different_keys_independent() -> None:
    deduper = HaltAlertDeduper(dedup_interval_seconds=600)
    assert deduper.should_send("key-A", now_monotonic=100.0) is True
    assert deduper.should_send("key-B", now_monotonic=110.0) is True  # different key, allowed


# ── Subject / content tests ─────────────────────────────────────────────


def test_subject_contains_critical_and_halt_reason() -> None:
    p = _payload()
    subject = _build_subject(p)
    assert "[ReclaimEdge]" in subject
    assert "[CRITICAL]" in subject
    assert "HALT" in subject
    assert "sidecar_tp_place_failed" in subject


# ── send_halt_alert_once tests ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_halt_alert_once_sends_on_first_call() -> None:
    email = FakeEmailSender()
    deduper = HaltAlertDeduper(dedup_interval_seconds=600)
    ok = await send_halt_alert_once(
        email_sender=email, payload=_payload(), deduper=deduper,
    )
    assert ok is True
    assert len(email.sent) == 1
    assert "CRITICAL" in email.sent[0]["subject"]


@pytest.mark.asyncio
async def test_send_halt_alert_once_suppresses_duplicate() -> None:
    email = FakeEmailSender()
    deduper = HaltAlertDeduper(dedup_interval_seconds=600)
    await send_halt_alert_once(email_sender=email, payload=_payload(), deduper=deduper)
    ok = await send_halt_alert_once(email_sender=email, payload=_payload(), deduper=deduper)
    assert ok is False  # suppressed
    assert len(email.sent) == 1  # only first one sent


@pytest.mark.asyncio
async def test_send_halt_alert_once_email_failure_returns_false_no_raise() -> None:
    email = FakeEmailSender(fail=True)
    deduper = HaltAlertDeduper(dedup_interval_seconds=600)
    ok = await send_halt_alert_once(
        email_sender=email, payload=_payload(), deduper=deduper,
    )
    assert ok is False  # email failed
    assert len(email.sent) == 0


@pytest.mark.asyncio
async def test_send_halt_alert_once_email_raises_returns_false_no_raise() -> None:
    class RaisingSender:
        async def send_email_async(self, subject, content, content_type="html"):
            raise RuntimeError("SMTP connection failed")

    deduper = HaltAlertDeduper(dedup_interval_seconds=600)
    ok = await send_halt_alert_once(
        email_sender=RaisingSender(), payload=_payload(), deduper=deduper,
    )
    assert ok is False  # exception caught, no propagation


# ── resolve_halt_mode integration ───────────────────────────────────────


def test_sidecar_tp_place_failed_maps_to_sidecar_dirty() -> None:
    assert resolve_halt_mode("sidecar_tp_place_failed") == SIDECAR_DIRTY_HALT


def test_sidecar_rate_limited_unprotected_maps_to_sidecar_dirty() -> None:
    assert resolve_halt_mode("sidecar_tp_place_rate_limited_unprotected") == SIDECAR_DIRTY_HALT


def test_rolling_loss_maps_to_entry_halt() -> None:
    assert resolve_halt_mode("rolling_loss_soft_halt") == "ENTRY_HALT_POSITION_MANAGEMENT_ALLOWED"


def test_unknown_halt_maps_to_full_halt() -> None:
    assert resolve_halt_mode("totally_unknown_reason") == FULL_HALT


def test_none_halt_maps_to_full_halt() -> None:
    assert resolve_halt_mode(None) == FULL_HALT
