from __future__ import annotations

import html
import time
from dataclasses import dataclass
from typing import Any, Protocol

from src.live.supervisor.alert_deduper import AlertDeduper
from src.live.supervisor.alert_policy import AlertPolicy
from src.live.supervisor.child_event_reader import (
    ChildEvent,
    ChildEventReadError,
    ChildEventReadResult,
    ChildEventReader,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LIFECYCLE_EVENT_TYPES = frozenset(
    {
        "WORKER_STARTED",
        "WORKER_STOPPING",
        "WORKER_STOPPED",
        "WORKER_STARTUP_RECOVERY_COMPLETED",
        "WORKER_STARTUP_RECOVERY_FAILED",
        "WORKER_TRADING_HALTED",
        "WORKER_HEARTBEAT_WRITE_FAILED",
        "WORKER_DRAIN_STARTED",
        "WORKER_DRAIN_COMPLETED",
        "WORKER_DRAIN_TIMEOUT",
    }
)

READ_ERROR_EVENT_TYPE = "CHILD_EVENT_READ_FAILED"

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SupervisorAlert:
    """A lifecycle / health alert built by the pipeline for publishing.

    Designed to carry only the minimum fields needed for email / alerting —
    never a full child payload.
    """

    symbol: str
    event_type: str
    severity: str
    reason: str | None
    subject: str
    body: str
    content_type: str = "html"
    source_path: str | None = None
    ts_ms: int | None = None


@dataclass(frozen=True)
class SupervisorEventPipelineResult:
    """Summary of one :meth:`SupervisorEventPipeline.process_once` call."""

    events_seen: int
    read_errors_seen: int
    alerts_built: int
    alerts_policy_suppressed: int
    alerts_allowed: int
    alerts_suppressed: int
    alerts_published: int
    publish_failures: int
    dropped_due_to_cycle_limit: int


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class SupervisorAlertPublisher(Protocol):
    """Protocol for publishing a :class:`SupervisorAlert`.

    Implementations must return ``True`` on success, ``False`` on failure.
    They may raise on unrecoverable errors; the pipeline catches exceptions
    and counts them as publish failures.
    """

    async def publish_alert(self, alert: SupervisorAlert) -> bool: ...


# ---------------------------------------------------------------------------
# SupervisorEventPipeline
# ---------------------------------------------------------------------------


class SupervisorEventPipeline:
    """Reads child lifecycle events, deduplicates, and publishes alerts.

    Orchestrates the supervisor-side event processing chain::

        ChildEventReader
        → SupervisorEventPipeline
            → AlertPolicy
            → AlertDeduper
            → Publisher

    This is a **supervisor control-plane** tool.  It must never be used
    inside the tick / trading path.

    Parameters
    ----------
    reader : ChildEventReader
        Reader that yields child events and read errors from JSONL outboxes.
    deduper : AlertDeduper
        Cooldown-based deduplication for alerts.
    publisher : SupervisorAlertPublisher
        Alert publisher (e.g. email adapter).  Test doubles inject a fake.
    alert_policy : AlertPolicy | None
        Policy that decides which alerts are critical enough to publish.
        ``None`` defaults to ``AlertPolicy()``.  Any object with a
        ``should_publish`` method is accepted (duck-typing for testing).
    max_alerts_per_cycle : int
        Maximum number of alerts to process per call (default 100).
    """

    def __init__(
        self,
        *,
        reader: ChildEventReader,
        deduper: AlertDeduper,
        publisher: SupervisorAlertPublisher,
        alert_policy: AlertPolicy | None = None,
        max_alerts_per_cycle: int = 100,
    ) -> None:
        # -- reader -----------------------------------------------------------
        if not hasattr(reader, "read_new_events"):
            raise ValueError(
                f"reader must have 'read_new_events' attribute, "
                f"got {type(reader).__name__}"
            )

        # -- deduper ----------------------------------------------------------
        if not hasattr(deduper, "should_send"):
            raise ValueError(
                f"deduper must have 'should_send' attribute, "
                f"got {type(deduper).__name__}"
            )

        # -- publisher --------------------------------------------------------
        if not hasattr(publisher, "publish_alert"):
            raise ValueError(
                f"publisher must have 'publish_alert' attribute, "
                f"got {type(publisher).__name__}"
            )

        # -- alert_policy -----------------------------------------------------
        if alert_policy is None:
            alert_policy = AlertPolicy()
        elif not hasattr(alert_policy, "should_publish"):
            raise ValueError(
                f"alert_policy must have 'should_publish' attribute, "
                f"got {type(alert_policy).__name__}"
            )

        # -- max_alerts_per_cycle ---------------------------------------------
        if type(max_alerts_per_cycle) is not int:
            raise ValueError(
                f"max_alerts_per_cycle must be int, "
                f"got {type(max_alerts_per_cycle).__name__}={max_alerts_per_cycle!r}"
            )
        if max_alerts_per_cycle <= 0:
            raise ValueError(
                f"max_alerts_per_cycle must be > 0, got {max_alerts_per_cycle}"
            )

        self._reader = reader
        self._deduper = deduper
        self._publisher = publisher
        self._alert_policy = alert_policy
        self._max_alerts_per_cycle = max_alerts_per_cycle

    # ------------------------------------------------------------------
    # process_once
    # ------------------------------------------------------------------

    async def process_once(
        self,
        *,
        now_ms: int | None = None,
    ) -> SupervisorEventPipelineResult:
        """Run one cycle of the event pipeline.

        Reads new events from the child outbox, builds lifecycle / read-error
        alerts, deduplicates them, and publishes allowed alerts via the
        injected publisher.

        Candidate alerts are streamed — when the cycle limit is reached
        remaining candidates are counted as dropped without building full
        ``SupervisorAlert`` objects or their HTML bodies.

        Parameters
        ----------
        now_ms : int | None
            Epoch milliseconds used for deduplication decisions.  When
            ``None``, generated internally via ``int(time.time() * 1000)``.
            The same value is used for all alerts in this cycle.

        Returns
        -------
        SupervisorEventPipelineResult
        """
        # -- validate now_ms --------------------------------------------------
        if now_ms is None:
            now_ms = int(time.time() * 1000)
        else:
            if type(now_ms) is not int:
                raise ValueError(
                    f"now_ms must be int, got {type(now_ms).__name__}={now_ms!r}"
                )
            if now_ms < 0:
                raise ValueError(f"now_ms must be >= 0, got {now_ms}")

        # -- read -------------------------------------------------------------
        result: ChildEventReadResult = self._reader.read_new_events()

        events_seen = len(result.events)
        read_errors_seen = len(result.errors)

        # -- counters ---------------------------------------------------------
        alerts_built = 0
        alerts_policy_suppressed = 0
        alerts_allowed = 0
        alerts_suppressed = 0
        alerts_published = 0
        publish_failures = 0
        dropped_due_to_cycle_limit = 0
        processed = 0

        # -- stream lifecycle events ------------------------------------------
        for event in result.events:
            if event.event_type not in LIFECYCLE_EVENT_TYPES:
                continue
            alerts_built += 1

            # -- policy check (before consuming limit) ------------------------
            severity = _extract_severity(event.payload)
            reason = _extract_reason(event.payload)
            policy_decision = self._alert_policy.should_publish(
                event_type=event.event_type,
                severity=severity,
                reason=reason,
            )
            if not policy_decision.allowed:
                alerts_policy_suppressed += 1
                continue

            # -- cycle limit (only for policy-allowed alerts) -----------------
            if processed >= self._max_alerts_per_cycle:
                dropped_due_to_cycle_limit += 1
                continue
            processed += 1

            alert = _build_alert_from_child_event(event)
            allowed, suppressed, failed = await self._dedupe_and_publish_alert(
                alert, now_ms
            )
            if suppressed:
                alerts_suppressed += 1
                continue
            if allowed:
                alerts_allowed += 1
            if failed:
                publish_failures += 1
                continue
            alerts_published += 1

        # -- stream read errors -----------------------------------------------
        for error in result.errors:
            alerts_built += 1

            # -- policy check (before consuming limit) ------------------------
            policy_decision = self._alert_policy.should_publish(
                event_type=READ_ERROR_EVENT_TYPE,
                severity="ERROR",
                reason=error.error_type,
            )
            if not policy_decision.allowed:
                alerts_policy_suppressed += 1
                continue

            # -- cycle limit (only for policy-allowed alerts) -----------------
            if processed >= self._max_alerts_per_cycle:
                dropped_due_to_cycle_limit += 1
                continue
            processed += 1

            alert = _build_alert_from_read_error(error)
            allowed, suppressed, failed = await self._dedupe_and_publish_alert(
                alert, now_ms
            )
            if suppressed:
                alerts_suppressed += 1
                continue
            if allowed:
                alerts_allowed += 1
            if failed:
                publish_failures += 1
                continue
            alerts_published += 1

        return SupervisorEventPipelineResult(
            events_seen=events_seen,
            read_errors_seen=read_errors_seen,
            alerts_built=alerts_built,
            alerts_policy_suppressed=alerts_policy_suppressed,
            alerts_allowed=alerts_allowed,
            alerts_suppressed=alerts_suppressed,
            alerts_published=alerts_published,
            publish_failures=publish_failures,
            dropped_due_to_cycle_limit=dropped_due_to_cycle_limit,
        )

    # ------------------------------------------------------------------
    # Internal: dedupe + publish
    # ------------------------------------------------------------------

    async def _dedupe_and_publish_alert(
        self,
        alert: SupervisorAlert,
        now_ms: int,
    ) -> tuple[bool, bool, bool]:
        """Dedupe and publish a single alert.

        Returns ``(allowed, suppressed, failed)``:

        * ``(False, True, False)`` — deduper suppressed the alert.
        * ``(True, False, False)`` — deduper allowed and publisher succeeded.
        * ``(True, False, True)`` — deduper allowed but publisher failed
          (returned ``False`` or raised).

        Dedupe state is recorded **only after** a successful publish.  A
        non-mutating ``record_send=False`` call is used first; if
        ``publish_alert`` returns ``True``, a second ``record_send=True``
        call persists the sent timestamp.
        """
        # Phase 1: non-mutating dedupe check.
        decision = self._deduper.should_send(
            symbol=alert.symbol,
            event_type=alert.event_type,
            severity=alert.severity,
            reason=alert.reason,
            payload=None,
            now_ms=now_ms,
            record_send=False,
        )

        if not decision.allowed:
            return False, True, False

        try:
            ok = await self._publisher.publish_alert(alert)
        except Exception:
            return True, False, True

        if not ok:
            return True, False, True

        # Phase 2: record dedupe state only after successful publish.
        self._deduper.should_send(
            symbol=alert.symbol,
            event_type=alert.event_type,
            severity=alert.severity,
            reason=alert.reason,
            payload=None,
            now_ms=now_ms,
            record_send=True,
        )

        return True, False, False


# ---------------------------------------------------------------------------
# Internal: alert builders
# ---------------------------------------------------------------------------


def _extract_symbol(payload: dict[str, Any]) -> str:
    """Extract symbol from payload, falling back to ``"UNKNOWN"``."""
    raw = payload.get("symbol")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return "UNKNOWN"


def _extract_severity(payload: dict[str, Any]) -> str:
    """Extract severity from payload, falling back to ``"INFO"``."""
    raw = payload.get("severity")
    if isinstance(raw, str) and raw.strip():
        return raw.strip().upper()
    return "INFO"


def _extract_reason(payload: dict[str, Any]) -> str | None:
    """Extract reason from a child event payload.

    Priority order:
    1. payload["reason"] (non-empty str)
    2. payload["halt_reason"] (non-empty str)
    3. payload["error_type"] (non-empty str)
    4. payload["data"]["reason"] (if data is dict, reason is non-empty str)
    5. payload["data"]["halt_reason"] (if data is dict, halt_reason is non-empty str)
    6. payload["data"]["error_type"] (if data is dict, error_type is non-empty str)
    7. None
    """
    # Top-level reason fields.
    for key in ("reason", "halt_reason", "error_type"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    # Nested data.* fields.
    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("reason", "halt_reason", "error_type"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

    return None


def _build_subject(symbol: str, severity: str, event_type: str) -> str:
    """Build the alert subject line."""
    return f"[ReclaimEdge][{severity}] {symbol} {event_type}"


def _build_child_event_body(
    symbol: str,
    event_type: str,
    severity: str,
    reason: str | None,
    ts_ms: int | None,
    source_path: str | None,
) -> str:
    """Build an HTML body for a child event alert."""
    parts: list[str] = [
        "<html><body>",
        "<table>",
        f"<tr><td><b>Symbol</b></td><td>{html.escape(symbol)}</td></tr>",
        f"<tr><td><b>Event Type</b></td><td>{html.escape(event_type)}</td></tr>",
        f"<tr><td><b>Severity</b></td><td>{html.escape(severity)}</td></tr>",
        f"<tr><td><b>Reason</b></td><td>{html.escape(reason if reason is not None else '-')}</td></tr>",
        (
            f"<tr><td><b>Event Time (ms)</b></td>"
            f"<td>{html.escape(str(ts_ms) if ts_ms is not None else '-')}</td></tr>"
        ),
        (
            f"<tr><td><b>Source Path</b></td>"
            f"<td>{html.escape(source_path if source_path is not None else '-')}</td></tr>"
        ),
        "</table>",
        "</body></html>",
    ]
    return "".join(parts)


def _build_read_error_body(
    error: ChildEventReadError,
    symbol: str,
    severity: str,
    event_type: str,
    reason: str | None,
) -> str:
    """Build an HTML body for a read-error alert."""
    parts: list[str] = [
        "<html><body>",
        "<table>",
        f"<tr><td><b>Symbol</b></td><td>{html.escape(symbol)}</td></tr>",
        f"<tr><td><b>Event Type</b></td><td>{html.escape(event_type)}</td></tr>",
        f"<tr><td><b>Severity</b></td><td>{html.escape(severity)}</td></tr>",
        f"<tr><td><b>Reason</b></td><td>{html.escape(reason if reason is not None else '-')}</td></tr>",
        f"<tr><td><b>Error Type</b></td><td>{html.escape(error.error_type)}</td></tr>",
        f"<tr><td><b>Message</b></td><td>{html.escape(error.message)}</td></tr>",
        (
            f"<tr><td><b>Offset Start</b></td>"
            f"<td>{html.escape(str(error.offset_start) if error.offset_start is not None else '-')}</td></tr>"
        ),
        (
            f"<tr><td><b>Offset End</b></td>"
            f"<td>{html.escape(str(error.offset_end) if error.offset_end is not None else '-')}</td></tr>"
        ),
        (
            f"<tr><td><b>Raw Preview</b></td>"
            f"<td><pre>{html.escape(error.raw_preview if error.raw_preview is not None else '-')}</pre></td></tr>"
        ),
        "</table>",
        "</body></html>",
    ]
    return "".join(parts)


def _build_alert_from_child_event(event: ChildEvent) -> SupervisorAlert:
    """Build a :class:`SupervisorAlert` from a lifecycle :class:`ChildEvent`."""
    payload = event.payload
    symbol = _extract_symbol(payload)
    severity = _extract_severity(payload)
    reason = _extract_reason(payload)
    subject = _build_subject(symbol, severity, event.event_type)
    body = _build_child_event_body(
        symbol=symbol,
        event_type=event.event_type,
        severity=severity,
        reason=reason,
        ts_ms=event.ts_ms,
        source_path=event.source_path,
    )
    return SupervisorAlert(
        symbol=symbol,
        event_type=event.event_type,
        severity=severity,
        reason=reason,
        subject=subject,
        body=body,
        content_type="html",
        source_path=event.source_path,
        ts_ms=event.ts_ms,
    )


def _build_alert_from_read_error(error: ChildEventReadError) -> SupervisorAlert:
    """Build a :class:`SupervisorAlert` from a :class:`ChildEventReadError`."""
    symbol = "SUPERVISOR"
    severity = "ERROR"
    event_type = READ_ERROR_EVENT_TYPE
    reason = error.error_type
    subject = _build_subject(symbol, severity, event_type)
    body = _build_read_error_body(
        error=error,
        symbol=symbol,
        severity=severity,
        event_type=event_type,
        reason=reason,
    )
    return SupervisorAlert(
        symbol=symbol,
        event_type=event_type,
        severity=severity,
        reason=reason,
        subject=subject,
        body=body,
        content_type="html",
        source_path=error.source_path,
        ts_ms=error.ts_ms,
    )
