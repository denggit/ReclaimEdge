from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CRITICAL_RUNTIME_EVENT_TYPES = frozenset(
    {
        "WORKER_STARTUP_RECOVERY_FAILED",
        "WORKER_TRADING_HALTED",
        "WORKER_HEARTBEAT_WRITE_FAILED",
        "WORKER_DRAIN_TIMEOUT",
        "CHILD_EVENT_READ_FAILED",
        # parent/supervisor future critical runtime alerts
        "SUPERVISOR_CHILD_EXITED_UNEXPECTED",
        "SUPERVISOR_RESTART_FAILED",
        "SUPERVISOR_HEARTBEAT_MISSING",
    }
)

DEFAULT_SUPPRESSED_RUNTIME_EVENT_TYPES = frozenset(
    {
        "WORKER_STARTED",
        "WORKER_STOPPING",
        "WORKER_STOPPED",
        "WORKER_STARTUP_RECOVERY_COMPLETED",
        "WORKER_DRAIN_STARTED",
        "WORKER_DRAIN_COMPLETED",
    }
)

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlertPolicyDecision:
    """Result of :meth:`AlertPolicy.should_publish`.

    Attributes
    ----------
    allowed : bool
        ``True`` if the alert is allowed to enter the email send path.
    event_type : str
        Normalised event type (stripped).
    severity : str
        Normalised severity (stripped, upper-cased).
    reason : str | None
        Stripped reason or ``None``.
    policy_reason : str
        Why the decision was made, e.g. ``"critical_event_type"``.
    """

    allowed: bool
    event_type: str
    severity: str
    reason: str | None
    policy_reason: str


# ---------------------------------------------------------------------------
# AlertPolicy
# ---------------------------------------------------------------------------


class AlertPolicy:
    """Determines whether a runtime alert should be published to email.

    Only **critical** runtime events are allowed by default.  Normal
    lifecycle events such as ``WORKER_STARTED``, ``WORKER_STOPPED``, etc.
    are suppressed.  Unknown event types are suppressed unless
    ``allow_unknown_error_severity`` is enabled.

    This is a **supervisor control-plane** tool.  It must never be used
    inside the tick / trading path.

    Parameters
    ----------
    critical_event_types : collection of str | None
        Event types that are always allowed.  Replaces the default set.
    suppressed_event_types : collection of str | None
        Event types that are always suppressed (normal lifecycle).
        Replaces the default set.
    allow_unknown_error_severity : bool
        If ``True``, unknown event types with severity ``ERROR`` or
        ``CRITICAL`` are allowed.  Default ``False``.
    """

    def __init__(
        self,
        *,
        critical_event_types: (
            frozenset[str] | set[str] | tuple[str, ...] | list[str] | None
        ) = None,
        suppressed_event_types: (
            frozenset[str] | set[str] | tuple[str, ...] | list[str] | None
        ) = None,
        allow_unknown_error_severity: bool = False,
    ) -> None:
        # -- allow_unknown_error_severity ------------------------------------
        if type(allow_unknown_error_severity) is not bool:
            raise ValueError(
                f"allow_unknown_error_severity must be bool, "
                f"got {type(allow_unknown_error_severity).__name__}"
            )

        # -- resolve critical set --------------------------------------------
        if critical_event_types is None:
            critical = DEFAULT_CRITICAL_RUNTIME_EVENT_TYPES
        else:
            critical = self._validate_event_type_collection(
                critical_event_types, "critical_event_types"
            )

        # -- resolve suppressed set ------------------------------------------
        if suppressed_event_types is None:
            suppressed = DEFAULT_SUPPRESSED_RUNTIME_EVENT_TYPES
        else:
            suppressed = self._validate_event_type_collection(
                suppressed_event_types, "suppressed_event_types"
            )

        # -- overlap check ---------------------------------------------------
        overlap = critical & suppressed
        if overlap:
            raise ValueError(
                f"critical_event_types and suppressed_event_types must not "
                f"overlap, got {sorted(overlap)!r}"
            )

        self._critical = critical
        self._suppressed = suppressed
        self._allow_unknown_error_severity = allow_unknown_error_severity

    # ------------------------------------------------------------------
    # should_publish
    # ------------------------------------------------------------------

    def should_publish(
        self,
        *,
        event_type: str,
        severity: str = "INFO",
        reason: str | None = None,
    ) -> AlertPolicyDecision:
        """Decide whether an alert should be published.

        Parameters
        ----------
        event_type : str
            Non-empty event type label.
        severity : str
            Severity label; normalised to uppercase.
        reason : str | None
            Optional reason string.

        Returns
        -------
        AlertPolicyDecision

        Raises
        ------
        ValueError
            If any required argument is invalid (e.g. empty event_type).
        """
        # -- validate event_type ---------------------------------------------
        if not isinstance(event_type, str):
            raise ValueError(
                f"event_type must be str, got {type(event_type).__name__}"
            )
        event_type = event_type.strip()
        if not event_type:
            raise ValueError("event_type must not be empty or whitespace-only")

        # -- validate severity -----------------------------------------------
        if not isinstance(severity, str):
            raise ValueError(
                f"severity must be str, got {type(severity).__name__}"
            )
        severity = severity.strip()
        if not severity:
            raise ValueError("severity must not be empty or whitespace-only")
        severity = severity.upper()

        # -- validate reason -------------------------------------------------
        if reason is not None:
            if not isinstance(reason, str):
                raise ValueError(
                    f"reason must be str or None, got {type(reason).__name__}"
                )
            reason = reason.strip()
            if not reason:
                reason = None

        return self._decide(event_type, severity, reason)

    # ------------------------------------------------------------------
    # should_publish_alert
    # ------------------------------------------------------------------

    def should_publish_alert(self, alert: object) -> AlertPolicyDecision:
        """Decide whether a duck-typed alert object should be published.

        Reads ``event_type``, ``severity``, and optionally ``reason``
        from *alert* via attribute access.  Never raises — invalid or
        missing attributes produce an ``allowed=False`` decision with
        ``policy_reason="invalid_alert_object"``.

        Parameters
        ----------
        alert : object
            Any object with ``event_type``, ``severity`` and optionally
            ``reason`` attributes.

        Returns
        -------
        AlertPolicyDecision
        """
        # -- extract event_type ----------------------------------------------
        raw_et = getattr(alert, "event_type", None)
        if not isinstance(raw_et, str):
            return AlertPolicyDecision(
                allowed=False,
                event_type="",
                severity="INFO",
                reason=None,
                policy_reason="invalid_alert_object",
            )
        event_type = raw_et.strip()
        if not event_type:
            return AlertPolicyDecision(
                allowed=False,
                event_type="",
                severity="INFO",
                reason=None,
                policy_reason="invalid_alert_object",
            )

        # -- extract severity ------------------------------------------------
        raw_sev = getattr(alert, "severity", None)
        if not isinstance(raw_sev, str):
            return AlertPolicyDecision(
                allowed=False,
                event_type=event_type,
                severity="INFO",
                reason=None,
                policy_reason="invalid_alert_object",
            )
        severity = raw_sev.strip()
        if not severity:
            return AlertPolicyDecision(
                allowed=False,
                event_type=event_type,
                severity="INFO",
                reason=None,
                policy_reason="invalid_alert_object",
            )
        severity = severity.upper()

        # -- extract reason --------------------------------------------------
        raw_reason = getattr(alert, "reason", None)
        reason: str | None = None
        if raw_reason is not None:
            if isinstance(raw_reason, str):
                stripped = raw_reason.strip()
                if stripped:
                    reason = stripped
            # Non-str, non-None reason is silently dropped (not an error).

        return self._decide(event_type, severity, reason)

    # ------------------------------------------------------------------
    # Internal: decision logic
    # ------------------------------------------------------------------

    def _decide(
        self,
        event_type: str,
        severity: str,
        reason: str | None,
    ) -> AlertPolicyDecision:
        """Core decision logic — shared by ``should_publish`` and
        ``should_publish_alert``.
        """
        # -- 1. explicit critical --------------------------------------------
        if event_type in self._critical:
            return AlertPolicyDecision(
                allowed=True,
                event_type=event_type,
                severity=severity,
                reason=reason,
                policy_reason="critical_event_type",
            )

        # -- 2. explicit suppressed ------------------------------------------
        if event_type in self._suppressed:
            return AlertPolicyDecision(
                allowed=False,
                event_type=event_type,
                severity=severity,
                reason=reason,
                policy_reason="suppressed_normal_lifecycle",
            )

        # -- 3. unknown + error severity -------------------------------------
        if self._allow_unknown_error_severity and severity in {"ERROR", "CRITICAL"}:
            return AlertPolicyDecision(
                allowed=True,
                event_type=event_type,
                severity=severity,
                reason=reason,
                policy_reason="unknown_error_severity",
            )

        # -- 4. unknown ------------------------------------------------------
        return AlertPolicyDecision(
            allowed=False,
            event_type=event_type,
            severity=severity,
            reason=reason,
            policy_reason="unknown_event_type",
        )

    # ------------------------------------------------------------------
    # Internal: validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_event_type_collection(
        value: object,
        param_name: str,
    ) -> frozenset[str]:
        """Validate a user-supplied event type collection.

        Returns a frozenset of stripped, non-empty str values.

        Raises ValueError for any invalid input.
        """
        # Must be list / tuple / set / frozenset.  str is iterable but
        # is NOT accepted — passing a bare string is almost certainly a
        # mistake.
        if isinstance(value, str):
            raise ValueError(
                f"{param_name} must be list/tuple/set/frozenset, not str"
            )

        if not isinstance(value, (list, tuple, set, frozenset)):
            raise ValueError(
                f"{param_name} must be list/tuple/set/frozenset, "
                f"got {type(value).__name__}"
            )

        result: set[str] = set()
        for i, item in enumerate(value):
            if not isinstance(item, str):
                raise ValueError(
                    f"{param_name}[{i}] must be str, "
                    f"got {type(item).__name__}={item!r}"
                )
            stripped = item.strip()
            if not stripped:
                raise ValueError(
                    f"{param_name}[{i}] must not be empty or whitespace-only"
                )
            result.add(stripped)

        return frozenset(result)
