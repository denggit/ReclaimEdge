from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from src.live.outbox.jsonl_outbox import JsonlOutbox

# ---------------------------------------------------------------------------
# Event type constants
# ---------------------------------------------------------------------------

WORKER_STARTED = "WORKER_STARTED"
WORKER_STOPPING = "WORKER_STOPPING"
WORKER_STOPPED = "WORKER_STOPPED"
WORKER_STARTUP_RECOVERY_COMPLETED = "WORKER_STARTUP_RECOVERY_COMPLETED"
WORKER_STARTUP_RECOVERY_FAILED = "WORKER_STARTUP_RECOVERY_FAILED"
WORKER_TRADING_HALTED = "WORKER_TRADING_HALTED"
WORKER_HEARTBEAT_WRITE_FAILED = "WORKER_HEARTBEAT_WRITE_FAILED"
WORKER_DRAIN_STARTED = "WORKER_DRAIN_STARTED"
WORKER_DRAIN_COMPLETED = "WORKER_DRAIN_COMPLETED"
WORKER_DRAIN_TIMEOUT = "WORKER_DRAIN_TIMEOUT"
WORKER_ROLLING_LOSS_GUARD = "WORKER_ROLLING_LOSS_GUARD"

WORKER_EVENT_TYPES = frozenset(
    {
        WORKER_STARTED,
        WORKER_STOPPING,
        WORKER_STOPPED,
        WORKER_STARTUP_RECOVERY_COMPLETED,
        WORKER_STARTUP_RECOVERY_FAILED,
        WORKER_TRADING_HALTED,
        WORKER_HEARTBEAT_WRITE_FAILED,
        WORKER_DRAIN_STARTED,
        WORKER_DRAIN_COMPLETED,
        WORKER_DRAIN_TIMEOUT,
        WORKER_ROLLING_LOSS_GUARD,
    }
)

WORKER_EVENT_SEVERITIES = frozenset({"INFO", "WARNING", "ERROR", "CRITICAL"})


# ---------------------------------------------------------------------------
# WorkerEvent
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkerEvent:
    """Normalised worker lifecycle / health / risk event.

    ``payload`` holds the **business data** — NOT the JSONL top-level
    ``"payload"`` wrapper.
    """

    ts_ms: int
    event_type: str
    symbol: str
    severity: str
    payload: dict[str, Any]


# ---------------------------------------------------------------------------
# WorkerEventEmitter
# ---------------------------------------------------------------------------


class WorkerEventEmitter:
    """Emit standardised worker events into a :class:`JsonlOutbox`.

    Every call to :meth:`emit` appends one JSONL line whose ``payload``
    field carries::

        {
            "symbol": "<symbol>",
            "severity": "<severity>",
            "data": { ... business payload ... }
        }

    The emitter is a **pure outbox tool** — it has no runtime side-effect
    imports and only depends on :class:`JsonlOutbox`.
    """

    def __init__(self, *, symbol: str, outbox: JsonlOutbox) -> None:
        # -- symbol validation --------------------------------------------------
        if not isinstance(symbol, str):
            raise ValueError(
                f"symbol must be str, got {type(symbol).__name__}"
            )
        stripped_symbol = symbol.strip()
        if not stripped_symbol:
            raise ValueError("symbol must not be empty or whitespace-only")
        self.symbol: str = stripped_symbol
        self._outbox: JsonlOutbox = outbox

    # ------------------------------------------------------------------
    # emit
    # ------------------------------------------------------------------

    def emit(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        severity: str = "INFO",
        ts_ms: int | None = None,
    ) -> WorkerEvent:
        """Write one event line to the outbox and return a :class:`WorkerEvent`.

        Parameters
        ----------
        event_type : str
            Non-empty event type label (e.g. ``"WORKER_STARTED"``).
        payload : dict[str, Any] | None
            Business data to nest inside ``data``.  Shallow-copied so the
            caller cannot mutate the stored event afterwards.
        severity : str
            One of ``INFO`` | ``WARNING`` | ``ERROR`` | ``CRITICAL``.
            Case-insensitive on input; always normalised to uppercase.
        ts_ms : int | None
            Epoch milliseconds.  Defaults to ``int(time.time() * 1000)``.

        Returns
        -------
        WorkerEvent
        """
        # -- event_type validation ----------------------------------------------
        if not isinstance(event_type, str):
            raise ValueError(
                f"event_type must be str, got {type(event_type).__name__}"
            )
        stripped_event_type = event_type.strip()
        if not stripped_event_type:
            raise ValueError("event_type must not be empty or whitespace-only")

        # -- severity normalisation + validation --------------------------------
        if not isinstance(severity, str):
            raise ValueError(
                f"severity must be str, got {type(severity).__name__}"
            )
        normalised_severity = severity.strip().upper()
        if normalised_severity not in WORKER_EVENT_SEVERITIES:
            raise ValueError(
                f"severity must be one of {sorted(WORKER_EVENT_SEVERITIES)}, "
                f"got {severity!r}"
            )

        # -- payload shallow-copy -----------------------------------------------
        if payload is None:
            data: dict[str, Any] = {}
        elif isinstance(payload, dict):
            data = dict(payload)
        else:
            raise ValueError(
                f"payload must be dict or None, got {type(payload).__name__}"
            )

        # -- ts_ms --------------------------------------------------------------
        if ts_ms is None:
            ts_ms = int(time.time() * 1000)

        # -- write through ------------------------------------------------------
        jsonl_event = self._outbox.append(
            event_type=stripped_event_type,
            payload={
                "symbol": self.symbol,
                "severity": normalised_severity,
                "data": data,
            },
            ts_ms=ts_ms,
        )

        return WorkerEvent(
            ts_ms=jsonl_event.ts_ms,
            event_type=jsonl_event.event_type,
            symbol=self.symbol,
            severity=normalised_severity,
            payload=data,
        )
