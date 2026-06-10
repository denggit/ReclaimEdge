from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.live.outbox.atomic_json import read_json_or_none, write_json_atomic

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlertDedupeDecision:
    """Result of :meth:`AlertDeduper.should_send`.

    Attributes
    ----------
    allowed : bool
        ``True`` if the alert should be sent.
    dedupe_key : str
        Stable deduplication key built from the alert fields.
    last_sent_ts_ms : int | None
        Epoch milliseconds of the most recent *send* (``None`` for first-time).
    next_allowed_ts_ms : int | None
        Earliest epoch millisecond at which the next identical alert will be
        allowed, based on ``last_sent_ts_ms + cooldown_ms``.  ``None`` when
        ``cooldown_seconds == 0``.
    now_ts_ms : int
        The ``now_ms`` value used for this decision.
    reason : str | None
        The reason preview that contributed to the dedupe key.
    """

    allowed: bool
    dedupe_key: str
    last_sent_ts_ms: int | None
    next_allowed_ts_ms: int | None
    now_ts_ms: int
    reason: str | None = None


# ---------------------------------------------------------------------------
# AlertDeduper
# ---------------------------------------------------------------------------


class AlertDeduper:
    """Cooldown-based alert deduplication for the supervisor control plane.

    Determines whether an alert with a given ``(symbol, event_type, severity,
    reason)`` tuple should be sent based on a configurable cooldown window.

    State is persisted to a single atomic JSON file — never appended,
    never unbounded.

    This is a **supervisor control-plane** tool.  It must never be used
    inside the tick / trading path.

    Parameters
    ----------
    state_path : Path
        Path to the atomic JSON dedupe state file.
    cooldown_seconds : int
        Minimum seconds between identical alerts (default 15 minutes).
    max_entries : int
        Maximum number of dedupe entries in the state file (default 2048).
    max_reason_chars : int
        Maximum characters for the reason preview stored in state (default 128).
    """

    def __init__(
        self,
        *,
        state_path: Path,
        cooldown_seconds: int = 15 * 60,
        max_entries: int = 2048,
        max_reason_chars: int = 128,
    ) -> None:
        # -- state_path ----------------------------------------------------------
        if not isinstance(state_path, Path):
            raise ValueError(
                f"state_path must be Path, got {type(state_path).__name__}"
            )

        # -- cooldown_seconds ----------------------------------------------------
        if type(cooldown_seconds) is not int:
            raise ValueError(
                f"cooldown_seconds must be int, got {type(cooldown_seconds).__name__}"
            )
        if cooldown_seconds < 0:
            raise ValueError(
                f"cooldown_seconds must be >= 0, got {cooldown_seconds}"
            )

        # -- max_entries ---------------------------------------------------------
        if type(max_entries) is not int:
            raise ValueError(
                f"max_entries must be int, got {type(max_entries).__name__}"
            )
        if max_entries <= 0:
            raise ValueError(
                f"max_entries must be > 0, got {max_entries}"
            )

        # -- max_reason_chars ----------------------------------------------------
        if type(max_reason_chars) is not int:
            raise ValueError(
                f"max_reason_chars must be int, got {type(max_reason_chars).__name__}"
            )
        if max_reason_chars <= 0:
            raise ValueError(
                f"max_reason_chars must be > 0, got {max_reason_chars}"
            )

        self._state_path = state_path
        self._cooldown_seconds = cooldown_seconds
        self._max_entries = max_entries
        self._max_reason_chars = max_reason_chars

    # ------------------------------------------------------------------
    # should_send
    # ------------------------------------------------------------------

    def should_send(
        self,
        *,
        symbol: str,
        event_type: str,
        severity: str = "INFO",
        reason: str | None = None,
        payload: dict[str, Any] | None = None,
        now_ms: int | None = None,
    ) -> AlertDedupeDecision:
        """Decide whether an alert should be sent.

        Parameters
        ----------
        symbol : str
            Non-empty symbol identifier (e.g. ``"ETH-USDT-SWAP"``).
        event_type : str
            Non-empty event type label (e.g. ``"WORKER_TRADING_HALTED"``).
        severity : str
            Severity label; normalised to uppercase on input.
        reason : str | None
            Explicit reason for the alert.  Takes precedence over any
            reason-like field in ``payload``.
        payload : dict[str, Any] | None
            Optional business-data dict.  Only inspected for fallback
            reason extraction — never stored in full.
        now_ms : int | None
            Epoch milliseconds for the decision.  Defaults to
            ``int(time.time() * 1000)``.

        Returns
        -------
        AlertDedupeDecision
        """
        # -- validate symbol -----------------------------------------------------
        if not isinstance(symbol, str):
            raise ValueError(
                f"symbol must be str, got {type(symbol).__name__}"
            )
        symbol = symbol.strip()
        if not symbol:
            raise ValueError("symbol must not be empty or whitespace-only")

        # -- validate event_type -------------------------------------------------
        if not isinstance(event_type, str):
            raise ValueError(
                f"event_type must be str, got {type(event_type).__name__}"
            )
        event_type = event_type.strip()
        if not event_type:
            raise ValueError("event_type must not be empty or whitespace-only")

        # -- validate severity ---------------------------------------------------
        if not isinstance(severity, str):
            raise ValueError(
                f"severity must be str, got {type(severity).__name__}"
            )
        severity = severity.strip().upper()
        if not severity:
            raise ValueError("severity must not be empty or whitespace-only")

        # -- validate reason -----------------------------------------------------
        if reason is not None and not isinstance(reason, str):
            raise ValueError(
                f"reason must be str or None, got {type(reason).__name__}"
            )

        # -- validate payload ----------------------------------------------------
        if payload is not None and not isinstance(payload, dict):
            raise ValueError(
                f"payload must be dict or None, got {type(payload).__name__}"
            )

        # -- validate now_ms -----------------------------------------------------
        if now_ms is None:
            now_ms = int(time.time() * 1000)
        else:
            if type(now_ms) is not int:
                raise ValueError(
                    f"now_ms must be int, got {type(now_ms).__name__}"
                )
            if now_ms < 0:
                raise ValueError(
                    f"now_ms must be >= 0, got {now_ms}"
                )

        # -- reason extraction ---------------------------------------------------
        reason_source = self._extract_reason(reason, payload)
        reason_preview = self._normalize_reason(reason_source)

        # -- build dedupe key ----------------------------------------------------
        dedupe_key = self._build_key(symbol, event_type, severity, reason_source)

        # -- load state ----------------------------------------------------------
        state = self._load_state()

        # -- decide --------------------------------------------------------------
        cooldown_ms = self._cooldown_seconds * 1000
        entries: dict[str, dict[str, Any]] = state.get("entries", {})
        if not isinstance(entries, dict):
            entries = {}

        entry = entries.get(dedupe_key)
        last_sent_ts_ms: int | None = None

        if entry is not None and isinstance(entry, dict):
            raw_last = entry.get("last_sent_ts_ms")
            if isinstance(raw_last, int) and raw_last >= 0:
                last_sent_ts_ms = raw_last

        if last_sent_ts_ms is None:
            # First time for this key.
            allowed = True
            next_allowed = now_ms + cooldown_ms if self._cooldown_seconds > 0 else None
            result = AlertDedupeDecision(
                allowed=True,
                dedupe_key=dedupe_key,
                last_sent_ts_ms=None,
                next_allowed_ts_ms=next_allowed,
                now_ts_ms=now_ms,
                reason=reason_preview,
            )
            self._update_entry(
                entries, dedupe_key, symbol, event_type, severity,
                reason_preview, now_ms, send_count=1,
            )
            self._prune_and_save(entries, now_ms, dedupe_key)
            return result

        # Cooldown == 0: always allow, still update state.
        if self._cooldown_seconds == 0:
            allowed = True
            send_count = entry.get("send_count", 1) if isinstance(entry, dict) else 1
            if isinstance(send_count, int) and send_count > 0:
                send_count += 1
            else:
                send_count = 1
            result = AlertDedupeDecision(
                allowed=True,
                dedupe_key=dedupe_key,
                last_sent_ts_ms=last_sent_ts_ms,
                next_allowed_ts_ms=None,
                now_ts_ms=now_ms,
                reason=reason_preview,
            )
            self._update_entry(
                entries, dedupe_key, symbol, event_type, severity,
                reason_preview, now_ms, send_count=send_count,
            )
            self._prune_and_save(entries, now_ms, dedupe_key)
            return result

        # Clock went backwards: suppress.
        if now_ms < last_sent_ts_ms:
            return AlertDedupeDecision(
                allowed=False,
                dedupe_key=dedupe_key,
                last_sent_ts_ms=last_sent_ts_ms,
                next_allowed_ts_ms=last_sent_ts_ms + cooldown_ms,
                now_ts_ms=now_ms,
                reason=reason_preview,
            )

        elapsed_ms = now_ms - last_sent_ts_ms

        if elapsed_ms >= cooldown_ms:
            # Cooldown expired: allow.
            allowed = True
            send_count = entry.get("send_count", 1) if isinstance(entry, dict) else 1
            if isinstance(send_count, int) and send_count > 0:
                send_count += 1
            else:
                send_count = 1
            result = AlertDedupeDecision(
                allowed=True,
                dedupe_key=dedupe_key,
                last_sent_ts_ms=last_sent_ts_ms,
                next_allowed_ts_ms=now_ms + cooldown_ms,
                now_ts_ms=now_ms,
                reason=reason_preview,
            )
            self._update_entry(
                entries, dedupe_key, symbol, event_type, severity,
                reason_preview, now_ms, send_count=send_count,
            )
            self._prune_and_save(entries, now_ms, dedupe_key)
            return result
        else:
            # Still within cooldown: suppress.
            return AlertDedupeDecision(
                allowed=False,
                dedupe_key=dedupe_key,
                last_sent_ts_ms=last_sent_ts_ms,
                next_allowed_ts_ms=last_sent_ts_ms + cooldown_ms,
                now_ts_ms=now_ms,
                reason=reason_preview,
            )

    # ------------------------------------------------------------------
    # Internal: reason extraction
    # ------------------------------------------------------------------

    def _extract_reason(
        self,
        explicit_reason: str | None,
        payload: dict[str, Any] | None,
    ) -> str | None:
        """Extract the canonical reason source.

        Priority:
        1. Explicit ``reason`` parameter.
        2. ``payload["reason"]`` if str.
        3. ``payload["halt_reason"]`` if str.
        4. ``payload["error_type"]`` if str.
        5. ``None``.
        """
        if explicit_reason is not None:
            return explicit_reason

        if payload is None:
            return None

        for key in ("reason", "halt_reason", "error_type"):
            val = payload.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()

        return None

    def _normalize_reason(self, reason_source: str | None) -> str | None:
        """Normalize and truncate a reason source for dedupe-key hashing."""
        if reason_source is None:
            return None
        stripped = reason_source.strip()
        if not stripped:
            return None
        return stripped[:self._max_reason_chars]

    # ------------------------------------------------------------------
    # Internal: dedupe key
    # ------------------------------------------------------------------

    def _build_key(
        self,
        symbol: str,
        event_type: str,
        severity: str,
        reason_source: str | None,
    ) -> str:
        """Build a stable, short, readable dedupe key.

        Format: ``symbol|event_type|severity|reason_hash``

        The reason is hashed (not stored in full) to keep keys short.
        """
        if reason_source:
            normalized = reason_source.strip()[:self._max_reason_chars]
            if normalized:
                reason_hash = hashlib.sha256(
                    normalized.encode("utf-8")
                ).hexdigest()[:16]
            else:
                reason_hash = "-"
        else:
            reason_hash = "-"

        return f"{symbol}|{event_type}|{severity}|{reason_hash}"

    # ------------------------------------------------------------------
    # Internal: state load / save
    # ------------------------------------------------------------------

    def _load_state(self) -> dict[str, Any]:
        """Load dedupe state from the atomic JSON file.

        Missing or invalid state is treated as empty — never raises.
        """
        raw = read_json_or_none(self._state_path)
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            return {}
        return raw

    def _update_entry(
        self,
        entries: dict[str, dict[str, Any]],
        dedupe_key: str,
        symbol: str,
        event_type: str,
        severity: str,
        reason_preview: str | None,
        now_ms: int,
        *,
        send_count: int = 1,
    ) -> None:
        """Update or create a dedupe entry in the entries dict (in-place)."""
        entries[dedupe_key] = {
            "last_sent_ts_ms": now_ms,
            "symbol": symbol,
            "event_type": event_type,
            "severity": severity,
            "reason_preview": reason_preview,
            "send_count": send_count,
        }

    # ------------------------------------------------------------------
    # Internal: prune + save
    # ------------------------------------------------------------------

    def _prune_and_save(
        self,
        entries: dict[str, dict[str, Any]],
        now_ms: int,
        current_key: str,
    ) -> None:
        """Prune invalid / old / excess entries and atomically save state."""
        pruned: dict[str, dict[str, Any]] = {}

        # -- cutoff for stale entries -------------------------------------------
        cutoff_ms: int | None = None
        if self._cooldown_seconds > 0:
            cutoff_ms = now_ms - (self._cooldown_seconds * 1000 * 4)

        for key, entry in entries.items():
            # -- 1. remove non-dict entries -------------------------------------
            if not isinstance(entry, dict):
                continue

            # -- 2. remove entries with invalid last_sent_ts_ms -----------------
            raw_last = entry.get("last_sent_ts_ms")
            if not isinstance(raw_last, int) or raw_last < 0:
                continue

            # -- trim reason_preview --------------------------------------------
            reason_preview = entry.get("reason_preview")
            if isinstance(reason_preview, str):
                entry["reason_preview"] = reason_preview[:self._max_reason_chars]

            # -- 3. remove stale entries (but keep current key) -----------------
            if cutoff_ms is not None and raw_last < cutoff_ms and key != current_key:
                continue

            pruned[key] = entry

        # -- 4. enforce max_entries (keep current key) --------------------------
        if len(pruned) > self._max_entries:
            # Sort by last_sent_ts_ms ascending (oldest first).
            sorted_keys = sorted(
                pruned.keys(),
                key=lambda k: (
                    pruned[k].get("last_sent_ts_ms", 0)
                    if isinstance(pruned[k], dict)
                    else 0
                ),
            )
            # Remove oldest entries, but always preserve current_key.
            to_remove = len(pruned) - self._max_entries
            removed = 0
            for k in sorted_keys:
                if removed >= to_remove:
                    break
                if k == current_key:
                    continue
                del pruned[k]
                removed += 1

        # -- save ---------------------------------------------------------------
        self._save_state(pruned, now_ms)

    def _save_state(
        self,
        entries: dict[str, dict[str, Any]],
        now_ms: int,
    ) -> None:
        """Atomically write the dedupe state."""
        state: dict[str, Any] = {
            "version": 1,
            "updated_ts_ms": now_ms,
            "entries": entries,
        }
        write_json_atomic(self._state_path, state, indent=2, sort_keys=True)
