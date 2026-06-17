"""Delayed confirmed swing extreme tracker for Reclaim V2.

Each tick, price is fed to a running candidate.  A candidate is only
**confirmed** as a swing extreme after:

- **retrace** — price reverses by at least ``confirm_retrace_pct`` from
  the candidate extreme; or
- **stable** — the candidate extreme has held for at least
  ``confirm_stable_seconds`` without being extended.

Only confirmed swing extremes participate in anchored CVD divergence
evaluation.  This eliminates the tick-level ``new_extreme_count`` spam
and dramatically reduces divergence evaluations / log noise.

This module is pure logic:

- No env reads
- No exchange calls
- No order placement
- No strategy state mutation

Usage::

    config = ConfirmedExtremeConfig(
        confirm_mode="RETRACE_OR_STABLE",
        confirm_retrace_pct=0.0008,
        confirm_stable_seconds=8,
        min_price_extension_pct=0.0002,
    )
    tracker = ConfirmedExtremeTracker(side="LOWER", config=config)

    # Each tick:
    result = tracker.update(price=1780.0, anchored_cvd=-50000.0, ts_ms=10000)
    if result is not None:
        print(f"Confirmed {result.side} at {result.price}, reason={result.confirm_reason}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ExtremeSide = Literal["LOWER", "UPPER"]


# ---------------------------------------------------------------------------
# value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfirmedExtreme:
    """A confirmed swing extreme event.

    This is produced when a running candidate has been confirmed via
    retrace or stability — not on every tick-level new high/low.
    """

    side: Literal["LOWER", "UPPER"]
    price: float
    anchored_cvd: float
    ts_ms: int
    confirm_ts_ms: int
    confirm_reason: str  # "retrace" | "stable"


@dataclass
class ExtremeCandidate:
    """Mutable running candidate for a single direction (LOWER or UPPER)."""

    price: float = 0.0
    anchored_cvd: float = 0.0
    ts_ms: int = 0
    last_update_ts_ms: int = 0

    @property
    def is_set(self) -> bool:
        return self.ts_ms > 0


@dataclass(frozen=True)
class ConfirmedExtremeConfig:
    """Configuration for :class:`ConfirmedExtremeTracker`.

    Parameters
    ----------
    confirm_mode : str
        ``RETRACE_OR_STABLE`` — confirm on retrace OR stability.
        Future modes may be added.
    confirm_retrace_pct : float
        Fractional retrace required to confirm a swing extreme.
        LOWER: price must rise by this fraction above the candidate low.
        UPPER: price must fall by this fraction below the candidate high.
    confirm_stable_seconds : int
        Seconds without a new candidate extension after which the
        candidate is confirmed (even without retrace).
    min_price_extension_pct : float
        Minimum fractional price extension required to update the
        candidate (avoids micro-tick noise).
    """

    confirm_mode: str = "RETRACE_OR_STABLE"
    confirm_retrace_pct: float = 0.0008
    confirm_stable_seconds: int = 8
    min_price_extension_pct: float = 0.0002


# ---------------------------------------------------------------------------
# tracker
# ---------------------------------------------------------------------------


class ConfirmedExtremeTracker:
    """Stateful tracker that confirms swing extremes.

    Call ``update()`` on each tick.  Returns a :class:`ConfirmedExtreme`
    when a candidate is confirmed; returns ``None`` otherwise.

    After a confirmed extreme is emitted the candidate is reset and a
    new accumulation cycle begins.
    """

    def __init__(
        self,
        *,
        side: Literal["LOWER", "UPPER"],
        config: ConfirmedExtremeConfig,
    ) -> None:
        if side not in ("LOWER", "UPPER"):
            raise ValueError(f"side must be LOWER or UPPER, got {side!r}")
        self.side: Literal["LOWER", "UPPER"] = side
        self.config = config
        self._candidate = ExtremeCandidate()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the running candidate (e.g. on strategy reset)."""
        self._candidate = ExtremeCandidate()

    def update(
        self,
        *,
        price: float,
        anchored_cvd: float,
        ts_ms: int,
    ) -> ConfirmedExtreme | None:
        """Process one tick and return a confirmed extreme if triggered.

        Returns ``None`` on most ticks.  A non-None return means a swing
        extreme has been confirmed and divergence evaluation should run.
        """
        if self.side == "LOWER":
            return self._update_lower(price=price, anchored_cvd=anchored_cvd, ts_ms=ts_ms)
        return self._update_upper(price=price, anchored_cvd=anchored_cvd, ts_ms=ts_ms)

    # ------------------------------------------------------------------
    # internal — LOWER
    # ------------------------------------------------------------------

    def _update_lower(
        self,
        *,
        price: float,
        anchored_cvd: float,
        ts_ms: int,
    ) -> ConfirmedExtreme | None:
        cfg = self.config
        cand = self._candidate

        # ── candidate update ─────────────────────────────────────────
        if not cand.is_set:
            cand.price = price
            cand.anchored_cvd = anchored_cvd
            cand.ts_ms = ts_ms
            cand.last_update_ts_ms = ts_ms
            return None

        # Extension: price is lower than candidate by min extension
        ext_threshold = cand.price * (1.0 - cfg.min_price_extension_pct)
        if price < ext_threshold:
            cand.price = price
            cand.anchored_cvd = anchored_cvd
            cand.ts_ms = ts_ms
            cand.last_update_ts_ms = ts_ms
            return None

        # ── confirmation checks ──────────────────────────────────────
        retrace_price = cand.price * (1.0 + cfg.confirm_retrace_pct)
        retrace_confirmed = price >= retrace_price

        stable_ms = cfg.confirm_stable_seconds * 1000
        stable_confirmed = (ts_ms - cand.last_update_ts_ms) >= stable_ms

        if cfg.confirm_mode == "RETRACE_OR_STABLE":
            if retrace_confirmed or stable_confirmed:
                reason = "retrace" if retrace_confirmed else "stable"
                result = ConfirmedExtreme(
                    side=self.side,
                    price=cand.price,
                    anchored_cvd=cand.anchored_cvd,
                    ts_ms=cand.ts_ms,
                    confirm_ts_ms=ts_ms,
                    confirm_reason=reason,
                )
                self._candidate = ExtremeCandidate()
                return result

        return None

    # ------------------------------------------------------------------
    # internal — UPPER
    # ------------------------------------------------------------------

    def _update_upper(
        self,
        *,
        price: float,
        anchored_cvd: float,
        ts_ms: int,
    ) -> ConfirmedExtreme | None:
        cfg = self.config
        cand = self._candidate

        # ── candidate update ─────────────────────────────────────────
        if not cand.is_set:
            cand.price = price
            cand.anchored_cvd = anchored_cvd
            cand.ts_ms = ts_ms
            cand.last_update_ts_ms = ts_ms
            return None

        # Extension: price is higher than candidate by min extension
        ext_threshold = cand.price * (1.0 + cfg.min_price_extension_pct)
        if price > ext_threshold:
            cand.price = price
            cand.anchored_cvd = anchored_cvd
            cand.ts_ms = ts_ms
            cand.last_update_ts_ms = ts_ms
            return None

        # ── confirmation checks ──────────────────────────────────────
        retrace_price = cand.price * (1.0 - cfg.confirm_retrace_pct)
        retrace_confirmed = price <= retrace_price

        stable_ms = cfg.confirm_stable_seconds * 1000
        stable_confirmed = (ts_ms - cand.last_update_ts_ms) >= stable_ms

        if cfg.confirm_mode == "RETRACE_OR_STABLE":
            if retrace_confirmed or stable_confirmed:
                reason = "retrace" if retrace_confirmed else "stable"
                result = ConfirmedExtreme(
                    side=self.side,
                    price=cand.price,
                    anchored_cvd=cand.anchored_cvd,
                    ts_ms=cand.ts_ms,
                    confirm_ts_ms=ts_ms,
                    confirm_reason=reason,
                )
                self._candidate = ExtremeCandidate()
                return result

        return None
