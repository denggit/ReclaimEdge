"""Event-anchored orderflow metrics — shared by Trend Breakout and Reclaim V2.

Tracks cumulative CVD, buy/sell volume, price move, new extremes, and
price-per-CVD / price-per-volume efficiency from an anchor event onward.

This module is pure logic:
- No env reads
- No exchange calls
- No order placement
- No strategy state mutation

Direction semantics:
  UP   — price breaking to the upside (trend LONG, reclaim SHORT upper)
  DOWN — price breaking to the downside (trend SHORT, reclaim LONG lower)

Usage::

    tracker = AnchoredOrderflowTracker()
    tracker.anchor(
        direction="DOWN",
        ts_ms=1000, price=1900.0,
        cumulative_cvd=-500_000.0,
        cumulative_buy_volume=1_000_000.0,
        cumulative_sell_volume=1_500_000.0,
    )
    snap = tracker.update(
        ts_ms=2000, price=1895.0,
        cumulative_cvd=-480_000.0,
        cumulative_buy_volume=1_050_000.0,
        cumulative_sell_volume=1_530_000.0,
    )
    print(snap.anchored_cvd)   # 20_000.0 (improved — less bearish)
    print(snap.price_move_pct) # ~0.0026
    print(snap.new_extreme_count)  # 1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

OrderflowDirection = Literal["UP", "DOWN"]

_EPS: float = 1e-12


# ---------------------------------------------------------------------------
# snapshot (frozen, read-only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnchoredOrderflowSnapshot:
    """Read-only snapshot of orderflow metrics since anchor.

    All volume values are deltas from anchor (current - anchor).
    """

    anchored_cvd: float
    """Current cumulative CVD minus anchor cumulative CVD."""

    buy_volume: float
    """Episode buy volume (current - anchor, clamped >= 0)."""

    sell_volume: float
    """Episode sell volume (current - anchor, clamped >= 0)."""

    total_volume: float
    """buy_volume + sell_volume."""

    price_move_pct: float
    """Direction-aligned absolute price move from anchor (>= 0)."""

    cvd_efficiency: float
    """price_move_pct / max(|anchored_cvd|, eps)."""

    volume_efficiency: float
    """price_move_pct / max(total_volume, eps)."""

    new_extreme_count: int
    """Number of new direction-aligned extremes since anchor."""

    last_extreme_price: float
    """Most recent direction-aligned extreme price (0 if none)."""

    last_extreme_anchored_cvd: float
    """Anchored CVD as of the most recent extreme (0 if none)."""


# ---------------------------------------------------------------------------
# tracker (mutable state)
# ---------------------------------------------------------------------------


@dataclass
class AnchoredOrderflowTracker:
    """Stateful tracker of orderflow metrics from an anchor event.

    Call ``anchor()`` to start a new episode, then ``update()`` on
    each tick.  ``reset()`` clears all state.

    The tracker does **not** make directional decisions — it only
    produces :class:`AnchoredOrderflowSnapshot` data.  Callers
    (Trend Breakout / Reclaim V2) interpret the snapshot for their
    own confirmation logic.
    """

    direction: OrderflowDirection | None = None
    anchor_ts_ms: int = 0
    anchor_price: float = 0.0
    anchor_cumulative_cvd: float = 0.0
    anchor_buy_volume: float = 0.0
    anchor_sell_volume: float = 0.0

    last_extreme_price: float = 0.0
    last_extreme_anchored_cvd: float = 0.0
    new_extreme_count: int = 0

    @property
    def initialised(self) -> bool:
        return self.direction is not None and self.anchor_ts_ms > 0

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def anchor(
        self,
        *,
        direction: OrderflowDirection,
        ts_ms: int,
        price: float,
        cumulative_cvd: float,
        cumulative_buy_volume: float,
        cumulative_sell_volume: float,
    ) -> None:
        """Start a new anchored orderflow episode.

        All subsequent ``update()`` calls compute deltas from this
        anchor point.
        """
        self.direction = direction
        self.anchor_ts_ms = ts_ms
        self.anchor_price = price
        self.anchor_cumulative_cvd = cumulative_cvd
        self.anchor_buy_volume = cumulative_buy_volume
        self.anchor_sell_volume = cumulative_sell_volume

        self.last_extreme_price = price
        self.last_extreme_anchored_cvd = 0.0
        self.new_extreme_count = 0

    def update(
        self,
        *,
        ts_ms: int,
        price: float,
        cumulative_cvd: float,
        cumulative_buy_volume: float,
        cumulative_sell_volume: float,
    ) -> AnchoredOrderflowSnapshot:
        """Record a tick and return the current anchored snapshot.

        Does **not** require ``ts_ms`` for computation (used only for
        future extensions).  All metrics are derived from price and
        cumulative volume deltas.
        """
        if not self.initialised:
            return _empty_snapshot()

        # ── Volume deltas ────────────────────────────────────────────
        buy_vol = max(cumulative_buy_volume - self.anchor_buy_volume, 0.0)
        sell_vol = max(cumulative_sell_volume - self.anchor_sell_volume, 0.0)
        total_vol = buy_vol + sell_vol
        anchored_cvd = cumulative_cvd - self.anchor_cumulative_cvd

        # ── Direction-aligned price move ─────────────────────────────
        if self.direction == "UP":
            price_move_pct = (
                max((price - self.anchor_price) / self.anchor_price, 0.0)
                if self.anchor_price > 0
                else 0.0
            )
        else:
            price_move_pct = (
                max((self.anchor_price - price) / self.anchor_price, 0.0)
                if self.anchor_price > 0
                else 0.0
            )

        # ── New extreme tracking ─────────────────────────────────────
        is_new_extreme = False
        if self.direction == "UP" and price > self.last_extreme_price:
            is_new_extreme = True
        elif self.direction == "DOWN" and price < self.last_extreme_price:
            is_new_extreme = True

        if is_new_extreme:
            self.new_extreme_count += 1
            self.last_extreme_price = price
            self.last_extreme_anchored_cvd = anchored_cvd

        # ── Efficiencies (zero-division safe) ────────────────────────
        abs_cvd = max(abs(anchored_cvd), _EPS)
        cvd_eff = price_move_pct / abs_cvd
        vol_eff = price_move_pct / max(total_vol, _EPS)

        return AnchoredOrderflowSnapshot(
            anchored_cvd=anchored_cvd,
            buy_volume=buy_vol,
            sell_volume=sell_vol,
            total_volume=total_vol,
            price_move_pct=price_move_pct,
            cvd_efficiency=cvd_eff,
            volume_efficiency=vol_eff,
            new_extreme_count=self.new_extreme_count,
            last_extreme_price=self.last_extreme_price,
            last_extreme_anchored_cvd=self.last_extreme_anchored_cvd,
        )

    def reset(self) -> None:
        """Clear all tracking state."""
        self.direction = None
        self.anchor_ts_ms = 0
        self.anchor_price = 0.0
        self.anchor_cumulative_cvd = 0.0
        self.anchor_buy_volume = 0.0
        self.anchor_sell_volume = 0.0
        self.last_extreme_price = 0.0
        self.last_extreme_anchored_cvd = 0.0
        self.new_extreme_count = 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _empty_snapshot() -> AnchoredOrderflowSnapshot:
    return AnchoredOrderflowSnapshot(
        anchored_cvd=0.0,
        buy_volume=0.0,
        sell_volume=0.0,
        total_volume=0.0,
        price_move_pct=0.0,
        cvd_efficiency=0.0,
        volume_efficiency=0.0,
        new_extreme_count=0,
        last_extreme_price=0.0,
        last_extreme_anchored_cvd=0.0,
    )
