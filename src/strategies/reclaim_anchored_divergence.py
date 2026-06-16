"""Event-anchored cumulative CVD divergence detector for Reclaim V2.

Reclaim is **not** "price comes back inside the band".  Reclaim is:
price continues making new extremes, but event-anchored cumulative CVD
moves in the OPPOSITE direction — signalling that the original move has
exhausted and the opposing side is taking over.

Uses ``AnchoredOrderflowSnapshot`` from the shared ``anchored_orderflow``
module.  This module only interprets the snapshot — it does not track
state or anchor its own episodes.

This module is pure logic; it has no dependency on strategy state or
exchange adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.strategies.anchored_orderflow import AnchoredOrderflowSnapshot

PositionSide = Literal["LONG", "SHORT"]


# ---------------------------------------------------------------------------
# decision value object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnchoredDivergenceDecision:
    """Result of evaluating whether anchored CVD divergence is confirmed.

    Attributes
    ----------
    confirmed : bool
        ``True`` when the divergence condition is met.
    reason : str
        Human-readable reason (ok / no_new_low / no_new_high /
        cvd_not_recovered / cvd_not_reversed / no_previous_extreme).
    price_extension_pct : float
        How far the current extreme extends beyond the previous one
        (positive fraction, e.g. 0.01 = 1 %).
    cvd_recovery : float
        Absolute change in anchored cumulative CVD between the two
        extremes (positive = CVD moved toward the recovery direction;
        i.e. less bearish for LONG, less bullish for SHORT).
    previous_extreme_price : float
    current_extreme_price : float
    previous_anchored_cvd : float
    current_anchored_cvd : float
    """

    confirmed: bool
    reason: str
    price_extension_pct: float
    cvd_recovery: float
    previous_extreme_price: float
    current_extreme_price: float
    previous_anchored_cvd: float
    current_anchored_cvd: float


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnchoredDivergenceConfig:
    """Configuration for anchored CVD divergence detection.

    Parameters
    ----------
    min_price_extension_pct : float
        Minimum fractional price extension required to consider a new
        valid extreme (e.g. 0.0001 = 0.01 % below previous low).
    min_cvd_recovery : float
        Minimum absolute CVD improvement required for divergence
        confirmation.  A positive value (e.g. 100_000) means CVD must
        recover by at least that much in the opposing direction.
    """
    min_price_extension_pct: float = 0.0
    min_cvd_recovery: float = 0.0


# ---------------------------------------------------------------------------
# public API — snapshot-based (preferred)
# ---------------------------------------------------------------------------


def evaluate_divergence_from_snapshots(
    *,
    side: PositionSide,
    previous_snapshot: AnchoredOrderflowSnapshot | None,
    current_snapshot: AnchoredOrderflowSnapshot | None,
    config: AnchoredDivergenceConfig | None = None,
) -> AnchoredDivergenceDecision:
    """Evaluate anchored CVD divergence using two orderflow snapshots.

    ``previous_snapshot`` is the snapshot at the last valid extreme.
    ``current_snapshot`` is the snapshot at the current (new) extreme.

    LONG (lower side):
        - Price must have made a new low (current.last_extreme_price
          < previous.last_extreme_price).
        - Anchored CVD must recover (current.last_extreme_anchored_cvd
          > previous.last_extreme_anchored_cvd + min_recovery).

    SHORT (upper side):
        - Price must have made a new high (current.last_extreme_price
          > previous.last_extreme_price).
        - Anchored CVD must reverse down (current.last_extreme_anchored_cvd
          < previous.last_extreme_anchored_cvd - min_recovery).
    """
    if previous_snapshot is None or current_snapshot is None:
        return AnchoredDivergenceDecision(
            confirmed=False,
            reason="snapshot_data_missing",
            price_extension_pct=0.0,
            cvd_recovery=0.0,
            previous_extreme_price=0.0,
            current_extreme_price=0.0,
            previous_anchored_cvd=0.0,
            current_anchored_cvd=0.0,
        )

    return evaluate_anchored_divergence(
        side=side,
        previous_extreme_price=previous_snapshot.last_extreme_price,
        previous_anchored_cvd=previous_snapshot.last_extreme_anchored_cvd,
        current_extreme_price=current_snapshot.last_extreme_price,
        current_anchored_cvd=current_snapshot.last_extreme_anchored_cvd,
        config=config,
    )


# ---------------------------------------------------------------------------
# public API — raw-value based (backward-compatible, also used by snapshot API)
# ---------------------------------------------------------------------------


def evaluate_anchored_divergence(
    *,
    side: PositionSide,
    previous_extreme_price: float | None,
    previous_anchored_cvd: float | None,
    current_extreme_price: float | None,
    current_anchored_cvd: float | None,
    config: AnchoredDivergenceConfig | None = None,
) -> AnchoredDivergenceDecision:
    """Evaluate whether anchored CVD divergence is confirmed.

    LONG (lower side):
        - Price must make a new low (current < previous * (1 - extension_pct)).
        - Anchored CVD must recover (current > previous + min_recovery).

    SHORT (upper side):
        - Price must make a new high (current > previous * (1 + extension_pct)).
        - Anchored CVD must reverse down (current < previous - min_recovery).

    Parameters
    ----------
    side : PositionSide
        The side being evaluated — ``LONG`` evaluates lower-side
        divergence; ``SHORT`` evaluates upper-side divergence.
    previous_extreme_price : float | None
        The price at the previous valid extreme.
    previous_anchored_cvd : float | None
        The event-anchored cumulative CVD at the previous extreme.
    current_extreme_price : float | None
        The price at the current (new) potential extreme.
    current_anchored_cvd : float | None
        The event-anchored cumulative CVD at the current extreme.
    config : AnchoredDivergenceConfig | None
        Optional configuration; defaults are used when omitted.

    Returns
    -------
    AnchoredDivergenceDecision
        Structured result — check ``.confirmed``.
    """
    cfg = config or AnchoredDivergenceConfig()
    min_ext_pct = max(float(cfg.min_price_extension_pct), 0.0)
    min_rec = max(float(cfg.min_cvd_recovery), 0.0)

    # ── data presence guards ──────────────────────────────────────────
    if previous_extreme_price is None or current_extreme_price is None:
        return AnchoredDivergenceDecision(
            confirmed=False,
            reason="price_data_missing",
            price_extension_pct=0.0,
            cvd_recovery=0.0,
            previous_extreme_price=previous_extreme_price or 0.0,
            current_extreme_price=current_extreme_price or 0.0,
            previous_anchored_cvd=previous_anchored_cvd or 0.0,
            current_anchored_cvd=current_anchored_cvd or 0.0,
        )
    if previous_anchored_cvd is None or current_anchored_cvd is None:
        return AnchoredDivergenceDecision(
            confirmed=False,
            reason="cvd_data_missing",
            price_extension_pct=0.0,
            cvd_recovery=0.0,
            previous_extreme_price=previous_extreme_price,
            current_extreme_price=current_extreme_price,
            previous_anchored_cvd=previous_anchored_cvd or 0.0,
            current_anchored_cvd=current_anchored_cvd or 0.0,
        )

    if previous_extreme_price <= 0 or current_extreme_price <= 0:
        return AnchoredDivergenceDecision(
            confirmed=False,
            reason="price_data_missing",
            price_extension_pct=0.0,
            cvd_recovery=0.0,
            previous_extreme_price=previous_extreme_price,
            current_extreme_price=current_extreme_price,
            previous_anchored_cvd=previous_anchored_cvd,
            current_anchored_cvd=current_anchored_cvd,
        )

    # ── evaluate ──────────────────────────────────────────────────────
    if side == "LONG":
        return _evaluate_long(
            prev_price=previous_extreme_price,
            prev_cvd=previous_anchored_cvd,
            curr_price=current_extreme_price,
            curr_cvd=current_anchored_cvd,
            min_ext_pct=min_ext_pct,
            min_rec=min_rec,
        )
    return _evaluate_short(
        prev_price=previous_extreme_price,
        prev_cvd=previous_anchored_cvd,
        curr_price=current_extreme_price,
        curr_cvd=current_anchored_cvd,
        min_ext_pct=min_ext_pct,
        min_rec=min_rec,
    )


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------


def _price_extension_pct(
    side: PositionSide,
    previous: float,
    current: float,
) -> float:
    """Return the absolute fractional price extension.

    LONG: (previous - current) / previous  (how much lower).
    SHORT: (current - previous) / previous (how much higher).
    """
    if previous <= 0:
        return 0.0
    if side == "LONG":
        return max((previous - current) / previous, 0.0)
    return max((current - previous) / previous, 0.0)


def _evaluate_long(
    *,
    prev_price: float,
    prev_cvd: float,
    curr_price: float,
    curr_cvd: float,
    min_ext_pct: float,
    min_rec: float,
) -> AnchoredDivergenceDecision:
    ext_pct = _price_extension_pct("LONG", prev_price, curr_price)
    cvd_recovery = curr_cvd - prev_cvd  # positive = CVD improved (less bearish)

    # ── price new low check ───────────────────────────────────────────
    if curr_price >= prev_price * (1.0 - min_ext_pct):
        return AnchoredDivergenceDecision(
            confirmed=False,
            reason="no_new_low",
            price_extension_pct=ext_pct,
            cvd_recovery=cvd_recovery,
            previous_extreme_price=prev_price,
            current_extreme_price=curr_price,
            previous_anchored_cvd=prev_cvd,
            current_anchored_cvd=curr_cvd,
        )

    # ── CVD recovery check ────────────────────────────────────────────
    if curr_cvd > prev_cvd + min_rec:
        return AnchoredDivergenceDecision(
            confirmed=True,
            reason="ok",
            price_extension_pct=ext_pct,
            cvd_recovery=cvd_recovery,
            previous_extreme_price=prev_price,
            current_extreme_price=curr_price,
            previous_anchored_cvd=prev_cvd,
            current_anchored_cvd=curr_cvd,
        )

    return AnchoredDivergenceDecision(
        confirmed=False,
        reason="cvd_not_recovered",
        price_extension_pct=ext_pct,
        cvd_recovery=cvd_recovery,
        previous_extreme_price=prev_price,
        current_extreme_price=curr_price,
        previous_anchored_cvd=prev_cvd,
        current_anchored_cvd=curr_cvd,
    )


def _evaluate_short(
    *,
    prev_price: float,
    prev_cvd: float,
    curr_price: float,
    curr_cvd: float,
    min_ext_pct: float,
    min_rec: float,
) -> AnchoredDivergenceDecision:
    ext_pct = _price_extension_pct("SHORT", prev_price, curr_price)
    cvd_recovery = prev_cvd - curr_cvd  # positive = CVD reversed down (less bullish)

    # ── price new high check ──────────────────────────────────────────
    if curr_price <= prev_price * (1.0 + min_ext_pct):
        return AnchoredDivergenceDecision(
            confirmed=False,
            reason="no_new_high",
            price_extension_pct=ext_pct,
            cvd_recovery=cvd_recovery,
            previous_extreme_price=prev_price,
            current_extreme_price=curr_price,
            previous_anchored_cvd=prev_cvd,
            current_anchored_cvd=curr_cvd,
        )

    # ── CVD reversal check ────────────────────────────────────────────
    if curr_cvd < prev_cvd - min_rec:
        return AnchoredDivergenceDecision(
            confirmed=True,
            reason="ok",
            price_extension_pct=ext_pct,
            cvd_recovery=cvd_recovery,
            previous_extreme_price=prev_price,
            current_extreme_price=curr_price,
            previous_anchored_cvd=prev_cvd,
            current_anchored_cvd=curr_cvd,
        )

    return AnchoredDivergenceDecision(
        confirmed=False,
        reason="cvd_not_reversed",
        price_extension_pct=ext_pct,
        cvd_recovery=cvd_recovery,
        previous_extreme_price=prev_price,
        current_extreme_price=curr_price,
        previous_anchored_cvd=prev_cvd,
        current_anchored_cvd=curr_cvd,
    )
