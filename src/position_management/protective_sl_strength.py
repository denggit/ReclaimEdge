"""Protective SL strength comparison — enforce no-loosen rule.

LONG:  higher SL = stronger (tighter to entry, more protective).
SHORT: lower SL = stronger (tighter to entry, more protective).

These helpers are used in protective_orders_phase to decide whether a
new candidate SL should replace an existing SL, or whether the existing
stronger SL should be kept.
"""

from __future__ import annotations


def stronger_sl_price(
    *,
    side: str | None,
    existing_sl_price: float | None,
    candidate_sl_price: float | None,
) -> float | None:
    """Return the stronger (more protective) SL price.

    LONG:
        higher SL is stronger.
    SHORT:
        lower SL is stronger.

    If one side is missing, return the other.
    If side is unknown, prefer existing if present, otherwise candidate.
    """
    if existing_sl_price is None and candidate_sl_price is None:
        return None
    if existing_sl_price is None:
        return candidate_sl_price
    if candidate_sl_price is None:
        return existing_sl_price

    if side == "LONG":
        # higher = stronger
        return existing_sl_price if existing_sl_price >= candidate_sl_price else candidate_sl_price
    elif side == "SHORT":
        # lower = stronger
        return existing_sl_price if existing_sl_price <= candidate_sl_price else candidate_sl_price
    else:
        # side unknown — prefer existing
        return existing_sl_price


def should_replace_sl(
    *,
    side: str | None,
    existing_sl_price: float | None,
    candidate_sl_price: float | None,
) -> bool:
    """True only when candidate exists and is strictly stronger than existing.

    - existing None, candidate not None => True
    - candidate None => False
    - LONG:  candidate > existing  => True
    - SHORT: candidate < existing  => True
    - side unknown: existing exists => False; existing None & candidate exists => True
    """
    if candidate_sl_price is None:
        return False
    if existing_sl_price is None:
        return True

    if side == "LONG":
        return candidate_sl_price > existing_sl_price
    elif side == "SHORT":
        return candidate_sl_price < existing_sl_price
    else:
        # side unknown — keep existing
        return False
