"""Trend middle trailing stop — pure logic module.

Calculates and tightens the BOLL20-middle-based trailing stop-loss for
trend breakout positions.  No exchange calls, no I/O.
"""

from __future__ import annotations


def calculate_trend_middle_sl(
    *,
    boll_middle: float,
    buffer_pct: float,
    side: str,
) -> float:
    """Calculate the trend trailing SL anchored at the BOLL middle band.

    For LONG: ``boll_middle * (1 - buffer_pct)``   (SL sits *below* middle)
    For SHORT: ``boll_middle * (1 + buffer_pct)``  (SL sits *above* middle)

    Raises:
        ValueError: if *boll_middle* is not positive or *buffer_pct* is negative.
    """
    if boll_middle <= 0:
        raise ValueError(f"boll_middle must be > 0, got {boll_middle}")
    if buffer_pct < 0:
        raise ValueError(f"buffer_pct must be >= 0, got {buffer_pct}")

    if side == "LONG":
        return boll_middle * (1.0 - buffer_pct)
    # SHORT
    return boll_middle * (1.0 + buffer_pct)


def tighten_trend_sl(
    *,
    old_sl: float | None,
    candidate_sl: float,
    current_price: float,
    side: str,
) -> float | None:
    """Tighten the trailing SL — only tightens, never loosens.

    Validation (per-side):
      * LONG:  candidate must be **below** current price.
      * SHORT: candidate must be **above** current price.

    Tightening logic (per-side):
      * LONG:  ``max(old_sl, candidate_sl)``  → higher SL = tighter
      * SHORT: ``min(old_sl, candidate_sl)``  → lower SL  = tighter

    Returns:
        The new SL price if it is a valid tightening, ``None`` otherwise
        (invalid candidate, no old SL to compare, unchanged, or loosening).
    """
    if side == "LONG":
        if candidate_sl >= current_price:
            return None  # invalid — SL would be at or above entry
        if old_sl is None:
            return candidate_sl
        new_sl = max(old_sl, candidate_sl)
        if new_sl == old_sl:
            return None  # no tightening
        return new_sl

    # SHORT
    if candidate_sl <= current_price:
        return None  # invalid — SL would be at or below entry
    if old_sl is None:
        return candidate_sl
    new_sl = min(old_sl, candidate_sl)
    if new_sl == old_sl:
        return None  # no tightening
    return new_sl


def is_trend_sl_tightened(
    *,
    old_sl: float | None,
    new_sl: float | None,
    side: str,
) -> bool:
    """Return ``True`` when *new_sl* is strictly tighter than *old_sl*.

    * ``None`` → ``float`` is always a tightening (initial SL).
    * ``None`` → ``None`` or ``float`` → ``None`` is never a tightening.
    """
    if old_sl is None and new_sl is not None:
        return True
    if old_sl is None or new_sl is None:
        return False
    if side == "LONG":
        return new_sl > old_sl  # higher = tighter
    return new_sl < old_sl      # lower  = tighter
