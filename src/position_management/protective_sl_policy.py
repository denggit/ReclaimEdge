from __future__ import annotations


def is_existing_sl_at_least_as_protective(
    *,
    side: str | None,
    existing_sl_price: float | None,
    candidate_sl_price: float | None,
) -> bool:
    if existing_sl_price is None or candidate_sl_price is None:
        return False
    if side == "LONG":
        return existing_sl_price >= candidate_sl_price
    if side == "SHORT":
        return existing_sl_price <= candidate_sl_price
    return False


def has_existing_protective_sl(
    *,
    old_sl_order_id: str | None,
    old_sl_price: float | None,
    old_protected: bool = False,
) -> bool:
    return bool(old_sl_order_id) and old_sl_price is not None and bool(old_protected)
