from __future__ import annotations

import pytest

from src.position_management.sidecar.fill_normalization import (
    SidecarFillSnapshot,
    normalize_sidecar_tp_fill,
    safe_positive_float,
)

# ── safe_positive_float tests ────────────────────────────────────────────────


def test_safe_positive_float_returns_float_for_positive() -> None:
    assert safe_positive_float(3.14) == 3.14
    assert safe_positive_float("2.0") == 2.0
    assert safe_positive_float(1) == 1.0


def test_safe_positive_float_returns_none_for_non_positive() -> None:
    assert safe_positive_float(0) is None
    assert safe_positive_float(-1) is None
    assert safe_positive_float("0") is None
    assert safe_positive_float("-0.5") is None


def test_safe_positive_float_returns_none_for_invalid() -> None:
    assert safe_positive_float(None) is None
    assert safe_positive_float("") is None
    assert safe_positive_float("abc") is None
    assert safe_positive_float([]) is None
    assert safe_positive_float({}) is None


# ── normalize_sidecar_tp_fill fallback tests ────────────────────────────────


def test_missing_contracts_falls_back_to_qty_divided_by_multiplier() -> None:
    """When leg has no 'contracts' key, filled_contracts must be computed
    as leg_qty / contract_multiplier (leg qty is ETH, not contracts)."""
    leg = {"qty": 0.095, "tp_price": 1627.34}
    status = {"filled_qty": None}

    snap = normalize_sidecar_tp_fill(leg=leg, status=status, contract_multiplier=0.1)

    # filled_eth_qty falls back to leg qty = 0.095 ETH
    assert snap.filled_eth_qty == pytest.approx(0.095)
    # filled_contracts = 0.095 / 0.1 = 0.95
    assert snap.filled_contracts == pytest.approx(0.95)


def test_missing_contracts_with_status_filled_qty_still_uses_status() -> None:
    """status filled_qty takes priority — contracts from OKX accFillSz."""
    leg = {"qty": 0.1}
    status = {"filled_qty": 0.95, "avg_fill_price": 1627.34}

    snap = normalize_sidecar_tp_fill(leg=leg, status=status, contract_multiplier=0.1)

    assert snap.filled_contracts == 0.95
    assert snap.filled_eth_qty == pytest.approx(0.095)


def test_contract_multiplier_zero_does_not_raise() -> None:
    """contract_multiplier=0 must not raise ZeroDivisionError."""
    leg = {"qty": 0.1, "tp_price": 3000.0}
    status = {"filled_qty": None}

    snap = normalize_sidecar_tp_fill(leg=leg, status=status, contract_multiplier=0)

    # filled_eth_qty falls back to leg qty
    assert snap.filled_eth_qty == pytest.approx(0.1)
    # filled_contracts: can't divide by 0, falls back to leg_qty as last resort
    assert snap.filled_contracts == pytest.approx(0.1)


def test_contract_multiplier_negative_does_not_raise() -> None:
    """contract_multiplier=-0.1 must not raise."""
    leg = {"qty": 0.2, "tp_price": 3000.0}
    status: dict = {}

    snap = normalize_sidecar_tp_fill(leg=leg, status=status, contract_multiplier=-0.1)

    # filled_eth_qty from status (empty) falls back to leg qty
    assert snap.filled_eth_qty == pytest.approx(0.2)
    # filled_contracts: contract_multiplier <= 0, falls back to leg_qty as last resort
    assert snap.filled_contracts == pytest.approx(0.2)


def test_both_qty_and_contracts_missing_returns_zero() -> None:
    """When leg has neither 'contracts' nor 'qty', both should be 0."""
    leg = {"leg_id": "empty"}
    status: dict = {}

    snap = normalize_sidecar_tp_fill(leg=leg, status=status)

    assert snap.filled_eth_qty == 0.0
    assert snap.filled_contracts == 0.0
    assert snap.filled_notional_usdt is None


def test_missing_contracts_with_qty_string() -> None:
    """leg qty as string should be parsed correctly."""
    leg = {"qty": "0.05", "tp_price": 2000.0}
    status: dict = {}

    snap = normalize_sidecar_tp_fill(leg=leg, status=status, contract_multiplier=0.1)

    assert snap.filled_eth_qty == pytest.approx(0.05)
    assert snap.filled_contracts == pytest.approx(0.5)


# ── Docstring / SidecarFillSnapshot verification tests ─────────────────────


def test_snapshot_docstring_clarifies_units() -> None:
    """Verify SidecarFillSnapshot docstring clearly states the unit for each field."""
    doc = SidecarFillSnapshot.__doc__ or ""
    assert "filled_contracts" in doc
    assert "filled_eth_qty" in doc
    assert "contracts" in doc.lower()
    assert "ETH" in doc
    # Must NOT say "All quantities are in ETH"
    assert "All quantities are in ETH" not in doc


def test_filled_eth_qty_and_filled_contracts_are_separate() -> None:
    """Verify that filled_eth_qty and filled_contracts are distinct fields
    with different semantics (ETH vs contracts)."""
    snap = SidecarFillSnapshot(
        leg_id="test",
        order_id="o1",
        filled_contracts=1.0,
        filled_eth_qty=0.1,
    )
    assert snap.filled_contracts == 1.0
    assert snap.filled_eth_qty == 0.1
    assert snap.filled_contracts != snap.filled_eth_qty
