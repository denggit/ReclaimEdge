"""Boundary tests that pin the sidecar pre-core reconciliation status query contract.

These tests are SOURCE-LEVEL assertions.  They ensure that
``pre_core_reconcile.py`` queries sidecar TP status exclusively through
``trader.fetch_sidecar_order_status(order_id)`` and does NOT reach behind the
Trader abstraction to call OKX endpoints or the broker_semantic_executor directly.

If a future refactor moves the adapter boundary, the change belongs in
``Trader.fetch_sidecar_order_status()`` or ``SidecarTpManager.fetch_sidecar_order_status()``,
not in ``pre_core_reconcile.py``.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pre_core_reconcile_text() -> str:
    return Path(
        "src/position_management/sidecar/pre_core_reconcile.py"
    ).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. MUST use trader.fetch_sidecar_order_status
# ---------------------------------------------------------------------------

def test_pre_core_reconcile_uses_trader_sidecar_status_boundary() -> None:
    """The ONLY sidecar TP status query in pre_core_reconcile is
    ``trader.fetch_sidecar_order_status(order_id)``."""
    text = _pre_core_reconcile_text()
    assert "trader.fetch_sidecar_order_status(order_id)" in text, (
        "pre_core_reconcile MUST query sidecar TP status through "
        "trader.fetch_sidecar_order_status(order_id)"
    )


# ---------------------------------------------------------------------------
# 2. MUST NOT directly query OKX order endpoint
# ---------------------------------------------------------------------------

def test_pre_core_reconcile_does_not_directly_query_okx_order_endpoint() -> None:
    """pre_core_reconcile MUST NOT contain any direct OKX REST endpoint
    reference or low-level HTTP request call."""
    text = _pre_core_reconcile_text()

    forbidden = [
        "/api/v5/trade/order",
        "orders-pending",
        "orders-algo-pending",
        'request("GET"',
        "request('GET'",
    ]
    for token in forbidden:
        assert token not in text, (
            f"'{token}' should not appear in pre_core_reconcile.py — "
            "the module must not directly query OKX endpoints"
        )


# ---------------------------------------------------------------------------
# 3. MUST NOT call broker_semantic_executor
# ---------------------------------------------------------------------------

def test_pre_core_reconcile_does_not_directly_call_broker_semantic_executor() -> None:
    """pre_core_reconcile MUST NOT import or call broker_semantic_executor
    or any fetch_broker variant."""
    text = _pre_core_reconcile_text()

    assert "broker_semantic_executor" not in text, (
        "pre_core_reconcile MUST NOT reference broker_semantic_executor"
    )
    assert "fetch_broker" not in text, (
        "pre_core_reconcile MUST NOT call fetch_broker"
    )


# ---------------------------------------------------------------------------
# 4. MUST consume a normalized status dict
# ---------------------------------------------------------------------------

def test_pre_core_reconcile_uses_normalized_status_dict_contract() -> None:
    """pre_core_reconcile reads ``status.get("status")`` and branches on
    the five known normalized status values."""
    text = _pre_core_reconcile_text()

    assert 'status.get("status")' in text, (
        "pre_core_reconcile MUST read the normalized 'status' key"
    )
    assert '"OPEN"' in text, (
        "pre_core_reconcile MUST handle normalized 'OPEN'"
    )
    assert '"FILLED"' in text, (
        "pre_core_reconcile MUST handle normalized 'FILLED'"
    )
    assert '"CANCELED"' in text, (
        "pre_core_reconcile MUST handle normalized 'CANCELED'"
    )
    assert '"NOT_FOUND"' in text, (
        "pre_core_reconcile MUST handle normalized 'NOT_FOUND'"
    )
    assert '"UNKNOWN"' in text, (
        "pre_core_reconcile MUST handle normalized 'UNKNOWN'"
    )
