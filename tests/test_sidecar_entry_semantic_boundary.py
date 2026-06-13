"""Source-level boundary tests for sidecar entry runtime.

These tests lock the boundary that the current sidecar "entry" is NOT a standalone
exchange order.  The sidecar leg is derived from the already-filled combined core
entry, then protected by placing a dedicated sidecar TP order.

Do NOT add broker_semantic_executor.sidecar_entry() wiring or
BROKER_SEMANTIC_SIDECAR_ENTRY_ENABLED until the runtime changes to place an
independent sidecar entry order with its own fill lifecycle.
"""

from __future__ import annotations

from pathlib import Path


_ENTRY_RUNTIME_PATH = "src/position_management/sidecar/entry_runtime.py"


def test_sidecar_entry_runtime_documents_no_standalone_entry_order() -> None:
    text = Path(_ENTRY_RUNTIME_PATH).read_text(encoding="utf-8")

    assert 'Current sidecar "entry" is not a standalone exchange order' in text
    assert "already-filled combined core entry" in text
    assert "dedicated sidecar TP order" in text


def test_sidecar_entry_runtime_does_not_wire_semantic_sidecar_entry() -> None:
    text = Path(_ENTRY_RUNTIME_PATH).read_text(encoding="utf-8")

    forbidden = [
        "broker_semantic_executor.sidecar_entry(",
        "BROKER_SEMANTIC_SIDECAR_ENTRY_ENABLED",
    ]
    for token in forbidden:
        assert token not in text, (
            f"{token} should not be wired until sidecar entry is a standalone order"
        )


def test_sidecar_entry_runtime_does_not_directly_place_exchange_order() -> None:
    text = Path(_ENTRY_RUNTIME_PATH).read_text(encoding="utf-8")

    forbidden = [
        'request("POST", "/api/v5/trade/order"',
        "request('POST', '/api/v5/trade/order'",
        "build_market_entry_order_body",
    ]
    for token in forbidden:
        assert token not in text, (
            f"{token} should not appear in sidecar entry runtime"
        )


def test_sidecar_entry_runtime_uses_sidecar_tp_attach_path() -> None:
    text = Path(_ENTRY_RUNTIME_PATH).read_text(encoding="utf-8")

    assert "_place_sidecar_tp_with_rate_limit_retry" in text
    assert "trader.place_sidecar_fixed_take_profit" in text
    assert "sidecar_leg_from_fill" in text
