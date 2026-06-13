"""Boundary tests that pin the account sync protective orders phase contract.

These tests are SOURCE-LEVEL assertions.  They ensure that
``protective_orders_phase.py`` handles protective stop (re)placement and
cancellation exclusively through high-level ``Trader`` methods and does NOT
reach behind the abstraction to call OKX endpoints, ``order_specs``, or
``broker_semantic_executor`` directly.

If a future refactor moves the exchange adapter boundary, the change belongs
in ``Trader`` or the TP/SL manager, not in ``protective_orders_phase.py``.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROTECTIVE_PHASE_PATH = Path("src/live/account_sync/protective_orders_phase.py")


def _protective_phase_text() -> str:
    return _PROTECTIVE_PHASE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. MUST NOT directly call OKX endpoints or low-level HTTP request
# ---------------------------------------------------------------------------

def test_protective_orders_phase_does_not_directly_call_okx_endpoints() -> None:
    """protective_orders_phase MUST NOT contain any direct OKX REST endpoint
    reference or low-level HTTP request call."""
    text = _protective_phase_text()

    forbidden = [
        "/api/v5/",
        'request("GET"',
        "request('GET'",
        'request("POST"',
        "request('POST'",
        "orders-pending",
        "orders-algo-pending",
        "trade/order",
        "trade/order-algo",
        "trade/cancel-order",
    ]

    for token in forbidden:
        assert token not in text, (
            f"'{token}' should not appear in protective_orders_phase.py — "
            "the module must not directly query OKX endpoints"
        )


# ---------------------------------------------------------------------------
# 2. MUST NOT import order_specs or exchange adapters
# ---------------------------------------------------------------------------

def test_protective_orders_phase_does_not_import_order_specs_or_exchange_adapters() -> None:
    """protective_orders_phase MUST NOT import order_specs or any exchange
    adapter classes."""
    text = _protective_phase_text()

    forbidden = [
        "from src.execution import order_specs",
        "import order_specs",
        "src.exchanges",
        "OkxBroker",
        "OkxBrokerClient",
        "OkxBrokerSemanticExecutor",
    ]

    for token in forbidden:
        assert token not in text, (
            f"'{token}' should not appear in protective_orders_phase.py — "
            "the module must not import order_specs or exchange adapters"
        )


# ---------------------------------------------------------------------------
# 3. MUST NOT call broker_semantic_executor / fetch_broker
# ---------------------------------------------------------------------------

def test_protective_orders_phase_does_not_directly_call_broker_semantic_executor() -> None:
    """protective_orders_phase MUST NOT reference broker_semantic_executor
    or fetch_broker."""
    text = _protective_phase_text()

    forbidden = [
        "broker_semantic_executor",
        "fetch_broker",
        "place_protective_stop(",
        "cancel_protective_stop(",
    ]

    for token in forbidden:
        assert token not in text, (
            f"'{token}' should not appear in protective_orders_phase.py — "
            "the module must not call broker_semantic_executor or low-level "
            "protective stop functions directly"
        )


# ---------------------------------------------------------------------------
# 4. MUST use high-level Trader protective order methods
# ---------------------------------------------------------------------------

def test_protective_orders_phase_uses_trader_high_level_protective_methods() -> None:
    """protective_orders_phase MUST handle protective stops exclusively
    through high-level Trader methods — these are the actual method names
    found in the current module."""
    text = _protective_phase_text()

    # Only methods that ACTUALLY appear in the current source.
    # Verified via: grep -n "trader\." src/live/account_sync/protective_orders_phase.py
    expected = [
        "trader.cancel_three_stage_post_tp1_protective_stop(",
        "trader.place_three_stage_post_tp1_protective_stop_with_retries(",
        "trader.cancel_middle_bucket_fast_protective_stop(",
        "trader.place_middle_runner_protective_stop_with_retries(",
        "trader.cancel_middle_runner_protective_stop(",
        "trader.place_middle_bucket_fast_protective_stop_with_retries(",
    ]

    for token in expected:
        assert token in text, (
            f"'{token}' must remain the protective order boundary in "
            "protective_orders_phase.py"
        )
