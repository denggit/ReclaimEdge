"""Broker-semantic bridge helpers.

Centralized request builders and side-mapping utilities that convert
execution-layer concepts (PositionSide strings, contracts, prices) into
``BrokerSemanticRequest`` objects.

These helpers do **not** make network requests, do **not** construct OKX
endpoint bodies, and do **not** import the Trader class.  They are pure
data converters.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from src.exchanges.models import BrokerOrderSide, BrokerPositionSide, ExchangeName
from src.exchanges.semantic_models import (
    BrokerSemanticAction,
    BrokerSemanticOrderRole,
    BrokerSemanticRequest,
)


# ---------------------------------------------------------------------------
# Side mapping
# ---------------------------------------------------------------------------


def close_order_side(position_side: str) -> BrokerOrderSide:
    """Map a ``PositionSide`` string to the close-side ``BrokerOrderSide``.

    >>> close_order_side("LONG")
    <BrokerOrderSide.SELL: 'SELL'>
    >>> close_order_side("SHORT")
    <BrokerOrderSide.BUY: 'BUY'>
    """
    return BrokerOrderSide.SELL if str(position_side).upper() == "LONG" else BrokerOrderSide.BUY


def entry_order_side(position_side: str) -> BrokerOrderSide:
    """Map a ``PositionSide`` string to the entry-side ``BrokerOrderSide``.

    >>> entry_order_side("LONG")
    <BrokerOrderSide.BUY: 'BUY'>
    >>> entry_order_side("SHORT")
    <BrokerOrderSide.SELL: 'SELL'>
    """
    return BrokerOrderSide.BUY if str(position_side).upper() == "LONG" else BrokerOrderSide.SELL


def broker_position_side(position_side: str) -> BrokerPositionSide:
    """Map a ``PositionSide`` string to ``BrokerPositionSide``.

    >>> broker_position_side("LONG")
    <BrokerPositionSide.LONG: 'LONG'>
    >>> broker_position_side("SHORT")
    <BrokerPositionSide.SHORT: 'SHORT'>
    """
    return BrokerPositionSide.LONG if str(position_side).upper() == "LONG" else BrokerPositionSide.SHORT


# ---------------------------------------------------------------------------
# Trader introspection
# ---------------------------------------------------------------------------


def get_broker_semantic_executor(trader: object) -> object | None:
    """Return the semantic executor attached to *trader*, or ``None``.

    Does **not** create a new executor — only reads the attribute.
    Callers should fall back to the legacy OKX body/request path when
    ``None`` is returned.
    """
    return getattr(trader, "broker_semantic_executor", None)


# ---------------------------------------------------------------------------
# Request builders — placement
# ---------------------------------------------------------------------------


def build_reduce_only_tp_request(
    *,
    symbol: str,
    side: str,
    contracts: Decimal,
    price: Decimal,
    role: BrokerSemanticOrderRole = BrokerSemanticOrderRole.CORE_TP,
    client_order_id: str | None = None,
    label: str | None = None,
    exchange: ExchangeName = ExchangeName.OKX,
) -> BrokerSemanticRequest:
    """Build a ``PLACE_REDUCE_ONLY_TP`` semantic request."""
    return BrokerSemanticRequest(
        exchange=exchange,
        symbol=symbol,
        action=BrokerSemanticAction.PLACE_REDUCE_ONLY_TP,
        role=role,
        side=close_order_side(side),
        position_side=broker_position_side(side),
        quantity=contracts,
        price=price,
        reduce_only=True,
        client_order_id=client_order_id,
        label=label,
    )


def build_protective_stop_request(
    *,
    symbol: str,
    side: str,
    contracts: Decimal,
    stop_price: Decimal,
    role: BrokerSemanticOrderRole = BrokerSemanticOrderRole.PROTECTIVE_SL,
    client_order_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    exchange: ExchangeName = ExchangeName.OKX,
) -> BrokerSemanticRequest:
    """Build a ``PLACE_PROTECTIVE_STOP`` semantic request."""
    return BrokerSemanticRequest(
        exchange=exchange,
        symbol=symbol,
        action=BrokerSemanticAction.PLACE_PROTECTIVE_STOP,
        role=role,
        side=close_order_side(side),
        position_side=broker_position_side(side),
        quantity=contracts,
        trigger_price=stop_price,
        reduce_only=True,
        close_position=True,
        client_order_id=client_order_id,
        metadata=metadata or {},
    )


def build_market_exit_request(
    *,
    symbol: str,
    side: str,
    contracts: Decimal,
    role: BrokerSemanticOrderRole = BrokerSemanticOrderRole.MARKET_EXIT,
    context: str = "generic",
    exchange: ExchangeName = ExchangeName.OKX,
) -> BrokerSemanticRequest:
    """Build a ``MARKET_EXIT`` or ``MARKET_EXIT_RUNNER`` semantic request."""
    action = (
        BrokerSemanticAction.MARKET_EXIT_RUNNER
        if "runner" in str(context)
        else BrokerSemanticAction.MARKET_EXIT
    )
    return BrokerSemanticRequest(
        exchange=exchange,
        symbol=symbol,
        action=action,
        role=role,
        side=close_order_side(side),
        position_side=broker_position_side(side),
        quantity=contracts,
        reduce_only=True,
        close_position=True,
        metadata={"context": context},
    )


# ---------------------------------------------------------------------------
# Request builders — cancel
# ---------------------------------------------------------------------------


def build_cancel_reduce_only_tp_request(
    *,
    symbol: str,
    order_id: str,
    role: BrokerSemanticOrderRole = BrokerSemanticOrderRole.CORE_TP,
    exchange: ExchangeName = ExchangeName.OKX,
) -> BrokerSemanticRequest:
    """Build a ``CANCEL_REDUCE_ONLY_TP`` semantic request."""
    return BrokerSemanticRequest(
        exchange=exchange,
        symbol=symbol,
        action=BrokerSemanticAction.CANCEL_REDUCE_ONLY_TP,
        role=role,
        order_id=order_id,
    )


def build_cancel_protective_stop_request(
    *,
    symbol: str,
    order_id: str,
    role: BrokerSemanticOrderRole = BrokerSemanticOrderRole.PROTECTIVE_SL,
    exchange: ExchangeName = ExchangeName.OKX,
) -> BrokerSemanticRequest:
    """Build a ``CANCEL_PROTECTIVE_STOP`` semantic request."""
    return BrokerSemanticRequest(
        exchange=exchange,
        symbol=symbol,
        action=BrokerSemanticAction.CANCEL_PROTECTIVE_STOP,
        role=role,
        order_id=order_id,
    )


# ---------------------------------------------------------------------------
# Request builders — sidecar
# ---------------------------------------------------------------------------


def build_sidecar_entry_request(
    *,
    symbol: str,
    side: str,
    contracts: Decimal,
    exchange: ExchangeName = ExchangeName.OKX,
) -> BrokerSemanticRequest:
    """Build a ``SIDECAR_ENTRY`` semantic request."""
    return BrokerSemanticRequest(
        exchange=exchange,
        symbol=symbol,
        action=BrokerSemanticAction.SIDECAR_ENTRY,
        role=BrokerSemanticOrderRole.SIDECAR_ENTRY,
        side=entry_order_side(side),
        position_side=broker_position_side(side),
        quantity=contracts,
    )


def build_sidecar_tp_request(
    *,
    symbol: str,
    side: str,
    contracts: Decimal,
    tp_price: Decimal,
    client_order_id: str | None = None,
    exchange: ExchangeName = ExchangeName.OKX,
) -> BrokerSemanticRequest:
    """Build a ``SIDECAR_TP`` semantic request."""
    return BrokerSemanticRequest(
        exchange=exchange,
        symbol=symbol,
        action=BrokerSemanticAction.SIDECAR_TP,
        role=BrokerSemanticOrderRole.SIDECAR_TP,
        side=close_order_side(side),
        position_side=broker_position_side(side),
        quantity=contracts,
        price=tp_price,
        reduce_only=True,
        client_order_id=client_order_id,
    )
