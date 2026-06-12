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
# Internal — side normalization (strict)
# ---------------------------------------------------------------------------


def _normalize_position_side(position_side: str) -> str:
    """Normalize *position_side* to ``"LONG"`` or ``"SHORT"``.

    Raises ``ValueError`` for unrecognised values so that silent
    mis-mapping cannot slip into a live trade.
    """
    side = str(position_side).upper().strip()
    if side not in {"LONG", "SHORT"}:
        raise ValueError(
            f"Unsupported position side for broker semantic bridge: {position_side!r}"
        )
    return side


# ---------------------------------------------------------------------------
# Public side mapping
# ---------------------------------------------------------------------------


def close_order_side(position_side: str) -> BrokerOrderSide:
    """Map a ``PositionSide`` string to the close-side ``BrokerOrderSide``.

    >>> close_order_side("LONG")
    <BrokerOrderSide.SELL: 'SELL'>
    >>> close_order_side("SHORT")
    <BrokerOrderSide.BUY: 'BUY'>
    """
    side = _normalize_position_side(position_side)
    if side == "LONG":
        return BrokerOrderSide.SELL
    return BrokerOrderSide.BUY


def entry_order_side(position_side: str) -> BrokerOrderSide:
    """Map a ``PositionSide`` string to the entry-side ``BrokerOrderSide``.

    >>> entry_order_side("LONG")
    <BrokerOrderSide.BUY: 'BUY'>
    >>> entry_order_side("SHORT")
    <BrokerOrderSide.SELL: 'SELL'>
    """
    side = _normalize_position_side(position_side)
    if side == "LONG":
        return BrokerOrderSide.BUY
    return BrokerOrderSide.SELL


def broker_position_side(position_side: str) -> BrokerPositionSide:
    """Map a ``PositionSide`` string to ``BrokerPositionSide``.

    >>> broker_position_side("LONG")
    <BrokerPositionSide.LONG: 'LONG'>
    >>> broker_position_side("SHORT")
    <BrokerPositionSide.SHORT: 'SHORT'>
    """
    side = _normalize_position_side(position_side)
    if side == "LONG":
        return BrokerPositionSide.LONG
    return BrokerPositionSide.SHORT


# ---------------------------------------------------------------------------
# TP role classifier
# ---------------------------------------------------------------------------


def semantic_tp_role(label: str | None) -> BrokerSemanticOrderRole:
    """Classify a TP order *label* into the closest semantic role.

    >>> semantic_tp_role("tp1")
    <BrokerSemanticOrderRole.TP1: 'TP1'>
    >>> semantic_tp_role("runner")
    <BrokerSemanticOrderRole.RUNNER_TP: 'RUNNER_TP'>
    >>> semantic_tp_role("unknown")
    <BrokerSemanticOrderRole.CORE_TP: 'CORE_TP'>
    """
    normalized = str(label or "").lower()
    if normalized in {
        "tp1", "tp1_middle", "tp1_middle_fast", "tp1_middle_slow",
        "middle", "middle_fast", "middle_slow",
    }:
        return BrokerSemanticOrderRole.TP1
    if normalized in {"tp2", "tp2_outer"}:
        return BrokerSemanticOrderRole.TP2
    if normalized == "runner":
        return BrokerSemanticOrderRole.RUNNER_TP
    return BrokerSemanticOrderRole.CORE_TP


# ---------------------------------------------------------------------------
# Trader introspection
# ---------------------------------------------------------------------------


def get_broker_semantic_executor(trader: object) -> object | None:
    """Return the semantic executor attached to *trader*, or ``None``.

    Checks both the ``broker_semantic_executor`` property (which may
    lazily initialise the executor) and the ``_broker_semantic_executor``
    private attribute.

    Callers should fall back to the legacy OKX body/request path when
    ``None`` is returned.
    """
    executor = getattr(trader, "broker_semantic_executor", None)
    if executor is not None:
        return executor
    return getattr(trader, "_broker_semantic_executor", None)


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
    """Build a ``MARKET_EXIT`` or ``MARKET_EXIT_RUNNER`` semantic request.

    The action is ``MARKET_EXIT_RUNNER`` when *context* contains
    ``"runner"`` (case-insensitive), otherwise ``MARKET_EXIT``.
    """
    action = (
        BrokerSemanticAction.MARKET_EXIT_RUNNER
        if "runner" in str(context).lower()
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


# ---------------------------------------------------------------------------
# Semantic result validation
# ---------------------------------------------------------------------------


def require_semantic_order_id(result: Any, *, action: str) -> str:
    """Validate a ``BrokerSemanticResult`` and return its ``order_id``.

    Raises ``RuntimeError`` when ``result.ok`` is ``False`` or when
    ``result.order_id`` is missing / falsy.

    Use this helper everywhere a placement path reads ``result.order_id``
    directly, so that ``ok=False`` is never treated as success.
    """
    if not result.ok:
        raise RuntimeError(
            f"Broker semantic action failed: {action}: {result.message}"
        )
    order_id: str | None = result.order_id
    if not order_id:
        raise RuntimeError(
            f"Broker semantic action returned no order_id: {action}"
        )
    return order_id


def require_semantic_ok(result: Any, *, action: str) -> None:
    """Validate that a ``BrokerSemanticResult`` is successful.

    Raises ``RuntimeError`` when ``result.ok`` is ``False``.

    Use this for cancel / query paths where the caller doesn't need an
    ``order_id`` — it just needs to know the operation succeeded.
    """
    if not result.ok:
        raise RuntimeError(
            f"Broker semantic action failed: {action}: {result.message}"
        )
