from __future__ import annotations

from typing import Any

from src.execution.trader import PositionSnapshot, Trader
from src.live import runtime_types as live_runtime_types
from src.reporting.live_state_store import LiveStateStore
from src.reporting.trade_journal import LiveTradeJournal
from src.strategies.boll_cvd_reclaim_strategy import BollCvdReclaimStrategy
from src.utils.log import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Reduce-only order selection helpers (compatible with OKX raw dicts and
# BrokerOrder-like objects – no hard dependency on any exchange model).
# ---------------------------------------------------------------------------


def _order_value(order: Any, *names: str) -> Any:
    """Return the first matching value from a dict or object attribute.

    Prefer to avoid importing BrokerOrder so startup recovery does not
    depend on a specific exchange semantic model.
    """
    if isinstance(order, dict):
        for name in names:
            if name in order:
                return order.get(name)
        return None
    for name in names:
        if hasattr(order, name):
            return getattr(order, name)
    return None


def _order_id(order: Any) -> str:
    value = _order_value(order, "ordId", "order_id", "id")
    return str(value or "")


def _order_symbol(order: Any) -> str:
    value = _order_value(order, "instId", "symbol")
    return str(value or "")


def _order_reduce_only(order: Any) -> bool:
    value = _order_value(order, "reduceOnly", "reduce_only")
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"true", "1", "yes", "y", "on"}


def _select_recoverable_reduce_only_orders(
    pending_orders: list[Any],
    *,
    symbol: str,
    protected_order_ids: set[str],
) -> list[Any]:
    """Select reduce-only orders matching *symbol* and excluding *protected_order_ids*.

    Returns the original order objects unchanged – callers can extract
    order ids via ``_order_id()`` when needed.

    Orders without a usable order id are silently skipped — startup
    recovery can only act on identifiable orders.
    """
    selected: list[Any] = []
    for order in pending_orders:
        order_id = _order_id(order)
        if not order_id:
            continue
        if _order_symbol(order) != symbol:
            continue
        if not _order_reduce_only(order):
            continue
        if order_id in protected_order_ids:
            continue
        selected.append(order)
    return selected


# Sidecar startup recovery has been removed.
# This file now only retains main TP startup recovery.


async def apply_main_tp_startup_recovery(
        *,
        execution_state: live_runtime_types.ExecutionState,
        saved_state: Any,
        startup_position: PositionSnapshot,
        trader: Trader,
        journal: LiveTradeJournal,
) -> None:
    if not startup_position.has_position:
        return
    restored_tp_order_id = getattr(saved_state, "tp_order_id", None) if saved_state is not None else None
    restored_tp_order_ids = list(getattr(saved_state, "tp_order_ids", []) or []) if saved_state is not None else []
    if not restored_tp_order_id and restored_tp_order_ids:
        restored_tp_order_id = ",".join(str(item) for item in restored_tp_order_ids if item)
    if restored_tp_order_id:
        trader.tp_order_id = str(restored_tp_order_id)
        return
    try:
        pending_orders = await trader.fetch_pending_orders()
    except Exception as exc:
        execution_state.trading_halted = True
        execution_state.halt_reason = "main_tp_order_id_missing_on_startup"
        if hasattr(journal, "append"):
            journal.append(
                "MAIN_TP_ORDER_ID_MISSING_ON_STARTUP",
                {"reason": "pending_order_check_failed", "error": str(exc), "manual_intervention_required": True},
                position_id=execution_state.current_position_id,
            )
        logger.error(
            "MAIN_TP_ORDER_ID_MISSING_ON_STARTUP | reason=pending_order_check_failed error=%s trading_halted=true manual_intervention_required=true",
            exc)
        return
    # Sidecar TP order IDs are no longer tracked — all pending reduce-only
    # orders are considered unprotected.
    protected_ids: set[str] = set()
    reduce_only_orders = _select_recoverable_reduce_only_orders(
        pending_orders,
        symbol=trader.symbol,
        protected_order_ids=protected_ids,
    )
    if reduce_only_orders:
        execution_state.trading_halted = True
        execution_state.halt_reason = "main_tp_order_id_missing_on_startup"
        if hasattr(journal, "append"):
            journal.append(
                "MAIN_TP_ORDER_ID_MISSING_ON_STARTUP",
                {
                    "pending_reduce_only_order_count": len(reduce_only_orders),
                    "pending_reduce_only_order_ids": [_order_id(item) for item in reduce_only_orders],
                    "manual_intervention_required": True,
                },
                position_id=execution_state.current_position_id,
            )
        logger.error(
            "MAIN_TP_ORDER_ID_MISSING_ON_STARTUP | pending_reduce_only_order_count=%s trading_halted=true manual_intervention_required=true",
            len(reduce_only_orders),
        )
