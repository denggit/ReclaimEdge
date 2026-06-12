from __future__ import annotations

from decimal import Decimal
from typing import Any, Mapping

from src.exchanges.errors import ExchangeError, ExchangeErrorDetail, ExchangeErrorKind
from src.exchanges.models import (
    BrokerInstrument,
    BrokerOrder,
    BrokerOrderRequest,
    BrokerOrderSide,
    BrokerOrderStatus,
    BrokerOrderType,
    BrokerPosition,
    BrokerPositionSide,
    ExchangeName,
)


# ---------------------------------------------------------------------------
# Symbol parsing
# ---------------------------------------------------------------------------


def parse_okx_swap_symbol(symbol: str) -> tuple[str, str]:
    """Parse an OKX swap symbol into (base_asset, quote_asset).

    >>> parse_okx_swap_symbol("ETH-USDT-SWAP")
    ("ETH", "USDT")
    >>> parse_okx_swap_symbol("BTC-USDT-SWAP")
    ("BTC", "USDT")

    If the symbol does not match the expected ``<base>-<quote>-SWAP`` pattern,
    returns ``(symbol, "USDT")`` as a conservative default.
    """
    if not isinstance(symbol, str) or not symbol:
        return (symbol, "USDT")

    normalized = symbol.strip().upper()
    parts = normalized.rsplit("-", 2)
    if len(parts) == 3 and parts[2] == "SWAP":
        base, quote = parts[0], parts[1]
        if base and quote and "-" not in base and "-" not in quote:
            return (base, quote)

    return (symbol, "USDT")


# ---------------------------------------------------------------------------
# Instrument mapping
# ---------------------------------------------------------------------------


def broker_instrument_from_trader(trader: object) -> BrokerInstrument:
    """Build a ``BrokerInstrument`` from an existing *trader* instance.

    Reads:
      - ``trader.symbol``
      - ``trader.instrument_metadata.contract_multiplier``
      - ``trader.instrument_metadata.contract_precision``
      - ``trader.instrument_metadata.min_contracts``

    This is a pure function — it does not access the network or OKX API.
    """
    base_asset, quote_asset = parse_okx_swap_symbol(trader.symbol)

    metadata = trader.instrument_metadata
    contract_multiplier = getattr(metadata, "contract_multiplier", Decimal("0.1"))
    contract_precision = getattr(metadata, "contract_precision", Decimal("0.01"))
    min_contracts = getattr(metadata, "min_contracts", Decimal("0.01"))
    price_tick = getattr(trader, "tick_size", None)
    if price_tick is None:
        price_tick = Decimal("0.01")
    elif not isinstance(price_tick, Decimal):
        price_tick = Decimal(str(price_tick))

    return BrokerInstrument(
        exchange=ExchangeName.OKX,
        symbol=trader.symbol,
        base_asset=base_asset,
        quote_asset=quote_asset,
        contract_type="SWAP",
        contract_size=Decimal(str(contract_multiplier)),
        qty_step=Decimal(str(contract_precision)),
        min_qty=Decimal(str(min_contracts)),
        min_notional=Decimal("0"),
        price_tick=price_tick,
        margin_asset="USDT",
    )


# ---------------------------------------------------------------------------
# Position mapping
# ---------------------------------------------------------------------------


def broker_position_from_snapshot(
    *,
    exchange: ExchangeName,
    symbol: str,
    snapshot: object,
    requested_side: BrokerPositionSide | None = None,
) -> BrokerPosition:
    """Build a ``BrokerPosition`` from a legacy ``PositionSnapshot``.

    Mapping rules:

    1. ``snapshot.side == "LONG"`` → ``BrokerPositionSide.LONG``
    2. ``snapshot.side == "SHORT"`` → ``BrokerPositionSide.SHORT``
    3. ``snapshot.side is None`` → *requested_side* or ``BrokerPositionSide.NET``
    4. If *requested_side* is given but differs from the resolved side, a flat
       position with ``side=requested_side`` is returned (for hedge-mode
       per-side queries when the requested side has no position).
    """
    raw_side = getattr(snapshot, "side", None)

    if raw_side == "LONG":
        resolved_side = BrokerPositionSide.LONG
    elif raw_side == "SHORT":
        resolved_side = BrokerPositionSide.SHORT
    else:
        resolved_side = requested_side or BrokerPositionSide.NET

    # Mismatch: caller asked for a specific side but the snapshot is a different side
    if requested_side is not None and raw_side is not None and requested_side != resolved_side:
        return BrokerPosition(
            exchange=exchange,
            symbol=symbol,
            side=requested_side,
            contracts=Decimal("0"),
            base_qty=Decimal("0"),
            avg_entry_price=Decimal("0"),
        )

    # Flat snapshot with a requested side
    if raw_side is None and requested_side is not None:
        return BrokerPosition(
            exchange=exchange,
            symbol=symbol,
            side=requested_side,
            contracts=Decimal("0"),
            base_qty=Decimal("0"),
            avg_entry_price=Decimal("0"),
        )

    contracts = Decimal(str(getattr(snapshot, "contracts", "0")))
    base_qty = Decimal(str(getattr(snapshot, "eth_qty", "0")))
    avg_entry_price = Decimal(str(getattr(snapshot, "avg_entry_price", "0")))
    raw = getattr(snapshot, "raw_pos", Decimal("0"))

    return BrokerPosition(
        exchange=exchange,
        symbol=symbol,
        side=resolved_side,
        contracts=contracts,
        base_qty=base_qty,
        avg_entry_price=avg_entry_price,
        raw={"raw_pos": str(raw)} if raw != Decimal("0") else {},
    )


# ---------------------------------------------------------------------------
# Order mapping
# ---------------------------------------------------------------------------


def _okx_side_to_broker_side(side: str) -> BrokerOrderSide:
    s = (side or "").strip().lower()
    if s == "buy":
        return BrokerOrderSide.BUY
    if s == "sell":
        return BrokerOrderSide.SELL
    return BrokerOrderSide.BUY


def _is_okx_side_recognized(raw_side: Any) -> bool:
    return str(raw_side or "").strip().lower() in {"buy", "sell"}


def broker_order_side_from_okx_side(raw_side: Any) -> BrokerOrderSide:
    """Map an OKX raw side string to ``BrokerOrderSide``.

    >>> broker_order_side_from_okx_side("buy")
    <BrokerOrderSide.BUY: 'BUY'>
    >>> broker_order_side_from_okx_side("sell")
    <BrokerOrderSide.SELL: 'SELL'>

    This legacy helper returns ``BUY`` for unrecognised values because
    ``BrokerOrderSide`` has no unknown member. Pending-order mapping marks such
    orders with ``UNKNOWN`` status so they are not treated as normal BUY orders.
    """
    return _okx_side_to_broker_side(str(raw_side) if raw_side is not None else "")


def _okx_pos_side_to_broker(pos_side: str) -> BrokerPositionSide:
    s = (pos_side or "").strip().lower()
    if s == "long":
        return BrokerPositionSide.LONG
    if s == "short":
        return BrokerPositionSide.SHORT
    return BrokerPositionSide.NET


def broker_position_side_from_okx_pos_side(raw_pos_side: Any) -> BrokerPositionSide:
    """Map an OKX raw posSide to ``BrokerPositionSide``.

    >>> broker_position_side_from_okx_pos_side("long")
    <BrokerPositionSide.LONG: 'LONG'>
    >>> broker_position_side_from_okx_pos_side("short")
    <BrokerPositionSide.SHORT: 'SHORT'>
    >>> broker_position_side_from_okx_pos_side("net")
    <BrokerPositionSide.NET: 'NET'>
    >>> broker_position_side_from_okx_pos_side(None)
    <BrokerPositionSide.NET: 'NET'>

    Falls back to ``NET`` for unrecognised values — never raises.
    """
    return _okx_pos_side_to_broker(str(raw_pos_side) if raw_pos_side is not None else "")


def _okx_state_to_broker_status(state: str) -> BrokerOrderStatus:
    s = (state or "").strip().lower()
    if s == "live":
        return BrokerOrderStatus.NEW
    if s in ("partially_filled", "partially-filled"):
        return BrokerOrderStatus.PARTIALLY_FILLED
    if s == "filled":
        return BrokerOrderStatus.FILLED
    if s in ("canceled", "cancelled"):
        return BrokerOrderStatus.CANCELED
    if s == "rejected":
        return BrokerOrderStatus.REJECTED
    if s == "expired":
        return BrokerOrderStatus.EXPIRED
    return BrokerOrderStatus.UNKNOWN


def broker_order_status_from_okx_state(raw_state: Any) -> BrokerOrderStatus:
    """Map an OKX raw order/algo state to ``BrokerOrderStatus``.

    >>> broker_order_status_from_okx_state("live")
    <BrokerOrderStatus.NEW: 'NEW'>
    >>> broker_order_status_from_okx_state("partially_filled")
    <BrokerOrderStatus.PARTIALLY_FILLED: 'PARTIALLY_FILLED'>
    >>> broker_order_status_from_okx_state("filled")
    <BrokerOrderStatus.FILLED: 'FILLED'>
    >>> broker_order_status_from_okx_state("expired")
    <BrokerOrderStatus.EXPIRED: 'EXPIRED'>

    Falls back to ``UNKNOWN`` for unrecognised values — never raises.
    """
    return _okx_state_to_broker_status(str(raw_state) if raw_state is not None else "")


def _okx_ord_type_to_broker(ord_type: str, *, is_algo: bool = False) -> BrokerOrderType:
    t = (ord_type or "").strip().lower()
    if is_algo:
        # Algo orders: conditional, oco, trigger, move_order_stop → STOP_MARKET
        if t in ("conditional", "oco", "trigger", "move_order_stop"):
            return BrokerOrderType.STOP_MARKET
        # Any other algo type → STOP_MARKET (conservative)
        return BrokerOrderType.STOP_MARKET
    # Ordinary orders
    if t == "market":
        return BrokerOrderType.MARKET
    if t == "limit":
        return BrokerOrderType.LIMIT
    return BrokerOrderType.LIMIT


def _is_okx_ordinary_ord_type_recognized(raw_ord_type: Any) -> bool:
    return str(raw_ord_type or "").strip().lower() in {"market", "limit"}


def broker_order_type_from_okx_ord_type(
    raw_ord_type: Any, *, is_algo: bool = False
) -> BrokerOrderType:
    """Map an OKX raw ordType to ``BrokerOrderType``.

    >>> broker_order_type_from_okx_ord_type("market")
    <BrokerOrderType.MARKET: 'MARKET'>
    >>> broker_order_type_from_okx_ord_type("limit")
    <BrokerOrderType.LIMIT: 'LIMIT'>
    >>> broker_order_type_from_okx_ord_type("conditional", is_algo=True)
    <BrokerOrderType.STOP_MARKET: 'STOP_MARKET'>
    >>> broker_order_type_from_okx_ord_type("oco", is_algo=True)
    <BrokerOrderType.STOP_MARKET: 'STOP_MARKET'>

    For ordinary orders, the legacy helper returns ``LIMIT`` for unrecognised
    values because ``BrokerOrderType`` has no unknown member. Pending-order
    mapping marks such ordinary orders with ``UNKNOWN`` status.
    """
    return _okx_ord_type_to_broker(
        str(raw_ord_type) if raw_ord_type is not None else "", is_algo=is_algo
    )


def _decimal_or_zero(value: Any) -> Decimal:
    try:
        if value in {None, ""}:
            return Decimal("0")
        return Decimal(str(value))
    except Exception:
        return Decimal("0")


def _decimal_or_none(value: Any) -> Decimal | None:
    try:
        if value in {None, ""}:
            return None
        return Decimal(str(value))
    except Exception:
        return None


def _to_decimal_or_none(value: Any) -> Decimal | None:
    """Convert *value* to ``Decimal``, returning ``None`` on failure.

    >>> _to_decimal_or_none("3.14")
    Decimal('3.14')
    >>> _to_decimal_or_none(None) is None
    True
    >>> _to_decimal_or_none("") is None
    True
    """
    if value is None or value == "":
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _to_decimal_or_zero(value: Any) -> Decimal:
    """Convert *value* to ``Decimal``, returning ``Decimal("0")`` on failure."""
    return _to_decimal_or_none(value) or Decimal("0")


# ---------------------------------------------------------------------------
# OKX raw-field helpers
# ---------------------------------------------------------------------------


def okx_reduce_only_flag(item: Mapping[str, Any]) -> bool:
    """Extract the reduce-only flag from an OKX raw order dict.

    Checks ``reduceOnly``, ``reduce_only``, and ``reduce-only`` keys.
    Accepts ``True``, ``"true"``, ``"1"``, ``"yes"``, ``"y"`` (case-insensitive).

    >>> okx_reduce_only_flag({"reduceOnly": "true"})
    True
    >>> okx_reduce_only_flag({"reduceOnly": True})
    True
    >>> okx_reduce_only_flag({"reduceOnly": "false"})
    False
    >>> okx_reduce_only_flag({})
    False
    """
    for key in ("reduceOnly", "reduce_only", "reduce-only"):
        raw = item.get(key)
        if raw is None:
            continue
        if raw is True:
            return True
        if isinstance(raw, str) and raw.strip().lower() in ("true", "1", "yes", "y"):
            return True
        return False
    return False


def okx_trigger_price(item: Mapping[str, Any]) -> Decimal | None:
    """Extract the trigger price from an OKX raw order dict.

    Checks fields in priority order:
    ``slTriggerPx`` → ``tpTriggerPx`` → ``triggerPx`` → ``ordPx`` → ``px``.

    Returns ``Decimal`` or ``None``.
    """
    for key in ("slTriggerPx", "tpTriggerPx", "triggerPx", "ordPx", "px"):
        raw = item.get(key)
        if raw is None or raw == "":
            continue
        try:
            return Decimal(str(raw))
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Order mapping
# ---------------------------------------------------------------------------


def broker_order_from_okx_pending_order(item: Mapping[str, Any], *, symbol: str) -> BrokerOrder:
    """Map an OKX pending order dict to a ``BrokerOrder``.

    This is a pure function — it does not access the network.
    """
    order_id = str(item.get("ordId", "") or "")

    if not order_id:
        # Missing ordId → return a placeholder with UNKNOWN status.
        # Do not raise to avoid crashing open-order recovery loops.
        return BrokerOrder(
            exchange=ExchangeName.OKX,
            symbol=symbol,
            order_id="",
            client_order_id=None,
            side=BrokerOrderSide.BUY,
            position_side=BrokerPositionSide.NET,
            order_type=BrokerOrderType.LIMIT,
            status=BrokerOrderStatus.UNKNOWN,
            price=None,
            quantity=Decimal("0"),
            filled_quantity=Decimal("0"),
            reduce_only=False,
            close_position=False,
            trigger_price=None,
            label=None,
            raw=dict(item),
        )

    raw_side = item.get("side", "")
    raw_ord_type = item.get("ordType", "")
    side_recognized = _is_okx_side_recognized(raw_side)
    ordinary_ord_type_recognized = _is_okx_ordinary_ord_type_recognized(raw_ord_type)

    side = _okx_side_to_broker_side(str(raw_side))
    pos_side = _okx_pos_side_to_broker(str(item.get("posSide", "")))
    order_type = _okx_ord_type_to_broker(str(raw_ord_type), is_algo=False)
    status = _okx_state_to_broker_status(str(item.get("state", "")))
    if not side_recognized or not ordinary_ord_type_recognized:
        status = BrokerOrderStatus.UNKNOWN
    price = _decimal_or_none(item.get("px"))
    quantity = _decimal_or_zero(item.get("sz"))
    filled_qty = _decimal_or_zero(item.get("accFillSz"))
    reduce_only = okx_reduce_only_flag(item)
    trigger_price = okx_trigger_price(item)
    client_order_id = str(item.get("clOrdId", "")) or str(item.get("tag", "")) or None
    label = item.get("label") or item.get("tag") or None

    return BrokerOrder(
        exchange=ExchangeName.OKX,
        symbol=symbol,
        order_id=order_id,
        client_order_id=client_order_id,
        side=side,
        position_side=pos_side,
        order_type=order_type,
        status=status,
        price=price,
        quantity=quantity,
        filled_quantity=filled_qty,
        reduce_only=reduce_only,
        close_position=False,
        trigger_price=trigger_price,
        label=label,
        raw=dict(item),
    )


def broker_order_from_okx_pending_algo_order(
    item: Mapping[str, Any], *, symbol: str
) -> BrokerOrder:
    """Map an OKX pending algo order dict to a ``BrokerOrder``.

    This is a pure function — it does not access the network.
    It does **not** cancel or modify any algo order.
    """
    order_id = str(item.get("algoId", "") or item.get("ordId", "") or "")

    if not order_id:
        # Missing id → return a placeholder with UNKNOWN status.
        return BrokerOrder(
            exchange=ExchangeName.OKX,
            symbol=symbol,
            order_id="",
            client_order_id=None,
            side=BrokerOrderSide.BUY,
            position_side=BrokerPositionSide.NET,
            order_type=BrokerOrderType.STOP_MARKET,
            status=BrokerOrderStatus.UNKNOWN,
            price=None,
            quantity=Decimal("0"),
            filled_quantity=Decimal("0"),
            reduce_only=False,
            close_position=False,
            trigger_price=None,
            label=None,
            raw=dict(item),
        )

    side = _okx_side_to_broker_side(str(item.get("side", "")))
    pos_side = _okx_pos_side_to_broker(str(item.get("posSide", "")))
    order_type = _okx_ord_type_to_broker(str(item.get("ordType", "")), is_algo=True)
    # algo orders may have state in "state" or "algoState"
    raw_state = item.get("state") or item.get("algoState")
    status = _okx_state_to_broker_status(str(raw_state) if raw_state else "")
    price = _decimal_or_none(item.get("px") or item.get("ordPx"))
    quantity = _decimal_or_zero(item.get("sz"))
    filled_qty = _decimal_or_zero(item.get("accFillSz"))
    reduce_only = okx_reduce_only_flag(item)
    trigger_price = okx_trigger_price(item)
    client_order_id = (
        str(item.get("algoClOrdId", ""))
        or str(item.get("clOrdId", ""))
        or str(item.get("tag", ""))
        or None
    )
    label = (
        item.get("label")
        or item.get("tag")
        or item.get("algoClOrdId")
        or None
    )

    return BrokerOrder(
        exchange=ExchangeName.OKX,
        symbol=symbol,
        order_id=order_id,
        client_order_id=client_order_id,
        side=side,
        position_side=pos_side,
        order_type=order_type,
        status=status,
        price=price,
        quantity=quantity,
        filled_quantity=filled_qty,
        reduce_only=reduce_only,
        close_position=False,
        trigger_price=trigger_price,
        label=label,
        raw=dict(item),
    )


# ---------------------------------------------------------------------------
# LiveTradeResult mapping (anti-corruption layer)
# ---------------------------------------------------------------------------


def broker_execution_result_from_live_trade_result(
    *,
    exchange: ExchangeName,
    symbol: str,
    result: object,
) -> "BrokerExecutionResult":
    """Map a legacy ``LiveTradeResult`` to a unified ``BrokerExecutionResult``.

    This is a pure function — it reads attributes from *result* and never
    imports or constructs a ``Trader`` instance.

    Compatible with both the real ``LiveTradeResult`` dataclass and lightweight
    test fakes that expose the same attribute names.
    """
    from src.exchanges.models import BrokerExecutionAction, BrokerExecutionResult

    # --- action ---
    raw_action = str(getattr(result, "action", "") or "")
    action_map: dict[str, BrokerExecutionAction] = {
        "OPEN_LONG": BrokerExecutionAction.OPEN_LONG,
        "OPEN_SHORT": BrokerExecutionAction.OPEN_SHORT,
        "ADD_LONG": BrokerExecutionAction.ADD_LONG,
        "ADD_SHORT": BrokerExecutionAction.ADD_SHORT,
        "UPDATE_TP": BrokerExecutionAction.UPDATE_TP,
        "NEAR_TP_REDUCE": BrokerExecutionAction.NEAR_TP_REDUCE,
        "MARKET_EXIT_RUNNER": BrokerExecutionAction.MARKET_EXIT_RUNNER,
    }
    action = action_map.get(raw_action, BrokerExecutionAction.UNKNOWN)

    # --- numeric fields ---
    contracts = _to_decimal_or_none(getattr(result, "contracts", None))
    tp_price = _to_decimal_or_none(getattr(result, "tp_price", None))
    protective_sl_price = _to_decimal_or_none(
        getattr(result, "protective_sl_price", None)
    )

    # --- tp_order_ids: string "id1,id2" or iterable → tuple ---
    raw_tp_ids = getattr(result, "tp_order_ids", None)
    tp_order_ids: tuple[str, ...] = ()
    if raw_tp_ids:
        if isinstance(raw_tp_ids, str) and raw_tp_ids.strip():
            tp_order_ids = tuple(
                oid.strip() for oid in raw_tp_ids.split(",") if oid.strip()
            )
        elif isinstance(raw_tp_ids, (list, tuple, set)):
            tp_order_ids = tuple(str(oid) for oid in raw_tp_ids if str(oid).strip())
        elif raw_tp_ids:
            # Single item fallback
            tp_order_ids = (str(raw_tp_ids),)

    # --- raw: lightweight metadata only ---
    raw: dict[str, Any] = {
        "legacy_type": type(result).__name__,
        "legacy_action": raw_action or None,
        "legacy_message": getattr(result, "message", None),
    }

    return BrokerExecutionResult(
        exchange=exchange,
        symbol=symbol,
        action=action,
        ok=bool(getattr(result, "ok", False)),
        message=str(getattr(result, "message", "") or ""),
        order_id=getattr(result, "order_id", None) or None,
        tp_order_id=getattr(result, "tp_order_id", None) or None,
        tp_order_ids=tp_order_ids,
        protective_sl_order_id=getattr(result, "protective_sl_order_id", None) or None,
        contracts=contracts,
        tp_price=tp_price,
        protective_sl_price=protective_sl_price,
        entry_filled=bool(getattr(result, "entry_filled", False)),
        tp_ok=(
            bool(getattr(result, "tp_ok", None))
            if getattr(result, "tp_ok", None) is not None
            else None
        ),
        protective_sl_ok=(
            bool(getattr(result, "protective_sl_ok", None))
            if getattr(result, "protective_sl_ok", None) is not None
            else None
        ),
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Unsupported operation helper
# ---------------------------------------------------------------------------


def unsupported_okx_order_request_error(request: BrokerOrderRequest, reason: str) -> ExchangeError:
    """Return an ``ExchangeError`` with kind ``UNSUPPORTED_OPERATION`` for the given request."""
    return ExchangeError(
        ExchangeErrorDetail(
            exchange=ExchangeName.OKX,
            kind=ExchangeErrorKind.UNSUPPORTED_OPERATION,
            message=f"Unsupported order request: {reason}",
            raw={
                "symbol": request.symbol,
                "order_type": request.order_type.value,
                "reduce_only": request.reduce_only,
                "position_side": request.position_side.value,
                "reason": reason,
            },
        )
    )
