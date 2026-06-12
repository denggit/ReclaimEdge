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


def _okx_pos_side_to_broker(pos_side: str) -> BrokerPositionSide:
    s = (pos_side or "").strip().lower()
    if s == "long":
        return BrokerPositionSide.LONG
    if s == "short":
        return BrokerPositionSide.SHORT
    return BrokerPositionSide.NET


def _okx_state_to_broker_status(state: str) -> BrokerOrderStatus:
    s = (state or "").strip().lower()
    if s == "live":
        return BrokerOrderStatus.NEW
    if s == "partially_filled":
        return BrokerOrderStatus.PARTIALLY_FILLED
    if s == "filled":
        return BrokerOrderStatus.FILLED
    if s in ("canceled", "cancelled"):
        return BrokerOrderStatus.CANCELED
    if s == "rejected":
        return BrokerOrderStatus.REJECTED
    return BrokerOrderStatus.UNKNOWN


def _okx_ord_type_to_broker(ord_type: str) -> BrokerOrderType:
    t = (ord_type or "").strip().lower()
    if t == "market":
        return BrokerOrderType.MARKET
    if t == "limit":
        return BrokerOrderType.LIMIT
    if t == "conditional":
        return BrokerOrderType.STOP_MARKET
    # Conservative fallback
    return BrokerOrderType.LIMIT


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
            raw=dict(item),
        )

    side = _okx_side_to_broker_side(str(item.get("side", "")))
    pos_side = _okx_pos_side_to_broker(str(item.get("posSide", "")))
    order_type = _okx_ord_type_to_broker(str(item.get("ordType", "")))
    status = _okx_state_to_broker_status(str(item.get("state", "")))
    price = _decimal_or_none(item.get("px"))
    quantity = _decimal_or_zero(item.get("sz"))
    filled_qty = _decimal_or_zero(item.get("accFillSz"))
    reduce_only = True if str(item.get("reduceOnly", "")).lower() == "true" else False
    client_order_id = str(item.get("clOrdId", "")) or None

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
        raw=dict(item),
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
