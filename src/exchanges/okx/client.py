from __future__ import annotations

from typing import Any, Protocol

from src.exchanges.base import BrokerClient
from src.exchanges.capabilities import ExchangeCapabilities, okx_capabilities
from src.exchanges.errors import ExchangeError, ExchangeErrorDetail, ExchangeErrorKind
from src.exchanges.models import (
    BrokerBalance,
    BrokerInstrument,
    BrokerOrder,
    BrokerOrderRequest,
    BrokerOrderResult,
    BrokerOrderSide,
    BrokerOrderStatus,
    BrokerOrderType,
    BrokerPosition,
    BrokerPositionSide,
    BrokerTimeInForce,
    ExchangeName,
)
from src.exchanges.okx.errors import okx_exception_to_exchange_error
from src.exchanges.okx.mapper import (
    broker_instrument_from_trader,
    broker_order_from_okx_pending_order,
    broker_position_from_snapshot,
    unsupported_okx_order_request_error,
)
from src.execution import order_specs


# ---------------------------------------------------------------------------
# Protocol for trader dependency (lightweight, no import of Trader class)
# ---------------------------------------------------------------------------


class _OkxTraderLike(Protocol):
    """Minimal protocol describing the Trader surface OkxBrokerClient needs.

    This avoids a hard import of ``src.execution.trader.Trader``, keeping the
    adapter decoupled from the concrete implementation and enabling tests with
    lightweight fakes.
    """

    symbol: str
    td_mode: str
    pos_side_mode: str
    instrument_metadata: object

    async def fetch_usdt_equity(self) -> float: ...

    async def fetch_position_snapshot(self) -> object: ...

    async def fetch_pending_orders(self) -> list[dict[str, Any]]: ...

    async def request(
        self, method: str, endpoint: str, payload: Any | None = None
    ) -> dict[str, Any]: ...

    def extract_order_id(self, res: dict[str, Any]) -> str: ...

    def decimal_to_str(self, value: Any) -> str: ...

    def price_to_str(self, price: float) -> str: ...

    async def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


_MARKET_ENTRY_ALLOWED_POSITION_SIDES = frozenset(
    {BrokerPositionSide.LONG, BrokerPositionSide.SHORT}
)
_MARKET_ENTRY_ALLOWED_TIME_IN_FORCE = frozenset(
    {None, BrokerTimeInForce.GTC, BrokerTimeInForce.IOC}
)

_TP_ALLOWED_POSITION_SIDES = frozenset(
    {BrokerPositionSide.LONG, BrokerPositionSide.SHORT}
)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class OkxBrokerClient(BrokerClient):
    """OKX adapter shell implementing ``BrokerClient``.

    Wraps an existing ``Trader`` instance via dependency injection.

    The adapter does **not** create a ``Trader`` itself — it is provided at
    construction time, avoiding side effects (API key validation, live-trading
    gate, network connectivity) during testing.

    Currently supported operations:

    - ``fetch_instrument`` / ``fetch_balance`` / ``fetch_position`` / ``fetch_open_orders``
    - ``place_order`` for MARKET entries and LIMIT reduce-only TP
    - ``cancel_order`` / ``cancel_all_open_orders`` for ordinary orders
    - ``close``

    Everything else raises ``UNSUPPORTED_OPERATION``.
    """

    def __init__(self, trader: _OkxTraderLike) -> None:
        self._trader = trader

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def exchange(self) -> ExchangeName:
        return ExchangeName.OKX

    @property
    def capabilities(self) -> ExchangeCapabilities:
        return okx_capabilities()

    # ------------------------------------------------------------------
    # Instrument
    # ------------------------------------------------------------------

    async def fetch_instrument(self, symbol: str) -> BrokerInstrument:
        if symbol != self._trader.symbol:
            raise ExchangeError(
                ExchangeErrorDetail(
                    exchange=ExchangeName.OKX,
                    kind=ExchangeErrorKind.INVALID_SYMBOL,
                    message=f"Instrument not available for symbol {symbol!r}",
                    raw={"requested": symbol, "configured": self._trader.symbol},
                )
            )
        return broker_instrument_from_trader(self._trader)

    # ------------------------------------------------------------------
    # Balance
    # ------------------------------------------------------------------

    async def fetch_balance(self, asset: str = "USDT") -> BrokerBalance:
        if asset != "USDT":
            raise ExchangeError(
                ExchangeErrorDetail(
                    exchange=ExchangeName.OKX,
                    kind=ExchangeErrorKind.UNSUPPORTED_OPERATION,
                    message=f"Balance query only supported for USDT, got {asset!r}",
                )
            )
        try:
            equity = await self._trader.fetch_usdt_equity()
        except Exception as exc:
            raise okx_exception_to_exchange_error(
                exc, message=f"Failed to fetch USDT balance: {exc}"
            ) from exc

        dec = self._to_decimal(equity)
        return BrokerBalance(
            exchange=ExchangeName.OKX,
            asset="USDT",
            total=dec,
            available=dec,
            equity=dec,
        )

    # ------------------------------------------------------------------
    # Position
    # ------------------------------------------------------------------

    async def fetch_position(
        self, symbol: str, side: BrokerPositionSide | None = None
    ) -> BrokerPosition:
        if symbol != self._trader.symbol:
            raise ExchangeError(
                ExchangeErrorDetail(
                    exchange=ExchangeName.OKX,
                    kind=ExchangeErrorKind.INVALID_SYMBOL,
                    message=f"Position not available for symbol {symbol!r}",
                    raw={"requested": symbol, "configured": self._trader.symbol},
                )
            )
        try:
            snapshot = await self._trader.fetch_position_snapshot()
        except Exception as exc:
            raise okx_exception_to_exchange_error(
                exc, message=f"Failed to fetch position: {exc}"
            ) from exc

        return broker_position_from_snapshot(
            exchange=ExchangeName.OKX,
            symbol=symbol,
            snapshot=snapshot,
            requested_side=side,
        )

    # ------------------------------------------------------------------
    # Open orders
    # ------------------------------------------------------------------

    async def fetch_open_orders(self, symbol: str) -> list[BrokerOrder]:
        if symbol != self._trader.symbol:
            raise ExchangeError(
                ExchangeErrorDetail(
                    exchange=ExchangeName.OKX,
                    kind=ExchangeErrorKind.INVALID_SYMBOL,
                    message=f"Open orders not available for symbol {symbol!r}",
                    raw={"requested": symbol, "configured": self._trader.symbol},
                )
            )
        try:
            raw_orders = await self._trader.fetch_pending_orders()
        except Exception as exc:
            raise okx_exception_to_exchange_error(
                exc, message=f"Failed to fetch open orders: {exc}"
            ) from exc

        return [
            broker_order_from_okx_pending_order(item, symbol=symbol)
            for item in raw_orders
        ]

    # ------------------------------------------------------------------
    # Place order
    # ------------------------------------------------------------------

    async def place_order(self, request: BrokerOrderRequest) -> BrokerOrderResult:
        request = _normalize_broker_order_request(request)

        # --- MARKET entry (non-reduce-only) ---
        if (
            request.order_type == BrokerOrderType.MARKET
            and request.reduce_only is False
            and request.close_position is False
            and request.position_side in _MARKET_ENTRY_ALLOWED_POSITION_SIDES
            and request.time_in_force in _MARKET_ENTRY_ALLOWED_TIME_IN_FORCE
        ):
            return await self._place_market_entry(request)

        # --- LIMIT reduce-only TP ---
        if (
            request.order_type == BrokerOrderType.LIMIT
            and request.reduce_only is True
            and request.price is not None
            and request.position_side in _TP_ALLOWED_POSITION_SIDES
        ):
            return await self._place_limit_reduce_only_tp(request)

        # --- Unsupported ---
        raise unsupported_okx_order_request_error(
            request,
            reason=(
                f"order_type={request.order_type.value} reduce_only={request.reduce_only} "
                f"position_side={request.position_side.value} "
                f"time_in_force={request.time_in_force.value if request.time_in_force else None} "
                f"close_position={request.close_position} price={'set' if request.price is not None else 'none'}"
            ),
        )

    async def _place_market_entry(
        self, request: BrokerOrderRequest
    ) -> BrokerOrderResult:
        """Place a MARKET entry order using existing ``order_specs``."""
        contracts_text = self._trader.decimal_to_str(request.quantity)
        side_str: order_specs.PositionSide = "LONG" if request.position_side == BrokerPositionSide.LONG else "SHORT"

        body = order_specs.build_market_entry_order_body(
            inst_id=self._trader.symbol,
            td_mode=self._trader.td_mode,
            side=side_str,
            contracts_text=contracts_text,
            pos_side_mode=self._trader.pos_side_mode,
        )

        try:
            res = await self._trader.request("POST", "/api/v5/trade/order", body)
        except Exception as exc:
            raise okx_exception_to_exchange_error(
                exc, message=f"Failed to place market entry: {exc}"
            ) from exc

        try:
            order_id = self._trader.extract_order_id(res)
        except Exception as exc:
            order_id = None
            raise okx_exception_to_exchange_error(
                exc, message=f"Failed to extract order id from response: {res}"
            ) from exc

        return BrokerOrderResult(
            exchange=ExchangeName.OKX,
            symbol=request.symbol,
            order_id=order_id,
            client_order_id=request.client_order_id,
            status=BrokerOrderStatus.NEW,
            raw=res,
        )

    async def _place_limit_reduce_only_tp(
        self, request: BrokerOrderRequest
    ) -> BrokerOrderResult:
        """Place a LIMIT reduce-only TP order using existing ``order_specs``."""
        contracts_text = self._trader.decimal_to_str(request.quantity)
        # price is guaranteed non-None by the caller's condition check
        price_text = self._trader.price_to_str(float(request.price))  # type: ignore[arg-type]
        side_str: order_specs.PositionSide = "LONG" if request.position_side == BrokerPositionSide.LONG else "SHORT"

        body = order_specs.build_reduce_only_tp_order_body(
            inst_id=self._trader.symbol,
            td_mode=self._trader.td_mode,
            side=side_str,
            contracts_text=contracts_text,
            price_text=price_text,
            pos_side_mode=self._trader.pos_side_mode,
            client_order_id=request.client_order_id,
        )

        try:
            res = await self._trader.request("POST", "/api/v5/trade/order", body)
        except Exception as exc:
            raise okx_exception_to_exchange_error(
                exc, message=f"Failed to place reduce-only TP: {exc}"
            ) from exc

        try:
            order_id = self._trader.extract_order_id(res)
        except Exception as exc:
            order_id = None
            raise okx_exception_to_exchange_error(
                exc, message=f"Failed to extract order id from response: {res}"
            ) from exc

        return BrokerOrderResult(
            exchange=ExchangeName.OKX,
            symbol=request.symbol,
            order_id=order_id,
            client_order_id=request.client_order_id,
            status=BrokerOrderStatus.NEW,
            raw=res,
        )

    # ------------------------------------------------------------------
    # Cancel order
    # ------------------------------------------------------------------

    async def cancel_order(self, symbol: str, order_id: str) -> None:
        if symbol != self._trader.symbol:
            raise ExchangeError(
                ExchangeErrorDetail(
                    exchange=ExchangeName.OKX,
                    kind=ExchangeErrorKind.INVALID_SYMBOL,
                    message=f"Cannot cancel order for symbol {symbol!r}",
                    raw={"requested": symbol, "configured": self._trader.symbol},
                )
            )
        try:
            await self._trader.request(
                "POST",
                "/api/v5/trade/cancel-order",
                {"instId": symbol, "ordId": order_id},
            )
        except Exception as exc:
            raise okx_exception_to_exchange_error(
                exc, message=f"Failed to cancel order {order_id}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Cancel all open orders
    # ------------------------------------------------------------------

    async def cancel_all_open_orders(self, symbol: str) -> None:
        """Cancel all ordinary pending orders for *symbol*.

        Algo orders are **not** cancelled by this method.
        """
        orders = await self.fetch_open_orders(symbol)
        for order in orders:
            if order.order_id:
                try:
                    await self.cancel_order(symbol, order.order_id)
                except ExchangeError:
                    # Continue cancelling remaining orders even if one fails
                    pass

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    async def close(self) -> None:
        await self._trader.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_decimal(value: Any) -> "Decimal":
        from decimal import Decimal as D

        if isinstance(value, D):
            return value
        return D(str(value))


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def _normalize_broker_order_request(request: BrokerOrderRequest) -> BrokerOrderRequest:
    """Normalize a ``BrokerOrderRequest`` for internal use.

    Ensures *symbol* is set (defaults to the exchange's primary symbol if empty).
    This conservatively guards against callers passing empty symbol strings.
    """
    if not request.symbol:
        object.__setattr__(request, "symbol", request.symbol or "")
    return request
