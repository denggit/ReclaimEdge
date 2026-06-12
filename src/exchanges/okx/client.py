from __future__ import annotations

from typing import Any, Protocol

from src.exchanges.base import BrokerClient
from src.exchanges.capabilities import ExchangeCapabilities, okx_capabilities
from src.exchanges.errors import ExchangeError, ExchangeErrorDetail, ExchangeErrorKind
from src.exchanges.models import (
    BrokerBalance,
    BrokerExecutionResult,
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
from src.exchanges.okx.errors import (
    okx_exception_to_exchange_error,
    raise_okx_exchange_error_from_response,
)
from src.exchanges.okx.mapper import (
    broker_execution_result_from_live_trade_result,
    broker_instrument_from_trader,
    broker_order_from_okx_pending_algo_order,
    broker_order_from_okx_pending_order,
    broker_position_from_snapshot,
    unsupported_okx_order_request_error,
)
from src.execution import order_specs
from src.utils.log import get_logger


logger = get_logger(__name__)


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

    async def fetch_pending_algo_orders(self) -> list[dict[str, Any]]: ...

    async def request(
        self, method: str, endpoint: str, payload: Any | None = None
    ) -> dict[str, Any]: ...

    def extract_order_id(self, res: dict[str, Any]) -> str: ...

    def extract_algo_id(self, res: dict[str, Any]) -> str: ...

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
            raw_fetch = getattr(self._trader, "fetch_pending_orders_raw", None)
            if callable(raw_fetch):
                raw_orders = await raw_fetch()
            else:
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
        _validate_okx_order_request(request, configured_symbol=self._trader.symbol)

        # --- MARKET entry (non-reduce-only) ---
        if (
            request.order_type == BrokerOrderType.MARKET
            and request.reduce_only is False
            and request.close_position is False
            and request.position_side in _MARKET_ENTRY_ALLOWED_POSITION_SIDES
            and request.time_in_force in _MARKET_ENTRY_ALLOWED_TIME_IN_FORCE
        ):
            return await self._place_market_entry(request)

        # --- MARKET reduce-only close / exit ---
        if (
            request.order_type == BrokerOrderType.MARKET
            and request.reduce_only is True
            and request.position_side in _TP_ALLOWED_POSITION_SIDES
            and request.time_in_force in _MARKET_ENTRY_ALLOWED_TIME_IN_FORCE
        ):
            return await self._place_reduce_only_market_close(request)

        # --- LIMIT reduce-only TP ---
        if (
            request.order_type == BrokerOrderType.LIMIT
            and request.reduce_only is True
            and request.price is not None
            and request.position_side in _TP_ALLOWED_POSITION_SIDES
        ):
            return await self._place_limit_reduce_only_tp(request)

        # --- STOP_MARKET protective SL ---
        if (
            request.order_type == BrokerOrderType.STOP_MARKET
            and (request.reduce_only is True or request.close_position is True)
            and request.trigger_price is not None
            and request.position_side in _TP_ALLOWED_POSITION_SIDES
        ):
            return await self._place_protective_stop_order(request)

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
            raise_okx_exchange_error_from_response(
                res,
                message="Failed to place market entry",
            )
        except ExchangeError:
            raise
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

    async def _place_reduce_only_market_close(
        self, request: BrokerOrderRequest
    ) -> BrokerOrderResult:
        """Place a MARKET reduce-only close order using existing ``order_specs``."""
        contracts_text = self._trader.decimal_to_str(request.quantity)
        side_str: order_specs.PositionSide = "LONG" if request.position_side == BrokerPositionSide.LONG else "SHORT"

        body = order_specs.build_reduce_only_market_order_body(
            inst_id=self._trader.symbol,
            td_mode=self._trader.td_mode,
            side=side_str,
            contracts_text=contracts_text,
            pos_side_mode=self._trader.pos_side_mode,
        )

        try:
            res = await self._trader.request("POST", "/api/v5/trade/order", body)
            raise_okx_exchange_error_from_response(
                res,
                message="Failed to place reduce-only market close",
            )
        except ExchangeError:
            raise
        except Exception as exc:
            raise okx_exception_to_exchange_error(
                exc, message=f"Failed to place reduce-only market close: {exc}"
            ) from exc

        try:
            order_id = self._trader.extract_order_id(res)
        except Exception as exc:
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
            raise_okx_exchange_error_from_response(
                res,
                message="Failed to place reduce-only TP",
            )
        except ExchangeError:
            raise
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

    async def _place_protective_stop_order(
        self, request: BrokerOrderRequest
    ) -> BrokerOrderResult:
        """Place a conditional protective SL algo order using existing ``order_specs``."""
        contracts_text = self._trader.decimal_to_str(request.quantity)
        # trigger_price is guaranteed non-None by the caller's condition check
        stop_price_text = self._trader.price_to_str(float(request.trigger_price))  # type: ignore[arg-type]
        side_str: order_specs.PositionSide = "LONG" if request.position_side == BrokerPositionSide.LONG else "SHORT"

        body = order_specs.build_conditional_protective_sl_algo_body(
            inst_id=self._trader.symbol,
            td_mode=self._trader.td_mode,
            side=side_str,
            contracts_text=contracts_text,
            stop_price_text=stop_price_text,
            pos_side_mode=self._trader.pos_side_mode,
        )

        try:
            res = await self._trader.request("POST", "/api/v5/trade/order-algo", body)
            raise_okx_exchange_error_from_response(
                res,
                message="Failed to place protective stop",
            )
        except ExchangeError:
            raise
        except Exception as exc:
            raise okx_exception_to_exchange_error(
                exc, message=f"Failed to place protective stop: {exc}"
            ) from exc

        try:
            order_id = _extract_algo_or_order_id(self._trader, res)
        except Exception as exc:
            raise okx_exception_to_exchange_error(
                exc, message=f"Failed to extract algo id from response: {res}"
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
            res = await self._trader.request(
                "POST",
                "/api/v5/trade/cancel-order",
                {"instId": symbol, "ordId": order_id},
            )
            raise_okx_exchange_error_from_response(
                res,
                message=f"Failed to cancel order {order_id}",
            )
        except ExchangeError:
            raise
        except Exception as exc:
            raise okx_exception_to_exchange_error(
                exc, message=f"Failed to cancel order {order_id}: {exc}"
            ) from exc

    async def cancel_algo_order(self, symbol: str, algo_id: str) -> None:
        if symbol != self._trader.symbol:
            raise ExchangeError(
                ExchangeErrorDetail(
                    exchange=ExchangeName.OKX,
                    kind=ExchangeErrorKind.INVALID_SYMBOL,
                    message=f"Cannot cancel algo order for symbol {symbol!r}",
                    raw={"requested": symbol, "configured": self._trader.symbol},
                )
            )
        try:
            res = await self._trader.request(
                "POST",
                "/api/v5/trade/cancel-algos",
                order_specs.build_cancel_algo_body(
                    inst_id=symbol,
                    algo_id=algo_id,
                ),
            )
            raise_okx_exchange_error_from_response(
                res,
                message=f"Failed to cancel algo order {algo_id}",
            )
        except ExchangeError:
            raise
        except Exception as exc:
            raise okx_exception_to_exchange_error(
                exc, message=f"Failed to cancel algo order {algo_id}: {exc}"
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
                except ExchangeError as exc:
                    # Continue cancelling remaining orders even if one fails
                    logger.warning(
                        "Failed to cancel open OKX order symbol=%s order_id=%s kind=%s message=%s",
                        symbol,
                        order.order_id,
                        exc.detail.kind.value,
                        exc.detail.message,
                    )

    # ------------------------------------------------------------------
    # Algo orders (adapter-specific, not part of BrokerClient interface)
    # ------------------------------------------------------------------

    async def fetch_algo_orders(self, symbol: str) -> list[BrokerOrder]:
        """Fetch pending algo orders for *symbol*.

        This is **not** part of the ``BrokerClient`` interface — it is an
        OKX-adapter-specific method for testing and future migration.
        It does **not** cancel or modify any algo order.
        """
        if symbol != self._trader.symbol:
            raise ExchangeError(
                ExchangeErrorDetail(
                    exchange=ExchangeName.OKX,
                    kind=ExchangeErrorKind.INVALID_SYMBOL,
                    message=f"Algo orders not available for symbol {symbol!r}",
                    raw={"requested": symbol, "configured": self._trader.symbol},
                )
            )
        try:
            raw_fetch = getattr(self._trader, "fetch_pending_algo_orders_raw", None)
            if callable(raw_fetch):
                raw_orders = await raw_fetch()
            else:
                raw_orders = await self._trader.fetch_pending_algo_orders()
        except Exception as exc:
            raise okx_exception_to_exchange_error(
                exc, message=f"Failed to fetch algo orders: {exc}"
            ) from exc

        return [
            broker_order_from_okx_pending_algo_order(item, symbol=symbol)
            for item in raw_orders
        ]

    # ------------------------------------------------------------------
    # LiveTradeResult mapping facade
    # ------------------------------------------------------------------

    def map_live_trade_result(self, result: object) -> BrokerExecutionResult:
        """Map a legacy ``LiveTradeResult`` to a unified ``BrokerExecutionResult``.

        This is a lightweight adapter method — it does **not** call
        ``Trader.execute_intent`` and does **not** change any live behaviour.
        """
        return broker_execution_result_from_live_trade_result(
            exchange=ExchangeName.OKX,
            symbol=self._trader.symbol,
            result=result,
        )

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

    Validation is responsible for symbol checks; frozen request objects are not
    mutated here.
    """
    return request


def _validate_okx_order_request(
    request: BrokerOrderRequest,
    *,
    configured_symbol: str,
) -> None:
    """Validate broker-level order semantics supported by the OKX adapter."""
    if request.exchange != ExchangeName.OKX:
        raise ExchangeError(
            ExchangeErrorDetail(
                exchange=ExchangeName.OKX,
                kind=ExchangeErrorKind.UNSUPPORTED_OPERATION,
                message=(
                    f"OKX adapter cannot place orders for exchange "
                    f"{request.exchange.value!r}"
                ),
                raw={
                    "requested_exchange": request.exchange.value,
                    "configured_exchange": ExchangeName.OKX.value,
                },
            )
        )

    if not request.symbol:
        raise ExchangeError(
            ExchangeErrorDetail(
                exchange=ExchangeName.OKX,
                kind=ExchangeErrorKind.INVALID_SYMBOL,
                message="Order symbol is required",
                raw={"requested": request.symbol, "configured": configured_symbol},
            )
        )

    if request.symbol != configured_symbol:
        raise ExchangeError(
            ExchangeErrorDetail(
                exchange=ExchangeName.OKX,
                kind=ExchangeErrorKind.INVALID_SYMBOL,
                message=(
                    f"Order symbol {request.symbol!r} does not match configured "
                    f"symbol {configured_symbol!r}"
                ),
                raw={"requested": request.symbol, "configured": configured_symbol},
            )
        )

    if request.order_type == BrokerOrderType.MARKET:
        if request.reduce_only is True or request.close_position is True:
            expected_side_by_position = {
                BrokerPositionSide.LONG: BrokerOrderSide.SELL,
                BrokerPositionSide.SHORT: BrokerOrderSide.BUY,
            }
            expected_side = expected_side_by_position.get(request.position_side)
            if expected_side is None:
                _raise_unsupported_order_validation_error(
                    request,
                    "MARKET reduce-only close orders require LONG or SHORT position side",
                )
            if request.side != expected_side:
                _raise_unsupported_order_validation_error(
                    request,
                    (
                        f"MARKET reduce-only close for {request.position_side.value} "
                        f"requires {expected_side.value} side"
                    ),
                )
            return

        expected_side_by_position = {
            BrokerPositionSide.LONG: BrokerOrderSide.BUY,
            BrokerPositionSide.SHORT: BrokerOrderSide.SELL,
        }
        expected_side = expected_side_by_position.get(request.position_side)
        if expected_side is None:
            _raise_unsupported_order_validation_error(
                request,
                "MARKET entry orders require LONG or SHORT position side",
            )
        if request.side != expected_side:
            _raise_unsupported_order_validation_error(
                request,
                (
                    f"MARKET {request.position_side.value} entry requires "
                    f"{expected_side.value} side"
                ),
            )
        return

    if request.order_type == BrokerOrderType.LIMIT and request.reduce_only is True:
        if request.price is None:
            _raise_unsupported_order_validation_error(
                request,
                "LIMIT reduce-only TP orders require price",
            )

        expected_side_by_position = {
            BrokerPositionSide.LONG: BrokerOrderSide.SELL,
            BrokerPositionSide.SHORT: BrokerOrderSide.BUY,
        }
        expected_side = expected_side_by_position.get(request.position_side)
        if expected_side is None:
            _raise_unsupported_order_validation_error(
                request,
                "LIMIT reduce-only TP orders require LONG or SHORT position side",
            )
        if request.side != expected_side:
            _raise_unsupported_order_validation_error(
                request,
                (
                    f"LIMIT reduce-only TP for {request.position_side.value} "
                    f"requires {expected_side.value} side"
                ),
            )
        return

    if request.order_type == BrokerOrderType.STOP_MARKET:
        if request.trigger_price is None:
            _raise_unsupported_order_validation_error(
                request,
                "STOP_MARKET protective SL orders require trigger_price",
            )
        if not (request.reduce_only is True or request.close_position is True):
            _raise_unsupported_order_validation_error(
                request,
                "STOP_MARKET protective SL orders must be reduce-only or close-position",
            )

        expected_side_by_position = {
            BrokerPositionSide.LONG: BrokerOrderSide.SELL,
            BrokerPositionSide.SHORT: BrokerOrderSide.BUY,
        }
        expected_side = expected_side_by_position.get(request.position_side)
        if expected_side is None:
            _raise_unsupported_order_validation_error(
                request,
                "STOP_MARKET protective SL orders require LONG or SHORT position side",
            )
        if request.side != expected_side:
            _raise_unsupported_order_validation_error(
                request,
                (
                    f"STOP_MARKET protective SL for {request.position_side.value} "
                    f"requires {expected_side.value} side"
                ),
            )
        return


def _extract_algo_or_order_id(trader: object, res: dict[str, Any]) -> str:
    extract_algo_id = getattr(trader, "extract_algo_id", None)
    if callable(extract_algo_id):
        return str(extract_algo_id(res))
    data = res.get("data", [])
    if data:
        item = data[0]
        order_id = item.get("algoId") or item.get("ordId")
        if order_id:
            return str(order_id)
    raise RuntimeError(f"Missing algoId/ordId in response: {res}")


def _raise_unsupported_order_validation_error(
    request: BrokerOrderRequest,
    reason: str,
) -> None:
    raise ExchangeError(
        ExchangeErrorDetail(
            exchange=ExchangeName.OKX,
            kind=ExchangeErrorKind.UNSUPPORTED_OPERATION,
            message=reason,
            raw={
                "symbol": request.symbol,
                "order_type": request.order_type.value,
                "side": request.side.value,
                "position_side": request.position_side.value,
                "reduce_only": request.reduce_only,
                "close_position": request.close_position,
                "price": str(request.price) if request.price is not None else None,
                "reason": reason,
            },
        )
    )
