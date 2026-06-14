#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : algo_orders.py
@Description: Binance USD-M Futures Algo Order client.

Provides a thin wrapper around the Binance Algo Order API
(``POST /fapi/v1/algoOrder``, ``DELETE /fapi/v1/algoOrder``,
``GET /fapi/v1/openAlgoOrders``) for placing and managing
CONDITIONAL stop-loss orders.

No live wiring.  No env reads.  No API key reads by itself.
No imports of strategy / execution / factory / runtime modules.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from src.exchanges.binance.signing import (
    BINANCE_USDM_ALGO_ORDER_PATH,
    BINANCE_USDM_BASE_URL,
    BINANCE_USDM_OPEN_ALGO_ORDERS_PATH,
    build_signed_request,
)
from src.exchanges.binance.transport import BinanceHttpTransport, BinanceTransportResponse
from src.exchanges.errors import ExchangeError, ExchangeErrorKind
from src.exchanges.models import (
    BrokerCancelResult,
    BrokerOrderResult,
    ExchangeName,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BINANCE_ETH_USDT_SYMBOL = "ETHUSDT"


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class BinanceAlgoOrderClient:
    """Thin wrapper around the Binance Algo Order API for stop-loss orders.

    Requires ``api_key``, ``api_secret``, and a ``transport``.  Without a
    transport the instance raises on every method — compatible with the
    same injection pattern used by ``BinanceBrokerClient``.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        transport: BinanceHttpTransport | None = None,
        base_url: str = BINANCE_USDM_BASE_URL,
        recv_window: int = 5000,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._transport = transport
        self._base_url = base_url
        self._recv_window = recv_window

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def place_stop_loss(
        self,
        *,
        symbol: str,
        side: str,
        quantity: Decimal,
        trigger_price: Decimal,
        client_algo_id: str,
        working_type: str = "MARK_PRICE",
    ) -> BrokerOrderResult:
        """Place a STOP_MARKET conditional algo order.

        Uses ``POST /fapi/v1/algoOrder`` with ``algoType=CONDITIONAL``,
        ``type=STOP_MARKET``, and ``reduceOnly=true``.

        Parameters
        ----------
        symbol:
            Trading symbol (must be ``ETHUSDT``).
        side:
            Order side — ``BUY`` to close a SHORT, ``SELL`` to close a LONG.
        quantity:
            Base-asset quantity (e.g. ``Decimal("0.1")`` for 0.1 ETH).
        trigger_price:
            The stop / trigger price.
        client_algo_id:
            Client-defined algo order ID.  Must start with ``RE_MAIN_``
            when used by the main strategy.
        working_type:
            Price type for trigger — ``MARK_PRICE`` (default) or ``CONTRACT_PRICE``.

        Returns
        -------
        BrokerOrderResult
            Result with ``order_id`` set to the Binance ``algoId`` and
            ``client_order_id`` set to ``clientAlgoId``.
        """
        self._ensure_transport("place_stop_loss")

        params: dict[str, Any] = {
            "algoType": "CONDITIONAL",
            "symbol": symbol,
            "side": side,
            "type": "STOP_MARKET",
            "quantity": _format_decimal(quantity),
            "triggerPrice": _format_decimal(trigger_price),
            "reduceOnly": "true",
            "workingType": working_type,
            "clientAlgoId": client_algo_id,
        }

        signed = build_signed_request(
            method="POST",
            path=BINANCE_USDM_ALGO_ORDER_PATH,
            params=params,
            api_key=self._api_key or "",
            api_secret=self._api_secret or "",
            base_url=self._base_url,
            recv_window=self._recv_window,
        )

        response = await self._transport.send(signed)  # type: ignore[union-attr]
        self._raise_for_transport_response(response, operation="place_stop_loss")

        if not isinstance(response.payload, dict):
            raise ExchangeError(
                exchange=ExchangeName.BINANCE,
                kind=ExchangeErrorKind.EXCHANGE_REJECTED,
                message="Binance algo order response payload is not a dict",
            )

        algo_id = response.payload.get("algoId")
        cid = response.payload.get("clientAlgoId") or client_algo_id

        return BrokerOrderResult(
            exchange=ExchangeName.BINANCE,
            symbol=symbol,
            ok=True,
            order_id=str(algo_id) if algo_id is not None else cid,
            client_order_id=cid,
            raw=response.payload,
        )

    async def cancel_algo_order(
        self,
        *,
        symbol: str,
        client_algo_id: str,
    ) -> BrokerCancelResult:
        """Cancel an algo order by its ``clientAlgoId``.

        Uses ``DELETE /fapi/v1/algoOrder``.
        """
        self._ensure_transport("cancel_algo_order")

        if not client_algo_id:
            raise ValueError("client_algo_id must not be empty")

        signed = build_signed_request(
            method="DELETE",
            path=BINANCE_USDM_ALGO_ORDER_PATH,
            params={"symbol": symbol, "clientAlgoId": client_algo_id},
            api_key=self._api_key or "",
            api_secret=self._api_secret or "",
            base_url=self._base_url,
            recv_window=self._recv_window,
        )

        response = await self._transport.send(signed)  # type: ignore[union-attr]

        if response.status_code >= 400:
            payload = response.payload if isinstance(response.payload, dict) else {}
            return BrokerCancelResult(
                exchange=ExchangeName.BINANCE,
                symbol=symbol,
                ok=False,
                order_id=None,
                client_order_id=client_algo_id,
                message=f"Algo cancel HTTP {response.status_code}: {payload}",
            )

        return BrokerCancelResult(
            exchange=ExchangeName.BINANCE,
            symbol=symbol,
            ok=True,
            order_id=None,
            client_order_id=client_algo_id,
            raw=response.payload if isinstance(response.payload, dict) else {},
        )

    async def fetch_open_algo_orders(
        self,
        *,
        symbol: str,
    ) -> list[dict[str, Any]]:
        """Fetch open algo orders via ``GET /fapi/v1/openAlgoOrders``.

        Returns the raw list of order dicts (may be empty).
        """
        self._ensure_transport("fetch_open_algo_orders")

        signed = build_signed_request(
            method="GET",
            path=BINANCE_USDM_OPEN_ALGO_ORDERS_PATH,
            params={"symbol": symbol},
            api_key=self._api_key or "",
            api_secret=self._api_secret or "",
            base_url=self._base_url,
            recv_window=self._recv_window,
        )

        response = await self._transport.send(signed)  # type: ignore[union-attr]

        if response.status_code >= 400:
            return []

        if not isinstance(response.payload, list):
            return []

        return response.payload

    # ------------------------------------------------------------------
    # Transport readiness
    # ------------------------------------------------------------------

    def _has_transport(self) -> bool:
        return bool(self._api_key and self._api_secret and self._transport)

    def _ensure_transport(self, operation: str) -> None:
        if not self._has_transport():
            raise ExchangeError(
                exchange=ExchangeName.BINANCE,
                kind=ExchangeErrorKind.UNSUPPORTED_OPERATION,
                message=f"BinanceAlgoOrderClient transport not wired for: {operation}",
                raw={"operation": operation},
            )

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def _raise_for_transport_response(
        self,
        response: BinanceTransportResponse,
        *,
        operation: str,
    ) -> None:
        if response.status_code >= 400:
            payload = (
                response.payload
                if isinstance(response.payload, dict)
                else {"message": str(response.payload)}
            )
            raise ExchangeError(
                exchange=ExchangeName.BINANCE,
                kind=ExchangeErrorKind.EXCHANGE_REJECTED,
                message=f"Algo {operation} HTTP {response.status_code}: {payload}",
            )

        if isinstance(response.payload, dict) and "code" in response.payload:
            code = response.payload.get("code")
            if isinstance(code, int) and code < 0:
                msg = response.payload.get("msg", "Unknown error")
                raise ExchangeError(
                    exchange=ExchangeName.BINANCE,
                    kind=ExchangeErrorKind.EXCHANGE_REJECTED,
                    message=f"Algo {operation} rejected: [{code}] {msg}",
                )


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------


def _format_decimal(value: Decimal) -> str:
    """Format a Decimal for Binance API params, avoiding scientific notation."""
    normalized = value.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


__all__ = ["BinanceAlgoOrderClient", "BINANCE_ETH_USDT_SYMBOL"]
