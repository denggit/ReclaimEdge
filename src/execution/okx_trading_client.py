#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : okx_trading_client.py
@Description: OKX implementation of TradingClientPort.

This class wraps an existing Trader instance.
It is NOT wired into production yet.
Quantity is currently interpreted as OKX contract quantity, matching the
existing OKX execution code path.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from src.execution import order_specs
from src.execution.trading_client_port import (
    AlgoOrderSnapshot,
    BalanceSnapshot,
    CancelResult,
    OrderResult,
    OrderSnapshot,
    OrderStatusSnapshot,
    PositionSnapshot,
    TradingClientPort,
)

if TYPE_CHECKING:
    from src.execution.trader import Trader


def _normalise_position_side(side: str) -> order_specs.PositionSide:
    """Convert a strategy-level side string into an order_specs PositionSide.

    Only ``LONG`` and ``SHORT`` are accepted.
    """
    value = side.strip().upper()
    if value not in {"LONG", "SHORT"}:
        raise ValueError(f"Unsupported position side: {side!r}")
    return value  # type: ignore[return-value]


def _normalise_client_order_id(client_order_id: str | None) -> str | None:
    """Return *client_order_id* stripped, or ``None`` when it is empty.

    An empty or whitespace-only client-order id is treated as "not
    provided" so the corresponding body key (``clOrdId`` /
    ``algoClOrdId``) is omitted from the request entirely.
    """
    if client_order_id is None:
        return None
    value = client_order_id.strip()
    return value or None


class OkxTradingClient(TradingClientPort):
    """OKX implementation of TradingClientPort.

    This class wraps an existing Trader instance.
    It is not wired into production yet.
    Quantity is currently interpreted as OKX contract quantity, matching the
    existing OKX execution code path.
    """

    def __init__(self, trader: Trader) -> None:
        self._trader = trader

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    async def fetch_balance(self) -> BalanceSnapshot:
        """Fetch USDT equity from the wrapped Trader."""
        equity = await self._trader.fetch_usdt_equity()
        return BalanceSnapshot(
            asset="USDT",
            total=Decimal(str(equity)),
            available=None,
            raw={"account_equity_usdt": equity},
        )

    async def fetch_position(self) -> PositionSnapshot:
        """Fetch current position snapshot from the wrapped Trader.

        Maps the Trader's internal ``PositionSnapshot`` to the port DTO.
        """
        snapshot = await self._trader.fetch_position_snapshot()
        return PositionSnapshot(
            side=snapshot.side,
            qty=snapshot.contracts,
            avg_entry_price=Decimal(str(snapshot.avg_entry_price)) if snapshot.avg_entry_price else None,
            raw={
                "contracts": str(snapshot.contracts),
                "eth_qty": snapshot.eth_qty,
                "raw_pos": str(snapshot.raw_pos),
            },
        )

    async def fetch_open_orders(self) -> list[OrderSnapshot]:
        """Fetch broker open orders and map them to OrderSnapshot DTOs."""
        orders = await self._trader.fetch_broker_open_orders()
        result: list[OrderSnapshot] = []
        for order in orders:
            result.append(
                OrderSnapshot(
                    order_id=order.order_id,
                    client_order_id=order.client_order_id,
                    side=str(order.side.value if hasattr(order.side, "value") else order.side),
                    qty=order.quantity or Decimal("0"),
                    price=order.price,
                    trigger_price=order.trigger_price,
                    reduce_only=order.reduce_only,
                    raw=dict(order.raw),
                )
            )
        return result

    # ------------------------------------------------------------------
    # Order placement methods
    # ------------------------------------------------------------------

    async def place_market_order(
        self,
        *,
        side: str,
        qty: Decimal,
        reduce_only: bool,
        client_order_id: str,
    ) -> OrderResult:
        """Place a market order via the wrapped Trader."""
        position_side = _normalise_position_side(side)
        contracts_text = self._trader.decimal_to_str(qty)
        normalised_cid = _normalise_client_order_id(client_order_id)

        if reduce_only:
            body = order_specs.build_reduce_only_market_order_body(
                inst_id=self._trader.symbol,
                td_mode=self._trader.td_mode,
                side=position_side,
                contracts_text=contracts_text,
                pos_side_mode=self._trader.pos_side_mode,
            )
        else:
            body = order_specs.build_market_entry_order_body(
                inst_id=self._trader.symbol,
                td_mode=self._trader.td_mode,
                side=position_side,
                contracts_text=contracts_text,
                pos_side_mode=self._trader.pos_side_mode,
            )

        if normalised_cid is not None:
            body["clOrdId"] = normalised_cid

        res = await self._trader.request("POST", "/api/v5/trade/order", body)
        order_id = self._trader.extract_order_id(res)

        return OrderResult(ok=True, order_id=order_id, client_order_id=normalised_cid, raw=res)

    async def place_limit_order(
        self,
        *,
        side: str,
        qty: Decimal,
        price: Decimal,
        reduce_only: bool,
        client_order_id: str,
    ) -> OrderResult:
        """Place a limit order.

        Currently only supports reduce-only limit orders (used for TP).
        """
        if not reduce_only:
            raise ValueError(
                "OkxTradingClient.place_limit_order currently supports reduce_only=True only"
            )

        position_side = _normalise_position_side(side)
        contracts_text = self._trader.decimal_to_str(qty)
        price_text = self._trader.price_to_str(float(price))
        normalised_cid = _normalise_client_order_id(client_order_id)

        body = order_specs.build_reduce_only_tp_order_body(
            inst_id=self._trader.symbol,
            td_mode=self._trader.td_mode,
            side=position_side,
            contracts_text=contracts_text,
            price_text=price_text,
            pos_side_mode=self._trader.pos_side_mode,
            client_order_id=normalised_cid,
        )

        res = await self._trader.request("POST", "/api/v5/trade/order", body)
        order_id = self._trader.extract_order_id(res)

        return OrderResult(ok=True, order_id=order_id, client_order_id=normalised_cid, raw=res)

    async def place_stop_market_order(
        self,
        *,
        side: str,
        qty: Decimal | None,
        trigger_price: Decimal,
        reduce_only: bool,
        client_order_id: str,
    ) -> OrderResult:
        """Place a stop-market (conditional) order.

        Currently only supports reduce-only stop-market orders (used for SL).
        If *qty* is ``None``, the current position quantity is used.
        """
        if not reduce_only:
            raise ValueError(
                "OkxTradingClient.place_stop_market_order currently supports reduce_only=True only"
            )

        # Resolve quantity
        effective_qty = qty
        if effective_qty is None:
            pos = await self.fetch_position()
            effective_qty = pos.qty

        if effective_qty <= 0:
            raise RuntimeError(
                f"OkxTradingClient.place_stop_market_order requires qty > 0, got {effective_qty}"
            )

        position_side = _normalise_position_side(side)
        contracts_text = self._trader.decimal_to_str(effective_qty)
        stop_price_text = self._trader.price_to_str(float(trigger_price))
        normalised_cid = _normalise_client_order_id(client_order_id)

        body = order_specs.build_conditional_protective_sl_algo_body(
            inst_id=self._trader.symbol,
            td_mode=self._trader.td_mode,
            side=position_side,
            contracts_text=contracts_text,
            stop_price_text=stop_price_text,
            pos_side_mode=self._trader.pos_side_mode,
        )
        if normalised_cid is not None:
            body["algoClOrdId"] = normalised_cid

        res = await self._trader.request("POST", "/api/v5/trade/order-algo", body)
        order_id = self._trader.extract_algo_id(res)

        return OrderResult(ok=True, order_id=order_id, client_order_id=normalised_cid, raw=res)

    # ------------------------------------------------------------------
    # Cancel method
    # ------------------------------------------------------------------

    async def cancel_order(
        self,
        *,
        order_id: str | None = None,
        client_order_id: str | None = None,
    ) -> CancelResult:
        """Cancel an order by *order_id* or *client_order_id*.

        Tries regular cancel first.  If that fails and *order_id* is
        available, falls back to algo cancel.  Fails fast on errors.
        """
        normalised_cid = _normalise_client_order_id(client_order_id)
        if order_id is None and normalised_cid is None:
            raise ValueError("cancel_order requires at least one of order_id or client_order_id")

        # Build the regular cancel body
        if order_id is not None:
            body: dict[str, Any] = order_specs.build_cancel_order_body(
                inst_id=self._trader.symbol,
                order_id=order_id,
            )
        else:
            body = {"instId": self._trader.symbol, "clOrdId": normalised_cid}

        try:
            res = await self._trader.request("POST", "/api/v5/trade/cancel-order", body)
        except Exception:
            if order_id is None:
                raise
            # Fallback: try algo cancel
            algo_body = order_specs.build_cancel_algo_body(
                inst_id=self._trader.symbol,
                algo_id=order_id,
            )
            res = await self._trader.request("POST", "/api/v5/trade/cancel-algos", algo_body)

        return CancelResult(ok=True, order_id=order_id, client_order_id=normalised_cid, raw=res)

    # ------------------------------------------------------------------
    # Instrument configuration
    # ------------------------------------------------------------------

    async def configure_instrument(self) -> None:
        """Configure instrument-level settings (leverage / margin mode).

        Current phase delegates to Trader.set_leverage() as a legacy
        bridge.  Future phases may inline the REST call here.
        """
        await self._trader.set_leverage()

    # ------------------------------------------------------------------
    # Order status query
    # ------------------------------------------------------------------

    async def fetch_order_status(
        self,
        *,
        order_id: str | None = None,
        client_order_id: str | None = None,
    ) -> OrderStatusSnapshot:
        """Fetch the current status of a single order.

        At least one of *order_id* or *client_order_id* must be provided.
        Prefers *order_id* when both are given.
        """
        if order_id is None and client_order_id is None:
            raise ValueError(
                "fetch_order_status requires at least one of order_id or client_order_id"
            )

        if order_id is not None:
            endpoint = (
                f"/api/v5/trade/order?instId={self._trader.symbol}&ordId={order_id}"
            )
        else:
            endpoint = (
                f"/api/v5/trade/order?instId={self._trader.symbol}&clOrdId={client_order_id}"
            )

        try:
            res = await self._trader.request("GET", endpoint)
        except Exception:
            return OrderStatusSnapshot(
                order_id=order_id,
                client_order_id=client_order_id,
                status="UNKNOWN",
                raw={},
            )

        data = res.get("data", [])
        if not data:
            return OrderStatusSnapshot(
                order_id=order_id,
                client_order_id=client_order_id,
                status="NOT_FOUND",
                raw=res,
            )

        item = data[0]
        state = str(item.get("state") or "").lower()
        if state in {"live", "partially_filled"}:
            status = "OPEN"
        elif state == "filled":
            status = "FILLED"
        elif state in {"canceled", "cancelled"}:
            status = "CANCELED"
        else:
            status = "UNKNOWN"

        return OrderStatusSnapshot(
            order_id=str(item.get("ordId")) if item.get("ordId") else order_id,
            client_order_id=str(item.get("clOrdId")) if item.get("clOrdId") else client_order_id,
            status=status,
            filled_qty=_safe_decimal(item.get("accFillSz")),
            avg_fill_price=_safe_decimal(item.get("avgPx")),
            raw=item,
        )

    # ------------------------------------------------------------------
    # Algo order query
    # ------------------------------------------------------------------

    async def fetch_open_algo_orders(self) -> tuple[AlgoOrderSnapshot, ...]:
        """Fetch all open algo (conditional) orders.

        Current phase delegates to Trader.fetch_pending_algo_orders()
        as a legacy bridge, then parses each raw item into an
        AlgoOrderSnapshot DTO.
        """
        raws = await self._trader.fetch_pending_algo_orders()
        results: list[AlgoOrderSnapshot] = []
        for item in raws:
            order_id = str(item.get("algoId") or item.get("ordId") or "")
            client_order_id = str(item.get("clOrdId")) if item.get("clOrdId") else None
            side = str(item.get("side", "")) if item.get("side") else None
            qty = _safe_decimal(item.get("sz"))
            trigger_price = _safe_decimal(
                item.get("slTriggerPx") or item.get("triggerPx")
            )
            status = str(item.get("state") or "OPEN")
            results.append(
                AlgoOrderSnapshot(
                    order_id=order_id or None,
                    client_order_id=client_order_id,
                    side=side,
                    qty=qty,
                    trigger_price=trigger_price,
                    status=status,
                    raw=item,
                )
            )
        return tuple(results)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _safe_decimal(value: Any) -> Decimal | None:
    """Convert *value* to Decimal, or return None on failure."""
    try:
        if value in {None, ""}:
            return None
        return Decimal(str(value))
    except Exception:
        return None
