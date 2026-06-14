#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : okx_trading_client.py
@Description: OKX implementation of TradingClientPort.

This class owns an OkxPrivateClient for direct OKX REST access.
It does NOT depend on Trader._client or Trader.request() for its
primary execution path.
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
    from src.execution.okx_private_client import OkxPrivateClient, PrivateWriteRateLimiter
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


def _normalise_okx_side(raw_side: str) -> str:
    """Normalise OKX side strings to uppercase: "buy" → "BUY", "sell" → "SELL"."""
    return raw_side.strip().upper()


class OkxTradingClient(TradingClientPort):
    """OKX implementation of TradingClientPort.

    Owns an OkxPrivateClient for direct OKX private REST access.
    References Trader only for exchange-agnostic config fields
    (symbol, td_mode, pos_side_mode, contract_multiplier, leverage)
    and formatting helpers (decimal_to_str, price_to_str, etc.).
    """

    def __init__(
        self,
        trader: Trader,
        *,
        private_client: OkxPrivateClient,
        rate_limiter: PrivateWriteRateLimiter | None = None,
    ) -> None:
        self._trader = trader
        self._client = private_client
        self._limiter = rate_limiter

    # ------------------------------------------------------------------
    # Lifecycle (NOT on TradingClientPort — called by runtime_factory)
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the underlying private REST session."""
        await self._client.start()

    async def close(self) -> None:
        """Close the underlying private REST session."""
        await self._client.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _post(self, endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
        """Rate-limited POST request to OKX private REST."""
        if self._limiter is not None:
            await self._limiter.acquire()
        return await self._client.request("POST", endpoint, body)

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    async def fetch_balance(self) -> BalanceSnapshot:
        """Fetch USDT equity directly from OKX private REST.

        Does NOT go through Trader.fetch_usdt_equity() to avoid recursion
        (Trader.fetch_usdt_equity() now delegates to this method).
        """
        res = await self._client.request(
            "GET", "/api/v5/account/balance?ccy=USDT"
        )
        data = res.get("data", [])
        equity = 0.0
        if data:
            details = data[0].get("details", [])
            for item in details:
                if item.get("ccy") == "USDT":
                    equity = float(
                        item.get("eq") or item.get("availEq") or item.get("availBal") or 0.0
                    )
                    break
            if equity == 0.0:
                equity = float(data[0].get("totalEq") or 0.0)
        return BalanceSnapshot(
            asset="USDT",
            total=Decimal(str(equity)),
            available=None,
            raw={"account_equity_usdt": equity},
        )

    async def fetch_position(self) -> PositionSnapshot:
        """Fetch current position directly from OKX private REST.

        Does NOT go through Trader.fetch_position_snapshot() to avoid recursion
        (Trader.fetch_position_snapshot() now delegates to this method).
        """
        symbol = self._trader.symbol
        pos_side_mode = self._trader.pos_side_mode
        contract_multiplier = self._trader.contract_multiplier

        res = await self._client.request(
            "GET", f"/api/v5/account/positions?instId={symbol}"
        )
        best_side: str | None = None
        best_qty: Decimal = Decimal("0")
        best_avg: Decimal | None = None
        best_raw_pos: str = "0"
        for item in res.get("data", []):
            if item.get("instId") != symbol:
                continue
            raw_pos = Decimal(str(item.get("pos", "0")))
            if raw_pos == 0:
                continue
            contracts = abs(raw_pos)
            avg_entry = float(item.get("avgPx") or item.get("avgPxUsd") or 0.0)
            if pos_side_mode == "long_short":
                pos_side = str(item.get("posSide", "")).lower()
                side = "LONG" if pos_side == "long" else "SHORT" if pos_side == "short" else None
            else:
                side = "LONG" if raw_pos > 0 else "SHORT"
            best_side = side
            best_qty = contracts
            best_avg = Decimal(str(avg_entry)) if avg_entry else None
            best_raw_pos = str(raw_pos)
            break

        if best_side is None:
            return PositionSnapshot(side=None, qty=Decimal("0"))

        return PositionSnapshot(
            side=best_side,
            qty=best_qty,
            avg_entry_price=best_avg,
            raw={
                "contracts": str(best_qty),
                "eth_qty": float(best_qty * contract_multiplier),
                "raw_pos": best_raw_pos,
            },
        )

    async def fetch_open_orders(self) -> list[OrderSnapshot]:
        """Fetch open orders directly from OKX private REST.

        Does NOT go through Trader broker semantic executor.
        """
        symbol = self._trader.symbol
        res = await self._client.request(
            "GET", f"/api/v5/trade/orders-pending?instId={symbol}"
        )
        results: list[OrderSnapshot] = []
        for item in res.get("data", []):
            if item.get("instId") != symbol:
                continue
            results.append(
                OrderSnapshot(
                    order_id=str(item.get("ordId", "")),
                    client_order_id=str(item.get("clOrdId")) if item.get("clOrdId") else None,
                    side=_normalise_okx_side(str(item.get("side", ""))),
                    qty=_safe_decimal(item.get("sz")) or Decimal("0"),
                    price=_safe_decimal(item.get("px")),
                    trigger_price=None,
                    reduce_only=str(item.get("reduceOnly", "")).lower() == "true",
                    raw=item,
                )
            )
        return results

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
        """Place a market order directly via OKX private REST."""
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

        res = await self._post("/api/v5/trade/order", body)
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

        res = await self._post("/api/v5/trade/order", body)
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

        res = await self._post("/api/v5/trade/order-algo", body)
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
            res = await self._post("/api/v5/trade/cancel-order", body)
        except Exception:
            if order_id is None:
                raise
            # Fallback: try algo cancel
            algo_body = order_specs.build_cancel_algo_body(
                inst_id=self._trader.symbol,
                algo_id=order_id,
            )
            res = await self._post("/api/v5/trade/cancel-algos", algo_body)

        return CancelResult(ok=True, order_id=order_id, client_order_id=normalised_cid, raw=res)

    async def cancel_algo_order(
        self,
        *,
        order_id: str | None = None,
        client_order_id: str | None = None,
    ) -> CancelResult:
        """Cancel an algo (conditional) order by *order_id* or *client_order_id*.

        This is the API-level cancel for algo/conditional orders — it always
        uses the ``/api/v5/trade/cancel-algos`` endpoint.
        """
        if order_id is None and client_order_id is None:
            raise ValueError(
                "cancel_algo_order requires at least one of order_id or client_order_id"
            )

        normalised_cid = _normalise_client_order_id(client_order_id)

        if order_id is not None:
            body: dict[str, Any] = order_specs.build_cancel_algo_body(
                inst_id=self._trader.symbol,
                algo_id=order_id,
            )
        else:
            body = {
                "instId": self._trader.symbol,
                "algoClOrdId": normalised_cid,
            }

        res = await self._client.request(
            "POST", "/api/v5/trade/cancel-algos", body
        )

        return CancelResult(
            ok=True, order_id=order_id, client_order_id=normalised_cid, raw=res
        )

    # ------------------------------------------------------------------
    # Instrument configuration
    # ------------------------------------------------------------------

    async def configure_instrument(self) -> None:
        """Configure instrument-level settings (leverage / margin mode).

        Calls OKX private REST directly — does NOT go through
        Trader.set_leverage() to avoid recursion.
        """
        bodies = order_specs.build_set_leverage_bodies(
            inst_id=self._trader.symbol,
            td_mode=self._trader.td_mode,
            leverage=self._trader.leverage,
            pos_side_mode=self._trader.pos_side_mode,
        )
        for body in bodies:
            await self._client.request(
                "POST", "/api/v5/account/set-leverage", body
            )

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
            res = await self._client.request("GET", endpoint)
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
        """Fetch all open algo (conditional) orders directly from OKX private REST.

        Does NOT go through Trader.fetch_pending_algo_orders() to avoid recursion
        (Trader.fetch_pending_algo_orders() now delegates to this method).
        """
        symbol = self._trader.symbol
        res = await self._client.request(
            "GET",
            f"/api/v5/trade/orders-algo-pending?instId={symbol}&ordType=conditional",
        )
        raws = list(res.get("data", []))
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
