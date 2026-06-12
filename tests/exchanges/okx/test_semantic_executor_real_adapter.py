"""Real-adapter combination tests for OkxBrokerSemanticExecutor.

Uses FakeTrader + real OkxBrokerClient + real OkxBrokerSemanticExecutor to
verify that the semantic executor wires correctly through the broker client to
the underlying trader (fake) endpoints.

No live OKX access is involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pytest

from src.exchanges.errors import ExchangeError, ExchangeErrorKind
from src.exchanges.models import (
    BrokerOrder,
    BrokerOrderRequest,
    BrokerOrderSide,
    BrokerOrderStatus,
    BrokerOrderType,
    BrokerPositionSide,
    ExchangeName,
)
from src.exchanges.okx.client import OkxBrokerClient
from src.exchanges.okx.semantic_executor import OkxBrokerSemanticExecutor
from src.exchanges.semantic_models import (
    BrokerSemanticAction,
    BrokerSemanticOrderQuery,
    BrokerSemanticOrderRole,
    BrokerSemanticRequest,
)


# ---------------------------------------------------------------------------
# FakeTrader (lightweight, mirrors the one in test_client.py)
# ---------------------------------------------------------------------------


@dataclass
class _FakeMetadata:
    contract_multiplier: Decimal = Decimal("0.1")
    contract_precision: Decimal = Decimal("0.01")
    min_contracts: Decimal = Decimal("0.01")


@dataclass
class _FakePositionSnapshot:
    side: str | None
    contracts: Decimal
    avg_entry_price: float
    eth_qty: float
    raw_pos: Decimal


class FakeTrader:
    """Lightweight fake implementing the _OkxTraderLike protocol."""

    def __init__(
        self,
        *,
        symbol: str = "ETH-USDT-SWAP",
        td_mode: str = "isolated",
        pos_side_mode: str = "net",
        equity: float = 100.0,
        position_snapshot: _FakePositionSnapshot | None = None,
        pending_orders: list[dict[str, Any]] | None = None,
        pending_algo_orders: list[dict[str, Any]] | None = None,
        order_response: dict[str, Any] | None = None,
        algo_order_response: dict[str, Any] | None = None,
        cancel_response: dict[str, Any] | None = None,
        cancel_algo_response: dict[str, Any] | None = None,
        fail_on_request: bool = False,
    ) -> None:
        self.symbol = symbol
        self.td_mode = td_mode
        self.pos_side_mode = pos_side_mode
        self.instrument_metadata = _FakeMetadata()
        self._equity = equity
        self._position_snapshot = position_snapshot or _FakePositionSnapshot(
            side=None,
            contracts=Decimal("0"),
            avg_entry_price=0.0,
            eth_qty=0.0,
            raw_pos=Decimal("0"),
        )
        self._pending_orders = pending_orders or []
        self._pending_algo_orders = pending_algo_orders or []
        self._order_response = order_response
        self._algo_order_response = algo_order_response
        self._cancel_response = cancel_response
        self._cancel_algo_response = cancel_algo_response
        self._fail_on_request = fail_on_request
        self.requests: list[dict[str, Any]] = []
        self._next_order_id = 1
        self.closed = False

    async def fetch_usdt_equity(self) -> float:
        return self._equity

    async def fetch_position_snapshot(self) -> _FakePositionSnapshot:
        return self._position_snapshot

    async def fetch_pending_orders(self) -> list[dict[str, Any]]:
        return list(self._pending_orders)

    async def fetch_pending_algo_orders(self) -> list[dict[str, Any]]:
        return list(self._pending_algo_orders)

    async def request(
        self, method: str, endpoint: str, payload: Any | None = None
    ) -> dict[str, Any]:
        if self._fail_on_request:
            raise RuntimeError("FakeTrader request failure")
        self.requests.append(
            {"method": method, "endpoint": endpoint, "payload": payload}
        )
        if endpoint == "/api/v5/trade/order":
            if self._order_response is not None:
                return self._order_response
            ord_id = str(self._next_order_id)
            self._next_order_id += 1
            return {
                "code": "0",
                "data": [
                    {
                        "ordId": ord_id,
                        "clOrdId": payload.get("clOrdId", "") if payload else "",
                        "sCode": "0",
                    }
                ],
            }
        if endpoint == "/api/v5/trade/order-algo":
            if self._algo_order_response is not None:
                return self._algo_order_response
            algo_id = f"algo-{self._next_order_id}"
            self._next_order_id += 1
            return {"code": "0", "data": [{"algoId": algo_id, "sCode": "0"}]}
        if endpoint == "/api/v5/trade/cancel-order":
            if self._cancel_response is not None:
                return self._cancel_response
            return {
                "code": "0",
                "data": [
                    {
                        "ordId": payload.get("ordId", ""),
                        "sCode": "0",
                    }
                ],
            }
        if endpoint == "/api/v5/trade/cancel-algos":
            if self._cancel_algo_response is not None:
                return self._cancel_algo_response
            return {
                "code": "0",
                "data": [
                    {
                        "algoId": payload[0].get("algoId", "") if payload else "",
                        "sCode": "0",
                    }
                ],
            }
        return {"code": "0", "data": []}

    @staticmethod
    def extract_order_id(res: dict[str, Any]) -> str:
        data = res.get("data", [])
        if not data or not data[0].get("ordId"):
            raise RuntimeError(f"Missing ordId in response: {res}")
        return str(data[0]["ordId"])

    @staticmethod
    def extract_algo_id(res: dict[str, Any]) -> str:
        data = res.get("data", [])
        if not data or not (data[0].get("algoId") or data[0].get("ordId")):
            raise RuntimeError(f"Missing algoId in response: {res}")
        return str(data[0].get("algoId") or data[0].get("ordId"))

    @staticmethod
    def decimal_to_str(value: Any) -> str:
        from decimal import Decimal as D

        if isinstance(value, D):
            return format(value.normalize(), "f")
        return format(D(str(value)).normalize(), "f")

    @staticmethod
    def price_to_str(price: float) -> str:
        return f"{price:.2f}"

    async def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _semantic_request(
    *,
    exchange: ExchangeName = ExchangeName.OKX,
    symbol: str = "ETH-USDT-SWAP",
    action: BrokerSemanticAction,
    role: BrokerSemanticOrderRole = BrokerSemanticOrderRole.UNKNOWN,
    side: BrokerOrderSide = BrokerOrderSide.BUY,
    position_side: BrokerPositionSide = BrokerPositionSide.LONG,
    quantity: Decimal = Decimal("1"),
    price: Decimal | None = None,
    trigger_price: Decimal | None = None,
    order_id: str | None = None,
) -> BrokerSemanticRequest:
    return BrokerSemanticRequest(
        exchange=exchange,
        symbol=symbol,
        action=action,
        role=role,
        side=side,
        position_side=position_side,
        quantity=quantity,
        price=price,
        trigger_price=trigger_price,
        order_id=order_id,
    )


def _pending_order(
    *,
    ord_id: str = "ord-1",
    side: str = "sell",
    pos_side: str = "long",
    ord_type: str = "limit",
    state: str = "live",
    px: str | None = None,
    sz: str = "1.0",
    label: str | None = None,
) -> dict[str, Any]:
    order: dict[str, Any] = {
        "ordId": ord_id,
        "side": side,
        "posSide": pos_side,
        "ordType": ord_type,
        "state": state,
        "sz": sz,
        "accFillSz": "0",
    }
    if px is not None:
        order["px"] = px
    if label is not None:
        order["label"] = label
    return order


def _pending_algo_order(
    *,
    algo_id: str = "algo-1",
    side: str = "sell",
    pos_side: str = "long",
    ord_type: str = "conditional",
    state: str = "live",
    sl_trigger_px: str = "3100.00",
    sz: str = "1.0",
    label: str | None = None,
) -> dict[str, Any]:
    order: dict[str, Any] = {
        "algoId": algo_id,
        "side": side,
        "posSide": pos_side,
        "ordType": ord_type,
        "state": state,
        "slTriggerPx": sl_trigger_px,
        "sz": sz,
        "accFillSz": "0",
    }
    if label is not None:
        order["label"] = label
    return order


# ---------------------------------------------------------------------------
# 1. PLACE_PROTECTIVE_STOP real adapter
# ---------------------------------------------------------------------------


class TestPlaceProtectiveStopRealAdapter:
    @pytest.mark.asyncio
    async def test_places_algo_order_via_correct_endpoint(self):
        trader = FakeTrader()
        client = OkxBrokerClient(trader)
        executor = OkxBrokerSemanticExecutor(client)

        request = _semantic_request(
            action=BrokerSemanticAction.PLACE_PROTECTIVE_STOP,
            role=BrokerSemanticOrderRole.PROTECTIVE_SL,
            side=BrokerOrderSide.SELL,
            position_side=BrokerPositionSide.LONG,
            quantity=Decimal("1"),
            trigger_price=Decimal("2800"),
        )

        result = await executor.execute_semantic_order(request)

        assert result.ok is True
        assert result.order_id == "algo-1"
        assert result.action == BrokerSemanticAction.PLACE_PROTECTIVE_STOP
        assert result.role == BrokerSemanticOrderRole.PROTECTIVE_SL

        # Verify FakeTrader received the right endpoint and payload
        assert len(trader.requests) == 1
        r = trader.requests[0]
        assert r["endpoint"] == "/api/v5/trade/order-algo"
        assert r["payload"]["ordType"] == "conditional"
        assert r["payload"]["slTriggerPx"] == "2800.00"
        assert r["payload"]["reduceOnly"] == "true"
        assert r["payload"]["side"] == "sell"

    @pytest.mark.asyncio
    async def test_protective_stop_short_side(self):
        trader = FakeTrader()
        client = OkxBrokerClient(trader)
        executor = OkxBrokerSemanticExecutor(client)

        request = _semantic_request(
            action=BrokerSemanticAction.PLACE_PROTECTIVE_STOP,
            role=BrokerSemanticOrderRole.PROTECTIVE_SL,
            side=BrokerOrderSide.BUY,
            position_side=BrokerPositionSide.SHORT,
            quantity=Decimal("2"),
            trigger_price=Decimal("4000"),
        )

        result = await executor.execute_semantic_order(request)

        assert result.ok is True
        r = trader.requests[0]
        assert r["endpoint"] == "/api/v5/trade/order-algo"
        assert r["payload"]["side"] == "buy"
        assert r["payload"]["slTriggerPx"] == "4000.00"
        assert r["payload"]["sz"] == "2"


# ---------------------------------------------------------------------------
# 2. MARKET_EXIT real adapter
# ---------------------------------------------------------------------------


class TestMarketExitRealAdapter:
    @pytest.mark.asyncio
    async def test_places_reduce_only_market_order(self):
        trader = FakeTrader()
        client = OkxBrokerClient(trader)
        executor = OkxBrokerSemanticExecutor(client)

        request = _semantic_request(
            action=BrokerSemanticAction.MARKET_EXIT,
            role=BrokerSemanticOrderRole.MARKET_EXIT,
            side=BrokerOrderSide.SELL,
            position_side=BrokerPositionSide.LONG,
            quantity=Decimal("1"),
        )

        result = await executor.execute_semantic_order(request)

        assert result.ok is True
        assert result.order_id == "1"
        assert result.action == BrokerSemanticAction.MARKET_EXIT

        # Verify endpoint and payload
        assert len(trader.requests) == 1
        r = trader.requests[0]
        assert r["endpoint"] == "/api/v5/trade/order"
        assert r["payload"]["ordType"] == "market"
        assert r["payload"]["reduceOnly"] == "true"
        assert r["payload"]["side"] == "sell"

    @pytest.mark.asyncio
    async def test_market_exit_short_position(self):
        trader = FakeTrader()
        client = OkxBrokerClient(trader)
        executor = OkxBrokerSemanticExecutor(client)

        request = _semantic_request(
            action=BrokerSemanticAction.MARKET_EXIT,
            role=BrokerSemanticOrderRole.MARKET_EXIT,
            side=BrokerOrderSide.BUY,
            position_side=BrokerPositionSide.SHORT,
            quantity=Decimal("3"),
        )

        result = await executor.execute_semantic_order(request)

        assert result.ok is True
        r = trader.requests[0]
        assert r["endpoint"] == "/api/v5/trade/order"
        assert r["payload"]["side"] == "buy"
        assert r["payload"]["reduceOnly"] == "true"


# ---------------------------------------------------------------------------
# 3. MARKET_EXIT_RUNNER real adapter
# ---------------------------------------------------------------------------


class TestMarketExitRunnerRealAdapter:
    @pytest.mark.asyncio
    async def test_market_exit_runner_uses_same_reduce_only_market_path(self):
        trader = FakeTrader()
        client = OkxBrokerClient(trader)
        executor = OkxBrokerSemanticExecutor(client)

        request = _semantic_request(
            action=BrokerSemanticAction.MARKET_EXIT_RUNNER,
            role=BrokerSemanticOrderRole.MARKET_EXIT,
            side=BrokerOrderSide.SELL,
            position_side=BrokerPositionSide.LONG,
            quantity=Decimal("1"),
        )

        result = await executor.execute_semantic_order(request)

        assert result.ok is True
        assert result.action == BrokerSemanticAction.MARKET_EXIT_RUNNER
        assert result.order_id == "1"

        r = trader.requests[0]
        assert r["endpoint"] == "/api/v5/trade/order"
        assert r["payload"]["ordType"] == "market"
        assert r["payload"]["reduceOnly"] == "true"


# ---------------------------------------------------------------------------
# 4. CANCEL_PROTECTIVE_STOP real adapter
# ---------------------------------------------------------------------------


class TestCancelProtectiveStopRealAdapter:
    @pytest.mark.asyncio
    async def test_cancels_via_algo_cancel_endpoint(self):
        trader = FakeTrader()
        client = OkxBrokerClient(trader)
        executor = OkxBrokerSemanticExecutor(client)

        request = BrokerSemanticRequest(
            exchange=ExchangeName.OKX,
            symbol="ETH-USDT-SWAP",
            action=BrokerSemanticAction.CANCEL_PROTECTIVE_STOP,
            role=BrokerSemanticOrderRole.PROTECTIVE_SL,
            order_id="algo-123",
        )

        result = await executor.execute(request)

        assert result.ok is True
        assert result.order_id == "algo-123"

        # Must use /api/v5/trade/cancel-algos, NOT /api/v5/trade/cancel-order
        assert len(trader.requests) == 1
        r = trader.requests[0]
        assert r["endpoint"] == "/api/v5/trade/cancel-algos"
        assert r["payload"] == [{"instId": "ETH-USDT-SWAP", "algoId": "algo-123"}]

    @pytest.mark.asyncio
    async def test_cancel_protective_stop_via_cancel_semantic_order(self):
        trader = FakeTrader()
        client = OkxBrokerClient(trader)
        executor = OkxBrokerSemanticExecutor(client)

        request = _semantic_request(
            action=BrokerSemanticAction.CANCEL_PROTECTIVE_STOP,
            role=BrokerSemanticOrderRole.PROTECTIVE_SL,
        )

        result = await executor.cancel_semantic_order(request, "algo-456")

        assert result.ok is True
        assert result.order_id == "algo-456"

        r = trader.requests[0]
        assert r["endpoint"] == "/api/v5/trade/cancel-algos"


# ---------------------------------------------------------------------------
# 5. CANCEL_REDUCE_ONLY_TP real adapter
# ---------------------------------------------------------------------------


class TestCancelReduceOnlyTPRealAdapter:
    @pytest.mark.asyncio
    async def test_cancels_via_ordinary_cancel_endpoint(self):
        trader = FakeTrader()
        client = OkxBrokerClient(trader)
        executor = OkxBrokerSemanticExecutor(client)

        request = BrokerSemanticRequest(
            exchange=ExchangeName.OKX,
            symbol="ETH-USDT-SWAP",
            action=BrokerSemanticAction.CANCEL_REDUCE_ONLY_TP,
            role=BrokerSemanticOrderRole.CORE_TP,
            order_id="ord-1",
        )

        result = await executor.execute(request)

        assert result.ok is True
        assert result.order_id == "ord-1"

        # Must use /api/v5/trade/cancel-order
        assert len(trader.requests) == 1
        r = trader.requests[0]
        assert r["endpoint"] == "/api/v5/trade/cancel-order"
        assert r["payload"]["instId"] == "ETH-USDT-SWAP"
        assert r["payload"]["ordId"] == "ord-1"


# ---------------------------------------------------------------------------
# 6. cancel_semantic_orders_by_role mixed ordinary/algo
# ---------------------------------------------------------------------------


class TestCancelSemanticOrdersByRoleMixed:
    @pytest.mark.asyncio
    async def test_routes_ordinary_to_cancel_order_and_algo_to_cancel_algo(self):
        trader = FakeTrader(
            pending_orders=[
                _pending_order(
                    ord_id="tp-1",
                    side="sell",
                    pos_side="long",
                    ord_type="limit",
                    px="3500.00",
                    label=BrokerSemanticOrderRole.CORE_TP.value,
                ),
            ],
            pending_algo_orders=[
                _pending_algo_order(
                    algo_id="sl-1",
                    side="sell",
                    pos_side="long",
                    label=BrokerSemanticOrderRole.PROTECTIVE_SL.value,
                ),
            ],
        )
        client = OkxBrokerClient(trader)
        executor = OkxBrokerSemanticExecutor(client)

        query = BrokerSemanticOrderQuery(
            exchange=ExchangeName.OKX,
            symbol="ETH-USDT-SWAP",
            roles=(BrokerSemanticOrderRole.CORE_TP, BrokerSemanticOrderRole.PROTECTIVE_SL),
            include_ordinary=True,
            include_algo=True,
        )

        results = await executor.cancel_semantic_orders_by_role(query)

        assert len(results) == 2
        assert all(r.ok for r in results)

        # Find which went to which endpoint
        cancel_order_requests = [
            r for r in trader.requests if r["endpoint"] == "/api/v5/trade/cancel-order"
        ]
        cancel_algo_requests = [
            r for r in trader.requests if r["endpoint"] == "/api/v5/trade/cancel-algos"
        ]

        assert len(cancel_order_requests) == 1
        assert cancel_order_requests[0]["payload"]["ordId"] == "tp-1"

        assert len(cancel_algo_requests) == 1
        assert cancel_algo_requests[0]["payload"] == [
            {"instId": "ETH-USDT-SWAP", "algoId": "sl-1"}
        ]

    @pytest.mark.asyncio
    async def test_algo_order_by_source_routed_to_cancel_algo(self):
        trader = FakeTrader(
            pending_orders=[
                _pending_order(
                    ord_id="ord-no-label",
                    side="sell",
                    pos_side="long",
                    ord_type="limit",
                    px="3500.00",
                    label=None,
                ),
            ],
            pending_algo_orders=[
                _pending_algo_order(
                    algo_id="algo-no-label",
                    side="sell",
                    pos_side="long",
                    label=None,
                ),
            ],
        )
        client = OkxBrokerClient(trader)
        executor = OkxBrokerSemanticExecutor(client)

        query = BrokerSemanticOrderQuery(
            exchange=ExchangeName.OKX,
            symbol="ETH-USDT-SWAP",
            roles=(BrokerSemanticOrderRole.UNKNOWN,),
            include_ordinary=True,
            include_algo=True,
        )

        results = await executor.cancel_semantic_orders_by_role(query)

        assert len(results) == 2

        # ordinary unknown-label → cancel-order
        cancel_order_ids = {
            r["payload"]["ordId"]
            for r in trader.requests
            if r["endpoint"] == "/api/v5/trade/cancel-order"
        }
        assert "ord-no-label" in cancel_order_ids

        # algo unknown-label (but source="algo") → cancel-algos
        cancel_algo_ids = {
            r["payload"][0]["algoId"]
            for r in trader.requests
            if r["endpoint"] == "/api/v5/trade/cancel-algos"
        }
        assert "algo-no-label" in cancel_algo_ids


# ---------------------------------------------------------------------------
# 7. RECOVER_OPEN_ORDERS
# ---------------------------------------------------------------------------


class TestRecoverOpenOrdersRealAdapter:
    @pytest.mark.asyncio
    async def test_returns_ordinary_and_algo_orders_with_source_preserved(self):
        trader = FakeTrader(
            pending_orders=[
                _pending_order(ord_id="ord-1", side="sell", pos_side="long"),
            ],
            pending_algo_orders=[
                _pending_algo_order(algo_id="algo-1", side="sell", pos_side="long"),
            ],
        )
        client = OkxBrokerClient(trader)
        executor = OkxBrokerSemanticExecutor(client)

        request = BrokerSemanticRequest(
            exchange=ExchangeName.OKX,
            symbol="ETH-USDT-SWAP",
            action=BrokerSemanticAction.RECOVER_OPEN_ORDERS,
        )

        result = await executor.execute(request)

        assert result.ok is True
        assert len(result.orders) == 2

        sources = {order.raw.get("source") for order in result.orders}
        assert sources == {"ordinary", "algo"}

        ord_ids = {order.order_id for order in result.orders}
        assert ord_ids == {"ord-1", "algo-1"}

    @pytest.mark.asyncio
    async def test_recover_only_ordinary_when_no_algo_orders(self):
        trader = FakeTrader(
            pending_orders=[
                _pending_order(ord_id="ord-1", side="sell", pos_side="long"),
            ],
            pending_algo_orders=[],
        )
        client = OkxBrokerClient(trader)
        executor = OkxBrokerSemanticExecutor(client)

        request = BrokerSemanticRequest(
            exchange=ExchangeName.OKX,
            symbol="ETH-USDT-SWAP",
            action=BrokerSemanticAction.RECOVER_OPEN_ORDERS,
        )

        result = await executor.execute(request)

        assert result.ok is True
        assert len(result.orders) == 1
        assert result.orders[0].raw["source"] == "ordinary"
        assert result.orders[0].order_id == "ord-1"


# ---------------------------------------------------------------------------
# Edge case: unsupported cancel_algo_order when broker lacks it
# ---------------------------------------------------------------------------


class TestCancelProtectiveStopWithoutAlgoSupport:
    @pytest.mark.asyncio
    async def test_cancel_semantic_order_raises_unsupported_when_no_algo_cancel(self):
        """When broker lacks cancel_algo_order, CANCEL_PROTECTIVE_STOP must raise."""

        # Create a broker without cancel_algo_order
        class BrokerWithoutAlgoCancel:
            exchange = ExchangeName.OKX

            async def cancel_order(self, symbol: str, order_id: str) -> None:
                pass

            async def fetch_open_orders(self, symbol: str) -> list:
                return []

            async def fetch_position(self, symbol, side=None):
                from src.exchanges.models import BrokerPosition
                return BrokerPosition(
                    exchange=ExchangeName.OKX,
                    symbol=symbol,
                    side=BrokerPositionSide.NET,
                    contracts=Decimal("0"),
                    base_qty=Decimal("0"),
                    avg_entry_price=Decimal("0"),
                )

            async def close(self) -> None:
                pass

        executor = OkxBrokerSemanticExecutor(BrokerWithoutAlgoCancel())
        request = _semantic_request(
            action=BrokerSemanticAction.CANCEL_PROTECTIVE_STOP,
            role=BrokerSemanticOrderRole.PROTECTIVE_SL,
        )

        with pytest.raises(ExchangeError) as exc_info:
            await executor.cancel_semantic_order(request, "algo-1")

        assert exc_info.value.detail.kind == ExchangeErrorKind.UNSUPPORTED_OPERATION
