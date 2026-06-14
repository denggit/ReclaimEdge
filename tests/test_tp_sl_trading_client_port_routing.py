"""Tests verifying TP/SL primitives route through TradingClientPort.

Ref: 20C-CLEAN-PORTS-05

Coverage:
1. TP limit order routes to trading_client.place_limit_order()
2. Protective SL algo order routes to trading_client.place_stop_market_order()
3. cancel regular order routes to trading_client.cancel_order()
4. No direct /api/v5/trade/order|order-algo|cancel-order|cancel-algos calls
   in the replaced code paths.
5. Parameter consistency: side, contracts, price, stop_price, client_order_id,
   reduce_only=True.
"""

from __future__ import annotations

from decimal import Decimal
from dataclasses import dataclass, field
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Fake TradingClientPort — records every call with full parameter snapshots
# ---------------------------------------------------------------------------


@dataclass
class FakeOrderResult:
    ok: bool = True
    order_id: str | None = "fake-order-id"
    client_order_id: str | None = None
    message: str = ""


@dataclass
class FakeCancelResult:
    ok: bool = True
    order_id: str | None = None
    client_order_id: str | None = None
    message: str = ""


class FakeTradingClient:
    """Record every TradingClientPort call so tests can assert routing."""

    def __init__(self) -> None:
        self.place_limit_calls: list[dict[str, Any]] = []
        self.place_stop_calls: list[dict[str, Any]] = []
        self.cancel_calls: list[dict[str, Any]] = []
        self.open_orders: list[Any] = []

        # Configurable per-call return values
        self._next_limit_result = FakeOrderResult(order_id="fake-tp-1")
        self._next_stop_result = FakeOrderResult(order_id="fake-sl-1")
        self._next_cancel_result = FakeCancelResult(order_id="cancelled-1")

    async def place_limit_order(self, *, side, qty, price, reduce_only,
                                client_order_id) -> FakeOrderResult:
        self.place_limit_calls.append({
            "side": side, "qty": qty, "price": price,
            "reduce_only": reduce_only, "client_order_id": client_order_id,
        })
        return self._next_limit_result

    async def place_stop_market_order(self, *, side, qty, trigger_price,
                                      reduce_only, client_order_id) -> FakeOrderResult:
        self.place_stop_calls.append({
            "side": side, "qty": qty, "trigger_price": trigger_price,
            "reduce_only": reduce_only, "client_order_id": client_order_id,
        })
        return self._next_stop_result

    async def cancel_order(self, *, order_id=None,
                           client_order_id=None) -> FakeCancelResult:
        self.cancel_calls.append({
            "order_id": order_id, "client_order_id": client_order_id,
        })
        return self._next_cancel_result

    async def fetch_open_orders(self) -> list[Any]:
        return list(self.open_orders)


# ---------------------------------------------------------------------------
# Fake Trader (minimal — only what the managers touch during test setup)
# ---------------------------------------------------------------------------


def _make_minimal_trader() -> Any:
    """Return a bare Trader-like object with attributes the managers read at init."""
    from src.execution.trader import Trader

    t = Trader.__new__(Trader)
    t.symbol = "ETH-USDT-SWAP"
    t.td_mode = "isolated"
    t.leverage = "50"
    t.pos_side_mode = "net"
    t.live_trading = True
    t.contract_multiplier = Decimal("0.1")
    t.contract_precision = Decimal("0.01")
    t.min_contracts = Decimal("0.01")
    t.position_contracts = Decimal("0")
    t.account_equity_usdt = 0.0
    t.tp_order_id = None
    t.near_tp_protective_sl_order_id = None
    t.middle_runner_protective_sl_order_id = None
    t.three_stage_post_tp1_protective_sl_order_id = None
    t.trend_runner_sl_order_id = None
    t._protected_reduce_only_order_ids: set[str] = set()
    t._managed_reduce_only_order_ids: set[str] = set()
    t._allow_cancel_unmanaged_reduce_only = True
    return t


# ===================================================================
# 1. TP limit order → place_limit_order()
# ===================================================================


class TestTpLimitOrderRoutesToPlaceLimitOrder:
    """TP reduce-only limit order calls trading_client.place_limit_order()."""

    @pytest.mark.asyncio
    async def test_tp_limit_order_routes_to_port(self) -> None:
        from src.execution.tp_sl_core_tp_manager import CoreTakeProfitManager

        trader = _make_minimal_trader()
        trader.position_contracts = Decimal("10")
        trader.decimal_to_str = lambda v: str(v)
        trader.price_to_str = lambda p: f"{float(p):.2f}"

        fake = FakeTradingClient()
        fake._next_limit_result = FakeOrderResult(order_id="tp-ord-1")

        manager = CoreTakeProfitManager(trader, protective_stops=None,
                                        trading_client=fake)

        # Create a minimal intent
        from types import SimpleNamespace
        intent = SimpleNamespace(side="LONG")

        order_ids = await manager._place_reduce_only_take_profit_orders(
            intent=intent,
            specs=[("final", Decimal("10"), 3100.0)],
        )

        # order_id comes from the port result
        assert order_ids == ["tp-ord-1"]
        assert len(fake.place_limit_calls) == 1
        call = fake.place_limit_calls[0]
        assert call["side"] == "LONG"
        assert call["qty"] == Decimal("10")
        assert call["price"] == Decimal("3100.0")
        assert call["reduce_only"] is True
        assert call["client_order_id"] == ""

    @pytest.mark.asyncio
    async def test_tp_limit_order_short_side(self) -> None:
        from src.execution.tp_sl_core_tp_manager import CoreTakeProfitManager

        trader = _make_minimal_trader()
        trader.position_contracts = Decimal("5")
        trader.decimal_to_str = lambda v: str(v)
        trader.price_to_str = lambda p: f"{float(p):.2f}"

        fake = FakeTradingClient()
        fake._next_limit_result = FakeOrderResult(order_id="tp-short-1")

        manager = CoreTakeProfitManager(trader, protective_stops=None,
                                        trading_client=fake)

        from types import SimpleNamespace
        intent = SimpleNamespace(side="SHORT")

        await manager._place_reduce_only_take_profit_orders(
            intent=intent,
            specs=[("final", Decimal("5"), 2900.0)],
        )

        assert fake.place_limit_calls[0]["side"] == "SHORT"
        assert fake.place_limit_calls[0]["price"] == Decimal("2900.0")

    @pytest.mark.asyncio
    async def test_multiple_tp_specs_each_routes_to_port(self) -> None:
        from src.execution.tp_sl_core_tp_manager import CoreTakeProfitManager

        trader = _make_minimal_trader()
        trader.position_contracts = Decimal("10")
        trader.decimal_to_str = lambda v: str(v)
        trader.price_to_str = lambda p: f"{float(p):.2f}"

        fake = FakeTradingClient()
        fake._next_limit_result = FakeOrderResult(order_id="tp-multi")

        manager = CoreTakeProfitManager(trader, protective_stops=None,
                                        trading_client=fake)

        from types import SimpleNamespace
        intent = SimpleNamespace(side="LONG")

        order_ids = await manager._place_reduce_only_take_profit_orders(
            intent=intent,
            specs=[
                ("tp1_middle_fast", Decimal("4"), 3050.0),
                ("tp2_outer", Decimal("6"), 3100.0),
            ],
        )

        assert order_ids == ["tp-multi", "tp-multi"]
        assert len(fake.place_limit_calls) == 2
        assert fake.place_limit_calls[0]["qty"] == Decimal("4")
        assert fake.place_limit_calls[1]["qty"] == Decimal("6")
        assert fake.place_limit_calls[0]["price"] == Decimal("3050.0")
        assert fake.place_limit_calls[1]["price"] == Decimal("3100.0")


# ===================================================================
# 2. Protective SL algo order → place_stop_market_order()
# ===================================================================


class TestProtectiveSlRoutesToPlaceStopMarketOrder:
    """Protective stop-loss routes to trading_client.place_stop_market_order()."""

    @pytest.mark.asyncio
    async def test_protective_sl_routes_to_port(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        trader = _make_minimal_trader()
        trader.decimal_to_str = lambda v: str(v)
        trader.price_to_str = lambda p: f"{float(p):.2f}"

        fake = FakeTradingClient()
        fake._next_stop_result = FakeOrderResult(order_id="sl-algo-1")

        manager = ProtectiveStopManager(trader, trading_client=fake)

        # Mock verify to succeed immediately
        async def fake_verify(_algo_id, _side, _contracts, _stop_price):
            return True

        trader.verify_near_tp_protective_stop = fake_verify

        ok, algo_id, message = await manager.place_near_tp_protective_stop_with_retries(
            side="LONG",
            contracts=Decimal("10"),
            stop_price=2950.0,
            retry_count=1,
            retry_interval_seconds=0,
        )

        assert ok is True
        assert algo_id == "sl-algo-1"
        assert message == "protective_sl_placed"
        assert len(fake.place_stop_calls) == 1
        call = fake.place_stop_calls[0]
        assert call["side"] == "LONG"
        assert call["qty"] == Decimal("10")
        assert call["trigger_price"] == Decimal("2950.0")
        assert call["reduce_only"] is True
        assert call["client_order_id"] == ""

    @pytest.mark.asyncio
    async def test_protective_sl_short_side(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        trader = _make_minimal_trader()
        trader.decimal_to_str = lambda v: str(v)
        trader.price_to_str = lambda p: f"{float(p):.2f}"

        fake = FakeTradingClient()
        fake._next_stop_result = FakeOrderResult(order_id="sl-short-1")

        manager = ProtectiveStopManager(trader, trading_client=fake)

        async def fake_verify(_algo_id, _side, _contracts, _stop_price):
            return True

        trader.verify_near_tp_protective_stop = fake_verify

        ok, algo_id, _msg = await manager.place_near_tp_protective_stop_with_retries(
            side="SHORT",
            contracts=Decimal("5"),
            stop_price=3700.0,
            retry_count=1,
            retry_interval_seconds=0,
        )

        assert ok is True
        assert algo_id == "sl-short-1"
        assert fake.place_stop_calls[0]["side"] == "SHORT"
        assert fake.place_stop_calls[0]["trigger_price"] == Decimal("3700.0")

    @pytest.mark.asyncio
    async def test_protective_sl_retries_on_port(self) -> None:
        """When verify fails, the port is called again on retry."""
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        trader = _make_minimal_trader()
        trader.decimal_to_str = lambda v: str(v)
        trader.price_to_str = lambda p: f"{float(p):.2f}"

        fake = FakeTradingClient()
        fake._next_stop_result = FakeOrderResult(order_id="sl-retry")

        manager = ProtectiveStopManager(trader, trading_client=fake)

        verify_count = [0]

        async def fake_verify(_algo_id, _side, _contracts, _stop_price):
            verify_count[0] += 1
            return verify_count[0] >= 2  # first attempt fails, retry succeeds

        async def fake_cancel_algo(_algo_id):
            return None

        trader.verify_near_tp_protective_stop = fake_verify
        trader._cancel_unverified_near_tp_algo = fake_cancel_algo
        trader.cancel_near_tp_protective_stop = lambda _: True

        ok, algo_id, message = await manager.place_near_tp_protective_stop_with_retries(
            side="LONG",
            contracts=Decimal("10"),
            stop_price=2950.0,
            retry_count=2,
            retry_interval_seconds=0,
        )

        assert ok is True
        assert algo_id == "sl-retry"
        assert message == "protective_sl_placed"
        # Called twice: once for failed attempt, once for successful retry
        assert len(fake.place_stop_calls) == 2


# ===================================================================
# 3. cancel regular order → cancel_order()
# ===================================================================


class TestCancelOrderRoutesToPort:
    """Regular order cancel routes to trading_client.cancel_order()."""

    @pytest.mark.asyncio
    async def test_cancel_reduce_only_order_routes_to_port(self) -> None:
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager
        from src.execution.trading_client_port import OrderSnapshot

        trader = _make_minimal_trader()
        fake = FakeTradingClient()
        fake.open_orders = [
            OrderSnapshot(
                order_id="core-old",
                client_order_id=None,
                side="sell",
                qty=Decimal("1"),
                reduce_only=True,
                raw={},
            ),
        ]

        manager = TpSlExecutionManager.__new__(TpSlExecutionManager)
        manager.trader = trader
        manager.trading_client = fake
        manager.protective_stops = None  # not needed for this test
        manager.core_tp = None
        manager.market_exit = None
        manager.near_tp = None
        manager.sidecar = None

        await manager.cancel_existing_reduce_only_orders()

        assert len(fake.cancel_calls) == 1
        assert fake.cancel_calls[0]["order_id"] == "core-old"

    @pytest.mark.asyncio
    async def test_cancel_skips_protected_orders(self) -> None:
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager
        from src.execution.trading_client_port import OrderSnapshot

        trader = _make_minimal_trader()
        trader._protected_reduce_only_order_ids = {"sidecar-tp"}
        fake = FakeTradingClient()
        fake.open_orders = [
            OrderSnapshot(
                order_id="core-old",
                client_order_id=None,
                side="sell",
                qty=Decimal("1"),
                reduce_only=True,
                raw={},
            ),
            OrderSnapshot(
                order_id="sidecar-tp",
                client_order_id=None,
                side="sell",
                qty=Decimal("1"),
                reduce_only=True,
                raw={},
            ),
        ]

        manager = TpSlExecutionManager.__new__(TpSlExecutionManager)
        manager.trader = trader
        manager.trading_client = fake
        manager.protective_stops = None
        manager.core_tp = None
        manager.market_exit = None
        manager.near_tp = None
        manager.sidecar = None

        await manager.cancel_existing_reduce_only_orders()

        cancelled_ids = [c["order_id"] for c in fake.cancel_calls]
        assert "core-old" in cancelled_ids
        assert "sidecar-tp" not in cancelled_ids

    @pytest.mark.asyncio
    async def test_cancel_exception_is_caught(self) -> None:
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager
        from src.execution.trading_client_port import OrderSnapshot

        trader = _make_minimal_trader()
        fake = FakeTradingClient()
        fake.open_orders = [
            OrderSnapshot(
                order_id="will-fail",
                client_order_id=None,
                side="sell",
                qty=Decimal("1"),
                reduce_only=True,
                raw={},
            ),
        ]

        async def failing_cancel(*, order_id=None, client_order_id=None):
            raise RuntimeError("cancel failed")

        fake.cancel_order = failing_cancel

        manager = TpSlExecutionManager.__new__(TpSlExecutionManager)
        manager.trader = trader
        manager.trading_client = fake
        manager.protective_stops = None
        manager.core_tp = None
        manager.market_exit = None
        manager.near_tp = None
        manager.sidecar = None

        # Must not raise — exception is caught and logged
        await manager.cancel_existing_reduce_only_orders()


# ===================================================================
# 4. No direct REST endpoint strings in replaced code paths
# ===================================================================


class TestNoDirectRestEndpointsInReplacedPaths:
    """The replaced code paths must not contain direct REST endpoint strings."""

    def test_no_direct_order_endpoint_in_place_tp_orders(self) -> None:
        """CoreTakeProfitManager._place_reduce_only_take_profit_orders
        no longer contains /api/v5/trade/order (the non-semantic branch)."""
        from pathlib import Path

        text = Path(
            "src/execution/tp_sl_core_tp_manager.py"
        ).read_text(encoding="utf-8")

        # The legacy semantic-free branch must route through trading_client,
        # NOT directly call /api/v5/trade/order.
        # The semantic path is unchanged and uses broker_semantic_executor.
        # We verify the non-semantic branch no longer has the direct endpoint.
        lines = text.splitlines()
        in_place_method = False
        direct_order_found = False
        for line in lines:
            if "def _place_reduce_only_take_profit_orders" in line:
                in_place_method = True
            elif in_place_method and line.startswith("    def "):
                in_place_method = False
            if in_place_method and '"/api/v5/trade/order"' in line:
                direct_order_found = True
        assert not direct_order_found, (
            "_place_reduce_only_take_profit_orders must not contain "
            "direct /api/v5/trade/order — route through trading_client"
        )

    def test_no_direct_order_algo_endpoint_in_place_sl_primary(self) -> None:
        """ProtectiveStopManager.place_near_tp_protective_stop_with_retries
        primary path no longer contains /api/v5/trade/order-algo."""
        from pathlib import Path

        text = Path(
            "src/execution/tp_sl_protective_stop_manager.py"
        ).read_text(encoding="utf-8")

        # The primary (non-semantic) branch must route through trading_client.
        # The fallback path intentionally keeps the legacy call.
        # We check that there is NO /api/v5/trade/order-algo in the primary
        # branch code (before the fallback section).
        lines = text.splitlines()
        in_method = False
        fallback_section = False
        primary_has_direct = False
        for line in lines:
            if "def place_near_tp_protective_stop_with_retries" in line:
                in_method = True
                continue
            if in_method and "fallback" in line.lower() and "conditional" in line.lower():
                fallback_section = True
            if in_method and line.startswith("    def "):
                break
            if in_method and not fallback_section and '"/api/v5/trade/order-algo"' in line:
                primary_has_direct = True
        assert not primary_has_direct, (
            "primary path must not contain direct /api/v5/trade/order-algo "
            "— route through trading_client"
        )

    def test_no_direct_cancel_order_endpoint_in_cancel_reduce_only(self) -> None:
        """TpSlExecutionManager.cancel_existing_reduce_only_orders non-semantic
        branch no longer contains /api/v5/trade/cancel-order."""
        from pathlib import Path

        text = Path(
            "src/execution/tp_sl_execution_manager.py"
        ).read_text(encoding="utf-8")

        lines = text.splitlines()
        in_method = False
        direct_cancel = False
        for line in lines:
            if "def cancel_existing_reduce_only_orders" in line:
                in_method = True
            elif in_method and line.startswith("    def "):
                in_method = False
            if in_method and '"/api/v5/trade/cancel-order"' in line:
                direct_cancel = True
        assert not direct_cancel, (
            "cancel_existing_reduce_only_orders must not contain "
            "direct /api/v5/trade/cancel-order — route through trading_client"
        )

    def test_algo_cancel_routed_through_trading_client_port(self) -> None:
        """cancel_near_tp_protective_stop now routes through
        TradingClientPort.cancel_algo_order()."""
        from pathlib import Path

        text = Path(
            "src/execution/tp_sl_execution_manager.py"
        ).read_text(encoding="utf-8")

        assert "self.trading_client.cancel_algo_order(" in text
        assert "/api/v5/trade/cancel-algos" not in text


# ===================================================================
# 5. Fallback protective SL path → place_stop_market_order()
# ===================================================================


class TestProtectiveSlFallbackRoutesToPort:
    """When primary retries are exhausted, the fallback loop also routes
    through trading_client.place_stop_market_order()."""

    @pytest.mark.asyncio
    async def test_fallback_routes_to_port(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        trader = _make_minimal_trader()
        trader.decimal_to_str = lambda v: str(v)
        trader.price_to_str = lambda p: f"{float(p):.2f}"

        fake = FakeTradingClient()
        fake._next_stop_result = FakeOrderResult(order_id="sl-fallback-1")

        manager = ProtectiveStopManager(trader, trading_client=fake)

        verify_count = [0]

        async def fake_verify(_algo_id, _side, _contracts, _stop_price):
            verify_count[0] += 1
            # Primary retries (3 attempts) all fail, fallback succeeds
            return verify_count[0] > 3

        async def fake_cancel_algo(_algo_id, *, phase=""):
            return None

        trader.verify_near_tp_protective_stop = fake_verify
        trader._cancel_unverified_near_tp_algo = fake_cancel_algo
        trader.cancel_near_tp_protective_stop = lambda _: True

        ok, algo_id, message = await manager.place_near_tp_protective_stop_with_retries(
            side="LONG",
            contracts=Decimal("10"),
            stop_price=2950.0,
            retry_count=3,
            retry_interval_seconds=0,
        )

        assert ok is True
        assert algo_id == "sl-fallback-1"
        assert message == "fallback_conditional_close_placed"
        # 3 primary attempts + 1 successful fallback = 4 total
        assert len(fake.place_stop_calls) == 4
        # All calls have the same parameters
        for call in fake.place_stop_calls:
            assert call["side"] == "LONG"
            assert call["qty"] == Decimal("10")
            assert call["trigger_price"] == Decimal("2950.0")
            assert call["reduce_only"] is True
            assert call["client_order_id"] == ""

    @pytest.mark.asyncio
    async def test_fallback_exhausted_returns_false(self) -> None:
        """When both primary and fallback are exhausted, returns False."""
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        trader = _make_minimal_trader()
        trader.decimal_to_str = lambda v: str(v)
        trader.price_to_str = lambda p: f"{float(p):.2f}"

        fake = FakeTradingClient()
        fake._next_stop_result = FakeOrderResult(order_id="sl-exhaust-1")

        manager = ProtectiveStopManager(trader, trading_client=fake)

        async def fake_verify(_algo_id, _side, _contracts, _stop_price):
            return False  # never verifies

        async def fake_cancel_algo(_algo_id, *, phase=""):
            return None

        trader.verify_near_tp_protective_stop = fake_verify
        trader._cancel_unverified_near_tp_algo = fake_cancel_algo
        trader.cancel_near_tp_protective_stop = lambda _: True

        ok, algo_id, message = await manager.place_near_tp_protective_stop_with_retries(
            side="SHORT",
            contracts=Decimal("5"),
            stop_price=3700.0,
            retry_count=2,
            retry_interval_seconds=0,
        )

        assert ok is False
        assert algo_id is None
        # 2 primary + 2 fallback = 4 total calls
        assert len(fake.place_stop_calls) == 4
        for call in fake.place_stop_calls:
            assert call["side"] == "SHORT"
            assert call["reduce_only"] is True


# ===================================================================
# 6. Delegating protective SL methods → place_stop_market_order()
# ===================================================================


class TestDelegatingProtectiveSlMethodsRouteToPort:
    """place_middle_runner / place_middle_bucket_fast / place_trend_runner /
    place_three_stage_post_tp1 all delegate to
    place_near_tp_protective_stop_with_retries and therefore route
    through trading_client.place_stop_market_order()."""

    @pytest.mark.asyncio
    async def test_middle_runner_routes_to_port(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        trader = _make_minimal_trader()
        trader.decimal_to_str = lambda v: str(v)
        trader.price_to_str = lambda p: f"{float(p):.2f}"

        fake = FakeTradingClient()
        fake._next_stop_result = FakeOrderResult(order_id="sl-middle-runner-1")

        manager = ProtectiveStopManager(trader, trading_client=fake)

        async def fake_verify(_algo_id, _side, _contracts, _stop_price):
            return True

        trader.verify_near_tp_protective_stop = fake_verify

        ok, algo_id, message = await manager.place_middle_runner_protective_stop_with_retries(
            side="LONG",
            contracts=Decimal("8"),
            stop_price=2900.0,
            retry_count=1,
            retry_interval_seconds=0,
        )

        assert ok is True
        assert algo_id == "sl-middle-runner-1"
        assert message == "protective_sl_placed"
        assert trader.middle_runner_protective_sl_order_id == "sl-middle-runner-1"
        assert len(fake.place_stop_calls) == 1
        call = fake.place_stop_calls[0]
        assert call["side"] == "LONG"
        assert call["qty"] == Decimal("8")
        assert call["trigger_price"] == Decimal("2900.0")
        assert call["reduce_only"] is True
        assert call["client_order_id"] == ""

    @pytest.mark.asyncio
    async def test_middle_bucket_fast_routes_to_port(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        trader = _make_minimal_trader()
        trader.decimal_to_str = lambda v: str(v)
        trader.price_to_str = lambda p: f"{float(p):.2f}"

        fake = FakeTradingClient()
        fake._next_stop_result = FakeOrderResult(order_id="sl-bucket-fast-1")

        manager = ProtectiveStopManager(trader, trading_client=fake)

        async def fake_verify(_algo_id, _side, _contracts, _stop_price):
            return True

        trader.verify_near_tp_protective_stop = fake_verify

        ok, algo_id, message = await manager.place_middle_bucket_fast_protective_stop_with_retries(
            side="SHORT",
            contracts=Decimal("6"),
            stop_price=3750.0,
            retry_count=1,
            retry_interval_seconds=0,
        )

        assert ok is True
        assert algo_id == "sl-bucket-fast-1"
        assert message == "protective_sl_placed"
        assert trader.middle_bucket_fast_sl_order_id == "sl-bucket-fast-1"
        assert len(fake.place_stop_calls) == 1
        call = fake.place_stop_calls[0]
        assert call["side"] == "SHORT"
        assert call["qty"] == Decimal("6")
        assert call["trigger_price"] == Decimal("3750.0")
        assert call["reduce_only"] is True

    @pytest.mark.asyncio
    async def test_trend_runner_routes_to_port(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        trader = _make_minimal_trader()
        trader.decimal_to_str = lambda v: str(v)
        trader.price_to_str = lambda p: f"{float(p):.2f}"

        fake = FakeTradingClient()
        fake._next_stop_result = FakeOrderResult(order_id="sl-trend-1")

        manager = ProtectiveStopManager(trader, trading_client=fake)

        async def fake_verify(_algo_id, _side, _contracts, _stop_price):
            return True

        trader.verify_near_tp_protective_stop = fake_verify

        ok, algo_id, message = await manager.place_trend_runner_protective_stop_with_retries(
            side="LONG",
            contracts=Decimal("12"),
            stop_price=2850.0,
            retry_count=1,
            retry_interval_seconds=0,
        )

        assert ok is True
        assert algo_id == "sl-trend-1"
        assert message == "protective_sl_placed"
        assert trader.trend_runner_sl_order_id == "sl-trend-1"
        assert len(fake.place_stop_calls) == 1
        call = fake.place_stop_calls[0]
        assert call["side"] == "LONG"
        assert call["qty"] == Decimal("12")
        assert call["trigger_price"] == Decimal("2850.0")
        assert call["reduce_only"] is True

    @pytest.mark.asyncio
    async def test_three_stage_post_tp1_routes_to_port(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        trader = _make_minimal_trader()
        trader.decimal_to_str = lambda v: str(v)
        trader.price_to_str = lambda p: f"{float(p):.2f}"

        fake = FakeTradingClient()
        fake._next_stop_result = FakeOrderResult(order_id="sl-three-stage-1")

        manager = ProtectiveStopManager(trader, trading_client=fake)

        async def fake_verify(_algo_id, _side, _contracts, _stop_price):
            return True

        trader.verify_near_tp_protective_stop = fake_verify

        ok, algo_id, message = await manager.place_three_stage_post_tp1_protective_stop_with_retries(
            side="SHORT",
            contracts=Decimal("4"),
            stop_price=3800.0,
            retry_count=1,
            retry_interval_seconds=0,
        )

        assert ok is True
        assert algo_id == "sl-three-stage-1"
        assert message == "protective_sl_placed"
        assert trader.three_stage_post_tp1_protective_sl_order_id == "sl-three-stage-1"
        assert len(fake.place_stop_calls) == 1
        call = fake.place_stop_calls[0]
        assert call["side"] == "SHORT"
        assert call["qty"] == Decimal("4")
        assert call["trigger_price"] == Decimal("3800.0")
        assert call["reduce_only"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
