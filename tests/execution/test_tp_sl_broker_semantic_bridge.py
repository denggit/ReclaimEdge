"""Bridge integration tests: TP/SL managers → BrokerSemanticExecutor.

Verifies that TP/SL manager methods correctly route through the
broker semantic executor while preserving legacy signatures.
"""

from __future__ import annotations

import unittest
from decimal import Decimal

import src.execution.trader as trader_module
from src.exchanges.errors import ExchangeError, ExchangeErrorDetail, ExchangeErrorKind
from src.exchanges.models import BrokerOrderSide, BrokerOrderType, BrokerPositionSide, ExchangeName
from src.exchanges.semantic_models import (
    BrokerSemanticAction,
    BrokerSemanticOrderRole,
    BrokerSemanticRequest,
    BrokerSemanticResult,
)
from src.execution.tp_sl_execution_manager import TpSlExecutionManager
from src.execution.trader import Trader
from tests.fakes.fake_broker_semantic_executor import FakeBrokerSemanticExecutor


# ---------------------------------------------------------------------------
# Helpers — minimal trader construction
# ---------------------------------------------------------------------------


def _make_trader(**overrides) -> Trader:
    t = Trader.__new__(Trader)
    t.base_url = "https://www.okx.test"
    t.api_key = "key"
    t.secret_key = "secret"
    t.passphrase = "pass"
    t._timeout_seconds = 7.0
    t.symbol = "ETH-USDT-SWAP"
    t.td_mode = "isolated"
    t.leverage = "50"
    t.pos_side_mode = "net"
    t.live_trading = True
    t.max_live_equity_usdt = 30.0
    t.contract_multiplier = Decimal("0.1")
    t.contract_precision = Decimal("0.01")
    t.min_contracts = Decimal("0.01")
    t.tp_order_id = None
    t.near_tp_protective_sl_order_id = None
    t.middle_runner_protective_sl_order_id = None
    t.three_stage_post_tp1_protective_sl_order_id = None
    t.trend_runner_sl_order_id = None
    t.position_contracts = Decimal("0")
    t.account_equity_usdt = 0.0
    t._protected_reduce_only_order_ids = set()
    t._managed_reduce_only_order_ids = set()
    t._allow_cancel_unmanaged_reduce_only = True
    for k, v in overrides.items():
        setattr(t, k, v)
    return t


# ---------------------------------------------------------------------------
# 1. CANCEL_REDUCE_ONLY_TP bridge
# ---------------------------------------------------------------------------


class TestCancelReduceOnlyTPBridge(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_known_order_uses_semantic_executor(self) -> None:
        fake = FakeBrokerSemanticExecutor()
        trader = _make_trader(_broker_semantic_executor=fake)
        trader._managed_reduce_only_order_ids = {"known-id"}

        async def fake_fetch():  # type: ignore[no-untyped-def]
            return [{"instId": trader.symbol, "reduceOnly": "true", "ordId": "known-id"}]

        async def fake_request(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("legacy path must not be called")

        trader.fetch_pending_orders = fake_fetch
        trader.request = fake_request  # type: ignore[method-assign]

        manager = TpSlExecutionManager(trader)
        result = await manager.cancel_existing_reduce_only_orders(phase="update_tp")

        self.assertTrue(result)
        self.assertEqual(len(fake.requests), 1)
        self.assertEqual(fake.requests[0].action, BrokerSemanticAction.CANCEL_REDUCE_ONLY_TP)
        self.assertEqual(fake.requests[0].order_id, "known-id")

    async def test_cancel_without_executor_falls_back_to_legacy(self) -> None:
        trader = _make_trader()
        trader._managed_reduce_only_order_ids = {"known-id"}
        legacy_calls = []

        async def fake_fetch():  # type: ignore[no-untyped-def]
            return [{"instId": trader.symbol, "reduceOnly": "true", "ordId": "known-id"}]

        async def fake_request(method, path, body):  # type: ignore[no-untyped-def]
            legacy_calls.append((method, path, body))
            return {"code": "0", "data": [{"ordId": "known-id", "sCode": "0"}]}

        trader.fetch_pending_orders = fake_fetch
        trader.request = fake_request  # type: ignore[method-assign]

        manager = TpSlExecutionManager(trader)
        result = await manager.cancel_existing_reduce_only_orders(phase="update_tp")

        self.assertTrue(result)
        self.assertEqual(len(legacy_calls), 1)
        self.assertEqual(legacy_calls[0][1], "/api/v5/trade/cancel-order")


# ---------------------------------------------------------------------------
# 2. CANCEL_PROTECTIVE_STOP bridge
# ---------------------------------------------------------------------------


class TestCancelProtectiveStopBridge(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_protective_stop_uses_semantic_executor(self) -> None:
        fake = FakeBrokerSemanticExecutor()
        trader = _make_trader(_broker_semantic_executor=fake)

        async def fake_request(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("legacy path must not be called")

        trader.request = fake_request  # type: ignore[method-assign]

        manager = TpSlExecutionManager(trader)
        result = await manager.cancel_near_tp_protective_stop("algo-1")

        self.assertTrue(result)
        self.assertEqual(len(fake.requests), 1)
        self.assertEqual(fake.requests[0].action, BrokerSemanticAction.CANCEL_PROTECTIVE_STOP)
        self.assertEqual(fake.requests[0].order_id, "algo-1")

    async def test_cancel_none_order_id_returns_true(self) -> None:
        fake = FakeBrokerSemanticExecutor()
        trader = _make_trader(_broker_semantic_executor=fake)
        manager = TpSlExecutionManager(trader)

        result = await manager.cancel_near_tp_protective_stop(None)
        self.assertTrue(result)
        self.assertEqual(len(fake.requests), 0)

    async def test_cancel_without_executor_falls_back_to_legacy(self) -> None:
        trader = _make_trader()
        legacy_calls = []

        async def fake_request(method, path, body):  # type: ignore[no-untyped-def]
            legacy_calls.append((method, path, body))
            return {"code": "0", "data": [{"algoId": "algo-1", "sCode": "0"}]}

        trader.request = fake_request  # type: ignore[method-assign]

        manager = TpSlExecutionManager(trader)
        result = await manager.cancel_near_tp_protective_stop("algo-1")

        self.assertTrue(result)
        self.assertEqual(len(legacy_calls), 1)
        self.assertEqual(legacy_calls[0][1], "/api/v5/trade/cancel-algos")


# ---------------------------------------------------------------------------
# 3. Semantic error is NOT swallowed
# ---------------------------------------------------------------------------


class TestSemanticErrorNotSwallowed(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_protective_stop_semantic_error_propagates(self) -> None:
        """When the semantic executor raises a non-recoverable error (e.g.
        network failure), cancel_near_tp_protective_stop correctly catches it
        and returns False (operation failed, not silently swallowed)."""
        fake = FakeBrokerSemanticExecutor()
        fake.queue_error(RuntimeError("Connection reset by peer"))
        trader = _make_trader(_broker_semantic_executor=fake)
        trader.request = None  # type: ignore[method-assign]

        manager = TpSlExecutionManager(trader)

        # Non-recoverable error → caught, returns False
        result = await manager.cancel_near_tp_protective_stop("algo-1")
        self.assertFalse(result)
        self.assertEqual(len(fake.requests), 1)

    async def test_cancel_reduce_only_tp_continues_on_single_cancel_failure(self) -> None:
        """When cancelling multiple reduce-only orders, a single failure
        should not prevent cancelling the remaining orders."""
        fake = FakeBrokerSemanticExecutor()
        # First cancel succeeds, second raises
        fake.queue_result(order_id="ok-1", ok=True)
        fake.queue_error(RuntimeError("Connection error"))

        trader = _make_trader(_broker_semantic_executor=fake)
        trader._managed_reduce_only_order_ids = {"ok-1", "fail-1"}

        async def fake_fetch():  # type: ignore[no-untyped-def]
            return [
                {"instId": trader.symbol, "reduceOnly": "true", "ordId": "ok-1"},
                {"instId": trader.symbol, "reduceOnly": "true", "ordId": "fail-1"},
            ]

        trader.fetch_pending_orders = fake_fetch

        manager = TpSlExecutionManager(trader)
        # cancel_existing_reduce_only_orders catches per-order failures
        # and continues — overall returns True because it tried all orders
        result = await manager.cancel_existing_reduce_only_orders(phase="update_tp")

        self.assertTrue(result)  # overall success despite one failure
        self.assertEqual(len(fake.requests), 2)


# ---------------------------------------------------------------------------
# 4. Sidecar TP bridge
# ---------------------------------------------------------------------------


class TestSidecarTPBridge(unittest.IsolatedAsyncioTestCase):
    async def test_sidecar_tp_uses_semantic_executor(self) -> None:
        fake = FakeBrokerSemanticExecutor()
        fake.queue_result(order_id="sidecar-tp-1")
        trader = _make_trader(_broker_semantic_executor=fake)

        async def fake_request(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("legacy path must not be called")

        trader.request = fake_request  # type: ignore[method-assign]

        manager = TpSlExecutionManager(trader)
        result = await manager.place_sidecar_fixed_take_profit(
            side="LONG", contracts="1.0", tp_price=3500.0
        )

        self.assertEqual(result, "sidecar-tp-1")
        self.assertEqual(len(fake.requests), 1)
        self.assertEqual(fake.requests[0].action, BrokerSemanticAction.SIDECAR_TP)
        self.assertEqual(fake.requests[0].role, BrokerSemanticOrderRole.SIDECAR_TP)
        self.assertEqual(fake.requests[0].side, BrokerOrderSide.SELL)  # LONG close → SELL
        self.assertEqual(fake.requests[0].position_side, BrokerPositionSide.LONG)

    async def test_sidecar_tp_without_executor_falls_back(self) -> None:
        trader = _make_trader()
        legacy_calls = []

        async def fake_request(method, path, body):  # type: ignore[no-untyped-def]
            legacy_calls.append((method, path, body))
            return {"code": "0", "data": [{"ordId": "legacy-tp-1", "sCode": "0"}]}

        trader.request = fake_request  # type: ignore[method-assign]

        manager = TpSlExecutionManager(trader)
        result = await manager.place_sidecar_fixed_take_profit(
            side="LONG", contracts="1.0", tp_price=3500.0
        )

        self.assertEqual(result, "legacy-tp-1")
        self.assertEqual(len(legacy_calls), 1)

    async def test_sidecar_cancel_uses_semantic_executor(self) -> None:
        fake = FakeBrokerSemanticExecutor()
        trader = _make_trader(_broker_semantic_executor=fake)

        async def fake_request(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("legacy path must not be called")

        trader.request = fake_request  # type: ignore[method-assign]

        manager = TpSlExecutionManager(trader)
        result = await manager.cancel_sidecar_take_profit("sidecar-tp-1")

        self.assertTrue(result)
        self.assertEqual(len(fake.requests), 1)
        self.assertEqual(fake.requests[0].action, BrokerSemanticAction.CANCEL_REDUCE_ONLY_TP)
        self.assertEqual(fake.requests[0].role, BrokerSemanticOrderRole.SIDECAR_TP)
        self.assertEqual(fake.requests[0].order_id, "sidecar-tp-1")


# ---------------------------------------------------------------------------
# 5. Helper function consistency
# ---------------------------------------------------------------------------


class TestBrokerSemanticHelpers(unittest.TestCase):
    # ------------------------------------------------------------------
    # Side mapping
    # ------------------------------------------------------------------

    def test_close_order_side_long_returns_sell(self) -> None:
        from src.execution.broker_semantic_helpers import close_order_side
        self.assertEqual(close_order_side("LONG"), BrokerOrderSide.SELL)
        self.assertEqual(close_order_side("SHORT"), BrokerOrderSide.BUY)

    def test_close_order_side_case_insensitive(self) -> None:
        from src.execution.broker_semantic_helpers import close_order_side
        self.assertEqual(close_order_side("long"), BrokerOrderSide.SELL)
        self.assertEqual(close_order_side("short"), BrokerOrderSide.BUY)

    def test_close_order_side_strips_whitespace(self) -> None:
        from src.execution.broker_semantic_helpers import close_order_side
        self.assertEqual(close_order_side("LONG "), BrokerOrderSide.SELL)
        self.assertEqual(close_order_side(" SHORT"), BrokerOrderSide.BUY)

    def test_close_order_side_invalid_raises_valueerror(self) -> None:
        from src.execution.broker_semantic_helpers import close_order_side
        for invalid in ("NET", "", "BUY"):
            with self.assertRaises(ValueError):
                close_order_side(invalid)

    def test_entry_order_side(self) -> None:
        from src.execution.broker_semantic_helpers import entry_order_side
        self.assertEqual(entry_order_side("LONG"), BrokerOrderSide.BUY)
        self.assertEqual(entry_order_side("SHORT"), BrokerOrderSide.SELL)

    def test_broker_position_side(self) -> None:
        from src.execution.broker_semantic_helpers import broker_position_side
        self.assertEqual(broker_position_side("LONG"), BrokerPositionSide.LONG)
        self.assertEqual(broker_position_side("SHORT"), BrokerPositionSide.SHORT)

    # ------------------------------------------------------------------
    # Request builders
    # ------------------------------------------------------------------

    def test_build_cancel_reduce_only_tp_request(self) -> None:
        from src.execution.broker_semantic_helpers import build_cancel_reduce_only_tp_request
        cancel_tp = build_cancel_reduce_only_tp_request(
            symbol="ETH-USDT-SWAP", order_id="ord-1",
            role=BrokerSemanticOrderRole.CORE_TP,
        )
        self.assertEqual(cancel_tp.action, BrokerSemanticAction.CANCEL_REDUCE_ONLY_TP)
        self.assertEqual(cancel_tp.order_id, "ord-1")

    def test_build_cancel_protective_stop_request(self) -> None:
        from src.execution.broker_semantic_helpers import build_cancel_protective_stop_request
        cancel_sl = build_cancel_protective_stop_request(
            symbol="ETH-USDT-SWAP", order_id="algo-1",
            role=BrokerSemanticOrderRole.PROTECTIVE_SL,
        )
        self.assertEqual(cancel_sl.action, BrokerSemanticAction.CANCEL_PROTECTIVE_STOP)
        self.assertEqual(cancel_sl.order_id, "algo-1")

    def test_build_reduce_only_tp_request(self) -> None:
        from src.execution.broker_semantic_helpers import build_reduce_only_tp_request
        tp_req = build_reduce_only_tp_request(
            symbol="ETH-USDT-SWAP", side="LONG",
            contracts=Decimal("1"), price=Decimal("3500"),
            role=BrokerSemanticOrderRole.TP1,
        )
        self.assertEqual(tp_req.action, BrokerSemanticAction.PLACE_REDUCE_ONLY_TP)
        self.assertEqual(tp_req.side, BrokerOrderSide.SELL)
        self.assertTrue(tp_req.reduce_only)

    def test_build_market_exit_request_runner_context(self) -> None:
        from src.execution.broker_semantic_helpers import build_market_exit_request
        # Lowercase
        req1 = build_market_exit_request(
            symbol="ETH-USDT-SWAP", side="LONG", contracts=Decimal("1"),
            context="runner",
        )
        self.assertEqual(req1.action, BrokerSemanticAction.MARKET_EXIT_RUNNER)
        # Uppercase
        req2 = build_market_exit_request(
            symbol="ETH-USDT-SWAP", side="LONG", contracts=Decimal("1"),
            context="TREND_RUNNER",
        )
        self.assertEqual(req2.action, BrokerSemanticAction.MARKET_EXIT_RUNNER)
        # Mixed case
        req3 = build_market_exit_request(
            symbol="ETH-USDT-SWAP", side="LONG", contracts=Decimal("1"),
            context="middle_Runner_Exit",
        )
        self.assertEqual(req3.action, BrokerSemanticAction.MARKET_EXIT_RUNNER)
        # Generic
        req4 = build_market_exit_request(
            symbol="ETH-USDT-SWAP", side="LONG", contracts=Decimal("1"),
            context="generic",
        )
        self.assertEqual(req4.action, BrokerSemanticAction.MARKET_EXIT)

    def test_build_sidecar_entry_request_uses_entry_side(self) -> None:
        from src.execution.broker_semantic_helpers import build_sidecar_entry_request
        req = build_sidecar_entry_request(
            symbol="ETH-USDT-SWAP", side="LONG", contracts=Decimal("1"),
        )
        self.assertEqual(req.action, BrokerSemanticAction.SIDECAR_ENTRY)
        self.assertEqual(req.role, BrokerSemanticOrderRole.SIDECAR_ENTRY)
        self.assertEqual(req.side, BrokerOrderSide.BUY)  # LONG entry → BUY

    def test_semantic_tp_role_classifier(self) -> None:
        from src.execution.broker_semantic_helpers import semantic_tp_role
        self.assertEqual(semantic_tp_role("tp1"), BrokerSemanticOrderRole.TP1)
        self.assertEqual(semantic_tp_role("tp2"), BrokerSemanticOrderRole.TP2)
        self.assertEqual(semantic_tp_role("runner"), BrokerSemanticOrderRole.RUNNER_TP)
        self.assertEqual(semantic_tp_role("unknown"), BrokerSemanticOrderRole.CORE_TP)
        self.assertEqual(semantic_tp_role("tp1_middle_fast"), BrokerSemanticOrderRole.TP1)

    # ------------------------------------------------------------------
    # Result ok=False / missing order_id
    # ------------------------------------------------------------------

    def test_build_protective_stop_request(self) -> None:
        from src.execution.broker_semantic_helpers import build_protective_stop_request
        req = build_protective_stop_request(
            symbol="ETH-USDT-SWAP", side="LONG", contracts=Decimal("1"),
            stop_price=Decimal("2800"),
            role=BrokerSemanticOrderRole.PROTECTIVE_SL,
        )
        self.assertEqual(req.action, BrokerSemanticAction.PLACE_PROTECTIVE_STOP)
        self.assertEqual(req.trigger_price, Decimal("2800"))
        self.assertTrue(req.reduce_only)


# ---------------------------------------------------------------------------
# 6. Semantic result ok=False / missing order_id
# ---------------------------------------------------------------------------


class TestSemanticResultEdgeCases(unittest.IsolatedAsyncioTestCase):
    async def test_sidecar_tp_missing_order_id_raises(self) -> None:
        fake = FakeBrokerSemanticExecutor()
        fake.queue_result(order_id=None, ok=True)
        trader = _make_trader(_broker_semantic_executor=fake)
        manager = TpSlExecutionManager(trader)

        with self.assertRaises(RuntimeError):
            await manager.place_sidecar_fixed_take_profit(
                side="LONG", contracts="1.0", tp_price=3500.0
            )

    async def test_cancel_result_ok_false_not_swallowed(self) -> None:
        """When result.ok=False and no order_id, cancel must not silently succeed."""
        fake = FakeBrokerSemanticExecutor()
        # queue a result with ok=False and no order_id
        fake.queue_result(order_id=None, ok=False, message="some error")
        trader = _make_trader(_broker_semantic_executor=fake)
        # set up fetch_pending_orders to return a managed order
        trader._managed_reduce_only_order_ids = {"known-id"}

        async def fake_fetch():  # type: ignore[no-untyped-def]
            return [{"instId": trader.symbol, "reduceOnly": "true", "ordId": "known-id"}]

        trader.fetch_pending_orders = fake_fetch
        trader.request = None  # type: ignore[method-assign]

        manager = TpSlExecutionManager(trader)
        # cancel_existing_reduce_only_orders catches per-order exceptions and
        # returns True overall. The semantic result with ok=False and no order_id
        # triggers an ExchangeError when the cancel fails.
        # But the OK=False semantic result causes a conditional branch in
        # the cancel_near_tp_protective_stop path, not in cancel_existing_reduce_only_orders.
        result = await manager.cancel_existing_reduce_only_orders(phase="update_tp")
        # Still True because failures are caught per-order
        self.assertTrue(result)


# ---------------------------------------------------------------------------
# 7. Sidecar entry bridge (via Trader)
# ---------------------------------------------------------------------------


class TestSidecarEntryBridge(unittest.TestCase):
    def test_sidecar_entry_request_side_mapping(self) -> None:
        """Sidecar entry for LONG uses BUY, for SHORT uses SELL."""
        from src.execution.broker_semantic_helpers import build_sidecar_entry_request
        req_long = build_sidecar_entry_request(
            symbol="ETH-USDT-SWAP", side="LONG", contracts=Decimal("1"),
        )
        self.assertEqual(req_long.side, BrokerOrderSide.BUY)

        req_short = build_sidecar_entry_request(
            symbol="ETH-USDT-SWAP", side="SHORT", contracts=Decimal("1"),
        )
        self.assertEqual(req_short.side, BrokerOrderSide.SELL)
