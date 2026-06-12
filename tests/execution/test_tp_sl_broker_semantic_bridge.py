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

    async def test_cancel_reduce_only_loop_catches_ok_false_per_order(self) -> None:
        """Per-order cancel with ok=False is caught by the loop; overall returns True."""
        fake = FakeBrokerSemanticExecutor()
        # queue a result with ok=False and no order_id — require_semantic_ok
        # raises RuntimeError, which is caught per-order by the cancel loop
        fake.queue_result(order_id=None, ok=False, message="some error")
        trader = _make_trader(_broker_semantic_executor=fake)
        trader._managed_reduce_only_order_ids = {"known-id"}

        async def fake_fetch():  # type: ignore[no-untyped-def]
            return [{"instId": trader.symbol, "reduceOnly": "true", "ordId": "known-id"}]

        trader.fetch_pending_orders = fake_fetch
        trader.request = None  # type: ignore[method-assign]

        manager = TpSlExecutionManager(trader)
        # Per-order RuntimeError from require_semantic_ok is caught by the
        # try/except around each order; the loop continues and returns True.
        result = await manager.cancel_existing_reduce_only_orders(phase="update_tp")
        self.assertTrue(result)
        # The single known-id order was attempted (its failure was caught)
        self.assertEqual(len(fake.requests), 1)


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


# ---------------------------------------------------------------------------
# 8. Trader broker_semantic_executor injection
# ---------------------------------------------------------------------------


class TestTraderBrokerSemanticExecutorInjection(unittest.TestCase):
    def test_fake_executor_via_private_attr(self) -> None:
        """Fake injected via _broker_semantic_executor is retrievable."""
        from src.execution.broker_semantic_helpers import get_broker_semantic_executor

        fake = FakeBrokerSemanticExecutor()
        t = _make_trader(_broker_semantic_executor=fake)

        executor = get_broker_semantic_executor(t)
        self.assertIs(executor, fake)

    def test_get_broker_semantic_executor_lazy_creates_executor(self) -> None:
        """When no _broker_semantic_executor is set, Trader.__getattr__ creates it lazily."""
        from src.execution.broker_semantic_helpers import get_broker_semantic_executor

        t = _make_trader()
        # Remove any pre-existing executor to test lazy creation
        if hasattr(t, '_broker_semantic_executor'):
            del t._broker_semantic_executor
        if hasattr(t, '_broker_client'):
            del t._broker_client

        # Accessing via get_broker_semantic_executor triggers __getattr__
        # which lazily constructs OkxBrokerSemanticExecutor
        executor = get_broker_semantic_executor(t)
        self.assertIsNotNone(executor)


# ---------------------------------------------------------------------------
# 9. Sidecar market entry bridge (async)
# ---------------------------------------------------------------------------


class TestSidecarMarketEntryBridge(unittest.IsolatedAsyncioTestCase):
    async def test_uses_sidcar_entry_and_legacy_not_called(self) -> None:
        """place_sidecar_market_order uses SIDECAR_ENTRY and skips legacy."""
        fake = FakeBrokerSemanticExecutor()
        fake.queue_result(order_id="sidecar-mkt-1", ok=True)
        t = _make_trader(_broker_semantic_executor=fake)
        t.td_mode = "cross"
        t.pos_side_mode = "net"
        t.contract_multiplier = Decimal("0.1")

        async def fake_request(*args, **kwargs):
            raise AssertionError("legacy path must not be called")

        t.request = fake_request  # type: ignore[method-assign]

        result = await t.place_sidecar_market_order(side="LONG", eth_qty=0.1)
        self.assertEqual(result["order_id"], "sidecar-mkt-1")
        self.assertIn("contracts", result)
        self.assertIn("qty", result)
        self.assertEqual(len(fake.requests), 1)
        self.assertEqual(fake.requests[0].action, BrokerSemanticAction.SIDECAR_ENTRY)
        self.assertEqual(fake.requests[0].role, BrokerSemanticOrderRole.SIDECAR_ENTRY)
        self.assertEqual(fake.requests[0].side, BrokerOrderSide.BUY)  # LONG entry

    async def test_missing_order_id_raises(self) -> None:
        """SIDECAR_ENTRY with no order_id must raise, not fake success."""
        fake = FakeBrokerSemanticExecutor()
        fake.queue_result(order_id=None, ok=True)
        t = _make_trader(_broker_semantic_executor=fake)
        t.td_mode = "cross"
        t.pos_side_mode = "net"
        t.contract_multiplier = Decimal("0.1")

        with self.assertRaises(RuntimeError):
            await t.place_sidecar_market_order(side="LONG", eth_qty=0.1)

    async def test_ok_false_raises(self) -> None:
        """SIDECAR_ENTRY with ok=False must raise, not fake success."""
        fake = FakeBrokerSemanticExecutor()
        fake.queue_result(order_id=None, ok=False, message="exchange error")
        t = _make_trader(_broker_semantic_executor=fake)
        t.td_mode = "cross"
        t.pos_side_mode = "net"
        t.contract_multiplier = Decimal("0.1")

        with self.assertRaises(RuntimeError):
            await t.place_sidecar_market_order(side="LONG", eth_qty=0.1)

    async def test_no_executor_falls_back_to_legacy(self) -> None:
        """When no executor, place_sidecar_market_order uses legacy path."""
        t = _make_trader()
        t.td_mode = "cross"
        t.pos_side_mode = "net"
        t.contract_multiplier = Decimal("0.1")
        if hasattr(t, '_broker_semantic_executor'):
            del t._broker_semantic_executor

        legacy_calls = []

        async def fake_request(method, path, body):
            legacy_calls.append((method, path, body))
            return {"code": "0", "data": [{"ordId": "legacy-mkt-1", "sCode": "0"}]}

        t.request = fake_request  # type: ignore[method-assign]

        result = await t.place_sidecar_market_order(side="SHORT", eth_qty=0.1)
        self.assertEqual(result["order_id"], "legacy-mkt-1")
        self.assertEqual(len(legacy_calls), 1)
        self.assertEqual(legacy_calls[0][1], "/api/v5/trade/order")


# ---------------------------------------------------------------------------
# 10. Placement path ok=False / missing order_id
# ---------------------------------------------------------------------------


class TestPlacementResultOkFalse(unittest.IsolatedAsyncioTestCase):
    """Verify that ok=False or missing order_id causes placement to fail."""

    async def test_core_tp_ok_false_raises(self) -> None:
        """PLACE_REDUCE_ONLY_TP with ok=False must raise."""
        fake = FakeBrokerSemanticExecutor()
        fake.queue_result(order_id=None, ok=False, message="tp error")
        t = _make_trader(_broker_semantic_executor=fake)
        from src.execution.tp_sl_core_tp_manager import CoreTakeProfitManager
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        mgr = CoreTakeProfitManager(t, protective_stops=ProtectiveStopManager(t))

        # Minimal intent mock with just the side attribute needed by the bridge
        class _FakeIntent:
            side = "LONG"

        with self.assertRaises(RuntimeError):
            await mgr._place_reduce_only_take_profit_orders(
                _FakeIntent(), [("tp1", Decimal("1"), 3500.0)]
            )

    async def test_protective_sl_ok_false_raises(self) -> None:
        """PLACE_PROTECTIVE_STOP with ok=False must raise."""
        fake = FakeBrokerSemanticExecutor()
        fake.queue_result(order_id=None, ok=False, message="sl error")
        t = _make_trader(_broker_semantic_executor=fake)
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager
        from src.exchanges.semantic_models import BrokerSemanticOrderRole

        mgr = ProtectiveStopManager(t)
        with self.assertRaises(RuntimeError):
            await mgr._place_protective_stop_semantic(
                side="LONG",
                contracts=Decimal("1"),
                stop_price=2800.0,
                role=BrokerSemanticOrderRole.PROTECTIVE_SL,
                metadata={"phase": "primary"},
            )


# ---------------------------------------------------------------------------
# 11. Cancel path ok=False is treated as failure
# ---------------------------------------------------------------------------


class TestCancelResultOkFalse(unittest.IsolatedAsyncioTestCase):
    """Verify that cancel paths treat ok=False as failure."""

    async def test_cancel_protective_stop_ok_false_returns_false(self) -> None:
        """CANCEL_PROTECTIVE_STOP with ok=False returns False (caught exception)."""
        fake = FakeBrokerSemanticExecutor()
        fake.queue_result(order_id=None, ok=False, message="cancel error")
        t = _make_trader(_broker_semantic_executor=fake)
        t.near_tp_protective_sl_order_id = "algo-1"
        t.request = None  # type: ignore[method-assign]

        from src.execution.tp_sl_execution_manager import TpSlExecutionManager
        mgr = TpSlExecutionManager(t)
        # cancel_near_tp_protective_stop catches exception → returns False
        result = await mgr.cancel_near_tp_protective_stop("algo-1")
        self.assertFalse(result)

    async def test_sidecar_cancel_ok_false_returns_false(self) -> None:
        """Sidecar cancel with ok=False returns False (caught exception)."""
        fake = FakeBrokerSemanticExecutor()
        fake.queue_result(order_id=None, ok=False, message="cancel error")
        t = _make_trader(_broker_semantic_executor=fake)
        t.request = None  # type: ignore[method-assign]

        from src.execution.tp_sl_execution_manager import TpSlExecutionManager
        mgr = TpSlExecutionManager(t)
        result = await mgr.cancel_sidecar_take_profit("sidecar-tp-1")
        self.assertFalse(result)

    async def test_cancel_reduce_only_loop_ok_false_still_continues(self) -> None:
        """When a per-order cancel returns ok=False, the loop catches it and continues."""
        fake = FakeBrokerSemanticExecutor()
        # First cancel: ok=False (caught, continues), second: ok=True
        fake.queue_result(order_id=None, ok=False, message="cancel error")
        fake.queue_result(order_id="ok-1", ok=True)
        t = _make_trader(_broker_semantic_executor=fake)
        t._managed_reduce_only_order_ids = {"bad-id", "ok-1"}

        async def fake_fetch():
            return [
                {"instId": t.symbol, "reduceOnly": "true", "ordId": "bad-id"},
                {"instId": t.symbol, "reduceOnly": "true", "ordId": "ok-1"},
            ]

        t.fetch_pending_orders = fake_fetch

        from src.execution.tp_sl_execution_manager import TpSlExecutionManager
        mgr = TpSlExecutionManager(t)
        result = await mgr.cancel_existing_reduce_only_orders(phase="update_tp")
        # Overall True because loop caught the first failure and continued
        self.assertTrue(result)
        self.assertEqual(len(fake.requests), 2)


# ---------------------------------------------------------------------------
# 12. require_semantic_order_id / require_semantic_ok unit tests
# ---------------------------------------------------------------------------


class TestRequireSemanticHelpers(unittest.TestCase):
    def test_require_semantic_order_id_ok_true_returns_order_id(self) -> None:
        from src.execution.broker_semantic_helpers import require_semantic_order_id
        from src.exchanges.models import ExchangeName
        r = BrokerSemanticResult(
            exchange=ExchangeName.OKX, symbol="X",
            action=BrokerSemanticAction.PLACE_REDUCE_ONLY_TP,
            role=BrokerSemanticOrderRole.CORE_TP,
            ok=True, order_id="ord-123",
        )
        self.assertEqual(require_semantic_order_id(r, action="TEST"), "ord-123")

    def test_require_semantic_order_id_ok_false_raises(self) -> None:
        from src.execution.broker_semantic_helpers import require_semantic_order_id
        from src.exchanges.models import ExchangeName
        r = BrokerSemanticResult(
            exchange=ExchangeName.OKX, symbol="X",
            action=BrokerSemanticAction.PLACE_REDUCE_ONLY_TP,
            role=BrokerSemanticOrderRole.CORE_TP,
            ok=False, message="exchange rejected",
        )
        with self.assertRaises(RuntimeError) as ctx:
            require_semantic_order_id(r, action="TEST_ACTION")
        self.assertIn("TEST_ACTION", str(ctx.exception))
        self.assertIn("exchange rejected", str(ctx.exception))

    def test_require_semantic_order_id_missing_order_id_raises(self) -> None:
        from src.execution.broker_semantic_helpers import require_semantic_order_id
        from src.exchanges.models import ExchangeName
        r = BrokerSemanticResult(
            exchange=ExchangeName.OKX, symbol="X",
            action=BrokerSemanticAction.PLACE_REDUCE_ONLY_TP,
            role=BrokerSemanticOrderRole.CORE_TP,
            ok=True, order_id=None,
        )
        with self.assertRaises(RuntimeError) as ctx:
            require_semantic_order_id(r, action="TEST_ACTION")
        self.assertIn("no order_id", str(ctx.exception))

    def test_require_semantic_ok_true_passes(self) -> None:
        from src.execution.broker_semantic_helpers import require_semantic_ok
        from src.exchanges.models import ExchangeName
        r = BrokerSemanticResult(
            exchange=ExchangeName.OKX, symbol="X",
            action=BrokerSemanticAction.CANCEL_REDUCE_ONLY_TP,
            role=BrokerSemanticOrderRole.CORE_TP,
            ok=True,
        )
        # Should not raise
        require_semantic_ok(r, action="TEST")

    def test_require_semantic_ok_false_raises(self) -> None:
        from src.execution.broker_semantic_helpers import require_semantic_ok
        from src.exchanges.models import ExchangeName
        r = BrokerSemanticResult(
            exchange=ExchangeName.OKX, symbol="X",
            action=BrokerSemanticAction.CANCEL_REDUCE_ONLY_TP,
            role=BrokerSemanticOrderRole.CORE_TP,
            ok=False, message="cancel rejected",
        )
        with self.assertRaises(RuntimeError) as ctx:
            require_semantic_ok(r, action="CANCEL_TEST")
        self.assertIn("CANCEL_TEST", str(ctx.exception))
