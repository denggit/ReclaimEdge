from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from decimal import Decimal

if importlib.util.find_spec("aiohttp") is None:
    aiohttp = types.ModuleType("aiohttp")
    sys.modules.setdefault("aiohttp", aiohttp)

import src.execution.trader as trader_module  # noqa: E402
from src.execution.trader import Trader  # noqa: E402
from src.risk.simple_position_sizer import PositionSize  # noqa: E402
from src.strategies.boll_cvd_reclaim_strategy import TradeIntent  # noqa: E402


def make_intent(**overrides) -> TradeIntent:
    kwargs = dict(
        intent_type="UPDATE_TP",
        side="LONG",
        price=3000.0,
        layer_index=1,
        tp_price=3100.0,
        reason="test",
        size=PositionSize(30.0, 1500.0, 0.5, 1, 1.0),
        fast_cvd=0.0,
        previous_fast_cvd=0.0,
        buy_ratio=0.0,
        sell_ratio=0.0,
        boll_upper=3100.0,
        boll_middle=3000.0,
        boll_lower=2900.0,
        ts_ms=1000,
        avg_entry_price=3000.0,
        breakeven_price=3003.0,
        tp_mode="MIDDLE",
    )
    kwargs.update(overrides)
    return TradeIntent(**kwargs)  # type: ignore[arg-type]


def make_trader(**overrides) -> Trader:
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


class TraderManagedCoreContractsTest(unittest.IsolatedAsyncioTestCase):
    """Tests for Trader._managed_core_contracts_from_intent."""

    def test_managed_core_contracts_valid_no_attribute_error(self) -> None:
        """replace_take_profit(managed_core_contracts="10") does not raise AttributeError."""
        trader = make_trader()
        intent = make_intent(managed_core_contracts="10")
        result = trader._managed_core_contracts_from_intent(intent)
        self.assertIsNotNone(result)
        self.assertEqual(result, Decimal("10"))

    async def test_sidecar_fixed_tp_sanitizes_client_order_id(self) -> None:
        trader = make_trader()
        requests = []

        async def fake_request(method, path, body):  # type: ignore[no-untyped-def]
            requests.append((method, path, dict(body)))
            return {"data": [{"ordId": "tp-1"}]}

        trader.request = fake_request  # type: ignore[method-assign]

        order_id = await trader.place_sidecar_fixed_take_profit(
            side="LONG",
            contracts="0.69",
            tp_price=3012.0,
            client_order_id="SC-97644895de-L1-47229",
        )

        self.assertEqual(order_id, "tp-1")
        self.assertEqual(requests[0][2]["clOrdId"], "SC97644895deL147229")
        self.assertLessEqual(len(requests[0][2]["clOrdId"]), 32)

    async def test_sidecar_fixed_tp_omits_empty_sanitized_client_order_id(self) -> None:
        trader = make_trader()
        requests = []

        async def fake_request(method, path, body):  # type: ignore[no-untyped-def]
            requests.append((method, path, dict(body)))
            return {"data": [{"ordId": "tp-1"}]}

        trader.request = fake_request  # type: ignore[method-assign]

        await trader.place_sidecar_fixed_take_profit(
            side="LONG",
            contracts="0.69",
            tp_price=3012.0,
            client_order_id="---___:::",
        )

        self.assertNotIn("clOrdId", requests[0][2])

    def test_managed_core_contracts_none_returns_none(self) -> None:
        """managed_core_contracts=None returns None (old logic path)."""
        trader = make_trader()
        intent = make_intent(managed_core_contracts=None)
        result = trader._managed_core_contracts_from_intent(intent)
        self.assertIsNone(result)

    def test_managed_core_contracts_empty_string_returns_none(self) -> None:
        """managed_core_contracts='' returns None."""
        trader = make_trader()
        intent = make_intent(managed_core_contracts="")
        result = trader._managed_core_contracts_from_intent(intent)
        self.assertIsNone(result)

    def test_managed_core_contracts_negative_returns_none(self) -> None:
        """managed_core_contracts <= 0 returns None."""
        trader = make_trader()
        intent = make_intent(managed_core_contracts="-5")
        result = trader._managed_core_contracts_from_intent(intent)
        self.assertIsNone(result)

    def test_managed_core_contracts_zero_returns_none(self) -> None:
        """managed_core_contracts=0 returns None."""
        trader = make_trader()
        intent = make_intent(managed_core_contracts="0")
        result = trader._managed_core_contracts_from_intent(intent)
        self.assertIsNone(result)

    def test_managed_core_contracts_invalid_string_raises_runtime_error(self) -> None:
        """managed_core_contracts invalid string raises RuntimeError."""
        trader = make_trader()
        intent = make_intent(managed_core_contracts="not_a_number")
        with self.assertRaises(RuntimeError) as ctx:
            trader._managed_core_contracts_from_intent(intent)
        self.assertIn("invalid managed_core_contracts", str(ctx.exception))

    def test_managed_core_contracts_below_min_contracts_raises_runtime_error(self) -> None:
        """managed_core_contracts < min_contracts raises RuntimeError."""
        trader = make_trader(min_contracts=Decimal("1"))
        # 0.01 < 1.0 min_contracts
        intent = make_intent(managed_core_contracts="0.001")
        with self.assertRaises(RuntimeError) as ctx:
            trader._managed_core_contracts_from_intent(intent)
        self.assertIn("managed_core_contracts below min_contracts", str(ctx.exception))

    def test_managed_core_contracts_rounded_down(self) -> None:
        """managed_core_contracts is rounded down to contract_precision."""
        trader = make_trader()
        # 10.005 should be rounded down to 10.00
        intent = make_intent(managed_core_contracts="10.005")
        result = trader._managed_core_contracts_from_intent(intent)
        self.assertEqual(result, Decimal("10.00"))

    async def test_replace_take_profit_uses_managed_core_contracts_for_tp(self) -> None:
        """When managed_core_contracts=10 and OKX net=12, replace_take_profit sets TP to 10 (core)."""
        trader = make_trader()
        trader.side = "LONG"
        placed: list[tuple] = []

        async def mock_fetch_snapshot():  # type: ignore[no-untyped-def]
            return trader_module.PositionSnapshot("LONG", Decimal("12"), 3000.0, 1.2, Decimal("12"))

        async def mock_fetch_pending():  # type: ignore[no-untyped-def]
            return []

        async def mock_place_tp(intent_, specs):  # type: ignore[no-untyped-def]
            placed.extend(specs)
            return [f"tp-{label}" for label, _c, _p in specs]

        async def mock_cancel_existing():  # type: ignore[no-untyped-def]
            return None

        trader.fetch_position_snapshot = mock_fetch_snapshot
        trader.fetch_pending_orders = mock_fetch_pending
        trader._place_reduce_only_take_profit_orders = mock_place_tp
        trader.cancel_existing_reduce_only_orders = mock_cancel_existing

        intent = make_intent(managed_core_contracts="10")
        result = await trader.replace_take_profit(intent)

        self.assertTrue(result.ok)
        self.assertEqual(result.contracts, "10")
        self.assertEqual(placed, [("final", Decimal("10"), 3100.0)])

    async def test_stale_update_tp_core_exceeds_net_skips_without_cancel_or_place(self) -> None:
        trader = make_trader()
        cancelled = False
        placed = False

        async def mock_fetch_snapshot():  # type: ignore[no-untyped-def]
            return trader_module.PositionSnapshot("LONG", Decimal("0.71"), 3000.0, 0.071, Decimal("0.71"))

        async def mock_cancel_existing():  # type: ignore[no-untyped-def]
            nonlocal cancelled
            cancelled = True

        async def mock_place_tp(_intent, _specs):  # type: ignore[no-untyped-def]
            nonlocal placed
            placed = True
            return ["tp-new"]

        trader.fetch_position_snapshot = mock_fetch_snapshot
        trader.cancel_existing_reduce_only_orders = mock_cancel_existing
        trader._place_reduce_only_take_profit_orders = mock_place_tp

        result = await trader.replace_take_profit(
            make_intent(managed_core_contracts="1.41", allow_stale_tp_update_skip=True)
        )

        self.assertTrue(result.ok)
        self.assertTrue(result.tp_ok)
        self.assertEqual(result.message, "stale_tp_update_skipped_net_reduced")
        self.assertFalse(cancelled)
        self.assertFalse(placed)

    async def test_non_update_tp_core_exceeds_net_still_raises(self) -> None:
        trader = make_trader()

        async def mock_fetch_snapshot():  # type: ignore[no-untyped-def]
            return trader_module.PositionSnapshot("LONG", Decimal("0.71"), 3000.0, 0.071, Decimal("0.71"))

        trader.fetch_position_snapshot = mock_fetch_snapshot

        with self.assertRaisesRegex(RuntimeError, "managed_core_contracts_exceeds_net_position"):
            await trader.replace_take_profit(
                make_intent(intent_type="OPEN_LONG", managed_core_contracts="1.41")
            )

    async def test_invalid_trend_runner_sl_with_old_sl_active_skips_ok(self) -> None:
        trader = make_trader()
        trader.tp_order_id = "old-tp"
        sl_place_called = False
        cancelled = False

        async def mock_fetch_snapshot():  # type: ignore[no-untyped-def]
            return trader_module.PositionSnapshot("LONG", Decimal("1.00"), 1670.0, 0.1, Decimal("1.00"))

        async def mock_cancel_existing(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal cancelled
            cancelled = True
            return True

        async def mock_place_tp(_intent, _specs):  # type: ignore[no-untyped-def]
            return ["tp-new"]

        async def mock_place_sl(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal sl_place_called
            sl_place_called = True
            return True, "sl-new", "ok"

        trader.fetch_position_snapshot = mock_fetch_snapshot
        trader.cancel_existing_reduce_only_orders = mock_cancel_existing
        trader._place_reduce_only_take_profit_orders = mock_place_tp
        trader.place_trend_runner_protective_stop_with_retries = mock_place_sl

        result = await trader.replace_take_profit(
            make_intent(
                price=1678.18,
                trend_runner_active=True,
                trend_runner_sl_price=1679.22,
                trend_runner_sl_order_id="old-sl",
                managed_core_contracts="1.00",
            )
        )

        self.assertTrue(result.ok)
        self.assertTrue(result.protective_sl_ok)
        self.assertEqual(result.message, "trend_runner_sl_update_skipped_invalid_but_old_sl_active")
        self.assertFalse(sl_place_called)
        self.assertTrue(cancelled)

    async def test_invalid_trend_runner_sl_without_old_sl_market_exits(self) -> None:
        trader = make_trader()
        sl_place_called = False
        market_exit_called = False
        cancel_called = False
        place_tp_called = False

        async def mock_fetch_snapshot():  # type: ignore[no-untyped-def]
            return trader_module.PositionSnapshot("LONG", Decimal("1.00"), 1670.0, 0.1, Decimal("1.00"))

        async def mock_cancel_existing(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal cancel_called
            cancel_called = True
            return True

        async def mock_place_tp(_intent, _specs):  # type: ignore[no-untyped-def]
            nonlocal place_tp_called
            place_tp_called = True
            return ["tp-new"]

        async def mock_fetch_pending_algo_orders():  # type: ignore[no-untyped-def]
            return []

        async def mock_place_sl(*args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal sl_place_called
            sl_place_called = True
            return True, "sl-new", "ok"

        async def mock_market_exit(intent):  # type: ignore[no-untyped-def]
            nonlocal market_exit_called
            market_exit_called = True
            return trader_module.LiveTradeResult(
                True,
                "MARKET_EXIT_RUNNER",
                None,
                None,
                "1",
                "1700.00",
                "market_exit_order_id=exit-1",
                reduce_filled=True,
                near_tp_exit_all=True,
                contracts_before="1",
                contracts_reduced="1",
                contracts_after="0",
            )

        trader.fetch_position_snapshot = mock_fetch_snapshot
        trader.cancel_existing_reduce_only_orders = mock_cancel_existing
        trader._place_reduce_only_take_profit_orders = mock_place_tp
        trader.fetch_pending_algo_orders = mock_fetch_pending_algo_orders
        trader.place_trend_runner_protective_stop_with_retries = mock_place_sl
        trader.execute_market_exit_runner = mock_market_exit

        result = await trader.replace_take_profit(
            make_intent(
                price=1678.18,
                trend_runner_active=True,
                trend_runner_sl_price=1679.22,
                trend_runner_sl_order_id=None,
                managed_core_contracts="1.00",
            )
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.action, "MARKET_EXIT_RUNNER")
        self.assertFalse(sl_place_called)
        self.assertFalse(cancel_called)
        self.assertFalse(place_tp_called)
        self.assertTrue(market_exit_called)

    async def test_trend_runner_sl_place_failed_with_old_sl_active_keeps_tp_update_ok(self) -> None:
        trader = make_trader()
        trader.tp_order_id = "old-tp"
        trader.trend_runner_sl_order_id = "old-sl"

        async def mock_fetch_snapshot():  # type: ignore[no-untyped-def]
            return trader_module.PositionSnapshot("LONG", Decimal("1.00"), 1670.0, 0.1, Decimal("1.00"))

        async def mock_cancel_existing(*args, **kwargs):  # type: ignore[no-untyped-def]
            return True

        async def mock_place_tp(_intent, _specs):  # type: ignore[no-untyped-def]
            return ["tp-new"]

        async def mock_place_sl(*args, **kwargs):  # type: ignore[no-untyped-def]
            return False, None, "exchange rejected"

        trader.fetch_position_snapshot = mock_fetch_snapshot
        trader.cancel_existing_reduce_only_orders = mock_cancel_existing
        trader._place_reduce_only_take_profit_orders = mock_place_tp
        trader.place_trend_runner_protective_stop_with_retries = mock_place_sl

        result = await trader.replace_take_profit(
            make_intent(
                price=1678.18,
                trend_runner_active=True,
                trend_runner_sl_price=1677.00,
                trend_runner_sl_order_id="old-sl",
                managed_core_contracts="1.00",
            )
        )

        self.assertTrue(result.ok)
        self.assertTrue(result.tp_ok)
        self.assertTrue(result.protective_sl_ok)
        self.assertEqual(result.message, "trend_runner_sl_update_failed_but_old_sl_active")
        self.assertEqual(result.tp_order_ids, ("tp-new",))
        self.assertEqual(result.protective_sl_order_id, "old-sl")


if __name__ == "__main__":
    unittest.main()
