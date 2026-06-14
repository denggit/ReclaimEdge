from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import os
import sys
import types
import unittest
from decimal import Decimal
from unittest.mock import patch

from tests.conftest import FakeOkxClient
from src.execution.okx_trading_client import OkxTradingClient
from src.execution.tp_sl_execution_manager import TpSlExecutionManager

if importlib.util.find_spec("dotenv") is None:
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules.setdefault("dotenv", dotenv)

from src.live.workers.account_position_sync_worker import account_position_sync_worker  # noqa: E402
from src.live.workers.execution_worker import execution_worker  # noqa: E402
from src.live.startup_recovery.basic_restore import restore_strategy_from_saved_state  # noqa: E402
from src.execution.trader import LiveTradeResult, PositionSnapshot, Trader  # noqa: E402
from src.indicators.cvd_tracker import CvdSnapshot  # noqa: E402
from src.live.runtime_types import AccountSnapshot, ExecutionState, TradeCommand  # noqa: E402
from src.monitors.boll_band_breakout_monitor import BollSnapshot  # noqa: E402
from src.reporting.live_state_store import LivePositionState, LiveStateStore  # noqa: E402
from src.risk.simple_position_sizer import PositionSize, SimplePositionSizer, SimplePositionSizerConfig  # noqa: E402
from src.strategies.boll_cvd_reclaim_strategy import (  # noqa: E402
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
    StrategyPositionState,
    TradeIntent,
)
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy  # noqa: E402


def boll() -> BollSnapshot:
    return BollSnapshot("ETH-USDT-SWAP", 1_000, 100.0, 110.0, 120.0, 90.0, 0.1, 0.1, True, True)


def cvd() -> CvdSnapshot:
    return CvdSnapshot(
        ts_ms=1_000,
        price=100.0,
        side="buy",
        size=1.0,
        signed_delta=1.0,
        total_cvd=1.0,
        fast_cvd=1.0,
        previous_fast_cvd=0.0,
        buy_volume=1.0,
        sell_volume=0.0,
        buy_ratio=1.0,
        sell_ratio=0.0,
        cross_positive=True,
        cross_negative=False,
        cvd_increasing=True,
        cvd_decreasing=False,
        no_new_low=True,
        no_new_high=True,
        window_low=99.0,
        window_high=101.0,
        burst_net_move_pct=0.0,
        burst_range_pct=0.0,
        baseline_range_pct=0.0,
        burst_move_ratio=0.0,
        burst_volume=0.0,
        baseline_volume=0.0,
        burst_volume_ratio=0.0,
        up_burst=False,
        down_burst=False,
    )


def size() -> PositionSize:
    return PositionSize(1.0, 50.0, 0.5, 1, 1.0)


def near_tp_config(**overrides) -> BollCvdReclaimStrategyConfig:
    values = dict(
        near_tp_enabled=True,
        near_tp_reduce_enabled=True,
        near_tp_giveback_usd=3.0,
        near_tp_giveback_pct=0.0015,
        near_tp_giveback_profit_ratio=0.25,
        near_tp_min_profit_pct=0.004,
        near_tp_min_reduce_profit_pct=0.004,
        near_tp_reduce_ratio=0.5,
    )
    values.update(overrides)
    return BollCvdReclaimStrategyConfig(**values)


def strategy(**config_overrides) -> BollCvdReclaimStrategy:
    return BollCvdReclaimStrategy(
        near_tp_config(**config_overrides),
        SimplePositionSizer(SimplePositionSizerConfig()),
    )


def intent(**overrides) -> TradeIntent:
    values = dict(
        intent_type="NEAR_TP_REDUCE",
        side="LONG",
        price=106.0,
        layer_index=2,
        tp_price=110.0,
        reason="near_tp_giveback_protection",
        size=size(),
        fast_cvd=0.0,
        previous_fast_cvd=0.0,
        buy_ratio=1.0,
        sell_ratio=0.0,
        boll_upper=120.0,
        boll_middle=110.0,
        boll_lower=90.0,
        ts_ms=1_000,
        avg_entry_price=100.0,
        breakeven_price=100.1,
        tp_mode="MIDDLE",
        tp_plan="SINGLE",
        partial_tp_consumed=True,
        near_tp_progress_ratio=0.6,
        near_tp_best_price=109.0,
        near_tp_giveback=3.0,
        near_tp_giveback_threshold=3.0,
        near_tp_reduce_ratio=0.5,
        near_tp_protective_sl_price=100.1,
    )
    values.update(overrides)
    return TradeIntent(**values)  # type: ignore[arg-type]


def seeded_long_state(**overrides) -> StrategyPositionState:
    values = dict(
        side="LONG",
        layers=2,
        last_entry_price=100.0,
        tp_price=110.0,
        total_entry_qty=1.0,
        total_entry_notional=100.0,
        avg_entry_price=100.0,
        breakeven_price=100.1,
        tp_mode="MIDDLE",
        tp_plan="SINGLE",
    )
    values.update(overrides)
    return StrategyPositionState(**values)


def flat_position() -> PositionSnapshot:
    return PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0"))


class NearTpStrategyTest(unittest.TestCase):
    def test_long_near_tp_arms_at_88_percent(self) -> None:
        strat = strategy()
        strat.state = seeded_long_state()

        result = strat._maybe_near_tp_reduce(108.8, 1_000, boll(), cvd())

        self.assertIsNone(result)
        self.assertTrue(strat.state.near_tp_armed)
        self.assertEqual(strat.state.near_tp_best_price, 108.8)

    def test_long_giveback_triggers_reduce_when_profit_still_above_0_4(self) -> None:
        strat = strategy()
        strat.state = seeded_long_state(near_tp_armed=True, near_tp_best_price=109.0)

        result = strat._maybe_near_tp_reduce(106.0, 2_000, boll(), cvd())

        self.assertIsNotNone(result)
        self.assertEqual(result.intent_type, "NEAR_TP_REDUCE")
        self.assertEqual(result.near_tp_reduce_ratio, 0.5)
        self.assertAlmostEqual(result.near_tp_protective_sl_price or 0, 100.1)

    def test_long_giveback_sets_pending_when_profit_below_0_4(self) -> None:
        strat = strategy()
        strat.state = seeded_long_state(near_tp_armed=True, near_tp_best_price=109.0)

        result = strat._maybe_near_tp_reduce(100.3, 2_000, boll(), cvd())

        self.assertIsNone(result)
        self.assertTrue(strat.state.near_tp_reduce_pending)
        self.assertEqual(strat.state.near_tp_pending_ts_ms, 2_000)

    def test_pending_reduce_executes_when_price_recovers_to_0_4_profit(self) -> None:
        strat = strategy()
        strat.state = seeded_long_state(near_tp_armed=True, near_tp_reduce_pending=True, near_tp_best_price=109.0)

        result = strat._maybe_near_tp_reduce(100.41, 3_000, boll(), cvd())

        self.assertIsNotNone(result)
        self.assertEqual(result.intent_type, "NEAR_TP_REDUCE")

    def test_short_symmetric_near_tp_reduce(self) -> None:
        strat = strategy()
        strat.state = StrategyPositionState(
            side="SHORT",
            layers=2,
            last_entry_price=100.0,
            tp_price=90.0,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            breakeven_price=99.9,
            tp_mode="MIDDLE",
            near_tp_armed=True,
            near_tp_best_price=91.0,
        )

        result = strat._maybe_near_tp_reduce(94.0, 2_000, boll(), cvd())

        self.assertIsNotNone(result)
        self.assertEqual(result.side, "SHORT")
        self.assertAlmostEqual(result.near_tp_protective_sl_price or 0, 99.9)

    def test_split_partial_not_consumed_blocks_near_tp(self) -> None:
        strat = strategy()
        strat.state = seeded_long_state(tp_plan="SPLIT_PARTIAL_FINAL", partial_tp_consumed=False)

        result = strat._maybe_near_tp_reduce(108.8, 1_000, boll(), cvd())

        self.assertIsNone(result)
        self.assertFalse(strat.state.near_tp_armed)

    def test_split_partial_consumed_allows_near_tp(self) -> None:
        strat = strategy()
        strat.state = seeded_long_state(
            tp_plan="SPLIT_PARTIAL_FINAL",
            partial_tp_consumed=True,
            near_tp_armed=True,
            near_tp_best_price=109.0,
        )

        result = strat._maybe_near_tp_reduce(106.0, 2_000, boll(), cvd())

        self.assertIsNotNone(result)
        self.assertEqual(result.intent_type, "NEAR_TP_REDUCE")

    def test_near_tp_protected_blocks_repeat_reduce(self) -> None:
        strat = strategy()
        strat.state = seeded_long_state(near_tp_protected=True, near_tp_armed=True, near_tp_best_price=109.0)

        self.assertIsNone(strat._maybe_near_tp_reduce(106.0, 2_000, boll(), cvd()))

    def test_near_tp_protected_tp_update_stays_single_final_tp(self) -> None:
        strat = strategy(split_tp_min_layers=4)
        strat.state = seeded_long_state(
            layers=4,
            tp_price=105.0,
            partial_tp_consumed=False,
            near_tp_protected=True,
            near_tp_add_disabled=True,
        )

        result = strat._maybe_update_tp(106.0, 2_000, boll(), cvd())

        self.assertIsNotNone(result)
        self.assertEqual(result.tp_plan, "SINGLE")
        self.assertIsNone(result.partial_tp_price)
        self.assertEqual(result.partial_tp_ratio, 0.0)
        self.assertEqual(strat.state.tp_plan, "SINGLE")

    def test_near_tp_add_disabled_blocks_add(self) -> None:
        strat = strategy()
        strat.state = seeded_long_state(near_tp_add_disabled=True)

        result = strat._maybe_open_or_add_long(99.0, 2_000, boll(), cvd())

        self.assertIsNone(result)
        self.assertEqual(strat.state.layers, 2)

    def test_open_resets_near_tp_state(self) -> None:
        strat = strategy()
        strat.state = StrategyPositionState(
            near_tp_armed=True,
            near_tp_reduce_pending=True,
            near_tp_protected=True,
            near_tp_best_price=109.0,
            near_tp_protective_sl_price=100.1,
            near_tp_protective_sl_order_id="algo-1",
            near_tp_add_disabled=True,
        )

        strat._open_position("LONG", "OPEN_LONG", 100.0, 1_000, boll(), cvd(), "open")

        self.assertFalse(strat.state.near_tp_armed)
        self.assertFalse(strat.state.near_tp_reduce_pending)
        self.assertFalse(strat.state.near_tp_protected)
        self.assertIsNone(strat.state.near_tp_best_price)
        self.assertIsNone(strat.state.near_tp_protective_sl_order_id)
        self.assertFalse(strat.state.near_tp_add_disabled)

    def test_flat_state_defaults_clear_near_tp(self) -> None:
        state = StrategyPositionState()

        self.assertFalse(state.near_tp_armed)
        self.assertFalse(state.near_tp_reduce_pending)
        self.assertFalse(state.near_tp_protected)
        self.assertIsNone(state.near_tp_best_price)
        self.assertIsNone(state.near_tp_protective_sl_order_id)
        self.assertFalse(state.near_tp_add_disabled)


class RecordingTrader(Trader):
    def __init__(self) -> None:
        self.symbol = "ETH-USDT-SWAP"
        self.td_mode = "isolated"
        self.pos_side_mode = "net"
        self.position_contracts = Decimal("1")
        self.contract_precision = Decimal("0.01")
        self.min_contracts = Decimal("0.01")
        self.tp_order_id = None
        self.near_tp_protective_sl_order_id = None
        self.middle_runner_protective_sl_order_id = None
        self.three_stage_post_tp1_protective_sl_order_id = None
        self.trend_runner_sl_order_id = None
        self.contract_multiplier = Decimal("0.1")
        self._client = FakeOkxClient(self)
        self.trading_client = OkxTradingClient(self)  # type: ignore[assignment]
        self._tp_sl_manager = TpSlExecutionManager(self, trading_client=self.trading_client)  # type: ignore[arg-type]
        self.positions: list[PositionSnapshot] = [
            PositionSnapshot("LONG", Decimal("1"), 100.0, 0.1, Decimal("1")),
            PositionSnapshot("LONG", Decimal("0.5"), 100.0, 0.05, Decimal("0.5")),
            PositionSnapshot("LONG", Decimal("0.5"), 100.0, 0.05, Decimal("0.5")),
        ]
        self.requests: list[tuple[str, str, object]] = []
        self.fail_algo_attempts = 0
        self.fail_fallback_attempts = 0
        self.fail_market_exit_attempts = 0
        self.market_order_count = 0
        self.market_orders: list[dict] = []
        self.algo_submit_count = 0
        self.secondary_algo_count = 0
        self.algo_orders: dict[str, dict] = {}
        self.verify_missing_attempts = 0
        self.raise_on_fetch_position = False
        self.fetch_position_exception_once = False
        self.cancel_reduce_only_calls = 0
        self.cancel_protective_calls = 0
        self.cancelled_algo_ids: list[str] = []
        self.fail_replace_take_profit = False

    async def fetch_position_snapshot(self) -> PositionSnapshot:
        if self.fetch_position_exception_once:
            self.fetch_position_exception_once = False
            raise RuntimeError("fetch position failed")
        if self.raise_on_fetch_position:
            raise RuntimeError("fetch position failed")
        if self.positions:
            return self.positions.pop(0)
        return PositionSnapshot("LONG", Decimal("0.5"), 100.0, 0.05, Decimal("0.5"))

    async def fetch_pending_orders(self) -> list[dict]:
        return []

    async def fetch_pending_algo_orders(self) -> list[dict]:
        if self.verify_missing_attempts > 0:
            self.verify_missing_attempts -= 1
            return []
        return list(self.algo_orders.values())

    async def cancel_existing_reduce_only_orders(self) -> None:
        self.cancel_reduce_only_calls += 1

    async def cancel_near_tp_protective_stop(self, order_id: str | None) -> bool:
        self.cancel_protective_calls += 1
        if order_id:
            self.cancelled_algo_ids.append(order_id)
            self.algo_orders.pop(order_id, None)
        return True

    async def replace_take_profit(self, trade_intent: TradeIntent) -> LiveTradeResult:
        if self.fail_replace_take_profit:
            raise RuntimeError("replace tp failed")
        return await super().replace_take_profit(trade_intent)

    async def request(self, method: str, endpoint: str, payload=None):  # type: ignore[no-untyped-def]
        self.requests.append((method, endpoint, payload or {}))
        if endpoint == "/api/v5/trade/order-algo":
            self.algo_submit_count += 1
            body = dict(payload or {})
            if self.algo_submit_count <= 3 and self.fail_algo_attempts > 0:
                self.fail_algo_attempts -= 1
                raise RuntimeError("primary algo failed")
            if self.algo_submit_count > 3 and self.fail_fallback_attempts > 0:
                self.fail_fallback_attempts -= 1
                raise RuntimeError("fallback algo failed")
            if self.algo_submit_count > 3:
                self.secondary_algo_count += 1
                algo_id = f"algo-sec-{self.secondary_algo_count}"
            else:
                algo_id = f"algo-{self.algo_submit_count}"
            body["algoId"] = algo_id
            self.algo_orders[algo_id] = body
            return {"code": "0", "data": [{"algoId": algo_id}]}
        if endpoint == "/api/v5/trade/order":
            if payload and payload.get("ordType") == "market" and payload.get("reduceOnly") == "true":
                self.market_order_count += 1
                self.market_orders.append(dict(payload))
            if (
                    payload
                    and payload.get("ordType") == "market"
                    and payload.get("reduceOnly") == "true"
                    and self.market_order_count > 1
                    and self.fail_market_exit_attempts > 0
            ):
                self.fail_market_exit_attempts -= 1
                raise RuntimeError("market exit failed")
            return {"code": "0", "data": [{"ordId": f"ord-{len(self.requests)}"}]}
        return {"code": "0", "data": [{"ordId": f"ord-{len(self.requests)}"}]}


class NearTpTraderTest(unittest.IsolatedAsyncioTestCase):
    async def test_execute_near_tp_reduce_market_reduces_half(self) -> None:
        trader = RecordingTrader()

        result = await trader.execute_near_tp_reduce(intent())

        self.assertTrue(result.ok)
        market_orders = [payload for _m, endpoint, payload in trader.requests if
                         endpoint == "/api/v5/trade/order" and payload.get("ordType") == "market"]
        self.assertEqual(market_orders[0]["side"], "sell")
        self.assertEqual(market_orders[0]["reduceOnly"], "true")
        self.assertEqual(market_orders[0]["sz"], "0.5")
        self.assertEqual(result.contracts_reduced, "0.5")

    async def test_execute_near_tp_reduce_replaces_final_tp_for_remaining_position(self) -> None:
        trader = RecordingTrader()

        await trader.execute_near_tp_reduce(
            intent(partial_tp_price=108.0, partial_tp_ratio=0.5, tp_plan="SPLIT_PARTIAL_FINAL"))

        tp_orders = [payload for _m, endpoint, payload in trader.requests if
                     endpoint == "/api/v5/trade/order" and payload.get("ordType") == "limit"]
        self.assertEqual(len(tp_orders), 1)
        self.assertEqual(tp_orders[0]["px"], "110.00")
        self.assertEqual(tp_orders[0]["sz"], "0.5")

    async def test_execute_near_tp_reduce_places_protective_sl(self) -> None:
        trader = RecordingTrader()

        result = await trader.execute_near_tp_reduce(intent())

        self.assertTrue(result.protective_sl_ok)
        self.assertTrue(result.protective_sl_order_id)
        algo_orders = [payload for _m, endpoint, payload in trader.requests if endpoint == "/api/v5/trade/order-algo"]
        self.assertEqual(algo_orders[-1]["slTriggerPx"], "100.10")
        self.assertEqual(algo_orders[-1]["reduceOnly"], "true")

    async def test_protective_sl_retry_then_success(self) -> None:
        trader = RecordingTrader()
        trader.fail_algo_attempts = 2

        ok, order_id, _message = await trader.place_near_tp_protective_stop_with_retries("LONG", Decimal("0.5"), 100.1,
                                                                                         3, 0)

        self.assertTrue(ok)
        self.assertTrue(order_id)
        self.assertEqual(len([r for r in trader.requests if r[1] == "/api/v5/trade/order-algo"]), 3)

    async def test_protective_sl_fallback_conditional_then_success(self) -> None:
        trader = RecordingTrader()
        trader.fail_algo_attempts = 3

        ok, order_id, message = await trader.place_near_tp_protective_stop_with_retries("LONG", Decimal("0.5"), 100.1,
                                                                                        3, 0)

        self.assertTrue(ok)
        self.assertTrue(order_id)
        self.assertEqual(message, "fallback_conditional_close_placed")
        fallback_orders = [payload for index, (_m, endpoint, payload) in enumerate(trader.requests, start=1) if
                           endpoint == "/api/v5/trade/order-algo" and index > 3]
        self.assertEqual(len(fallback_orders), 1)
        self.assertIn("slTriggerPx", fallback_orders[0])
        self.assertNotIn("triggerPx", fallback_orders[0])

    async def test_reduce_filled_final_tp_replace_fails_still_places_protective_sl(self) -> None:
        trader = RecordingTrader()
        trader.fail_replace_take_profit = True

        result = await trader.execute_near_tp_reduce(intent())

        self.assertTrue(result.ok)
        self.assertTrue(result.reduce_filled)
        self.assertFalse(result.tp_ok)
        self.assertTrue(result.protective_sl_ok)
        self.assertIn("final_tp_failed", result.message)
        self.assertIn("near_tp_reduce_done_final_tp_and_protective_sl_placed", result.message)

    async def test_reduce_filled_final_tp_replace_fails_and_sl_fails_no_market_exit(self) -> None:
        """Final TP replace fails and protective SL fails → no immediate market exit."""
        trader = RecordingTrader()
        trader.fail_replace_take_profit = True
        trader.fail_algo_attempts = 3
        trader.fail_fallback_attempts = 3
        trader.positions.append(PositionSnapshot("LONG", Decimal("0.5"), 100.0, 0.05, Decimal("0.5")))

        with patch.dict(os.environ, {"NEAR_TP_PROTECTIVE_SL_RETRY_INTERVAL_SECONDS": "0"}):
            result = await trader.execute_near_tp_reduce(intent())

        self.assertFalse(result.ok)
        self.assertFalse(result.tp_ok)
        self.assertFalse(result.protective_sl_ok)
        self.assertFalse(getattr(result, "near_tp_exit_all", True))
        self.assertIn("final_tp_failed", result.message)
        self.assertIn("protective_sl_failed", result.message)

    async def test_protective_sl_api_success_but_verify_missing_retries(self) -> None:
        trader = RecordingTrader()
        trader.verify_missing_attempts = 1

        with patch.dict(
                os.environ,
                {
                    "NEAR_TP_PROTECTIVE_SL_VERIFY_ATTEMPTS": "1",
                    "NEAR_TP_PROTECTIVE_SL_VERIFY_INTERVAL_SECONDS": "0",
                },
        ):
            ok, order_id, message = await trader.place_near_tp_protective_stop_with_retries("LONG", Decimal("0.5"),
                                                                                            100.1, 2, 0)

        self.assertTrue(ok)
        self.assertTrue(order_id)
        self.assertEqual(message, "protective_sl_placed")
        algo_submits = [payload for _m, endpoint, payload in trader.requests if endpoint == "/api/v5/trade/order-algo"]
        self.assertEqual(len(algo_submits), 2)

    async def test_verify_failed_algo_is_cancelled_before_retry(self) -> None:
        trader = RecordingTrader()
        trader.verify_missing_attempts = 1

        with patch.dict(
                os.environ,
                {
                    "NEAR_TP_PROTECTIVE_SL_VERIFY_ATTEMPTS": "1",
                    "NEAR_TP_PROTECTIVE_SL_VERIFY_INTERVAL_SECONDS": "0",
                },
        ):
            ok, order_id, _message = await trader.place_near_tp_protective_stop_with_retries("LONG", Decimal("0.5"),
                                                                                             100.1, 2, 0)

        self.assertTrue(ok)
        self.assertEqual(order_id, "algo-2")
        self.assertEqual(trader.cancelled_algo_ids, ["algo-1"])
        self.assertNotIn("algo-2", trader.cancelled_algo_ids)
        self.assertEqual(len([r for r in trader.requests if r[1] == "/api/v5/trade/order-algo"]), 2)

    async def test_verify_failed_secondary_algo_is_cancelled_before_retry(self) -> None:
        trader = RecordingTrader()
        trader.fail_algo_attempts = 3
        trader.verify_missing_attempts = 1

        with patch.dict(
                os.environ,
                {
                    "NEAR_TP_PROTECTIVE_SL_VERIFY_ATTEMPTS": "1",
                    "NEAR_TP_PROTECTIVE_SL_VERIFY_INTERVAL_SECONDS": "0",
                },
        ):
            ok, order_id, _message = await trader.place_near_tp_protective_stop_with_retries("LONG", Decimal("0.5"),
                                                                                             100.1, 3, 0)

        self.assertTrue(ok)
        self.assertEqual(order_id, "algo-sec-2")
        self.assertIn("algo-sec-1", trader.cancelled_algo_ids)
        self.assertNotIn("algo-sec-2", trader.cancelled_algo_ids)

    async def test_market_exit_confirms_flat_before_success(self) -> None:
        trader = RecordingTrader()
        trader.positions = [
            PositionSnapshot("LONG", Decimal("0.5"), 100.0, 0.05, Decimal("0.5")),
            PositionSnapshot("LONG", Decimal("0.25"), 100.0, 0.025, Decimal("0.25")),
            flat_position(),
        ]

        ok, message = await trader.market_exit_remaining_position_with_retries("LONG", 3)

        self.assertTrue(ok)
        self.assertEqual(message, "already_flat")
        self.assertEqual(trader.market_order_count, 1)

    async def test_market_exit_closes_exact_min_contract_position(self) -> None:
        trader = RecordingTrader()
        trader.min_contracts = Decimal("0.01")
        trader.positions = [
            PositionSnapshot("LONG", Decimal("0.01"), 100.0, 0.001, Decimal("0.01")),
            flat_position(),
        ]

        ok, message = await trader.market_exit_remaining_position_with_retries("LONG", 1)

        self.assertTrue(ok)
        self.assertIn("market_exit_order_id", message)
        self.assertEqual(trader.market_order_count, 1)
        self.assertEqual(trader.market_orders[0]["sz"], "0.01")
        self.assertEqual(trader.cancel_reduce_only_calls, 1)
        self.assertNotEqual(message, "already_flat")

    async def test_market_exit_does_not_treat_min_contract_as_flat(self) -> None:
        trader = RecordingTrader()
        trader.min_contracts = Decimal("0.01")
        trader.positions = [
            PositionSnapshot("LONG", Decimal("0.01"), 100.0, 0.001, Decimal("0.01")),
            PositionSnapshot("LONG", Decimal("0.01"), 100.0, 0.001, Decimal("0.01")),
        ]

        ok, message = await trader.market_exit_remaining_position_with_retries("LONG", 1)

        self.assertFalse(ok)
        self.assertIn("market_exit_not_flat_after_order", message)
        self.assertEqual(trader.market_order_count, 1)

    async def test_market_exit_dust_below_min_contract_returns_failure(self) -> None:
        trader = RecordingTrader()
        trader.min_contracts = Decimal("0.01")
        trader.positions = [PositionSnapshot("LONG", Decimal("0.005"), 100.0, 0.0005, Decimal("0.005"))]

        ok, message = await trader.market_exit_remaining_position_with_retries("LONG", 1)

        self.assertFalse(ok)
        self.assertIn("dust_position_below_min_contracts", message)
        self.assertEqual(trader.market_order_count, 0)
        self.assertEqual(trader.cancel_reduce_only_calls, 0)

    async def test_market_exit_success_cancels_leftover_reduce_only_tp(self) -> None:
        trader = RecordingTrader()
        trader.near_tp_protective_sl_order_id = "algo-old"
        trader.positions = [
            PositionSnapshot("LONG", Decimal("0.5"), 100.0, 0.05, Decimal("0.5")),
            flat_position(),
        ]

        ok, _message = await trader.market_exit_remaining_position_with_retries("LONG", 1)

        self.assertTrue(ok)
        self.assertEqual(trader.cancel_reduce_only_calls, 1)
        self.assertEqual(trader.cancel_protective_calls, 1)

    async def test_protective_sl_all_retries_fail_no_market_exit(self) -> None:
        """Protective SL failure now returns result.ok=False without immediate market exit."""
        trader = RecordingTrader()
        trader.fail_algo_attempts = 3
        trader.fail_fallback_attempts = 3
        trader.positions.append(PositionSnapshot("LONG", Decimal("0.5"), 100.0, 0.05, Decimal("0.5")))

        with patch.dict(os.environ, {"NEAR_TP_PROTECTIVE_SL_RETRY_INTERVAL_SECONDS": "0"}):
            result = await trader.execute_near_tp_reduce(intent())

        self.assertFalse(result.ok)
        self.assertFalse(result.protective_sl_ok)
        self.assertFalse(getattr(result, "near_tp_exit_all", True))
        self.assertIn("protective_sl_failed", result.message)
        # No near_tp_exit_all means no market exit was attempted by the trader

    async def test_protective_sl_fail_returns_error_without_market_exit(self) -> None:
        """Protective SL failure returns error without attempting market exit."""
        trader = RecordingTrader()
        trader.fail_algo_attempts = 3
        trader.fail_fallback_attempts = 3
        trader.fail_market_exit_attempts = 3
        trader.positions.append(PositionSnapshot("LONG", Decimal("0.5"), 100.0, 0.05, Decimal("0.5")))

        with patch.dict(
                os.environ,
                {
                    "NEAR_TP_PROTECTIVE_SL_RETRY_INTERVAL_SECONDS": "0",
                },
        ):
            result = await trader.execute_near_tp_reduce(intent())

        self.assertFalse(result.ok)
        self.assertTrue(result.reduce_filled)
        self.assertFalse(result.protective_sl_ok)
        self.assertIn("protective_sl_failed", result.message)


class FakeJournal:
    def __init__(self) -> None:
        self.entries = []
        self.near_tp_reduces = []
        self.errors = []
        self.flats = []

    def new_position_id(self, symbol: str, side: str, ts_ms: int | None = None) -> str:
        return f"{symbol}:{side}:{ts_ms}"

    def record_entry(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.entries.append(kwargs)

    def record_tp_update(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        pass

    def record_near_tp_reduce(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.near_tp_reduces.append(kwargs)

    def record_error(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.errors.append(kwargs)

    def record_flat(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.flats.append(kwargs)

    def record_cash_transfer(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        pass

    def record_account_cash_drift(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        pass


class RecordingStateStore:
    def __init__(self) -> None:
        self.saved_states = []
        self.clear_calls = 0

    def save(self, state) -> None:  # type: ignore[no-untyped-def]
        self.saved_states.append(state)

    def clear(self) -> None:
        self.clear_calls += 1


class RecordingEmailSender:
    def __init__(self) -> None:
        self.subjects: list[str] = []

    async def send_email_async(self, subject, content, content_type="html") -> bool:  # type: ignore[no-untyped-def]
        self.subjects.append(subject)
        return True


class RunnerTrader:
    def __init__(self, result: LiveTradeResult, positions: list[PositionSnapshot] | None = None,
                 raise_on_fetch_position: bool = False) -> None:
        self.symbol = "ETH-USDT-SWAP"
        self.position_contracts = Decimal("0.5")
        self.account_equity_usdt = 100.0
        self.result = result
        self.cancelled_algo_ids: list[str] = []
        self.positions = positions or []
        self.raise_on_fetch_position = raise_on_fetch_position

    async def execute_intent(self, trade_intent: TradeIntent) -> LiveTradeResult:
        return self.result

    async def fetch_position_snapshot(self) -> PositionSnapshot:
        if self.raise_on_fetch_position:
            raise RuntimeError("fetch position failed")
        if self.positions:
            return self.positions.pop(0)
        return flat_position()

    async def fetch_usdt_equity(self) -> float:
        return 101.0

    async def request(self, method: str, endpoint: str, payload=None):  # type: ignore[no-untyped-def]
        return {"data": [{"details": [{"ccy": "USDT", "cashBal": "101"}]}]}

    async def cancel_near_tp_protective_stop(self, order_id: str | None) -> bool:
        if order_id:
            self.cancelled_algo_ids.append(order_id)
        return True

    def mark_flat(self) -> None:
        self.position_contracts = Decimal("0")


class FakeStrategy:
    def __init__(self) -> None:
        self.state = StrategyPositionState()


async def run_execution_worker_once(
        *,
        result: LiveTradeResult,
        strat_state: StrategyPositionState,
        execution_state: ExecutionState,
        journal: FakeJournal,
        state_store: RecordingStateStore,
        email_sender: RecordingEmailSender,
        trader_positions: list[PositionSnapshot] | None = None,
        raise_on_fetch_position: bool = False,
) -> RunnerTrader:
    queue: asyncio.Queue[TradeCommand] = asyncio.Queue(maxsize=1000)
    near_intent = intent()
    await queue.put(TradeCommand(near_intent, strat_state, near_intent.ts_ms, asyncio.get_running_loop().time(), 0,
                                 near_intent.reason))
    trader = RunnerTrader(result, trader_positions, raise_on_fetch_position=raise_on_fetch_position)
    strategy_obj = FakeStrategy()
    strategy_obj.state = strat_state
    task = asyncio.create_task(
        execution_worker(
            execution_queue=queue,
            state_lock=asyncio.Lock(),
            execution_state=execution_state,
            account_snapshot=AccountSnapshot(PositionSnapshot("LONG", Decimal("0.5"), 100.0, 0.05, Decimal("0.5")),
                                             100.0, 100.0, asyncio.get_running_loop().time(), 0, 1),
            trader=trader,  # type: ignore[arg-type]
            strategy=strategy_obj,  # type: ignore[arg-type]
            journal=journal,  # type: ignore[arg-type]
            state_store=state_store,  # type: ignore[arg-type]
            email_sender=email_sender,  # type: ignore[arg-type]
            backlog_log_seconds=999,
        )
    )
    await asyncio.wait_for(queue.join(), timeout=1)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    return trader


class NearTpRunnerTest(unittest.IsolatedAsyncioTestCase):
    async def test_execution_worker_near_tp_reduce_success_updates_state(self) -> None:
        strat_state = seeded_long_state(near_tp_reduce_pending=True)
        journal = FakeJournal()
        state_store = RecordingStateStore()
        email_sender = RecordingEmailSender()
        result = LiveTradeResult(
            True,
            "NEAR_TP_REDUCE",
            "ord-1",
            "tp-1",
            "0.5",
            "110.00",
            "ok",
            tp_ok=True,
            protective_sl_order_id="algo-1",
            protective_sl_price="100.10",
            protective_sl_ok=True,
            contracts_before="1",
            contracts_reduced="0.5",
            contracts_after="0.5",
            reduce_filled=True,
        )

        await run_execution_worker_once(
            result=result,
            strat_state=strat_state,
            execution_state=ExecutionState("pos-1", 100.0, pending_order_count=1),
            journal=journal,
            state_store=state_store,
            email_sender=email_sender,
            trader_positions=[PositionSnapshot("LONG", Decimal("0.5"), 100.0, 0.05, Decimal("0.5"))],
        )

        self.assertEqual(len(journal.near_tp_reduces), 1)
        self.assertEqual(journal.entries, [])
        self.assertTrue(strat_state.near_tp_protected)
        self.assertTrue(strat_state.near_tp_add_disabled)
        self.assertTrue(strat_state.partial_tp_consumed)
        self.assertEqual(strat_state.near_tp_protective_sl_order_id, "algo-1")
        self.assertEqual(len(state_store.saved_states), 1)

    async def test_near_tp_exit_all_arms_delayed_market_exit(self) -> None:
        """Near-TP protective SL failure arms delayed market exit, not immediate exit."""
        strat_state = seeded_long_state()
        journal = FakeJournal()
        state_store = RecordingStateStore()
        email_sender = RecordingEmailSender()
        result = LiveTradeResult(
            True,
            "NEAR_TP_REDUCE",
            "ord-1",
            "tp-1",
            "0.5",
            "110.00",
            "protective_sl_failed",
            tp_ok=True,
            protective_sl_ok=False,
            contracts_before="1",
            contracts_reduced="0.5",
            contracts_after="0",
            near_tp_exit_all=True,
            reduce_filled=True,
        )
        execution_state = ExecutionState("pos-1", 100.0, pending_order_count=1)

        await run_execution_worker_once(
            result=result,
            strat_state=strat_state,
            execution_state=execution_state,
            journal=journal,
            state_store=state_store,
            email_sender=email_sender,
        )

        self.assertTrue(execution_state.trading_halted)
        self.assertTrue(getattr(strat_state, "delayed_market_exit_armed", False))
        # Email should mention delayed market exit (not market-exited success)
        self.assertTrue(any("delayed market exit" in s.lower() or "Near-TP protective SL failed" in s for s in email_sender.subjects))

        cleared = asyncio.Event()

        class ClearingStateStore(RecordingStateStore):
            def clear(inner_self) -> None:
                super().clear()
                cleared.set()

        strategy_obj = FakeStrategy()
        strategy_obj.state = strat_state
        clearing_store = ClearingStateStore()
        with patch.dict(
                os.environ,
                {
                    "FLAT_BALANCE_CONFIRM_ATTEMPTS": "2",
                    "FLAT_BALANCE_CONFIRM_INTERVAL_SECONDS": "0",
                    "CASH_TRANSFER_MIN_DELTA_USDT": "0.5",
                    "CASH_TRANSFER_SETTLE_SECONDS": "0",
                    "CASH_TRANSFER_AFTER_FLAT_COOLDOWN_SECONDS": "180",
                    "CASH_DRIFT_MIN_DELTA_USDT": "0.5",
                },
        ):
            task = asyncio.create_task(
                account_position_sync_worker(
                    state_lock=asyncio.Lock(),
                    account_snapshot=AccountSnapshot(
                        PositionSnapshot("LONG", Decimal("0.5"), 100.0, 0.05, Decimal("0.5")), 100.0, 100.0,
                        asyncio.get_running_loop().time(), 0, 1),
                    execution_state=execution_state,
                    trader=RunnerTrader(LiveTradeResult(True, "noop", None, None, "0", "", "ok")),
                    # type: ignore[arg-type]
                    sizer=SimplePositionSizer(SimplePositionSizerConfig()),
                    strategy=strategy_obj,  # type: ignore[arg-type]
                    journal=journal,  # type: ignore[arg-type]
                    state_store=clearing_store,  # type: ignore[arg-type]
                    position_sync_seconds=0,
                    account_sync_seconds=999,
                    cash_log_min_delta_usdt=999,
                )
            )
            await asyncio.wait_for(cleared.wait(), timeout=1)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        self.assertFalse(execution_state.trading_halted)
        self.assertEqual(strategy_obj.state, StrategyPositionState())

    async def test_near_tp_success_syncs_remaining_position_before_state_save(self) -> None:
        strat_state = seeded_long_state(total_entry_qty=1.0, total_entry_notional=100.0, avg_entry_price=100.0)
        journal = FakeJournal()
        state_store = RecordingStateStore()
        email_sender = RecordingEmailSender()
        result = LiveTradeResult(
            True,
            "NEAR_TP_REDUCE",
            "ord-1",
            "tp-1",
            "0.5",
            "110.00",
            "ok",
            tp_ok=True,
            protective_sl_order_id="algo-1",
            protective_sl_price="100.10",
            protective_sl_ok=True,
            contracts_before="1",
            contracts_reduced="0.5",
            contracts_after="0.5",
            reduce_filled=True,
        )

        await run_execution_worker_once(
            result=result,
            strat_state=strat_state,
            execution_state=ExecutionState("pos-1", 100.0, pending_order_count=1),
            journal=journal,
            state_store=state_store,
            email_sender=email_sender,
            trader_positions=[PositionSnapshot("LONG", Decimal("0.5"), 101.0, 0.05, Decimal("0.5"))],
        )

        self.assertAlmostEqual(strat_state.total_entry_qty, 0.05)
        self.assertAlmostEqual(strat_state.total_entry_notional, 5.05)
        self.assertAlmostEqual(strat_state.avg_entry_price, 101.0)
        self.assertEqual(len(state_store.saved_states), 1)
        saved = state_store.saved_states[0]
        self.assertAlmostEqual(saved.total_entry_qty, 0.05)
        self.assertAlmostEqual(saved.avg_entry_price, 101.0)

    async def test_near_tp_protective_sl_success_but_position_sync_fails_halts_and_saves_minimal_protected_state(
            self) -> None:
        strat_state = seeded_long_state(total_entry_qty=1.0, total_entry_notional=100.0, avg_entry_price=100.0)
        journal = FakeJournal()
        state_store = RecordingStateStore()
        email_sender = RecordingEmailSender()
        result = LiveTradeResult(
            True,
            "NEAR_TP_REDUCE",
            "ord-1",
            "tp-1",
            "0.5",
            "110.00",
            "ok",
            tp_ok=True,
            protective_sl_order_id="algo-1",
            protective_sl_price="100.10",
            protective_sl_ok=True,
            contracts_before="1",
            contracts_reduced="0.5",
            contracts_after="0.5",
            reduce_filled=True,
        )
        execution_state = ExecutionState("pos-1", 100.0, pending_order_count=1)

        await run_execution_worker_once(
            result=result,
            strat_state=strat_state,
            execution_state=execution_state,
            journal=journal,
            state_store=state_store,
            email_sender=email_sender,
            raise_on_fetch_position=True,
        )

        self.assertTrue(execution_state.trading_halted)
        self.assertEqual(execution_state.halt_reason, "near_tp_protected_sync_failed")
        self.assertTrue(strat_state.near_tp_protected)
        self.assertTrue(strat_state.near_tp_add_disabled)
        self.assertEqual(strat_state.near_tp_protective_sl_order_id, "algo-1")
        self.assertEqual(len(state_store.saved_states), 1)
        saved = state_store.saved_states[0]
        self.assertTrue(saved.near_tp_protected)
        self.assertTrue(saved.near_tp_add_disabled)
        self.assertEqual(saved.near_tp_protective_sl_order_id, "algo-1")
        self.assertEqual(len(journal.near_tp_reduces), 1)
        self.assertEqual(journal.errors, [])
        self.assertIn("Near-TP protected but position sync failed", email_sender.subjects)

    async def test_account_sync_recovers_near_tp_protected_sync_failed_halt(self) -> None:
        saved = asyncio.Event()

        class SavingStateStore(RecordingStateStore):
            def save(inner_self, state) -> None:  # type: ignore[no-untyped-def]
                super().save(state)
                saved.set()

        strategy_obj = FakeStrategy()
        strategy_obj.state = seeded_long_state(near_tp_protected=True, near_tp_add_disabled=True)
        execution_state = ExecutionState("pos-1", 100.0, trading_halted=True,
                                         halt_reason="near_tp_protected_sync_failed")
        state_store = SavingStateStore()
        task = asyncio.create_task(
            account_position_sync_worker(
                state_lock=asyncio.Lock(),
                account_snapshot=AccountSnapshot(flat_position(), 100.0, 100.0, asyncio.get_running_loop().time(), 0,
                                                 1),
                execution_state=execution_state,
                trader=RunnerTrader(
                    LiveTradeResult(True, "noop", None, None, "0", "", "ok"),
                    [PositionSnapshot("LONG", Decimal("0.5"), 101.0, 0.05, Decimal("0.5")) for _ in range(10)],
                ),  # type: ignore[arg-type]
                sizer=SimplePositionSizer(SimplePositionSizerConfig()),
                strategy=strategy_obj,  # type: ignore[arg-type]
                journal=FakeJournal(),  # type: ignore[arg-type]
                state_store=state_store,  # type: ignore[arg-type]
                position_sync_seconds=0,
                account_sync_seconds=999,
                cash_log_min_delta_usdt=999,
            )
        )
        await asyncio.wait_for(saved.wait(), timeout=1)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        self.assertFalse(execution_state.trading_halted)
        self.assertIsNone(execution_state.halt_reason)
        self.assertTrue(strategy_obj.state.near_tp_protected)
        self.assertGreaterEqual(len(state_store.saved_states), 1)

    async def test_account_sync_does_not_recover_critical_halt(self) -> None:
        saved = asyncio.Event()

        class SavingStateStore(RecordingStateStore):
            def save(inner_self, state) -> None:  # type: ignore[no-untyped-def]
                super().save(state)
                saved.set()

        strategy_obj = FakeStrategy()
        strategy_obj.state = seeded_long_state(near_tp_protected=True, near_tp_add_disabled=True)
        execution_state = ExecutionState("pos-1", 100.0, trading_halted=True, halt_reason="near_tp_reduce_failure")
        state_store = SavingStateStore()
        task = asyncio.create_task(
            account_position_sync_worker(
                state_lock=asyncio.Lock(),
                account_snapshot=AccountSnapshot(flat_position(), 100.0, 100.0, asyncio.get_running_loop().time(), 0,
                                                 1),
                execution_state=execution_state,
                trader=RunnerTrader(
                    LiveTradeResult(True, "noop", None, None, "0", "", "ok"),
                    [PositionSnapshot("LONG", Decimal("0.5"), 101.0, 0.05, Decimal("0.5")) for _ in range(10)],
                ),  # type: ignore[arg-type]
                sizer=SimplePositionSizer(SimplePositionSizerConfig()),
                strategy=strategy_obj,  # type: ignore[arg-type]
                journal=FakeJournal(),  # type: ignore[arg-type]
                state_store=state_store,  # type: ignore[arg-type]
                position_sync_seconds=0,
                account_sync_seconds=999,
                cash_log_min_delta_usdt=999,
            )
        )
        await asyncio.wait_for(saved.wait(), timeout=1)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        self.assertTrue(execution_state.trading_halted)
        self.assertEqual(execution_state.halt_reason, "near_tp_reduce_failure")

    async def test_execution_worker_near_tp_total_failure_halts(self) -> None:
        strat_state = seeded_long_state(near_tp_reduce_pending=True)
        journal = FakeJournal()
        state_store = RecordingStateStore()
        email_sender = RecordingEmailSender()
        result = LiveTradeResult(
            False,
            "NEAR_TP_REDUCE",
            "ord-1",
            "tp-1",
            "0.5",
            "110.00",
            "protective_sl_failed_and_market_exit_failed",
            tp_ok=True,
            protective_sl_ok=False,
            contracts_before="1",
            contracts_reduced="0.5",
            contracts_after="0.5",
            reduce_filled=True,
        )
        snapshot = seeded_long_state(near_tp_reduce_pending=False)
        execution_state = ExecutionState("pos-1", 100.0, pending_order_count=1)
        await run_execution_worker_once(
            result=result,
            strat_state=strat_state,
            execution_state=execution_state,
            journal=journal,
            state_store=state_store,
            email_sender=email_sender,
        )

        self.assertTrue(execution_state.trading_halted)
        self.assertTrue(journal.errors)
        self.assertIn("CRITICAL: Near-TP protective SL and market exit failed", email_sender.subjects)
        self.assertTrue(strat_state.near_tp_reduce_pending)
        self.assertFalse(snapshot.near_tp_reduce_pending)

    async def test_flat_reset_clears_near_tp_protected_and_allows_new_open(self) -> None:
        cleared = asyncio.Event()
        strat = FakeStrategy()
        strat.state = seeded_long_state(
            near_tp_protected=True,
            near_tp_add_disabled=True,
            near_tp_protective_sl_order_id="algo-1",
        )
        journal = FakeJournal()
        state_store = RecordingStateStore()

        class ClearingStateStore(RecordingStateStore):
            def clear(inner_self) -> None:
                super().clear()
                cleared.set()

        trader = RunnerTrader(LiveTradeResult(True, "noop", None, None, "0", "", "ok"))
        with patch.dict(
                os.environ,
                {
                    "FLAT_BALANCE_CONFIRM_ATTEMPTS": "2",
                    "FLAT_BALANCE_CONFIRM_INTERVAL_SECONDS": "0",
                    "CASH_TRANSFER_MIN_DELTA_USDT": "0.5",
                    "CASH_TRANSFER_SETTLE_SECONDS": "0",
                    "CASH_TRANSFER_AFTER_FLAT_COOLDOWN_SECONDS": "180",
                    "CASH_DRIFT_MIN_DELTA_USDT": "0.5",
                },
        ):
            store = ClearingStateStore()
            task = asyncio.create_task(
                account_position_sync_worker(
                    state_lock=asyncio.Lock(),
                    account_snapshot=AccountSnapshot(
                        PositionSnapshot("LONG", Decimal("0.5"), 100.0, 0.05, Decimal("0.5")), 100.0, 100.0,
                        asyncio.get_running_loop().time(), 0, 1),
                    execution_state=ExecutionState("pos-1", 100.0),
                    trader=trader,  # type: ignore[arg-type]
                    sizer=SimplePositionSizer(SimplePositionSizerConfig()),
                    strategy=strat,  # type: ignore[arg-type]
                    journal=journal,  # type: ignore[arg-type]
                    state_store=store,  # type: ignore[arg-type]
                    position_sync_seconds=0,
                    account_sync_seconds=999,
                    cash_log_min_delta_usdt=999,
                )
            )
            await asyncio.wait_for(cleared.wait(), timeout=1)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        self.assertEqual(strat.state, StrategyPositionState())
        self.assertFalse(strat.state.near_tp_add_disabled)
        self.assertEqual(store.clear_calls, 1)
        self.assertEqual(trader.cancelled_algo_ids, ["algo-1"])

    def test_startup_flat_clears_stale_near_tp_live_state(self) -> None:
        strat = BollCvdReclaimStrategy(BollCvdReclaimStrategyConfig(), SimplePositionSizer(SimplePositionSizerConfig()))
        saved = LivePositionState(
            position_id="pos-1",
            side="LONG",
            layers=1,
            near_tp_protected=True,
            near_tp_add_disabled=True,
        )
        restore_strategy_from_saved_state(strat, saved)
        self.assertTrue(strat.state.near_tp_protected)

        state_store = RecordingStateStore()
        startup_position = flat_position()
        if not startup_position.has_position:
            state_store.clear()
            strat.state = StrategyPositionState()

        self.assertEqual(state_store.clear_calls, 1)
        self.assertFalse(strat.state.near_tp_protected)
        self.assertFalse(strat.state.near_tp_add_disabled)

    def test_saved_state_restores_first_entry_clock_for_first_add_block(self) -> None:
        state = StrategyPositionState(
            side="LONG",
            layers=2,
            last_entry_price=99.0,
            tp_price=110.0,
            last_order_ts_ms=600_000,
            first_entry_ts_ms=1_000,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            breakeven_price=100.1,
        )
        saved = LiveStateStore.from_strategy_state(
            position_id="pos-1",
            symbol="ETH-USDT-SWAP",
            strategy_state=state,
            cash_before_position=None,
        )
        strat = BollCvdShockReclaimStrategy(BollCvdReclaimStrategyConfig(),
                                            SimplePositionSizer(SimplePositionSizerConfig()))

        restore_strategy_from_saved_state(strat, saved)

        self.assertEqual(strat.state.first_entry_ts_ms, 1_000)
        self.assertEqual(strat.state.last_order_ts_ms, 600_000)
        self.assertEqual(strat._first_entry_elapsed_seconds(601_000), 600.0)


if __name__ == "__main__":
    unittest.main()
