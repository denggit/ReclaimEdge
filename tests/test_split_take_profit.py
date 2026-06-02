from __future__ import annotations

import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from types import MethodType

from scripts.run_boll_cvd_live import mark_partial_tp_consumed_if_position_reduced
from src.execution.trader import PositionSnapshot, Trader
from src.indicators.cvd_tracker import CvdSnapshot
from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.reporting.trade_journal import LiveTradeJournal
from src.risk.simple_position_sizer import PositionSize, SimplePositionSizer, SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
    StrategyPositionState,
    TradeIntent,
)


def strategy() -> BollCvdReclaimStrategy:
    return BollCvdReclaimStrategy(
        BollCvdReclaimStrategyConfig(
            split_tp_min_layers=4,
            split_tp_path_ratio=0.8,
            split_tp_partial_ratio=0.5,
            split_tp_min_profit_pct=0.004,
        ),
        SimplePositionSizer(SimplePositionSizerConfig()),
    )


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


def intent(**overrides) -> TradeIntent:
    values = dict(
        intent_type="UPDATE_TP",
        side="LONG",
        price=100.0,
        layer_index=4,
        tp_price=110.0,
        reason="test",
        size=PositionSize(1.0, 50.0, 0.5, 4, 1.45),
        fast_cvd=0.0,
        previous_fast_cvd=0.0,
        buy_ratio=0.0,
        sell_ratio=0.0,
        boll_upper=110.0,
        boll_middle=105.0,
        boll_lower=95.0,
        ts_ms=1_000,
        avg_entry_price=100.0,
        breakeven_price=100.1,
        tp_mode="MIDDLE",
        partial_tp_price=None,
        partial_tp_ratio=0.0,
        tp_plan="SINGLE",
        partial_tp_consumed=False,
    )
    values.update(overrides)
    return TradeIntent(**values)  # type: ignore[arg-type]


class SplitTakeProfitStrategyTest(unittest.TestCase):
    def test_long_split_tp_uses_80_pct_path_when_min_profit_is_inside_final_tp(self) -> None:
        strat = strategy()
        strat.state.avg_entry_price = 100.0

        partial_tp, partial_ratio, plan = strat._select_tp_plan("LONG", 110.0, 4)

        self.assertEqual(plan, "SPLIT_PARTIAL_FINAL")
        self.assertEqual(partial_ratio, 0.5)
        self.assertAlmostEqual(partial_tp or 0, 108.0)

    def test_long_does_not_split_when_min_profit_would_exceed_final_tp(self) -> None:
        strat = strategy()
        strat.state.avg_entry_price = 100.0

        partial_tp, partial_ratio, plan = strat._select_tp_plan("LONG", 100.3, 4)

        self.assertEqual(plan, "SINGLE")
        self.assertEqual(partial_ratio, 0.0)
        self.assertIsNone(partial_tp)

    def test_short_split_tp_uses_80_pct_path_when_min_profit_is_inside_final_tp(self) -> None:
        strat = strategy()
        strat.state.avg_entry_price = 100.0

        partial_tp, partial_ratio, plan = strat._select_tp_plan("SHORT", 90.0, 4)

        self.assertEqual(plan, "SPLIT_PARTIAL_FINAL")
        self.assertEqual(partial_ratio, 0.5)
        self.assertAlmostEqual(partial_tp or 0, 92.0)

    def test_short_does_not_split_when_min_profit_would_exceed_final_tp(self) -> None:
        strat = strategy()
        strat.state.avg_entry_price = 100.0

        partial_tp, partial_ratio, plan = strat._select_tp_plan("SHORT", 99.7, 4)

        self.assertEqual(plan, "SINGLE")
        self.assertEqual(partial_ratio, 0.0)
        self.assertIsNone(partial_tp)

    def test_partial_tp_consumed_prevents_repeated_split(self) -> None:
        strat = strategy()
        strat.state.avg_entry_price = 100.0
        strat.state.partial_tp_consumed = True

        partial_tp, partial_ratio, plan = strat._select_tp_plan("LONG", 110.0, 4)

        self.assertEqual(plan, "SINGLE")
        self.assertEqual(partial_ratio, 0.0)
        self.assertIsNone(partial_tp)

    def test_new_add_rearms_split_after_partial_consumed(self) -> None:
        strat = strategy()
        strat.state = StrategyPositionState(
            side="LONG",
            layers=4,
            last_entry_price=100.0,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            partial_tp_consumed=True,
            tp_plan="SINGLE",
        )

        new_intent = strat._open_position("LONG", "ADD_LONG", 90.0, 2_000, boll(), cvd(), "test add")

        self.assertFalse(new_intent.partial_tp_consumed)
        self.assertFalse(strat.state.partial_tp_consumed)
        self.assertEqual(new_intent.tp_plan, "SPLIT_PARTIAL_FINAL")
        self.assertIsNotNone(new_intent.partial_tp_price)

    def test_layers_below_threshold_keep_single_tp(self) -> None:
        strat = strategy()
        strat.state.avg_entry_price = 100.0

        partial_tp, partial_ratio, plan = strat._select_tp_plan("LONG", 110.0, 3)

        self.assertEqual(plan, "SINGLE")
        self.assertEqual(partial_ratio, 0.0)
        self.assertIsNone(partial_tp)


class SplitTakeProfitTraderTest(unittest.IsolatedAsyncioTestCase):
    def make_trader(self) -> Trader:
        trader = Trader.__new__(Trader)
        trader.symbol = "ETH-USDT-SWAP"
        trader.td_mode = "isolated"
        trader.pos_side_mode = "net"
        trader.position_contracts = Decimal("4")
        trader.contract_precision = Decimal("0.01")
        trader.min_contracts = Decimal("0.01")
        trader.tp_order_id = None
        return trader

    def test_build_split_order_specs_rounds_half_position(self) -> None:
        trader = self.make_trader()
        specs = trader._build_take_profit_order_specs(
            intent(partial_tp_price=108.0, partial_tp_ratio=0.5, tp_plan="SPLIT_PARTIAL_FINAL")
        )

        self.assertEqual(specs, [("partial", Decimal("2.00"), 108.0), ("final", Decimal("2.00"), 110.0)])

    async def test_replace_take_profit_places_two_reduce_only_orders_for_split_plan(self) -> None:
        trader = self.make_trader()
        posted: list[dict] = []

        async def fetch_position_snapshot(self) -> PositionSnapshot:  # type: ignore[no-untyped-def]
            return PositionSnapshot("LONG", Decimal("4"), 100.0, 0.4, Decimal("4"))

        async def fetch_pending_orders(self) -> list[dict]:  # type: ignore[no-untyped-def]
            return []

        async def request(self, method: str, endpoint: str, payload=None):  # type: ignore[no-untyped-def]
            posted.append(dict(payload or {}))
            return {"code": "0", "data": [{"ordId": f"tp-{len(posted)}"}]}

        trader.fetch_position_snapshot = MethodType(fetch_position_snapshot, trader)  # type: ignore[method-assign]
        trader.fetch_pending_orders = MethodType(fetch_pending_orders, trader)  # type: ignore[method-assign]
        trader.request = MethodType(request, trader)  # type: ignore[method-assign]

        result = await trader.replace_take_profit(
            intent(partial_tp_price=108.0, partial_tp_ratio=0.5, tp_plan="SPLIT_PARTIAL_FINAL")
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.tp_order_ids, ("tp-1", "tp-2"))
        self.assertEqual(result.tp_order_id, "tp-1,tp-2")
        self.assertEqual(result.tp_price, "partial:108.00,final:110.00")
        self.assertEqual([item["sz"] for item in posted], ["2", "2"])
        self.assertEqual([item["px"] for item in posted], ["108.00", "110.00"])
        self.assertTrue(all(item["reduceOnly"] == "true" for item in posted))


class SplitTakeProfitLifecycleTest(unittest.TestCase):
    def test_mark_partial_tp_consumed_when_position_reduced_enough(self) -> None:
        strat = strategy()
        strat.state = StrategyPositionState(
            side="LONG",
            layers=4,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            partial_tp_price=108.0,
            partial_tp_ratio=0.5,
            tp_plan="SPLIT_PARTIAL_FINAL",
        )
        position = PositionSnapshot("LONG", Decimal("5"), 100.0, 0.5, Decimal("5"))

        consumed = mark_partial_tp_consumed_if_position_reduced(strat, position)

        self.assertTrue(consumed)
        self.assertTrue(strat.state.partial_tp_consumed)
        self.assertEqual(strat.state.tp_plan, "SINGLE")
        self.assertIsNone(strat.state.partial_tp_price)
        self.assertEqual(strat.state.partial_tp_ratio, 0.0)

    def test_does_not_mark_partial_consumed_for_small_rounding_difference(self) -> None:
        strat = strategy()
        strat.state = StrategyPositionState(
            side="LONG",
            layers=4,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            avg_entry_price=100.0,
            partial_tp_price=108.0,
            partial_tp_ratio=0.5,
            tp_plan="SPLIT_PARTIAL_FINAL",
        )
        position = PositionSnapshot("LONG", Decimal("9.98"), 100.0, 0.998, Decimal("9.98"))

        consumed = mark_partial_tp_consumed_if_position_reduced(strat, position)

        self.assertFalse(consumed)
        self.assertFalse(strat.state.partial_tp_consumed)
        self.assertEqual(strat.state.tp_plan, "SPLIT_PARTIAL_FINAL")

    def test_record_flat_accepts_split_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            journal = LiveTradeJournal(Path(tmp) / "live_trade_events.jsonl")
            journal.record_flat(
                position_id="pos-1",
                symbol="ETH-USDT-SWAP",
                side="LONG",
                cash_before_position=100.0,
                cash_after=101.0,
                equity_after=101.0,
                reason="test",
                layers=4,
                avg_entry_price=100.0,
                last_tp_price=110.0,
                last_partial_tp_price=108.0,
                last_tp_plan="SPLIT_PARTIAL_FINAL",
                partial_tp_consumed=True,
            )

            events = journal.load_events()

        self.assertEqual(len(events), 1)
        payload = events[0].payload
        self.assertEqual(payload["last_partial_tp_price"], 108.0)
        self.assertEqual(payload["last_tp_plan"], "SPLIT_PARTIAL_FINAL")
        self.assertTrue(payload["partial_tp_consumed"])


if __name__ == "__main__":
    unittest.main()
