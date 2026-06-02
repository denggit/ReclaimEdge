from __future__ import annotations

import unittest
from decimal import Decimal
from types import MethodType

from src.execution.trader import PositionSnapshot, Trader
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
    )
    values.update(overrides)
    return TradeIntent(**values)  # type: ignore[arg-type]


class SplitTakeProfitStrategyTest(unittest.TestCase):
    def test_long_split_tp_uses_80_pct_path_when_min_profit_is_inside_final_tp(self) -> None:
        strat = strategy()
        strat.state.avg_entry_price = 100.0

        partial_tp, partial_ratio, plan = strat._select_tp_plan("LONG", 110.0, 4)

        self.assertEqual(plan, "SPLIT_50_50")
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

        self.assertEqual(plan, "SPLIT_50_50")
        self.assertEqual(partial_ratio, 0.5)
        self.assertAlmostEqual(partial_tp or 0, 92.0)

    def test_short_does_not_split_when_min_profit_would_exceed_final_tp(self) -> None:
        strat = strategy()
        strat.state.avg_entry_price = 100.0

        partial_tp, partial_ratio, plan = strat._select_tp_plan("SHORT", 99.7, 4)

        self.assertEqual(plan, "SINGLE")
        self.assertEqual(partial_ratio, 0.0)
        self.assertIsNone(partial_tp)

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
            intent(partial_tp_price=108.0, partial_tp_ratio=0.5, tp_plan="SPLIT_50_50")
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
            intent(partial_tp_price=108.0, partial_tp_ratio=0.5, tp_plan="SPLIT_50_50")
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.tp_order_ids, ("tp-1", "tp-2"))
        self.assertEqual(result.tp_order_id, "tp-1,tp-2")
        self.assertEqual(result.tp_price, "partial:108.00,final:110.00")
        self.assertEqual([item["sz"] for item in posted], ["2", "2"])
        self.assertEqual([item["px"] for item in posted], ["108.00", "110.00"])
        self.assertTrue(all(item["reduceOnly"] == "true" for item in posted))


if __name__ == "__main__":
    unittest.main()
