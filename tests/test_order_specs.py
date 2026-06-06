from __future__ import annotations

import unittest
from decimal import Decimal

from src.execution.order_specs import (
    TakeProfitOrderSpec,
    TakeProfitSpecsDecision,
    build_cancel_algo_body,
    build_cancel_order_body,
    build_conditional_protective_sl_algo_body,
    build_market_entry_order_body,
    build_reduce_only_market_order_body,
    build_reduce_only_tp_order_body,
    build_set_leverage_bodies,
    build_take_profit_order_specs,
    close_order_side,
    maybe_add_pos_side,
    open_order_side,
    pos_side_for_mode,
    round_contracts_down,
    trend_runner_sl_contracts,
)


class PosSideForModeTest(unittest.TestCase):
    def test_net_mode_returns_none(self) -> None:
        self.assertIsNone(pos_side_for_mode(side="LONG", pos_side_mode="net"))
        self.assertIsNone(pos_side_for_mode(side="SHORT", pos_side_mode="net"))

    def test_long_short_long_returns_long(self) -> None:
        self.assertEqual(pos_side_for_mode(side="LONG", pos_side_mode="long_short"), "long")

    def test_long_short_short_returns_short(self) -> None:
        self.assertEqual(pos_side_for_mode(side="SHORT", pos_side_mode="long_short"), "short")


class OpenOrderSideTest(unittest.TestCase):
    def test_long_is_buy(self) -> None:
        self.assertEqual(open_order_side(side="LONG"), "buy")

    def test_short_is_sell(self) -> None:
        self.assertEqual(open_order_side(side="SHORT"), "sell")


class CloseOrderSideTest(unittest.TestCase):
    def test_long_is_sell(self) -> None:
        self.assertEqual(close_order_side(side="LONG"), "sell")

    def test_short_is_buy(self) -> None:
        self.assertEqual(close_order_side(side="SHORT"), "buy")


class MaybeAddPosSideTest(unittest.TestCase):
    def test_net_mode_does_not_add_pos_side(self) -> None:
        body = maybe_add_pos_side({"instId": "X"}, side="LONG", pos_side_mode="net")
        self.assertNotIn("posSide", body)

    def test_long_short_adds_pos_side(self) -> None:
        body = maybe_add_pos_side({"instId": "X"}, side="LONG", pos_side_mode="long_short")
        self.assertEqual(body["posSide"], "long")

    def test_does_not_mutate_original(self) -> None:
        original = {"instId": "X"}
        result = maybe_add_pos_side(original, side="LONG", pos_side_mode="long_short")
        self.assertNotIn("posSide", original)
        self.assertIn("posSide", result)


class BuildMarketEntryOrderBodyTest(unittest.TestCase):
    def test_long_net_buy_market_no_pos_side(self) -> None:
        body = build_market_entry_order_body(
            inst_id="ETH-USDT-SWAP",
            td_mode="isolated",
            side="LONG",
            contracts_text="0.10",
            pos_side_mode="net",
        )
        self.assertEqual(body["instId"], "ETH-USDT-SWAP")
        self.assertEqual(body["tdMode"], "isolated")
        self.assertEqual(body["side"], "buy")
        self.assertEqual(body["ordType"], "market")
        self.assertEqual(body["sz"], "0.10")
        self.assertNotIn("posSide", body)

    def test_short_net_sell_market_no_pos_side(self) -> None:
        body = build_market_entry_order_body(
            inst_id="ETH-USDT-SWAP",
            td_mode="cross",
            side="SHORT",
            contracts_text="1.00",
            pos_side_mode="net",
        )
        self.assertEqual(body["side"], "sell")
        self.assertNotIn("posSide", body)

    def test_long_long_short_includes_pos_side(self) -> None:
        body = build_market_entry_order_body(
            inst_id="ETH-USDT-SWAP",
            td_mode="isolated",
            side="LONG",
            contracts_text="0.10",
            pos_side_mode="long_short",
        )
        self.assertEqual(body["posSide"], "long")

    def test_short_long_short_includes_pos_side(self) -> None:
        body = build_market_entry_order_body(
            inst_id="ETH-USDT-SWAP",
            td_mode="isolated",
            side="SHORT",
            contracts_text="0.10",
            pos_side_mode="long_short",
        )
        self.assertEqual(body["posSide"], "short")


class BuildReduceOnlyMarketOrderBodyTest(unittest.TestCase):
    def test_long_close_side_sell(self) -> None:
        body = build_reduce_only_market_order_body(
            inst_id="ETH-USDT-SWAP",
            td_mode="isolated",
            side="LONG",
            contracts_text="0.10",
            pos_side_mode="net",
        )
        self.assertEqual(body["side"], "sell")
        self.assertEqual(body["reduceOnly"], "true")
        self.assertIsInstance(body["reduceOnly"], str)

    def test_short_close_side_buy(self) -> None:
        body = build_reduce_only_market_order_body(
            inst_id="ETH-USDT-SWAP",
            td_mode="isolated",
            side="SHORT",
            contracts_text="0.10",
            pos_side_mode="net",
        )
        self.assertEqual(body["side"], "buy")
        self.assertEqual(body["reduceOnly"], "true")

    def test_long_short_includes_pos_side(self) -> None:
        body = build_reduce_only_market_order_body(
            inst_id="ETH-USDT-SWAP",
            td_mode="isolated",
            side="LONG",
            contracts_text="0.10",
            pos_side_mode="long_short",
        )
        self.assertEqual(body["posSide"], "long")


class BuildReduceOnlyTpOrderBodyTest(unittest.TestCase):
    def test_all_fields(self) -> None:
        body = build_reduce_only_tp_order_body(
            inst_id="ETH-USDT-SWAP",
            td_mode="isolated",
            side="LONG",
            contracts_text="0.10",
            price_text="3000.00",
            pos_side_mode="net",
        )
        self.assertEqual(body["instId"], "ETH-USDT-SWAP")
        self.assertEqual(body["tdMode"], "isolated")
        self.assertEqual(body["side"], "sell")
        self.assertEqual(body["ordType"], "limit")
        self.assertEqual(body["px"], "3000.00")
        self.assertEqual(body["sz"], "0.10")
        self.assertEqual(body["reduceOnly"], "true")

    def test_long_short_includes_pos_side(self) -> None:
        body = build_reduce_only_tp_order_body(
            inst_id="ETH-USDT-SWAP",
            td_mode="isolated",
            side="SHORT",
            contracts_text="0.10",
            price_text="3000.00",
            pos_side_mode="long_short",
        )
        self.assertEqual(body["posSide"], "short")

    def test_client_order_id_adds_cl_ord_id(self) -> None:
        body = build_reduce_only_tp_order_body(
            inst_id="ETH-USDT-SWAP",
            td_mode="isolated",
            side="LONG",
            contracts_text="0.10",
            price_text="3000.00",
            pos_side_mode="net",
            client_order_id="test-clordid-123",
        )
        self.assertIn("clOrdId", body)
        self.assertEqual(body["clOrdId"], "test-clordid-123")

    def test_client_order_id_none_does_not_add(self) -> None:
        body = build_reduce_only_tp_order_body(
            inst_id="ETH-USDT-SWAP",
            td_mode="isolated",
            side="LONG",
            contracts_text="0.10",
            price_text="3000.00",
            pos_side_mode="net",
            client_order_id=None,
        )
        self.assertNotIn("clOrdId", body)


class BuildConditionalProtectiveSlAlgoBodyTest(unittest.TestCase):
    def test_basic_body(self) -> None:
        body = build_conditional_protective_sl_algo_body(
            inst_id="ETH-USDT-SWAP",
            td_mode="isolated",
            side="LONG",
            contracts_text="0.10",
            stop_price_text="2950.00",
            pos_side_mode="net",
        )
        self.assertEqual(body["instId"], "ETH-USDT-SWAP")
        self.assertEqual(body["tdMode"], "isolated")
        self.assertEqual(body["side"], "sell")
        self.assertEqual(body["ordType"], "conditional")
        self.assertEqual(body["sz"], "0.10")
        self.assertEqual(body["slTriggerPx"], "2950.00")
        self.assertEqual(body["slOrdPx"], "-1")
        self.assertEqual(body["slTriggerPxType"], "last")
        self.assertEqual(body["reduceOnly"], "true")

    def test_long_short_adds_pos_side(self) -> None:
        body = build_conditional_protective_sl_algo_body(
            inst_id="ETH-USDT-SWAP",
            td_mode="isolated",
            side="SHORT",
            contracts_text="0.10",
            stop_price_text="3050.00",
            pos_side_mode="long_short",
        )
        self.assertEqual(body["posSide"], "short")


class BuildCancelOrderBodyTest(unittest.TestCase):
    def test_cancel_order_body(self) -> None:
        body = build_cancel_order_body(inst_id="ETH-USDT-SWAP", order_id="ord-123")
        self.assertEqual(body, {"instId": "ETH-USDT-SWAP", "ordId": "ord-123"})


class BuildCancelAlgoBodyTest(unittest.TestCase):
    def test_cancel_algo_body(self) -> None:
        body = build_cancel_algo_body(inst_id="ETH-USDT-SWAP", algo_id="algo-456")
        self.assertEqual(body, [{"instId": "ETH-USDT-SWAP", "algoId": "algo-456"}])


class BuildSetLeverageBodiesTest(unittest.TestCase):
    def test_net_one_body(self) -> None:
        bodies = build_set_leverage_bodies(
            inst_id="ETH-USDT-SWAP",
            td_mode="isolated",
            leverage="50",
            pos_side_mode="net",
        )
        self.assertEqual(len(bodies), 1)
        self.assertEqual(bodies[0], {"instId": "ETH-USDT-SWAP", "lever": "50", "mgnMode": "isolated"})

    def test_long_short_two_bodies_long_then_short(self) -> None:
        bodies = build_set_leverage_bodies(
            inst_id="ETH-USDT-SWAP",
            td_mode="cross",
            leverage="20",
            pos_side_mode="long_short",
        )
        self.assertEqual(len(bodies), 2)
        self.assertEqual(bodies[0], {"instId": "ETH-USDT-SWAP", "lever": "20", "mgnMode": "cross", "posSide": "long"})
        self.assertEqual(bodies[1], {"instId": "ETH-USDT-SWAP", "lever": "20", "mgnMode": "cross", "posSide": "short"})

    def test_leverage_always_string(self) -> None:
        bodies = build_set_leverage_bodies(
            inst_id="ETH-USDT-SWAP",
            td_mode="isolated",
            leverage="50",
            pos_side_mode="net",
        )
        self.assertIsInstance(bodies[0]["lever"], str)


class RoundContractsDownTest(unittest.TestCase):
    def test_rounds_down_correctly(self) -> None:
        result = round_contracts_down(contracts=Decimal("1.239"), contract_precision=Decimal("0.01"))
        self.assertEqual(result, Decimal("1.23"))

    def test_exact_multiple(self) -> None:
        result = round_contracts_down(contracts=Decimal("1.00"), contract_precision=Decimal("0.01"))
        self.assertEqual(result, Decimal("1.00"))

    def test_zero(self) -> None:
        result = round_contracts_down(contracts=Decimal("0.00"), contract_precision=Decimal("0.01"))
        self.assertEqual(result, Decimal("0.00"))

    def test_tiny_contracts(self) -> None:
        result = round_contracts_down(contracts=Decimal("0.009"), contract_precision=Decimal("0.01"))
        self.assertEqual(result, Decimal("0.00"))

    def test_different_precision(self) -> None:
        result = round_contracts_down(contracts=Decimal("1.499"), contract_precision=Decimal("0.1"))
        self.assertEqual(result, Decimal("1.4"))


class BuildTakeProfitOrderSpecsTest(unittest.TestCase):
    PREC = Decimal("0.01")
    MIN = Decimal("0.01")

    def _call(self, **overrides) -> TakeProfitSpecsDecision:
        defaults: dict = {
            "position_contracts": Decimal("10"),
            "min_contracts": self.MIN,
            "contract_precision": self.PREC,
            "tp_plan": "SINGLE",
            "final_tp_price": 100.0,
            "partial_tp_price": None,
            "partial_tp_ratio": Decimal("0"),
            "partial_tp_consumed": False,
            "middle_runner_active": False,
            "three_stage_tp1_price": None,
            "three_stage_tp2_price": None,
            "three_stage_tp1_ratio": Decimal("0"),
            "three_stage_tp2_ratio": Decimal("0"),
            "three_stage_tp1_consumed": False,
            "three_stage_tp2_consumed": False,
            "three_stage_runner_ratio": Decimal("0"),
        }
        defaults.update(overrides)
        return build_take_profit_order_specs(**defaults)

    def _specs(self, decision: TakeProfitSpecsDecision) -> list[tuple[str, Decimal, float]]:
        return [(s.label, s.contracts, s.price) for s in decision.specs]

    # ── SINGLE ──
    def test_single_returns_final_full(self) -> None:
        d = self._call(tp_plan="SINGLE", final_tp_price=100.0)
        self.assertEqual(self._specs(d), [("final", Decimal("10"), 100.0)])
        self.assertIsNone(d.fallback_reason)

    # ── SPLIT_PARTIAL_FINAL ──
    def test_split_partial_final_valid(self) -> None:
        d = self._call(
            tp_plan="SPLIT_PARTIAL_FINAL",
            partial_tp_price=90.0,
            partial_tp_ratio=Decimal("0.6"),
            final_tp_price=100.0,
        )
        self.assertEqual(
            self._specs(d),
            [("partial", Decimal("6.00"), 90.0), ("final", Decimal("4.00"), 100.0)],
        )
        self.assertIsNone(d.fallback_reason)

    def test_split_size_too_small_partial(self) -> None:
        d = self._call(
            position_contracts=Decimal("0.01"),
            tp_plan="SPLIT_PARTIAL_FINAL",
            partial_tp_price=90.0,
            partial_tp_ratio=Decimal("0.5"),
            final_tp_price=100.0,
        )
        self.assertEqual(self._specs(d), [("final", Decimal("0.01"), 100.0)])
        self.assertEqual(d.fallback_reason, "SPLIT_TP_FALLBACK_SINGLE_SIZE_TOO_SMALL")
        self.assertIsNotNone(d.fallback_context)
        ctx = d.fallback_context
        self.assertIn("total_contracts", ctx)
        self.assertIn("partial_contracts", ctx)
        self.assertIn("final_contracts", ctx)
        self.assertIn("min_contracts", ctx)

    def test_split_size_too_small_final(self) -> None:
        d = self._call(
            position_contracts=Decimal("0.019"),
            tp_plan="SPLIT_PARTIAL_FINAL",
            partial_tp_price=90.0,
            partial_tp_ratio=Decimal("0.99"),
            final_tp_price=100.0,
        )
        self.assertEqual(self._specs(d), [("final", Decimal("0.019"), 100.0)])
        self.assertEqual(d.fallback_reason, "SPLIT_TP_FALLBACK_SINGLE_SIZE_TOO_SMALL")

    def test_split_invalid_ratio_zero(self) -> None:
        d = self._call(
            tp_plan="SPLIT_PARTIAL_FINAL",
            partial_tp_price=90.0,
            partial_tp_ratio=Decimal("0"),
            final_tp_price=100.0,
        )
        self.assertEqual(self._specs(d), [("final", Decimal("10"), 100.0)])
        self.assertIsNone(d.fallback_reason)

    def test_split_invalid_ratio_one(self) -> None:
        d = self._call(
            tp_plan="SPLIT_PARTIAL_FINAL",
            partial_tp_price=90.0,
            partial_tp_ratio=Decimal("1"),
            final_tp_price=100.0,
        )
        self.assertEqual(self._specs(d), [("final", Decimal("10"), 100.0)])
        self.assertIsNone(d.fallback_reason)

    def test_split_partial_price_none(self) -> None:
        d = self._call(
            tp_plan="SPLIT_PARTIAL_FINAL",
            partial_tp_price=None,
            partial_tp_ratio=Decimal("0.5"),
            final_tp_price=100.0,
        )
        self.assertEqual(self._specs(d), [("final", Decimal("10"), 100.0)])

    # ── MIDDLE_RUNNER ──
    def test_middle_runner_valid(self) -> None:
        d = self._call(
            tp_plan="MIDDLE_RUNNER",
            partial_tp_price=90.0,
            partial_tp_ratio=Decimal("0.5"),
            final_tp_price=100.0,
        )
        self.assertEqual(
            self._specs(d),
            [("middle", Decimal("5.00"), 90.0), ("runner", Decimal("5.00"), 100.0)],
        )
        self.assertIsNone(d.fallback_reason)

    def test_middle_runner_active_returns_final_only(self) -> None:
        d = self._call(
            tp_plan="MIDDLE_RUNNER",
            middle_runner_active=True,
            partial_tp_price=90.0,
            partial_tp_ratio=Decimal("0.5"),
            final_tp_price=100.0,
        )
        self.assertEqual(self._specs(d), [("final", Decimal("10"), 100.0)])
        self.assertIsNone(d.fallback_reason)

    def test_partial_consumed_returns_final_only(self) -> None:
        d = self._call(
            tp_plan="SPLIT_PARTIAL_FINAL",
            partial_tp_consumed=True,
            partial_tp_price=90.0,
            partial_tp_ratio=Decimal("0.5"),
            final_tp_price=100.0,
        )
        self.assertEqual(self._specs(d), [("final", Decimal("10"), 100.0)])

    def test_unknown_tp_plan_returns_final(self) -> None:
        d = self._call(tp_plan="UNKNOWN", final_tp_price=100.0)
        self.assertEqual(self._specs(d), [("final", Decimal("10"), 100.0)])

    # ── THREE_STAGE_RUNNER: normal ──
    def test_three_stage_normal_valid(self) -> None:
        d = self._call(
            tp_plan="THREE_STAGE_RUNNER",
            final_tp_price=110.0,
            three_stage_tp1_price=101.0,
            three_stage_tp2_price=110.0,
            three_stage_tp1_ratio=Decimal("0.6"),
            three_stage_tp2_ratio=Decimal("0.2"),
            three_stage_runner_ratio=Decimal("0.2"),
        )
        self.assertEqual(
            self._specs(d),
            [
                ("tp1_middle", Decimal("6.00"), 101.0),
                ("tp2_outer", Decimal("2.00"), 110.0),
            ],
        )
        self.assertIsNone(d.fallback_reason)

    def test_three_stage_missing_tp1_price(self) -> None:
        d = self._call(
            tp_plan="THREE_STAGE_RUNNER",
            final_tp_price=110.0,
            three_stage_tp1_price=None,
            three_stage_tp2_price=110.0,
            three_stage_tp1_ratio=Decimal("0.6"),
            three_stage_tp2_ratio=Decimal("0.2"),
        )
        self.assertEqual(self._specs(d), [("final", Decimal("10"), 110.0)])
        self.assertIsNone(d.fallback_reason)

    def test_three_stage_missing_tp2_price(self) -> None:
        d = self._call(
            tp_plan="THREE_STAGE_RUNNER",
            final_tp_price=110.0,
            three_stage_tp1_price=101.0,
            three_stage_tp2_price=None,
            three_stage_tp1_ratio=Decimal("0.6"),
            three_stage_tp2_ratio=Decimal("0.2"),
        )
        self.assertEqual(self._specs(d), [("final", Decimal("10"), 110.0)])

    def test_three_stage_size_too_small(self) -> None:
        d = self._call(
            position_contracts=Decimal("0.01"),
            tp_plan="THREE_STAGE_RUNNER",
            final_tp_price=110.0,
            three_stage_tp1_price=101.0,
            three_stage_tp2_price=110.0,
            three_stage_tp1_ratio=Decimal("0.6"),
            three_stage_tp2_ratio=Decimal("0.2"),
        )
        self.assertEqual(self._specs(d), [("final", Decimal("0.01"), 110.0)])
        self.assertEqual(d.fallback_reason, "THREE_STAGE_TP_FALLBACK_SINGLE_SIZE_TOO_SMALL")

    def test_three_stage_zero_ratios(self) -> None:
        d = self._call(
            tp_plan="THREE_STAGE_RUNNER",
            final_tp_price=110.0,
            three_stage_tp1_price=101.0,
            three_stage_tp2_price=110.0,
            three_stage_tp1_ratio=Decimal("0"),
            three_stage_tp2_ratio=Decimal("0"),
        )
        self.assertEqual(self._specs(d), [("final", Decimal("10"), 110.0)])

    # ── THREE_STAGE_RUNNER: after TP1 ──
    def test_three_stage_after_tp1_valid(self) -> None:
        d = self._call(
            tp_plan="THREE_STAGE_RUNNER",
            final_tp_price=110.0,
            three_stage_tp1_price=101.0,
            three_stage_tp2_price=110.0,
            three_stage_tp1_ratio=Decimal("0.6"),
            three_stage_tp2_ratio=Decimal("0.2"),
            three_stage_runner_ratio=Decimal("0.2"),
            three_stage_tp1_consumed=True,
            three_stage_tp2_consumed=False,
        )
        self.assertEqual(
            self._specs(d),
            [("tp2_outer", Decimal("5.00"), 110.0)],
        )
        self.assertIsNone(d.fallback_reason)

    def test_three_stage_after_tp1_invalid_ratios(self) -> None:
        d = self._call(
            tp_plan="THREE_STAGE_RUNNER",
            final_tp_price=110.0,
            three_stage_tp2_price=None,
            three_stage_tp2_ratio=Decimal("0"),
            three_stage_runner_ratio=Decimal("0"),
            three_stage_tp1_consumed=True,
            three_stage_tp2_consumed=False,
        )
        self.assertEqual(self._specs(d), [("final", Decimal("10"), 110.0)])
        self.assertEqual(d.fallback_reason, "THREE_STAGE_TP2_AFTER_TP1_INVALID_RATIOS")

    def test_three_stage_after_tp1_tp2_too_small(self) -> None:
        d = self._call(
            position_contracts=Decimal("0.03"),
            tp_plan="THREE_STAGE_RUNNER",
            final_tp_price=110.0,
            three_stage_tp1_price=101.0,
            three_stage_tp2_price=110.0,
            three_stage_tp2_ratio=Decimal("0.1"),
            three_stage_runner_ratio=Decimal("0.3"),
            three_stage_tp1_consumed=True,
            three_stage_tp2_consumed=False,
        )
        self.assertEqual(self._specs(d), [("final", Decimal("0.03"), 110.0)])
        self.assertEqual(d.fallback_reason, "THREE_STAGE_TP2_AFTER_TP1_TP2_TOO_SMALL")

    def test_three_stage_after_tp1_runner_too_small(self) -> None:
        d = self._call(
            position_contracts=Decimal("0.019"),
            tp_plan="THREE_STAGE_RUNNER",
            final_tp_price=110.0,
            three_stage_tp1_price=101.0,
            three_stage_tp2_price=110.0,
            three_stage_tp2_ratio=Decimal("0.8"),
            three_stage_runner_ratio=Decimal("0.2"),
            three_stage_tp1_consumed=True,
            three_stage_tp2_consumed=False,
        )
        self.assertEqual(self._specs(d), [("tp2_outer", Decimal("0.019"), 110.0)])
        self.assertEqual(d.fallback_reason, "THREE_STAGE_TP2_AFTER_TP1_RUNNER_TOO_SMALL")


class TrendRunnerSlContractsTest(unittest.TestCase):
    def test_trend_runner_active_returns_net(self) -> None:
        result = trend_runner_sl_contracts(
            net_contracts_for_sl=Decimal("10"),
            runner_ratio=Decimal("0.5"),
            min_contracts=Decimal("0.01"),
            contract_precision=Decimal("0.01"),
            trend_runner_active=True,
        )
        self.assertEqual(result, Decimal("10"))

    def test_invalid_ratio_returns_net(self) -> None:
        result = trend_runner_sl_contracts(
            net_contracts_for_sl=Decimal("10"),
            runner_ratio=Decimal("0"),
            min_contracts=Decimal("0.01"),
            contract_precision=Decimal("0.01"),
            trend_runner_active=False,
        )
        self.assertEqual(result, Decimal("10"))

    def test_ratio_one_returns_net(self) -> None:
        result = trend_runner_sl_contracts(
            net_contracts_for_sl=Decimal("10"),
            runner_ratio=Decimal("1"),
            min_contracts=Decimal("0.01"),
            contract_precision=Decimal("0.01"),
            trend_runner_active=False,
        )
        self.assertEqual(result, Decimal("10"))

    def test_rounded_below_min_returns_net(self) -> None:
        result = trend_runner_sl_contracts(
            net_contracts_for_sl=Decimal("0.01"),
            runner_ratio=Decimal("0.5"),
            min_contracts=Decimal("0.01"),
            contract_precision=Decimal("0.01"),
            trend_runner_active=False,
        )
        self.assertEqual(result, Decimal("0.01"))

    def test_valid_ratio_returns_rounded(self) -> None:
        result = trend_runner_sl_contracts(
            net_contracts_for_sl=Decimal("10"),
            runner_ratio=Decimal("0.33"),
            min_contracts=Decimal("0.01"),
            contract_precision=Decimal("0.01"),
            trend_runner_active=False,
        )
        self.assertEqual(result, Decimal("3.30"))


if __name__ == "__main__":
    unittest.main()
