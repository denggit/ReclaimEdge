from __future__ import annotations

from decimal import Decimal

import pytest

from src.execution.trader import PositionSnapshot, Trader
from src.risk.simple_position_sizer import PositionSize
from src.strategies.boll_cvd_reclaim_strategy import TradeIntent


def intent(**overrides) -> TradeIntent:
    values = dict(
        intent_type="UPDATE_TP",
        side="LONG",
        price=100.0,
        layer_index=1,
        tp_price=101.0,
        reason="test",
        size=PositionSize(1, 50, 0.5, 1, 1.0),
        fast_cvd=0.0,
        previous_fast_cvd=0.0,
        buy_ratio=0.0,
        sell_ratio=0.0,
        boll_upper=101.0,
        boll_middle=100.0,
        boll_lower=99.0,
        ts_ms=1,
        avg_entry_price=100.0,
        breakeven_price=100.0,
        tp_mode="MIDDLE",
    )
    values.update(overrides)
    return TradeIntent(**values)  # type: ignore[arg-type]


class IsolationTrader(Trader):
    def __init__(self) -> None:
        self.symbol = "ETH-USDT-SWAP"
        self.td_mode = "isolated"
        self.pos_side_mode = "net"
        self.position_contracts = Decimal("10")
        self.contract_precision = Decimal("0.01")
        self.min_contracts = Decimal("0.01")
        self.tp_order_id = "core-old"
        self.near_tp_protective_sl_order_id = "near-sl"
        self.middle_runner_protective_sl_order_id = None
        self.three_stage_post_tp1_protective_sl_order_id = None
        self.trend_runner_sl_order_id = None
        self.cancelled: list[str] = []
        self.cancelled_middle_runner_stops: list[str | None] = []
        self.cancelled_post_tp1_stops: list[str | None] = []
        self.cancelled_trend_runner_stops: list[str | None] = []
        self.placed = []

    async def fetch_position_snapshot(self) -> PositionSnapshot:
        return PositionSnapshot("LONG", Decimal("12"), 100.0, 1.2, Decimal("12"))

    async def fetch_pending_orders(self):  # type: ignore[no-untyped-def]
        return [
            {"instId": self.symbol, "reduceOnly": "true", "ordId": "core-old"},
            {"instId": self.symbol, "reduceOnly": "true", "ordId": "sidecar-tp"},
            {"instId": self.symbol, "reduceOnly": "false", "ordId": "entry"},
        ]

    async def request(self, method, endpoint, payload=None):  # type: ignore[no-untyped-def]
        if endpoint == "/api/v5/trade/cancel-order":
            self.cancelled.append(payload["ordId"])
            return {"code": "0", "data": [{"ordId": payload["ordId"]}]}
        return {"code": "0", "data": [{"ordId": "new-tp"}]}

    async def _place_reduce_only_take_profit_orders(self, intent_: TradeIntent, specs):  # type: ignore[no-untyped-def]
        self.placed = specs
        return ["new-tp"]

    async def cancel_middle_runner_protective_stop(self, order_id: str | None) -> bool:
        self.cancelled_middle_runner_stops.append(order_id)
        if self.middle_runner_protective_sl_order_id == order_id:
            self.middle_runner_protective_sl_order_id = None
        return True

    async def cancel_three_stage_post_tp1_protective_stop(self, order_id: str | None) -> bool:
        self.cancelled_post_tp1_stops.append(order_id)
        if self.three_stage_post_tp1_protective_sl_order_id == order_id:
            self.three_stage_post_tp1_protective_sl_order_id = None
        return True

    async def cancel_trend_runner_protective_stop(self, order_id: str | None) -> bool:
        self.cancelled_trend_runner_stops.append(order_id)
        if self.trend_runner_sl_order_id == order_id:
            self.trend_runner_sl_order_id = None
        return True


@pytest.mark.asyncio
async def test_main_tp_update_does_not_cancel_sidecar_tp() -> None:
    trader = IsolationTrader()

    result = await trader.replace_take_profit(
        intent(protected_order_ids=("sidecar-tp", "near-sl"), managed_core_contracts="10"))

    assert result.ok
    assert trader.cancelled == ["core-old"]
    assert "sidecar-tp" not in trader.cancelled
    assert trader.placed == [("final", Decimal("10"), 101.0)]


@pytest.mark.asyncio
async def test_update_tp_uses_core_contracts_not_okx_net_contracts() -> None:
    trader = IsolationTrader()

    result = await trader.replace_take_profit(intent(protected_order_ids=("sidecar-tp",), managed_core_contracts="10"))

    assert result.ok
    assert result.contracts == "10"
    assert trader.placed == [("final", Decimal("10"), 101.0)]


@pytest.mark.asyncio
async def test_three_stage_degrade_to_single_does_not_cancel_sidecar_tp() -> None:
    trader = IsolationTrader()
    trader.middle_runner_protective_sl_order_id = "middle-sl"
    trader.three_stage_post_tp1_protective_sl_order_id = "post-tp1-sl"
    trader.trend_runner_sl_order_id = "trend-sl"

    result = await trader.replace_take_profit(
        intent(
            reason="three_stage_pre_tp1_degraded_to_single",
            protected_order_ids=("sidecar-tp",),
            managed_core_contracts="10",
            tp_plan="SINGLE",
        )
    )

    assert result.ok
    assert trader.cancelled == ["core-old"]
    assert "sidecar-tp" not in trader.cancelled
    assert trader.cancelled_middle_runner_stops == ["middle-sl"]
    assert trader.cancelled_post_tp1_stops == ["post-tp1-sl"]
    assert trader.cancelled_trend_runner_stops == ["trend-sl"]
    assert trader.placed == [("final", Decimal("10"), 101.0)]


@pytest.mark.asyncio
async def test_three_stage_degrade_to_middle_runner_does_not_cancel_sidecar_tp() -> None:
    trader = IsolationTrader()
    trader.three_stage_post_tp1_protective_sl_order_id = "post-tp1-sl"

    result = await trader.replace_take_profit(
        intent(
            reason="three_stage_pre_tp1_degraded_to_middle_runner",
            protected_order_ids=("sidecar-tp",),
            managed_core_contracts="10",
            tp_plan="MIDDLE_RUNNER",
            partial_tp_price=100.0,
            partial_tp_ratio=0.8,
        )
    )

    assert result.ok
    assert trader.cancelled == ["core-old"]
    assert "sidecar-tp" not in trader.cancelled
    assert trader.cancelled_post_tp1_stops == ["post-tp1-sl"]
    assert trader.placed == [("middle", Decimal("8.00"), 100.0), ("runner", Decimal("2.00"), 101.0)]


@pytest.mark.asyncio
async def test_three_stage_degrade_cancels_stale_sl_even_if_intent_protected_ids_contains_them() -> None:
    trader = IsolationTrader()
    trader.middle_runner_protective_sl_order_id = "middle-sl"
    trader.three_stage_post_tp1_protective_sl_order_id = "post-tp1-sl"
    trader.trend_runner_sl_order_id = "trend-sl"

    result = await trader.replace_take_profit(
        intent(
            reason="three_stage_pre_tp1_degraded_to_single",
            protected_order_ids=("sidecar-tp", "middle-sl", "post-tp1-sl", "trend-sl"),
            managed_core_contracts="10",
            tp_plan="SINGLE",
        )
    )

    assert result.ok
    assert trader.cancelled == ["core-old"]
    assert "sidecar-tp" not in trader.cancelled
    assert trader.cancelled_middle_runner_stops == ["middle-sl"]
    assert trader.cancelled_post_tp1_stops == ["post-tp1-sl"]
    assert trader.cancelled_trend_runner_stops == ["trend-sl"]
    assert trader.placed == [("final", Decimal("10"), 101.0)]


class UnknownReduceOnlyTrader(IsolationTrader):
    async def fetch_pending_orders(self):  # type: ignore[no-untyped-def]
        return [{"instId": self.symbol, "reduceOnly": "true", "ordId": "unknown"}]


@pytest.mark.asyncio
async def test_unknown_reduce_only_order_blocks_tp_update() -> None:
    trader = UnknownReduceOnlyTrader()
    trader.tp_order_id = None

    result = await trader.replace_take_profit(intent(protected_order_ids=("sidecar-tp",)))

    assert result.ok
    assert result.message == "reduce_only_order_identity_unknown_update_tp_skipped"
    assert trader.cancelled == []
    assert trader.placed == []
