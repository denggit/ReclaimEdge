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


@pytest.mark.asyncio
async def test_main_tp_update_does_not_cancel_sidecar_tp() -> None:
    trader = IsolationTrader()

    result = await trader.replace_take_profit(intent(protected_order_ids=("sidecar-tp", "near-sl"), managed_core_contracts="10"))

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


class UnknownReduceOnlyTrader(IsolationTrader):
    async def fetch_pending_orders(self):  # type: ignore[no-untyped-def]
        return [{"instId": self.symbol, "reduceOnly": "true", "ordId": "unknown"}]


@pytest.mark.asyncio
async def test_unknown_reduce_only_order_blocks_tp_update() -> None:
    trader = UnknownReduceOnlyTrader()
    trader.tp_order_id = None

    with pytest.raises(RuntimeError, match="reduce_only_order_identity_unknown"):
        await trader.replace_take_profit(intent(protected_order_ids=("sidecar-tp",)))
