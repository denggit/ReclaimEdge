from __future__ import annotations

from decimal import Decimal

import pytest

from src.execution.trader import PositionSnapshot, Trader
from src.risk.simple_position_sizer import PositionSize
from src.strategies.boll_cvd_reclaim_strategy import TradeIntent


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


class SlNetContractsTrader(Trader):
    """Test trader that records SL placement calls with their contracts argument."""

    def __init__(self) -> None:
        self.symbol = "ETH-USDT-SWAP"
        self.td_mode = "isolated"
        self.pos_side_mode = "net"
        self.position_contracts = Decimal("0")
        self.contract_precision = Decimal("0.01")
        self.min_contracts = Decimal("0.01")
        self.tp_order_id = None
        self.near_tp_protective_sl_order_id = None
        self.middle_runner_protective_sl_order_id = None
        self.three_stage_post_tp1_protective_sl_order_id = None
        self.trend_runner_sl_order_id = None
        self._protected_reduce_only_order_ids: set[str] = set()
        self._managed_reduce_only_order_ids: set[str] = set()
        self._allow_cancel_unmanaged_reduce_only = True

        # Record SL contract arguments
        self.post_tp1_sl_contracts: list[Decimal] = []
        self.middle_runner_sl_contracts: list[Decimal] = []
        self.trend_runner_sl_contracts: list[Decimal] = []
        self.placed_tp_specs: list[tuple] = []

    async def fetch_position_snapshot(self) -> PositionSnapshot:
        # Default: OKX net = 12
        return PositionSnapshot("LONG", Decimal("12"), 3000.0, 1.2, Decimal("12"))

    async def fetch_pending_orders(self):  # type: ignore[no-untyped-def]
        return []

    async def _place_reduce_only_take_profit_orders(self, intent_: TradeIntent, specs):  # type: ignore[no-untyped-def]
        self.placed_tp_specs = specs
        return [f"tp-{label}" for label, _contracts, _price in specs]

    async def place_three_stage_post_tp1_protective_stop_with_retries(
        self, side, contracts, stop_price, retry_count, retry_interval_seconds,  # type: ignore[no-untyped-def]
    ):
        self.post_tp1_sl_contracts.append(contracts)
        self.three_stage_post_tp1_protective_sl_order_id = "algo-post-tp1"
        return True, "algo-post-tp1", "protective_sl_placed"

    async def place_middle_runner_protective_stop_with_retries(
        self, side, contracts, stop_price, retry_count, retry_interval_seconds,  # type: ignore[no-untyped-def]
    ):
        self.middle_runner_sl_contracts.append(contracts)
        self.middle_runner_protective_sl_order_id = "algo-middle-runner"
        return True, "algo-middle-runner", "protective_sl_placed"

    async def place_trend_runner_protective_stop_with_retries(
        self, side, contracts, stop_price, retry_count, retry_interval_seconds,  # type: ignore[no-untyped-def]
    ):
        self.trend_runner_sl_contracts.append(contracts)
        self.trend_runner_sl_order_id = "algo-trend-runner"
        return True, "algo-trend-runner", "protective_sl_placed"

    async def cancel_three_stage_post_tp1_protective_stop(self, order_id: str | None) -> bool:
        return True

    async def cancel_middle_runner_protective_stop(self, order_id: str | None) -> bool:
        return True

    async def cancel_trend_runner_protective_stop(self, order_id: str | None) -> bool:
        return True

    async def cancel_existing_reduce_only_orders(self) -> None:
        return None


# ============================================================
# Test 1: post-TP1 SL uses net contracts (12), TP uses core (10)
# ============================================================
@pytest.mark.asyncio
async def test_post_tp1_sl_uses_net_contracts() -> None:
    """post-TP1 protective SL must cover full OKX net position, not just core."""
    trader = SlNetContractsTrader()

    result = await trader.replace_take_profit(
        make_intent(
            managed_core_contracts="10",
            tp_plan="THREE_STAGE_RUNNER",
            three_stage_tp1_consumed=True,
            three_stage_tp2_consumed=False,
            three_stage_post_tp1_protective_sl_price=2990.0,
            trend_runner_active=False,
        )
    )

    assert result.ok
    assert result.protective_sl_ok
    # SL must use net contracts (12), not core (10)
    assert trader.post_tp1_sl_contracts == [Decimal("12")], (
        f"post-TP1 SL should use net=12, got {trader.post_tp1_sl_contracts}"
    )
    # result.contracts is TP contracts (core)
    assert result.contracts == "10"


# ============================================================
# Test 2: Middle Runner SL uses net contracts (12), TP uses core (10)
# ============================================================
@pytest.mark.asyncio
async def test_middle_runner_sl_uses_net_contracts() -> None:
    """Middle Runner protective SL must cover full OKX net position, not just core."""
    trader = SlNetContractsTrader()

    result = await trader.replace_take_profit(
        make_intent(
            managed_core_contracts="10",
            tp_plan="SINGLE",
            middle_runner_active=True,
            middle_runner_protective_sl_price=2980.0,
        )
    )

    assert result.ok
    assert result.protective_sl_ok
    # SL must use net contracts (12), not core (10)
    assert trader.middle_runner_sl_contracts == [Decimal("12")], (
        f"middle runner SL should use net=12, got {trader.middle_runner_sl_contracts}"
    )
    # TP must use core contracts (10)
    assert trader.placed_tp_specs == [("final", Decimal("10"), 3100.0)], (
        f"TP should use core=10, got {trader.placed_tp_specs}"
    )


# ============================================================
# Test 3: Trend Runner SL uses net contracts (12), TP uses core (10)
# ============================================================
@pytest.mark.asyncio
async def test_trend_runner_sl_uses_net_contracts() -> None:
    """Trend Runner protective SL must cover full OKX net position, not just core."""
    trader = SlNetContractsTrader()

    result = await trader.replace_take_profit(
        make_intent(
            managed_core_contracts="10",
            tp_plan="SINGLE",
            trend_runner_active=True,
            trend_runner_sl_price=2950.0,
        )
    )

    assert result.ok
    assert result.protective_sl_ok
    # SL must use net contracts (12), not core (10)
    assert trader.trend_runner_sl_contracts == [Decimal("12")], (
        f"trend runner SL should use net=12, got {trader.trend_runner_sl_contracts}"
    )
    # TP must use core contracts (10)
    assert trader.placed_tp_specs == [("final", Decimal("10"), 3100.0)], (
        f"TP should use core=10, got {trader.placed_tp_specs}"
    )


# ============================================================
# Test 4: TP still uses core contracts when managed_core_contracts set
# ============================================================
@pytest.mark.asyncio
async def test_tp_uses_core_contracts() -> None:
    """When managed_core_contracts is set, TP orders must use core, not net."""
    trader = SlNetContractsTrader()

    result = await trader.replace_take_profit(
        make_intent(
            managed_core_contracts="10",
            tp_plan="SINGLE",
        )
    )

    assert result.ok
    assert result.contracts == "10"
    assert trader.placed_tp_specs == [("final", Decimal("10"), 3100.0)], (
        f"TP should use core=10, got {trader.placed_tp_specs}"
    )


# ============================================================
# Test 5: managed_core_contracts > net contracts → RuntimeError
# ============================================================
@pytest.mark.asyncio
async def test_managed_core_exceeds_net_raises_runtime_error() -> None:
    """When managed_core_contracts > OKX net position, raise RuntimeError."""
    trader = SlNetContractsTrader()

    # OKX net=8, core=10 → should raise
    # Override fetch_position_snapshot to return 8
    original_fetch = trader.fetch_position_snapshot

    async def fetch_8():  # type: ignore[no-untyped-def]
        return PositionSnapshot("LONG", Decimal("8"), 3000.0, 0.8, Decimal("8"))

    trader.fetch_position_snapshot = fetch_8  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="managed_core_contracts_exceeds_net_position"):
        await trader.replace_take_profit(
            make_intent(
                managed_core_contracts="10",
                tp_plan="SINGLE",
            )
        )

    # Restore
    trader.fetch_position_snapshot = original_fetch  # type: ignore[method-assign]


# ============================================================
# Test 6: Without managed_core_contracts, TP and SL both use net
# ============================================================
@pytest.mark.asyncio
async def test_without_managed_core_contracts_tp_and_sl_use_net() -> None:
    """Without managed_core_contracts, both TP and SL use OKX net position."""
    trader = SlNetContractsTrader()

    result = await trader.replace_take_profit(
        make_intent(
            managed_core_contracts=None,
            tp_plan="SINGLE",
            trend_runner_active=True,
            trend_runner_sl_price=2950.0,
        )
    )

    assert result.ok
    assert result.protective_sl_ok
    # SL uses net = 12
    assert trader.trend_runner_sl_contracts == [Decimal("12")]
    # TP uses net = 12 (same as core since no managed_core_contracts)
    assert trader.placed_tp_specs == [("final", Decimal("12"), 3100.0)]
    assert result.contracts == "12"


# ============================================================
# Test 7: _trend_runner_sl_contracts accepts net_contracts_for_sl
# ============================================================
def test_trend_runner_sl_contracts_accepts_net_param() -> None:
    """_trend_runner_sl_contracts uses net_contracts_for_sl when available."""
    trader = Trader.__new__(Trader)
    trader.position_contracts = Decimal("5")
    trader.contract_precision = Decimal("0.01")
    trader.min_contracts = Decimal("0.01")

    intent = make_intent(trend_runner_active=True)

    # When trend_runner_active=True, returns net_contracts_for_sl
    result = trader._trend_runner_sl_contracts(intent, Decimal("12"))
    assert result == Decimal("12"), f"active runner should return net=12, got {result}"

    # When not active, uses net_contracts_for_sl with runner_ratio
    intent2 = make_intent(
        trend_runner_active=False,
        three_stage_runner_ratio=0.5,
    )
    result2 = trader._trend_runner_sl_contracts(intent2, Decimal("12"))
    assert result2 == Decimal("6.00"), f"runner ratio 0.5 × net=12 should be 6.00, got {result2}"


# ============================================================
# Test 8: No position → returns error with net_contracts_for_sl
# ============================================================
@pytest.mark.asyncio
async def test_no_position_returns_error() -> None:
    """When OKX net position is 0, replace_take_profit returns error."""
    trader = SlNetContractsTrader()

    async def fetch_flat():  # type: ignore[no-untyped-def]
        return PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0"))

    trader.fetch_position_snapshot = fetch_flat  # type: ignore[method-assign]

    result = await trader.replace_take_profit(
        make_intent(
            managed_core_contracts="10",
            tp_plan="SINGLE",
        )
    )

    assert not result.ok
    assert "no position to protect" in result.message
