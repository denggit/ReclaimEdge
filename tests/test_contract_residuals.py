from __future__ import annotations

from decimal import Decimal, ROUND_DOWN

import pytest

from src.execution.trader import PositionSnapshot, Trader
from src.risk.simple_position_sizer import PositionSize
from src.strategies.boll_cvd_reclaim_strategy import TradeIntent


# ============================================================
# Test helpers
# ============================================================

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
    t._session = None
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


# ============================================================
# Test 1: Split TP _build_take_profit_order_specs exact sum (calls production)
# ============================================================
def test_split_tp_build_specs_exact_sum_to_core_contracts() -> None:
    """Split TP via _build_take_profit_order_specs: partial + final must exactly sum to core."""
    trader = make_trader(position_contracts=Decimal("10.01"))
    intent = make_intent(
        tp_plan="SPLIT_PARTIAL_FINAL",
        partial_tp_price=3060.0,
        partial_tp_ratio=0.60,
    )

    specs = trader._build_take_profit_order_specs(intent)
    total = sum(contracts for _, contracts, _ in specs)

    assert total == Decimal("10.01"), (
        f"partial + final must equal 10.01, got total={total} specs={specs}"
    )
    # Should not fallback to single (which returns only "final")
    labels = [label for label, _, _ in specs]
    assert "partial" in labels, f"should have partial TP, got labels={labels}"
    assert "final" in labels, f"should have final TP, got labels={labels}"
    # Verify no 0.01 residual: partial should be round_down(10.01*0.6) = 6.00
    partial = next(c for l, c, _ in specs if l == "partial")
    assert partial == Decimal("6.00"), f"partial should be 6.00, got {partial}"
    final = next(c for l, c, _ in specs if l == "final")
    assert final == Decimal("4.01"), f"final should be 4.01, got {final}"


def test_split_partial_consumed_builds_only_final_tp() -> None:
    trader = make_trader(position_contracts=Decimal("4.01"))
    intent = make_intent(
        tp_plan="SPLIT_PARTIAL_FINAL",
        partial_tp_price=3060.0,
        partial_tp_ratio=0.60,
        partial_tp_consumed=True,
        tp_price=3120.0,
    )

    assert trader._build_take_profit_order_specs(intent) == [("final", Decimal("4.01"), 3120.0)]


def test_middle_runner_active_builds_only_final_tp() -> None:
    trader = make_trader(position_contracts=Decimal("2.00"))
    intent = make_intent(
        tp_plan="MIDDLE_RUNNER",
        partial_tp_price=3050.0,
        partial_tp_ratio=0.80,
        middle_runner_active=True,
        tp_price=3120.0,
    )

    assert trader._build_take_profit_order_specs(intent) == [("final", Decimal("2.00"), 3120.0)]


# ============================================================
# Test 2: Three-Stage _build_three_stage_order_specs exact sum (calls production)
# ============================================================
def test_three_stage_build_specs_exact_sum_to_core_contracts() -> None:
    """Three-Stage via _build_three_stage_order_specs: TP1 + TP2 + runner must exactly sum to core."""
    trader = make_trader(position_contracts=Decimal("10.01"))

    # _build_three_stage_order_specs is called when tp_plan="THREE_STAGE_RUNNER"
    # It reads three_stage_tp1_price, three_stage_tp2_price, ratios from the intent
    intent = make_intent(
        tp_plan="THREE_STAGE_RUNNER",
        tp_price=3200.0,
        three_stage_tp1_price=3050.0,
        three_stage_tp1_ratio=0.60,
        three_stage_tp2_price=3100.0,
        three_stage_tp2_ratio=0.20,
        three_stage_runner_ratio=0.20,
    )

    specs = trader._build_three_stage_order_specs(intent)
    # specs returns TP1 and TP2 only; runner is implicit
    placed = sum(contracts for _, contracts, _ in specs)
    runner = trader.position_contracts - placed

    # Total must equal core
    assert placed + runner == Decimal("10.01"), (
        f"tp1 + tp2 + runner must equal 10.01, got placed={placed} runner={runner}"
    )
    # Runner must be >= min_contracts
    assert runner >= trader.min_contracts, (
        f"runner must be >= 0.01, got {runner}"
    )
    # TP1 should be round_down(10.01 * 0.60) = 6.00
    tp1 = next(c for l, c, _ in specs if l == "tp1_middle")
    assert tp1 == Decimal("6.00"), f"tp1 should be 6.00, got {tp1}"
    # TP2 should be round_down(10.01 * 0.20) = 2.00
    tp2 = next(c for l, c, _ in specs if l == "tp2_outer")
    assert tp2 == Decimal("2.00"), f"tp2 should be 2.00, got {tp2}"
    # Runner should be 10.01 - 6.00 - 2.00 = 2.01
    assert runner == Decimal("2.01"), f"runner should be 2.01, got {runner}"


def test_three_stage_tp1_consumed_builds_only_tp2_share_from_remaining_core() -> None:
    trader = make_trader(position_contracts=Decimal("0.40"))
    intent = make_intent(
        tp_plan="THREE_STAGE_RUNNER",
        tp_price=3120.0,
        three_stage_tp1_price=3050.0,
        three_stage_tp1_ratio=0.60,
        three_stage_tp2_price=3120.0,
        three_stage_tp2_ratio=0.20,
        three_stage_runner_ratio=0.20,
        three_stage_tp1_consumed=True,
        three_stage_tp2_consumed=False,
    )

    assert trader._build_three_stage_order_specs(intent) == [("tp2_outer", Decimal("0.20"), 3120.0)]


def test_three_stage_tp1_consumed_keeps_runner_on_larger_remaining_core() -> None:
    trader = make_trader(position_contracts=Decimal("4.00"))
    intent = make_intent(
        tp_plan="THREE_STAGE_RUNNER",
        tp_price=3120.0,
        three_stage_tp1_price=3050.0,
        three_stage_tp1_ratio=0.60,
        three_stage_tp2_price=3120.0,
        three_stage_tp2_ratio=0.20,
        three_stage_runner_ratio=0.20,
        three_stage_tp1_consumed=True,
        three_stage_tp2_consumed=False,
    )

    specs = trader._build_three_stage_order_specs(intent)

    assert specs == [("tp2_outer", Decimal("2.00"), 3120.0)]
    assert trader.position_contracts - specs[0][1] == Decimal("2.00")


# ============================================================
# Test 3: market_exit_remaining_position closes exact min_contracts (0.01)
# ============================================================
@pytest.mark.asyncio
async def test_market_exit_closes_exact_min_contracts() -> None:
    """market_exit_remaining_position_with_retries must submit reduce-only market sz=0.01."""
    trader = make_trader()

    submitted_body: dict | None = None
    _fetch_count = 0

    async def mock_fetch_snapshot():
        nonlocal _fetch_count
        _fetch_count += 1
        if _fetch_count == 1:
            return PositionSnapshot("LONG", Decimal("0.01"), 3000.0, 0.001, Decimal("0.01"))
        # After order placed, return flat
        return PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0"))

    async def mock_request(_method, _endpoint, body=None):
        nonlocal submitted_body
        submitted_body = body
        return {"code": "0", "data": [{"ordId": "exit-001"}]}

    async def mock_cleanup():
        pass

    async def mock_fetch_pending():
        return []

    trader.fetch_position_snapshot = mock_fetch_snapshot
    trader.request = mock_request
    trader._cleanup_after_near_tp_market_exit = mock_cleanup
    trader.fetch_pending_orders = mock_fetch_pending

    # The exit should submit the order with sz=0.01 (the position contracts)
    ok, message = await trader.market_exit_remaining_position_with_retries("LONG", retry_count=1)

    assert ok, f"market exit should succeed, got message={message}"
    assert submitted_body is not None, "order body should have been submitted"
    assert submitted_body["sz"] == "0.01", (
        f"should submit sz=0.01 for exact min_contracts position, got sz={submitted_body.get('sz')}"
    )
    assert str(submitted_body.get("reduceOnly", "")).lower() == "true"


# ============================================================
# Test 4: market_exit does not pretend below-min dust is flat
# ============================================================
@pytest.mark.asyncio
async def test_market_exit_refuses_below_min_dust() -> None:
    """market_exit_remaining_position_with_retries returns ok=False for 0.005 < min_contracts=0.01."""
    trader = make_trader()

    async def mock_fetch_snapshot():
        return PositionSnapshot("LONG", Decimal("0.005"), 3000.0, 0.0005, Decimal("0.005"))

    trader.fetch_position_snapshot = mock_fetch_snapshot

    ok, message = await trader.market_exit_remaining_position_with_retries("LONG", retry_count=1)

    assert not ok, f"market exit should fail for below-min dust, got ok={ok}"
    assert "dust_position_below_min_contracts" in message, (
        f"message should contain dust_position_below_min_contracts, got: {message}"
    )
    assert "0.005" in message, f"message should mention 0.005 contracts, got: {message}"


# ============================================================
# Test 5: managed_core_contracts_exceeds_net_position still raises RuntimeError
# ============================================================
@pytest.mark.asyncio
async def test_managed_core_exceeds_net_position_raises_runtime_error() -> None:
    """When core=10.01 > net=10, replace_take_profit must raise RuntimeError."""
    trader = make_trader()

    async def mock_fetch_snapshot():
        return PositionSnapshot("LONG", Decimal("10"), 3000.0, 1.0, Decimal("10"))

    async def mock_fetch_pending():
        return []

    trader.fetch_position_snapshot = mock_fetch_snapshot
    trader.fetch_pending_orders = mock_fetch_pending

    intent = make_intent(managed_core_contracts="10.01")

    with pytest.raises(RuntimeError, match="managed_core_contracts_exceeds_net_position"):
        await trader.replace_take_profit(intent)


# ============================================================
# Test 6: replace_take_profit fetch position fails → RuntimeError (not fallback)
# ============================================================
@pytest.mark.asyncio
async def test_replace_tp_fetch_fails_raises_runtime_error() -> None:
    """When fetch_position_snapshot raises and managed_core_contracts is set,
    replace_take_profit must raise RuntimeError ('failed_to_fetch_net_position_for_global_sl'),
    not fallback to net=managed_core_contracts. No TP/SL orders submitted."""
    trader = make_trader()
    tp_place_called = False
    sl_place_called = False

    async def mock_fetch_snapshot():
        raise RuntimeError("OKX API timeout")

    async def mock_fetch_pending():
        return []

    async def mock_place_tp(_intent, _specs):
        nonlocal tp_place_called
        tp_place_called = True
        return ["tp-fake"]

    async def mock_cancel_existing():
        return None

    trader.fetch_position_snapshot = mock_fetch_snapshot
    trader.fetch_pending_orders = mock_fetch_pending
    trader._place_reduce_only_take_profit_orders = mock_place_tp
    trader.cancel_existing_reduce_only_orders = mock_cancel_existing

    intent = make_intent(managed_core_contracts="10", trend_runner_active=True, trend_runner_sl_price=2950.0)

    with pytest.raises(RuntimeError, match="failed_to_fetch_net_position_for_global_sl"):
        await trader.replace_take_profit(intent)

    # Verify no TP/SL orders were placed
    assert not tp_place_called, "No TP orders should have been placed after fetch failure"
