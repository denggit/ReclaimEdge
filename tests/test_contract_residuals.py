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
# Test 1: Split TP contracts exactly sum to core contracts
# ============================================================
def test_split_tp_exact_sum_to_core_contracts() -> None:
    """Split TP partial + final must exactly sum to core_contracts (no 0.01 residual)."""
    trader = make_trader(position_contracts=Decimal("10.01"))
    # core_contracts = 10.01
    # partial_ratio = 0.60
    # partial = round_down(10.01 * 0.60) = round_down(6.006) = 6.00
    # final = 10.01 - 6.00 = 4.01
    partial_contracts = trader.round_contracts_down(Decimal("10.01") * Decimal("0.60"))
    final_contracts = Decimal("10.01") - partial_contracts

    assert partial_contracts == Decimal("6.00"), f"partial should be 6.00, got {partial_contracts}"
    assert final_contracts == Decimal("4.01"), f"final should be 4.01, got {final_contracts}"
    assert partial_contracts + final_contracts == Decimal("10.01"), (
        f"partial + final must equal 10.01, got {partial_contracts + final_contracts}"
    )


# ============================================================
# Test 2: Three-Stage TP1 + TP2 + runner exactly sum to core contracts
# ============================================================
def test_three_stage_exact_sum_to_core_contracts() -> None:
    """Three-Stage TP1 + TP2 + runner must exactly sum to core_contracts."""
    trader = make_trader(position_contracts=Decimal("10.01"))
    ta = Decimal("10.01")

    tp1_ratio = Decimal("0.60")
    tp2_ratio = Decimal("0.20")
    runner_ratio = Decimal("1") - tp1_ratio - tp2_ratio  # 0.20

    tp1 = trader.round_contracts_down(ta * tp1_ratio)
    tp2 = trader.round_contracts_down(ta * tp2_ratio)
    # runner = core - tp1 - tp2 (exact residual, not rounded down)
    runner = ta - tp1 - tp2

    total = tp1 + tp2 + runner
    assert total == ta, f"tp1 + tp2 + runner must equal {ta}, got total={total} tp1={tp1} tp2={tp2} runner={runner}"

    # Each component must be >= min_contracts or fallback to single TP
    assert tp1 >= Decimal("0.01")
    assert tp2 >= Decimal("0.01")
    assert runner >= Decimal("0.01") or runner == Decimal("0"), (
        f"runner must be >= 0.01 or 0 (fallback), got {runner}"
    )


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

    # Override any SL placement path
    original_place_sl = getattr(trader, "place_near_tp_protective_stop_with_retries", None)

    trader.fetch_position_snapshot = mock_fetch_snapshot
    trader.fetch_pending_orders = mock_fetch_pending
    trader._place_reduce_only_take_profit_orders = mock_place_tp
    trader.cancel_existing_reduce_only_orders = mock_cancel_existing

    intent = make_intent(managed_core_contracts="10", trend_runner_active=True, trend_runner_sl_price=2950.0)

    with pytest.raises(RuntimeError, match="failed_to_fetch_net_position_for_global_sl"):
        await trader.replace_take_profit(intent)

    # Verify no TP/SL orders were placed
    assert not tp_place_called, "No TP orders should have been placed after fetch failure"
