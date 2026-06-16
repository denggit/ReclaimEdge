"""Tests for entry protective SL execution correctness.

Covers:
1. Trader no longer references the non-existent method
2. Entry protective SL placement success path
3. SL placement exception -> NO market exit, manual_intervention_required
4. SL placement returns failure -> NO market exit, manual_intervention_required
5. Missing entry_protective_sl_price -> NO market exit, manual_intervention_required
6. Trend SL update success path uses correct method
7. Trend SL update exception -> no old SL cancel, returns ok=False
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.execution.trader import Trader, LiveTradeResult, PositionSnapshot


# ======================================================================
# Helpers
# ======================================================================


@dataclass
class _FakeSize:
    margin_usdt: float = 10.0
    notional_usdt: float = 500.0
    eth_qty: float = 0.1
    layer_index: int = 1
    layer_multiplier: float = 1.0
    sizing_mode: str = "risk_budget"
    risk_usdt: float = 2.0
    stop_price: float | None = 3000.0
    stop_distance_pct: float = 0.02
    effective_risk_pct: float = 0.021


def _flat_position() -> PositionSnapshot:
    return PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0"))


def _open_long_intent(**overrides):
    from src.strategies.boll_cvd_reclaim_strategy import TradeIntent

    kwargs = dict(
        intent_type="OPEN_LONG",
        side="LONG",
        price=3200.0,
        layer_index=1,
        tp_price=3300.0,
        reason="test_entry",
        size=_FakeSize(),
        fast_cvd=0.1,
        previous_fast_cvd=0.09,
        buy_ratio=0.7,
        sell_ratio=0.3,
        boll_upper=3300.0,
        boll_middle=3100.0,
        boll_lower=2900.0,
        ts_ms=1000000,
        avg_entry_price=3200.0,
        breakeven_price=3205.0,
        tp_mode="MIDDLE",
        entry_protective_sl_price=3096.9,
    )
    kwargs.update(overrides)
    return TradeIntent(**kwargs)


def _update_trend_sl_intent(sl_price=2990.0, side="LONG"):
    from src.strategies.boll_cvd_reclaim_strategy import TradeIntent

    return TradeIntent(
        intent_type="UPDATE_TREND_SL",
        side=side,
        price=3050.0,
        layer_index=1,
        tp_price=3500.0,
        reason="trend_trailing_sl_tightened",
        size=_FakeSize(eth_qty=0.1),
        fast_cvd=0.01,
        previous_fast_cvd=0.005,
        buy_ratio=0.6,
        sell_ratio=0.4,
        boll_upper=3500.0,
        boll_middle=3000.0,
        boll_lower=2500.0,
        ts_ms=2000000,
        avg_entry_price=3200.0,
        breakeven_price=3205.0,
        tp_mode="UPPER",
        entry_protective_sl_price=sl_price,
    )


def _make_minimal_trader(*, set_position: bool = True):
    """Build a minimal Trader (via object.__new__) with mocked deps for testing."""
    trader = object.__new__(Trader)
    trader.position_contracts = Decimal("1") if set_position else Decimal("0")
    trader.symbol = "ETH-USDT-SWAP"
    trader.td_mode = "isolated"
    trader.pos_side_mode = "net"
    trader.contract_multiplier = Decimal("0.1")
    trader.contract_precision = Decimal("0.01")
    trader.min_contracts = Decimal("0.01")
    trader.entry_protective_sl_order_id = None

    # Position snapshot
    if set_position:
        trader.fetch_position_snapshot = AsyncMock(
            return_value=PositionSnapshot(
                side="LONG",
                contracts=Decimal("1"),
                avg_entry_price=3200.0,
                eth_qty=0.1,
                raw_pos=Decimal("1"),
            )
        )
    else:
        trader.fetch_position_snapshot = AsyncMock(return_value=_flat_position())

    return trader


def _make_fake_tc():
    """Build a minimal FakeTradingClient that returns successful market entry."""
    from src.execution.trading_client_port import OrderResult

    class FakeTC:
        async def place_market_order(self, *, side, qty, reduce_only, client_order_id):
            return OrderResult(ok=True, order_id="entry-1", client_order_id=None, raw={})

        async def fetch_balance(self):
            class FB:
                total = 500.0
            return FB()

        async def configure_instrument(self):
            pass

        async def fetch_position(self):
            class FakePos:
                has_position = True
                side = "LONG"
                qty = Decimal("1")
                avg_entry_price = 3200.0
                raw = {"raw_pos": "1"}
            return FakePos()

        async def fetch_open_orders(self):
            return []

        async def fetch_open_algo_orders(self):
            return []

    return FakeTC()


# ======================================================================
# 1. Trader no longer references the non-existent method
# ======================================================================


def test_trader_does_not_reference_missing_entry_protective_stop_method():
    """Source code must not contain place_entry_protective_stop_with_retries."""
    source = Path("src/execution/trader.py").read_text()
    assert "place_entry_protective_stop_with_retries" not in source, (
        "trader.py must not reference place_entry_protective_stop_with_retries — "
        "use place_protective_stop_with_retries instead"
    )


def test_trader_references_correct_place_protective_stop():
    """Source code must reference the correct method."""
    source = Path("src/execution/trader.py").read_text()
    assert "place_protective_stop_with_retries" in source, (
        "trader.py must reference place_protective_stop_with_retries"
    )


# ======================================================================
# 2. Entry protective SL placement success
# ======================================================================


class TestEntryProtectiveSLSuccess:
    """Entry protective SL is placed successfully."""

    @pytest.mark.asyncio
    async def test_entry_sl_success_returns_correct_result(self):
        """When SL placement succeeds, LiveTradeResult reflects it."""
        trader = _make_minimal_trader()
        fake_tc = _make_fake_tc()
        trader.trading_client = fake_tc

        from src.execution.tp_sl_execution_manager import TpSlExecutionManager
        trader._tp_sl_manager = TpSlExecutionManager(trader, trading_client=fake_tc)

        # Monkey-patch place_protective_stop_with_retries at the trader level
        trader.place_protective_stop_with_retries = AsyncMock(
            return_value=(True, "entry-sl-1", "protective_sl_placed"),
        )

        # Must also mock replace_take_profit for mean-reversion entries
        trader.replace_take_profit = AsyncMock(
            return_value=LiveTradeResult(
                ok=True, action="OPEN_LONG", order_id="entry-1",
                tp_order_id="tp-1", contracts="1.00", tp_price="3300.00",
                message="tp placed", entry_filled=True, tp_ok=True,
                tp_order_ids=("tp-1",),
                protective_sl_order_id=None, protective_sl_price="",
                protective_sl_ok=True,
            ),
        )

        intent = _open_long_intent()
        result = await trader.execute_intent(intent)

        assert result.entry_filled is True, f"entry_filled must be True, got {result}"
        assert result.protective_sl_ok is True, (
            f"protective_sl_ok must be True, got {result}"
        )

        # Verify the correct method was called
        trader.place_protective_stop_with_retries.assert_called_once()

    @pytest.mark.asyncio
    async def test_entry_sl_success_for_trend_entry(self):
        """Trend entries skip fixed TP but still place entry protective SL."""
        trader = _make_minimal_trader()
        fake_tc = _make_fake_tc()
        trader.trading_client = fake_tc

        from src.execution.tp_sl_execution_manager import TpSlExecutionManager
        trader._tp_sl_manager = TpSlExecutionManager(trader, trading_client=fake_tc)

        trader.place_protective_stop_with_retries = AsyncMock(
            return_value=(True, "entry-sl-1", "protective_sl_placed"),
        )

        intent = _open_long_intent(entry_regime="TREND_BREAKOUT", tp_price=0.0)
        result = await trader.execute_intent(intent)

        assert result.entry_filled is True
        assert result.protective_sl_ok is True
        assert "no_fixed_tp" in result.message or result.tp_order_id is None
        trader.place_protective_stop_with_retries.assert_called_once()


# ======================================================================
# 3. Entry protective SL placement exception -> NO market exit
# ======================================================================


class TestEntryProtectiveSLException:
    """When SL placement raises an exception, NO market exit — halt only."""

    @pytest.mark.asyncio
    async def test_sl_exception_does_not_market_exit(self):
        """SL placement exception -> result.ok=False, manual_intervention_required, NO market exit."""
        trader = _make_minimal_trader()
        fake_tc = _make_fake_tc()
        trader.trading_client = fake_tc

        from src.execution.tp_sl_execution_manager import TpSlExecutionManager
        trader._tp_sl_manager = TpSlExecutionManager(trader, trading_client=fake_tc)

        # SL placement RAISES exception
        trader.place_protective_stop_with_retries = AsyncMock(
            side_effect=RuntimeError("Connection reset"),
        )

        # Track market exit — must NOT be called
        trader.market_exit_remaining_position_with_retries = AsyncMock(
            return_value=(True, "should_not_be_called"),
        )

        intent = _open_long_intent()

        # Exception must NOT bubble to caller
        result = await trader.execute_intent(intent)

        assert result.ok is False, f"Result must be not ok, got {result}"
        assert result.entry_filled is True, "entry_filled must be True"
        assert result.protective_sl_ok is False, "protective_sl_ok must be False"
        assert "entry_filled_but_entry_protective_sl_exception" in result.message, (
            f"message must indicate SL exception, got: {result.message}"
        )
        assert "manual_intervention_required=true" in result.message, (
            f"message must contain manual_intervention_required=true, got: {result.message}"
        )
        assert "Connection reset" in result.message, (
            f"message must include exception text, got: {result.message}"
        )
        # CRITICAL: market exit must NOT be called
        trader.market_exit_remaining_position_with_retries.assert_not_called()


# ======================================================================
# 4. Entry protective SL placement returns failure -> NO market exit
# ======================================================================


class TestEntryProtectiveSLFailure:
    """When SL placement returns ok=False, NO market exit — halt only."""

    @pytest.mark.asyncio
    async def test_sl_returns_false_does_not_market_exit(self):
        """SL placement returns ok=False -> manual_intervention_required, NO market exit."""
        trader = _make_minimal_trader()
        fake_tc = _make_fake_tc()
        trader.trading_client = fake_tc

        from src.execution.tp_sl_execution_manager import TpSlExecutionManager
        trader._tp_sl_manager = TpSlExecutionManager(trader, trading_client=fake_tc)

        # SL placement FAILS
        trader.place_protective_stop_with_retries = AsyncMock(
            return_value=(False, None, "SL placement failed: insufficient margin"),
        )

        # Track market exit — must NOT be called
        trader.market_exit_remaining_position_with_retries = AsyncMock(
            return_value=(True, "should_not_be_called"),
        )

        intent = _open_long_intent()
        result = await trader.execute_intent(intent)

        assert result.ok is False, f"Result must be not ok, got {result}"
        assert result.entry_filled is True, "entry_filled must be True"
        assert result.protective_sl_ok is False, "protective_sl_ok must be False"
        assert "entry_filled_but_entry_protective_sl_failed" in result.message, (
            f"message must indicate SL failure, got: {result.message}"
        )
        assert "manual_intervention_required=true" in result.message, (
            f"message must contain manual_intervention_required=true, got: {result.message}"
        )
        # CRITICAL: market exit must NOT be called
        trader.market_exit_remaining_position_with_retries.assert_not_called()


# ======================================================================
# 5. Missing entry_protective_sl_price -> NO market exit
# ======================================================================


class TestEntryProtectiveSLMissing:
    """When entry_protective_sl_price is None, NO market exit — halt only."""

    @pytest.mark.asyncio
    async def test_missing_sl_price_does_not_market_exit(self):
        """Missing SL price -> manual_intervention_required, NO market exit."""
        trader = _make_minimal_trader()
        fake_tc = _make_fake_tc()
        trader.trading_client = fake_tc

        # Track market exit — must NOT be called
        trader.market_exit_remaining_position_with_retries = AsyncMock(
            return_value=(True, "should_not_be_called"),
        )

        # Intent without entry_protective_sl_price
        intent = _open_long_intent()
        # Remove entry_protective_sl_price by constructing dict and popping
        intent_dict = dict(intent.__dict__)
        intent_dict.pop("entry_protective_sl_price", None)
        from src.strategies.boll_cvd_reclaim_strategy import TradeIntent
        intent_no_sl = TradeIntent(**intent_dict)

        result = await trader.execute_intent(intent_no_sl)

        assert result.ok is False, f"Result must be not ok, got {result}"
        assert result.entry_filled is True, "entry_filled must be True"
        assert result.protective_sl_ok is False, "protective_sl_ok must be False"
        assert "manual_intervention_required=true" in result.message, (
            f"message must contain manual_intervention_required=true, got: {result.message}"
        )
        assert "missing_entry_protective_sl" in result.message, (
            f"message must indicate missing SL, got: {result.message}"
        )
        # CRITICAL: market exit must NOT be called
        trader.market_exit_remaining_position_with_retries.assert_not_called()


# ======================================================================
# 6. Trend SL update success path uses correct method
# ======================================================================


class TestTrendSLUpdateCorrectMethod:
    """UPDATE_TREND_SL uses place_protective_stop_with_retries (not the old name)."""

    @pytest.mark.asyncio
    async def test_trend_sl_update_success_uses_correct_method(self):
        """UPDATE_TREND_SL success uses place_protective_stop_with_retries."""
        trader = _make_minimal_trader()
        trader.entry_protective_sl_order_id = "old-sl-1"

        from src.execution.tp_sl_execution_manager import TpSlExecutionManager

        fake_tc = _make_fake_tc()
        trader.trading_client = fake_tc
        trader._tp_sl_manager = TpSlExecutionManager(trader, trading_client=fake_tc)

        # Mock the actual method that should be called
        trader.place_protective_stop_with_retries = AsyncMock(
            return_value=(True, "new-sl-1", "sl_placed"),
        )

        # Mock cancel_protective_stop so old SL cancellation works
        with patch.object(
            trader._tp_sl_manager, "cancel_protective_stop",
            return_value=True,
        ):
            intent = _update_trend_sl_intent(sl_price=2990.0)
            result = await trader._execute_update_trend_sl(intent)

        assert result.ok is True, f"Result must be ok, got: {result.message}"
        assert result.protective_sl_ok is True
        assert result.protective_sl_order_id == "new-sl-1"
        trader.place_protective_stop_with_retries.assert_called_once()

    @pytest.mark.asyncio
    async def test_trend_sl_update_success_does_not_call_missing_method(self):
        """place_entry_protective_stop_with_retries must have 0 calls."""
        trader = _make_minimal_trader()
        trader.entry_protective_sl_order_id = "old-sl-1"

        from src.execution.tp_sl_execution_manager import TpSlExecutionManager

        fake_tc = _make_fake_tc()
        trader.trading_client = fake_tc
        trader._tp_sl_manager = TpSlExecutionManager(trader, trading_client=fake_tc)

        # Monkey-patch the correct method directly
        trader.place_protective_stop_with_retries = AsyncMock(
            return_value=(True, "new-sl-1", "sl_placed"),
        )

        # If any code path still references the old name,
        # set it up so we can detect it
        trader.place_entry_protective_stop_with_retries = AsyncMock(
            side_effect=AssertionError(
                "place_entry_protective_stop_with_retries should not be called"
            ),
        )

        with patch.object(
            trader._tp_sl_manager, "cancel_protective_stop",
            return_value=True,
        ):
            intent = _update_trend_sl_intent(sl_price=2990.0)
            result = await trader._execute_update_trend_sl(intent)

        assert result.ok is True
        trader.place_entry_protective_stop_with_retries.assert_not_called()


# ======================================================================
# 7. Trend SL update exception -> no old SL cancel, returns ok=False
# ======================================================================


class TestTrendSLUpdateException:
    """When new SL placement raises an exception, old SL is NOT cancelled."""

    @pytest.mark.asyncio
    async def test_trend_sl_exception_does_not_cancel_old_sl(self):
        """New SL exception -> old SL kept, result.ok=False, no bubble."""
        trader = _make_minimal_trader()
        trader.entry_protective_sl_order_id = "old-sl-1"

        from src.execution.tp_sl_execution_manager import TpSlExecutionManager

        fake_tc = _make_fake_tc()
        trader.trading_client = fake_tc
        trader._tp_sl_manager = TpSlExecutionManager(trader, trading_client=fake_tc)

        # SL placement RAISES
        trader.place_protective_stop_with_retries = AsyncMock(
            side_effect=RuntimeError("SL API timeout"),
        )

        cancel_called_with = []

        async def track_cancel(algo_id):
            cancel_called_with.append(algo_id)
            return True

        with patch.object(
            trader._tp_sl_manager, "cancel_protective_stop",
            side_effect=track_cancel,
        ):
            intent = _update_trend_sl_intent(sl_price=2990.0)

            # Must NOT raise exception
            result = await trader._execute_update_trend_sl(intent)

        assert result.ok is False, f"Result must be not ok, got {result}"
        assert result.protective_sl_ok is False
        assert "trend_sl_update_failed_place_new_sl_exception" in result.message, (
            f"message must indicate SL exception, got: {result.message}"
        )
        assert "SL API timeout" in result.message, (
            f"message must include exception text, got: {result.message}"
        )
        # Old SL must NOT have been cancelled
        assert len(cancel_called_with) == 0, (
            f"cancel_protective_stop must NOT be called on exception, "
            f"but was called with: {cancel_called_with}"
        )
        # Old SL ID must still be tracked
        assert trader.entry_protective_sl_order_id == "old-sl-1", (
            "entry_protective_sl_order_id must remain unchanged on exception"
        )
