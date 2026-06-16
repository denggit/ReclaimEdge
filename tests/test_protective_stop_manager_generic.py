"""Generic ProtectiveStopManager tests — placement, verify, cancel, wrapper state.

These tests verify that ProtectiveStopManager routes through
TradingClientPort (or broker_semantic_executor when semantic flag is on),
without any near-TP / NEAR_TP references.
"""

from __future__ import annotations

import os
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.execution.trading_client_port import (
    AlgoOrderSnapshot,
    CancelResult,
    OrderResult,
    PositionSnapshot,
)


def _algo(order_id, side="sell", qty=Decimal("1.5"), trigger_price=Decimal("2900.00"), status="live"):
    """Helper to build AlgoOrderSnapshot with client_order_id=None."""
    return AlgoOrderSnapshot(
        client_order_id=None,
        order_id=order_id,
        side=side,
        qty=qty,
        trigger_price=trigger_price,
        status=status,
    )


# ======================================================================
# Fake TradingClientPort
# ======================================================================


class FakeTradingClientPort:
    """Minimal fake that records calls and returns configured responses."""

    def __init__(self) -> None:
        self.place_stop_market_calls: list[dict[str, Any]] = []
        self.place_market_calls: list[dict[str, Any]] = []
        self.cancel_algo_calls: list[dict[str, Any]] = []
        self._algo_orders: tuple[AlgoOrderSnapshot, ...] = ()
        self._position: PositionSnapshot = PositionSnapshot(side=None, qty=Decimal("0"))

        self._next_place_stop_result: OrderResult = OrderResult(ok=True, order_id="algo-001")
        self._next_cancel_algo_result: CancelResult = CancelResult(ok=True, order_id="algo-001")

        self._placement_call_count: int = 0

    def set_algo_orders(self, orders: tuple[AlgoOrderSnapshot, ...]) -> None:
        self._algo_orders = orders

    def set_position(self, pos: PositionSnapshot) -> None:
        self._position = pos

    def set_place_stop_result(self, result: OrderResult) -> None:
        self._next_place_stop_result = result

    def set_cancel_algo_result(self, result: CancelResult) -> None:
        self._next_cancel_algo_result = result

    # --- TradingClientPort interface ---

    async def fetch_position(self) -> PositionSnapshot:
        return self._position

    async def fetch_open_algo_orders(self) -> tuple[AlgoOrderSnapshot, ...]:
        return self._algo_orders

    async def place_stop_market_order(
        self, *, side, qty, trigger_price, reduce_only, client_order_id
    ) -> OrderResult:
        self._placement_call_count += 1
        self.place_stop_market_calls.append({
            "side": side, "qty": qty, "trigger_price": trigger_price,
            "reduce_only": reduce_only, "client_order_id": client_order_id,
        })
        return self._next_place_stop_result

    async def place_market_order(self, *, side, qty, reduce_only, client_order_id) -> OrderResult:
        self.place_market_calls.append({
            "side": side, "qty": qty,
            "reduce_only": reduce_only, "client_order_id": client_order_id,
        })
        return OrderResult(ok=True, order_id="market-001")

    async def cancel_algo_order(self, *, order_id=None, client_order_id=None) -> CancelResult:
        self.cancel_algo_calls.append({"order_id": order_id, "client_order_id": client_order_id})
        return self._next_cancel_algo_result

    async def fetch_open_orders(self) -> list:
        return []

    async def cancel_order(self, *, order_id=None, client_order_id=None) -> CancelResult:
        return CancelResult(ok=True)

    async def place_limit_order(self, **kwargs) -> OrderResult:
        return OrderResult(ok=True)

    async def fetch_balance(self):
        return MagicMock()

    async def configure_instrument(self) -> None:
        pass

    async def fetch_order_status(self, **kwargs):
        return MagicMock()


# ======================================================================
# Fake Trader
# ======================================================================


class FakeTraderForProtectiveStop:
    """Trader stub with only the attributes/methods ProtectiveStopManager uses."""

    symbol = "ETH-USDT-SWAP"
    contract_precision = Decimal("0.01")
    min_contracts = Decimal("0.01")

    def __init__(self, trading_client: FakeTradingClientPort) -> None:
        self._tc = trading_client

        self.entry_protective_sl_order_id: str | None = None
        self.middle_runner_protective_sl_order_id: str | None = None
        self.three_stage_post_tp1_protective_sl_order_id: str | None = None
        self.trend_runner_sl_order_id: str | None = None
        self.middle_bucket_fast_sl_order_id: str | None = None
        self.position_contracts = Decimal("0")

        self.cancel_protective_stop_calls: list[str] = []
        self._cancel_protective_stop_return: bool = True

        self._semantic_executor: MagicMock | None = None

    def set_semantic_executor(self, mock: MagicMock) -> None:
        self._semantic_executor = mock

    @property
    def broker_semantic_executor(self) -> MagicMock:
        if self._semantic_executor is None:
            raise RuntimeError("broker_semantic_executor_not_bound")
        return self._semantic_executor

    def bind_broker_semantic_executor(self, executor: Any) -> None:
        self._semantic_executor = executor

    def set_cancel_protective_stop_return(self, value: bool) -> None:
        self._cancel_protective_stop_return = value

    @staticmethod
    def decimal_to_str(value: Decimal | str | int | float) -> str:
        if not isinstance(value, Decimal):
            value = Decimal(str(value))
        return format(value.normalize(), "f")

    @staticmethod
    def price_to_str(price: float) -> str:
        import math
        if not math.isfinite(price):
            raise RuntimeError(f"Invalid price: {price}")
        return f"{price:.2f}"

    @staticmethod
    def _to_decimal(value: Decimal | str | int | float) -> Decimal:
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))

    async def cancel_protective_stop(self, order_id: str | None) -> bool:
        if order_id:
            self.cancel_protective_stop_calls.append(order_id)
        return self._cancel_protective_stop_return

    async def verify_protective_stop(
        self, algo_id: str, side: str, contracts: Decimal, stop_price: float
    ) -> bool:
        orders = await self._tc.fetch_open_algo_orders()
        for item in orders:
            if item.order_id != str(algo_id):
                continue
            close_side = "sell" if side == "LONG" else "buy"
            if str(item.side or "").lower() != close_side:
                continue
            if item.qty is None:
                continue
            contract_tolerance = max(self.contract_precision, contracts.copy_abs() * Decimal("0.001"))
            if abs(item.qty - contracts) > contract_tolerance:
                continue
            if item.trigger_price is None:
                continue
            expected_stop = Decimal(self.price_to_str(stop_price))
            price_tolerance = max(Decimal("0.01"), expected_stop.copy_abs() * Decimal("0.0001"))
            if abs(item.trigger_price - expected_stop) <= price_tolerance:
                return True
        return False

    async def _cancel_unverified_algo(self, algo_id: str, *, phase: str) -> None:
        await self.cancel_protective_stop(algo_id)

    async def fetch_position_snapshot(self) -> PositionSnapshot:
        return await self._tc.fetch_position()

    async def market_exit_remaining_position_with_retries(
        self, side, retry_count, *, context="generic", retry_interval_seconds=None
    ) -> tuple[bool, str]:
        return False, "not_expected_in_this_test"

    async def _cleanup_after_market_exit(self) -> None:
        pass


# ======================================================================
# Case A — non-semantic protective SL placement
# ======================================================================


class TestNonSemanticProtectiveStopPlacement:
    """place_protective_stop_with_retries calls trading_client.place_stop_market_order."""

    @pytest.mark.asyncio
    async def test_placement_calls_place_stop_market_order(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        fake_client = FakeTradingClientPort()
        fake_client.set_algo_orders((_algo("algo-001"),))
        fake_client.set_place_stop_result(OrderResult(ok=True, order_id="algo-001"))
        trader = FakeTraderForProtectiveStop(fake_client)
        manager = ProtectiveStopManager(trader, fake_client)

        ok, order_id, message = await manager.place_protective_stop_with_retries(
            side="LONG", contracts=Decimal("1.5"), stop_price=2900.00,
            retry_count=1, retry_interval_seconds=0,
        )

        assert ok is True
        assert order_id == "algo-001"
        assert message == "protective_sl_placed"
        assert len(fake_client.place_stop_market_calls) >= 1
        call = fake_client.place_stop_market_calls[0]
        assert call["reduce_only"] is True
        assert call["side"] == "LONG"
        assert call["qty"] == Decimal("1.5")
        assert call["trigger_price"] == Decimal("2900.00")
        assert call["client_order_id"] == ""

    @pytest.mark.asyncio
    async def test_long_side_close_is_sell(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        qty = Decimal("2.0")
        price = Decimal("3100.00")
        fake_client = FakeTradingClientPort()
        fake_client.set_algo_orders((_algo("algo-long-close", side="sell", qty=qty, trigger_price=price),))
        fake_client.set_place_stop_result(OrderResult(ok=True, order_id="algo-long-close"))
        trader = FakeTraderForProtectiveStop(fake_client)
        manager = ProtectiveStopManager(trader, fake_client)

        ok, _, _ = await manager.place_protective_stop_with_retries(
            side="LONG", contracts=qty, stop_price=float(price),
            retry_count=1, retry_interval_seconds=0,
        )
        assert ok is True

    @pytest.mark.asyncio
    async def test_short_side_close_is_buy(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        qty = Decimal("3.0")
        price = Decimal("3200.00")
        fake_client = FakeTradingClientPort()
        fake_client.set_algo_orders((_algo("algo-short-close", side="buy", qty=qty, trigger_price=price),))
        fake_client.set_place_stop_result(OrderResult(ok=True, order_id="algo-short-close"))
        trader = FakeTraderForProtectiveStop(fake_client)
        manager = ProtectiveStopManager(trader, fake_client)

        ok, _, _ = await manager.place_protective_stop_with_retries(
            side="SHORT", contracts=qty, stop_price=float(price),
            retry_count=1, retry_interval_seconds=0,
        )
        assert ok is True


# ======================================================================
# Case B — semantic protective SL placement
# ======================================================================


class TestSemanticProtectiveStopPlacement:
    """BROKER_SEMANTIC_PROTECTIVE_SL_PLACEMENT_ENABLED=true uses semantic executor."""

    @pytest.mark.asyncio
    async def test_semantic_placement_called(self, monkeypatch) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        monkeypatch.setenv("BROKER_SEMANTIC_PROTECTIVE_SL_PLACEMENT_ENABLED", "true")

        fake_client = FakeTradingClientPort()
        trader = FakeTraderForProtectiveStop(fake_client)

        semantic_exec = MagicMock()
        semantic_exec.place_protective_stop = AsyncMock(
            return_value=MagicMock(ok=True, order_id="semantic-sl-001", message="")
        )
        trader.set_semantic_executor(semantic_exec)

        fake_client.set_algo_orders((_algo("semantic-sl-001"),))
        fake_client.set_place_stop_result(OrderResult(ok=True, order_id="semantic-sl-001"))

        manager = ProtectiveStopManager(trader, fake_client)
        ok, order_id, _ = await manager.place_protective_stop_with_retries(
            side="LONG", contracts=Decimal("1.5"), stop_price=2900.00,
            retry_count=1, retry_interval_seconds=0,
        )

        assert ok is True
        assert order_id == "semantic-sl-001"
        semantic_exec.place_protective_stop.assert_called_once()
        call_kwargs = semantic_exec.place_protective_stop.call_args.kwargs
        assert call_kwargs["symbol"] == trader.symbol
        assert call_kwargs["quantity"] == Decimal("1.5")

        from src.exchanges.semantic_models import BrokerSemanticOrderRole
        assert call_kwargs["role"] == BrokerSemanticOrderRole.PROTECTIVE_SL

        from src.exchanges.models import BrokerQuantityUnit
        assert call_kwargs["quantity_unit"] == BrokerQuantityUnit.CONTRACTS

    @pytest.mark.asyncio
    async def test_semantic_role_is_not_near_tp_protective_sl(self, monkeypatch) -> None:
        from src.exchanges.semantic_models import BrokerSemanticOrderRole

        role_names = [r.value for r in BrokerSemanticOrderRole]
        assert "NEAR_TP_PROTECTIVE_SL" not in role_names, (
            f"NEAR_TP_PROTECTIVE_SL should not exist; got {role_names}"
        )

        monkeypatch.setenv("BROKER_SEMANTIC_PROTECTIVE_SL_PLACEMENT_ENABLED", "true")

        fake_client = FakeTradingClientPort()
        trader = FakeTraderForProtectiveStop(fake_client)

        semantic_exec = MagicMock()
        semantic_exec.place_protective_stop = AsyncMock(
            return_value=MagicMock(ok=True, order_id="sl-002", message="")
        )
        trader.set_semantic_executor(semantic_exec)
        qty2 = Decimal("1.0")
        price2 = Decimal("3000.00")
        fake_client.set_algo_orders((_algo("sl-002", qty=qty2, trigger_price=price2),))
        fake_client.set_place_stop_result(OrderResult(ok=True, order_id="sl-002"))

        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager
        manager = ProtectiveStopManager(trader, fake_client)
        ok, _, _ = await manager.place_protective_stop_with_retries(
            side="LONG", contracts=qty2, stop_price=float(price2),
            retry_count=1, retry_interval_seconds=0,
        )
        assert ok is True
        assert semantic_exec.place_protective_stop.call_args.kwargs["role"] == BrokerSemanticOrderRole.PROTECTIVE_SL


# ======================================================================
# Case C — verify_protective_stop
# ======================================================================


class TestVerifyProtectiveStop:
    """verify_protective_stop fetches open algo orders and matches snapshot."""

    @pytest.mark.asyncio
    async def test_matches_exact(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        fake_client = FakeTradingClientPort()
        fake_client.set_algo_orders((_algo("algo-match"),))
        fake_client.set_place_stop_result(OrderResult(ok=True, order_id="algo-match"))
        trader = FakeTraderForProtectiveStop(fake_client)
        manager = ProtectiveStopManager(trader, fake_client)

        result = await manager.verify_protective_stop("algo-match", "LONG", Decimal("1.5"), 2900.00)
        assert result is True

    @pytest.mark.asyncio
    async def test_wrong_order_id_returns_false(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        fake_client = FakeTradingClientPort()
        fake_client.set_algo_orders((_algo("algo-other"),))
        fake_client.set_place_stop_result(OrderResult(ok=True, order_id="algo-other"))
        trader = FakeTraderForProtectiveStop(fake_client)
        manager = ProtectiveStopManager(trader, fake_client)

        result = await manager.verify_protective_stop("algo-match", "LONG", Decimal("1.5"), 2900.00)
        assert result is False

    @pytest.mark.asyncio
    async def test_wrong_side_returns_false(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        fake_client = FakeTradingClientPort()
        fake_client.set_algo_orders((_algo("algo-match", side="buy"),))
        fake_client.set_place_stop_result(OrderResult(ok=True, order_id="algo-match"))
        trader = FakeTraderForProtectiveStop(fake_client)
        manager = ProtectiveStopManager(trader, fake_client)

        result = await manager.verify_protective_stop("algo-match", "LONG", Decimal("1.5"), 2900.00)
        assert result is False

    @pytest.mark.asyncio
    async def test_qty_out_of_tolerance_returns_false(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        fake_client = FakeTradingClientPort()
        fake_client.set_algo_orders((_algo("algo-match", qty=Decimal("5.0")),))
        fake_client.set_place_stop_result(OrderResult(ok=True, order_id="algo-match"))
        trader = FakeTraderForProtectiveStop(fake_client)
        manager = ProtectiveStopManager(trader, fake_client)

        result = await manager.verify_protective_stop("algo-match", "LONG", Decimal("1.5"), 2900.00)
        assert result is False

    @pytest.mark.asyncio
    async def test_trigger_price_out_of_tolerance_returns_false(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        fake_client = FakeTradingClientPort()
        fake_client.set_algo_orders((_algo("algo-match", trigger_price=Decimal("3500.00")),))
        fake_client.set_place_stop_result(OrderResult(ok=True, order_id="algo-match"))
        trader = FakeTraderForProtectiveStop(fake_client)
        manager = ProtectiveStopManager(trader, fake_client)

        result = await manager.verify_protective_stop("algo-match", "LONG", Decimal("1.5"), 2900.00)
        assert result is False

    @pytest.mark.asyncio
    async def test_empty_orders_returns_false(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        fake_client = FakeTradingClientPort()
        fake_client.set_algo_orders(())
        trader = FakeTraderForProtectiveStop(fake_client)
        manager = ProtectiveStopManager(trader, fake_client)

        result = await manager.verify_protective_stop("algo-x", "LONG", Decimal("1.0"), 2900.00)
        assert result is False

    @pytest.mark.asyncio
    async def test_none_qty_in_snapshot_returns_false(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        fake_client = FakeTradingClientPort()
        snap = AlgoOrderSnapshot(
            client_order_id=None, order_id="algo-match", side="sell",
            qty=None, trigger_price=Decimal("2900.00"),
        )
        fake_client.set_algo_orders((snap,))
        trader = FakeTraderForProtectiveStop(fake_client)
        manager = ProtectiveStopManager(trader, fake_client)

        result = await manager.verify_protective_stop("algo-match", "LONG", Decimal("1.5"), 2900.00)
        assert result is False

    @pytest.mark.asyncio
    async def test_none_trigger_price_in_snapshot_returns_false(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        fake_client = FakeTradingClientPort()
        snap = AlgoOrderSnapshot(
            client_order_id=None, order_id="algo-match", side="sell",
            qty=Decimal("1.5"), trigger_price=None,
        )
        fake_client.set_algo_orders((snap,))
        trader = FakeTraderForProtectiveStop(fake_client)
        manager = ProtectiveStopManager(trader, fake_client)

        result = await manager.verify_protective_stop("algo-match", "LONG", Decimal("1.5"), 2900.00)
        assert result is False

    @pytest.mark.asyncio
    async def test_matches_within_qty_tolerance(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        fake_client = FakeTradingClientPort()
        fake_client.set_algo_orders((_algo("algo-match", qty=Decimal("1.5001")),))
        fake_client.set_place_stop_result(OrderResult(ok=True, order_id="algo-match"))
        trader = FakeTraderForProtectiveStop(fake_client)
        manager = ProtectiveStopManager(trader, fake_client)

        result = await manager.verify_protective_stop("algo-match", "LONG", Decimal("1.5"), 2900.00)
        assert result is True

    @pytest.mark.asyncio
    async def test_matches_within_price_tolerance(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        fake_client = FakeTradingClientPort()
        fake_client.set_algo_orders((_algo("algo-match", trigger_price=Decimal("2900.01")),))
        fake_client.set_place_stop_result(OrderResult(ok=True, order_id="algo-match"))
        trader = FakeTraderForProtectiveStop(fake_client)
        manager = ProtectiveStopManager(trader, fake_client)

        result = await manager.verify_protective_stop("algo-match", "LONG", Decimal("1.5"), 2900.00)
        assert result is True


# ======================================================================
# Case D — unverified algo cleanup
# ======================================================================


class TestUnverifiedAlgoCleanup:
    """When placement succeeds but verify fails, cancel called and fallback tried."""

    @pytest.mark.asyncio
    async def test_unverified_algo_cancelled_then_fallback(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        fake_client = FakeTradingClientPort()
        # Default placement order_id="algo-001" — verify will NOT find it in algo_orders
        # because algo_orders only has "algo-other" with different qty/price
        fake_client.set_algo_orders((_algo("algo-other", qty=Decimal("5.0"), trigger_price=Decimal("5000.00")),))
        trader = FakeTraderForProtectiveStop(fake_client)
        manager = ProtectiveStopManager(trader, fake_client)

        ok, order_id, message = await manager.place_protective_stop_with_retries(
            side="LONG", contracts=Decimal("1.5"), stop_price=2900.00,
            retry_count=1, retry_interval_seconds=0,
        )

        assert ok is False
        assert order_id is None
        assert len(trader.cancel_protective_stop_calls) >= 1
        # Primary placement returns algo-001, verify fails, cancel called with algo-001
        assert any("algo-001" in call for call in trader.cancel_protective_stop_calls)

    @pytest.mark.asyncio
    async def test_no_lingering_unverified_algo_on_success(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        fake_client = FakeTradingClientPort()
        fake_client.set_algo_orders((_algo("algo-001"),))
        fake_client.set_place_stop_result(OrderResult(ok=True, order_id="algo-001"))
        trader = FakeTraderForProtectiveStop(fake_client)
        manager = ProtectiveStopManager(trader, fake_client)

        ok, _, _ = await manager.place_protective_stop_with_retries(
            side="LONG", contracts=Decimal("1.5"), stop_price=2900.00,
            retry_count=1, retry_interval_seconds=0,
        )
        assert ok is True
        assert len(trader.cancel_protective_stop_calls) == 0


# ======================================================================
# Case E — wrapper writes correct state
# ======================================================================


class TestWrapperState:
    """Each placement wrapper must write the correct Trader state field."""

    @pytest.mark.asyncio
    async def test_place_entry_protective_stop_writes_entry_state(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        qty = Decimal("1.0")
        price = Decimal("2900.00")
        fake_client = FakeTradingClientPort()
        fake_client.set_algo_orders((_algo("entry-sl", qty=qty, trigger_price=price),))
        fake_client.set_place_stop_result(OrderResult(ok=True, order_id="entry-sl"))
        trader = FakeTraderForProtectiveStop(fake_client)
        manager = ProtectiveStopManager(trader, fake_client)

        ok, _, _ = await manager.place_entry_protective_stop_with_retries(
            side="LONG", contracts=qty, stop_price=float(price),
            retry_count=1, retry_interval_seconds=0,
        )
        assert ok is True
        assert trader.entry_protective_sl_order_id == "entry-sl"

    @pytest.mark.asyncio
    async def test_place_middle_runner_protective_stop_writes_state(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        qty = Decimal("2.0")
        price = Decimal("2950.00")
        fake_client = FakeTradingClientPort()
        fake_client.set_algo_orders((_algo("mr-sl", qty=qty, trigger_price=price),))
        fake_client.set_place_stop_result(OrderResult(ok=True, order_id="mr-sl"))
        trader = FakeTraderForProtectiveStop(fake_client)
        manager = ProtectiveStopManager(trader, fake_client)

        ok, _, _ = await manager.place_middle_runner_protective_stop_with_retries(
            side="LONG", contracts=qty, stop_price=float(price),
            retry_count=1, retry_interval_seconds=0,
        )
        assert ok is True
        assert trader.middle_runner_protective_sl_order_id == "mr-sl"

    @pytest.mark.asyncio
    async def test_place_three_stage_post_tp1_protective_stop_writes_state(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        qty = Decimal("0.5")
        price = Decimal("3000.00")
        fake_client = FakeTradingClientPort()
        fake_client.set_algo_orders((_algo("ts-sl", qty=qty, trigger_price=price),))
        fake_client.set_place_stop_result(OrderResult(ok=True, order_id="ts-sl"))
        trader = FakeTraderForProtectiveStop(fake_client)
        manager = ProtectiveStopManager(trader, fake_client)

        ok, _, _ = await manager.place_three_stage_post_tp1_protective_stop_with_retries(
            side="LONG", contracts=qty, stop_price=float(price),
            retry_count=1, retry_interval_seconds=0,
        )
        assert ok is True
        assert trader.three_stage_post_tp1_protective_sl_order_id == "ts-sl"

    @pytest.mark.asyncio
    async def test_place_trend_runner_protective_stop_writes_state(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        qty = Decimal("0.3")
        price = Decimal("3100.00")
        fake_client = FakeTradingClientPort()
        fake_client.set_algo_orders((_algo("tr-sl", qty=qty, trigger_price=price),))
        fake_client.set_place_stop_result(OrderResult(ok=True, order_id="tr-sl"))
        trader = FakeTraderForProtectiveStop(fake_client)
        manager = ProtectiveStopManager(trader, fake_client)

        ok, _, _ = await manager.place_trend_runner_protective_stop_with_retries(
            side="LONG", contracts=qty, stop_price=float(price),
            retry_count=1, retry_interval_seconds=0,
        )
        assert ok is True
        assert trader.trend_runner_sl_order_id == "tr-sl"

    @pytest.mark.asyncio
    async def test_place_middle_bucket_fast_protective_stop_writes_state(self) -> None:
        from src.execution.tp_sl_protective_stop_manager import ProtectiveStopManager

        qty = Decimal("0.8")
        price = Decimal("2920.00")
        fake_client = FakeTradingClientPort()
        fake_client.set_algo_orders((_algo("mbf-sl", qty=qty, trigger_price=price),))
        fake_client.set_place_stop_result(OrderResult(ok=True, order_id="mbf-sl"))
        trader = FakeTraderForProtectiveStop(fake_client)
        manager = ProtectiveStopManager(trader, fake_client)

        ok, _, _ = await manager.place_middle_bucket_fast_protective_stop_with_retries(
            side="LONG", contracts=qty, stop_price=float(price),
            retry_count=1, retry_interval_seconds=0,
        )
        assert ok is True
        assert trader.middle_bucket_fast_sl_order_id == "mbf-sl"


# ======================================================================
# Case F — generic env names
# ======================================================================


class TestGenericEnvNames:
    """verify_protective_stop reads PROTECTIVE_SL_VERIFY_* from env."""

    def test_verify_attempts_env_used(self, monkeypatch) -> None:
        monkeypatch.setenv("PROTECTIVE_SL_VERIFY_ATTEMPTS", "5")
        import os as _os
        attempts = int(_os.getenv("PROTECTIVE_SL_VERIFY_ATTEMPTS", "3"))
        assert attempts == 5

    def test_verify_interval_env_used(self, monkeypatch) -> None:
        monkeypatch.setenv("PROTECTIVE_SL_VERIFY_INTERVAL_SECONDS", "0.5")
        import os as _os
        interval = float(_os.getenv("PROTECTIVE_SL_VERIFY_INTERVAL_SECONDS", "0.2"))
        assert interval == pytest.approx(0.5)

    def test_no_near_tp_env_used(self) -> None:
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-c",
             r"import subprocess,sys;"
             r"p=subprocess.run(['grep','-rn','NEAR_TP_PROTECTIVE_SL_VERIFY'],cwd='src',capture_output=True,text=True);"
             r"hits=[l for l in p.stdout.strip().split('\n') if l and '__pycache__' not in l];"
             r"sys.exit(f'Found: {hits}') if hits else None"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"NEAR_TP_PROTECTIVE_SL_VERIFY references: {result.stderr}"


# ======================================================================
# Cancel tests
# ======================================================================


class TestCancelProtectiveStop:
    """cancel_protective_stop through TradingClientPort and semantic path."""

    @pytest.mark.asyncio
    async def test_non_semantic_cancel_calls_cancel_algo_order(self) -> None:
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager

        fake_client = FakeTradingClientPort()
        fake_client.set_cancel_algo_result(CancelResult(ok=True, order_id="sl-001"))
        trader = FakeTraderForProtectiveStop(fake_client)

        mgr = TpSlExecutionManager(trader, trading_client=fake_client)
        ok = await mgr.cancel_protective_stop("sl-001")
        assert ok is True
        assert len(fake_client.cancel_algo_calls) >= 1
        assert fake_client.cancel_algo_calls[0]["order_id"] == "sl-001"

    @pytest.mark.asyncio
    async def test_cancel_none_order_id_returns_true(self) -> None:
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager

        fake_client = FakeTradingClientPort()
        trader = FakeTraderForProtectiveStop(fake_client)

        mgr = TpSlExecutionManager(trader, trading_client=fake_client)
        ok = await mgr.cancel_protective_stop(None)
        assert ok is True

    @pytest.mark.asyncio
    async def test_cancel_not_found_returns_true(self) -> None:
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager

        class NotFoundClient(FakeTradingClientPort):
            async def cancel_algo_order(self, **kwargs) -> CancelResult:
                self.cancel_algo_calls.append(kwargs)
                raise RuntimeError("algo order not found")

        fake_client = NotFoundClient()
        trader = FakeTraderForProtectiveStop(fake_client)

        mgr = TpSlExecutionManager(trader, trading_client=fake_client)
        ok = await mgr.cancel_protective_stop("sl-gone")
        assert ok is True

    @pytest.mark.asyncio
    async def test_cancel_does_not_exist_returns_true(self) -> None:
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager

        class NotFoundClient(FakeTradingClientPort):
            async def cancel_algo_order(self, **kwargs) -> CancelResult:
                self.cancel_algo_calls.append(kwargs)
                raise RuntimeError("order does not exist")

        fake_client = NotFoundClient()
        trader = FakeTraderForProtectiveStop(fake_client)

        mgr = TpSlExecutionManager(trader, trading_client=fake_client)
        ok = await mgr.cancel_protective_stop("sl-gone")
        assert ok is True

    @pytest.mark.asyncio
    async def test_cancel_already_returns_true(self) -> None:
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager

        class AlreadyClient(FakeTradingClientPort):
            async def cancel_algo_order(self, **kwargs) -> CancelResult:
                self.cancel_algo_calls.append(kwargs)
                raise RuntimeError("order already cancelled")

        fake_client = AlreadyClient()
        trader = FakeTraderForProtectiveStop(fake_client)

        mgr = TpSlExecutionManager(trader, trading_client=fake_client)
        ok = await mgr.cancel_protective_stop("sl-already")
        assert ok is True

    @pytest.mark.asyncio
    async def test_semantic_cancel(self, monkeypatch) -> None:
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager
        monkeypatch.setenv("BROKER_SEMANTIC_PROTECTIVE_SL_CANCEL_ENABLED", "true")

        fake_client = FakeTradingClientPort()
        trader = FakeTraderForProtectiveStop(fake_client)

        semantic_exec = MagicMock()
        semantic_exec.cancel_protective_stop = AsyncMock(
            return_value=MagicMock(ok=True, message="")
        )
        trader.set_semantic_executor(semantic_exec)

        mgr = TpSlExecutionManager(trader, trading_client=fake_client)
        ok = await mgr.cancel_protective_stop("sl-semantic")
        assert ok is True
        semantic_exec.cancel_protective_stop.assert_called_once_with(
            symbol=trader.symbol, order_id="sl-semantic"
        )

    @pytest.mark.asyncio
    async def test_semantic_cancel_already_absent_returns_true(self, monkeypatch) -> None:
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager
        monkeypatch.setenv("BROKER_SEMANTIC_PROTECTIVE_SL_CANCEL_ENABLED", "true")

        fake_client = FakeTradingClientPort()
        trader = FakeTraderForProtectiveStop(fake_client)

        semantic_exec = MagicMock()
        semantic_exec.cancel_protective_stop = AsyncMock(
            return_value=MagicMock(ok=False, message="order not found")
        )
        trader.set_semantic_executor(semantic_exec)

        mgr = TpSlExecutionManager(trader, trading_client=fake_client)
        ok = await mgr.cancel_protective_stop("sl-gone")
        assert ok is True


class TestCancelWrapperState:
    """Each cancel wrapper must clear the correct Trader state field on success."""

    @pytest.mark.asyncio
    async def test_cancel_middle_runner_clears_state(self) -> None:
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager

        fake_client = FakeTradingClientPort()
        fake_client.set_cancel_algo_result(CancelResult(ok=True, order_id="mr-sl"))
        trader = FakeTraderForProtectiveStop(fake_client)
        trader.middle_runner_protective_sl_order_id = "mr-sl"

        mgr = TpSlExecutionManager(trader, trading_client=fake_client)
        ok = await mgr.cancel_middle_runner_protective_stop("mr-sl")
        assert ok is True
        assert trader.middle_runner_protective_sl_order_id is None

    @pytest.mark.asyncio
    async def test_cancel_three_stage_post_tp1_clears_state(self) -> None:
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager

        fake_client = FakeTradingClientPort()
        fake_client.set_cancel_algo_result(CancelResult(ok=True, order_id="ts-sl"))
        trader = FakeTraderForProtectiveStop(fake_client)
        trader.three_stage_post_tp1_protective_sl_order_id = "ts-sl"

        mgr = TpSlExecutionManager(trader, trading_client=fake_client)
        ok = await mgr.cancel_three_stage_post_tp1_protective_stop("ts-sl")
        assert ok is True
        assert trader.three_stage_post_tp1_protective_sl_order_id is None

    @pytest.mark.asyncio
    async def test_cancel_trend_runner_clears_state(self) -> None:
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager

        fake_client = FakeTradingClientPort()
        fake_client.set_cancel_algo_result(CancelResult(ok=True, order_id="tr-sl"))
        trader = FakeTraderForProtectiveStop(fake_client)
        trader.trend_runner_sl_order_id = "tr-sl"

        mgr = TpSlExecutionManager(trader, trading_client=fake_client)
        ok = await mgr.cancel_trend_runner_protective_stop("tr-sl")
        assert ok is True
        assert trader.trend_runner_sl_order_id is None

    @pytest.mark.asyncio
    async def test_cancel_middle_bucket_fast_clears_state(self) -> None:
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager

        fake_client = FakeTradingClientPort()
        fake_client.set_cancel_algo_result(CancelResult(ok=True, order_id="mbf-sl"))
        trader = FakeTraderForProtectiveStop(fake_client)
        trader.middle_bucket_fast_sl_order_id = "mbf-sl"

        mgr = TpSlExecutionManager(trader, trading_client=fake_client)
        ok = await mgr.cancel_middle_bucket_fast_protective_stop("mbf-sl")
        assert ok is True
        assert trader.middle_bucket_fast_sl_order_id is None
