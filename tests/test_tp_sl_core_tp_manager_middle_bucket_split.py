"""Tests for middle_bucket_split_actual_order_mode in CoreTakeProfitManager.

Verifies that LiveTradeResult carries the correct actual_order_mode value
based on whether the split succeeded, fell back to unsplit, or fell back
to a full-size final TP.
"""

from __future__ import annotations

from decimal import Decimal
from unittest import mock

import pytest


class TestSplitNormalReturnsSplitFastSlow:
    """When split succeeds, actual_order_mode == SPLIT_FAST_SLOW."""

    @pytest.mark.asyncio
    async def test_split_normal_returns_split_fast_slow(self):
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager
        from src.execution.trader import PositionSnapshot, Trader

        trader = Trader.__new__(Trader)
        trader.symbol = "ETH-USDT-SWAP"
        trader.td_mode = "isolated"
        trader.leverage = "50"
        trader.pos_side_mode = "net"
        trader.live_trading = True
        trader.contract_multiplier = Decimal("0.1")
        trader.contract_precision = Decimal("0.01")
        trader.min_contracts = Decimal("0.01")
        trader.position_contracts = Decimal("10")
        trader.tp_order_id = None
        trader.near_tp_protective_sl_order_id = None
        trader.middle_runner_protective_sl_order_id = None
        trader.three_stage_post_tp1_protective_sl_order_id = None
        trader.trend_runner_sl_order_id = None
        trader.account_equity_usdt = 0.0
        trader._protected_reduce_only_order_ids = set()
        trader._managed_reduce_only_order_ids = set()
        trader._allow_cancel_unmanaged_reduce_only = True
        trader.decimal_to_str = lambda d: str(d)
        trader.price_to_str = lambda p: f"{p:.1f}"
        trader.round_contracts_down = lambda c: c
        trader._tp_price_summary = lambda specs: trader.price_to_str(specs[0][2])

        async def fake_fetch():
            return PositionSnapshot("LONG", Decimal("10"), 3000.0, Decimal("1"), Decimal("10"))
        trader.fetch_position_snapshot = fake_fetch

        facade = TpSlExecutionManager(trader)

        from src.strategies.boll_cvd_reclaim_strategy import TradeIntent
        from src.risk.simple_position_sizer import PositionSize

        intent = TradeIntent(
            intent_type="UPDATE_TP",
            side="LONG",
            price=3000.0,
            layer_index=1,
            tp_price=3100.0,
            reason="test_split_normal",
            size=PositionSize(
                eth_qty=0.1, margin_usdt=300.0, notional_usdt=3000.0,
                layer_index=1, layer_multiplier=1.0,
            ),
            fast_cvd=0.0, previous_fast_cvd=0.0,
            buy_ratio=0.5, sell_ratio=0.5,
            boll_upper=3200.0, boll_middle=3100.0, boll_lower=3000.0,
            ts_ms=1000,
            avg_entry_price=3000.0,
            breakeven_price=3000.0,
            tp_mode="MIDDLE",
            tp_plan="THREE_STAGE_RUNNER",
            three_stage_tp1_price=3050.0,
            three_stage_tp2_price=3200.0,
            three_stage_tp1_ratio=0.70,
            three_stage_tp2_ratio=0.20,
            three_stage_runner_ratio=0.10,
            middle_bucket_split_active=True,
            middle_bucket_split_fast_price=3060.0,
            middle_bucket_split_slow_price=3040.0,
            middle_bucket_split_effective_price=3054.0,
            middle_bucket_split_middle_bucket_ratio=0.70,
            middle_bucket_split_fast_ratio_of_bucket=0.70,
            middle_bucket_split_slow_ratio_of_bucket=0.30,
            middle_bucket_split_fast_total_ratio=0.49,
            middle_bucket_split_slow_total_ratio=0.21,
        )

        # Build specs with split (3 labels)
        multi_specs = [
            ("tp1_middle_fast", Decimal("3.43"), 3060.0),
            ("tp1_middle_slow", Decimal("1.47"), 3040.0),
            ("tp2_outer", Decimal("2.00"), 3200.0),
        ]
        facade.core_tp._build_take_profit_order_specs = mock.MagicMock(
            return_value=(multi_specs, None)  # split_disabled_reason=None → split succeeded
        )

        trader._cancel_existing_take_profit_orders_for_intent = mock.AsyncMock()
        trader._cancel_stale_runner_protective_stops_for_degrade = mock.AsyncMock()
        trader._place_reduce_only_take_profit_orders = mock.AsyncMock(
            return_value=["fast-order", "slow-order", "outer-order"]
        )

        result = await facade.replace_take_profit(intent)

        assert result.ok is True
        assert result.middle_bucket_split_executed is True
        assert result.middle_bucket_split_actual_order_mode == "SPLIT_FAST_SLOW"
        assert result.middle_bucket_split_disabled_reason is None


class TestSublegTooSmallReturnsUnsplitMiddleBucket:
    """When subleg too small, actual_order_mode == UNSPLIT_MIDDLE_BUCKET."""

    @pytest.mark.asyncio
    async def test_subleg_too_small_returns_unsplit_middle_bucket(self):
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager
        from src.execution.trader import PositionSnapshot, Trader

        trader = Trader.__new__(Trader)
        trader.symbol = "ETH-USDT-SWAP"
        trader.td_mode = "isolated"
        trader.leverage = "50"
        trader.pos_side_mode = "net"
        trader.live_trading = True
        trader.contract_multiplier = Decimal("0.1")
        trader.contract_precision = Decimal("0.01")
        trader.min_contracts = Decimal("0.01")
        trader.position_contracts = Decimal("10")
        trader.tp_order_id = None
        trader.near_tp_protective_sl_order_id = None
        trader.middle_runner_protective_sl_order_id = None
        trader.three_stage_post_tp1_protective_sl_order_id = None
        trader.trend_runner_sl_order_id = None
        trader.account_equity_usdt = 0.0
        trader._protected_reduce_only_order_ids = set()
        trader._managed_reduce_only_order_ids = set()
        trader._allow_cancel_unmanaged_reduce_only = True
        trader.decimal_to_str = lambda d: str(d)
        trader.price_to_str = lambda p: f"{p:.1f}"
        trader.round_contracts_down = lambda c: c
        trader._tp_price_summary = lambda specs: trader.price_to_str(specs[0][2])

        async def fake_fetch():
            return PositionSnapshot("LONG", Decimal("10"), 3000.0, Decimal("1"), Decimal("10"))
        trader.fetch_position_snapshot = fake_fetch

        facade = TpSlExecutionManager(trader)

        from src.strategies.boll_cvd_reclaim_strategy import TradeIntent
        from src.risk.simple_position_sizer import PositionSize

        intent = TradeIntent(
            intent_type="UPDATE_TP",
            side="LONG",
            price=3000.0,
            layer_index=1,
            tp_price=3100.0,
            reason="test_subleg_too_small",
            size=PositionSize(
                eth_qty=0.1, margin_usdt=300.0, notional_usdt=3000.0,
                layer_index=1, layer_multiplier=1.0,
            ),
            fast_cvd=0.0, previous_fast_cvd=0.0,
            buy_ratio=0.5, sell_ratio=0.5,
            boll_upper=3200.0, boll_middle=3100.0, boll_lower=3000.0,
            ts_ms=1000,
            avg_entry_price=3000.0,
            breakeven_price=3000.0,
            tp_mode="MIDDLE",
            tp_plan="MIDDLE_RUNNER",
            partial_tp_price=3050.0,
            partial_tp_ratio=0.80,
            middle_bucket_split_active=True,
            middle_bucket_split_fast_price=3060.0,
            middle_bucket_split_slow_price=3040.0,
            middle_bucket_split_effective_price=3054.0,
            middle_bucket_split_middle_bucket_ratio=0.80,
            middle_bucket_split_fast_ratio_of_bucket=0.70,
            middle_bucket_split_slow_ratio_of_bucket=0.30,
            middle_bucket_split_fast_total_ratio=0.56,
            middle_bucket_split_slow_total_ratio=0.24,
        )

        # Build specs with unsplit middle bucket (split_disabled=subleg_too_small).
        # Real order_specs produces ("middle", partial_contracts, partial_tp_price)
        # and ("runner", final_contracts, final_tp_price) for Middle Runner unsplit.
        unsplit_specs = [
            ("middle", Decimal("8"), 3050.0),
            ("runner", Decimal("2"), 3100.0),
        ]
        facade.core_tp._build_take_profit_order_specs = mock.MagicMock(
            return_value=(unsplit_specs, "subleg_too_small")
        )

        trader._cancel_existing_take_profit_orders_for_intent = mock.AsyncMock()
        trader._cancel_stale_runner_protective_stops_for_degrade = mock.AsyncMock()
        trader._place_reduce_only_take_profit_orders = mock.AsyncMock(
            return_value=["middle-order", "final-order"]
        )

        result = await facade.replace_take_profit(intent)

        assert result.ok is True
        assert result.middle_bucket_split_executed is False
        assert result.middle_bucket_split_disabled_reason == "subleg_too_small"
        assert result.middle_bucket_split_actual_order_mode == "UNSPLIT_MIDDLE_BUCKET"


class TestPlacementFailedReturnsFinalFullSize:
    """When placement fails and fallback is full-size final,
    actual_order_mode == FINAL_FULL_SIZE."""

    @pytest.mark.asyncio
    async def test_placement_failed_returns_final_full_size(self):
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager
        from src.execution.trader import PositionSnapshot, Trader

        trader = Trader.__new__(Trader)
        trader.symbol = "ETH-USDT-SWAP"
        trader.td_mode = "isolated"
        trader.leverage = "50"
        trader.pos_side_mode = "net"
        trader.live_trading = True
        trader.contract_multiplier = Decimal("0.1")
        trader.contract_precision = Decimal("0.01")
        trader.min_contracts = Decimal("0.01")
        trader.position_contracts = Decimal("10")
        trader.tp_order_id = None
        trader.near_tp_protective_sl_order_id = None
        trader.middle_runner_protective_sl_order_id = None
        trader.three_stage_post_tp1_protective_sl_order_id = None
        trader.trend_runner_sl_order_id = None
        trader.account_equity_usdt = 0.0
        trader._protected_reduce_only_order_ids = set()
        trader._managed_reduce_only_order_ids = set()
        trader._allow_cancel_unmanaged_reduce_only = True
        trader.decimal_to_str = lambda d: str(d)
        trader.price_to_str = lambda p: f"{p:.1f}"
        trader.round_contracts_down = lambda c: c
        trader._tp_price_summary = lambda specs: trader.price_to_str(specs[0][2])

        async def fake_fetch():
            return PositionSnapshot("LONG", Decimal("10"), 3000.0, Decimal("1"), Decimal("10"))
        trader.fetch_position_snapshot = fake_fetch

        facade = TpSlExecutionManager(trader)

        from src.strategies.boll_cvd_reclaim_strategy import TradeIntent
        from src.risk.simple_position_sizer import PositionSize

        intent = TradeIntent(
            intent_type="UPDATE_TP",
            side="LONG",
            price=3000.0,
            layer_index=1,
            tp_price=3100.0,
            reason="test_placement_failed",
            size=PositionSize(
                eth_qty=0.1, margin_usdt=300.0, notional_usdt=3000.0,
                layer_index=1, layer_multiplier=1.0,
            ),
            fast_cvd=0.0, previous_fast_cvd=0.0,
            buy_ratio=0.5, sell_ratio=0.5,
            boll_upper=3200.0, boll_middle=3100.0, boll_lower=3000.0,
            ts_ms=1000,
            avg_entry_price=3000.0,
            breakeven_price=3000.0,
            tp_mode="MIDDLE",
            tp_plan="THREE_STAGE_RUNNER",
            three_stage_tp1_price=3050.0,
            three_stage_tp2_price=3200.0,
            three_stage_tp1_ratio=0.70,
            three_stage_tp2_ratio=0.20,
            three_stage_runner_ratio=0.10,
            middle_bucket_split_active=True,
            middle_bucket_split_fast_price=3060.0,
            middle_bucket_split_slow_price=3040.0,
            middle_bucket_split_effective_price=3054.0,
            middle_bucket_split_middle_bucket_ratio=0.70,
            middle_bucket_split_fast_ratio_of_bucket=0.70,
            middle_bucket_split_slow_ratio_of_bucket=0.30,
            middle_bucket_split_fast_total_ratio=0.49,
            middle_bucket_split_slow_total_ratio=0.21,
        )

        # Return multiple specs so except handler triggers split→fallback logic
        multi_specs = [
            ("tp1_middle_fast", Decimal("3.43"), 3060.0),
            ("tp1_middle_slow", Decimal("1.47"), 3040.0),
            ("tp2_outer", Decimal("2.00"), 3200.0),
        ]
        facade.core_tp._build_take_profit_order_specs = mock.MagicMock(
            return_value=(multi_specs, None)
        )

        trader._cancel_existing_take_profit_orders_for_intent = mock.AsyncMock()
        trader._cancel_stale_runner_protective_stops_for_degrade = mock.AsyncMock()

        # First placement raises, second (fallback final) succeeds
        call_count = [0]

        async def mock_place(inner_intent, specs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("simulated placement failure")
            return ["fallback-final-order"]

        trader._place_reduce_only_take_profit_orders = mock_place

        result = await facade.replace_take_profit(intent)

        assert result.ok is True
        assert result.middle_bucket_split_executed is False
        assert (
            result.middle_bucket_split_disabled_reason
            == "split_order_placement_failed_fallback_final"
        )
        assert result.middle_bucket_split_actual_order_mode == "FINAL_FULL_SIZE"
        assert "fallback" in result.message


class TestNoSplitActiveReturnsNone:
    """When split is not active, actual_order_mode is None."""

    @pytest.mark.asyncio
    async def test_no_split_active_returns_none(self):
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager
        from src.execution.trader import PositionSnapshot, Trader

        trader = Trader.__new__(Trader)
        trader.symbol = "ETH-USDT-SWAP"
        trader.td_mode = "isolated"
        trader.leverage = "50"
        trader.pos_side_mode = "net"
        trader.live_trading = True
        trader.contract_multiplier = Decimal("0.1")
        trader.contract_precision = Decimal("0.01")
        trader.min_contracts = Decimal("0.01")
        trader.position_contracts = Decimal("10")
        trader.tp_order_id = None
        trader.near_tp_protective_sl_order_id = None
        trader.middle_runner_protective_sl_order_id = None
        trader.three_stage_post_tp1_protective_sl_order_id = None
        trader.trend_runner_sl_order_id = None
        trader.account_equity_usdt = 0.0
        trader._protected_reduce_only_order_ids = set()
        trader._managed_reduce_only_order_ids = set()
        trader._allow_cancel_unmanaged_reduce_only = True
        trader.decimal_to_str = lambda d: str(d)
        trader.price_to_str = lambda p: f"{p:.1f}"
        trader.round_contracts_down = lambda c: c
        trader._tp_price_summary = lambda specs: trader.price_to_str(specs[0][2])

        async def fake_fetch():
            return PositionSnapshot("LONG", Decimal("10"), 3000.0, Decimal("1"), Decimal("10"))
        trader.fetch_position_snapshot = fake_fetch

        facade = TpSlExecutionManager(trader)

        from src.strategies.boll_cvd_reclaim_strategy import TradeIntent
        from src.risk.simple_position_sizer import PositionSize

        intent = TradeIntent(
            intent_type="UPDATE_TP",
            side="LONG",
            price=3000.0,
            layer_index=1,
            tp_price=3100.0,
            reason="test_no_split",
            size=PositionSize(
                eth_qty=0.1, margin_usdt=300.0, notional_usdt=3000.0,
                layer_index=1, layer_multiplier=1.0,
            ),
            fast_cvd=0.0, previous_fast_cvd=0.0,
            buy_ratio=0.5, sell_ratio=0.5,
            boll_upper=3200.0, boll_middle=3100.0, boll_lower=3000.0,
            ts_ms=1000,
            avg_entry_price=3000.0,
            breakeven_price=3000.0,
            tp_mode="MIDDLE",
            tp_plan="SINGLE",
            middle_bucket_split_active=False,
        )

        specs = [("final", Decimal("10"), 3100.0)]
        facade.core_tp._build_take_profit_order_specs = mock.MagicMock(
            return_value=(specs, None)
        )

        trader._cancel_existing_take_profit_orders_for_intent = mock.AsyncMock()
        trader._cancel_stale_runner_protective_stops_for_degrade = mock.AsyncMock()
        trader._place_reduce_only_take_profit_orders = mock.AsyncMock(
            return_value=["final-order"]
        )

        result = await facade.replace_take_profit(intent)

        assert result.ok is True
        assert result.middle_bucket_split_executed is None
        assert result.middle_bucket_split_actual_order_mode is None


# ── Label-based classifier direct tests ─────────────────────────────────

class TestClassifierDirect:
    """Direct tests for _classify_middle_bucket_split_actual_order_mode."""

    def _classify(self, *, split_was_active, labels, reason=None):
        from src.execution.tp_sl_core_tp_manager import (
            _classify_middle_bucket_split_actual_order_mode,
        )
        from decimal import Decimal
        specs = [(label, Decimal("1"), 3000.0) for label in labels]
        return _classify_middle_bucket_split_actual_order_mode(
            split_was_active=split_was_active,
            specs=specs,
            split_disabled_reason=reason,
        )

    def test_not_active_returns_none(self):
        executed, reason, mode = self._classify(
            split_was_active=False, labels=["final"],
        )
        assert executed is None
        assert reason is None
        assert mode is None

    def test_three_stage_split_labels(self):
        """tp1_middle_fast + tp1_middle_slow + tp2_outer → SPLIT_FAST_SLOW."""
        executed, reason, mode = self._classify(
            split_was_active=True,
            labels=["tp1_middle_fast", "tp1_middle_slow", "tp2_outer"],
        )
        assert executed is True
        assert reason is None
        assert mode == "SPLIT_FAST_SLOW"

    def test_middle_runner_split_labels(self):
        """middle_fast + middle_slow + runner → SPLIT_FAST_SLOW."""
        executed, reason, mode = self._classify(
            split_was_active=True,
            labels=["middle_fast", "middle_slow", "runner"],
        )
        assert executed is True
        assert reason is None
        assert mode == "SPLIT_FAST_SLOW"

    def test_final_label_produces_final_full_size(self):
        """Single "final" label → FINAL_FULL_SIZE, even without reason."""
        executed, reason, mode = self._classify(
            split_was_active=True, labels=["final"],
        )
        assert executed is False
        assert reason == "split_fallback_final_order_structure"
        assert mode == "FINAL_FULL_SIZE"

    def test_final_label_preserves_existing_reason(self):
        """When split_disabled_reason is already set, it is preserved."""
        executed, reason, mode = self._classify(
            split_was_active=True, labels=["final"],
            reason="split_order_placement_failed_fallback_final",
        )
        assert executed is False
        assert reason == "split_order_placement_failed_fallback_final"
        assert mode == "FINAL_FULL_SIZE"

    def test_three_stage_unsplit_labels(self):
        """tp1_middle + tp2_outer → UNSPLIT_MIDDLE_BUCKET."""
        executed, reason, mode = self._classify(
            split_was_active=True,
            labels=["tp1_middle", "tp2_outer"],
        )
        assert executed is False
        assert reason == "split_fallback_unsplit_middle_bucket"
        assert mode == "UNSPLIT_MIDDLE_BUCKET"

    def test_middle_runner_unsplit_labels(self):
        """middle + runner → UNSPLIT_MIDDLE_BUCKET."""
        executed, reason, mode = self._classify(
            split_was_active=True,
            labels=["middle", "runner"],
        )
        assert executed is False
        assert reason == "split_fallback_unsplit_middle_bucket"
        assert mode == "UNSPLIT_MIDDLE_BUCKET"

    def test_unknown_labels_fails_safe(self):
        """Unknown labels → FINAL_FULL_SIZE (never returns executed=True)."""
        executed, reason, mode = self._classify(
            split_was_active=True,
            labels=["weird_label", "another_weird"],
        )
        assert executed is False
        assert reason == "split_unknown_order_structure_fallback_final"
        assert mode == "FINAL_FULL_SIZE"

    def test_post_tp1_tp2_only_order_structure(self):
        """labels={"tp2_outer"} with split_was_active=True → POST_TP1_TP2_ONLY.

        This is the Three-Stage post-TP1 waiting-TP2 phase where only the
        tp2_outer order is placed.  It is a legitimate structure that must
        NOT trigger unknown fallback or degrade to SINGLE.
        """
        executed, reason, mode = self._classify(
            split_was_active=True,
            labels=["tp2_outer"],
        )
        assert executed is True
        assert reason is None
        assert mode == "POST_TP1_TP2_ONLY"

    def test_post_tp1_tp2_only_with_reason_preserves_none(self):
        """Even if split_disabled_reason is passed, POST_TP1_TP2_ONLY ignores it.

        The split_disabled_reason is irrelevant for this mode — the order
        structure is intentional, not a fallback.
        """
        executed, reason, mode = self._classify(
            split_was_active=True,
            labels=["tp2_outer"],
            reason="some_other_reason",
        )
        assert executed is True
        assert reason is None
        assert mode == "POST_TP1_TP2_ONLY"

    def test_unknown_labels_preserves_existing_reason(self):
        """Unknown labels with existing reason preserves it."""
        executed, reason, mode = self._classify(
            split_was_active=True,
            labels=["weird_label"],
            reason="custom_reason",
        )
        assert executed is False
        assert reason == "custom_reason"
        assert mode == "FINAL_FULL_SIZE"

    def test_three_stage_final_with_size_fallback_reason(self):
        """THREE_STAGE_TP_SPLIT_FALLBACK_SINGLE_SIZE_TOO_SMALL → FINAL_FULL_SIZE."""
        executed, reason, mode = self._classify(
            split_was_active=True, labels=["final"],
            reason="split_fallback_final_order_structure",
        )
        assert executed is False
        assert reason == "split_fallback_final_order_structure"
        assert mode == "FINAL_FULL_SIZE"

    # ── Test 1 & 2: Partial split structures are recognized as valid ────

    def test_fast_consumed_slow_pending_is_valid_partial_split(self):
        """fast consumed, only slow + tp2 remain → PARTIAL_SPLIT_SLOW_PENDING.
        Must NOT return FINAL_FULL_SIZE. Must NOT trigger degrade."""
        executed, reason, mode = self._classify(
            split_was_active=True,
            labels=["tp1_middle_slow", "tp2_outer"],
        )
        assert executed is True, (
            "Partial split (slow + tp2) must be recognized as valid"
        )
        assert reason is None
        assert mode == "PARTIAL_SPLIT_SLOW_PENDING"

    def test_slow_consumed_fast_pending_is_valid_partial_split(self):
        """slow consumed, only fast + tp2 remain → PARTIAL_SPLIT_FAST_PENDING.
        Must NOT return FINAL_FULL_SIZE. Must NOT trigger degrade."""
        executed, reason, mode = self._classify(
            split_was_active=True,
            labels=["tp1_middle_fast", "tp2_outer"],
        )
        assert executed is True, (
            "Partial split (fast + tp2) must be recognized as valid"
        )
        assert reason is None
        assert mode == "PARTIAL_SPLIT_FAST_PENDING"

    # ── Test 3: Unknown structures still fall back ────────────────────

    def test_only_fast_without_tp2_is_unknown(self):
        """tp1_middle_fast alone (no tp2_outer) → unknown → FINAL_FULL_SIZE."""
        executed, reason, mode = self._classify(
            split_was_active=True,
            labels=["tp1_middle_fast"],
        )
        assert executed is False
        assert mode == "FINAL_FULL_SIZE"
        assert reason is not None

    # ── Test 4: Full split still SPLIT_FAST_SLOW ──────────────────────

    def test_full_split_still_split_fast_slow(self):
        """labels={"tp1_middle_fast", "tp1_middle_slow", "tp2_outer"} → SPLIT_FAST_SLOW."""
        executed, reason, mode = self._classify(
            split_was_active=True,
            labels=["tp1_middle_fast", "tp1_middle_slow", "tp2_outer"],
        )
        assert executed is True
        assert reason is None
        assert mode == "SPLIT_FAST_SLOW"

    # ── Test 5: Post-TP1 tp2 only still POST_TP1_TP2_ONLY ─────────────

    def test_post_tp1_tp2_only_still_post_tp1_tp2_only(self):
        """labels={"tp2_outer"} → POST_TP1_TP2_ONLY (existing behavior unchanged)."""
        executed, reason, mode = self._classify(
            split_was_active=True,
            labels=["tp2_outer"],
        )
        assert executed is True
        assert reason is None
        assert mode == "POST_TP1_TP2_ONLY"

    # ── Test: Partial split with reason still returns None reason ─────

    def test_partial_split_slow_pending_ignores_disabled_reason(self):
        """Even if split_disabled_reason is passed, partial split ignores it."""
        executed, reason, mode = self._classify(
            split_was_active=True,
            labels=["tp1_middle_slow", "tp2_outer"],
            reason="some_other_reason",
        )
        assert executed is True
        assert reason is None
        assert mode == "PARTIAL_SPLIT_SLOW_PENDING"

    def test_partial_split_fast_pending_ignores_disabled_reason(self):
        """Even if split_disabled_reason is passed, partial split ignores it."""
        executed, reason, mode = self._classify(
            split_was_active=True,
            labels=["tp1_middle_fast", "tp2_outer"],
            reason="some_other_reason",
        )
        assert executed is True
        assert reason is None
        assert mode == "PARTIAL_SPLIT_FAST_PENDING"


# ── Integration: Three-Stage TP2/runner too small → FINAL_FULL_SIZE ────

class TestThreeStageTp2TooSmallClassifiesFinalFullSize:
    """When the pre-check passes but order_specs falls back to single final,
    the classifier must return FINAL_FULL_SIZE."""

    @pytest.mark.asyncio
    async def test_three_stage_split_tp2_or_runner_too_small_classifies_final_full_size(self):
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager
        from src.execution.trader import PositionSnapshot, Trader
        from decimal import Decimal

        trader = Trader.__new__(Trader)
        trader.symbol = "ETH-USDT-SWAP"
        trader.td_mode = "isolated"
        trader.leverage = "50"
        trader.pos_side_mode = "net"
        trader.live_trading = True
        trader.contract_multiplier = Decimal("0.1")
        trader.contract_precision = Decimal("0.01")
        trader.min_contracts = Decimal("0.01")
        trader.position_contracts = Decimal("10")
        trader.tp_order_id = None
        trader.near_tp_protective_sl_order_id = None
        trader.middle_runner_protective_sl_order_id = None
        trader.three_stage_post_tp1_protective_sl_order_id = None
        trader.trend_runner_sl_order_id = None
        trader.account_equity_usdt = 0.0
        trader._protected_reduce_only_order_ids = set()
        trader._managed_reduce_only_order_ids = set()
        trader._allow_cancel_unmanaged_reduce_only = True
        trader.decimal_to_str = lambda d: str(d)
        trader.price_to_str = lambda p: f"{p:.1f}"
        trader.round_contracts_down = lambda c: c
        trader._tp_price_summary = lambda specs: trader.price_to_str(specs[0][2])

        async def fake_fetch():
            return PositionSnapshot("LONG", Decimal("10"), 3000.0, Decimal("1"), Decimal("10"))
        trader.fetch_position_snapshot = fake_fetch

        facade = TpSlExecutionManager(trader)

        from src.strategies.boll_cvd_reclaim_strategy import TradeIntent
        from src.risk.simple_position_sizer import PositionSize

        intent = TradeIntent(
            intent_type="UPDATE_TP",
            side="LONG",
            price=3000.0,
            layer_index=1,
            tp_price=3100.0,
            reason="test_tp2_too_small",
            size=PositionSize(eth_qty=0.1, margin_usdt=300.0, notional_usdt=3000.0,
                              layer_index=1, layer_multiplier=1.0),
            fast_cvd=0.0, previous_fast_cvd=0.0,
            buy_ratio=0.5, sell_ratio=0.5,
            boll_upper=3200.0, boll_middle=3100.0, boll_lower=3000.0,
            ts_ms=1000,
            avg_entry_price=3000.0, breakeven_price=3000.0,
            tp_mode="MIDDLE",
            tp_plan="THREE_STAGE_RUNNER",
            three_stage_tp1_price=3050.0,
            three_stage_tp2_price=3200.0,
            three_stage_tp1_ratio=0.70,
            three_stage_tp2_ratio=0.20,
            three_stage_runner_ratio=0.10,
            middle_bucket_split_active=True,
            middle_bucket_split_fast_price=3060.0,
            middle_bucket_split_slow_price=3040.0,
            middle_bucket_split_effective_price=3054.0,
            middle_bucket_split_middle_bucket_ratio=0.70,
            middle_bucket_split_fast_ratio_of_bucket=0.70,
            middle_bucket_split_slow_ratio_of_bucket=0.30,
            middle_bucket_split_fast_total_ratio=0.49,
            middle_bucket_split_slow_total_ratio=0.21,
        )

        # Simulate order_specs fallback: pre-check passes (split_disabled_reason=None)
        # but order_specs returns single final due to TP2/runner too small.
        final_specs = [("final", Decimal("10"), 3100.0)]
        facade.core_tp._build_take_profit_order_specs = mock.MagicMock(
            return_value=(final_specs, None)
        )

        trader._cancel_existing_take_profit_orders_for_intent = mock.AsyncMock()
        trader._cancel_stale_runner_protective_stops_for_degrade = mock.AsyncMock()
        trader._place_reduce_only_take_profit_orders = mock.AsyncMock(
            return_value=["final-order"]
        )

        result = await facade.replace_take_profit(intent)

        assert result.ok is True
        # The classifier sees labels={"final"} → FINAL_FULL_SIZE, NOT SPLIT_FAST_SLOW
        assert result.middle_bucket_split_executed is False
        assert result.middle_bucket_split_actual_order_mode == "FINAL_FULL_SIZE"
        assert result.middle_bucket_split_disabled_reason is not None


# ── Integration: Middle Runner runner too small → FINAL_FULL_SIZE ───────

class TestMiddleRunnerRunnerTooSmallClassifiesFinalFullSize:
    """When the pre-check passes but runner is too small for Middle Runner,
    the classifier must return FINAL_FULL_SIZE."""

    @pytest.mark.asyncio
    async def test_middle_runner_split_runner_too_small_classifies_final_full_size(self):
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager
        from src.execution.trader import PositionSnapshot, Trader
        from decimal import Decimal

        trader = Trader.__new__(Trader)
        trader.symbol = "ETH-USDT-SWAP"
        trader.td_mode = "isolated"
        trader.leverage = "50"
        trader.pos_side_mode = "net"
        trader.live_trading = True
        trader.contract_multiplier = Decimal("0.1")
        trader.contract_precision = Decimal("0.01")
        trader.min_contracts = Decimal("0.01")
        trader.position_contracts = Decimal("10")
        trader.tp_order_id = None
        trader.near_tp_protective_sl_order_id = None
        trader.middle_runner_protective_sl_order_id = None
        trader.three_stage_post_tp1_protective_sl_order_id = None
        trader.trend_runner_sl_order_id = None
        trader.account_equity_usdt = 0.0
        trader._protected_reduce_only_order_ids = set()
        trader._managed_reduce_only_order_ids = set()
        trader._allow_cancel_unmanaged_reduce_only = True
        trader.decimal_to_str = lambda d: str(d)
        trader.price_to_str = lambda p: f"{p:.1f}"
        trader.round_contracts_down = lambda c: c
        trader._tp_price_summary = lambda specs: trader.price_to_str(specs[0][2])

        async def fake_fetch():
            return PositionSnapshot("LONG", Decimal("10"), 3000.0, Decimal("1"), Decimal("10"))
        trader.fetch_position_snapshot = fake_fetch

        facade = TpSlExecutionManager(trader)

        from src.strategies.boll_cvd_reclaim_strategy import TradeIntent
        from src.risk.simple_position_sizer import PositionSize

        intent = TradeIntent(
            intent_type="UPDATE_TP",
            side="LONG",
            price=3000.0,
            layer_index=1,
            tp_price=3100.0,
            reason="test_runner_too_small",
            size=PositionSize(eth_qty=0.1, margin_usdt=300.0, notional_usdt=3000.0,
                              layer_index=1, layer_multiplier=1.0),
            fast_cvd=0.0, previous_fast_cvd=0.0,
            buy_ratio=0.5, sell_ratio=0.5,
            boll_upper=3200.0, boll_middle=3100.0, boll_lower=3000.0,
            ts_ms=1000,
            avg_entry_price=3000.0, breakeven_price=3000.0,
            tp_mode="MIDDLE",
            tp_plan="MIDDLE_RUNNER",
            partial_tp_price=3050.0,
            partial_tp_ratio=0.80,
            middle_bucket_split_active=True,
            middle_bucket_split_fast_price=3060.0,
            middle_bucket_split_slow_price=3040.0,
            middle_bucket_split_effective_price=3054.0,
            middle_bucket_split_middle_bucket_ratio=0.80,
            middle_bucket_split_fast_ratio_of_bucket=0.70,
            middle_bucket_split_slow_ratio_of_bucket=0.30,
            middle_bucket_split_fast_total_ratio=0.56,
            middle_bucket_split_slow_total_ratio=0.24,
        )

        # Simulate: pre-check passes but order_specs returns final (runner too small)
        final_specs = [("final", Decimal("10"), 3100.0)]
        facade.core_tp._build_take_profit_order_specs = mock.MagicMock(
            return_value=(final_specs, None)
        )

        trader._cancel_existing_take_profit_orders_for_intent = mock.AsyncMock()
        trader._cancel_stale_runner_protective_stops_for_degrade = mock.AsyncMock()
        trader._place_reduce_only_take_profit_orders = mock.AsyncMock(
            return_value=["final-order"]
        )

        result = await facade.replace_take_profit(intent)

        assert result.ok is True
        assert result.middle_bucket_split_executed is False
        assert result.middle_bucket_split_actual_order_mode == "FINAL_FULL_SIZE"
        assert result.middle_bucket_split_disabled_reason is not None


# ── Protective SL failure paths preserve actual_order_mode ───────────────


class TestMiddleRunnerSlFailurePreservesActualOrderModeFinalFullSize:
    """When middle runner protective SL fails and specs are final,
    the classifier (called BEFORE protective SL) must still classify
    as FINAL_FULL_SIZE and the fields must be carried in the early return."""

    @pytest.mark.asyncio
    async def test_middle_runner_sl_failure_preserves_actual_order_mode_final_full_size(self):
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager
        from src.execution.trader import PositionSnapshot, Trader

        trader = Trader.__new__(Trader)
        trader.symbol = "ETH-USDT-SWAP"
        trader.td_mode = "isolated"
        trader.leverage = "50"
        trader.pos_side_mode = "net"
        trader.live_trading = True
        trader.contract_multiplier = Decimal("0.1")
        trader.contract_precision = Decimal("0.01")
        trader.min_contracts = Decimal("0.01")
        trader.position_contracts = Decimal("10")
        trader.tp_order_id = None
        trader.near_tp_protective_sl_order_id = None
        trader.middle_runner_protective_sl_order_id = None
        trader.three_stage_post_tp1_protective_sl_order_id = None
        trader.trend_runner_sl_order_id = None
        trader.account_equity_usdt = 0.0
        trader._protected_reduce_only_order_ids = set()
        trader._managed_reduce_only_order_ids = set()
        trader._allow_cancel_unmanaged_reduce_only = True
        trader.decimal_to_str = lambda d: str(d)
        trader.price_to_str = lambda p: f"{p:.1f}"
        trader.round_contracts_down = lambda c: c
        trader._tp_price_summary = lambda specs: trader.price_to_str(specs[0][2])

        async def fake_fetch():
            return PositionSnapshot("LONG", Decimal("10"), 3000.0, Decimal("1"), Decimal("10"))
        trader.fetch_position_snapshot = fake_fetch

        facade = TpSlExecutionManager(trader)

        from src.strategies.boll_cvd_reclaim_strategy import TradeIntent
        from src.risk.simple_position_sizer import PositionSize

        intent = TradeIntent(
            intent_type="UPDATE_TP",
            side="LONG",
            price=3000.0,
            layer_index=1,
            tp_price=3100.0,
            reason="test_middle_runner_sl_failure",
            size=PositionSize(
                eth_qty=0.1, margin_usdt=300.0, notional_usdt=3000.0,
                layer_index=1, layer_multiplier=1.0,
            ),
            fast_cvd=0.0, previous_fast_cvd=0.0,
            buy_ratio=0.5, sell_ratio=0.5,
            boll_upper=3200.0, boll_middle=3100.0, boll_lower=3000.0,
            ts_ms=1000,
            avg_entry_price=3000.0,
            breakeven_price=3000.0,
            tp_mode="MIDDLE",
            tp_plan="THREE_STAGE_RUNNER",
            three_stage_tp1_price=3050.0,
            three_stage_tp2_price=3200.0,
            three_stage_tp1_ratio=0.70,
            three_stage_tp2_ratio=0.20,
            three_stage_runner_ratio=0.10,
            middle_bucket_split_active=True,
            middle_bucket_split_fast_price=3060.0,
            middle_bucket_split_slow_price=3040.0,
            middle_bucket_split_effective_price=3054.0,
            middle_bucket_split_middle_bucket_ratio=0.70,
            middle_bucket_split_fast_ratio_of_bucket=0.70,
            middle_bucket_split_slow_ratio_of_bucket=0.30,
            middle_bucket_split_fast_total_ratio=0.49,
            middle_bucket_split_slow_total_ratio=0.21,
            # Trigger middle runner protective SL path
            middle_runner_active=True,
            middle_runner_protective_sl_price=2900.0,
        )

        # Build specs that yield single "final" (e.g. fallback to final)
        final_specs = [("final", Decimal("10"), 3100.0)]
        facade.core_tp._build_take_profit_order_specs = mock.MagicMock(
            return_value=(final_specs, None)
        )

        trader._cancel_existing_take_profit_orders_for_intent = mock.AsyncMock()
        trader._cancel_stale_runner_protective_stops_for_degrade = mock.AsyncMock()
        trader._place_reduce_only_take_profit_orders = mock.AsyncMock(
            return_value=["final-order"]
        )

        # Mock middle runner SL failure
        trader.place_middle_runner_protective_stop_with_retries = mock.AsyncMock(
            return_value=(False, None, "simulated middle runner SL failure")
        )

        result = await facade.replace_take_profit(intent)

        assert result.ok is False
        assert "middle_runner_protective_sl_failed" in result.message
        assert result.middle_bucket_split_executed is False
        assert result.middle_bucket_split_actual_order_mode == "FINAL_FULL_SIZE"
        assert result.middle_bucket_split_disabled_reason is not None


class TestThreeStagePostTp1SlFailurePreservesActualOrderModeUnsplitMiddleBucket:
    """When three-stage post-TP1 protective SL fails and specs are unsplit
    middle bucket, the classifier must return UNSPLIT_MIDDLE_BUCKET."""

    @pytest.mark.asyncio
    async def test_three_stage_post_tp1_sl_failure_preserves_actual_order_mode_unsplit_middle_bucket(self):
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager
        from src.execution.trader import PositionSnapshot, Trader

        trader = Trader.__new__(Trader)
        trader.symbol = "ETH-USDT-SWAP"
        trader.td_mode = "isolated"
        trader.leverage = "50"
        trader.pos_side_mode = "net"
        trader.live_trading = True
        trader.contract_multiplier = Decimal("0.1")
        trader.contract_precision = Decimal("0.01")
        trader.min_contracts = Decimal("0.01")
        trader.position_contracts = Decimal("10")
        trader.tp_order_id = None
        trader.near_tp_protective_sl_order_id = None
        trader.middle_runner_protective_sl_order_id = None
        trader.three_stage_post_tp1_protective_sl_order_id = None
        trader.trend_runner_sl_order_id = None
        trader.account_equity_usdt = 0.0
        trader._protected_reduce_only_order_ids = set()
        trader._managed_reduce_only_order_ids = set()
        trader._allow_cancel_unmanaged_reduce_only = True
        trader.decimal_to_str = lambda d: str(d)
        trader.price_to_str = lambda p: f"{p:.1f}"
        trader.round_contracts_down = lambda c: c
        trader._tp_price_summary = lambda specs: trader.price_to_str(specs[0][2])

        async def fake_fetch():
            return PositionSnapshot("LONG", Decimal("10"), 3000.0, Decimal("1"), Decimal("10"))
        trader.fetch_position_snapshot = fake_fetch

        facade = TpSlExecutionManager(trader)

        from src.strategies.boll_cvd_reclaim_strategy import TradeIntent
        from src.risk.simple_position_sizer import PositionSize

        intent = TradeIntent(
            intent_type="UPDATE_TP",
            side="LONG",
            price=3000.0,
            layer_index=1,
            tp_price=3100.0,
            reason="test_post_tp1_sl_failure",
            size=PositionSize(
                eth_qty=0.1, margin_usdt=300.0, notional_usdt=3000.0,
                layer_index=1, layer_multiplier=1.0,
            ),
            fast_cvd=0.0, previous_fast_cvd=0.0,
            buy_ratio=0.5, sell_ratio=0.5,
            boll_upper=3200.0, boll_middle=3100.0, boll_lower=3000.0,
            ts_ms=1000,
            avg_entry_price=3000.0,
            breakeven_price=3000.0,
            tp_mode="MIDDLE",
            tp_plan="THREE_STAGE_RUNNER",
            three_stage_tp1_price=3050.0,
            three_stage_tp2_price=3200.0,
            three_stage_tp1_ratio=0.70,
            three_stage_tp2_ratio=0.20,
            three_stage_runner_ratio=0.10,
            middle_bucket_split_active=True,
            middle_bucket_split_fast_price=3060.0,
            middle_bucket_split_slow_price=3040.0,
            middle_bucket_split_effective_price=3054.0,
            middle_bucket_split_middle_bucket_ratio=0.70,
            middle_bucket_split_fast_ratio_of_bucket=0.70,
            middle_bucket_split_slow_ratio_of_bucket=0.30,
            middle_bucket_split_fast_total_ratio=0.49,
            middle_bucket_split_slow_total_ratio=0.21,
            # Trigger post-TP1 protective SL path (NOT middle runner, NOT trend runner)
            middle_runner_active=False,
            trend_runner_active=False,
            three_stage_tp1_consumed=True,
            three_stage_tp2_consumed=False,
            three_stage_post_tp1_protective_sl_price=2950.0,
        )

        # Build specs with unsplit middle bucket labels
        unsplit_specs = [
            ("tp1_middle", Decimal("7"), 3050.0),
            ("tp2_outer", Decimal("3"), 3200.0),
        ]
        facade.core_tp._build_take_profit_order_specs = mock.MagicMock(
            return_value=(unsplit_specs, "subleg_too_small")
        )

        trader._cancel_existing_take_profit_orders_for_intent = mock.AsyncMock()
        trader._cancel_stale_runner_protective_stops_for_degrade = mock.AsyncMock()
        trader._place_reduce_only_take_profit_orders = mock.AsyncMock(
            return_value=["tp1-middle", "tp2-outer"]
        )

        # Mock three-stage post-TP1 SL failure
        trader.place_three_stage_post_tp1_protective_stop_with_retries = mock.AsyncMock(
            return_value=(False, None, "simulated post_tp1 SL failure")
        )

        result = await facade.replace_take_profit(intent)

        assert result.ok is False
        assert "three_stage_post_tp1_protective_sl_failed" in result.message
        assert result.middle_bucket_split_executed is False
        assert result.middle_bucket_split_actual_order_mode == "UNSPLIT_MIDDLE_BUCKET"
        assert result.middle_bucket_split_disabled_reason == "subleg_too_small"


class TestTrendRunnerSlFailurePreservesActualOrderModeSplitFastSlow:
    """When trend runner protective SL fails and specs were real split labels,
    the classifier must return SPLIT_FAST_SLOW and the early return must carry it."""

    @pytest.mark.asyncio
    async def test_trend_runner_sl_failure_preserves_actual_order_mode_split_fast_slow(self):
        from src.execution.tp_sl_execution_manager import TpSlExecutionManager
        from src.execution.trader import PositionSnapshot, Trader

        trader = Trader.__new__(Trader)
        trader.symbol = "ETH-USDT-SWAP"
        trader.td_mode = "isolated"
        trader.leverage = "50"
        trader.pos_side_mode = "net"
        trader.live_trading = True
        trader.contract_multiplier = Decimal("0.1")
        trader.contract_precision = Decimal("0.01")
        trader.min_contracts = Decimal("0.01")
        trader.position_contracts = Decimal("10")
        trader.tp_order_id = None
        trader.near_tp_protective_sl_order_id = None
        trader.middle_runner_protective_sl_order_id = None
        trader.three_stage_post_tp1_protective_sl_order_id = None
        trader.trend_runner_sl_order_id = None
        trader.account_equity_usdt = 0.0
        trader._protected_reduce_only_order_ids = set()
        trader._managed_reduce_only_order_ids = set()
        trader._allow_cancel_unmanaged_reduce_only = True
        trader.decimal_to_str = lambda d: str(d)
        trader.price_to_str = lambda p: f"{p:.1f}"
        trader.round_contracts_down = lambda c: c
        trader._tp_price_summary = lambda specs: trader.price_to_str(specs[0][2])

        # _trend_runner_sl_contracts is needed by the trend runner SL path
        trader._trend_runner_sl_contracts = lambda intent, net: Decimal("5")

        async def fake_fetch():
            return PositionSnapshot("LONG", Decimal("10"), 3000.0, Decimal("1"), Decimal("10"))
        trader.fetch_position_snapshot = fake_fetch

        facade = TpSlExecutionManager(trader)

        from src.strategies.boll_cvd_reclaim_strategy import TradeIntent
        from src.risk.simple_position_sizer import PositionSize

        intent = TradeIntent(
            intent_type="UPDATE_TP",
            side="LONG",
            price=3000.0,
            layer_index=1,
            tp_price=3100.0,
            reason="test_trend_runner_sl_failure",
            size=PositionSize(
                eth_qty=0.1, margin_usdt=300.0, notional_usdt=3000.0,
                layer_index=1, layer_multiplier=1.0,
            ),
            fast_cvd=0.0, previous_fast_cvd=0.0,
            buy_ratio=0.5, sell_ratio=0.5,
            boll_upper=3200.0, boll_middle=3100.0, boll_lower=3000.0,
            ts_ms=1000,
            avg_entry_price=3000.0,
            breakeven_price=3000.0,
            tp_mode="MIDDLE",
            tp_plan="THREE_STAGE_RUNNER",
            three_stage_tp1_price=3050.0,
            three_stage_tp2_price=3200.0,
            three_stage_tp1_ratio=0.70,
            three_stage_tp2_ratio=0.20,
            three_stage_runner_ratio=0.10,
            middle_bucket_split_active=True,
            middle_bucket_split_fast_price=3060.0,
            middle_bucket_split_slow_price=3040.0,
            middle_bucket_split_effective_price=3054.0,
            middle_bucket_split_middle_bucket_ratio=0.70,
            middle_bucket_split_fast_ratio_of_bucket=0.70,
            middle_bucket_split_slow_ratio_of_bucket=0.30,
            middle_bucket_split_fast_total_ratio=0.49,
            middle_bucket_split_slow_total_ratio=0.21,
            # Skip middle runner SL so we reach trend runner
            middle_runner_active=False,
            # Trigger trend runner protective SL path
            trend_runner_active=True,
            trend_runner_sl_price=2800.0,
        )

        # Build specs with real split labels
        split_specs = [
            ("tp1_middle_fast", Decimal("3.43"), 3060.0),
            ("tp1_middle_slow", Decimal("1.47"), 3040.0),
            ("tp2_outer", Decimal("2.00"), 3200.0),
        ]
        facade.core_tp._build_take_profit_order_specs = mock.MagicMock(
            return_value=(split_specs, None)
        )

        trader._cancel_existing_take_profit_orders_for_intent = mock.AsyncMock()
        trader._cancel_stale_runner_protective_stops_for_degrade = mock.AsyncMock()
        trader._place_reduce_only_take_profit_orders = mock.AsyncMock(
            return_value=["fast-order", "slow-order", "outer-order"]
        )

        # Mock trend runner SL failure
        trader.place_trend_runner_protective_stop_with_retries = mock.AsyncMock(
            return_value=(False, None, "simulated trend runner SL failure")
        )

        result = await facade.replace_take_profit(intent)

        assert result.ok is False
        assert "trend_runner_protective_sl_failed" in result.message
        assert result.middle_bucket_split_executed is True
        assert result.middle_bucket_split_actual_order_mode == "SPLIT_FAST_SLOW"
        assert result.middle_bucket_split_disabled_reason is None


# ── Integration: order_specs partial consumed + classifier round-trip ───


class TestPartialSplitOrderSpecsClassifierIntegration:
    """Integration test: build_take_profit_order_specs with partial consumed
    flags → classifier recognizes the resulting labels as valid partial split."""

    def _classify(self, *, split_was_active, labels, reason=None):
        from src.execution.tp_sl_core_tp_manager import (
            _classify_middle_bucket_split_actual_order_mode,
        )
        from decimal import Decimal
        specs = [(label, Decimal("1"), 3000.0) for label in labels]
        return _classify_middle_bucket_split_actual_order_mode(
            split_was_active=split_was_active,
            specs=specs,
            split_disabled_reason=reason,
        )

    def test_fast_consumed_order_specs_plus_classifier(self):
        """fast_consumed=True, slow_consumed=False → labels=["tp1_middle_slow", "tp2_outer"]
        → classifier returns PARTIAL_SPLIT_SLOW_PENDING (not FINAL_FULL_SIZE)."""
        from src.execution.order_specs import (
            MiddleBucketSplitOrderInput,
            build_take_profit_order_specs,
        )
        from decimal import Decimal

        fast_total = Decimal("0.70") * Decimal("0.70")  # 0.49
        slow_total = Decimal("0.70") * Decimal("0.30")  # 0.21
        split = MiddleBucketSplitOrderInput(
            active=True,
            fast_price=1638.73,
            slow_price=1634.46,
            effective_price=1637.0,
            middle_bucket_ratio=Decimal("0.70"),
            fast_ratio_of_bucket=Decimal("0.70"),
            slow_ratio_of_bucket=Decimal("0.30"),
            fast_total_ratio=fast_total,
            slow_total_ratio=slow_total,
            fast_consumed=True,
            slow_consumed=False,
        )
        decision = build_take_profit_order_specs(
            position_contracts=Decimal("100"),
            min_contracts=Decimal("1"),
            contract_precision=Decimal("1"),
            tp_plan="THREE_STAGE_RUNNER",
            final_tp_price=1609.44,
            partial_tp_price=None,
            partial_tp_ratio=Decimal("0"),
            partial_tp_consumed=False,
            middle_runner_active=False,
            three_stage_tp1_price=1637.0,
            three_stage_tp2_price=1609.44,
            three_stage_tp1_ratio=Decimal("0.70"),
            three_stage_tp2_ratio=Decimal("0.20"),
            three_stage_tp1_consumed=False,
            three_stage_tp2_consumed=False,
            three_stage_runner_ratio=Decimal("0.10"),
            middle_bucket_split=split,
        )

        labels = [s.label for s in decision.specs]
        assert labels == ["tp1_middle_slow", "tp2_outer"], (
            f"Expected partial (slow + tp2), got: {labels}"
        )

        # Now classify the resulting labels
        executed, reason, mode = self._classify(
            split_was_active=True, labels=labels,
        )
        assert executed is True, (
            f"Classifier must return True for partial split, got: executed={executed}"
        )
        assert reason is None
        assert mode == "PARTIAL_SPLIT_SLOW_PENDING", (
            f"Expected PARTIAL_SPLIT_SLOW_PENDING, got: {mode}"
        )
        assert mode != "FINAL_FULL_SIZE", (
            "Partial split must NOT be classified as FINAL_FULL_SIZE"
        )

    def test_slow_consumed_order_specs_plus_classifier(self):
        """slow_consumed=True, fast_consumed=False → labels=["tp1_middle_fast", "tp2_outer"]
        → classifier returns PARTIAL_SPLIT_FAST_PENDING (not FINAL_FULL_SIZE)."""
        from src.execution.order_specs import (
            MiddleBucketSplitOrderInput,
            build_take_profit_order_specs,
        )
        from decimal import Decimal

        fast_total = Decimal("0.70") * Decimal("0.70")
        slow_total = Decimal("0.70") * Decimal("0.30")
        split = MiddleBucketSplitOrderInput(
            active=True,
            fast_price=1638.73,
            slow_price=1634.46,
            effective_price=1637.0,
            middle_bucket_ratio=Decimal("0.70"),
            fast_ratio_of_bucket=Decimal("0.70"),
            slow_ratio_of_bucket=Decimal("0.30"),
            fast_total_ratio=fast_total,
            slow_total_ratio=slow_total,
            fast_consumed=False,
            slow_consumed=True,
        )
        decision = build_take_profit_order_specs(
            position_contracts=Decimal("100"),
            min_contracts=Decimal("1"),
            contract_precision=Decimal("1"),
            tp_plan="THREE_STAGE_RUNNER",
            final_tp_price=1609.44,
            partial_tp_price=None,
            partial_tp_ratio=Decimal("0"),
            partial_tp_consumed=False,
            middle_runner_active=False,
            three_stage_tp1_price=1637.0,
            three_stage_tp2_price=1609.44,
            three_stage_tp1_ratio=Decimal("0.70"),
            three_stage_tp2_ratio=Decimal("0.20"),
            three_stage_tp1_consumed=False,
            three_stage_tp2_consumed=False,
            three_stage_runner_ratio=Decimal("0.10"),
            middle_bucket_split=split,
        )

        labels = [s.label for s in decision.specs]
        assert labels == ["tp1_middle_fast", "tp2_outer"], (
            f"Expected partial (fast + tp2), got: {labels}"
        )

        executed, reason, mode = self._classify(
            split_was_active=True, labels=labels,
        )
        assert executed is True, (
            f"Classifier must return True for partial split, got: executed={executed}"
        )
        assert reason is None
        assert mode == "PARTIAL_SPLIT_FAST_PENDING", (
            f"Expected PARTIAL_SPLIT_FAST_PENDING, got: {mode}"
        )
        assert mode != "FINAL_FULL_SIZE"
