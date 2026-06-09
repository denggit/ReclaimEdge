"""Tests for startup TP reconcile — pre-TP1 degrade stage refresh and Middle Bucket Split recovery.

Verifies that when TpUpdateCoordinator runs with ``force_reconcile=True``
(startup path):

1. Stale ``three_stage_pre_tp1_degrade_stage`` is refreshed based on position age
   so that a saved SINGLE cap doesn't permanently block recovery to
   THREE_STAGE_RUNNER or MIDDLE_RUNNER.

2. Middle Bucket Split is re-applied for THREE_STAGE_RUNNER and MIDDLE_RUNNER
   plans when ``force_reconcile=True``.

3. Normal (non-startup) TP updates do NOT clear stale degrade stage.

4. Non-Three-Stage strategies are not erroneously capped by the pre-TP1 degrade
   cap during startup reconcile.

Covers tests 1–6 from the fix specification.
"""

from __future__ import annotations

import unittest
from decimal import Decimal

from src.indicators.cvd_tracker import CvdSnapshot
from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
)
from src.strategies.tp_update_coordinator import TpUpdateCoordinator
from src.execution.order_specs import (
    MiddleBucketSplitOrderInput,
    build_take_profit_order_specs,
)

# ── shared constants ───────────────────────────────────────────────────

HOUR_MS = 3_600_000

# Timestamp anchored in ~2026-06 to guarantee first_entry_ts_ms stays
# strictly positive after all age subtractions (up to 24 h).
BASE_TS_MS = 1_781_000_000_000


# ── reusable helpers ────────────────────────────────────────────────────


def _boll(
    middle: float = 2000.0,
    upper: float = 2100.0,
    lower: float = 1900.0,
    tp_middle: float | None = None,
    tp_upper: float | None = None,
    tp_lower: float | None = None,
    candle_ts_ms: int = BASE_TS_MS,
) -> BollSnapshot:
    return BollSnapshot(
        inst_id="ETH-USDT-SWAP",
        candle_ts_ms=candle_ts_ms,
        close=middle,
        middle=middle,
        upper=upper,
        lower=lower,
        upper_distance_pct=0.05,
        lower_distance_pct=0.05,
        alert_switch_on=True,
        live_mode=True,
        tp_middle=tp_middle,
        tp_upper=tp_upper,
        tp_lower=tp_lower,
    )


def _cvd(buy_ratio: float = 0.6, sell_ratio: float = 0.4) -> CvdSnapshot:
    return CvdSnapshot(
        ts_ms=BASE_TS_MS,
        price=2000.0,
        side="buy",
        size=1.0,
        signed_delta=0.1,
        total_cvd=0.5,
        fast_cvd=0.1,
        previous_fast_cvd=0.09,
        buy_volume=60.0,
        sell_volume=40.0,
        buy_ratio=buy_ratio,
        sell_ratio=sell_ratio,
        cross_positive=True,
        cross_negative=False,
        cvd_increasing=True,
        cvd_decreasing=False,
        no_new_low=True,
        no_new_high=False,
        window_low=1990.0,
        window_high=2010.0,
        burst_net_move_pct=0.0,
        burst_range_pct=0.01,
        baseline_range_pct=0.001,
        burst_move_ratio=10.0,
        burst_volume=10.0,
        baseline_volume=1.0,
        burst_volume_ratio=10.0,
        up_burst=False,
        down_burst=False,
    )


def _sizer() -> SimplePositionSizer:
    cfg = SimplePositionSizerConfig()
    return SimplePositionSizer(cfg)


def _coordinator(strategy: BollCvdReclaimStrategy) -> TpUpdateCoordinator:
    return TpUpdateCoordinator(strategy)


def _setup_strategy_for_tp_update(
    *,
    strategy: BollCvdReclaimStrategy,
    first_entry_ts_ms: int,
    avg_entry_notional: float,
    eth_qty: float,
    side: str = "LONG",
    layers: int = 2,
    old_degrade_stage: str | None = "SINGLE",
    old_degraded_ts_ms: int = 0,
    old_tp_plan: str = "SINGLE",
    three_stage_enabled_for_position: bool = False,
    startup_force_tp_reconcile: bool = True,
) -> None:
    """Configure strategy state to simulate an existing position before TP update."""
    strategy.state.side = side  # type: ignore[assignment]
    strategy.state.layers = layers
    strategy.state.first_entry_ts_ms = first_entry_ts_ms
    strategy.state.last_entry_price = 2000.0
    strategy.state.last_order_ts_ms = first_entry_ts_ms
    strategy.state.total_entry_qty = eth_qty
    strategy.state.total_entry_notional = avg_entry_notional  # type: ignore[assignment]
    strategy.state.avg_entry_price = avg_entry_notional / eth_qty  # type: ignore[assignment]
    strategy.state.position_cost_entry_notional = avg_entry_notional  # type: ignore[assignment]
    strategy.state.position_cost_exit_notional = 0.0
    strategy.state.position_cost_remaining_qty = eth_qty  # type: ignore[assignment]
    strategy.state.tp_plan = old_tp_plan
    strategy.state.three_stage_pre_tp1_degrade_stage = old_degrade_stage
    strategy.state.three_stage_pre_tp1_degraded_ts_ms = old_degraded_ts_ms
    strategy.state.three_stage_runner_enabled_for_position = three_stage_enabled_for_position
    strategy.state.startup_force_tp_reconcile = startup_force_tp_reconcile
    # trigger breakeven computation
    strategy._refresh_net_remaining_breakeven_price()


# ── Test 1: startup reconcile age <3h + old SINGLE → THREE_STAGE_RUNNER + split ──


class TestStartupReconcileAgeUnder3hRecoverThreeStageWithSplit(unittest.TestCase):
    """age < 3h + old SINGLE + sufficient middle profit → THREE_STAGE_RUNNER + split."""

    def test_age_under_3h_recovers_three_stage_runner_with_split(self) -> None:
        ts_ms = BASE_TS_MS
        age_2h = 2 * 3600
        first_entry_ts_ms = ts_ms - age_2h * 1000

        config = BollCvdReclaimStrategyConfig(
            three_stage_runner_enabled=True,
            middle_bucket_split_enabled=True,
            tp_boll_enabled=True,
            tp_min_net_profit_pct=0.004,
            three_stage_pre_tp1_degrade_enabled=True,
            three_stage_pre_tp1_middle_runner_after_seconds=10800,
            three_stage_pre_tp1_single_after_seconds=21600,
            middle_runner_enabled=False,
        )
        strategy = BollCvdReclaimStrategy(config, _sizer())

        # existing position: 2 ETH @ 1980 avg, age 2h
        _setup_strategy_for_tp_update(
            strategy=strategy,
            first_entry_ts_ms=first_entry_ts_ms,
            avg_entry_notional=3960.0,
            eth_qty=2.0,
            side="LONG",
            layers=2,
            old_degrade_stage="SINGLE",
            old_degraded_ts_ms=first_entry_ts_ms + 60_000,
            old_tp_plan="SINGLE",
            startup_force_tp_reconcile=True,
        )

        # BOLL with middle prices well above breakeven
        boll = _boll(
            middle=2000.0,
            upper=2100.0,
            lower=1900.0,
            tp_middle=2008.0,  # BOLL15 middle sufficient for profit
            tp_upper=2110.0,
            tp_lower=1890.0,
            candle_ts_ms=ts_ms,
        )
        cvd = _cvd()

        coord = _coordinator(strategy)
        intent = coord.maybe_update_tp(
            price=2000.0,
            ts_ms=ts_ms,
            boll=boll,
            cvd=cvd,
        )

        # Intent assertions
        self.assertIsNotNone(intent, "startup force reconcile should emit an intent")
        assert intent is not None
        self.assertEqual(intent.tp_plan, "THREE_STAGE_RUNNER")

        # State: stale SINGLE degrade_stage must be cleared
        self.assertIsNone(strategy.state.three_stage_pre_tp1_degrade_stage)
        self.assertEqual(strategy.state.three_stage_pre_tp1_degraded_ts_ms, 0)

        # Middle Bucket Split must be active
        self.assertTrue(strategy.state.middle_bucket_split_active)
        self.assertTrue(intent.middle_bucket_split_active)

        # Split prices must be set
        self.assertIsNotNone(strategy.state.middle_bucket_split_fast_price)
        self.assertIsNotNone(strategy.state.middle_bucket_split_slow_price)

        # Three-Stage TP fields must be populated
        self.assertIsNotNone(strategy.state.three_stage_tp1_price)
        self.assertIsNotNone(strategy.state.three_stage_tp2_price)


# ── Test 2: startup reconcile intent can generate split order labels ────


class TestStartupReconcileSplitOrderLabels(unittest.TestCase):
    """THREE_STAGE_RUNNER + middle_bucket_split → labels = [tp1_middle_fast, tp1_middle_slow, tp2_outer]."""

    def test_split_order_labels_from_startup_reconcile(self) -> None:
        ts_ms = BASE_TS_MS
        age_2h = 2 * 3600
        first_entry_ts_ms = ts_ms - age_2h * 1000

        config = BollCvdReclaimStrategyConfig(
            three_stage_runner_enabled=True,
            middle_bucket_split_enabled=True,
            tp_boll_enabled=True,
            tp_min_net_profit_pct=0.004,
            three_stage_pre_tp1_degrade_enabled=True,
            three_stage_pre_tp1_middle_runner_after_seconds=10800,
            three_stage_pre_tp1_single_after_seconds=21600,
            middle_runner_enabled=False,
        )
        strategy = BollCvdReclaimStrategy(config, _sizer())

        _setup_strategy_for_tp_update(
            strategy=strategy,
            first_entry_ts_ms=first_entry_ts_ms,
            avg_entry_notional=3960.0,
            eth_qty=2.0,
            side="LONG",
            layers=2,
            old_degrade_stage="SINGLE",
            old_degraded_ts_ms=first_entry_ts_ms + 60_000,
            old_tp_plan="SINGLE",
            startup_force_tp_reconcile=True,
        )

        boll = _boll(
            middle=2000.0,
            upper=2100.0,
            lower=1900.0,
            tp_middle=2008.0,
            tp_upper=2110.0,
            tp_lower=1890.0,
            candle_ts_ms=ts_ms,
        )
        cvd = _cvd()

        coord = _coordinator(strategy)
        intent = coord.maybe_update_tp(
            price=2000.0,
            ts_ms=ts_ms,
            boll=boll,
            cvd=cvd,
        )

        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.tp_plan, "THREE_STAGE_RUNNER")
        self.assertTrue(intent.middle_bucket_split_active)

        # Build order specs using the intent/state
        split_input = MiddleBucketSplitOrderInput(
            active=True,
            fast_price=float(strategy.state.middle_bucket_split_fast_price),
            slow_price=float(strategy.state.middle_bucket_split_slow_price),
            effective_price=float(strategy.state.middle_bucket_split_effective_price),
            middle_bucket_ratio=Decimal(
                str(strategy.state.middle_bucket_split_middle_bucket_ratio)
            ),
            fast_ratio_of_bucket=Decimal(
                str(strategy.state.middle_bucket_split_fast_ratio_of_bucket)
            ),
            slow_ratio_of_bucket=Decimal(
                str(strategy.state.middle_bucket_split_slow_ratio_of_bucket)
            ),
            fast_total_ratio=Decimal(
                str(strategy.state.middle_bucket_split_fast_total_ratio)
            ),
            slow_total_ratio=Decimal(
                str(strategy.state.middle_bucket_split_slow_total_ratio)
            ),
        )

        final_tp_price = strategy.state.tp_price or 2110.0
        partial_tp_price = strategy.state.partial_tp_price or 2005.0
        partial_tp_ratio = Decimal(str(strategy.state.partial_tp_ratio))

        decision = build_take_profit_order_specs(
            position_contracts=Decimal("6.79"),
            min_contracts=Decimal("0.01"),
            contract_precision=Decimal("0.01"),
            tp_plan="THREE_STAGE_RUNNER",
            final_tp_price=final_tp_price,
            partial_tp_price=partial_tp_price,
            partial_tp_ratio=partial_tp_ratio,
            partial_tp_consumed=False,
            middle_runner_active=False,
            three_stage_tp1_price=strategy.state.three_stage_tp1_price,
            three_stage_tp2_price=strategy.state.three_stage_tp2_price,
            three_stage_tp1_ratio=Decimal(str(strategy.state.three_stage_tp1_ratio)),
            three_stage_tp2_ratio=Decimal(str(strategy.state.three_stage_tp2_ratio)),
            three_stage_tp1_consumed=False,
            three_stage_tp2_consumed=False,
            three_stage_runner_ratio=Decimal(str(strategy.state.three_stage_runner_ratio)),
            middle_bucket_split=split_input,
        )
        labels = [s.label for s in decision.specs]
        self.assertEqual(labels, ["tp1_middle_fast", "tp1_middle_slow", "tp2_outer"])


# ── Test 3: startup reconcile age 4h → MIDDLE_RUNNER + split ──────────────


class TestStartupReconcileAge4hRecoverMiddleRunnerWithSplit(unittest.TestCase):
    """age 4h + old SINGLE + sufficient middle profit → MIDDLE_RUNNER + split."""

    def test_age_4h_recovers_middle_runner_with_split(self) -> None:
        ts_ms = BASE_TS_MS
        age_4h = 4 * 3600
        first_entry_ts_ms = ts_ms - age_4h * 1000

        config = BollCvdReclaimStrategyConfig(
            three_stage_runner_enabled=True,
            middle_bucket_split_enabled=True,
            middle_runner_enabled=True,
            tp_boll_enabled=True,
            tp_min_net_profit_pct=0.004,
            three_stage_pre_tp1_degrade_enabled=True,
            three_stage_pre_tp1_middle_runner_after_seconds=10800,
            three_stage_pre_tp1_single_after_seconds=21600,
        )
        strategy = BollCvdReclaimStrategy(config, _sizer())

        _setup_strategy_for_tp_update(
            strategy=strategy,
            first_entry_ts_ms=first_entry_ts_ms,
            avg_entry_notional=3960.0,
            eth_qty=2.0,
            side="LONG",
            layers=2,
            old_degrade_stage="SINGLE",
            old_degraded_ts_ms=first_entry_ts_ms + 60_000,
            old_tp_plan="SINGLE",
            startup_force_tp_reconcile=True,
        )

        boll = _boll(
            middle=2000.0,
            upper=2100.0,
            lower=1900.0,
            tp_middle=2008.0,
            tp_upper=2110.0,
            tp_lower=1890.0,
            candle_ts_ms=ts_ms,
        )
        cvd = _cvd()

        coord = _coordinator(strategy)
        intent = coord.maybe_update_tp(
            price=2000.0,
            ts_ms=ts_ms,
            boll=boll,
            cvd=cvd,
        )

        self.assertIsNotNone(intent)
        assert intent is not None

        # Degrade stage must be MIDDLE_RUNNER
        self.assertEqual(strategy.state.three_stage_pre_tp1_degrade_stage, "MIDDLE_RUNNER")
        self.assertEqual(strategy.state.three_stage_pre_tp1_degraded_ts_ms, ts_ms)

        # Plan must be MIDDLE_RUNNER (not THREE_STAGE_RUNNER)
        self.assertEqual(intent.tp_plan, "MIDDLE_RUNNER")

        # Middle Bucket Split should be active
        self.assertTrue(strategy.state.middle_bucket_split_active)

        # Generate order labels to confirm split
        if strategy.state.middle_bucket_split_active:
            split_input = MiddleBucketSplitOrderInput(
                active=True,
                fast_price=float(strategy.state.middle_bucket_split_fast_price),
                slow_price=float(strategy.state.middle_bucket_split_slow_price),
                effective_price=float(strategy.state.middle_bucket_split_effective_price),
                middle_bucket_ratio=Decimal(
                    str(strategy.state.middle_bucket_split_middle_bucket_ratio)
                ),
                fast_ratio_of_bucket=Decimal(
                    str(strategy.state.middle_bucket_split_fast_ratio_of_bucket)
                ),
                slow_ratio_of_bucket=Decimal(
                    str(strategy.state.middle_bucket_split_slow_ratio_of_bucket)
                ),
                fast_total_ratio=Decimal(
                    str(strategy.state.middle_bucket_split_fast_total_ratio)
                ),
                slow_total_ratio=Decimal(
                    str(strategy.state.middle_bucket_split_slow_total_ratio)
                ),
            )

            final_tp_price = strategy.state.tp_price or 2100.0
            partial_tp_price = strategy.state.partial_tp_price or 2005.0
            partial_tp_ratio = Decimal(str(strategy.state.partial_tp_ratio))

            decision = build_take_profit_order_specs(
                position_contracts=Decimal("5.0"),
                min_contracts=Decimal("0.01"),
                contract_precision=Decimal("0.01"),
                tp_plan="MIDDLE_RUNNER",
                final_tp_price=final_tp_price,
                partial_tp_price=partial_tp_price,
                partial_tp_ratio=partial_tp_ratio,
                partial_tp_consumed=False,
                middle_runner_active=False,
                three_stage_tp1_price=None,
                three_stage_tp2_price=None,
                three_stage_tp1_ratio=Decimal("0"),
                three_stage_tp2_ratio=Decimal("0"),
                three_stage_tp1_consumed=False,
                three_stage_tp2_consumed=False,
                three_stage_runner_ratio=Decimal("0"),
                middle_bucket_split=split_input,
            )
            labels = [s.label for s in decision.specs]
            self.assertEqual(labels, ["middle_fast", "middle_slow", "runner"])


# ── Test 4: startup reconcile age 7h → SINGLE (no recovery) ───────────────


class TestStartupReconcileAge7hKeepsSingle(unittest.TestCase):
    """age ≥ 6h → SINGLE — no recovery to MIDDLE_RUNNER or THREE_STAGE_RUNNER."""

    def test_age_7h_keeps_single(self) -> None:
        ts_ms = BASE_TS_MS
        age_7h = 7 * 3600
        first_entry_ts_ms = ts_ms - age_7h * 1000

        config = BollCvdReclaimStrategyConfig(
            three_stage_runner_enabled=True,
            middle_bucket_split_enabled=True,
            middle_runner_enabled=True,
            tp_boll_enabled=True,
            tp_min_net_profit_pct=0.004,
            three_stage_pre_tp1_degrade_enabled=True,
            three_stage_pre_tp1_middle_runner_after_seconds=10800,
            three_stage_pre_tp1_single_after_seconds=21600,
        )
        strategy = BollCvdReclaimStrategy(config, _sizer())

        _setup_strategy_for_tp_update(
            strategy=strategy,
            first_entry_ts_ms=first_entry_ts_ms,
            avg_entry_notional=3960.0,
            eth_qty=2.0,
            side="LONG",
            layers=2,
            old_degrade_stage="SINGLE",
            old_degraded_ts_ms=first_entry_ts_ms + 60_000,
            old_tp_plan="SINGLE",
            startup_force_tp_reconcile=True,
        )

        boll = _boll(
            middle=2000.0,
            upper=2100.0,
            lower=1900.0,
            tp_middle=2008.0,
            tp_upper=2110.0,
            tp_lower=1890.0,
            candle_ts_ms=ts_ms,
        )
        cvd = _cvd()

        coord = _coordinator(strategy)
        intent = coord.maybe_update_tp(
            price=2000.0,
            ts_ms=ts_ms,
            boll=boll,
            cvd=cvd,
        )

        # Degrade stage must be SINGLE
        self.assertEqual(strategy.state.three_stage_pre_tp1_degrade_stage, "SINGLE")

        # Even with profit sufficient, plan must be SINGLE
        if intent is not None:
            self.assertEqual(intent.tp_plan, "SINGLE")
            self.assertNotEqual(intent.tp_plan, "THREE_STAGE_RUNNER")
            self.assertNotEqual(intent.tp_plan, "MIDDLE_RUNNER")

            # Middle Bucket Split must be off
            self.assertFalse(intent.middle_bucket_split_active)
            self.assertFalse(strategy.state.middle_bucket_split_active)
        else:
            # If intent is None (plan unchanged in state), state should still be SINGLE
            self.assertEqual(strategy.state.tp_plan, "SINGLE")


# ── Test 5: non-startup normal TP update does NOT clear stale SINGLE ───────


class TestNormalTpUpdateDoesNotClearStaleSingle(unittest.TestCase):
    """force_reconcile=False must NOT trigger stale degrade stage refresh."""

    def test_normal_update_keeps_stale_single(self) -> None:
        ts_ms = BASE_TS_MS
        age_2h = 2 * 3600
        first_entry_ts_ms = ts_ms - age_2h * 1000

        config = BollCvdReclaimStrategyConfig(
            three_stage_runner_enabled=True,
            middle_bucket_split_enabled=True,
            tp_boll_enabled=True,
            tp_min_net_profit_pct=0.004,
            three_stage_pre_tp1_degrade_enabled=True,
            three_stage_pre_tp1_middle_runner_after_seconds=10800,
            three_stage_pre_tp1_single_after_seconds=21600,
            middle_runner_enabled=False,
        )
        strategy = BollCvdReclaimStrategy(config, _sizer())

        _setup_strategy_for_tp_update(
            strategy=strategy,
            first_entry_ts_ms=first_entry_ts_ms,
            avg_entry_notional=3960.0,
            eth_qty=2.0,
            side="LONG",
            layers=2,
            old_degrade_stage="SINGLE",
            old_degraded_ts_ms=first_entry_ts_ms + 60_000,
            old_tp_plan="SINGLE",
            startup_force_tp_reconcile=False,  # <-- normal update, NOT startup
        )

        # Set a distinct candle_ts to avoid the "same candle" early return
        strategy.state.last_tp_update_candle_ts_ms = 0

        boll = _boll(
            middle=2000.0,
            upper=2100.0,
            lower=1900.0,
            tp_middle=2008.0,
            tp_upper=2110.0,
            tp_lower=1890.0,
            candle_ts_ms=ts_ms,
        )
        cvd = _cvd()

        coord = _coordinator(strategy)
        intent = coord.maybe_update_tp(
            price=2000.0,
            ts_ms=ts_ms,
            boll=boll,
            cvd=cvd,
        )

        # Degrade stage must STILL be SINGLE — NOT cleared
        self.assertEqual(
            strategy.state.three_stage_pre_tp1_degrade_stage,
            "SINGLE",
            "Normal tick update must NOT clear stale degrade stage",
        )

        # The plan selected by the normal tick update may still be SINGLE
        # because _three_stage_pre_tp1_degrade_target returns None when
        # the current stage is already SINGLE (sticky guard).
        # This verifies the non-startup path does NOT trigger the new logic.
        if intent is not None:
            self.assertEqual(
                intent.tp_plan, "SINGLE",
                "Normal tick update should follow old sticky SINGLE behavior",
            )


# ── Test 6: non-Three-Stage config not capped by startup cap ───────────────


class TestNonThreeStageNotCappedByStartupReconcile(unittest.TestCase):
    """Non-Three-Stage config + force_reconcile=True must NOT cap to MIDDLE_RUNNER or SINGLE."""

    def test_non_three_stage_age_4h_no_cap(self) -> None:
        ts_ms = BASE_TS_MS
        age_4h = 4 * 3600
        first_entry_ts_ms = ts_ms - age_4h * 1000

        config = BollCvdReclaimStrategyConfig(
            three_stage_runner_enabled=False,
            middle_runner_enabled=False,
            middle_bucket_split_enabled=False,
            tp_boll_enabled=True,
            tp_min_net_profit_pct=0.004,
            three_stage_pre_tp1_degrade_enabled=True,
            three_stage_pre_tp1_middle_runner_after_seconds=10800,
            three_stage_pre_tp1_single_after_seconds=21600,
        )
        strategy = BollCvdReclaimStrategy(config, _sizer())

        _setup_strategy_for_tp_update(
            strategy=strategy,
            first_entry_ts_ms=first_entry_ts_ms,
            avg_entry_notional=3960.0,
            eth_qty=2.0,
            side="LONG",
            layers=2,
            old_degrade_stage="SINGLE",
            old_degraded_ts_ms=first_entry_ts_ms + 60_000,
            old_tp_plan="SINGLE",
            startup_force_tp_reconcile=True,
        )

        boll = _boll(
            middle=2000.0,
            upper=2100.0,
            lower=1900.0,
            tp_middle=2008.0,
            tp_upper=2110.0,
            tp_lower=1890.0,
            candle_ts_ms=ts_ms,
        )
        cvd = _cvd()

        coord = _coordinator(strategy)
        intent = coord.maybe_update_tp(
            price=2000.0,
            ts_ms=ts_ms,
            boll=boll,
            cvd=cvd,
        )

        # Degrade stage must be None (not Three-Stage lifecycle)
        self.assertIsNone(strategy.state.three_stage_pre_tp1_degrade_stage)
        self.assertEqual(strategy.state.three_stage_pre_tp1_degraded_ts_ms, 0)

        # Plan must NOT be MIDDLE_RUNNER or THREE_STAGE_RUNNER
        if intent is not None:
            self.assertNotEqual(intent.tp_plan, "MIDDLE_RUNNER")
            self.assertNotEqual(intent.tp_plan, "THREE_STAGE_RUNNER")

    def test_non_three_stage_age_7h_no_cap(self) -> None:
        ts_ms = BASE_TS_MS
        age_7h = 7 * 3600
        first_entry_ts_ms = ts_ms - age_7h * 1000

        config = BollCvdReclaimStrategyConfig(
            three_stage_runner_enabled=False,
            middle_runner_enabled=False,
            middle_bucket_split_enabled=False,
            tp_boll_enabled=True,
            tp_min_net_profit_pct=0.004,
            three_stage_pre_tp1_degrade_enabled=True,
            three_stage_pre_tp1_middle_runner_after_seconds=10800,
            three_stage_pre_tp1_single_after_seconds=21600,
        )
        strategy = BollCvdReclaimStrategy(config, _sizer())

        _setup_strategy_for_tp_update(
            strategy=strategy,
            first_entry_ts_ms=first_entry_ts_ms,
            avg_entry_notional=3960.0,
            eth_qty=2.0,
            side="LONG",
            layers=2,
            old_degrade_stage="MIDDLE_RUNNER",
            old_degraded_ts_ms=first_entry_ts_ms + 60_000,
            old_tp_plan="SINGLE",
            startup_force_tp_reconcile=True,
        )

        boll = _boll(
            middle=2000.0,
            upper=2100.0,
            lower=1900.0,
            tp_middle=2008.0,
            tp_upper=2110.0,
            tp_lower=1890.0,
            candle_ts_ms=ts_ms,
        )
        cvd = _cvd()

        coord = _coordinator(strategy)
        intent = coord.maybe_update_tp(
            price=2000.0,
            ts_ms=ts_ms,
            boll=boll,
            cvd=cvd,
        )

        # Degrade stage must be None (not Three-Stage lifecycle)
        self.assertIsNone(strategy.state.three_stage_pre_tp1_degrade_stage)
        self.assertEqual(strategy.state.three_stage_pre_tp1_degraded_ts_ms, 0)

        # Plan must NOT be MIDDLE_RUNNER or THREE_STAGE_RUNNER
        if intent is not None:
            self.assertNotEqual(intent.tp_plan, "MIDDLE_RUNNER")
            self.assertNotEqual(intent.tp_plan, "THREE_STAGE_RUNNER")


if __name__ == "__main__":
    unittest.main()
