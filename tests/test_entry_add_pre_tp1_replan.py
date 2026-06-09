"""Tests for pre-TP1 degrade stage refresh on position add/replan.

Verifies that when a stale ``three_stage_pre_tp1_degrade_stage`` (e.g.
"SINGLE" from a previous tick update) blocks TP plan recovery after a
position-size increase, the EntryAddFlowCoordinator refreshes the cap
based on current position age before re-selecting the TP plan.

Covers tests A–G from the fix specification.
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
from src.strategies.entry_add_flow_coordinator import EntryAddFlowCoordinator
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


def _coordinator(strategy: BollCvdReclaimStrategy) -> EntryAddFlowCoordinator:
    return EntryAddFlowCoordinator(strategy)


def _setup_strategy_for_add(
    *,
    strategy: BollCvdReclaimStrategy,
    first_entry_ts_ms: int,
    avg_entry_notional: float,
    eth_qty: float,
    side: str = "LONG",
    old_degrade_stage: str | None = "SINGLE",
    old_degraded_ts_ms: int = 0,
) -> None:
    """Configure strategy state to simulate an existing position before an add."""
    strategy.state.side = side  # type: ignore[assignment]
    strategy.state.layers = 1
    strategy.state.first_entry_ts_ms = first_entry_ts_ms
    strategy.state.last_entry_price = 2000.0
    strategy.state.last_order_ts_ms = first_entry_ts_ms
    strategy.state.total_entry_qty = eth_qty
    strategy.state.total_entry_notional = avg_entry_notional  # type: ignore[assignment]
    strategy.state.avg_entry_price = avg_entry_notional / eth_qty  # type: ignore[assignment]
    strategy.state.position_cost_entry_notional = avg_entry_notional  # type: ignore[assignment]
    strategy.state.position_cost_exit_notional = 0.0
    strategy.state.position_cost_remaining_qty = eth_qty  # type: ignore[assignment]
    strategy.state.tp_plan = "SINGLE"
    strategy.state.three_stage_pre_tp1_degrade_stage = old_degrade_stage
    strategy.state.three_stage_pre_tp1_degraded_ts_ms = old_degraded_ts_ms
    # trigger breakeven computation
    strategy._refresh_net_remaining_breakeven_price()


# ── Test A: age < 3h, old SINGLE, profit sufficient → THREE_STAGE_RUNNER ──


class TestAddRefreshPreTp1RecoverThreeStage(unittest.TestCase):
    """age < 3h + old SINGLE + sufficient middle profit → THREE_STAGE_RUNNER."""

    def test_age_under_3h_recovers_three_stage_runner(self) -> None:
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

        # existing position: 1 ETH @ 1980 avg
        _setup_strategy_for_add(
            strategy=strategy,
            first_entry_ts_ms=first_entry_ts_ms,
            avg_entry_notional=1980.0,
            eth_qty=1.0,
            side="LONG",
            old_degrade_stage="SINGLE",
            old_degraded_ts_ms=first_entry_ts_ms + 60_000,
        )

        # BOLL with middle prices well above breakeven
        boll = _boll(
            middle=2000.0,
            upper=2100.0,
            lower=1900.0,
            tp_middle=2008.0,
            tp_upper=2110.0,
            tp_lower=1890.0,
        )
        cvd = _cvd()

        coord = _coordinator(strategy)
        intent = coord.open_position(
            side="LONG",
            intent_type="ADD_LONG",
            price=1970.0,
            ts_ms=ts_ms,
            boll=boll,
            cvd=cvd,
            reason="test_age_under_3h_recovers_three_stage_runner",
        )

        # Intent assertions
        self.assertEqual(intent.tp_plan, "THREE_STAGE_RUNNER")

        # State assertions
        self.assertEqual(strategy.state.tp_plan, "THREE_STAGE_RUNNER")
        self.assertIsNone(strategy.state.three_stage_pre_tp1_degrade_stage)
        self.assertEqual(strategy.state.three_stage_pre_tp1_degraded_ts_ms, 0)

        # Three-Stage TP fields must be populated
        self.assertIsNotNone(intent.three_stage_tp1_price)
        self.assertIsNotNone(intent.three_stage_tp2_price)

        # Middle Bucket Split must be active
        self.assertTrue(strategy.state.middle_bucket_split_active)
        self.assertTrue(intent.middle_bucket_split_active)


# ── Test B: age < 3h, old SINGLE, profit still insufficient → SINGLE ───


class TestAddRefreshPreTp1KeepSingleWhenProfitInsufficient(unittest.TestCase):
    """age < 3h + old SINGLE + insufficient middle profit → SINGLE."""

    def test_age_under_3h_keeps_single_when_middle_profit_insufficient(self) -> None:
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

        # avg_entry at 1998 → breakeven ≈ 2000 → required middle ≈ 2008
        # BOLL20 middle=2005 < 2008 → profit check fails → outer mode → SINGLE
        _setup_strategy_for_add(
            strategy=strategy,
            first_entry_ts_ms=first_entry_ts_ms,
            avg_entry_notional=1998.0,
            eth_qty=1.0,
            side="LONG",
            old_degrade_stage="SINGLE",
            old_degraded_ts_ms=first_entry_ts_ms + 60_000,
        )

        # BOLL20 middle too low → profit fails. No TP_BOLL15 available.
        boll = _boll(
            middle=2005.0,
            upper=2100.0,
            lower=1900.0,
            tp_middle=None,
            tp_upper=None,
            tp_lower=None,
        )
        cvd = _cvd()

        coord = _coordinator(strategy)
        intent = coord.open_position(
            side="LONG",
            intent_type="ADD_LONG",
            price=1995.0,  # small improvement, avg stays near 1997.7
            ts_ms=ts_ms,
            boll=boll,
            cvd=cvd,
            reason="test_age_under_3h_keeps_single_when_middle_profit_insufficient",
        )

        # Must be SINGLE because middle profit insufficient
        self.assertEqual(intent.tp_plan, "SINGLE")
        self.assertEqual(strategy.state.tp_plan, "SINGLE")

        # Middle Bucket Split must be off
        self.assertFalse(strategy.state.middle_bucket_split_active)
        self.assertFalse(intent.middle_bucket_split_active)

        # Three-Stage fields must NOT be populated
        self.assertIsNone(intent.three_stage_tp1_price)
        self.assertIsNone(intent.three_stage_tp2_price)


# ── Test C: 3h ≤ age < 6h, old SINGLE, profit sufficient → MIDDLE_RUNNER ──


class TestAddRefreshPreTp1CapMiddleRunner(unittest.TestCase):
    """3h ≤ age < 6h + old SINGLE + sufficient middle profit → MIDDLE_RUNNER."""

    def test_age_4h_caps_at_middle_runner(self) -> None:
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

        _setup_strategy_for_add(
            strategy=strategy,
            first_entry_ts_ms=first_entry_ts_ms,
            avg_entry_notional=1980.0,
            eth_qty=1.0,
            side="LONG",
            old_degrade_stage="SINGLE",
            old_degraded_ts_ms=first_entry_ts_ms + 60_000,
        )

        boll = _boll(
            middle=2000.0,
            upper=2100.0,
            lower=1900.0,
            tp_middle=2008.0,
            tp_upper=2110.0,
            tp_lower=1890.0,
        )
        cvd = _cvd()

        coord = _coordinator(strategy)
        intent = coord.open_position(
            side="LONG",
            intent_type="ADD_LONG",
            price=1970.0,
            ts_ms=ts_ms,
            boll=boll,
            cvd=cvd,
            reason="test_age_4h_caps_at_middle_runner",
        )

        # Degrade stage must be MIDDLE_RUNNER
        self.assertEqual(strategy.state.three_stage_pre_tp1_degrade_stage, "MIDDLE_RUNNER")
        self.assertEqual(strategy.state.three_stage_pre_tp1_degraded_ts_ms, ts_ms)

        # Plan must be MIDDLE_RUNNER (not THREE_STAGE_RUNNER)
        self.assertEqual(intent.tp_plan, "MIDDLE_RUNNER")
        self.assertNotEqual(intent.tp_plan, "THREE_STAGE_RUNNER")

        # TP2 outer must NOT be populated
        self.assertIsNone(intent.three_stage_tp2_price)

        # Middle Bucket Split should be active (prices satisfy profit)
        self.assertTrue(strategy.state.middle_bucket_split_active)
        self.assertTrue(intent.middle_bucket_split_active)


# ── Test D: age ≥ 6h, profit sufficient → SINGLE (no recovery) ────────────


class TestAddRefreshPreTp1CapSingle(unittest.TestCase):
    """age ≥ 6h + profit sufficient → SINGLE (no recovery allowed)."""

    def test_age_over_6h_caps_at_single(self) -> None:
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

        _setup_strategy_for_add(
            strategy=strategy,
            first_entry_ts_ms=first_entry_ts_ms,
            avg_entry_notional=1980.0,
            eth_qty=1.0,
            side="LONG",
            old_degrade_stage="SINGLE",
            old_degraded_ts_ms=first_entry_ts_ms + 60_000,
        )

        boll = _boll(
            middle=2000.0,
            upper=2100.0,
            lower=1900.0,
            tp_middle=2008.0,
            tp_upper=2110.0,
            tp_lower=1890.0,
        )
        cvd = _cvd()

        coord = _coordinator(strategy)
        intent = coord.open_position(
            side="LONG",
            intent_type="ADD_LONG",
            price=1970.0,
            ts_ms=ts_ms,
            boll=boll,
            cvd=cvd,
            reason="test_age_over_6h_caps_at_single",
        )

        # Degrade stage must be SINGLE
        self.assertEqual(strategy.state.three_stage_pre_tp1_degrade_stage, "SINGLE")

        # Plan must be SINGLE
        self.assertEqual(intent.tp_plan, "SINGLE")
        self.assertNotEqual(intent.tp_plan, "THREE_STAGE_RUNNER")
        self.assertNotEqual(intent.tp_plan, "MIDDLE_RUNNER")

        # Middle Bucket Split must be off
        self.assertFalse(strategy.state.middle_bucket_split_active)
        self.assertFalse(intent.middle_bucket_split_active)

        # Three-Stage fields must NOT be populated
        self.assertIsNone(intent.three_stage_tp1_price)
        self.assertIsNone(intent.three_stage_tp2_price)

    def test_age_over_6h_old_stage_none(self) -> None:
        """Even when old degrade_stage was None, age ≥ 6h forces SINGLE."""
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

        _setup_strategy_for_add(
            strategy=strategy,
            first_entry_ts_ms=first_entry_ts_ms,
            avg_entry_notional=1980.0,
            eth_qty=1.0,
            side="LONG",
            old_degrade_stage=None,
            old_degraded_ts_ms=0,
        )

        boll = _boll(
            middle=2000.0,
            upper=2100.0,
            lower=1900.0,
            tp_middle=2008.0,
            tp_upper=2110.0,
            tp_lower=1890.0,
        )
        cvd = _cvd()

        coord = _coordinator(strategy)
        intent = coord.open_position(
            side="LONG",
            intent_type="ADD_LONG",
            price=1970.0,
            ts_ms=ts_ms,
            boll=boll,
            cvd=cvd,
            reason="test_age_over_6h_old_stage_none",
        )

        self.assertEqual(strategy.state.three_stage_pre_tp1_degrade_stage, "SINGLE")
        self.assertEqual(intent.tp_plan, "SINGLE")

    def test_age_over_6h_old_stage_middle_runner(self) -> None:
        """When old degrade_stage was MIDDLE_RUNNER, age ≥ 6h forces SINGLE."""
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

        _setup_strategy_for_add(
            strategy=strategy,
            first_entry_ts_ms=first_entry_ts_ms,
            avg_entry_notional=1980.0,
            eth_qty=1.0,
            side="LONG",
            old_degrade_stage="MIDDLE_RUNNER",
            old_degraded_ts_ms=first_entry_ts_ms + 60_000,
        )

        boll = _boll(
            middle=2000.0,
            upper=2100.0,
            lower=1900.0,
            tp_middle=2008.0,
            tp_upper=2110.0,
            tp_lower=1890.0,
        )
        cvd = _cvd()

        coord = _coordinator(strategy)
        intent = coord.open_position(
            side="LONG",
            intent_type="ADD_LONG",
            price=1970.0,
            ts_ms=ts_ms,
            boll=boll,
            cvd=cvd,
            reason="test_age_over_6h_old_stage_middle_runner",
        )

        self.assertEqual(strategy.state.three_stage_pre_tp1_degrade_stage, "SINGLE")
        self.assertEqual(intent.tp_plan, "SINGLE")


# ── Test E: Add does NOT reset first_entry_ts_ms ───────────────────────


class TestAddPreservesFirstEntryTsMs(unittest.TestCase):
    """An add must never reset ``first_entry_ts_ms``."""

    def test_add_preserves_first_entry_ts_ms(self) -> None:
        ts_ms = BASE_TS_MS
        known_old_ts = ts_ms - 5 * 3600 * 1000  # 5 hours ago

        config = BollCvdReclaimStrategyConfig(
            three_stage_runner_enabled=True,
            middle_bucket_split_enabled=True,
            tp_boll_enabled=True,
            tp_min_net_profit_pct=0.004,
            three_stage_pre_tp1_degrade_enabled=True,
            three_stage_pre_tp1_middle_runner_after_seconds=10800,
            three_stage_pre_tp1_single_after_seconds=21600,
            middle_runner_enabled=True,
        )
        strategy = BollCvdReclaimStrategy(config, _sizer())

        _setup_strategy_for_add(
            strategy=strategy,
            first_entry_ts_ms=known_old_ts,
            avg_entry_notional=1980.0,
            eth_qty=1.0,
            side="LONG",
            old_degrade_stage="SINGLE",
            old_degraded_ts_ms=known_old_ts + 60_000,
        )

        boll = _boll(
            middle=2000.0,
            upper=2100.0,
            lower=1900.0,
            tp_middle=2008.0,
            tp_upper=2110.0,
            tp_lower=1890.0,
        )
        cvd = _cvd()

        coord = _coordinator(strategy)
        coord.open_position(
            side="LONG",
            intent_type="ADD_LONG",
            price=1970.0,
            ts_ms=ts_ms,
            boll=boll,
            cvd=cvd,
            reason="test_add_preserves_first_entry_ts_ms",
        )

        # first_entry_ts_ms MUST be unchanged
        self.assertEqual(strategy.state.first_entry_ts_ms, known_old_ts)
        # layers increased from 1 to 2 (so this was indeed an add, not first entry)
        self.assertEqual(strategy.state.layers, 2)


# ── Test F: Middle Bucket Split old state does not contaminate new replan ──


class TestMiddleBucketSplitNotContaminated(unittest.TestCase):
    """Stale middle_bucket_split_* state must be cleared before replan."""

    def test_old_split_state_cleared_for_three_stage_replan(self) -> None:
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

        _setup_strategy_for_add(
            strategy=strategy,
            first_entry_ts_ms=first_entry_ts_ms,
            avg_entry_notional=1980.0,
            eth_qty=1.0,
            side="LONG",
            old_degrade_stage="SINGLE",
            old_degraded_ts_ms=first_entry_ts_ms + 60_000,
        )

        # Inject stale middle_bucket_split state with OLD prices
        strategy.state.middle_bucket_split_active = True
        strategy.state.middle_bucket_split_fast_price = 1990.0  # old stale price
        strategy.state.middle_bucket_split_slow_price = 1995.0  # old stale price

        boll = _boll(
            middle=2000.0,
            upper=2100.0,
            lower=1900.0,
            tp_middle=2008.0,  # NEW tp_middle (BOLL15)
            tp_upper=2110.0,
            tp_lower=1890.0,
        )
        cvd = _cvd()

        coord = _coordinator(strategy)
        intent = coord.open_position(
            side="LONG",
            intent_type="ADD_LONG",
            price=1970.0,
            ts_ms=ts_ms,
            boll=boll,
            cvd=cvd,
            reason="test_old_split_state_cleared_for_three_stage_replan",
        )

        # Should recover to THREE_STAGE_RUNNER with new split prices
        self.assertEqual(intent.tp_plan, "THREE_STAGE_RUNNER")
        self.assertTrue(strategy.state.middle_bucket_split_active)

        # fast_price must be the NEW tp_middle (2008), NOT the stale 1990
        self.assertNotEqual(strategy.state.middle_bucket_split_fast_price, 1990.0)
        if strategy.state.middle_bucket_split_fast_price is not None:
            # fast price should come from new BOLL15 middle
            self.assertAlmostEqual(
                float(strategy.state.middle_bucket_split_fast_price),
                2008.0,
                delta=1.0,
                msg="fast_price should reflect new BOLL15 middle, not stale price",
            )

    def test_old_split_state_cleared_for_single_replan(self) -> None:
        ts_ms = BASE_TS_MS
        age_7h = 7 * 3600
        first_entry_ts_ms = ts_ms - age_7h * 1000

        config = BollCvdReclaimStrategyConfig(
            three_stage_runner_enabled=True,
            middle_bucket_split_enabled=True,
            tp_boll_enabled=True,
            tp_min_net_profit_pct=0.004,
            three_stage_pre_tp1_degrade_enabled=True,
            three_stage_pre_tp1_middle_runner_after_seconds=10800,
            three_stage_pre_tp1_single_after_seconds=21600,
            middle_runner_enabled=True,
        )
        strategy = BollCvdReclaimStrategy(config, _sizer())

        _setup_strategy_for_add(
            strategy=strategy,
            first_entry_ts_ms=first_entry_ts_ms,
            avg_entry_notional=1980.0,
            eth_qty=1.0,
            side="LONG",
            old_degrade_stage="SINGLE",
            old_degraded_ts_ms=first_entry_ts_ms + 60_000,
        )

        # Inject stale split state
        strategy.state.middle_bucket_split_active = True
        strategy.state.middle_bucket_split_fast_price = 1990.0
        strategy.state.middle_bucket_split_slow_price = 1995.0

        boll = _boll(
            middle=2000.0,
            upper=2100.0,
            lower=1900.0,
            tp_middle=2008.0,
            tp_upper=2110.0,
            tp_lower=1890.0,
        )
        cvd = _cvd()

        coord = _coordinator(strategy)
        intent = coord.open_position(
            side="LONG",
            intent_type="ADD_LONG",
            price=1970.0,
            ts_ms=ts_ms,
            boll=boll,
            cvd=cvd,
            reason="test_old_split_state_cleared_for_single_replan",
        )

        # Age ≥ 6h → SINGLE
        self.assertEqual(intent.tp_plan, "SINGLE")
        self.assertFalse(strategy.state.middle_bucket_split_active)

        # All split price fields must be None
        self.assertIsNone(strategy.state.middle_bucket_split_fast_price)
        self.assertIsNone(strategy.state.middle_bucket_split_slow_price)


# ── Test G: Pure order-spec regression (execution layer unchanged) ─────


class TestOrderSpecRegression(unittest.TestCase):
    """Ensure ``build_take_profit_order_specs`` output labels are unchanged."""

    def _make_split_input(self, active: bool = True) -> MiddleBucketSplitOrderInput:
        return MiddleBucketSplitOrderInput(
            active=active,
            fast_price=2008.0,
            slow_price=2000.0,
            effective_price=2005.0,
            middle_bucket_ratio=Decimal("0.60"),
            fast_ratio_of_bucket=Decimal("0.70"),
            slow_ratio_of_bucket=Decimal("0.30"),
            fast_total_ratio=Decimal("0.42"),
            slow_total_ratio=Decimal("0.18"),
        )

    def test_g1_three_stage_with_split(self) -> None:
        """THREE_STAGE_RUNNER + middle_bucket_split → tp1_middle_fast/slow/outer."""
        split = self._make_split_input(active=True)
        decision = build_take_profit_order_specs(
            position_contracts=Decimal("6.79"),
            min_contracts=Decimal("0.01"),
            contract_precision=Decimal("0.01"),
            tp_plan="THREE_STAGE_RUNNER",
            final_tp_price=2110.0,
            partial_tp_price=2005.0,
            partial_tp_ratio=Decimal("0.60"),
            partial_tp_consumed=False,
            middle_runner_active=False,
            three_stage_tp1_price=2005.0,
            three_stage_tp2_price=2110.0,
            three_stage_tp1_ratio=Decimal("0.60"),
            three_stage_tp2_ratio=Decimal("0.20"),
            three_stage_tp1_consumed=False,
            three_stage_tp2_consumed=False,
            three_stage_runner_ratio=Decimal("0.20"),
            middle_bucket_split=split,
        )
        labels = [s.label for s in decision.specs]
        self.assertIn("tp1_middle_fast", labels)
        self.assertIn("tp1_middle_slow", labels)
        self.assertIn("tp2_outer", labels)
        self.assertEqual(len(decision.specs), 3)

    def test_g2_middle_runner_with_split(self) -> None:
        """MIDDLE_RUNNER + middle_bucket_split → middle_fast/slow/runner."""
        split = self._make_split_input(active=True)
        decision = build_take_profit_order_specs(
            position_contracts=Decimal("5.0"),
            min_contracts=Decimal("0.01"),
            contract_precision=Decimal("0.01"),
            tp_plan="MIDDLE_RUNNER",
            final_tp_price=2100.0,
            partial_tp_price=2005.0,
            partial_tp_ratio=Decimal("0.80"),
            partial_tp_consumed=False,
            middle_runner_active=False,
            three_stage_tp1_price=None,
            three_stage_tp2_price=None,
            three_stage_tp1_ratio=Decimal("0"),
            three_stage_tp2_ratio=Decimal("0"),
            three_stage_tp1_consumed=False,
            three_stage_tp2_consumed=False,
            three_stage_runner_ratio=Decimal("0"),
            middle_bucket_split=split,
        )
        labels = [s.label for s in decision.specs]
        self.assertIn("middle_fast", labels)
        self.assertIn("middle_slow", labels)
        self.assertIn("runner", labels)
        self.assertEqual(len(decision.specs), 3)

    def test_g3_single(self) -> None:
        """SINGLE plan → one 'final' order spec."""
        decision = build_take_profit_order_specs(
            position_contracts=Decimal("2.0"),
            min_contracts=Decimal("0.01"),
            contract_precision=Decimal("0.01"),
            tp_plan="SINGLE",
            final_tp_price=2100.0,
            partial_tp_price=None,
            partial_tp_ratio=Decimal("0"),
            partial_tp_consumed=False,
            middle_runner_active=False,
            three_stage_tp1_price=None,
            three_stage_tp2_price=None,
            three_stage_tp1_ratio=Decimal("0"),
            three_stage_tp2_ratio=Decimal("0"),
            three_stage_tp1_consumed=False,
            three_stage_tp2_consumed=False,
            three_stage_runner_ratio=Decimal("0"),
            middle_bucket_split=None,
        )
        labels = [s.label for s in decision.specs]
        self.assertEqual(labels, ["final"])
        self.assertEqual(len(decision.specs), 1)


if __name__ == "__main__":
    unittest.main()
