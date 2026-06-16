"""Verify that legacy SPLIT_PARTIAL_FINAL is fully removed.

After the legacy layer-based split TP removal:
- BollCvdReclaimStrategyConfig no longer accepts split_tp_enabled / split_tp_min_layers / etc.
- tp_plan selection never returns "SPLIT_PARTIAL_FINAL"
- open_position only generates SINGLE / MIDDLE_RUNNER / THREE_STAGE_RUNNER
- no-add + layers does NOT trigger legacy split TP fallback
- Middle Runner / Three-stage Runner plans remain intact
"""

from __future__ import annotations

import unittest

from src.indicators.cvd_tracker import CvdSnapshot
from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.risk.simple_position_sizer import PositionSize, SimplePositionSizer, SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
    StrategyPositionState,
)
from src.strategies.tp_plan_selector import select_tp_plan


# ── helpers ──────────────────────────────────────────────────────────────


def _sizer() -> SimplePositionSizer:
    return SimplePositionSizer(SimplePositionSizerConfig())


def _boll(
    middle: float = 2000.0,
    upper: float = 2150.0,
    lower: float = 1900.0,
    candle_ts_ms: int = 1000,
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
    )


def _cvd() -> CvdSnapshot:
    return CvdSnapshot(
        ts_ms=1000,
        price=2000.0,
        side="buy",
        size=1.0,
        signed_delta=1.0,
        total_cvd=1.0,
        fast_cvd=1.0,
        previous_fast_cvd=0.0,
        buy_volume=1.0,
        sell_volume=0.0,
        buy_ratio=1.0,
        sell_ratio=0.0,
        cross_positive=True,
        cross_negative=False,
        cvd_increasing=True,
        cvd_decreasing=False,
        no_new_low=True,
        no_new_high=True,
        window_low=1990.0,
        window_high=2010.0,
        burst_net_move_pct=0.0,
        burst_range_pct=0.0,
        baseline_range_pct=0.0,
        burst_move_ratio=0.0,
        burst_volume=0.0,
        baseline_volume=0.0,
        burst_volume_ratio=0.0,
        up_burst=False,
        down_burst=False,
    )


def _strategy(**config_overrides) -> BollCvdReclaimStrategy:
    config_values = dict(entry_rr_target="FINAL_TP", entry_max_stop_distance_pct=0.0)
    config_values.update(config_overrides)
    return BollCvdReclaimStrategy(
        BollCvdReclaimStrategyConfig(**config_values),
        _sizer(),
    )


# ── tests ────────────────────────────────────────────────────────────────


class NoLegacySplitTpConfigTest(unittest.TestCase):
    """Verify split_tp_* fields are removed from config."""

    def test_config_rejects_split_tp_enabled(self) -> None:
        """BollCvdReclaimStrategyConfig must not accept split_tp_enabled."""
        with self.assertRaises(TypeError):
            BollCvdReclaimStrategyConfig(split_tp_enabled=True)

    def test_config_rejects_split_tp_min_layers(self) -> None:
        """BollCvdReclaimStrategyConfig must not accept split_tp_min_layers."""
        with self.assertRaises(TypeError):
            BollCvdReclaimStrategyConfig(split_tp_min_layers=4)

    def test_config_rejects_split_tp_path_ratio(self) -> None:
        """BollCvdReclaimStrategyConfig must not accept split_tp_path_ratio."""
        with self.assertRaises(TypeError):
            BollCvdReclaimStrategyConfig(split_tp_path_ratio=0.8)

    def test_config_rejects_split_tp_partial_ratio(self) -> None:
        """BollCvdReclaimStrategyConfig must not accept split_tp_partial_ratio."""
        with self.assertRaises(TypeError):
            BollCvdReclaimStrategyConfig(split_tp_partial_ratio=0.5)

    def test_config_rejects_split_tp_min_profit_pct(self) -> None:
        """BollCvdReclaimStrategyConfig must not accept split_tp_min_profit_pct."""
        with self.assertRaises(TypeError):
            BollCvdReclaimStrategyConfig(split_tp_min_profit_pct=0.004)


class NoLegacySplitTpPlanTest(unittest.TestCase):
    """Verify select_tp_plan() never returns SPLIT_PARTIAL_FINAL."""

    def test_no_split_plan_when_all_runners_disabled(self) -> None:
        """When no runner plan is allowed, only SINGLE is returned."""
        sel = select_tp_plan(
            side="LONG",
            final_tp=2200.0,
            layers=5,
            tp_mode="MIDDLE",
            boll_exists=True,
            three_stage_pre_tp1_degrade_stage=None,
            middle_runner_first_close_ratio=0.8,
            tp_middle_profit_fallback_price=2100.0,
            three_stage_runner_plan_allowed=False,
            three_stage_tp1_ratio=0.4,
            three_stage_runner_enabled=False,
            middle_runner_plan_allowed=False,
        )
        self.assertEqual(sel.tp_plan, "SINGLE")
        self.assertIsNone(sel.partial_tp_price)
        self.assertEqual(sel.partial_tp_ratio, 0.0)

    def test_split_tp_params_not_accepted_by_function(self) -> None:
        """select_tp_plan() must reject legacy split_tp kwargs."""
        with self.assertRaises(TypeError):
            select_tp_plan(
                side="LONG",
                final_tp=2200.0,
                layers=5,
                tp_mode="MIDDLE",
                boll_exists=True,
                three_stage_pre_tp1_degrade_stage=None,
                middle_runner_first_close_ratio=0.8,
                tp_middle_profit_fallback_price=2100.0,
                three_stage_runner_plan_allowed=False,
                three_stage_tp1_ratio=0.4,
                three_stage_runner_enabled=False,
                middle_runner_plan_allowed=False,
                split_tp_enabled=True,
                split_tp_min_layers=3,
            )

    def test_plan_only_single_middle_three_stage(self) -> None:
        """tp_plan is always one of SINGLE / MIDDLE_RUNNER / THREE_STAGE_RUNNER."""
        valid_plans = {"SINGLE", "MIDDLE_RUNNER", "THREE_STAGE_RUNNER"}

        # Case 1: All disabled → SINGLE
        sel1 = select_tp_plan(
            side="LONG", final_tp=2200.0, layers=1, tp_mode="MIDDLE", boll_exists=True,
            three_stage_pre_tp1_degrade_stage=None, middle_runner_first_close_ratio=0.8,
            tp_middle_profit_fallback_price=2100.0, three_stage_runner_plan_allowed=False,
            three_stage_tp1_ratio=0.4, three_stage_runner_enabled=False,
            middle_runner_plan_allowed=False,
        )
        self.assertIn(sel1.tp_plan, valid_plans)

        # Case 2: Middle runner allowed → MIDDLE_RUNNER
        sel2 = select_tp_plan(
            side="LONG", final_tp=2200.0, layers=1, tp_mode="MIDDLE", boll_exists=True,
            three_stage_pre_tp1_degrade_stage=None, middle_runner_first_close_ratio=0.8,
            tp_middle_profit_fallback_price=2100.0, three_stage_runner_plan_allowed=False,
            three_stage_tp1_ratio=0.4, three_stage_runner_enabled=False,
            middle_runner_plan_allowed=True,
        )
        self.assertIn(sel2.tp_plan, valid_plans)

        # Case 3: Three-stage allowed → THREE_STAGE_RUNNER
        sel3 = select_tp_plan(
            side="LONG", final_tp=2200.0, layers=1, tp_mode="MIDDLE", boll_exists=True,
            three_stage_pre_tp1_degrade_stage=None, middle_runner_first_close_ratio=0.8,
            tp_middle_profit_fallback_price=2100.0, three_stage_runner_plan_allowed=True,
            three_stage_tp1_ratio=0.4, three_stage_runner_enabled=True,
            middle_runner_plan_allowed=False,
        )
        self.assertIn(sel3.tp_plan, valid_plans)

        # None should be SPLIT_PARTIAL_FINAL
        self.assertNotEqual(sel1.tp_plan, "SPLIT_PARTIAL_FINAL")
        self.assertNotEqual(sel2.tp_plan, "SPLIT_PARTIAL_FINAL")
        self.assertNotEqual(sel3.tp_plan, "SPLIT_PARTIAL_FINAL")


class NoLegacySplitTpOpenPositionTest(unittest.TestCase):
    """Verify open_position never sets SPLIT_PARTIAL_FINAL."""

    def test_open_long_with_middle_runner_uses_middle_runner(self) -> None:
        """open_position LONG with middle_runner_enabled → MIDDLE_RUNNER (not SPLIT)."""
        strat = _strategy(
            middle_runner_enabled=True,
            breakeven_fee_buffer_pct=0.001,
            tp_min_net_profit_pct=0.002,
            entry_min_reward_risk=0.0,
        )
        strat.state.avg_entry_price = 2000.0
        intent = strat._open_position(
            "LONG", "OPEN_LONG", 2000.0, 2000,
            _boll(middle=2010.0, upper=2150.0, lower=1900.0),
            _cvd(), "test",
        )
        self.assertIsNotNone(intent)
        self.assertIn(intent.tp_plan, {"SINGLE", "MIDDLE_RUNNER", "THREE_STAGE_RUNNER"})
        self.assertNotEqual(intent.tp_plan, "SPLIT_PARTIAL_FINAL")

    def test_open_long_without_runners_uses_single(self) -> None:
        """open_position LONG with all runners disabled → SINGLE (not SPLIT)."""
        strat = _strategy(
            middle_runner_enabled=False,
            three_stage_runner_enabled=False,
            entry_min_reward_risk=0.0,
        )
        strat.state.avg_entry_price = 2000.0
        intent = strat._open_position(
            "LONG", "OPEN_LONG", 2000.0, 2000,
            _boll(middle=2050.0, upper=2150.0, lower=1900.0),
            _cvd(), "test",
        )
        self.assertIsNotNone(intent)
        self.assertEqual(intent.tp_plan, "SINGLE")

    def test_layers_do_not_trigger_split_tp(self) -> None:
        """Even with many layers, SPLIT_PARTIAL_FINAL must not be selected."""
        strat = _strategy(
            middle_runner_enabled=False,
            three_stage_runner_enabled=False,
            entry_min_reward_risk=0.0,
        )
        strat.state.avg_entry_price = 2000.0
        strat.state.layers = 10  # Many layers, but no legacy split trigger
        intent = strat._open_position(
            "LONG", "OPEN_LONG", 2000.0, 2000,
            _boll(middle=2050.0, upper=2150.0, lower=1900.0),
            _cvd(), "test",
        )
        self.assertIsNotNone(intent)
        self.assertNotEqual(intent.tp_plan, "SPLIT_PARTIAL_FINAL")
        # Should be SINGLE since no runner is enabled
        self.assertEqual(intent.tp_plan, "SINGLE")


class MiddleRunnerThreeStagePreservedTest(unittest.TestCase):
    """Verify Middle Runner and Three-stage Runner are preserved."""

    def test_middle_runner_plan_is_still_generated(self) -> None:
        """select_tp_plan still returns MIDDLE_RUNNER when allowed."""
        sel = select_tp_plan(
            side="LONG",
            final_tp=2200.0,
            layers=1,
            tp_mode="MIDDLE",
            boll_exists=True,
            three_stage_pre_tp1_degrade_stage=None,
            middle_runner_first_close_ratio=0.8,
            tp_middle_profit_fallback_price=2100.0,
            three_stage_runner_plan_allowed=False,
            three_stage_tp1_ratio=0.4,
            three_stage_runner_enabled=False,
            middle_runner_plan_allowed=True,
        )
        self.assertEqual(sel.tp_plan, "MIDDLE_RUNNER")
        self.assertEqual(sel.partial_tp_price, 2100.0)
        self.assertAlmostEqual(sel.partial_tp_ratio, 0.8)

    def test_three_stage_runner_plan_is_still_generated(self) -> None:
        """select_tp_plan still returns THREE_STAGE_RUNNER when allowed."""
        sel = select_tp_plan(
            side="LONG",
            final_tp=2200.0,
            layers=1,
            tp_mode="MIDDLE",
            boll_exists=True,
            three_stage_pre_tp1_degrade_stage=None,
            middle_runner_first_close_ratio=0.8,
            tp_middle_profit_fallback_price=2100.0,
            three_stage_runner_plan_allowed=True,
            three_stage_tp1_ratio=0.4,
            three_stage_runner_enabled=True,
            middle_runner_plan_allowed=False,
        )
        self.assertEqual(sel.tp_plan, "THREE_STAGE_RUNNER")
        self.assertEqual(sel.partial_tp_price, 2100.0)
        self.assertAlmostEqual(sel.partial_tp_ratio, 0.4)

    def test_partial_tp_fields_still_present(self) -> None:
        """TpPlanSelection still carries partial_tp_price/partial_tp_ratio."""
        sel = select_tp_plan(
            side="LONG",
            final_tp=2200.0,
            layers=1,
            tp_mode="MIDDLE",
            boll_exists=True,
            three_stage_pre_tp1_degrade_stage=None,
            middle_runner_first_close_ratio=0.8,
            tp_middle_profit_fallback_price=2100.0,
            three_stage_runner_plan_allowed=False,
            three_stage_tp1_ratio=0.4,
            three_stage_runner_enabled=False,
            middle_runner_plan_allowed=True,
        )
        # partial_tp_price and partial_tp_ratio must exist (used by Middle Runner / Three-stage)
        self.assertIsNotNone(sel.partial_tp_price)
        self.assertGreater(sel.partial_tp_ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
