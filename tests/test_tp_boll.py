"""Tests for TP-only BOLL window (TP_BOLL_WINDOW=15).

Verifies:
- BollSnapshot carries both structure BOLL20 and TP BOLL15 fields.
- TP resolver prefers TP_BOLL15, falls back to structure BOLL20.
- SINGLE / MIDDLE RUNNER use TP_BOLL15 for TP prices.
- Three-Stage TP1 uses TP_BOLL15 middle (with profit fallback to BOLL20 middle).
- Three-Stage TP2 uses structure BOLL20 outer by default
  (TP2 is the structural confirmation gate before Trend Runner).
- Runner and Sidecar are NOT affected.
- Profit distance / fallback logic is preserved.
"""

from __future__ import annotations

import os
from dataclasses import replace
from unittest import mock

from src.monitors.boll_band_breakout_monitor import (
    BollBandBreakoutMonitorConfig,
    BollCalculator,
    BollSnapshot,
)
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
    StrategyPositionState,
)


# ── helpers ────────────────────────────────────────────────────────────

def _boll_structure_20(
        middle: float = 100.0,
        upper: float = 110.0,
        lower: float = 90.0,
) -> BollSnapshot:
    return BollSnapshot(
        inst_id="ETH-USDT-SWAP",
        candle_ts_ms=1000,
        close=middle,
        middle=middle,
        upper=upper,
        lower=lower,
        upper_distance_pct=0.01,
        lower_distance_pct=0.01,
        alert_switch_on=True,
        live_mode=True,
    )


def _boll_with_tp(
        middle: float = 100.0,
        upper: float = 110.0,
        lower: float = 90.0,
        tp_middle: float | None = 101.0,
        tp_upper: float | None = 108.0,
        tp_lower: float | None = 92.0,
        tp_window: int | None = 15,
) -> BollSnapshot:
    return BollSnapshot(
        inst_id="ETH-USDT-SWAP",
        candle_ts_ms=1000,
        close=middle,
        middle=middle,
        upper=upper,
        lower=lower,
        upper_distance_pct=0.01,
        lower_distance_pct=0.01,
        alert_switch_on=True,
        live_mode=True,
        tp_lower=tp_lower,
        tp_middle=tp_middle,
        tp_upper=tp_upper,
        tp_window=tp_window,
    )


def _cvd() -> "CvdSnapshot":
    from src.indicators.cvd_tracker import CvdSnapshot
    return CvdSnapshot(
        ts_ms=1000,
        price=100.0,
        side="buy",
        size=1.0,
        signed_delta=0.1,
        total_cvd=0.5,
        fast_cvd=0.1,
        previous_fast_cvd=0.09,
        buy_volume=60.0,
        sell_volume=40.0,
        buy_ratio=0.6,
        sell_ratio=0.4,
        cross_positive=True,
        cross_negative=False,
        cvd_increasing=True,
        cvd_decreasing=False,
        no_new_low=True,
        no_new_high=False,
        window_low=99.0,
        window_high=101.0,
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


def _sizer() -> SimplePositionSizer:
    sizer_config = SimplePositionSizerConfig(
        dry_run_equity_usdt=1000.0,
        layer_margin_pct=0.03,
        leverage=50.0,
    )
    return SimplePositionSizer(sizer_config)


def _strategy(**overrides) -> BollCvdReclaimStrategy:
    cfg = BollCvdReclaimStrategyConfig(**overrides)
    return BollCvdReclaimStrategy(cfg, _sizer())


def _min_profit_long(middle: float, be: float, min_pct: float = 0.002) -> bool:
    return middle >= be * (1 + min_pct)


def _min_profit_short(middle: float, be: float, min_pct: float = 0.002) -> bool:
    return middle <= be * (1 - min_pct)


# ═══════════════════════════════════════════════════════════════════════
# 1. BollSnapshot / BollCalculator tests
# ═══════════════════════════════════════════════════════════════════════

class TestBollSnapshotTpFields:
    def test_default_boll_snapshot_has_tp_fields_none(self):
        b = _boll_structure_20()
        assert b.tp_lower is None
        assert b.tp_middle is None
        assert b.tp_upper is None
        assert b.tp_window is None

    def test_boll_with_tp_fields_populated(self):
        b = _boll_with_tp()
        assert b.tp_lower == 92.0
        assert b.tp_middle == 101.0
        assert b.tp_upper == 108.0
        assert b.tp_window == 15
        # structure BOLL unchanged
        assert b.middle == 100.0
        assert b.upper == 110.0
        assert b.lower == 90.0

    def test_tp_boll_window_does_not_replace_structure_boll_via_calculator(self):
        """Given same closes, window=20 vs window=15 produce different values."""
        import random
        random.seed(42)
        closes = [100.0 + random.uniform(-3, 3) for _ in range(25)]

        mid_20, up_20, lo_20 = BollCalculator.calculate(closes, 20, 2.0)
        mid_15, up_15, lo_15 = BollCalculator.calculate(closes, 15, 2.0)

        # They should differ because windows are different.
        assert abs(mid_20 - mid_15) > 0.0001 or abs(up_20 - up_15) > 0.0001 or abs(lo_20 - lo_15) > 0.0001


# ═══════════════════════════════════════════════════════════════════════
# 2. TP resolver tests
# ═══════════════════════════════════════════════════════════════════════

class TestTpResolver:
    def test_select_tp_middle_prefers_tp_boll_when_available(self):
        s = _strategy()
        b = _boll_with_tp(middle=100.0, tp_middle=101.0)
        price, source = s._select_tp_middle(b)
        assert price == 101.0
        assert source == "TP_BOLL"

    def test_select_tp_middle_falls_back_to_structure_boll(self):
        s = _strategy()
        b = _boll_structure_20(middle=100.0)
        price, source = s._select_tp_middle(b)
        assert price == 100.0
        assert source == "STRUCTURE_BOLL"

    def test_select_tp_outer_prefers_tp_boll_long(self):
        s = _strategy()
        b = _boll_with_tp(upper=110.0, tp_upper=108.0)
        price, source = s._select_tp_outer("LONG", b)
        assert price == 108.0
        assert source == "TP_BOLL"

    def test_select_tp_outer_prefers_tp_boll_short(self):
        s = _strategy()
        b = _boll_with_tp(lower=90.0, tp_lower=92.0)
        price, source = s._select_tp_outer("SHORT", b)
        assert price == 92.0
        assert source == "TP_BOLL"

    def test_select_tp_outer_falls_back_long(self):
        s = _strategy()
        b = _boll_structure_20(upper=110.0)
        price, source = s._select_tp_outer("LONG", b)
        assert price == 110.0
        assert source == "STRUCTURE_BOLL"

    def test_select_tp_outer_falls_back_short(self):
        s = _strategy()
        b = _boll_structure_20(lower=90.0)
        price, source = s._select_tp_outer("SHORT", b)
        assert price == 90.0
        assert source == "STRUCTURE_BOLL"

    def test_tp_boll_available_when_fields_present(self):
        s = _strategy()
        b = _boll_with_tp()
        assert s._tp_boll_available(b) is True

    def test_tp_boll_not_available_when_tp_middle_none(self):
        s = _strategy()
        b = _boll_with_tp(tp_middle=None)
        assert s._tp_boll_available(b) is False

    def test_tp_boll_not_available_when_disabled(self):
        s = _strategy(tp_boll_enabled=False)
        b = _boll_with_tp()
        assert s._tp_boll_available(b) is False

    def test_tp_boll_disabled_config_leaves_resolver_returning_structure(self):
        s = _strategy(tp_boll_enabled=False)
        b = _boll_with_tp(middle=100.0, tp_middle=105.0, upper=110.0, tp_upper=108.0)
        price, source = s._select_tp_middle(b)
        assert price == 100.0
        assert source == "STRUCTURE_BOLL"
        price2, source2 = s._select_tp_outer("LONG", b)
        assert price2 == 110.0
        assert source2 == "STRUCTURE_BOLL"


# ═══════════════════════════════════════════════════════════════════════
# 3. SINGLE TP tests
# ═══════════════════════════════════════════════════════════════════════

class TestSingleTpWithTpBoll:
    def test_single_tp_uses_tp_boll15_middle_when_middle_target_long(self):
        """LONG: structure middle=100, tp middle=101 → SINGLE middle TP uses 101."""
        s = _strategy(tp_min_net_profit_pct=0.001)
        b = _boll_with_tp(middle=100.0, tp_middle=101.0)

        # Simulate a position with avg_entry well below middle
        s.state.side = "LONG"
        s.state.layers = 2
        s.state.avg_entry_price = 98.0
        s.state.total_entry_qty = 1.0
        s.state.total_entry_notional = 98.0
        s.state.position_cost_entry_notional = 98.0
        s.state.position_cost_remaining_qty = 1.0
        s._refresh_net_remaining_breakeven_price()

        price, mode = s._select_tp_price("LONG", b)
        # TP_BOLL15 middle (101) should be chosen since it meets net profit
        assert price == 101.0
        assert mode == "MIDDLE"

    def test_single_tp_uses_tp_boll15_middle_when_middle_target_short(self):
        """SHORT: structure middle=100, tp middle=99 → SINGLE middle TP uses 99."""
        s = _strategy(tp_min_net_profit_pct=0.001)
        b = _boll_with_tp(middle=100.0, tp_middle=99.0)

        s.state.side = "SHORT"
        s.state.layers = 2
        s.state.avg_entry_price = 102.0
        s.state.total_entry_qty = 1.0
        s.state.total_entry_notional = 102.0
        s.state.position_cost_entry_notional = 102.0
        s.state.position_cost_remaining_qty = 1.0
        s._refresh_net_remaining_breakeven_price()

        price, mode = s._select_tp_price("SHORT", b)
        assert price == 99.0
        assert mode == "MIDDLE"

    def test_single_tp_uses_tp_boll15_outer_when_outer_target_long(self):
        """LONG: structure upper=110, tp upper=108 → SINGLE outer TP uses 108."""
        s = _strategy()
        b = _boll_with_tp(upper=110.0, tp_upper=108.0, middle=100.0, tp_middle=101.0)

        s.state.side = "LONG"
        s.state.layers = 2
        # Set avg_entry high enough that neither middle meets profit → outer used
        s.state.avg_entry_price = 100.9
        s.state.total_entry_qty = 1.0
        s.state.total_entry_notional = 100.9
        s.state.position_cost_entry_notional = 100.9
        s.state.position_cost_remaining_qty = 1.0
        s._refresh_net_remaining_breakeven_price()

        price, mode = s._select_tp_price("LONG", b)
        # Should use outer since middle profit is insufficient
        assert price == 108.0
        assert mode == "UPPER"

    def test_single_tp_uses_tp_boll15_outer_when_outer_target_short(self):
        """SHORT: structure lower=90, tp lower=92 → SINGLE outer TP uses 92."""
        s = _strategy()
        b = _boll_with_tp(lower=90.0, tp_lower=92.0, middle=100.0, tp_middle=99.0)

        s.state.side = "SHORT"
        s.state.layers = 2
        # Set avg_entry low enough that neither middle meets profit → outer used
        s.state.avg_entry_price = 99.1
        s.state.total_entry_qty = 1.0
        s.state.total_entry_notional = 99.1
        s.state.position_cost_entry_notional = 99.1
        s.state.position_cost_remaining_qty = 1.0
        s._refresh_net_remaining_breakeven_price()

        price, mode = s._select_tp_price("SHORT", b)
        assert price == 92.0
        assert mode == "LOWER"

    def test_single_tp_middle_profit_fallback_verified(self):
        """TP_BOLL15 middle has insufficient profit, BOLL20 middle sufficient → fallback."""
        s = _strategy(tp_min_net_profit_pct=0.005)  # 0.5% minimum
        b = _boll_with_tp(
            middle=101.0,  # structure BOLL20 middle = 101 → ~1% profit from 100
            tp_middle=100.3,  # TP_BOLL15 middle = 100.3 → ~0.3% profit, below 0.5%
        )

        s.state.side = "LONG"
        s.state.layers = 2
        s.state.avg_entry_price = 100.0
        s.state.total_entry_qty = 1.0
        s.state.total_entry_notional = 100.0
        s.state.position_cost_entry_notional = 100.0
        s.state.position_cost_remaining_qty = 1.0
        s._refresh_net_remaining_breakeven_price()

        price, mode = s._select_tp_price("LONG", b)
        # TP_BOLL15 middle (100.3) gives 0.3% < 0.5% → skip
        # BOLL20 middle (101.0) gives 1% >= 0.5% → use it
        assert price == 101.0
        assert mode == "MIDDLE"


# ═══════════════════════════════════════════════════════════════════════
# 4. MIDDLE RUNNER tests
# ═══════════════════════════════════════════════════════════════════════

class TestMiddleRunnerTpBoll:
    def test_middle_runner_initial_tp_uses_tp_boll15_for_first_and_final_long(self):
        """LONG: first TP=TP_BOLL15 middle, final TP=TP_BOLL15 upper."""
        s = _strategy(middle_runner_enabled=True)
        b = _boll_with_tp(middle=100.0, tp_middle=101.0, upper=110.0, tp_upper=108.0)

        first_tp, src_mid = s._select_tp_middle(b)
        final_tp, src_out = s._select_tp_outer("LONG", b)

        assert first_tp == 101.0
        assert final_tp == 108.0
        assert src_mid == "TP_BOLL"
        assert src_out == "TP_BOLL"

    def test_middle_runner_initial_tp_uses_tp_boll15_for_first_and_final_short(self):
        """SHORT: first TP=TP_BOLL15 middle, final TP=TP_BOLL15 lower."""
        s = _strategy(middle_runner_enabled=True)
        b = _boll_with_tp(middle=100.0, tp_middle=99.0, lower=90.0, tp_lower=92.0)

        first_tp, src_mid = s._select_tp_middle(b)
        final_tp, src_out = s._select_tp_outer("SHORT", b)

        assert first_tp == 99.0
        assert final_tp == 92.0
        assert src_out == "TP_BOLL"

    def test_middle_runner_tp_boll_missing_falls_back_to_structure(self):
        """When TP_BOLL is not available, middle runner uses structure BOLL20."""
        s = _strategy(middle_runner_enabled=True)
        b = _boll_structure_20(middle=100.0, upper=110.0)

        first_tp, src_mid = s._select_tp_middle(b)
        final_tp, src_out = s._select_tp_outer("LONG", b)

        assert first_tp == 100.0
        assert final_tp == 110.0
        assert src_mid == "STRUCTURE_BOLL"
        assert src_out == "STRUCTURE_BOLL"

    def test_middle_runner_planned_sets_tp_boll_prices_long(self):
        """_set_middle_runner_planned uses TP_BOLL15 prices."""
        s = _strategy(middle_runner_enabled=True)
        b = _boll_with_tp(middle=100.0, tp_middle=101.0, upper=110.0, tp_upper=108.0)
        s.state.side = "LONG"

        s._set_middle_runner_planned(101.0, 108.0)
        assert s.state.middle_runner_first_tp_price == 101.0
        assert s.state.middle_runner_final_tp_price == 108.0


# ═══════════════════════════════════════════════════════════════════════
# 5. THREE-STAGE tests
# ═══════════════════════════════════════════════════════════════════════

class TestThreeStageTpBoll:
    def test_three_stage_initial_tp1_tp2_use_tp_boll15_long(self):
        """LONG: TP1=TP_BOLL15 middle, TP2=structure BOLL20 upper.

        Three-Stage TP2 changed by design:
        TP2 is the structural confirmation gate before Trend Runner,
        therefore it uses structure BOLL20 outer by default.
        TP_BOLL15 outer remains available only when THREE_STAGE_TP2_USE_STRUCTURE_BOLL=false.
        """
        s = _strategy(three_stage_runner_enabled=True, tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(middle=100.0, tp_middle=101.0, upper=110.0, tp_upper=108.0)
        _setup_position_state(s, "LONG", 100.0)

        s._set_three_stage_runner_planned("LONG", b)
        assert s.state.three_stage_tp1_price == 101.0
        # TP2 uses structure BOLL20 outer (110.0), not TP_BOLL15 outer (108.0)
        assert s.state.three_stage_tp2_price == 110.0
        assert s.state.three_stage_tp2_price != 108.0

    def test_three_stage_initial_tp1_tp2_use_tp_boll15_short(self):
        """SHORT: TP1=TP_BOLL15 middle, TP2=structure BOLL20 lower.

        Three-Stage TP2 changed by design:
        TP2 is the structural confirmation gate before Trend Runner,
        therefore it uses structure BOLL20 outer by default.
        TP_BOLL15 outer remains available only when THREE_STAGE_TP2_USE_STRUCTURE_BOLL=false.
        """
        s = _strategy(three_stage_runner_enabled=True, tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(middle=100.0, tp_middle=99.0, lower=90.0, tp_lower=92.0)
        _setup_position_state(s, "SHORT", 100.0)

        s._set_three_stage_runner_planned("SHORT", b)
        assert s.state.three_stage_tp1_price == 99.0
        # TP2 uses structure BOLL20 lower (90.0), not TP_BOLL15 lower (92.0)
        assert s.state.three_stage_tp2_price == 90.0
        assert s.state.three_stage_tp2_price != 92.0

    def test_three_stage_update_tp_uses_tp_boll15(self):
        """_update_three_stage_dynamic_targets_without_reset keeps TP1 TP_BOLL/profit
        fallback and uses structure BOLL20 for TP2.

        Three-Stage TP2 changed by design:
        TP2 is the structural confirmation gate before Trend Runner,
        therefore it uses structure BOLL20 outer by default.
        TP_BOLL15 outer remains available only when THREE_STAGE_TP2_USE_STRUCTURE_BOLL=false.
        """
        s = _strategy(three_stage_runner_enabled=True, tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(middle=100.0, tp_middle=101.0, upper=110.0, tp_upper=108.0)
        _setup_position_state(s, "LONG", 100.0)
        s.state.side = "LONG"
        s.state.three_stage_runner_enabled_for_position = True

        updated = s._update_three_stage_dynamic_targets_without_reset("LONG", b)
        assert updated is True
        assert s.state.three_stage_tp1_price == 101.0
        # TP2 uses structure BOLL20 outer (110.0), not TP_BOLL15 outer (108.0)
        assert s.state.three_stage_tp2_price == 110.0
        assert s.state.three_stage_tp2_price != 108.0

    def test_three_stage_tp_boll_unavailable_fallback_long(self):
        """When TP_BOLL not available, Three-Stage falls back to structure BOLL20."""
        s = _strategy(three_stage_runner_enabled=True, tp_min_net_profit_pct=0.005)
        b = _boll_structure_20(middle=100.0, upper=110.0)
        _setup_position_state(s, "LONG", 99.0)

        s._set_three_stage_runner_planned("LONG", b)
        assert s.state.three_stage_tp1_price == 100.0
        assert s.state.three_stage_tp2_price == 110.0

    def test_three_stage_tp_boll_unavailable_fallback_short(self):
        """SHORT: TP_BOLL unavailable → structure BOLL20."""
        s = _strategy(three_stage_runner_enabled=True, tp_min_net_profit_pct=0.005)
        b = _boll_structure_20(middle=100.0, lower=90.0)
        _setup_position_state(s, "SHORT", 101.0)

        s._set_three_stage_runner_planned("SHORT", b)
        assert s.state.three_stage_tp1_price == 100.0
        assert s.state.three_stage_tp2_price == 90.0

    def test_three_stage_plan_skipped_when_effective_breakeven_missing(self):
        """Without effective breakeven, Three-Stage plan must be skipped → SINGLE outer."""
        s = _strategy(three_stage_runner_enabled=True)
        s.state.side = "LONG"
        # Deliberately do NOT set avg_entry_price / net_remaining_breakeven_price

        b = _boll_with_tp(middle=100.0, tp_middle=101.0, upper=110.0, tp_upper=108.0)

        s._set_three_stage_runner_planned("LONG", b)

        assert s.state.three_stage_tp1_price is None, (
            "TP1 must be None when effective breakeven is missing"
        )
        assert s.state.three_stage_tp2_price is None, (
            "TP2 must be None when effective breakeven is missing"
        )
        assert s.state.tp_plan == "SINGLE", (
            "Must fall back to SINGLE when breakeven is missing"
        )
        assert s.state.tp_price == 108.0, (
            "SINGLE TP must be TP_BOLL15 outer (108.0)"
        )
        assert s.state.three_stage_pre_tp1_degrade_stage == "SINGLE", (
            "Degrade stage must be locked to SINGLE"
        )

    def test_three_stage_waiting_tp2_outer_price_uses_tp_boll15(self):
        """Waiting TP2: outer price should come from TP_BOLL15."""
        s = _strategy(three_stage_runner_enabled=True)
        b = _boll_with_tp(upper=110.0, tp_upper=108.0)
        s.state.side = "LONG"

        tp2_price, src = s._select_tp_outer("LONG", b)
        assert tp2_price == 108.0
        assert src == "TP_BOLL"

    def test_three_stage_profit_fallback_verified(self):
        """TP_BOLL15 middle doesn't meet profit, BOLL20 middle does → fallback to BOLL20 middle."""
        s = _strategy(three_stage_runner_enabled=True, tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(
            middle=101.0,  # structure middle = 101 → ~1% profit from 100
            tp_middle=100.3,  # TP_BOLL15 middle = 100.3 → ~0.3% profit, below 0.5%
        )
        s.state.side = "LONG"
        s.state.layers = 2
        s.state.avg_entry_price = 100.0
        s.state.total_entry_qty = 1.0
        s.state.total_entry_notional = 100.0
        s.state.position_cost_entry_notional = 100.0
        s.state.position_cost_remaining_qty = 1.0
        s._refresh_net_remaining_breakeven_price()

        price, mode = s._select_tp_price("LONG", b)
        # TP_BOLL15 middle (100.3) fails → BOLL20 middle (101.0) passes → use it
        assert price == 101.0
        assert mode == "MIDDLE"


# ═══════════════════════════════════════════════════════════════════════
# 6. Runner / Sidecar isolation tests
# ═══════════════════════════════════════════════════════════════════════

class TestRunnerNotAffectedByTpBoll:
    def test_trend_runner_still_uses_structure_boll20(self):
        """Trend Runner TP/SL calculations must still use structure BOLL20."""
        s = _strategy()
        b = _boll_with_tp(upper=110.0, tp_upper=108.0, middle=100.0, tp_middle=101.0)

        # _calculate_runner_initial_tp uses boll.upper directly
        tp = s._calculate_runner_initial_tp("LONG", b)
        # Should use BOLL20 upper (110) not TP_BOLL15 upper (108)
        expected = 110.0 * (1 + s.config.runner_tp_initial_outer_extra_pct)
        assert tp == expected

    def test_trend_runner_dynamic_orders_use_structure_boll20(self):
        """Dynamic runner orders must use structure BOLL20."""
        s = _strategy()
        b = _boll_with_tp(upper=110.0, tp_upper=108.0, middle=100.0, tp_middle=101.0)

        tp, sl, extra, dist = s._calculate_trend_runner_dynamic_orders("LONG", b, 0, None)
        # TP should be based on boll.upper (110), not tp_upper (108)
        expected_tp = 110.0 * (
                1 + max(s.config.runner_tp_min_outer_extra_pct, s.config.runner_tp_initial_outer_extra_pct))
        assert tp == expected_tp


class TestSidecarNotAffectedByTpBoll:
    def test_sidecar_fixed_tp_unchanged_by_tp_boll(self):
        """Sidecar fixed TP is entry_price based, unaffected by BOLL."""
        entry_price = 100.0
        sidecar_tp_pct = 0.004
        # Sidecar TP = entry_price * (1 + tp_pct)
        tp_long = entry_price * (1 + sidecar_tp_pct)
        tp_short = entry_price * (1 - sidecar_tp_pct)
        assert tp_long == 100.4
        assert tp_short == 99.6
        # These are purely entry-price based, independent of BOLL


# ═══════════════════════════════════════════════════════════════════════
# 7. Monitor config test
# ═══════════════════════════════════════════════════════════════════════

class TestMonitorConfigTpBoll:
    def test_monitor_config_tp_boll_defaults(self):
        cfg = BollBandBreakoutMonitorConfig()
        assert cfg.tp_boll_enabled is True
        assert cfg.tp_boll_window == 15
        assert cfg.boll_window == 20  # structure BOLL unchanged

    def test_monitor_config_from_env_with_tp_boll(self):
        with mock.patch.dict(os.environ, {
            "TP_BOLL_ENABLED": "false",
            "TP_BOLL_WINDOW": "10",
            "BOLL_WINDOW": "20",
        }):
            cfg = BollBandBreakoutMonitorConfig.from_env()
            assert cfg.tp_boll_enabled is False
            assert cfg.tp_boll_window == 10
            assert cfg.boll_window == 20  # structure BOLL unchanged

    def test_monitor_config_tp_boll_disabled_when_window_zero(self):
        """TP_BOLL_WINDOW <= 0 means effectively disabled (will not compute)."""
        cfg = BollBandBreakoutMonitorConfig(tp_boll_enabled=True, tp_boll_window=0)
        # The monitor will skip computing TP BOLL when window <= 0
        assert cfg.tp_boll_window == 0


# ═══════════════════════════════════════════════════════════════════════
# 8. Strategy config tests
# ═══════════════════════════════════════════════════════════════════════

class TestStrategyConfigTpBoll:
    def test_strategy_config_tp_boll_defaults(self):
        cfg = BollCvdReclaimStrategyConfig()
        assert cfg.tp_boll_enabled is True
        assert cfg.tp_boll_window == 15

    def test_strategy_config_from_env_with_tp_boll(self):
        with mock.patch.dict(os.environ, {
            "TP_BOLL_ENABLED": "false",
            "TP_BOLL_WINDOW": "10",
        }):
            cfg = BollCvdReclaimStrategyConfig.from_env()
            assert cfg.tp_boll_enabled is False
            assert cfg.tp_boll_window == 10

    def test_strategy_config_tp_boll_disabled(self):
        cfg = BollCvdReclaimStrategyConfig(tp_boll_enabled=False)
        assert cfg.tp_boll_enabled is False
        # All logic should fall back to structure BOLL
        s = BollCvdReclaimStrategy(cfg, _sizer())
        b = _boll_with_tp()
        assert s._tp_boll_available(b) is False


# ═══════════════════════════════════════════════════════════════════════
# 9. Entry path tests
# ═══════════════════════════════════════════════════════════════════════

class TestEntryPathTpBoll:
    def test_open_position_long_selects_tp_boll_middle(self):
        """_open_position LONG uses TP_BOLL15 middle for SINGLE TP."""
        s = _strategy(tp_min_net_profit_pct=0.001)
        b = _boll_with_tp(middle=100.0, tp_middle=101.0, upper=110.0, tp_upper=108.0)
        s.state.lower_armed = True
        s.state.lower_deep_enough = True
        s.state.lower_extreme_price = 99.0

        intent = s._open_position("LONG", "OPEN_LONG", 99.5, 1000, b, _cvd(), "test")
        # With tp_min_net_profit_pct=0.001, middle should pass
        # TP_BOLL15 middle (101) >= be * 1.001 → use 101
        assert intent.tp_price == 101.0

    def test_open_position_short_selects_tp_boll_middle(self):
        """_open_position SHORT uses TP_BOLL15 middle for SINGLE TP."""
        s = _strategy(tp_min_net_profit_pct=0.001)
        b = _boll_with_tp(middle=100.0, tp_middle=99.0, lower=90.0, tp_lower=92.0)
        s.state.upper_armed = True
        s.state.upper_deep_enough = True
        s.state.upper_extreme_price = 101.0

        intent = s._open_position("SHORT", "OPEN_SHORT", 100.5, 1000, b, _cvd(), "test")
        assert intent.tp_price == 99.0


# ═══════════════════════════════════════════════════════════════════════
# 10. Degrade path test
# ═══════════════════════════════════════════════════════════════════════

class TestDegradeTpBoll:
    def test_degrade_to_middle_runner_uses_tp_boll15(self):
        """Pre-TP1 degrade to Middle Runner uses TP_BOLL15 prices."""
        s = _strategy(three_stage_runner_enabled=True, tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(middle=100.0, tp_middle=101.0, upper=110.0, tp_upper=108.0)
        _setup_position_state(s, "LONG", 100.0)
        s.state.side = "LONG"
        s.state.three_stage_runner_enabled_for_position = True
        s.state.tp_plan = "THREE_STAGE_RUNNER"
        s.state.first_entry_ts_ms = 1000

        s._degrade_three_stage_pre_tp1_to_middle_runner(1000 + 10801_000, b)

        assert s.state.middle_runner_first_tp_price == 101.0
        assert s.state.middle_runner_final_tp_price == 108.0
        assert s.state.tp_plan == "MIDDLE_RUNNER"


# ═══════════════════════════════════════════════════════════════════════
# 11. Structural BOLL unchanged tests
# ═══════════════════════════════════════════════════════════════════════

class TestStructuralBollUnchanged:
    def test_armed_state_still_uses_structure_boll(self):
        """_update_armed_state continues using structure BOLL lower/upper/middle."""
        s = _strategy()
        b = _boll_with_tp(lower=90.0, tp_lower=92.0, upper=110.0, tp_upper=108.0, middle=100.0, tp_middle=101.0)

        s._update_armed_state(89.0, 1000, b)
        # Price 89 is below structure lower (90), not TP lower (92)
        # If TP_BOLL leaked, 89 would be above tp_lower=92 (no arm)
        assert s.state.lower_armed is True, "Armed state must use structure BOLL lower, not TP BOLL lower"

    def test_deep_enough_uses_structure_boll(self):
        """Deep-enough check uses structure BOLL, not TP BOLL."""
        s = _strategy()
        b = _boll_with_tp(lower=90.0, tp_lower=92.0)

        # threshold = 90 * (1 - 0.001) = 89.91
        # extreme=89.95 > 89.91 → NOT deep enough
        s.state.lower_armed = True
        s.state.lower_extreme_price = 89.95
        s._update_lower_deep_enough(b)
        assert s.state.lower_deep_enough is False

        # Reset and test deep enough
        s.state.lower_deep_enough = False
        s.state.lower_extreme_price = 89.0  # 89.0 <= 89.91 → deep enough
        s._update_lower_deep_enough(b)
        assert s.state.lower_deep_enough is True


# ═══════════════════════════════════════════════════════════════════════
# 12. Profit fallback tests (TP_BOLL15 middle profit insufficient)
# ═══════════════════════════════════════════════════════════════════════

def _setup_position_state(s: BollCvdReclaimStrategy, side: str, avg_entry: float) -> None:
    """Configure minimal position state for _effective_breakeven_for_tp_selection."""
    s.state.side = side
    s.state.layers = 2
    s.state.avg_entry_price = avg_entry
    s.state.total_entry_qty = 1.0
    s.state.total_entry_notional = avg_entry
    s.state.position_cost_entry_notional = avg_entry
    s.state.position_cost_remaining_qty = 1.0
    s._refresh_net_remaining_breakeven_price()


class TestThreeStageTp1ProfitFallback:
    """Verify Three-Stage TP1 uses BOLL20 middle when TP_BOLL15 middle profit is insufficient."""

    def test_tp1_uses_structure_middle_when_tp_boll_middle_profit_too_small_long(self):
        """LONG: TP_BOLL15 middle=100.3 (< 0.5% profit), BOLL20 middle=101.0 (OK) → TP1=101.0."""
        s = _strategy(three_stage_runner_enabled=True, tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(
            middle=101.0, tp_middle=100.3,
            upper=110.0, tp_upper=108.0,
        )
        _setup_position_state(s, "LONG", 100.0)

        # _select_tp_price should return BOLL20 middle after profit fallback
        price, mode = s._select_tp_price("LONG", b)
        assert price == 101.0
        assert mode == "MIDDLE"

        # _set_three_stage_runner_planned should also use BOLL20 middle for TP1
        s._set_three_stage_runner_planned("LONG", b)
        assert s.state.three_stage_tp1_price == 101.0, (
            "TP1 must use BOLL20 middle (101.0) when TP_BOLL15 middle (100.3) "
            "fails the profit check"
        )
        # TP2 uses structure BOLL20 outer (110.0), not TP_BOLL15 outer (108.0)
        # Three-Stage TP2 changed by design: TP2 is the structural confirmation
        # gate before Trend Runner, therefore it uses structure BOLL20 outer.
        assert s.state.three_stage_tp2_price == 110.0, (
            "TP2 should use structure BOLL20 outer (110.0)"
        )
        assert s.state.three_stage_tp2_price != 108.0

    def test_tp1_uses_structure_middle_when_tp_boll_middle_profit_too_small_short(self):
        """SHORT: TP_BOLL15 middle=99.7 (> 0.5% profit), BOLL20 middle=99.0 (OK) → TP1=99.0."""
        s = _strategy(three_stage_runner_enabled=True, tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(
            middle=99.0, tp_middle=99.7,
            lower=90.0, tp_lower=92.0,
        )
        _setup_position_state(s, "SHORT", 100.0)

        price, mode = s._select_tp_price("SHORT", b)
        assert price == 99.0
        assert mode == "MIDDLE"

        s._set_three_stage_runner_planned("SHORT", b)
        assert s.state.three_stage_tp1_price == 99.0
        # TP2 uses structure BOLL20 lower (90.0), not TP_BOLL15 lower (92.0)
        assert s.state.three_stage_tp2_price == 90.0
        assert s.state.three_stage_tp2_price != 92.0

    def test_tp1_uses_tp_boll_when_profit_ok_long(self):
        """LONG: TP_BOLL15 middle=101.0 (>= 0.5% profit) → TP1=101.0 (no fallback)."""
        s = _strategy(three_stage_runner_enabled=True, tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(
            middle=101.0, tp_middle=101.0,
            upper=110.0, tp_upper=108.0,
        )
        _setup_position_state(s, "LONG", 100.0)

        s._set_three_stage_runner_planned("LONG", b)
        assert s.state.three_stage_tp1_price == 101.0, "TP_BOLL15 middle meets profit → use it"
        # TP2 uses structure BOLL20 outer (110.0), not TP_BOLL15 outer (108.0)
        assert s.state.three_stage_tp2_price == 110.0
        assert s.state.three_stage_tp2_price != 108.0

    def test_update_dynamic_targets_keeps_profit_fallback(self):
        """_update_three_stage_dynamic_targets_without_reset honours profit fallback."""
        s = _strategy(three_stage_runner_enabled=True, tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(
            middle=101.0, tp_middle=100.3,
            upper=110.0, tp_upper=108.0,
        )
        _setup_position_state(s, "LONG", 100.0)

        s._update_three_stage_dynamic_targets_without_reset("LONG", b)
        assert s.state.three_stage_tp1_price == 101.0, "profit fallback applies to dynamic update"
        # TP2 uses structure BOLL20 outer (110.0), not TP_BOLL15 outer (108.0)
        assert s.state.three_stage_tp2_price == 110.0
        assert s.state.three_stage_tp2_price != 108.0


class TestMiddleRunnerFirstTpProfitFallback:
    """Verify Middle Runner first TP uses BOLL20 middle when TP_BOLL15 profit insufficient."""

    def test_first_tp_uses_structure_middle_when_tp_boll_profit_too_small_long(self):
        """LONG: TP_BOLL15 middle=100.3, BOLL20 middle=101.0 → first TP=101.0."""
        s = _strategy(middle_runner_enabled=True, tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(
            middle=101.0, tp_middle=100.3,
            upper=110.0, tp_upper=108.0,
        )
        _setup_position_state(s, "LONG", 100.0)

        tp_mid, src = s._select_valid_tp_middle_with_profit_fallback("LONG", b)
        assert tp_mid == 101.0
        assert src == "STRUCTURE_BOLL_PROFIT_FALLBACK"

    def test_first_tp_uses_structure_middle_when_tp_boll_profit_too_small_short(self):
        """SHORT: TP_BOLL15 middle=99.7, BOLL20 middle=99.0 → first TP=99.0."""
        s = _strategy(middle_runner_enabled=True, tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(
            middle=99.0, tp_middle=99.7,
            lower=90.0, tp_lower=92.0,
        )
        _setup_position_state(s, "SHORT", 100.0)

        tp_mid, src = s._select_valid_tp_middle_with_profit_fallback("SHORT", b)
        assert tp_mid == 99.0
        assert src == "STRUCTURE_BOLL_PROFIT_FALLBACK"

    def test_first_tp_uses_tp_boll_when_profit_ok_long(self):
        """TP_BOLL15 middle=101.5 → enough profit → use it."""
        s = _strategy(middle_runner_enabled=True, tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(
            middle=101.0, tp_middle=101.5,
            upper=110.0, tp_upper=108.0,
        )
        _setup_position_state(s, "LONG", 100.0)

        tp_mid, src = s._select_valid_tp_middle_with_profit_fallback("LONG", b)
        assert tp_mid == 101.5
        assert src == "TP_BOLL"

    def test_planned_sets_fallback_prices_long(self):
        """_set_middle_runner_planned with profit fallback."""
        s = _strategy(middle_runner_enabled=True, tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(
            middle=101.0, tp_middle=100.3,
            upper=110.0, tp_upper=108.0,
        )
        _setup_position_state(s, "LONG", 100.0)
        s.state.side = "LONG"

        # Simulate what _select_tp_plan returns with fallback
        tp_mid, _ = s._select_valid_tp_middle_with_profit_fallback("LONG", b)
        tp_outer, _ = s._select_tp_outer("LONG", b)
        s._set_middle_runner_planned(tp_mid, tp_outer)

        assert s.state.middle_runner_first_tp_price == 101.0
        assert s.state.middle_runner_final_tp_price == 108.0


class TestDegradeProfitFallback:
    """Verify degrade to Middle Runner uses profit fallback for first TP."""

    def test_degrade_uses_structure_middle_when_tp_boll_profit_too_small(self):
        """Three-Stage degrade → Middle Runner: BOLL20 middle when TP_BOLL15 insufficient."""
        s = _strategy(
            three_stage_runner_enabled=True,
            tp_min_net_profit_pct=0.005,
            three_stage_pre_tp1_degrade_enabled=False,  # prevent auto-degrade
        )
        b = _boll_with_tp(
            middle=101.0, tp_middle=100.3,
            upper=110.0, tp_upper=108.0,
        )
        _setup_position_state(s, "LONG", 100.0)
        s.state.side = "LONG"
        s.state.three_stage_runner_enabled_for_position = True
        s.state.tp_plan = "THREE_STAGE_RUNNER"
        s.state.first_entry_ts_ms = 1000

        s._degrade_three_stage_pre_tp1_to_middle_runner(2000000, b)

        assert s.state.middle_runner_first_tp_price == 101.0, (
            "Degrade first TP must fall back to BOLL20 middle when TP_BOLL15 profit insufficient"
        )
        assert s.state.middle_runner_final_tp_price == 108.0
        assert s.state.tp_plan == "MIDDLE_RUNNER"


class TestUpdateTpProfitFallback:
    """Verify 15m UPDATE_TP keeps profit fallback for TP1/first TP."""

    def test_three_stage_update_keeps_profit_fallback_for_tp1(self):
        """_update_three_stage_dynamic_targets_without_reset honours fallback during update."""
        s = _strategy(three_stage_runner_enabled=True, tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(
            middle=101.0, tp_middle=100.3,
            upper=110.0, tp_upper=108.0,
        )
        _setup_position_state(s, "LONG", 100.0)
        s.state.side = "LONG"
        s.state.three_stage_runner_enabled_for_position = True

        # This is what _maybe_update_tp calls for three_stage pre-TP1
        s._update_three_stage_dynamic_targets_without_reset("LONG", b)

        assert s.state.three_stage_tp1_price == 101.0
        # TP2 uses structure BOLL20 outer (110.0), not TP_BOLL15 outer (108.0)
        assert s.state.three_stage_tp2_price == 110.0
        assert s.state.three_stage_tp2_price != 108.0

    def test_middle_runner_pending_update_keeps_profit_fallback_for_first_tp(self):
        """_select_valid_tp_middle_with_profit_fallback used in middle_runner_pending path."""
        s = _strategy(middle_runner_enabled=True, tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(
            middle=101.0, tp_middle=100.3,
            upper=110.0, tp_upper=108.0,
        )
        _setup_position_state(s, "LONG", 100.0)
        s.state.side = "LONG"

        tp_mid, src = s._select_valid_tp_middle_with_profit_fallback("LONG", b)
        assert tp_mid == 101.0
        assert src == "STRUCTURE_BOLL_PROFIT_FALLBACK"

    def test_select_tp_plan_three_stage_uses_fallback(self):
        """_select_tp_plan THREE_STAGE path uses profit fallback."""
        s = _strategy(three_stage_runner_enabled=True, tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(
            middle=101.0, tp_middle=100.3,
            upper=110.0, tp_upper=108.0,
        )
        _setup_position_state(s, "LONG", 100.0)
        s.state.side = "LONG"
        s.state.breakeven_price = s.state.avg_entry_price * (1 + s.config.breakeven_fee_buffer_pct)

        partial_tp, ratio, plan = s._select_tp_plan("LONG", 110.0, 5, tp_mode="MIDDLE", boll=b)
        assert plan == "THREE_STAGE_RUNNER"
        # partial TP (TP1) should be BOLL20 middle (101.0) due to profit fallback
        assert partial_tp == 101.0, (
            f"Expected TP1=101.0 (BOLL20 fallback), got {partial_tp}"
        )

    def test_select_tp_plan_middle_runner_uses_fallback(self):
        """_select_tp_plan MIDDLE_RUNNER path uses profit fallback."""
        s = _strategy(middle_runner_enabled=True, tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(
            middle=101.0, tp_middle=100.3,
            upper=110.0, tp_upper=108.0,
        )
        _setup_position_state(s, "LONG", 100.0)
        s.state.side = "LONG"
        s.state.breakeven_price = s.state.avg_entry_price * (1 + s.config.breakeven_fee_buffer_pct)

        partial_tp, ratio, plan = s._select_tp_plan("LONG", 110.0, 5, tp_mode="MIDDLE", boll=b)
        assert plan == "MIDDLE_RUNNER"
        assert partial_tp == 101.0, f"Expected first TP=101.0 (BOLL20 fallback), got {partial_tp}"


class TestTp2OuterStillUsesTpBoll:
    """Verify TP2 outer price selection behaviour.

    Three-Stage TP2 changed by design: TP2 is the structural confirmation
    gate before Trend Runner, therefore it uses structure BOLL20 outer by default.
    TP_BOLL15 outer remains available only when THREE_STAGE_TP2_USE_STRUCTURE_BOLL=false.

    These tests verify the _select_tp_outer resolver (which still prefers TP_BOLL15),
    used by Middle Runner and SINGLE TP paths, separately from Three-Stage TP2 selection.
    """

    def test_tp2_still_uses_tp_boll_outer_when_tp1_falls_back_long(self):
        """TP1 falls back to BOLL20, TP2 still uses TP_BOLL15 outer."""
        s = _strategy(three_stage_runner_enabled=True, tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(
            middle=101.0, tp_middle=100.3,
            upper=110.0, tp_upper=108.0,
        )
        _setup_position_state(s, "LONG", 100.0)

        tp_outer, src = s._select_tp_outer("LONG", b)
        assert tp_outer == 108.0
        assert src == "TP_BOLL"


class TestInvalidMiddleProfitSafety:
    def test_valid_middle_helper_returns_none_when_tp_and_structure_middle_profit_insufficient_short(self):
        s = _strategy(tp_min_net_profit_pct=0.003)
        s.state = StrategyPositionState(
            side="SHORT",
            layers=1,
            avg_entry_price=1579.7668,
            net_remaining_breakeven_price=1578.2440,
        )
        b = _boll_with_tp(
            middle=1576.1145,
            lower=1556.0,
            tp_middle=1581.3680,
            tp_lower=1558.3630,
        )

        price, source = s._select_valid_tp_middle_with_profit_fallback("SHORT", b)

        assert price is None
        assert source == "MIDDLE_PROFIT_INSUFFICIENT"

    def test_degrade_to_middle_runner_skipped_when_middle_profit_insufficient_short(self):
        s = _strategy(
            three_stage_runner_enabled=True,
            tp_min_net_profit_pct=0.003,
        )
        b = _boll_with_tp(
            middle=1576.1145,
            lower=1556.0,
            tp_middle=1581.3680,
            tp_lower=1558.3630,
        )
        s.state = StrategyPositionState(
            side="SHORT",
            layers=1,
            first_entry_ts_ms=1000,
            last_order_ts_ms=1000,
            avg_entry_price=1579.7668,
            total_entry_qty=1.0,
            total_entry_notional=1579.7668,
            net_remaining_breakeven_price=1578.2440,
            tp_price=1558.3630,
            tp_mode="LOWER",
            tp_plan="THREE_STAGE_RUNNER",
            partial_tp_price=1576.1145,
            partial_tp_ratio=0.6,
            three_stage_runner_enabled_for_position=True,
            three_stage_tp1_price=1576.1145,
            three_stage_tp2_price=1558.3630,
            three_stage_tp1_ratio=0.6,
            three_stage_tp2_ratio=0.2,
            three_stage_runner_ratio=0.2,
            three_stage_tp1_consumed=False,
            three_stage_tp2_consumed=False,
        )

        s._degrade_three_stage_pre_tp1_to_middle_runner(1000 + 10_801_000, b)

        assert s.state.tp_plan == "SINGLE"
        assert s.state.tp_price == 1558.3630
        assert s.state.tp_mode == "LOWER"
        assert s.state.partial_tp_price is None
        assert s.state.middle_runner_pending is False
        assert s.state.middle_runner_active is False
        assert s.state.three_stage_runner_enabled_for_position is False
        assert s.state.three_stage_pre_tp1_degrade_stage == "SINGLE"
        assert s.state.middle_runner_first_tp_price is None

    def test_middle_profit_insufficient_single_lock_prevents_later_middle_runner_degrade(self):
        s = _strategy(
            three_stage_runner_enabled=True,
            tp_min_net_profit_pct=0.003,
            three_stage_pre_tp1_middle_runner_after_seconds=10_800,
            three_stage_pre_tp1_single_after_seconds=21_600,
        )
        b = _boll_with_tp(
            middle=1576.1145,
            lower=1556.0,
            tp_middle=1581.3680,
            tp_lower=1558.3630,
        )
        first_ts = 1000
        s.state = StrategyPositionState(
            side="SHORT",
            layers=1,
            first_entry_ts_ms=first_ts,
            last_order_ts_ms=first_ts,
            avg_entry_price=1579.7668,
            total_entry_qty=1.0,
            total_entry_notional=1579.7668,
            net_remaining_breakeven_price=1578.2440,
            tp_price=1558.3630,
            tp_mode="LOWER",
            tp_plan="THREE_STAGE_RUNNER",
            partial_tp_price=1576.1145,
            partial_tp_ratio=0.6,
            three_stage_runner_enabled_for_position=True,
            three_stage_tp1_price=1576.1145,
            three_stage_tp2_price=1558.3630,
            three_stage_tp1_ratio=0.6,
            three_stage_tp2_ratio=0.2,
            three_stage_runner_ratio=0.2,
            last_tp_update_candle_ts_ms=0,
        )

        first = s._maybe_update_tp(1579.0, first_ts + 60_000, b, _cvd())
        assert first is not None
        assert s.state.tp_plan == "SINGLE"
        assert s.state.three_stage_pre_tp1_degrade_stage == "SINGLE"
        assert s.state.three_stage_runner_enabled_for_position is False

        b2 = replace(b, candle_ts_ms=2_000)
        s._maybe_update_tp(1579.0, first_ts + 10_901_000, b2, _cvd())
        assert s.state.tp_plan == "SINGLE"
        assert s.state.middle_runner_pending is False
        assert s.state.partial_tp_price is None
        assert s.state.tp_price == 1558.3630

    def test_middle_runner_pending_invalid_middle_falls_back_single_outer_short(self):
        s = _strategy(middle_runner_enabled=True, tp_min_net_profit_pct=0.003)
        b = _boll_with_tp(
            middle=1576.1145,
            lower=1556.0,
            tp_middle=1581.3680,
            tp_lower=1558.3630,
        )
        s.state = StrategyPositionState(
            side="SHORT",
            layers=1,
            avg_entry_price=1579.7668,
            total_entry_qty=1.0,
            total_entry_notional=1579.7668,
            net_remaining_breakeven_price=1578.2440,
            tp_price=1558.3630,
            tp_mode="LOWER",
            tp_plan="MIDDLE_RUNNER",
            partial_tp_price=1581.3680,
            partial_tp_ratio=0.8,
            middle_runner_enabled_for_position=True,
            middle_runner_pending=True,
            middle_runner_active=False,
            middle_runner_first_tp_price=1581.3680,
            middle_runner_final_tp_price=1558.3630,
            last_tp_update_candle_ts_ms=0,
        )

        got = s._maybe_update_tp(1579.0, 2_000, b, _cvd())

        assert got is not None
        assert s.state.tp_plan == "SINGLE"
        assert s.state.tp_price == 1558.3630
        assert s.state.partial_tp_price is None
        assert s.state.middle_runner_pending is False

    def test_invalid_middle_profit_fallback_long(self):
        s = _strategy(three_stage_runner_enabled=True, tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(
            middle=100.4,
            upper=110.0,
            tp_middle=100.3,
            tp_upper=108.0,
        )
        s.state = StrategyPositionState(
            side="LONG",
            layers=1,
            avg_entry_price=100.0,
            total_entry_qty=1.0,
            total_entry_notional=100.0,
            net_remaining_breakeven_price=100.0,
            tp_price=108.0,
            tp_mode="UPPER",
            tp_plan="THREE_STAGE_RUNNER",
            partial_tp_price=100.4,
            partial_tp_ratio=0.6,
            three_stage_runner_enabled_for_position=True,
            three_stage_tp1_price=100.4,
            three_stage_tp2_price=108.0,
            three_stage_tp1_ratio=0.6,
            three_stage_tp2_ratio=0.2,
            three_stage_runner_ratio=0.2,
            last_tp_update_candle_ts_ms=0,
        )

        got = s._maybe_update_tp(100.0, 2_000, b, _cvd())

        assert got is not None
        assert s.state.tp_plan == "SINGLE"
        assert s.state.tp_price == 108.0
        assert s.state.partial_tp_price is None
        assert s.state.three_stage_runner_enabled_for_position is False

    def test_valid_tp_boll_middle_still_enables_three_stage(self):
        s = _strategy(three_stage_runner_enabled=True, tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(
            middle=100.8,
            upper=110.0,
            tp_middle=101.0,
            tp_upper=108.0,
        )
        _setup_position_state(s, "LONG", 100.0)

        s._set_three_stage_runner_planned("LONG", b)

        assert s.state.three_stage_tp1_price == 101.0
        # TP2 uses structure BOLL20 outer (110.0), not TP_BOLL15 outer (108.0)
        assert s.state.three_stage_tp2_price == 110.0
        assert s.state.three_stage_tp2_price != 108.0

    def test_tp_boll_middle_invalid_but_structure_middle_valid_uses_structure_middle(self):
        s = _strategy(three_stage_runner_enabled=True, tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(
            middle=101.0,
            upper=110.0,
            tp_middle=100.3,
            tp_upper=108.0,
        )
        _setup_position_state(s, "LONG", 100.0)

        s._set_three_stage_runner_planned("LONG", b)

        assert s.state.three_stage_tp1_price == 101.0
        # TP2 uses structure BOLL20 outer (110.0), not TP_BOLL15 outer (108.0)
        assert s.state.three_stage_tp2_price == 110.0
        assert s.state.three_stage_tp2_price != 108.0


class TestTp2OuterStillUsesTpBollShortAndFinal:
    def test_tp2_still_uses_tp_boll_outer_when_tp1_falls_back_short(self):
        """SHORT: TP1 falls back, TP2 still uses TP_BOLL15 lower."""
        s = _strategy(three_stage_runner_enabled=True, tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(
            middle=99.0, tp_middle=99.7,
            lower=90.0, tp_lower=92.0,
        )
        _setup_position_state(s, "SHORT", 100.0)

        tp_outer, src = s._select_tp_outer("SHORT", b)
        assert tp_outer == 92.0
        assert src == "TP_BOLL"

    def test_final_tp_still_uses_tp_boll_outer_when_first_tp_falls_back(self):
        """Middle Runner: first TP falls back, final outer still TP_BOLL15."""
        s = _strategy(middle_runner_enabled=True, tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(
            middle=101.0, tp_middle=100.3,
            upper=110.0, tp_upper=108.0,
        )
        _setup_position_state(s, "LONG", 100.0)

        tp_outer, src = s._select_tp_outer("LONG", b)
        assert tp_outer == 108.0
        assert src == "TP_BOLL"


# ═══════════════════════════════════════════════════════════════════════
# 13. Outer profit fallback tests
# ═══════════════════════════════════════════════════════════════════════

class TestOuterProfitFallback:
    """Verify outer TP profit-distance fallback: TP_BOLL15 → BOLL20 → last resort."""

    def test_long_tp_boll_outer_insufficient_structure_outer_fallback(self):
        """LONG: TP_BOLL15 upper insufficient profit, BOLL20 upper sufficient → fallback."""
        s = _strategy(tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(
            middle=100.0, tp_middle=100.8,
            upper=110.0, tp_upper=104.5,
        )
        _setup_position_state(s, "LONG", 104.0)
        # effective_be ~ 104.0, required_outer = 104.0 * 1.005 = 104.52
        # tp_upper=104.5 < 104.52 → insufficient
        # upper=110.0 >= 104.52 → fallback to BOLL20 upper

        price, mode = s._select_tp_price("LONG", b)
        assert price == 110.0
        assert mode == "UPPER"

    def test_short_tp_boll_outer_insufficient_structure_outer_fallback(self):
        """SHORT: TP_BOLL15 lower insufficient, BOLL20 lower sufficient → fallback."""
        s = _strategy(tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(
            middle=99.3, tp_middle=99.5,
            lower=90.0, tp_lower=95.8,
        )
        _setup_position_state(s, "SHORT", 96.0)
        # effective_be ~ 96.0, required = 96.0 * 0.995 = 95.52
        # tp_middle=99.5 > 95.52 → insufficient middle
        # middle=99.3 > 95.52 → insufficient middle
        # tp_lower=95.8 > 95.52 → insufficient outer
        # lower=90.0 <= 95.52 → fallback to BOLL20 lower

        price, mode = s._select_tp_price("SHORT", b)
        assert price == 90.0
        assert mode == "LOWER"

    def test_long_both_outer_insufficient_warning_fallback(self):
        """LONG: both outers insufficient but NOT at loss → farther outer with WARNING source."""
        s = _strategy(tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(
            middle=107.5, tp_middle=107.6,
            upper=108.1, tp_upper=107.8,
        )
        _setup_position_state(s, "LONG", 107.5)
        # effective_be ~ 107.6075, required = 107.6075 * 1.005 = 108.1455
        # tp_upper=107.8 > 107.6075 (not at loss), but < 108.1455 (insufficient)
        # upper=108.1 > 107.6075 (not at loss), but < 108.1455 (insufficient)
        # farther = max(107.8, 108.1) = 108.1

        tp_outer, src = s._select_valid_tp_outer_with_profit_fallback("LONG", b)
        assert tp_outer == 108.1
        assert src == "TP_OUTER_PROFIT_INSUFFICIENT_FALLBACK"

    def test_short_both_outer_insufficient_warning_fallback(self):
        """SHORT: both outers insufficient → farther outer with WARNING source."""
        s = _strategy(tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(
            middle=93.5, tp_middle=94.0,
            lower=93.55, tp_lower=93.6,
        )
        _setup_position_state(s, "SHORT", 94.0)
        # effective_be ~ 94.0, required = 94.0 * 0.995 = 93.53
        # tp_lower=93.6 > 93.53 → insufficient
        # lower=93.55 > 93.53 → insufficient
        # farther = min(93.6, 93.55) = 93.55

        tp_outer, src = s._select_valid_tp_outer_with_profit_fallback("SHORT", b)
        assert tp_outer == 93.55
        assert src == "TP_OUTER_PROFIT_INSUFFICIENT_FALLBACK"

    def test_tp2_uses_outer_fallback_via_three_stage_planned_long(self):
        """Three-Stage TP2 uses profit-validated outer when TP1 passes profit check."""
        s = _strategy(three_stage_runner_enabled=True, tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(
            middle=106.5, tp_middle=106.5,
            upper=110.0, tp_upper=105.0,
        )
        _setup_position_state(s, "LONG", 104.5)
        # effective_be ~ 104.5, required_mid = 105.0225
        # tp_middle=106.5 >= 105.0225 → TP1 passes
        # required_outer = 105.0225
        # tp_upper=105.0 < 105.0225 → insufficient
        # upper=110.0 >= 105.0225 → fallback for TP2

        s._set_three_stage_runner_planned("LONG", b)
        assert s.state.three_stage_tp1_price == 106.5
        assert s.state.three_stage_tp2_price == 110.0, (
            "TP2 must fall back to BOLL20 upper when TP_BOLL15 upper insufficient"
        )

    def test_degrade_final_tp_uses_outer_fallback_long(self):
        """Degrade to Middle Runner: final TP uses profit-validated outer."""
        s = _strategy(
            three_stage_runner_enabled=True,
            tp_min_net_profit_pct=0.005,
            three_stage_pre_tp1_degrade_enabled=False,
        )
        b = _boll_with_tp(
            middle=106.5, tp_middle=106.5,
            upper=110.0, tp_upper=105.0,
        )
        _setup_position_state(s, "LONG", 104.5)
        s.state.side = "LONG"
        s.state.three_stage_runner_enabled_for_position = True
        s.state.tp_plan = "THREE_STAGE_RUNNER"
        s.state.first_entry_ts_ms = 1000

        s._degrade_three_stage_pre_tp1_to_middle_runner(2000000, b)

        assert s.state.tp_plan == "MIDDLE_RUNNER"
        assert s.state.middle_runner_first_tp_price == 106.5
        assert s.state.middle_runner_final_tp_price == 110.0, (
            "Final TP must fall back to BOLL20 upper when TP_BOLL15 upper insufficient"
        )

    def test_waiting_tp2_update_falls_back_to_structure_outer_when_tp_boll_outer_lacks_profit(self):
        """Waiting TP2: new TP2 via _maybe_update_tp uses BOLL20 upper when TP_BOLL15 insufficient."""
        s = _strategy(
            three_stage_runner_enabled=True,
            tp_min_net_profit_pct=0.005,
            three_stage_post_tp1_protective_sl_enabled=False,
        )
        b = _boll_with_tp(
            middle=106.5, tp_middle=106.5,
            upper=110.0, tp_upper=105.0,
        )
        _setup_position_state(s, "LONG", 104.5)
        s.state.side = "LONG"
        s.state.three_stage_runner_enabled_for_position = True
        s.state.three_stage_tp1_consumed = True
        s.state.three_stage_tp2_consumed = False
        s.state.trend_runner_active = False
        s.state.tp_plan = "THREE_STAGE_RUNNER"
        s.state.tp_mode = "UPPER"
        s.state.three_stage_tp1_price = 106.5
        s.state.three_stage_tp2_price = 105.0
        s.state.tp_price = 105.0
        s.state.last_tp_update_candle_ts_ms = 0
        s.state.partial_tp_consumed = True
        # effective_be ~ 104.5, required_outer = 104.5 * 1.005 = 105.0225
        # tp_upper=105.0 < 105.0225 → insufficient
        # upper=110.0 >= 105.0225 → fallback

        intent = s._maybe_update_tp(105.0, 2000, b, _cvd())
        assert intent is not None, "waiting TP2 should produce UPDATE_TP intent"
        assert intent.tp_price == 110.0, (
            "UPDATE_TP intent tp_price must be BOLL20 upper (110.0)"
        )
        assert s.state.three_stage_tp2_price == 110.0, (
            "three_stage_tp2_price must fall back to BOLL20 upper (110.0)"
        )
        assert s.state.tp_price == 110.0

    def test_middle_runner_active_final_tp_falls_back_to_structure_outer_when_tp_boll_outer_lacks_profit(self):
        """Middle Runner active: final TP update uses BOLL20 upper when TP_BOLL15 insufficient."""
        s = _strategy(middle_runner_enabled=True, tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(
            middle=106.5, tp_middle=106.5,
            upper=110.0, tp_upper=105.0,
        )
        _setup_position_state(s, "LONG", 104.5)
        s.state.side = "LONG"
        s.state.middle_runner_active = True
        s.state.tp_plan = "MIDDLE_RUNNER"
        s.state.tp_mode = "UPPER"
        s.state.tp_price = 105.0
        s.state.middle_runner_first_tp_price = 106.5
        s.state.middle_runner_final_tp_price = 105.0
        s.state.last_tp_update_candle_ts_ms = 0
        s.state.partial_tp_consumed = True
        s.state.middle_runner_enabled_for_position = True
        s.state.first_entry_ts_ms = 1000
        s.state.last_order_ts_ms = 1000
        # effective_be ~ 104.5, required_outer = 105.0225
        # tp_upper=105.0 < 105.0225 → insufficient
        # upper=110.0 >= 105.0225 → fallback

        intent = s._maybe_update_tp(105.0, 2000, b, _cvd())
        assert intent is not None, "middle_runner_active should produce UPDATE_TP intent"
        assert intent.tp_price == 110.0
        assert s.state.middle_runner_final_tp_price == 110.0, (
            "Final TP must fall back to BOLL20 upper (110.0)"
        )
        assert s.state.tp_price == 110.0

    def test_split_fallback_uses_valid_outer_when_tp_boll_outer_lacks_profit(self):
        """SPLIT_PARTIAL_FINAL fallback: outer uses BOLL20 upper when TP_BOLL15 insufficient."""
        s = _strategy(tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(
            middle=106.5, tp_middle=106.5,
            upper=110.0, tp_upper=107.0,
        )
        _setup_position_state(s, "LONG", 106.5)
        s.state.side = "LONG"
        s.state.tp_plan = "SPLIT_PARTIAL_FINAL"
        s.state.partial_tp_consumed = False
        s.state.tp_price = 105.0
        s.state.tp_mode = "UPPER"
        s.state.partial_tp_price = 106.5
        s.state.partial_tp_ratio = 0.5
        s.state.last_tp_update_candle_ts_ms = 0
        s.state.middle_runner_active = False
        s.state.trend_runner_active = False
        s.state.three_stage_tp1_consumed = False
        s.state.three_stage_tp2_consumed = False
        # effective_be ~ 106.6065, required_middle = 106.6065 * 1.005 = 107.1395
        # tp_middle=106.5 < 107.1395 → insufficient middle
        # middle=106.5 < 107.1395 → insufficient middle
        # → tp_mode becomes UPPER
        # tp_upper=107.0 > 106.6065 (not at loss), but < 107.1395 (insufficient)
        # upper=110.0 >= 107.1395 → structure fallback to 110.0

        intent = s._maybe_update_tp(106.5, 2000, b, _cvd())
        assert intent is not None
        assert intent.tp_price == 110.0, (
            "SPLIT fallback outer must use BOLL20 upper (110.0)"
        )
        assert s.state.tp_plan == "SINGLE"
        assert s.state.tp_price == 110.0
        assert s.state.tp_mode == "UPPER"

    def test_generic_complex_fallback_uses_valid_outer_when_tp_boll_outer_lacks_profit(self):
        """Generic complex plan fallback uses BOLL20 upper when TP_BOLL15 insufficient."""
        s = _strategy(tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(
            middle=106.5, tp_middle=106.5,
            upper=110.0, tp_upper=107.0,
        )
        _setup_position_state(s, "LONG", 106.5)
        s.state.side = "LONG"
        # Use a non-SINGLE, non-SPLIT, non-THREE_STAGE plan to hit the generic branch
        s.state.tp_plan = "MIDDLE_RUNNER"
        s.state.middle_runner_pending = False
        s.state.middle_runner_active = False
        s.state.middle_runner_enabled_for_position = False
        s.state.trend_runner_active = False
        s.state.three_stage_tp1_consumed = False
        s.state.three_stage_tp2_consumed = False
        s.state.three_stage_runner_enabled_for_position = False
        s.state.tp_price = 107.0
        s.state.tp_mode = "UPPER"
        s.state.last_tp_update_candle_ts_ms = 0
        s.state.partial_tp_price = None
        s.state.partial_tp_consumed = False
        # effective_be ~ 106.6065, required_middle = 107.1395
        # tp_middle=106.5 < 107.1395 → insufficient
        # middle=106.5 < 107.1395 → insufficient
        # → tp_mode becomes UPPER
        # tp_upper=107.0 > 106.6065 (not at loss), but < 107.1395 (insufficient)
        # upper=110.0 >= 107.1395 → structure fallback to 110.0

        intent = s._maybe_update_tp(106.5, 2000, b, _cvd())
        assert intent is not None
        assert intent.tp_price == 110.0, (
            "COMPLEX fallback outer must use BOLL20 upper (110.0)"
        )
        assert s.state.tp_plan == "SINGLE"
        assert s.state.tp_price == 110.0

    def test_tp_boll_log_source_reports_structure_outer_profit_fallback(self):
        """_select_valid_tp_outer_with_profit_fallback(log_warning=False) reports correct source."""
        s = _strategy(tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(
            middle=106.5, tp_middle=106.5,
            upper=110.0, tp_upper=105.0,
        )
        _setup_position_state(s, "LONG", 104.5)
        # effective_be ~ 104.5, required_outer = 105.0225
        # tp_upper=105.0 < 105.0225 → insufficient
        # upper=110.0 >= 105.0225 → fallback to BOLL20

        outer_price, outer_src = s._select_valid_tp_outer_with_profit_fallback(
            "LONG", b, log_warning=False)
        assert outer_price == 110.0
        assert outer_src == "STRUCTURE_BOLL_OUTER_PROFIT_FALLBACK", (
            f"Log source must report STRUCTURE_BOLL_OUTER_PROFIT_FALLBACK, got {outer_src}"
        )

    def test_valid_outer_with_log_warning_true_still_logs(self):
        """log_warning=True on TP_OUTER_PROFIT_INSUFFICIENT_FALLBACK still emits warning."""
        s = _strategy(tp_min_net_profit_pct=0.005)
        b = _boll_with_tp(
            middle=107.5, tp_middle=107.6,
            upper=108.1, tp_upper=107.8,
        )
        _setup_position_state(s, "LONG", 107.5)
        # effective_be ~ 107.6075, required_outer = 107.6075 * 1.005 = 108.1455
        # tp_upper=107.8 > 107.6075 (not at loss), < 108.1455 → insufficient
        # upper=108.1 > 107.6075 (not at loss), < 108.1455 → insufficient
        # → TP_OUTER_PROFIT_INSUFFICIENT_FALLBACK with WARNING

        outer_price, outer_src = s._select_valid_tp_outer_with_profit_fallback(
            "LONG", b, log_warning=True)
        assert outer_price == 108.1
        assert outer_src == "TP_OUTER_PROFIT_INSUFFICIENT_FALLBACK"
