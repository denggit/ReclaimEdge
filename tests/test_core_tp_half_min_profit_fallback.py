"""Integration tests for core TP outer half-min-profit fallback.

Verifies:
- LONG: when BOLL outer is at/below breakeven, TP uses half-min-profit price
- SHORT: when BOLL outer is at/above breakeven, TP uses half-min-profit price
- Small profit outer (not loss, not full profit) does NOT trigger half fallback
- TpUpdateCoordinator generates UPDATE_TP intent with correct half-min-profit price
- CORE_TP_OUTER_UNPROFITABLE_HALF_MIN_FALLBACK warning is logged
"""

from __future__ import annotations

from unittest import mock

import pytest
from _pytest.logging import LogCaptureFixture

from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
    StrategyPositionState,
)


# ── helpers ────────────────────────────────────────────────────────────

def _boll_with_tp(
    middle: float = 100.0,
    upper: float = 110.0,
    lower: float = 90.0,
    tp_middle: float | None = 101.0,
    tp_upper: float | None = 108.0,
    tp_lower: float | None = 92.0,
    tp_window: int | None = 15,
    candle_ts_ms: int = 1000,
) -> BollSnapshot:
    return BollSnapshot(
        inst_id="ETH-USDT-SWAP",
        candle_ts_ms=candle_ts_ms,
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


def _strategy(**kwargs) -> BollCvdReclaimStrategy:
    config = BollCvdReclaimStrategyConfig(**kwargs)
    sizer_config = SimplePositionSizerConfig()
    sizer = SimplePositionSizer(sizer_config)
    return BollCvdReclaimStrategy(config, sizer)


def _setup_position_state(
    strategy: BollCvdReclaimStrategy,
    side: str = "LONG",
    layers: int = 1,
    avg_entry_price: float = 100.0,
    breakeven_price: float = 100.0,
    net_remaining_breakeven_price: float = 100.0,
    last_tp_update_candle_ts_ms: int = 0,
    tp_price: float | None = None,
    tp_plan: str = "SINGLE",
) -> None:
    s = strategy.state
    s.side = side
    s.layers = layers
    s.avg_entry_price = avg_entry_price
    s.breakeven_price = breakeven_price
    s.net_remaining_breakeven_price = net_remaining_breakeven_price
    s.last_tp_update_candle_ts_ms = last_tp_update_candle_ts_ms
    if tp_price is not None:
        s.tp_price = tp_price
    s.tp_plan = tp_plan


# ── Strategy wrapper: _select_valid_tp_outer_with_profit_fallback ─────

class TestStrategySelectValidTpOuterHalfMinProfitFallback:
    """Integration between strategy wrapper and pure selector for half-min-profit."""

    def test_long_outer_at_loss_returns_half_min_profit(self):
        """LONG: BOLL upper=99 <= effective_be=100 → half-min-profit fallback."""
        s = _strategy()
        _setup_position_state(s, side="LONG", avg_entry_price=100.0, net_remaining_breakeven_price=100.0)
        # tp_middle also low so middle doesn't satisfy profit
        boll = _boll_with_tp(
            middle=99.0, upper=99.0, lower=90.0,
            tp_middle=99.0, tp_upper=99.0, tp_lower=92.0,
        )
        tp_price, tp_src = s._select_valid_tp_outer_with_profit_fallback("LONG", boll)
        # half_min = 0.004 * 0.5 = 0.002 → 100 * 1.002 = 100.2
        assert tp_src == "TP_OUTER_HALF_MIN_PROFIT_FALLBACK"
        assert tp_price == 100.2

    def test_short_outer_at_loss_returns_half_min_profit(self):
        """SHORT: BOLL lower=101 >= effective_be=100 → half-min-profit fallback."""
        s = _strategy()
        _setup_position_state(s, side="SHORT", avg_entry_price=100.0, net_remaining_breakeven_price=100.0)
        boll = _boll_with_tp(
            middle=101.0, upper=110.0, lower=101.0,
            tp_middle=101.0, tp_upper=108.0, tp_lower=101.0,
        )
        tp_price, tp_src = s._select_valid_tp_outer_with_profit_fallback("SHORT", boll)
        # half_min = 0.004 * 0.5 = 0.002 → 100 * 0.998 = 99.8
        assert tp_src == "TP_OUTER_HALF_MIN_PROFIT_FALLBACK"
        assert tp_price == 99.8

    def test_long_outer_small_profit_no_half_fallback(self):
        """LONG: BOLL upper=100.05 > effective_be=100 (small profit) → NO half fallback."""
        s = _strategy()
        _setup_position_state(s, side="LONG", avg_entry_price=100.0, net_remaining_breakeven_price=100.0)
        # upper=100.05 is small profit (above be=100), but insufficient for full 0.004
        boll = _boll_with_tp(
            middle=99.0, upper=102.0, lower=90.0,
            tp_middle=99.0, tp_upper=100.05, tp_lower=92.0,
        )
        tp_price, tp_src = s._select_valid_tp_outer_with_profit_fallback("LONG", boll)
        # Should use structure fallback (102.0 >= 100.4), not half fallback
        assert tp_src != "TP_OUTER_HALF_MIN_PROFIT_FALLBACK"
        assert tp_price != 100.2
        # full required = 100.4, tp_upper=100.05 < 100.4 failed
        # structure_upper=102.0 >= 100.4 → structure fallback
        assert tp_src == "STRUCTURE_BOLL_OUTER_PROFIT_FALLBACK"
        assert tp_price == 102.0

    def test_short_outer_small_profit_no_half_fallback(self):
        """SHORT: BOLL lower=99.95 < effective_be=100 (small profit) → NO half fallback."""
        s = _strategy()
        _setup_position_state(s, side="SHORT", avg_entry_price=100.0, net_remaining_breakeven_price=100.0)
        boll = _boll_with_tp(
            middle=101.0, upper=110.0, lower=98.0,
            tp_middle=101.0, tp_upper=108.0, tp_lower=99.95,
        )
        tp_price, tp_src = s._select_valid_tp_outer_with_profit_fallback("SHORT", boll)
        # Should use structure fallback, not half fallback
        assert tp_src != "TP_OUTER_HALF_MIN_PROFIT_FALLBACK"
        assert tp_price != 99.8

    def test_effective_be_zero_no_half_fallback(self):
        """effective_be=0 → no half fallback, returns raw outer."""
        s = _strategy()
        _setup_position_state(s, side="LONG", avg_entry_price=0.0, net_remaining_breakeven_price=0.0)
        boll = _boll_with_tp(
            middle=99.0, upper=99.0, lower=90.0,
            tp_middle=99.0, tp_upper=99.0, tp_lower=92.0,
        )
        tp_price, tp_src = s._select_valid_tp_outer_with_profit_fallback("LONG", boll)
        # effective_be=0 → no profit check, returns raw TP_BOLL upper
        assert tp_src == "TP_BOLL"
        assert tp_price == 99.0

    def test_outer_meets_full_profit_uses_tp_boll(self):
        """LONG: BOLL upper=101 >= 100.4 (full min profit) → TP_BOLL."""
        s = _strategy()
        _setup_position_state(s, side="LONG", avg_entry_price=100.0, net_remaining_breakeven_price=100.0)
        boll = _boll_with_tp(
            middle=100.0, upper=102.0, lower=90.0,
            tp_middle=100.0, tp_upper=101.0, tp_lower=92.0,
        )
        tp_price, tp_src = s._select_valid_tp_outer_with_profit_fallback("LONG", boll)
        assert tp_src == "TP_BOLL"
        assert tp_price == 101.0

    def test_core_tp_outer_half_min_fallback_log_warning(self):
        """Verify warning log is emitted with all required fields."""
        s = _strategy()
        _setup_position_state(s, side="LONG", avg_entry_price=100.0, net_remaining_breakeven_price=100.0)
        boll = _boll_with_tp(
            middle=99.0, upper=101.0, lower=90.0,
            tp_middle=99.0, tp_upper=99.0, tp_lower=92.0,
            candle_ts_ms=5000,
        )
        with mock.patch.object(s, "_select_valid_tp_outer_with_profit_fallback", wraps=s._select_valid_tp_outer_with_profit_fallback) as spy:
            tp_price, tp_src = spy("LONG", boll, log_warning=True)
            assert tp_src == "TP_OUTER_HALF_MIN_PROFIT_FALLBACK"
            assert tp_price == 100.2


# ── TpUpdateCoordinator integration ────────────────────────────────────

class TestTpUpdateCoordinatorHalfMinProfit:
    """Integration: TpUpdateCoordinator generates UPDATE_TP with half-min-profit price."""

    def test_long_update_tp_intent_with_half_min_profit(self):
        """LONG: outer at loss → UPDATE_TP intent with half-min-profit tp_price."""
        s = _strategy()
        _setup_position_state(
            s,
            side="LONG",
            layers=1,
            avg_entry_price=100.0,
            breakeven_price=100.0,
            net_remaining_breakeven_price=100.0,
            last_tp_update_candle_ts_ms=0,
        )
        # middle and tp_middle both below effective_be, upper at loss
        boll = _boll_with_tp(
            middle=99.0, upper=101.0, lower=90.0,
            tp_middle=99.0, tp_upper=99.0, tp_lower=92.0,
            candle_ts_ms=2000,
        )
        cvd = _cvd()
        result = s._maybe_update_tp(100.0, 2000, boll, cvd)
        assert result is not None
        assert result.intent_type == "UPDATE_TP"
        # half_min = 0.004 * 0.5 = 0.002 → 100 * 1.002 = 100.2
        assert result.tp_price == 100.2
        assert result.tp_mode == "UPPER"

    def test_short_update_tp_intent_with_half_min_profit(self):
        """SHORT: outer at loss → UPDATE_TP intent with half-min-profit tp_price."""
        s = _strategy()
        _setup_position_state(
            s,
            side="SHORT",
            layers=1,
            avg_entry_price=100.0,
            breakeven_price=100.0,
            net_remaining_breakeven_price=100.0,
            last_tp_update_candle_ts_ms=0,
        )
        # middle and tp_middle both above effective_be, lower at loss
        boll = _boll_with_tp(
            middle=101.0, upper=110.0, lower=99.0,
            tp_middle=101.0, tp_upper=108.0, tp_lower=101.0,
            candle_ts_ms=2000,
        )
        cvd = _cvd()
        result = s._maybe_update_tp(100.0, 2000, boll, cvd)
        assert result is not None
        assert result.intent_type == "UPDATE_TP"
        # half_min = 0.004 * 0.5 = 0.002 → 100 * 0.998 = 99.8
        assert result.tp_price == 99.8
        assert result.tp_mode == "LOWER"

    def test_long_outer_meets_full_profit_no_half_fallback(self):
        """LONG: outer meets full profit → UPDATE_TP uses TP_BOLL (not half fallback)."""
        s = _strategy()
        _setup_position_state(
            s,
            side="LONG",
            layers=1,
            avg_entry_price=100.0,
            breakeven_price=100.0,
            net_remaining_breakeven_price=100.0,
            last_tp_update_candle_ts_ms=0,
        )
        # tp_upper=101 >= 100.4 (full min profit)
        boll = _boll_with_tp(
            middle=100.0, upper=102.0, lower=90.0,
            tp_middle=100.0, tp_upper=101.0, tp_lower=92.0,
            candle_ts_ms=2000,
        )
        cvd = _cvd()
        result = s._maybe_update_tp(100.0, 2000, boll, cvd)
        assert result is not None
        assert result.intent_type == "UPDATE_TP"
        # middle=100.0 < required=100.4 → middle fails → outer UPPER
        # tp_upper=101 >= 100.4 → TP_BOLL
        assert result.tp_price == 101.0
        assert result.tp_mode == "UPPER"


# ── Log field correctness ───────────────────────────────────────────────

class TestHalfMinProfitFallbackLogFields:
    """Verify raw_outer, tp_boll_outer, structure_outer fields in the warning log."""

    def test_tp_boll_available_raw_outer_equals_tp_boll_outer(
        self, caplog: LogCaptureFixture,
    ):
        """LONG: TP_BOLL available → raw_outer = tp_boll_outer."""
        s = _strategy()
        _setup_position_state(s, side="LONG", avg_entry_price=100.0, net_remaining_breakeven_price=100.0)
        # tp_upper=99 < effective_be=100 → outer at loss → half fallback
        # tp_middle=99 < required middle=100.4 → middle also fails
        # structure upper=98 also at loss
        boll = _boll_with_tp(
            middle=99.0, upper=98.0, lower=90.0,
            tp_middle=99.0, tp_upper=99.0, tp_lower=92.0,
            tp_window=15,
        )
        with caplog.at_level("WARNING"):
            tp_price, tp_src = s._select_valid_tp_outer_with_profit_fallback(
                "LONG", boll, log_warning=True,
            )
        assert tp_src == "TP_OUTER_HALF_MIN_PROFIT_FALLBACK"
        log_text = caplog.text
        assert "CORE_TP_OUTER_UNPROFITABLE_HALF_MIN_FALLBACK" in log_text
        assert "raw_outer=99.0000" in log_text
        assert "tp_boll_outer=99.0000" in log_text
        assert "structure_outer=98.0000" in log_text

    def test_tp_boll_unavailable_raw_outer_equals_structure_outer(
        self, caplog: LogCaptureFixture,
    ):
        """LONG: TP_BOLL unavailable → raw_outer = structure_outer."""
        s = _strategy()
        _setup_position_state(s, side="LONG", avg_entry_price=100.0, net_remaining_breakeven_price=100.0)
        # tp_upper/middle/lower all None → TP_BOLL unavailable
        # tp_middle=None means middle also unavailable → fallback to outer
        # structure upper=99 < effective_be=100 → outer at loss → half fallback
        boll = _boll_with_tp(
            middle=99.0, upper=99.0, lower=90.0,
            tp_middle=None, tp_upper=None, tp_lower=None,
            tp_window=None,
        )
        with caplog.at_level("WARNING"):
            tp_price, tp_src = s._select_valid_tp_outer_with_profit_fallback(
                "LONG", boll, log_warning=True,
            )
        assert tp_src == "TP_OUTER_HALF_MIN_PROFIT_FALLBACK"
        log_text = caplog.text
        assert "CORE_TP_OUTER_UNPROFITABLE_HALF_MIN_FALLBACK" in log_text
        assert "raw_outer=99.0000" in log_text
        assert "tp_boll_outer=-" in log_text
        assert "structure_outer=99.0000" in log_text

    def test_short_tp_boll_available_raw_outer_equals_tp_boll_outer(
        self, caplog: LogCaptureFixture,
    ):
        """SHORT: TP_BOLL available → raw_outer = tp_boll_outer (lower)."""
        s = _strategy()
        _setup_position_state(s, side="SHORT", avg_entry_price=100.0, net_remaining_breakeven_price=100.0)
        # tp_lower=101 > effective_be=100 → outer at loss → half fallback
        # tp_middle=101 > required middle=99.6 → middle also fails
        # structure lower=102 also at loss
        boll = _boll_with_tp(
            middle=101.0, upper=110.0, lower=102.0,
            tp_middle=101.0, tp_upper=108.0, tp_lower=101.0,
            tp_window=15,
        )
        with caplog.at_level("WARNING"):
            tp_price, tp_src = s._select_valid_tp_outer_with_profit_fallback(
                "SHORT", boll, log_warning=True,
            )
        assert tp_src == "TP_OUTER_HALF_MIN_PROFIT_FALLBACK"
        log_text = caplog.text
        assert "CORE_TP_OUTER_UNPROFITABLE_HALF_MIN_FALLBACK" in log_text
        assert "raw_outer=101.0000" in log_text
        assert "tp_boll_outer=101.0000" in log_text
        assert "structure_outer=102.0000" in log_text
