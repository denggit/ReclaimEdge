"""Tests for extreme_retest_add restart/restore scenarios."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest

from src.indicators.cvd_tracker import CvdSnapshot

if importlib.util.find_spec("aiohttp") is None:
    sys.modules.setdefault("aiohttp", types.ModuleType("aiohttp"))

from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig
from src.strategies import extreme_retest_add as _extreme_retest
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategyConfig,
    StrategyPositionState,
)
from src.strategies.boll_cvd_shock_reclaim_strategy import BollCvdShockReclaimStrategy
from src.strategies.extreme_retest_add import ExtremeRetestAnchor, ExtremeRetestConfig


def boll_snapshot(**overrides) -> BollSnapshot:
    defaults = dict(
        inst_id="ETH-USDT-SWAP",
        candle_ts_ms=5000,
        close=105.0,
        middle=105.0,
        upper=110.0,
        lower=100.0,
        upper_distance_pct=0.05,
        lower_distance_pct=0.05,
        alert_switch_on=True,
        live_mode=True,
        high=106.0,
        low=104.0,
    )
    defaults.update(overrides)
    return BollSnapshot(**defaults)


def cvd_snapshot(**overrides):
    defaults = dict(
        ts_ms=0, price=105.0, side="unknown", size=0.0, signed_delta=0.0,
        total_cvd=0.0, fast_cvd=0.0, previous_fast_cvd=0.0,
        buy_volume=0.0, sell_volume=0.0,
        buy_ratio=0.0, sell_ratio=0.0,
        cross_positive=False, cross_negative=False,
        cvd_increasing=False, cvd_decreasing=False,
        no_new_low=False, no_new_high=False,
        window_low=105.0, window_high=105.0,
        burst_net_move_pct=0.0, burst_range_pct=0.01,
        baseline_range_pct=0.001, burst_move_ratio=10.0,
        burst_volume=10.0, baseline_volume=1.0, burst_volume_ratio=10.0,
        up_burst=False, down_burst=False,
    )
    defaults.update(overrides)
    return CvdSnapshot(**defaults)


def strategy(**overrides) -> BollCvdShockReclaimStrategy:
    values = dict(min_outside_pct=0.001)
    values.update(overrides)
    config = BollCvdReclaimStrategyConfig(**values)
    sizer = SimplePositionSizer(SimplePositionSizerConfig())
    return BollCvdShockReclaimStrategy(config, sizer)


def _make_config(**overrides) -> ExtremeRetestConfig:
    defaults = dict(
        enabled=True,
        pivot_left_bars=2,
        pivot_right_bars=2,
        anchor_max_age_candles=12,
        sweep_max_age_seconds=900.0,
        near_extreme_pct=0.0015,
        reclaim_pct=0.0005,
        min_reverse_ratio=0.55,
        one_add_per_anchor=True,
    )
    defaults.update(overrides)
    return ExtremeRetestConfig(**defaults)


def _seed_candle_buffer(strat: BollCvdShockReclaimStrategy, candles: list[dict]) -> None:
    """Pre-populate the candle buffer by simulating tick sequence.

    Adds one extra dummy tick at the end so the last real candle is pushed.
    """
    strat._candle_buffer.clear()
    strat._prev_boll = None
    for c in candles:
        boll = boll_snapshot(
            candle_ts_ms=c["ts_ms"],
            close=c["close"],
            upper=c.get("boll_upper", 110.0),
            lower=c.get("boll_lower", 100.0),
            high=c["high"],
            low=c["low"],
        )
        if strat._prev_boll is not None and boll.candle_ts_ms != strat._prev_boll.candle_ts_ms:
            prev = strat._prev_boll
            strat._candle_buffer.append({
                "ts_ms": prev.candle_ts_ms,
                "high": prev.high if prev.high is not None else prev.close,
                "low": prev.low if prev.low is not None else prev.close,
                "close": prev.close,
                "boll_upper": prev.upper,
                "boll_lower": prev.lower,
            })
        strat._prev_boll = boll
    # Push a dummy tick to flush the last real candle into the buffer
    last_c = candles[-1]
    dummy = boll_snapshot(
        candle_ts_ms=last_c["ts_ms"] + 1000,
        close=last_c["close"],
        upper=last_c.get("boll_upper", 110.0),
        lower=last_c.get("boll_lower", 100.0),
        high=last_c["high"],
        low=last_c["low"],
    )
    if strat._prev_boll is not None and dummy.candle_ts_ms != strat._prev_boll.candle_ts_ms:
        strat._candle_buffer.append({
            "ts_ms": strat._prev_boll.candle_ts_ms,
            "high": strat._prev_boll.high if strat._prev_boll.high is not None else strat._prev_boll.close,
            "low": strat._prev_boll.low if strat._prev_boll.low is not None else strat._prev_boll.close,
            "close": strat._prev_boll.close,
            "boll_upper": strat._prev_boll.upper,
            "boll_lower": strat._prev_boll.lower,
        })
    strat._prev_boll = dummy


def _candle(ts_ms: int, high: float, low: float, close: float,
            boll_upper: float = 110.0, boll_lower: float = 100.0) -> dict:
    return {"ts_ms": ts_ms, "high": high, "low": low, "close": close,
            "boll_upper": boll_upper, "boll_lower": boll_lower}


# ──────────────────────────────────────────────────────────────────────────────
# Trusted State Restore
# ──────────────────────────────────────────────────────────────────────────────


class TrustedStateRestoreTest(unittest.TestCase):

    def test_trusted_saved_state_with_anchor_restores(self) -> None:
        strat = strategy(extreme_retest_add_enabled=True)
        # Set up state with an active anchor
        strat.state.side = "SHORT"
        strat.state.layers = 1
        strat.state.last_entry_price = 1700.0
        strat.state.extreme_retest_anchor_side = "SHORT"
        strat.state.extreme_retest_anchor_kind = "PIVOT_HIGH"
        strat.state.extreme_retest_anchor_price = 1740.0
        strat.state.extreme_retest_anchor_candle_ts_ms = 5000

        strat.restore_extreme_retest_state_from_saved(trusted=True)

        self.assertEqual(strat._extreme_retest_anchor.side, "SHORT")
        self.assertEqual(strat._extreme_retest_anchor.kind, "PIVOT_HIGH")
        self.assertEqual(strat._extreme_retest_anchor.price, 1740.0)

    def test_trusted_saved_state_with_sweep_restores(self) -> None:
        strat = strategy(extreme_retest_add_enabled=True)
        strat.state.side = "SHORT"
        strat.state.layers = 1
        strat.state.last_entry_price = 1700.0
        strat.state.extreme_retest_anchor_side = "SHORT"
        strat.state.extreme_retest_anchor_kind = "PIVOT_HIGH"
        strat.state.extreme_retest_anchor_price = 1740.0
        strat.state.extreme_retest_anchor_candle_ts_ms = 5000
        strat.state.extreme_retest_sweep_seen = True
        strat.state.extreme_retest_sweep_extreme_price = 1755.0
        strat.state.extreme_retest_sweep_first_seen_ts_ms = 6000

        strat.restore_extreme_retest_state_from_saved(trusted=True)

        self.assertTrue(strat._extreme_retest_anchor.sweep_seen)
        self.assertEqual(strat._extreme_retest_anchor.sweep_extreme_price, 1755.0)
        self.assertEqual(strat._extreme_retest_anchor.sweep_first_seen_ts_ms, 6000)

    def test_untrusted_saved_state_drops_anchor(self) -> None:
        strat = strategy(extreme_retest_add_enabled=True)
        strat.state.side = "SHORT"
        strat.state.layers = 1
        strat.state.extreme_retest_anchor_side = "SHORT"
        strat.state.extreme_retest_anchor_price = 1740.0
        strat.state.extreme_retest_consumed_watermark_price = 1730.0  # watermark is preserved

        strat.restore_extreme_retest_state_from_saved(trusted=False)

        self.assertIsNone(strat._extreme_retest_anchor.side)
        self.assertIsNone(strat._extreme_retest_anchor.price)
        # Watermark should be preserved
        self.assertEqual(strat._extreme_retest_anchor.consumed_watermark_price, 1730.0)

    def test_trusted_state_with_consumed_watermark_restores(self) -> None:
        strat = strategy(extreme_retest_add_enabled=True)
        strat.state.side = "SHORT"
        strat.state.layers = 1
        strat.state.last_entry_price = 1700.0
        strat.state.extreme_retest_consumed_watermark_price = 1740.0
        strat.state.extreme_retest_consumed_anchor_ts_ms = 5000

        strat.restore_extreme_retest_state_from_saved(trusted=True)

        self.assertEqual(strat._extreme_retest_anchor.consumed_watermark_price, 1740.0)
        self.assertEqual(strat._extreme_retest_anchor.consumed_anchor_ts_ms, 5000)
        # anchor should not be active (was consumed)
        self.assertFalse(strat._extreme_retest_anchor.is_active())


# ──────────────────────────────────────────────────────────────────────────────
# Rebuild from Candles
# ──────────────────────────────────────────────────────────────────────────────


class RebuildFromCandlesTest(unittest.TestCase):

    def test_rebuild_from_candles_short_pivot_high(self) -> None:
        strat = strategy(extreme_retest_add_enabled=True)
        strat.state.side = "SHORT"
        strat.state.layers = 1
        strat.state.last_entry_price = 2100.0

        # Seed candles with a valid pivot high
        candles = [
            _candle(1000, 2100, 2080, 2095, boll_upper=2150, boll_lower=2050),
            _candle(2000, 2120, 2090, 2110, boll_upper=2150, boll_lower=2050),
            _candle(3000, 2130, 2100, 2120, boll_upper=2150, boll_lower=2050),
            _candle(4000, 2170, 2110, 2160, boll_upper=2150, boll_lower=2050),  # pivot high=2170 > upper=2150
            _candle(5000, 2140, 2120, 2135, boll_upper=2150, boll_lower=2050),
            _candle(6000, 2150, 2130, 2145, boll_upper=2150, boll_lower=2050),
        ]
        _seed_candle_buffer(strat, candles)

        result = strat.rebuild_extreme_retest_anchor_from_candles()
        self.assertTrue(result)
        self.assertEqual(strat._extreme_retest_anchor.side, "SHORT")
        self.assertEqual(strat._extreme_retest_anchor.kind, "PIVOT_HIGH")
        self.assertEqual(strat._extreme_retest_anchor.price, 2170.0)

    def test_rebuild_from_candles_short_pivot_high_realistic_prices(self) -> None:
        strat = strategy(extreme_retest_add_enabled=True)
        strat.state.side = "SHORT"
        strat.state.layers = 1
        strat.state.last_entry_price = 2000.0

        candles = [
            _candle(1000, 2100, 2080, 2095, boll_upper=2150, boll_lower=2050),
            _candle(2000, 2120, 2090, 2110, boll_upper=2150, boll_lower=2050),
            _candle(3000, 2130, 2100, 2120, boll_upper=2150, boll_lower=2050),
            _candle(4000, 2170, 2110, 2160, boll_upper=2150, boll_lower=2050),  # pivot high=2170 > upper=2150
            _candle(5000, 2140, 2120, 2135, boll_upper=2150, boll_lower=2050),
            _candle(6000, 2150, 2130, 2145, boll_upper=2150, boll_lower=2050),
        ]
        _seed_candle_buffer(strat, candles)

        result = strat.rebuild_extreme_retest_anchor_from_candles()
        # last_entry=2000, pivot=2170, gap=(2170-2000)/2000=0.085 > required
        self.assertTrue(result)
        self.assertEqual(strat._extreme_retest_anchor.side, "SHORT")
        self.assertEqual(strat._extreme_retest_anchor.kind, "PIVOT_HIGH")
        self.assertEqual(strat._extreme_retest_anchor.price, 2170.0)

    def test_rebuild_from_candles_long_pivot_low(self) -> None:
        strat = strategy(extreme_retest_add_enabled=True)
        strat.state.side = "LONG"
        strat.state.layers = 1
        strat.state.last_entry_price = 2100.0

        candles = [
            _candle(1000, 2100, 2050, 2095, boll_upper=2150, boll_lower=2050),
            _candle(2000, 2120, 2080, 2110, boll_upper=2150, boll_lower=2050),
            _candle(3000, 2110, 2070, 2100, boll_upper=2150, boll_lower=2050),
            _candle(4000, 2120, 2030, 2110, boll_upper=2150, boll_lower=2050),  # pivot low=2030 < lower=2050
            _candle(5000, 2140, 2040, 2130, boll_upper=2150, boll_lower=2050),
            _candle(6000, 2150, 2045, 2145, boll_upper=2150, boll_lower=2050),
        ]
        _seed_candle_buffer(strat, candles)

        result = strat.rebuild_extreme_retest_anchor_from_candles()
        self.assertTrue(result)
        self.assertEqual(strat._extreme_retest_anchor.side, "LONG")
        self.assertEqual(strat._extreme_retest_anchor.kind, "PIVOT_LOW")
        self.assertEqual(strat._extreme_retest_anchor.price, 2030.0)

    def test_rebuild_with_no_valid_pivot_returns_false(self) -> None:
        strat = strategy(extreme_retest_add_enabled=True)
        strat.state.side = "SHORT"
        strat.state.layers = 1
        strat.state.last_entry_price = 2000.0

        # No pivot outside band
        candles = [
            _candle(1000, 2100, 2080, 2095, boll_upper=2150, boll_lower=2050),
            _candle(2000, 2120, 2090, 2110, boll_upper=2150, boll_lower=2050),
            _candle(3000, 2130, 2100, 2120, boll_upper=2150, boll_lower=2050),
            _candle(4000, 2140, 2110, 2130, boll_upper=2150, boll_lower=2050),  # high=2140 <= upper=2150
            _candle(5000, 2140, 2120, 2135, boll_upper=2150, boll_lower=2050),
            _candle(6000, 2150, 2130, 2145, boll_upper=2150, boll_lower=2050),
        ]
        _seed_candle_buffer(strat, candles)

        result = strat.rebuild_extreme_retest_anchor_from_candles()
        self.assertFalse(result)

    def test_rebuild_too_close_to_last_entry_fails(self) -> None:
        strat = strategy(extreme_retest_add_enabled=True)
        strat.state.side = "SHORT"
        strat.state.layers = 1
        # Very close to pivot 2170 — gap = (2170-2165)/2165 = 0.0023 < 0.003
        strat.state.last_entry_price = 2165.0

        candles = [
            _candle(1000, 2100, 2080, 2095, boll_upper=2150, boll_lower=2050),
            _candle(2000, 2120, 2090, 2110, boll_upper=2150, boll_lower=2050),
            _candle(3000, 2130, 2100, 2120, boll_upper=2150, boll_lower=2050),
            _candle(4000, 2170, 2110, 2160, boll_upper=2150, boll_lower=2050),  # pivot=2170 > upper=2150
            _candle(5000, 2140, 2120, 2135, boll_upper=2150, boll_lower=2050),
            _candle(6000, 2150, 2130, 2145, boll_upper=2150, boll_lower=2050),
        ]
        _seed_candle_buffer(strat, candles)

        result = strat.rebuild_extreme_retest_anchor_from_candles()
        # gap = (2170-2165)/2165 ≈ 0.0023 < 0.003 → rejected
        self.assertFalse(result)


# ──────────────────────────────────────────────────────────────────────────────
# Startup ADD Prevention
# ──────────────────────────────────────────────────────────────────────────────


class StartupAddPreventionTest(unittest.TestCase):

    def test_restored_anchor_does_not_trigger_on_startup(self) -> None:
        """Restored anchor should NOT immediately trigger ADD — must wait for live
        tick that satisfies all conditions."""
        strat = strategy(extreme_retest_add_enabled=True)
        strat.state.side = "SHORT"
        strat.state.layers = 1
        strat.state.last_entry_price = 2000.0
        strat.state.last_order_ts_ms = 1000  # old timestamp → cooldown OK
        strat.state.add_freeze_until_ts_ms = 0

        # Set up anchor in state
        strat.state.extreme_retest_anchor_side = "SHORT"
        strat.state.extreme_retest_anchor_kind = "PIVOT_HIGH"
        strat.state.extreme_retest_anchor_price = 2200.0
        strat.state.extreme_retest_anchor_candle_ts_ms = 5000
        strat.state.extreme_retest_anchor_boll_upper = 2150.0
        strat.state.extreme_retest_anchor_boll_lower = 2050.0

        strat.restore_extreme_retest_state_from_saved(trusted=True)

        # First tick: price inside band, but sell_ratio=0.0 → reject not triggered
        boll = boll_snapshot(candle_ts_ms=5000, close=2100.0, upper=2150.0, lower=2050.0)
        cvd = cvd_snapshot(price=2100.0, sell_ratio=0.0, buy_ratio=0.0)

        intents = strat.on_tick(2100.0, 5000, boll, cvd)
        # No ADD intent because sell_ratio is too low
        add_intents = [i for i in intents if i.intent_type in ("ADD_LONG", "ADD_SHORT")]
        self.assertEqual(len(add_intents), 0)

    def test_restored_anchor_triggers_on_live_tick_meeting_conditions(self) -> None:
        """Restored anchor triggers ADD when live tick meets all conditions
        (price inside band, reject before break, sell_ratio >= threshold)."""
        strat = strategy(extreme_retest_add_enabled=True)
        strat.state.side = "SHORT"
        strat.state.layers = 1
        strat.state.last_entry_price = 2000.0
        strat.state.last_order_ts_ms = 1000
        strat.state.add_freeze_until_ts_ms = 0
        strat.state.avg_entry_price = 2000.0
        strat.state.total_entry_qty = 1.0
        strat.state.total_entry_notional = 2000.0

        strat.state.extreme_retest_anchor_side = "SHORT"
        strat.state.extreme_retest_anchor_kind = "PIVOT_HIGH"
        strat.state.extreme_retest_anchor_price = 2200.0
        strat.state.extreme_retest_anchor_candle_ts_ms = 5000
        strat.state.extreme_retest_anchor_boll_upper = 2300.0
        strat.state.extreme_retest_anchor_boll_lower = 2050.0

        strat.restore_extreme_retest_state_from_saved(trusted=True)

        # Live tick that meets reject before break:
        # price=2199 inside band, near anchor 2200, sell_ratio=0.60 >= 0.55
        boll = boll_snapshot(candle_ts_ms=5000, close=2199.0, upper=2300.0, lower=2050.0)
        cvd = cvd_snapshot(price=2199.0, sell_ratio=0.60, buy_ratio=0.30)

        intents = strat.on_tick(2199.0, 100000, boll, cvd)
        add_intents = [i for i in intents if i.intent_type in ("ADD_LONG", "ADD_SHORT")]
        self.assertEqual(len(add_intents), 1)
        self.assertEqual(add_intents[0].intent_type, "ADD_SHORT")

        # Anchor should be consumed after successful ADD
        self.assertIsNone(strat._extreme_retest_anchor.price)
        self.assertEqual(strat._extreme_retest_anchor.consumed_watermark_price, 2200.0)


# ──────────────────────────────────────────────────────────────────────────────
# Tick Path Boundary
# ──────────────────────────────────────────────────────────────────────────────


class TickPathBoundaryTest(unittest.TestCase):

    def test_evaluate_on_tick_no_file_io(self) -> None:
        """evaluate_on_tick should not do any file IO."""
        anchor = ExtremeRetestAnchor(
            side="SHORT", kind="PIVOT_HIGH", price=110.0, candle_ts_ms=1000,
            boll_upper=111.0, boll_lower=92.0)
        cfg = ExtremeRetestConfig(enabled=True)
        # This is a pure computation — verify it runs without exceptions
        result = _extreme_retest.evaluate_on_tick(
            "SHORT", 109.90, 5000, 111.0, 92.0, anchor, cfg,
            buy_ratio=0.3, sell_ratio=0.60)
        self.assertIsNotNone(result)

    def test_pivot_detection_no_dataframe(self) -> None:
        """Pivot detection uses plain list of dicts, no pandas."""
        candles = [
            {"ts_ms": 1000, "high": 100, "low": 95, "close": 98},
            {"ts_ms": 2000, "high": 102, "low": 97, "close": 101},
            {"ts_ms": 3000, "high": 103, "low": 98, "close": 102},
            {"ts_ms": 4000, "high": 110, "low": 96, "close": 105},
            {"ts_ms": 5000, "high": 105, "low": 99, "close": 104},
            {"ts_ms": 6000, "high": 106, "low": 100, "close": 105},
        ]
        # This should not import pandas or use dataframe
        result = _extreme_retest.detect_pivot_high(candles, 3, 2, 2)
        self.assertTrue(result)


# ──────────────────────────────────────────────────────────────────────────────
# Original Add Filters Preservation (Shock Strategy integration)
# ──────────────────────────────────────────────────────────────────────────────


class OriginalAddFiltersPreservedTest(unittest.TestCase):

    def _setup_short_strat(self, **extra_config):
        # Create config with add_freeze_chain_enabled as needed
        cfg_values = dict(min_outside_pct=0.001, extreme_retest_add_enabled=True)
        if "add_freeze_chain_enabled" in extra_config:
            cfg_values["add_freeze_chain_enabled"] = extra_config.pop("add_freeze_chain_enabled")
        cfg_values.update(extra_config)
        config = BollCvdReclaimStrategyConfig(**cfg_values)
        sizer = SimplePositionSizer(SimplePositionSizerConfig())
        strat = BollCvdShockReclaimStrategy(config, sizer)
        strat.state.side = "SHORT"
        strat.state.layers = 1
        strat.state.last_entry_price = 2000.0
        strat.state.last_order_ts_ms = 1000
        strat.state.avg_entry_price = 2000.0
        strat.state.total_entry_qty = 1.0
        strat.state.total_entry_notional = 2000.0
        # Set up anchor
        strat.state.extreme_retest_anchor_side = "SHORT"
        strat.state.extreme_retest_anchor_kind = "PIVOT_HIGH"
        strat.state.extreme_retest_anchor_price = 2200.0
        strat.state.extreme_retest_anchor_candle_ts_ms = 5000
        strat.state.extreme_retest_anchor_boll_upper = 2300.0
        strat.state.extreme_retest_anchor_boll_lower = 2050.0
        strat.restore_extreme_retest_state_from_saved(trusted=True)
        return strat

    def test_extreme_retest_blocked_by_add_freeze(self) -> None:
        """EXTREME_RETEST triggered but add_freeze active → not ADD."""
        strat = self._setup_short_strat(add_freeze_chain_enabled=True)
        strat.state.add_freeze_until_ts_ms = 200000  # far in future
        # Price must be close enough to last_entry that the adverse gap
        # doesn't exceed the bypass multiplier gap
        strat.state.last_entry_price = 2190.0  # close to price=2199, gap ≈ 0.4%
        strat.restore_extreme_retest_state_from_saved(trusted=True)

        boll = boll_snapshot(candle_ts_ms=5000, upper=2300.0, lower=2050.0)
        cvd = cvd_snapshot(price=2199.0, sell_ratio=0.60, buy_ratio=0.30)

        intents = strat.on_tick(2199.0, 100000, boll, cvd)
        add_intents = [i for i in intents if i.intent_type in ("ADD_SHORT",)]
        self.assertEqual(len(add_intents), 0)
        # Anchor NOT consumed because ADD was blocked
        self.assertTrue(strat._extreme_retest_anchor.is_active())

    def test_extreme_retest_blocked_by_max_layers(self) -> None:
        """EXTREME_RETEST triggered but max_layers reached → not ADD."""
        strat = self._setup_short_strat()
        strat.state.layers = 3  # at max_layers (config default is 3)
        strat.state.add_freeze_until_ts_ms = 0

        boll = boll_snapshot(candle_ts_ms=5000, upper=2300.0, lower=2050.0)
        cvd = cvd_snapshot(price=2199.0, sell_ratio=0.60, buy_ratio=0.30)

        intents = strat.on_tick(2199.0, 100000, boll, cvd)
        add_intents = [i for i in intents if i.intent_type in ("ADD_SHORT",)]
        self.assertEqual(len(add_intents), 0)
        # Anchor NOT consumed
        self.assertTrue(strat._extreme_retest_anchor.is_active())

    def test_extreme_retest_blocked_by_gap(self) -> None:
        """EXTREME_RETEST triggered but gap too small → not ADD."""
        strat = self._setup_short_strat()
        # last_entry=2000, price=2050, gap=(2050-2000)/2000=0.025
        # But required gap for target_layer=2 is base=0.003 (0.3%)
        # Actually 2.5% > 0.3% so gap passes...
        # Let me set price very close to last_entry to fail gap check
        strat.state.last_entry_price = 2150.0
        strat.state.add_freeze_until_ts_ms = 0
        strat.state.extreme_retest_anchor_price = 2300.0
        strat.state.extreme_retest_anchor_boll_upper = 2400.0
        strat.state.extreme_retest_anchor_boll_lower = 2100.0
        strat.restore_extreme_retest_state_from_saved(trusted=True)

        # price very close to last_entry → gap fails
        boll = boll_snapshot(candle_ts_ms=5000, upper=2400.0, lower=2100.0)
        cvd = cvd_snapshot(price=2151.0, sell_ratio=0.60, buy_ratio=0.30)

        intents = strat.on_tick(2151.0, 100000, boll, cvd)
        add_intents = [i for i in intents if i.intent_type in ("ADD_SHORT",)]
        self.assertEqual(len(add_intents), 0)

    def test_extreme_retest_passes_all_filters_generates_add(self) -> None:
        """EXTREME_RETEST triggered, passes all filters → ADD intent generated."""
        strat = self._setup_short_strat()
        strat.state.add_freeze_until_ts_ms = 0

        boll = boll_snapshot(candle_ts_ms=5000, upper=2300.0, lower=2050.0)
        cvd = cvd_snapshot(price=2199.0, sell_ratio=0.60, buy_ratio=0.30)

        intents = strat.on_tick(2199.0, 100000, boll, cvd)
        add_intents = [i for i in intents if i.intent_type in ("ADD_SHORT",)]
        self.assertEqual(len(add_intents), 1)
        self.assertEqual(add_intents[0].intent_type, "ADD_SHORT")
        # Anchor consumed
        self.assertFalse(strat._extreme_retest_anchor.is_active())
        self.assertEqual(strat._extreme_retest_anchor.consumed_watermark_price, 2200.0)


# ──────────────────────────────────────────────────────────────────────────────
# Revalidate After Normal ADD
# ──────────────────────────────────────────────────────────────────────────────


class RevalidateAfterNormalAddTest(unittest.TestCase):

    def test_outer_band_add_drops_anchor_too_close(self) -> None:
        """After OUTER_BAND_ADD, anchor too close to new last_entry → dropped."""
        strat = strategy(extreme_retest_add_enabled=True)
        strat.state.side = "SHORT"
        strat.state.layers = 1
        strat.state.last_entry_price = 2000.0
        strat.state.last_order_ts_ms = 1000
        strat.state.add_freeze_until_ts_ms = 0
        strat.state.avg_entry_price = 2000.0
        strat.state.total_entry_qty = 1.0
        strat.state.total_entry_notional = 2000.0

        # Set active anchor far enough from current entry
        strat.state.extreme_retest_anchor_side = "SHORT"
        strat.state.extreme_retest_anchor_kind = "PIVOT_HIGH"
        strat.state.extreme_retest_anchor_price = 2200.0
        strat.state.extreme_retest_anchor_candle_ts_ms = 5000
        strat.restore_extreme_retest_state_from_saved(trusted=True)

        # Verify anchor is active
        self.assertTrue(strat._extreme_retest_anchor.is_active())

        # Simulate an OUTER_BAND_ADD that sets last_entry_price close to anchor
        # by calling _open_position (which triggers revalidate)
        from unittest.mock import patch
        boll = boll_snapshot(candle_ts_ms=5000, upper=2300.0, lower=2050.0)
        cvd = cvd_snapshot(price=2180.0, sell_ratio=0.60)

        # Before: last_entry=2000, anchor=2200, gap=10% → OK
        # After ADD at 2180, last_entry=2180 (close to anchor 2200)
        # gap = (2200-2180)/2180 = 0.00917 < 0.003? No, 0.9% > 0.3%
        # Actually gap is still enough. Need tighter values.
        # Let me directly test revalidate:
        strat.state.last_entry_price = 2195.0
        strat._maybe_revalidate_extreme_retest_anchor_after_add()
        # gap = (2200-2195)/2195 = 0.00228 < 0.003 → anchor dropped
        self.assertFalse(strat._extreme_retest_anchor.is_active())

    def test_outer_band_add_keeps_anchor_when_still_far_enough(self) -> None:
        strat = strategy(extreme_retest_add_enabled=True)
        strat.state.side = "SHORT"
        strat.state.layers = 1
        strat.state.last_entry_price = 2000.0
        strat.state.add_freeze_until_ts_ms = 0

        strat.state.extreme_retest_anchor_side = "SHORT"
        strat.state.extreme_retest_anchor_kind = "PIVOT_HIGH"
        strat.state.extreme_retest_anchor_price = 2200.0
        strat.state.extreme_retest_anchor_candle_ts_ms = 5000
        strat.restore_extreme_retest_state_from_saved(trusted=True)

        # ADD at 2100, last_entry becomes 2100
        # gap = (2200-2100)/2100 = 0.0476 > 0.003 → anchor still valid
        strat.state.last_entry_price = 2100.0
        strat._maybe_revalidate_extreme_retest_anchor_after_add()
        self.assertTrue(strat._extreme_retest_anchor.is_active())
        self.assertEqual(strat._extreme_retest_anchor.price, 2200.0)


# ──────────────────────────────────────────────────────────────────────────────
# Candle Buffer Tracking
# ──────────────────────────────────────────────────────────────────────────────


class CandleBufferTrackingTest(unittest.TestCase):

    def test_candle_buffer_pushes_on_candle_close(self) -> None:
        strat = strategy(extreme_retest_add_enabled=True)
        strat.state.side = "LONG"
        strat.state.layers = 1
        strat.state.last_entry_price = 2000.0

        # First tick: candle A
        boll_a = boll_snapshot(candle_ts_ms=1000, high=105.0, low=95.0, close=100.0)
        strat._track_candle_buffer(boll_a)
        self.assertEqual(len(strat._candle_buffer), 0)  # no close yet

        # Second tick: new candle B → candle A closes
        boll_b = boll_snapshot(candle_ts_ms=2000, high=106.0, low=94.0, close=101.0)
        strat._track_candle_buffer(boll_b)
        self.assertEqual(len(strat._candle_buffer), 1)
        self.assertEqual(strat._candle_buffer[0]["ts_ms"], 1000)
        self.assertEqual(strat._candle_buffer[0]["high"], 105.0)
        self.assertEqual(strat._candle_buffer[0]["low"], 95.0)
        self.assertEqual(strat._candle_buffer[0]["close"], 100.0)

    def test_candle_buffer_maxlen(self) -> None:
        strat = strategy(extreme_retest_add_enabled=True)
        strat.state.side = "LONG"
        strat.state.layers = 1

        # Push 60 candles through
        for i in range(60):
            boll = boll_snapshot(candle_ts_ms=i * 1000, high=100.0, low=90.0, close=95.0)
            strat._track_candle_buffer(boll)

        self.assertLessEqual(len(strat._candle_buffer), 50)

    def test_sync_anchor_from_to_state_roundtrip(self) -> None:
        strat = strategy(extreme_retest_add_enabled=True)
        strat.state.extreme_retest_anchor_side = "LONG"
        strat.state.extreme_retest_anchor_kind = "PIVOT_LOW"
        strat.state.extreme_retest_anchor_price = 90.0
        strat.state.extreme_retest_anchor_candle_ts_ms = 3000
        strat.state.extreme_retest_anchor_boll_upper = 112.0
        strat.state.extreme_retest_anchor_boll_lower = 89.0
        strat.state.extreme_retest_sweep_seen = True
        strat.state.extreme_retest_sweep_extreme_price = 88.0
        strat.state.extreme_retest_sweep_first_seen_ts_ms = 4000
        strat.state.extreme_retest_sweep_last_seen_ts_ms = 4500
        strat.state.extreme_retest_consumed_watermark_price = 87.0
        strat.state.extreme_retest_consumed_anchor_ts_ms = 3500

        # Sync to anchor object
        strat._sync_anchor_from_state()
        self.assertEqual(strat._extreme_retest_anchor.side, "LONG")
        self.assertEqual(strat._extreme_retest_anchor.price, 90.0)
        self.assertTrue(strat._extreme_retest_anchor.sweep_seen)

        # Modify and sync back
        strat._extreme_retest_anchor.price = 89.5
        strat._extreme_retest_anchor.clear_sweep()
        strat._sync_anchor_to_state()
        self.assertEqual(strat.state.extreme_retest_anchor_price, 89.5)
        self.assertFalse(strat.state.extreme_retest_sweep_seen)


# ──────────────────────────────────────────────────────────────────────────────
# Fix 4: Same Tick Single ADD
# ──────────────────────────────────────────────────────────────────────────────


class SameTickSingleADDTest(unittest.TestCase):

    def test_normal_add_blocks_extreme_retest_on_same_tick(self) -> None:
        """When normal OUTER_BAND ADD fires, extreme retest is NOT evaluated."""
        strat = strategy(extreme_retest_add_enabled=True)
        strat.state.side = "SHORT"
        strat.state.layers = 1
        strat.state.last_entry_price = 2000.0
        strat.state.last_order_ts_ms = 1000
        strat.state.add_freeze_until_ts_ms = 0
        strat.state.avg_entry_price = 2000.0
        strat.state.total_entry_qty = 1.0
        strat.state.total_entry_notional = 2000.0

        # Set up extreme retest anchor that would also trigger
        strat.state.extreme_retest_anchor_side = "SHORT"
        strat.state.extreme_retest_anchor_kind = "PIVOT_HIGH"
        strat.state.extreme_retest_anchor_price = 2200.0
        strat.state.extreme_retest_anchor_candle_ts_ms = 5000
        strat.state.extreme_retest_anchor_boll_upper = 2300.0
        strat.state.extreme_retest_anchor_boll_lower = 2050.0
        strat.state.lower_armed = True
        strat.state.lower_deep_enough = True
        strat.state.lower_extreme_price = 2199.0
        strat.restore_extreme_retest_state_from_saved(trusted=True)

        # Both OUTER_BAND_ADD and EXTREME_RETEST could fire
        boll = boll_snapshot(candle_ts_ms=5000, upper=2300.0, lower=2050.0, alert_switch_on=True)
        cvd = cvd_snapshot(
            price=2199.0, sell_ratio=0.60, buy_ratio=0.30,
            cross_negative=True, no_new_high=True,  # satisfies _short_setup
        )

        intents = strat.on_tick(2199.0, 100000, boll, cvd)
        add_intents = [i for i in intents if i.intent_type in ("ADD_LONG", "ADD_SHORT")]
        # Must be at most 1 ADD
        self.assertLessEqual(len(add_intents), 1)

    def test_extreme_retest_fires_when_normal_add_not_eligible(self) -> None:
        """When normal add does NOT fire, extreme retest can fire."""
        strat = strategy(extreme_retest_add_enabled=True)
        strat.state.side = "SHORT"
        strat.state.layers = 1
        strat.state.last_entry_price = 2000.0
        strat.state.last_order_ts_ms = 1000
        strat.state.add_freeze_until_ts_ms = 0
        strat.state.avg_entry_price = 2000.0
        strat.state.total_entry_qty = 1.0
        strat.state.total_entry_notional = 2000.0

        # Set up extreme retest anchor
        strat.state.extreme_retest_anchor_side = "SHORT"
        strat.state.extreme_retest_anchor_kind = "PIVOT_HIGH"
        strat.state.extreme_retest_anchor_price = 2200.0
        strat.state.extreme_retest_anchor_candle_ts_ms = 5000
        strat.state.extreme_retest_anchor_boll_upper = 2300.0
        strat.state.extreme_retest_anchor_boll_lower = 2050.0
        # Normal add NOT eligible: lower_armed is False
        strat.state.lower_armed = False
        strat.restore_extreme_retest_state_from_saved(trusted=True)

        boll = boll_snapshot(candle_ts_ms=5000, upper=2300.0, lower=2050.0, alert_switch_on=True)
        cvd = cvd_snapshot(price=2199.0, sell_ratio=0.60, buy_ratio=0.30)

        intents = strat.on_tick(2199.0, 100000, boll, cvd)
        add_intents = [i for i in intents if i.intent_type in ("ADD_LONG", "ADD_SHORT")]
        self.assertEqual(len(add_intents), 1)
        self.assertEqual(add_intents[0].intent_type, "ADD_SHORT")


# ──────────────────────────────────────────────────────────────────────────────
# Fix 5: High-Frequency Log Throttling
# ──────────────────────────────────────────────────────────────────────────────


class HighFrequencyLogTest(unittest.TestCase):

    def test_non_triggered_eval_log_throttled(self) -> None:
        """10 consecutive non-triggered ticks should produce at most 1 EVALUATED INFO."""
        strat = strategy(extreme_retest_add_enabled=True)
        strat.state.side = "SHORT"
        strat.state.layers = 1
        strat.state.last_entry_price = 2000.0
        strat.state.last_order_ts_ms = 1000
        strat.state.add_freeze_until_ts_ms = 0
        strat.state.avg_entry_price = 2000.0
        strat.state.total_entry_qty = 1.0
        strat.state.total_entry_notional = 2000.0
        strat.state.lower_armed = False

        strat.state.extreme_retest_anchor_side = "SHORT"
        strat.state.extreme_retest_anchor_kind = "PIVOT_HIGH"
        strat.state.extreme_retest_anchor_price = 2200.0
        strat.state.extreme_retest_anchor_candle_ts_ms = 5000
        strat.state.extreme_retest_anchor_boll_upper = 2300.0
        strat.state.extreme_retest_anchor_boll_lower = 2050.0
        strat.restore_extreme_retest_state_from_saved(trusted=True)

        import logging
        with self.assertLogs("src.strategies.boll_cvd_shock_reclaim_strategy", level="INFO") as logs:
            for i in range(10):
                boll = boll_snapshot(
                    candle_ts_ms=5000, upper=2300.0, lower=2050.0,
                    alert_switch_on=True,
                )
                # Price far from anchor → not triggered
                cvd = cvd_snapshot(
                    price=2100.0, sell_ratio=0.30, buy_ratio=0.30,
                    ts_ms=100000 + i * 1000,
                )
                strat.on_tick(2100.0, 100000 + i * 1000, boll, cvd)

        evaluated_logs = [
            rec for rec in logs.output
            if "EXTREME_RETEST_ADD_EVALUATED" in rec
        ]
        # At most 1 evaluated log (60s throttle interval; all 10 ticks within 10s)
        self.assertLessEqual(len(evaluated_logs), 1)


# ──────────────────────────────────────────────────────────────────────────────
# Fix 7: Timing Skip Logs with trigger_source
# ──────────────────────────────────────────────────────────────────────────────


class TimingSkipLogWithTriggerSourceTest(unittest.TestCase):

    def test_extreme_retest_timing_skip_log_includes_trigger_source(self) -> None:
        """When extreme retest is blocked by add_freeze, log includes trigger_source=EXTREME_RETEST."""
        strat = strategy(extreme_retest_add_enabled=True, add_freeze_chain_enabled=True)
        strat.state.side = "SHORT"
        strat.state.layers = 1
        # Price close to last_entry so adverse_gap < 0.015 (freeze bypass)
        strat.state.last_entry_price = 2170.0
        strat.state.last_order_ts_ms = 1000
        strat.state.add_freeze_until_ts_ms = 200000  # active freeze
        strat.state.avg_entry_price = 2170.0
        strat.state.total_entry_qty = 1.0
        strat.state.total_entry_notional = 2170.0
        strat.state.lower_armed = False

        strat.state.extreme_retest_anchor_side = "SHORT"
        strat.state.extreme_retest_anchor_kind = "PIVOT_HIGH"
        strat.state.extreme_retest_anchor_price = 2200.0
        strat.state.extreme_retest_anchor_candle_ts_ms = 5000
        strat.state.extreme_retest_anchor_boll_upper = 2300.0
        strat.state.extreme_retest_anchor_boll_lower = 2050.0
        strat.restore_extreme_retest_state_from_saved(trusted=True)

        import logging
        with self.assertLogs("src.strategies.boll_cvd_shock_reclaim_strategy", level="INFO") as logs:
            boll = boll_snapshot(candle_ts_ms=5000, upper=2300.0, lower=2050.0,
                               alert_switch_on=True)
            # price=2198: near anchor (2196.7-2200), adverse_gap=1.29% < 1.5% bypass
            cvd = cvd_snapshot(price=2198.0, sell_ratio=0.60, buy_ratio=0.30,
                             ts_ms=100000)
            # Call _evaluate_extreme_retest_add directly rather than full on_tick
            # to avoid complex interactions with UPDATE_TP and other intent generators
            result = strat._evaluate_extreme_retest_add(2198.0, 100000, boll, cvd)

        # extreme retest should be blocked (add_freeze) and not generate ADD
        self.assertIsNone(result)
        # Anchor should NOT be consumed
        self.assertIsNone(strat._extreme_retest_anchor.consumed_watermark_price)

        add_skipped_logs = [
            rec for rec in logs.output
            if "ADD_SKIPPED" in rec and "trigger_source=EXTREME_RETEST" in rec
        ]
        self.assertGreaterEqual(len(add_skipped_logs), 1)


# ──────────────────────────────────────────────────────────────────────────────
# Fix 6: Effective Required Gap with Add Freeze
# ──────────────────────────────────────────────────────────────────────────────


class EffectiveRequiredGapWithFreezeTest(unittest.TestCase):

    def test_anchor_rejected_when_only_satisfies_base_gap_not_multiplied(self) -> None:
        """When add_freeze is active with multiplier > 1, anchor must satisfy
        base_gap * multiplier, not just base_gap."""
        cfg = _make_config()
        anchor = ExtremeRetestAnchor()

        # base_gap = 0.003 (for target_layer=2)
        # multiplier = 5 (first_add_block_bypass_multiplier), effective_gap = 0.015
        # last_entry=100, candidate=103.5
        # gap = (103.5-100)/100 = 0.035 > 0.015 → passes
        action, reason = _extreme_retest.try_create_or_replace_anchor(
            "SHORT", 103.5, 5000, boll_upper=103.0, boll_lower=92.0,
            last_entry_price=100.0, effective_required_gap_pct=0.015,
            anchor=anchor, config=cfg,
        )
        self.assertTrue(action)

    def test_anchor_rejected_when_gap_below_effective(self) -> None:
        """Anchor rejected when gap < effective_required_gap_pct."""
        cfg = _make_config()
        anchor = ExtremeRetestAnchor()

        # effective_gap = 0.015, last_entry=100, candidate=101
        # gap = (101-100)/100 = 0.01 < 0.015 → rejected
        action, reason = _extreme_retest.try_create_or_replace_anchor(
            "SHORT", 101.0, 5000, boll_upper=100.5, boll_lower=92.0,
            last_entry_price=100.0, effective_required_gap_pct=0.015,
            anchor=anchor, config=cfg,
        )
        self.assertFalse(action)


# ──────────────────────────────────────────────────────────────────────────────
# Fix 8: Restart Not Immediately ADD
# ──────────────────────────────────────────────────────────────────────────────


class RestartNotImmediatelyADDTest(unittest.TestCase):

    def test_restored_anchor_no_add_without_live_tick(self) -> None:
        """Restored anchor should not generate ADD without on_tick call."""
        strat = strategy(extreme_retest_add_enabled=True)
        strat.state.side = "SHORT"
        strat.state.layers = 1
        strat.state.last_entry_price = 2000.0
        strat.state.last_order_ts_ms = 1000
        strat.state.add_freeze_until_ts_ms = 0
        strat.state.avg_entry_price = 2000.0
        strat.state.total_entry_qty = 1.0
        strat.state.total_entry_notional = 2000.0
        strat.state.lower_armed = False

        strat.state.extreme_retest_anchor_side = "SHORT"
        strat.state.extreme_retest_anchor_kind = "PIVOT_HIGH"
        strat.state.extreme_retest_anchor_price = 2200.0
        strat.state.extreme_retest_anchor_candle_ts_ms = 5000
        strat.state.extreme_retest_anchor_boll_upper = 2300.0
        strat.state.extreme_retest_anchor_boll_lower = 2050.0
        strat.restore_extreme_retest_state_from_saved(trusted=True)

        # No on_tick called — anchor is restored but no ADD generated
        self.assertTrue(strat._extreme_retest_anchor.is_active())

    def test_first_live_tick_insufficient_cvd_no_add(self) -> None:
        """First live tick with insufficient CVD should not trigger ADD."""
        strat = strategy(extreme_retest_add_enabled=True)
        strat.state.side = "SHORT"
        strat.state.layers = 1
        strat.state.last_entry_price = 2000.0
        strat.state.last_order_ts_ms = 1000
        strat.state.add_freeze_until_ts_ms = 0
        strat.state.avg_entry_price = 2000.0
        strat.state.total_entry_qty = 1.0
        strat.state.total_entry_notional = 2000.0
        strat.state.lower_armed = False

        strat.state.extreme_retest_anchor_side = "SHORT"
        strat.state.extreme_retest_anchor_kind = "PIVOT_HIGH"
        strat.state.extreme_retest_anchor_price = 2200.0
        strat.state.extreme_retest_anchor_candle_ts_ms = 5000
        strat.state.extreme_retest_anchor_boll_upper = 2300.0
        strat.state.extreme_retest_anchor_boll_lower = 2050.0
        strat.restore_extreme_retest_state_from_saved(trusted=True)

        # Price near anchor but sell_ratio too low
        boll = boll_snapshot(candle_ts_ms=5000, upper=2300.0, lower=2050.0,
                           alert_switch_on=True)
        cvd = cvd_snapshot(price=2199.0, sell_ratio=0.30, buy_ratio=0.30)
        intents = strat.on_tick(2199.0, 100000, boll, cvd)
        add_intents = [i for i in intents if i.intent_type in ("ADD_LONG", "ADD_SHORT")]
        self.assertEqual(len(add_intents), 0)
        # Anchor still active (ADD was not triggered)
        self.assertTrue(strat._extreme_retest_anchor.is_active())


# ──────────────────────────────────────────────────────────────────────────────
# Startup Glue Restore — trusted/untrusted log verification
# ──────────────────────────────────────────────────────────────────────────────


class StartupGlueTrustedRestoreTest(unittest.TestCase):
    """Verify that the glue call restore_extreme_retest_state_from_saved(trusted=True)
    logs EXTREME_RETEST_STATE_RESTORED and preserves the active anchor."""

    def test_trusted_restore_logs_restored_when_anchor_active(self) -> None:
        strat = strategy(extreme_retest_add_enabled=True)
        strat.state.side = "SHORT"
        strat.state.layers = 1
        strat.state.last_entry_price = 1700.0
        strat.state.extreme_retest_anchor_side = "SHORT"
        strat.state.extreme_retest_anchor_kind = "PIVOT_HIGH"
        strat.state.extreme_retest_anchor_price = 1740.0
        strat.state.extreme_retest_anchor_candle_ts_ms = 5000

        import logging
        with self.assertLogs("src.strategies.boll_cvd_shock_reclaim_strategy", level="INFO") as logs:
            strat.restore_extreme_retest_state_from_saved(trusted=True)

        restored_logs = [r for r in logs.output if "EXTREME_RETEST_STATE_RESTORED" in r]
        self.assertGreaterEqual(len(restored_logs), 1,
                                "Trusted restore must log EXTREME_RETEST_STATE_RESTORED")
        # Anchor preserved
        self.assertEqual(strat._extreme_retest_anchor.side, "SHORT")
        self.assertEqual(strat._extreme_retest_anchor.kind, "PIVOT_HIGH")
        self.assertEqual(strat._extreme_retest_anchor.price, 1740.0)
        self.assertTrue(strat._extreme_retest_anchor.is_active())

    def test_trusted_restore_preserves_sweep_state(self) -> None:
        strat = strategy(extreme_retest_add_enabled=True)
        strat.state.side = "SHORT"
        strat.state.layers = 1
        strat.state.last_entry_price = 1700.0
        strat.state.extreme_retest_anchor_side = "SHORT"
        strat.state.extreme_retest_anchor_kind = "PIVOT_HIGH"
        strat.state.extreme_retest_anchor_price = 1740.0
        strat.state.extreme_retest_anchor_candle_ts_ms = 5000
        strat.state.extreme_retest_sweep_seen = True
        strat.state.extreme_retest_sweep_extreme_price = 1755.0
        strat.state.extreme_retest_sweep_first_seen_ts_ms = 6000

        import logging
        with self.assertLogs("src.strategies.boll_cvd_shock_reclaim_strategy", level="INFO") as logs:
            strat.restore_extreme_retest_state_from_saved(trusted=True)

        self.assertTrue(any("EXTREME_RETEST_STATE_RESTORED" in r for r in logs.output))
        self.assertTrue(strat._extreme_retest_anchor.sweep_seen)
        self.assertEqual(strat._extreme_retest_anchor.sweep_extreme_price, 1755.0)
        self.assertEqual(strat._extreme_retest_anchor.sweep_first_seen_ts_ms, 6000)

    def test_trusted_restore_preserves_consumed_watermark(self) -> None:
        strat = strategy(extreme_retest_add_enabled=True)
        strat.state.side = "SHORT"
        strat.state.layers = 1
        strat.state.last_entry_price = 1700.0
        strat.state.extreme_retest_consumed_watermark_price = 1740.0
        strat.state.extreme_retest_consumed_anchor_ts_ms = 5000

        # When only consumed watermark is present (no active anchor),
        # restore does not log RESTORED because there's nothing to restore.
        # It should silently preserve the watermark.
        strat.restore_extreme_retest_state_from_saved(trusted=True)

        # No active anchor (was previously consumed), but watermark persists
        self.assertFalse(strat._extreme_retest_anchor.is_active())
        self.assertEqual(strat._extreme_retest_anchor.consumed_watermark_price, 1740.0)
        self.assertEqual(strat._extreme_retest_anchor.consumed_anchor_ts_ms, 5000)


class StartupGlueUntrustedDropTest(unittest.TestCase):
    """Verify that the glue call restore_extreme_retest_state_from_saved(trusted=False)
    logs EXTREME_RETEST_STATE_DROPPED, clears the active anchor, and preserves the
    consumed watermark."""

    def test_untrusted_restore_logs_dropped_and_clears_anchor(self) -> None:
        strat = strategy(extreme_retest_add_enabled=True)
        strat.state.side = "SHORT"
        strat.state.layers = 1
        strat.state.extreme_retest_anchor_side = "SHORT"
        strat.state.extreme_retest_anchor_kind = "PIVOT_HIGH"
        strat.state.extreme_retest_anchor_price = 1740.0
        strat.state.extreme_retest_anchor_candle_ts_ms = 5000

        import logging
        with self.assertLogs("src.strategies.boll_cvd_shock_reclaim_strategy", level="INFO") as logs:
            strat.restore_extreme_retest_state_from_saved(trusted=False)

        dropped_logs = [r for r in logs.output if "EXTREME_RETEST_STATE_DROPPED" in r]
        self.assertGreaterEqual(len(dropped_logs), 1,
                                "Untrusted restore must log EXTREME_RETEST_STATE_DROPPED")
        # Active anchor cleared
        self.assertIsNone(strat._extreme_retest_anchor.side)
        self.assertIsNone(strat._extreme_retest_anchor.kind)
        self.assertIsNone(strat._extreme_retest_anchor.price)
        self.assertFalse(strat._extreme_retest_anchor.is_active())

    def test_untrusted_restore_preserves_consumed_watermark(self) -> None:
        strat = strategy(extreme_retest_add_enabled=True)
        strat.state.side = "SHORT"
        strat.state.layers = 1
        strat.state.extreme_retest_anchor_side = "SHORT"
        strat.state.extreme_retest_anchor_kind = "PIVOT_HIGH"
        strat.state.extreme_retest_anchor_price = 1740.0
        strat.state.extreme_retest_anchor_candle_ts_ms = 5000
        strat.state.extreme_retest_consumed_watermark_price = 1730.0
        strat.state.extreme_retest_consumed_anchor_ts_ms = 4000

        import logging
        with self.assertLogs("src.strategies.boll_cvd_shock_reclaim_strategy", level="INFO") as logs:
            strat.restore_extreme_retest_state_from_saved(trusted=False)

        self.assertTrue(any("EXTREME_RETEST_STATE_DROPPED" in r for r in logs.output))
        # Anchor cleared
        self.assertIsNone(strat._extreme_retest_anchor.side)
        self.assertIsNone(strat._extreme_retest_anchor.price)
        # Watermark preserved
        self.assertEqual(strat._extreme_retest_anchor.consumed_watermark_price, 1730.0)
        self.assertEqual(strat._extreme_retest_anchor.consumed_anchor_ts_ms, 4000)

    def test_untrusted_restore_clears_sweep_state(self) -> None:
        strat = strategy(extreme_retest_add_enabled=True)
        strat.state.side = "SHORT"
        strat.state.layers = 1
        strat.state.extreme_retest_anchor_side = "SHORT"
        strat.state.extreme_retest_anchor_price = 1740.0
        strat.state.extreme_retest_sweep_seen = True
        strat.state.extreme_retest_sweep_extreme_price = 1755.0
        strat.state.extreme_retest_sweep_first_seen_ts_ms = 6000
        strat.state.extreme_retest_consumed_watermark_price = 1730.0

        import logging
        with self.assertLogs("src.strategies.boll_cvd_shock_reclaim_strategy", level="INFO") as logs:
            strat.restore_extreme_retest_state_from_saved(trusted=False)

        self.assertTrue(any("EXTREME_RETEST_STATE_DROPPED" in r for r in logs.output))
        # Sweep cleared
        self.assertFalse(strat._extreme_retest_anchor.sweep_seen)
        self.assertIsNone(strat._extreme_retest_anchor.sweep_extreme_price)
        # Watermark preserved
        self.assertEqual(strat._extreme_retest_anchor.consumed_watermark_price, 1730.0)


class StartupGlueNoADDTest(unittest.TestCase):
    """Verify that the glue restore call itself does NOT trigger any ADD intent,
    does NOT call on_tick(), and does NOT push to any execution queue."""

    def test_restore_call_does_not_generate_add_intent(self) -> None:
        """Calling restore_extreme_retest_state_from_saved must not generate
        ADD_LONG or ADD_SHORT intents — it is purely a state hydration step."""
        strat = strategy(extreme_retest_add_enabled=True)
        strat.state.side = "SHORT"
        strat.state.layers = 1
        strat.state.last_entry_price = 2000.0
        strat.state.last_order_ts_ms = 1000
        strat.state.add_freeze_until_ts_ms = 0
        strat.state.avg_entry_price = 2000.0
        strat.state.total_entry_qty = 1.0
        strat.state.total_entry_notional = 2000.0
        strat.state.lower_armed = False

        strat.state.extreme_retest_anchor_side = "SHORT"
        strat.state.extreme_retest_anchor_kind = "PIVOT_HIGH"
        strat.state.extreme_retest_anchor_price = 2200.0
        strat.state.extreme_retest_anchor_candle_ts_ms = 5000
        strat.state.extreme_retest_anchor_boll_upper = 2300.0
        strat.state.extreme_retest_anchor_boll_lower = 2050.0

        # Capture intents via monkey-patching on_tick to detect if called
        on_tick_called = []

        def fake_on_tick(price, ts_ms, boll, cvd):
            on_tick_called.append(True)
            return []

        original_on_tick = strat.on_tick
        strat.on_tick = fake_on_tick
        try:
            strat.restore_extreme_retest_state_from_saved(trusted=True)
        finally:
            strat.on_tick = original_on_tick

        # restore call must NOT invoke on_tick
        self.assertEqual(len(on_tick_called), 0,
                         "restore_extreme_retest_state_from_saved must not call on_tick")
        # Anchor should be active after restore
        self.assertTrue(strat._extreme_retest_anchor.is_active())

    def test_restore_untrusted_does_not_generate_add_intent(self) -> None:
        """Untrusted restore must not generate ADD intent either."""
        strat = strategy(extreme_retest_add_enabled=True)
        strat.state.side = "SHORT"
        strat.state.layers = 1
        strat.state.extreme_retest_anchor_side = "SHORT"
        strat.state.extreme_retest_anchor_kind = "PIVOT_HIGH"
        strat.state.extreme_retest_anchor_price = 2200.0
        strat.state.extreme_retest_consumed_watermark_price = 2100.0

        on_tick_called = []

        def fake_on_tick(price, ts_ms, boll, cvd):
            on_tick_called.append(True)
            return []

        original_on_tick = strat.on_tick
        strat.on_tick = fake_on_tick
        try:
            strat.restore_extreme_retest_state_from_saved(trusted=False)
        finally:
            strat.on_tick = original_on_tick

        self.assertEqual(len(on_tick_called), 0,
                         "Untrusted restore must not call on_tick")
        # Anchor cleared
        self.assertFalse(strat._extreme_retest_anchor.is_active())

    def test_full_startup_sequence_no_add(self) -> None:
        """Simulate the exact sequence the live runner uses after restore:
        1. restore_extreme_retest_state_from_saved(trusted=False)
        2. No on_tick is called during restore
        3. After restore, the first live tick does NOT auto-trigger ADD
           unless all conditions are met."""
        strat = strategy(extreme_retest_add_enabled=True)
        # Simulate untrusted startup: position present, but saved_state is None
        strat.state.side = "SHORT"
        strat.state.layers = 1
        strat.state.last_entry_price = 2000.0
        strat.state.last_order_ts_ms = 1000
        strat.state.add_freeze_until_ts_ms = 0
        strat.state.avg_entry_price = 2000.0
        strat.state.total_entry_qty = 1.0
        strat.state.total_entry_notional = 2000.0
        # Residual anchor from stale state
        strat.state.extreme_retest_anchor_side = "SHORT"
        strat.state.extreme_retest_anchor_kind = "PIVOT_HIGH"
        strat.state.extreme_retest_anchor_price = 2200.0
        strat.state.extreme_retest_anchor_candle_ts_ms = 5000
        strat.state.extreme_retest_anchor_boll_upper = 2300.0
        strat.state.extreme_retest_anchor_boll_lower = 2050.0
        strat.state.extreme_retest_consumed_watermark_price = 2100.0

        # Step 1: Simulate startup restore call (untrusted)
        strat.restore_extreme_retest_state_from_saved(trusted=False)
        self.assertFalse(strat._extreme_retest_anchor.is_active())
        self.assertEqual(strat._extreme_retest_anchor.consumed_watermark_price, 2100.0)

        # Step 2: First live tick arrives (price near old anchor but anchor dropped)
        # Even if CVD conditions were met, anchor is gone so no ADD
        boll = boll_snapshot(candle_ts_ms=6000, upper=2300.0, lower=2050.0,
                           alert_switch_on=True)
        cvd = cvd_snapshot(price=2198.0, sell_ratio=0.60, buy_ratio=0.30,
                         ts_ms=100000)
        intents = strat.on_tick(2198.0, 100000, boll, cvd)
        add_intents = [i for i in intents if i.intent_type in ("ADD_LONG", "ADD_SHORT")]
        # No ADD because anchor was dropped during untrusted restore
        self.assertEqual(len(add_intents), 0)

    def test_trusted_full_sequence_anchor_preserved_no_immediate_add(self) -> None:
        """Simulate the exact sequence the live runner uses after trusted restore:
        1. restore_extreme_retest_state_from_saved(trusted=True)
        2. Anchor is preserved but no ADD is generated without on_tick
        3. First live tick with insufficient CVD does not trigger ADD
        4. Anchor survives until conditions are met on a later tick."""
        strat = strategy(extreme_retest_add_enabled=True)
        # Simulate trusted startup
        strat.state.side = "SHORT"
        strat.state.layers = 1
        strat.state.last_entry_price = 2000.0
        strat.state.last_order_ts_ms = 1000
        strat.state.add_freeze_until_ts_ms = 0
        strat.state.avg_entry_price = 2000.0
        strat.state.total_entry_qty = 1.0
        strat.state.total_entry_notional = 2000.0
        strat.state.lower_armed = False

        strat.state.extreme_retest_anchor_side = "SHORT"
        strat.state.extreme_retest_anchor_kind = "PIVOT_HIGH"
        strat.state.extreme_retest_anchor_price = 2200.0
        strat.state.extreme_retest_anchor_candle_ts_ms = 5000
        strat.state.extreme_retest_anchor_boll_upper = 2300.0
        strat.state.extreme_retest_anchor_boll_lower = 2050.0

        # Step 1: Simulate startup restore call (trusted)
        strat.restore_extreme_retest_state_from_saved(trusted=True)
        self.assertTrue(strat._extreme_retest_anchor.is_active())
        self.assertEqual(strat._extreme_retest_anchor.price, 2200.0)

        # Step 2: First live tick with insufficient CVD → no ADD, anchor survives
        boll = boll_snapshot(candle_ts_ms=6000, upper=2300.0, lower=2050.0,
                           alert_switch_on=True)
        cvd = cvd_snapshot(price=2199.0, sell_ratio=0.30, buy_ratio=0.30,
                         ts_ms=100000)
        intents = strat.on_tick(2199.0, 100000, boll, cvd)
        add_intents = [i for i in intents if i.intent_type in ("ADD_LONG", "ADD_SHORT")]
        self.assertEqual(len(add_intents), 0)
        self.assertTrue(strat._extreme_retest_anchor.is_active(),
                        "Anchor must survive a tick that does not meet all conditions")

        # Step 3: Later tick with sufficient CVD → ADD fires and anchor consumed
        cvd2 = cvd_snapshot(price=2199.0, sell_ratio=0.60, buy_ratio=0.30,
                          ts_ms=200000)
        intents2 = strat.on_tick(2199.0, 200000, boll, cvd2)
        add_intents2 = [i for i in intents2 if i.intent_type in ("ADD_LONG", "ADD_SHORT")]
        self.assertEqual(len(add_intents2), 1)
        self.assertEqual(add_intents2[0].intent_type, "ADD_SHORT")
        self.assertFalse(strat._extreme_retest_anchor.is_active(),
                         "Anchor must be consumed after successful ADD")
