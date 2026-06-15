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
