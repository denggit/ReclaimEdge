"""Tests for trend breakout live state persistence via LiveStateStore."""

import tempfile
import unittest
from pathlib import Path

from src.reporting.live_state_store import LivePositionState, LiveStateStore
from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState


class TestTrendBreakoutLiveState(unittest.TestCase):
    """Verify that trend breakout fields round-trip through LiveStateStore."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = LiveStateStore(Path(self._tmp.name) / "live_state.json")

    def tearDown(self):
        self._tmp.cleanup()

    # ── helpers ────────────────────────────────────────────────────────

    def _make_strategy_state(self, **overrides) -> StrategyPositionState:
        state = StrategyPositionState()
        for key, value in overrides.items():
            setattr(state, key, value)
        return state

    # ── tests ──────────────────────────────────────────────────────────

    def test_entry_regime_round_trip(self):
        """entry_regime survives save → load."""
        strat_state = self._make_strategy_state(
            side="LONG", layers=1,
            entry_regime="TREND_BREAKOUT",
            trend_trailing_sl_price=2950.0,
            trend_last_sl_update_ts_ms=1000000,
        )
        live = self.store.from_strategy_state(
            position_id="test:1", symbol="ETH-USDT-SWAP",
            strategy_state=strat_state, cash_before_position=None,
        )
        self.store.save(live)

        loaded = self.store.load()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.entry_regime, "TREND_BREAKOUT")
        self.assertEqual(loaded.trend_trailing_sl_price, 2950.0)
        self.assertEqual(loaded.trend_last_sl_update_ts_ms, 1000000)

    def test_entry_regime_mean_reversion(self):
        """MEAN_REVERSION regime round-trips."""
        strat_state = self._make_strategy_state(
            side="SHORT", layers=1,
            entry_regime="MEAN_REVERSION",
        )
        live = self.store.from_strategy_state(
            position_id="test:2", symbol="ETH-USDT-SWAP",
            strategy_state=strat_state, cash_before_position=None,
        )
        self.store.save(live)

        loaded = self.store.load()
        self.assertEqual(loaded.entry_regime, "MEAN_REVERSION")

    def test_entry_regime_none_default(self):
        """entry_regime None is the default for old or unset state."""
        strat_state = self._make_strategy_state(side="LONG", layers=1)
        live = self.store.from_strategy_state(
            position_id="test:3", symbol="ETH-USDT-SWAP",
            strategy_state=strat_state, cash_before_position=None,
        )
        self.store.save(live)

        loaded = self.store.load()
        self.assertIsNone(loaded.entry_regime)
        self.assertIsNone(loaded.trend_trailing_sl_price)
        self.assertEqual(loaded.trend_last_sl_update_ts_ms, 0)

    def test_old_state_without_trend_fields_loads(self):
        """Old live_state.json without trend fields must not fail on load.

        ``LivePositionState`` uses field defaults for missing keys, so
        old state restores cleanly.
        """
        # Write a minimal JSON that lacks trend fields (simulating old state)
        self.store.path.write_text("""
{
  "side": "LONG",
  "layers": 1,
  "tp_plan": "SINGLE",
  "avg_entry_price": 3000.0
}
""")
        loaded = self.store.load()
        self.assertIsNotNone(loaded)
        # Old state → trend fields default to None / 0
        self.assertIsNone(loaded.entry_regime)
        self.assertIsNone(loaded.trend_trailing_sl_price)
        self.assertEqual(loaded.trend_last_sl_update_ts_ms, 0)

    def test_trend_trailing_sl_price_none_round_trips(self):
        """None trend_sl round-trips without becoming 0."""
        strat_state = self._make_strategy_state(
            side="LONG", layers=1,
            entry_regime="TREND_BREAKOUT",
            trend_trailing_sl_price=None,
        )
        live = self.store.from_strategy_state(
            position_id="test:4", symbol="ETH-USDT-SWAP",
            strategy_state=strat_state, cash_before_position=None,
        )
        self.store.save(live)
        loaded = self.store.load()
        self.assertIsNone(loaded.trend_trailing_sl_price)

    def test_empty_json_does_not_crash(self):
        """Corrupt / empty JSON should return None, not crash."""
        self.store.path.write_text("")
        loaded = self.store.load()
        self.assertIsNone(loaded)
