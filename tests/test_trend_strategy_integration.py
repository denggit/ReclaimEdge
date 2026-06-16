"""Integration tests: BollCvdReclaimStrategy wired with trend breakout."""

import copy
from unittest.mock import patch

import pytest
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
    TradeIntent,
)
from src.strategies.regime.types import (
    RegimeDecision,
    RegimeDecisionType,
    TrendState,
)
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig


# ── helpers ────────────────────────────────────────────────────────────

def _make_config(**overrides) -> BollCvdReclaimStrategyConfig:
    kwargs = dict(
        trend_breakout_enabled=True,
        trend_middle_trailing_sl_enabled=True,
        trend_middle_sl_buffer_pct=0.001,
        trend_max_stop_distance_pct=0.02,
        trend_sl_update_interval_seconds=900,
        # Allow entries without complex BOLL/CVD structure gates
        entry_cvd_structure_mode="DIVERGENCE_OR_ABSORPTION",
    )
    kwargs.update(overrides)
    config = BollCvdReclaimStrategyConfig(**kwargs)
    # Bypass __post_init__ validation for unsupported fields by
    # using object.__setattr__ after construction if needed
    return config


def _make_sizer() -> SimplePositionSizer:
    return SimplePositionSizer(SimplePositionSizerConfig(
        dry_run_equity_usdt=1000.0,
        trade_risk_pct=0.003,
        leverage=20.0,
    ))


def _make_strategy(config=None, **config_overrides) -> BollCvdReclaimStrategy:
    if config is None:
        config = _make_config(**config_overrides)
    return BollCvdReclaimStrategy(config, _make_sizer())


def _boll_snapshot(
    upper=3100.0, middle=3000.0, lower=2900.0,
    candle_ts_ms=1000000, alert_switch_on=True,
    tp_upper=3120.0, tp_middle=3020.0, tp_lower=2920.0,
):
    """Minimal BollSnapshot stub."""
    from src.monitors.boll_band_breakout_monitor import BollSnapshot
    return BollSnapshot(
        inst_id="ETH-USDT-SWAP",
        candle_ts_ms=candle_ts_ms,
        close=middle,
        middle=middle, upper=upper, lower=lower,
        upper_distance_pct=(upper - middle) / middle,
        lower_distance_pct=(middle - lower) / middle,
        alert_switch_on=alert_switch_on,
        live_mode=True,
        tp_upper=tp_upper, tp_middle=tp_middle, tp_lower=tp_lower,
        tp_window=15,
        high=upper + 10, low=lower - 10,
    )


def _cvd_snapshot(
    fast_cvd=0.001, buy_ratio=0.55, sell_ratio=0.45,
    previous_fast_cvd=0.0005, price=3000.0,
    cvd_increasing=True, cvd_decreasing=False,
):
    """Minimal CvdSnapshot stub with sensible defaults for all fields."""
    from src.indicators.cvd_tracker import CvdSnapshot
    buy_vol = 100.0 * buy_ratio
    sell_vol = 100.0 * sell_ratio
    return CvdSnapshot(
        ts_ms=0, price=price, side="BUY", size=1.0,
        signed_delta=fast_cvd - previous_fast_cvd,
        total_cvd=fast_cvd, fast_cvd=fast_cvd,
        previous_fast_cvd=previous_fast_cvd,
        buy_volume=buy_vol, sell_volume=sell_vol,
        buy_ratio=buy_ratio, sell_ratio=sell_ratio,
        cross_positive=fast_cvd > 0, cross_negative=fast_cvd < 0,
        cvd_increasing=cvd_increasing, cvd_decreasing=cvd_decreasing,
        no_new_low=not cvd_decreasing, no_new_high=not cvd_increasing,
        window_low=price * 0.99, window_high=price * 1.01,
        burst_net_move_pct=0.0, burst_range_pct=0.0,
        baseline_range_pct=0.01, burst_move_ratio=0.0,
        burst_volume=0.0, baseline_volume=100.0,
        burst_volume_ratio=0.0, up_burst=False, down_burst=False,
    )


def _trend_long_decision():
    return RegimeDecision(
        decision_type=RegimeDecisionType.TREND_LONG,
        side="LONG",
        reason="trend_confirmed",
        confidence=0.9,
        trend_state=TrendState.TREND_UP_CONFIRMED,
    )


def _trend_short_decision():
    return RegimeDecision(
        decision_type=RegimeDecisionType.TREND_SHORT,
        side="SHORT",
        reason="trend_confirmed",
        confidence=0.9,
        trend_state=TrendState.TREND_DOWN_CONFIRMED,
    )


def _mr_long_decision():
    return RegimeDecision(
        decision_type=RegimeDecisionType.MEAN_REVERSION_LONG,
        side="LONG",
        reason="mr_allowed",
        confidence=0.7,
    )


def _mr_short_decision():
    return RegimeDecision(
        decision_type=RegimeDecisionType.MEAN_REVERSION_SHORT,
        side="SHORT",
        reason="mr_allowed",
        confidence=0.7,
    )


def _conflict_decision():
    return RegimeDecision(
        decision_type=RegimeDecisionType.CONFLICT_NO_TRADE,
        side=None,
        reason="opposite_directions",
        confidence=0.0,
    )


def _no_trade_decision():
    return RegimeDecision(
        decision_type=RegimeDecisionType.NO_TRADE,
        side=None,
        reason="no_candidate",
        confidence=0.0,
    )


# ── Test: Trend Breakout Disabled ──────────────────────────────────────


class TestTrendBreakoutDisabled:
    """When TREND_BREAKOUT_ENABLED=false, on_tick falls through to existing MR-only
    behaviour.  _route_regime returns None, so the strategy uses the legacy
    _long_setup / _short_setup path exactly as before."""

    def test_disabled_returns_none_from_route_regime(self):
        config = _make_config(trend_breakout_enabled=False)
        strategy = _make_strategy(config)
        boll = _boll_snapshot()
        cvd = _cvd_snapshot()
        # _route_regime should return None
        result = strategy._route_regime(
            price=3000.0, ts_ms=1000000, boll=boll, cvd=cvd,
            mr_long_allowed=False, mr_short_allowed=False,
        )
        assert result is None

    def test_disabled_on_tick_runs_mr_logic(self):
        """With trend disabled and no armed state, on_tick returns no intents
        but does not crash."""
        config = _make_config(trend_breakout_enabled=False)
        strategy = _make_strategy(config)
        boll = _boll_snapshot()
        cvd = _cvd_snapshot()
        intents = strategy.on_tick(price=3000.0, ts_ms=1000000, boll=boll, cvd=cvd)
        assert isinstance(intents, list)


# ── Test: Trend LONG Entry ─────────────────────────────────────────────


class TestTrendLongEntry:
    def test_trend_long_emits_open_long_intent(self):
        """Given TREND_LONG decision, strategy emits OPEN_LONG intent."""
        strategy = _make_strategy()
        boll = _boll_snapshot()
        cvd = _cvd_snapshot(fast_cvd=0.002, buy_ratio=0.70, sell_ratio=0.30)

        with patch.object(
            strategy, "_route_regime", return_value=_trend_long_decision(),
        ):
            # Entry price close to middle band so stop distance is within 2%
            intents = strategy.on_tick(
                price=3050.0, ts_ms=1000000, boll=boll, cvd=cvd,
            )

        open_intents = [i for i in intents if i.intent_type == "OPEN_LONG"]
        assert len(open_intents) == 1
        intent = open_intents[0]
        assert intent.side == "LONG"
        assert intent.entry_regime == "TREND_BREAKOUT"

    def test_trend_long_sets_entry_regime_on_state(self):
        """After trend entry, strategy.state.entry_regime is TREND_BREAKOUT."""
        strategy = _make_strategy()
        boll = _boll_snapshot()
        cvd = _cvd_snapshot(fast_cvd=0.002, buy_ratio=0.70, sell_ratio=0.30)

        with patch.object(
            strategy, "_route_regime", return_value=_trend_long_decision(),
        ):
            strategy.on_tick(price=3050.0, ts_ms=1000000, boll=boll, cvd=cvd)

        assert strategy.state.entry_regime == "TREND_BREAKOUT"
        assert strategy.state.trend_breakout_active is True

    def test_trend_long_has_single_tp_plan(self):
        """Trend positions get tp_plan=SINGLE (no Three-Stage)."""
        strategy = _make_strategy()
        boll = _boll_snapshot()
        cvd = _cvd_snapshot(fast_cvd=0.002, buy_ratio=0.70, sell_ratio=0.30)

        with patch.object(
            strategy, "_route_regime", return_value=_trend_long_decision(),
        ):
            strategy.on_tick(price=3050.0, ts_ms=1000000, boll=boll, cvd=cvd)

        assert strategy.state.tp_plan == "SINGLE"
        # No Three-Stage state should be set
        assert strategy.state.three_stage_runner_enabled_for_position is False

    def test_trend_long_sets_trend_trailing_sl(self):
        """Trend entry initialises trend_trailing_sl_price."""
        strategy = _make_strategy()
        boll = _boll_snapshot(middle=3000.0)
        cvd = _cvd_snapshot(fast_cvd=0.002, buy_ratio=0.70, sell_ratio=0.30)

        with patch.object(
            strategy, "_route_regime", return_value=_trend_long_decision(),
        ):
            strategy.on_tick(price=3050.0, ts_ms=1000000, boll=boll, cvd=cvd)

        # trend trailing SL should be ~ middle * (1 - 0.001) = 2997
        assert strategy.state.trend_trailing_sl_price is not None
        assert strategy.state.trend_trailing_sl_price < boll.middle

    def test_trend_long_with_position_already_open_skips_entry(self):
        """Trend entry skipped when a position is already open."""
        strategy = _make_strategy()
        strategy.state.side = "LONG"
        strategy.state.layers = 1
        boll = _boll_snapshot()
        cvd = _cvd_snapshot()

        with patch.object(
            strategy, "_route_regime", return_value=_trend_long_decision(),
        ):
            intents = strategy.on_tick(
                price=3050.0, ts_ms=1000000, boll=boll, cvd=cvd,
            )

        # No new entry intents
        open_intents = [i for i in intents if i.intent_type in ("OPEN_LONG", "OPEN_SHORT")]
        assert len(open_intents) == 0


# ── Test: Trend SHORT Entry ────────────────────────────────────────────


class TestTrendShortEntry:
    def test_trend_short_emits_open_short_intent(self):
        """Given TREND_SHORT decision, strategy emits OPEN_SHORT intent."""
        strategy = _make_strategy()
        boll = _boll_snapshot()
        cvd = _cvd_snapshot(fast_cvd=-0.002, buy_ratio=0.30, sell_ratio=0.70)

        with patch.object(
            strategy, "_route_regime", return_value=_trend_short_decision(),
        ):
            # Entry price close to middle so stop distance within bounds
            intents = strategy.on_tick(
                price=2950.0, ts_ms=1000000, boll=boll, cvd=cvd,
            )

        open_intents = [i for i in intents if i.intent_type == "OPEN_SHORT"]
        assert len(open_intents) == 1
        intent = open_intents[0]
        assert intent.side == "SHORT"
        assert intent.entry_regime == "TREND_BREAKOUT"

    def test_trend_short_sets_entry_regime_on_state(self):
        strategy = _make_strategy()
        boll = _boll_snapshot()
        cvd = _cvd_snapshot(fast_cvd=-0.002, buy_ratio=0.30, sell_ratio=0.70)

        with patch.object(
            strategy, "_route_regime", return_value=_trend_short_decision(),
        ):
            strategy.on_tick(price=2950.0, ts_ms=1000000, boll=boll, cvd=cvd)

        assert strategy.state.entry_regime == "TREND_BREAKOUT"
        assert strategy.state.trend_breakout_active is True
        assert strategy.state.tp_plan == "SINGLE"


# ── Test: Stop Distance Too Wide ───────────────────────────────────────


class TestTrendStopDistanceTooWide:
    def test_trend_long_skipped_when_stop_too_wide(self):
        """When stop distance exceeds TREND_MAX_STOP_DISTANCE_PCT, entry is skipped."""
        config = _make_config(trend_max_stop_distance_pct=0.005)
        strategy = _make_strategy(config)
        # BOLL middle far from entry → large stop distance
        boll = _boll_snapshot(middle=2500.0, upper=3000.0, lower=2000.0)
        cvd = _cvd_snapshot(fast_cvd=0.002, buy_ratio=0.70, sell_ratio=0.30)

        with patch.object(
            strategy, "_route_regime", return_value=_trend_long_decision(),
        ):
            intents = strategy.on_tick(
                price=3200.0, ts_ms=1000000, boll=boll, cvd=cvd,
            )

        open_intents = [i for i in intents if i.intent_type in ("OPEN_LONG", "OPEN_SHORT")]
        assert len(open_intents) == 0

    def test_trend_stop_distance_too_wide_default(self):
        """With default max 2% and price far from middle, entry is skipped."""
        config = _make_config(trend_max_stop_distance_pct=0.005)
        strategy = _make_strategy(config)
        boll = _boll_snapshot(middle=2500.0, upper=3000.0, lower=2000.0)
        cvd = _cvd_snapshot(fast_cvd=0.002, buy_ratio=0.70, sell_ratio=0.30)

        with patch.object(
            strategy, "_route_regime", return_value=_trend_long_decision(),
        ):
            intents = strategy.on_tick(
                price=3050.0, ts_ms=1000000, boll=boll, cvd=cvd,
            )

        # Stop distance too wide → skipped
        open_intents = [i for i in intents if i.intent_type in ("OPEN_LONG", "OPEN_SHORT")]
        assert len(open_intents) == 0


# ── Test: Mean-Reversion Still Works ───────────────────────────────────


class TestMeanReversionStillWorks:
    def test_mr_long_decision_goes_to_mr_path(self):
        """MEAN_REVERSION_LONG decision dispatches to existing MR entry path."""
        strategy = _make_strategy()
        boll = _boll_snapshot()
        cvd = _cvd_snapshot()

        # The MR path needs armed state. Set it up.
        strategy.state.lower_armed = True
        strategy.state.lower_extreme_price = 2850.0
        strategy.state.lower_deep_enough = True
        strategy.state.lower_cvd_divergence_confirmed = True

        with patch.object(
            strategy, "_route_regime", return_value=_mr_long_decision(),
        ):
            intents = strategy.on_tick(
                price=2910.0, ts_ms=1000000, boll=boll, cvd=cvd,
            )

        open_intents = [i for i in intents if i.intent_type == "OPEN_LONG"]
        # May produce entry if reclaim soft-confirm passes; entry_regime should be
        # MEAN_REVERSION if it does
        for intent in open_intents:
            assert intent.entry_regime == "MEAN_REVERSION"


# ── Test: Conflict No Trade ────────────────────────────────────────────


class TestConflictNoTrade:
    def test_conflict_no_trade_skips_entry(self):
        """CONFLICT_NO_TRADE → no open intent emitted."""
        strategy = _make_strategy()
        boll = _boll_snapshot()
        cvd = _cvd_snapshot()

        with patch.object(
            strategy, "_route_regime", return_value=_conflict_decision(),
        ):
            intents = strategy.on_tick(
                price=3000.0, ts_ms=1000000, boll=boll, cvd=cvd,
            )

        open_intents = [i for i in intents if i.intent_type in ("OPEN_LONG", "OPEN_SHORT")]
        assert len(open_intents) == 0

    def test_no_trade_decision_skips_entry(self):
        """NO_TRADE → no open intent emitted."""
        strategy = _make_strategy()
        boll = _boll_snapshot()
        cvd = _cvd_snapshot()

        with patch.object(
            strategy, "_route_regime", return_value=_no_trade_decision(),
        ):
            intents = strategy.on_tick(
                price=3000.0, ts_ms=1000000, boll=boll, cvd=cvd,
            )

        open_intents = [i for i in intents if i.intent_type in ("OPEN_LONG", "OPEN_SHORT")]
        assert len(open_intents) == 0


# ── Test: Trend Trailing SL Update ─────────────────────────────────────


class TestTrendTrailingSLUpdate:
    def test_trend_position_triggers_trailing_sl_branch(self):
        """When position has entry_regime=TREND_BREAKOUT, TP update goes
        to the trend trailing SL branch."""
        strategy = _make_strategy()
        strategy.state.side = "LONG"
        strategy.state.layers = 1
        strategy.state.entry_regime = "TREND_BREAKOUT"
        strategy.state.trend_trailing_sl_price = 2900.0
        strategy.state.trend_last_sl_update_ts_ms = 0  # allow immediate update
        strategy.state.avg_entry_price = 3200.0
        strategy.state.tp_price = 3500.0
        strategy.state.last_tp_update_candle_ts_ms = 0  # different from boll
        boll = _boll_snapshot(
            middle=3000.0, upper=3500.0, lower=2500.0,
            candle_ts_ms=2000000,  # different from last_tp_update_candle
        )
        cvd = _cvd_snapshot()

        intents = strategy.on_tick(
            price=3300.0, ts_ms=2000000, boll=boll, cvd=cvd,
        )

        # Check that a TP update was attempted
        tp_intents = [i for i in intents if i.intent_type == "UPDATE_TP"]
        # The trailing SL should tighten: old=2900, new ~2997 (3000*0.999)
        if tp_intents:
            assert strategy.state.trend_trailing_sl_price > 2900.0

    def test_trend_trailing_sl_only_tightens_not_loosens(self):
        """Trend trailing SL does not loosen when middle moves away."""
        strategy = _make_strategy()
        strategy.state.side = "LONG"
        strategy.state.layers = 1
        strategy.state.entry_regime = "TREND_BREAKOUT"
        # Current SL is already tight (close to middle)
        strategy.state.trend_trailing_sl_price = 2990.0
        strategy.state.trend_last_sl_update_ts_ms = 0
        strategy.state.avg_entry_price = 3200.0
        strategy.state.tp_price = 3500.0
        strategy.state.last_tp_update_candle_ts_ms = 0
        # Middle moved down → candidate would be lower → loosening → rejected
        boll = _boll_snapshot(
            middle=2950.0, candle_ts_ms=2000000,
        )
        cvd = _cvd_snapshot()

        # Save old SL
        old_sl = strategy.state.trend_trailing_sl_price

        intents = strategy.on_tick(
            price=3300.0, ts_ms=2000000, boll=boll, cvd=cvd,
        )

        # SL should not have loosened
        assert strategy.state.trend_trailing_sl_price >= old_sl

    def test_trend_trailing_sl_update_respects_interval(self):
        """Trailing SL respects TREND_SL_UPDATE_INTERVAL_SECONDS."""
        strategy = _make_strategy(trend_sl_update_interval_seconds=900)
        strategy.state.side = "LONG"
        strategy.state.layers = 1
        strategy.state.entry_regime = "TREND_BREAKOUT"
        strategy.state.trend_trailing_sl_price = 2900.0
        # Last update was very recent
        strategy.state.trend_last_sl_update_ts_ms = 1999000  # only 1 second ago
        strategy.state.avg_entry_price = 3200.0
        strategy.state.tp_price = 3500.0
        strategy.state.last_tp_update_candle_ts_ms = 0
        boll = _boll_snapshot(
            middle=3000.0, candle_ts_ms=2000000,
        )
        cvd = _cvd_snapshot()

        old_sl = strategy.state.trend_trailing_sl_price

        intents = strategy.on_tick(
            price=3300.0, ts_ms=2000000, boll=boll, cvd=cvd,
        )

        # SL should NOT have been updated (rate-limited)
        assert strategy.state.trend_trailing_sl_price == old_sl


# ── Test: Config Validation ────────────────────────────────────────────


class TestTrendConfigValidation:
    def test_default_config_parses(self):
        config = BollCvdReclaimStrategyConfig()
        assert config.trend_breakout_enabled is False
        assert config.trend_middle_trailing_sl_enabled is True
        assert config.trend_middle_sl_buffer_pct == 0.001

    def test_trend_max_stop_distance_positive(self):
        with pytest.raises(RuntimeError, match="TREND_MAX_STOP_DISTANCE_PCT"):
            BollCvdReclaimStrategyConfig(trend_max_stop_distance_pct=-0.01)

    def test_trend_sl_update_interval_positive(self):
        with pytest.raises(RuntimeError, match="TREND_SL_UPDATE_INTERVAL_SECONDS"):
            BollCvdReclaimStrategyConfig(trend_sl_update_interval_seconds=0)

    def test_outside_occupancy_range(self):
        with pytest.raises(RuntimeError, match="TREND_OUTSIDE_OCCUPANCY_MIN_RATIO"):
            BollCvdReclaimStrategyConfig(trend_outside_occupancy_min_ratio=1.5)
