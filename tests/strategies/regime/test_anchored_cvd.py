from __future__ import annotations

from src.strategies.regime.anchored_cvd import (
    AnchoredCvdConfig,
    build_anchored_cvd_state,
    is_cvd_confirming_trend,
    is_cvd_diverging_from_price,
)


def make_config(**kwargs) -> AnchoredCvdConfig:
    defaults = dict(
        min_buy_ratio=0.58,
        min_sell_ratio=0.58,
        max_pullback_ratio=0.45,
    )
    defaults.update(kwargs)
    return AnchoredCvdConfig(**defaults)


# ── build_anchored_cvd_state ──────────────────────────────────────────


class TestBuildAnchoredCvdState:
    """Test 1: anchor_cvd=100, current_cvd=160 → episode_delta=60."""

    def test_positive_delta(self):
        state = build_anchored_cvd_state(
            anchor_ts_ms=1000,
            current_ts_ms=2000,
            anchor_cvd=100.0,
            current_cvd=160.0,
            episode_buy_volume=80.0,
            episode_sell_volume=20.0,
            episode_cvd_max=160.0,
            episode_cvd_min=100.0,
        )
        assert state.episode_cvd_delta == 60.0
        assert state.episode_buy_ratio == 0.8
        assert state.episode_sell_ratio == 0.2
        assert state.cvd_slope == 60.0  # 60 delta / 1 second


class TestNegativeDelta:
    def test_negative_delta(self):
        state = build_anchored_cvd_state(
            anchor_ts_ms=1000,
            current_ts_ms=5000,
            anchor_cvd=200.0,
            current_cvd=150.0,
            episode_buy_volume=10.0,
            episode_sell_volume=90.0,
            episode_cvd_max=200.0,
            episode_cvd_min=150.0,
        )
        assert state.episode_cvd_delta == -50.0
        assert state.episode_buy_ratio == 0.1
        assert state.episode_sell_ratio == 0.9
        assert state.cvd_slope == -12.5  # -50 / 4 seconds


# ── is_cvd_confirming_trend ───────────────────────────────────────────


class TestCvdConfirmsTrendUp:
    """Test 2: UP + delta positive + buy_ratio high → confirms."""

    def test_up_breakout_confirms(self):
        config = make_config(min_buy_ratio=0.58)
        state = build_anchored_cvd_state(
            anchor_ts_ms=1000, current_ts_ms=2000,
            anchor_cvd=100.0, current_cvd=160.0,
            episode_buy_volume=80.0, episode_sell_volume=20.0,
            episode_cvd_max=160.0, episode_cvd_min=100.0,
        )
        assert is_cvd_confirming_trend("UP", state, config) is True

    def test_up_breakout_delta_negative_no_confirm(self):
        config = make_config()
        state = build_anchored_cvd_state(
            anchor_ts_ms=1000, current_ts_ms=2000,
            anchor_cvd=100.0, current_cvd=90.0,
            episode_buy_volume=80.0, episode_sell_volume=20.0,
            episode_cvd_max=100.0, episode_cvd_min=90.0,
        )
        assert is_cvd_confirming_trend("UP", state, config) is False

    def test_up_breakout_buy_ratio_too_low(self):
        config = make_config(min_buy_ratio=0.58)
        state = build_anchored_cvd_state(
            anchor_ts_ms=1000, current_ts_ms=2000,
            anchor_cvd=100.0, current_cvd=160.0,
            episode_buy_volume=50.0, episode_sell_volume=50.0,
            episode_cvd_max=160.0, episode_cvd_min=100.0,
        )
        assert is_cvd_confirming_trend("UP", state, config) is False


class TestCvdConfirmsTrendDown:
    """Test 4: DOWN + delta negative + sell_ratio high → confirms."""

    def test_down_breakout_confirms(self):
        config = make_config(min_sell_ratio=0.58)
        state = build_anchored_cvd_state(
            anchor_ts_ms=1000, current_ts_ms=2000,
            anchor_cvd=200.0, current_cvd=150.0,
            episode_buy_volume=20.0, episode_sell_volume=80.0,
            episode_cvd_max=200.0, episode_cvd_min=150.0,
        )
        assert is_cvd_confirming_trend("DOWN", state, config) is True

    def test_down_breakout_sell_ratio_too_low(self):
        config = make_config(min_sell_ratio=0.58)
        state = build_anchored_cvd_state(
            anchor_ts_ms=1000, current_ts_ms=2000,
            anchor_cvd=200.0, current_cvd=150.0,
            episode_buy_volume=50.0, episode_sell_volume=50.0,
            episode_cvd_max=200.0, episode_cvd_min=150.0,
        )
        assert is_cvd_confirming_trend("DOWN", state, config) is False


# ── is_cvd_diverging_from_price ───────────────────────────────────────


class TestCvdDivergesUpBreakout:
    """Test 3: UP + price new high + delta negative → diverges (mean-reversion)."""

    def test_up_price_new_high_cvd_negative(self):
        config = make_config()
        state = build_anchored_cvd_state(
            anchor_ts_ms=1000, current_ts_ms=2000,
            anchor_cvd=100.0, current_cvd=90.0,
            episode_buy_volume=30.0, episode_sell_volume=70.0,
            episode_cvd_max=100.0, episode_cvd_min=90.0,
        )
        assert is_cvd_diverging_from_price("UP", state, True, config) is True

    def test_up_no_new_extreme_no_divergence(self):
        config = make_config()
        state = build_anchored_cvd_state(
            anchor_ts_ms=1000, current_ts_ms=2000,
            anchor_cvd=100.0, current_cvd=90.0,
            episode_buy_volume=30.0, episode_sell_volume=70.0,
            episode_cvd_max=100.0, episode_cvd_min=90.0,
        )
        # No new extreme → no divergence reported
        assert is_cvd_diverging_from_price("UP", state, False, config) is False


class TestCvdDivergesDownBreakout:
    """Test 5: DOWN + price new low + delta positive → diverges (mean-reversion)."""

    def test_down_price_new_low_cvd_positive(self):
        config = make_config()
        state = build_anchored_cvd_state(
            anchor_ts_ms=1000, current_ts_ms=2000,
            anchor_cvd=100.0, current_cvd=120.0,
            episode_buy_volume=70.0, episode_sell_volume=30.0,
            episode_cvd_max=120.0, episode_cvd_min=100.0,
        )
        assert is_cvd_diverging_from_price("DOWN", state, True, config) is True

    def test_down_no_new_extreme_no_divergence(self):
        config = make_config()
        state = build_anchored_cvd_state(
            anchor_ts_ms=1000, current_ts_ms=2000,
            anchor_cvd=100.0, current_cvd=120.0,
            episode_buy_volume=70.0, episode_sell_volume=30.0,
            episode_cvd_max=120.0, episode_cvd_min=100.0,
        )
        assert is_cvd_diverging_from_price("DOWN", state, False, config) is False


class TestCvdDrawdownBlocksConfirm:
    """Test: Deep drawdown should block trend confirmation."""

    def test_up_breakout_deep_drawdown_no_confirm(self):
        config = make_config(max_pullback_ratio=0.45)
        # Episode max was 160 but current is back to 110 → high drawdown
        state = build_anchored_cvd_state(
            anchor_ts_ms=1000, current_ts_ms=2000,
            anchor_cvd=100.0, current_cvd=110.0,  # pulled back from 160
            episode_buy_volume=70.0, episode_sell_volume=30.0,
            episode_cvd_max=160.0, episode_cvd_min=100.0,
        )
        # delta = 10, drawdown = (160-110)/10 = 5.0 > 0.45
        assert is_cvd_confirming_trend("UP", state, config) is False


class TestFiveSecondCvdNotCore:
    """Test 6: 5-second CVD window is NOT part of core trend confirmation.

    This test verifies that our anchored CVD functions do not even accept
    a 5-second-window parameter — they are purely event-anchored.
    """

    def test_no_five_second_window_in_api(self):
        import inspect
        sig = inspect.signature(build_anchored_cvd_state)
        params = list(sig.parameters.keys())
        assert "fast_cvd" not in params
        assert "window_seconds" not in params
        assert "5s" not in str(params).lower()

    def test_no_five_second_window_in_confirm(self):
        import inspect
        sig = inspect.signature(is_cvd_confirming_trend)
        params = list(sig.parameters.keys())
        assert "fast_cvd" not in params
        assert "window" not in params
