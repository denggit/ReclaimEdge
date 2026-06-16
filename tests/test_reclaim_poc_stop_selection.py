"""Tests for POC / Extreme adaptive entry SL selection."""
from __future__ import annotations

import pytest
from src.indicators.cvd_tracker import CvdSnapshot
from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
)
from src.strategies.sweep_volume_profile import SweepVolumeProfile


def _boll(middle=2000, upper=2100, lower=1900, alert_switch_on=True) -> BollSnapshot:
    return BollSnapshot(
        inst_id="ETH-USDT-SWAP", candle_ts_ms=1000, close=middle,
        middle=middle, upper=upper, lower=lower,
        upper_distance_pct=0.0, lower_distance_pct=0.001,
        alert_switch_on=alert_switch_on, live_mode=True,
    )


def _cvd(ts_ms=1000, price=1901, fast_cvd=1.0, previous_fast_cvd=0.0,
         buy_ratio=0.7, sell_ratio=0.3,
         buy_volume=70.0, sell_volume=30.0,
         cumulative_buy_volume=500000.0, cumulative_sell_volume=1500000.0,
         cross_positive=True, cross_negative=False,
         cvd_increasing=True, cvd_decreasing=False,
         no_new_low=True, no_new_high=True,
         up_burst=False, down_burst=False) -> CvdSnapshot:
    return CvdSnapshot(
        ts_ms=ts_ms, price=price,
        side="buy" if buy_ratio >= sell_ratio else "sell",
        size=1.0, signed_delta=1.0, total_cvd=10.0,
        fast_cvd=fast_cvd, previous_fast_cvd=previous_fast_cvd,
        buy_volume=buy_volume, sell_volume=sell_volume,
        buy_ratio=buy_ratio, sell_ratio=sell_ratio,
        cross_positive=cross_positive, cross_negative=cross_negative,
        cvd_increasing=cvd_increasing, cvd_decreasing=cvd_decreasing,
        no_new_low=no_new_low, no_new_high=no_new_high,
        window_low=1897.0, window_high=1905.0,
        burst_net_move_pct=0.0, burst_range_pct=0.002, baseline_range_pct=0.001,
        burst_move_ratio=2.0, burst_volume=10.0, baseline_volume=5.0,
        burst_volume_ratio=2.0,
        up_burst=up_burst, down_burst=down_burst,
        cumulative_buy_volume=cumulative_buy_volume,
        cumulative_sell_volume=cumulative_sell_volume,
    )


def _strategy(**overrides) -> BollCvdReclaimStrategy:
    cfg_kwargs = dict(
        min_outside_pct=0.001,
        entry_min_reward_risk=0.0,
        entry_fee_slippage_buffer_pct=0.0,
        order_cooldown_seconds=0,
        entry_reclaim_v2_enabled=True,
        entry_reclaim_require_anchored_divergence=True,
        entry_sweep_profile_enabled=True,
        entry_sweep_profile_bucket_pct=0.0002,
        entry_poc_stop_enabled=True,
        entry_poc_stop_min_tail_pct=0.003,
        entry_poc_stop_buffer_pct=0.001,
        entry_extreme_stop_buffer_pct=0.001,
        entry_sl_buffer_pct=0.0005,
        entry_reclaim_confirm_seconds=0,
        entry_reclaim_inside_band=False,
        entry_cvd_divergence_enabled=False,
        entry_cvd_absorption_enabled=False,
        entry_max_stop_distance_pct=0,
    )
    cfg_kwargs.update(overrides)
    cfg = BollCvdReclaimStrategyConfig(**cfg_kwargs)
    sizer = SimplePositionSizer(SimplePositionSizerConfig(
        dry_run_equity_usdt=10_000, leverage=20, trade_risk_pct=0.01,
        fee_slippage_buffer_pct=0.001,
    ))
    return BollCvdReclaimStrategy(cfg, sizer)


# ── LONG: extreme far from POC → POC_OUTWARD ────────────────────────

def test_long_poc_stop_when_extreme_is_far_tail() -> None:
    """When extreme is far below POC (sweep tail), use POC-based stop."""
    strat = _strategy(
        entry_poc_stop_min_tail_pct=0.003,  # 0.3% tail minimum
        entry_poc_stop_buffer_pct=0.001,
    )
    entry = 100.0
    extreme = 95.0   # 5% below entry — extreme is a distant tail
    poc = 98.0       # POC is close to entry, far from extreme

    # Manually set up state
    strat.state.lower_extreme_price = extreme
    sp = SweepVolumeProfile(bucket_pct=0.0002)
    sp.add(poc, 100.0)
    sp.add(poc + 0.01, 10.0)
    strat.state.lower_sweep_profile = sp

    sl, mode = strat._select_entry_stop_price(side="LONG", entry_price=entry)
    assert sl is not None
    assert mode == "POC_OUTWARD"
    expected_poc_sl = poc * (1 - 0.001)
    assert abs(sl - expected_poc_sl) < 0.0001
    assert sl < entry  # LONG SL must be below entry


# ── LONG: extreme close to POC → EXTREME_OUTWARD ─────────────────────

def test_long_extreme_stop_when_tail_is_small() -> None:
    """When extreme is close to POC, use extreme-based stop."""
    strat = _strategy(
        entry_poc_stop_min_tail_pct=0.01,  # 1% tail minimum
        entry_extreme_stop_buffer_pct=0.001,
    )
    entry = 100.0
    extreme = 96.0  # 4% below entry
    poc = 96.2      # POC close to extreme, tail is small

    strat.state.lower_extreme_price = extreme
    sp = SweepVolumeProfile(bucket_pct=0.0002)
    sp.add(poc, 100.0)
    strat.state.lower_sweep_profile = sp

    sl, mode = strat._select_entry_stop_price(side="LONG", entry_price=entry)
    assert sl is not None
    assert mode == "EXTREME_OUTWARD"
    expected = extreme * (1 - 0.001)
    assert abs(sl - expected) < 0.0001


# ── SHORT: extreme far from POC → POC_OUTWARD ───────────────────────

def test_short_poc_stop_when_extreme_is_far_tail() -> None:
    """When extreme is far above POC (sweep tail), use POC-based stop."""
    strat = _strategy(
        entry_poc_stop_min_tail_pct=0.003,
        entry_poc_stop_buffer_pct=0.001,
    )
    entry = 100.0
    extreme = 105.0  # 5% above entry — extreme is a distant tail
    poc = 102.0      # POC close to entry, far from extreme

    strat.state.upper_extreme_price = extreme
    sp = SweepVolumeProfile(bucket_pct=0.0002)
    sp.add(poc, 100.0)
    sp.add(poc - 0.01, 10.0)
    strat.state.upper_sweep_profile = sp

    sl, mode = strat._select_entry_stop_price(side="SHORT", entry_price=entry)
    assert sl is not None
    assert mode == "POC_OUTWARD"
    expected_poc_sl = poc * (1 + 0.001)
    assert abs(sl - expected_poc_sl) < 0.0001
    assert sl > entry  # SHORT SL must be above entry


# ── SHORT: extreme close to POC → EXTREME_OUTWARD ────────────────────

def test_short_extreme_stop_when_tail_is_small() -> None:
    """When extreme is close to POC, use extreme-based stop."""
    strat = _strategy(
        entry_poc_stop_min_tail_pct=0.01,
        entry_extreme_stop_buffer_pct=0.001,
    )
    entry = 100.0
    extreme = 104.0
    poc = 103.8  # POC close to extreme

    strat.state.upper_extreme_price = extreme
    sp = SweepVolumeProfile(bucket_pct=0.0002)
    sp.add(poc, 100.0)
    strat.state.upper_sweep_profile = sp

    sl, mode = strat._select_entry_stop_price(side="SHORT", entry_price=entry)
    assert sl is not None
    assert mode == "EXTREME_OUTWARD"
    expected = extreme * (1 + 0.001)
    assert abs(sl - expected) < 0.0001


# ── SL validity ──────────────────────────────────────────────────────

def test_long_sl_below_entry_price() -> None:
    """LONG SL must always be below entry price."""
    strat = _strategy(entry_poc_stop_buffer_pct=0.001, entry_extreme_stop_buffer_pct=0.001)
    entry = 100.0
    strat.state.lower_extreme_price = 96.0
    sp = SweepVolumeProfile(bucket_pct=0.0002)
    sp.add(98.0, 100.0)
    strat.state.lower_sweep_profile = sp

    sl, _mode = strat._select_entry_stop_price(side="LONG", entry_price=entry)
    assert sl is not None
    assert sl < entry


def test_short_sl_above_entry_price() -> None:
    """SHORT SL must always be above entry price."""
    strat = _strategy(entry_poc_stop_buffer_pct=0.001, entry_extreme_stop_buffer_pct=0.001)
    entry = 100.0
    strat.state.upper_extreme_price = 104.0
    sp = SweepVolumeProfile(bucket_pct=0.0002)
    sp.add(102.0, 100.0)
    strat.state.upper_sweep_profile = sp

    sl, _mode = strat._select_entry_stop_price(side="SHORT", entry_price=entry)
    assert sl is not None
    assert sl > entry


# ── POC disabled → extreme classic ───────────────────────────────────

def test_poc_disabled_uses_classic_extreme() -> None:
    """When POC stop is disabled, use classic extreme-based SL."""
    strat = _strategy(entry_poc_stop_enabled=False, entry_sl_buffer_pct=0.0005)
    entry = 100.0
    strat.state.lower_extreme_price = 95.0

    sl, mode = strat._select_entry_stop_price(side="LONG", entry_price=entry)
    assert sl is not None
    assert mode == "EXTREME_CLASSIC"
    expected = 95.0 * (1 - 0.0005)
    assert abs(sl - expected) < 0.0001


# ── Missing extreme → None ───────────────────────────────────────────

def test_missing_extreme_returns_none() -> None:
    strat = _strategy()
    sl, mode = strat._select_entry_stop_price(side="LONG", entry_price=100.0)
    assert sl is None
    assert mode == "MISSING_EXTREME"
