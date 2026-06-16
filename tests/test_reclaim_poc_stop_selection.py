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
         up_burst=False, down_burst=False,
         size=1.0) -> CvdSnapshot:
    return CvdSnapshot(
        ts_ms=ts_ms, price=price,
        side="buy" if buy_ratio >= sell_ratio else "sell",
        size=size, signed_delta=1.0, total_cvd=10.0,
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
        entry_poc_stop_min_tail_pct=0.008,
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
        entry_poc_stop_min_tail_pct=0.008,  # 0.3% tail minimum
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

def test_long_extreme_stop_when_entry_extreme_distance_small() -> None:
    """When entry-extreme distance is small (< 0.8%), use extreme-based stop regardless of POC."""
    strat = _strategy(
        entry_poc_stop_min_tail_pct=0.008,
        entry_extreme_stop_buffer_pct=0.001,
    )
    entry = 2000.0
    extreme = 1988.0  # 0.6% below entry (< 0.8% threshold)
    poc = 1990.0      # POC is valid, but entry-extreme distance is too small

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
        entry_poc_stop_min_tail_pct=0.008,
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

def test_short_extreme_stop_when_entry_extreme_distance_small() -> None:
    """When entry-extreme distance is small (< 0.8%), use extreme-based stop regardless of POC."""
    strat = _strategy(
        entry_poc_stop_min_tail_pct=0.008,
        entry_extreme_stop_buffer_pct=0.001,
    )
    entry = 2000.0
    extreme = 2012.0  # 0.6% above entry (< 0.8% threshold)
    poc = 2010.0      # POC is valid, but entry-extreme distance is too small

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


# ── POC volume uses cvd.size ───────────────────────────────────────────

def test_poc_volume_uses_cvd_size_not_rolling_window() -> None:
    """POC volume must come from cvd.size (current tick), not buy_volume+sell_volume (rolling window)."""
    strat = _strategy(entry_sweep_profile_enabled=True, entry_sweep_profile_bucket_pct=0.001)
    from src.strategies.sweep_volume_profile import SweepVolumeProfile

    sp = SweepVolumeProfile(bucket_pct=0.001)
    strat.state.lower_sweep_profile = sp

    # Simulate a tick: cvd.size determines POC, even though buy_volume + sell_volume is huge
    cvd = _cvd(
        ts_ms=1000, price=1900.0,
        size=50.0,  # current tick volume → this should determine POC bucket weight
        buy_volume=999999.0,  # large rolling window buy volume — should NOT influence POC
        sell_volume=999999.0,  # large rolling window sell volume — should NOT influence POC
    )
    strat._record_sweep_volume("LOWER", 1900.0, cvd)
    assert sp.poc_price() is not None

    # Add a competing tick with larger cvd.size at a different price
    cvd2 = _cvd(
        ts_ms=2000, price=1910.0,
        size=100.0,  # higher tick volume → should make 1910.0 bucket the POC
        buy_volume=1.0,
        sell_volume=1.0,
    )
    strat._record_sweep_volume("LOWER", 1910.0, cvd2)

    poc = sp.poc_price()
    assert poc is not None
    # The POC should be near 1910 because that's where the most cvd.size went
    bucket_size_1910 = 1910.0 * 0.001
    expected_bucket = round(1910.0 / bucket_size_1910) * bucket_size_1910
    assert abs(poc - expected_bucket) < 0.01


def test_poc_volume_uses_cvd_size_upper() -> None:
    """Same for UPPER side — cvd.size not buy_volume+sell_volume."""
    strat = _strategy(entry_sweep_profile_enabled=True, entry_sweep_profile_bucket_pct=0.001)
    from src.strategies.sweep_volume_profile import SweepVolumeProfile

    sp = SweepVolumeProfile(bucket_pct=0.001)
    strat.state.upper_sweep_profile = sp

    cvd = _cvd(
        ts_ms=1000, price=2100.0,
        size=75.0,
        buy_volume=500000.0,
        sell_volume=500000.0,
    )
    strat._record_sweep_volume("UPPER", 2100.0, cvd)
    assert sp.poc_price() is not None


# ═══════════════════════════════════════════════════════════════════════════════
# New tests: entry-extreme distance replaces POC-extreme distance
# ═══════════════════════════════════════════════════════════════════════════════

# ── LONG: entry-extreme >= 0.8%, POC valid → POC_OUTWARD ─────────────

def test_long_entry_extreme_wide_poc_valid() -> None:
    """When entry-extreme distance >= 0.8% and POC stop is reasonable, use POC stop."""
    strat = _strategy(
        entry_poc_stop_min_tail_pct=0.008,
        entry_poc_stop_buffer_pct=0.001,
        entry_extreme_stop_buffer_pct=0.001,
    )
    entry = 2000.0
    extreme = 1982.0  # 0.9% below entry ( >= 0.8% )
    poc = 1990.0      # POC near entry, far from extreme

    strat.state.lower_extreme_price = extreme
    sp = SweepVolumeProfile(bucket_pct=0.0002)
    sp.add(poc, 100.0)
    sp.add(poc + 1.0, 10.0)
    strat.state.lower_sweep_profile = sp

    sl, mode = strat._select_entry_stop_price(side="LONG", entry_price=entry)
    assert sl is not None
    assert mode == "POC_OUTWARD", (
        f"entry-extreme distance={(entry-extreme)/entry*100:.1f}% >= 0.8%, "
        f"POC valid → POC_OUTWARD"
    )
    expected_poc_sl = poc * (1.0 - 0.001)
    assert abs(sl - expected_poc_sl) < 0.0001
    assert sl < entry


# ── LONG: entry-extreme >= 0.8% but POC invalid → EXTREME_OUTWARD ────

def test_long_entry_extreme_wide_poc_invalid() -> None:
    """When entry-extreme >= 0.8% but POC stop >= entry_price, fallback to extreme."""
    strat = _strategy(
        entry_poc_stop_min_tail_pct=0.008,
        entry_poc_stop_buffer_pct=0.001,
        entry_extreme_stop_buffer_pct=0.001,
    )
    entry = 2000.0
    extreme = 1982.0  # 0.9% below entry ( >= 0.8% )
    # POC very close to entry — poc_stop = 1999.8 * 0.999 ≈ 1998.0, still < entry
    # Actually need poc so close that poc_stop >= entry
    # poc_stop = poc * 0.999 >= 2000 → poc >= 2002.003
    # But poc should be < entry for a valid LONG POC... let me use a different approach
    # Make poc very close to entry: poc=1999.9 → poc_stop = 1999.9*0.999=1997.9 < 2000
    # Hmm, let me think differently. The user wants poc_stop >= entry_price
    # For that, poc needs to be very close to or above entry
    # Let me use poc = 2000.5 (slightly above entry, unusual but possible with sweep)
    poc = 2000.5  # POC slightly above entry → poc_stop = 2000.5*0.999 = 1998.5 < 2000, still valid...

    # Wait, poc_stop < entry is one of the conditions for use_poc. To make it invalid:
    # Option 1: poc_stop >= entry_price → poc * 0.999 >= 2000 → poc >= 2002.003
    # Option 2: poc_stop <= extreme_stop → poc*0.999 <= extreme*0.999 → poc <= extreme (unlikely)
    # Let me use poc very near entry: poc=1999.0 → poc_stop=1999*0.999=1997.001
    # extreme_stop = 1982 * 0.999 = 1980.018, poc_stop > extreme_stop ✓
    # poc_stop < entry ✓
    # → this would be POC_OUTWARD!

    # To make it invalid, let me use a case where poc_stop <= extreme_stop
    # That can happen when poc is close to extreme
    # poc=1985.0 → poc_stop=1985*0.999=1983.015, extreme_stop=1982*0.999=1980.018
    # poc_stop > extreme_stop still... hmm

    # Actually for the test to show EXTREME_OUTWARD when POC is invalid,
    # the simplest case: poc > entry_price (POC stop would be above entry on LONG)
    # With poc=2001.0: poc_stop=2001*0.999=1998.999, still < 2000 (entry)
    # Actually (2001 * 0.999) = 1998.999 < 2000, so it's still valid...

    # Let me try with poc=2003.0: poc_stop=2003*0.999=2000.997 >= 2000 → INVALID!
    # extreme_stop=1982*0.999=1980.018, 2000.997 > extreme_stop
    # So: distance >= 0.8% but poc_stop >= entry → EXTREME_OUTWARD
    poc = 2003.0  # poc_stop = 2003 * 0.999 = 2000.997 >= 2000 (entry)

    strat.state.lower_extreme_price = extreme
    sp = SweepVolumeProfile(bucket_pct=0.0002)
    sp.add(poc, 100.0)
    strat.state.lower_sweep_profile = sp

    sl, mode = strat._select_entry_stop_price(side="LONG", entry_price=entry)
    assert sl is not None
    assert mode == "EXTREME_OUTWARD", (
        f"POC stop (2000.997) >= entry (2000), must fallback to EXTREME_OUTWARD"
    )
    expected = extreme * (1.0 - 0.001)
    assert abs(sl - expected) < 0.0001


# ── SHORT: entry-extreme >= 0.8%, POC valid → POC_OUTWARD ────────────

def test_short_entry_extreme_wide_poc_valid() -> None:
    """When entry-extreme distance >= 0.8% and POC stop is reasonable, use POC stop."""
    strat = _strategy(
        entry_poc_stop_min_tail_pct=0.008,
        entry_poc_stop_buffer_pct=0.001,
        entry_extreme_stop_buffer_pct=0.001,
    )
    entry = 2000.0
    extreme = 2018.0  # 0.9% above entry ( >= 0.8% )
    poc = 2010.0      # POC near entry, far from extreme

    strat.state.upper_extreme_price = extreme
    sp = SweepVolumeProfile(bucket_pct=0.0002)
    sp.add(poc, 100.0)
    sp.add(poc - 1.0, 10.0)
    strat.state.upper_sweep_profile = sp

    sl, mode = strat._select_entry_stop_price(side="SHORT", entry_price=entry)
    assert sl is not None
    assert mode == "POC_OUTWARD", (
        f"entry-extreme distance={(extreme-entry)/entry*100:.1f}% >= 0.8%, "
        f"POC valid → POC_OUTWARD"
    )
    expected_poc_sl = poc * (1.0 + 0.001)
    assert abs(sl - expected_poc_sl) < 0.0001
    assert sl > entry
