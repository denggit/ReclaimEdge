from __future__ import annotations

from src.indicators.cvd_tracker import CvdSnapshot
from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import BollCvdReclaimStrategy, BollCvdReclaimStrategyConfig


def _boll(middle=2000, upper=2100, lower=1900, alert_switch_on=True) -> BollSnapshot:
    return BollSnapshot(
        inst_id="ETH-USDT-SWAP", candle_ts_ms=1000, close=middle,
        middle=middle, upper=upper, lower=lower,
        upper_distance_pct=0.0, lower_distance_pct=0.001,
        alert_switch_on=alert_switch_on, live_mode=True,
    )


def _cvd(ts_ms=1000, price=1901, fast_cvd=1.0, previous_fast_cvd=0.0,
         buy_ratio=0.7, sell_ratio=0.3, cross_positive=True, cross_negative=False,
         cvd_increasing=True, cvd_decreasing=False, no_new_low=True, no_new_high=True,
         up_burst=False, down_burst=False) -> CvdSnapshot:
    return CvdSnapshot(
        ts_ms=ts_ms, price=price,
        side="buy" if buy_ratio >= sell_ratio else "sell",
        size=1.0, signed_delta=1.0, total_cvd=10.0,
        fast_cvd=fast_cvd, previous_fast_cvd=previous_fast_cvd,
        buy_volume=70.0, sell_volume=30.0, buy_ratio=buy_ratio, sell_ratio=sell_ratio,
        cross_positive=cross_positive, cross_negative=cross_negative,
        cvd_increasing=cvd_increasing, cvd_decreasing=cvd_decreasing,
        no_new_low=no_new_low, no_new_high=no_new_high,
        window_low=1897.0, window_high=1905.0,
        burst_net_move_pct=0.0, burst_range_pct=0.002, baseline_range_pct=0.001,
        burst_move_ratio=2.0, burst_volume=10.0, baseline_volume=5.0, burst_volume_ratio=2.0,
        up_burst=up_burst, down_burst=down_burst,
    )


def _strategy(**overrides) -> BollCvdReclaimStrategy:
    cfg_kwargs = dict(
        min_outside_pct=0.001,
        entry_min_reward_risk=0.0, entry_fee_slippage_buffer_pct=0.0,
        order_cooldown_seconds=0,
        entry_reclaim_v2_enabled=False,  # legacy path for structure tests
        entry_cvd_structure_mode="DIVERGENCE_OR_ABSORPTION",
        entry_cvd_divergence_enabled=True,
        entry_cvd_absorption_enabled=True,
        entry_cvd_structure_min_outside_pct=0.001,
        entry_reclaim_confirm_seconds=0,  # no wait for structure tests
        entry_reclaim_inside_band=True,
    )
    cfg_kwargs.update(overrides)
    cfg = BollCvdReclaimStrategyConfig(**cfg_kwargs)
    sizer = SimplePositionSizer(SimplePositionSizerConfig(
        dry_run_equity_usdt=10_000, leverage=20, trade_risk_pct=0.01, fee_slippage_buffer_pct=0.001,
    ))
    return BollCvdReclaimStrategy(cfg, sizer)


# ── Test 1: upper breach < 0.1% → no entry ────────────────────────────

def test_short_breach_below_01pct_no_entry() -> None:
    strat = _strategy()
    boll = _boll()
    price = 2100 * 1.0005  # ~0.05% outside
    cvd = _cvd(ts_ms=1000, price=price, fast_cvd=100.0,
               cross_positive=False, cross_negative=True, cvd_increasing=False, cvd_decreasing=True)
    strat.on_tick(price, 1000, boll, cvd)
    assert strat.state.upper_armed is True
    assert strat.state.upper_deep_enough is False
    reclaim_price = boll.upper * 0.999
    cvd2 = _cvd(ts_ms=5000, price=reclaim_price, fast_cvd=90.0,
                buy_ratio=0.3, sell_ratio=0.7, no_new_high=True,
                cross_positive=False, cross_negative=True, cvd_increasing=False, cvd_decreasing=True)
    intents = strat.on_tick(reclaim_price, 5000, boll, cvd2)
    assert len(intents) == 0


# ── Test 2: lower breach < 0.1% → no entry ────────────────────────────

def test_long_breach_below_01pct_no_entry() -> None:
    strat = _strategy()
    boll = _boll()
    price = 1900 * 0.9995
    cvd = _cvd(ts_ms=1000, price=price, fast_cvd=-100.0)
    strat.on_tick(price, 1000, boll, cvd)
    assert strat.state.lower_deep_enough is False
    reclaim_price = boll.lower * 1.001
    cvd2 = _cvd(ts_ms=5000, price=reclaim_price, fast_cvd=-90.0)
    intents = strat.on_tick(reclaim_price, 5000, boll, cvd2)
    assert len(intents) == 0


# ── Test 3-4: upper divergence ────────────────────────────────────────

def test_short_no_divergence_cvd_follows() -> None:
    strat = _strategy(entry_cvd_structure_mode="DIVERGENCE_ONLY",
                       entry_cvd_absorption_enabled=False)
    boll = _boll()
    # First extreme deep enough
    p1 = 2100 * 1.0015
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, fast_cvd=100.0))
    assert strat.state.upper_deep_enough is True
    # New extreme: price higher, CVD also higher → no divergence
    p2 = 2100 * 1.003
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, fast_cvd=120.0))
    assert strat.state.upper_cvd_divergence_confirmed is False


def test_short_divergence_confirmed() -> None:
    strat = _strategy(entry_cvd_structure_mode="DIVERGENCE_ONLY",
                       entry_cvd_absorption_enabled=False)
    boll = _boll()
    p1 = 2100 * 1.0015
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, fast_cvd=100.0))
    # New extreme: price higher, CVD LOWER → divergence!
    p2 = 2100 * 1.003
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, fast_cvd=80.0))
    assert strat.state.upper_cvd_divergence_confirmed is True


# ── Test 5-6: lower divergence ────────────────────────────────────────

def test_long_no_divergence_cvd_follows() -> None:
    strat = _strategy(entry_cvd_structure_mode="DIVERGENCE_ONLY",
                       entry_cvd_absorption_enabled=False)
    boll = _boll()
    p1 = 1900 * 0.9985
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, fast_cvd=-100.0))
    # New low, CVD also lower → no divergence
    p2 = 1900 * 0.997
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, fast_cvd=-120.0))
    assert strat.state.lower_cvd_divergence_confirmed is False


def test_long_divergence_confirmed() -> None:
    strat = _strategy(entry_cvd_structure_mode="DIVERGENCE_ONLY",
                       entry_cvd_absorption_enabled=False)
    boll = _boll()
    p1 = 1900 * 0.9985
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, fast_cvd=-100.0))
    # New low, CVD NOT lower → divergence!
    p2 = 1900 * 0.997
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, fast_cvd=-70.0))
    assert strat.state.lower_cvd_divergence_confirmed is True


# ── Test 7-8: upper absorption ────────────────────────────────────────

def test_short_absorption_confirmed() -> None:
    strat = _strategy(entry_cvd_structure_mode="ABSORPTION_ONLY",
                       entry_cvd_divergence_enabled=False)
    boll = _boll()
    # First arm: reference_fast_cvd = 100
    p1 = 2100 * 1.0015
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, fast_cvd=100.0))
    assert strat.state.upper_reference_fast_cvd == 100.0
    # extreme_fast_cvd recorded by _check_upper_cvd_structure
    # extreme_fast_cvd=100.0, reference_fast_cvd=100.0 → NOT absorption (no weakening)
    # Need extreme_fast_cvd to be <= reference to confirm absorption
    # Actually: extreme_fast_cvd <= reference → absorption
    # 100 <= 100 → True → absorption confirmed!
    assert strat.state.upper_cvd_absorption_confirmed is True


def test_short_no_absorption_cvd_strengthens() -> None:
    strat = _strategy(entry_cvd_structure_mode="ABSORPTION_ONLY",
                       entry_cvd_divergence_enabled=False)
    boll = _boll()
    # Tick 1: price just barely outside, NOT deep enough
    # reference_fast_cvd = 50 (weak CVD initially)
    p1 = 2100 * 1.0002  # ~2100.42, 0.02% outside (not deep enough)
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, fast_cvd=50.0,
                cross_positive=False, cross_negative=False,
                cvd_increasing=False, cvd_decreasing=False))
    assert strat.state.upper_armed is True
    assert strat.state.upper_deep_enough is False
    # Tick 2: price goes much deeper, now deep enough, CVD surged to 120
    # reference = 50, extreme_fast_cvd = 120 → 120 <= 50 → False → no absorption
    p2 = 2100 * 1.003  # ~2106.30, deep enough
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, fast_cvd=120.0,
                cross_positive=False, cross_negative=False,
                cvd_increasing=False, cvd_decreasing=False))
    assert strat.state.upper_deep_enough is True
    assert strat.state.upper_cvd_absorption_confirmed is False


# ── Test 9-10: lower absorption ───────────────────────────────────────

def test_long_absorption_confirmed() -> None:
    strat = _strategy(entry_cvd_structure_mode="ABSORPTION_ONLY",
                       entry_cvd_divergence_enabled=False)
    boll = _boll()
    # reference_fast_cvd = -100
    p1 = 1900 * 0.9985
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, fast_cvd=-100.0))
    # extreme_fast_cvd = -100, reference = -100
    # absorption: extreme >= reference → -100 >= -100 → True
    assert strat.state.lower_cvd_absorption_confirmed is True


def test_long_no_absorption_cvd_weakens() -> None:
    strat = _strategy(entry_cvd_structure_mode="ABSORPTION_ONLY",
                       entry_cvd_divergence_enabled=False)
    boll = _boll()
    # Tick 1: price just barely outside, NOT deep enough
    # reference_fast_cvd = -50 (moderate CVD)
    p1 = 1900 * 0.9998  # ~1899.62, 0.02% outside (not deep enough)
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, fast_cvd=-50.0))
    assert strat.state.lower_armed is True
    assert strat.state.lower_deep_enough is False
    # Tick 2: price goes deeper, now deep enough, CVD goes much lower (-150)
    # reference = -50, extreme_fast_cvd = -150 → -150 >= -50 → False → no absorption
    p2 = 1900 * 0.997  # ~1894.30, deep enough
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, fast_cvd=-150.0))
    assert strat.state.lower_deep_enough is True
    assert strat.state.lower_cvd_absorption_confirmed is False


# ── Test 11-13: structure mode ────────────────────────────────────────

def test_divergence_only_ignores_absorption() -> None:
    strat = _strategy(entry_cvd_structure_mode="DIVERGENCE_ONLY",
                       entry_cvd_absorption_enabled=False)
    boll = _boll()
    p1 = 1900 * 0.9985
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, fast_cvd=-100.0))
    # absorption may be true from first extreme but DIVERGENCE_ONLY ignores it
    # divergence requires a second extreme → not confirmed
    assert strat._lower_cvd_structure_ok() is False


def test_absorption_only_ignores_divergence() -> None:
    strat = _strategy(entry_cvd_structure_mode="ABSORPTION_ONLY",
                       entry_cvd_divergence_enabled=False)
    boll = _boll()
    # Two extremes with divergence
    p1 = 2100 * 1.0015
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, fast_cvd=100.0))
    p2 = 2100 * 1.003
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, fast_cvd=80.0))
    # divergence would be true but ABSORPTION_ONLY ignores it
    # absorption needs extreme_fast_cvd <= reference: at this point extreme_fast_cvd=80, reference=100
    # 80 <= 100 → absorption confirmed
    assert strat._upper_cvd_structure_ok() is True  # absorption confirmed


def test_divergence_or_absorption_either_ok() -> None:
    strat = _strategy(entry_cvd_structure_mode="DIVERGENCE_OR_ABSORPTION")
    boll = _boll()
    # Single extreme with absorption confirmed
    p1 = 2100 * 1.0015
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, fast_cvd=100.0))
    # absorption confirmed immediately (extreme_fast_cvd <= reference_fast_cvd)
    assert strat._upper_cvd_structure_ok() is True


# ── Fix 3: divergence only on new price extreme ─────────────────────────


def test_upper_no_new_extreme_no_divergence() -> None:
    """Price did NOT break a new high, fast_cvd dropped — divergence must NOT
    be confirmed because there's no new price extreme."""
    strat = _strategy(entry_cvd_structure_mode="DIVERGENCE_ONLY",
                       entry_cvd_absorption_enabled=False)
    boll = _boll()
    # First extreme
    p1 = 2100 * 1.0015
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, fast_cvd=100.0))
    assert strat.state.upper_deep_enough is True
    assert strat.state.upper_extreme_fast_cvd == 100.0

    # Same extreme price (within buffer), fast_cvd dropped
    # This should NOT trigger divergence because price didn't break new extreme
    p2 = p1  # same price, not a new extreme
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, fast_cvd=80.0))
    assert strat.state.upper_cvd_divergence_confirmed is False  # no new extreme → no divergence check


def test_upper_new_extreme_divergence_confirmed() -> None:
    """Price breaks new high, fast_cvd does NOT — divergence confirmed."""
    strat = _strategy(entry_cvd_structure_mode="DIVERGENCE_ONLY",
                       entry_cvd_absorption_enabled=False)
    boll = _boll()
    p1 = 2100 * 1.0015
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, fast_cvd=100.0))
    # New extreme: price higher, fast_cvd lower → divergence
    p2 = 2100 * 1.003
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, fast_cvd=80.0))
    assert strat.state.upper_cvd_divergence_confirmed is True


def test_upper_new_extreme_cvd_confirms_no_divergence() -> None:
    """Price breaks new high, fast_cvd also makes new high — no divergence,
    extreme_fast_cvd updated."""
    strat = _strategy(entry_cvd_structure_mode="DIVERGENCE_ONLY",
                       entry_cvd_absorption_enabled=False)
    boll = _boll()
    p1 = 2100 * 1.0015
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, fast_cvd=100.0))
    assert strat.state.upper_extreme_fast_cvd == 100.0
    # New extreme: price higher, fast_cvd ALSO higher → CVD confirms → update reference
    p2 = 2100 * 1.003
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, fast_cvd=120.0))
    assert strat.state.upper_cvd_divergence_confirmed is False  # CVD confirms → no divergence
    assert strat.state.upper_extreme_fast_cvd == 120.0  # updated for next comparison


def test_lower_no_new_extreme_no_divergence() -> None:
    """Price did NOT break a new low, fast_cvd rose — divergence must NOT
    be confirmed because there's no new price extreme."""
    strat = _strategy(entry_cvd_structure_mode="DIVERGENCE_ONLY",
                       entry_cvd_absorption_enabled=False)
    boll = _boll()
    p1 = 1900 * 0.9985
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, fast_cvd=-100.0))
    assert strat.state.lower_deep_enough is True
    assert strat.state.lower_extreme_fast_cvd == -100.0

    # Same extreme price, fast_cvd rose → no new extreme, no divergence check
    p2 = p1  # same price, not a new extreme
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, fast_cvd=-70.0))
    assert strat.state.lower_cvd_divergence_confirmed is False  # no new extreme → no divergence check


def test_lower_new_extreme_divergence_confirmed() -> None:
    """Price breaks new low, fast_cvd does NOT — divergence confirmed."""
    strat = _strategy(entry_cvd_structure_mode="DIVERGENCE_ONLY",
                       entry_cvd_absorption_enabled=False)
    boll = _boll()
    p1 = 1900 * 0.9985
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, fast_cvd=-100.0))
    # New extreme: price lower, fast_cvd higher → divergence
    p2 = 1900 * 0.997
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, fast_cvd=-70.0))
    assert strat.state.lower_cvd_divergence_confirmed is True


def test_lower_new_extreme_cvd_confirms_no_divergence() -> None:
    """Price breaks new low, fast_cvd also makes new low — no divergence,
    extreme_fast_cvd updated."""
    strat = _strategy(entry_cvd_structure_mode="DIVERGENCE_ONLY",
                       entry_cvd_absorption_enabled=False)
    boll = _boll()
    p1 = 1900 * 0.9985
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, fast_cvd=-100.0))
    assert strat.state.lower_extreme_fast_cvd == -100.0
    # New extreme: price lower, fast_cvd ALSO lower → CVD confirms → update reference
    p2 = 1900 * 0.997
    strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, fast_cvd=-120.0))
    assert strat.state.lower_cvd_divergence_confirmed is False  # CVD confirms → no divergence
    assert strat.state.lower_extreme_fast_cvd == -120.0  # updated for next comparison


def test_absorption_still_works_on_single_extreme() -> None:
    """Absorption should still be confirmed on the first valid extreme tick
    (within-buffer micro-breaches do not count as new extremes)."""
    strat = _strategy(entry_cvd_structure_mode="ABSORPTION_ONLY",
                       entry_cvd_divergence_enabled=False)
    boll = _boll()
    # First arm: reference_fast_cvd set
    p1 = 2100 * 1.0015
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, fast_cvd=100.0))
    # extreme_fast_cvd = 100.0, reference_fast_cvd = 100.0
    # extreme <= reference → 100 <= 100 → absorption confirmed on first extreme
    assert strat.state.upper_cvd_absorption_confirmed is True
