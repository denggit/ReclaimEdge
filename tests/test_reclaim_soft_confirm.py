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
        entry_reclaim_v2_enabled=False,  # legacy reclaim path
        entry_cvd_structure_mode="DIVERGENCE_OR_ABSORPTION",
        entry_cvd_divergence_enabled=True,
        entry_cvd_absorption_enabled=True,
        entry_cvd_structure_min_outside_pct=0.001,
        entry_reclaim_confirm_seconds=1.0,
        entry_reclaim_outside_tolerance_pct=0.0002,
        entry_reclaim_new_extreme_buffer_pct=0.0001,
        entry_max_extreme_to_reclaim_seconds=900,
        entry_max_total_setup_seconds=1800,
        entry_max_reclaim_cycles=3,
        entry_reclaim_inside_band=True,
        entry_reclaim_buffer_pct=0.0,
    )
    cfg_kwargs.update(overrides)
    cfg = BollCvdReclaimStrategyConfig(**cfg_kwargs)
    sizer = SimplePositionSizer(SimplePositionSizerConfig(
        dry_run_equity_usdt=10_000, leverage=20, trade_risk_pct=0.01, fee_slippage_buffer_pct=0.001,
    ))
    return BollCvdReclaimStrategy(cfg, sizer)


def _arm_and_get_divergence(strat, boll, side="LONG") -> None:
    """Helper: arm, reach deep enough, confirm divergence."""
    if side == "LONG":
        p1 = 1900 * 0.9985
        strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, fast_cvd=-100.0))
        p2 = 1900 * 0.997
        strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, fast_cvd=-70.0))
    else:
        p1 = 2100 * 1.0015
        strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, fast_cvd=100.0,
                                            cross_positive=False, cross_negative=False,
                                            cvd_increasing=False, cvd_decreasing=False))
        p2 = 2100 * 1.003
        strat.on_tick(p2, 2000, boll, _cvd(ts_ms=2000, price=p2, fast_cvd=70.0,
                                            cross_positive=False, cross_negative=False,
                                            cvd_increasing=False, cvd_decreasing=False))


# ── Test 1: first reclaim tick → pending, no entry ────────────────────

def test_first_reclaim_pending_no_entry() -> None:
    strat = _strategy()
    boll = _boll()
    _arm_and_get_divergence(strat, boll, "LONG")
    reclaim_price = boll.lower * 1.001
    cvd = _cvd(ts_ms=5000, price=reclaim_price, fast_cvd=-65.0,
               cross_positive=True, cvd_increasing=True,
               buy_ratio=0.7, sell_ratio=0.3, no_new_low=True)
    intents = strat.on_tick(reclaim_price, 5000, boll, cvd)
    assert len(intents) == 0
    assert strat.state.lower_reclaim_seen is True
    assert strat.state.lower_reclaim_ts_ms == 5000


# ── Test 2: reclaim < 1s → no entry ──────────────────────────────────

def test_reclaim_below_confirm_seconds_no_entry() -> None:
    strat = _strategy()
    boll = _boll()
    _arm_and_get_divergence(strat, boll, "LONG")
    reclaim_price = boll.lower * 1.001
    cvd = _cvd(ts_ms=5000, price=reclaim_price, fast_cvd=-65.0,
               cross_positive=True, cvd_increasing=True,
               buy_ratio=0.7, sell_ratio=0.3, no_new_low=True)
    strat.on_tick(reclaim_price, 5000, boll, cvd)  # first reclaim → pending
    cvd2 = _cvd(ts_ms=5500, price=reclaim_price, fast_cvd=-65.0,
                cross_positive=True, cvd_increasing=True,
                buy_ratio=0.7, sell_ratio=0.3, no_new_low=True)
    intents = strat.on_tick(reclaim_price, 5500, boll, cvd2)
    assert len(intents) == 0  # only 500ms elapsed


# ── Test 3: reclaim ≥ 1s + CVD direction ok → entry ──────────────────

def test_reclaim_confirmed_enters() -> None:
    strat = _strategy()
    boll = _boll()
    _arm_and_get_divergence(strat, boll, "LONG")
    reclaim_price = boll.lower * 1.001
    cvd1 = _cvd(ts_ms=5000, price=reclaim_price, fast_cvd=-65.0,
                cross_positive=True, cvd_increasing=True,
                buy_ratio=0.7, sell_ratio=0.3, no_new_low=True)
    strat.on_tick(reclaim_price, 5000, boll, cvd1)  # pending
    cvd2 = _cvd(ts_ms=6500, price=reclaim_price, fast_cvd=-65.0,
                cross_positive=True, cvd_increasing=True,
                buy_ratio=0.7, sell_ratio=0.3, no_new_low=True)
    intents = strat.on_tick(reclaim_price, 6500, boll, cvd2)
    assert len(intents) == 1
    assert intents[0].intent_type == "OPEN_LONG"


# ── Test 4: minor breach → timer reset, NOT armed reset ───────────────

def test_minor_breach_resets_timer_not_armed() -> None:
    strat = _strategy()
    boll = _boll()
    _arm_and_get_divergence(strat, boll, "LONG")
    # First reclaim
    reclaim_price = boll.lower * 1.001
    cvd1 = _cvd(ts_ms=5000, price=reclaim_price, fast_cvd=-65.0,
                cross_positive=True, cvd_increasing=True,
                buy_ratio=0.7, sell_ratio=0.3, no_new_low=True)
    strat.on_tick(reclaim_price, 5000, boll, cvd1)
    assert strat.state.lower_reclaim_seen is True
    # Minor breach (within tolerance, does NOT break extreme)
    minor_breach = boll.lower * 0.9999  # just barely outside
    cvd2 = _cvd(ts_ms=5200, price=minor_breach, fast_cvd=-60.0)
    strat.on_tick(minor_breach, 5200, boll, cvd2)
    # Armed should still be True, reclaim_seen still True, timer reset
    assert strat.state.lower_armed is True
    assert strat.state.lower_reclaim_seen is True
    assert strat.state.lower_reclaim_ts_ms == 0  # timer reset


# ── Test 5: major breach → cancel pending, new extreme, cycle++ ───────

def test_major_breach_cancels_pending_new_extreme() -> None:
    strat = _strategy()
    boll = _boll()
    _arm_and_get_divergence(strat, boll, "LONG")
    # First reclaim
    reclaim_price = boll.lower * 1.001
    cvd1 = _cvd(ts_ms=5000, price=reclaim_price, fast_cvd=-65.0,
                cross_positive=True, cvd_increasing=True,
                buy_ratio=0.7, sell_ratio=0.3, no_new_low=True)
    strat.on_tick(reclaim_price, 5000, boll, cvd1)
    # Major breach: new price below old extreme (1894.3 * 0.9999 ≈ 1894.1)
    # Old extreme was ~1894.3 (1900 * 0.997). New price = 1893.0 breaks it.
    major_breach = 1893.0  # clearly below old extreme
    cvd2 = _cvd(ts_ms=5200, price=major_breach, fast_cvd=-80.0)
    strat.on_tick(major_breach, 5200, boll, cvd2)
    # Pending cancelled, new extreme set, cycle incremented
    assert strat.state.lower_reclaim_seen is False
    assert strat.state.lower_reclaim_cycle_count == 1
    assert strat.state.lower_armed is True


# ── Test 6: max cycles exceeded → reset armed ─────────────────────────

def test_max_reclaim_cycles_resets_armed() -> None:
    strat = _strategy(entry_max_reclaim_cycles=1)
    boll = _boll()
    _arm_and_get_divergence(strat, boll, "LONG")
    # First reclaim
    reclaim_price = boll.lower * 1.001
    strat.on_tick(reclaim_price, 5000, boll, _cvd(ts_ms=5000, price=reclaim_price,
                  fast_cvd=-65.0, cross_positive=True, cvd_increasing=True,
                  buy_ratio=0.7, sell_ratio=0.3, no_new_low=True))
    # Major breach → pending cancelled, cycle=1
    strat.on_tick(1893.0, 5200, boll, _cvd(ts_ms=5200, price=1893.0, fast_cvd=-80.0))
    assert strat.state.lower_reclaim_cycle_count == 1
    # Second reclaim
    reclaim2 = boll.lower * 1.001
    strat.on_tick(reclaim2, 6000, boll, _cvd(ts_ms=6000, price=reclaim2,
                  fast_cvd=-75.0, cross_positive=True, cvd_increasing=True,
                  buy_ratio=0.7, sell_ratio=0.3, no_new_low=True))
    # Second major breach → cycle would be 2 > max=1 → reset
    strat.on_tick(1892.0, 6200, boll, _cvd(ts_ms=6200, price=1892.0, fast_cvd=-85.0))
    assert strat.state.lower_armed is False


# ── Test 7: extreme-to-reclaim timeout → reset ────────────────────────

def test_extreme_to_reclaim_timeout_resets() -> None:
    strat = _strategy(entry_max_extreme_to_reclaim_seconds=900)
    boll = _boll()
    _arm_and_get_divergence(strat, boll, "LONG")
    extreme_ts = strat.state.lower_extreme_ts_ms
    # Reclaim after 901 seconds
    reclaim_ts = extreme_ts + 901_000
    reclaim_price = boll.lower * 1.001
    cvd = _cvd(ts_ms=reclaim_ts, price=reclaim_price, fast_cvd=-65.0,
               cross_positive=True, cvd_increasing=True,
               buy_ratio=0.7, sell_ratio=0.3, no_new_low=True)
    intents = strat.on_tick(reclaim_price, reclaim_ts, boll, cvd)
    assert len(intents) == 0
    assert strat.state.lower_armed is False


# ── Test 8: total setup timeout → reset ───────────────────────────────

def test_total_setup_timeout_resets() -> None:
    strat = _strategy(entry_max_total_setup_seconds=1800, max_armed_seconds=3600)
    boll = _boll()
    # First arm at ts=1000
    p1 = 1900 * 0.9985
    strat.on_tick(p1, 1000, boll, _cvd(ts_ms=1000, price=p1, fast_cvd=-100.0))
    assert strat.state.lower_first_armed_ts_ms == 1000
    # Tick at ts=1_801_001 (just over 1800s) with price inside band → total setup timeout fires
    inside_price = boll.lower * 1.001
    cvd = _cvd(ts_ms=1_801_001, price=inside_price, fast_cvd=-100.0)
    strat.on_tick(inside_price, 1_801_001, boll, cvd)
    assert strat.state.lower_armed is False


# ── Test 9: reclaim with wrong CVD direction → no entry ───────────────

def test_reclaim_wrong_cvd_direction_no_entry() -> None:
    strat = _strategy()
    boll = _boll()
    _arm_and_get_divergence(strat, boll, "LONG")
    # First reclaim
    reclaim_price = boll.lower * 1.001
    cvd1 = _cvd(ts_ms=5000, price=reclaim_price, fast_cvd=-65.0,
                cross_positive=False, cvd_increasing=False,
                buy_ratio=0.4, sell_ratio=0.6, no_new_low=False)
    strat.on_tick(reclaim_price, 5000, boll, cvd1)
    # After 1s, CVD still wrong
    cvd2 = _cvd(ts_ms=6500, price=reclaim_price, fast_cvd=-65.0,
                cross_positive=False, cvd_increasing=False,
                buy_ratio=0.4, sell_ratio=0.6, no_new_low=False)
    intents = strat.on_tick(reclaim_price, 6500, boll, cvd2)
    assert len(intents) == 0


# ── Test 10: confirm_seconds=0 → immediate entry ──────────────────────

def test_confirm_seconds_zero_immediate_entry() -> None:
    strat = _strategy(entry_reclaim_confirm_seconds=0)
    boll = _boll()
    _arm_and_get_divergence(strat, boll, "LONG")
    reclaim_price = boll.lower * 1.001
    cvd = _cvd(ts_ms=5000, price=reclaim_price, fast_cvd=-65.0,
               cross_positive=True, cvd_increasing=True,
               buy_ratio=0.7, sell_ratio=0.3, no_new_low=True)
    intents = strat.on_tick(reclaim_price, 5000, boll, cvd)
    assert len(intents) == 1
    assert intents[0].intent_type == "OPEN_LONG"


# ── Fix 2: max_armed_seconds does NOT preempt new extreme window ────────


def test_new_extreme_resets_timeout_window() -> None:
    """A new price extreme within the armed window should extend the valid
    reclaim window — max_armed_seconds must not preempt the new extreme.
    """
    strat = _strategy(
        max_armed_seconds=900,          # old fallback: 15 min
        entry_max_extreme_to_reclaim_seconds=900,  # 15 min per extreme
        entry_max_total_setup_seconds=1800,        # 30 min total
        entry_reclaim_confirm_seconds=0,           # no confirm wait
        entry_cvd_divergence_enabled=False,
        entry_cvd_absorption_enabled=False,
    )
    boll = _boll()

    # t=0 min: first arm
    p1 = 1900 * 0.9985  # outside lower band
    strat.on_tick(p1, 0, boll, _cvd(ts_ms=0, price=p1, fast_cvd=-100.0))
    assert strat.state.lower_armed is True
    assert strat.state.lower_armed_ts_ms == 0
    first_extreme_ts = strat.state.lower_extreme_ts_ms

    # t=10 min (600_000 ms): new extreme
    p2 = 1893.0  # lower price = new extreme
    strat.on_tick(p2, 600_000, boll, _cvd(ts_ms=600_000, price=p2, fast_cvd=-120.0))
    assert strat.state.lower_armed is True  # still armed
    assert strat.state.lower_extreme_ts_ms == 600_000  # updated
    assert strat.state.lower_extreme_price == 1893.0

    # t=20 min (1_200_000 ms): 20 min from first arm but only 10 min from new extreme
    # old max_armed_seconds=900 would expire at 900_000, but new extreme at 600_000
    # gives a new window until 600_000 + 900_000 = 1_500_000.
    # The old bug would reset here because 1_200_000 - 0 > 900_000.
    inside_price = boll.lower * 1.001
    strat.on_tick(inside_price, 1_200_000, boll, _cvd(ts_ms=1_200_000, price=inside_price, fast_cvd=-110.0))
    assert strat.state.lower_armed is True  # NOT reset by max_armed_seconds

    # t=26 min (1_560_000 ms): still within new extreme window (600k + 900k = 1_500k)
    # Actually 1_560_000 > 1_500_000 so should expire the extreme window
    # But total setup is 1800s = 1_800_000 ms, so armed should still be here
    # Wait, 1_560_000 > 600_000 + 900_000 = 1_500_000, so _expire_armed_state would reset
    # because the extreme_to_reclaim_timeout fires.
    strat.on_tick(inside_price, 1_560_000, boll, _cvd(ts_ms=1_560_000, price=inside_price, fast_cvd=-110.0))
    # extreme_to_reclaim_timeout: 1_560_000 - 600_000 = 960_000 > 900_000 → reset
    assert strat.state.lower_armed is False  # expired by extreme timeout


def test_no_new_extreme_falls_back_to_max_armed_seconds() -> None:
    """When no extreme timestamp exists (e.g. price went outside but never
    reached min_outside_pct depth), max_armed_seconds is used as fallback.
    """
    strat = _strategy(
        max_armed_seconds=900,
        entry_max_extreme_to_reclaim_seconds=900,
        entry_max_total_setup_seconds=3600,
        entry_reclaim_confirm_seconds=0,
        entry_cvd_divergence_enabled=False,
        entry_cvd_absorption_enabled=False,
    )
    boll = _boll()
    # Arm lower but never reaches deep enough → no extreme timestamp
    p1 = 1900 * 0.9995  # barely outside, NOT deep enough
    strat.on_tick(p1, 0, boll, _cvd(ts_ms=0, price=p1, fast_cvd=-50.0))
    assert strat.state.lower_armed is True
    # extreme_ts_ms is 0 because never deep enough → no extreme recorded
    assert strat.state.lower_extreme_ts_ms == 0

    # t=901s (901_000 ms): max_armed_seconds=900 → should expire
    inside_price = boll.lower * 1.001
    strat.on_tick(inside_price, 901_000, boll, _cvd(ts_ms=901_000, price=inside_price, fast_cvd=-50.0))
    assert strat.state.lower_armed is False  # expired by max_armed_seconds fallback


def test_total_setup_timeout_still_fires() -> None:
    """The total setup timeout (entry_max_total_setup_seconds) is enforced
    via _update_armed_state() and fires independently of extreme timeouts.
    """
    strat = _strategy(
        max_armed_seconds=3600,
        entry_max_extreme_to_reclaim_seconds=900,
        entry_max_total_setup_seconds=1800,
        entry_reclaim_confirm_seconds=0,
        entry_cvd_divergence_enabled=False,
        entry_cvd_absorption_enabled=False,
    )
    boll = _boll()
    # Arm lower, deep enough
    p1 = 1900 * 0.9985
    strat.on_tick(p1, 0, boll, _cvd(ts_ms=0, price=p1, fast_cvd=-100.0))
    assert strat.state.lower_first_armed_ts_ms == 0

    # t=10 min: new extreme refreshes the extreme window
    p2 = 1893.0
    strat.on_tick(p2, 600_000, boll, _cvd(ts_ms=600_000, price=p2, fast_cvd=-120.0))
    assert strat.state.lower_armed is True

    # t=31 min (1_860_000 ms): total setup = 1800s = 1_800_000 → should expire
    inside_price = boll.lower * 1.001
    strat.on_tick(inside_price, 1_860_000, boll, _cvd(ts_ms=1_860_000, price=inside_price, fast_cvd=-110.0))
    assert strat.state.lower_armed is False  # expired by total setup timeout
