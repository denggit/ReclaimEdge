from __future__ import annotations

from src.indicators.cvd_tracker import CvdSnapshot
from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import BollCvdReclaimStrategy, BollCvdReclaimStrategyConfig


def _boll() -> BollSnapshot:
    return BollSnapshot(
        inst_id="ETH-USDT-SWAP",
        candle_ts_ms=1000,
        close=1901,
        middle=2000,
        upper=2100,
        lower=1900,
        upper_distance_pct=0.0,
        lower_distance_pct=0.001,
        alert_switch_on=True,
        live_mode=True,
    )


def _cvd() -> CvdSnapshot:
    return CvdSnapshot(
        ts_ms=1000,
        price=1901,
        side="buy",
        size=1,
        signed_delta=1,
        total_cvd=1,
        fast_cvd=1,
        previous_fast_cvd=0,
        buy_volume=70,
        sell_volume=30,
        buy_ratio=0.7,
        sell_ratio=0.3,
        cross_positive=True,
        cross_negative=False,
        cvd_increasing=True,
        cvd_decreasing=False,
        no_new_low=True,
        no_new_high=True,
        window_low=1897,
        window_high=1905,
        burst_net_move_pct=0.0,
        burst_range_pct=0.002,
        baseline_range_pct=0.001,
        burst_move_ratio=2.0,
        burst_volume=10,
        baseline_volume=5,
        burst_volume_ratio=2.0,
        up_burst=False,
        down_burst=False,
    )


def _strategy() -> BollCvdReclaimStrategy:
    cfg = BollCvdReclaimStrategyConfig(
        min_outside_pct=0.001,
        entry_min_reward_risk=1.0,
        entry_fee_slippage_buffer_pct=0.001,
        order_cooldown_seconds=0,
        entry_reclaim_v2_enabled=False,  # legacy path
        entry_cvd_divergence_enabled=False,
        entry_cvd_absorption_enabled=False,
        entry_reclaim_confirm_seconds=0,
    )
    sizer = SimplePositionSizer(
        SimplePositionSizerConfig(
            dry_run_equity_usdt=10_000,
            leverage=20,
            trade_risk_pct=0.01,
            fee_slippage_buffer_pct=0.001,
        )
    )
    return BollCvdReclaimStrategy(cfg, sizer)


def test_long_enters_only_after_reclaim_inside_band_and_uses_risk_size() -> None:
    strat = _strategy()
    boll = _boll()
    cvd = _cvd()

    # First tick arms the lower-band false breakout.
    assert strat.on_tick(1897.0, 1000, boll, cvd) == []
    assert strat.state.lower_armed is True

    # Reclaim back inside the lower band with reverse CVD should open once.
    intents = strat.on_tick(1901.0, 2000, boll, cvd)

    assert len(intents) == 1
    intent = intents[0]
    assert intent.intent_type == "OPEN_LONG"
    assert intent.entry_protective_sl_price is not None
    assert intent.entry_protective_sl_price < 1897.0
    assert intent.size.sizing_mode == "risk"
    assert intent.size.risk_usdt == 100.0
    assert strat.state.layers == 1


def test_existing_position_never_generates_add_intent() -> None:
    strat = _strategy()
    boll = _boll()
    cvd = _cvd()
    strat.on_tick(1897.0, 1000, boll, cvd)
    strat.on_tick(1901.0, 2000, boll, cvd)

    intents = strat.on_tick(1901.5, 3000, boll, cvd)

    assert all(i.intent_type not in {"ADD_LONG", "ADD_SHORT"} for i in intents)
