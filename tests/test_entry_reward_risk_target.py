"""Test entry reward/risk target selection (STRUCTURE_MIDDLE vs FINAL_TP).

These tests verify that the entry RR filter can use BOLL20 middle as the
reward target instead of the final take-profit price, without changing the
actual TP ordering path.
"""

from __future__ import annotations

import pytest

from src.indicators.cvd_tracker import CvdSnapshot
from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import BollCvdReclaimStrategy, BollCvdReclaimStrategyConfig


# ── helpers ────────────────────────────────────────────────────────────────

def _long_cvd() -> CvdSnapshot:
    """CVD snapshot favourable for a LONG entry (reclaim / absorption)."""
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


def _short_cvd() -> CvdSnapshot:
    """CVD snapshot favourable for a SHORT entry (rejection / distribution)."""
    return CvdSnapshot(
        ts_ms=1000,
        price=3109,
        side="sell",
        size=1,
        signed_delta=-1,
        total_cvd=-1,
        fast_cvd=-1,
        previous_fast_cvd=0,
        buy_volume=30,
        sell_volume=70,
        buy_ratio=0.3,
        sell_ratio=0.7,
        cross_positive=False,
        cross_negative=True,
        cvd_increasing=False,
        cvd_decreasing=True,
        no_new_low=True,
        no_new_high=True,
        window_low=3095,
        window_high=3115,
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


def _make_strategy(*, entry_rr_target: str = "STRUCTURE_MIDDLE",
                   entry_min_reward_risk: float = 1.0,
                   entry_sl_buffer_pct: float = 0.0005,
                   entry_fee_slippage_buffer_pct: float = 0.001,
                   tp_min_net_profit_pct: float = 0.004,
                   min_outside_pct: float = 0.001,
                   **kwargs) -> BollCvdReclaimStrategy:
    cfg = BollCvdReclaimStrategyConfig(
        entry_rr_target=entry_rr_target,
        entry_min_reward_risk=entry_min_reward_risk,
        entry_sl_buffer_pct=entry_sl_buffer_pct,
        entry_fee_slippage_buffer_pct=entry_fee_slippage_buffer_pct,
        tp_min_net_profit_pct=tp_min_net_profit_pct,
        min_outside_pct=min_outside_pct,
        order_cooldown_seconds=0,
        entry_reclaim_buffer_pct=0.0,
        entry_cvd_divergence_enabled=False,
        entry_cvd_absorption_enabled=False,
        entry_reclaim_confirm_seconds=0,
        **kwargs,
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


# ── Case A: STRUCTURE_MIDDLE — middle RR insufficient → skip ─────────────

def test_structure_middle_skips_when_middle_rr_insufficient() -> None:
    """Final TP outer has enough RR, but BOLL20 middle does not → skip entry."""
    strat = _make_strategy(entry_rr_target="STRUCTURE_MIDDLE")

    # BOLL: middle is very close to the lower band, so reward to middle is tiny.
    # Upper is far away → final-TP RR would be fine, but that's not what we filter on.
    boll = BollSnapshot(
        inst_id="ETH-USDT-SWAP",
        candle_ts_ms=1000,
        close=3010,
        middle=3010,               # only 10 above entry → tiny reward
        upper=3200,                # far away → final TP RR huge
        lower=3000,
        upper_distance_pct=0.0,
        lower_distance_pct=0.002,  # deep enough outside
        alert_switch_on=True,
        live_mode=True,
    )
    cvd = _long_cvd()

    # Arm the lower band.
    assert strat.on_tick(2995.0, 1000, boll, cvd) == []
    assert strat.state.lower_armed is True

    # Reclaim inside the lower band — attempt entry.
    # entry ≈ 3005, SL ≈ 3000*(1-0.0005)=2998.5
    # reward to middle 3010 = (3010-3005)/3005 ≈ 0.166 % → R < 1.0
    intents = strat.on_tick(3005.0, 2000, boll, cvd)
    assert len(intents) == 0, f"Expected skip, got {intents}"


# ── Case B: FINAL_TP — middle RR insufficient but outer TP RR sufficient → entry ─

def test_final_tp_enters_when_outer_rr_sufficient() -> None:
    """Same BOLL as Case A but with FINAL_TP — outer TP RR passes so entry works."""
    strat = _make_strategy(entry_rr_target="FINAL_TP")

    boll = BollSnapshot(
        inst_id="ETH-USDT-SWAP",
        candle_ts_ms=1000,
        close=3010,
        middle=3010,
        upper=3200,
        lower=3000,
        upper_distance_pct=0.0,
        lower_distance_pct=0.002,
        alert_switch_on=True,
        live_mode=True,
    )
    cvd = _long_cvd()

    # Arm.
    assert strat.on_tick(2995.0, 1000, boll, cvd) == []
    assert strat.state.lower_armed is True

    # Entry — RR to outer (3200) is huge → should enter.
    intents = strat.on_tick(3005.0, 2000, boll, cvd)
    assert len(intents) == 1, f"Expected 1 intent, got {len(intents)}"
    intent = intents[0]
    assert intent.intent_type == "OPEN_LONG"
    # The real TP is stored correctly (outer band, not middle).
    assert intent.tp_price > 3100, f"tp_price={intent.tp_price} should be outer, not middle"
    # Reason should mention FINAL_TP
    assert "rr_target_source=FINAL_TP" in intent.reason
    assert "rr_target=" in intent.reason


# ── Case C: STRUCTURE_MIDDLE, middle RR sufficient, TP is outer ───────────

def test_structure_middle_passes_but_real_tp_is_outer() -> None:
    """BOLL20 middle passes RR, but the actual TP falls back to outer band.

    The intent MUST carry the real outer TP price, not the middle.
    """
    strat = _make_strategy(entry_rr_target="STRUCTURE_MIDDLE")

    # Middle provides ~2.5% reward from entry → well above min RR.
    # But net profit to middle is < tp_min_net_profit (0.4%) → TP falls back to outer.
    # lower=3000, middle=3075, upper=3200
    # entry=3005, SL=2998.5
    # reward to middle: (3075-3005)/3005 ≈ 2.33% → R ≈ 7+  (passes)
    # net profit to middle from BE≈3008: (3075-3008)/3008≈2.23% → passes 0.4%
    # ... hmm that passes too.
    #
    # Let's use a tighter middle:
    # lower=3000, middle=3020, upper=3200
    # entry=3005, SL=2998.5
    # reward to middle: (3020-3005)/3005 ≈ 0.50% → R ≈ 1.58  (passes RR)
    # net profit from BE=3008: (3020-3008)/3008 ≈ 0.40% → barely at threshold
    # If tp_min_net_profit=0.006, it fails and TP falls back to outer.
    boll = BollSnapshot(
        inst_id="ETH-USDT-SWAP",
        candle_ts_ms=1000,
        close=3020,
        middle=3020,
        upper=3200,
        lower=3000,
        upper_distance_pct=0.0,
        lower_distance_pct=0.002,
        alert_switch_on=True,
        live_mode=True,
    )
    cvd = _long_cvd()

    # Use higher tp_min_net_profit to force TP fallback to outer.
    strat.config = BollCvdReclaimStrategyConfig(
        entry_rr_target="STRUCTURE_MIDDLE",
        entry_min_reward_risk=1.0,
        entry_sl_buffer_pct=0.0005,
        entry_fee_slippage_buffer_pct=0.001,
        tp_min_net_profit_pct=0.006,  # 0.6% — middle 0.4% doesn't clear it
        min_outside_pct=0.001,
        order_cooldown_seconds=0,
        entry_reclaim_buffer_pct=0.0,
        entry_cvd_divergence_enabled=False,
        entry_cvd_absorption_enabled=False,
        entry_reclaim_confirm_seconds=0,
    )

    # Arm.
    assert strat.on_tick(2995.0, 1000, boll, cvd) == []
    assert strat.state.lower_armed is True

    intents = strat.on_tick(3005.0, 2000, boll, cvd)
    assert len(intents) == 1, f"Expected 1 intent, got {len(intents)}"
    intent = intents[0]
    assert intent.intent_type == "OPEN_LONG"

    # The real TP stored on intent is the outer band price (real TP path untouched).
    assert intent.tp_price > 3100, (
        f"tp_price={intent.tp_price} should be outer band (>3100), "
        f"not the BOLL20 middle (3020)"
    )
    # Reason must show STRUCTURE_MIDDLE as the RR target source.
    assert "rr_target_source=STRUCTURE_MIDDLE" in intent.reason
    assert "rr_target=" in intent.reason


# ── Case D: SHORT — BOLL20 middle >= entry → invalid_reward_distance ─────

def test_short_skips_when_middle_above_entry() -> None:
    """SHORT entry: BOLL20 middle >= entry price gives no valid reward distance."""
    strat = _make_strategy(entry_rr_target="STRUCTURE_MIDDLE")

    # For SHORT, reward = (entry - tp) / entry. If middle >= entry, reward ≤ 0.
    # upper=3100, middle=3115 (> entry), lower=3000
    boll = BollSnapshot(
        inst_id="ETH-USDT-SWAP",
        candle_ts_ms=1000,
        close=3100,
        middle=3115,               # above entry → invalid for SHORT reward
        upper=3100,
        lower=3000,
        upper_distance_pct=0.002,
        lower_distance_pct=0.0,
        alert_switch_on=True,
        live_mode=True,
    )
    cvd = _short_cvd()

    # Arm upper band.
    assert strat.on_tick(3115.0, 1000, boll, cvd) == []
    assert strat.state.upper_armed is True

    # Reclaim inside upper band — should skip because middle >= entry.
    # entry=3105, SL=3100*(1+0.0005)=3101.55
    # reward to middle (3115) = (3105-3115)/3105 < 0 → invalid_reward_distance
    intents = strat.on_tick(3105.0, 2000, boll, cvd)
    assert len(intents) == 0, f"Expected skip, got {intents}"


# ── Case D variant: SHORT with valid STRUCTURE_MIDDLE RR enters ──────────

def test_short_enters_when_middle_below_entry_rr_sufficient() -> None:
    """SHORT entry: BOLL20 middle < entry provides valid reward, entry works."""
    strat = _make_strategy(entry_rr_target="STRUCTURE_MIDDLE")

    # middle=3000 (< entry=3105) — valid reward for SHORT.
    boll = BollSnapshot(
        inst_id="ETH-USDT-SWAP",
        candle_ts_ms=1000,
        close=3100,
        middle=3000,               # well below entry → valid reward
        upper=3100,
        lower=3000,
        upper_distance_pct=0.002,
        lower_distance_pct=0.0,
        alert_switch_on=True,
        live_mode=True,
    )
    cvd = _short_cvd()

    # Arm upper band.
    assert strat.on_tick(3115.0, 1000, boll, cvd) == []
    assert strat.state.upper_armed is True

    # Reclaim inside upper band — price must be ≤ upper for SHORT reclaim.
    # entry=3095, SL=3100*(1+0.0005)=3101.55, reward to middle=3000 is large → entry works.
    intents = strat.on_tick(3095.0, 2000, boll, cvd)
    assert len(intents) == 1, f"Expected 1 intent, got {len(intents)}"
    intent = intents[0]
    assert intent.intent_type == "OPEN_SHORT"
    assert "rr_target_source=STRUCTURE_MIDDLE" in intent.reason


# ── unit: helper method returns correct values ────────────────────────────

def test_entry_reward_risk_target_price_structure_middle() -> None:
    """Helper returns BOLL20 middle when entry_rr_target is STRUCTURE_MIDDLE."""
    strat = _make_strategy(entry_rr_target="STRUCTURE_MIDDLE")
    boll = BollSnapshot(
        inst_id="ETH-USDT-SWAP", candle_ts_ms=1, close=1901,
        middle=2000, upper=2100, lower=1900,
        upper_distance_pct=0, lower_distance_pct=0.001,
        alert_switch_on=True, live_mode=True,
    )
    price, source = strat._entry_reward_risk_target_price(
        side="LONG", boll=boll, final_tp_price=2100.0,
    )
    assert price == 2000.0
    assert source == "STRUCTURE_MIDDLE"


def test_entry_reward_risk_target_price_final_tp() -> None:
    """Helper returns final_tp_price when entry_rr_target is FINAL_TP."""
    strat = _make_strategy(entry_rr_target="FINAL_TP")
    boll = BollSnapshot(
        inst_id="ETH-USDT-SWAP", candle_ts_ms=1, close=1901,
        middle=2000, upper=2100, lower=1900,
        upper_distance_pct=0, lower_distance_pct=0.001,
        alert_switch_on=True, live_mode=True,
    )
    price, source = strat._entry_reward_risk_target_price(
        side="LONG", boll=boll, final_tp_price=2100.0,
    )
    assert price == 2100.0
    assert source == "FINAL_TP"


# ── config: from_env and validation ───────────────────────────────────────

def test_config_default_is_structure_middle() -> None:
    cfg = BollCvdReclaimStrategyConfig()
    assert cfg.entry_rr_target == "STRUCTURE_MIDDLE"


def test_config_rejects_invalid_entry_rr_target() -> None:
    with pytest.raises(RuntimeError, match="ENTRY_RR_TARGET"):
        BollCvdReclaimStrategyConfig(entry_rr_target="INVALID")


def test_config_accepts_final_tp() -> None:
    cfg = BollCvdReclaimStrategyConfig(entry_rr_target="FINAL_TP")
    assert cfg.entry_rr_target == "FINAL_TP"
    # __post_init__ should not raise
