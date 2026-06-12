from __future__ import annotations

from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
)
from src.strategies.middle_bucket_split_apply import (
    apply_three_stage_middle_bucket_split,
)


def _strategy() -> BollCvdReclaimStrategy:
    config = BollCvdReclaimStrategyConfig(
        middle_bucket_split_enabled=True,
        middle_bucket_split_fast_ratio=0.70,
        tp_min_net_profit_pct=0.0,
    )
    return BollCvdReclaimStrategy(config, SimplePositionSizer(SimplePositionSizerConfig()))


def _boll() -> BollSnapshot:
    return BollSnapshot(
        inst_id="ETH-USDT-SWAP",
        candle_ts_ms=2000,
        close=102.0,
        middle=102.0,
        upper=110.0,
        lower=90.0,
        upper_distance_pct=0.01,
        lower_distance_pct=0.01,
        alert_switch_on=True,
        live_mode=True,
        tp_middle=103.0,
        tp_upper=108.0,
        tp_lower=92.0,
        tp_window=15,
    )


def _setup_three_stage_state(strategy: BollCvdReclaimStrategy) -> None:
    state = strategy.state
    state.side = "LONG"
    state.layers = 1
    state.avg_entry_price = 100.0
    state.breakeven_price = 100.0
    state.net_remaining_breakeven_price = 100.0
    state.three_stage_tp1_ratio = 0.70
    state.three_stage_tp2_ratio = 0.20
    state.three_stage_runner_ratio = 0.10


def test_apply_three_stage_split_preserves_fast_consumed() -> None:
    strategy = _strategy()
    _setup_three_stage_state(strategy)
    strategy.state.middle_bucket_split_fast_consumed = True

    result = apply_three_stage_middle_bucket_split(strategy=strategy, boll=_boll())

    assert result.action == "SPLIT"
    assert strategy.state.middle_bucket_split_fast_consumed is True
    assert strategy.state.middle_bucket_split_slow_consumed is False


def test_apply_three_stage_split_preserves_slow_consumed() -> None:
    strategy = _strategy()
    _setup_three_stage_state(strategy)
    strategy.state.middle_bucket_split_slow_consumed = True

    result = apply_three_stage_middle_bucket_split(strategy=strategy, boll=_boll())

    assert result.action == "SPLIT"
    assert strategy.state.middle_bucket_split_fast_consumed is False
    assert strategy.state.middle_bucket_split_slow_consumed is True


def test_apply_three_stage_split_initial_progress_remains_unconsumed() -> None:
    strategy = _strategy()
    _setup_three_stage_state(strategy)

    result = apply_three_stage_middle_bucket_split(strategy=strategy, boll=_boll())

    assert result.action == "SPLIT"
    assert strategy.state.middle_bucket_split_fast_consumed is False
    assert strategy.state.middle_bucket_split_slow_consumed is False
