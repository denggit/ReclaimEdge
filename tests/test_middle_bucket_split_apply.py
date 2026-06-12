from __future__ import annotations

from decimal import Decimal

from src.execution.order_specs import (
    MiddleBucketSplitOrderInput,
    build_take_profit_order_specs,
)
from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
)
from src.strategies.middle_bucket_split_apply import (
    apply_three_stage_middle_bucket_split,
)


def _strategy(*, tp_min_net_profit_pct: float = 0.0) -> BollCvdReclaimStrategy:
    config = BollCvdReclaimStrategyConfig(
        middle_bucket_split_enabled=True,
        middle_bucket_split_fast_ratio=0.70,
        tp_min_net_profit_pct=tp_min_net_profit_pct,
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


def _boll_unsplit_slow_middle() -> BollSnapshot:
    return BollSnapshot(
        inst_id="ETH-USDT-SWAP",
        candle_ts_ms=3000,
        close=103.0,
        middle=103.0,
        upper=110.0,
        lower=90.0,
        upper_distance_pct=0.01,
        lower_distance_pct=0.01,
        alert_switch_on=True,
        live_mode=True,
        tp_middle=101.0,
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


def _seed_existing_split_state(strategy: BollCvdReclaimStrategy) -> None:
    state = strategy.state
    state.middle_bucket_split_active = True
    state.middle_bucket_split_fast_price = 106.0
    state.middle_bucket_split_slow_price = 104.0
    state.middle_bucket_split_effective_price = 105.4
    state.middle_bucket_split_middle_bucket_ratio = 0.70
    state.middle_bucket_split_fast_ratio_of_bucket = 0.70
    state.middle_bucket_split_slow_ratio_of_bucket = 0.30
    state.middle_bucket_split_fast_total_ratio = 0.49
    state.middle_bucket_split_slow_total_ratio = 0.21
    state.middle_bucket_split_reason = "split_enabled"


def _split_input_from_state(strategy: BollCvdReclaimStrategy) -> MiddleBucketSplitOrderInput:
    state = strategy.state
    return MiddleBucketSplitOrderInput(
        active=state.middle_bucket_split_active,
        fast_price=state.middle_bucket_split_fast_price,
        slow_price=state.middle_bucket_split_slow_price,
        effective_price=state.middle_bucket_split_effective_price,
        middle_bucket_ratio=Decimal(str(state.middle_bucket_split_middle_bucket_ratio)),
        fast_ratio_of_bucket=Decimal(str(state.middle_bucket_split_fast_ratio_of_bucket)),
        slow_ratio_of_bucket=Decimal(str(state.middle_bucket_split_slow_ratio_of_bucket)),
        fast_total_ratio=Decimal(str(state.middle_bucket_split_fast_total_ratio)),
        slow_total_ratio=Decimal(str(state.middle_bucket_split_slow_total_ratio)),
        fast_consumed=state.middle_bucket_split_fast_consumed,
        slow_consumed=state.middle_bucket_split_slow_consumed,
    )


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


def test_fast_consumed_unsplit_slow_middle_preserves_partial_split() -> None:
    strategy = _strategy(tp_min_net_profit_pct=0.02)
    _setup_three_stage_state(strategy)
    _seed_existing_split_state(strategy)
    strategy.state.middle_bucket_split_fast_consumed = True

    result = apply_three_stage_middle_bucket_split(
        strategy=strategy,
        boll=_boll_unsplit_slow_middle(),
    )

    assert result.action in {"SPLIT", "PARTIAL_SPLIT_PRESERVED"}
    assert result.split_active is True
    assert strategy.state.middle_bucket_split_active is True
    assert strategy.state.middle_bucket_split_fast_consumed is True
    assert strategy.state.middle_bucket_split_slow_consumed is False

    decision = build_take_profit_order_specs(
        position_contracts=Decimal("100"),
        min_contracts=Decimal("1"),
        contract_precision=Decimal("1"),
        tp_plan="THREE_STAGE_RUNNER",
        final_tp_price=110.0,
        partial_tp_price=result.partial_tp_price,
        partial_tp_ratio=Decimal(str(result.partial_tp_ratio)),
        partial_tp_consumed=False,
        middle_runner_active=False,
        three_stage_tp1_price=strategy.state.three_stage_tp1_price,
        three_stage_tp2_price=110.0,
        three_stage_tp1_ratio=Decimal("0.70"),
        three_stage_tp2_ratio=Decimal("0.20"),
        three_stage_tp1_consumed=False,
        three_stage_tp2_consumed=False,
        three_stage_runner_ratio=Decimal("0.10"),
        middle_bucket_split=_split_input_from_state(strategy),
    )
    labels = [spec.label for spec in decision.specs]
    assert labels == ["tp1_middle_slow", "tp2_outer"]
    assert "tp1_middle_fast" not in labels


def test_slow_consumed_unsplit_slow_middle_preserves_partial_split() -> None:
    strategy = _strategy(tp_min_net_profit_pct=0.02)
    _setup_three_stage_state(strategy)
    _seed_existing_split_state(strategy)
    strategy.state.middle_bucket_split_slow_consumed = True

    result = apply_three_stage_middle_bucket_split(
        strategy=strategy,
        boll=_boll_unsplit_slow_middle(),
    )

    assert result.action in {"SPLIT", "PARTIAL_SPLIT_PRESERVED"}
    assert result.split_active is True
    assert strategy.state.middle_bucket_split_active is True
    assert strategy.state.middle_bucket_split_fast_consumed is False
    assert strategy.state.middle_bucket_split_slow_consumed is True


def test_unconsumed_unsplit_slow_middle_keeps_original_unsplit_behavior() -> None:
    strategy = _strategy(tp_min_net_profit_pct=0.02)
    _setup_three_stage_state(strategy)
    _seed_existing_split_state(strategy)

    result = apply_three_stage_middle_bucket_split(
        strategy=strategy,
        boll=_boll_unsplit_slow_middle(),
    )

    assert result.action == "UNSPLIT_SLOW_MIDDLE"
    assert result.split_active is False
    assert result.partial_tp_price == 103.0
    assert strategy.state.middle_bucket_split_active is False
    assert strategy.state.middle_bucket_split_fast_consumed is False
    assert strategy.state.middle_bucket_split_slow_consumed is False
    assert strategy.state.middle_bucket_split_fast_price is None
    assert strategy.state.middle_bucket_split_slow_price is None
    assert strategy.state.three_stage_tp1_price == 103.0
