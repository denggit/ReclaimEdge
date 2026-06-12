from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from src.execution.trader import PositionSnapshot
from src.live.account_sync import tp_progress_phase as phase_module
from src.live import runtime_types as live_runtime_types
from src.live.account_sync.tp_progress_phase import run_account_sync_tp_progress_phase
from src.position_management import cost_runtime as position_cost_runtime
from src.position_management import tp_progress


class FakeTrader:
    symbol = "ETH-USDT-SWAP"
    position_contracts = Decimal("0")


class FakeJournal:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict, str | None]] = []

    def append(self, event: str, payload: dict, position_id: str | None = None) -> None:
        self.events.append((event, dict(payload), position_id))


class FakeStateStore:
    def __init__(self) -> None:
        self.saved: list = []

    def save(self, state) -> None:
        self.saved.append(state)


class FakeStrategy:
    def __init__(self, state: SimpleNamespace) -> None:
        self.state = state
        self.config = SimpleNamespace(
            breakeven_fee_buffer_pct=0.001,
            middle_bucket_split_fast_sl_fee_buffer_pct=0.001,
            middle_bucket_split_fast_sl_invalid_action="MARKET_EXIT",
            middle_bucket_split_fast_sl_enabled=True,
            three_stage_post_tp1_protective_sl_enabled=True,
            middle_runner_protective_sl_enabled=True,
            middle_runner_disable_add_after_partial=True,
        )

    def _reset_middle_runner_sl_time_tighten_state(self) -> None:
        self.state.middle_runner_sl_time_tighten_candle_count = 0

    def _reset_three_stage_post_tp1_sl_time_tighten_state(self) -> None:
        self.state.three_stage_post_tp1_sl_time_tighten_candle_count = 0

    def _seed_runner_sl_time_tighten_activation_candle(self, *, target: str, candle_ts_ms: int) -> None:
        self.state.seeded_runner_sl_target = target
        self.state.seeded_runner_sl_candle_ts_ms = candle_ts_ms

    def _calculate_three_stage_post_tp1_protective_sl(self, side, current_price, post_tp1_boll) -> float:
        return 99.0

    def _apply_three_stage_post_tp1_extension_trigger(self, side, current_price, post_tp1_boll, base_sl) -> float:
        return base_sl

    def _tighten_optional_three_stage_post_tp1_sl(self, side, base_sl, extension_sl) -> float:
        return extension_sl

    def _calculate_middle_runner_protective_sl(self, side, current_price, runner_boll) -> float:
        return 98.0


def position(eth_qty: float) -> PositionSnapshot:
    return PositionSnapshot("LONG", Decimal(str(eth_qty)), 100.0, eth_qty, Decimal(str(eth_qty)))


def split_state(**overrides) -> SimpleNamespace:
    values = dict(
        side="LONG",
        layers=1,
        total_entry_qty=1.0,
        total_entry_notional=100.0,
        avg_entry_price=100.0,
        last_entry_price=100.0,
        tp_plan="THREE_STAGE_RUNNER",
        partial_tp_consumed=False,
        partial_tp_price=101.0,
        partial_tp_ratio=0.5,
        middle_bucket_split_active=True,
        middle_bucket_split_fast_consumed=False,
        middle_bucket_split_slow_consumed=False,
        middle_bucket_split_add_disabled=False,
        middle_bucket_split_middle_bucket_ratio=0.5,
        middle_bucket_split_fast_ratio_of_bucket=0.6,
        middle_bucket_split_slow_ratio_of_bucket=0.4,
        middle_bucket_split_fast_total_ratio=0.3,
        middle_bucket_split_slow_total_ratio=0.2,
        middle_bucket_split_fast_price=110.0,
        middle_bucket_split_slow_price=105.0,
        middle_bucket_split_effective_price=108.0,
        middle_bucket_split_fast_sl_price=None,
        middle_bucket_split_fast_sl_order_id="fast-sl-old",
        three_stage_runner_enabled_for_position=True,
        three_stage_tp1_consumed=False,
        three_stage_tp2_consumed=False,
        three_stage_tp1_price=105.0,
        three_stage_tp2_price=115.0,
        three_stage_tp1_ratio=0.5,
        three_stage_tp2_ratio=0.3,
        three_stage_runner_ratio=0.2,
        three_stage_post_tp1_protective_sl_order_id="post-old",
        trend_runner_active=False,
        trend_runner_adjust_count=0,
        trend_runner_trend_start_ts_ms=0,
        trend_runner_tp_price=None,
        trend_runner_sl_price=None,
        middle_runner_pending=False,
        middle_runner_active=False,
        middle_runner_add_disabled=False,
        middle_runner_keep_ratio=0.5,
        middle_runner_first_close_ratio=0.5,
        middle_runner_first_tp_price=105.0,
        middle_runner_final_tp_price=115.0,
        middle_runner_protective_sl_order_id="middle-old",
        last_tp_update_candle_ts_ms=1234,
        sidecar_legs=[],
        position_cost_entry_notional=100.0,
        position_cost_exit_notional=0.0,
        position_cost_remaining_qty=1.0,
        net_remaining_breakeven_price=100.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def run_phase(strategy: FakeStrategy, core_position: PositionSnapshot):
    account_snapshot = live_runtime_types.AccountSnapshot(
        position=core_position,
        cash=1000.0,
        equity=1000.0,
        updated_monotonic=0.0,
        updated_ts_ms=0,
        latest_market_price=None,
        latest_market_price_ts_ms=0,
    )
    execution_state = live_runtime_types.ExecutionState("pos-1", 1000.0)
    journal = FakeJournal()

    result = run_account_sync_tp_progress_phase(
        account_snapshot=account_snapshot,
        execution_state=execution_state,
        trader=FakeTrader(),
        strategy=strategy,
        journal=journal,
        state_store=FakeStateStore(),
        position=core_position,
        core_position=core_position,
        current_position_key=("LONG", core_position.contracts, core_position.eth_qty),
        pending_order_count=0,
        last_logged_position_key=None,
    )
    return result, journal


def patch_cost(monkeypatch) -> None:
    monkeypatch.setattr(tp_progress.position_cost_runtime, "record_core_position_reduction_exit", lambda *a, **k: None)
    monkeypatch.setattr(position_cost_runtime, "sync_strategy_cost_from_position", lambda *a, **k: None)


def patch_no_split_progress(monkeypatch, *, three_stage_event=None, middle_runner_activated=False) -> None:
    monkeypatch.setattr(
        phase_module.tp_progress_helpers,
        "mark_middle_bucket_split_progress_if_position_reduced",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        phase_module.tp_progress_helpers,
        "mark_middle_runner_active_if_position_reduced",
        lambda *_args, **_kwargs: middle_runner_activated,
    )
    monkeypatch.setattr(
        phase_module.tp_progress_helpers,
        "mark_three_stage_progress_if_position_reduced",
        lambda *_args, **_kwargs: three_stage_event,
    )
    monkeypatch.setattr(
        phase_module.tp_progress_helpers,
        "mark_partial_tp_consumed_if_position_reduced",
        lambda *_args, **_kwargs: None,
    )


def patch_runner_boll(monkeypatch) -> None:
    monkeypatch.setattr(
        phase_module.runner_live_helpers,
        "three_stage_post_tp1_boll",
        lambda *_args, **_kwargs: SimpleNamespace(middle=106.0),
    )
    monkeypatch.setattr(
        phase_module.runner_live_helpers,
        "three_stage_post_tp1_current_price",
        lambda *_args, **_kwargs: (106.0, "test_price"),
    )
    monkeypatch.setattr(
        phase_module.runner_live_helpers,
        "middle_runner_activation_boll",
        lambda *_args, **_kwargs: SimpleNamespace(middle=106.0),
    )


def test_three_stage_post_tp1_payload_keeps_old_sl_price_without_state_candidate_pollution(monkeypatch) -> None:
    patch_cost(monkeypatch)
    patch_no_split_progress(monkeypatch, three_stage_event="TP1")
    patch_runner_boll(monkeypatch)
    state = split_state(
        middle_bucket_split_active=False,
        three_stage_post_tp1_protective_sl_order_id="old-post-sl",
        three_stage_post_tp1_protective_sl_price=101.0,
        three_stage_post_tp1_protected=True,
    )
    strategy = FakeStrategy(state)

    result, _journal = run_phase(strategy, position(0.50))

    assert result.three_stage_post_tp1_sl_payload is not None
    assert result.three_stage_post_tp1_sl_payload["protective_sl_price"] == 99.0
    assert result.three_stage_post_tp1_sl_payload["old_sl_order_id"] == "old-post-sl"
    assert result.three_stage_post_tp1_sl_payload["old_sl_price"] == 101.0
    assert result.three_stage_post_tp1_sl_payload["old_protected"] is True
    assert state.three_stage_post_tp1_protective_sl_price == 101.0


def test_middle_runner_payload_keeps_old_sl_price_without_state_candidate_pollution(monkeypatch) -> None:
    patch_cost(monkeypatch)
    patch_no_split_progress(monkeypatch, middle_runner_activated=True)
    patch_runner_boll(monkeypatch)
    state = split_state(
        middle_bucket_split_active=False,
        middle_runner_protective_sl_order_id="old-middle-sl",
        middle_runner_protective_sl_price=97.0,
    )
    strategy = FakeStrategy(state)

    result, _journal = run_phase(strategy, position(0.50))

    assert result.middle_runner_sl_payload is not None
    assert result.middle_runner_sl_payload["protective_sl_price"] == 98.0
    assert result.middle_runner_sl_payload["old_sl_order_id"] == "old-middle-sl"
    assert result.middle_runner_sl_payload["old_sl_price"] == 97.0
    assert result.middle_runner_sl_payload["old_protected"] is True
    assert state.middle_runner_protective_sl_price == 97.0


def test_middle_bucket_slow_only_does_not_trigger_post_tp1_or_middle_runner_sl(monkeypatch) -> None:
    patch_cost(monkeypatch)
    strategy = FakeStrategy(split_state())

    result, journal = run_phase(strategy, position(0.80))

    assert result.middle_bucket_split_event_payload is not None
    assert result.middle_bucket_split_event_payload["event"] == "MIDDLE_BUCKET_SLOW_ONLY"
    assert result.three_stage_post_tp1_sl_payload is None
    assert result.middle_runner_sl_payload is None
    assert result.middle_runner_activation_payload is None
    assert [event for event, _payload, _position_id in journal.events] == ["MIDDLE_BUCKET_SLOW_ONLY_FILLED"]


def test_middle_bucket_full_three_stage_uses_pre_split_plan_for_post_tp1_sl(monkeypatch) -> None:
    patch_cost(monkeypatch)
    strategy = FakeStrategy(split_state())

    result, journal = run_phase(strategy, position(0.50))

    assert result.middle_bucket_split_event_payload is not None
    assert result.middle_bucket_split_event_payload["event"] == "MIDDLE_BUCKET_FULL"
    assert result.middle_bucket_split_event_payload["pre_split_tp_plan"] == "THREE_STAGE_RUNNER"
    assert result.three_stage_post_tp1_sl_payload is not None
    assert result.three_stage_post_tp1_sl_payload["reason"] == "middle_bucket_full_filled"
    assert result.middle_runner_sl_payload is None
    assert result.three_stage_event_payload is not None
    assert result.three_stage_event_payload["split_source"] == "middle_bucket_full_filled"
    assert [event for event, _payload, _position_id in journal.events] == [
        "MIDDLE_BUCKET_FULL_FILLED",
        "MIDDLE_BUCKET_SPLIT_COMPLETED",
    ]


def test_middle_bucket_full_after_fast_fill_uses_fast_sl_as_old_protection(monkeypatch) -> None:
    patch_cost(monkeypatch)
    patch_runner_boll(monkeypatch)
    state = split_state(
        middle_bucket_split_fast_consumed=True,
        middle_bucket_split_fast_sl_order_id="old-fast-sl",
        middle_bucket_split_fast_sl_price=102.0,
        middle_bucket_split_fast_sl_protected=True,
        three_stage_post_tp1_protective_sl_price=97.0,
    )
    strategy = FakeStrategy(state)

    result, _journal = run_phase(strategy, position(0.50))

    assert result.middle_bucket_split_event_payload is not None
    assert result.middle_bucket_split_event_payload["event"] == "MIDDLE_BUCKET_FULL"
    assert result.middle_bucket_split_event_payload["pre_split_tp_plan"] == "THREE_STAGE_RUNNER"
    assert result.three_stage_post_tp1_sl_payload is not None
    assert result.three_stage_post_tp1_sl_payload["reason"] == "middle_bucket_full_filled"
    assert result.three_stage_post_tp1_sl_payload["protective_sl_price"] == 99.0
    assert result.three_stage_post_tp1_sl_payload["old_sl_order_id"] == "old-fast-sl"
    assert result.three_stage_post_tp1_sl_payload["old_sl_price"] == 102.0
    assert result.three_stage_post_tp1_sl_payload["old_protected"] is True
    assert state.three_stage_post_tp1_protective_sl_price == 97.0


def test_middle_bucket_full_middle_runner_uses_pre_split_plan_after_state_tp_plan_becomes_single(monkeypatch) -> None:
    patch_cost(monkeypatch)
    state = split_state(
        tp_plan="MIDDLE_RUNNER",
        three_stage_runner_enabled_for_position=False,
        middle_runner_pending=True,
        middle_runner_active=False,
    )
    strategy = FakeStrategy(state)

    result, _journal = run_phase(strategy, position(0.50))

    assert state.tp_plan == "SINGLE"
    assert result.middle_bucket_split_event_payload is not None
    assert result.middle_bucket_split_event_payload["event"] == "MIDDLE_BUCKET_FULL"
    assert result.middle_bucket_split_event_payload["pre_split_tp_plan"] == "MIDDLE_RUNNER"
    assert result.middle_runner_sl_payload is not None
    assert result.middle_runner_sl_payload["reason"] == "middle_bucket_full_filled"
    assert result.middle_runner_activation_payload is not None
    assert result.middle_runner_activation_payload["reason"] == "middle_bucket_full_filled"
    assert result.three_stage_post_tp1_sl_payload is None
