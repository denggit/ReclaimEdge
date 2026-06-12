from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from src.execution.trader import PositionSnapshot
from src.position_management import tp_progress


class RecordingJournal:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict, str | None]] = []

    def append(self, event: str, payload: dict, position_id: str | None = None) -> None:
        self.events.append((event, dict(payload), position_id))


class FakeStrategy:
    def __init__(self, state: SimpleNamespace) -> None:
        self.state = state
        self.config = SimpleNamespace(
            breakeven_fee_buffer_pct=0.001,
            middle_bucket_split_fast_sl_fee_buffer_pct=0.001,
        )
        self.reset_calls: list[str] = []
        self.seed_calls: list[tuple[str, int]] = []

    def _reset_middle_runner_sl_time_tighten_state(self) -> None:
        self.reset_calls.append("middle_runner")

    def _reset_three_stage_post_tp1_sl_time_tighten_state(self) -> None:
        self.reset_calls.append("three_stage_post_tp1")

    def _seed_runner_sl_time_tighten_activation_candle(self, *, target: str, candle_ts_ms: int) -> None:
        self.seed_calls.append((target, candle_ts_ms))


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
        three_stage_runner_enabled_for_position=True,
        three_stage_tp1_consumed=False,
        three_stage_tp2_consumed=False,
        three_stage_tp1_ratio=0.5,
        three_stage_tp2_ratio=0.3,
        three_stage_runner_ratio=0.2,
        middle_runner_pending=False,
        middle_runner_active=False,
        middle_runner_add_disabled=False,
        middle_runner_keep_ratio=0.5,
        middle_runner_first_tp_price=105.0,
        middle_runner_final_tp_price=115.0,
        last_tp_update_candle_ts_ms=1234,
        sidecar_legs=[],
        position_cost_entry_notional=100.0,
        position_cost_exit_notional=0.0,
        position_cost_remaining_qty=1.0,
        net_remaining_breakeven_price=100.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def patch_cost_record(monkeypatch):
    calls: list[float | None] = []

    def fake_record(_state, _position, *, exit_price, fee_buffer_pct=0.001, expected_remaining_qty=None):
        calls.append(exit_price)

    monkeypatch.setattr(tp_progress.position_cost_runtime, "record_core_position_reduction_exit", fake_record)
    return calls


def test_middle_bucket_split_fast_only(monkeypatch) -> None:
    calls = patch_cost_record(monkeypatch)
    state = split_state()
    result = tp_progress.mark_middle_bucket_split_progress_if_position_reduced(FakeStrategy(state), position(0.70))

    assert result is not None
    assert result.event == "MIDDLE_BUCKET_FAST"
    assert result.pre_split_tp_plan == "THREE_STAGE_RUNNER"
    assert state.middle_bucket_split_fast_consumed is True
    assert state.middle_bucket_split_slow_consumed is False
    assert state.three_stage_tp1_consumed is False
    assert state.partial_tp_consumed is False
    assert calls == [110.0]


def test_middle_bucket_split_slow_only(monkeypatch) -> None:
    calls = patch_cost_record(monkeypatch)
    state = split_state()
    result = tp_progress.mark_middle_bucket_split_progress_if_position_reduced(FakeStrategy(state), position(0.80))

    assert result is not None
    assert result.event == "MIDDLE_BUCKET_SLOW_ONLY"
    assert result.pre_split_tp_plan == "THREE_STAGE_RUNNER"
    assert state.middle_bucket_split_fast_consumed is False
    assert state.middle_bucket_split_slow_consumed is True
    assert state.middle_bucket_split_add_disabled is True
    assert state.three_stage_tp1_consumed is False
    assert state.partial_tp_consumed is False
    assert calls == [105.0]


def test_middle_bucket_split_same_sync_full(monkeypatch) -> None:
    calls = patch_cost_record(monkeypatch)
    state = split_state()
    strategy = FakeStrategy(state)
    result = tp_progress.mark_middle_bucket_split_progress_if_position_reduced(strategy, position(0.50))

    assert result is not None
    assert result.event == "MIDDLE_BUCKET_FULL"
    assert result.pre_split_tp_plan == "THREE_STAGE_RUNNER"
    assert result.full_completed is True
    assert state.middle_bucket_split_fast_consumed is True
    assert state.middle_bucket_split_slow_consumed is True
    assert state.three_stage_tp1_consumed is True
    assert state.partial_tp_consumed is True
    assert calls == [108.0]


def test_middle_bucket_split_fast_then_slow_full(monkeypatch) -> None:
    calls = patch_cost_record(monkeypatch)
    state = split_state(middle_bucket_split_fast_consumed=True)
    result = tp_progress.mark_middle_bucket_split_progress_if_position_reduced(FakeStrategy(state), position(0.50))

    assert result is not None
    assert result.event == "MIDDLE_BUCKET_FULL"
    assert result.completed_leg == "slow"
    assert state.middle_bucket_split_slow_consumed is True
    assert state.three_stage_tp1_consumed is True
    assert calls == [105.0]


def test_middle_bucket_split_slow_then_fast_full(monkeypatch) -> None:
    calls = patch_cost_record(monkeypatch)
    state = split_state(middle_bucket_split_slow_consumed=True)
    result = tp_progress.mark_middle_bucket_split_progress_if_position_reduced(FakeStrategy(state), position(0.50))

    assert result is not None
    assert result.event == "MIDDLE_BUCKET_FULL"
    assert result.completed_leg == "fast"
    assert state.middle_bucket_split_fast_consumed is True
    assert state.three_stage_tp1_consumed is True
    assert calls == [110.0]


def test_middle_bucket_split_fast_then_unchanged_position_does_not_mark_slow(monkeypatch) -> None:
    calls = patch_cost_record(monkeypatch)
    state = split_state(middle_bucket_split_fast_consumed=True)
    result = tp_progress.mark_middle_bucket_split_progress_if_position_reduced(FakeStrategy(state), position(0.70))

    assert result is None
    assert state.middle_bucket_split_slow_consumed is False
    assert state.three_stage_tp1_consumed is False
    assert calls == []


def test_middle_bucket_split_slow_only_blocks_old_progress(monkeypatch) -> None:
    patch_cost_record(monkeypatch)
    state = split_state(middle_bucket_split_slow_consumed=True)
    strategy = FakeStrategy(state)

    three_stage_event = tp_progress.mark_three_stage_progress_if_position_reduced(strategy, position(0.50), 10_000)
    middle_runner_activated = tp_progress.mark_middle_runner_active_if_position_reduced(
        strategy,
        position(0.50),
    )

    assert three_stage_event is None
    assert middle_runner_activated is False
    assert state.three_stage_tp1_consumed is False
    assert state.middle_runner_active is False


def test_append_middle_bucket_split_journal_events_supports_new_and_legacy_events() -> None:
    journal = RecordingJournal()

    tp_progress.append_middle_bucket_split_journal_events(
        journal,
        {"event": "MIDDLE_BUCKET_SLOW_ONLY", "position_id": "pos-1"},
    )
    tp_progress.append_middle_bucket_split_journal_events(
        journal,
        {"event": "MIDDLE_BUCKET_FULL", "position_id": "pos-2"},
    )
    tp_progress.append_middle_bucket_split_journal_events(
        journal,
        {"event": "MIDDLE_BUCKET_SLOW", "position_id": "pos-3"},
    )

    assert [event for event, _payload, _position_id in journal.events] == [
        "MIDDLE_BUCKET_SLOW_ONLY_FILLED",
        "MIDDLE_BUCKET_FULL_FILLED",
        "MIDDLE_BUCKET_SPLIT_COMPLETED",
        "MIDDLE_BUCKET_SLOW_FILLED",
        "MIDDLE_BUCKET_SPLIT_COMPLETED",
    ]
