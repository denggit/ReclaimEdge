from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pytest

from src.live.runtime_paths import RuntimePaths
from src.reporting.live_state_store import (
    DEFAULT_STATE_PATH,
    LivePositionState,
    LiveStateStore,
    ROOT,
)
from src.strategies.boll_cvd_reclaim_strategy import StrategyPositionState


class LiveStateStoreTest(unittest.TestCase):
    def test_three_stage_post_tp1_fields_save_and_restore(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "live_state.json"
            store = LiveStateStore(path)
            strategy_state = StrategyPositionState(
                side="LONG",
                layers=1,
                total_entry_qty=1.0,
                total_entry_notional=100.0,
                avg_entry_price=100.0,
                position_cost_entry_notional=120.0,
                position_cost_exit_notional=20.0,
                position_cost_remaining_qty=0.8,
                net_remaining_breakeven_price=125.125,
                tp_plan="THREE_STAGE_RUNNER",
                three_stage_runner_enabled_for_position=True,
                three_stage_tp1_price=101.0,
                three_stage_tp2_price=110.0,
                three_stage_tp1_ratio=0.6,
                three_stage_tp2_ratio=0.2,
                three_stage_runner_ratio=0.2,
                three_stage_tp1_consumed=True,
                three_stage_post_tp1_protective_sl_price=101.0,
                three_stage_post_tp1_protective_sl_order_id="old-post",
                three_stage_post_tp1_sl_extension_triggered=True,
                three_stage_post_tp1_protected=True,
                three_stage_post_tp1_sl_time_tighten_candle_count=4,
                three_stage_post_tp1_sl_time_tighten_last_candle_ts_ms=4_000,
            )

            store.save(
                LiveStateStore.from_strategy_state(
                    position_id="pos-1",
                    symbol="ETH-USDT-SWAP",
                    strategy_state=strategy_state,
                    cash_before_position=100.0,
                )
            )

            loaded = store.load()

            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.position_id, "pos-1")
            self.assertEqual(loaded.three_stage_post_tp1_protective_sl_order_id, "old-post")
            self.assertEqual(loaded.three_stage_post_tp1_protective_sl_price, 101.0)
            self.assertTrue(loaded.three_stage_post_tp1_sl_extension_triggered)
            self.assertTrue(loaded.three_stage_post_tp1_protected)
            self.assertEqual(loaded.three_stage_post_tp1_sl_time_tighten_candle_count, 4)
            self.assertEqual(loaded.three_stage_post_tp1_sl_time_tighten_last_candle_ts_ms, 4_000)
            self.assertEqual(loaded.position_cost_entry_notional, 120.0)
            self.assertEqual(loaded.position_cost_exit_notional, 20.0)
            self.assertEqual(loaded.position_cost_remaining_qty, 0.8)
            self.assertEqual(loaded.net_remaining_breakeven_price, 125.125)

    def test_middle_runner_size_mismatch_fields_save_and_restore(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "live_state.json"
            store = LiveStateStore(path)
            strategy_state = StrategyPositionState(
                side="LONG",
                layers=1,
                total_entry_qty=1.0,
                total_entry_notional=100.0,
                avg_entry_price=100.0,
                tp_plan="MIDDLE_RUNNER",
                middle_runner_enabled_for_position=True,
                middle_runner_pending=True,
                middle_runner_add_disabled=True,
                middle_runner_size_mismatch_protected=True,
                middle_runner_size_mismatch_warning_ts_ms=123_456,
                middle_runner_sl_time_tighten_candle_count=3,
                middle_runner_sl_time_tighten_last_candle_ts_ms=3_000,
            )

            store.save(
                LiveStateStore.from_strategy_state(
                    position_id="pos-1",
                    symbol="ETH-USDT-SWAP",
                    strategy_state=strategy_state,
                    cash_before_position=100.0,
                )
            )

            loaded = store.load()

            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertTrue(loaded.middle_runner_size_mismatch_protected)
            self.assertEqual(loaded.middle_runner_size_mismatch_warning_ts_ms, 123_456)
            self.assertEqual(loaded.middle_runner_sl_time_tighten_candle_count, 3)
            self.assertEqual(loaded.middle_runner_sl_time_tighten_last_candle_ts_ms, 3_000)

    def test_time_tighten_state_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "live_state.json"
            store = LiveStateStore(path)
            strategy_state = StrategyPositionState(
                side="LONG",
                layers=1,
                middle_runner_sl_time_tighten_candle_count=7,
                middle_runner_sl_time_tighten_last_candle_ts_ms=70_000,
                three_stage_post_tp1_sl_time_tighten_candle_count=8,
                three_stage_post_tp1_sl_time_tighten_last_candle_ts_ms=80_000,
            )

            store.save(
                LiveStateStore.from_strategy_state(
                    position_id="pos-1",
                    symbol="ETH-USDT-SWAP",
                    strategy_state=strategy_state,
                    cash_before_position=100.0,
                )
            )

            loaded = store.load()

            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.middle_runner_sl_time_tighten_candle_count, 7)
            self.assertEqual(loaded.middle_runner_sl_time_tighten_last_candle_ts_ms, 70_000)
            self.assertEqual(loaded.three_stage_post_tp1_sl_time_tighten_candle_count, 8)
            self.assertEqual(loaded.three_stage_post_tp1_sl_time_tighten_last_candle_ts_ms, 80_000)

    def test_add_freeze_and_three_stage_degrade_fields_save_and_restore(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "live_state.json"
            store = LiveStateStore(path)
            strategy_state = StrategyPositionState(
                side="LONG",
                layers=2,
                total_entry_qty=1.0,
                total_entry_notional=100.0,
                avg_entry_price=100.0,
                add_freeze_until_ts_ms=3_700_000,
                add_freeze_penalty_count=2,
                three_stage_pre_tp1_degrade_stage="MIDDLE_RUNNER",
                three_stage_pre_tp1_degraded_ts_ms=10_800_001,
            )

            store.save(
                LiveStateStore.from_strategy_state(
                    position_id="pos-1",
                    symbol="ETH-USDT-SWAP",
                    strategy_state=strategy_state,
                    cash_before_position=100.0,
                )
            )

            loaded = store.load()

            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.add_freeze_until_ts_ms, 3_700_000)
            self.assertEqual(loaded.add_freeze_penalty_count, 2)
            self.assertEqual(loaded.three_stage_pre_tp1_degrade_stage, "MIDDLE_RUNNER")
            self.assertEqual(loaded.three_stage_pre_tp1_degraded_ts_ms, 10_800_001)


# ===========================================================================
# B02 – RuntimePaths integration (pytest‑style, uses fixtures)
# ===========================================================================


def test_default_state_store_path_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``DEFAULT_STATE_PATH`` still points to the legacy single‑coin path,
    and ``LiveStateStore()`` with no arguments uses that path."""
    import src.reporting.live_state_store as live_state_store_module

    default_path = tmp_path / "live_state.json"
    monkeypatch.setattr(live_state_store_module, "DEFAULT_STATE_PATH", default_path)
    monkeypatch.delenv("LIVE_STATE_PATH", raising=False)

    store = LiveStateStore()

    assert store.path == default_path


def test_env_live_state_path_still_overrides_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``LIVE_STATE_PATH`` env var overrides ``DEFAULT_STATE_PATH``."""
    custom = tmp_path / "custom_state.json"
    monkeypatch.setenv("LIVE_STATE_PATH", str(custom))
    store = LiveStateStore()
    assert store.path == custom


def test_explicit_path_still_wins_over_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit ``path=`` wins over ``LIVE_STATE_PATH`` env."""
    monkeypatch.setenv("LIVE_STATE_PATH", str(tmp_path / "env_state.json"))
    explicit = tmp_path / "explicit_state.json"
    store = LiveStateStore(path=explicit)
    assert store.path == explicit


def test_from_runtime_paths_uses_symbol_scoped_state_file(
    tmp_path: Path,
) -> None:
    """``from_runtime_paths`` delegates to ``RuntimePaths.state_file``."""
    runtime_paths = RuntimePaths(
        runtime_dir=tmp_path / "runtime", inst_id="ETH-USDT-SWAP",
    )
    store = LiveStateStore.from_runtime_paths(runtime_paths)
    expected = tmp_path / "runtime" / "state" / "live_state_ETH-USDT-SWAP.json"
    assert store.path == expected


def test_symbol_scoped_state_store_save_load_clear_roundtrip(
    tmp_path: Path,
) -> None:
    """Full save → load → clear cycle via ``from_runtime_paths``."""
    runtime_paths = RuntimePaths(
        runtime_dir=tmp_path / "runtime", inst_id="ETH-USDT-SWAP",
    )
    store = LiveStateStore.from_runtime_paths(runtime_paths)

    state = LivePositionState(
        position_id="ETH-USDT-SWAP:LONG:1:test",
        symbol="ETH-USDT-SWAP",
        side="LONG",
        layers=2,
        avg_entry_price=1680.0,
    )
    store.save(state)

    loaded = store.load()
    assert loaded is not None
    assert loaded.position_id == state.position_id
    assert loaded.symbol == "ETH-USDT-SWAP"
    assert loaded.layers == 2
    assert store.path.exists()

    store.clear()
    assert not store.path.exists()


def test_from_runtime_paths_accepts_btc_for_path_only(
    tmp_path: Path,
) -> None:
    """Path builder is generic — BTC symbol works for path generation only.

    This does NOT enable BTC live trading, create a TOML, or modify the
    validator.  It only proves that the path builder is symbol‑agnostic.
    """
    runtime_paths = RuntimePaths(
        runtime_dir=tmp_path / "runtime", inst_id="BTC-USDT-SWAP",
    )
    store = LiveStateStore.from_runtime_paths(runtime_paths)
    expected = tmp_path / "runtime" / "state" / "live_state_BTC-USDT-SWAP.json"
    assert store.path == expected


if __name__ == "__main__":
    unittest.main()
