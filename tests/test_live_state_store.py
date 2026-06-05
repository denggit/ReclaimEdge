from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.reporting.live_state_store import LiveStateStore
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


if __name__ == "__main__":
    unittest.main()
