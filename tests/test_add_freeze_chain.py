from __future__ import annotations

import unittest

from src.strategies import add_freeze_chain


class AddFreezeActiveTest(unittest.TestCase):
    def test_enabled_until_gt_ts_returns_true(self) -> None:
        self.assertTrue(add_freeze_chain.add_freeze_active(
            add_freeze_chain_enabled=True,
            add_freeze_until_ts_ms=2_000_000,
            ts_ms=1_000_000,
        ))

    def test_enabled_until_eq_ts_returns_false(self) -> None:
        self.assertFalse(add_freeze_chain.add_freeze_active(
            add_freeze_chain_enabled=True,
            add_freeze_until_ts_ms=1_000_000,
            ts_ms=1_000_000,
        ))

    def test_enabled_until_lt_ts_returns_false(self) -> None:
        self.assertFalse(add_freeze_chain.add_freeze_active(
            add_freeze_chain_enabled=True,
            add_freeze_until_ts_ms=500_000,
            ts_ms=1_000_000,
        ))

    def test_disabled_returns_false(self) -> None:
        self.assertFalse(add_freeze_chain.add_freeze_active(
            add_freeze_chain_enabled=False,
            add_freeze_until_ts_ms=2_000_000,
            ts_ms=1_000_000,
        ))

    def test_zero_until_returns_false(self) -> None:
        self.assertFalse(add_freeze_chain.add_freeze_active(
            add_freeze_chain_enabled=True,
            add_freeze_until_ts_ms=0,
            ts_ms=1_000_000,
        ))


class AddFreezeRemainingSecondsTest(unittest.TestCase):
    def test_remaining_30_seconds(self) -> None:
        self.assertAlmostEqual(
            add_freeze_chain.add_freeze_remaining_seconds(
                add_freeze_until_ts_ms=1_030_000,
                ts_ms=1_000_000,
            ),
            30.0,
        )

    def test_remaining_zero_when_expired(self) -> None:
        self.assertAlmostEqual(
            add_freeze_chain.add_freeze_remaining_seconds(
                add_freeze_until_ts_ms=1_000_000,
                ts_ms=1_000_000,
            ),
            0.0,
        )

    def test_remaining_zero_when_past(self) -> None:
        self.assertAlmostEqual(
            add_freeze_chain.add_freeze_remaining_seconds(
                add_freeze_until_ts_ms=500_000,
                ts_ms=1_000_000,
            ),
            0.0,
        )

    def test_remaining_zero_when_zero_until(self) -> None:
        self.assertAlmostEqual(
            add_freeze_chain.add_freeze_remaining_seconds(
                add_freeze_until_ts_ms=0,
                ts_ms=1_000_000,
            ),
            0.0,
        )


class ShouldResetAddFreezeIfExpiredTest(unittest.TestCase):
    def test_until_le_ts_returns_true(self) -> None:
        self.assertTrue(add_freeze_chain.should_reset_add_freeze_if_expired(
            add_freeze_until_ts_ms=1_000_000,
            ts_ms=1_000_000,
        ))

    def test_until_lt_ts_returns_true(self) -> None:
        self.assertTrue(add_freeze_chain.should_reset_add_freeze_if_expired(
            add_freeze_until_ts_ms=500_000,
            ts_ms=1_000_000,
        ))

    def test_until_gt_ts_returns_false(self) -> None:
        self.assertFalse(add_freeze_chain.should_reset_add_freeze_if_expired(
            add_freeze_until_ts_ms=2_000_000,
            ts_ms=1_000_000,
        ))

    def test_zero_until_returns_true(self) -> None:
        self.assertTrue(add_freeze_chain.should_reset_add_freeze_if_expired(
            add_freeze_until_ts_ms=0,
            ts_ms=1_000_000,
        ))


class ActiveAddFreezeBypassMultiplierTest(unittest.TestCase):
    def test_layers_1_penalty_0_returns_first_add_block_multiplier(self) -> None:
        self.assertAlmostEqual(
            add_freeze_chain.active_add_freeze_bypass_multiplier(
                layers=1,
                penalty_count=0,
                first_add_block_bypass_multiplier=5.0,
                add_min_interval_bypass_multiplier=2.0,
            ),
            5.0,
        )

    def test_layers_1_penalty_1_returns_add_min_interval_plus_1(self) -> None:
        self.assertAlmostEqual(
            add_freeze_chain.active_add_freeze_bypass_multiplier(
                layers=1,
                penalty_count=1,
                first_add_block_bypass_multiplier=5.0,
                add_min_interval_bypass_multiplier=2.0,
            ),
            3.0,
        )

    def test_layers_2_penalty_0_returns_add_min_interval(self) -> None:
        self.assertAlmostEqual(
            add_freeze_chain.active_add_freeze_bypass_multiplier(
                layers=2,
                penalty_count=0,
                first_add_block_bypass_multiplier=5.0,
                add_min_interval_bypass_multiplier=2.0,
            ),
            2.0,
        )

    def test_layers_2_penalty_2_returns_add_min_interval_plus_2(self) -> None:
        self.assertAlmostEqual(
            add_freeze_chain.active_add_freeze_bypass_multiplier(
                layers=2,
                penalty_count=2,
                first_add_block_bypass_multiplier=5.0,
                add_min_interval_bypass_multiplier=2.0,
            ),
            4.0,
        )


class FirstEntryElapsedSecondsTest(unittest.TestCase):
    def test_uses_first_entry_when_positive(self) -> None:
        self.assertAlmostEqual(
            add_freeze_chain.first_entry_elapsed_seconds(
                ts_ms=1_030_000,
                first_entry_ts_ms=1_000_000,
                last_order_ts_ms=500_000,
            ),
            30.0,
        )

    def test_falls_back_to_last_order_when_first_entry_is_zero(self) -> None:
        self.assertAlmostEqual(
            add_freeze_chain.first_entry_elapsed_seconds(
                ts_ms=1_030_000,
                first_entry_ts_ms=0,
                last_order_ts_ms=1_000_000,
            ),
            30.0,
        )

    def test_falls_back_to_last_order_when_first_entry_is_negative(self) -> None:
        self.assertAlmostEqual(
            add_freeze_chain.first_entry_elapsed_seconds(
                ts_ms=1_030_000,
                first_entry_ts_ms=-1,
                last_order_ts_ms=1_000_000,
            ),
            30.0,
        )

    def test_returns_zero_when_both_are_after_ts(self) -> None:
        self.assertAlmostEqual(
            add_freeze_chain.first_entry_elapsed_seconds(
                ts_ms=1_000_000,
                first_entry_ts_ms=2_000_000,
                last_order_ts_ms=2_000_000,
            ),
            0.0,
        )


class FirstAddBlockRequiredGapPctTest(unittest.TestCase):
    def test_target_0_003_multiplier_5_returns_0_015(self) -> None:
        self.assertAlmostEqual(
            add_freeze_chain.first_add_block_required_gap_pct(
                target_layer_gap_pct=0.003,
                first_add_block_bypass_multiplier=5.0,
            ),
            0.015,
        )


class CheckShockAddTimingDisabledFreezeTest(unittest.TestCase):
    def _defaults(self, **overrides):
        values = dict(
            side="LONG",
            price=99.0,
            ts_ms=1_500_000,
            target_layer=2,
            layers=1,
            last_entry_price=100.0,
            last_order_ts_ms=1_000_000,
            first_entry_ts_ms=1_000_000,
            add_freeze_chain_enabled=False,
            add_freeze_until_ts_ms=0,
            add_freeze_penalty_count=0,
            first_add_block_seconds=1800,
            add_min_interval_seconds=600,
            add_min_interval_bypass_multiplier=2.0,
            first_add_block_bypass_multiplier=5.0,
            target_layer_gap_pct=0.003,
        )
        values.update(overrides)
        return values

    def test_first_add_block_adverse_insufficient_returns_false(self) -> None:
        decision = add_freeze_chain.check_shock_add_timing(**self._defaults(
            price=99.0,
            ts_ms=1_500_000,  # 500s < 1800s
        ))
        self.assertFalse(decision.ok)
        self.assertEqual(decision.reason, "first_add_block")
        self.assertAlmostEqual(decision.adverse_gap_pct, 0.01)
        self.assertAlmostEqual(decision.required_gap_pct, 0.015)  # 0.003 * 5
        self.assertAlmostEqual(decision.multiplier, 5.0)
        self.assertAlmostEqual(decision.first_elapsed_seconds, 500.0)

    def test_first_add_block_adverse_sufficient_returns_true(self) -> None:
        # adverse = (100 - 98.4) / 100 = 0.016 >= 0.015
        decision = add_freeze_chain.check_shock_add_timing(**self._defaults(
            price=98.4,
            ts_ms=1_500_000,
        ))
        self.assertTrue(decision.ok)
        self.assertEqual(decision.reason, "first_add_block_bypassed")
        self.assertAlmostEqual(decision.adverse_gap_pct, 0.016)
        self.assertAlmostEqual(decision.required_gap_pct, 0.015)
        self.assertAlmostEqual(decision.first_elapsed_seconds, 500.0)

    def test_layers_1_after_block_time_returns_ok(self) -> None:
        decision = add_freeze_chain.check_shock_add_timing(**self._defaults(
            ts_ms=3_000_000,  # 2000s > 1800s
        ))
        self.assertTrue(decision.ok)
        self.assertEqual(decision.reason, "ok")

    def test_missing_last_entry_returns_false(self) -> None:
        for bad_price in (None, 0.0, -1.0):
            with self.subTest(last_entry_price=bad_price):
                decision = add_freeze_chain.check_shock_add_timing(**self._defaults(
                    last_entry_price=bad_price,
                ))
                self.assertFalse(decision.ok)
                self.assertEqual(decision.reason, "missing_last_entry")


class CheckShockAddTimingActiveFreezeTest(unittest.TestCase):
    def _defaults(self, **overrides):
        values = dict(
            side="LONG",
            price=99.0,
            ts_ms=1_500_000,
            target_layer=2,
            layers=1,
            last_entry_price=100.0,
            last_order_ts_ms=1_000_000,
            first_entry_ts_ms=1_000_000,
            add_freeze_chain_enabled=True,
            add_freeze_until_ts_ms=2_000_000,
            add_freeze_penalty_count=0,
            first_add_block_seconds=1800,
            add_min_interval_seconds=600,
            add_min_interval_bypass_multiplier=2.0,
            first_add_block_bypass_multiplier=5.0,
            target_layer_gap_pct=0.003,
        )
        values.update(overrides)
        return values

    def test_adverse_insufficient_returns_add_freeze(self) -> None:
        # adverse = (100 - 99.2) / 100 = 0.008
        # required = 0.003 * 5 = 0.015 (layers=1, penalty=0, first_add_block_bypass_multiplier=5)
        decision = add_freeze_chain.check_shock_add_timing(**self._defaults(
            price=99.2,
            layers=1,
            add_freeze_penalty_count=0,
        ))
        self.assertFalse(decision.ok)
        self.assertEqual(decision.reason, "add_freeze")
        self.assertAlmostEqual(decision.adverse_gap_pct, 0.008)
        self.assertAlmostEqual(decision.required_gap_pct, 0.015)
        self.assertAlmostEqual(decision.multiplier, 5.0)
        self.assertGreater(decision.freeze_remaining_seconds, 0)

    def test_layers_1_penalty_0_adverse_sufficient_returns_first_add_block_bypassed(self) -> None:
        # adverse = (100 - 98.4) / 100 = 0.016 >= 0.015
        decision = add_freeze_chain.check_shock_add_timing(**self._defaults(
            price=98.4,
            layers=1,
            add_freeze_penalty_count=0,
        ))
        self.assertTrue(decision.ok)
        self.assertEqual(decision.reason, "first_add_block_bypassed")
        self.assertAlmostEqual(decision.adverse_gap_pct, 0.016)
        self.assertAlmostEqual(decision.required_gap_pct, 0.015)
        self.assertGreater(decision.freeze_remaining_seconds, 0)

    def test_layers_2_adverse_sufficient_returns_add_freeze_bypassed(self) -> None:
        # layers=2, penalty=1 => multiplier = add_min_interval(2.0) + 1 = 3.0
        # required = 0.003 * 3 = 0.009
        # adverse = (100 - 99.0) / 100 = 0.01 >= 0.009
        decision = add_freeze_chain.check_shock_add_timing(**self._defaults(
            price=99.0,
            layers=2,
            add_freeze_penalty_count=1,
        ))
        self.assertTrue(decision.ok)
        self.assertEqual(decision.reason, "add_freeze_bypassed")
        self.assertAlmostEqual(decision.adverse_gap_pct, 0.01)
        self.assertAlmostEqual(decision.required_gap_pct, 0.009)
        self.assertAlmostEqual(decision.multiplier, 3.0)


class CheckShockAddTimingInactiveFreezeTest(unittest.TestCase):
    def _defaults(self, **overrides):
        values = dict(
            side="LONG",
            price=99.0,
            ts_ms=1_500_000,
            target_layer=2,
            layers=2,
            last_entry_price=100.0,
            last_order_ts_ms=1_000_000,
            first_entry_ts_ms=1_000_000,
            add_freeze_chain_enabled=True,
            add_freeze_until_ts_ms=0,
            add_freeze_penalty_count=0,
            first_add_block_seconds=1800,
            add_min_interval_seconds=600,
            add_min_interval_bypass_multiplier=2.0,
            first_add_block_bypass_multiplier=5.0,
            target_layer_gap_pct=0.003,
        )
        values.update(overrides)
        return values

    def test_layers_1_returns_ok(self) -> None:
        decision = add_freeze_chain.check_shock_add_timing(**self._defaults(
            layers=1,
        ))
        self.assertTrue(decision.ok)
        self.assertEqual(decision.reason, "ok")

    def test_layers_2_interval_not_met_and_adverse_insufficient_returns_add_interval(self) -> None:
        # elapsed = 500s < 600s, adverse = 0.01, bypass = 0.003 * 2 = 0.006
        # adverse (0.01) < bypass (0.006)? No, 0.01 >= 0.006. So it WOULD pass the bypass.
        # Let me use a price that gives a smaller adverse.
        # price=99.5 => adverse = (100-99.5)/100 = 0.005, which is < 0.006
        decision = add_freeze_chain.check_shock_add_timing(**self._defaults(
            price=99.5,
            layers=2,
            last_order_ts_ms=1_000_000,
            ts_ms=1_500_000,
        ))
        self.assertFalse(decision.ok)
        self.assertEqual(decision.reason, "add_interval")
        self.assertAlmostEqual(decision.adverse_gap_pct, 0.005)
        self.assertAlmostEqual(decision.required_gap_pct, 0.006)  # 0.003 * 2

    def test_layers_2_interval_bypassed_on_adverse_sufficient(self) -> None:
        # adverse = (100-99.39)/100 = 0.0061 >= 0.006
        decision = add_freeze_chain.check_shock_add_timing(**self._defaults(
            price=99.39,
            layers=2,
            last_order_ts_ms=1_000_000,
            ts_ms=1_500_000,
        ))
        self.assertTrue(decision.ok)
        self.assertEqual(decision.reason, "ok")

    def test_layers_2_interval_passed_on_elapsed_sufficient(self) -> None:
        decision = add_freeze_chain.check_shock_add_timing(**self._defaults(
            price=99.5,
            layers=2,
            last_order_ts_ms=1_000_000,
            ts_ms=2_000_000,  # 1000s > 600s
        ))
        self.assertTrue(decision.ok)
        self.assertEqual(decision.reason, "ok")


class StartAddFreezeAfterFirstEntryTest(unittest.TestCase):
    def test_enabled_returns_ts_plus_first_add_block(self) -> None:
        decision = add_freeze_chain.start_add_freeze_after_first_entry(
            ts_ms=1_000_000,
            add_freeze_chain_enabled=True,
            first_add_block_seconds=2700,
        )
        self.assertTrue(decision.enabled)
        self.assertEqual(decision.freeze_until_ts_ms, 1_000_000 + 2_700_000)
        self.assertEqual(decision.penalty_count, 0)

    def test_disabled_returns_not_enabled(self) -> None:
        decision = add_freeze_chain.start_add_freeze_after_first_entry(
            ts_ms=1_000_000,
            add_freeze_chain_enabled=False,
            first_add_block_seconds=2700,
        )
        self.assertFalse(decision.enabled)
        self.assertEqual(decision.freeze_until_ts_ms, 0)
        self.assertEqual(decision.penalty_count, 0)


class ExtendAddFreezeAfterSuccessfulAddTest(unittest.TestCase):
    def test_disabled_returns_unchanged(self) -> None:
        decision = add_freeze_chain.extend_add_freeze_after_successful_add(
            ts_ms=1_000_000,
            add_freeze_chain_enabled=False,
            add_min_interval_seconds=1800,
            add_freeze_until_ts_ms=2_000_000,
            add_freeze_penalty_count=1,
            was_active_freeze=True,
        )
        self.assertFalse(decision.changed)

    def test_zero_extension_ms_returns_unchanged(self) -> None:
        decision = add_freeze_chain.extend_add_freeze_after_successful_add(
            ts_ms=1_000_000,
            add_freeze_chain_enabled=True,
            add_min_interval_seconds=0,
            add_freeze_until_ts_ms=2_000_000,
            add_freeze_penalty_count=1,
            was_active_freeze=True,
        )
        self.assertFalse(decision.changed)

    def test_was_active_freeze_extends_from_max_of_old_until_and_ts(self) -> None:
        # old_until=2_000_000, ts=1_500_000, max=2_000_000
        # new_until = 2_000_000 + 1_800_000 = 3_800_000
        # penalty: 1 + 1 = 2
        decision = add_freeze_chain.extend_add_freeze_after_successful_add(
            ts_ms=1_500_000,
            add_freeze_chain_enabled=True,
            add_min_interval_seconds=1800,
            add_freeze_until_ts_ms=2_000_000,
            add_freeze_penalty_count=1,
            was_active_freeze=True,
        )
        self.assertTrue(decision.changed)
        self.assertEqual(decision.freeze_until_ts_ms, 3_800_000)
        self.assertEqual(decision.penalty_count, 2)
        self.assertEqual(decision.extension_seconds, 1800)

    def test_was_active_freeze_uses_ts_when_greater_than_old_until(self) -> None:
        # old_until=2_000_000, ts=3_000_000, max=3_000_000
        # new_until = 3_000_000 + 1_800_000 = 4_800_000
        decision = add_freeze_chain.extend_add_freeze_after_successful_add(
            ts_ms=3_000_000,
            add_freeze_chain_enabled=True,
            add_min_interval_seconds=1800,
            add_freeze_until_ts_ms=2_000_000,
            add_freeze_penalty_count=1,
            was_active_freeze=True,
        )
        self.assertTrue(decision.changed)
        self.assertEqual(decision.freeze_until_ts_ms, 4_800_000)
        self.assertEqual(decision.penalty_count, 2)

    def test_was_inactive_freeze_starts_from_ts_with_penalty_zero(self) -> None:
        decision = add_freeze_chain.extend_add_freeze_after_successful_add(
            ts_ms=3_000_000,
            add_freeze_chain_enabled=True,
            add_min_interval_seconds=1800,
            add_freeze_until_ts_ms=500_000,
            add_freeze_penalty_count=5,
            was_active_freeze=False,
        )
        self.assertTrue(decision.changed)
        self.assertEqual(decision.freeze_until_ts_ms, 4_800_000)  # 3_000_000 + 1_800_000
        self.assertEqual(decision.penalty_count, 0)


class AddFreezeSkipLogKeyTest(unittest.TestCase):
    def test_key_is_tuple_of_side_layers_target_layer_penalty_rounded_multiplier(self) -> None:
        key = add_freeze_chain.add_freeze_skip_log_key(
            side="LONG",
            layers=2,
            target_layer=3,
            penalty_count=1,
            multiplier=3.0,
        )
        self.assertEqual(key, ("LONG", 2, 3, 1, 3.0))

    def test_penalty_none_treated_as_zero(self) -> None:
        key = add_freeze_chain.add_freeze_skip_log_key(
            side="SHORT",
            layers=1,
            target_layer=2,
            penalty_count=0,
            multiplier=5.1,
        )
        self.assertEqual(key[3], 0)

    def test_multiplier_rounded_to_6_decimal_places(self) -> None:
        key = add_freeze_chain.add_freeze_skip_log_key(
            side="LONG",
            layers=1,
            target_layer=2,
            penalty_count=0,
            multiplier=5.123456789,
        )
        self.assertEqual(key[4], 5.123457)


class ShouldEmitAddFreezeSkipLogTest(unittest.TestCase):
    def test_same_key_within_interval_returns_false(self) -> None:
        key = ("LONG", 2, 3, 1, 3.0)
        self.assertFalse(add_freeze_chain.should_emit_add_freeze_skip_log(
            last_key=key,
            current_key=key,
            last_ts_ms=1_000_000,
            ts_ms=1_010_000,
            interval_ms=30_000,
        ))

    def test_same_key_after_interval_returns_true(self) -> None:
        key = ("LONG", 2, 3, 1, 3.0)
        self.assertTrue(add_freeze_chain.should_emit_add_freeze_skip_log(
            last_key=key,
            current_key=key,
            last_ts_ms=1_000_000,
            ts_ms=1_031_000,
            interval_ms=30_000,
        ))

    def test_different_key_within_interval_returns_true(self) -> None:
        self.assertTrue(add_freeze_chain.should_emit_add_freeze_skip_log(
            last_key=("LONG", 2, 3, 1, 3.0),
            current_key=("LONG", 2, 4, 1, 3.0),
            last_ts_ms=1_000_000,
            ts_ms=1_010_000,
            interval_ms=30_000,
        ))

    def test_none_last_key_returns_true(self) -> None:
        self.assertTrue(add_freeze_chain.should_emit_add_freeze_skip_log(
            last_key=None,
            current_key=("LONG", 2, 3, 1, 3.0),
            last_ts_ms=0,
            ts_ms=1_000_000,
            interval_ms=30_000,
        ))


class CheckShockAddTimingShortSideTest(unittest.TestCase):
    def test_short_active_freeze_returns_add_freeze(self) -> None:
        # adverse for SHORT: (101.0 - 100.0) / 100.0 = 0.01
        # required: 0.003 * 5 = 0.015, 0.01 < 0.015 => add_freeze
        decision = add_freeze_chain.check_shock_add_timing(
            side="SHORT",
            price=101.0,
            ts_ms=1_500_000,
            target_layer=2,
            layers=1,
            last_entry_price=100.0,
            last_order_ts_ms=1_000_000,
            first_entry_ts_ms=1_000_000,
            add_freeze_chain_enabled=True,
            add_freeze_until_ts_ms=2_000_000,
            add_freeze_penalty_count=0,
            first_add_block_seconds=1800,
            add_min_interval_seconds=600,
            add_min_interval_bypass_multiplier=2.0,
            first_add_block_bypass_multiplier=5.0,
            target_layer_gap_pct=0.003,
        )
        self.assertFalse(decision.ok)
        self.assertEqual(decision.reason, "add_freeze")


class LinearGapFreezeChainComboTest(unittest.TestCase):
    """Verify linear add gap combines correctly with freeze chain multipliers."""

    def test_linear_l4_gap_0_005_with_penalty_0_multiplier_2(self) -> None:
        """L4 gap=0.005, penalty=0 → multiplier=2.0, required=0.010"""
        target_layer_gap_pct = 0.005
        multiplier = add_freeze_chain.active_add_freeze_bypass_multiplier(
            layers=2,
            penalty_count=0,
            first_add_block_bypass_multiplier=5.0,
            add_min_interval_bypass_multiplier=2.0,
        )
        self.assertAlmostEqual(multiplier, 2.0)
        required = target_layer_gap_pct * multiplier
        self.assertAlmostEqual(required, 0.010)

    def test_linear_l4_gap_0_005_with_penalty_1_multiplier_3(self) -> None:
        """L4 gap=0.005, penalty=1 → multiplier=3.0, required=0.015"""
        target_layer_gap_pct = 0.005
        multiplier = add_freeze_chain.active_add_freeze_bypass_multiplier(
            layers=2,
            penalty_count=1,
            first_add_block_bypass_multiplier=5.0,
            add_min_interval_bypass_multiplier=2.0,
        )
        self.assertAlmostEqual(multiplier, 3.0)
        required = target_layer_gap_pct * multiplier
        self.assertAlmostEqual(required, 0.015)

    def test_linear_l4_gap_0_005_with_penalty_2_multiplier_4(self) -> None:
        """L4 gap=0.005, penalty=2 → multiplier=4.0, required=0.020"""
        target_layer_gap_pct = 0.005
        multiplier = add_freeze_chain.active_add_freeze_bypass_multiplier(
            layers=2,
            penalty_count=2,
            first_add_block_bypass_multiplier=5.0,
            add_min_interval_bypass_multiplier=2.0,
        )
        self.assertAlmostEqual(multiplier, 4.0)
        required = target_layer_gap_pct * multiplier
        self.assertAlmostEqual(required, 0.020)

    def test_linear_l8_gap_0_009_with_penalty_1_multiplier_3(self) -> None:
        """L8 gap=0.009, penalty=1 → multiplier=3.0, required=0.027"""
        target_layer_gap_pct = 0.009
        multiplier = add_freeze_chain.active_add_freeze_bypass_multiplier(
            layers=2,
            penalty_count=1,
            first_add_block_bypass_multiplier=5.0,
            add_min_interval_bypass_multiplier=2.0,
        )
        self.assertAlmostEqual(multiplier, 3.0)
        required = target_layer_gap_pct * multiplier
        self.assertAlmostEqual(required, 0.027)


if __name__ == "__main__":
    unittest.main()
