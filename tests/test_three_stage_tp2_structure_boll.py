"""Tests for Three-Stage TP2 using structure BOLL20 outer.

Verifies:
- Default: THREE_STAGE_TP2_USE_STRUCTURE_BOLL=true → TP2 = structure BOLL20 outer.
- SHORT uses structure BOLL20 lower.
- TP1 split still uses BOLL15/BOLL20 middle when split is enabled.
- Config off preserves old TP_BOLL15 behavior.
"""

from __future__ import annotations

import unittest

from src.monitors.boll_band_breakout_monitor import BollSnapshot
from src.risk.simple_position_sizer import SimplePositionSizer, SimplePositionSizerConfig
from src.strategies.boll_cvd_reclaim_strategy import (
    BollCvdReclaimStrategy,
    BollCvdReclaimStrategyConfig,
)


def _boll_with_tp(
    middle: float = 100.0,
    upper: float = 110.0,
    lower: float = 90.0,
    tp_middle: float | None = 101.0,
    tp_upper: float | None = 108.0,
    tp_lower: float | None = 92.0,
    tp_window: int | None = 15,
    candle_ts_ms: int = 1000,
) -> BollSnapshot:
    return BollSnapshot(
        inst_id="ETH-USDT-SWAP",
        candle_ts_ms=candle_ts_ms,
        close=middle,
        middle=middle,
        upper=upper,
        lower=lower,
        upper_distance_pct=0.01,
        lower_distance_pct=0.01,
        alert_switch_on=True,
        live_mode=True,
        tp_lower=tp_lower,
        tp_middle=tp_middle,
        tp_upper=tp_upper,
        tp_window=tp_window,
    )


def _boll_structure_only(
    middle: float = 100.0,
    upper: float = 110.0,
    lower: float = 90.0,
    candle_ts_ms: int = 1000,
) -> BollSnapshot:
    """BOLL snapshot without TP_BOLL15 fields (tp_* = None)."""
    return BollSnapshot(
        inst_id="ETH-USDT-SWAP",
        candle_ts_ms=candle_ts_ms,
        close=middle,
        middle=middle,
        upper=upper,
        lower=lower,
        upper_distance_pct=0.01,
        lower_distance_pct=0.01,
        alert_switch_on=True,
        live_mode=True,
    )


def _strategy(**kwargs) -> BollCvdReclaimStrategy:
    config = BollCvdReclaimStrategyConfig(**kwargs)
    sizer_config = SimplePositionSizerConfig()
    sizer = SimplePositionSizer(sizer_config)
    return BollCvdReclaimStrategy(config, sizer)


def _setup_state(
    strategy: BollCvdReclaimStrategy,
    side: str = "LONG",
    layers: int = 1,
    avg_entry_price: float = 100.0,
    breakeven_price: float = 100.2,
    net_remaining_breakeven_price: float = 100.2,
) -> None:
    s = strategy.state
    s.side = side
    s.layers = layers
    s.avg_entry_price = avg_entry_price
    s.breakeven_price = breakeven_price
    s.net_remaining_breakeven_price = net_remaining_breakeven_price


# ── Test 1: LONG default uses BOLL20 structure upper ────────────────────

class TestThreeStageTp2LongDefault(unittest.TestCase):
    """LONG Three-Stage TP2 defaults to structure BOLL20 upper."""

    def test_long_tp2_uses_structure_upper(self) -> None:
        boll = _boll_with_tp(
            middle=1650.0,
            upper=1700.0,
            lower=1600.0,
            tp_middle=1660.0,
            tp_upper=1670.0,
            tp_lower=1610.0,
            tp_window=15,
        )
        strategy = _strategy(
            tp_boll_enabled=True,
            three_stage_runner_enabled=True,
            three_stage_tp2_use_structure_boll=True,
        )
        _setup_state(strategy, "LONG", layers=1, avg_entry_price=1640.0)

        price, source = strategy._select_three_stage_tp2_outer("LONG", boll)

        self.assertAlmostEqual(price, 1700.0, places=4,
                               msg="TP2 must be structure BOLL20 upper (1700), not TP_BOLL15 upper (1670)")
        self.assertNotAlmostEqual(price, 1670.0, places=4,
                                  msg="TP2 must NOT be TP_BOLL15 upper")
        self.assertEqual(source, "STRUCTURE_BOLL_THREE_STAGE_TP2")

    def test_long_tp2_profit_fallback_when_structure_too_close(self) -> None:
        """When structure BOLL20 upper is too close to breakeven, fallback to farther outer."""
        boll = _boll_with_tp(
            middle=100.0,
            upper=100.15,  # too close — only 0.15% above mid
            lower=90.0,
            tp_middle=100.5,
            tp_upper=102.0,  # TP_BOLL15 upper is farther
            tp_lower=92.0,
            tp_window=15,
        )
        strategy = _strategy(
            tp_boll_enabled=True,
            three_stage_runner_enabled=True,
            three_stage_tp2_use_structure_boll=True,
            tp_min_net_profit_pct=0.002,
        )
        _setup_state(strategy, "LONG", layers=1, avg_entry_price=100.0,
                     breakeven_price=100.2, net_remaining_breakeven_price=100.2)

        price, source = strategy._select_three_stage_tp2_outer("LONG", boll)

        # required = 100.2 * 1.002 ≈ 100.4004
        # structure_upper = 100.15 < required → fallback
        # fallback_candidates = [100.15, 102.0] → max = 102.0
        self.assertAlmostEqual(price, 102.0, places=4)
        self.assertEqual(source, "THREE_STAGE_TP2_PROFIT_FALLBACK")

    def test_long_tp2_no_effective_be_uses_structure(self) -> None:
        """When effective breakeven ≤ 0, default to structure outer."""
        boll = _boll_with_tp(
            middle=100.0,
            upper=110.0,
            lower=90.0,
            tp_upper=108.0,
            tp_lower=92.0,
            tp_middle=101.0,
        )
        strategy = _strategy(
            tp_boll_enabled=True,
            three_stage_tp2_use_structure_boll=True,
        )
        # net_remaining_breakeven_price=0 → effective_be=0
        _setup_state(strategy, "LONG", layers=1, avg_entry_price=0.0,
                     breakeven_price=0.0, net_remaining_breakeven_price=0.0)

        price, source = strategy._select_three_stage_tp2_outer("LONG", boll)

        self.assertAlmostEqual(price, 110.0, places=4)
        self.assertEqual(source, "STRUCTURE_BOLL_THREE_STAGE_TP2")


# ── Test 2: SHORT default uses BOLL20 structure lower ──────────────────

class TestThreeStageTp2ShortDefault(unittest.TestCase):
    """SHORT Three-Stage TP2 defaults to structure BOLL20 lower."""

    def test_short_tp2_uses_structure_lower(self) -> None:
        boll = _boll_with_tp(
            middle=1650.0,
            upper=1700.0,
            lower=1600.0,
            tp_middle=1640.0,
            tp_upper=1670.0,
            tp_lower=1620.0,  # TP_BOLL15 lower is higher (closer to mid)
            tp_window=15,
        )
        strategy = _strategy(
            tp_boll_enabled=True,
            three_stage_runner_enabled=True,
            three_stage_tp2_use_structure_boll=True,
        )
        _setup_state(strategy, "SHORT", layers=1, avg_entry_price=1700.0,
                     breakeven_price=1700.0, net_remaining_breakeven_price=1700.0)

        price, source = strategy._select_three_stage_tp2_outer("SHORT", boll)

        self.assertAlmostEqual(price, 1600.0, places=4,
                               msg="TP2 must be structure BOLL20 lower (1600), not TP_BOLL15 lower (1620)")
        self.assertNotAlmostEqual(price, 1620.0, places=4,
                                  msg="TP2 must NOT be TP_BOLL15 lower")
        self.assertEqual(source, "STRUCTURE_BOLL_THREE_STAGE_TP2")

    def test_short_tp2_profit_fallback_when_structure_too_close(self) -> None:
        """When structure BOLL20 lower is too close to breakeven, fallback to farther outer."""
        boll = _boll_with_tp(
            middle=100.0,
            upper=110.0,
            lower=99.85,  # too close — only 0.15% below mid
            tp_middle=99.5,
            tp_upper=108.0,
            tp_lower=98.0,  # TP_BOLL15 lower is farther (lower = better for SHORT)
            tp_window=15,
        )
        strategy = _strategy(
            tp_boll_enabled=True,
            three_stage_runner_enabled=True,
            three_stage_tp2_use_structure_boll=True,
            tp_min_net_profit_pct=0.002,
        )
        _setup_state(strategy, "SHORT", layers=1, avg_entry_price=100.0,
                     breakeven_price=99.8, net_remaining_breakeven_price=99.8)

        price, source = strategy._select_three_stage_tp2_outer("SHORT", boll)

        # required = 99.8 * (1 - 0.002) ≈ 99.6004
        # structure_lower = 99.85 > required → fallback
        # fallback_candidates = [99.85, 98.0] → min = 98.0 (farther for SHORT)
        self.assertAlmostEqual(price, 98.0, places=4)
        self.assertEqual(source, "THREE_STAGE_TP2_PROFIT_FALLBACK")


# ── Test 3: TP1 split still uses BOLL15/BOLL20 middle ──────────────────

class TestTp1SplitPreserved(unittest.TestCase):
    """TP1 middle bucket split is unaffected by TP2 structure BOLL changes."""

    def test_tp1_split_uses_boll15_and_boll20_middle(self) -> None:
        """TP1 middle bucket split uses TP_BOLL15 middle (fast) and
        structure BOLL20 middle (slow). TP2 uses structure BOLL20 outer."""
        boll = _boll_with_tp(
            middle=1650.0,
            upper=1700.0,
            lower=1600.0,
            tp_middle=1660.0,  # TP_BOLL15 middle
            tp_upper=1670.0,
            tp_lower=1610.0,
            tp_window=15,
        )
        strategy = _strategy(
            tp_boll_enabled=True,
            tp_boll_window=15,
            middle_bucket_split_enabled=True,
            three_stage_runner_enabled=True,
            three_stage_tp2_use_structure_boll=True,
        )
        _setup_state(strategy, "LONG", layers=1, avg_entry_price=1640.0)

        # TP1 middle uses TP_BOLL15 middle (fast) by default
        tp_mid, tp_mid_src = strategy._select_valid_tp_middle_with_profit_fallback("LONG", boll)
        # TP2 outer uses structure BOLL20
        tp2_outer, tp2_src = strategy._select_three_stage_tp2_outer("LONG", boll)

        # TP1 middle should be TP_BOLL15 middle (1660)
        self.assertIsNotNone(tp_mid)
        self.assertAlmostEqual(tp_mid, 1660.0, places=4,
                               msg="TP1 middle should prefer TP_BOLL15 middle")
        # TP2 outer should be structure BOLL20 upper (1700)
        self.assertAlmostEqual(tp2_outer, 1700.0, places=4,
                               msg="TP2 outer should be structure BOLL20 upper")
        self.assertEqual(tp2_src, "STRUCTURE_BOLL_THREE_STAGE_TP2")
        # TP2 should NOT equal TP_BOLL15 upper
        self.assertNotEqual(tp2_outer, 1670.0,
                            msg="TP2 must not be TP_BOLL15 upper even when split is active")


# ── Test 4: Config off preserves old behavior ──────────────────────────

class TestThreeStageTp2ConfigOff(unittest.TestCase):
    """When THREE_STAGE_TP2_USE_STRUCTURE_BOLL=false, preserve old
    _select_valid_tp_outer_with_profit_fallback behavior."""

    def test_config_off_uses_tp_boll15_outer(self) -> None:
        boll = _boll_with_tp(
            middle=100.0,
            upper=110.0,
            lower=90.0,
            tp_middle=101.0,
            tp_upper=108.0,
            tp_lower=92.0,
            tp_window=15,
        )
        strategy = _strategy(
            tp_boll_enabled=True,
            three_stage_runner_enabled=True,
            three_stage_tp2_use_structure_boll=False,
            tp_min_net_profit_pct=0.002,
        )
        _setup_state(strategy, "LONG", layers=1, avg_entry_price=100.0)

        price, source = strategy._select_three_stage_tp2_outer("LONG", boll)

        # With old behavior (TP_BOLL15 first), TP2 may be TP_BOLL15 upper (108)
        # as long as it satisfies profit. Expected: TP_BOLL15 outer.
        self.assertAlmostEqual(price, 108.0, places=4,
                               msg="When config is off, TP2 should use TP_BOLL15 outer (old behavior)")
        # Source should NOT be STRUCTURE_BOLL_THREE_STAGE_TP2 when config is off
        self.assertNotEqual(source, "STRUCTURE_BOLL_THREE_STAGE_TP2")

    def test_config_off_delegates_to_fallback_method(self) -> None:
        """Explicitly verify delegation to the old method."""
        boll = _boll_with_tp(
            middle=100.0,
            upper=110.0,
            lower=90.0,
            tp_upper=108.0,
            tp_lower=92.0,
            tp_middle=101.0,
        )
        strategy = _strategy(
            tp_boll_enabled=True,
            three_stage_runner_enabled=True,
            three_stage_tp2_use_structure_boll=False,
        )
        _setup_state(strategy, "LONG", layers=1, avg_entry_price=100.0)

        # When config off, _select_three_stage_tp2_outer delegates to
        # _select_valid_tp_outer_with_profit_fallback. Both should return same result.
        price_new, src_new = strategy._select_three_stage_tp2_outer("LONG", boll)
        price_old, src_old = strategy._select_valid_tp_outer_with_profit_fallback("LONG", boll)

        self.assertAlmostEqual(price_new, price_old, places=4)
        self.assertEqual(src_new, src_old)


# ── Test 5: No TP_BOLL available ──────────────────────────────────────

class TestThreeStageTp2NoTpBoll(unittest.TestCase):
    """When TP_BOLL15 is not available, fallback uses structure BOLL20 outer only."""

    def test_long_no_tp_boll_uses_structure_upper(self) -> None:
        boll = _boll_structure_only(
            middle=100.0,
            upper=110.0,
            lower=90.0,
        )
        strategy = _strategy(
            tp_boll_enabled=False,
            three_stage_runner_enabled=True,
            three_stage_tp2_use_structure_boll=True,
        )
        _setup_state(strategy, "LONG", layers=1, avg_entry_price=100.0)

        price, source = strategy._select_three_stage_tp2_outer("LONG", boll)

        self.assertAlmostEqual(price, 110.0, places=4)
        self.assertEqual(source, "STRUCTURE_BOLL_THREE_STAGE_TP2")

    def test_short_no_tp_boll_uses_structure_lower(self) -> None:
        boll = _boll_structure_only(
            middle=100.0,
            upper=110.0,
            lower=90.0,
        )
        strategy = _strategy(
            tp_boll_enabled=False,
            three_stage_runner_enabled=True,
            three_stage_tp2_use_structure_boll=True,
        )
        _setup_state(strategy, "SHORT", layers=1, avg_entry_price=100.0)

        price, source = strategy._select_three_stage_tp2_outer("SHORT", boll)

        self.assertAlmostEqual(price, 90.0, places=4)
        self.assertEqual(source, "STRUCTURE_BOLL_THREE_STAGE_TP2")


if __name__ == "__main__":
    unittest.main()
