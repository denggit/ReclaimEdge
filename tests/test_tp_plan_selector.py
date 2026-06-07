"""Tests for the pure TP plan selector functions.

These tests verify the pure functions in src/strategies/tp_plan_selector.py
work correctly without any dependency on strategy class, state, logger, or env.
"""

from __future__ import annotations

from src.strategies.tp_plan_selector import (
    TpBandSnapshot,
    effective_breakeven_for_tp_selection,
    middle_runner_plan_allowed,
    select_tp_middle,
    select_tp_middle_with_profit_fallback,
    select_tp_outer,
    select_tp_plan,
    select_tp_price,
    three_stage_runner_plan_allowed,
    tp_boll_available,
    tp_plan_unchanged,
)


# ── helpers ────────────────────────────────────────────────────────────

def _tp_band(
        middle: float = 100.0,
        upper: float = 110.0,
        lower: float = 90.0,
        tp_middle: float | None = 101.0,
        tp_upper: float | None = 108.0,
        tp_lower: float | None = 92.0,
        tp_window: int | None = 15,
) -> TpBandSnapshot:
    return TpBandSnapshot(
        middle=middle,
        upper=upper,
        lower=lower,
        tp_middle=tp_middle,
        tp_upper=tp_upper,
        tp_lower=tp_lower,
        tp_window=tp_window,
    )


def _tp_band_no_tp() -> TpBandSnapshot:
    return TpBandSnapshot(
        middle=100.0,
        upper=110.0,
        lower=90.0,
        tp_middle=None,
        tp_upper=None,
        tp_lower=None,
        tp_window=None,
    )


# ═══════════════════════════════════════════════════════════════════════
# 1. tp_boll_available
# ═══════════════════════════════════════════════════════════════════════

class TestTpBollAvailable:
    def test_enabled_all_fields_present_returns_true(self):
        assert tp_boll_available(
            tp_boll_enabled=True,
            tp_middle=101.0,
            tp_upper=108.0,
            tp_lower=92.0,
        ) is True

    def test_disabled_returns_false(self):
        assert tp_boll_available(
            tp_boll_enabled=False,
            tp_middle=101.0,
            tp_upper=108.0,
            tp_lower=92.0,
        ) is False

    def test_tp_middle_none_returns_false(self):
        assert tp_boll_available(
            tp_boll_enabled=True,
            tp_middle=None,
            tp_upper=108.0,
            tp_lower=92.0,
        ) is False

    def test_tp_upper_none_returns_false(self):
        assert tp_boll_available(
            tp_boll_enabled=True,
            tp_middle=101.0,
            tp_upper=None,
            tp_lower=92.0,
        ) is False

    def test_tp_lower_none_returns_false(self):
        assert tp_boll_available(
            tp_boll_enabled=True,
            tp_middle=101.0,
            tp_upper=108.0,
            tp_lower=None,
        ) is False

    def test_all_fields_none_returns_false(self):
        assert tp_boll_available(
            tp_boll_enabled=True,
            tp_middle=None,
            tp_upper=None,
            tp_lower=None,
        ) is False


# ═══════════════════════════════════════════════════════════════════════
# 2. select_tp_middle
# ═══════════════════════════════════════════════════════════════════════

class TestSelectTpMiddle:
    def test_tp_boll_available_returns_tp_middle(self):
        tp_band = _tp_band(tp_middle=101.0)
        sel = select_tp_middle(tp_band=tp_band, tp_boll_enabled=True)
        assert sel.price == 101.0
        assert sel.source == "TP_BOLL"

    def test_tp_boll_disabled_returns_structure_middle(self):
        tp_band = _tp_band(middle=100.0, tp_middle=101.0)
        sel = select_tp_middle(tp_band=tp_band, tp_boll_enabled=False)
        assert sel.price == 100.0
        assert sel.source == "STRUCTURE_BOLL"

    def test_tp_boll_unavailable_returns_structure_middle(self):
        tp_band = _tp_band_no_tp()
        sel = select_tp_middle(tp_band=tp_band, tp_boll_enabled=True)
        assert sel.price == 100.0
        assert sel.source == "STRUCTURE_BOLL"


# ═══════════════════════════════════════════════════════════════════════
# 3. select_tp_outer
# ═══════════════════════════════════════════════════════════════════════

class TestSelectTpOuter:
    def test_long_tp_boll_available_returns_tp_upper(self):
        tp_band = _tp_band(tp_upper=108.0)
        sel = select_tp_outer(side="LONG", tp_band=tp_band, tp_boll_enabled=True)
        assert sel.price == 108.0
        assert sel.source == "TP_BOLL"

    def test_short_tp_boll_available_returns_tp_lower(self):
        tp_band = _tp_band(tp_lower=92.0)
        sel = select_tp_outer(side="SHORT", tp_band=tp_band, tp_boll_enabled=True)
        assert sel.price == 92.0
        assert sel.source == "TP_BOLL"

    def test_long_tp_boll_unavailable_returns_structure_upper(self):
        tp_band = _tp_band_no_tp()
        sel = select_tp_outer(side="LONG", tp_band=tp_band, tp_boll_enabled=True)
        assert sel.price == 110.0
        assert sel.source == "STRUCTURE_BOLL"

    def test_short_tp_boll_unavailable_returns_structure_lower(self):
        tp_band = _tp_band_no_tp()
        sel = select_tp_outer(side="SHORT", tp_band=tp_band, tp_boll_enabled=True)
        assert sel.price == 90.0
        assert sel.source == "STRUCTURE_BOLL"

    def test_long_tp_boll_disabled_returns_structure(self):
        tp_band = _tp_band(tp_upper=108.0, upper=110.0)
        sel = select_tp_outer(side="LONG", tp_band=tp_band, tp_boll_enabled=False)
        assert sel.price == 110.0
        assert sel.source == "STRUCTURE_BOLL"


# ═══════════════════════════════════════════════════════════════════════
# 4. select_tp_middle_with_profit_fallback (LONG)
# ═══════════════════════════════════════════════════════════════════════

class TestSelectTpMiddleWithProfitFallbackLong:
    def test_effective_be_zero_returns_none(self):
        tp_band = _tp_band(tp_middle=101.0, middle=100.0)
        sel = select_tp_middle_with_profit_fallback(
            side="LONG", effective_be=0.0, min_net_profit=0.002,
            tp_band=tp_band, tp_boll_enabled=True,
        )
        assert sel is None, "must return None when effective_be <= 0"

    def test_tp_boll_middle_meets_profit(self):
        tp_band = _tp_band(tp_middle=101.0, middle=100.0)
        # effective_be=100.0, required=100.2, tp_mid=101.0 >= 100.2 → OK
        sel = select_tp_middle_with_profit_fallback(
            side="LONG", effective_be=100.0, min_net_profit=0.002,
            tp_band=tp_band, tp_boll_enabled=True,
        )
        assert sel.price == 101.0
        assert sel.source == "TP_BOLL"

    def test_tp_boll_insufficient_structure_sufficient_returns_fallback(self):
        tp_band = _tp_band(tp_middle=100.3, middle=101.0)
        # effective_be=100.0, required=100.5, tp_mid=100.3 < 100.5 → fail
        # structure middle=101.0 >= 100.5 → fallback
        sel = select_tp_middle_with_profit_fallback(
            side="LONG", effective_be=100.0, min_net_profit=0.005,
            tp_band=tp_band, tp_boll_enabled=True,
        )
        assert sel.price == 101.0
        assert sel.source == "STRUCTURE_BOLL_PROFIT_FALLBACK"

    def test_neither_meets_profit_returns_none(self):
        tp_band = _tp_band(tp_middle=100.3, middle=100.4)
        # effective_be=100.0, required=101.0, neither >= 101.0 → None
        sel = select_tp_middle_with_profit_fallback(
            side="LONG", effective_be=100.0, min_net_profit=0.01,
            tp_band=tp_band, tp_boll_enabled=True,
        )
        assert sel is None, "must return None when neither middle meets profit"


# ═══════════════════════════════════════════════════════════════════════
# 5. select_tp_middle_with_profit_fallback (SHORT)
# ═══════════════════════════════════════════════════════════════════════

class TestSelectTpMiddleWithProfitFallbackShort:
    def test_effective_be_zero_returns_none(self):
        tp_band = _tp_band(tp_middle=99.0, middle=100.0)
        sel = select_tp_middle_with_profit_fallback(
            side="SHORT", effective_be=0.0, min_net_profit=0.002,
            tp_band=tp_band, tp_boll_enabled=True,
        )
        assert sel is None, "must return None when effective_be <= 0"

    def test_tp_boll_middle_meets_profit(self):
        tp_band = _tp_band(tp_middle=99.0, middle=100.0)
        # effective_be=100.0, required=99.8, tp_mid=99.0 <= 99.8 → OK
        sel = select_tp_middle_with_profit_fallback(
            side="SHORT", effective_be=100.0, min_net_profit=0.002,
            tp_band=tp_band, tp_boll_enabled=True,
        )
        assert sel.price == 99.0
        assert sel.source == "TP_BOLL"

    def test_tp_boll_insufficient_structure_sufficient_returns_fallback(self):
        tp_band = _tp_band(tp_middle=99.7, middle=99.0)
        # effective_be=100.0, required=99.5, tp_mid=99.7 > 99.5 → fail
        # structure middle=99.0 <= 99.5 → fallback
        sel = select_tp_middle_with_profit_fallback(
            side="SHORT", effective_be=100.0, min_net_profit=0.005,
            tp_band=tp_band, tp_boll_enabled=True,
        )
        assert sel.price == 99.0
        assert sel.source == "STRUCTURE_BOLL_PROFIT_FALLBACK"

    def test_neither_meets_profit_returns_none(self):
        tp_band = _tp_band(tp_middle=99.7, middle=99.6)
        # effective_be=100.0, required=99.0, neither <= 99.0 → None
        sel = select_tp_middle_with_profit_fallback(
            side="SHORT", effective_be=100.0, min_net_profit=0.01,
            tp_band=tp_band, tp_boll_enabled=True,
        )
        assert sel is None, "must return None when neither middle meets profit"


# ═══════════════════════════════════════════════════════════════════════
# 6. effective_breakeven_for_tp_selection
# ═══════════════════════════════════════════════════════════════════════

class TestEffectiveBreakevenForTpSelection:
    def test_net_remaining_be_positive_takes_priority(self):
        result = effective_breakeven_for_tp_selection(
            side="LONG",
            net_remaining_breakeven_price=99.0,
            avg_entry_price=100.0,
            breakeven_fee_buffer_pct=0.001,
        )
        assert result == 99.0

    def test_avg_entry_zero_or_negative_returns_zero(self):
        result = effective_breakeven_for_tp_selection(
            side="LONG",
            net_remaining_breakeven_price=0.0,
            avg_entry_price=0.0,
            breakeven_fee_buffer_pct=0.001,
        )
        assert result == 0.0

        result2 = effective_breakeven_for_tp_selection(
            side="LONG",
            net_remaining_breakeven_price=0.0,
            avg_entry_price=-1.0,
            breakeven_fee_buffer_pct=0.001,
        )
        assert result2 == 0.0

    def test_long_fee_buffer(self):
        result = effective_breakeven_for_tp_selection(
            side="LONG",
            net_remaining_breakeven_price=0.0,
            avg_entry_price=100.0,
            breakeven_fee_buffer_pct=0.001,
        )
        assert result == 100.1  # 100 * 1.001

    def test_short_fee_buffer(self):
        result = effective_breakeven_for_tp_selection(
            side="SHORT",
            net_remaining_breakeven_price=0.0,
            avg_entry_price=100.0,
            breakeven_fee_buffer_pct=0.001,
        )
        assert result == 99.9  # 100 * 0.999


# ═══════════════════════════════════════════════════════════════════════
# 7. select_tp_price
# ═══════════════════════════════════════════════════════════════════════

class TestSelectTpPrice:
    def test_effective_be_zero_returns_middle(self):
        tp_band = _tp_band(middle=100.0)
        sel = select_tp_price(
            side="LONG", effective_be=0.0, min_net_profit=0.002,
            tp_band=tp_band, tp_boll_enabled=True,
        )
        assert sel.price == 100.0
        assert sel.mode == "MIDDLE"

    def test_long_middle_meets_profit_returns_middle(self):
        tp_band = _tp_band(tp_middle=101.0)
        sel = select_tp_price(
            side="LONG", effective_be=98.0, min_net_profit=0.002,
            tp_band=tp_band, tp_boll_enabled=True,
        )
        # required = 98.0 * 1.002 = 98.196, tp_mid=101.0 >= 98.196 → MIDDLE
        assert sel.price == 101.0
        assert sel.mode == "MIDDLE"

    def test_long_structure_middle_fallback(self):
        tp_band = _tp_band(tp_middle=100.3, middle=101.0)
        sel = select_tp_price(
            side="LONG", effective_be=100.0, min_net_profit=0.005,
            tp_band=tp_band, tp_boll_enabled=True,
        )
        # required = 100.0 * 1.005 = 100.5
        # tp_mid=100.3 < 100.5 → skip
        # boll.middle=101.0 >= 100.5 → use structure middle
        assert sel.price == 101.0
        assert sel.mode == "MIDDLE"

    def test_long_middle_fails_returns_outer_upper(self):
        tp_band = _tp_band(middle=100.0, tp_middle=100.0, tp_upper=108.0, upper=110.0)
        sel = select_tp_price(
            side="LONG", effective_be=100.0, min_net_profit=0.05,
            tp_band=tp_band, tp_boll_enabled=True,
        )
        # required = 100.0 * 1.05 = 105.0, neither middle >= 105 → outer
        assert sel.price == 108.0
        assert sel.mode == "UPPER"

    def test_short_middle_meets_profit_returns_middle(self):
        tp_band = _tp_band(tp_middle=99.0)
        sel = select_tp_price(
            side="SHORT", effective_be=102.0, min_net_profit=0.002,
            tp_band=tp_band, tp_boll_enabled=True,
        )
        # required = 102.0 * 0.998 = 101.796, tp_mid=99.0 <= 101.796 → MIDDLE
        assert sel.price == 99.0
        assert sel.mode == "MIDDLE"

    def test_short_structure_middle_fallback(self):
        tp_band = _tp_band(tp_middle=99.7, middle=99.0)
        sel = select_tp_price(
            side="SHORT", effective_be=100.0, min_net_profit=0.005,
            tp_band=tp_band, tp_boll_enabled=True,
        )
        # required = 100.0 * 0.995 = 99.5
        # tp_mid=99.7 > 99.5 → skip
        # boll.middle=99.0 <= 99.5 → use structure middle
        assert sel.price == 99.0
        assert sel.mode == "MIDDLE"

    def test_short_middle_fails_returns_outer_lower(self):
        tp_band = _tp_band(middle=100.0, tp_middle=100.0, tp_lower=92.0, lower=90.0)
        sel = select_tp_price(
            side="SHORT", effective_be=100.0, min_net_profit=0.05,
            tp_band=tp_band, tp_boll_enabled=True,
        )
        # required = 100.0 * 0.95 = 95.0, neither middle <= 95 → outer
        assert sel.price == 92.0
        assert sel.mode == "LOWER"

    def test_tp_boll_disabled_uses_structure(self):
        tp_band = _tp_band(tp_middle=101.0, middle=100.0, upper=110.0, tp_upper=108.0)
        sel = select_tp_price(
            side="LONG", effective_be=100.0, min_net_profit=0.005,
            tp_band=tp_band, tp_boll_enabled=False,
        )
        # With tp_boll_enabled=False, middle=100 < required=100.5 → outer
        # outer uses structure upper=110.0
        assert sel.price == 110.0
        assert sel.mode == "UPPER"


# ═══════════════════════════════════════════════════════════════════════
# 8. three_stage_runner_plan_allowed
# ═══════════════════════════════════════════════════════════════════════

class TestThreeStageRunnerPlanAllowed:
    def _default_allowed_kwargs(self, **overrides):
        kwargs = dict(
            three_stage_runner_enabled=True,
            three_stage_pre_tp1_degrade_stage=None,
            tp_mode="MIDDLE",
            boll_exists=True,
            near_tp_protected=False,
            near_tp_add_disabled=False,
            partial_tp_consumed=False,
            middle_runner_enabled_for_position=False,
            middle_runner_pending=False,
            middle_runner_active=False,
            tp_plan=None,
            trend_runner_active=False,
        )
        kwargs.update(overrides)
        return kwargs

    def test_happy_path_true(self):
        assert three_stage_runner_plan_allowed(**self._default_allowed_kwargs()) is True

    def test_not_enabled_returns_false(self):
        assert three_stage_runner_plan_allowed(
            **self._default_allowed_kwargs(three_stage_runner_enabled=False)
        ) is False

    def test_degrade_stage_not_none_returns_false(self):
        assert three_stage_runner_plan_allowed(
            **self._default_allowed_kwargs(three_stage_pre_tp1_degrade_stage="SINGLE")
        ) is False

    def test_tp_mode_not_middle_returns_false(self):
        assert three_stage_runner_plan_allowed(
            **self._default_allowed_kwargs(tp_mode="UPPER")
        ) is False

    def test_boll_not_exists_returns_false(self):
        assert three_stage_runner_plan_allowed(
            **self._default_allowed_kwargs(boll_exists=False)
        ) is False

    def test_near_tp_protected_returns_false(self):
        assert three_stage_runner_plan_allowed(
            **self._default_allowed_kwargs(near_tp_protected=True)
        ) is False

    def test_near_tp_add_disabled_returns_false(self):
        assert three_stage_runner_plan_allowed(
            **self._default_allowed_kwargs(near_tp_add_disabled=True)
        ) is False

    def test_partial_tp_consumed_returns_false(self):
        assert three_stage_runner_plan_allowed(
            **self._default_allowed_kwargs(partial_tp_consumed=True)
        ) is False

    def test_middle_runner_enabled_returns_false(self):
        assert three_stage_runner_plan_allowed(
            **self._default_allowed_kwargs(middle_runner_enabled_for_position=True)
        ) is False

    def test_middle_runner_pending_returns_false(self):
        assert three_stage_runner_plan_allowed(
            **self._default_allowed_kwargs(middle_runner_pending=True)
        ) is False

    def test_middle_runner_active_returns_false(self):
        assert three_stage_runner_plan_allowed(
            **self._default_allowed_kwargs(middle_runner_active=True)
        ) is False

    def test_tp_plan_middle_runner_returns_false(self):
        assert three_stage_runner_plan_allowed(
            **self._default_allowed_kwargs(tp_plan="MIDDLE_RUNNER")
        ) is False

    def test_trend_runner_active_returns_false(self):
        assert three_stage_runner_plan_allowed(
            **self._default_allowed_kwargs(trend_runner_active=True)
        ) is False


# ═══════════════════════════════════════════════════════════════════════
# 9. middle_runner_plan_allowed
# ═══════════════════════════════════════════════════════════════════════

class TestMiddleRunnerPlanAllowed:
    def _default_allowed_kwargs(self, **overrides):
        kwargs = dict(
            middle_runner_enabled=True,
            tp_mode="MIDDLE",
            boll_exists=True,
            near_tp_protected=False,
            near_tp_add_disabled=False,
            partial_tp_consumed=False,
            middle_runner_active=False,
            three_stage_runner_enabled_for_position=False,
            tp_plan=None,
            three_stage_tp1_consumed=False,
            three_stage_tp2_consumed=False,
        )
        kwargs.update(overrides)
        return kwargs

    def test_happy_path_true(self):
        assert middle_runner_plan_allowed(**self._default_allowed_kwargs()) is True

    def test_not_enabled_returns_false(self):
        assert middle_runner_plan_allowed(
            **self._default_allowed_kwargs(middle_runner_enabled=False)
        ) is False

    def test_tp_mode_not_middle_returns_false(self):
        assert middle_runner_plan_allowed(
            **self._default_allowed_kwargs(tp_mode="UPPER")
        ) is False

    def test_boll_not_exists_returns_false(self):
        assert middle_runner_plan_allowed(
            **self._default_allowed_kwargs(boll_exists=False)
        ) is False

    def test_near_tp_protected_returns_false(self):
        assert middle_runner_plan_allowed(
            **self._default_allowed_kwargs(near_tp_protected=True)
        ) is False

    def test_partial_tp_consumed_returns_false(self):
        assert middle_runner_plan_allowed(
            **self._default_allowed_kwargs(partial_tp_consumed=True)
        ) is False

    def test_middle_runner_active_returns_false(self):
        assert middle_runner_plan_allowed(
            **self._default_allowed_kwargs(middle_runner_active=True)
        ) is False

    def test_three_stage_enabled_returns_false(self):
        assert middle_runner_plan_allowed(
            **self._default_allowed_kwargs(three_stage_runner_enabled_for_position=True)
        ) is False

    def test_tp_plan_three_stage_returns_false(self):
        assert middle_runner_plan_allowed(
            **self._default_allowed_kwargs(tp_plan="THREE_STAGE_RUNNER")
        ) is False

    def test_three_stage_tp1_consumed_returns_false(self):
        assert middle_runner_plan_allowed(
            **self._default_allowed_kwargs(three_stage_tp1_consumed=True)
        ) is False

    def test_three_stage_tp2_consumed_returns_false(self):
        assert middle_runner_plan_allowed(
            **self._default_allowed_kwargs(three_stage_tp2_consumed=True)
        ) is False


# ═══════════════════════════════════════════════════════════════════════
# 10. select_tp_plan
# ═══════════════════════════════════════════════════════════════════════

class TestSelectTpPlan:
    def _default_kwargs(self, **overrides):
        kwargs = dict(
            side="LONG",
            final_tp=110.0,
            layers=5,
            tp_mode="MIDDLE",
            boll_exists=True,
            three_stage_pre_tp1_degrade_stage=None,
            middle_runner_first_close_ratio=0.5,
            tp_middle_profit_fallback_price=101.0,
            three_stage_runner_plan_allowed=False,
            three_stage_tp1_ratio=0.4,
            three_stage_runner_enabled=False,
            middle_runner_plan_allowed=False,
            split_tp_enabled=True,
            split_tp_min_layers=3,
            partial_tp_consumed=False,
            avg_entry=100.0,
            split_tp_partial_ratio=0.5,
            split_tp_path_ratio=0.5,
            split_tp_min_profit_pct=0.005,
        )
        kwargs.update(overrides)
        return kwargs

    def test_degrade_single_returns_single(self):
        sel = select_tp_plan(**self._default_kwargs(three_stage_pre_tp1_degrade_stage="SINGLE"))
        assert sel.tp_plan == "SINGLE"
        assert sel.partial_tp_price is None
        assert sel.partial_tp_ratio == 0.0

    def test_degrade_middle_runner_returns_middle_runner(self):
        sel = select_tp_plan(**self._default_kwargs(
            three_stage_pre_tp1_degrade_stage="MIDDLE_RUNNER",
        ))
        assert sel.tp_plan == "MIDDLE_RUNNER"
        assert sel.partial_tp_price == 101.0
        assert sel.partial_tp_ratio == 0.5

    def test_degrade_middle_runner_clamps_ratio(self):
        sel = select_tp_plan(**self._default_kwargs(
            three_stage_pre_tp1_degrade_stage="MIDDLE_RUNNER",
            middle_runner_first_close_ratio=0.05,  # below 0.1 → clamp to 0.1
        ))
        assert sel.partial_tp_ratio == 0.1

        sel2 = select_tp_plan(**self._default_kwargs(
            three_stage_pre_tp1_degrade_stage="MIDDLE_RUNNER",
            middle_runner_first_close_ratio=0.99,  # above 0.95 → clamp to 0.95
        ))
        assert sel2.partial_tp_ratio == 0.95

    def test_degrade_middle_runner_not_middle_mode_returns_single(self):
        sel = select_tp_plan(**self._default_kwargs(
            three_stage_pre_tp1_degrade_stage="MIDDLE_RUNNER",
            tp_mode="UPPER",
        ))
        assert sel.tp_plan == "SINGLE"

    def test_degrade_middle_runner_no_boll_returns_single(self):
        sel = select_tp_plan(**self._default_kwargs(
            three_stage_pre_tp1_degrade_stage="MIDDLE_RUNNER",
            boll_exists=False,
        ))
        assert sel.tp_plan == "SINGLE"

    def test_three_stage_allowed_returns_three_stage(self):
        sel = select_tp_plan(**self._default_kwargs(
            three_stage_runner_plan_allowed=True,
            three_stage_tp1_ratio=0.4,
        ))
        assert sel.tp_plan == "THREE_STAGE_RUNNER"
        assert sel.partial_tp_price == 101.0
        assert sel.partial_tp_ratio == 0.4

    def test_three_stage_enabled_but_not_allowed_returns_single(self):
        sel = select_tp_plan(**self._default_kwargs(
            three_stage_runner_enabled=True,
            three_stage_runner_plan_allowed=False,
        ))
        assert sel.tp_plan == "SINGLE"

    def test_middle_runner_allowed_returns_middle_runner(self):
        sel = select_tp_plan(**self._default_kwargs(
            middle_runner_plan_allowed=True,
        ))
        assert sel.tp_plan == "MIDDLE_RUNNER"
        assert sel.partial_tp_price == 101.0
        assert sel.partial_tp_ratio == 0.5

    def test_middle_runner_allowed_clamps_ratio(self):
        sel = select_tp_plan(**self._default_kwargs(
            middle_runner_plan_allowed=True,
            middle_runner_first_close_ratio=0.05,
        ))
        assert sel.partial_tp_ratio == 0.1

    def test_tp_mode_not_middle_returns_single(self):
        sel = select_tp_plan(**self._default_kwargs(tp_mode="UPPER"))
        assert sel.tp_plan == "SINGLE"

    def test_split_disabled_returns_single(self):
        sel = select_tp_plan(**self._default_kwargs(split_tp_enabled=False))
        assert sel.tp_plan == "SINGLE"

    def test_layers_insufficient_returns_single(self):
        sel = select_tp_plan(**self._default_kwargs(layers=2, split_tp_min_layers=3))
        assert sel.tp_plan == "SINGLE"

    def test_partial_tp_consumed_returns_single(self):
        sel = select_tp_plan(**self._default_kwargs(partial_tp_consumed=True))
        assert sel.tp_plan == "SINGLE"

    def test_split_long_valid_returns_split_partial_final(self):
        sel = select_tp_plan(**self._default_kwargs(
            side="LONG", final_tp=110.0, avg_entry=100.0,
        ))
        assert sel.tp_plan == "SPLIT_PARTIAL_FINAL"
        # path_tp = 100 + (110 - 100) * 0.5 = 105.0
        # min_tp = 100 * (1 + 0.005) = 100.5
        # partial_tp = max(105.0, 100.5) = 105.0
        assert sel.partial_tp_price == 105.0
        assert sel.partial_tp_ratio == 0.5

    def test_split_short_valid_returns_split_partial_final(self):
        sel = select_tp_plan(**self._default_kwargs(
            side="SHORT", final_tp=90.0, avg_entry=100.0,
        ))
        assert sel.tp_plan == "SPLIT_PARTIAL_FINAL"
        # path_tp = 100 - (100 - 90) * 0.5 = 95.0
        # min_tp = 100 * (1 - 0.005) = 99.5
        # partial_tp = min(95.0, 99.5) = 95.0
        assert sel.partial_tp_price == 95.0
        assert sel.partial_tp_ratio == 0.5

    def test_split_long_final_tp_too_close_to_entry_returns_single(self):
        # final_tp <= min_tp → SINGLE
        sel = select_tp_plan(**self._default_kwargs(
            side="LONG", final_tp=100.3, avg_entry=100.0,
            split_tp_min_profit_pct=0.005,  # min_tp = 100.5
        ))
        assert sel.tp_plan == "SINGLE"

    def test_split_long_partial_tp_exceeds_final_returns_single(self):
        # partial_tp >= final_tp → SINGLE
        sel = select_tp_plan(**self._default_kwargs(
            side="LONG", final_tp=101.0, avg_entry=100.0,
            split_tp_min_profit_pct=0.0,  # min_tp = 100.0
            split_tp_path_ratio=0.99,  # path_tp = 100 + (101-100)*0.99 = 100.99
        ))
        # partial_tp = max(100.99, 100.0) = 100.99
        # partial_tp < 101.0 → valid → SPLIT
        assert sel.tp_plan == "SPLIT_PARTIAL_FINAL"

    def test_split_short_final_tp_too_close_to_entry_returns_single(self):
        sel = select_tp_plan(**self._default_kwargs(
            side="SHORT", final_tp=99.7, avg_entry=100.0,
            split_tp_min_profit_pct=0.005,  # min_tp = 99.5, final_tp=99.7 >= 99.5 → SINGLE
        ))
        assert sel.tp_plan == "SINGLE"

    def test_invalid_partial_ratio_returns_single(self):
        sel = select_tp_plan(**self._default_kwargs(
            split_tp_partial_ratio=0.0,  # clamp → 0.0, <= 0 → invalid
        ))
        assert sel.tp_plan == "SINGLE"

        sel2 = select_tp_plan(**self._default_kwargs(
            split_tp_partial_ratio=1.0,  # clamp → 1.0, >= 1 → invalid
        ))
        assert sel2.tp_plan == "SINGLE"

    def test_invalid_path_ratio_returns_single(self):
        sel = select_tp_plan(**self._default_kwargs(
            split_tp_path_ratio=0.0,
        ))
        assert sel.tp_plan == "SINGLE"

    def test_avg_entry_zero_returns_single(self):
        sel = select_tp_plan(**self._default_kwargs(avg_entry=0.0))
        assert sel.tp_plan == "SINGLE"

    def test_final_tp_zero_returns_single(self):
        sel = select_tp_plan(**self._default_kwargs(final_tp=0.0))
        assert sel.tp_plan == "SINGLE"

    def test_long_split_min_profit_uses_abs_value(self):
        """split_tp_min_profit_pct uses abs(), negative values work same."""
        sel1 = select_tp_plan(**self._default_kwargs(
            side="LONG", final_tp=110.0, avg_entry=100.0,
            split_tp_min_profit_pct=0.005,
        ))
        sel2 = select_tp_plan(**self._default_kwargs(
            side="LONG", final_tp=110.0, avg_entry=100.0,
            split_tp_min_profit_pct=-0.005,
        ))
        assert sel1.tp_plan == sel2.tp_plan
        assert sel1.partial_tp_price == sel2.partial_tp_price


# ═══════════════════════════════════════════════════════════════════════
# 11. tp_plan_unchanged
# ═══════════════════════════════════════════════════════════════════════

class TestTpPlanUnchanged:
    def test_no_current_tp_price_returns_false(self):
        decision = tp_plan_unchanged(
            current_tp_price=None,
            current_tp_plan="SINGLE",
            current_partial_tp_price=None,
            current_partial_tp_ratio=0.0,
            new_tp_price=100.0,
            new_partial_tp_price=None,
            new_partial_tp_ratio=0.0,
            new_tp_plan="SINGLE",
        )
        assert decision.unchanged is False

    def test_exact_same_returns_true(self):
        decision = tp_plan_unchanged(
            current_tp_price=100.0,
            current_tp_plan="SINGLE",
            current_partial_tp_price=None,
            current_partial_tp_ratio=0.0,
            new_tp_price=100.0,
            new_partial_tp_price=None,
            new_partial_tp_ratio=0.0,
            new_tp_plan="SINGLE",
        )
        assert decision.unchanged is True

    def test_tp_price_diff_returns_false(self):
        decision = tp_plan_unchanged(
            current_tp_price=100.0,
            current_tp_plan="SINGLE",
            current_partial_tp_price=None,
            current_partial_tp_ratio=0.0,
            new_tp_price=101.0,
            new_partial_tp_price=None,
            new_partial_tp_ratio=0.0,
            new_tp_plan="SINGLE",
        )
        assert decision.unchanged is False

    def test_plan_diff_returns_false(self):
        decision = tp_plan_unchanged(
            current_tp_price=100.0,
            current_tp_plan="SINGLE",
            current_partial_tp_price=None,
            current_partial_tp_ratio=0.0,
            new_tp_price=100.0,
            new_partial_tp_price=None,
            new_partial_tp_ratio=0.0,
            new_tp_plan="SPLIT_PARTIAL_FINAL",
        )
        assert decision.unchanged is False

    def test_partial_ratio_diff_returns_false(self):
        decision = tp_plan_unchanged(
            current_tp_price=100.0,
            current_tp_plan="SINGLE",
            current_partial_tp_price=None,
            current_partial_tp_ratio=0.0,
            new_tp_price=100.0,
            new_partial_tp_price=None,
            new_partial_tp_ratio=0.5,
            new_tp_plan="SINGLE",
        )
        assert decision.unchanged is False

    def test_one_partial_none_mismatch_returns_false(self):
        decision = tp_plan_unchanged(
            current_tp_price=100.0,
            current_tp_plan="SPLIT_PARTIAL_FINAL",
            current_partial_tp_price=None,
            current_partial_tp_ratio=0.5,
            new_tp_price=100.0,
            new_partial_tp_price=101.0,
            new_partial_tp_ratio=0.5,
            new_tp_plan="SPLIT_PARTIAL_FINAL",
        )
        assert decision.unchanged is False

        decision2 = tp_plan_unchanged(
            current_tp_price=100.0,
            current_tp_plan="SPLIT_PARTIAL_FINAL",
            current_partial_tp_price=101.0,
            current_partial_tp_ratio=0.5,
            new_tp_price=100.0,
            new_partial_tp_price=None,
            new_partial_tp_ratio=0.5,
            new_tp_plan="SPLIT_PARTIAL_FINAL",
        )
        assert decision2.unchanged is False

    def test_both_partial_none_returns_true(self):
        decision = tp_plan_unchanged(
            current_tp_price=100.0,
            current_tp_plan="SINGLE",
            current_partial_tp_price=None,
            current_partial_tp_ratio=0.0,
            new_tp_price=100.0,
            new_partial_tp_price=None,
            new_partial_tp_ratio=0.0,
            new_tp_plan="SINGLE",
        )
        assert decision.unchanged is True

    def test_partial_prices_close_enough_returns_true(self):
        decision = tp_plan_unchanged(
            current_tp_price=100.0,
            current_tp_plan="SPLIT_PARTIAL_FINAL",
            current_partial_tp_price=101.0,
            current_partial_tp_ratio=0.5,
            new_tp_price=100.0,
            new_partial_tp_price=101.00005,  # diff < 0.0001 relative
            new_partial_tp_ratio=0.5,
            new_tp_plan="SPLIT_PARTIAL_FINAL",
        )
        assert decision.unchanged is True

    def test_partial_prices_differ_too_much_returns_false(self):
        decision = tp_plan_unchanged(
            current_tp_price=100.0,
            current_tp_plan="SPLIT_PARTIAL_FINAL",
            current_partial_tp_price=101.0,
            current_partial_tp_ratio=0.5,
            new_tp_price=100.0,
            new_partial_tp_price=102.0,
            new_partial_tp_ratio=0.5,
            new_tp_plan="SPLIT_PARTIAL_FINAL",
        )
        assert decision.unchanged is False

    def test_tp_price_below_threshold_returns_true(self):
        """Relative diff < 0.0001 is considered unchanged."""
        decision = tp_plan_unchanged(
            current_tp_price=100.0,
            current_tp_plan="SINGLE",
            current_partial_tp_price=None,
            current_partial_tp_ratio=0.0,
            new_tp_price=100.005,  # diff = 0.005, relative = 0.005/100.005 ≈ 0.0000499 < 0.0001
            new_partial_tp_price=None,
            new_partial_tp_ratio=0.0,
            new_tp_plan="SINGLE",
        )
        assert decision.unchanged is True
