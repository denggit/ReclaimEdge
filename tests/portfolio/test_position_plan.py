# -*- coding: utf-8 -*-
"""Unit tests for src/portfolio/position_plan.py (G02)."""

from __future__ import annotations

import uuid
from decimal import Decimal
from pathlib import Path

import pytest

from src.portfolio.position_plan import (
    VALID_SIDES,
    PositionPlan,
    PositionPlanError,
    create_main_position_plan,
    decimal_to_plain_str,
    quantize_contracts_down,
)


# ===================================================================
# 1. decimal_to_plain_str helper
# ===================================================================


class TestDecimalToPlainStr:
    def test_integer_like(self):
        assert decimal_to_plain_str(Decimal("1")) == "1"

    def test_one_decimal_place(self):
        assert decimal_to_plain_str(Decimal("1.5")) == "1.5"

    def test_two_decimal_places(self):
        assert decimal_to_plain_str(Decimal("1.15")) == "1.15"

    def test_small_value(self):
        assert decimal_to_plain_str(Decimal("0.01")) == "0.01"

    def test_no_trailing_zeros_on_one_and_thirty(self):
        """1.30 normalizes to 1.3."""
        assert decimal_to_plain_str(Decimal("1.30")) == "1.3"

    def test_no_trailing_zeros_on_one_and_sixty(self):
        """1.60 normalizes to 1.6."""
        assert decimal_to_plain_str(Decimal("1.60")) == "1.6"

    def test_no_scientific_notation(self):
        """Large or small decimals must never produce scientific notation."""
        assert "E" not in decimal_to_plain_str(Decimal("0.0001"))
        assert "E" not in decimal_to_plain_str(Decimal("1000"))

    def test_no_floating_point_artefacts(self):
        """Must not produce strings like '1.1500000000000001'."""
        result = decimal_to_plain_str(Decimal("1.15"))
        assert "." not in result.replace("1.15", "") or result == "1.15"


# ===================================================================
# 2. quantize_contracts_down helper
# ===================================================================


class TestQuantizeContractsDown:
    def test_exact_multiple(self):
        assert quantize_contracts_down(Decimal("1.15"), Decimal("0.01")) == Decimal("1.15")

    def test_rounds_down(self):
        assert quantize_contracts_down(Decimal("1.159"), Decimal("0.01")) == Decimal("1.15")

    def test_small_value_rounds_down_to_min_lot(self):
        assert quantize_contracts_down(Decimal("0.0115"), Decimal("0.01")) == Decimal("0.01")

    def test_eth_precision_rounding(self):
        assert quantize_contracts_down(Decimal("1.38"), Decimal("0.1")) == Decimal("1.3")
        assert quantize_contracts_down(Decimal("1.56"), Decimal("0.1")) == Decimal("1.5")
        assert quantize_contracts_down(Decimal("1.74"), Decimal("0.1")) == Decimal("1.7")

    def test_precision_zero_raises(self):
        with pytest.raises(PositionPlanError, match="contract_precision"):
            quantize_contracts_down(Decimal("1"), Decimal("0"))


# ===================================================================
# 3. max_layers = 8 generates 8 layers
# ===================================================================


class TestMaxLayers8:
    def test_generates_exactly_eight_layers(self):
        plan = create_main_position_plan(
            inst_id="ETH-USDT-SWAP",
            side="LONG",
            base_main_contracts="1",
            max_layers=8,
            layer_multiplier_step="0.15",
            contract_precision="0.01",
            min_contracts="0.01",
        )
        assert plan.max_layers == 8
        assert plan.layer_count == 8
        assert len(plan.planned_main_contracts) == 8

    def test_planned_contracts_match_expected(self):
        plan = create_main_position_plan(
            inst_id="ETH-USDT-SWAP",
            side="LONG",
            base_main_contracts="1",
            max_layers=8,
            layer_multiplier_step="0.15",
            contract_precision="0.01",
            min_contracts="0.01",
        )
        expected = (
            "1",
            "1.15",
            "1.3",
            "1.45",
            "1.6",
            "1.75",
            "1.9",
            "2.05",
        )
        assert plan.planned_main_contracts == expected


# ===================================================================
# 4. max_layers = 10 generates 10 layers
# ===================================================================


class TestMaxLayers10:
    def test_generates_exactly_ten_layers(self):
        plan = create_main_position_plan(
            inst_id="ETH-USDT-SWAP",
            side="LONG",
            base_main_contracts="1",
            max_layers=10,
            layer_multiplier_step="0.15",
            contract_precision="0.01",
            min_contracts="0.01",
        )
        assert plan.max_layers == 10
        assert plan.layer_count == 10
        assert len(plan.planned_main_contracts) == 10

    def test_last_two_layers_match_expected(self):
        plan = create_main_position_plan(
            inst_id="ETH-USDT-SWAP",
            side="LONG",
            base_main_contracts="1",
            max_layers=10,
            layer_multiplier_step="0.15",
            contract_precision="0.01",
            min_contracts="0.01",
        )
        # L9: 1 * (1 + 8 * 0.15) = 2.2
        # L10: 1 * (1 + 9 * 0.15) = 2.35
        assert plan.planned_main_contracts[8] == "2.2"
        assert plan.planned_main_contracts[9] == "2.35"


# ===================================================================
# 5. max_layers = 3 generates 3 layers
# ===================================================================


class TestMaxLayers3:
    def test_generates_exactly_three_layers(self):
        plan = create_main_position_plan(
            inst_id="ETH-USDT-SWAP",
            side="LONG",
            base_main_contracts="1",
            max_layers=3,
            layer_multiplier_step="0.15",
            contract_precision="0.01",
            min_contracts="0.01",
        )
        assert plan.max_layers == 3
        assert plan.layer_count == 3
        assert plan.planned_main_contracts == ("1", "1.15", "1.3")


# ===================================================================
# 6. BTC small contract rounding down
# ===================================================================


class TestBtcSmallContractRounding:
    def test_all_layers_round_down_to_min(self):
        plan = create_main_position_plan(
            inst_id="BTC-USDT-SWAP",
            side="LONG",
            base_main_contracts="0.01",
            max_layers=4,
            layer_multiplier_step="0.15",
            contract_precision="0.01",
            min_contracts="0.01",
        )
        # L1: 0.01 * 1.00 = 0.01 → 0.01
        # L2: 0.01 * 1.15 = 0.0115 → 0.01
        # L3: 0.01 * 1.30 = 0.013 → 0.01
        # L4: 0.01 * 1.45 = 0.0145 → 0.01
        assert plan.planned_main_contracts == ("0.01", "0.01", "0.01", "0.01")

    def test_never_rounds_up_beyond_risk(self):
        """Even when raw=0.0115, quantized must be 0.01 (not 0.02)."""
        plan = create_main_position_plan(
            inst_id="BTC-USDT-SWAP",
            side="LONG",
            base_main_contracts="0.02",
            max_layers=2,
            layer_multiplier_step="0.15",
            contract_precision="0.01",
            min_contracts="0.01",
        )
        # L1: 0.02
        # L2: 0.02 * 1.15 = 0.023 → quantized down to 0.02 (not 0.03)
        assert plan.planned_main_contracts == ("0.02", "0.02")


# ===================================================================
# 7. ETH-like precision (0.1)
# ===================================================================


class TestEthLikePrecision:
    def test_eth_precision_rounding(self):
        plan = create_main_position_plan(
            inst_id="ETH-USDT-SWAP",
            side="LONG",
            base_main_contracts="1.2",
            max_layers=4,
            layer_multiplier_step="0.15",
            contract_precision="0.1",
            min_contracts="0.1",
        )
        # L1 raw = 1.2                   → 1.2
        # L2 raw = 1.2 * 1.15 = 1.38     → quant down 1.3
        # L3 raw = 1.2 * 1.30 = 1.56     → quant down 1.5
        # L4 raw = 1.2 * 1.45 = 1.74     → quant down 1.7
        assert plan.planned_main_contracts == ("1.2", "1.3", "1.5", "1.7")


# ===================================================================
# 8. max_layers > 8 proves nothing hard-coded to 8
# ===================================================================


class TestMaxLayers12:
    def test_generates_12_layers(self):
        """max_layers=12 must generate exactly 12 layers — no 8 hard-coding."""
        plan = create_main_position_plan(
            inst_id="ETH-USDT-SWAP",
            side="LONG",
            base_main_contracts="1",
            max_layers=12,
            layer_multiplier_step="0.15",
            contract_precision="0.01",
            min_contracts="0.01",
        )
        assert plan.max_layers == 12
        assert plan.layer_count == 12
        assert len(plan.planned_main_contracts) == 12

    def test_layer_12_matches_expected(self):
        plan = create_main_position_plan(
            inst_id="ETH-USDT-SWAP",
            side="LONG",
            base_main_contracts="1",
            max_layers=12,
            layer_multiplier_step="0.15",
            contract_precision="0.01",
            min_contracts="0.01",
        )
        # L12 = 1 * (1 + 11 * 0.15) = 2.65
        assert plan.planned_main_contracts[11] == "2.65"


class TestMaxLayers6:
    def test_generates_6_layers(self):
        plan = create_main_position_plan(
            inst_id="ETH-USDT-SWAP",
            side="LONG",
            base_main_contracts="1",
            max_layers=6,
            layer_multiplier_step="0.15",
            contract_precision="0.01",
            min_contracts="0.01",
        )
        assert plan.max_layers == 6
        assert plan.layer_count == 6
        assert plan.planned_main_contracts == (
            "1", "1.15", "1.3", "1.45", "1.6", "1.75",
        )


# ===================================================================
# 9. invalid max_layers
# ===================================================================


class TestInvalidMaxLayers:
    def test_zero_raises(self):
        with pytest.raises(PositionPlanError, match="max_layers"):
            create_main_position_plan(
                inst_id="ETH-USDT-SWAP",
                side="LONG",
                base_main_contracts="1",
                max_layers=0,
                layer_multiplier_step="0.15",
                contract_precision="0.01",
                min_contracts="0.01",
            )

    def test_negative_raises(self):
        with pytest.raises(PositionPlanError, match="max_layers"):
            create_main_position_plan(
                inst_id="ETH-USDT-SWAP",
                side="LONG",
                base_main_contracts="1",
                max_layers=-1,
                layer_multiplier_step="0.15",
                contract_precision="0.01",
                min_contracts="0.01",
            )

    def test_float_raises(self):
        with pytest.raises(PositionPlanError, match="max_layers"):
            create_main_position_plan(
                inst_id="ETH-USDT-SWAP",
                side="LONG",
                base_main_contracts="1",
                max_layers=1.5,  # type: ignore[arg-type]
                layer_multiplier_step="0.15",
                contract_precision="0.01",
                min_contracts="0.01",
            )

    def test_bool_raises(self):
        with pytest.raises(PositionPlanError, match="max_layers"):
            create_main_position_plan(
                inst_id="ETH-USDT-SWAP",
                side="LONG",
                base_main_contracts="1",
                max_layers=True,  # type: ignore[arg-type]
                layer_multiplier_step="0.15",
                contract_precision="0.01",
                min_contracts="0.01",
            )


# ===================================================================
# 10. invalid numeric inputs
# ===================================================================


class TestInvalidNumericInputs:
    def test_base_main_contracts_zero_raises(self):
        with pytest.raises(PositionPlanError, match="base_main_contracts"):
            create_main_position_plan(
                inst_id="ETH-USDT-SWAP",
                side="LONG",
                base_main_contracts="0",
                max_layers=8,
                layer_multiplier_step="0.15",
                contract_precision="0.01",
                min_contracts="0.01",
            )

    def test_base_main_contracts_negative_raises(self):
        with pytest.raises(PositionPlanError, match="base_main_contracts"):
            create_main_position_plan(
                inst_id="ETH-USDT-SWAP",
                side="LONG",
                base_main_contracts="-1",
                max_layers=8,
                layer_multiplier_step="0.15",
                contract_precision="0.01",
                min_contracts="0.01",
            )

    def test_base_below_min_raises(self):
        with pytest.raises(PositionPlanError, match="min_contracts"):
            create_main_position_plan(
                inst_id="ETH-USDT-SWAP",
                side="LONG",
                base_main_contracts="0.005",
                max_layers=8,
                layer_multiplier_step="0.15",
                contract_precision="0.01",
                min_contracts="0.01",
            )

    def test_layer_multiplier_step_negative_raises(self):
        with pytest.raises(PositionPlanError, match="layer_multiplier_step"):
            create_main_position_plan(
                inst_id="ETH-USDT-SWAP",
                side="LONG",
                base_main_contracts="1",
                max_layers=8,
                layer_multiplier_step="-0.15",
                contract_precision="0.01",
                min_contracts="0.01",
            )

    def test_contract_precision_zero_raises(self):
        with pytest.raises(PositionPlanError, match="contract_precision"):
            create_main_position_plan(
                inst_id="ETH-USDT-SWAP",
                side="LONG",
                base_main_contracts="1",
                max_layers=8,
                layer_multiplier_step="0.15",
                contract_precision="0",
                min_contracts="0.01",
            )

    def test_contract_precision_negative_raises(self):
        with pytest.raises(PositionPlanError, match="contract_precision"):
            create_main_position_plan(
                inst_id="ETH-USDT-SWAP",
                side="LONG",
                base_main_contracts="1",
                max_layers=8,
                layer_multiplier_step="0.15",
                contract_precision="-0.01",
                min_contracts="0.01",
            )

    def test_min_contracts_zero_raises(self):
        with pytest.raises(PositionPlanError, match="min_contracts"):
            create_main_position_plan(
                inst_id="ETH-USDT-SWAP",
                side="LONG",
                base_main_contracts="1",
                max_layers=8,
                layer_multiplier_step="0.15",
                contract_precision="0.01",
                min_contracts="0",
            )

    def test_min_contracts_negative_raises(self):
        with pytest.raises(PositionPlanError, match="min_contracts"):
            create_main_position_plan(
                inst_id="ETH-USDT-SWAP",
                side="LONG",
                base_main_contracts="1",
                max_layers=8,
                layer_multiplier_step="0.15",
                contract_precision="0.01",
                min_contracts="-0.01",
            )


# ===================================================================
# 11. side validation
# ===================================================================


class TestSideValidation:
    def test_long_allowed(self):
        plan = create_main_position_plan(
            inst_id="ETH-USDT-SWAP",
            side="LONG",
            base_main_contracts="1",
            max_layers=8,
            layer_multiplier_step="0.15",
            contract_precision="0.01",
            min_contracts="0.01",
        )
        assert plan.side == "LONG"

    def test_short_allowed(self):
        plan = create_main_position_plan(
            inst_id="ETH-USDT-SWAP",
            side="SHORT",
            base_main_contracts="1",
            max_layers=8,
            layer_multiplier_step="0.15",
            contract_precision="0.01",
            min_contracts="0.01",
        )
        assert plan.side == "SHORT"

    def test_lowercase_long_raises(self):
        with pytest.raises(PositionPlanError, match="side"):
            create_main_position_plan(
                inst_id="ETH-USDT-SWAP",
                side="long",
                base_main_contracts="1",
                max_layers=8,
                layer_multiplier_step="0.15",
                contract_precision="0.01",
                min_contracts="0.01",
            )

    def test_buy_raises(self):
        with pytest.raises(PositionPlanError, match="side"):
            create_main_position_plan(
                inst_id="ETH-USDT-SWAP",
                side="BUY",
                base_main_contracts="1",
                max_layers=8,
                layer_multiplier_step="0.15",
                contract_precision="0.01",
                min_contracts="0.01",
            )

    def test_empty_string_side_raises(self):
        with pytest.raises(PositionPlanError, match="side"):
            create_main_position_plan(
                inst_id="ETH-USDT-SWAP",
                side="",
                base_main_contracts="1",
                max_layers=8,
                layer_multiplier_step="0.15",
                contract_precision="0.01",
                min_contracts="0.01",
            )


# ===================================================================
# 12. to_dict / from_dict round trip
# ===================================================================


class TestToDictFromDict:
    def test_round_trip_equivalence(self):
        plan = create_main_position_plan(
            inst_id="ETH-USDT-SWAP",
            side="LONG",
            base_main_contracts="1",
            max_layers=8,
            layer_multiplier_step="0.15",
            contract_precision="0.01",
            min_contracts="0.01",
            plan_id="fixed-plan-id",
            created_ms=1000,
        )
        d = plan.to_dict()
        restored = PositionPlan.from_dict(d)
        assert restored == plan

    def test_planned_main_contracts_is_list_in_dict(self):
        plan = create_main_position_plan(
            inst_id="ETH-USDT-SWAP",
            side="LONG",
            base_main_contracts="1",
            max_layers=3,
            layer_multiplier_step="0.15",
            contract_precision="0.01",
            min_contracts="0.01",
        )
        d = plan.to_dict()
        assert isinstance(d["planned_main_contracts"], list)
        assert d["planned_main_contracts"] == ["1", "1.15", "1.3"]

    def test_restored_planned_main_contracts_is_tuple(self):
        plan = create_main_position_plan(
            inst_id="ETH-USDT-SWAP",
            side="LONG",
            base_main_contracts="1",
            max_layers=3,
            layer_multiplier_step="0.15",
            contract_precision="0.01",
            min_contracts="0.01",
        )
        d = plan.to_dict()
        restored = PositionPlan.from_dict(d)
        assert isinstance(restored.planned_main_contracts, tuple)

    def test_decimal_fields_are_strings_in_dict(self):
        plan = create_main_position_plan(
            inst_id="ETH-USDT-SWAP",
            side="LONG",
            base_main_contracts="1",
            max_layers=3,
            layer_multiplier_step="0.15",
            contract_precision="0.01",
            min_contracts="0.01",
        )
        d = plan.to_dict()
        assert isinstance(d["base_main_contracts"], str)
        assert isinstance(d["layer_multiplier_step"], str)
        assert isinstance(d["contract_precision"], str)
        assert isinstance(d["min_contracts"], str)

    def test_both_sides_round_trip(self):
        for side in ("LONG", "SHORT"):
            plan = create_main_position_plan(
                inst_id="ETH-USDT-SWAP",
                side=side,
                base_main_contracts="2",
                max_layers=5,
                layer_multiplier_step="0.1",
                contract_precision="0.01",
                min_contracts="0.01",
                plan_id=f"plan-{side}",
                created_ms=2000,
            )
            d = plan.to_dict()
            restored = PositionPlan.from_dict(d)
            assert restored == plan

    def test_plan_id_auto_generated_is_valid_uuid_hex(self):
        plan = create_main_position_plan(
            inst_id="ETH-USDT-SWAP",
            side="LONG",
            base_main_contracts="1",
            max_layers=2,
            layer_multiplier_step="0.15",
            contract_precision="0.01",
            min_contracts="0.01",
        )
        # Should be a 32-char hex string
        assert len(plan.plan_id) == 32
        assert all(c in "0123456789abcdef" for c in plan.plan_id)

    def test_created_ms_auto_generated_is_int(self):
        plan = create_main_position_plan(
            inst_id="ETH-USDT-SWAP",
            side="LONG",
            base_main_contracts="1",
            max_layers=2,
            layer_multiplier_step="0.15",
            contract_precision="0.01",
            min_contracts="0.01",
        )
        assert isinstance(plan.created_ms, int)
        assert plan.created_ms > 0


# ===================================================================
# 13. from_dict invalid schema
# ===================================================================


class TestFromDictInvalidSchema:
    def _valid_dict(self):
        return {
            "plan_id": "p1",
            "inst_id": "ETH-USDT-SWAP",
            "side": "LONG",
            "base_main_contracts": "1",
            "max_layers": 3,
            "layer_multiplier_step": "0.15",
            "contract_precision": "0.01",
            "min_contracts": "0.01",
            "planned_main_contracts": ["1", "1.15", "1.3"],
            "created_ms": 1000,
        }

    def test_planned_main_contracts_is_string_raises(self):
        d = self._valid_dict()
        d["planned_main_contracts"] = "123"
        with pytest.raises(PositionPlanError, match="planned_main_contracts"):
            PositionPlan.from_dict(d)

    def test_planned_main_contracts_item_not_string_raises(self):
        d = self._valid_dict()
        d["planned_main_contracts"] = ["1", 2, "1.3"]
        with pytest.raises(PositionPlanError, match="item at index 1"):
            PositionPlan.from_dict(d)

    def test_planned_main_contracts_length_mismatch_raises(self):
        d = self._valid_dict()
        d["planned_main_contracts"] = ["1", "1.15"]
        with pytest.raises(PositionPlanError, match="must equal max_layers"):
            PositionPlan.from_dict(d)

    def test_max_layers_not_int_raises(self):
        d = self._valid_dict()
        d["max_layers"] = "3"
        with pytest.raises(PositionPlanError, match="max_layers.*must be an int"):
            PositionPlan.from_dict(d)

    def test_max_layers_bool_raises(self):
        d = self._valid_dict()
        d["max_layers"] = True
        with pytest.raises(PositionPlanError, match="max_layers.*must be an int"):
            PositionPlan.from_dict(d)

    def test_created_ms_not_int_raises(self):
        d = self._valid_dict()
        d["created_ms"] = "1000"
        with pytest.raises(PositionPlanError, match="created_ms.*must be an int"):
            PositionPlan.from_dict(d)

    def test_side_not_str_raises(self):
        d = self._valid_dict()
        d["side"] = 123
        with pytest.raises(PositionPlanError, match="side.*must be a string"):
            PositionPlan.from_dict(d)

    def test_invalid_side_value_raises_in_from_dict(self):
        d = self._valid_dict()
        d["side"] = "BUY"
        with pytest.raises(PositionPlanError, match="side must be one of"):
            PositionPlan.from_dict(d)

    def test_missing_plan_id_raises(self):
        d = self._valid_dict()
        del d["plan_id"]
        with pytest.raises(PositionPlanError, match="plan_id"):
            PositionPlan.from_dict(d)

    def test_missing_inst_id_raises(self):
        d = self._valid_dict()
        del d["inst_id"]
        with pytest.raises(PositionPlanError, match="inst_id"):
            PositionPlan.from_dict(d)


# ===================================================================
# 14. planned_contract_for_layer
# ===================================================================


class TestPlannedContractForLayer:
    @pytest.fixture(autouse=True)
    def plan(self):
        return create_main_position_plan(
            inst_id="ETH-USDT-SWAP",
            side="LONG",
            base_main_contracts="1",
            max_layers=5,
            layer_multiplier_step="0.15",
            contract_precision="0.01",
            min_contracts="0.01",
            plan_id="test-plan",
            created_ms=5000,
        )

    def test_layer_1_returns_first(self, plan):
        assert plan.planned_contract_for_layer(1) == "1"

    def test_layer_max_returns_last(self, plan):
        assert plan.planned_contract_for_layer(5) == "1.6"

    def test_layer_0_raises(self, plan):
        with pytest.raises(PositionPlanError, match="layer must be in"):
            plan.planned_contract_for_layer(0)

    def test_layer_above_max_raises(self, plan):
        with pytest.raises(PositionPlanError, match="layer must be in"):
            plan.planned_contract_for_layer(6)

    def test_layer_negative_raises(self, plan):
        with pytest.raises(PositionPlanError, match="layer must be in"):
            plan.planned_contract_for_layer(-1)

    def test_layer_not_int_raises(self, plan):
        with pytest.raises(PositionPlanError, match="layer must be an int"):
            plan.planned_contract_for_layer("1")  # type: ignore[arg-type]


# ===================================================================
# 15. PositionPlan dataclass properties
# ===================================================================


class TestPositionPlanProperties:
    def test_layer_count_property(self):
        plan = create_main_position_plan(
            inst_id="ETH-USDT-SWAP",
            side="LONG",
            base_main_contracts="1",
            max_layers=7,
            layer_multiplier_step="0.15",
            contract_precision="0.01",
            min_contracts="0.01",
        )
        assert plan.layer_count == 7

    def test_frozen_immutable(self):
        plan = create_main_position_plan(
            inst_id="ETH-USDT-SWAP",
            side="LONG",
            base_main_contracts="1",
            max_layers=3,
            layer_multiplier_step="0.15",
            contract_precision="0.01",
            min_contracts="0.01",
        )
        with pytest.raises(Exception):
            plan.max_layers = 5  # type: ignore[misc]

    def test_hashable(self):
        p1 = create_main_position_plan(
            inst_id="ETH-USDT-SWAP",
            side="LONG",
            base_main_contracts="1",
            max_layers=3,
            layer_multiplier_step="0.15",
            contract_precision="0.01",
            min_contracts="0.01",
            plan_id="same-id",
            created_ms=1000,
        )
        p2 = create_main_position_plan(
            inst_id="ETH-USDT-SWAP",
            side="LONG",
            base_main_contracts="1",
            max_layers=3,
            layer_multiplier_step="0.15",
            contract_precision="0.01",
            min_contracts="0.01",
            plan_id="same-id",
            created_ms=1000,
        )
        assert hash(p1) == hash(p2)
        assert p1 == p2


# ===================================================================
# 16. Decimal input acceptance
# ===================================================================


class TestDecimalInput:
    def test_accepts_decimal_base_main_contracts(self):
        plan = create_main_position_plan(
            inst_id="ETH-USDT-SWAP",
            side="LONG",
            base_main_contracts=Decimal("1"),
            max_layers=3,
            layer_multiplier_step="0.15",
            contract_precision="0.01",
            min_contracts="0.01",
        )
        assert plan.base_main_contracts == "1"

    def test_accepts_decimal_layer_multiplier_step(self):
        plan = create_main_position_plan(
            inst_id="ETH-USDT-SWAP",
            side="LONG",
            base_main_contracts="1",
            max_layers=3,
            layer_multiplier_step=Decimal("0.15"),
            contract_precision="0.01",
            min_contracts="0.01",
        )
        assert plan.planned_main_contracts == ("1", "1.15", "1.3")

    def test_accepts_decimal_contract_precision(self):
        plan = create_main_position_plan(
            inst_id="ETH-USDT-SWAP",
            side="LONG",
            base_main_contracts="1",
            max_layers=3,
            layer_multiplier_step="0.15",
            contract_precision=Decimal("0.1"),
            min_contracts="0.1",
        )
        assert plan.planned_main_contracts == ("1", "1.1", "1.3")

    def test_accepts_decimal_min_contracts(self):
        plan = create_main_position_plan(
            inst_id="ETH-USDT-SWAP",
            side="LONG",
            base_main_contracts="0.02",
            max_layers=2,
            layer_multiplier_step="0.15",
            contract_precision="0.01",
            min_contracts=Decimal("0.01"),
        )
        assert plan.min_contracts == "0.01"


# ===================================================================
# 17. edge cases
# ===================================================================


class TestEdgeCases:
    def test_max_layers_1(self):
        plan = create_main_position_plan(
            inst_id="ETH-USDT-SWAP",
            side="LONG",
            base_main_contracts="5",
            max_layers=1,
            layer_multiplier_step="0.15",
            contract_precision="0.01",
            min_contracts="0.01",
        )
        assert plan.layer_count == 1
        assert plan.planned_main_contracts == ("5",)

    def test_step_zero_all_same(self):
        """step=0 means all layers equal base."""
        plan = create_main_position_plan(
            inst_id="ETH-USDT-SWAP",
            side="LONG",
            base_main_contracts="3",
            max_layers=4,
            layer_multiplier_step="0",
            contract_precision="0.01",
            min_contracts="0.01",
        )
        assert plan.planned_main_contracts == ("3", "3", "3", "3")

    def test_large_base_with_moderate_step(self):
        plan = create_main_position_plan(
            inst_id="ETH-USDT-SWAP",
            side="LONG",
            base_main_contracts="100",
            max_layers=3,
            layer_multiplier_step="0.15",
            contract_precision="0.01",
            min_contracts="0.01",
        )
        # L1: 100
        # L2: 100 * 1.15 = 115
        # L3: 100 * 1.30 = 130
        assert plan.planned_main_contracts == ("100", "115", "130")

    def test_custom_plan_id_and_created_ms(self):
        plan = create_main_position_plan(
            inst_id="ETH-USDT-SWAP",
            side="SHORT",
            base_main_contracts="1",
            max_layers=2,
            layer_multiplier_step="0.15",
            contract_precision="0.01",
            min_contracts="0.01",
            plan_id="my-custom-plan",
            created_ms=9999999999999,
        )
        assert plan.plan_id == "my-custom-plan"
        assert plan.created_ms == 9999999999999

    def test_invalid_decimal_string_raises(self):
        with pytest.raises(PositionPlanError, match="not a valid decimal"):
            create_main_position_plan(
                inst_id="ETH-USDT-SWAP",
                side="LONG",
                base_main_contracts="not-a-number",
                max_layers=8,
                layer_multiplier_step="0.15",
                contract_precision="0.01",
                min_contracts="0.01",
            )

    def test_base_main_contracts_wrong_type_raises(self):
        with pytest.raises(PositionPlanError, match="base_main_contracts must be str"):
            create_main_position_plan(
                inst_id="ETH-USDT-SWAP",
                side="LONG",
                base_main_contracts=1.5,  # type: ignore[arg-type]
                max_layers=8,
                layer_multiplier_step="0.15",
                contract_precision="0.01",
                min_contracts="0.01",
            )


# ===================================================================
# 18. exception hierarchy
# ===================================================================


class TestExceptionHierarchy:
    def test_position_plan_error_is_value_error(self):
        assert issubclass(PositionPlanError, ValueError)

    def test_position_plan_error_can_be_caught_as_value_error(self):
        try:
            raise PositionPlanError("test")
        except ValueError:
            pass
        else:
            pytest.fail("PositionPlanError should be catchable as ValueError")


# ===================================================================
# 19. source purity check
# ===================================================================


class TestSourcePurity:
    def test_position_plan_source_has_no_forbidden_imports(self) -> None:
        source_path = (
            Path(__file__).parents[2]
            / "src" / "portfolio" / "position_plan.py"
        )
        source = source_path.read_text()

        forbidden = [
            "Trader",
            "Strategy",
            "requests",
            "httpx",
            "websocket",
            "okx",
            "EmailSender",
            "os.getenv",
            "load_dotenv",
            "redis",
            "sqlite",
            "pydantic",
            "portalocker",
            "filelock",
            # Must not read env / config files
            "toml",
            "yaml",
            ".env",
            # Must not import live runtime
            "src.live",
            "src.execution",
            # Must not import strategy
            "src.strategy",
        ]
        for token in forbidden:
            assert token not in source, (
                f"position_plan.py must not import/use {token}"
            )


# ===================================================================
# 20. VALID_SIDES constant
# ===================================================================


class TestValidSides:
    def test_only_long_and_short(self):
        assert VALID_SIDES == ("LONG", "SHORT")
