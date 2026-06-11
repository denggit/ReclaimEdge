# -*- coding: utf-8 -*-
"""
G02: PositionPlan —— 开仓时生成完整 layer 计划。

纯逻辑模块，无 IO / 网络 / 环境变量 / live runtime 依赖。

职责:
  - 根据配置参数生成一轮主仓的完整 layer 张数计划
  - 每层张数固定，按 base * (1 + (layer - 1) * step) 计算
  - 按 contract_precision 向下量化
  - to_dict / from_dict 序列化

不负责:
  - leader/follower 判断
  - permission_max_layers / add_gap / add_freeze
  - can_open / can_add / allow / reject
  - allocator decision
  - 写入 CapitalLedger
  - 订单下单 / OKX 请求 / 邮件发送 / 策略信号判断
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal, InvalidOperation
from typing import Any, Mapping

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PositionPlanError(ValueError):
    """PositionPlan 模块基础异常。"""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_SIDES: tuple[str, str] = ("LONG", "SHORT")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def decimal_to_plain_str(value: Decimal) -> str:
    """Convert a Decimal to a plain string without scientific notation.

    Uses ``format(value.normalize(), 'f')`` to produce stable output like
    ``"1.15"``, ``"0.01"``, ``"1"``, never ``"1.1500000000000001"`` or ``"1E-2"``.
    """
    return format(value.normalize(), "f")


def quantize_contracts_down(value: Decimal, step: Decimal) -> Decimal:
    """Quantize *value* down to the nearest multiple of *step*.

    Always rounds **down** (toward zero for positive, away from zero for
    negative) so that we never exceed the risk budget.

    >>> quantize_contracts_down(Decimal("1.159"), Decimal("0.01"))
    Decimal('1.15')
    >>> quantize_contracts_down(Decimal("0.0115"), Decimal("0.01"))
    Decimal('0.01')
    """
    if step <= 0:
        raise PositionPlanError(f"contract_precision must be > 0, got {step}")
    # Use integer division to floor toward zero.
    units = (value / step).to_integral_value(rounding=ROUND_FLOOR)
    return units * step


def _to_decimal(value: str | Decimal, label: str = "value") -> Decimal:
    """Coerce a str or Decimal to Decimal, raising PositionPlanError on failure."""
    if isinstance(value, Decimal):
        return value
    if not isinstance(value, str):
        raise PositionPlanError(
            f"{label} must be str or Decimal, got {type(value).__name__}: {value!r}"
        )
    try:
        return Decimal(value)
    except InvalidOperation:
        raise PositionPlanError(f"{label} is not a valid decimal string: {value!r}") from None


def _require_positive(value: Decimal, label: str) -> None:
    """Raise PositionPlanError if *value* <= 0."""
    if value <= 0:
        raise PositionPlanError(f"{label} must be > 0, got {value}")


# ---------------------------------------------------------------------------
# DTO
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PositionPlan:
    """一轮主仓的完整 layer 张数计划。

    All decimal values are stored as plain strings so that JSON round-trips
    are lossless and never contain floating-point artefacts.
    """

    plan_id: str
    inst_id: str
    side: str
    base_main_contracts: str
    max_layers: int
    layer_multiplier_step: str
    contract_precision: str
    min_contracts: str
    planned_main_contracts: tuple[str, ...]
    created_ms: int

    # -- helpers ---------------------------------------------------------------

    @property
    def layer_count(self) -> int:
        """Return the number of planned layers (equal to max_layers)."""
        return len(self.planned_main_contracts)

    def planned_contract_for_layer(self, layer: int) -> str:
        """Return the planned contract count for *layer* (1-indexed).

        Raises ``PositionPlanError`` if *layer* is out of range.
        """
        if not isinstance(layer, int):
            raise PositionPlanError(
                f"layer must be an int, got {type(layer).__name__}: {layer!r}"
            )
        if layer < 1 or layer > self.max_layers:
            raise PositionPlanError(
                f"layer must be in [1, {self.max_layers}], got {layer}"
            )
        return self.planned_main_contracts[layer - 1]

    # -- serialization ---------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation.

        All decimal fields are plain strings.  ``planned_main_contracts`` is a
        ``list[str]``.
        """
        return {
            "plan_id": self.plan_id,
            "inst_id": self.inst_id,
            "side": self.side,
            "base_main_contracts": self.base_main_contracts,
            "max_layers": self.max_layers,
            "layer_multiplier_step": self.layer_multiplier_step,
            "contract_precision": self.contract_precision,
            "min_contracts": self.min_contracts,
            "planned_main_contracts": list(self.planned_main_contracts),
            "created_ms": self.created_ms,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PositionPlan:
        """Reconstruct a PositionPlan from a dict produced by :meth:`to_dict`.

        Strictly validates all field types and constraints.
        """
        # -- required string fields --------------------------------------------
        plan_id = _require_str(data, "plan_id")
        inst_id = _require_str(data, "inst_id")
        side = _require_str(data, "side")
        base_main_contracts = _require_str(data, "base_main_contracts")
        layer_multiplier_step = _require_str(data, "layer_multiplier_step")
        contract_precision = _require_str(data, "contract_precision")
        min_contracts = _require_str(data, "min_contracts")

        # -- required int fields -----------------------------------------------
        max_layers = _require_int(data, "max_layers")
        created_ms = _require_int(data, "created_ms")

        # -- planned_main_contracts --------------------------------------------
        planned_main_contracts = _require_str_tuple(data, "planned_main_contracts")
        if len(planned_main_contracts) != max_layers:
            raise PositionPlanError(
                f"planned_main_contracts length ({len(planned_main_contracts)}) "
                f"must equal max_layers ({max_layers})"
            )

        # -- side validation ---------------------------------------------------
        if side not in VALID_SIDES:
            raise PositionPlanError(
                f"side must be one of {VALID_SIDES}, got {side!r}"
            )

        return cls(
            plan_id=plan_id,
            inst_id=inst_id,
            side=side,
            base_main_contracts=base_main_contracts,
            max_layers=max_layers,
            layer_multiplier_step=layer_multiplier_step,
            contract_precision=contract_precision,
            min_contracts=min_contracts,
            planned_main_contracts=planned_main_contracts,
            created_ms=created_ms,
        )


# ---------------------------------------------------------------------------
# Internal validators for from_dict
# ---------------------------------------------------------------------------


def _require_str(d: Mapping[str, Any], key: str) -> str:
    val = d.get(key)
    if not isinstance(val, str):
        raise PositionPlanError(
            f"'{key}' must be a string, got {type(val).__name__}: {val!r}"
        )
    return val


def _require_int(d: Mapping[str, Any], key: str) -> int:
    val = d.get(key)
    # Reject bool because bool is a subclass of int.
    if isinstance(val, bool) or not isinstance(val, int):
        raise PositionPlanError(
            f"'{key}' must be an int, got {type(val).__name__}: {val!r}"
        )
    return val


def _require_str_tuple(d: Mapping[str, Any], key: str) -> tuple[str, ...]:
    val = d.get(key)
    if not isinstance(val, (list, tuple)):
        raise PositionPlanError(
            f"'{key}' must be a list or tuple, "
            f"got {type(val).__name__}: {val!r}"
        )
    for i, item in enumerate(val):
        if not isinstance(item, str):
            raise PositionPlanError(
                f"'{key}' item at index {i} must be a string, "
                f"got {type(item).__name__}: {item!r}"
            )
    return tuple(val)


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------


def create_main_position_plan(
    *,
    inst_id: str,
    side: str,
    base_main_contracts: str | Decimal,
    max_layers: int,
    layer_multiplier_step: str | Decimal,
    contract_precision: str | Decimal,
    min_contracts: str | Decimal,
    plan_id: str | None = None,
    created_ms: int | None = None,
) -> PositionPlan:
    """Generate a full PositionPlan for a new main position.

    Parameters
    ----------
    inst_id:
        Instrument ID, e.g. ``"ETH-USDT-SWAP"``.
    side:
        ``"LONG"`` or ``"SHORT"``.
    base_main_contracts:
        Number of contracts for layer 1 (the base layer).  Must be >= min_contracts.
    max_layers:
        Total number of layers.  Must be an int >= 1.  The plan always generates
        exactly *max_layers* layers — it never hardcodes a fixed number.
    layer_multiplier_step:
        Per-layer multiplier increment, e.g. ``"0.15"``.  Must be >= 0.
    contract_precision:
        The lot-size / tick-size the exchange uses, e.g. ``"0.1"`` (ETH) or
        ``"0.01"`` (BTC).  Every layer's contract count is quantized **down**
        to this precision.
    min_contracts:
        Minimum allowed contracts per layer.  Must be > 0.
    plan_id:
        Optional stable plan identifier.  Auto-generated (uuid4 hex) when omitted.
    created_ms:
        Optional creation timestamp in milliseconds.  Auto-generated when omitted.

    Returns
    -------
    PositionPlan
        A frozen dataclass with the full layer schedule.

    Raises
    ------
    PositionPlanError
        If any input fails validation.
    """
    # -- coerce decimals -------------------------------------------------------
    _base = _to_decimal(base_main_contracts, "base_main_contracts")
    _step = _to_decimal(layer_multiplier_step, "layer_multiplier_step")
    _precision = _to_decimal(contract_precision, "contract_precision")
    _min = _to_decimal(min_contracts, "min_contracts")

    # -- validate side ---------------------------------------------------------
    if not isinstance(side, str) or side not in VALID_SIDES:
        raise PositionPlanError(
            f"side must be one of {VALID_SIDES}, got {side!r}"
        )

    # -- validate max_layers ---------------------------------------------------
    if not isinstance(max_layers, int) or isinstance(max_layers, bool):
        raise PositionPlanError(
            f"max_layers must be an int >= 1, got {type(max_layers).__name__}: {max_layers!r}"
        )
    if max_layers < 1:
        raise PositionPlanError(f"max_layers must be >= 1, got {max_layers}")

    # -- validate numeric inputs -----------------------------------------------
    _require_positive(_base, "base_main_contracts")
    if _step < 0:
        raise PositionPlanError(f"layer_multiplier_step must be >= 0, got {_step}")
    _require_positive(_precision, "contract_precision")
    _require_positive(_min, "min_contracts")

    if _base < _min:
        raise PositionPlanError(
            f"base_main_contracts ({_base}) must be >= min_contracts ({_min})"
        )

    # -- generate layers -------------------------------------------------------
    planned: list[str] = []
    one = Decimal("1")

    for layer in range(1, max_layers + 1):
        multiplier = one + (layer - 1) * _step
        raw = _base * multiplier
        quantized = quantize_contracts_down(raw, _precision)
        if quantized < _min:
            raise PositionPlanError(
                f"Layer {layer}: quantized contracts ({quantized}) "
                f"is below min_contracts ({_min}). "
                f"raw={raw}, precision={_precision}"
            )
        planned.append(decimal_to_plain_str(quantized))

    # -- auto-generate id / timestamp ------------------------------------------
    _plan_id = plan_id if plan_id is not None else uuid.uuid4().hex
    _created_ms = created_ms if created_ms is not None else int(time.time() * 1000)

    return PositionPlan(
        plan_id=_plan_id,
        inst_id=inst_id,
        side=side,
        base_main_contracts=decimal_to_plain_str(_base),
        max_layers=max_layers,
        layer_multiplier_step=decimal_to_plain_str(_step),
        contract_precision=decimal_to_plain_str(_precision),
        min_contracts=decimal_to_plain_str(_min),
        planned_main_contracts=tuple(planned),
        created_ms=_created_ms,
    )
