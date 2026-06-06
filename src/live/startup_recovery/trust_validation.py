from __future__ import annotations

import os
from typing import Any

from src.execution.trader import PositionSnapshot
from src.position_management.sidecar.model import sidecar_open_qty
from src.utils.log import get_logger

logger = get_logger(__name__)


def expected_saved_state_remaining_qty(saved_state: Any) -> tuple[float, str]:  # type: ignore[no-untyped-def]
    """Estimate the current remaining position qty from saved state.

    total_entry_qty is the original entry quantity, which may have been
    reduced by partial TP / Three-Stage TP1 / Middle Runner first close.
    This function computes the *expected current remaining* qty using the
    best available information, prioritized:

      1. Sidecar: core_eth_qty + sidecar_open_qty(sidecar_legs)
      2. core_eth_qty (when > 0, even without sidecar)
      3. position_cost_remaining_qty
      4. Three-Stage deduction from consumed flags
      5. Middle Runner keep_ratio
      6. Fallback: total_entry_qty

    Returns (qty, source_label).
    """
    # ── 1. Sidecar: net position = core + sidecar open ──────────────────
    if bool(getattr(saved_state, "sidecar_enabled_for_position", False)):
        core_qty = float(getattr(saved_state, "core_eth_qty", 0.0) or 0.0)
        sidecar_legs = list(getattr(saved_state, "sidecar_legs", []) or [])
        sc_open = sidecar_open_qty(sidecar_legs)
        expected = core_qty + sc_open
        if expected > 0:
            return expected, "sidecar_core_plus_open"

    # ── 2. core_eth_qty (present even when sidecar is disabled) ────────
    core_qty = float(getattr(saved_state, "core_eth_qty", 0.0) or 0.0)
    if core_qty > 0:
        return core_qty, "core_eth_qty"

    # ── 3. position_cost_remaining_qty ──────────────────────────────────
    cost_remaining = float(getattr(saved_state, "position_cost_remaining_qty", 0.0) or 0.0)
    if cost_remaining > 0:
        return cost_remaining, "position_cost_remaining_qty"

    # ── 4. Three-Stage deduction ────────────────────────────────────────
    total_entry = float(getattr(saved_state, "total_entry_qty", 0.0) or 0.0)
    if (
            total_entry > 0
            and bool(getattr(saved_state, "three_stage_runner_enabled_for_position", False))
    ):
        tp2_consumed = bool(getattr(saved_state, "three_stage_tp2_consumed", False))
        tp1_consumed = bool(getattr(saved_state, "three_stage_tp1_consumed", False))
        runner_ratio = float(getattr(saved_state, "three_stage_runner_ratio", 0.0) or 0.0)
        tp2_ratio = float(getattr(saved_state, "three_stage_tp2_ratio", 0.0) or 0.0)
        if tp2_consumed and runner_ratio > 0:
            return total_entry * runner_ratio, "three_stage_runner"
        if tp1_consumed and (tp2_ratio + runner_ratio) > 0:
            return total_entry * (tp2_ratio + runner_ratio), "three_stage_after_tp1"

    # ── 5. Middle Runner active ─────────────────────────────────────────
    if total_entry > 0 and bool(getattr(saved_state, "middle_runner_active", False)):
        keep_ratio = float(getattr(saved_state, "middle_runner_keep_ratio", 0.0) or 0.0)
        if keep_ratio > 0:
            return total_entry * keep_ratio, "middle_runner_active"

    # ── 6. Fallback: total_entry_qty ────────────────────────────────────
    if total_entry > 0:
        return total_entry, "total_entry_qty"

    return 0.0, "none"


def trusted_startup_saved_state(  # type: ignore[no-untyped-def]
        saved_state: Any,
        startup_position: PositionSnapshot,
        max_avg_diff_pct: float | None = None,
        max_qty_diff_pct: float | None = None,
) -> Any:
    """Return saved_state only when it matches the current OKX position.

    A saved_state is trusted when ALL of these hold:
      1. saved_state is not None
      2. startup_position.has_position is True
      3. saved_state.side == startup_position.side
      4. saved_state.layers > 0
      5. avg_entry_price is within max_avg_diff_pct of OKX avg
      6. expected remaining qty is within max_qty_diff_pct of OKX qty

    The expected remaining qty accounts for partial exits (Three-Stage TP1,
    Middle Runner first close, etc.) via expected_saved_state_remaining_qty().

    Tolerance defaults are read from env vars with fallback values:
      STARTUP_SAVED_STATE_MAX_AVG_DIFF_PCT  → 0.003  (0.3%)
      STARTUP_SAVED_STATE_MAX_QTY_DIFF_PCT  → 0.05   (5%)
    """
    if max_avg_diff_pct is None:
        max_avg_diff_pct = float(os.getenv("STARTUP_SAVED_STATE_MAX_AVG_DIFF_PCT", "0.003"))
    if max_qty_diff_pct is None:
        max_qty_diff_pct = float(os.getenv("STARTUP_SAVED_STATE_MAX_QTY_DIFF_PCT", "0.05"))

    # ── basic identity checks ─────────────────────────────────────────
    if saved_state is None:
        return None
    if not startup_position.has_position:
        return None
    if getattr(saved_state, "side", None) != startup_position.side:
        logger.warning(
            "STARTUP_SAVED_STATE_UNTRUSTED | reason=side_mismatch saved_side=%s okx_side=%s",
            getattr(saved_state, "side", None),
            startup_position.side,
        )
        return None
    if int(getattr(saved_state, "layers", 0) or 0) <= 0:
        logger.warning(
            "STARTUP_SAVED_STATE_UNTRUSTED | reason=layers_zero_or_missing saved_layers=%s",
            getattr(saved_state, "layers", None),
        )
        return None

    # ── avg_entry check ───────────────────────────────────────────────
    saved_avg = float(getattr(saved_state, "avg_entry_price", 0.0) or 0.0)
    pos_avg = float(startup_position.avg_entry_price or 0.0)
    if saved_avg <= 0:
        logger.warning(
            "STARTUP_SAVED_STATE_UNTRUSTED | reason=avg_entry_missing_or_zero saved_avg=%s okx_avg=%.4f",
            getattr(saved_state, "avg_entry_price", None),
            pos_avg,
        )
        return None
    if pos_avg <= 0:
        logger.warning(
            "STARTUP_SAVED_STATE_UNTRUSTED | reason=okx_avg_entry_zero saved_avg=%.4f okx_avg=%.4f",
            saved_avg,
            pos_avg,
        )
        return None
    avg_diff = abs(saved_avg - pos_avg) / pos_avg
    if avg_diff > max_avg_diff_pct:
        logger.warning(
            "STARTUP_SAVED_STATE_UNTRUSTED | reason=avg_entry_mismatch saved_avg=%.4f okx_avg=%.4f diff_pct=%.6f max_diff_pct=%.6f",
            saved_avg,
            pos_avg,
            avg_diff,
            max_avg_diff_pct,
        )
        return None

    # ── size check ────────────────────────────────────────────────────
    expected_qty, qty_source = expected_saved_state_remaining_qty(saved_state)
    pos_qty = float(startup_position.eth_qty or 0.0)

    if expected_qty <= 0:
        logger.warning(
            "STARTUP_SAVED_STATE_UNTRUSTED | reason=qty_missing_or_zero expected_qty=%.8f qty_source=%s saved_total_entry_qty=%s saved_core_eth_qty=%s",
            expected_qty,
            qty_source,
            getattr(saved_state, "total_entry_qty", None),
            getattr(saved_state, "core_eth_qty", None),
        )
        return None
    if pos_qty <= 0:
        logger.warning(
            "STARTUP_SAVED_STATE_UNTRUSTED | reason=okx_qty_zero expected_qty=%.8f qty_source=%s okx_qty=%.8f",
            expected_qty,
            qty_source,
            pos_qty,
        )
        return None

    logger.info(
        "STARTUP_SAVED_STATE_QTY_EXPECTED | source=%s expected_qty=%.8f okx_qty=%.8f total_entry_qty=%s sidecar_enabled=%s",
        qty_source,
        expected_qty,
        pos_qty,
        getattr(saved_state, "total_entry_qty", None),
        bool(getattr(saved_state, "sidecar_enabled_for_position", False)),
    )

    qty_diff = abs(expected_qty - pos_qty) / pos_qty
    if qty_diff > max_qty_diff_pct:
        logger.warning(
            "STARTUP_SAVED_STATE_UNTRUSTED | reason=qty_mismatch expected_qty=%.8f okx_qty=%.8f diff_pct=%.6f max_diff_pct=%.6f qty_source=%s",
            expected_qty,
            pos_qty,
            qty_diff,
            max_qty_diff_pct,
            qty_source,
        )
        return None

    return saved_state
