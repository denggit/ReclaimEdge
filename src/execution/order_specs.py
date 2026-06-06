from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Any, Literal

PositionSide = Literal["LONG", "SHORT"]


@dataclass(frozen=True)
class TakeProfitOrderSpec:
    label: str
    contracts: Decimal
    price: float


@dataclass(frozen=True)
class TakeProfitSpecsDecision:
    specs: tuple[TakeProfitOrderSpec, ...]
    fallback_reason: str | None = None
    fallback_context: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Side helpers
# ---------------------------------------------------------------------------


def pos_side_for_mode(*, side: PositionSide, pos_side_mode: str) -> str | None:
    if pos_side_mode != "long_short":
        return None
    return "long" if side == "LONG" else "short"


def open_order_side(*, side: PositionSide) -> str:
    return "buy" if side == "LONG" else "sell"


def close_order_side(*, side: PositionSide) -> str:
    return "sell" if side == "LONG" else "buy"


def maybe_add_pos_side(
        body: dict[str, Any],
        *,
        side: PositionSide,
        pos_side_mode: str,
) -> dict[str, Any]:
    body = dict(body)
    ps = pos_side_for_mode(side=side, pos_side_mode=pos_side_mode)
    if ps is not None:
        body["posSide"] = ps
    return body


# ---------------------------------------------------------------------------
# Rounding
# ---------------------------------------------------------------------------


def round_contracts_down(
        *,
        contracts: Decimal,
        contract_precision: Decimal,
) -> Decimal:
    lots = (contracts / contract_precision).to_integral_value(rounding=ROUND_DOWN)
    return lots * contract_precision


# ---------------------------------------------------------------------------
# OKX order body builders
# ---------------------------------------------------------------------------


def build_market_entry_order_body(
        *,
        inst_id: str,
        td_mode: str,
        side: PositionSide,
        contracts_text: str,
        pos_side_mode: str,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "instId": inst_id,
        "tdMode": td_mode,
        "side": open_order_side(side=side),
        "ordType": "market",
        "sz": contracts_text,
    }
    return maybe_add_pos_side(body, side=side, pos_side_mode=pos_side_mode)


def build_reduce_only_market_order_body(
        *,
        inst_id: str,
        td_mode: str,
        side: PositionSide,
        contracts_text: str,
        pos_side_mode: str,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "instId": inst_id,
        "tdMode": td_mode,
        "side": close_order_side(side=side),
        "ordType": "market",
        "sz": contracts_text,
        "reduceOnly": "true",
    }
    return maybe_add_pos_side(body, side=side, pos_side_mode=pos_side_mode)


def build_reduce_only_tp_order_body(
        *,
        inst_id: str,
        td_mode: str,
        side: PositionSide,
        contracts_text: str,
        price_text: str,
        pos_side_mode: str,
        client_order_id: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "instId": inst_id,
        "tdMode": td_mode,
        "side": close_order_side(side=side),
        "ordType": "limit",
        "px": price_text,
        "sz": contracts_text,
        "reduceOnly": "true",
    }
    body = maybe_add_pos_side(body, side=side, pos_side_mode=pos_side_mode)
    if client_order_id:
        body["clOrdId"] = client_order_id
    return body


def build_conditional_protective_sl_algo_body(
        *,
        inst_id: str,
        td_mode: str,
        side: PositionSide,
        contracts_text: str,
        stop_price_text: str,
        pos_side_mode: str,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "instId": inst_id,
        "tdMode": td_mode,
        "side": close_order_side(side=side),
        "ordType": "conditional",
        "sz": contracts_text,
        "slTriggerPx": stop_price_text,
        "slOrdPx": "-1",
        "slTriggerPxType": "last",
        "reduceOnly": "true",
    }
    return maybe_add_pos_side(body, side=side, pos_side_mode=pos_side_mode)


def build_cancel_order_body(
        *,
        inst_id: str,
        order_id: str,
) -> dict[str, Any]:
    return {"instId": inst_id, "ordId": order_id}


def build_cancel_algo_body(
        *,
        inst_id: str,
        algo_id: str,
) -> list[dict[str, Any]]:
    return [{"instId": inst_id, "algoId": algo_id}]


def build_set_leverage_bodies(
        *,
        inst_id: str,
        td_mode: str,
        leverage: str,
        pos_side_mode: str,
) -> tuple[dict[str, Any], ...]:
    base: dict[str, Any] = {"instId": inst_id, "lever": str(leverage), "mgnMode": td_mode}
    if pos_side_mode == "long_short":
        long_body = dict(base)
        long_body["posSide"] = "long"
        short_body = dict(base)
        short_body["posSide"] = "short"
        return (long_body, short_body)
    return (base,)


# ---------------------------------------------------------------------------
# TP spec calculation (pure)
# ---------------------------------------------------------------------------


def build_take_profit_order_specs(
        *,
        position_contracts: Decimal,
        min_contracts: Decimal,
        contract_precision: Decimal,
        tp_plan: str,
        final_tp_price: float,
        partial_tp_price: float | None,
        partial_tp_ratio: Decimal,
        partial_tp_consumed: bool,
        middle_runner_active: bool,
        three_stage_tp1_price: float | None,
        three_stage_tp2_price: float | None,
        three_stage_tp1_ratio: Decimal,
        three_stage_tp2_ratio: Decimal,
        three_stage_tp1_consumed: bool,
        three_stage_tp2_consumed: bool,
        three_stage_runner_ratio: Decimal,
) -> TakeProfitSpecsDecision:
    _rnd = lambda c: round_contracts_down(contracts=c, contract_precision=contract_precision)

    # ── Three-Stage branch ──
    if tp_plan == "THREE_STAGE_RUNNER":
        # Case A: after TP1 consumed, TP2 still pending
        if three_stage_tp1_consumed and not three_stage_tp2_consumed:
            remaining_ratio = three_stage_tp2_ratio + three_stage_runner_ratio
            if three_stage_tp2_price is None or three_stage_tp2_ratio <= 0 or remaining_ratio <= 0:
                return TakeProfitSpecsDecision(
                    specs=(TakeProfitOrderSpec(label="final", contracts=position_contracts, price=final_tp_price),),
                    fallback_reason="THREE_STAGE_TP2_AFTER_TP1_INVALID_RATIOS",
                    fallback_context={
                        "total_contracts": position_contracts,
                        "tp2_ratio": three_stage_tp2_ratio,
                        "runner_ratio": three_stage_runner_ratio,
                        "tp2_price": three_stage_tp2_price,
                    },
                )
            tp2_contracts = _rnd(position_contracts * three_stage_tp2_ratio / remaining_ratio)
            runner_contracts = position_contracts - tp2_contracts
            if tp2_contracts < min_contracts:
                return TakeProfitSpecsDecision(
                    specs=(TakeProfitOrderSpec(label="final", contracts=position_contracts, price=final_tp_price),),
                    fallback_reason="THREE_STAGE_TP2_AFTER_TP1_TP2_TOO_SMALL",
                    fallback_context={
                        "total_contracts": position_contracts,
                        "tp2_contracts": tp2_contracts,
                        "runner_contracts": runner_contracts,
                        "min_contracts": min_contracts,
                    },
                )
            if runner_contracts < min_contracts:
                return TakeProfitSpecsDecision(
                    specs=(TakeProfitOrderSpec(label="tp2_outer", contracts=position_contracts,
                                               price=float(three_stage_tp2_price)),),
                    fallback_reason="THREE_STAGE_TP2_AFTER_TP1_RUNNER_TOO_SMALL",
                    fallback_context={
                        "total_contracts": position_contracts,
                        "tp2_contracts": tp2_contracts,
                        "runner_contracts": runner_contracts,
                        "min_contracts": min_contracts,
                    },
                )
            return TakeProfitSpecsDecision(
                specs=(
                TakeProfitOrderSpec(label="tp2_outer", contracts=tp2_contracts, price=float(three_stage_tp2_price)),),
            )

        # Case B: normal three-stage (fresh)
        if three_stage_tp1_price is None or three_stage_tp2_price is None or three_stage_tp1_ratio <= 0 or three_stage_tp2_ratio <= 0:
            return TakeProfitSpecsDecision(
                specs=(TakeProfitOrderSpec(label="final", contracts=position_contracts, price=final_tp_price),),
            )
        tp1_contracts = _rnd(position_contracts * three_stage_tp1_ratio)
        tp2_contracts = _rnd(position_contracts * three_stage_tp2_ratio)
        runner_contracts = position_contracts - tp1_contracts - tp2_contracts
        if tp1_contracts < min_contracts or tp2_contracts < min_contracts or runner_contracts < min_contracts:
            return TakeProfitSpecsDecision(
                specs=(TakeProfitOrderSpec(label="final", contracts=position_contracts, price=final_tp_price),),
                fallback_reason="THREE_STAGE_TP_FALLBACK_SINGLE_SIZE_TOO_SMALL",
                fallback_context={
                    "total_contracts": position_contracts,
                    "tp1_contracts": tp1_contracts,
                    "tp2_contracts": tp2_contracts,
                    "runner_contracts": runner_contracts,
                    "min_contracts": min_contracts,
                },
            )
        return TakeProfitSpecsDecision(
            specs=(
                TakeProfitOrderSpec(label="tp1_middle", contracts=tp1_contracts, price=float(three_stage_tp1_price)),
                TakeProfitOrderSpec(label="tp2_outer", contracts=tp2_contracts, price=float(three_stage_tp2_price)),
            ),
        )

    # ── Non-Three-Stage branch ──
    if partial_tp_consumed or (tp_plan == "MIDDLE_RUNNER" and middle_runner_active):
        return TakeProfitSpecsDecision(
            specs=(TakeProfitOrderSpec(label="final", contracts=position_contracts, price=final_tp_price),),
        )
    if tp_plan not in {"SPLIT_PARTIAL_FINAL", "SPLIT_50_50",
                       "MIDDLE_RUNNER"} or partial_tp_price is None or partial_tp_ratio <= 0 or partial_tp_ratio >= 1:
        return TakeProfitSpecsDecision(
            specs=(TakeProfitOrderSpec(label="final", contracts=position_contracts, price=final_tp_price),),
        )

    partial_contracts = _rnd(position_contracts * partial_tp_ratio)
    final_contracts = position_contracts - partial_contracts
    if partial_contracts < min_contracts or final_contracts < min_contracts:
        return TakeProfitSpecsDecision(
            specs=(TakeProfitOrderSpec(label="final", contracts=position_contracts, price=final_tp_price),),
            fallback_reason="SPLIT_TP_FALLBACK_SINGLE_SIZE_TOO_SMALL",
            fallback_context={
                "total_contracts": position_contracts,
                "partial_contracts": partial_contracts,
                "final_contracts": final_contracts,
                "min_contracts": min_contracts,
            },
        )
    if tp_plan == "MIDDLE_RUNNER":
        return TakeProfitSpecsDecision(
            specs=(
                TakeProfitOrderSpec(label="middle", contracts=partial_contracts, price=float(partial_tp_price)),
                TakeProfitOrderSpec(label="runner", contracts=final_contracts, price=final_tp_price),
            ),
        )
    return TakeProfitSpecsDecision(
        specs=(
            TakeProfitOrderSpec(label="partial", contracts=partial_contracts, price=float(partial_tp_price)),
            TakeProfitOrderSpec(label="final", contracts=final_contracts, price=final_tp_price),
        ),
    )


# ---------------------------------------------------------------------------
# Trend Runner SL contract sizing (pure)
# ---------------------------------------------------------------------------


def trend_runner_sl_contracts(
        *,
        net_contracts_for_sl: Decimal,
        runner_ratio: Decimal,
        min_contracts: Decimal,
        contract_precision: Decimal,
        trend_runner_active: bool,
) -> Decimal:
    if trend_runner_active:
        return net_contracts_for_sl
    if runner_ratio <= 0 or runner_ratio >= 1:
        return net_contracts_for_sl
    contracts = round_contracts_down(contracts=net_contracts_for_sl * runner_ratio,
                                     contract_precision=contract_precision)
    if contracts < min_contracts:
        return net_contracts_for_sl
    return contracts
