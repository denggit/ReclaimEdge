from __future__ import annotations

import base64
import asyncio
import datetime
import hmac
import json
import math
import os
from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_DOWN
from typing import Any, Optional

import aiohttp

from config.env_loader import OKX_CONFIG
from src.strategies.boll_cvd_reclaim_strategy import PositionSide, TradeIntent
from src.utils.log import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class LiveTradeResult:
    ok: bool
    action: str
    order_id: Optional[str]
    tp_order_id: Optional[str]
    contracts: str
    tp_price: str
    message: str
    entry_filled: bool = False
    tp_ok: bool = False
    tp_order_ids: tuple[str, ...] = ()
    protective_sl_order_id: Optional[str] = None
    protective_sl_price: str = ""
    protective_sl_ok: bool = False
    contracts_before: str = ""
    contracts_reduced: str = ""
    contracts_after: str = ""
    near_tp_exit_all: bool = False
    reduce_filled: bool = False


@dataclass(frozen=True)
class PositionSnapshot:
    side: Optional[PositionSide]
    contracts: Decimal
    avg_entry_price: float
    eth_qty: float
    raw_pos: Decimal

    @property
    def has_position(self) -> bool:
        return self.side is not None and self.contracts > 0


class Trader:
    """Simple OKX live trader for ReclaimEdge.

    This module is intentionally small. It only supports the current strategy needs:
    - verify account balance cap
    - set leverage
    - market open long/short
    - place reduce-only take-profit limit order at BOLL target
    - replace take-profit when needed
    - recover existing ETH-USDT-SWAP position on restart
    """

    def __init__(self) -> None:
        self.symbol = os.getenv("OKX_INST_ID", "ETH-USDT-SWAP")
        if self.symbol != "ETH-USDT-SWAP":
            raise RuntimeError("Live trader only supports ETH-USDT-SWAP for now.")

        self.base_url = os.getenv("OKX_BASE_URL", "https://www.okx.com")
        self.td_mode = os.getenv("OKX_TD_MODE", "isolated")
        self.leverage = os.getenv("LEVERAGE", "50")
        self.pos_side_mode = os.getenv("OKX_POS_SIDE_MODE", "net")
        self.live_trading = os.getenv("LIVE_TRADING", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
        self.max_live_equity_usdt = float(os.getenv("MAX_LIVE_EQUITY_USDT", "30"))
        self.contract_multiplier = Decimal("0.1")
        self.contract_precision = Decimal("0.01")
        self.min_contracts = Decimal("0.01")

        self.api_key = OKX_CONFIG.get("api_key", "")
        self.secret_key = OKX_CONFIG.get("secret_key", "")
        self.passphrase = OKX_CONFIG.get("passphrase", "")

        self.tp_order_id: str | None = None
        self.near_tp_protective_sl_order_id: str | None = None
        self.middle_runner_protective_sl_order_id: str | None = None
        self.trend_runner_sl_order_id: str | None = None
        self.position_contracts = Decimal("0")
        self.account_equity_usdt: float = 0.0
        self._session: aiohttp.ClientSession | None = None
        self._timeout_seconds = float(os.getenv("OKX_PRIVATE_REST_TIMEOUT_SECONDS", "10"))

        if not self.api_key or not self.secret_key or not self.passphrase:
            raise ValueError("OKX API config is incomplete. Check OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHASE.")
        if not self.live_trading:
            raise RuntimeError("LIVE_TRADING is not true. Refusing to initialize live trader.")

    async def start(self) -> None:
        if self._session is not None and not self._session.closed:
            return
        self._session = aiohttp.ClientSession()

    async def close(self) -> None:
        if self._session is None:
            return
        await self._session.close()
        self._session = None

    async def initialize(self) -> None:
        equity = await self.fetch_usdt_equity()
        self.account_equity_usdt = equity
        if equity > self.max_live_equity_usdt:
            raise RuntimeError(
                f"USDT equity {equity:.4f} > MAX_LIVE_EQUITY_USDT {self.max_live_equity_usdt:.4f}. Refusing live trading."
            )
        await self.set_leverage()
        position = await self.fetch_position_snapshot()
        self.position_contracts = position.contracts
        logger.warning(
            "LIVE trader initialized | symbol=%s td_mode=%s leverage=%s equity=%.4f existing_side=%s existing_contracts=%s existing_avg=%.4f contract_multiplier=%s min_contracts=%s",
            self.symbol,
            self.td_mode,
            self.leverage,
            equity,
            position.side,
            self.position_contracts,
            position.avg_entry_price,
            self.contract_multiplier,
            self.min_contracts,
        )

    async def execute_intent(self, intent: TradeIntent) -> LiveTradeResult:
        if intent.intent_type == "NEAR_TP_REDUCE":
            return await self.execute_near_tp_reduce(intent)
        if intent.intent_type == "MARKET_EXIT_RUNNER":
            return await self.execute_market_exit_runner(intent)
        if intent.intent_type == "UPDATE_TP":
            return await self.replace_take_profit(intent)

        contracts = self.eth_qty_to_contracts(Decimal(str(intent.size.eth_qty)))
        side = "buy" if intent.side == "LONG" else "sell"
        body: dict[str, Any] = {
            "instId": self.symbol,
            "tdMode": self.td_mode,
            "side": side,
            "ordType": "market",
            "sz": self.decimal_to_str(contracts),
        }
        pos_side = self.pos_side(intent.side)
        if pos_side:
            body["posSide"] = pos_side

        res = await self.request("POST", "/api/v5/trade/order", body)
        order_id = self.extract_order_id(res)

        # From here on, assume the entry may already be live. Never let a later TP
        # failure look like a pre-entry failure to the caller.
        try:
            position = await self.fetch_position_snapshot()
            self.position_contracts = position.contracts if position.contracts > 0 else self.position_contracts + contracts
        except Exception:
            logger.exception("Failed to refresh position after entry; using requested contracts as fallback")
            self.position_contracts += contracts

        try:
            tp = await self.replace_take_profit(intent)
            if not tp.ok:
                return LiveTradeResult(
                    ok=False,
                    action=intent.intent_type,
                    order_id=order_id,
                    tp_order_id=tp.tp_order_id,
                    contracts=self.decimal_to_str(contracts),
                    tp_price=tp.tp_price,
                    message=f"entry_filled_but_tp_failed: {tp.message}",
                    entry_filled=True,
                    tp_ok=False,
                    tp_order_ids=tp.tp_order_ids,
                    protective_sl_order_id=tp.protective_sl_order_id,
                    protective_sl_price=tp.protective_sl_price,
                    protective_sl_ok=tp.protective_sl_ok,
                )
            return LiveTradeResult(
                ok=True,
                action=intent.intent_type,
                order_id=order_id,
                tp_order_id=tp.tp_order_id,
                contracts=self.decimal_to_str(contracts),
                tp_price=tp.tp_price,
                message="market order placed and take-profit protected",
                entry_filled=True,
                tp_ok=True,
                tp_order_ids=tp.tp_order_ids,
                protective_sl_order_id=tp.protective_sl_order_id,
                protective_sl_price=tp.protective_sl_price,
                protective_sl_ok=tp.protective_sl_ok,
            )
        except Exception as exc:
            logger.exception("Entry appears filled, but TP placement raised an exception")
            return LiveTradeResult(
                ok=False,
                action=intent.intent_type,
                order_id=order_id,
                tp_order_id=None,
                contracts=self.decimal_to_str(contracts),
                tp_price=self.price_to_str(intent.tp_price),
                message=f"entry_filled_but_tp_exception: {exc}",
                entry_filled=True,
                tp_ok=False,
            )

    async def execute_near_tp_reduce(self, intent: TradeIntent) -> LiveTradeResult:
        action = "NEAR_TP_REDUCE"
        position = await self.fetch_position_snapshot()
        if not position.has_position:
            return LiveTradeResult(False, action, None, None, "0", self.price_to_str(intent.tp_price), "no position")
        if position.side != intent.side:
            return LiveTradeResult(False, action, None, None, "0", self.price_to_str(intent.tp_price), "position side mismatch")

        contracts_before = position.contracts
        reduce_ratio = Decimal(str(getattr(intent, "near_tp_reduce_ratio", 0.0) or os.getenv("NEAR_TP_REDUCE_RATIO", "0.5")))
        reduce_ratio = min(max(reduce_ratio, Decimal("0")), Decimal("1"))
        reduce_contracts = self.round_contracts_down(contracts_before * reduce_ratio)
        if reduce_contracts < self.min_contracts:
            return LiveTradeResult(
                False,
                action,
                None,
                None,
                self.decimal_to_str(contracts_before),
                self.price_to_str(intent.tp_price),
                "reduce size too small",
                contracts_before=self.decimal_to_str(contracts_before),
            )

        body = self._reduce_only_market_order_body(intent.side, reduce_contracts)
        res = await self.request("POST", "/api/v5/trade/order", body)
        order_id = self.extract_order_id(res)
        logger.warning(
            "NEAR_TP_REDUCE_ORDER_PLACED | side=%s contracts=%s ordId=%s",
            intent.side,
            self.decimal_to_str(reduce_contracts),
            order_id,
        )

        try:
            refreshed = await self.fetch_position_snapshot()
        except Exception:
            logger.exception("Failed to refresh position after Near-TP reduce")
            refreshed = PositionSnapshot(intent.side, contracts_before - reduce_contracts, position.avg_entry_price, 0.0, Decimal("0"))
        contracts_after = refreshed.contracts if refreshed.has_position and refreshed.side == intent.side else Decimal("0")
        self.position_contracts = contracts_after
        logger.warning(
            "NEAR_TP_REDUCE_FILLED | side=%s contracts_before=%s contracts_reduced=%s contracts_after=%s",
            intent.side,
            self.decimal_to_str(contracts_before),
            self.decimal_to_str(reduce_contracts),
            self.decimal_to_str(contracts_after),
        )

        base_result_kwargs = {
            "order_id": order_id,
            "contracts": self.decimal_to_str(reduce_contracts),
            "tp_price": self.price_to_str(intent.tp_price),
            "entry_filled": False,
            "reduce_filled": True,
            "contracts_before": self.decimal_to_str(contracts_before),
            "contracts_reduced": self.decimal_to_str(reduce_contracts),
            "contracts_after": self.decimal_to_str(contracts_after),
        }
        if contracts_after <= 0:
            return LiveTradeResult(
                True,
                action,
                tp_order_id=None,
                message="near_tp_reduce_closed_position",
                tp_ok=True,
                protective_sl_ok=True,
                near_tp_exit_all=True,
                **base_result_kwargs,
            )

        single_intent = replace(intent, partial_tp_price=None, partial_tp_ratio=0.0, tp_plan="SINGLE", partial_tp_consumed=True)
        tp_order_id: str | None = None
        tp_order_ids: tuple[str, ...] = ()
        tp_ok = False
        final_tp_failure = ""
        try:
            tp = await self.replace_take_profit(single_intent)
        except Exception as exc:
            logger.exception("Near-TP reduce filled but final TP replacement raised")
            final_tp_failure = f"final_tp_failed_exception: {exc}"
        else:
            tp_order_id = tp.tp_order_id
            tp_order_ids = tp.tp_order_ids
            tp_ok = bool(tp.ok)
            if tp.ok:
                logger.warning(
                    "NEAR_TP_FINAL_TP_REPLACED | side=%s contracts=%s tp_price=%s tp_order_id=%s",
                    intent.side,
                    tp.contracts,
                    tp.tp_price,
                    tp.tp_order_id,
                )
            else:
                final_tp_failure = f"final_tp_failed: {tp.message}"

        protective_sl_price = getattr(intent, "near_tp_protective_sl_price", None)
        if protective_sl_price is None:
            pct = float(os.getenv("NEAR_TP_PROTECTIVE_SL_PROFIT_PCT", "0.001"))
            protective_sl_price = intent.avg_entry_price * (1 + pct) if intent.side == "LONG" else intent.avg_entry_price * (1 - pct)

        if os.getenv("NEAR_TP_PROTECTIVE_SL_ENABLED", "true").strip().lower() not in {"1", "true", "yes", "y", "on"}:
            if not tp_ok:
                sl_ok = False
                sl_order_id = None
                sl_message = f"{final_tp_failure}; protective_sl_disabled"
            else:
                return LiveTradeResult(
                    True,
                    action,
                    tp_order_id=tp_order_id,
                    message="near_tp_reduce_done_protective_sl_disabled",
                    tp_ok=True,
                    tp_order_ids=tp_order_ids,
                    protective_sl_price=self.price_to_str(float(protective_sl_price)),
                    protective_sl_ok=True,
                    **base_result_kwargs,
                )
        else:
            sl_ok, sl_order_id, sl_message = await self.place_near_tp_protective_stop_with_retries(
                intent.side,
                contracts_after,
                float(protective_sl_price),
                retry_count=int(os.getenv("NEAR_TP_PROTECTIVE_SL_RETRY_COUNT", "3")),
                retry_interval_seconds=float(os.getenv("NEAR_TP_PROTECTIVE_SL_RETRY_INTERVAL_SECONDS", "1")),
            )
        message_prefix = f"{final_tp_failure}; " if final_tp_failure else ""
        if sl_ok:
            self.near_tp_protective_sl_order_id = sl_order_id
            logger.warning(
                "NEAR_TP_PROTECTIVE_SL_PLACED | side=%s contracts=%s stop_price=%s algoId=%s",
                intent.side,
                self.decimal_to_str(contracts_after),
                self.price_to_str(float(protective_sl_price)),
                sl_order_id,
            )
            return LiveTradeResult(
                True,
                action,
                tp_order_id=tp_order_id,
                message=f"{message_prefix}near_tp_reduce_done_final_tp_and_protective_sl_placed",
                tp_ok=tp_ok,
                tp_order_ids=tp_order_ids,
                protective_sl_order_id=sl_order_id,
                protective_sl_price=self.price_to_str(float(protective_sl_price)),
                protective_sl_ok=True,
                **base_result_kwargs,
            )

        fail_action = os.getenv("NEAR_TP_SL_FAIL_ACTION", "MARKET_EXIT").strip().upper()
        if fail_action == "MARKET_EXIT":
            logger.error("NEAR_TP_PROTECTIVE_SL_FAILED_MARKET_EXIT | side=%s message=%s", intent.side, sl_message)
            exit_ok, exit_message = await self.market_exit_remaining_position_with_retries(
                intent.side,
                retry_count=int(os.getenv("NEAR_TP_SL_FAIL_MARKET_EXIT_RETRY_COUNT", "3")),
            )
            if exit_ok:
                return LiveTradeResult(
                    True,
                    action,
                    tp_order_id=tp_order_id,
                    message=f"{message_prefix}protective_sl_failed_market_exit_success: {sl_message}; {exit_message}",
                    tp_ok=tp_ok,
                    tp_order_ids=tp_order_ids,
                    protective_sl_price=self.price_to_str(float(protective_sl_price)),
                    protective_sl_ok=False,
                    near_tp_exit_all=True,
                    contracts_after="0",
                    **{k: v for k, v in base_result_kwargs.items() if k != "contracts_after"},
                )
            return LiveTradeResult(
                False,
                action,
                tp_order_id=tp_order_id,
                message=f"{message_prefix}protective_sl_failed_and_market_exit_failed: {sl_message}; {exit_message}",
                tp_ok=tp_ok,
                tp_order_ids=tp_order_ids,
                protective_sl_price=self.price_to_str(float(protective_sl_price)),
                protective_sl_ok=False,
                **base_result_kwargs,
            )

        return LiveTradeResult(
            False,
            action,
            tp_order_id=tp_order_id,
            message=f"{message_prefix}protective_sl_failed_halt_only: {sl_message}",
            tp_ok=tp_ok,
            tp_order_ids=tp_order_ids,
            protective_sl_price=self.price_to_str(float(protective_sl_price)),
            protective_sl_ok=False,
            **base_result_kwargs,
        )

    async def execute_market_exit_runner(self, intent: TradeIntent) -> LiveTradeResult:
        action = "MARKET_EXIT_RUNNER"
        restored_trend_runner_sl_order_id = getattr(intent, "trend_runner_sl_order_id", None)
        if restored_trend_runner_sl_order_id and not self.trend_runner_sl_order_id:
            self.trend_runner_sl_order_id = restored_trend_runner_sl_order_id
        position = await self.fetch_position_snapshot()
        if not position.has_position:
            await self._cleanup_after_near_tp_market_exit()
            return LiveTradeResult(True, action, None, None, "0", self.price_to_str(intent.tp_price), "runner_already_flat", near_tp_exit_all=True)
        if position.side != intent.side:
            await self._cleanup_after_near_tp_market_exit()
            return LiveTradeResult(True, action, None, None, "0", self.price_to_str(intent.tp_price), "runner_side_absent", near_tp_exit_all=True)

        contracts_before = position.contracts
        ok, message = await self.market_exit_remaining_position_with_retries(
            intent.side,
            retry_count=int(os.getenv("NEAR_TP_SL_FAIL_MARKET_EXIT_RETRY_COUNT", "3")),
        )
        refreshed = await self.fetch_position_snapshot()
        contracts_after = refreshed.contracts if refreshed.has_position and refreshed.side == intent.side else Decimal("0")
        self.position_contracts = contracts_after
        if ok:
            return LiveTradeResult(
                True,
                action,
                None,
                None,
                self.decimal_to_str(contracts_before),
                self.price_to_str(intent.tp_price),
                message,
                reduce_filled=True,
                near_tp_exit_all=True,
                contracts_before=self.decimal_to_str(contracts_before),
                contracts_reduced=self.decimal_to_str(contracts_before),
                contracts_after=self.decimal_to_str(contracts_after),
            )
        return LiveTradeResult(
            False,
            action,
            None,
            None,
            self.decimal_to_str(contracts_before),
            self.price_to_str(intent.tp_price),
            message,
            reduce_filled=True,
            near_tp_exit_all=False,
            contracts_before=self.decimal_to_str(contracts_before),
            contracts_reduced="",
            contracts_after=self.decimal_to_str(contracts_after),
        )

    async def replace_take_profit(self, intent: TradeIntent) -> LiveTradeResult:
        try:
            position = await self.fetch_position_snapshot()
            if position.contracts > 0:
                self.position_contracts = position.contracts
        except Exception:
            logger.exception("Failed to refresh position before replacing TP")

        if self.position_contracts <= 0:
            return LiveTradeResult(False, intent.intent_type, None, None, "0", self.price_to_str(intent.tp_price), "no position to protect")

        await self.cancel_existing_reduce_only_orders()

        specs = self._build_take_profit_order_specs(intent)
        placed_order_ids: list[str] = []
        message = "take-profit replaced"
        try:
            placed_order_ids = await self._place_reduce_only_take_profit_orders(intent, specs)
        except Exception:
            if len(specs) <= 1:
                raise
            logger.exception("Failed to place split take-profit orders; falling back to one full-size final TP")
            await self.cancel_existing_reduce_only_orders()
            fallback_specs = [("final", self.position_contracts, intent.tp_price)]
            placed_order_ids = await self._place_reduce_only_take_profit_orders(intent, fallback_specs)
            specs = fallback_specs
            message = "split take-profit fallback to single final TP"

        tp_order_id = ",".join(placed_order_ids)
        self.tp_order_id = tp_order_id
        tp_price_text = self._tp_price_summary(specs)
        protective_sl_order_id: str | None = None
        protective_sl_price_text = ""
        protective_sl_ok = False
        runner_sl_price = getattr(intent, "middle_runner_protective_sl_price", None)
        if getattr(intent, "middle_runner_active", False) and runner_sl_price is not None:
            old_sl_order_id = getattr(intent, "middle_runner_protective_sl_order_id", None) or self.middle_runner_protective_sl_order_id
            sl_ok, sl_order_id, sl_message = await self.place_middle_runner_protective_stop_with_retries(
                intent.side,
                self.position_contracts,
                float(runner_sl_price),
                retry_count=int(os.getenv("NEAR_TP_PROTECTIVE_SL_RETRY_COUNT", "3")),
                retry_interval_seconds=float(os.getenv("NEAR_TP_PROTECTIVE_SL_RETRY_INTERVAL_SECONDS", "1")),
            )
            protective_sl_price_text = self.price_to_str(float(runner_sl_price))
            if not sl_ok:
                return LiveTradeResult(
                    False,
                    intent.intent_type,
                    None,
                    tp_order_id,
                    self.decimal_to_str(self.position_contracts),
                    tp_price_text,
                    f"middle_runner_protective_sl_failed: {sl_message}",
                    entry_filled=False,
                    tp_ok=True,
                    tp_order_ids=tuple(placed_order_ids),
                    protective_sl_price=protective_sl_price_text,
                    protective_sl_ok=False,
                )
            protective_sl_order_id = sl_order_id
            protective_sl_ok = True
            self.middle_runner_protective_sl_order_id = sl_order_id
            if old_sl_order_id and old_sl_order_id != sl_order_id:
                await self.cancel_middle_runner_protective_stop(old_sl_order_id)
            logger.warning(
                "MIDDLE_RUNNER_SL_UPDATED | side=%s contracts=%s protective_sl_price=%s old_sl_order_id=%s new_sl_order_id=%s",
                intent.side,
                self.decimal_to_str(self.position_contracts),
                protective_sl_price_text,
                old_sl_order_id,
                sl_order_id,
            )
        trend_runner_sl_price = getattr(intent, "trend_runner_sl_price", None) or getattr(intent, "three_stage_runner_sl_price", None)
        if (
            getattr(intent, "trend_runner_active", False)
            and trend_runner_sl_price is not None
        ):
            old_sl_order_id = getattr(intent, "trend_runner_sl_order_id", None) or self.trend_runner_sl_order_id
            sl_contracts = self._trend_runner_sl_contracts(intent)
            sl_ok, sl_order_id, sl_message = await self.place_trend_runner_protective_stop_with_retries(
                intent.side,
                sl_contracts,
                float(trend_runner_sl_price),
                retry_count=int(os.getenv("NEAR_TP_PROTECTIVE_SL_RETRY_COUNT", "3")),
                retry_interval_seconds=float(os.getenv("NEAR_TP_PROTECTIVE_SL_RETRY_INTERVAL_SECONDS", "1")),
            )
            protective_sl_price_text = self.price_to_str(float(trend_runner_sl_price))
            if not sl_ok:
                return LiveTradeResult(
                    False,
                    intent.intent_type,
                    None,
                    tp_order_id,
                    self.decimal_to_str(self.position_contracts),
                    tp_price_text,
                    f"trend_runner_protective_sl_failed: {sl_message}",
                    entry_filled=False,
                    tp_ok=True,
                    tp_order_ids=tuple(placed_order_ids),
                    protective_sl_price=protective_sl_price_text,
                    protective_sl_ok=False,
                )
            protective_sl_order_id = sl_order_id
            protective_sl_ok = True
            self.trend_runner_sl_order_id = sl_order_id
            if old_sl_order_id and old_sl_order_id != sl_order_id:
                await self.cancel_trend_runner_protective_stop(old_sl_order_id)
            logger.warning(
                "TREND_RUNNER_SL_UPDATED | side=%s contracts=%s protective_sl_price=%s old_sl_order_id=%s new_sl_order_id=%s",
                intent.side,
                self.decimal_to_str(sl_contracts),
                protective_sl_price_text,
                old_sl_order_id,
                sl_order_id,
            )
        return LiveTradeResult(
            True,
            intent.intent_type,
            None,
            tp_order_id,
            self.decimal_to_str(self.position_contracts),
            tp_price_text,
            message,
            entry_filled=False,
            tp_ok=True,
            tp_order_ids=tuple(placed_order_ids),
            protective_sl_order_id=protective_sl_order_id,
            protective_sl_price=protective_sl_price_text,
            protective_sl_ok=protective_sl_ok,
        )

    def _build_take_profit_order_specs(self, intent: TradeIntent) -> list[tuple[str, Decimal, float]]:
        partial_tp_price = getattr(intent, "partial_tp_price", None)
        partial_tp_ratio = Decimal(str(getattr(intent, "partial_tp_ratio", 0.0)))
        tp_plan = getattr(intent, "tp_plan", "SINGLE")
        if tp_plan == "THREE_STAGE_RUNNER":
            return self._build_three_stage_order_specs(intent)
        if tp_plan not in {"SPLIT_PARTIAL_FINAL", "SPLIT_50_50", "MIDDLE_RUNNER"} or partial_tp_price is None or partial_tp_ratio <= 0 or partial_tp_ratio >= 1:
            return [("final", self.position_contracts, intent.tp_price)]

        partial_contracts = self.round_contracts_down(self.position_contracts * partial_tp_ratio)
        final_contracts = self.position_contracts - partial_contracts
        if partial_contracts < self.min_contracts or final_contracts < self.min_contracts:
            logger.warning(
                "SPLIT_TP_FALLBACK_SINGLE | reason=size_too_small total_contracts=%s partial_contracts=%s final_contracts=%s min_contracts=%s",
                self.position_contracts,
                partial_contracts,
                final_contracts,
                self.min_contracts,
            )
            return [("final", self.position_contracts, intent.tp_price)]
        if tp_plan == "MIDDLE_RUNNER":
            return [("middle", partial_contracts, float(partial_tp_price)), ("runner", final_contracts, intent.tp_price)]
        return [("partial", partial_contracts, float(partial_tp_price)), ("final", final_contracts, intent.tp_price)]

    def _build_three_stage_order_specs(self, intent: TradeIntent) -> list[tuple[str, Decimal, float]]:
        tp1_price = getattr(intent, "three_stage_tp1_price", None)
        tp2_price = getattr(intent, "three_stage_tp2_price", None)
        tp1_ratio = Decimal(str(getattr(intent, "three_stage_tp1_ratio", 0.0)))
        tp2_ratio = Decimal(str(getattr(intent, "three_stage_tp2_ratio", 0.0)))
        if tp1_price is None or tp2_price is None or tp1_ratio <= 0 or tp2_ratio <= 0:
            return [("final", self.position_contracts, intent.tp_price)]
        tp1_contracts = self.round_contracts_down(self.position_contracts * tp1_ratio)
        tp2_contracts = self.round_contracts_down(self.position_contracts * tp2_ratio)
        runner_contracts = self.position_contracts - tp1_contracts - tp2_contracts
        if tp1_contracts < self.min_contracts or tp2_contracts < self.min_contracts or runner_contracts < self.min_contracts:
            logger.warning(
                "THREE_STAGE_TP_FALLBACK_SINGLE | reason=size_too_small total_contracts=%s tp1_contracts=%s tp2_contracts=%s runner_contracts=%s min_contracts=%s",
                self.position_contracts,
                tp1_contracts,
                tp2_contracts,
                runner_contracts,
                self.min_contracts,
            )
            return [("final", self.position_contracts, intent.tp_price)]
        return [
            ("tp1_middle", tp1_contracts, float(tp1_price)),
            ("tp2_outer", tp2_contracts, float(tp2_price)),
        ]

    def _trend_runner_sl_contracts(self, intent: TradeIntent) -> Decimal:
        if getattr(intent, "trend_runner_active", False):
            return self.position_contracts
        runner_ratio = Decimal(str(getattr(intent, "three_stage_runner_ratio", 0.0)))
        if runner_ratio <= 0 or runner_ratio >= 1:
            return self.position_contracts
        contracts = self.round_contracts_down(self.position_contracts * runner_ratio)
        if contracts < self.min_contracts:
            return self.position_contracts
        return contracts

    async def _place_reduce_only_take_profit_orders(self, intent: TradeIntent, specs: list[tuple[str, Decimal, float]]) -> list[str]:
        placed_order_ids: list[str] = []
        for label, contracts, price in specs:
            body = self._reduce_only_tp_order_body(intent.side, contracts, price)
            res = await self.request("POST", "/api/v5/trade/order", body)
            order_id = self.extract_order_id(res)
            placed_order_ids.append(order_id)
            logger.info(
                "TP_ORDER_PLACED | label=%s side=%s contracts=%s price=%s ordId=%s",
                label,
                intent.side,
                self.decimal_to_str(contracts),
                self.price_to_str(price),
                order_id,
            )
        return placed_order_ids

    def _reduce_only_tp_order_body(self, side: PositionSide, contracts: Decimal, price: float) -> dict[str, Any]:
        close_side = "sell" if side == "LONG" else "buy"
        body: dict[str, Any] = {
            "instId": self.symbol,
            "tdMode": self.td_mode,
            "side": close_side,
            "ordType": "limit",
            "px": self.price_to_str(price),
            "sz": self.decimal_to_str(contracts),
            "reduceOnly": "true",
        }
        pos_side = self.pos_side(side)
        if pos_side:
            body["posSide"] = pos_side
        return body

    def _reduce_only_market_order_body(self, side: PositionSide, contracts: Decimal) -> dict[str, Any]:
        close_side = "sell" if side == "LONG" else "buy"
        body: dict[str, Any] = {
            "instId": self.symbol,
            "tdMode": self.td_mode,
            "side": close_side,
            "ordType": "market",
            "sz": self.decimal_to_str(contracts),
            "reduceOnly": "true",
        }
        pos_side = self.pos_side(side)
        if pos_side:
            body["posSide"] = pos_side
        return body

    async def place_near_tp_protective_stop_with_retries(
        self,
        side: PositionSide,
        contracts: Decimal,
        stop_price: float,
        retry_count: int,
        retry_interval_seconds: float,
    ) -> tuple[bool, str | None, str]:
        retry_count = max(int(retry_count), 1)
        last_error = ""
        for attempt in range(1, retry_count + 1):
            try:
                body = self._near_tp_protective_sl_algo_body(side, contracts, stop_price)
                res = await self.request("POST", "/api/v5/trade/order-algo", body)
                algo_id = self.extract_algo_id(res)
                if await self.verify_near_tp_protective_stop(algo_id, side, contracts, stop_price):
                    return True, algo_id, "protective_sl_placed"
                await self._cancel_unverified_near_tp_algo(algo_id, phase="primary")
                last_error = f"protective_sl_verify_failed algoId={algo_id}"
                raise RuntimeError(last_error)
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "NEAR_TP_PROTECTIVE_SL_RETRY | attempt=%s/%s side=%s contracts=%s stop_price=%s error=%s",
                    attempt,
                    retry_count,
                    side,
                    self.decimal_to_str(contracts),
                    self.price_to_str(stop_price),
                    exc,
                )
                if attempt < retry_count and retry_interval_seconds > 0:
                    await asyncio.sleep(retry_interval_seconds)

        for attempt in range(1, retry_count + 1):
            try:
                body = self._near_tp_fallback_conditional_close_body(side, contracts, stop_price)
                res = await self.request("POST", "/api/v5/trade/order-algo", body)
                algo_id = self.extract_algo_id(res)
                if await self.verify_near_tp_protective_stop(algo_id, side, contracts, stop_price):
                    return True, algo_id, "fallback_conditional_close_placed"
                await self._cancel_unverified_near_tp_algo(algo_id, phase="secondary")
                last_error = f"fallback_conditional_verify_failed algoId={algo_id}"
                raise RuntimeError(last_error)
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "NEAR_TP_PROTECTIVE_SL_FALLBACK_RETRY | attempt=%s/%s side=%s contracts=%s stop_price=%s error=%s",
                    attempt,
                    retry_count,
                    side,
                    self.decimal_to_str(contracts),
                    self.price_to_str(stop_price),
                    exc,
                )
                if attempt < retry_count and retry_interval_seconds > 0:
                    await asyncio.sleep(retry_interval_seconds)
        return False, None, last_error or "protective_sl_retries_exhausted"

    async def place_middle_runner_protective_stop_with_retries(
        self,
        side: PositionSide,
        contracts: Decimal,
        stop_price: float,
        retry_count: int,
        retry_interval_seconds: float,
    ) -> tuple[bool, str | None, str]:
        ok, order_id, message = await self.place_near_tp_protective_stop_with_retries(
            side,
            contracts,
            stop_price,
            retry_count=retry_count,
            retry_interval_seconds=retry_interval_seconds,
        )
        if ok:
            self.middle_runner_protective_sl_order_id = order_id
        return ok, order_id, message

    async def place_trend_runner_protective_stop_with_retries(
        self,
        side: PositionSide,
        contracts: Decimal,
        stop_price: float,
        retry_count: int,
        retry_interval_seconds: float,
    ) -> tuple[bool, str | None, str]:
        ok, order_id, message = await self.place_near_tp_protective_stop_with_retries(
            side,
            contracts,
            stop_price,
            retry_count=retry_count,
            retry_interval_seconds=retry_interval_seconds,
        )
        if ok:
            self.trend_runner_sl_order_id = order_id
        return ok, order_id, message

    async def _cancel_unverified_near_tp_algo(self, algo_id: str, *, phase: str) -> None:
        try:
            ok = await self.cancel_near_tp_protective_stop(algo_id)
            logger.warning(
                "NEAR_TP_PROTECTIVE_SL_VERIFY_CANCELLED | phase=%s algoId=%s ok=%s",
                phase,
                algo_id,
                ok,
            )
        except Exception as exc:
            logger.warning(
                "NEAR_TP_PROTECTIVE_SL_VERIFY_CANCEL_FAILED | phase=%s algoId=%s error=%s",
                phase,
                algo_id,
                exc,
            )

    async def verify_near_tp_protective_stop(self, algo_id: str, side: PositionSide, contracts: Decimal, stop_price: float) -> bool:
        attempts = max(int(os.getenv("NEAR_TP_PROTECTIVE_SL_VERIFY_ATTEMPTS", "3")), 1)
        interval_seconds = float(os.getenv("NEAR_TP_PROTECTIVE_SL_VERIFY_INTERVAL_SECONDS", "0.2"))
        for attempt in range(1, attempts + 1):
            try:
                orders = await self.fetch_pending_algo_orders()
                for item in orders:
                    if self._near_tp_protective_stop_matches(item, algo_id, side, contracts, stop_price):
                        return True
            except Exception as exc:
                logger.warning("NEAR_TP_PROTECTIVE_SL_VERIFY_FAILED | attempt=%s/%s algoId=%s error=%s", attempt, attempts, algo_id, exc)
            if attempt < attempts and interval_seconds > 0:
                await asyncio.sleep(interval_seconds)
        logger.warning("NEAR_TP_PROTECTIVE_SL_VERIFY_MISSING | algoId=%s side=%s contracts=%s stop_price=%s", algo_id, side, self.decimal_to_str(contracts), self.price_to_str(stop_price))
        return False

    def _near_tp_protective_stop_matches(self, item: dict[str, Any], algo_id: str, side: PositionSide, contracts: Decimal, stop_price: float) -> bool:
        item_algo_id = str(item.get("algoId") or item.get("ordId") or "")
        if item_algo_id != str(algo_id):
            return False
        if item.get("instId") != self.symbol:
            return False
        close_side = "sell" if side == "LONG" else "buy"
        if str(item.get("side", "")).lower() != close_side:
            return False
        try:
            item_contracts = Decimal(str(item.get("sz", "0")))
        except Exception:
            return False
        contract_tolerance = max(self.contract_precision, contracts.copy_abs() * Decimal("0.001"))
        if abs(item_contracts - contracts) > contract_tolerance:
            return False
        raw_trigger = item.get("slTriggerPx") or item.get("triggerPx")
        if raw_trigger is None:
            return False
        try:
            item_stop = Decimal(str(raw_trigger))
            expected_stop = Decimal(self.price_to_str(stop_price))
        except Exception:
            return False
        price_tolerance = max(Decimal("0.01"), expected_stop.copy_abs() * Decimal("0.0001"))
        return abs(item_stop - expected_stop) <= price_tolerance

    def _near_tp_protective_sl_algo_body(self, side: PositionSide, contracts: Decimal, stop_price: float) -> dict[str, Any]:
        close_side = "sell" if side == "LONG" else "buy"
        body: dict[str, Any] = {
            "instId": self.symbol,
            "tdMode": self.td_mode,
            "side": close_side,
            "ordType": "conditional",
            "sz": self.decimal_to_str(contracts),
            "slTriggerPx": self.price_to_str(stop_price),
            "slOrdPx": "-1",
            "slTriggerPxType": "last",
            "reduceOnly": "true",
        }
        pos_side = self.pos_side(side)
        if pos_side:
            body["posSide"] = pos_side
        return body

    def _near_tp_fallback_conditional_close_body(self, side: PositionSide, contracts: Decimal, stop_price: float) -> dict[str, Any]:
        close_side = "sell" if side == "LONG" else "buy"
        body: dict[str, Any] = {
            "instId": self.symbol,
            "tdMode": self.td_mode,
            "side": close_side,
            "ordType": "conditional",
            "sz": self.decimal_to_str(contracts),
            "slTriggerPx": self.price_to_str(stop_price),
            "slOrdPx": "-1",
            "slTriggerPxType": "last",
            "reduceOnly": "true",
        }
        pos_side = self.pos_side(side)
        if pos_side:
            body["posSide"] = pos_side
        return body

    async def market_exit_remaining_position_with_retries(self, side: PositionSide, retry_count: int) -> tuple[bool, str]:
        retry_count = max(int(retry_count), 1)
        last_error = ""
        for attempt in range(1, retry_count + 1):
            try:
                position = await self.fetch_position_snapshot()
                if not position.has_position or position.contracts <= 0:
                    self.position_contracts = Decimal("0")
                    await self._cleanup_after_near_tp_market_exit()
                    logger.warning("NEAR_TP_MARKET_EXIT_SUCCESS | reason=already_flat")
                    return True, "already_flat"
                if position.side != side:
                    self.position_contracts = Decimal("0")
                    await self._cleanup_after_near_tp_market_exit()
                    logger.warning(
                        "NEAR_TP_MARKET_EXIT_SUCCESS | reason=target_side_absent expected_side=%s actual_side=%s contracts=%s",
                        side,
                        position.side,
                        self.decimal_to_str(position.contracts),
                    )
                    return True, "target_side_absent"
                if Decimal("0") < position.contracts < self.min_contracts:
                    last_error = (
                        f"dust_position_below_min_contracts contracts={self.decimal_to_str(position.contracts)} "
                        f"min_contracts={self.decimal_to_str(self.min_contracts)}"
                    )
                    logger.error("NEAR_TP_MARKET_EXIT_FAILED | reason=%s", last_error)
                    return False, last_error

                body = self._reduce_only_market_order_body(side, position.contracts)
                res = await self.request("POST", "/api/v5/trade/order", body)
                order_id = self.extract_order_id(res)
                refreshed = await self.fetch_position_snapshot()
                if not refreshed.has_position or refreshed.contracts <= 0:
                    self.position_contracts = Decimal("0")
                    await self._cleanup_after_near_tp_market_exit()
                    logger.warning(
                        "NEAR_TP_MARKET_EXIT_SUCCESS | side=%s contracts=%s ordId=%s attempt=%s",
                        side,
                        self.decimal_to_str(position.contracts),
                        order_id,
                        attempt,
                    )
                    return True, f"market_exit_order_id={order_id}"
                if refreshed.side != side:
                    self.position_contracts = Decimal("0")
                    await self._cleanup_after_near_tp_market_exit()
                    logger.warning(
                        "NEAR_TP_MARKET_EXIT_SUCCESS | reason=target_side_absent_after_order side=%s actual_side=%s ordId=%s attempt=%s",
                        side,
                        refreshed.side,
                        order_id,
                        attempt,
                    )
                    return True, f"market_exit_order_id={order_id};target_side_absent_after_order"
                if Decimal("0") < refreshed.contracts < self.min_contracts:
                    last_error = (
                        f"dust_position_below_min_contracts_after_order contracts={self.decimal_to_str(refreshed.contracts)} "
                        f"min_contracts={self.decimal_to_str(self.min_contracts)}"
                    )
                    logger.error(
                        "NEAR_TP_MARKET_EXIT_FAILED | reason=%s ordId=%s attempt=%s/%s",
                        last_error,
                        order_id,
                        attempt,
                        retry_count,
                    )
                    return False, last_error

                self.position_contracts = refreshed.contracts
                last_error = f"market_exit_not_flat_after_order contracts={self.decimal_to_str(refreshed.contracts)}"
                logger.error(
                    "NEAR_TP_MARKET_EXIT_FAILED | reason=not_flat_after_order attempt=%s/%s side=%s remaining_contracts=%s ordId=%s",
                    attempt,
                    retry_count,
                    side,
                    self.decimal_to_str(refreshed.contracts),
                    order_id,
                )
            except Exception as exc:
                last_error = str(exc)
                logger.error("NEAR_TP_MARKET_EXIT_FAILED | attempt=%s/%s side=%s error=%s", attempt, retry_count, side, exc)
        return False, last_error or "market_exit_failed"

    async def _cleanup_after_near_tp_market_exit(self) -> None:
        try:
            await self.cancel_existing_reduce_only_orders()
        except Exception:
            logger.warning("NEAR_TP_MARKET_EXIT_SUCCESS | cleanup=cancel_reduce_only_tp_failed")
        if self.near_tp_protective_sl_order_id:
            await self.cancel_near_tp_protective_stop(self.near_tp_protective_sl_order_id)
        middle_runner_sl_order_id = getattr(self, "middle_runner_protective_sl_order_id", None)
        if middle_runner_sl_order_id:
            await self.cancel_middle_runner_protective_stop(middle_runner_sl_order_id)
        trend_runner_sl_order_id = getattr(self, "trend_runner_sl_order_id", None)
        if trend_runner_sl_order_id:
            await self.cancel_trend_runner_protective_stop(trend_runner_sl_order_id)

    def _tp_price_summary(self, specs: list[tuple[str, Decimal, float]]) -> str:
        if len(specs) == 1:
            return self.price_to_str(specs[0][2])
        return ",".join(f"{label}:{self.price_to_str(price)}" for label, _contracts, price in specs)

    async def cancel_existing_reduce_only_orders(self) -> None:
        orders = await self.fetch_pending_orders()
        for item in orders:
            if item.get("instId") != self.symbol:
                continue
            if str(item.get("reduceOnly", "")).lower() != "true":
                continue
            ord_id = item.get("ordId")
            if not ord_id:
                continue
            try:
                await self.request("POST", "/api/v5/trade/cancel-order", {"instId": self.symbol, "ordId": ord_id})
                logger.info("Canceled existing reduce-only order | ordId=%s", ord_id)
            except Exception:
                logger.exception("Failed to cancel existing reduce-only order | ordId=%s", ord_id)

    async def fetch_pending_orders(self) -> list[dict[str, Any]]:
        res = await self.request("GET", f"/api/v5/trade/orders-pending?instId={self.symbol}")
        return list(res.get("data", []))

    async def fetch_pending_algo_orders(self) -> list[dict[str, Any]]:
        res = await self.request("GET", f"/api/v5/trade/orders-algo-pending?instId={self.symbol}&ordType=conditional")
        return list(res.get("data", []))

    async def cancel_near_tp_protective_stop(self, order_id: str | None) -> bool:
        if not order_id:
            return True
        try:
            await self.request("POST", "/api/v5/trade/cancel-algos", [{"instId": self.symbol, "algoId": order_id}])
            if self.near_tp_protective_sl_order_id == order_id:
                self.near_tp_protective_sl_order_id = None
            logger.warning("NEAR_TP_PROTECTIVE_SL_CANCEL_ON_FLAT | algoId=%s", order_id)
            return True
        except Exception as exc:
            text = str(exc).lower()
            if "not found" in text or "not exist" in text or "does not exist" in text or "already" in text:
                logger.info("NEAR_TP_PROTECTIVE_SL_CANCEL_ON_FLAT | algoId=%s already_absent message=%s", order_id, exc)
                return True
            logger.warning("NEAR_TP_PROTECTIVE_SL_CANCEL_ON_FLAT | algoId=%s failed=%s", order_id, exc)
            return False

    async def cancel_middle_runner_protective_stop(self, order_id: str | None) -> bool:
        if not order_id:
            return True
        ok = await self.cancel_near_tp_protective_stop(order_id)
        if ok and getattr(self, "middle_runner_protective_sl_order_id", None) == order_id:
            self.middle_runner_protective_sl_order_id = None
        if ok:
            logger.warning("MIDDLE_RUNNER_SL_CANCELLED | algoId=%s", order_id)
        return ok

    async def cancel_trend_runner_protective_stop(self, order_id: str | None) -> bool:
        if not order_id:
            return True
        ok = await self.cancel_near_tp_protective_stop(order_id)
        if ok and getattr(self, "trend_runner_sl_order_id", None) == order_id:
            self.trend_runner_sl_order_id = None
        if ok:
            logger.warning("TREND_RUNNER_SL_CANCELLED | algoId=%s", order_id)
        return ok

    async def fetch_usdt_equity(self) -> float:
        res = await self.request("GET", "/api/v5/account/balance?ccy=USDT")
        data = res.get("data", [])
        if not data:
            return 0.0
        details = data[0].get("details", [])
        for item in details:
            if item.get("ccy") == "USDT":
                return float(item.get("eq") or item.get("availEq") or item.get("availBal") or 0.0)
        return float(data[0].get("totalEq") or 0.0)

    async def fetch_position_contracts(self) -> Decimal:
        return (await self.fetch_position_snapshot()).contracts

    async def fetch_position_snapshot(self) -> PositionSnapshot:
        res = await self.request("GET", f"/api/v5/account/positions?instId={self.symbol}")
        best: PositionSnapshot | None = None
        for item in res.get("data", []):
            if item.get("instId") != self.symbol:
                continue
            raw_pos = Decimal(str(item.get("pos", "0")))
            if raw_pos == 0:
                continue
            contracts = abs(raw_pos)
            avg_entry = float(item.get("avgPx") or item.get("avgPxUsd") or 0.0)
            if self.pos_side_mode == "long_short":
                pos_side = str(item.get("posSide", "")).lower()
                side: PositionSide | None = "LONG" if pos_side == "long" else "SHORT" if pos_side == "short" else None
            else:
                side = "LONG" if raw_pos > 0 else "SHORT"
            best = PositionSnapshot(
                side=side,
                contracts=contracts,
                avg_entry_price=avg_entry,
                eth_qty=float(contracts * self.contract_multiplier),
                raw_pos=raw_pos,
            )
            break
        if best is None:
            return PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0"))
        return best

    def mark_flat(self) -> None:
        self.position_contracts = Decimal("0")
        self.tp_order_id = None
        self.middle_runner_protective_sl_order_id = None
        self.trend_runner_sl_order_id = None

    async def set_leverage(self) -> None:
        body = {"instId": self.symbol, "lever": str(self.leverage), "mgnMode": self.td_mode}
        if self.pos_side_mode == "long_short":
            for pos_side in ("long", "short"):
                side_body = dict(body)
                side_body["posSide"] = pos_side
                await self.request("POST", "/api/v5/account/set-leverage", side_body)
        else:
            await self.request("POST", "/api/v5/account/set-leverage", body)

    async def request(self, method: str, endpoint: str, payload: Any | None = None) -> dict[str, Any]:
        await self.start()
        if self._session is None:
            raise RuntimeError("OKX private REST session is not initialized")
        method = method.upper()
        body = "" if method == "GET" else json.dumps(payload or {}, separators=(",", ":"))
        headers = self.headers(method, endpoint, body)
        if method == "GET":
            async with self._session.get(self.base_url + endpoint, headers=headers, timeout=self._timeout_seconds) as resp:
                res = await resp.json()
        else:
            async with self._session.post(self.base_url + endpoint, headers=headers, data=body, timeout=self._timeout_seconds) as resp:
                res = await resp.json()
        if res.get("code") != "0":
            raise RuntimeError(f"OKX API error: method={method} endpoint={endpoint} response={res}")
        return res

    def headers(self, method: str, endpoint: str, body: str) -> dict[str, str]:
        timestamp = datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
        message = timestamp + method + endpoint + body
        digest = hmac.new(self.secret_key.encode("utf-8"), message.encode("utf-8"), digestmod="sha256").digest()
        signature = base64.b64encode(digest).decode("utf-8")
        return {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
        }

    def eth_qty_to_contracts(self, eth_qty: Decimal) -> Decimal:
        raw_contracts = eth_qty / self.contract_multiplier
        contracts = self.round_contracts_down(raw_contracts)
        if contracts < self.min_contracts:
            raise RuntimeError(f"Order size {contracts} contracts is below minimum {self.min_contracts}")
        return contracts

    def round_contracts_down(self, contracts: Decimal) -> Decimal:
        lots = (contracts / self.contract_precision).to_integral_value(rounding=ROUND_DOWN)
        return lots * self.contract_precision

    def pos_side(self, side: str) -> str | None:
        if self.pos_side_mode != "long_short":
            return None
        return "long" if side == "LONG" else "short"

    @staticmethod
    def extract_order_id(res: dict[str, Any]) -> str:
        data = res.get("data", [])
        if not data or not data[0].get("ordId"):
            raise RuntimeError(f"Missing ordId in response: {res}")
        return str(data[0]["ordId"])

    @staticmethod
    def extract_algo_id(res: dict[str, Any]) -> str:
        data = res.get("data", [])
        if not data:
            raise RuntimeError(f"Missing algoId in response: {res}")
        algo_id = data[0].get("algoId") or data[0].get("ordId")
        if not algo_id:
            raise RuntimeError(f"Missing algoId in response: {res}")
        return str(algo_id)

    @staticmethod
    def decimal_to_str(value: Decimal) -> str:
        return format(value.normalize(), "f")

    @staticmethod
    def price_to_str(price: float) -> str:
        if not math.isfinite(price):
            raise RuntimeError(f"Invalid price: {price}")
        return f"{price:.2f}"
