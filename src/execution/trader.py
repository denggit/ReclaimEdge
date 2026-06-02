from __future__ import annotations

import base64
import datetime
import hmac
import json
import math
import os
from dataclasses import dataclass
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
        )

    def _build_take_profit_order_specs(self, intent: TradeIntent) -> list[tuple[str, Decimal, float]]:
        partial_tp_price = getattr(intent, "partial_tp_price", None)
        partial_tp_ratio = Decimal(str(getattr(intent, "partial_tp_ratio", 0.0)))
        tp_plan = getattr(intent, "tp_plan", "SINGLE")
        if tp_plan != "SPLIT_50_50" or partial_tp_price is None or partial_tp_ratio <= 0 or partial_tp_ratio >= 1:
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
        return [("partial", partial_contracts, float(partial_tp_price)), ("final", final_contracts, intent.tp_price)]

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

    async def set_leverage(self) -> None:
        body = {"instId": self.symbol, "lever": str(self.leverage), "mgnMode": self.td_mode}
        if self.pos_side_mode == "long_short":
            for pos_side in ("long", "short"):
                side_body = dict(body)
                side_body["posSide"] = pos_side
                await self.request("POST", "/api/v5/account/set-leverage", side_body)
        else:
            await self.request("POST", "/api/v5/account/set-leverage", body)

    async def request(self, method: str, endpoint: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
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
    def decimal_to_str(value: Decimal) -> str:
        return format(value.normalize(), "f")

    @staticmethod
    def price_to_str(price: float) -> str:
        if not math.isfinite(price):
            raise RuntimeError(f"Invalid price: {price}")
        return f"{price:.2f}"
