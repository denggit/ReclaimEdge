from __future__ import annotations

import base64
import datetime
import hmac
import json
import logging
import math
import os
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Any, Optional

import aiohttp

from config.env_loader import OKX_CONFIG
from src.strategies.boll_cvd_reclaim_strategy import TradeIntent

logger = logging.getLogger(__name__)


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


class Trader:
    """Simple OKX live trader for ReclaimEdge.

    This module is intentionally small. It only supports the current strategy needs:
    - verify account balance cap
    - set leverage
    - market open long/short
    - place reduce-only take-profit limit order at BOLL middle
    - replace take-profit when needed
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

        if not self.api_key or not self.secret_key or not self.passphrase:
            raise ValueError("OKX API config is incomplete. Check OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHASE.")
        if not self.live_trading:
            raise RuntimeError("LIVE_TRADING is not true. Refusing to initialize live trader.")

    async def initialize(self) -> None:
        equity = await self.fetch_usdt_equity()
        if equity > self.max_live_equity_usdt:
            raise RuntimeError(
                f"USDT equity {equity:.4f} > MAX_LIVE_EQUITY_USDT {self.max_live_equity_usdt:.4f}. Refusing live trading."
            )
        await self.set_leverage()
        self.position_contracts = await self.fetch_position_contracts()
        logger.warning(
            "LIVE trader initialized | symbol=%s td_mode=%s leverage=%s equity=%.4f existing_contracts=%s contract_multiplier=%s min_contracts=%s",
            self.symbol,
            self.td_mode,
            self.leverage,
            equity,
            self.position_contracts,
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
            current_pos = await self.fetch_position_contracts()
            self.position_contracts = current_pos if current_pos > 0 else self.position_contracts + contracts
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
                    tp_price=self.price_to_str(intent.tp_price),
                    message=f"entry_filled_but_tp_failed: {tp.message}",
                    entry_filled=True,
                    tp_ok=False,
                )
            return LiveTradeResult(
                ok=True,
                action=intent.intent_type,
                order_id=order_id,
                tp_order_id=tp.tp_order_id,
                contracts=self.decimal_to_str(contracts),
                tp_price=self.price_to_str(intent.tp_price),
                message="market order placed and take-profit protected",
                entry_filled=True,
                tp_ok=True,
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
            current_pos = await self.fetch_position_contracts()
            if current_pos > 0:
                self.position_contracts = current_pos
        except Exception:
            logger.exception("Failed to refresh position before replacing TP")

        if self.position_contracts <= 0:
            return LiveTradeResult(False, intent.intent_type, None, None, "0", self.price_to_str(intent.tp_price), "no position to protect")

        if self.tp_order_id:
            try:
                await self.request("POST", "/api/v5/trade/cancel-order", {"instId": self.symbol, "ordId": self.tp_order_id})
            except Exception:
                logger.exception("Cancel previous TP failed; will try to place a new reduce-only TP anyway")

        close_side = "sell" if intent.side == "LONG" else "buy"
        body: dict[str, Any] = {
            "instId": self.symbol,
            "tdMode": self.td_mode,
            "side": close_side,
            "ordType": "limit",
            "px": self.price_to_str(intent.tp_price),
            "sz": self.decimal_to_str(self.position_contracts),
            "reduceOnly": "true",
        }
        pos_side = self.pos_side(intent.side)
        if pos_side:
            body["posSide"] = pos_side

        res = await self.request("POST", "/api/v5/trade/order", body)
        tp_order_id = self.extract_order_id(res)
        self.tp_order_id = tp_order_id
        return LiveTradeResult(
            True,
            intent.intent_type,
            None,
            tp_order_id,
            self.decimal_to_str(self.position_contracts),
            self.price_to_str(intent.tp_price),
            "take-profit replaced",
            entry_filled=False,
            tp_ok=True,
        )

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
        res = await self.request("GET", f"/api/v5/account/positions?instId={self.symbol}")
        total = Decimal("0")
        for item in res.get("data", []):
            if item.get("instId") == self.symbol:
                total += abs(Decimal(str(item.get("pos", "0"))))
        return total

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
        method = method.upper()
        body = "" if method == "GET" else json.dumps(payload or {}, separators=(",", ":"))
        headers = self.headers(method, endpoint, body)
        async with aiohttp.ClientSession() as session:
            if method == "GET":
                async with session.get(self.base_url + endpoint, headers=headers, timeout=10) as resp:
                    res = await resp.json()
            else:
                async with session.post(self.base_url + endpoint, headers=headers, data=body, timeout=10) as resp:
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
        lots = (raw_contracts / self.contract_precision).to_integral_value(rounding=ROUND_DOWN)
        contracts = lots * self.contract_precision
        if contracts < self.min_contracts:
            raise RuntimeError(f"Order size {contracts} contracts is below minimum {self.min_contracts}")
        return contracts

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
