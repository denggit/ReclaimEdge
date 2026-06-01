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
class InstrumentSpec:
    inst_id: str
    ct_val: Decimal
    lot_sz: Decimal
    min_sz: Decimal


@dataclass(frozen=True)
class LiveTradeResult:
    ok: bool
    action: str
    order_id: Optional[str]
    tp_order_id: Optional[str]
    contracts: str
    tp_price: str
    message: str


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
        self.base_url = os.getenv("OKX_BASE_URL", "https://www.okx.com")
        self.td_mode = os.getenv("OKX_TD_MODE", "isolated")
        self.leverage = os.getenv("LEVERAGE", "50")
        self.pos_side_mode = os.getenv("OKX_POS_SIDE_MODE", "net")
        self.live_trading = os.getenv("LIVE_TRADING", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
        self.max_live_equity_usdt = float(os.getenv("MAX_LIVE_EQUITY_USDT", "30"))

        self.api_key = OKX_CONFIG.get("api_key", "")
        self.secret_key = OKX_CONFIG.get("secret_key", "")
        self.passphrase = OKX_CONFIG.get("passphrase", "")

        self.instrument: InstrumentSpec | None = None
        self.tp_order_id: str | None = None
        self.position_contracts = Decimal("0")

        if not self.api_key or not self.secret_key or not self.passphrase:
            raise ValueError("OKX API config is incomplete. Check OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSPHASE.")
        if not self.live_trading:
            raise RuntimeError("LIVE_TRADING is not true. Refusing to initialize live trader.")

    async def initialize(self) -> None:
        self.instrument = await self.fetch_instrument_spec()
        equity = await self.fetch_usdt_equity()
        if equity > self.max_live_equity_usdt:
            raise RuntimeError(
                f"USDT equity {equity:.4f} > MAX_LIVE_EQUITY_USDT {self.max_live_equity_usdt:.4f}. Refusing live trading."
            )
        await self.set_leverage()
        logger.warning(
            "LIVE trader initialized | symbol=%s td_mode=%s leverage=%s equity=%.4f ctVal=%s lotSz=%s minSz=%s",
            self.symbol,
            self.td_mode,
            self.leverage,
            equity,
            self.instrument.ct_val,
            self.instrument.lot_sz,
            self.instrument.min_sz,
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
        self.position_contracts += contracts
        tp = await self.replace_take_profit(intent)
        return LiveTradeResult(True, intent.intent_type, order_id, tp.tp_order_id, self.decimal_to_str(contracts), self.price_to_str(intent.tp_price), "market order placed")

    async def replace_take_profit(self, intent: TradeIntent) -> LiveTradeResult:
        if self.position_contracts <= 0:
            current_pos = await self.fetch_position_contracts()
            self.position_contracts = current_pos
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
        return LiveTradeResult(True, intent.intent_type, None, tp_order_id, self.decimal_to_str(self.position_contracts), self.price_to_str(intent.tp_price), "take-profit replaced")

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

    async def fetch_instrument_spec(self) -> InstrumentSpec:
        url = f"{self.base_url}/api/v5/public/instruments?instType=SWAP&instId={self.symbol}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as resp:
                payload = await resp.json()
        if payload.get("code") != "0" or not payload.get("data"):
            raise RuntimeError(f"Failed to fetch instrument spec: {payload}")
        item = payload["data"][0]
        return InstrumentSpec(
            inst_id=item["instId"],
            ct_val=Decimal(str(item["ctVal"])),
            lot_sz=Decimal(str(item["lotSz"])),
            min_sz=Decimal(str(item["minSz"])),
        )

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
        if self.instrument is None:
            raise RuntimeError("Instrument spec not initialized")
        raw = eth_qty / self.instrument.ct_val
        lots = (raw / self.instrument.lot_sz).to_integral_value(rounding=ROUND_DOWN)
        contracts = lots * self.instrument.lot_sz
        if contracts < self.instrument.min_sz:
            raise RuntimeError(f"Order size {contracts} is below minSz {self.instrument.min_sz}")
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
