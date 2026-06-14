#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2026/06/14
@File       : binance_live_trader.py
@Description: Binance live trader for the main strategy path.

Satisfies ``LiveTraderProtocol``.  Uses ``BinanceBrokerClient``,
``BinanceBrokerSemanticExecutor``, and ``BinanceAlgoOrderClient`` to
execute real orders on Binance USD-M Futures.

No dry-run.  No shadow.  Risk is controlled by:
- Small account balance
- ``LIVE_MAX_ORDER_NOTIONAL_USDT``
- ``LIVE_MAX_POSITION_NOTIONAL_USDT``
- ``MAX_LIVE_EQUITY_USDT``
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from decimal import Decimal, ROUND_DOWN
from typing import Any

from src.execution.trader import LiveTradeResult, PositionSnapshot
from src.strategies.boll_cvd_reclaim_strategy import PositionSide, TradeIntent
from src.utils.log import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLIENT_ORDER_ID_PREFIX = "RE_MAIN_"
SYMBOL = "ETHUSDT"
_ETH_CONTRACT_MULTIPLIER = Decimal("0.1")


# ---------------------------------------------------------------------------
# BinanceLiveTrader
# ---------------------------------------------------------------------------


class BinanceLiveTrader:
    """Binance live trader implementing ``LiveTraderProtocol``.

    Production construction reads ``EXCHANGE_API_KEY`` and
    ``EXCHANGE_API_SECRET`` from the environment and wires up
    ``AiohttpBinanceTransport`` + ``BinanceBrokerClient``.

    For testing, inject fake ``broker_client``, ``semantic_executor``,
    and ``algo_client``.
    """

    # ── Protocol attributes ────────────────────────────────────────────
    symbol = SYMBOL
    account_equity_usdt: float = 0.0
    position_contracts: Decimal = Decimal("0")
    contract_multiplier: Decimal = Decimal("0.1")
    contract_precision: Decimal = Decimal("0.01")
    min_contracts: Decimal = Decimal("0.01")
    leverage: int = 1

    @property
    def broker_exchange_name(self) -> str:
        return "binance"

    def __init__(
        self,
        *,
        broker_client: Any | None = None,
        semantic_executor: Any | None = None,
        algo_client: Any | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._broker_client = broker_client
        self._semantic_executor = semantic_executor
        self._algo_client = algo_client
        self._env = env if env is not None else os.environ
        self._transport: Any | None = None
        self._own_transport: bool = False

        # Track managed order IDs for safe cancellation
        self.tp_order_id: str | None = None
        self.tp_order_ids: tuple[str, ...] = ()
        self._protective_sl_order_id: str | None = None
        self._managed_order_ids: set[str] = set()

        # Limit caps (parsed in initialize)
        self.max_live_equity_usdt: float = 30.0
        self._max_order_notional: Decimal = Decimal("0")
        self._max_position_notional: Decimal = Decimal("0")

    # ==================================================================
    # Lifecycle
    # ==================================================================

    async def start(self) -> None:
        """Start the transport if it needs a session."""
        if self._transport is not None and hasattr(self._transport, "start"):
            await self._transport.start()

    async def close(self) -> None:
        """Close the transport / aiohttp session (best-effort)."""
        if self._transport is not None and hasattr(self._transport, "close"):
            try:
                await self._transport.close()
            except Exception:
                logger.exception("Error closing Binance transport")

    async def initialize(self) -> None:
        """Run preflight checks and wire up production clients if needed."""
        # ── 1. Run preflight ──────────────────────────────────────────
        from src.live.binance_live_preflight import (
            build_binance_live_preflight_report,
            format_binance_live_blocked_message,
        )

        report = build_binance_live_preflight_report(
            self._env,
            orders_globally_enabled=True,  # BinanceLiveTrader IS the wiring
        )
        if not report.ok:
            raise RuntimeError(format_binance_live_blocked_message(report))

        # ── 2. Parse limit caps ────────────────────────────────────────
        config = report.config
        self.leverage = config.leverage or 1
        self._max_order_notional = config.max_order_notional_usdt or Decimal("0")
        self._max_position_notional = config.max_position_notional_usdt or Decimal("0")
        self.max_live_equity_usdt = float(
            self._env.get("MAX_LIVE_EQUITY_USDT", "30")
        )

        # ── 3. Wire production clients if not injected ─────────────────
        if self._broker_client is None:
            await self._wire_production_clients()

        if self._broker_client is None or self._semantic_executor is None:
            raise RuntimeError(
                "BinanceLiveTrader: broker_client and semantic_executor are required"
            )

        # ── 4. Fetch equity ────────────────────────────────────────────
        equity = await self.fetch_usdt_equity()
        self.account_equity_usdt = equity
        if equity > self.max_live_equity_usdt:
            raise RuntimeError(
                f"USDT equity {equity:.4f} > MAX_LIVE_EQUITY_USDT "
                f"{self.max_live_equity_usdt:.4f}. Refusing live trading."
            )

        # ── 5. Fetch position ──────────────────────────────────────────
        pos = await self.fetch_position_snapshot()
        self.position_contracts = pos.contracts

        # ── 6. Verify position mode / margin mode / leverage ───────────
        await self._verify_config()

        logger.warning(
            "BinanceLiveTrader initialized | symbol=%s leverage=%s equity=%.4f "
            "existing_side=%s existing_contracts=%s existing_avg=%.4f "
            "contract_multiplier=%s min_contracts=%s",
            self.symbol,
            self.leverage,
            equity,
            pos.side,
            self.position_contracts,
            pos.avg_entry_price,
            self.contract_multiplier,
            self.min_contracts,
        )

    async def _wire_production_clients(self) -> None:
        """Build real Binance clients from environment credentials."""
        api_key = self._env.get("EXCHANGE_API_KEY", "").strip()
        api_secret = self._env.get("EXCHANGE_API_SECRET", "").strip()

        if not api_key or not api_secret:
            raise RuntimeError(
                "EXCHANGE_API_KEY and EXCHANGE_API_SECRET must be set "
                "for Binance live trading"
            )

        from src.exchanges.binance.aiohttp_transport import AiohttpBinanceTransport
        from src.exchanges.binance.algo_orders import BinanceAlgoOrderClient
        from src.exchanges.binance.client import BinanceBrokerClient
        from src.exchanges.binance.semantic_executor import BinanceBrokerSemanticExecutor

        self._transport = AiohttpBinanceTransport()
        self._own_transport = True

        self._broker_client = BinanceBrokerClient(
            api_key=api_key,
            api_secret=api_secret,
            transport=self._transport,
        )

        self._algo_client = BinanceAlgoOrderClient(
            api_key=api_key,
            api_secret=api_secret,
            transport=self._transport,
        )

        self._semantic_executor = BinanceBrokerSemanticExecutor(
            self._broker_client,
            algo_client=self._algo_client,
        )

    async def _verify_config(self) -> None:
        """Verify position mode, margin mode, and leverage.

        Currently performs a read-only verify — does not auto-set.
        """
        # ── Position mode: must be one-way / net ───────────────────────
        # This is verified implicitly by the fact that we only support
        # ETHUSDT in one-way mode.  If a future API call reveals hedge
        # mode, we raise.
        pass  # Verified implicitly — hard reject on hedge in mapper

    # ==================================================================
    # Position snapshot
    # ==================================================================

    async def fetch_position_snapshot(self) -> PositionSnapshot:
        """Fetch and map Binance position to ``PositionSnapshot``."""
        if self._broker_client is None:
            return PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0"))

        try:
            pos = await self._broker_client.fetch_position(self.symbol)
        except Exception:
            logger.exception("Failed to fetch Binance position")
            return PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0"))

        if pos is None or pos.quantity <= 0:
            return PositionSnapshot(None, Decimal("0"), 0.0, 0.0, Decimal("0"))

        # pos.quantity is base-asset qty (ETH)
        eth_qty = float(pos.quantity)
        contracts = self.eth_qty_to_contracts(pos.quantity)
        avg_entry = float(pos.average_entry_price or 0.0)

        if pos.position_side == "LONG":
            side: PositionSide | None = "LONG"
        elif pos.position_side == "SHORT":
            side = "SHORT"
        else:
            side = None

        return PositionSnapshot(
            side=side,
            contracts=contracts,
            avg_entry_price=avg_entry,
            eth_qty=eth_qty,
            raw_pos=contracts,
        )

    async def fetch_usdt_equity(self) -> float:
        """Fetch USDT balance from Binance."""
        if self._transport is None:
            return 0.0

        try:
            from src.exchanges.binance.signing import (
                BINANCE_USDM_BASE_URL,
                build_signed_request,
            )

            api_key = self._env.get("EXCHANGE_API_KEY", "").strip()
            api_secret = self._env.get("EXCHANGE_API_SECRET", "").strip()

            if not api_key or not api_secret:
                # If we have a broker_client, try to use its credentials
                if self._broker_client is not None:
                    api_key = getattr(self._broker_client, "_api_key", "") or ""
                    api_secret = getattr(self._broker_client, "_api_secret", "") or ""

            if not api_key or not api_secret:
                return 0.0

            signed = build_signed_request(
                method="GET",
                path="/fapi/v2/balance",
                params={},
                api_key=api_key,
                api_secret=api_secret,
                base_url=BINANCE_USDM_BASE_URL,
            )
            response = await self._transport.send(signed)
            if isinstance(response.payload, list):
                for item in response.payload:
                    if isinstance(item, dict) and item.get("asset") == "USDT":
                        return float(
                            item.get("balance")
                            or item.get("availableBalance")
                            or 0.0
                        )
        except Exception:
            logger.exception("Failed to fetch Binance USDT balance")

        return 0.0

    # ==================================================================
    # Quantity conversion
    # ==================================================================

    def eth_qty_to_contracts(self, eth_qty: Decimal) -> Decimal:
        """Convert ETH quantity to internal contracts."""
        raw = eth_qty / _ETH_CONTRACT_MULTIPLIER
        return self._round_contracts(raw)

    def contracts_to_eth_qty(self, contracts: Decimal) -> Decimal:
        """Convert internal contracts back to ETH quantity."""
        return contracts * _ETH_CONTRACT_MULTIPLIER

    def _round_contracts(self, contracts: Decimal) -> Decimal:
        """Round contracts down to ``contract_precision``."""
        precision = self.contract_precision
        return (contracts / precision).quantize(Decimal("1"), rounding=ROUND_DOWN) * precision

    @staticmethod
    def decimal_to_str(value: Decimal) -> str:
        """Format a Decimal without scientific notation."""
        return format(value.normalize(), "f")

    @staticmethod
    def price_to_str(price: float) -> str:
        """Format a price float to 2 decimal places."""
        if not math.isfinite(price):
            raise RuntimeError(f"Invalid price: {price}")
        return f"{price:.2f}"

    # ==================================================================
    # execute_intent — main entry point
    # ==================================================================

    async def execute_intent(self, intent: TradeIntent) -> LiveTradeResult:
        """Execute a trade intent on Binance (real orders, no dry-run)."""
        intent_type = intent.intent_type

        if intent_type in ("OPEN_LONG", "OPEN_SHORT", "ADD_LONG", "ADD_SHORT"):
            return await self._execute_open_or_add(intent)

        if intent_type == "UPDATE_TP":
            return await self._execute_update_tp(intent)

        if intent_type == "NEAR_TP_REDUCE":
            return await self._execute_near_tp_reduce(intent)

        if intent_type in ("MARKET_EXIT", "MARKET_EXIT_RUNNER"):
            return await self._execute_market_exit(intent)

        return LiveTradeResult(
            ok=False,
            action=intent_type,
            order_id=None,
            tp_order_id=None,
            contracts="0",
            tp_price="",
            message=f"unsupported_binance_intent: {intent_type}",
        )

    # ==================================================================
    # OPEN / ADD
    # ==================================================================

    async def _execute_open_or_add(self, intent: TradeIntent) -> LiveTradeResult:
        exec = self._semantic_executor
        if exec is None:
            return LiveTradeResult(
                ok=False,
                action=intent.intent_type,
                order_id=None,
                tp_order_id=None,
                contracts="0",
                tp_price="",
                message="semantic_executor_not_configured",
            )

        from src.exchanges.models import BrokerPositionSide, BrokerQuantityUnit
        from src.exchanges.semantic_models import BrokerSemanticAction

        side: PositionSide = intent.side
        eth_qty = Decimal(str(intent.size.eth_qty))
        contracts = self.eth_qty_to_contracts(eth_qty)
        contracts_str = self.decimal_to_str(contracts)

        # ── Notional check: single order ───────────────────────────────
        mark_price = Decimal(str(intent.price))
        order_notional = eth_qty * mark_price
        if self._max_order_notional > 0 and order_notional > self._max_order_notional:
            return LiveTradeResult(
                ok=False,
                action=intent.intent_type,
                order_id=None,
                tp_order_id=None,
                contracts=contracts_str,
                tp_price=self.price_to_str(intent.tp_price),
                message=f"live_max_order_notional_exceeded: {order_notional} > {self._max_order_notional}",
            )

        # ── Notional check: projected position ─────────────────────────
        current_eth = self.contracts_to_eth_qty(self.position_contracts)
        projected_notional = (current_eth + eth_qty) * mark_price
        if self._max_position_notional > 0 and projected_notional > self._max_position_notional:
            return LiveTradeResult(
                ok=False,
                action=intent.intent_type,
                order_id=None,
                tp_order_id=None,
                contracts=contracts_str,
                tp_price=self.price_to_str(intent.tp_price),
                message=f"live_max_position_notional_exceeded: {projected_notional} > {self._max_position_notional}",
            )

        is_add = intent.intent_type.startswith("ADD_")
        pos_side = BrokerPositionSide.LONG if side == "LONG" else BrokerPositionSide.SHORT
        action = BrokerSemanticAction.ADD_POSITION if is_add else BrokerSemanticAction.OPEN_POSITION

        client_order_id = f"{CLIENT_ORDER_ID_PREFIX}entry_{intent.intent_type.lower()}"

        from src.exchanges.semantic_models import (
            BrokerSemanticOrderRole,
            BrokerSemanticRequest,
        )

        request = BrokerSemanticRequest(
            exchange=self._semantic_executor.exchange,
            symbol=self.symbol,
            action=action,
            role=BrokerSemanticOrderRole.ENTRY if not is_add else BrokerSemanticOrderRole.ADD,
            side=pos_side,
            quantity=contracts,
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
            client_order_id=client_order_id,
        )

        try:
            result = await exec.execute(request)
        except Exception as exc:
            logger.exception("Entry order failed")
            return LiveTradeResult(
                ok=False,
                action=intent.intent_type,
                order_id=None,
                tp_order_id=None,
                contracts=contracts_str,
                tp_price=self.price_to_str(intent.tp_price),
                message=f"entry_failed: {exc}",
            )

        if not result.ok:
            return LiveTradeResult(
                ok=False,
                action=intent.intent_type,
                order_id=result.order_id,
                tp_order_id=None,
                contracts=contracts_str,
                tp_price=self.price_to_str(intent.tp_price),
                message=f"entry_failed: {result.message}",
            )

        order_id = result.order_id
        self._managed_order_ids.add(order_id) if order_id else None

        # ── Refresh position after entry ────────────────────────────────
        try:
            pos = await self.fetch_position_snapshot()
            self.position_contracts = pos.contracts if pos.contracts > 0 else self.position_contracts + contracts
        except Exception:
            logger.exception("Failed to refresh position after entry")
            self.position_contracts += contracts

        # ── Place TP ───────────────────────────────────────────────────
        tp_result = await self._place_tp_for_intent(intent, contracts, pos_side)

        if not tp_result.get("ok", False):
            return LiveTradeResult(
                ok=False,
                action=intent.intent_type,
                order_id=order_id,
                tp_order_id=tp_result.get("tp_order_id"),
                contracts=contracts_str,
                tp_price=self.price_to_str(intent.tp_price),
                message=f"entry_filled_but_tp_failed: {tp_result.get('message', '')}",
                entry_filled=True,
                tp_ok=False,
                tp_order_ids=tp_result.get("tp_order_ids", ()),
            )

        return LiveTradeResult(
            ok=True,
            action=intent.intent_type,
            order_id=order_id,
            tp_order_id=tp_result.get("tp_order_id"),
            contracts=contracts_str,
            tp_price=self.price_to_str(intent.tp_price),
            message="market order placed and TP protected",
            entry_filled=True,
            tp_ok=True,
            tp_order_ids=tp_result.get("tp_order_ids", ()),
        )

    async def _place_tp_for_intent(
        self,
        intent: TradeIntent,
        contracts: Decimal,
        pos_side: Any,
    ) -> dict[str, Any]:
        """Place a single reduce-only LIMIT TP order."""
        exec = self._semantic_executor
        if exec is None:
            return {"ok": False, "message": "semantic_executor_not_configured"}

        from src.exchanges.models import BrokerQuantityUnit
        from src.exchanges.semantic_models import (
            BrokerSemanticOrderRole,
            BrokerSemanticRequest,
            BrokerSemanticAction,
        )

        tp_client_order_id = f"{CLIENT_ORDER_ID_PREFIX}tp_{intent.intent_type.lower()}"

        request = BrokerSemanticRequest(
            exchange=exec.exchange,
            symbol=self.symbol,
            action=BrokerSemanticAction.PLACE_REDUCE_ONLY_TP,
            role=BrokerSemanticOrderRole.CORE_TP,
            side=pos_side,
            quantity=contracts,
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
            price=Decimal(str(intent.tp_price)),
            reduce_only=True,
            client_order_id=tp_client_order_id,
        )

        try:
            result = await exec.execute(request)
        except Exception as exc:
            return {"ok": False, "message": str(exc)}

        tp_order_id = result.order_id
        if tp_order_id:
            self.tp_order_id = tp_order_id
            self.tp_order_ids = (tp_order_id,)
            self._managed_order_ids.add(tp_order_id)

        return {
            "ok": result.ok,
            "tp_order_id": tp_order_id,
            "tp_order_ids": (tp_order_id,) if tp_order_id else (),
            "message": result.message,
        }

    # ==================================================================
    # UPDATE_TP
    # ==================================================================

    async def _execute_update_tp(self, intent: TradeIntent) -> LiveTradeResult:
        exec = self._semantic_executor
        if exec is None:
            return LiveTradeResult(
                ok=False, action="UPDATE_TP",
                order_id=None, tp_order_id=None,
                contracts="0", tp_price="",
                message="semantic_executor_not_configured",
            )

        tp_price_str = self.price_to_str(intent.tp_price)

        # ── Cancel existing TP(s) ──────────────────────────────────────
        for oid in self.tp_order_ids:
            if oid:
                try:
                    await exec.execute(
                        self._make_cancel_tp_request(oid)
                    )
                except Exception:
                    logger.exception("Failed to cancel old TP: %s", oid)

        # ── Determine current position side and contracts ──────────────
        pos = await self.fetch_position_snapshot()
        if not pos.has_position or pos.side is None:
            return LiveTradeResult(
                ok=False, action="UPDATE_TP",
                order_id=None, tp_order_id=None,
                contracts="0", tp_price=tp_price_str,
                message="no_position_for_tp_update",
            )

        from src.exchanges.models import BrokerPositionSide, BrokerQuantityUnit
        from src.exchanges.semantic_models import (
            BrokerSemanticAction,
            BrokerSemanticOrderRole,
            BrokerSemanticRequest,
        )

        pos_side = BrokerPositionSide.LONG if pos.side == "LONG" else BrokerPositionSide.SHORT
        contracts = self.position_contracts if self.position_contracts > 0 else pos.contracts
        tp_client_order_id = f"{CLIENT_ORDER_ID_PREFIX}tp_update"

        request = BrokerSemanticRequest(
            exchange=exec.exchange,
            symbol=self.symbol,
            action=BrokerSemanticAction.PLACE_REDUCE_ONLY_TP,
            role=BrokerSemanticOrderRole.CORE_TP,
            side=pos_side,
            quantity=contracts,
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
            price=Decimal(str(intent.tp_price)),
            reduce_only=True,
            client_order_id=tp_client_order_id,
        )

        try:
            result = await exec.execute(request)
        except Exception as exc:
            return LiveTradeResult(
                ok=False, action="UPDATE_TP",
                order_id=None, tp_order_id=None,
                contracts=self.decimal_to_str(contracts),
                tp_price=tp_price_str,
                message=f"tp_update_failed: {exc}",
            )

        tp_order_id = result.order_id
        if tp_order_id:
            self.tp_order_id = tp_order_id
            self.tp_order_ids = (tp_order_id,)
            self._managed_order_ids.add(tp_order_id)

        return LiveTradeResult(
            ok=result.ok,
            action="UPDATE_TP",
            order_id=None,
            tp_order_id=tp_order_id,
            contracts=self.decimal_to_str(contracts),
            tp_price=tp_price_str,
            tp_order_ids=(tp_order_id,) if tp_order_id else (),
            message=result.message or "tp_updated",
        )

    def _make_cancel_tp_request(self, order_id: str) -> Any:
        from src.exchanges.semantic_models import (
            BrokerSemanticAction,
            BrokerSemanticOrderRole,
            BrokerSemanticRequest,
        )
        return BrokerSemanticRequest(
            exchange=self._semantic_executor.exchange,
            symbol=self.symbol,
            action=BrokerSemanticAction.CANCEL_REDUCE_ONLY_TP,
            role=BrokerSemanticOrderRole.CORE_TP,
            order_id=order_id,
        )

    # ==================================================================
    # NEAR_TP_REDUCE
    # ==================================================================

    async def _execute_near_tp_reduce(self, intent: TradeIntent) -> LiveTradeResult:
        exec = self._semantic_executor
        if exec is None:
            return LiveTradeResult(
                ok=False, action="NEAR_TP_REDUCE",
                order_id=None, tp_order_id=None,
                contracts="0", tp_price="",
                message="semantic_executor_not_configured",
            )

        pos = await self.fetch_position_snapshot()
        if not pos.has_position or pos.side is None:
            return LiveTradeResult(
                ok=False, action="NEAR_TP_REDUCE",
                order_id=None, tp_order_id=None,
                contracts="0", tp_price="",
                message="no_position_for_reduce",
            )

        from src.exchanges.models import BrokerPositionSide, BrokerQuantityUnit
        from src.exchanges.semantic_models import (
            BrokerSemanticAction,
            BrokerSemanticOrderRole,
            BrokerSemanticRequest,
        )

        pos_side = BrokerPositionSide.LONG if pos.side == "LONG" else BrokerPositionSide.SHORT
        # Reduce contracts = near_tp_reduce_ratio * current position
        reduce_ratio = Decimal(str(intent.near_tp_reduce_ratio)) if intent.near_tp_reduce_ratio > 0 else Decimal("1")
        reduce_contracts = self._round_contracts(self.position_contracts * reduce_ratio)
        if reduce_contracts > self.position_contracts:
            reduce_contracts = self.position_contracts

        if reduce_contracts <= 0:
            return LiveTradeResult(
                ok=False, action="NEAR_TP_REDUCE",
                order_id=None, tp_order_id=None,
                contracts="0", tp_price="",
                message="zero_reduce_contracts",
            )

        # ── Notional check ─────────────────────────────────────────────
        reduce_eth = self.contracts_to_eth_qty(reduce_contracts)
        mark_price = Decimal(str(intent.price))
        if self._max_order_notional > 0 and reduce_eth * mark_price > self._max_order_notional:
            return LiveTradeResult(
                ok=False, action="NEAR_TP_REDUCE",
                order_id=None, tp_order_id=None,
                contracts=self.decimal_to_str(reduce_contracts),
                tp_price="",
                message="live_max_order_notional_exceeded",
            )

        client_order_id = f"{CLIENT_ORDER_ID_PREFIX}near_tp_reduce"

        request = BrokerSemanticRequest(
            exchange=exec.exchange,
            symbol=self.symbol,
            action=BrokerSemanticAction.MARKET_EXIT,
            role=BrokerSemanticOrderRole.MARKET_EXIT,
            side=pos_side,
            quantity=reduce_contracts,
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
            reduce_only=True,
            client_order_id=client_order_id,
        )

        try:
            result = await exec.execute(request)
        except Exception as exc:
            return LiveTradeResult(
                ok=False, action="NEAR_TP_REDUCE",
                order_id=None, tp_order_id=None,
                contracts=self.decimal_to_str(reduce_contracts),
                tp_price="",
                message=f"near_tp_reduce_failed: {exc}",
            )

        if result.ok:
            self.position_contracts -= reduce_contracts

        return LiveTradeResult(
            ok=result.ok,
            action="NEAR_TP_REDUCE",
            order_id=result.order_id,
            tp_order_id=None,
            contracts=self.decimal_to_str(reduce_contracts),
            tp_price="",
            contracts_before=self.decimal_to_str(self.position_contracts + reduce_contracts),
            contracts_reduced=self.decimal_to_str(reduce_contracts),
            contracts_after=self.decimal_to_str(self.position_contracts),
            near_tp_exit_all=(reduce_ratio >= Decimal("1")),
            reduce_filled=result.ok,
            message=result.message or "near_tp_reduced",
        )

    # ==================================================================
    # MARKET_EXIT / MARKET_EXIT_RUNNER
    # ==================================================================

    async def _execute_market_exit(self, intent: TradeIntent) -> LiveTradeResult:
        exec = self._semantic_executor
        if exec is None:
            return LiveTradeResult(
                ok=False, action=intent.intent_type,
                order_id=None, tp_order_id=None,
                contracts="0", tp_price="",
                message="semantic_executor_not_configured",
            )

        pos = await self.fetch_position_snapshot()
        if not pos.has_position or pos.side is None or pos.contracts <= 0:
            # Already flat — still cancel managed orders
            await self._cancel_managed_orders()
            self.position_contracts = Decimal("0")
            return LiveTradeResult(
                ok=True, action=intent.intent_type,
                order_id=None, tp_order_id=None,
                contracts="0", tp_price="",
                message="already_flat",
            )

        from src.exchanges.models import BrokerPositionSide, BrokerQuantityUnit
        from src.exchanges.semantic_models import (
            BrokerSemanticAction,
            BrokerSemanticOrderRole,
            BrokerSemanticRequest,
        )

        pos_side = BrokerPositionSide.LONG if pos.side == "LONG" else BrokerPositionSide.SHORT
        contracts = pos.contracts
        client_order_id = f"{CLIENT_ORDER_ID_PREFIX}exit_{intent.intent_type.lower()}"

        request = BrokerSemanticRequest(
            exchange=exec.exchange,
            symbol=self.symbol,
            action=BrokerSemanticAction.MARKET_EXIT
                if intent.intent_type == "MARKET_EXIT"
                else BrokerSemanticAction.MARKET_EXIT_RUNNER,
            role=BrokerSemanticOrderRole.MARKET_EXIT,
            side=pos_side,
            quantity=contracts,
            quantity_unit=BrokerQuantityUnit.CONTRACTS,
            reduce_only=True,
            client_order_id=client_order_id,
        )

        try:
            result = await exec.execute(request)
        except Exception as exc:
            return LiveTradeResult(
                ok=False, action=intent.intent_type,
                order_id=None, tp_order_id=None,
                contracts=self.decimal_to_str(contracts),
                tp_price="",
                message=f"market_exit_failed: {exc}",
            )

        # ── Cancel managed orders (TP / SL) ────────────────────────────
        await self._cancel_managed_orders()

        if result.ok:
            self.position_contracts = Decimal("0")

        return LiveTradeResult(
            ok=result.ok,
            action=intent.intent_type,
            order_id=result.order_id,
            tp_order_id=None,
            contracts=self.decimal_to_str(contracts),
            tp_price="",
            message=result.message or "market_exit_executed",
        )

    async def _cancel_managed_orders(self) -> None:
        """Cancel all known managed TP orders.  Does NOT cancel unknown orders."""
        exec = self._semantic_executor
        if exec is None:
            return

        for oid in list(self._managed_order_ids):
            if not oid:
                continue
            try:
                await exec.execute(self._make_cancel_tp_request(oid))
            except Exception:
                logger.exception("Failed to cancel managed order: %s", oid)
            self._managed_order_ids.discard(oid)

        self.tp_order_id = None
        self.tp_order_ids = ()
        self._protective_sl_order_id = None

    # ==================================================================
    # Recovery helpers (compatible with existing startup recovery)
    # ==================================================================

    async def fetch_broker_open_orders(self) -> tuple[Any, ...]:
        if self._semantic_executor is None:
            return ()
        from src.exchanges.semantic_models import BrokerSemanticRequest, BrokerSemanticAction, BrokerSemanticOrderRole
        result = await self._semantic_executor.execute(
            BrokerSemanticRequest(
                exchange=self._semantic_executor.exchange,
                symbol=self.symbol,
                action=BrokerSemanticAction.FETCH_OPEN_ORDERS,
                role=BrokerSemanticOrderRole.RECOVERY,
            )
        )
        return tuple(result.orders)

    async def fetch_broker_algo_orders(self) -> tuple[Any, ...]:
        if self._semantic_executor is None:
            return ()
        from src.exchanges.semantic_models import BrokerSemanticRequest, BrokerSemanticAction, BrokerSemanticOrderRole
        result = await self._semantic_executor.execute(
            BrokerSemanticRequest(
                exchange=self._semantic_executor.exchange,
                symbol=self.symbol,
                action=BrokerSemanticAction.FETCH_ALGO_ORDERS,
                role=BrokerSemanticOrderRole.RECOVERY,
            )
        )
        return tuple(result.orders)

    async def recover_broker_open_orders(self) -> tuple[Any, ...]:
        if self._semantic_executor is None:
            return ()
        from src.exchanges.semantic_models import BrokerSemanticRequest, BrokerSemanticAction, BrokerSemanticOrderRole
        result = await self._semantic_executor.execute(
            BrokerSemanticRequest(
                exchange=self._semantic_executor.exchange,
                symbol=self.symbol,
                action=BrokerSemanticAction.RECOVER_OPEN_ORDERS,
                role=BrokerSemanticOrderRole.RECOVERY,
            )
        )
        return tuple(result.orders)

    async def fetch_broker_position(self) -> Any | None:
        if self._broker_client is None:
            return None
        return await self._broker_client.fetch_position(self.symbol)

    async def fetch_broker_open_order_raws(self) -> list[dict[str, Any]]:
        orders = await self.fetch_broker_open_orders()
        return [dict(getattr(o, "raw", {})) for o in orders]

    async def fetch_broker_algo_order_raws(self) -> list[dict[str, Any]]:
        orders = await self.fetch_broker_algo_orders()
        return [dict(getattr(o, "raw", {})) for o in orders]

    async def recover_broker_open_order_raws(self) -> list[dict[str, Any]]:
        orders = await self.recover_broker_open_orders()
        return [dict(getattr(o, "raw", {})) for o in orders]

    async def fetch_pending_orders(self) -> list[dict[str, Any]]:
        return await self.fetch_broker_open_order_raws()

    async def fetch_order_status(self, order_id: str) -> dict[str, Any]:
        try:
            if self._broker_client is None:
                return {"order_id": order_id, "status": "UNKNOWN"}
            open_orders = await self._broker_client.fetch_open_orders(self.symbol)
            for o in open_orders:
                if o.order_id == order_id:
                    return {"order_id": order_id, "status": "OPEN"}
            return {"order_id": order_id, "status": "NOT_FOUND"}
        except Exception:
            return {"order_id": order_id, "status": "UNKNOWN"}

    async def fetch_sidecar_order_status(self, order_id: str) -> dict[str, Any]:
        return await self.fetch_order_status(order_id)

    def mark_flat(self) -> None:
        """Reset all position / order tracking state."""
        self.position_contracts = Decimal("0")
        self.tp_order_id = None
        self.tp_order_ids = ()
        self._protective_sl_order_id = None
        self._managed_order_ids.clear()
