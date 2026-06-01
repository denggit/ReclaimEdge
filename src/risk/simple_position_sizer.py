from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SimplePositionSizerConfig:
    dry_run_equity_usdt: float = 1000.0
    layer_margin_pct: float = 0.03
    leverage: float = 50.0
    layer_multiplier_step: float = 0.15

    @classmethod
    def from_env(cls) -> "SimplePositionSizerConfig":
        return cls(
            dry_run_equity_usdt=float(os.getenv("DRY_RUN_EQUITY_USDT", "1000")),
            layer_margin_pct=float(os.getenv("LAYER_MARGIN_PCT", "0.03")),
            leverage=float(os.getenv("LEVERAGE", "50")),
        )

    @classmethod
    def from_account_equity(cls, account_equity_usdt: float) -> "SimplePositionSizerConfig":
        return cls(
            dry_run_equity_usdt=account_equity_usdt,
            layer_margin_pct=float(os.getenv("LAYER_MARGIN_PCT", "0.03")),
            leverage=float(os.getenv("LEVERAGE", "50")),
        )


@dataclass(frozen=True)
class PositionSize:
    margin_usdt: float
    notional_usdt: float
    eth_qty: float
    layer_index: int
    layer_multiplier: float


class SimplePositionSizer:
    def __init__(self, config: SimplePositionSizerConfig):
        self.config = config

    def update_account_equity(self, account_equity_usdt: float) -> None:
        self.config = SimplePositionSizerConfig(
            dry_run_equity_usdt=account_equity_usdt,
            layer_margin_pct=self.config.layer_margin_pct,
            leverage=self.config.leverage,
            layer_multiplier_step=self.config.layer_multiplier_step,
        )

    @property
    def account_equity_usdt(self) -> float:
        return self.config.dry_run_equity_usdt

    def calculate(self, price: float, layer_index: int = 1) -> PositionSize:
        safe_layer_index = max(int(layer_index), 1)
        multiplier = 1.0 + (safe_layer_index - 1) * self.config.layer_multiplier_step
        base_margin = self.config.dry_run_equity_usdt * self.config.layer_margin_pct
        margin = base_margin * multiplier
        notional = margin * self.config.leverage
        eth_qty = notional / price if price > 0 else 0.0
        return PositionSize(
            margin_usdt=margin,
            notional_usdt=notional,
            eth_qty=eth_qty,
            layer_index=safe_layer_index,
            layer_multiplier=multiplier,
        )
