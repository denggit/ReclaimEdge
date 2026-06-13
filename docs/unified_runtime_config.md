# Unified Runtime Config

## 目标

OKX 和 Binance 两个目录除了 `EXCHANGE` 和 API key / secret / passphrase 之外，其他参数保持一致。

所有业务逻辑读取同一个 `ExchangeRuntimeConfig` 对象，不再各自散落读取 env。

## 通用配置

```env
EXCHANGE=okx                 # okx 或 binance，默认 okx

TRADE_ASSET=ETH              # 只允许 ETH
QUOTE_ASSET=USDT             # 只允许 USDT
MARKET_TYPE=PERPETUAL        # 只允许 PERPETUAL

MARGIN_MODE=isolated         # 只允许 isolated
POSITION_MODE=net            # 只允许 net（one-way 语义）
LEVERAGE=20                  # 正整数，默认 20
KLINE_INTERVAL=15m           # 只允许 15m
```

## OKX 目录

```env
EXCHANGE=okx
EXCHANGE_API_KEY=your_okx_api_key
EXCHANGE_API_SECRET=your_okx_api_secret
EXCHANGE_API_PASSPHRASE=your_okx_passphrase
```

内部映射：

| canonical | OKX instrument ID |
|---|---|
| `ETH-USDT-PERP` | `ETH-USDT-SWAP` |

## Binance 目录

```env
EXCHANGE=binance
EXCHANGE_API_KEY=your_binance_api_key
EXCHANGE_API_SECRET=your_binance_api_secret
EXCHANGE_API_PASSPHRASE=           # Binance 无 passphrase，留空
```

内部映射：

| canonical | Binance symbol |
|---|---|
| `ETH-USDT-PERP` | `ETHUSDT` |

## 代码入口

```python
from src.exchanges.runtime_config import load_unified_runtime_config

rt = load_unified_runtime_config()

print(rt.exchange)          # ExchangeName.BINANCE
print(rt.canonical_symbol)  # "ETH-USDT-PERP"
print(rt.binance_symbol)    # "ETHUSDT"
print(rt.okx_inst_id)       # "ETH-USDT-SWAP"
print(rt.position_mode)     # "net"
print(rt.margin_mode)       # "isolated"
print(rt.kline_interval)    # "15m"
```

所有 API credentials 使用 `repr=False`，不会被意外日志泄露。

## Legacy OKX raw env

以下环境变量是旧 OKX 专用 raw 配置：

- `OKX_TD_MODE`
- `OKX_POS_SIDE_MODE`
- `OKX_INST_ID`
- `OKX_BAR`

新多平台目录 **不要** 依赖它们。统一 config 通过 `load_unified_runtime_config` 加载，不会消费这些 legacy 变量。
