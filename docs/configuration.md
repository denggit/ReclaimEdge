# ReclaimEdge Configuration Guide

## A08+ Configuration Split

As of A08, ReclaimEdge uses a **two-layer configuration model**:

| Layer | Location | Purpose |
|-------|----------|---------|
| **.env** | `.env` (copy from `.env.example`) | Secrets, live gate, Trader env, runtime orchestration, report/email, temporary legacy fallback |
| **Symbol TOML** | `config/symbols/*.toml` | Per-symbol strategy/trading parameters (BOLL, CVD, TP, Sidecar, position sizing) |

### Key Files

| File | Role |
|------|------|
| `config/symbols/ETH-USDT-SWAP.toml` | **Default live strategy source** after A08 |
| `config/symbols/sample.toml` | Template for future symbols in later phases |
| `.env.example` | Reference for all `.env`-owned configuration |

## Default Behaviour (A08)

- **`RECLAIM_USE_SYMBOL_TOML` defaults to `true`.**
- When `true`, all strategy/trading parameters are loaded from `config/symbols/ETH-USDT-SWAP.toml`.
- The TOML is loaded once at startup — never on the tick/strategy path.
- `account_equity_usdt` from the live OKX account overrides `dry_run_equity_usdt` at startup.

## What to Edit?

| You want to change … | Edit … |
|----------------------|--------|
| BOLL window | `config/symbols/ETH-USDT-SWAP.toml` → `[market].boll_window` |
| CVD burst windows | TOML → `[cvd]` |
| TP ratios (three-stage) | TOML → `[tp]` |
| Sidecar settings | TOML → `[sidecar]` |
| Position sizing / equity | TOML → `[capital]` |
| OKX API keys | `.env` → `OKX_API_KEY`, `OKX_SECRET_KEY`, `OKX_PASSPHASE` |
| LIVE_TRADING on/off | `.env` → `LIVE_TRADING` |
| Report time | `.env` → `DAILY_REPORT_TIME`, `WEEKLY_SUMMARY_TIME` |
| Temporarily use old env strategy params | `.env` → `RECLAIM_USE_SYMBOL_TOML=false` |

## .env Still Controls

- **`LIVE_TRADING`** — the master live/dry-run gate. `symbol_config.symbol.live_trading` in TOML is data only and does **not** control the real trading switch.
- **OKX credentials** — `OKX_API_KEY`, `OKX_SECRET_KEY`, `OKX_PASSPHASE`.
- **Trader environment** — `OKX_INST_ID`, `OKX_TD_MODE`, `OKX_POS_SIDE_MODE`, `LEVERAGE`, `OKX_BASE_URL`, `MAX_LIVE_EQUITY_USDT`.
- **Runtime orchestration** — `RECLAIM_RUN_MODE`, `RECLAIM_SYMBOLS`, `RECLAIM_SYMBOL_CONFIG_DIR`, `RECLAIM_RUNTIME_DIR`.
- **Feature flags** — `RECLAIM_USE_SYMBOL_TOML`.
- **Email / reporting** — `EMAIL_*`, `SMTP_*`, `DAILY_REPORT_TIME`, `WEEKLY_SUMMARY_*`.
- **Worker / queue tuning** — `STRATEGY_TICK_QUEUE_MAXSIZE`, `EXECUTION_QUEUE_MAXSIZE`, `POSITION_SYNC_SECONDS`, etc.

## TOML/env Consistency

When `RECLAIM_USE_SYMBOL_TOML=true` (the default), the live entrypoint checks that these `.env` values match their TOML counterparts:

| .env | TOML field |
|------|-----------|
| `OKX_INST_ID` | `[symbol].inst_id` |
| `OKX_TD_MODE` | `[market].td_mode` |
| `OKX_POS_SIDE_MODE` | `[market].pos_side_mode` |
| `LEVERAGE` | `[capital].leverage` |

**A mismatch causes startup failure** with a clear error message listing every inconsistency. This prevents dangerous silent divergence between what Trader configures on OKX and what the strategy/sizer assumes.

## Legacy .env Strategy Parameters

The `.env.example` file retains all legacy strategy parameters in a clearly marked section. These parameters are **ignored by default** after A08 because `RECLAIM_USE_SYMBOL_TOML=true`. They only take effect when `RECLAIM_USE_SYMBOL_TOML=false`.

Prefer editing `config/symbols/ETH-USDT-SWAP.toml` instead.

## OKX_PASSPHASE

The project uses `OKX_PASSPHASE` (not `OKX_PASSPHRASE`). This is a deliberate historical spelling preserved across the codebase. Do not use `OKX_PASSPHRASE` — it is ignored.

## Symbol Support

- **Currently only `ETH-USDT-SWAP` is supported.** Attempting to set `RECLAIM_SYMBOLS=BTC-USDT-SWAP` or any other symbol will cause a startup error.
- BTC / SOL support will be added in a future F-phase after safety work is complete.
- Do not create `config/symbols/BTC-USDT-SWAP.toml` with `enabled = true` yet.

## Safety Rules

1. **Never set `LIVE_TRADING=true` before verifying TOML/env consistency.** Run in dry-run mode first.
2. **`LEVERAGE` in `.env` must match `[capital].leverage` in TOML.**
3. **`OKX_INST_ID` must match `[symbol].inst_id`.**
4. **`OKX_TD_MODE` must match `[market].td_mode`.**
5. **`OKX_POS_SIDE_MODE` must match `[market].pos_side_mode`.**
6. **Do not create `BTC-USDT-SWAP.toml` as enabled yet.**
7. **Keep `sidecar.tp_rate_limit_fail_action = "HALT_ONLY"`.**
8. **Keep `risk.order_failure_market_exit_delay_seconds >= 1800`.**
