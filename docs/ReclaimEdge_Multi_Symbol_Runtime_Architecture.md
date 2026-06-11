# ReclaimEdge Multi-Symbol Runtime 架构升级方案

> 目标：在不破坏当前 ETH 单品种稳定交易逻辑的前提下，把 ReclaimEdge 逐步升级为可安全运行 BTC + ETH，未来再扩展 SOL 的多 symbol 架构。
>
> 核心原则：先做运行时架构升级，不急着直接多币种实盘。第一阶段只把现有 ETH 交易进程子进程化，父进程负责监管。

---

## 1. 背景

当前 ReclaimEdge 是 OKX `ETH-USDT-SWAP` 单品种实盘交易系统。

当前 live 入口：

```text
scripts/run_boll_cvd_live.py
```

当前核心链路：

```text
BollBandBreakoutMonitor
    -> CvdTracker
    -> BollCvdShockReclaimStrategy / BollCvdReclaimStrategy
    -> TradeIntent
    -> ExecutionCommandProcessor
    -> Trader / OKX REST
    -> Account Sync / LiveStateStore / LiveTradeJournal / RollingLossGuard
```

当前系统已经具备：

```text
1. BOLL + CVD Shock Reclaim 入场
2. Three-Stage Runner
3. Middle Runner
4. BOLL15/BOLL20 Middle Bucket Split
5. Sidecar 副仓
6. Delayed Market Exit 30 分钟人工窗口
7. Halt Mode
8. Live State / Journal / Report
9. RollingLossGuard
10. OKX private write limiter
```

未来目标：

```text
ETH-USDT-SWAP
BTC-USDT-SWAP
未来可能 SOL-USDT-SWAP
```

但当前代码本质仍是单 symbol runtime。多币种不能简单把 `inst_id` 改成列表。

---

## 2. 总体架构原则

采用 **Control Plane / Data Plane** 分离：

```text
父进程 = Control Plane：监管、配置、全局状态、通知、报告、组合风控
子进程 = Data Plane：单 symbol 行情、策略、执行、account sync
```

### 2.1 父进程不碰 tick path

父进程绝对不要：

```text
接收 tick
分发 tick
计算 CVD
计算 BOLL
调用 strategy.on_tick()
决定入场
决定 TP/SL
```

这些全部留在子进程。父进程不能成为延迟瓶颈。

### 2.2 子进程只管一个 symbol

```text
ETH worker 只处理 ETH-USDT-SWAP
BTC worker 只处理 BTC-USDT-SWAP
```

子进程不读取其他 symbol 的 state，不判断组合风险，不写其他 symbol 的 journal。

### 2.3 父子进程只低频通信

父子进程之间只做低频通信：

```text
heartbeat
event outbox
global halt
lifecycle command
```

不要每个 tick 做 IPC。

---

## 3. 父进程职责

父进程建议命名为：

```text
ReclaimSupervisor
```

负责：

```text
1. 读取 .env
2. 读取 RECLAIM_SYMBOLS
3. 找到每个 symbol 对应配置文件
4. 启动 symbol 子进程
5. 监控子进程 heartbeat
6. 子进程异常退出后重启或 global halt
7. 维护 global_halt.json
8. 读取子进程事件 outbox
9. 统一发邮件
10. 聚合日报 / 周报
11. 后续做 PortfolioRiskGuard
12. 后续管理共享 OKX private write limiter
```

父进程禁止：

```text
处理 tick
调用 strategy.on_tick()
替子进程下单
直接改子进程 state
逐 tick 做组合风控
```

---

## 4. 子进程职责

子进程建议命名为：

```text
SymbolWorker
```

每个子进程负责一个 symbol。

负责：

```text
1. 当前 symbol websocket
2. 当前 symbol BOLL monitor
3. 当前 symbol CVD tracker
4. 当前 symbol strategy.on_tick()
5. 当前 symbol TradeIntent
6. 当前 symbol execution queue
7. 当前 symbol Trader / OKX order
8. 当前 symbol account sync
9. 当前 symbol TP progress / flat settlement / DME
10. 当前 symbol state store
11. 当前 symbol trade journal
12. 当前 symbol heartbeat 写入
13. 当前 symbol event outbox 写入
14. 低频读取 global halt
```

子进程禁止：

```text
读取其他 symbol 的 state
写其他 symbol 的 journal
判断 portfolio-level 风险
高频依赖父进程
把 BTC/ETH 的日志混在一起
```

---

## 5. 推荐设计模式

### 5.1 Supervisor Pattern

用于父进程管理子进程。

建议模块：

```text
src/runtime/supervisor/process_supervisor.py
src/runtime/supervisor/child_process.py
src/runtime/supervisor/heartbeat_monitor.py
src/runtime/supervisor/child_restart_policy.py
```

职责：

```text
start_child(symbol)
stop_child(symbol)
restart_child(symbol)
shutdown_all()
poll_children()
```

典型逻辑：

```text
child dead
    -> restart count +1
    -> restart

restart count > limit
    -> write global halt
    -> send CRITICAL email
```

### 5.2 Actor / Process-per-Symbol Pattern

每个 symbol 是一个 actor：

```text
SymbolWorker[ETH-USDT-SWAP]
SymbolWorker[BTC-USDT-SWAP]
```

每个 actor 拥有独立：

```text
state
journal
logs
strategy
execution queue
account sync
DME
sidecar
```

actor 之间不共享内存。

### 5.3 Factory Pattern

新增：

```text
src/runtime/worker/symbol_worker_factory.py
```

负责构造：

```text
BollBandBreakoutMonitor
CvdTracker
BollCvdShockReclaimStrategy
Trader
ExecutionCommandProcessor
LiveStateStore
LiveTradeJournal
RollingLossGuard
```

避免 `run_symbol_worker.py` 变成巨型文件。

### 5.4 Repository Pattern

配置、状态、journal、heartbeat、events 都通过 repository 读写。

建议：

```text
SymbolConfigRepository
LiveStateRepository
TradeJournalRepository
HeartbeatRepository
EventOutboxRepository
GlobalHaltRepository
```

好处：

```text
文件路径隔离
原子写入统一
测试容易 mock
后续从文件换 SQLite / Redis 不影响业务
```

### 5.5 Event Bus / Outbox Pattern

子进程不直接阻塞等待父进程处理事件。

子进程只写 durable event：

```text
runtime/events/ETH-USDT-SWAP.events.jsonl
```

父进程低频读取、去重、发邮件、聚合报告。

事件示例：

```json
{
  "event_id": "ETH-USDT-SWAP-1780000000000-sidecar_tp_failed",
  "ts_ms": 1780000000000,
  "symbol": "ETH-USDT-SWAP",
  "level": "CRITICAL",
  "event_type": "SIDECAR_TP_PLACE_FAILED",
  "position_id": "xxx",
  "halt_reason": "sidecar_tp_place_failed_delayed_market_exit_armed",
  "manual_intervention_required": true,
  "payload": {
    "side": "SHORT",
    "contracts": "2.47",
    "error": "50011 rate limit"
  }
}
```

### 5.6 Command Pattern

父进程对子进程使用低频命令文件。

```text
runtime/commands/ETH-USDT-SWAP.command.json
runtime/global_halt.json
```

命令类型：

```text
PAUSE_ENTRY
RESUME_ENTRY
SYMBOL_HALT
ACCOUNT_HALT
GRACEFUL_STOP
```

子进程每 1 秒读取一次即可。

### 5.7 State Machine Pattern

明确两层状态机。

Symbol 状态机：

```text
RUNNING
SYMBOL_HALTED
DME_ARMED
DME_WAITING_FLAT
DME_FAILED
FLAT
```

Supervisor 状态机：

```text
STARTING
RUNNING
CHILD_STALE
CHILD_RESTARTING
GLOBAL_HALTED
STOPPING
```

### 5.8 Circuit Breaker Pattern

用于子进程重启和 API 异常。

```text
child 5 分钟内崩溃 3 次
    -> 不再重启
    -> global halt
    -> 发邮件

OKX private API 连续失败 N 次
    -> symbol halt 或 account halt
```

### 5.9 Token Bucket / Leaky Bucket Pattern

用于共享 OKX private write limiter。

第一阶段可以先不做，但设计要预留：

```text
SharedPrivateWriteLimiter.acquire()
```

短期可用：

```text
SQLite lock
file lock
```

后续可以换 Redis。

### 5.10 Adapter Pattern

OKX、Email、文件系统都应该作为 adapter。

```text
OkxPrivateClientAdapter
EmailSenderAdapter
FileHeartbeatAdapter
FileEventOutboxAdapter
```

核心业务不要直接依赖底层 IO 细节。

---

## 6. 推荐模块结构

### 6.1 Supervisor 模块

```text
src/runtime/supervisor/
    supervisor_app.py
    supervisor_config.py
    process_supervisor.py
    child_process.py
    heartbeat_monitor.py
    child_restart_policy.py
    global_halt_manager.py
    supervisor_event_loop.py
    supervisor_paths.py
```

### 6.2 Worker 模块

```text
src/runtime/worker/
    symbol_worker_app.py
    symbol_worker_config.py
    symbol_worker_factory.py
    heartbeat_writer.py
    event_emitter.py
    global_halt_reader.py
    worker_paths.py
```

### 6.3 Config 模块

```text
src/config/
    env_runtime_config.py
    symbol_config.py
    symbol_config_loader.py
    symbol_config_validator.py
    symbol_config_mapper.py
```

说明：

```text
env_runtime_config.py 只读 .env 的全局配置和隐私配置。
symbol_config.py 定义 symbol config schema。
symbol_config_loader.py 读取 config/symbols/*.toml。
symbol_config_mapper.py 把 symbol config 映射到现有策略 / sizer / monitor / trader config。
```

推荐使用 TOML：

```text
config/symbols/ETH-USDT-SWAP.toml
config/symbols/BTC-USDT-SWAP.toml
```

原因：Python 标准库有 `tomllib`，不用引入 PyYAML。

### 6.4 IPC / Persistence 模块

```text
src/runtime/ipc/
    atomic_json.py
    jsonl_outbox.py
    file_lock.py
    command_file.py
```

功能：

```text
原子写 heartbeat
原子写 global halt
append event jsonl
避免半写文件
```

### 6.5 Alert / Report 模块

```text
src/runtime/alerts/
    parent_alert_router.py
    alert_deduper.py
    child_event_reader.py
    email_payload_builder.py
```

```text
src/runtime/reports/
    parent_report_scheduler.py
    symbol_journal_reader.py
    portfolio_report_builder.py
```

### 6.6 Portfolio 模块

```text
src/portfolio/
    portfolio_snapshot.py
    portfolio_risk_guard.py
    portfolio_position_reader.py
    portfolio_journal_aggregator.py
```

Phase 1 先不做交易拦截，只预留。

---

## 7. 配置分层

### 7.1 `.env`

`.env` 只放全局 / 隐私 / 启动编排配置：

```env
RECLAIM_RUN_MODE=live
RECLAIM_SYMBOLS=ETH-USDT-SWAP

RECLAIM_SYMBOL_CONFIG_DIR=config/symbols
RECLAIM_RUNTIME_DIR=runtime

OKX_API_KEY=xxx
OKX_SECRET_KEY=xxx
OKX_PASSPHASE=xxx

EMAIL_ENABLED=true
SMTP_HOST=xxx
SMTP_USER=xxx
SMTP_PASSWORD=xxx
ALERT_EMAIL_TO=xxx
```

`.env` 不应该继续堆：

```text
THREE_STAGE_TP1_RATIO
MIDDLE_BUCKET_SPLIT_FAST_RATIO
SIDECAR_TP_PCT
ADD_GAP
MAX_LAYERS
CVD_THRESHOLD
```

这些应该进入 symbol config。

### 7.2 Symbol TOML 配置

示例：

```toml
[symbol]
inst_id = "ETH-USDT-SWAP"
enabled = true

[market]
bar = "15m"
td_mode = "isolated"
pos_side_mode = "net"
contract_value = 0.1
min_contracts = 0.01
contract_precision = 0.01
price_precision = 0.01

[capital]
allocated_cash_pct = 0.40
layer_margin_pct = 0.04
leverage = 10
max_layers = 3

[entry]
add_gap_pct = 0.006
first_add_block_seconds = 3600
add_min_interval_seconds = 1800

[tp]
tp_min_net_profit_pct = 0.002
tp_boll_enabled = true
tp_boll_window = 15
three_stage_runner_enabled = true
three_stage_tp1_ratio = 0.70
three_stage_tp2_ratio = 0.20
three_stage_runner_ratio = 0.10
three_stage_tp2_use_structure_boll = true

[middle_bucket_split]
enabled = true
fast_ratio = 0.60
fast_sl_enabled = true
fast_sl_fee_buffer_pct = 0.001

[sidecar]
enabled = true
margin_pct = 0.02
tp_pct = 0.006
skip_first_layer = true

[risk]
rolling_loss_guard_enabled = true
symbol_daily_loss_limit_pct = 0.02
order_failure_market_exit_delay_seconds = 1800

[execution]
private_write_min_interval_seconds = 0.6
max_order_retries = 3
```

---

## 8. 父子进程通信

### 8.1 Heartbeat

路径：

```text
runtime/heartbeats/ETH-USDT-SWAP.heartbeat.json
```

内容：

```json
{
  "symbol": "ETH-USDT-SWAP",
  "pid": 12345,
  "status": "RUNNING",
  "ts_ms": 1780000000000,
  "last_tick_ts_ms": 1780000000000,
  "last_boll_candle_ts_ms": 1780000000000,
  "last_account_sync_ts_ms": 1780000000000,
  "trading_halted": false,
  "halt_reason": null,
  "has_position": true,
  "side": "SHORT",
  "layers": 1
}
```

频率建议：

```text
1~3 秒一次
```

### 8.2 Event Outbox

路径：

```text
runtime/events/ETH-USDT-SWAP.events.jsonl
```

只写重要事件：

```text
HALT
DME_ARMED
DME_FAILED
CHILD_STARTED
CHILD_STOPPING
ORDER_FAILURE
FLAT_SETTLED
DAILY_REPORT_READY
```

不要每 tick 写 event。

### 8.3 Global Halt

路径：

```text
runtime/global_halt.json
```

内容：

```json
{
  "enabled": true,
  "scope": "ACCOUNT",
  "reason": "portfolio_daily_loss_limit_hit",
  "created_ts_ms": 1780000000000
}
```

子进程每 1 秒低频读取一次即可。

---

## 9. Runtime 目录结构

推荐：

```text
runtime/
  state/
    live_state_ETH-USDT-SWAP.json
    live_state_BTC-USDT-SWAP.json

  journals/
    live_trades_ETH-USDT-SWAP.jsonl
    live_trades_BTC-USDT-SWAP.jsonl

  logs/
    live_ETH-USDT-SWAP.log
    live_BTC-USDT-SWAP.log
    supervisor.log

  heartbeats/
    ETH-USDT-SWAP.heartbeat.json
    BTC-USDT-SWAP.heartbeat.json

  events/
    ETH-USDT-SWAP.events.jsonl
    BTC-USDT-SWAP.events.jsonl

  commands/
    ETH-USDT-SWAP.command.json
    BTC-USDT-SWAP.command.json

  shared/
    okx_private_write_limiter.json
    okx_private_write_limiter.lock

  global_halt.json
```

---

## 10. 分阶段实施计划

### Phase 1：Supervisor + ETH child skeleton

Commit message 建议：

```text
feat(runtime): add supervisor and single-symbol worker process
```

目标：

```text
1. 新增父进程脚本
2. 新增子进程脚本
3. 父进程启动 ETH 子进程
4. 子进程继续跑现有 ETH live loop
5. 子进程写 heartbeat
6. 父进程监控 heartbeat
7. 父进程发现子进程退出 / stale 后发邮件
8. 支持优雅 shutdown
```

不做：

```text
不启动 BTC
不迁移所有策略配置
不重写交易逻辑
不迁移所有邮件
不做 portfolio guard
不做 shared limiter
```

目标运行形态：

```text
watchdog -> supervisor -> ETH child
```

---

### Phase 2：Per-symbol runtime paths

Commit message 建议：

```text
feat(runtime): isolate state journal logs heartbeat by symbol
```

目标：

```text
state path 带 symbol
journal path 带 symbol
log path 带 symbol
heartbeat path 带 symbol
event path 带 symbol
```

---

### Phase 3：Symbol config loader

Commit message 建议：

```text
feat(config): load per-symbol strategy config from config directory
```

目标：

```text
新增 config/symbols/ETH-USDT-SWAP.toml
新增 SymbolConfig dataclass
新增 symbol_config_loader
新增 symbol_config_validator
新增 symbol_config_mapper
把 TOML 映射到现有 config 对象
```

---

### Phase 4：Parent event outbox and alert router

Commit message 建议：

```text
feat(runtime): route child events through supervisor alert router
```

目标：

```text
子进程写 events jsonl
父进程读取 events
父进程 dedupe
父进程发邮件
父进程聚合重要事件
```

第一阶段可以保留子进程原有交易关键邮件，后续逐步迁移。

---

### Phase 5：BTC child

Commit message 建议：

```text
feat(runtime): enable BTC and ETH workers under supervisor
```

目标：

```text
RECLAIM_SYMBOLS=ETH-USDT-SWAP,BTC-USDT-SWAP
新增 BTC TOML config
父进程启动两个子进程
BTC 小仓运行
日志 / state / journal 完全隔离
```

---

### Phase 6：Shared private write limiter

Commit message 建议：

```text
feat(execution): add cross-process OKX private write limiter
```

目标：

```text
所有子进程 OKX private POST 前共享 limiter
支持 place / cancel / amend / algo cancel
用 file lock / SQLite lock 实现
不阻塞 tick path
```

---

### Phase 7：PortfolioRiskGuard

Commit message 建议：

```text
feat(risk): add portfolio-level risk guard for multi-symbol runtime
```

目标：

```text
父进程低频读取账户 equity / margin / positions
聚合所有 symbol journal
控制全账户总风险
触发 global halt
```

至少支持：

```text
GLOBAL_MAX_TOTAL_MARGIN_PCT
GLOBAL_MAX_DAILY_LOSS_PCT
GLOBAL_MAX_OPEN_SYMBOLS
GLOBAL_MAX_SAME_DIRECTION_SYMBOLS
```

---

## 11. 关键红线

```text
1. 父进程是监管层，不是交易层。
2. 子进程是交易层，不是组合风控层。
3. tick path 不能依赖父进程。
4. 每个 symbol 的 state/journal/log 必须隔离。
5. .env 只放隐私 / 全局 / 启动编排配置。
6. symbol 策略参数必须进 config/symbols/*.toml。
7. 第一阶段只做 ETH child，不直接上 BTC。
8. 多币种前必须有账户级资金预算和 shared private write limiter。
```

---

## 12. 推荐第一张开发票

第一张开发票建议：

```text
feat(runtime): add supervisor and single-symbol worker process
```

目标只做：

```text
1. 新增 scripts/run_reclaim_supervisor.py
2. 新增 scripts/run_symbol_worker.py
3. 新增 runtime/supervisor 基础模块
4. 新增 runtime/worker 基础模块
5. 父进程读取 RECLAIM_SYMBOLS
6. 父进程只启动 ETH child
7. ETH child 复用现有 live loop
8. child 写 heartbeat
9. supervisor 监控 heartbeat
10. supervisor 支持优雅 shutdown
11. supervisor 发现 child dead/stale 发邮件
```

禁止：

```text
1. 禁止改策略逻辑
2. 禁止改 TP/SL/DME/Sidecar
3. 禁止直接启动 BTC
4. 禁止迁移所有邮件
5. 禁止迁移所有配置
6. 禁止父进程处理 tick
7. 禁止执行 git commit
```
