# Account Capital Allocator Design for ETH + BTC Runtime

## 1. Background

当前 ReclaimEdge 已经完成 F 阶段：

* ETH-USDT-SWAP 是当前主实盘 symbol。
* BTC-USDT-SWAP 已经有 TOML 配置，但 `enabled=false`、`live_trading=false`。
* BTC 当前只能 config-check / dry-run preview，不会启动 worker，不会创建 Trader，不会连接 OKX，不会下单。
* 下一阶段 G 要解决 ETH/BTC 多 worker 共用同一个 OKX 账户的问题。
* 最大风险不是单个 worker 算错，而是两个 worker 各自以为账户还有资金，然后同时开仓、同时加仓，把账户总风险放大。

G 阶段不能直接进入 BTC dry-run 或 BTC 实盘。必须先把账户资金管理、leader/follower 规则、完整 layer 计划、动态执行权限、sidecar 资金释放、跨 worker 并发处理等规则设计清楚。

---

## 2. Core Goal

G 阶段的目标：

* 允许未来 ETH worker 和 BTC worker 共用同一个 OKX 账户。
* 每个 worker 仍然只负责自己的交易信号和订单执行。
* 账户级资金和风险由 Account Capital Allocator 统一判断。
* 不允许两个 worker 各自独立按账户权益百分比开仓。
* 不允许两个币同时自由打满完整 layer 计划。
* 不改变每个仓位计划中每一层的张数结构。
* 不让 allocator 进入 tick path，避免影响 tick 处理延迟。
* 在 BTC 真正实盘前，先让 ETH 以 shadow mode / enforce mode 逐步接入 allocator，确认不会破坏当前 ETH 实盘行为。

---

## 3. Key Design Principle

核心原则：

```text
完整计划 ≠ 完整执行权限
plan 固定
permission 动态
张数不变
限制可变
```

解释：

* 每个 symbol 开首仓时，可以生成完整 8 层 layer 计划。
* 这张计划表里的每一层张数固定，不因为账户权益变化、另一个币平仓、副仓止盈而改变。
* 实际允许执行到第几层，是动态判断的。
* 如果账户里出现 pressure leader，follower 会被限制最大可执行层数、加仓价格距离、冻结时间。
* follower 虽然有完整计划，但不代表可以一路打到 layer8。
* leader 平仓后，follower 可以恢复更大的执行权限，但原始 layer 张数计划仍然不变。
* 账户权益变多，只能让 allocator 放宽 permission，不能重新放大已有 plan 的每层张数。
* 账户权益变少，也不能让已有 plan 的张数缩水；只能让 allocator 拒绝下一层新增风险。

---

## 4. Terminology

### 4.1 Position Plan

Position Plan 是开首仓时生成的一整轮仓位计划。

它包括：

* main position layer1-layer8 每层张数。
* 本轮 `plan_max_layers`。
* 首仓张数 `base_main_contracts`。
* sidecar 是否启用。
* sidecar 可用预算。
* 当前 symbol 是本轮已有仓位，还是 flat。

一旦 Position Plan 生成，本轮内每层张数不升级、不缩水。

例子：

```text
layer1 = 1.00
layer2 = 1.15
layer3 = 1.30
layer4 = 1.45
layer5 = 1.60
layer6 = 1.75
layer7 = 1.90
layer8 = 2.05
```

如果 layer1 是 1 张，那么本轮后续层数永远按这张表执行。不会因为另一个币平仓、sidecar 止盈、账户权益增加而改成更大张数。

---

### 4.2 Permission

Permission 是当前允许这个 symbol 实际执行到第几层。

Permission 可以随着账户状态变化而变化。

它可以控制：

* 当前最多允许执行到第几层。
* 当前是否允许新开仓。
* 当前是否允许主仓加仓。
* 当前是否允许 sidecar 新开腿。
* 当前加仓价格距离是否需要变宽。
* 当前加仓冻结时间是否需要变长。

Permission 变化不改变 Position Plan 里的张数。

简单说：

```text
Position Plan 决定：如果允许加，这一层加多少张。
Permission 决定：现在到底允不允许加。
```

---

### 4.3 Pressure Leader

Pressure Leader 是当前账户里最有压力的 symbol。

第一版规则：

* 不是谁先开首仓谁永远是 leader。
* 谁先进入较高风险层数，例如 first reaches layer3，谁成为 pressure leader。
* 谁先到 layer3，谁成为 leader。
* leader 不平仓，不主动切换。
* 如果 leader 平仓，另一个仍持仓的 symbol 成为新的 leader。
* 如果两个 symbol 都 <= layer2，可以没有 leader。
* 如果所有 symbol 都 flat，也没有 leader。

leader 自己继续按自己的 Position Plan 正常管理。

leader 的加仓张数、加仓价格距离、冻结时间，不因为 follower 存在而被主动放大或缩小。

---

### 4.4 Follower

当账户里已有 pressure leader 时，其他持仓 symbol 是 follower。

follower 的规则：

* follower 的 layer 张数计划不变。
* follower 的执行权限会变保守。
* follower 的加仓价格距离可以变宽。
* follower 的冻结时间可以变长。
* follower 的最大可执行层数会被限制。
* 当 leader 到达高层，例如 layer5+，follower 禁止新增风险。
* follower 仍然可以止盈、止损、减仓、平仓、撤单、补保护单。

---

### 4.5 New Risk

“新增风险”包括：

* 新开仓。
* 主仓加仓。
* sidecar 新开腿。
* 增加合约张数的动作。

“新增风险”不包括：

* 止盈。
* 止损。
* 减仓。
* 平仓。
* 撤风险单。
* 补保护单。

风险降低动作永远优先，不应被 allocator 阻止。

---

## 5. Leader / Follower Rules

### 5.1 No Leader State

当 ETH 和 BTC 都 <= layer2：

```text
ETH used_layers <= 2
BTC used_layers <= 2
```

规则：

* 没有 pressure leader。
* 两边都可以正常管理。
* 加仓价格距离使用原配置。
* 加仓冻结时间使用原配置。
* 两个 symbol 都可以从 layer2 加到 layer3。
* 但如果两个 worker 同时要加 layer3，需要通过账本文件锁决定顺序。
* 谁先成功写入 layer3，谁成为 leader。
* 第二个 worker 再拿锁时，必须读取第一个 worker 已写入的新状态，再按 follower 规则重新判断。

---

### 5.2 Leader Reaches Layer3

当某个 symbol 先到 layer3：

```text
leader used_layers = 3
```

规则：

* leader 正常按自己的 Position Plan 管理。
* leader 的张数、加仓距离、冻结时间不因为 follower 存在而改变。
* follower 张数计划不变。
* follower 新增风险开始变保守。
* follower 加仓价格距离乘以 1.5。
* follower 加仓冻结时间可以乘以 1.5。
* follower 最大可执行层数默认限制为 5。
* follower 仍然可以止盈、止损、减仓、平仓、撤单、补保护单。

默认参数：

```text
leader used_layers = 3:
  follower.permission_max_layers = 5
  follower.add_gap_multiplier = 1.5
  follower.add_freeze_multiplier = 1.5
```

这些参数后续可以配置化，但第一版先用明确默认值，避免实现时摇摆。

---

### 5.3 Leader Reaches Layer4

当 leader 到 layer4：

```text
leader used_layers = 4
```

规则：

* leader 仍然正常按原计划。
* follower 张数计划不变。
* follower 加仓价格距离乘以 2.0。
* follower 加仓冻结时间可以乘以 2.0。
* follower 最大可执行层数默认限制为 4。
* follower 仍然可以止盈、止损、减仓、平仓、撤单、补保护单。

默认参数：

```text
leader used_layers = 4:
  follower.permission_max_layers = 4
  follower.add_gap_multiplier = 2.0
  follower.add_freeze_multiplier = 2.0
```

---

### 5.4 Leader Reaches Layer5+

当 leader 到 layer5 或更高：

```text
leader used_layers >= 5
```

规则：

* follower 禁止新开仓。
* follower 禁止主仓继续加仓。
* follower 禁止 sidecar 新开腿。
* follower 仍然允许止盈、止损、减仓、平仓、撤单、补保护单。
* 如果 follower 已经有仓，继续管理已有仓位，但不能新增风险。

默认参数：

```text
leader used_layers >= 5:
  follower.no_new_entry = true
  follower.no_add_layer = true
  follower.no_new_sidecar_leg = true
```

---

### 5.5 Leader Closes

当 leader 平仓后：

* 如果另一个 symbol 仍然持仓，它成为新的 leader。
* 新 leader 的原始 Position Plan 不变。
* 新 leader 可以恢复更大的执行权限。
* 如果账本显示账户资金足够，新 leader 可以恢复到自己的原始 `plan_max_layers`。
* 新 leader 的加仓价格距离和冻结时间可以恢复正常。
* 但绝对不能因为 leader 平仓、账户权益增加、副仓止盈，而重新放大已有 Position Plan 的每层张数。

例子：

BTC 原本是 follower：

```text
BTC planned_main_contracts = [0.020, 0.023, 0.026, 0.029, ...]
BTC permission_max_layers = 4
BTC add_gap_multiplier = 2.0
```

ETH leader 平仓后，BTC 成为新 leader。

可以恢复：

```text
BTC permission_max_layers = 8
BTC add_gap_multiplier = 1.0
BTC add_freeze_multiplier = 1.0
```

但 BTC 的 `planned_main_contracts` 不变。

---

### 5.6 All Symbols Flat

当所有 symbols 都 FLAT：

* 没有 leader。
* 没有 follower。
* 下一次谁先到 layer3，谁成为 pressure leader。
* 新仓位会重新生成新的 Position Plan。

---

## 6. Full Plan vs Execution Permission

Position Plan 是完整计划。

Execution Permission 是当前允许执行到哪一步。

例子：

BTC 开仓时生成完整 8 层计划：

```text
layer1 = 0.020
layer2 = 0.023
layer3 = 0.026
layer4 = 0.029
layer5 = 0.032
layer6 = 0.035
layer7 = 0.038
layer8 = 0.041
```

但如果 ETH 已经是 leader layer4，那么 BTC 虽然有完整 layer 表，但 permission 可能只有：

```text
allowed_max_layer = 4
add_gap_multiplier = 2.0
add_freeze_multiplier = 2.0
```

因此 BTC 不能执行 layer5-layer8，除非 leader 平仓后 allocator 重新放宽 permission。

重点：

* 已有 plan 不因 permission 变化而改变。
* permission 放宽只允许执行原计划中更靠后的层。
* permission 放宽不重新计算更大的 layer 张数。
* permission 收紧也不缩小已有 plan 的张数，只是拒绝执行更高层。

---

## 7. Capital Share Design

第一版不要做 ETH/BTC 固定死分配。

不要设计成：

```text
ETH 永远 50%
BTC 永远 50%
```

原因：

* 资金利用率低。
* 很多时候只有一个币有机会。
* 当前策略普遍只打 1-3 层，少数到 6 层。
* 固定分配会浪费空闲 symbol 的资金空间。
* 开仓时生成完整计划，不等于开仓时锁死全部资金。

建议：

* 不在 FLAT 阶段锁死 ETH/BTC 资金份额。
* 每个 symbol 开仓时生成完整张数计划。
* allocator 只在每次新增风险前检查当前账户是否允许执行下一层。
* follower 的最大可执行层数动态限制。
* 全局必须有账户级最大计划风险限制。
* 两个币不能同时自由打满 layer8。
* follower 想执行更高层，必须看 leader 状态、账户空间和当前 permission。

第一版建议参数：

```text
single_symbol_full_main_plan:
  entry_margin_pct = 4%
  max_layers = 8
  layer_multiplier_step = 0.15
  theoretical_layer_sum = 12.2
  theoretical_main_margin_pct = 48.8%

global_main_plan_cap:
  suggested = 60% to 70%

follower_permission:
  when leader layer3:
    max_allowed_layer = 5
    add_gap_multiplier = 1.5
    add_freeze_multiplier = 1.5

  when leader layer4:
    max_allowed_layer = 4
    add_gap_multiplier = 2.0
    add_freeze_multiplier = 2.0

  when leader layer5+:
    no_new_entry = true
    no_add_layer = true
    no_new_sidecar_leg = true
```

说明：

* 以上是第一版默认建议参数。
* 后续可以配置化。
* 实现时应先写成清晰常量或 config 字段，不要散落在交易逻辑里。
* allocator 判断账户空间时，必须同时看已有 main 仓位、sidecar 仓位、pending order、leader/follower 状态。

---

## 8. Sidecar Rules

sidecar 也必须纳入同一个账户账本。

第一版规则：

* ETH 第一版可以保留 sidecar。
* BTC 第一版禁用 sidecar。
* sidecar 开仓会增加 `sidecar_used_margin_usdt`。
* sidecar 提前止盈后，释放对应资金。
* 释放资金回到账户池。
* 释放资金可以提高账户可用空间。
* 但释放资金不能改变已有主仓 Position Plan 的每层张数。
* 释放资金能不能被另一个 symbol 用于新增风险，要由 allocator 根据 leader/follower 状态判断。
* leader layer5+ 时，即使 sidecar 释放了资金，follower 仍然不能新增风险。
* sidecar 止盈释放资金后，follower 如果原本被限制最大层数，只有在 leader 状态变化或账户风险下降后，permission 才能放宽。

简单说：

```text
sidecar 释放资金可以增加账户空间。
sidecar 释放资金不能放大已有主仓张数计划。
sidecar 释放资金不能绕过 leader/follower 限制。
```

---

## 9. Allocator Runtime Location

第一版 allocator 的运行位置：

* 第一版 allocator 不做新进程。
* 第一版 allocator 不做常驻线程。
* 第一版 allocator 不是 supervisor 的内部服务。
* 第一版 allocator 是普通 Python 模块。
* worker 在 execution path 调用 allocator。
* allocator 绝对不能运行在 tick path。
* tick path 只产生交易意图，不读写资金账本。
* execution worker 在真正下单前调用 allocator。
* allocator 对外可以提供 async 接口。
* 内部文件 IO 可以用 `asyncio.to_thread()` 或等价方式，避免阻塞 event loop。
* 本地文件锁和 JSON 读写延迟通常远小于 OKX 下单网络延迟，但仍然禁止放入 tick path。

建议未来模块名：

```text
src/portfolio/capital_ledger.py
src/portfolio/capital_allocator.py
src/portfolio/position_plan.py
src/portfolio/leader_follower.py
```

这些只是设计建议，G00 不创建这些代码文件。

---

## 10. Ledger Storage

第一版使用：

```text
runtime/portfolio/capital_ledger.json
runtime/portfolio/capital_ledger.lock
```

原因：

* 简单。
* 重启后状态仍在。
* ETH/BTC worker 都能读写。
* 容易人工排查。
* 不需要 Redis / 数据库 / 网络服务。
* 用文件锁保证两个 worker 同时开仓时不会同时认为自己是 leader。

账本写入必须是原子写入，不能写一半损坏。

建议账本包含字段：

```json
{
  "version": 1,
  "updated_ms": 0,
  "leader_symbol": null,
  "global_no_new_entry": false,
  "symbols": {
    "ETH-USDT-SWAP": {
      "state": "FLAT",
      "side": null,
      "used_layers": 0,
      "position_plan_id": null,
      "planned_main_contracts": [],
      "base_main_contracts": "0",
      "plan_max_layers": 8,
      "permission_max_layers": 8,
      "add_gap_multiplier": "1.0",
      "add_freeze_multiplier": "1.0",
      "main_used_margin_usdt": "0",
      "sidecar_enabled": true,
      "sidecar_used_margin_usdt": "0"
    },
    "BTC-USDT-SWAP": {
      "state": "FLAT",
      "side": null,
      "used_layers": 0,
      "position_plan_id": null,
      "planned_main_contracts": [],
      "base_main_contracts": "0",
      "plan_max_layers": 8,
      "permission_max_layers": 8,
      "add_gap_multiplier": "1.0",
      "add_freeze_multiplier": "1.0",
      "main_used_margin_usdt": "0",
      "sidecar_enabled": false,
      "sidecar_used_margin_usdt": "0"
    }
  }
}
```

字段名后续开发时可以微调，但含义必须保留。

必须保留的含义：

* 当前 leader 是谁。
* 每个 symbol 当前是否 FLAT / OPEN。
* 每个 symbol 当前 used_layers。
* 每个 symbol 的完整 Position Plan。
* 每个 symbol 当前 permission。
* 每个 symbol 当前 add gap / freeze multiplier。
* 每个 symbol 当前 main used margin。
* 每个 symbol 当前 sidecar used margin。
* BTC 第一版 sidecar disabled。
* 全局是否禁止新开仓。

---

## 11. Concurrent Open / Add Handling

两个 worker 同时开仓或加仓时：

* 两个 worker 都调用 allocator。
* allocator 竞争同一个 ledger lock。
* 谁先拿到锁，谁先读最新账本并写入。
* 第二个 worker 拿到锁后，必须读取第一个 worker 已写入的新状态。
* 所以不会出现两个 worker 同时认为自己是 leader。
* 如果两个都从 layer2 几乎同时想加 layer3，谁先成功写入 layer3，谁成为 leader。
* 另一个再判断时，会看到对方已经是 leader，然后按 follower 规则处理。
* 如果 follower 当前 permission 不允许执行下一层，allocator 应该拒绝，不应该缩小张数下单。
* 如果 allocator 拒绝，需要记录原因，方便后续复盘。

例子：

```text
ETH layer2
BTC layer2
```

ETH 和 BTC 同时想加 layer3。

如果 ETH 先拿到锁：

```text
ETH -> layer3
leader_symbol = ETH-USDT-SWAP
```

BTC 再拿锁时看到：

```text
leader_symbol = ETH-USDT-SWAP
leader used_layers = 3
```

BTC 变成 follower。

BTC 是否能加 layer3，要看 follower permission：

```text
permission_max_layers >= 3
add_gap_multiplier = 1.5
add_freeze_multiplier = 1.5
```

如果 BTC 当前价格距离满足 follower 的加宽距离，才允许加；否则拒绝。

---

## 12. Worker Integration Principles

后续接入时：

* tick path 不调用 allocator。
* strategy 只产生 open/add/reduce intent。
* execution path 处理 intent。
* execution path 下单前必须调用 allocator。
* allocator 拒绝时，不下单，只记录原因。
* allocator 通过时，worker 按原计划张数下单。
* allocator 不负责下单，只负责判断和账本更新。
* 下单成功后更新 used_layers / used_margin。
* 下单失败或撤单后释放对应 pending 状态。
* 平仓后更新 symbol state。
* sidecar TP 后释放 sidecar margin。
* allocator 不能发 OKX 请求。
* allocator 不能发邮件。
* allocator 不能直接改策略信号。
* allocator 只判断账户级是否允许新增风险。

推荐执行流程：

```text
tick path:
  1. 收到 tick
  2. 策略快速判断
  3. 如果出现交易意图，丢到 execution queue
  4. 立刻返回，不读写资金账本

execution path:
  1. 收到 open/add/sidecar intent
  2. 调用 allocator
  3. allocator 拿 ledger lock
  4. 读取最新 capital_ledger.json
  5. 判断 leader/follower/permission/account space
  6. 写回账本中的 pending/permission/leader 状态
  7. 释放锁
  8. 如果允许，worker 才下单
  9. 下单成功/失败后再更新账本
```

---

## 13. BTC First Version Constraints

BTC 第一版约束：

* BTC 第一版仍然 `enabled=false`。
* BTC 第一版不实盘。
* BTC 第一版不启用 sidecar。
* BTC 第一版先做 paper/dry-run。
* BTC 真实下单必须等 allocator、ledger、leader/follower、shared private write limiter 全部稳定后再开。
* BTC 小仓实盘必须有单独开关，不能因为配置检查通过就自动启用。
* BTC 即使进入 dry-run，也只能模拟申请 permission、生成 plan、记录 allocator decision，不得真实下单。
* BTC sidecar 进入长期评估，不属于第一版目标。

---

## 14. Updated G Stage Plan

```text
G 阶段：ETH + BTC 共享账户资金管理 / dry-run / 实盘前安全层

[x] G00 Account Capital Allocator 设计文档
[ ] G01 账户资金账本 CapitalLedger：JSON 文件 + 文件锁
[ ] G02 PositionPlan：开仓时生成完整 layer 计划，张数固定
[ ] G03 Leader/Follower Permission：动态限制 follower 最大层数、加仓距离、冻结时间
[ ] G04 Allocator dry-run checker：只判断，不接真实下单
[ ] G05 ETH 接入 allocator shadow mode：只记录判断，不改变 ETH 实盘行为
[ ] G06 ETH 接入 allocator enforce mode：ETH 下单前必须通过 allocator
[ ] G07 SharedPrivateWriteLimiter：ETH/BTC 共用 OKX private write 限流
[ ] G08 BTC worker paper/dry-run：只模拟，不真实下单
[ ] G09 BTC 主仓小仓实盘开关：BTC sidecar 仍禁用
[ ] G10 BTC sidecar 评估：长期任务，第一版不启用
```

---

## 15. Non-Goals

G00 设计文档不做以下事情：

* 不启用 BTC。
* 不启动 BTC worker。
* 不创建 BTC Trader。
* 不连接 OKX。
* 不改 ETH 实盘行为。
* 不改下单逻辑。
* 不改 TP / SL / Sidecar / DME。
* 不新增运行代码。
* 不新增测试。
* 不引入 Redis / SQLite / 独立服务进程。

这些内容留到后续 G01-G10 分阶段实现。

---

## 16. Final Summary

最终设计原则：

```text
1. 每个 symbol 开仓时生成完整 Position Plan。
2. Position Plan 里的每层张数固定，不升级、不缩水。
3. 当前能执行到哪一层由 Permission 动态决定。
4. 谁先到 layer3，谁成为 pressure leader。
5. leader 不平仓，不主动切换。
6. follower 根据 leader 层数限制最大可执行层数。
7. follower 根据 leader 层数加宽加仓价格距离。
8. follower 根据 leader 层数延长冻结时间。
9. leader layer5+ 时，follower 禁止新增风险。
10. leader 平仓后，另一个仍持仓 symbol 可以成为新 leader。
11. sidecar 资金纳入账本，止盈后释放，但不改变主仓 plan。
12. allocator 是普通模块，不是独立进程。
13. allocator 只能在 execution path 调用，绝对不能进入 tick path。
14. allocator 使用 JSON 账本 + 文件锁解决 ETH/BTC 并发开仓/加仓。
15. BTC 第一版只做 paper/dry-run，不实盘，不启用 sidecar。
```
