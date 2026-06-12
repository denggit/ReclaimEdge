# BOLL + CVD Reclaim Strategy V1

## 1. Strategy Positioning

This strategy is not a pure BOLL breakout alert anymore. It is a layered execution strategy built on top of independent market-state modules.

Core idea:

- Price moves outside the 15m BOLL band.
- The system enters a focused monitoring state.
- If CVD shows aggressive-flow exhaustion or reversal while price can no longer continue in the breakout direction, the strategy opens a contrarian mean-reversion position.
- Take profit targets the BOLL middle band.
- The target can be adjusted every 15 minutes as the BOLL middle band updates.

The first version should run in dry-run / paper mode before live trading.

---

## 2. Module Boundaries

### 2.1 BOLL Market State Module

Path suggestion:

```text
src/monitors/boll_band_breakout_monitor.py
```

Responsibilities:

- Maintain 15m BOLL snapshot.
- Support live-candle mode.
- Know whether price is inside, above upper band, or below lower band.
- Emit BOLL state and breakout events.

Non-responsibilities:

- No CVD logic.
- No position logic.
- No order execution.
- No email sending.

---

### 2.2 CVD Indicator Module

Path suggestion:

```text
src/indicators/cvd_tracker.py
```

Responsibilities:

- Consume OKX trade ticks.
- Convert buy trades into positive delta.
- Convert sell trades into negative delta.
- Maintain fast and slow rolling CVD windows.
- Detect aggressive-flow reclaim or rejection.

Non-responsibilities:

- No BOLL judgment.
- No position logic.
- No order execution.

---

### 2.3 Strategy Decision Module

Path suggestion:

```text
src/strategies/boll_cvd_reclaim_strategy.py
```

Responsibilities:

- Combine BOLL state and CVD state.
- Decide whether a long/short setup is valid.
- Decide whether a new layer can be opened.
- Emit trade intents.

Non-responsibilities:

- No direct OKX API order call.
- No SMTP email sending.
- No raw websocket handling.

---

### 2.4 Risk and Position Sizing Module

Path suggestion:

```text
src/risk/position_sizer.py
```

Responsibilities:

- Read account equity.
- Calculate per-layer margin.
- Calculate notional exposure.
- Convert notional exposure into ETH quantity / OKX order size.
- Enforce maximum layers and maximum total margin usage.

Core formula:

```text
per_layer_margin = equity * layer_margin_pct
per_layer_notional = per_layer_margin * leverage
eth_qty = per_layer_notional / current_price
```

For example:

```text
layer_margin_pct = 3%
leverage = 50x
per_layer_notional = equity * 0.03 * 50 = equity * 1.5
```

So each layer is approximately 150% of account equity in notional exposure.

---

### 2.5 Execution Module

Path suggestion:

```text
src/execution/okx_executor.py
```

Responsibilities:

- Convert strategy intents into OKX orders.
- Place open orders.
- Place or amend take-profit orders.
- Query positions and account state.
- Support dry-run mode.

Non-responsibilities:

- No signal calculation.
- No BOLL calculation.
- No CVD calculation.

---

### 2.6 Notification Module

Use existing module:

```text
src/utils/email_sender.py
```

Responsibilities:

- Send alerts.
- Send dry-run trade notifications.
- Send order failure notifications.

Non-responsibilities:

- No signal calculation.
- No position logic.

---

## 3. Long Setup Logic

The long setup is activated only after price moves below the lower BOLL band.

### 3.1 Monitoring State

Enter `LOWER_OUTSIDE_MONITORING` when:

```text
boll_switch_on = True
price < boll_lower
```

While in this state, monitor CVD and price behavior.

---

### 3.2 Long Entry Trigger A: CVD Reclaim

Open long when:

```text
price < boll_lower
fast_cvd crosses from <= 0 to > 0
buy_ratio >= min_buy_ratio
price has not made a new low for stall_seconds
```

Meaning:

- Price is still outside the lower band.
- Aggressive flow has shifted from sell-dominant to buy-dominant.
- Price has stopped falling temporarily.

Signal name:

```text
LOWER_FAST_CVD_RECLAIM_LONG
```

---

### 3.3 Long Entry Trigger B: Absorption-style Divergence

Open long when:

```text
price < boll_lower
fast_cvd keeps increasing
price cannot make meaningful new lows
```

Meaning:

- Aggressive selling or buying pressure is being absorbed.
- Flow improves but price no longer extends lower.

A practical first version can define this as:

```text
fast_cvd_current > fast_cvd_previous
lowest_price_current >= previous_lowest_price - max_new_low_tolerance_pct
```

Signal name:

```text
LOWER_CVD_PRICE_STALL_LONG
```

---

## 4. Short Setup Logic

The short setup is symmetrical.

Enter `UPPER_OUTSIDE_MONITORING` when:

```text
boll_switch_on = True
price > boll_upper
```

Open short when:

```text
price > boll_upper
fast_cvd crosses from >= 0 to < 0
sell_ratio >= min_sell_ratio
price has not made a new high for stall_seconds
```

Signal name:

```text
UPPER_FAST_CVD_REJECT_SHORT
```

Or:

```text
price > boll_upper
fast_cvd keeps decreasing
price cannot make meaningful new highs
```

Signal name:

```text
UPPER_CVD_PRICE_STALL_SHORT
```

---

## 5. Layering Logic

After the first long entry:

Open another long layer only when:

```text
current_price <= last_entry_price * (1 - gap_pct)
AND long setup conditions are valid again
AND layer_count < max_layers
AND total_margin_used_pct < max_total_margin_pct
```

Gap_pct uses linear mode:

```text
ADD_GAP_MODE=linear
ADD_GAP_BASE_PCT=0.003   (L2 base)
ADD_GAP_STEP_PCT=0.001   (per-layer increment)
L2: 0.3%, L3: 0.4%, L4: 0.5%, ... (linear, no upper bound)
```

For short:

```text
current_price >= last_entry_price * (1 + gap_pct)
AND short setup conditions are valid again
```

---

## 6. Position Sizing

Default parameters:

```text
layer_margin_pct = 0.03
leverage = 50
```

Per-layer notional:

```text
per_layer_notional = account_equity * layer_margin_pct * leverage
```

ETH quantity:

```text
eth_qty = per_layer_notional / current_price
```

The actual OKX order size must be converted using instrument metadata. Do not hard-code contract size.

---

## 7. Take Profit Logic

For long:

```text
take_profit_price = boll_middle
```

For short:

```text
take_profit_price = boll_middle
```

Adjust take-profit every 15 minutes when a new 15m BOLL middle band is available.

Rule:

```text
If position is open and new 15m BOLL middle changes, amend TP order to the latest middle band.
```

---

## 8. Mandatory Risk Guards

These guards are required before live trading:

```text
DRY_RUN=true by default
MAX_LAYERS=3
MAX_TOTAL_MARGIN_PCT=0.12
MAX_DAILY_LOSS_PCT=0.05
HARD_STOP_PCT optional but strongly recommended
ORDER_COOLDOWN_SECONDS=10
```

Reason:

At 50x leverage and 3% margin per layer, each layer is about 150% notional exposure. Multiple layers can create very large directional exposure quickly.

---

## 9. Suggested Environment Parameters

```env
# CVD
ENABLE_CVD_RECLAIM_ALERT=true
CVD_FAST_WINDOW_SECONDS=8
CVD_SLOW_WINDOW_SECONDS=30
CVD_MIN_BUY_RATIO=0.56
CVD_MIN_SELL_RATIO=0.56
CVD_STALL_SECONDS=2
CVD_PRICE_STALL_TOLERANCE_PCT=0.0005

# Layering
LAYER_MARGIN_PCT=0.03
LEVERAGE=50
ADD_GAP_MODE=linear
ADD_GAP_BASE_PCT=0.003
ADD_GAP_STEP_PCT=0.001
MAX_LAYERS=3
MAX_TOTAL_MARGIN_PCT=0.12

# Execution
DRY_RUN=true
ORDER_COOLDOWN_SECONDS=10
TP_UPDATE_INTERVAL_SECONDS=900
```

---

## 10. First Implementation Order

1. Extend CVD tracker to support fast/slow windows and buy/sell ratio.
2. Add a pure strategy module that emits trade intents only.
3. Add dry-run executor.
4. Add email notification for trade intents.
5. Add OKX live executor only after dry-run logs are stable.
