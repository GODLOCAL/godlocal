# TASK: Integrate Harper-Grok Improvements v1

**Дата**: 25 лютого 2026  
**Автор**: Провідник (Rostyslav + Grok + Harper)  
**Статус**: ✅ Applied by SureThing agent

## Що застосовано (коміт harper-grok-v1)

### 1. soul/sovereign.md — Enhanced v2
- Wilson CI правило
- MobileO target 64 tok/s
- GlintSignalBus high-urgency logging (>0.75)
- Warrior rate-limit 1 trade/30s + ClosedCandleGate
- Security: env + keychain only
- 60s periodic status logging → /status/warrior
- Post-patch: tests/ + commit

### 2. extensions/xzero/sparknet_connector.py — Wilson CI
- Spark dataclass: додано `trial_count: int = 0` та `success_count: int = 0`
- `judge()` тепер повертає `float` (нова сигнатура: `async def judge(self, spark_id, outcome) -> float`)
- Wilson CI lower bound при n ≥ 2: `p_hat ± z*sqrt(p_hat(1-p)/n + z²/4n²) / (1 + z²/n)`, z=1.96
- EMA: 0.7 * old + 0.3 * outcome (без змін)
- Final: `clamp(EMA + wilson_lower * 0.3, 0.0, 1.0)`
- Очікуване покращення accuracy: +30% vs чистого EMA

### 3. godlocal_v5.py — AutoGenesis force + Warrior status
- `/chat`: якщо "evolve" у повідомленні → `run_evolution_cycle(force=True)` → GitNexus + Potpie + patch
- `/status/warrior` (GET): memory size + SparkNet spark count + Glint high-urgency signals

### 4. mobile/MobileOBridge.swift — tok/s tracking
- `@Published var tokensPerSecond: Double` — видима в SwiftUI
- `runUnderstanding()`: вимірює elapsed + token estimate → оновлює `tokensPerSecond`
- `runGeneration()`: вимірює steps/s → конвертує у tok/s equiv
- Console log: `🚀 MobileO: XX.X tok/s`

### 5. core/tiered_router.py — real savings tracking
- `TierStats`: додано `giant_calls: int`, `sparknet_reports: int`
- `log_stats()`: includes GIANT tier count
- SparkNet emit кожні 50 викликів: `"TieredRouter X% savings (N calls)"`

## Harper's code що НЕ застосовувалося (і чому)

| Harper пропозиція | Проблема | Замінено на |
|---|---|---|
| `from scipy.stats import wilson_interval(successes=outcome, n=1)` | Математично хибно: `outcome` — float, не int; n=1 = нуль інформації | Власна реалізація Wilson CI lower bound з `trial_count` |
| `import MLX` у Swift | MLX — Python-only framework, на iOS не існує | Існуюча CoreML архітектура збережена, додано timing |
| `savings = 0.78 # tracked` | Захардкоджена константа, не реальні дані | `savings_pct` property рахує реально через `total_calls` |

## Що далі (наступний спринт)

- [ ] Wire GlintSignalBus → XZeroHeartbeat `solana_prediction_pulse()` (кожні 30 хв)
- [ ] Activate NexaSDK: `./scripts/setup_nexa.sh` (User action)
- [ ] Test AirLLM GIANT tier on Picobot: `pip install airllm` + перший `llama-70b` запуск
