# AlgoForge Phase 1B 実装計画草案（v0.2）

**日付**: 2026-11-22  
**前提**: 統合設計書 v0.2（承認済み）、Phase 1A 互換性検証（条件付き受入）  
**状態**: 計画承認待ち（実装・実行未着手）  

---

## 1. Entry/Exit の対応範囲と根拠コード

- **根拠コード**: `src/research_pipeline/backtest/causal_fx_backtest.py` の `run_causal_fx_backtest` は `long_signal`, `short_signal`（新規エントリー boolean 配列）のみを受け取り、決済は SL/TP・最大保有時間・反転シグナル・週末強制決済で処理される（独立した `long_exit` / `short_exit` 配列入力は未実装）。
- **初期対応方針（Entry のみ）**:
  - AlgoForge 形式では `long: {"entry": {...}}`, `short: {"entry": {...}}` を必須とし、`exit` は省略または `null` / `{}` のみ許容する。
  - **非空の `exit` 条件が指定された場合**: 黙って無視したり SL/TP へ暗黙置換したりせず、**明示的に `UNSUPPORTED_EXIT_CONDITION` として棄却** し、Capability Request JSON へ記録する。

---

## 2. 初期対応 Strategy JSON 部分集合と条件評価マトリクス

- **スキーマ（`schema_version = "algoforge_v1"`）**: `{symbol, indicators, long: {entry}, short: {entry}, risk}`
- **初期対応指標**: `ema`, `sma`, `rsi`, `atr`, `bb`, `donchian`, `macd`, `adx`, `stochastic`, `roc`, `rvol`

| 条件種別 | Phase 1B-1 対応 | 評価方式 | 備考 |
|---|---|---|---|
| 単純比較 (`>`, `<`, `>=`, `<=`, `==`, `!=`) | **対応** | ベクトル化（逐次と100%一致検証） | 数値・列名比較 |
| 基本論理 (`and`, `or`, `not`) | **対応** | ベクトル化（逐次と100%一致検証） | 交換法則が成立する範囲で正規化 |
| `cross_above`, `cross_below` | **1B-1 参照検証のみ** | 逐次評価を参照実装として検証 | 欠損・初期状態確認後に対応判断 |
| `rolling`, `streak`, `regime` | **未対応（1B-2以降）** | `UNSUPPORTED_CONDITION` として保存 | Capability Request へ記録 |

---

## 3. StrategyRegistry の互換性保証（新旧完全分離）

- 既存 `pipeline_v1` の正規化・ハッシュ・キャッシュ検索経路は **一切変更しない**。
- `algoforge_v1` 用に独立した正規化経路（`canonicalize_algoforge_strategy`）を新設し、`indicators`（`id`, `type`, `params`）、`long.entry`、`short.entry`、`risk` を保持する。
- `evaluation_hash` に `schema_version="algoforge_v1"` と `unit_versions` を含め、旧形式と新形式で同一評価キャッシュを参照しないことをテストする。

---

## 4. 2段階実装計画（Phase 1B-1 と Phase 1B-2）

### Phase 1B-1：固定戦略の統合検証（今回の承認対象）
- **固定戦略（2件）**:
  1. `af_ema_cross_fixed`: `ema12 > ema48` (Long) / `ema12 < ema48` (Short)
  2. `af_rsi_bb_fixed`: `rsi14 < 30 AND close < bb20_lower` (Long) / `rsi14 > 70 AND close > bb20_upper` (Short)
- **テストケースと受入基準**:
  1. JSON 検証・`algoforge_v1` 正規化（非空 `exit` の明示的棄却と Capability Request 保存、旧 `pipeline_v1` とのハッシュ非衝突）。
  2. 指標計算＋ベクトル化 Entry シグナル生成（AlgoForge 逐次 `evaluate_conditions` と **100% 完全一致**）。
  3. `causal_fx_backtest.py` による評価・`StrategyRegistry` 保存・同一入力でのキャッシュ再利用（2回目実行回数 0）。
  4. **性能計測**: 55,201バー 1候補あたりの実測時間・ボトルネック報告（$\le 1.0\text{秒}$ は性能目標とし、正しさの検証とは独立して扱う）。

### Phase 1B-2：小規模探索への開放（1B-1 承認後に別途実施）
- 1B-1 受入後、`CandidateGenerator` から AlgoForge JSON を生成し 20〜30候補の小規模探索を実施。

---

## 5. 変更対象ファイルと概算工数
- **変更対象（`algoforge` 本体は無変更）**:
  1. `src/research_pipeline/backtest/strategy_registry.py`（`algoforge_v1` 経路追加、`pipeline_v1` 無変更）
  2. `src/research_pipeline/integration/algoforge_adapter.py`（ベクトル化評価・`exit` ガード・Capability Request 保存）
  3. `scripts/test_phase1b1_integration.py`（1B-1 固定戦略検証スクリプト）
- **概算工数**: 小（1日、API コスト `$0.00 USD`）。
