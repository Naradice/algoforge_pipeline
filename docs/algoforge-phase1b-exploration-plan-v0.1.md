# AlgoForge Phase 1B 実装計画草案（v0.1）

**日付**: 2026-11-22  
**前提**: 統合設計書 v0.2（承認済み）、Phase 1A 互換性検証（条件付き受入）  
**状態**: 計画承認待ち（実装・実行未着手）  

---

## 1. 初期対応する AlgoForge Strategy JSON 部分集合と指標意味論

- **対象スキーマ（`schema_version = "algoforge_v1"`）**:
  `{symbol, indicators, long: {entry, exit}, short: {entry, exit}, risk}`
- **初期対応指標**: `ema`, `sma`, `rsi`, `atr`, `bb`, `donchian`, `macd`, `adx`, `stochastic`, `roc`, `rvol`（AlgoForge `apply_indicators` で計算可能な決定論的指標群）。
- **指標定義差の扱い（意味論の保持）**:
  - `algoforge_v1` では AlgoForge 側の数理定義（Wilder 平滑 RSI/ATR、母標準偏差 BB `ddof=0`、当バー含む Donchian）を正本として明示し、Pipeline 旧定義（`pipeline_v1`）と無条件に置換・混同しない。
  - 既存の `causal_fx_backtest.py` の約定・会計・SL/TP 優先順位・サイジング用 `ATR_20` は一切変更しない。

---

## 2. Entry/Exit・Long/Short 契約と因果性保証

- **評価契約**: `long.entry` / `short.entry` / `long.exit` / `short.exit` をそれぞれ boolean 配列へ評価し、`run_causal_fx_backtest` へ渡す。
- **因果性保証の拡張**:
  - M5 バー確定時刻 $t_{\text{close}} = t_{\text{bar\_start}} + 5\text{分}$ 時点で確定した特徴量のみを参照し、約定は次バー $t+1$ 始値とする（Donchian 等もバー $t$ 終値確定後に評価するため先読みなし）。
  - 未確定バー・上位時間足（H1/H4 `close_time <= t_close`）・外部データ（`information_cutoff <= t_close`）を厳格に区別する。

---

## 3. StrategyRegistry の新旧分離仕様

既存の Phase 0〜4 データ（`pipeline_v1`）は変更・削除・再計算せず、以下の通り名前空間を分離する：
- **`strategy_hash`**: `SHA256("algoforge_v1" + canonicalize_algoforge(strategy_json))`（`indicators` を `id` 順、AND/OR 条件木を辞書順正規化）。
- **`evaluation_hash`**: `SHA256(strategy_hash + schema_version + dataset_sha256 + eval_period + engine_version + unit_versions + cost_config)`。

---

## 4. 条件評価の高速化設計（ベクトル化 vs 逐次評価）

Phase 1A の実測で `evaluate_conditions` 逐次評価は 55,201バーあたり約 13.5秒かかるため、意味論を変えない最小限の高速化を行う：
- **ベクトル化可能（一括 Pandas/NumPy 演算）**: 単純比較（`>`, `<`, `>=`, `<=`, `==`, `!=`）、`cross_above`/`cross_below`、`and`/`or`/`not`、`rolling`。
- **逐次評価が必要**: `streak` 等 `df_upto` に依存する状態付き条件。
- **等価性保証**: ベクトル化評価の結果が AlgoForge `evaluate_conditions` 逐次評価と **100% 一致** することを契約テストで検証する。

---

## 5. 未知戦略（Capability Request）の保存と Phase 1A 証跡確認

- 既存機能で表現できない LLM 提案は破棄せず、Capability Request JSON（名称・根拠・入出力・不可理由・状態）として保存する（自動実装・モデル学習は対象外）。
- Phase 1A 未監査証跡については、Phase 1B 実装前に `test_phase1a_compatibility.py` のログと `algoforge` `git status`（0変更）を添付確認する。

---

## 6. 受入条件・変更対象ファイル・段階的実装順序

- **受入条件**:
  1. **因果性**: 未来改変不変性 0 差分。
  2. **再現性**: ベクトル化評価 vs `evaluate_conditions` 100% 一致。
  3. **性能**: 55,201バー 1候補あたり $\le 1.0\text{秒}$。
- **変更対象ファイルと実装順序（`algoforge` 本体は無変更）**:
  1. `src/research_pipeline/backtest/strategy_registry.py`: `algoforge_v1` 正規化・`evaluation_hash` 拡張。
  2. `src/research_pipeline/integration/algoforge_adapter.py`: ベクトル化条件評価＋`evaluate_conditions` 等価性テスト。
  3. `src/research_pipeline/backtest/candidate_generators.py`: AlgoForge JSON 生成対応＋Capability Request 保存。
  4. `scripts/run_phase1b_exploration.py`: 20〜30候補の小規模疎通探索（2024年データ）。
