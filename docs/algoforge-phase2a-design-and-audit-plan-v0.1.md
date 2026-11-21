# AlgoForge Phase 2A 設計・監査計画（v0.1）

**日付**: 2026-11-22  
**前提**: Phase 1B-2 条件付き受入（コミット `cfe8126`, `dea8a3f`）  
**状態**: 計画承認待ち（新規 Unit 実装・モデル学習・探索実験は未着手）  

---

## 1. Phase 1B 完了整理（証跡と対応範囲）

- **Phase 1A / 1B-1 / 1B-2 対応範囲**:
  - 決定論的既存指標（`ema`, `sma`, `rsi`, `atr`, `bb`, `donchian`, `macd`, `adx`, `stochastic`, `roc`, `rvol`）の計算、1階層 `and`/`or` 条件のベクトル化 Entry 評価、`algoforge_v1` 名前空間での `StrategyRegistry` 保存・キャッシュ再利用、25試行進化的探索（root cap=3、Capability Request 保存）。
- **未対応機能（明記）**:
  - **多段入れ子論理式**（AlgoForge `evaluate_conditions` は 1階層のみ対応）、`streak` / `rolling` / `regime` 条件、非空 `exit` 条件、拡張 `risk`（`trailing_stop` 等）。
- **確定証跡**:
  - コミット ID: `cfe8126`（Phase 1B-1）、`dea8a3f`（Phase 1B-2）
  - 結果 JSON: `data/phase1b2_exploration_results.json`
  - 基準データ SHA-256: `bbc7a9c24b701f78d0d4716e7f633246f05445a198f6a2cdca771861e43939a0`
  - 評価設定: `causal_fx_backtest_v1.0`、`spread=0.010, slippage=0.002`

---

## 2. Phase 2A 監査対象コードと設計項目（8項目）

### 監査対象コード（読み取り専用）
- `algoforge/backend/strategy/engine/{indicators,conditions,execution,backtest}.py`
- `algoforge_pipeline/src/research_pipeline/discovery/econometric_modules.py`
- `algoforge_pipeline/src/research_pipeline/integration/algoforge_adapter.py`
- `algoforge_pipeline/src/research_pipeline/backtest/{causal_fx_backtest,strategy_registry}.py`

### 設計成果物（明確化する 8項目）
1. **`UNIT_REGISTRY` の構造・登録・実行・依存関係**: ホワイトリスト制レジストリ（JSON からの任意 Python 実行禁止）、Unit メタデータ（`unit_id`, `version`, `input_cols`, `output_schema`, `warmup_bars`）。
2. **Strategy JSON からの Unit 参照方式**: `indicators` ブロック内での `{"id": "...", "type": "unit:<name>", "params": {...}}` 形式。
3. **4層責務境界**: Feature/Model Output $\to$ Signal/Decision $\to$ Order Intent $\to$ Execution/Risk。
4. **Signal の型と意味**: `boolean`（エントリー/ゲート）、`continuous`（スコア/強度）、`probability`（レジーム確率 $p \in [0,1]$）、`state`（離散状態）。
5. **Unit 入出力スキーマ・時刻・欠損処理**: `information_cutoff <= t_close` の厳守、ウォームアップ期間中の `NaN` 出力、欠損時の安全側（エントリー不可）処理。
6. **Unit バージョンと `StrategyRegistry`**: `unit_versions` 辞書を `compute_algoforge_evaluation_hash` へ組み込み、Unit 更新時のキャッシュ自動分離を保証。
7. **既存バックテスターへの接続範囲と未対応 Signal**: 当面 `causal_fx_backtest.py` へは `boolean` シグナルのみ接続し、`Order Intent`（指値/動的SL）を要する Signal は未対応として Capability Request へ記録。
8. **Capability Request $\to$ Unit 実装接続契約**: Capability Request JSON の `missing_capability` と `UNIT_REGISTRY` の `unit_id` を 1対1 対応させる最小契約。

---

## 3. 互換性方針・受入基準・残余リスク

- **互換性方針**: Phase 0〜4 および Phase 1B の既存成果・`StrategyRegistry` データ・AlgoForge 本体は一切変更しない。
- **受入基準**: 上記 8項目の設計仕様書（`docs/algoforge-phase2a-unit-architecture-v0.1.md`）を提出し、既存コードとの整合性を確認すること。
- **残余リスク**: `Order Intent` 層（指値・動的 TP/SL）を `causal_fx_backtest.py` へ接続する際は、Phase 2B 以降で別途バックテスター側の契約テストが必要となる。
