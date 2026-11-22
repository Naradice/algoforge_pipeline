# AlgoForge_Pipeline 評価境界分離計画 v0.2（`EvaluatorProtocol` / `LocalCausalEvaluator`）

> **ステータス**: 設計・計画のみ（実装・実験実行・AlgoForge/Study Manager 変更は未実施）
> **変更前基準コミット**: `8f52f02`

## R1. キャッシュ互換性の証明

- **既存呼び出し**（`scripts/run_phase1b2_exploration.py` L34, L216）:
  `compute_algoforge_evaluation_hash(s_hash, snap_sha, "2024_132d", ENGINE_VERSION, COST_CFG, UNIT_VERSIONS)`
  ここで `ENGINE_VERSION = "causal_fx_backtest_v1.0"`、名前空間は既存の `algoforge_v1`（`strategy_registry.py` 既定）。
- **互換性維持**: `LocalCausalEvaluator.engine_id` を `"causal_fx_backtest_v1.0"` に固定し、`compute_algoforge_evaluation_hash` への入力（`s_hash`, `snap_sha`, `"2024_132d"`, `evaluator.engine_id`, `COST_CFG`, `UNIT_VERSIONS`）を一切変更しない。
- **将来分離**: 将来 `AlgoForgeApiEvaluator` を導入する際は `engine_id = "algoforge_api_v1"` のように異なる `engine_id` を与えることで、既存キャッシュと混在しない分離を実現する。

## R2. 既存評価インターフェースの完全な対応付け

- **既存入力**: `af_ind`（計算済み指標 DataFrame）、`pl_df`（Polars OHLCV）、`eval_start_idx`、`cand["long"]["entry"]`、`cand["short"]["entry"]`、`BacktestConfig`、`GateConfig`。
- **既存出力・参照フィールド**: `summary` 辞書（`total_trades`, `win_rate`, `profit_factor`, `total_return_pct`, `max_drawdown_pct`, `sharpe_ratio` 等）。
- **最小 `LocalDatasetRef` と `EvalResult`**:
  - `LocalDatasetRef(af_ind: pd.DataFrame, pl_df: Any, eval_start_idx: int, dataset_sha256: str)` によりローカル評価入力と将来の外部参照（`dataset_id`）を明確に区別する。
  - `EvalResult(status: str, summary: Dict[str, Any], engine_id: str, dataset_sha256: str, eval_config_hash: str, run_ref: Optional[str] = None, error: Optional[str] = None)` とし、既存の `summary` 辞書・候補拒否・Capability Request 記録情報を欠落なく保持する。

## R3. 独立した回帰基準の固定

- **基準取得**: 変更前コミット `8f52f02` の `run_phase1b2_exploration.py`（シード `42`、最大 `25` 試行、`usdjpy_m5_2024_oanda_snapshot.parquet`）から基準結果 JSON を取得し、回帰テストで以下 5 項目を検証する：
  1. キャッシュなしでの探索結果の一致
  2. 既存キャッシュを利用した探索結果の一致（再利用の可否は実測後に確定）
  3. 候補順序、`strategy_hash` / `root_hash`、重複排除、Capability Request（3 件）の一致
  4. 評価メトリクス（`summary` 全項目）の一致
  5. 入力不正・未対応機能・評価失敗時の既存動作維持

## R4. 宣言的 JSON と Capability Request の境界維持

- 既存 Strategy JSON の意味・正規化・ハッシュ・Capability Request 記録動作は変更しない。
- 将来の探索・モデル・評価設定も宣言的 JSON で外部定義し、未対応機能は Capability Request として記録する（実装課題は GitHub Issues、研究上の必要性は Study Manager `#774` で管理。自動起票・自動登録は今回対象外）。

## R5. 変更規模の再見積もり

- **変更対象ファイル**:
  1. `src/research_pipeline/backtest/evaluator_protocol.py`（新規：~70 行）
  2. `scripts/run_phase1b2_exploration.py`（局所修正：~15 行）
  3. `scripts/test_phase1b2_evaluator_regression.py`（新規：~90 行）
- **無変更ファイル**: `causal_fx_backtest.py`, `strategy_registry.py`, `integration/*`, Phase 0〜4C, `algoforge/*`, `study_manager/*`。
- 探索アルゴリズム・バックテスト計算・キャッシュ実装の変更は不要であり、もし必要が判明した場合は実装範囲を拡大せず未解決事項として報告する。
