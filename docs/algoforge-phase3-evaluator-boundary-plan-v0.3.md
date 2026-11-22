# AlgoForge_Pipeline 評価境界分離計画 v0.3（`EvaluatorProtocol` / `LocalCausalEvaluator`）

> **ステータス**: 設計・計画のみ（実装・実験実行・AlgoForge/Study Manager 変更は未実施）
> **変更前基準コミット**: `8f52f02`
> **対象レビュー**: Issue #1 `issuecomment-7074253583`

## レビュー指摘（1〜5）への対応表

| # | 指摘事項 | v0.3 での対応 |
|---|---|---|
| **1** | **R1 ハッシュ互換性** | `compute_algoforge_evaluation_hash(s_hash, snap_sha, "2024_132d", "causal_fx_backtest_v1.0", COST_CFG, UNIT_VERSIONS)` の入力と期待ハッシュを固定例で検証し、キャッシュ空／既存キャッシュの双方で名前空間 `algoforge_v1` の不変を確認する。 |
| **2** | **R2 情報境界** | 候補拒否・Capability Request は従来どおり**評価器の外**（`run_phase1b2_exploration.py` の `StrategyRegistry` / `validate_candidate`）で処理し、`EvalResult` はバックテスト評価結果（`summary`, `error`）のみを返す。制御フローは不変。 |
| **3** | **R3 独立した期待値** | 基準コミット `8f52f02` から取得した結果を `tests/fixtures/phase1b2_baseline_8f52f02.json` として事前固定し、変更後コードからの再生成を禁止する。全 `summary` 値（非有限値 `NaN`/`Inf` 含む）・例外・異常系を厳密比較する。 |
| **4** | **評価設定識別** | `EvalResult.eval_config_hash` は既存 `COST_CFG` + `UNIT_VERSIONS` + `BacktestConfig`/`GateConfig` の正規化 JSON SHA-256（識別用メタデータのみ）とし、既存キャッシュ計算（`compute_algoforge_evaluation_hash`）は一切変更しない。 |
| **5** | **将来 API 経路** | 別 `engine_id` のみで全設定差・データ差を分離できるとは断定せず、将来の API 接続時の識別契約課題（コスト・約定・データセット参照の統合ハッシュ設計）として記録する。今回実装はしない。 |

## 1. 情報経路の対応表（制御フロー不変）

| 情報 | 処理主体（変更前＝変更後） | `EvalResult` での扱い |
|---|---|---|
| 未対応機能・Capability Request 記録 | 探索ループ側（評価器呼出し前） | 評価器へ渡らない（対象外） |
| `StrategyRegistry` 重複排除・キャッシュ照合 | 探索ループ側（評価器呼出し前） | キャッシュヒット時は評価器を呼ばない |
| 条件評価・バックテスト実行・例外捕捉 | `LocalCausalEvaluator.evaluate()` | `EvalResult(status, summary, engine_id, dataset_sha256, eval_config_hash, run_ref=None, error)` |

## 2. 独立した期待値と異常系比較規則（R3）

- **基準成果物**: `8f52f02` 実行時の出力を `tests/fixtures/phase1b2_baseline_8f52f02.json` に固定保存（変更後コードによる上書き禁止）。
- **比較規則**:
  - 数値フィールド：`math.isnan` 同士は一致、有限値は `rtol=0, atol=0`（完全一致）。
  - 異常系（入力不正・評価失敗）：既存コードが例外を外側へ伝播させる場合は `EvalResult(status="error")` へ変換せず、**既存の例外種別・メッセージ・伝播動作をそのまま維持**する（既存動作優先）。

## 3. 変更対象ファイル（4 ファイル限定）

1. `src/research_pipeline/backtest/evaluator_protocol.py`（新規）
2. `scripts/run_phase1b2_exploration.py`（評価呼出しのみ置換）
3. `scripts/test_phase1b2_evaluator_regression.py`（新規）
4. `tests/fixtures/phase1b2_baseline_8f52f02.json`（新規：変更前基準 `8f52f02` の固定成果物）
