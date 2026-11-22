# Phase 3B / B3: 評価結果・失敗・実行証跡契約 詳細設計案 v0.1

> **ステータス**: 詳細設計案 v0.1（未承認 — コード変更・実験実行は未実施）
> **対象 Issue**: [Issue #4](https://github.com/Naradice/algoforge_pipeline/issues/4)
> **前提**: Phase 3A 受入済み（Commit `441641d`）、B1 承認済み（Issue #2）、B2 承認済み（Issue #3）

## 1. 現行実装・既存資料との対応

- **現行 `EvalResult`**: `src/research_pipeline/backtest/evaluator_protocol.py` の `EvalResult(status, summary, engine_id, dataset_sha256, eval_config_hash, run_ref=None, error=None)`。
- **例外伝播**: `LocalCausalEvaluator.evaluate()` は例外を捕捉・握りつぶしせず外側へ伝播する（Phase 3A `test_exception_propagation_preserved` で検証済み）。
- **探索側責務との分離**: 候補拒否（バリデーション失敗）および Capability Request 記録は評価器呼出し前に探索ループ側で完結する（Issue #1 R2 契約）。

## 2. 提案する結果・失敗・実行証跡契約

### 2.1 結果ステータスと失敗分類（`EvalStatus` / `EvalErrorKind`）
- **`EvalStatus`**:
  - `"ok"`: 評価正常完了（`summary` が有効）。
  - `"eval_failed"`: 評価エンジン側の実行失敗（例：バックテスト異常終了、非有限メトリクス）。
  - `"infra_error"`: 実行環境エラー（ネットワーク断、API 5xx、タイムアウト等）。
  - `"unsupported"`: 評価器側で未対応機能に到達した場合（原則として探索側で事前検出するが、防御的境界として定義）。
- **`EvalError`（構造化エラー）**:
  - `kind: str`（`"eval_failed" | "infra_error" | "unsupported"`）, `message: str`, `retryable: bool`, `timeout_sec: Optional[float] = None`。
  - **再試行可否**: `"infra_error"`（タイムアウト・5xx・一時通信断）のみ `retryable=True` とし、`"eval_failed"` / `"unsupported"` は `retryable=False`。

### 2.2 実行証跡（`EvalTrace` / `run_ref`）
- **`run_ref`**: ローカル時は `None`、外部 API 時は `algoforge_run:<strategy_id>:<run_id>` の形式で実行 ID を保持。
- **証跡メタデータ**: `engine_id`, `dataset_sha256`, `eval_config_hash`, `identity_verified`（B2 整合：`False` 時はキャッシュ再利用禁止・証跡保存のみ）, `artifact_uri`（任意：詳細ログ・トレード一覧への参照 URI）。

## 3. 後方互換性・例外互換方針・責務境界

- **既存例外伝播の維持**: `LocalCausalEvaluator.evaluate()` は従来どおり例外を外側へ伝播し、`EvalResult` への無断変換は行わない（`status="ok"` のみ返却）。構造化エラー返却は将来の `AlgoForgeApiEvaluator`（B4）側で採用する。
- **探索側との責務境界**: 探索側の候補拒否・Capability Request は評価器へ渡る前の責務であり、`EvalResult` の失敗分類と混同しない。

## 4. 今回の設計対象と後続の境界

- **今回対象**: B3（評価結果・失敗・実行証跡契約）の仕様設計のみ。
- **対象外**: コード変更、実験実行、`AlgoForgeApiEvaluator` 実装（B4 以降）。
- **受入条件（B5 で検証）**: `LocalCausalEvaluator` の例外伝播不変、`EvalStatus`/`retryable` 分類の一意性、`identity_verified=False` 時の証跡保存・キャッシュ非再利用。
