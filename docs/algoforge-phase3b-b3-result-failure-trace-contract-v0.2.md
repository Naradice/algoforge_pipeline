# Phase 3B / B3: 評価結果・失敗・実行証跡契約 詳細設計案 v0.2

> **ステータス**: 詳細設計案 v0.2（未承認 — コード変更・実験実行は未実施）
> **対象 Issue**: [Issue #4](https://github.com/Naradice/algoforge_pipeline/issues/4)
> **対象レビュー**: `issuecomment-7076514180`（修正依頼 3 点反映）

## レビュー指摘（3 点）への対応

1. **失敗分類の境界（例外 vs `EvalResult`）**:
   - **ローカル評価 (`LocalCausalEvaluator`)**: 従来どおり例外を外側へ伝播する（`EvalResult` へ変換しない）。
   - **外部 API 評価 (`AlgoForgeApiEvaluator`, B4)**：
     - **呼び出し前入力契約違反**（`dataset_sha256` 欠落等、B1/B2 違反）：送信前に `ValueError` を送出（Fail-Closed）。
     - **API 4xx（認証 401/権限 403/不正入力 400・422）**：`EvalResult(status="infra_error" or "eval_failed", error=EvalError(..., retryable=False))`（認証/権限は `infra_error`・再試行不可、入力不備は `eval_failed`・再試行不可）。
     - **評価実行失敗・非有限メトリクス（`NaN`/`Inf`）**：`EvalResult(status="eval_failed", error=EvalError(..., retryable=False))`。
     - **評価器側未対応機能**：`EvalResult(status="unsupported", error=EvalError(..., retryable=False))`（探索側の事前 Capability Request とは明確に区別）。
2. **再試行条件と重複実行防止**:
   - **再試行可（`retryable=True`）**：一時通信断・接続タイムアウト・502/503/504 のみに限定。認証（401/403）・設定不備（4xx）・恒久的エラーは `retryable=False`。
   - **タイムアウト時の重複実行防止**：ジョブ投入時の冪等キー（または既存 `run_ref` の状態照会 `GET .../runs/{run_id}`）を優先し、タイムアウト後に API 側で継続中の実行を二重起動しない契約とする（具体的 API マッピングは B4 で確定）。
3. **B2 同一性と証跡の整合**:
   - 実行証跡に `engine_id`, `engine_version`, `engine_impl_id`, `engine_verified`, `identity_verified`, `eval_config_hash`, `run_ref`, `artifact_uri` を関連付ける。
   - `run_ref` / `artifact_uri` が未取得（`None`）の場合も証跡メタデータ自体は記録可能とするが、B2 契約に従い `identity_verified == False` または `engine_verified == False` の場合は **キャッシュ再利用禁止** とする。
