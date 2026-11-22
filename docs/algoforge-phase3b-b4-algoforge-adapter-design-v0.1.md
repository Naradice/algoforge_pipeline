# Phase 3B / B4: AlgoForge Evaluator Adapter 詳細設計案 v0.1

> **ステータス**: 詳細設計案 v0.1（未承認 — コード変更・実験実行は未実施）
> **対象 Issue**: [Issue #5](https://github.com/Naradice/algoforge_pipeline/issues/5)
> **前提**: Phase 3A 受入済み（Commit `441641d`）、B1 v0.2 / B2 v0.2 / B3 v0.2 承認済み

## 1. 現行実装・既存資料との対応

- **EvaluatorProtocol**: `src/research_pipeline/backtest/evaluator_protocol.py`（`LocalCausalEvaluator` は回帰比較用として維持し、既存 `causal_fx_backtest.py` は削除しない）。
- **AlgoForge API（既存監査済み）**: `algoforge/backend/strategy/router.py`（`POST /strategies`, `POST /strategies/{id}/runs`, `GET /strategies/{id}/runs/{run_id}`, `GET .../metrics`, `GET .../trades`）および `data/router.py`（`GET /datasets/{id}`）。

## 2. `AlgoForgeApiEvaluator` 設計とマッピング表

### 2.1 インターフェースと共存構成
- `AlgoForgeApiEvaluator` は `EvaluatorProtocol` を実装し、`engine_id = "algoforge_api_v1"` とする。
- `LocalCausalEvaluator` と同一の `evaluate(strategy_json, dataset_ref, eval_config) -> EvalResult` を提供し、探索ループ側は評価器の差し替えのみで切り替え可能とする。

### 2.2 入出力マッピング表

| 項目 | Pipeline 契約（B1〜B3） | AlgoForge API 対応 |
|---|---|---|
| **Strategy JSON** | 宣言的 Strategy JSON（1階 AND/OR） | `POST /strategies` (`StrategyCreate`) へ変換送信。未対応機能は送信前に探索側で Capability Request 化 |
| **データセット** | `DatasetRef(dataset_sha256, external_dataset_id, eval_start, eval_end, timeframe)` | `external_dataset_id` を指定。`GET /datasets/{id}` でハッシュ照合不能時は `identity_verified=False`（B2: キャッシュ再利用禁止） |
| **評価設定** | `eval_config`（`algoforge_eval_v2` キャッシュキー用正規化 JSON SHA-256、メタデータ `eval_config_hash` と区別） | `POST /strategies/{id}/runs` (`StrategyRunCreate`) へ評価期間 `[eval_start, eval_end)`・コスト設定をマッピング |
| **エンジン同一性** | `engine_id`, `engine_version`, `engine_impl_id`, `engine_verified` | API が実装リビジョンを返せない場合は `engine_verified=False`（B2: キャッシュ再利用禁止） |
| **実行・結果・証跡** | `EvalResult(status, summary, run_ref, error)` | `202` 後に `GET .../runs/{run_id}` をポーリングし、完了後 `GET .../metrics` を `summary` へ変換、`run_ref = "algoforge_run:<strategy_id>:<run_id>"` |

## 3. 失敗・タイムアウト・重複実行防止（B3 引き継ぎ反映）

- **B3 失敗分類**: 送信前入力違反は `ValueError`、401/403 は再試行不可の `infra_error`、400/422 は再試行不可の `eval_failed`、502/503/504・接続タイムアウト・一時通信断のみ `retryable=True`。
- **タイムアウト時の重複防止**: タイムアウト後はまず `run_ref` による状態照会（`GET .../runs/{run_id}`）を行う。**冪等キーも実行状態照会も利用できない場合は、タイムアウト後の自動再投入を禁止**し、`infra_error(retryable=False)` として返す。

## 4. 今回の設計対象と後続の境界

- **今回対象**: B4（`AlgoForgeApiEvaluator` 設計）のみ。コード変更・実装・実験実行は行わない。
- **移行計画**: 後続 B5（回帰検証計画）でモック API テスト・仕様差分検証を設計し、B6 統合レビュー後に実装フェーズ（Phase 3C）へ進む。
