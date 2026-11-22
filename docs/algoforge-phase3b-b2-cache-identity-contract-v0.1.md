# Phase 3B / B2: 評価設定・同一性・キャッシュ契約 詳細設計案 v0.1

> **ステータス**: 詳細設計案 v0.1（未承認 — コード変更・実験実行は未実施）
> **対象 Issue**: [Issue #3](https://github.com/Naradice/algoforge_pipeline/issues/3)
> **前提**: Phase 3A 受入済み（Commit `441641d`）、B1 承認済み（[Issue #2](https://github.com/Naradice/algoforge_pipeline/issues/2) `v0.2`）

## 1. 現行実装・既存資料との対応

- **既存キャッシュキー**: `src/research_pipeline/backtest/strategy_registry.py` の `compute_algoforge_evaluation_hash(strategy_hash, dataset_sha256, eval_period, engine_version, cost_config, unit_versions)`（名前空間 `algoforge_v1`）。
- **既存メタデータハッシュ**: `src/research_pipeline/backtest/evaluator_protocol.py` の `compute_eval_config_hash(eval_config)`（`dataclasses.asdict` による 16 桁メタデータ識別子、キャッシュキーには未使用）。
- **Unit 版**: `src/research_pipeline/integration/unit_registry.py` の `compute_algoforge_unit_evaluation_hash`（名前空間 `algoforge_unit_v1`）。

## 2. 提案する評価設定・同一性・キャッシュ契約

### 2.1 旧キャッシュとの完全共存（名前空間分離）
- **ローカル PoC 経路 (`algoforge_v1`)**: `LocalCausalEvaluator`（`engine_id = "causal_fx_backtest_v1.0"`）は既存 `compute_algoforge_evaluation_hash` をそのまま維持し、旧キャッシュとの 100% 互換性を保つ。
- **新評価契約経路 (`algoforge_eval_v2`)**: 外部 API 評価や拡張設定を伴う新規経路では、名前空間を `"algoforge_eval_v2"` に分離し、旧 `algoforge_v1` キャッシュと物理的・論理的に混在させない。

### 2.2 `algoforge_eval_v2` 評価ハッシュの構成要素
異なる評価条件を誤って同一視しないため、以下の正規化 JSON の SHA-256（64 桁）をキャッシュキーとする：
1. `namespace`: `"algoforge_eval_v2"`
2. `engine_id` + `engine_version`: 例 `{"engine_id": "algoforge_api_v1", "engine_version": "..."}`
3. `dataset_identity`（B1 整合）: `{"dataset_sha256": ..., "timeframe": ..., "eval_start": ..., "eval_end": ..., "external_dataset_id": ..., "identity_verified": bool}`
   - `identity_verified == False`（`unverified`）の場合は、原則としてキャッシュ再利用を禁止（または `unverified` 専用タグを付与し検証済みキャッシュと分離）。
4. `eval_config`（正規化済み設定）: `backtest_config`（SL/TP/最大保有バー）、`gate_config`、`cost_config`（スプレッド/スリッページ/手数料）、`execution_timing`（`next_bar_open` 等）、`unit_versions`。
5. `strategy_hash`: 正規化 Strategy JSON の SHA-256。

## 3. 後方互換性・失敗時の扱い・衝突/誤再利用テストケース

- **後方互換性**: 既存 `compute_algoforge_evaluation_hash` および `StrategyRegistry` のスキーマは変更しない。
- **失敗時の扱い（Fail-Closed）**: 必須識別子（`engine_id`, `dataset_sha256`, `eval_start`, `eval_end`, `strategy_hash`）の欠落や非決定論的オブジェクト（`asdict` 不可）が含まれる場合はハッシュ算出前に `ValueError` で拒否する。
- **検証テストケース（後続 B5/実装時）**:
  1. `LocalCausalEvaluator` で旧 `algoforge_v1` ハッシュが `8f52f02` と一致すること。
  2. `cost_config`・`gate_config`・`eval_start`/`eval_end`・`engine_id` のいずれか 1 項目を変更したとき、`algoforge_eval_v2` ハッシュが必ず変化すること。
  3. `identity_verified=False`（`unverified`）の外部データセット結果が、検証済みキャッシュと誤再利用されないこと。
  4. `algoforge_v1` と `algoforge_eval_v2` で同一戦略・同一データでもキャッシュキーが衝突しないこと。

## 4. 今回の設計対象と後続の境界

- **今回対象**: B2（評価設定・同一性・キャッシュ契約）の仕様設計のみ。
- **対象外**: コード変更、実験実行、`AlgoForgeApiEvaluator` 実装（B4 以降）。
