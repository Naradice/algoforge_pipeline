# Phase 3B / B1: 評価入力契約 詳細設計案 v0.1（Strategy JSON / DatasetRef / 評価期間・前処理）

> **ステータス**: 詳細設計案（未承認 — コード変更・実験実行は未実施）
> **対象 Issue**: [Issue #2](https://github.com/Naradice/algoforge_pipeline/issues/2)
> **前提**: Phase 3A 受入済み（Commit `441641d`, [Issue #1](https://github.com/Naradice/algoforge_pipeline/issues/1)）

## 1. 現行実装・既存資料との対応

- **Strategy JSON**: `scripts/run_phase1b2_exploration.py` が生成する AlgoForge 形式（`indicators`, `long.entry`, `short.entry`, `risk`）。Phase 1B-2 で 1 階 AND/OR・EMA/RSI を検証済み。
- **DatasetRef**: `src/research_pipeline/backtest/evaluator_protocol.py` の `LocalDatasetRef(af_ind, pl_df, eval_start_idx, dataset_sha256)`。現在は DataFrame (`af_ind`, `pl_df`) を直接保持している。
- **評価期間・前処理**: `run_phase1b2_exploration.py` では `"2024_132d"` 固定、指標計算（`compute_algoforge_indicators`）と Polars 変換を呼び出し側で実施し、`eval_start_idx` でウォームアップ境界を指定。

## 2. 提案する評価入力契約（共通部とローカル/API 境界）

### 2.1 Strategy 入力契約
- **正規形式**: 宣言的 Strategy JSON（Python コード・実行スクリプト禁止）。
- **検証境界**: 評価器へ渡す前に呼び出し側（探索ループ）でスキーマ・未対応機能検証を行い、未対応機能は Capability Request として記録する（Phase 3A 契約を維持）。

### 2.2 `DatasetRef` 契約（共通部 vs ローカル専用部）
```python
@dataclass(frozen=True)
class DatasetRef:
    """ローカル評価・外部API評価で共通利用するデータ参照識別子"""
    dataset_sha256: str           # 必須：データ内容の不変識別子
    timeframe: str                # 例: "M5"
    eval_start: str               # ISO8601 UTC（例: "2024-07-01T00:00:00Z"）
    eval_end: str                 # ISO8601 UTC（例: "2024-12-31T23:55:00Z"）
    external_dataset_id: Optional[str] = None  # API評価時の AlgoForge dataset_id
```
- **ローカル専用部 (`LocalDatasetRef`)**: 既存 `LocalDatasetRef(af_ind, pl_df, eval_start_idx, dataset_sha256)` を維持し、`DatasetRef` のメタデータ（`dataset_sha256`, `eval_start`, `eval_end`）と 1:1 対応させる。
- **変換境界**:
  - `LocalCausalEvaluator`: 既存どおり `LocalDatasetRef`（メモリ上 DataFrame + `eval_start_idx`）を受け取る。
  - 将来の `AlgoForgeApiEvaluator`: `DatasetRef.external_dataset_id`（および `eval_start` / `eval_end` / `timeframe`）のみを送信し、DataFrame 本体は送信しない。

### 2.3 前処理の責務境界
- **ローカル経路**: `compute_algoforge_indicators` を呼び出し側（または `LocalDatasetRef` 構築時）で実行し、既存計算結果・`eval_start_idx` と 100% 一致させる。
- **API 経路**: AlgoForge 本体側（`POST /strategies/{id}/runs`）が `external_dataset_id` に基づき指標計算・ウォームアップ処理を行う。

## 3. 後方互換性・失敗時の扱い・未解決論点

- **後方互換性**: `LocalDatasetRef` の既存フィールド・型は変更せず、`LocalCausalEvaluator.evaluate()` の動作を 100% 維持する。
- **失敗時の扱い（Fail-Closed）**:
  - `dataset_sha256` 欠落・空文字、または `eval_start >= eval_end` は `ValueError` で即時拒否。
  - API 経路で `external_dataset_id is None` の場合は送信前に `ValueError` で拒否。
- **未解決論点（B2〜B4 へ引き継ぐ事項）**:
  1. AlgoForge `StrategyRunCreate` における `eval_start` / `eval_end` / `timeframe` の正確なフィールド名・型対応（B4 で確定）。
  2. ローカル `eval_start_idx` と API 側ウォームアップバー数の境界差異の扱い（B5 回帰検証で確認）。

## 4. 最小限の移行手順・検証・受入条件・境界

- **今回の設計対象**: B1（評価入力契約）の仕様定義のみ。
- **境界**: コード変更・実装・研究実験は行わない（`【計画承認】` 後に B4/B5 と合わせて実装または個別実装を判断）。
- **受入条件**:
  1. 既存 `LocalDatasetRef`・`LocalCausalEvaluator` との 100% 後方互換が保たれること。
  2. 共通 `DatasetRef` が DataFrame 非依存で API 評価に必要な識別子（`dataset_sha256`, `external_dataset_id`, `eval_start`, `eval_end`, `timeframe`）を過不足なく表現できること。
