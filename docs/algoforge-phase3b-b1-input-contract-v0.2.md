# Phase 3B / B1: 評価入力契約 詳細設計案 v0.2（Strategy JSON / DatasetRef / 評価期間・前処理）

> **ステータス**: 詳細設計案 v0.2（未承認 — コード変更・実験実行は未実施）
> **対象 Issue**: [Issue #2](https://github.com/Naradice/algoforge_pipeline/issues/2)
> **対象レビュー**: `issuecomment-7075584869`（修正依頼 2 点反映）

## レビュー指摘（2 点）への対応

1. **データセット同一性**:
   - `dataset_sha256` はデータ内容の不変識別子、`external_dataset_id` は外部システム上の参照 ID として役割を分離する。
   - 外部 API が `dataset_sha256` を返却・照合できない場合は「**同一性未検証（unverified）**」として扱い、異なるデータを同一と断定しない（具体的な API 照合マッピングは B4 に委譲）。
2. **評価期間の意味**:
   - 共通契約における `eval_start` / `eval_end` は **UTC の半開区間 `[eval_start, eval_end)`**、バー時刻は **バー開始時刻（`bar_start`）** として定義する。
   - **ウォームアップ期間**は評価対象期間 `[eval_start, eval_end)` の**外側（直前）**として扱い、評価集計には含めない（既存 `eval_start_idx` との厳密な対応・API 差異検証は B4/B5 に委譲）。

## 1. 契約仕様（v0.1 からの維持事項）

- **Strategy JSON**: 宣言的 JSON のみ（Python コード禁止）。未対応機能は評価器呼出し前に探索ループ側で Capability Request として記録。
- **`DatasetRef`**: 共通 `DatasetRef(dataset_sha256, timeframe, eval_start, eval_end, external_dataset_id=None)` とローカル専用 `LocalDatasetRef(af_ind, pl_df, eval_start_idx, dataset_sha256)` を維持（`LocalCausalEvaluator`・旧キャッシュキー変更なし）。
- **Fail-Closed**: `dataset_sha256` 欠落、`eval_start >= eval_end`、API 経路での `external_dataset_id` 欠落は即時 `ValueError`。
