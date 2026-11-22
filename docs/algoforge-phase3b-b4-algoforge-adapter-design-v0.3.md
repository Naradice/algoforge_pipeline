# Phase 3B / B4: AlgoForge Evaluator Adapter 詳細設計案 v0.3

> **ステータス**: 詳細設計案 v0.3（未承認 — コード変更・実験実行は未実施）
> **対象 Issue**: [Issue #5](https://github.com/Naradice/algoforge_pipeline/issues/5)
> **対象レビュー**: `issuecomment-7076648567`（残り 2 点反映）

## レビュー指摘（2 点）への対応

1. **非同等評価の混入禁止（Fail-Closed 一本化）**:
   - Phase 3C の初期 `EvaluatorProtocol.evaluate()` では、評価期間・ウォームアップ・コスト・ゲート等の同等性が確認できない条件は **送信前拒否（`ValueError`, Fail-Closed）に一本化** する。
   - 「非同等フラグ付き評価」は通常 `status="ok"` 結果へ混入させず、将来の明示的な別モードに留める。
2. **再試行操作の限定（新規 `POST /runs` 禁止）**:
   - `run_ref` 取得後のポーリングタイムアウトや `metrics` 取得 5xx における `retryable=True` は、**既存 `run_ref` に対する状態照会（`GET .../runs/{run_id}`）／結果取得（`GET .../metrics`）の再試行のみ** を意味し、`evaluate()` を再呼出しして新規 `POST /runs` を行う許可ではない。
   - `run_id` 不明・冪等性未確認時の自動再投入禁止を維持し、`metrics` 未取得のまま `status="ok"` を返さない（`eval_failed` または `infra_error` とする）。

## 1. v0.2 からの維持事項

- Strategy 変換表（`indicators`, `long/short.entry` 1 階 AND/OR, `risk`、未対応は送信前拒否 → Capability Request）。
- `eval_config_hash`（B3 メタデータ）と `algoforge_eval_v2`（B2 キャッシュキー）の分離、および `identity_verified=False` / `engine_verified=False` 時のキャッシュ再利用禁止。
- 既存 `LocalCausalEvaluator`・`causal_fx_backtest.py`・`algoforge_v1` キャッシュの維持。
