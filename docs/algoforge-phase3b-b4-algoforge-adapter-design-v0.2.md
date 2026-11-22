# Phase 3B / B4: AlgoForge Evaluator Adapter 詳細設計案 v0.2

> **ステータス**: 詳細設計案 v0.2（未承認 — コード変更・実験実行は未実施）
> **対象 Issue**: [Issue #5](https://github.com/Naradice/algoforge_pipeline/issues/5)
> **対象レビュー**: `issuecomment-7076618040`（修正依頼 3 点反映）

## 1. Strategy 変換のフィールド対応（`StrategyCreate`）

| Pipeline Strategy JSON | AlgoForge `StrategyCreate` | 変換・既定値・振分け |
|---|---|---|
| `indicators` (`ema`, `rsi` 等) | `indicators` / `features` | 許可済み指標 ID のみ変換。未対応指標・外部 Unit は送信前拒否 → 探索側 Capability Request |
| `long.entry` / `short.entry` | `entry_rules` | 1 階 `AND`/`OR` のみ変換。多段入れ子論理式・`NOT` 等の未対応構文は送信前拒否 → Capability Request |
| `risk` (`k_sl`, `k_tp`, `max_hold_bars`) | `risk_params` | 数値・必須キーをそのまま対応付け。欠落時は送信前 `ValueError` |

## 2. データ・評価条件のマッピング（`StrategyRunCreate` / Dataset API）

- **対応可能（既知）**: `external_dataset_id`, `timeframe`, 評価期間 `[eval_start, eval_end)`（UTC, `bar_start` 基準）, `cost_config`（スプレッド・スリッページ）。
- **未確認／非同等の扱い（Fail-Closed）**:
  - AlgoForge API 側で表現不能または仕様未確認の条件（例：Pipeline 独自 `GateConfig` の細部、ウォームアップ境界の厳密一致、手数料体系差分）は **黙って無視せず、送信前拒否（`ValueError`）または明示的な非同等フラグ付き評価として扱う**（B5 回帰検証対象）。
  - `GET /datasets/{id}` で `dataset_sha256` が照合不能時は `identity_verified=False`（キャッシュ再利用禁止）。
- **ハッシュ対応**:
  - `eval_config_hash`: B3 証跡用メタデータ（`compute_eval_config_hash`、16 桁）。
  - `algoforge_eval_v2` キャッシュキー: B2 契約に基づく 64 桁 SHA-256（`engine_id/version/impl_id`, `dataset_identity`, 正規化全 `eval_config`, `strategy_hash`）。

## 3. ジョブ状態遷移・失敗処理

1. **投入 (`POST /strategies/{id}/runs`)**: `202 Accepted` から `run_id` を取得（`run_ref = "algoforge_run:<strategy_id>:<run_id>"`）。
   - **投入応答消失（`run_id` 不明）かつ冪等性未確認時**: **自動再投入を禁止**し、`EvalResult(status="infra_error", error=EvalError(retryable=False))` を返す。
2. **ポーリング (`GET .../runs/{run_id}`)**:
   - `running`/`pending` → 間隔ポーリング継続。タイムアウト超過時は `run_ref` を保持したまま `infra_error(retryable=True)`（再試行時は新規投入ではなく同 `run_id` の状態照会を優先）。
   - `failed` → `eval_failed(retryable=False)`。
3. **結果取得 (`GET .../metrics`)**:
   - 完了後の `metrics` 取得失敗・HTTP 5xx → `infra_error(retryable=True)`。非有限値（`NaN`/`Inf`）検出 → `eval_failed(retryable=False)`。
