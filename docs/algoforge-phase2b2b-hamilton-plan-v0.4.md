# AlgoForge Phase 2B-2b Hamilton 実装計画 v0.4

> **ステータス**: 実装未承認（Phase 2B-2a 修正完了・受入後に実装開始）

## 1. 正式仕様と日次添字・時刻契約

1. **60日推定窓と日次添字**:
   - `HamiltonLogRVRegimeModel.step_daily_update_with_fallback` の **60日推定窓 (`y_window`)** を正式仕様とする。
   - 最終観測日 $d-1$（`00:00 UTC` 日足確定）、更新実行時刻 $d$ `00:00:00 UTC`、予測対象日 `target_day` $= d$、予測確率 $\xi_{d \mid d-1} = P^\top \xi_{d-1 \mid d-1}$。
2. **EM・フィルタリングの計算遅延ゼロ仮定**:
   - 日次 RV 確定、EM 推定、フォールバック判定、および `forward_filter` 実行の計算遅延を **ゼロ** と仮定し、`available_at = target_day 00:00:00 UTC` とする。
3. **日次 RV の有効観測条件**:
   - 日次 Log-RV の算出には、当該 UTC 日に有効な M5 リターン観測が存在すること（全欠損日・非有限値 `NaN`/`Inf` を除外）を条件とし、60日窓内に欠損がある場合は `is_valid = False`（Fail-Closed）とする。

## 2. M5 as-of join・UTC 日付境界・フォールバック整合性

1. **`target_day` と M5 バー開始日の一致**:
   - 日次出力は `target_day`、`available_at`（`d 00:00:00 UTC`）、`valid_until`（`(d+1) 00:00:00 UTC`）を保持する。
   - M5 結合時は `bar_start_utc.date() == target_day` かつ `available_at <= decision_time < valid_until` を検証する。
2. **UTC 日付境界 `23:55 -> 00:00` の負例テスト**:
   - `23:55 UTC`（日付 $d-1$）のバーに日付 $d$ 向けの $\xi_{d \mid d-1}$ が混入した場合、および `00:00 UTC`（日付 $d$）のバーに日付 $d-1$ の古い値が残存した場合を **Fail-Closed（拒否または `is_valid = False`）** とする負例テストを実装する。
3. **フォールバック時の状態ラベル整合性**:
   - `quantile_initial`（`day_idx == 0` 失敗時）および `lag1_carryover`（`day_idx >= 1` 失敗時）において、状態 0（低ボラティリティ）・状態 1（高ボラティリティ）の順序（$\mu_0 < \mu_1$）が維持されることを検証する。

## 3. 未来改変不変性と評価ハッシュ分離

- 日次 RV 集計、EM 推定、フォールバック、フィルタリング、M5 結合の全工程を通じた未来改変不変性テストを実施する。
- モデル実装 SHA-256、60日推定窓、フォールバック設定、入力データ SHA-256 を `algoforge_unit_v1` 評価ハッシュに含めて分離する。
