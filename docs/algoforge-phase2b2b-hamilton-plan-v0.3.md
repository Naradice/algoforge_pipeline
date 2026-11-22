# AlgoForge Phase 2B-2b Hamilton 計画 v0.3

## 1. 正式仕様と日次添字の厳密定義

1. **60日推定窓**:
   - `HamiltonLogRVRegimeModel.step_daily_update_with_fallback` の **60日推定窓 (`y_window`)** を正式仕様とする。60日未満のウォームアップ区間は `is_valid = False`。
2. **最終観測日・更新実行時刻・予測対象日の分離**:
   - **最終観測日**: $d - 1$（日次 Log-RV $y_{d-1}$ は日付 $d$ の `00:00:00 UTC` に確定）。
   - **更新実行時刻**: 日付 $d$ の `00:00:00 UTC`（計算遅延ゼロ仮定）。
   - **予測対象日 (`target_day`)**: 日付 $d$（`00:00:00 UTC` 〜 `23:55:00 UTC`）。
   - **予測確率の定義**: 日付 $d$ の `00:00:00 UTC` に利用可能な情報 $y_{\le d-1}$ から生成される日付 $d$ の予測確率を **$\xi_{d \mid d-1} = P^\top \xi_{d-1 \mid d-1}$** として定義する。

## 2. 非対称数値例による遷移行列・予測確率の手計算照合

- 行確率行列 $P$（`P[i, j] = P(S_d = j | S_{d-1} = i)`）の非対称例として：
  $$P = \begin{pmatrix} 0.90 & 0.10 \\ 0.25 & 0.75 \end{pmatrix}, \quad \xi_{d-1 \mid d-1} = \begin{pmatrix} 0.80 \\ 0.20 \end{pmatrix}$$
- 手計算：
  $$\xi_{d \mid d-1} = P^\top \xi_{d-1 \mid d-1} = \begin{pmatrix} 0.90 & 0.25 \\ 0.10 & 0.75 \end{pmatrix} \begin{pmatrix} 0.80 \\ 0.20 \end{pmatrix} = \begin{pmatrix} 0.72 + 0.05 \\ 0.08 + 0.15 \end{pmatrix} = \begin{pmatrix} 0.77 \\ 0.23 \end{pmatrix}$$
- `forward_filter` (L552) および `predict_next_day_state_prob` (L729) の `self.P.T @ xi` と一致することを契約テストで照合する。

## 3. EM 失敗時のフォールバックと M5 as-of join 契約

1. **EM 失敗時のパラメータ引継ぎとフィルタ再計算**:
   - `day_idx == 0` 失敗時は `quantile_initial`、`day_idx >= 1` 失敗時は `lag1_carryover` を採用し、いずれも窓内 $y_{\le d-1}$ のみを用いて `forward_filter` を再実行して $\xi_{d-1 \mid d-1}$ を得るため未来情報を含まない。
2. **`target_day` と `valid_until` による M5 結合検証**:
   - 日次出力に `target_day`（日付 $d$）、`available_at`（`d 00:00:00 UTC`）、`valid_until`（`(d+1) 00:00:00 UTC`）を付与する。
   - M5 結合時は `available_at <= decision_time < valid_until` かつバーの UTC 日付が `target_day` と一致することを検証し、欠損・期限切れは `is_valid = False`（Fail-Closed）とする。
3. **未来改変テストと評価ハッシュ**:
   - 日次 RV 集計、EM 推定、フォールバック、フィルタリング、M5 結合の全工程を通じた未来改変不変性テストを実施する。
   - モデル実装 SHA-256、推定窓（60日）、フォールバック設定、入力データ SHA-256 を `algoforge_unit_v1` 評価ハッシュに含めて識別する。
