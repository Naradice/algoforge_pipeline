# AlgoForge Phase 2B-2b Hamilton 監査補足・計画 v0.2

## 1. 読み取り専用監査結果 (`HamiltonLogRVRegimeModel` in `econometric_modules.py`)

1. **60日推定窓とウォームアップ**:
   - `step_daily_update_with_fallback` (L733-775) は日次対数実現ボラティリティ（Log-RV）の **60日窓 (`y_window`)** を用いて `fit_em(y_window)` を実行する。
   - ウォームアップ期間は日次 60 日分（M5 バー換算で 60 取引日分）とし、60 日未満の区間は `is_valid = False` とする。
2. **遷移行列の定義と予測確率の手計算照合**:
   - `forward_filter` (L526-575) L552 にて `pred = self.P.T @ xi_curr_filt` を計算し、`predict_next_day_state_prob` (L722-731) L729 にて `xi_d_plus_1 = self.P.T @ xi_d_d` を計算している。
   - したがって `self.P` は行確率行列（`P[i, j] = P(S_{d+1} = j | S_d = i)`）であり、$\xi_{d+1 \mid d} = P^\top \xi_{d \mid d}$ が成立する。
3. **EM 失敗時のパラメータ・フィルタ状態の扱い**:
   - `day_idx == 0` または `prev_params is None` の失敗時：`quantile_initial`（25/75 パーセンタイル、`P = [[0.90, 0.10], [0.20, 0.80]]`）へフォールバック。
   - `day_idx >= 1` の失敗時：`lag1_carryover`（前日 `prev_params` を引き継ぐ）。
   - いずれの場合も `predict_next_day_state_prob(y_window)` は `forward_filter` を初期事前分布 `[0.80, 0.20]` から窓内で再実行して `xi_filt[-1]` を得る。
4. **日次 RV 確定・EM 推定完了・Signal 利用可能時刻**:
   - 日次 Log-RV は UTC 00:00 の日足確定時に確定し、同時刻に `step_daily_update_with_fallback` が完了する（計算遅延ゼロ仮定）。
   - よって翌日 $d+1$ の全 M5 バーに対する `available_at` は `day_{d+1} 00:00:00 UTC` であり、`available_at <= decision_time` を満たす。
5. **M5 への as-of join と古い値の失効条件**:
   - 日次予測確率 $\xi_{d+1 \mid d}$ は `day_{d+1} 00:00 UTC` から `day_{d+2} 00:00 UTC` 未満の M5 バーへ前方 as-of join する。
   - 日次データが欠損し `available_at` が 24 時間以上更新されない場合（古い値の失効条件）は `is_valid = False` とし、意図しない Entry 発火を禁止する。

## 2. Phase 2B-2b 実装計画（承認待ち — 現時点では未実装）

- Phase 2B-2a 修正計画 v0.1 の完了・受入後に、`unit:hamilton_pred_prob` の実装、契約テスト（手計算照合・EM 失敗フォールバック・未来改変不変性・as-of join 失効）、および `algoforge_unit_v1` キャッシュ分離を実施する。
