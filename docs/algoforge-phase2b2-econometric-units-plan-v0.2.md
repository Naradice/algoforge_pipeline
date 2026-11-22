# AlgoForge Phase 2B-2 監査補足・実装計画草案（v0.2）

**日付**: 2026-11-22  
**前提**: Phase 2B-1 条件付き受入（コミット `e364fea`）  
**状態**: 読み取り専用コード監査完了・計画承認待ち（実装・再学習・バックテスト未着手）  

---

## 1. Hawkes 因果性のコードレベル確定

- **参照コード**: `src/research_pipeline/discovery/econometric_modules.py` (`HawkesDiscreteHazardModel.extract_events_from_returns`, `fit`, `lfilter`)
- **処理順序と因果性証明**:
  1. **イベント生成**: `window_returns = returns[t - 20 : t]`（バー $t$ 自身を厳密に除外した過去 20バー標準偏差で $y_t = \mathbb{I}(|r_t| > 2.0 \sigma_{\text{roll},t})$ を算出）。
  2. **強度記録**: バー $t$ の出力強度 `h[t] = curr_h` は、**$y_t$ 観測前かつバー $t$ の `fit()` 実行前**に記録される（すなわち `events[0..t-1]` と時刻 $\le t-1$ の推定パラメータのみに依存）。
  3. **MLE 500バー窓・50バー周期更新**: `t >= 500` かつ `(t - 500) % 50 == 0` のバー $t$ 終値後に `events[t-500 : t+1]` でパラメータを再推定し、次バー $t+1$ 以降に適用する。
  4. **ギャップ処理**: `diff_sec > 300` の再開バーでは `ret = 0.0` とし、再帰強度を `curr_h = mu` へリセットする。
- **3時刻契約**: `event_time = bar_start[t]`、`available_at = bar_start[t]`（$y_{t-1}$ までから確定）$\le$ `decision_time = bar_close[t]`。

---

## 2. Hamilton 因果性のコードレベル確定

- **参照コード**: `src/research_pipeline/discovery/econometric_modules.py` (`HamiltonLogRVRegimeModel.fit_em`, `forward_filter`)
- **因果性証明**:
  1. **日次 RV 集計区間**: UTC 取引日 $d$ の全 M5 バー（`00:00`〜`23:55 UTC` 開始バー）の対数リターン二乗和 $\text{RV}_d = \sum r_{d,b}^2$、確定時刻は $d+1$ `00:00 UTC`。
  2. **EM 推定・Quantile 初期化**: `y_win = daily_rv.iloc[d - 30 : d]["log_rv"].values`（過去 30有効取引日 $d-30 \dots d-1$ のみ参照、当日 $d$ は完全除外）。
  3. **予測確率**: `p_pred = model.P.T @ xi_filt[-1]` は $p_{d|d-1} = P(S_d = 1 \mid \mathcal{F}_{d-1})$ であり、真に 1期先予測確率。
  4. **利用可能時刻と禁止系列**: `available_at = d 00:00 UTC <= decision_time = m5_close[t]`（当日日中の M5 判断に因果的に利用可能）。フィルタ確率（`xi_filtered_at_day_close`）および平滑化確率は当日 Entry への使用を構造的に禁止する。

---

## 3. 時刻結合（as-of join）・欠損/古い値・評価再現性

- **as-of join**: `pd.merge_asof(m5_df, daily_pred_df, left_on="decision_time", right_on="available_at", direction="backward")` により `available_at <= decision_time` を実行時に強制検証。
- **未確定・欠損・古い値**: ウォームアップ期間（Hawkes `500` バー、Hamilton `30` 日）および `NaN` 時は `is_valid = False`（Entry 発火禁止）。日次データ欠落日（2日以上古い `available_at`）は `tolerance=pd.Timedelta(days=1)` で無効化し `is_valid = False` とする。
- **段階的実装順**: まず `unit:hawkes_hazard` と `unit:hamilton_pred_prob` を個別に `UNIT_REGISTRY` へ登録し、それぞれ独立に推定処理を含む未来改変不変性テスト（0 差分）を実施する。
- **Phase 2B-1 残余3項目の別枠検証**: ①依存ファイル書き換えによる `impl_id` 自動変更テスト、②`available_at > decision_time` 負例テスト、③`unit:ema` 経由での固定戦略バックテスト取引・損益一致テストを別枠で実施する。
- **不変性制約**: 既存 `causal_fx_backtest.py`、AlgoForge 本体、既存 `StrategyRegistry` 保存データは一切変更しない。
