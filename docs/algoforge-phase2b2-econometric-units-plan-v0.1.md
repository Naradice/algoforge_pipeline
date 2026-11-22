# AlgoForge Phase 2B-2 監査結果および実装計画草案（v0.1）

**日付**: 2026-11-22  
**前提**: Phase 2B-1 条件付き受入（コミット `e364fea`）  
**状態**: 監査完了・計画承認待ち（実装・再学習・バックテスト未着手）  

---

## 1. Phase 2B-1 残余確認（証跡と未確認項目）

- **確定証跡**: コミット `e364fea`、`unit_registry.py`、`algoforge_adapter.py`、`test_phase2b1_unit_registry.py`、`algoforge-phase2b1-unit-registry-plan-v0.3.md`。
- **未確認項目（Phase 2B-2 テストへ組み込み解消予定）**:
  1. 依存ファイル内容の実際の書き換えによる `impl_id` 自動変更テスト（2B-1 ではモック `impl_id` 変更のみ確認）。
  2. `available_at > decision_time` となる時刻契約違反の負例テスト。
  3. Phase 2B-1 `unit:*` 経由での固定戦略バックテスト取引・損益一致確認（1B-1 では確認済みだが 2B-1 経由では未実行）。

---

## 2. Phase 2B-2 読み取り専用監査結果（`econometric_modules.py`）

### 2.1 `HawkesDiscreteHazardModel`
- **入力・粒度・履歴長**: M5 対数リターン `ret`（ギャップ時 `0.0`）、20バー後方標準偏差の `2.0x` 超過イベント $y_t \in \{0, 1\}$。最小推定履歴長 `500` バー。
- **推定方法・推論範囲**: 直近 `500` バーの MLE（50バー周期更新）。強度は全履歴 IIR 再帰 `lfilter` により $h_t > 0$ を算出（ギャップ時は $h_{\text{reopen}} = \mu$ リセット）。
- **3時刻契約**: `event_time = bar_start[t]`。バー $t$ の強度 $h_t$ は $y_{t-1}$ までから確定するため `available_at = bar_start[t] <= decision_time = bar_close[t]`。未来情報混入なし。

### 2.2 `HamiltonLogRVRegimeModel`
- **入力・粒度・履歴長**: 日次実現ボラティリティ対数 $\log(\text{RV}_d)$（M5 288バー二乗和）。最小推定履歴長 `30` 有効取引日。
- **推定方法・出力の区別**: 直前 30取引日 `[d-30 : d-1]` での EM 推定（Quantile 初期化）。
  - **`xi_predicted_before_day` ($p_{d|d-1} \in [0, 1]$)**: 取引日 $d$ の `00:00 UTC` に確定（`available_at = d 00:00 UTC <= decision_time = m5_close[t]`）。**因果的に利用可能（Phase 2B-2 実装対象）**。
  - **`xi_filtered_at_day_close` ($\xi_{d|d}$)**: 日次確定後（$d+1$ `00:00 UTC`）にのみ利用可能。当日 $d$ の日中は未来情報のため使用禁止。
  - **平滑化確率・`xi_next_fit`**: 未来データ参照を含むため **バックテスト利用禁止**。

---

## 3. Phase 2B-2 実装計画草案（v0.1）

- **実装対象 Unit（因果性確認済みのみ）**:
  1. `unit:hawkes_hazard`（`continuous`、ウォームアップ `500` バー）
  2. `unit:hamilton_pred_prob`（`probability` = `xi_predicted_before_day` のみ、ウォームアップ `30` 取引日）
- **M5 × 日次 as-of join**: `merge_asof(..., direction='backward')` により `available_at <= decision_time` を実行時に強制検証。
- **Phase 2B-1 契約の再利用**: ウォームアップ期間および非有限値は `is_valid = False` とし、`eval_condition_with_mask`（Decision 層閾値比較）から `causal_fx_backtest.py` へ boolean Entry-only 接続を行う。
- **Unit Manifest**: `econometric_modules.py` と `unit_registry.py` のファイル SHA-256 を `impl_id` に含め、パラメータ・推定設定を `evaluation_hash` へ反映。
- **受入テスト**: ①Phase 2B-1 残余3項目（依存ファイル変更 `impl_id` テスト、時刻違反負例、固定戦略バックテスト一致）、②Hawkes/Hamilton 未来改変不変性（複数切断位置で 0 差分）、③`algoforge_unit_v1` キャッシュ分離。
