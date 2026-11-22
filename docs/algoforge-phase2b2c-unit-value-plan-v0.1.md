# AlgoForge Phase 2B-2c: Unit 情報価値検証 研究計画草案 v0.1

> **ステータス**: 計画草案（未承認 — 戦略探索・バックテスト拡大・新規学習・コード変更は未実施）

## 1. 目的

Phase 2B-2a/2B-2b で因果性・時刻契約・キャッシュ分離が検証された `unit:hawkes_hazard` および `unit:hamilton_pred_prob` について、単純ベースラインに対する**情報価値（Information Value）**を既存の固定データ・限定評価枠組みで定量検証する。

## 2. 対象 Unit と予測対象・単純ベースライン

| Unit | 予測対象 | 単純ベースライン |
| :--- | :--- | :--- |
| `unit:hawkes_hazard` | 次バー（M5）における極端リターンイベント（$2\sigma$ 超）の発生強度 $h_t$ | 20バー移動標準偏差・ATR、およびイベント発生率の単純移動平均 |
| `unit:hamilton_pred_prob` | 翌日（UTC 00:00〜23:55）の高ボラティリティ・レジーム確率 $\xi_{d \mid d-1}[1]$ | 前日 Log-RV（ラグ1持続モデル）および 60日移動平均 Log-RV 閾値 |

## 3. 評価指標

- **確率・強度予測精度**: 対数尤度（Log-Likelihood）、Brier Score、ROC-AUC（イベント発生 / 高ボラ日判定）。
- **戦略条件付け有用性**: Unit 条件（例：`hawkes > θ` でのフィルタリング、`hamilton_pred_prob < θ` での低ボラ限定）を課した場合の OOS 勝率・Profit Factor・Sharpe 比のベースライン比較（過剰最適化を避けるため事前固定閾値のみ使用）。

## 4. 時系列分割と因果性検証

- **時系列分割**: `usdjpy_m5_2024_oanda_snapshot.parquet` 上で、ウォームアップ後の期間を前半（In-Sample 閾値確認）・後半（Out-of-Sample 検証）に厳密分割し、境界での情報漏洩を禁止する。
- **因果性検証**: 全評価において `apply_units_with_time_check` を経由し、`available_at <= decision_time`、未来改変不変性、および次バー始値約定を維持する。

## 5. 推定コストとキャッシュ再利用条件

- **推定コスト**: CPU 実行のみ（GPU 実験・新規深層学習なし）。Hawkes は 50 バーごとの L-BFGS-B、Hamilton は日次 EM のため、スナップショット全体でも数秒〜数十秒規模に収まる。
- **キャッシュ再利用条件**: `algoforge_unit_v1` 名前空間の `compute_algoforge_unit_evaluation_hash` を使用し、`impl_id`（`unit_registry.py` + `econometric_modules.py` SHA-256）、`params`、`dataset_sha256`、`eval_period`、`cost_config` が完全一致する場合のみ再利用する。
