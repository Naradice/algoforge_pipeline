# AlgoForge Phase 2B-2a 限定修正計画 v0.2

## 1. 時刻契約の厳格化 (`algoforge_adapter.py` / `unit_registry.py`)

1. **無条件上書きの廃止**:
   - `apply_units_with_time_check` は `work["available_at"] = work["decision_time"]` のような無条件上書きを行わない。
2. **Unit ごとの時刻契約 (`time_contract`)**:
   - `UnitSpec` に `time_contract`（`"bar_close"` または `"explicit_available_at"`）を定義する。
   - **バー終値確定型 Unit**（`unit:ema`, `unit:rsi`, `unit:hawkes_hazard`）：`available_at` を返さない場合のみ、明示された契約に従い `bar_close`（`time + 5min`）を `available_at` として生成する。
   - **独自時刻を持つ Unit**（`"explicit_available_at"`）：出力 DataFrame に `available_at` が欠落している場合は **Fail-Closed（`ValueError`）** とする。
3. **入力時刻のタイムゾーン検証（暗黙 UTC 解釈の禁止）**:
   - 入力 `df["time"]` および Unit 出力 `u_df["available_at"]` について、**tz-naive（タイムゾーン情報なし）を暗黙に `utc=True` で解釈しない**。
   - tz-aware（例：UTC やオフセット付き）であることを検証した上で `.dt.tz_convert("UTC")` で正規化し、tz-naive・`NaT`・型不整合は **Fail-Closed（`ValueError`）** とする。
   - すべてのバーで `available_at <= decision_time` を検証し、違反時は `RuntimeError("CAUSALITY_VIOLATION")` を送出する。

## 2. Unit 識別子と返却列名の契約統一

1. **本番 Strategy JSON の正規形式**:
   - 正規キーを `"type"` とする。互換キー（`"unit_id"`, `"unit"`）は、`type` が未指定かつ単一キーのみ指定された場合のみ許容し、**複数の識別子キーが矛盾した値で同時指定された場合は Fail-Closed（`ValueError`）** とする。
2. **返却列名の統一**:
   - 全 Unit の返却 DataFrame を `{col_id: values, f"{col_id}_valid": is_valid}`（および必要時 `"available_at"`）に統一する。`_unit_hawkes_hazard` の `{"value", "is_valid"}` 返却を修正する。

## 3. Hawkes `h[500]` の定義確定

- `_unit_hawkes_hazard` における `h[500]` は、バー 500 終値確定後の `events[0:501]`（501件）を用いた初回 MLE 後のパラメータ `(mu, rho, gamma)` と `events[500]` を用いた **事後更新強度** として定義する。
- したがって `h[500]` の利用可能時刻は **バー 500 終値確定時刻（`bar_500_start + 5min == decision_time`）** であり、バー 500 開始時点では利用不可とする。

## 4. 回帰テスト計画 (`scripts/test_phase2b2a_hawkes.py`)

1. 時刻負例テスト：`available_at > decision_time`、`NaT`、tz-naive（タイムゾーン不整合）、独自時刻 Unit での `available_at` 欠落、矛盾した Unit 識別子指定のすべてで Fail-Closed となること。
2. `unit:ema` 固定戦略のバックテスト取引・損益一致テスト（`"type": "unit:ema"` 使用）。
3. Hawkes 推定設定（`mle_window`, `refit_interval`）変更時の評価ハッシュ分離および同条件再現性。
