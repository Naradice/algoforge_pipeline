# AlgoForge Phase 2B-2a 契約不具合の原因分析・限定修正計画 v0.1

## 1. 原因分析（読み取り専用調査結果）

### 1.1 最優先：`apply_units_with_time_check` の時刻契約不具合
- **該当箇所**: `src/research_pipeline/integration/algoforge_adapter.py` L188-211
- **原因**:
  1. L190-191 で `work["decision_time"] = work["time"] + pd.Timedelta(minutes=5)` および `work["available_at"] = work["decision_time"]` と無条件に上書きしており、入力 `df` の `available_at` / `decision_time` を検証していない。
  2. L202 で `u_df = u.fn(work, col_id, spec.get("params", {}))` を呼んだ後、`u_df` に含まれる `available_at` 列を一切検証せず、`work[col_id]` と `work[f"{col_id}_valid"]` のみをコピーしている。
  3. 結果として、Unit が `available_at > decision_time`（未来時刻）や `NaT`、タイムゾーン不整合を返しても `RuntimeError` / `ValueError` が発生せず通過していた。

### 1.2 第二優先：Hawkes 初回有効バーと `h[500]` の依存関係
- **該当箇所**: `src/research_pipeline/integration/unit_registry.py` L176-188 (`_unit_hawkes_hazard`)
- **現状**:
  - `t = 500`（501本目）で `window_events = events[0:501]` を用いて `model.fit(...)` を実行し、その直後に得られた `(mu, rho, gamma)` を用いて `values[500]` (`h[500]`) を計算している。
  - バー 500 の終値確定後（`decision_time = bar_500_start + 5min`）に `events[500]` と初回 MLE が確定するため、バー 500 終値時点のシグナルとしては `available_at == decision_time` となるが、バー 500 開始時点では利用不可。
  - また `_unit_hawkes_hazard` の返却列名が `{"value", "is_valid"}` となっており、`apply_units_with_time_check` が期待する `{col_id, f"{col_id}_valid"}` と不一致であった。
- **ギャップ時の補完・リセット**:
  - `diffs > gap_seconds`（300秒超）のバーで `events[gap_mask] = 0` とし、`curr_h = model.mu` にリセットしている。

### 1.3 第三優先：Phase 2B-1 残余テスト（`unit` vs `unit_id` 契約）
- **該当箇所**: `algoforge_adapter.py` L197 (`uid = spec.get("type", "")`)
- **原因**: `apply_units_with_time_check` は `spec.get("type", "")` を参照していたが、`test_phase2b2a_hawkes.py` では `"unit"` を渡したため `UNKNOWN_UNIT:` となった。

## 2. 限定修正計画（承認後に実施）

1. **`algoforge_adapter.py` (`apply_units_with_time_check`)**:
   - `u_df` が `available_at` を返す場合（および返さない場合は `work["decision_time"]` を既定値とする）、`pd.to_datetime(..., utc=True)` で UTC 正規化し、**欠落・`NaT`・タイムゾーン不整合・`available_at > decision_time`** のいずれかを検出した場合に **Fail-Closed（`ValueError` / `RuntimeError` 送出）** とする。
   - `spec.get("type") or spec.get("unit_id") or spec.get("unit")` の優先順位で Unit ID を解決し、API 契約を統一する。
2. **`unit_registry.py` (`_unit_hawkes_hazard`)**:
   - 返却 DataFrame の列名を `{col_id: values, f"{col_id}_valid": is_valid, "available_at": available_at}` に統一する。
   - `h[500]` はバー 500 終値確定後（`available_at = bar_500_close == decision_time`）にのみ有効とし、バー 500 開始時点での先読みを禁止する。
3. **`scripts/test_phase2b2a_hawkes.py`**:
   - 時刻契約負例テスト（未来時刻・`NaT`・タイムゾーン不整合）を追加し全件 PASS を確認する。
   - `unit:ema` 固定戦略のバックテスト取引・損益一致テストを再検証する。
   - Hawkes 推定設定（`mle_window`, `refit_interval`）変更時の評価ハッシュ分離・同条件安定性を検証する。
