# AlgoForge Phase 2B-1 実装計画草案（v0.2）

**日付**: 2026-11-22  
**前提**: Phase 2A 設計書 v0.1（条件付き受入）  
**状態**: 計画承認待ち（実装・実験未着手）  

---

## 1. Phase 1B 互換性と新マスク仕様の分離

- **対象指標（2種）**: `unit:ema`、`unit:rsi`（Hawkes/Hamilton は Phase 2B-2 へ分離）。
- **数値・NaN 一致**: `unit:ema` / `unit:rsi` の数値および `NaN` 位置は既存 AlgoForge `apply_indicators` と **完全一致** させる。
- **有効区間 vs 無効区間**: 有効入力区間では条件評価結果を Phase 1B と完全一致させ、無効入力区間（ウォームアップ・`NaN`・非有限値）では新マスク仕様（`is_valid = False` による発火禁止）を優先する。バックテスト結果に差異が生じた場合は発生位置と原因を報告する。
- **既存経路の保護**: 既存の `pipeline_v1` および `algoforge_v1` 経路・保存データは一切変更しない。

---

## 2. Unit 識別とキャッシュ契約（`algoforge_unit_v1`）

- **`impl_id` の生成・検証**: Unit 実装関数のソースコード（`inspect.getsource(fn)`）の SHA-256 先頭 16文字として自動生成し、実装変更時に必ず変化することを検証する。
- **依存情報の正規化**: `unit_versions` として `{unit_id: {"version": str, "impl_id": str, "dependencies": sorted(list)}}` を `canonicalize_condition_node` で再帰正規化し、`compute_algoforge_evaluation_hash` へ渡す（ネストした依存情報も決定論的にハッシュ化）。
- **新旧キャッシュ混用禁止**: `schema_version = "algoforge_unit_v1"` を新設し、既存の `pipeline_v1` / `algoforge_v1` とキャッシュを完全に分離する。

---

## 3. マスク・時刻契約と対応範囲

- **定数と有効性判定**: 有限の定数は有効、非有限の定数（`NaN`, `inf`, `-inf`）はバリデーションで拒否する。Unit 出力の有効性判定には `np.isfinite(...)` を使用する。
- **AND / OR / NOT 有効性伝播**: 保守的伝播（`NOT(A)` は `(~A.val) & A.valid`、`AND`/`OR` は `A.valid & B.valid`）を維持し、欠損・ウォームアップ・無限大・`NOT` を含むテストを実施する。
- **時刻契約**: `available_at <= decision_time`（M5 バー確定時刻 $t_{\text{close}} = t_{\text{bar\_start}} + 5\text{分}$、次バー始値約定）を満たすデータのみを Decision 層へ渡す。
- **対応範囲と名前衝突**: 多段入れ子論理式を暗黙に有効化せず（1階層 `and`/`or` のみ）、Unit 出力列と `{id}_valid` マスク列の名前衝突を検出・拒否する。

---

## 4. 変更対象ファイル・テスト・停止条件

- **変更対象**: `src/research_pipeline/integration/unit_registry.py`（新規）、`src/research_pipeline/integration/algoforge_adapter.py`（`unit:*` 解決とマスク伝播）、`scripts/test_phase2b1_unit_registry.py`（新規）。
- **停止条件**: 因果性違反、マスク誤発火（欠損/無限大/`NOT`）、キャッシュ誤再利用を検出した場合は即時停止する。
