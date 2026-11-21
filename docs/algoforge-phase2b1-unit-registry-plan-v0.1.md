# AlgoForge Phase 2B-1 実装計画草案（v0.1）

**日付**: 2026-11-22  
**前提**: Phase 2A 設計書 v0.1（条件付き受入）  
**状態**: 計画承認待ち（実装・実験未着手）  

---

## 1. Phase 2B-1 の対象と境界

- **検証対象（決定論的既存指標 2種）**: `unit:ema` および `unit:rsi`
- **分離事項**: Hawkes / Hamilton の接続は **Phase 2B-2** として分離し、今回は実装しない。既存バックテスター（`causal_fx_backtest.py`）、新規モデル学習、探索実験、GPU 実験、AlgoForge 本体の変更はすべて対象外。

---

## 2. Unit インターフェース契約と有効性マスク伝播仕様

### 2.1 Unit インターフェース (`UnitSpec`)
- **メタデータ**: `unit_id`（例: `"unit:ema"`）、`version`（例: `"1.0.0"`）、`impl_id`（実装関数ハッシュ/識別子）、`dependencies`（依存 Unit リスト）、`warmup_bars`、`output_schema`（`{col: "continuous"}`）。
- **出力契約**: すべての Unit は値列 `df[id]` と対になる有効性マスク列 `df[f"{id}_valid"]`（boolean）を出力する。ウォームアップ期間中および入力 `NaN` 時は `is_valid = False`。

### 2.2 Decision 層での有効性伝播規則（AND / OR / NOT）
- **原子比較 `(L op R)`**: `valid = L_valid & R_valid & (~isnan(L)) & (~isnan(R))`、`val = (L op R) & valid`。
- **`NOT(A)`**: `val = (~A.val) & A.valid`、`valid = A.valid`（**欠損・ウォームアップ時に `NOT` が `True` へ誤反転することを完全禁止**）。
- **`AND(A, B)`**: `val = A.val & B.val`、`valid = A.valid & B.valid`。
- **`OR(A, B)`**: `val = (A.val | B.val) & (A.valid & B.valid)`、`valid = A.valid & B.valid`（安全側：両方が有効な場合のみ発火）。
- **最終 Entry シグナル**: `entry = decision_val & decision_valid`（欠損時の意図しない Entry 発火を構造的に禁止）。

---

## 3. Strategy JSON 参照と `evaluation_hash` 構成

- **JSON 参照**: `indicators` ブロック内に `{"id": "ema12", "type": "unit:ema", "params": {"period": 12}}` を記述。
- **キャッシュ識別（`evaluation_hash`）**: 既存 `compute_algoforge_evaluation_hash` の `unit_versions` 引数に `{unit_id: {"version": ..., "impl_id": ..., "dependencies": ...}}` を渡すことで、Unit 実装・バージョン変更時のキャッシュ誤再利用を防止する（既存 `pipeline_v1` / `algoforge_v1` の保存データは一切変更しない）。

---

## 4. 変更対象ファイル・テスト・受入基準・停止条件

- **変更対象**:
  1. `src/research_pipeline/integration/unit_registry.py`（新規：`UnitSpec` と `UNIT_REGISTRY` 最小実装、`unit:ema`, `unit:rsi` 登録）
  2. `src/research_pipeline/integration/algoforge_adapter.py`（`unit:*` 解決と `is_valid` マスク伝播対応）
  3. `scripts/test_phase2b1_unit_registry.py`（新規：契約・回帰・因果性・キャッシュテスト）
- **必須テストと受入基準**:
  1. **Phase 1B 回帰一致**: `unit:ema` / `unit:rsi` 経由の指標値・シグナル・バックテスト結果が Phase 1B-1 と 100% 一致。
  2. **有効性マスク伝播**: ウォームアップ期間中および意図的 `NaN` 注入時に `NOT` を含む条件が一切発火しない（誤発火 0件）。
  3. **因果性**: 複数切断位置での未来改変不変性 **0 差分**。
  4. **キャッシュ分離**: `unit_versions`（`version` / `impl_id`）変更時に `evaluation_hash` が変化し、キャッシュ誤再利用が起きないこと。
- **停止条件・残余リスク**: 未来改変不変性違反またはマスク誤発火が 1件でも検出された場合は即座に停止。残余リスクとして、Phase 2B-2 で Hawkes/Hamilton を接続する際には学習期間・推定期間・予測/フィルタ確率の分離監査が別途必要となる。
