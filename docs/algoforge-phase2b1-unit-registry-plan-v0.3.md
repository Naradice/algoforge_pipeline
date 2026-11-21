# AlgoForge Phase 2B-1 実装計画草案（v0.3）

**日付**: 2026-11-22  
**前提**: Phase 2A 設計書 v0.1（条件付き受入）  
**状態**: 計画承認待ち（実装・実験未着手）  

---

## 1. 実装識別子（`impl_id`）のファイル SHA-256 化と Fail-Closed

- `inspect.getsource(fn)` 単独の SHA-256 を廃止する。
- `impl_id` は **Unit 実装ファイル（`unit_registry.py`）および依存する計算コード（`algoforge/backend/strategy/engine/indicators.py` 等）のファイル SHA-256** をソート済みパス順で連結・ハッシュ化して生成する。
- 識別に必要なファイルが存在しない・読み取れない場合は **fail-closed**（例外送出・Unit 登録拒否）とする。
- 依存ファイル内容が変化した場合に `impl_id` および評価キャッシュが分離されることを契約テストで確認する。

---

## 2. Unit Manifest 正規化と `algoforge_unit_v1` 評価ハッシュ

- 条件式正規化とは独立した `canonicalize_unit_manifest(units)` を新設し、辞書順序・依存順序の正規化、**重複 Unit ID 検出**、および **循環依存検出**（検出時は例外送出）を実装する。
- 既存の `pipeline_v1` と `algoforge_v1` のハッシュ・キャッシュ経路は一切変更しない。
- `algoforge_unit_v1` 専用の評価ハッシュ生成ラッパー `compute_algoforge_unit_evaluation_hash`（`schema_version="algoforge_unit_v1"`＋正規化済み Unit Manifest＋戦略ハッシュ＋データ/期間/コスト）を設ける。

---

## 3. 条件式の対応範囲とマスク契約

- **本番 Strategy JSON**: **1階層 AND/OR と原子比較（`>`, `<`, `>=`, `<=`, `==`, `!=`）のみ** 対応。多段入れ子式は暗黙に有効化せず拒否する。
- **`NOT` の扱い**: `NOT` は Decision 内部のマスク伝播契約テスト（`(~A.val) & A.valid`）に限定する。
- **有効性契約**: 有限定数のみ許可（非有限定数は拒否）、Unit 出力は `np.isfinite` で有効性マスクを生成し、欠損・無限大・ウォームアップ時の誤発火を **0件** とする。出力列と `{id}_valid` の名前衝突を検出・拒否する。

---

## 4. 時刻・因果性契約と実施範囲

- **時刻契約**: バー開始時刻 `bar_start`・終了時刻 $t_{\text{close}} = t_{\text{bar\_start}} + 5\text{分}$ を明示し、`available_at <= decision_time` を実行時に検証する（時刻列欠落・規約不明な入力は拒否）。複数切断位置で未来改変不変性を確認する。
- **変更対象（3ファイル限定）**:
  1. `src/research_pipeline/integration/unit_registry.py`
  2. `src/research_pipeline/integration/algoforge_adapter.py`
  3. `scripts/test_phase2b1_unit_registry.py`
- 既存バックテスター、AlgoForge 本体、既存 Registry 保存データ、Phase 0〜4 履歴は変更しない。
