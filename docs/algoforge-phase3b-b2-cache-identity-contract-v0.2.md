# Phase 3B / B2: 評価設定・同一性・キャッシュ契約 詳細設計案 v0.2

> **ステータス**: 詳細設計案 v0.2（未承認 — コード変更・実験実行は未実施）
> **対象 Issue**: [Issue #3](https://github.com/Naradice/algoforge_pipeline/issues/3)
> **対象レビュー**: `issuecomment-7076356708`（修正依頼 3 点反映）

## レビュー指摘（3 点）への対応

1. **評価条件の正規化契約（全項目対象・Fail-Closed）**:
   - 代表例に限定せず、**評価結果に影響する全設定項目**（`backtest_config`, `gate_config`, `cost_config`, `execution_timing`, `unit_versions`, 特徴量/前処理設定等）を正規化対象とする。
   - **正規化規則**:
     - 辞書キーは再帰的にソート（`sort_keys=True`）、リスト順序は意味を持つものとして保持。
     - 数値は `float` / `int` をそのまま保持し、`NaN` / `Inf` / `-Inf` 等の非有限値は **Fail-Closed（`ValueError`）** とする。
     - 日時は UTC ISO8601 文字列へ正規化、デフォルト値は展開済みの明示値として含める。
     - シリアライズ不能値（`asdict` 不可・JSON 変換不可）は **Fail-Closed（`ValueError`）** とする。
   - **テスト追加**: 等価な設定（キー順違い等）で同一ハッシュ、評価条件の任意 1 項目差分で異なるハッシュとなることを検証する。
2. **`unverified` の再利用方針を一本化**:
   - データ内容の同一性を検証できない評価（`identity_verified == False`）は、結果の証跡保存は可能であっても **キャッシュヒット／再利用を全面禁止** とする（「専用タグで分離するだけ」の代替案は採用しない）。
3. **エンジン同一性**:
   - `engine_id` / `engine_version` に加え、評価結果を変えうる実装リビジョン（`engine_impl_id`）および実行環境設定の識別責務を定義する。
   - API 側で実装識別ができない場合は「同一性未検証（`engine_verified == False`）」として **キャッシュ再利用禁止** とする（具体的 API マッピングは B4 で確定）。

## 1. 旧キャッシュ互換維持（v0.1 からの維持事項）

- `LocalCausalEvaluator`（`causal_fx_backtest_v1.0`）の既存 `compute_algoforge_evaluation_hash`（`algoforge_v1`）および `StrategyRegistry` スキーマは一切変更しない。
- 新評価契約名前空間 `algoforge_eval_v2` は旧 `algoforge_v1` と物理的・論理的に分離する。
