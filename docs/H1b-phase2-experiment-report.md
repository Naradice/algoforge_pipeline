# HDE Phase 2 — 探索実験（H1a-R2, H1b-RE, H1b-ER）総合報告書

**実験期間:** 2026-11-16  
**Primary Model:** `gpt-4o-mini-2024-07-18` (temperature=0.0, seed=42)  
**評価対象データ:** Development Set（10文書：論文7件、OSS 3件）に基づく7クラスベンチマーク  
**累計消費費用（Phase 2単体）:** \$0.057821 USD（全通算でも約 \$0.134 USD / 上限 \$10.00 USD）  
**監査台帳:** data/budget_ledger.jsonl  
**生データ・実験結果:**  
- Step 2.1: data/experiments/step21_h1a_r2/  
- Step 2.2: data/experiments/step22_h1b_re/  
- Step 2.3: data/experiments/step23_h1b_er/  

---

## 1. 総合要約

レビュー承認に基づき、上流抽出と下流推論を厳密に切り分けるため、以下の3つの独立探索実験を完了しました。

1. **Step 2.1 (H1a-R2: Capability Recall & 7クラス比較):**
   - 7クラスオントロジーの導入により、OSSツール実体の弁別（Software F1: 0.40 $\to$ 0.50）および制約（Limitation Recall: 0.67 $\to$ 1.00）が大幅に向上。
   - 2段階抽出（C5）により、Capability Recallは 0.150 $\to$ 0.200（+33.3% 相対向上）を達成。一方で細粒度な動詞句候補が増加し、Precisionは低下。
   - 引用整合性（Quote Integrity）は全条件で100%を完全維持。
2. **Step 2.2 (H1b-RE: 専用関係抽出 Oracle評価):**
   - 型共起制約（RE-C0）が F1=0.8800 の極めて強力なベースラインとして機能し、負例の73%を即座に排除。
   - Few-shot CoT（RE-C2）は正例の有向性（矢印の向き）を 100% の精度（Directional Accuracy 1.0000）で獲得（Recall 100%, F1 0.6875）。
   - エビデンス厳密化（RE-C3）では境界引用不一致により1件脱落（Recall 100% $\to$ 90.9%）のトレードオフを実測。
   - **数理的ボトムラインの実証:** 上流エンティティRecallが50%の場合、下流Oracle REが90.9%であってもEnd-to-End関係獲得率は高々 $0.50^2 \times 0.909 = 22.7\%$ に制約されることを確認。
3. **Step 2.3 (H1b-ER: 文書横断エンティティ解決 Oracle評価):**
   - 密埋め込み完全連結（ER-C2, $\tau=0.82$）が **過統合 0.0%（Hard Negative 5件を完全分離）かつ F1=0.8000** を記録し最良の性能を発揮。
   - LLM境界裁定（ER-C3）は過統合0%を達成したものの、完全同一の判定基準に対して過度に保守的（過分割 100%）に倒れ、`SNPE` や `Multivariate Hawkes` を別概念として分離。同一概念（`SAME_AS`）だけでなく特殊化（`VARIANT_OF`）を区別する多値オントロジーの必要性が実証。
