# HDE H1a — Capability Extraction 実験計画書 v0.1

**Status:** Proposed (Review Pending)  
**Date:** 2026-11-15  
**Version:** 0.1  
**Target:** Hypothesis Discovery Engine (HDE) Phase 1 / Step H1a  
**Parent Specifications:**
- [architecture.md](file:///c:/Users/ksato/workspace/finance/research_pipeline/docs/architecture.md)
- [hypothesis_discovery_engine.md](file:///c:/Users/ksato/workspace/finance/research_pipeline/docs/hypothesis_discovery_engine.md)
- [H0-completion-report.md](file:///c:/Users/ksato/workspace/finance/research_pipeline/docs/H0-completion-report.md)

---

## 1. 目的と位置づけ

### 1.1 目的
Step H1a の目的は、H0 で収集・保存された `ResearchDocument`（論文 Abstract および OSS README、メタデータ）を入力とし、そこから **Model（モデル）**、**Method（手法）**、**Capability（操作可能な能力）**、**Assumption（前提仮定）**、**Limitation（適用制約）** を高精度かつ再現可能な形で抽出し、原典版に厳密に紐付いた構造化研究知識（`ExtractedKnowledgeRecord`）を生成することである。

### 1.2 設計原則と境界制約
1. **Evidence Grounding（原典根拠性の絶対担保）**:
   - 抽出されるすべての項目は、入力テキスト中の直接の引用文（`verbatim_quote`）を伴わなければならない。根拠のない推測（Hallucination）や、一般的な知識の補完による捏造を厳格に排除する。
2. **Scope-Constrained Extraction（情報範囲に応じた抽出制限）**:
   - 入力が `abstract_only` または `readme_only` である場合、原典のすべての詳細（完全なハイパーパラメータや数学的証明）が含まれていないことを前提とし、抽出される知識のエビデンス状態は **`reported`（原典著者の主張）** に限定する。**「実証済み（reproduced）」や「完全検証済み」として扱ってはならない**。
3. **Controlled Vocabulary 準拠**:
   - 後段の Cross-Domain Composition（H2）で入出力型や時間解像度を安全に接続できるよう、入出力データ型は自由記述ではなく、定義された標準データ型語彙（Controlled Vocabulary）にマッピングする。
4. **実装コード・実験の開始禁止**:
   - 本書は実験計画の策定のみをスコープとし、**コード実装、モデル API 呼び出し、実験実行は計画承認まで一切行わない**。

---

## 2. 抽出対象の定義とスキーマ

抽出対象は 5 つの主要エンティティから構成される。各エンティティは必ず原典の引用（`EvidenceSpan`）を保持する。

```text
       ┌─────────────────────────────────────────────────────────────┐
       │                 ExtractedKnowledgeRecord                    │
       │  (source_id, version_id, extractor_version, content_scope)  │
       └──────┬──────────────┬──────────────┬─────────────┬──────────┘
              │              │              │             │
              ▼              ▼              ▼             ▼
          [Model]        [Method]     [Capability]   [Limitation]
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                        [Assumption]
```

### 2.1 スキーマ定義（Pydantic 設計案）

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

class EvidenceSpan(BaseModel):
    """テキスト内の直接引用と位置情報"""
    verbatim_quote: str = Field(..., description="原文からの一致抜粋文字列 (完全一致)")
    start_char: int = Field(..., description="抽出元テキスト内の開始文字インデックス")
    end_char: int = Field(..., description="抽出元テキスト内の終了文字インデックス")
    field_context: Literal["abstract", "readme", "title", "metadata"] = Field(..., description="引用箇所")

class AssumptionItem(BaseModel):
    """成立に必要な前提条件・仮定"""
    assumption_id: str = Field(..., description="一時識別子 (例: 'asm_01')")
    statement: str = Field(..., description="仮定の内容 (例: 'エージェントは有限の記憶容量を持つ')")
    is_explicit: bool = Field(..., description="原文に明示されているか (True)、論理的必然として明記された前提か")
    evidence: EvidenceSpan

class LimitationItem(BaseModel):
    """適用上の制約・未解決課題"""
    limitation_id: str = Field(..., description="一時識別子 (例: 'lim_01')")
    statement: str = Field(..., description="制約の内容 (例: '高ボラティリティ環境では収束が遅い')")
    evidence: EvidenceSpan

class ModelEntity(BaseModel):
    """数学モデル・シミュレータ・アーキテクチャ"""
    model_id: str = Field(..., description="一時識別子 (例: 'mod_01')")
    name: str = Field(..., description="モデル名称 (例: 'Minority Game', 'Transformer-XL')")
    category: Literal["agent_based", "statistical_physics", "state_space", "deep_learning", "stochastic_process", "other"]
    description: str = Field(..., description="モデルの数理的・構造的説明")
    evidence: EvidenceSpan

class MethodEntity(BaseModel):
    """推定・学習・分析・検証アルゴリズム"""
    method_id: str = Field(..., description="一時識別子 (例: 'met_01')")
    name: str = Field(..., description="手法名称 (例: 'Neural Posterior Estimation', 'ADF Test')")
    category: Literal["parameter_inference", "anomaly_detection", "state_estimation", "representation_learning", "optimization", "other"]
    description: str = Field(..., description="アルゴリズムの機能説明")
    evidence: EvidenceSpan

class CapabilityIO(BaseModel):
    """入出力データの仕様"""
    data_type: str = Field(..., description="Controlled Vocabulary 準拠のデータ型 (例: 'price_timeseries')")
    description: str = Field(..., description="データの具体的意味 (例: 'M1 USDJPY Close価格系列')")
    time_resolution: Optional[Literal["tick", "sub_minute", "minute", "hour", "daily", "static"]] = Field(None)
    dimensionality: Optional[str] = Field(None, description="次元数 (例: '1D', 'OHLCV_5D', 'N_agents')")

class CapabilityEntity(BaseModel):
    """モデルや手法が提供する操作可能な能力"""
    capability_id: str = Field(..., description="一時識別子 (例: 'cap_01')")
    name: str = Field(..., description="能力名称 (例: 'Parameterized Price Generation')")
    operation: Literal["generate", "infer", "detect", "predict", "transform", "optimize"] = Field(...)
    inputs: List[CapabilityIO] = Field(..., min_length=1, description="必要な入力データ型")
    outputs: List[CapabilityIO] = Field(..., min_length=1, description="生成・推定される出力データ型")
    associated_model_id: Optional[str] = Field(None, description="紐付く ModelEntity の ID")
    associated_method_id: Optional[str] = Field(None, description="紐付く MethodEntity の ID")
    preconditions: List[str] = Field(default_factory=list, description="実行に必要な前提条件")
    limitations: List[str] = Field(default_factory=list, description="適用限界")
    evidence: EvidenceSpan

class ExtractedKnowledgeRecord(BaseModel):
    """H1a の完全な出力レコード"""
    source_id: str = Field(..., description="論理資料ID (H0)")
    version_id: str = Field(..., description="資料版ID (H0)")
    content_scope: Literal["abstract_only", "readme_only", "metadata_only", "full_text"]
    evidence_status: Literal["reported", "unverified"] = Field(
        "reported",
        description="Abstract/README からの抽出物は常に 'reported' (著者の主張) として扱う"
    )
    models: List[ModelEntity] = Field(default_factory=list)
    methods: List[MethodEntity] = Field(default_factory=list)
    capabilities: List[CapabilityEntity] = Field(default_factory=list)
    assumptions: List[AssumptionItem] = Field(default_factory=list)
    limitations: List[LimitationItem] = Field(default_factory=list)
    extractor_model: str = Field(..., description="抽出に使用した LLM モデル名と版")
    extractor_prompt_version: str = Field(..., description="プロンプトテンプレートの版 (例: 'prompt_v1.0')")
    extracted_at: str = Field(..., description="抽出実行日時 (ISO8601)")
```

---

## 3. Capability の入出力・制約の表現（Controlled Vocabulary）

異なるドメインの研究 Capability を後段で自動接続（Composition）できるようにするため、入出力型を以下の標準データ型語彙（第 1 版）に制限・正規化する。

### 3.1 標準データ型語彙（Controlled Vocabulary v1.0）

| データ型名 (`data_type`) | 説明 | 主な属性（解像度・次元） |
|---|---|---|
| `price_timeseries` | 1次元の価格系列（Close, Mid等） | tick, minute, hour, daily |
| `ohlcv_timeseries` | 5次元の標準ローソク足時系列 | minute, hour, daily |
| `orderbook_snapshot` | 板情報（Bid/Ask の価格・数量リスト） | tick, depth=N |
| `trade_events` | 約定イベント系列（Time, Price, Size, Side） | tick |
| `agent_parameters` | エージェントモデルの構造パラメータ群 | static / vector |
| `latent_state` | 隠れマルコフや状態空間の潜在状態系列 | sub_minute 〜 daily, vector |
| `probability_distribution`| 将来予測やパラメータの事後確率分布 | parametric (Normal, Student-t) / empirical |
| `prediction_score` | 回帰スコア、期待リターン等の連続値 | point-in-time scalar |
| `anomaly_score` | 異常度スコア、ジャンプ検知指標 | point-in-time scalar / flag |
| `feature_vector` | 前処理済み多次元特徴量ベクトル | static / time-series 2D |

---

## 4. Abstract と README に応じた抽出制限

入力情報範囲（`content_scope`）に応じ、LLM 抽出器に対してプロンプトおよび後処理で以下の厳格な制限を課す。

1. **`abstract_only`（論文要約）の場合**:
   - **抽出許可**: 主要な現象（Phenomenon）、提案モデル（Model）、コア推定手法（Method）、主要な主張（Capability: "〜を推定可能"）、明示的な適用限界（Limitation）。
   - **抽出禁止・拒否**: 論文本文にしか書かれていないはずの詳細な数値パラメータ、学習率、レイヤー数、数学的証明の完全性。
   - **エビデンス状態**: 必ず **`reported`** とする（著者による報告・主張であり、当システムでの再現・実証ではない）。
2. **`readme_only`（OSS README）の場合**:
   - **抽出許可**: 提供される API/CLI 操作（Capability）、対応データ形式（Inputs/Outputs）、実行前提（依存関係・Pythonバージョン等の Preconditions）、明示された注意点。
   - **抽出禁止・拒否**: README に書かれていないアルゴリズムの内部数理。
   - **エビデンス状態**: `reported` または `unverified`（実装の存在報告）。
3. **`metadata_only`（README 欠落リポジトリ等）の場合**:
   - LLM による深い推論はスキップし、リポジトリ説明文（description）やトピック（topics）から最低限のカテゴリ分類のみを抽出し、Capability 抽出は行わない。

---

## 5. Evidence の引用・位置特定・原典版への紐付け

抽出の信頼性を担保するため、抽出された全エンティティに `EvidenceSpan` の付与を義務付ける。

### 5.1 Evidence 検証規則（バリデーションアルゴリズム）
1. **完全一致性（Exact Match）**:
   - `input_text[start_char:end_char] == verbatim_quote` であることをプログラム（正規表現・スライス）で決定論的に検証する。1 文字でも不一致（句読点、空白の改変）がある場合は抽出不正としてリジェクト。
2. **最小長・最大長制約**:
   - `verbatim_quote` の長さは 10 文字以上、300 文字以下とする（単語 1 語の引用や、段落全体の無意味な長文引用を禁止）。
3. **原典版の固定**:
   - 抽出結果は、H0 の `source_versions.version_id`（例: `arxiv:2301.12345:v3`）に外部キーとして 1 対 1 で紐付けられ、原典の改定時に知識が混濁しないようにする。

---

## 6. LLM 抽出器の構成とプロンプト・モデル管理

### 6.1 抽出器の構成
- **モデル**: `Gemini 1.5 Pro`（または `Claude 3.5 Sonnet` / `Gemini 2.0 Flash`）
  - ※決定論的再現性を担保するため、**Temperature = 0.0**, `top_p = 1.0`, `seed = 42`（サポート時）に固定する。
- **モデル版管理**: 使用したモデル識別子（例: `gemini-1.5-pro-002`）を `ExtractedKnowledgeRecord.extractor_model` に完全記録する。
- **プロンプト版管理**: プロンプトテンプレートを `prompts/extraction_v1_0.txt` などの不変ファイルとしてバージョン管理し、ハッシュとともに `extractor_prompt_version` に記録する。

### 6.2 構造化出力（Structured Outputs / JSON Schema）
LLM の API に Pydantic から生成した厳密な JSON Schema を渡し、JSON 形式を強制する。

### 6.3 失敗時のハンドリング手順
1. **JSON パースエラー / スキーマ違反**:
   - 最大 2 回まで、エラーメッセージをフィードバックして再抽出（Self-Correction）。
   - それでも修正されない場合は、当該ドキュメントの抽出を `failed_validation` として隔離ログに記録。
2. **Evidence 引用検証エラー（引用文が原文に存在しない）**:
   - Hallucination とみなし、当該エンティティを破棄。
   - 全体として 1 つも有効なエンティティが残らない場合は `failed_ungrounded` として記録。

---

## 7. 人手確認済み評価データセット（Gold Standard）の構築

LLM 抽出器の性能と信頼性を科学的に検証するため、H1a 実装・比較実験に先立ち **15 件のゴールドスタンダード（正解アノテーション）データセット** を構築する。

### 7.1 対象ドキュメントの選定基準（15 件）
多様性を担保するため、以下の 3 カテゴリから選定する。

1. **金融・市場モデリング論文（5 件）**:
   - エージェントベース市場モデル（ABM）
   - 指数平滑化・ボラティリティ推定
   - 高頻度板情報・オーダーフローダイナミクス
2. **他分野・機械学習・複雑系論文（5 件）**:
   - Simulation-Based Inference（SBI / NPE）
   - 状態空間モデル・カルマンフィルタ
   - 時系列異常検知（時系列表現学習）
3. **金融・時系列関連 OSS リポジトリ（5 件）**:
   - 金融シミュレータ（Gym/Gymnasium 環境）
   - テクニカル指標・時系列特徴量計算ライブラリ
   - README が充実しているもの 4 件、短い/最小限のもの 1 件

### 7.2 アノテーション仕様
- 人手（専門家レビュー）により、各ドキュメントから以下をアノテーションする:
  - 正解 Model 名およびその根拠文
  - 正解 Method 名およびその根拠文
  - 正解 Capability（Operation, Inputs, Outputs）およびその根拠文
  - 正解 Assumption / Limitation
- アノテーション結果は `tests/fixtures/gold_standard/h1a_ground_truth.json` に保存する。

---

## 8. ベースラインと評価指標

抽出器の優位性を測定するため、ベースライン手法と比較する。

### 8.1 比較対象（ベースライン）
1. **Baseline-A (Rule-based Regex / Spacy NER)**:
   - 既知のモデル名・手法名辞書および正規表現による単純キーワード抽出。入出力型は固定ルールで推定。
2. **Baseline-B (Direct Prompting without Strict Schema)**:
   - スキーマ制約や Evidence 引用を強制しない、LLM に対する自由文要約プロンプト。
3. **Proposed (H1a Grounded Structured Extractor)**:
   - 本仕様に基づく、Evidence 引用・Controlled Vocabulary・スキーマ強制を備えた抽出器。

### 8.2 評価指標（Evaluation Metrics）

| 指標名 | 定義・計算方法 | 目標基準（受入基準） |
|---|---|---|
| **Grounding Precision** | 抽出された全エンティティのうち、`EvidenceSpan` が原文に実在し、かつ主張と合致している割合。 | **95.0% 以上** |
| **Hallucination Rate** | 原文に根拠のない推測、捏造された入出力型、存在しないモデル名が含まれる割合。 | **2.0% 以下** |
| **Entity Precision** | ゴールドスタンダードと照合した Model/Method/Capability の適合率。 | **85.0% 以上** |
| **Entity Recall** | ゴールドスタンダードに存在する主要 Model/Method/Capability の再現率。 | **80.0% 以上** |
| **Controlled Vocabulary Coverage** | 抽出された Capability の入出力型が、Controlled Vocabulary に正しく適合・分類された割合。 | **90.0% 以上** |
| **Schema Validation Rate** | 初回出力で Pydantic バリデーションおよび Evidence 完全一致をパスした割合。 | **98.0% 以上** |

---

## 9. 診断的分析（過剰推論・根拠なし抽出・前提欠落の診断）

定量的指標に加え、以下の 3 大エラーパターンに対する診断パイプラインを実装する。

1. **過剰推論（Over-generalization / Extrapolation）の診断**:
   - 原文が「toy market simulation」と記述しているにもかかわらず、「general financial forecasting」と誇大抽出していないかを診断。
   - 診断手法: 抽出された Capability の説明文と引用文の意味的包含関係（NLI: Natural Language Inference モデルまたは批判的プロンプト）による矛盾・飛躍検知。
2. **根拠なし抽出（Ungrounded Claims）の診断**:
   - `start_char:end_char` のスライステストを自動実行。
   - 引用文の中に「suggests」「might」「hypothesize」などのヘッジ（不確実表現）が含まれている場合、確定的な Capability として抽出していないかをフラグ付け。
3. **前提条件欠落（Missing Preconditions）の診断**:
   - モデルが「定常性を仮定」している文脈において、Assumption が空になっていないかをチェック。
   - 典型的な前提キーワード（`assume`, `under the condition of`, `assuming`, `given that`）が原文に含まれている場合の Assumption 抽出率をモニタリング。

---

## 10. H1b（Entity Resolution）への引き渡し契約

抽出されたエンティティは、同一概念の表記揺れ（例: "NPE" と "Neural Posterior Estimation"、"ABM" と "Agent-based model"）の統合を担当する **H1b（Entity Resolution）** へ引き渡される。

### 10.1 H1b 引き渡しデータ構造
H1b は `ExtractedKnowledgeRecord` のリストを入力とし、各エンティティに対して正規化 ID（`canonical_entity_id`）と関係エッジ（`SAME_AS`, `IS_A`, `PROVIDES`）を付与する。

```python
class H1bInputBatch(BaseModel):
    records: List[ExtractedKnowledgeRecord]
    batch_id: str
    created_at: str
```

H1a はエンティティ間の同一性判定（名寄せ）は行わず、あくまで「1つのドキュメントから忠実に抽出すること」に専念する（関心の分離）。

---

## 11. 実装ステップと受入基準

計画承認後の実装は以下のステップに従って進める。

### 11.1 実装ステップ

```text
Step 1: データセット構築 & スキーマ定義
        - tests/fixtures/gold_standard/h1a_ground_truth.json (15件) の作成
        - src/research_pipeline/extractors/models.py (スキーマ・バリデータ)
        - Controlled Vocabulary 定義 (vocab.py)

Step 2: Evidence 検証器 & プロンプト設計
        - src/research_pipeline/extractors/validator.py (完全一致引用チェッカー)
        - src/research_pipeline/extractors/prompts/ (抽出プロンプトテンプレート v1.0)

Step 3: LLM 抽出器クライアントの実装
        - src/research_pipeline/extractors/llm_extractor.py (JSON Schema強制, リトライ)
        - tests/unit/test_validator.py, test_llm_extractor.py (モックテスト)

Step 4: 診断パイプラインの実装
        - src/research_pipeline/extractors/diagnostics.py (過剰推論・根拠なし検知)

Step 5: 抽出サービス & H1b 契約インターフェースの実装
        - src/research_pipeline/service/extraction_service.py (H0 ResearchDocument 受領 → 抽出 → DB保存)
        - SQLite テーブル追加 (extracted_knowledge テーブル)

Step 6: ゴールドスタンダード評価実験の実行 & 受入検査
        - 15件のデータセットに対するベースライン比較実験
        - メトリクス（Precision, Recall, Grounding Rate）の自動集計
        - 受入基準判定
```

### 11.2 H1a 受入基準
1. **Grounding Rate 100%**: 抽出された全エンティティの `verbatim_quote` が原文から 100% 機械的に照合可能であること（0 件の捏造引用）。
2. **ゴールドスタンダード検証**: 15 件の評価セットにおいて、Precision 85% 以上、Recall 80% 以上を達成すること。
3. **Controlled Vocabulary 適合率**: Capability の入出力型の 90% 以上が語彙集にマッピングされること。
4. **AlgoForge 無侵害**: AlgoForge 側のコード・DB に一切変更がないこと。

---
**Plan Prepared by:** Antigravity Agent  
**Awaiting Review:** H1a 実験計画草案 v0.1 の承認判定待ち
