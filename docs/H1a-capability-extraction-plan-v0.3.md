# HDE H1a — Capability Extraction 科学的実験計画書 v0.3

**Status:** Proposed (Scientific Review Pending)  
**Date:** 2026-11-15  
**Version:** 0.3  
**Target:** Hypothesis Discovery Engine (HDE) Phase 1 / Step H1a  
**Parent Specifications:**
- [architecture.md](file:///c:/Users/ksato/workspace/finance/research_pipeline/docs/architecture.md)
- [hypothesis_discovery_engine.md](file:///c:/Users/ksato/workspace/finance/research_pipeline/docs/hypothesis_discovery_engine.md)
- [H0-completion-report.md](file:///c:/Users/ksato/workspace/finance/research_pipeline/docs/H0-completion-report.md)
- [H1a-capability-extraction-plan-v0.2.md](file:///c:/Users/ksato/workspace/finance/research_pipeline/docs/H1a-capability-extraction-plan-v0.2.md)

---

## 1. 目的と科学的課題

### 1.1 目的
Step H1a の目的は、H0 で収集・保存された `ResearchDocument`（論文 Abstract および OSS README、メタデータ）を入力とし、そこから **Phenomenon（対象現象）**、**Model（モデル）**、**Method（手法）**、**Capability（操作可能な能力）**、**Assumption（前提仮定）**、**Limitation（適用制約）**、およびそれらの **文書内局所関係（Local Relations）** を高精度かつ再現可能な形で抽出し、原典版に厳密に紐付いた構造化知識レコード（`ExtractedKnowledgeRecord`）を生成することである。

### 1.2 科学的課題と計画 v0.3 の改訂方針
先行する計画 v0.2 の科学的レビューにおいて指摘された評価設計上の課題を解決するため、本計画 v0.3 では以下の改訂を行う：
1. **実装受入と科学的一般化性能の分離**: 12件のHoldoutデータセットを「初期パイロット評価（実装受入ゲート）」と明確に位置づけ、小標本（$N=12$）における統計的限界（95% Wilson信頼区間）を明記。「1%以下の幻覚率を実証した」等の母集団に対する過大な統計的主張を行わない。
2. **意味的同一性に基づく一対一マッチング（主指標）**: 文字列一致や Embedding コサイン類似度は補助指標とし、専門家が定義した意味的同値関係に基づく二部グラフラベルマッチング（一対一対応）を主評価指標とする。
3. **抽出3態（Raw / Accepted / Rejected）の完全保存と検証器効果の分離**: 検証器が棄却した項目を理由とともに保存し、検証器による Precision / Recall / Coverage のトレードオフを定量化する。Claim Support の最終正解基準は独立した人手評価とする。
4. **CapabilityIO および LocalRelation の Evidence 義務付け**: 明示された入出力属性および局所関係には原文引用（EvidenceSpan）を必須とし、未記載属性には根拠捏造を禁止して `is_unspecified=True` を義務付ける。Pydantic バリデータで状態整合性を厳格に保証する。
5. **C1〜C4共通評価形式への正規化パイプライン**: 構文/変換失敗、プロンプト引用生成失敗、検証器棄却の各段階を明確に分離して追跡・記録する。

---

## 2. 抽出対象の定義と厳密スキーマ

### 2.1 エンティティの厳密な概念定義

| エンティティ種別 | 厳密な定義 | 具体例（正例） | 混同しやすい非該当例（負例） |
|---|---|---|---|
| **Phenomenon** | 説明・予測・再現の対象となる市場現象、統計的性質、または実世界挙動。 | 「ボラティリティ・クラスタリング」「ファットテール」「注文板の非対称性」 | 「ニューラルネットワーク」（Model）、「パラメータ推定」（Method） |
| **Model** | 現象を数理的・論理的に説明・模倣・生成するための構造的枠組み、アーキテクチャ、またはシミュレータ。 | 「Minority Game」「Hawkes過程」「Transformer-XL」 | 「最尤推定法」（Method）、「価格系列生成」（Capability） |
| **Method** | モデルのパラメータ推定、表現学習、状態識別、最適化、仮説検定を実行する具体的なアルゴリズムまたは計算手順。 | 「Neural Posterior Estimation (NPE)」「Augmented Dickey-Fuller検定」「AdamW」 | 「エージェントベースモデル」（Model）、「異常検出」（Capability） |
| **Capability** | モデルや手法によって実行可能な「操作的機能・入出力変換能力」。入出力の接続点となる。 | 「パラメータから価格系列を生成する」「板情報から潜在レジームを分類する」 | 単なる手法名「NPE」、単なるデータ名「Tickデータ」 |
| **Assumption** | モデルや手法が成立・収束・正当化されるために前提とされている仮定・環境条件。 | 「市場参加者は有限の記憶容量を持つ」「価格変動は弱定常である」「無裁定条件」 | 「高精度である」（主張）、「計算が遅い」（Limitation） |
| **Limitation** | 著者が明示または理論上認められている適用限界、計算コスト問題、未解決課題。 | 「極端なショック時には推定が不安定」「多変量拡張時の次元の呪い」「遅延大」 | 前提条件（Assumption） |

### 2.2 Pydantic スキーマ設計

```python
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Literal, Dict, Any

class EvidenceSpan(BaseModel):
    """原文からの完全一致引用と位置情報（固定文字数制限は撤廃）"""
    verbatim_quote: str = Field(..., min_length=1, description="原文からの一致抜粋文字列 (完全一致)")
    start_char: int = Field(..., ge=0, description="抽出元テキスト内の開始文字インデックス")
    end_char: int = Field(..., ge=0, description="抽出元テキスト内の終了文字インデックス")
    field_context: Literal["abstract", "readme", "title", "metadata"] = Field(..., description="引用元フィールド")

    @model_validator(mode="after")
    def validate_indices(self) -> "EvidenceSpan":
        if self.end_char <= self.start_char:
            raise ValueError(f"end_char ({self.end_char}) must be strictly greater than start_char ({self.start_char})")
        return self

class PhenomenonEntity(BaseModel):
    """対象現象エンティティ"""
    local_id: str = Field(..., description="文書内局所ID (例: 'phe_01')")
    name: str = Field(..., description="現象の標準的名称")
    description: str = Field(..., description="現象の内容説明")
    evidence: EvidenceSpan

class ModelEntity(BaseModel):
    """モデルエンティティ"""
    local_id: str = Field(..., description="文書内局所ID (例: 'mod_01')")
    name: str = Field(..., description="モデル名称")
    category: Literal["agent_based", "statistical_physics", "state_space", "deep_learning", "stochastic_process", "other"]
    description: str = Field(..., description="数理的・構造的説明")
    evidence: EvidenceSpan

class MethodEntity(BaseModel):
    """手法エンティティ"""
    local_id: str = Field(..., description="文書内局所ID (例: 'met_01')")
    name: str = Field(..., description="手法名称")
    category: Literal["parameter_inference", "anomaly_detection", "state_estimation", "representation_learning", "optimization", "hypothesis_testing", "other"]
    description: str = Field(..., description="計算・推定手順の説明")
    evidence: EvidenceSpan

class CapabilityIO(BaseModel):
    """Capability の入出力データ仕様（状態整合性を厳格検証）"""
    data_type_l1: Literal[
        "timeseries_1d", "timeseries_multivariate", "tabular_record",
        "graph_network", "discrete_sequence", "probability_distribution",
        "parameter_vector", "scalar_value", "event_stream", "unknown"
    ] = Field(..., description="L1 汎用データ型")
    data_type_l2: Optional[str] = Field(None, description="L2 ドメイン固有型 (例: 'price_timeseries')")
    description: Optional[str] = Field(None, description="データの具体的意味")
    time_resolution: Optional[Literal["tick", "sub_minute", "minute", "hour", "daily", "static", "unspecified"]] = Field("unspecified")
    dimensionality: Optional[str] = Field(None, description="次元数 (例: '1D', '5D', 'unknown')")
    is_unspecified: bool = Field(False, description="原文に入出力仕様の記載がない場合は True")
    evidence: Optional[EvidenceSpan] = Field(None, description="明示属性の原文根拠 (is_unspecified=True時はNone必須)")

    @model_validator(mode="after")
    def validate_io_consistency(self) -> "CapabilityIO":
        if self.is_unspecified:
            if self.data_type_l1 != "unknown":
                raise ValueError("is_unspecified is True, so data_type_l1 must be 'unknown'")
            if self.data_type_l2 is not None:
                raise ValueError("is_unspecified is True, so data_type_l2 must be None")
            if self.evidence is not None:
                raise ValueError("is_unspecified is True, so evidence must not be fabricated (must be None)")
        else:
            if self.data_type_l1 == "unknown":
                raise ValueError("is_unspecified is False, so data_type_l1 cannot be 'unknown'")
            if self.evidence is None:
                raise ValueError("is_unspecified is False, so explicit IO attribute must have evidence")
        return self

class AssumptionItem(BaseModel):
    """前提仮定アイテム"""
    local_id: str = Field(..., description="文書内局所ID (例: 'asm_01')")
    statement: str = Field(..., description="仮定の内容")
    is_explicit: bool = Field(..., description="原文に明示されているか")
    evidence: EvidenceSpan

class LimitationItem(BaseModel):
    """制約・限界アイテム"""
    local_id: str = Field(..., description="文書内局所ID (例: 'lim_01')")
    statement: str = Field(..., description="制約の内容")
    evidence: EvidenceSpan

class CapabilityEntity(BaseModel):
    """操作的機能（Capability）エンティティ"""
    local_id: str = Field(..., description="文書内局所ID (例: 'cap_01')")
    name: str = Field(..., description="能力名称 (例: 'Parameterized Order Flow Simulation')")
    operation: Literal["generate", "infer", "detect", "predict", "transform", "optimize"] = Field(...)
    inputs: List[CapabilityIO] = Field(..., min_length=1, description="入力仕様")
    outputs: List[CapabilityIO] = Field(..., min_length=1, description="出力仕様")
    preconditions: List[str] = Field(default_factory=list, description="実行に必要な前提条件")
    limitations: List[str] = Field(default_factory=list, description="既知の適用限界")
    evidence: EvidenceSpan

class LocalRelation(BaseModel):
    """文書内のエンティティ間局所関係（Evidence 必須化）"""
    relation_id: str = Field(..., description="関係ID (例: 'rel_01')")
    src_local_id: str = Field(..., description="関係元局所ID")
    dst_local_id: str = Field(..., description="関係先局所ID")
    relation_type: Literal[
        "EXPLAINS",       # Model/Method -> Phenomenon
        "APPLIES_TO",     # Method -> Model
        "PROVIDED_BY",    # Capability -> Model/Method
        "REQUIRES",       # Capability/Model -> Assumption
        "CONSTRAINED_BY"  # Capability/Model/Method -> Limitation
    ]
    evidence: EvidenceSpan = Field(..., description="関係性を示す原文引用 (必須)")

class GenerationMetadata(BaseModel):
    """推論再現性のための完全メタデータ（機密情報は除外・マスク済み）"""
    model_name: str
    model_checkpoint: str
    prompt_template_name: str
    prompt_template_version: str
    prompt_hash: str
    input_text_hash: str
    temperature: float
    seed: Optional[int]
    generation_params: Dict[str, Any]
    raw_response_text: str
    sanitized: bool = Field(True, description="APIキー等の機密情報がマスクされているフラグ")
    executed_at: str

class ExtractedKnowledgeRecord(BaseModel):
    """H1a の構造化知識レコード（共通正規化形式）"""
    source_id: str = Field(..., description="論理資料ID (H0)")
    version_id: str = Field(..., description="資料版ID (H0)")
    version_hash: str = Field(..., description="コンテンツSHA-256")
    content_scope: Literal["abstract_only", "readme_only", "metadata_only", "full_text"]
    evidence_status: Literal["reported", "unverified"] = Field("reported")
    
    phenomena: List[PhenomenonEntity] = Field(default_factory=list)
    models: List[ModelEntity] = Field(default_factory=list)
    methods: List[MethodEntity] = Field(default_factory=list)
    capabilities: List[CapabilityEntity] = Field(default_factory=list)
    assumptions: List[AssumptionItem] = Field(default_factory=list)
    limitations: List[LimitationItem] = Field(default_factory=list)
    local_relations: List[LocalRelation] = Field(default_factory=list)
    
    generation_metadata: GenerationMetadata
```

### 2.3 機密情報のログおよび保存データからの完全除外
- **サニタイズ規則**:
  - LLM APIキー（`GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`）、GitHub トークン（`GITHUB_TOKEN`）、連絡先メールアドレス（`HDE_CONTACT_EMAIL`）、およびローカルユーザー名パス（`C:\Users\<user>\...`）は、ログ出力および保存 JSON（`raw_response_text`, `generation_params` 等）から自動マスクされる。
  - パスマスク: `c:\Users\[A-Za-z0-9_-]+\workspace` $\to$ `[WORKSPACE]`
  - トークンマスク: `(AIzaSy[A-Za-z0-9_-]{33}|sk-[A-Za-z0-9_-]{32,}|ghp_[A-Za-z0-9]{36})` $\to$ `[REDACTED_API_KEY]`

---

## 3. Evidence 検証の 3 層分離と抽出結果 3 態の保存

### 3.1 3層検証パイプライン（固定文字数制限撤廃）
原文引用の存在をもって主張の正しさを保証したと扱わないため、以下の独立した 3 段階で検証する。従来の「10文字〜300文字」といった固定文字数制限は撤廃し、意味的・タイポグラフィ的完全一致で判定する。

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. Quote Validity (機械的・構文的検証)                      │
│    - 完全一致: input_text[start:end] == verbatim_quote      │
│    - 空白・改行の正規化比較（非ゼロ文字長）                 │
└──────────────────────────────┬──────────────────────────────┘
                               │ PASS
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Claim Support (意味的・論理的支持検証)                   │
│    - 引用文が主張（Entity/Relation）を論理的に支持しているか │
│    - NLI/LLM検証器によるスクリーニング                      │
│    - 【重要】最終正解判定は独立した人手評価を基準とする     │
└──────────────────────────────┬──────────────────────────────┘
                               │ ENTAILED
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Claim Completeness (文脈完全性・歪曲検証)                │
│    - 否定語 (not, failed)、前提条件、ヘッジ (might, suggest)│
│      が切り落とされ、断定的な主張に歪曲されていないか       │
└─────────────────────────────────────────────────────────────┘
```

1. **Quote Validity（引用の正確性）**:
   - `input_text[start_char:end_char] == verbatim_quote` を決定論的に検証。不一致は即時リジェクト。
2. **Claim Support（主張の論理的支持）**:
   - 引用文が抽出された主張を直接支持（含意: Entailment）しているかを検証。自動スクリーニングにはNLIを用いるが、**評価実験における最終正解基準（Ground Truth）は独立した人手アノテーションによる判定とする**。
3. **Claim Completeness（文脈完全性・非過剰主張）**:
   - 引用周辺の文脈で「〜条件下でのみ成立」「〜には失敗した」「〜と推測されるにとどまる」といった否定・制約表現（Hedges）が存在する場合に、それを無視して無条件の Capability として抽出したものを **過剰主張（Overclaim）** として棄却。

### 3.2 抽出結果 3 態（Raw / Accepted / Rejected）の完全保存
検証器の棄却効果を透明化するため、推論結果を以下の 3 つのカテゴリに分類・保存する。

```text
┌───────────────────────────────────────────────────────────┐
│ Raw Extraction (LLMの生抽出レコード)                      │
└─────────────────────────────┬─────────────────────────────┘
                              │ Validator Execution
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐
│ Accepted Extraction       │   │ Rejected Extraction       │
│ - 全検証 (QV, CS, CC) 合格│   │ - 棄却理由・不整合内容記録│
│ - H1b 引き渡し候補        │   │ - 偽陽性・過剰抽出分析用  │
└───────────────────────────┘   └───────────────────────────┘
```

```python
class RejectionRecord(BaseModel):
    """検証器により棄却された抽出項目と理由"""
    local_id: str
    entity_type: str
    rejection_stage: Literal["quote_validity", "claim_support", "claim_completeness", "io_consistency", "schema_violation"]
    rejection_reason: str
    raw_entity_data: Dict[str, Any]

class ExtractionTrioBundle(BaseModel):
    """資料ごとの 3 態完全保存バンドル"""
    source_id: str
    version_id: str
    raw_record: ExtractedKnowledgeRecord
    accepted_record: ExtractedKnowledgeRecord
    rejected_items: List[RejectionRecord]
```

### 3.3 検証器の棄却によるトレードオフ定量化
検証器が適合率（Precision）向上に寄与する一方で、再現率（Recall）や文書カバレッジ（Coverage）をどの程度損なうかを以下の指標で明示的に測定・報告する。

$$\Delta P = \text{Precision}_{\text{Accepted}} - \text{Precision}_{\text{Raw}}$$
$$\Delta R = \text{Recall}_{\text{Accepted}} - \text{Recall}_{\text{Raw}}$$
$$\Delta \text{Coverage} = \frac{\text{Accepted Capability が 1 件以上残存した文書数}}{N_{\text{total}}} - \frac{\text{Raw Capability が 1 件以上抽出された文書数}}{N_{\text{total}}}$$

---

## 4. Controlled Vocabulary の二層構造

### 4.1 L1 (Universal Types) と L2 (Domain Specializations)
他分野（物理、生物、複雑系、情報理論等）の研究成果を汎用的に扱い、かつ金融応用に特化できるよう二層化する。

| L1 汎用型 (`data_type_l1`) | 意味・数理表現 | 対応する L2 金融ドメイン型 (`data_type_l2`) | 対応する L2 他分野ドメイン型（例） |
|---|---|---|---|
| `timeseries_1d` | 単一変数の時系列 $x_t \in \mathbb{R}$ | `price_timeseries`, `return_series` | 心拍系列、地震波形系列、流速系列 |
| `timeseries_multivariate` | 多次元時系列 $\mathbf{x}_t \in \mathbb{R}^D$ | `ohlcv_bars`, `macroeconomic_indicators` | 多点気象時系列、脳波多チャネル系列 |
| `tabular_record` | 異種属性のテーブル・レコード | `financial_report_ratios`, `analyst_forecast` | 臨床データ、国勢調査レコード |
| `graph_network` | ノード・エッジの幾何構造 $(V, E)$ | `supply_chain_network`, `cross_currency_graph` | ソーシャルグラフ、分子構造グラフ |
| `discrete_sequence` | 離散シンボルの系列 $s_t \in \Sigma$ | `order_event_stream`, `news_token_stream` | DNA配列、タンパク質アミノ酸配列 |
| `probability_distribution` | パラメトリック/非パラメトリック確率分布 | `predictive_return_distribution`, `posterior_params`| 状態遷移確率、事後分布 $p(\theta \mid x)$ |
| `parameter_vector` | 状態を持たない静的パラメータベクトル | `agent_behavior_parameters`, `model_weights` | 物理定数、シミュレータ構成パラメータ |
| `scalar_value` | 単一の数値指標・スコア | `sharpe_ratio`, `volatility_estimate`, `anomaly_score`| 臨界点パラメータ、Lyapunov指数 |
| `event_stream` | タイムスタンプ付きの不定期イベント | `tick_trades`, `earnings_announcement` | パルス信号、障害ログイベント |
| `unknown` | 原文からデータ型が特定不能 | `None` (未指定必須) | `None` (未指定必須) |

### 4.2 未知属性の根拠捏造防止ルール
- **禁止事項**: 原文に入出力データ型の明記がない場合、LLM が一般的知識に基づいて「価格データ」「リターン系列」等と補完・推測することを禁止する。
- **義務付け**: 未記載時は必ず `is_unspecified=True`, `data_type_l1="unknown"`, `data_type_l2=None`, `evidence=None` を設定する。Pydantic バリデータにより、これらに違反する出力は即座にスキーマ不整合としてリジェクトされる。

---

## 5. 評価データセット設計と隔離・汚染防止

### 5.1 3分割構成と役割の明確化
評価データセット（全 30 件）を以下の 3 群に分離する。特に Holdout 12 件は「初期パイロット評価（実装受入ゲート）」とし、母集団に対する科学的一般化の統計的実証とは区別する。

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                  H1a Gold Standard Dataset (全 30 件)                       │
├──────────────────────────────┬──────────────────────────────┬───────────────┤
│ 1. Development Set (10件)    │ 2. Holdout Pilot Set (12件)  │ 3. Challenge  │
│    - プロンプト改善          │    - 実装受入ゲート検証      │    Set (8件)  │
│    - エラー分析              │    - 調整中はアクセス禁止    │    - 罠・難関 │
│    - 変換器チューニング      │    - パイロット小標本CI報告  │    - 欠落・否定│
│    - 内部反復評価            │    - 母集団への一般化主張不可│      ヘッジ文 │
└──────────────────────────────┴──────────────────────────────┴───────────────┘
```

1. **Development Set（10 件: 開発・調整専用）**:
   - 金融市場論文 4 件、他分野・ML 論文 3 件、OSS README 3 件。プロンプト調整、正規化器の改善、エラー分析に利用。
2. **Holdout Pilot Set（12 件: 実装受入ゲート検証・完全隔離）**:
   - 金融市場論文 4 件、他分野論文 4 件、OSS README 4 件。
   - **運用制約**: プロンプト作成・調整・モデル選定フェーズでは一切使用せず、実装完了時に 1 回のみブラインド実行。
   - **位置づけ**: パイロット検証であり、統計的受入判定には Wilson スコア区間を併記する。
3. **Challenge Set（8 件: ストレステスト専用）**:
   - README 欠落（1件）、仮説棄却・否定文多用論文（2件）、ヘッジ・不確実表現論文（2件）、入出力未記載の抽象理論論文（2件）、過剰主張論文（1件）。

### 5.2 資料重複・フォーク混入防止ルール（Cross-Split Isolation）
データセット分割間での情報漏洩（Contamination）を完全に防止するため、以下の独立性検証スクリプトを適用する：
1. **arXiv**:
   - 同一の arXiv Base ID（版番号を除いた識別子）の重複を禁止。
   - 同一筆頭著者による同一テーマの派生論文の混入を禁止。
2. **GitHub**:
   - 同一の `repository_id` の重複を禁止。
   - Fork リポジトリの混入を禁止（GitHub API により `fork: true` または `parent.id == candidate.id` であるペアは同一 split または除外）。
   - 同一の README コンテンツハッシュ（`extracted_text_hash`）を持つ資料の重複を禁止。

### 5.3 アノテーション品質と Cohen's Kappa の評価単位・負例集合の定義
- **評価単位の厳密定義**:
  - 情報抽出における Cohen's Kappa の不定性を防ぐため、評価単位を **「文単位（Sentence-level）の概念分類」** と定義する。
  - 対象文書を文単位（$S_1, S_2, \dots, S_M$）に分割。
  - 各文に対してアノテーターが以下の排他ラベルを付与：
    $$\mathcal{Y} = \{\text{Phenomenon}, \text{Model}, \text{Method}, \text{Capability}, \text{Assumption}, \text{Limitation}, \text{NONE}\}$$
  - **負例集合（Negative Class: `NONE`）**: いかなるターゲット概念にも該当しない一般的な背景文、謝辞、導入文、参考文献言及等。これにより明確な $7 \times 7$ 分割表が定義され、負例過多や不定性を回避した厳密な Cohen's Kappa および Fleiss' Kappa が算出可能となる。
- **目標一致度**: 2名のアノテーター間で $\kappa \ge 0.75$（Substantial Agreement）を達成し、不一致文は Adjudicator により裁定。

---

## 6. 段階的アブレーション比較実験 (C0〜C4) と共通正規化

### 6.1 実験条件（C0〜C4）の構成
抽出器の各構成要素の効果を単離するため、以下の 5 条件で比較を行う。

| 条件ID | 手法名 | 構成要素・プロンプト | 出力形式 |
|---|---|---|---|
| **C0** | Rule-based Baseline | 正規表現・キーワード辞書マッピング | ヒューリスティック辞書 |
| **C1** | Free-text LLM | 自由形式テキスト抽出プロンプト | 非構造化プレーンテキスト |
| **C2** | Structured Output Only | Pydantic JSON Schema 強制 (Evidenceなし) | JSON (スキーマ準拠) |
| **C3** | C2 + Evidence Grounding | Pydantic Schema + Verbatim Quote 引用強制 | JSON (EvidenceSpan 含む) |
| **C4 (Proposed)** | C3 + Controlled Vocab | Pydantic + Evidence + L1/L2 語彙 + unknown 許容 | JSON (完全スキーマ準拠) |

### 6.2 共通評価形式への正規化変換パイプライン（Standardization Pipeline）
全条件の出力を同一の評価関数で公正に比較するため、固定の正規化コンバータを介して共通の `ExtractedKnowledgeRecord` に変換する。

```text
各条件の出力 (C0〜C4)
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ Deterministic Normalization Converter                       │
│ - C0: 抽出辞書から Entity を生成 (evidence は空 placeholder)│
│ - C1: 正規表現・セクションパーサーで JSON 構造へマッピング   │
│ - C2: スキーマ JSON をロード (evidence は空 placeholder)    │
│ - C3/C4: 直接 Pydantic レコードへデシリアライズ             │
└──────────────────────────────┬──────────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
       [変換成功: Record]              [変換失敗: Logged]
       - 評価指標の分母・分子へ        - conversion_failure_rate 記録
```

### 6.3 失敗モードの完全分離
引用要求や検証器による棄却効果と、構文上の変換失敗を混同しないよう、以下の 3 つの失敗率を分離して計測する：
1. **構文・変換失敗率 ($R_{\text{conv\_fail}}$)**:
   $$R_{\text{conv\_fail}} = \frac{\text{正規化コンバータでパース・変換に失敗した文書数}}{N_{\text{total}}}$$
2. **引用生成失敗率 ($R_{\text{quote\_gen\_fail}}$)**:
   $$R_{\text{quote\_gen\_fail}} = \frac{\text{LLMが出力したが quote 文字列が空または欠落していたエンティティ数}}{N_{\text{raw\_entities}}}$$
3. **検証器棄却率 ($R_{\text{val\_reject}}$)**:
   $$R_{\text{val\_reject}} = \frac{\text{構文正常だが検証器 (QV, CS, CC, IO) によりリジェクトされた数}}{N_{\text{parsed\_entities}}}$$

---

## 7. 定量的評価指標・照合規則・統計的推定

### 7.1 エンティティ照合規則（Entity Matching）
正解エンティティ集合 $G$ と抽出エンティティ集合 $P$ の照合において、**人手定義の意味的同一性に基づく一対一マッチング（主指標）** を採用する。

1. **主指標: Human-defined Semantic Equivalence with 1-to-1 Bipartite Matching**:
   - ゴールドスタンダード構築時に定義された「同一概念とみなす同義語・表現規則」に基づき、重み付き二部グラフ $B = (G, P, W)$ を構築。
   - 重み $W(g, p) \in \{0, 1\}$ は専門家人手評価による意味的等価性（Semantic Equivalence）。
   - **Kuhn-Munkres (Hungarian) アルゴリズム** により最大重み一対一マッチングを解き、マッチしたペア数を $|G \cap P|_{\text{semantic}}$ とする。1つの正解に複数の抽出エンティティが多重マッチすることは許却されない。
2. **補助指標（Auxiliary Metrics）**:
   - **Strict String Match**: 名称の完全一致（小文字正規化）かつ引用 Jaccard $\ge 0.5$。
   - **Embedding Similarity Match**: テキスト埋め込み（`text-embedding-3-small` 等）のコサイン類似度 $\ge 0.85$ によるマッチング。

### 7.2 指標定義

1. **Semantic Entity Precision ($P$) / Recall ($R$) / F1 (主指標)**:
   $$P_{\text{sem}} = \frac{|G \cap P|_{\text{semantic}}}{|P|}, \quad R_{\text{sem}} = \frac{|G \cap P|_{\text{semantic}}}{|G|}, \quad F1_{\text{sem}} = \frac{2 P_{\text{sem}} R_{\text{sem}}}{P_{\text{sem}} + R_{\text{sem}}}$$
2. **Quote Validity Rate ($R_{QV}$)**:
   $$R_{QV} = \frac{\sum_{p \in P} \mathbb{I}(\text{input\_text}[\text{start}:\text{end}] == \text{verbatim\_quote})}{|P|}$$
3. **Claim Support Rate ($R_{CS}$)**:
   $$R_{CS} = \frac{\sum_{p \in P} \mathbb{I}(\text{HumanJudgeSupport}(\text{evidence}, p) == \text{Entailed})}{|P|}$$
   ※ 最終評価は独立した専門家人手判定に基づく。
4. **Overclaim Rate ($R_{OC}$)**:
   $$R_{OC} = \frac{\text{文脈上の制約・否定・不確実性を無視して抽出された過剰主張エンティティ数}}{|P|}$$
5. **Observed Hallucination Rate ($R_{HAL}$)**:
   $$R_{HAL} = \frac{\text{原文に根拠が一切存在せず捏造されたエンティティ数}}{|P|}$$
6. **Vocabulary Compliance Rate ($R_{VC}$)**:
   $$R_{VC} = \frac{\text{L1語彙準拠かつ整合性バリデータを通過したIO数}}{\text{全抽出IO数}}$$

### 7.3 文書単位クラスターブートストラップ（Cluster Bootstrap 95% CI）
同一文書内の複数エンティティは互いに独立でない（相関を持つ）ため、エンティティ単位のナイーブなブートストラップは信頼区間を過小評価する。本計画では **文書単位のクラスターブートストラップ（Document-level Cluster Bootstrap）** を使用する。

- **手順**:
  1. データセット内の $N$ 件の文書インデックス $\{d_1, \dots, d_N\}$ から、重複を許して $N$ 件をリサンプリング（復元抽出）。これを $B = 1000$ 回反復。
  2. サンプリングされた文書群に含まれる全エンティティを集計し、各指標（$F1_{\text{sem}}, R_{QV}, R_{CS}, R_{OC}$ 等）を再計算。
  3. ブートストラップ分布の 2.5 パーセンタイルと 97.5 パーセンタイルを **95% クラスター信頼区間** として報告。

### 7.4 小標本（$N=12$）における統計的限界の明記
- **標本数と信頼区間**: Holdout Pilot Set は 12 件（抽出エンティティ総数は推定 50〜80 件）の小標本である。
- **Wilson Score 区間の適用**: 比率 $p = k/n$ に対し、小標本でも正確な Wilson Score 区間（または Clopper-Pearson 厳密区間）を算出する。
- **統計的主張の制限**:
  - 例えばパイロット標本 60 個のエンティティで幻覚が 0 件観察された場合（$k=0, n=60$）、95% 信頼区間の上限は約 $3/60 = 5.0\%$（Rule of Three）となる。
  - したがって、**「パイロット評価において幻覚が 0 件観察された」事実のみを報告し、「母集団に対する幻覚率が 1.0% 以下であることを実証した」という過大な統計的主張は厳格に禁止する**。

---

## 8. 実行制御・反復評価・費用とリソース上限

### 8.1 実装利用 LLM の選定
本実験パイプラインでは、以下のモデルを選定・固定する：
- **Primary Model**: `gemini-2.5-flash`（高速性、長文脈対応、構造化出力 JSON Schema の完全サポート、経済性）。
- **Secondary / Verification Model**: `gemini-2.5-pro`（高度推論能力による Claim Support スクリーニングおよびアブレーション検証）。
- **Fallback / Alternative**: `gpt-4o-2024-08-06` または `claude-3-5-sonnet-20241022`。
- **バージョン固定**: API 呼び出し時はモデルのエイリアスではなく、ピン留めされた特定チェックポイント（日付指定バージョン）を使用し、`generation_metadata.model_checkpoint` に記録。

### 8.2 反復実験回数（$R=3$）と分散評価
LLM の非決定論性（GPU浮動小数点演算順序や並列推論による微細な揺らぎ）を測定するため、各条件（C1〜C4）について同一パラメータ（Temperature=0.0, Seed=42）で **$R = 3$ 回の独立試行** を実施する。
- 3 回の試行間での出力エンティティの Jaccard 類似度および $F1$ スコアの標準偏差を算出し、再現性の安定度を報告する。

### 8.3 トークン・費用予算上限と安全停止機構
- **総予算上限**: **30.00 USD（または最大 5,000,000 トークン）** を本 H1a 実験の上限として厳格に設定。
- **試算**:
  - 1 文書あたり平均入力 4,000 トークン、出力 2,000 トークン（計 6,000 トークン）。
  - 30 文書 $\times$ 6,000 トークン = 180,000 トークン / 試行。
  - 4 条件 $\times$ 3 反復 = 12 試行 $\times$ 180,000 $\approx$ 2,160,000 トークン（予算枠内で十分実行可能）。
- **安全停止機構（Hard Budget Stop）**:
  - 実験ランナーは API コールごとに累積トークン数および概算費用を追跡。
  - 予算の 80%（\$24.00）到達で警告ログを出力。
  - 予算の 100%（\$30.00）到達で即座に実験プロセスを安全停止（Fail-safe Termination）し、部分結果を保存して終了する。

---

## 9. H1b（Entity Resolution & Cross-Document Graph）への引き渡し契約

### 9.1 グローバル一意 ID 仕様
文書内局所 ID（`cap_01`, `mod_01`）が複数文書の統合時に衝突することを防ぐため、H1b 引き渡し時に以下の決定論的グローバル一意 ID を付与する。

```python
# グローバルエンティティID形式:
# <source_id>:<version_hash_prefix>:<entity_type>:<local_id>
global_entity_id = f"{source_id}:{version_hash[:8]}:{entity_type}:{local_id}"
# 例: "arxiv_2301_01234:a8f13b2c:capability:cap_01"

# グローバルエッジID形式:
# <src_global_id>-><relation_type>-><dst_global_id>
global_edge_id = f"{src_global_id}->{relation_type}->{dst_global_id}"
```

### 9.2 引き渡しデータ構造 (`H1bDocumentPackage`)

```python
class GlobalEntityRef(BaseModel):
    global_id: str
    source_id: str
    version_id: str
    entity_type: Literal["phenomenon", "model", "method", "capability", "assumption", "limitation"]
    name_or_statement: str
    verbatim_quote: str
    quote_start_char: int
    quote_end_char: int

class GlobalEdgeRef(BaseModel):
    global_edge_id: str
    src_global_id: str
    dst_global_id: str
    relation_type: str
    evidence_quote: str

class H1bDocumentPackage(BaseModel):
    """H1a から H1b に引き渡される 1 資料単位のパッケージ"""
    source_id: str
    version_id: str
    version_hash: str
    source_type: Literal["paper", "oss"]
    primary_name: str
    content_scope: Literal["abstract_only", "readme_only", "metadata_only", "full_text"]
    evidence_status: Literal["reported", "unverified"]
    
    # Accepted のみがグローバルID化されて引き渡される
    entities: List[GlobalEntityRef]
    local_relations: List[GlobalEdgeRef]
    
    # 監査用完全バンドル参照
    bundle: ExtractionTrioBundle
```

---

## 10. 実装ステップと受入判定基準の分離

### 10.1 実装・実験ステップ

```text
Step 1: Gold Standard データセットの構築 (30件)
        - Development Set (10件) の人手アノテーション作成
        - Challenge Set (8件) の罠・エッジケース作成
        - Holdout Pilot Set (12件) のアノテーション作成・完全隔離
        - Cross-split 重複・フォーク排除テスト (test_dataset_isolation.py)

Step 2: スキーマ・二層 Controlled Vocabulary の実装
        - src/research_pipeline/extractors/models.py (Pydantic スキーマ, IO整合性バリデータ)
        - src/research_pipeline/extractors/vocab.py (L1/L2 語彙集)
        - tests/unit/test_h1a_schemas.py で検証

Step 3: 3層 Evidence 検証器 & 3態バンドラーの実装
        - src/research_pipeline/extractors/validator.py (QV, CS, CC, IO-Consistency)
        - src/research_pipeline/extractors/bundle.py (Raw, Accepted, Rejected 追跡)
        - tests/unit/test_evidence_validator.py で検証

Step 4: LLM 抽出器・正規化コンバータ・予算トラッカーの実装
        - src/research_pipeline/extractors/llm_extractor.py (JSON Schema, トークン上限)
        - src/research_pipeline/extractors/converter.py (C0〜C4 共通正規化パイプライン)
        - tests/unit/test_llm_extractor_mock.py で検証

Step 5: 評価スクリプト & クラスターブートストラップの実装
        - src/research_pipeline/evaluation/evaluator.py (Hungarian 二部マッチング)
        - src/research_pipeline/evaluation/bootstrap.py (Document-level Cluster Bootstrap)

Step 6: Development Set によるアブレーション実験 (C0〜C4, R=3)
        - 失敗モード分離分析、プロンプト微調整、予算消費監視

Step 7: Holdout Pilot Set ブラインド評価 & 受入ゲート判定
        - 1 回限りのブラインド推論
        - 実装受入ゲート判定（合否）と小標本 95% 信頼区間付き結果報告
```

### 10.2 実装受入ゲート（パイロット検証合否）と科学的一般化判定の分離

#### A. 実装受入ゲート（パイロット合格基準）
本エンジニアリング実装が H1b への連携に耐えうるかを判断するための判定基準。Holdout Pilot Set（12件）において以下をすべて満たすこと：
1. **Quote Validity Rate ($R_{QV}$)**: **100.0%**（機械的完全一致）。
2. **Claim Support Rate ($R_{CS}$)**: 独立人手判定において **90.0% 以上**。
3. **Observed Hallucination ($R_{HAL}$)**: パイロット標本において **0 件（$0.0\%$）**。
4. **Semantic Entity F1 ($F1_{\text{sem}}$)**: 人手意味的同値マッチングにおいて **80.0% 以上**。
5. **Vocabulary Compliance Rate ($R_{VC}$)**: **95.0% 以上**（未記載時の `unknown` / `is_unspecified=True` 整合率 100%）。
6. **IO 状態整合性違反**: **0 件**（Pydantic バリデータ違反なし）。
7. **再現性 (Self-Consistency)**: 3 回反復試行での Jaccard 一致度が **90.0% 以上**。
8. **AlgoForge 無侵害**: AlgoForge のコード・DB に一切変更がないこと。

#### B. 科学的一般化性能の統計的報告要件
- パイロット結果は、必ず **「文書単位クラスターブートストラップ 95% 信頼区間」** または **「小標本 Wilson スコア区間」** を併記して報告する。
- 標本数 $N=12$ から得られた数値について、開かれた文献母集団全体に対する一般化性能を統計的に実証したと解釈・主張してはならない。本格的な一般化性能評価は、後続のスケールアップ評価フェーズ（大規模アノテーションデータ）にて実施することを明記する。

---
**Plan Prepared by:** Antigravity Agent  
**Awaiting Review:** H1a 科学的実験計画書 v0.3 の承認判定待ち
