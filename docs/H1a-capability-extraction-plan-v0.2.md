# HDE H1a — Capability Extraction 科学的実験計画書 v0.2

**Status:** Proposed (Scientific Review Pending)  
**Date:** 2026-11-15  
**Version:** 0.2  
**Target:** Hypothesis Discovery Engine (HDE) Phase 1 / Step H1a  
**Parent Specifications:**
- [architecture.md](file:///c:/Users/ksato/workspace/finance/research_pipeline/docs/architecture.md)
- [hypothesis_discovery_engine.md](file:///c:/Users/ksato/workspace/finance/research_pipeline/docs/hypothesis_discovery_engine.md)
- [H0-completion-report.md](file:///c:/Users/ksato/workspace/finance/research_pipeline/docs/H0-completion-report.md)
- [H1a-capability-extraction-plan-v0.1.md](file:///c:/Users/ksato/workspace/finance/research_pipeline/docs/H1a-capability-extraction-plan-v0.1.md)

---

## 1. 目的と科学的課題

### 1.1 目的
Step H1a の目的は、H0 で収集・保存された `ResearchDocument`（論文 Abstract および OSS README、メタデータ）を入力とし、そこから **Phenomenon（対象現象）**、**Model（モデル）**、**Method（手法）**、**Capability（操作可能な能力）**、**Assumption（前提仮定）**、**Limitation（適用制約）**、およびそれらの **文書内局所関係（Local Relations）** を高精度かつ再現可能な形で抽出し、原典版に厳密に紐付いた構造化知識（`ExtractedKnowledgeRecord`）を生成することである。

### 1.2 主要な科学的課題
1. **引用の存在と主張の支持の分離**: 原文の文字列が引用（Quote）として存在していても、その引用が抽出された主張を論理的に支持しているとは限らない。完全一致抜粋だけでなく、意味的な支持関係（Claim Support）をいかに検証するか。
2. **情報の欠落と過剰推論の防止**: Abstract や README に入出力仕様や制約が明記されていない場合、LLM は常識的な知識で空白を補完（過剰推論）しやすい。原典に未記載の属性を「推測」せず、正しく「未知（Unknown / Unspecified）」として扱えるか。
3. **異分野知識の表現互換性**: 金融市場論文だけでなく、統計物理、エージェントベースモデル（ABM）、Simulation-Based Inference（SBI）、制御理論などの他分野文献から、互換性を保ちつつ操作可能な Capability を抽出できるか。
4. **評価の客観性とデータ隔離**: プロンプト改善に用いる開発セットと、最終的な一般化性能を測る評価セットを厳格に分離し、過学習や基準変更による見かけの性能向上を排除できるか。

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
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

class EvidenceSpan(BaseModel):
    """原文からの完全一致引用と位置情報"""
    verbatim_quote: str = Field(..., description="原文からの一致抜粋文字列 (完全一致)")
    start_char: int = Field(..., description="抽出元テキスト内の開始文字インデックス")
    end_char: int = Field(..., description="抽出元テキスト内の終了文字インデックス")
    field_context: Literal["abstract", "readme", "title", "metadata"] = Field(..., description="引用元フィールド")

class PhenomenonEntity(BaseModel):
    """対象現象エンティティ"""
    phenomenon_id: str = Field(..., description="文書内一意ID (例: 'phe_01')")
    name: str = Field(..., description="現象の標準的名称")
    description: str = Field(..., description="現象の内容説明")
    evidence: EvidenceSpan

class ModelEntity(BaseModel):
    """モデルエンティティ"""
    model_id: str = Field(..., description="文書内一意ID (例: 'mod_01')")
    name: str = Field(..., description="モデル名称")
    category: Literal["agent_based", "statistical_physics", "state_space", "deep_learning", "stochastic_process", "other"]
    description: str = Field(..., description="数理的・構造的説明")
    evidence: EvidenceSpan

class MethodEntity(BaseModel):
    """手法エンティティ"""
    method_id: str = Field(..., description="文書内一意ID (例: 'met_01')")
    name: str = Field(..., description="手法名称")
    category: Literal["parameter_inference", "anomaly_detection", "state_estimation", "representation_learning", "optimization", "hypothesis_testing", "other"]
    description: str = Field(..., description="計算・推定手順の説明")
    evidence: EvidenceSpan

class CapabilityIO(BaseModel):
    """Capability の入出力データ仕様（未記載時は is_unspecified=True）"""
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

class AssumptionItem(BaseModel):
    """前提仮定アイテム"""
    assumption_id: str = Field(..., description="文書内一意ID (例: 'asm_01')")
    statement: str = Field(..., description="仮定の内容")
    is_explicit: bool = Field(..., description="原文に明示されているか")
    evidence: EvidenceSpan

class LimitationItem(BaseModel):
    """制約・限界アイテム"""
    limitation_id: str = Field(..., description="文書内一意ID (例: 'lim_01')")
    statement: str = Field(..., description="制約の内容")
    evidence: EvidenceSpan

class CapabilityEntity(BaseModel):
    """操作的機能（Capability）エンティティ"""
    capability_id: str = Field(..., description="文書内一意ID (例: 'cap_01')")
    name: str = Field(..., description="能力名称 (例: 'Parameterized Order Flow Simulation')")
    operation: Literal["generate", "infer", "detect", "predict", "transform", "optimize"] = Field(...)
    inputs: List[CapabilityIO] = Field(..., min_length=1, description="必要な入力データ仕様 (未記載時は is_unspecified=True のIOを格納)")
    outputs: List[CapabilityIO] = Field(..., min_length=1, description="生成される出力データ仕様 (未記載時は is_unspecified=True のIOを格納)")
    preconditions: List[str] = Field(default_factory=list, description="実行に必要な前提条件")
    limitations: List[str] = Field(default_factory=list, description="既知の適用限界")
    evidence: EvidenceSpan

class LocalRelation(BaseModel):
    """文書内のエンティティ間局所関係"""
    src_id: str = Field(..., description="関係元エンティティID")
    dst_id: str = Field(..., description="関係先エンティティID")
    relation_type: Literal[
        "EXPLAINS",       # Model/Method -> Phenomenon
        "APPLIES_TO",     # Method -> Model
        "PROVIDED_BY",    # Capability -> Model/Method
        "REQUIRES",       # Capability/Model -> Assumption
        "CONSTRAINED_BY"  # Capability/Model/Method -> Limitation
    ]
    evidence: Optional[EvidenceSpan] = Field(None, description="関係性を示す文脈引用")

class GenerationMetadata(BaseModel):
    """推論再現性のための完全メタデータ"""
    model_name: str
    model_version: str
    prompt_template_version: str
    prompt_hash: str
    input_text_hash: str
    temperature: float
    seed: Optional[int]
    raw_response_text: str
    executed_at: str

class ExtractedKnowledgeRecord(BaseModel):
    """H1a の完全な出力レコード"""
    source_id: str = Field(..., description="論理資料ID (H0)")
    version_id: str = Field(..., description="資料版ID (H0)")
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

---

## 3. Evidence 検証の 3 層分離

完全一致引用（Quote）の存在をもって「主張の正しさ」を検証したとみなす誤りを排除するため、Evidence の妥当性を以下の 3 つの独立した層で検証する。

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. Quote Validity (機械的・構文的検証)                      │
│    - 原文スライスとの完全一致: text[start:end] == quote     │
│    - 最小/最大長制約: 10文字 <= len(quote) <= 300文字       │
└──────────────────────────────┬──────────────────────────────┘
                               │ PASS
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Claim Support (意味的・論理的支持検証)                   │
│    - 引用文が抽出された主張を論理的に支持しているか         │
│    - NLI (含意/矛盾) または 批判的レビューによる検証        │
│    - 支持判定: ENTAILED (支持), NEUTRAL (無関係), CONTRADICT│
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
   - `input_text[start_char:end_char] == verbatim_quote` を決定論的スライス比較で 100% 検証する。不一致は機械的に即リジェクト。
2. **Claim Support（主張の論理的支持）**:
   - 抽出された主張（Statement / Description）と引用文（Quote）のペアを評価。引用文が主張を直接支持（含意: Entailment）しているかを検証。引用文に書かれていない内容を主張している場合は `unsupported_claim` としてリジェクト。
3. **Claim Completeness（文脈の完全性と非歪曲性）**:
   - 引用文周辺の文脈で「〜条件下でのみ成立」「〜には失敗した」「〜と推測されるにとどまる」といった制約・否定・不確実表現（Hedges）が存在する場合に、それを無視して無条件の Capability として抽出したものを **過剰主張（Overclaim）** として検出・除外。

---

## 4. Controlled Vocabulary の二層構造

他分野（物理、生物、複雑系、情報理論等）の研究成果を汎用的に扱い、かつ金融応用に特化できるよう、Controlled Vocabulary を二層化する。

```text
       ┌────────────────────────────────────────────────────────┐
       │ Layer 1: Universal Types (汎用データ構造 - 必須)        │
       │ (timeseries_1d, graph_network, parameter_vector, etc.) │
       └───────────────────────────┬────────────────────────────┘
                                   │ 特化 (Subtyping)
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │ Layer 2: Domain-Specific Specializations (任意・ドメイン)│
       │ (price_timeseries, orderbook_snapshot, etc.)           │
       └────────────────────────────────────────────────────────┘
```

### 4.1 L1 (Universal Types) と L2 (Domain Specializations) の定義

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
| `unknown` | 原文からデータ型が特定不能 | `unspecified` | `unspecified` |

**ルール**: 原文に入出力データ型の明記がない場合は、**LLM に推測させず `data_type_l1='unknown'`, `is_unspecified=True` を義務付ける**。

---

## 5. Gold Standard 評価データセットの分離設計

評価の客観性を保証し、プロンプトの過学習（P-hacking）を防ぐため、評価用アノテーションデータセットを **3 つの独立集合（計 30 件）** に完全分離する。

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                  H1a Gold Standard Dataset (全 30 件)                       │
├──────────────────────────────┬──────────────────────────────┬───────────────┤
│ 1. Development Set (10件)    │ 2. Holdout Set (12件)        │ 3. Challenge  │
│    - プロンプト調整          │    - 最終評価専用            │    Set (8件)  │
│    - エラー分析              │    - 調整中はアクセス禁止    │    - 罠・難関 │
│    - パラメータ選択          │    - プロンプト確定後に1回   │    - 欠落・否定│
│    - 内部評価にのみ利用      │      のみブラインド評価      │      ヘッジ文 │
└──────────────────────────────┴──────────────────────────────┴───────────────┘
```

### 5.1 データセットの内訳

1. **Development Set（10 件: 開発・チューニング専用）**:
   - 金融市場論文 4 件（ABM, Hawkes過程, 注文板ダイナミクス）
   - 他分野・ML 論文 3 件（SBI, 状態空間モデル, 異常検知）
   - OSS リポジトリ 3 件（Gym環境, 特徴量計算ライブラリ）
2. **Holdout Set（12 件: 最終評価専用・完全隔離）**:
   - 金融市場論文 4 件（暗号資産微細構造, レジームスイッチング等）
   - 他分野論文 4 件（統計物理スピングラス, 複雑系カオス, 因果推論）
   - OSS リポジトリ 4 件（シミュレータ, 統計ライブラリ）
   - **運用制約**: プロンプト作成・調整・モデル選定フェーズでは一切使用せず、評価確定直前に 1 回のみ実行（ブラインド評価）。
3. **Challenge Set（8 件: 診断・ストレステスト専用）**:
   - README 欠落リポジトリ（1 件）
   - 否定文多用・実験失敗論文（「〜の仮説は棄却された」論文 2 件）
   - ヘッジ・不確実表現多用論文（「〜かもしれない」「〜と推測される」2 件）
   - 入出力データ型が一切書かれていない抽象的理論論文（2 件）
   - 誇大なタイトルに対して本文が限定的な過剰主張論文（1 件）

### 5.2 アノテーションの品質担保（Inter-Annotator Agreement）
- **アノテーション手順**:
  1. 専門的知識を持つ 2 名のアノテーターが独立に同一文書をアノテーション。
  2. 一致度を **Cohen's Kappa（エンティティ境界・分類）** および **Token-level F1** で測定。
  3. 不一致箇所は Adjudicator（主設計者）を交えた審議会議で解決（Adjudication）。
  4. 最終正解の判定理由をアノテーションメタデータに保存。

---

## 6. 段階的アブレーション比較実験の設計

抽出器の各構成要素が性能に与える影響を単離・実証するため、以下の 5 条件による段階的比較実験を設計する。

| 実験条件 | 手法名 | 構成要素 |
|---|---|---|
| **C0 (Baseline)** | Rule-based / Regex | 既知のモデル・手法キーワード辞書と正規表現による抽出。入出力は固定ルール。 |
| **C1** | Free-text LLM | 自由形式テキストプロンプト（「この論文のModel, Method, Capabilityを要約せよ」）。 |
| **C2** | Structured Output Only | Pydantic JSON Schema 強制。Evidence 引用制約なし。入出力は自由テキスト。 |
| **C3** | C2 + Evidence Grounding | Pydantic Schema + `EvidenceSpan` 引用強制（Quote Validity + Claim Support 検証）。 |
| **C4 (Proposed)** | C3 + Controlled Vocab | Pydantic Schema + Evidence 引用 + L1/L2 二層 Controlled Vocabulary + 未知入出力の unknown 許容。 |

### 6.1 統制条件
- **モデル**: 同一 LLM（Gemini 1.5 Pro または Claude 3.5 Sonnet）を全 LLM 条件（C1〜C4）で使用。
- **入力テキスト**: 同一の `ResearchDocument.content_text` を全条件に投入。
- **生成パラメータ**: Temperature = 0.0, top_p = 1.0, seed = 42 を固定。
- **評価基準**: 同一の評価スクリプトおよびゴールドスタンダード正解集を使用。

---

## 7. 定量的評価指標と計算規則

各指標の分母・分子・照合規則を数学的に厳密に定義する。

### 7.1 エンティティ照合規則（Entity Matching）
抽出されたエンティティ $e_{pred}$ とゴールドスタンダード正解 $e_{gold}$ の一致判定：
- **Strict Match**: エンティティ名が一致（正規化小文字一致）かつ `EvidenceSpan` の Jaccard 係数 $\ge 0.5$。
- **Soft Match**: 同義語マッピング下での名称一致、または意味的類似度（Embedding cosine similarity $\ge 0.85$）。

### 7.2 指標定義

1. **Entity Precision ($P$) / Recall ($R$) / F1**:
   $$P = \frac{|E_{pred} \cap E_{gold}|}{|E_{pred}|}, \quad R = \frac{|E_{pred} \cap E_{gold}|}{|E_{gold}|}, \quad F1 = \frac{2PR}{P+R}$$
   ※ Model, Method, Capability, Assumption, Limitation の各エンティティ種別ごとに個別集計し、マクロ平均も算出。
2. **Quote Validity Rate ($R_{QV}$)**:
   $$R_{QV} = \frac{\sum_{e \in E_{pred}} \mathbb{I}(\text{input\_text}[\text{start}:\text{end}] == \text{verbatim\_quote})}{|E_{pred}|}$$
   （目標基準: **100.0%**）
3. **Claim Support Rate ($R_{CS}$)**:
   $$R_{CS} = \frac{\sum_{e \in E_{pred}} \mathbb{I}(\text{ClaimIsSupported}(\text{evidence}, e))}{|E_{pred}|}$$
   分母: 抽出された全エンティティ。分子: 引用文によって意味的に支持されているエンティティ数。（目標基準: **92.0% 以上**）
4. **Overclaim Rate ($R_{OC}$)**:
   $$R_{OC} = \frac{\text{原文の文脈（制約・否定・ヘッジ）を切り落として誇大抽出されたクレーム数}}{|E_{pred}|}$$
   （目標基準: **3.0% 以下**）
5. **Hallucination Rate ($R_{HAL}$)**:
   $$R_{HAL} = \frac{\text{原文に根拠が一切存在しない捏造エンティティ数}}{|E_{pred}|}$$
   （目標基準: **1.0% 以下**）
6. **Vocabulary Compliance Rate ($R_{VC}$)**:
   $$R_{VC} = \frac{\text{Capability の入出力のうち L1 語彙に正しくマッピングされた数}}{\text{抽出された Capability の全入出力数}}$$
   （目標基準: **95.0% 以上**）
7. **統計的信頼区間**:
   - 各スコアについて、Bootstrap リサンプリング（$B=1000$）による **95% 信頼区間 (CI)** を算出して報告。

---

## 8. 非決定論性への対策と実験メタデータ管理

### 8.1 Temperature 0 の不完全性と反復評価
- **課題**: Temperature = 0 でも、クラウド LLM API では並列推論スレッドや GPU 浮動小数点丸め、内部キャッシュ、マイナーリビジョン更新により出力が揺らぐ場合がある。
- **対策**:
  1. 同一のプロンプト・入力に対し、**独立に 3 回の反復実行 ($N=3$)** を実施。
  2. 3 回の出力間におけるエンティティ一致度（Self-Consistency Jaccard 類似度）を測定。
  3. 抽出結果レコード（`ExtractedKnowledgeRecord.generation_metadata`）に以下を完全保存:
     - 使用モデルの完全な checkpoint 名
     - プロンプトテンプレートの SHA-256
     - 入力テキストの SHA-256
     - 生レスポンス JSON テキスト全文
     - 生成ハイパーパラメータ（seed, temperature, top_p, top_k 等）

---

## 9. 文書内局所関係（Local Relations）と構造化

抽出されたエンティティ間の関係を孤立させず、同一文書内の局所グラフとして保持する。

### 9.1 関係性エッジの定義
1. **`EXPLAINS`**: Model または Method が Phenomenon を説明・再現する関係。
   - 例: `Minority Game` --EXPLAINS--> `Volatility Clustering`
2. **`APPLIES_TO`**: Method が Model を対象として適用される関係。
   - 例: `Neural Posterior Estimation` --APPLIES_TO--> `Agent-Based Model`
3. **`PROVIDED_BY`**: Capability が特定の Model または Method によって提供される関係。
   - 例: `Parameterized Price Generation` --PROVIDED_BY--> `Agent-Based Model`
4. **`REQUIRES`**: Capability または Model が特定の Assumption を必須前提とする関係。
   - 例: `Black-Scholes Pricing` --REQUIRES--> `Constant Volatility Assumption`
5. **`CONSTRAINED_BY`**: Capability または Method が特定の Limitation に制約される関係。
   - 例: `High-dimensional NPE` --CONSTRAINED_BY--> `Curse of Dimensionality`

---

## 10. H1b（Entity Resolution）への引き渡し契約

H1a の出力は、表記揺れの統一や異分野エンティティ間の名寄せを行う **H1b** へ引き渡される。

### 10.1 引き渡しデータ構造 (`H1bInputBatch`)

```python
class LocalEntityRef(BaseModel):
    local_id: str
    entity_type: Literal["phenomenon", "model", "method", "capability", "assumption", "limitation"]
    name_or_statement: str
    evidence_quote: str

class LocalEdgeRef(BaseModel):
    src_local_id: str
    dst_local_id: str
    relation_type: str

class H1bDocumentPackage(BaseModel):
    """H1a から H1b に引き渡される 1 資料単位のパッケージ"""
    source_id: str
    version_id: str
    source_type: Literal["paper", "oss"]
    primary_name: str
    content_scope: Literal["abstract_only", "readme_only", "metadata_only", "full_text"]
    evidence_status: Literal["reported", "unverified"]
    
    # 局所エンティティ一覧
    entities: List[LocalEntityRef]
    # 局所関係一覧
    local_relations: List[LocalEdgeRef]
    # 完全な詳細構造
    raw_extracted_record: ExtractedKnowledgeRecord
```

---

## 11. 実装ステップと受入基準

計画承認後の実装・実験は以下の厳密な順序で行う。

### 11.1 実装ステップ

```text
Step 1: Gold Standard データセットの構築 (30件)
        - Development Set (10件) の人手アノテーション作成
        - Challenge Set (8件) の罠・エッジケース作成
        - Holdout Set (12件) のアノテーション作成・完全隔離暗号化

Step 2: スキーマ・二層 Controlled Vocabulary の実装
        - src/research_pipeline/extractors/models.py (Pydantic スキーマ)
        - src/research_pipeline/extractors/vocab.py (L1/L2 語彙集)
        - tests/unit/test_h1a_schemas.py で検証

Step 3: 3層 Evidence 検証器の実装
        - src/research_pipeline/extractors/validator.py (Quote Validity, Claim Support, Claim Completeness)
        - tests/unit/test_evidence_validator.py で検証

Step 4: LLM 抽出器プロンプト & クライアント実装
        - prompts/extraction_v2_0.txt (二層型, unknown許容, 局所関係抽出)
        - src/research_pipeline/extractors/llm_extractor.py (JSON Schema強制, メタデータ記録)
        - tests/unit/test_llm_extractor_mock.py で検証

Step 5: 診断パイプラインの実装
        - src/research_pipeline/extractors/diagnostics.py (Overclaim, Missing Preconditions)

Step 6: Development Set によるアブレーション実験 (C0〜C4)
        - プロンプトの反復改善とエラー分析
        - 信頼区間の算出

Step 7: Holdout Set によるブラインド最終評価 & 受入検査
        - Holdout Set への 1 回限りの推論実行
        - 指標判定と受入合否の確認
```

### 11.2 H1a 受入基準（最終合格ライン）

1. **Quote Validity Rate**: Holdout Set において **100.0%**（機械的完全一致）。
2. **Claim Support Rate**: Holdout Set において **92.0% 以上**（論理的裏付け）。
3. **Overclaim Rate**: Challenge Set / Holdout Set において **3.0% 以下**。
4. **Hallucination Rate**: 全評価セットにおいて **1.0% 以下**。
5. **Entity F1 (Strict Match)**: Holdout Set において **80.0% 以上**（Soft Match では **85.0% 以上**）。
6. **Vocabulary Compliance**: 抽出 Capability の入出力型の **95.0% 以上** が L1 語彙にマッピングされ、未記載時は確実に `unknown` / `is_unspecified=True` となること。
7. **再現性**: 同一入力・同一条件の 3 回実行で、出力エンティティの Jaccard 一致度が **95.0% 以上** であること。
8. **AlgoForge 無侵害**: AlgoForge のコード・DB に一切変更がないこと。

---
**Plan Prepared by:** Antigravity Agent  
**Awaiting Review:** H1a 科学的実験計画書 v0.2 の承認判定待ち
