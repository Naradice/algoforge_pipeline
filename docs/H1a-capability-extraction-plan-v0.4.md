# HDE H1a — Capability Extraction 科学的実験計画書 v0.4

**Status:** Proposed (Scientific Review Pending)  
**Date:** 2026-11-15  
**Version:** 0.4  
**Target:** Hypothesis Discovery Engine (HDE) Phase 1 / Step H1a  
**Parent Specifications:**
- [architecture.md](file:///c:/Users/ksato/workspace/finance/research_pipeline/docs/architecture.md)
- [hypothesis_discovery_engine.md](file:///c:/Users/ksato/workspace/finance/research_pipeline/docs/hypothesis_discovery_engine.md)
- [H0-completion-report.md](file:///c:/Users/ksato/workspace/finance/research_pipeline/docs/H0-completion-report.md)
- [H1a-capability-extraction-plan-v0.3.md](file:///c:/Users/ksato/workspace/finance/research_pipeline/docs/H1a-capability-extraction-plan-v0.3.md)

---

## 1. 目的と科学的課題

### 1.1 目的
Step H1a の目的は、H0 で収集・保存された `ResearchDocument`（論文 Abstract および OSS README、メタデータ）を入力とし、そこから **Phenomenon（対象現象）**、**Model（モデル）**、**Method（手法）**、**Capability（操作可能な能力）**、**Assumption（前提仮定）**、**Limitation（適用制約）**、およびそれらの **文書内局所関係（Local Relations）** を高精度かつ再現可能な形で抽出し、原典版に厳密に紐付いた構造化知識レコード（`ExtractedKnowledgeRecord`）を生成することである。

### 1.2 計画 v0.4 における改訂方針
計画 v0.3 のレビューにおいて指摘された「データ整合性」「引用検証の厳密性」「ID体系の試行分離」「予算制御の予約精算」「アノテーションのマルチラベル化」を解決するため、本計画 v0.4 では以下の改訂を行う：
1. **Raw LLM 応答の検証前独立保存アーキテクチャ**: Pydantic モデル検証の成否にかかわらず、生の文字列応答（`raw_response_text`）、構文パース結果（`parsed_candidates`）、パースエラー（`parse_error`）を独立保存し、JSON構文エラーや Evidence 欠落、IO 整合性違反も漏れなく監査・分析できる設計とする。
2. **EvidenceSpan の厳密定義と決定論的一致**: 参照対象フィールド名（`target_field_name`）、フィールド正規化仕様（Unicode NFC, 改行統一 `\n`）、フィールドハッシュ（`target_field_hash`）、および Unicode Codepoint 文字オフセットを定義。Quote Validity は曖昧な正規化比較ではなく、正規化済み対象文字列に対する厳密な部分文字列一致（Strict Codepoint Equality）とする。
3. **`extraction_run_id` を含む候補エンティティ ID の策定**: 同一文書版に対する反復試行（$R=3$）や条件比較（C1〜C4）のエンティティを区別するため、H1a 出力 ID に `extraction_run_id` を含めた `candidate_entity_id` を定義し、H1b の `canonical_entity_id` と明確に分離する。
4. **二相型予算管理システム（事前予約・事後精算）**: LLM 呼び出し前に最悪トークン数・費用を事前予約し、30.00 USD または 5,000,000 トークンのいずれかに達した時点で即座に安全停止（Hard Stop）する。リトライおよび検証用（Claim Support）LLM も予算管理に完全算入する。
5. **複数概念文へのマルチラベル対応**: 文単位アノテーションにおいて、1文に複数概念（例: Method と Capability）が含まれる実態に合わせ、概念ごとのバイナリマルチラベル分類として Cohen's / Fleiss' Kappa を定義。
6. **文書単位クラスターブートストラップの主CI化**: 文書相関を考慮し、文書単位クラスターブートストラップ（$B=1000$）を全指標の主信頼区間（Primary CI）とする。
7. **GitHub Fork の根リポジトリ（Root Parent）重複排除**: フォーク元・フォーク先・兄弟フォークを `root_parent_repo_id` に基づいて同一グループとしてクラスタリングし、split 間の混入を完全遮断。
8. **機密情報の許可リスト方式（Allowlist）保護**: ブラックリストではなく、保存を許可されたメタデータキーのみを抽出保存する許可リスト方式を採用。
9. **固定モデル廃止時の移行プロトコル**: チェックポイントが提供終了（Sunset）となった場合のモデル移行判定（Dev 10件での $\Delta F1 \le 2.0\%$ 検証）を規定。

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
    """
    原文からの完全一致引用と位置情報。
    target_field_name で指定された正規化テキストに対する厳密なスライス一致を義務付ける。
    """
    target_field_name: Literal["abstract_text", "readme_text", "title_text"] = Field(
        ..., description="参照対象の正規化テキストフィールド名"
    )
    target_field_hash: str = Field(
        ..., description="参照対象正規化テキストの SHA-256 ハッシュ (64進数小文字)"
    )
    verbatim_quote: str = Field(
        ..., min_length=1, description="正規化テキストからの一致抜粋文字列 (完全一致)"
    )
    start_char: int = Field(..., ge=0, description="Unicode Codepoint 開始インデックス")
    end_char: int = Field(..., ge=0, description="Unicode Codepoint 終了インデックス")

    @model_validator(mode="after")
    def validate_indices(self) -> "EvidenceSpan":
        if self.end_char <= self.start_char:
            raise ValueError(
                f"end_char ({self.end_char}) must be strictly greater than start_char ({self.start_char})"
            )
        return self

class PhenomenonEntity(BaseModel):
    local_id: str = Field(..., description="文書内局所ID (例: 'phe_01')")
    name: str = Field(..., description="現象の標準的名称")
    description: str = Field(..., description="現象の内容説明")
    evidence: EvidenceSpan

class ModelEntity(BaseModel):
    local_id: str = Field(..., description="文書内局所ID (例: 'mod_01')")
    name: str = Field(..., description="モデル名称")
    category: Literal["agent_based", "statistical_physics", "state_space", "deep_learning", "stochastic_process", "other"]
    description: str = Field(..., description="数理的・構造的説明")
    evidence: EvidenceSpan

class MethodEntity(BaseModel):
    local_id: str = Field(..., description="文書内局所ID (例: 'met_01')")
    name: str = Field(..., description="手法名称")
    category: Literal["parameter_inference", "anomaly_detection", "state_estimation", "representation_learning", "optimization", "hypothesis_testing", "other"]
    description: str = Field(..., description="計算・推定手順の説明")
    evidence: EvidenceSpan

class CapabilityIO(BaseModel):
    """Capability の入出力仕様（状態整合性を厳格検証）"""
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
    local_id: str = Field(..., description="文書内局所ID (例: 'asm_01')")
    statement: str = Field(..., description="仮定の内容")
    is_explicit: bool = Field(..., description="原文に明示されているか")
    evidence: EvidenceSpan

class LimitationItem(BaseModel):
    local_id: str = Field(..., description="文書内局所ID (例: 'lim_01')")
    statement: str = Field(..., description="制約の内容")
    evidence: EvidenceSpan

class CapabilityEntity(BaseModel):
    local_id: str = Field(..., description="文書内局所ID (例: 'cap_01')")
    name: str = Field(..., description="能力名称 (例: 'Parameterized Order Flow Simulation')")
    operation: Literal["generate", "infer", "detect", "predict", "transform", "optimize"] = Field(...)
    inputs: List[CapabilityIO] = Field(..., min_length=1, description="入力仕様")
    outputs: List[CapabilityIO] = Field(..., min_length=1, description="出力仕様")
    preconditions: List[str] = Field(default_factory=list, description="実行に必要な前提条件")
    limitations: List[str] = Field(default_factory=list, description="既知の適用限界")
    evidence: EvidenceSpan

class LocalRelation(BaseModel):
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

class ExtractedKnowledgeRecord(BaseModel):
    """H1a の構造化知識レコード（Accepted 確定用）"""
    source_id: str = Field(..., description="論理資料ID (H0)")
    version_id: str = Field(..., description="資料版ID (H0)")
    version_hash: str = Field(..., description="コンテンツSHA-256")
    extraction_run_id: str = Field(..., description="抽出試行一意ID")
    content_scope: Literal["abstract_only", "readme_only", "metadata_only", "full_text"]
    evidence_status: Literal["reported", "unverified"] = Field("reported")
    
    phenomena: List[PhenomenonEntity] = Field(default_factory=list)
    models: List[ModelEntity] = Field(default_factory=list)
    methods: List[MethodEntity] = Field(default_factory=list)
    capabilities: List[CapabilityEntity] = Field(default_factory=list)
    assumptions: List[AssumptionItem] = Field(default_factory=list)
    limitations: List[LimitationItem] = Field(default_factory=list)
    local_relations: List[LocalRelation] = Field(default_factory=list)
```

### 2.3 機密情報保護の許可リスト方式（Allowlist）
- **背景**: ブラックリスト（正規表現による置換）は、未知の環境変数や新しいトークン形式が漏洩するリスクがある。
- **許可リスト方式（Allowlist）**:
  保存されるメタデータ辞書（`GenerationMetadata.generation_params`）には、事前に明示的に許可されたキーのみを転記し、未定義キーは自動破棄する。
  ```python
  ALLOWED_GENERATION_PARAM_KEYS = {
      "model_checkpoint",
      "temperature",
      "top_p",
      "top_k",
      "seed",
      "max_output_tokens",
      "response_mime_type",
  }
  ```
  HTTPリクエストヘッダー（`Authorization`, `x-goog-api-key` 等）やクライアント内部設定は一切メタデータ辞書に含めない。
- **防御的二重マスク**: ログ出力層においても、万一のエラーダンプに備え、正規表現マスク（パス `[WORKSPACE]`, トークン `[REDACTED_API_KEY]`）を適用する。

---

## 3. 抽出結果の 3 階層分離と生応答監査アーキテクチャ

### 3.1 Raw LLM 応答の検証前独立保存
LLM の応答が不正な JSON であったり、Pydantic スキーマ違反を起こした場合でも、クラッシュせずに生応答を確実に永続化し、監査・原因分析を可能にするため、Pydantic 検証前に `RawExtractionArtifact` を保存する。

```python
class RawExtractionArtifact(BaseModel):
    """Pydantic スキーマ検証前に保存される生抽出アーティファクト"""
    extraction_run_id: str = Field(..., description="抽出試行一意ID")
    source_id: str
    version_id: str
    condition_id: Literal["C0", "C1", "C2", "C3", "C4"]
    raw_response_text: str = Field(..., description="LLM から受信した生の完全文字列")
    http_status: int = Field(200, description="HTTP ステータスコード")
    token_usage: Dict[str, int] = Field(..., description="消費トークン数 (prompt, completion, total)")
    executed_at: str
    model_checkpoint: str
    prompt_hash: str
    target_field_hashes: Dict[str, str] = Field(..., description="対象フィールド名とSHA-256")
    
    # 構文解析状態
    json_parse_status: Literal["success", "invalid_json", "empty_response", "not_applicable"]
    parse_error: Optional[str] = Field(None, description="構文エラーまたはスキーマ検証エラーの詳細")
    parsed_candidates: Optional[Dict[str, Any]] = Field(None, description="json.loads 成功時の生辞書データ")
```

### 3.2 抽出 3 態バンドル（Raw / Accepted / Rejected）
生抽出アーティファクトをもとに検証パイプラインを実行し、合格したエンティティのみを `ExtractedKnowledgeRecord` に格納し、不合格の項目は棄却理由を付与して `RejectionRecord` に記録する。

```python
class RejectionRecord(BaseModel):
    """検証器により棄却された抽出項目と理由"""
    local_id: str
    entity_type: str
    rejection_stage: Literal[
        "syntax_parse",
        "quote_validity",
        "claim_support",
        "claim_completeness",
        "io_consistency",
        "schema_violation"
    ]
    rejection_reason: str
    raw_entity_data: Dict[str, Any]

class ExtractionTrioBundle(BaseModel):
    """資料・試行ごとの完全監査バンドル"""
    extraction_run_id: str
    source_id: str
    version_id: str
    raw_artifact: RawExtractionArtifact
    accepted_record: Optional[ExtractedKnowledgeRecord]
    rejected_items: List[RejectionRecord]
```

### 3.3 検証器棄却によるトレードオフの定量化
検証器が適合率（Precision）向上に寄与する一方で、再現率（Recall）や文書カバレッジ（Coverage）をどの程度損なうかを以下の指標で明示的に測定・報告する。

$$\Delta P = \text{Precision}_{\text{Accepted}} - \text{Precision}_{\text{Raw}}$$
$$\Delta R = \text{Recall}_{\text{Accepted}} - \text{Recall}_{\text{Raw}}$$
$$\Delta \text{Coverage} = \frac{\text{Accepted Capability が 1 件以上残存した文書数}}{N_{\text{total}}} - \frac{\text{Raw 候補に Capability が 1 件以上存在した文書数}}{N_{\text{total}}}$$

### 3.4 Claim Support の最終正解基準（Ground Truth）
- 自動 NLI モデルや LLM 判定器は、実験実行時の候補スクリーニングまたはアブレーション比較にのみ用いる。
- **評価実験における Claim Support の最終正解基準（Ground Truth）は、独立した専門家 2 名による人手ブラインド判定（不一致時は Adjudicator による裁定）とする**。モデル自身の自己評価や自動 LLM 判定を正解基準と混同することを禁止する。

---

## 4. 厳密引用検証（Quote Validity）の数学的・文字列表現定義

### 4.1 参照対象フィールドと文字列正規化仕様
引用の開始・終了オフセット（`start_char`, `end_char`）が曖昧になることを防ぐため、以下の決定論的正規化パイプラインを定義する。

1. **対象フィールド**:
   - `abstract_text`: 論文 Abstract のテキスト本文。
   - `readme_text`: GitHub README のテキスト本文。
   - `title_text`: 論文・リポジトリのタイトル文字列。
2. **決定論的正規化仕様（Canonical Normalization）**:
   - **Unicode 正規化**: 全文字列を Unicode NFC（`unicodedata.normalize('NFC', text)`）に変換。
   - **改行コードの統一**: すべての `\r\n` および `\r` を `\n` に置換。
   - **外側空白の除去**: 文字列全体の先頭・末尾の空白文字を除去（`strip()`）。
   - 本文内部のスペース、インデント、改行は一切変更せず保持する。
3. **フィールドハッシュ（`target_field_hash`）**:
   - 正規化後の UTF-8 バイト列に対する SHA-256 ハッシュを計算し、`EvidenceSpan` に必須記録する。
   $$\text{target\_field\_hash} = \text{SHA256}(\text{CanonicalText}_{\text{NFC, LF}})$$

### 4.2 文字オフセットと厳格一致（Strict Codepoint Equality）
- **文字オフセット定義**:
  - `start_char` および `end_char` は、正規化済み文字列に対する **Python Unicode Codepoint インデックス**（0-indexed, 半開区間 `[start_char:end_char]`）とする。
- **厳密一致判定**:
  Quote Validity は以下の厳密一致式によって判定される。
  $$\text{CanonicalText}[\text{start\_char}:\text{end\_char}] == \text{verbatim\_quote}$$
  曖昧な正規化比較（スペースの自動圧縮やケース非依存比較など）は判定ロジックから排除し、**1 文字・1 空白の狂いもなく完全一致すること** を必須とする。不一致の場合は即座に `quote_validity` ステージで棄却される。

---

## 5. Controlled Vocabulary の二層構造

### 5.1 L1 (Universal Types) と L2 (Domain Specializations)
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

### 5.2 未知属性の根拠捏造防止と整合性
- **ルール**: 原文に入出力仕様の記載がない場合は、LLM による常識的補完を厳禁とし、`is_unspecified=True, data_type_l1='unknown', data_type_l2=None, evidence=None` を強制する。これに違反する出力は Pydantic バリデータにより即時棄却される。

---

## 6. 評価データセット設計と隔離・汚染防止

### 6.1 3分割構成
評価データセット（全 30 件）を以下の 3 群に分離する。Holdout 12 件は「初期パイロット評価（実装受入ゲート）」とし、母集団に対する一般化の統計的実証とは明確に区別する。

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

### 6.2 GitHub Fork の根リポジトリ（Root Parent）重複排除
データセット分割間での情報漏洩（Contamination）を完全に防止するため、以下の独立性検証ルールを適用する：
1. **GitHub Fork の根リポジトリ解決**:
   - リポジトリ $R$ に対し、GitHub API を再帰的に辿って最上位の親リポジトリ `root_parent_repo_id` を特定する。
   $$\text{root\_id}(R) = \begin{cases} R.\text{id} & \text{if } R.\text{parent is None} \\ \text{root\_id}(R.\text{parent}) & \text{otherwise} \end{cases}$$
   - **制約**: 同一の `root_parent_repo_id` を共有するすべてのリポジトリ（フォーク元、フォーク先、および同一親を持つ兄弟フォーク）は、同一の split にのみ割り当て可能とし、異なる split への分散配置を厳格に禁止する。
   - 同一の README コンテンツハッシュ（`extracted_text_hash`）を持つリポジトリも同様に split 間での重複を禁止する。
2. **arXiv**:
   - 同一の arXiv Base ID の重複禁止。
   - 同一筆頭著者チームによる同一テーマの派生論文の split 間重複禁止。

### 6.3 複数概念を含む文への対応：マルチラベル評価
- **課題**: 1 つの文の中に複数の概念（例: 「我々はボラティリティを予測する（Capability）Hawkes過程（Model）を提案する」）が含まれる場合、単一排他ラベルではアノテーションの不一致や情報落ちが発生する。
- **マルチラベル設計**:
  - 文書内の各文 $S_i$ に対し、アノテーターは以下の 6 概念それぞれについて独立した 2 値判定（Present = 1, Absent = 0）を行う。
    $$\mathbf{y}_i = (y_{i, \text{phe}}, y_{i, \text{mod}}, y_{i, \text{met}}, y_{i, \text{cap}}, y_{i, \text{asm}}, y_{i, \text{lim}}) \in \{0, 1\}^6$$
  - **負例文（`NONE`）**: いずれのフラグも立たない文（$\sum_k y_{i, k} = 0$）。
  - **一致度算出**:
    - 各概念クラス $k$ ごとに独立した 2 値判定表（$2 \times 2$）を構成し、クラス別 Cohen's Kappa $\kappa_k$ を算出。
    - 全概念のマクロ平均 $\bar{\kappa} = \frac{1}{6} \sum_k \kappa_k$ を報告し、各概念で $\kappa_k \ge 0.70$、マクロ平均で $\bar{\kappa} \ge 0.75$ をアノテーション受入基準とする。

---

## 7. 段階的アブレーション比較実験 (C0〜C4) と正規化変換

### 7.1 実験条件（C0〜C4）の構成

| 条件ID | 手法名 | 構成要素・プロンプト | 出力形式 |
|---|---|---|---|
| **C0** | Rule-based Baseline | 正規表現・キーワード辞書マッピング | ヒューリスティック辞書 |
| **C1** | Free-text LLM | 自由形式テキスト抽出プロンプト | 非構造化プレーンテキスト |
| **C2** | Structured Output Only | Pydantic JSON Schema 強制 (Evidenceなし) | JSON (スキーマ準拠) |
| **C3** | C2 + Evidence Grounding | Pydantic Schema + Verbatim Quote 引用強制 | JSON (EvidenceSpan 含む) |
| **C4 (Proposed)** | C3 + Controlled Vocab | Pydantic + Evidence + L1/L2 語彙 + unknown 許容 | JSON (完全スキーマ準拠) |

### 7.2 共通正規化変換パイプライン（Standardization Converter）
全条件の出力を同一の評価関数で公平に比較するため、決定論的正規化コンバータを適用する。
- **C1 変換失敗の独立報告**:
  C1（Free-text LLM）は非構造化テキストを出力するため、正規表現およびヒューリスティック抽出器によって構造化を試みる。
  - パース不能またはセクション欠落により変換できなかったケースは、エンティティ抽出失敗ではなく **構文・変換失敗（`conversion_failure`）** として独立集計する。
  - 評価レポートには以下の内訳表を明記する：
    $$\text{Total Raw Responses} = N_{\text{converted}} + N_{\text{conv\_fail}}$$
    これにより、「プロンプトによる抽出能力の限界」と「構文出力の非構造性に起因するパースエラー」を完全に分離して評価する。

---

## 8. 定量的評価指標・照合規則・統計的推定

### 8.1 人手定義意味的同値一対一マッチング（主指標）
ゴールドスタンダード正解集合 $G$ と抽出エンティティ集合 $P$ の照合には、専門家が定義した意味的同値ルールに基づく **重み付き二部グラフ最大マッチング（Kuhn-Munkres法）** を主指標として採用する。
- 重み行列 $W_{ij} \in \{0, 1\}$: $g_i \in G$ と $p_j \in P$ が意味的に同一の概念を表現している場合のみ 1。
- 1 つの正解に複数の抽出がマッチすること、およびその逆を完全に排除した一対一マッチング数を $|G \cap P|_{\text{semantic}}$ とする。
- 補助指標として Strict String Match および Embedding Cosine Similarity（$\ge 0.85$）を併記する。

### 8.2 指標定義

1. **Semantic Entity Precision ($P_{\text{sem}}$) / Recall ($R_{\text{sem}}$) / F1 (主指標)**:
   $$P_{\text{sem}} = \frac{|G \cap P|_{\text{semantic}}}{|P|}, \quad R_{\text{sem}} = \frac{|G \cap P|_{\text{semantic}}}{|G|}, \quad F1_{\text{sem}} = \frac{2 P_{\text{sem}} R_{\text{sem}}}{P_{\text{sem}} + R_{\text{sem}}}$$
2. **Quote Validity Rate ($R_{QV}$)**:
   $$R_{QV} = \frac{\sum_{p \in P} \mathbb{I}(\text{CanonicalText}[\text{start}:\text{end}] == \text{verbatim\_quote})}{|P|}$$
3. **Claim Support Rate ($R_{CS}$)**:
   $$R_{CS} = \frac{\sum_{p \in P} \mathbb{I}(\text{HumanJudgeSupport}(\text{evidence}, p) == \text{Entailed})}{|P|}$$
   ※ 最終評価は独立した専門家人手判定に基づく。
4. **Overclaim Rate ($R_{OC}$)**:
   $$R_{OC} = \frac{\text{文脈上の制約・否定・不確実性を無視して抽出された過剰主張エンティティ数}}{|P|}$$
5. **Observed Hallucination ($R_{HAL}$)**:
   $$R_{HAL} = \frac{\text{原文に根拠が一切存在せず捏造されたエンティティ数}}{|P|}$$
6. **Vocabulary Compliance Rate ($R_{VC}$)**:
   $$R_{VC} = \frac{\text{L1語彙準拠かつ整合性バリデータを通過したIO数}}{\text{全抽出IO数}}$$

### 8.3 文書単位クラスターブートストラップ（主信頼区間）
同一文書内の複数エンティティは互いに独立でない（相関を持つ）ため、エンティティ単位のブートストラップは分散を過小評価する。本実験では **文書単位クラスターブートストラップ（Document-level Cluster Bootstrap）** を全評価指標の **主信頼区間（Primary CI）** とする。
- **手順**: $N$ 件の文書インデックスから重複を許して $N$ 件をリサンプリング（$B=1000$ 回）。サンプリングされた文書群の全エンティティから各指標を再計算し、経験分布の 2.5% と 97.5% 点を 95% 信頼区間とする。

### 8.4 小標本（$N=12$）における統計的限界の明記
- Holdout Pilot Set（12件）で得られるエンティティ総数は 50〜80 件程度と想定される。
- 比率指標には Wilson Score 区間を併記し、例えばパイロットで幻覚が 0 件観察された場合でも、その統計的意味は「95% 信頼区間の上限が約 $3/60 = 5.0\%$ であること」に留まり、「母集団に対して 1.0% 以下の幻覚率を実証した」とは主張しないことを厳命する。

---

## 9. 実行制御・反復評価・予算予約精算システム

### 9.1 実装利用 LLM の選定
- **Primary Model**: `gemini-2.5-flash`（高速性、長文脈対応、JSON Schema 完全対応）。
- **Secondary / Verification Model**: `gemini-2.5-pro`（推論検証、アブレーション照合）。
- **API バージョン固定**: 日付指定の特定チェックポイントを使用し、`generation_metadata.model_checkpoint` に完全記録する。

### 9.2 固定モデル版が廃止（Sunset）された場合の移行プロトコル
クラウドプロバイダにより固定チェックポイントが廃止された場合は、以下の移行プロトコルを厳格に適用する：
1. **移行判定テスト（Migration Transition Test）**:
   - 新しい後継チェックポイントに対し、Development Set（10件）を同一設定で再実行する。
   - 旧チェックポイントの保存結果との間で、Jaccard 一致度 $\ge 90.0\%$、および $\Delta F1 \le 2.0\%$ であることを検証する。
2. **移行記録の義務付け**:
   - 移行テスト結果、API レスポンスの `system_fingerprint`、および移行差分ログを明記した上で、実行環境バージョンをインクリメントする。無断・未検証でのモデル切替は一切禁止する。

### 9.3 独立反復試行（$R=3$）
同一パラメータ（Temperature=0.0, Seed=42）で各条件について **$R = 3$ 回の独立試行** を行い、試行間の分散および Jaccard 一致度（Self-Consistency）を測定する。

### 9.4 事前予約・事後精算型予算管理システム（Two-phase Budget Tracker）
実験中に予算オーバーランが発生することを構造的に防止するため、事前予約・事後精算型の予算管理を導入する。

```text
               LLM 呼び出し要求
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Pre-call Reservation (事前枠予約)                        │
│    - 最悪見積: est_tokens = len(prompt) + max_output_tokens │
│    - 最悪費用: est_cost = est_tokens * worst_rate           │
│    - 判定: cumulative + reserved + est > 30.00 USD          │
│            OR cumulative_tokens + est > 5,000,000 tokens    │
└──────────────────────┬──────────────────────────────────────┘
                       │
       ┌───────────────┴───────────────┐
       │ 超過                          │ 枠内OK
       ▼                               ▼
[Hard Stop 停止]             [予約確保: reserved += est]
- 即時例外送出                         │
- 部分結果保存                         ▼
                             [LLM API 呼び出し実行]
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Post-call Settlement (事後実測精算)                      │
│    - レスポンスから実消費トークン数 (actual_tokens) を取得  │
│    - 実消費費用 (actual_cost) を計算                        │
│    - 予約解除: reserved -= est                              │
│    - 実績加算: cumulative += actual                         │
└─────────────────────────────────────────────────────────────┘
```

- **予算上限**: **30.00 USD または 5,000,000 トークンのいずれかに達した時点で即時停止（Hard Stop）**。
- **対象範囲**: Primary 抽出器の全条件（C1〜C4）、リトライ呼び出し（タイムアウト・レート制限）、および検証用 LLM（Claim Support）のすべての API コールを同一トラッカーで合算管理する。いかなる呼び出しも本予約精算をバイパスできない。

---

## 10. H1b（Entity Resolution & Cross-Document Graph）への引き渡し契約

### 10.1 `extraction_run_id` を導入した候補エンティティ ID 仕様
同一文書版に対する反復試行や複数手法（C1〜C4）の抽出結果が混同されることを防ぎ、H1b の名寄せ対象（Mention）として一意に識別できるよう、以下の ID 体系を定義する。

```python
# H1a 候補エンティティID形式:
# <source_id>:<version_hash[:8]>:<extraction_run_id>:<entity_type>:<local_id>
candidate_entity_id = f"{source_id}:{version_hash[:8]}:{extraction_run_id}:{entity_type}:{local_id}"
# 例: "arxiv_2301_01234:a8f13b2c:run_c4_trial1:capability:cap_01"

# H1a 候補エッジID形式:
# <src_candidate_id>-><relation_type>-><dst_candidate_id>
candidate_edge_id = f"{src_candidate_id}->{relation_type}->{dst_candidate_id}"
```

- **H1b Canonical ID との責務分離**:
  - `candidate_entity_id`: H1a が抽出した 1 試行・1 箇所の出現（Mention）を一意に示す ID。
  - `canonical_entity_id`: H1b が複数文書・複数試行の名寄せ（Entity Resolution）を経て合成・決定する代表 ID（例: `canonical:model:minority_game`）。H1a 側では canonical ID を勝手に付与せず、H1b に名寄せを完全に委ねる。

### 10.2 引き渡しデータ構造 (`H1bDocumentPackage`)

```python
class CandidateEntityRef(BaseModel):
    candidate_id: str
    source_id: str
    version_id: str
    extraction_run_id: str
    entity_type: Literal["phenomenon", "model", "method", "capability", "assumption", "limitation"]
    name_or_statement: str
    verbatim_quote: str
    quote_start_char: int
    quote_end_char: int
    target_field_name: str
    target_field_hash: str

class CandidateEdgeRef(BaseModel):
    candidate_edge_id: str
    src_candidate_id: str
    dst_candidate_id: str
    relation_type: str
    evidence_quote: str

class H1bDocumentPackage(BaseModel):
    """H1a から H1b に引き渡される 1 資料・1 試行単位のパッケージ"""
    source_id: str
    version_id: str
    version_hash: str
    extraction_run_id: str
    source_type: Literal["paper", "oss"]
    primary_name: str
    content_scope: Literal["abstract_only", "readme_only", "metadata_only", "full_text"]
    evidence_status: Literal["reported", "unverified"]
    
    # Accepted のみが候補エンティティとして引き渡される
    entities: List[CandidateEntityRef]
    local_relations: List[CandidateEdgeRef]
    
    # 監査用完全バンドル参照
    bundle: ExtractionTrioBundle
```

---

## 11. 実装ステップと受入判定基準の分離

### 11.1 実装・実験ステップ

```text
Step 1: Gold Standard データセットの構築 (30件)
        - Development Set (10件) のマルチラベル人手アノテーション作成
        - Challenge Set (8件) の罠・エッジケース作成
        - Holdout Pilot Set (12件) のアノテーション作成・完全隔離
        - Root Parent Repository ID による Fork 排除テスト (test_dataset_isolation.py)

Step 2: スキーマ・二層 Controlled Vocabulary の実装
        - src/research_pipeline/extractors/models.py (EvidenceSpan厳密定義, IO整合性バリデータ)
        - src/research_pipeline/extractors/vocab.py (L1/L2 語彙集)
        - tests/unit/test_h1a_schemas.py で検証

Step 3: 3層 Evidence 検証器 & 生応答アーティファクト保存の実装
        - src/research_pipeline/extractors/raw_artifact.py (Pydantic検証前の生応答保存)
        - src/research_pipeline/extractors/validator.py (Strict Codepoint QV, CS, CC, IO整合性)
        - tests/unit/test_evidence_validator.py で検証

Step 4: LLM クライアント・正規化コンバータ・二相予算管理の実装
        - src/research_pipeline/extractors/budget_tracker.py (Pre-reservation & Post-settlement)
        - src/research_pipeline/extractors/llm_extractor.py (JSON Schema, 許可リスト保護)
        - src/research_pipeline/extractors/converter.py (C0〜C4 共通正規化 & C1 変換失敗分離)
        - tests/unit/test_budget_and_converter.py で検証

Step 5: 評価スクリプト & クラスターブートストラップの実装
        - src/research_pipeline/evaluation/evaluator.py (Hungarian 二部マッチング)
        - src/research_pipeline/evaluation/bootstrap.py (Document-level Cluster Bootstrap)

Step 6: Development Set によるアブレーション実験 (C0〜C4, R=3)
        - 失敗モード分離分析、プロンプト微調整、予算消費監視

Step 7: Holdout Pilot Set ブラインド評価 & 受入ゲート判定
        - 1 回限りのブラインド推論
        - 実装受入ゲート判定（合否）と小標本 95% 信頼区間付き結果報告
```

### 11.2 実装受入ゲート（パイロット検証合否）と科学的一般化判定の分離

#### A. 実装受入ゲート（パイロット合格基準）
本エンジニアリング実装が H1b への連携に耐えうるかを判断するための判定基準。Holdout Pilot Set（12件）において以下をすべて満たすこと：
1. **Quote Validity Rate ($R_{QV}$)**: **100.0%**（正規化後文字列に対する厳密コードポイント一致）。
2. **Claim Support Rate ($R_{CS}$)**: 独立人手判定において **90.0% 以上**。
3. **Observed Hallucination ($R_{HAL}$)**: パイロット標本において **0 件（$0.0\%$）**。
4. **Semantic Entity F1 ($F1_{\text{sem}}$)**: 人手意味的同値一対一マッチングにおいて **80.0% 以上**。
5. **Vocabulary Compliance Rate ($R_{VC}$)**: **95.0% 以上**（未記載時の `unknown` / `is_unspecified=True` 整合率 100%）。
6. **IO 状態整合性違反**: **0 件**（Pydantic バリデータ違反なし）。
7. **再現性 (Self-Consistency)**: 3 回反復試行での Jaccard 一致度が **90.0% 以上**。
8. **予算規律遵守**: 30.00 USD / 5,000,000 トークン超過なし。
9. **AlgoForge 無侵害**: AlgoForge のコード・DB に一切変更がないこと。

#### B. 科学的一般化性能の統計的報告要件
- パイロット結果は、必ず **「文書単位クラスターブートストラップ 95% 信頼区間」** および **「小標本 Wilson スコア区間」** を併記して報告する。
- 標本数 $N=12$ から得られた数値について、開かれた文献母集団全体に対する一般化性能を統計的に実証したと解釈・主張してはならない。本格的な一般化性能評価は、後続のスケールアップ評価フェーズ（大規模アノテーションデータ）にて実施することを明記する。

---
**Plan Prepared by:** Antigravity Agent  
**Awaiting Review:** H1a 科学的実験計画書 v0.4 の承認判定待ち
