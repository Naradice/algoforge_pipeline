# HDE H1a — Capability Extraction 科学的実験計画書 v0.5

**Status:** Proposed — Scientific Review Pending  
**Date:** 2026-11-15  
**Version:** 0.5  
**Target:** Hypothesis Discovery Engine (HDE), Phase 1 / Step H1a  
**Supersedes:** H1a Capability Extraction Plan v0.4

## 0. v0.5 変更概要

本改訂では、v0.4の科学的設計を維持しながら、実装・評価時の不整合を解消する。

| ID   | 修正内容                     | 目的                 |
| ---- | ------------------------ | ------------------ |
| R4-1 | トークン数と費用の保守的な事前予約        | 予算超過防止             |
| R5   | `EvaluationCandidate`の導入 | Evidenceなし条件の公平な比較 |
| R6   | Holdoutの反復評価規則を固定        | 評価の恣意性防止           |
| R7   | Evidenceのフィールドハッシュ検証     | 原典版との厳密な対応         |
| R8   | IO未指定属性の整合性強化            | 根拠のない補完の防止         |
| R9   | LocalRelation参照整合性の検証    | H1bへの不正なエッジ流入防止    |
| R10  | ID一意性と試行識別の強化            | 抽出結果の混同防止          |
| R11  | 人手評価のブラインド化              | 評価者バイアス低減          |
| R12  | API固有メタデータの任意化           | プロバイダ互換性確保         |

本計画はH1aの実装および科学的評価を対象とする。H1bの実装、仮説生成、実験の自動実行は対象外とする。

---

## 1. 研究目的

H0で収集・保存された`ResearchDocument`を入力として、以下の6種類のエンティティと文書内関係を抽出する。

1. **Phenomenon**：説明・予測・再現の対象となる現象
2. **Model**：現象を表現する数理的・計算的構造
3. **Method**：推定・学習・最適化・検証の具体的手順
4. **Capability**：モデルや手法によって実行可能な操作的機能
5. **Assumption**：成立条件・前提仮定
6. **Limitation**：適用制約・計算上の限界

さらに、エンティティ間の`LocalRelation`を抽出する。

研究上の中心課題は、**原典に存在する情報を、根拠を失わず、後続の知識統合と仮説生成に利用可能な形式へ変換できるか**である。

本段階では、論文の主張が実世界で真であることは検証しない。評価するのは、文書に記述された内容を正確に抽出できたかである。

### 1.1 検証する仮説

**H1：** 構造化出力、Evidence Grounding、Controlled Vocabularyの段階的導入により、自由形式LLM抽出と比較して、根拠付き知識抽出の品質が改善する。

**H2：** Evidence検証器は誤った抽出の一部を除去できるが、正しい抽出も棄却する可能性があり、PrecisionとRecallの間にトレードオフが生じる。

**H3：** 入出力の未指定状態を明示的に表現することで、原文にないCapability IO仕様の補完を減らせる。

これらは検証対象の仮説であり、事前に成立を仮定しない。

---

## 2. 抽出スキーマと責務分離

### 2.1 3種類のデータ表現

抽出処理では、次の3種類を区別する。

**RawExtractionArtifact**

LLMまたはRule-based抽出器の出力を、構文解析・スキーマ検証より前に保存する。

**EvaluationCandidate**

全実験条件C0〜C4を公平に比較するための中間表現。Evidenceを持たない候補も許容する。

**ExtractedKnowledgeRecord**

厳密なEvidence検証と構造検証に合格したエンティティを保存する。H1bへの引き渡し対象となる。

### 2.2 EvidenceSpan

```python
from pydantic import BaseModel, Field, model_validator
from typing import Literal

class EvidenceSpan(BaseModel):
    target_field_name: Literal[
        "abstract_text",
        "readme_text",
        "title_text",
    ]
    target_field_hash: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    verbatim_quote: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_indices(self):
        if self.end_char <= self.start_char:
            raise ValueError("Invalid evidence interval")
        return self
```

引用の正当性は、Pydantic単独ではなく、対象文書の正規化済みフィールドを取得する検証器で判定する。

```python
def validate_evidence(evidence, canonical_fields):
    text = canonical_fields[evidence.target_field_name]

    expected_hash = sha256(
        text.encode("utf-8")
    ).hexdigest()

    return (
        evidence.target_field_hash == expected_hash
        and evidence.end_char <= len(text)
        and text[
            evidence.start_char:evidence.end_char
        ] == evidence.verbatim_quote
    )
```

### 2.3 正規化仕様

引用対象の文字列は次の順序で正規化する。

1. 改行コードをLFへ統一する。
2. Unicode NFC正規化を適用する。
3. 文字列全体の先頭・末尾の空白を除去する。
4. 内部の空白・インデント・改行は変更しない。

文字位置は正規化後のPython Unicode Codepointインデックスとする。

正規化アルゴリズムのバージョンを保存し、変更時には旧版の引用位置を自動的に再解釈しない。

### 2.4 CapabilityIOの整合性

```python
class CapabilityIO(BaseModel):
    data_type_l1: str
    data_type_l2: str | None = None
    description: str | None = None
    time_resolution: str = "unspecified"
    dimensionality: str | None = None
    is_unspecified: bool = False
    evidence: EvidenceSpan | None = None

    @model_validator(mode="after")
    def validate_consistency(self):
        if self.is_unspecified:
            if self.data_type_l1 != "unknown":
                raise ValueError("Expected unknown L1 type")
            if self.data_type_l2 is not None:
                raise ValueError("L2 must be absent")
            if self.description is not None:
                raise ValueError("Description must be absent")
            if self.dimensionality is not None:
                raise ValueError("Dimensionality must be absent")
            if self.time_resolution != "unspecified":
                raise ValueError("Time resolution must be unspecified")
            if self.evidence is not None:
                raise ValueError("Evidence must be absent")
        else:
            if self.data_type_l1 == "unknown":
                raise ValueError("Explicit IO requires known L1")
            if self.evidence is None:
                raise ValueError("Explicit IO requires evidence")
        return self
```

L1の許容値はv0.4のControlled Vocabularyを維持し、実装では`Literal`またはEnumによって強制する。

なお、入出力自体は記載されていても、時間解像度や次元数が記載されていない場合がある。この場合、既知属性のみを記録し、未知属性を補完しない。

### 2.5 LocalRelationの整合性

`LocalRelation`について、以下を検証する。

- `src_local_id`と`dst_local_id`がAcceptedエンティティに存在する。
- 関係型に対する始点・終点のエンティティ種別が許容される。
- 関係を支持するEvidenceが存在する。
- 自己参照および重複関係を検出する。
- エンティティが棄却された場合、そのエンティティを参照する関係も棄却する。

棄却された関係には`dangling_reference`、`invalid_relation_type`等の理由を記録する。

---

## 3. Raw Extractionと共通評価表現

### 3.1 RawExtractionArtifact

v0.4で定義した次の情報を維持する。

- `extraction_run_id`
- `source_id`
- `version_id`
- `condition_id`
- `raw_response_text`
- `http_status`
- `token_usage`
- `executed_at`
- `model_checkpoint`
- `prompt_hash`
- `target_field_hashes`
- `json_parse_status`
- `parse_error`
- `parsed_candidates`

生応答は構文検証前に独立して保存する。

保存失敗時には抽出処理を成功扱いにしない。

### 3.2 EvaluationCandidate

```python
class EvaluationCandidate(BaseModel):
    candidate_local_id: str
    entity_type: Literal[
        "phenomenon",
        "model",
        "method",
        "capability",
        "assumption",
        "limitation",
    ]
    name_or_statement: str
    evidence: EvidenceSpan | None = None
    condition_id: Literal["C0", "C1", "C2", "C3", "C4"]
    extraction_run_id: str
    conversion_status: Literal[
        "success",
        "partial",
        "failed",
    ]
```

C0〜C2ではEvidenceが存在しなくても正常な評価候補として扱う。

ただし、Evidenceを要求するC3/C4でEvidenceが欠落した場合は、`missing_evidence`として記録する。

### 3.3 評価指標の適用範囲

| 指標                               | C0〜C2            | C3〜C4          |
| -------------------------------- | ---------------- | -------------- |
| Semantic Precision / Recall / F1 | 評価               | 評価             |
| Conversion Failure Rate          | 評価               | 評価             |
| Quote Validity                   | N/A              | 評価             |
| Claim Support                    | 原文に対する主張の支持を別途評価 | Evidenceに基づき評価 |
| Overclaim Rate                   | 評価               | 評価             |
| Vocabulary Compliance            | N/Aまたは参考値        | 評価             |
| H1b Accepted適格性                  | 対象外              | 評価             |

Semantic F1は変換成功例だけでなく、全対象文書を分母に含める。変換失敗文書の予測エンティティ集合は空集合として扱う。

変換成功文書のみのF1は診断用の補助指標とする。

---

## 4. Evidence検証と3態保存

### 4.1 検証段階

**Stage 1：Quote Validity**

正規化フィールドハッシュ、オフセット、引用文字列の完全一致を検証する。

**Stage 2：Claim Support**

引用が抽出された主張を支持するか検証する。

**Stage 3：Claim Completeness**

否定、条件、ヘッジ、適用制限の省略による過剰主張を検出する。

**Stage 4：Structural Consistency**

IO状態、エンティティID、関係参照、語彙整合性を検証する。

自動LLM/NLI検証器は運用上のスクリーニングに使用するが、科学的評価のGround Truthには使用しない。

### 4.2 3態保存

- **Raw**：検証前の応答と候補。
- **Accepted**：すべての必須検証に合格した候補。
- **Rejected**：棄却された候補とその理由。

棄却された候補を削除せず、再評価できる状態で保持する。

検証器の有効性はRawとAcceptedのPrecision、Recall、Coverageの差分で評価する。

---

## 5. Controlled Vocabulary

v0.4の二層語彙を維持する。

### 5.1 L1 Universal Types

- `timeseries_1d`
- `timeseries_multivariate`
- `tabular_record`
- `graph_network`
- `discrete_sequence`
- `probability_distribution`
- `parameter_vector`
- `scalar_value`
- `event_stream`
- `unknown`

### 5.2 L2 Domain Specializations

L2は金融・物理・生物・情報科学などのドメイン固有型を表す。

L2の未知語を発見した場合、勝手に既存語彙へ置換せず、原文表現と候補型を記録する。

語彙体系の拡張はDevelopment Setの評価期間中に行い、Holdout評価開始前に固定する。

---

## 6. Gold Standard Dataset

### 6.1 データセット構成

| 分割            | 文書数    | 用途               |
| ------------- | ------ | ---------------- |
| Development   | 10     | プロンプト・変換器・検証器の調整 |
| Holdout Pilot | 12     | 最終実装受入評価         |
| Challenge     | 8      | 否定・ヘッジ・欠落等の診断    |
| **合計**        | **30** | <br>             |

Developmentは金融論文4件、他分野論文3件、OSS README3件とする。

Holdoutは金融論文4件、他分野論文4件、OSS README4件とする。

ChallengeはREADME欠落1件、否定・仮説棄却2件、ヘッジ2件、IO未記載2件、過剰主張1件を基本構成とする。

### 6.2 Cross-Split Isolation

arXivではBase IDを単位に分割する。

GitHubでは`repository_id`、Root Parent Repository ID、READMEコンテンツハッシュを用いて重複を検出する。

同一テーマの派生論文や近似重複READMEは、人手で確認して分割間の漏洩を防止する。

### 6.3 アノテーション

2名の独立したアノテーターが、6概念のマルチラベル分類とエンティティ単位の正解を作成する。

各概念についてCohen's Kappaを計算する。

受入目標：

- 各クラスのKappa：0.70以上
- マクロ平均Kappa：0.75以上

希少クラスでKappaが定義不能となる場合は、無理に数値化せず、陽性件数、観測一致率、不一致内容を報告する。

不一致はAdjudicatorが裁定する。

---

## 7. 比較実験 C0〜C4

| 条件 | 内容                                            |
| -- | --------------------------------------------- |
| C0 | Rule-based Baseline                           |
| C1 | Free-text LLM                                 |
| C2 | Structured Output Only                        |
| C3 | Structured Output + Evidence Grounding        |
| C4 | C3 + Controlled Vocabulary + Unknown Handling |

同一の入力文書版、同一の対象フィールドを使用する。

LLM条件間では、可能な限り同一モデル・生成設定を使用する。

C1の非構造化応答は、固定された決定論的コンバータで評価形式へ変換する。

変換失敗は別途記録するが、エンドツーエンドF1から除外しない。

### 7.1 アブレーションの解釈

- C1とC2：構造化出力の影響
- C2とC3：Evidence要求の影響
- C3とC4：Controlled Vocabularyの影響
- C0とLLM条件：Rule-based方式との差

これらは複数の要素が同時に変化する場合があるため、厳密な単一因子の因果効果と断定しない。

---

## 8. 評価方法

### 8.1 Semantic Entity Matching

GoldとPredictionの間に意味的同値関係を定義し、一対一最大二部マッチングを行う。

同一概念の重複抽出は複数のTrue Positiveとして数えない。

### 8.2 主評価指標

- Semantic Precision
- Semantic Recall
- Semantic F1

### 8.3 補助指標

- Quote Validity Rate
- Claim Support Rate
- Overclaim Rate
- Observed Hallucination Rate
- Vocabulary Compliance Rate
- Conversion Failure Rate
- Capability Coverage
- Raw-to-Accepted Precision / Recall差分
- Self-Consistency

空集合などで分母が0となる場合の取り扱いは評価コードで事前固定する。該当例数とともにN/Aを報告し、都合のよい値を代入しない。

### 8.4 人手評価のブラインド化

人手評価者には、可能な範囲で以下を伏せる。

- C0〜C4の条件ID
- 使用モデル名
- 抽出試行番号
- 自動検証器の合否

候補の提示順序は固定Seedでランダム化する。

Claim Support判定では、引用と必要な前後文脈を提示する。

### 8.5 信頼区間

主信頼区間は文書単位クラスターブートストラップとする。

- リサンプリング単位：文書
- 反復数：1,000
- 区間：2.5〜97.5パーセンタイル

比率指標にはWilson区間を補助的に併記する。ただし、エンティティ間の独立性を仮定したWilson区間は参考値であり、主CIとして解釈しない。

Holdoutは12文書の小標本であるため、母集団への一般化性能を実証したと主張しない。

---

## 9. LLM実行と再現性

### 9.1 モデル

Primary候補：

`gemini-2.5-flash`

Verification候補：

`gemini-2.5-pro`

実際の実験開始前に、利用可能なモデルID、バージョン固定の可否、構造化出力対応、料金を確認する。

特定のモデルチェックポイントが提供されない場合は、固定できないことを明記し、リクエストしたモデルIDとAPIが返すモデル情報を保存する。

`system_fingerprint`等の情報は、APIが提供する場合のみ保存する。

### 9.2 Development Set

C0〜C4を評価する。

LLM条件C1〜C4については、各条件3反復を実施する。

生成パラメータは以下を基本とする。

- Temperature：0.0
- Seed：42（API対応時）
- 同一プロンプトテンプレート版
- 同一入力文書版

試行ごとの抽出集合のJaccard類似度とSemantic F1を報告する。

### 9.3 Holdout Pilot Set

Holdout評価では、Developmentで固定した最終C4を**3回実行する**。

**Trial 1を主性能評価として事前指定する。**

Trial 2、Trial 3は再現性の測定に使用する。

3試行の中から最も良い結果を選択して主性能とすることは禁止する。

Holdoutの結果を確認した後にプロンプト、語彙、検証閾値を変更した場合、同じHoldoutを新しい独立評価として再利用しない。

### 9.4 モデル移行

モデル廃止時はDevelopment Setで移行試験を行う。

比較指標：

- Entity Jaccard
- Semantic F1差分
- Quote Validity
- Claim Support
- Conversion Failure

v0.4のJaccard 90%以上、F1差2ポイント以内を移行時の目標として維持する。

ただし、移行により科学的比較条件が変化した場合は、新しい実験系列として記録する。

---

## 10. 予算管理

### 10.1 上限

- 最大費用：30.00 USD
- 最大トークン数：5,000,000

いずれかの上限に到達する予約は許可しない。

すべての抽出、再試行、検証用LLM呼び出しを同一の予算管理対象とする。

### 10.2 事前予約

API呼び出し前に、入力トークン数と最大出力トークン数を別々に見積もる。

```python
estimated_input_tokens = count_input_tokens(
    model=model_id,
    request=request,
)

reserved_output_tokens = max_output_tokens

reserved_tokens = (
    estimated_input_tokens
    + reserved_output_tokens
    + provider_specific_token_allowance
)

reserved_cost = estimate_worst_case_cost(
    model=model_id,
    input_tokens=estimated_input_tokens,
    output_tokens=reserved_output_tokens,
    additional_tokens=provider_specific_token_allowance,
)
```

`len(prompt)`をトークン数として使用することは禁止する。

プロバイダの計数APIまたは対応トークナイザを利用し、誤差がある場合は保守的な上限を予約する。

モデルごとの料金区分と追加課金項目を反映する。

### 10.3 予約判定

```python
if spent_usd + reserved_usd + new_cost > 30.00:
    raise BudgetExceeded()

if spent_tokens + reserved_tokens + new_tokens > 5_000_000:
    raise BudgetExceeded()
```

予約と実績の更新は原子的に行い、同時実行による二重予約を防止する。

### 10.4 事後精算

API応答から実際の使用量を取得できた場合、予約額と実績額を精算する。

使用量が不明な場合は、予約額を即時解除せず、`unsettled`状態で保持する。

APIの課金仕様によって厳密な上限保証が不可能な場合、保守的な安全余裕を設け、保証できない部分を報告する。

### 10.5 安全停止テスト

以下を必須テストとする。

- 予算上限直前の呼び出し拒否
- 複数同時予約
- APIタイムアウト
- リトライ時の二重計上防止
- 使用量不明時の予約保持
- 検証用LLMの予算算入
- プロセス再起動後の残予算復元

---

## 11. ID体系とH1b引き渡し

### 11.1 Candidate Entity ID

```python
candidate_entity_id = (
    f"{source_id}:"
    f"{version_hash}:"
    f"{extraction_run_id}:"
    f"{entity_type}:"
    f"{local_id}"
)
```

`version_hash`は原則として完全なSHA-256値を使用し、短縮表示はUI用途に限定する。

`extraction_run_id`は条件、試行、実行を識別できる一意なIDとする。

同一試行内で`local_id`が重複する場合はスキーマ違反として棄却する。

### 11.2 Candidate Edge ID

```python
candidate_edge_id = (
    f"{src_candidate_id}->"
    f"{relation_type}->"
    f"{dst_candidate_id}"
)
```

複数のEvidenceを持つ同種関係が存在する場合は、関係インスタンスIDを追加して衝突を回避する。

### 11.3 H1bとの責務分離

H1aは候補エンティティと候補関係を出力する。

H1bは複数文書間の名寄せを担当し、Canonical Entity IDを付与する。

H1aではCanonical IDを生成しない。

H1bへ引き渡すのはAcceptedエンティティと、参照整合性に合格したAccepted関係のみとする。

RawおよびRejectedは監査参照として保持する。

---

## 12. 機密情報と監査

保存する生成パラメータは許可リスト方式で制限する。

許可する主な項目：

- model_checkpoint
- temperature
- top_p
- top_k
- seed
- max_output_tokens
- response_mime_type

HTTP認証ヘッダー、APIキー、環境変数全体、クライアント内部設定は保存しない。

ログ出力には防御的マスクを適用する。

なお、Raw LLM応答には文書由来の情報が含まれるため、保存先のアクセス制御と保持方針も定義する。

---

## 13. 実装・実験手順

### Step 1：Gold Standard Dataset構築

30文書を選定し、分割間の重複を検査する。

2名による独立アノテーションと裁定を実施する。

Holdoutの正解データは評価担当者以外から隔離する。

### Step 2：スキーマ実装

`EvidenceSpan`、`CapabilityIO`、`LocalRelation`、`ExtractedKnowledgeRecord`を実装する。

正常系・異常系の単体テストを作成する。

### Step 3：Raw保存・評価中間表現

`RawExtractionArtifact`と`EvaluationCandidate`を実装する。

不正JSON、欠落フィールド、部分変換を監査可能にする。

### Step 4：Evidence検証器

Quote Validity、Claim Support、Claim Completeness、Structural Consistencyを実装する。

棄却理由を保存する。

### Step 5：LLM抽出器・予算管理

C0〜C4の抽出器とコンバータを実装する。

事前予約・事後精算型Budget Trackerを実装する。

### Step 6：評価基盤

Semantic Matching、Precision、Recall、F1、Cluster Bootstrap、Kappa、Self-Consistencyを実装する。

### Step 7：Development実験

C0〜C4を比較し、必要な調整を行う。

Holdout実行前にモデル、プロンプト、語彙、評価コード、閾値を固定する。

### Step 8：Holdout Pilot評価

C4を3反復する。

Trial 1を主性能、3試行全体を再現性評価に使用する。

結果確認後の調整を禁止する。

### Step 9：科学的評価報告

以下を報告する。

- 各条件のSemantic F1
- RawとAcceptedの差
- 検証器によるPrecision / Recall変化
- Capability IOの未知属性率
- Quote Validity
- Claim Support
- Overclaim
- Hallucination
- 文書別失敗事例
- 95%信頼区間
- API費用・トークン数
- H1b引き渡し可能なAcceptedレコード数

---

## 14. 実装受入基準

Holdout Pilot SetのTrial 1について、以下を評価する。

| 指標                     | 受入基準  |
| ---------------------- | ----- |
| Quote Validity         | 100%  |
| Claim Support          | 90%以上 |
| Observed Hallucination | 0件    |
| Semantic Entity F1     | 80%以上 |
| Vocabulary Compliance  | 95%以上 |
| IO未指定状態の整合率            | 100%  |
| IO状態整合性違反              | 0件    |
| LocalRelation参照整合性違反   | 0件    |
| Budget Overrun         | 0件    |
| AlgoForgeへの変更          | 0件    |

再現性はHoldoutの3試行について、意味的に正規化されたエンティティ集合のペアワイズJaccard平均90%以上を受入基準とする。

抽出順序や`local_id`の差だけで不一致と判定しない。

### 14.1 合否の解釈

本基準は**H1aの初期実装受入基準**であり、科学的一般化性能の証明ではない。

Holdoutの小標本で達成した数値を、未知の文献母集団に対する保証として扱わない。

失敗した場合は、失敗原因を分析して次の実験計画を立案する。同じHoldoutを繰り返し利用して合格するまで調整することは禁止する。

---

## 15. 実験開始前の必須確認

実装担当Agentは、実験開始前に次の事項を報告する。

1. H0の入力データ契約との互換性。
2. Gold Standardの分割と隔離の完了。
3. スキーマ・引用検証・ID整合性テストの結果。
4. C0〜C4共通評価コンバータのテスト結果。
5. Budget Trackerの障害系テスト結果。
6. 実際に利用するモデルID、料金、トークン計数方式。
7. 予定API呼び出し数と最悪費用見積もり。
8. Holdout実行前の設定固定方法。

**実験の実行には、科学的レビューによる明示的な承認を必要とする。**

---

**Plan Status:** Proposed v0.5  
**Prepared for:** HDE Scientific Review  
**Execution Authorization:** Not granted  
**Next Action:** v0.5の設計レビューと承認判定
