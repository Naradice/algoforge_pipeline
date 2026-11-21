# StockNet — Hypothesis Discovery Engine 詳細設計仕様書

**Version:** 0.2  
**Status:** Draft  
**Date:** 2026-11-15  
**Parent Specification:** StockNet FXトレード戦略探索システム 統合設計仕様 v0.3  
**Study Manager Root Goal:** #774「FXのトレード戦略の探索」

---

# 1. 概要

## 1.1 目的

Hypothesis Discovery Engine（以下HDE）は、AI Agentが既存研究、論文、OSS、実験結果を調査し、新しい市場仮説と研究方向を継続的に発見するためのサブシステムである。

本システムは、単純な取引戦略の生成やパラメータ最適化ではなく、**市場現象を説明・予測する新しい研究仮説の発見**を目的とする。

既存の研究成果を知識として蓄積し、異なる研究分野のモデル・アルゴリズム・実験手法を組み合わせることで、従来の戦略探索では発見しにくい研究方向を生成する。

## 1.2 設計原則

1. **Hypothesis First:** 取引戦略より先に研究仮説を立案する。
2. **Evidence Grounding:** 研究上の主張には追跡可能な出典を持たせる。
3. **Capability-Centric:** 論文の分類だけでなく、研究が提供する能力を抽出する。
4. **Cross-Domain Discovery:** 金融以外の研究分野も探索する。
5. **Falsifiability:** 仮説は観測可能な予測と反証条件を持つ。
6. **Knowledge Accumulation:** 成功・失敗の双方を知識として蓄積する。
7. **Reproducibility:** 検索・抽出・推論の過程を追跡できる。
8. **Separation of Concerns:** 資料収集、知識抽出、仮説生成、実験実行を分離する。
9. **Controlled Autonomy:** AIは研究方向を提案するが、実験実行には別途承認を必要とする。

## 1.3 スコープ

HDEが担当する処理：

- 研究資料の検索計画
- 論文・OSSの検索
- 研究情報の構造化
- Capability抽出
- Knowledge Graphの構築・更新
- Entity Resolution
- 研究間の関係探索
- 新しい仮説の生成
- 仮説の重複検出
- 仮説ポートフォリオの構築
- Study Managerへの研究提案
- 探索履歴と失敗履歴の管理

HDEが直接担当しない処理：

- バックテストの実行
- MLモデルの学習
- 実取引
- 実験の最終承認
- AlgoForgeの本番コード変更

これらはResearch Orchestrator、Experiment Designer、AlgoForgeなどが担当する。

---

# 2. システム構成

## 2.1 論理アーキテクチャ

```text
                    Research Goal
              FXのトレード戦略の探索
                           |
                           v
                 Research Orchestrator
                           |
            +--------------+--------------+
            |                             |
            v                             v
    Continuous Discovery          Problem-Driven Discovery
            |                             |
            +--------------+--------------+
                           |
                           v
                  Search Planner
                           |
                           v
               Research Collection Layer
                 /                 \
                v                   v
         Paper Collector       OSS Collector
                \                   /
                 +--------+--------+
                          |
                          v
                  Source Repository
                          |
                          v
                Capability Extraction
                          |
                          v
                  Entity Resolution
                          |
                          v
                 Research Knowledge Graph
                          |
                          v
                  Operator Scheduler
                          |
                          v
                 Hypothesis Generator
                          |
                          v
                    Review Gate
                          |
                          v
                 Hypothesis Portfolio
                          |
                          v
                    Study Manager
```

## 2.2 コンポーネント

| コンポーネント | 責務 |
|---|---|
| Search Planner | 検索対象・検索クエリ・探索予算の決定 |
| Collection Layer | 資料取得、定期実行、更新検知 |
| Source Repository | 出典、取得内容、版、ハッシュの保存 |
| Capability Extractor | 研究から利用可能な能力を抽出 |
| Entity Resolver | 同一概念・包含関係・類似概念の識別 |
| Knowledge Graph | 研究知識と関係の保存 |
| Operator Scheduler | 探索操作の適用条件・順序・予算管理 |
| Hypothesis Generator | 検証可能な研究仮説の生成 |
| Research Critic | 科学的妥当性と実現可能性の確認 |
| Portfolio Manager | 仮説の多様性と研究優先度の管理 |
| Study Manager Adapter | 研究ツリー・仮説・レビューとの連携 |

---

# 3. 研究探索モード

## 3.1 Continuous Discovery

継続的研究探索は、現在進行中の実験に依存せず、新しい研究方向を発見する。

探索対象は金融分野に限定しない。

対象例：

- エージェントベース市場モデル
- 複雑系・統計物理
- 情報理論
- 確率過程
- 状態空間モデル
- Simulation-Based Inference
- 異常検知
- 表現学習
- 制御理論
- 因果推論

探索はKnowledge Graph上の未探索領域を起点とする。

定期的に研究資料を収集し、新しいCapabilityや研究間の関係を発見する。

## 3.2 Problem-Driven Discovery

課題駆動探索は、実験結果や研究上の未解決問題を起点とする。

主な入力：

- Study ManagerのFinding
- 実験結果の批判的レビュー
- モデルの汎化失敗
- パラメータ識別の失敗
- 統計的な曖昧さ
- 既存実装の機能不足
- 研究系列の停滞

問題を構造化したうえで、関連する研究手法や代替モデルを探索する。

## 3.3 探索予算

Continuous DiscoveryとProblem-Driven Discoveryには独立した予算を設定する。

初期比較では同程度の予算を割り当てる。

記録する資源：

- LLMトークン
- 検索回数
- 取得資料数
- Agent実行時間
- 計算資源
- 生成候補数

予算配分は後続の評価実験によって見直す。

---

# 4. Research Collection Layer

## 4.1 基本方針

AlgoForgeには金融レポートを定期的に取得する仕組みが存在する。

HDEでは、この収集基盤と同様の仕組みを利用して論文・OSSを取得する。

既存のスケジューラ、取得アダプタ、リトライ、保存、更新検知などを可能な範囲で再利用する。

ただし、既存実装の詳細は未調査であるため、再利用可能な機能はH0で確認する。

## 4.2 Paper Collector

対象：

- 論文
- プレプリント
- 技術報告
- 公式実装への参照
- 引用・関連研究情報

取得情報：

```json
{
  "source_type": "paper",
  "title": "...",
  "authors": [],
  "doi": null,
  "arxiv_id": null,
  "url": "...",
  "published_at": null,
  "updated_at": null,
  "retrieved_at": "...",
  "content_hash": "...",
  "license": null
}
```

一次資料を優先する。

## 4.3 OSS Collector

対象：

- GitHub等の公開リポジトリ
- README
- ドキュメント
- リリース
- 実装例
- テスト・再現手順

取得情報：

- Repository URL
- Commit SHA
- Release Version
- License
- 依存関係
- 実装言語
- 実行条件
- 最終更新日時

論文で主張された能力と、公開実装で実際に利用できる能力を区別する。

## 4.4 更新検知と重複排除

同一資料の再取得によって重複処理が発生しないようにする。

識別子の優先順位：

1. DOI、arXiv IDなどの永続識別子
2. Repository URLとCommit SHA
3. 正規化URL
4. コンテンツハッシュ
5. タイトル・著者などによる補助判定

異なるバージョンの論文やOSSは、同一資料の異なる版として保持する。

## 4.5 検索計画

検索クエリはAgentが生成する。

資料取得、キャッシュ、更新検知、実行履歴は決定論的なPython処理で管理する。

これにより、Coding AgentとPython Agentの切替時にも同じ資料を再利用できる。

---

# 5. Research Knowledge Graph

## 5.1 ノード

| Type | 内容 |
|---|---|
| Source | 論文・OSS・技術報告 |
| Phenomenon | 市場現象・統計的性質 |
| Model | 数学モデル・シミュレータ |
| Method | 推定・学習・検証手法 |
| Capability | モデルや手法が提供する能力 |
| Assumption | 成立に必要な仮定 |
| Limitation | 適用上の制約 |
| Hypothesis | 研究仮説 |
| FindingRef | Study Managerの知見への参照 |

## 5.2 エッジ

主な関係：

- `PROVIDES`
- `REQUIRES`
- `PRODUCES`
- `CONSUMES`
- `EXPLAINS`
- `CONTRADICTS`
- `EXTENDS`
- `IS_A`
- `SAME_AS`
- `RELATED_TO`
- `DISTINCT_FROM`
- `TESTED_BY`
- `SUPPORTED_BY`
- `REFUTED_BY`

関係には出典、根拠状態、確信度を付与する。

## 5.3 根拠状態

| Status | 意味 |
|---|---|
| reported | 出典で主張されている |
| reproduced | 独立に再現された |
| inferred | 証拠から推論された |
| proposed | 新しい未検証の提案 |

`reported`は科学的な正しさを保証しない。

`proposed`は既存研究の実証結果と混同しない。

---

# 6. Capability Data Model

## 6.1 定義

Capabilityは、モデルや手法が提供する操作可能な能力を表す。

例：

- パラメータから価格系列を生成する
- 観測系列から潜在状態を推定する
- 時系列の異常を検出する
- 複数の時系列を比較する
- 将来の確率分布を予測する

## 6.2 スキーマ

```json
{
  "capability_id": "cap_market_simulation",
  "name": "Parameterized Market Simulation",
  "operation": "generate",
  "inputs": [
    {
      "type": "agent_parameters"
    }
  ],
  "outputs": [
    {
      "type": "price_timeseries"
    }
  ],
  "preconditions": [],
  "assumptions": [],
  "limitations": [],
  "evidence": [],
  "implementation": {
    "repository": null,
    "reproducibility_status": "unverified",
    "implementation_gap_level": 3
  }
}
```

## 6.3 Controlled Vocabulary

Capabilityの入出力型は自由記述にしない。

初期段階では、標準データ型語彙を定義する。

例：

- `price_timeseries`
- `ohlcv_timeseries`
- `tick_timeseries`
- `orderbook_snapshot`
- `trade_events`
- `agent_parameters`
- `latent_state`
- `probability_distribution`
- `prediction_score`
- `anomaly_score`
- `simulation_trajectory`

型には必要に応じて以下の属性を持たせる。

- 時間解像度
- 単位
- 通貨ペア
- 観測対象
- 時間範囲
- 利用可能時刻
- 欠損条件
- 確率分布の表現

標準語彙の最終的な種類と数は、実際の抽出結果から調整する。

---

# 7. Entity Resolution

## 7.1 目的

表記揺れ、同義語、概念の包含関係によってKnowledge Graphが分断されることを防ぐ。

## 7.2 正規化パイプライン

```text
Extracted Entity
       |
       v
Name Normalization
       |
       v
Exact Identifier Matching
       |
       v
Vector Similarity Search
       |
       v
Top-K Candidates
       |
       v
LLM / Rule-Based Relation Classification
       |
       v
Merge / Link / Keep Separate / Defer
```

## 7.3 判定結果

| Relation | 処理 |
|---|---|
| SAME_AS | 同一概念として統合可能 |
| IS_A | 上位・下位関係を登録 |
| RELATED_TO | 関連エッジを追加 |
| DISTINCT_FROM | 別概念として保持 |
| UNCERTAIN | 判定を保留 |

誤統合は後続の探索に大きな影響を与えるため、確信できない場合は統合しない。

## 7.4 統合履歴

統合時には以下を記録する。

- 統合前のID
- 統合後のCanonical ID
- 判定理由
- 使用したモデル・ルール
- 根拠
- 判定日時

誤統合を修正できるように、元の抽出記録を保持する。

---

# 8. Discovery Operators

## 8.1 操作一覧

| Operator | 役割 |
|---|---|
| Expand | 隣接分野へ探索を拡張 |
| Compose | 複数Capabilityを組み合わせる |
| Transfer | 他分野の手法を移転する |
| Invert | 生成・予測モデルを逆問題として利用する |
| Challenge | 仮定・説明・結論を疑う |
| Generalize | 特定の発見を一般化する |

## 8.2 操作の層別化

探索操作を大きく二つに分ける。

**Knowledge Expansion**

- Expand
- Transfer

**Hypothesis Transformation**

- Compose
- Invert
- Challenge
- Generalize

ただし、実行順序は固定しない。

現在のFrontier、Knowledge Graphの状態、研究課題に応じて次の操作を選択する。

## 8.3 Operator Eligibility

各操作には適用条件を定義する。

例：

Composeの場合：

- 入出力型が接続可能か
- 時間解像度が整合するか
- 必要データが存在するか
- 前提条件が矛盾しないか
- 計算量が許容範囲か
- 既存の組み合わせと重複しないか

## 8.4 Negative Constraints

明確に不適切な組み合わせは、LLMによる詳細検討前に除外する。

対象：

- 型の不一致
- 単位の不一致
- 利用可能時刻の矛盾
- 因果的な未来参照
- 必須データの欠落
- 計算資源の超過

ただし、変換や追加観測によって解消できる不一致は、完全に棄却せず保留できるようにする。

## 8.5 探索予算制御

制限する項目：

- 探索深度
- Frontier数
- Operator適用回数
- 候補生成数
- LLMトークン
- 検索回数
- 実行時間

同一組み合わせの再試行は、根拠や条件に変化がある場合に限定する。

---

# 9. Implementation Gap

## 9.1 定義

Capabilityや仮説を実際の研究環境で検証するために必要な実装コストを分類する。

| Level | 定義 |
|---|---|
| L1 | 既存AlgoForge機能のみで実行可能 |
| L2 | OSS導入または小規模な拡張が必要 |
| L3 | 新規モデルや大規模実装が必要 |

## 9.2 運用

L1は通常の実験候補とする。

L2は実装見積もりを行ったうえで採用を判断する。

L3は原則としてKnowledge Graphに保持し、初期実験では優先度を下げる。

ただし、簡略化した概念実証によって重要な仮説を検証できる場合は例外とする。

実装難度と科学的価値は独立して評価する。

---

# 10. Problem Triage

## 10.1 目的

単純な実装上の問題によって、不要な研究探索が大量に発生することを防ぐ。

## 10.2 分類

| Type | 内容 | 処理 |
|---|---|---|
| Engineering Failure | バグ、設定ミス、環境障害 | 修正タスクへ |
| Scientific Failure | 仮説の反証、識別不能、仮定の破綻 | HDEへ |
| Capability Gap | 必要な機能・手法がない | 実装方法を探索 |
| Ambiguous Failure | 原因を特定できない | 診断を優先 |
| Repeated Engineering Blocker | 既存技術では解消困難 | HDEへの昇格を検討 |

## 10.3 Problem Record

```json
{
  "problem_id": "problem_001",
  "source_experiment_id": 123,
  "failure_type": "scientific_failure",
  "observed_result": "...",
  "expected_result": "...",
  "candidate_explanations": [],
  "missing_evidence": [],
  "constraints": [],
  "triage_status": "ready_for_discovery"
}
```

観測事実と原因の推測を分離する。

---

# 11. Hypothesis Generation

## 11.1 生成手順

1. 研究対象の現象を定義する。
2. 候補メカニズムを特定する。
3. 観測可能な予測を導出する。
4. 帰無仮説・代替説明を定義する。
5. 反証条件を明確にする。
6. 必要データと実験方法を検討する。
7. 既存研究との重複を確認する。
8. 実装ギャップを評価する。
9. 仮説候補として保存する。

## 11.2 Hypothesis Schema

```json
{
  "title": "...",
  "research_question": "...",
  "mechanism": "...",
  "predictions": [],
  "null_hypothesis": "...",
  "alternative_explanations": [],
  "falsification_conditions": [],
  "required_data": [],
  "required_capabilities": [],
  "related_sources": [],
  "implementation_gap_level": 1,
  "potential_trading_application": "...",
  "known_limitations": []
}
```

## 11.3 状態

- Candidate
- Grounded
- Testable
- Proposed

Candidateは着想段階。

Groundedは根拠を確認した状態。

Testableは反証可能な実験案がある状態。

ProposedはStudy Managerへの研究提案対象となった状態。

これらはHDE内部の状態であり、Study Managerの正式なステータスとは区別する。

---

# 12. Review Gate

仮説候補に対して以下を評価する。

- 根拠の妥当性
- 既存研究との差異
- 反証可能性
- 識別可能性
- 必要データの入手性
- 実装ギャップ
- 計算コスト
- 研究上の情報価値
- 最上位Goalとの関係

レビューは仮説生成とは独立した処理として実施する。

レビューに失敗した候補も、理由とともに保存する。

---

# 13. Hypothesis Portfolio

## 13.1 目的

研究候補を単一スコアで順位付けせず、多様性を維持する。

## 13.2 初期探索枠

| 枠 | 初期配分 |
|---|---:|
| 新しい研究方向 | 40% |
| 既存研究の発展 | 30% |
| 反証・代替説明 | 20% |
| 高リスク探索 | 10% |

これらは比較実験用の仮設定であり、確定した最適配分ではない。

## 13.3 類似性評価

以下の構造情報を比較する。

- 対象現象
- メカニズム
- 使用モデル
- 使用手法
- 観測量
- 予測
- 反証条件
- Capability

文章Embeddingだけによる重複判定は行わない。

---

# 14. 探索失敗履歴

## 14.1 目的

探索で失敗した組み合わせや研究方向を、次の探索で再利用する。

## 14.2 記録対象

- 型不一致
- 時間解像度の不一致
- 仮定の矛盾
- データ不足
- パラメータ識別不能
- 計算コスト超過
- 既存仮説との重複
- 再現実験の失敗

## 14.3 再評価

失敗した候補を永久に除外しない。

次の条件が変化した場合、再評価できる。

- 新しいデータの追加
- AlgoForgeの機能拡張
- 新しい研究手法の発見
- 計算資源の増加
- 既存仮説に対する新しいFinding

---

# 15. データ永続化

## 15.1 初期データベース

初期実装ではSQLiteを使用する。

主要テーブル：

```text
nodes
edges
sources
discovery_runs
```

## 15.2 nodes

```text
id
type
canonical_name
payload_json
created_at
updated_at
```

## 15.3 edges

```text
id
src_id
dst_id
relation_type
evidence_status
source_id
confidence
payload_json
created_at
```

## 15.4 sources

```text
id
source_type
canonical_identifier
url
version
content_hash
metadata_json
retrieved_at
```

## 15.5 discovery_runs

```text
id
mode
root_goal_id
query_log_json
operator_log_json
agent_backend
model_version
token_usage
status
started_at
completed_at
```

探索失敗履歴は初期段階では`discovery_runs`内の構造化ログとして保持し、検索・再評価要件に応じて独立テーブル化する。

---

# 16. Agent Interface

## 16.1 DiscoveryRequest

```python
from dataclasses import dataclass
from enum import Enum


class DiscoveryMode(str, Enum):
    CONTINUOUS = "continuous"
    PROBLEM_DRIVEN = "problem_driven"


@dataclass
class DiscoveryRequest:
    root_goal_id: int
    mode: DiscoveryMode
    problem_id: str | None
    max_sources: int
    max_hypotheses: int
    token_budget: int
```

## 16.2 DiscoveryResult

```python
@dataclass
class DiscoveryResult:
    hypothesis_ids: list[str]
    capability_ids: list[str]
    source_ids: list[str]
    consumed_tokens: int
    stop_reason: str
```

## 16.3 Agent Backend

```python
class AgentBackend:
    async def execute(
        self,
        task: dict,
        context: dict,
    ) -> dict:
        ...
```

Coding Agent、Python Agent、Local LLMの実行方式を切り替え可能にする。

Agentの役割と実行Backendは分離する。

---

# 17. Study Manager連携

## 17.1 Root Goal

登録済みの最上位Goal：

**#774「FXのトレード戦略の探索」**

HDEはこのGoalに関連する研究仮説を生成する。

## 17.2 登録方針

すべてのCandidateをStudy Managerへ登録しない。

Grounding、重複検査、Review Gateを通過した候補を登録対象とする。

## 17.3 研究サイクル

```text
Discovery
    |
Hypothesis Proposal
    |
Study Manager
    |
Experiment Planning
    |
Plan Review
    |
Approved Experiment
    |
AlgoForge / Scientific Execution
    |
Finding
    |
Problem Triage
    |
Next Discovery
```

HDEは研究提案までを担当する。

実験実行はOrchestratorが管理し、承認された計画に従う。

---

# 18. 評価・比較実験

## 18.1 比較対象

| 条件 | 探索方式 |
|---|---|
| A | LLMによる直接仮説生成 |
| B | 論文検索・要約からの生成 |
| C | Capability抽出と単一研究の応用 |
| D | Capability抽出とCross-Domain Composition |
| E | DにProblem-Driven Discoveryを追加 |

## 18.2 統制条件

- 同一LLM
- 同一初期情報
- 同一検索予算
- 同一トークン予算
- 同一評価基準
- 複数独立試行

課題駆動探索を比較する際は、各条件に同じ問題情報を与えるなど、入力情報量の差を管理する。

## 18.3 評価指標

- Grounding Precision
- Distinct Hypotheses
- Testability Rate
- Cross-Domain Rate
- Novelty
- Information Gain
- Cost per Valid Hypothesis
- Duplicate Rate

新規性と研究価値は、可能な範囲で生成方式を伏せた独立レビューによって評価する。

## 18.4 診断課題

初期診断には市場生成モデルの研究探索を使用する。

既知の到達候補：

- エージェントベース市場モデル
- 生成モデルの逆問題
- パラメータ識別可能性
- 合成データによる異常検知
- 市場状態推定

ただし、これらを正解リストとして固定せず、他の有効な研究方向も評価する。

---

# 19. 開発フェーズ

## H0：収集・保存基盤

実装内容：

- AlgoForgeの既存収集ロジック調査
- Paper Collector
- OSS Collector
- SQLite保存
- 出典追跡
- 更新検知
- 重複排除

完了条件：

- 同一資料を識別できる
- 更新版を追跡できる
- 検索・取得履歴を再現できる

## H1a：Capability Extraction

実装内容：

- Model・Method・Capability抽出
- 根拠状態の付与
- 出典との対応付け

品質ゲート：

- 10件以上の抽出結果を独立に目視レビューする
- 誤抽出、欠落、根拠の不一致を記録する

10件レビューは初期診断であり、抽出精度の統計的保証ではない。

## H1b：Entity Resolution

実装内容：

- 正規化
- 類似候補検索
- SAME_AS / IS_A等の分類
- Controlled Vocabulary
- 統合履歴

完了条件：

- 同一概念と関連概念を区別できる
- 誤統合を修正できる

## H2：Discovery Operators

実装内容：

- 6 Operators
- Operator Eligibility
- Negative Constraints
- 探索予算制御
- 失敗履歴

完了条件：

- 適用理由と棄却理由を追跡できる

## H3：Hypothesis Generation

実装内容：

- 仮説生成
- 帰無仮説
- 反証条件
- 実験案
- 実装ギャップ評価

完了条件：

- 根拠付きで検証可能な仮説を生成できる

## H4：Portfolio・Study Manager連携

実装内容：

- 仮説重複検出
- 多目的選択
- Review Gate
- Study Manager登録

完了条件：

- 採用理由と棄却理由を保存できる

## H5：比較実験

実装内容：

- 直接生成方式との比較
- Capability方式の評価
- Cross-Domain Compositionの評価
- Continuous / Problem-Driven比較

完了条件：

- 同一予算で各方式を比較できる
- 探索方式の効果を分析できる

---

# 20. セキュリティと運用

## 20.1 外部資料

取得した論文、README、Webページの内容は信頼できない入力として扱う。

資料中の命令文をAgentへの実行指示として扱わない。

## 20.2 OSS

公開OSSの取得と実行を区別する。

未知のリポジトリを自動実行しない。

コード実行が必要な場合は、隔離環境と承認プロセスを利用する。

## 20.3 資源制御

検索回数、取得容量、LLMトークン、実行時間を制限する。

APIのレート制限と利用条件を遵守する。

## 20.4 研究運用

仮説の生成・登録は実験実行の承認を意味しない。

実験計画のレビューと承認を別工程とする。

---

# 21. 未決定事項

1. AlgoForgeの金融レポート収集基盤の内部構造
2. 論文検索APIとOSS検索APIの選定
3. 取得資料の本文保存方式
4. Controlled Vocabularyの初期定義
5. Entity Resolutionの閾値と評価方法
6. Operator Eligibilityの具体的ルール
7. 探索予算の初期値
8. 仮説の新規性評価方法
9. Portfolio配分の最適化方法
10. Study Manager既存機能との責務境界
11. Agent Backend切替条件
12. Knowledge Graphの将来的な拡張方式

---

# 22. 最終設計方針

HDEは、研究資料の検索から仮説生成までを一体化するが、各工程を独立して検証できる構成とする。

中心となるサイクルは以下である。

```text
Continuous Discovery
         |
         +-------------------+
                             |
Problem-Driven Discovery     |
         |                   |
         +---------+---------+
                   |
                   v
           Research Collection
                   |
                   v
           Capability Extraction
                   |
                   v
             Entity Resolution
                   |
                   v
            Knowledge Graph
                   |
                   v
           Discovery Operators
                   |
                   v
           Hypothesis Generation
                   |
                   v
                Review
                   |
                   v
          Hypothesis Portfolio
                   |
                   v
             Study Manager
                   |
                   v
           Scientific Experiment
                   |
                   v
                Finding
                   |
                   +----> Next Discovery
```

本システムでは、研究分野そのものではなく、**研究が提供するCapabilityと、それらを組み合わせることで検証可能になる新しい仮説**を探索の中心に置く。

さらに、探索の失敗履歴、研究上の制約、既存のFindingを蓄積することで、研究サイクルを繰り返すほど探索対象と判断根拠が充実する構成を目指す。

---

**Document Status:** Draft v0.2  
**Implementation Status:** 未着手  
**Experiment Approval:** 未承認  