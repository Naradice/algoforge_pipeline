# StockNet FXトレード戦略探索システム
## 統合設計仕様書 v0.3

**Status:** Draft  
**Date:** 2026-11-15  
**Project:** StockNet  
**Study Manager Root Goal:** #774「FXのトレード戦略の探索」

---

# 1. 目的と基本方針

## 1.1 研究目的

本システムの最上位研究目的は、**FXのトレード戦略の探索**である。

AI Agentが自律的に研究方向を発見し、市場現象に関する仮説を立案し、実験によって検証する。その成果を取引戦略へ発展させ、バックテストによって収益性と汎化能力を評価する。

既存のテクニカル指標の組み合わせやパラメータ最適化だけでなく、市場生成モデル、統計的性質、機械学習、異常検知、他分野の研究成果などを幅広く対象とする。

## 1.2 主要な研究課題

1. AIは既存戦略の単純な変種にとどまらず、新しい市場仮説を発見できるか。
2. 論文・OSS・研究手法を組み合わせることで、新しい研究方向を生成できるか。
3. 市場現象の発見を、再現性のある取引戦略へ変換できるか。
4. バックテスト結果のフィードバックによって探索効率は改善するか。
5. 単一Agent、役割分担した複数Agent、統括Agentのどの構成が有効か。
6. 探索の退化、重複、過学習を自動検出・抑制できるか。

## 1.3 基本原則

- **研究仮説を起点とする。** 戦略コードやJSONの生成を研究の出発点にしない。
- **研究方向はAIが立案する。** 最上位Goal以外の研究課題は原則としてAIが提案する。
- **探索方法も研究対象とする。** 特定のAgent構成や探索アルゴリズムを最初から最良と仮定しない。
- **科学的検証と収益性評価を区別する。** 市場現象の存在と取引可能な優位性は別々に検証する。
- **失敗した研究も保存する。** 再現性と知識の蓄積を重視する。
- **既存基盤を再利用する。** AlgoForgeとStudy Managerの機能を重複実装しない。

---

# 2. システム全体構成

## 2.1 論理アーキテクチャ

```text
                    Research Goal
                FXのトレード戦略の探索
                           |
                           v
                 Research Orchestrator
                    (Python Manager)
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
    Discovery Agent   Strategy Agent   Research Critic
          |                |                |
          +----------------+----------------+
                           |
                           v
                 Hypothesis Portfolio
                           |
          +----------------+----------------+
          |                                 |
          v                                 v
    Scientific Experiment             Strategy Synthesis
          |                                 |
          v                                 v
    Study Manager MCP                  AlgoForge MCP
          |                                 |
          |                      +----------+----------+
          |                      |          |          |
          |                     Data       Model     Strategy
          |                      |          |          |
          |                      +----------+----------+
          |                                 |
          |                            Backtesting
          |                                 |
          +----------------+----------------+
                           |
                           v
                 Evaluation & Findings
                           |
                           v
                   Knowledge Base
                           |
                           v
                  Next Research Cycle
```

## 2.2 コンポーネントの責務

| コンポーネント | 主な責務 |
|---|---|
| Research Orchestrator | 研究サイクル、Agent割当、予算、ジョブ管理 |
| Hypothesis Discovery Engine | 研究検索、知識抽出、仮説生成 |
| Strategy Synthesis | 仮説からAlgoForge戦略定義への変換 |
| Research Critic | 実験計画・結果・次の方向性の批判的評価 |
| Meta-Researcher | 複数の研究系列の統括と探索資源配分 |
| Degeneracy Detector | 重複、探索停滞、類似戦略の検出 |
| AlgoForge | データ、モデル学習、戦略実行、バックテスト |
| Study Manager | 研究目的、仮説、実験、知見、レビューの管理 |
| Knowledge Base | 論文、OSS、Capability、研究間の関係 |

---

# 3. 研究探索アーキテクチャ

## 3.1 二系統の探索方式

本システムでは、次の二つの研究探索を併用する。

### A. 継続的研究探索（Continuous Discovery）

特定の実験上の問題に依存せず、新しい研究方向を継続的に発見する。

対象はFX研究に限定しない。

探索対象：

- 金融市場の生成モデル
- エージェントベースモデル
- 統計物理・複雑系
- 情報理論
- 状態空間モデル
- Simulation-Based Inference
- 異常検知
- 表現学習
- 制御理論
- その他の関連分野

目的は、未知の研究方向や利用可能な研究技術を発見することである。

### B. 課題駆動研究探索（Problem-Driven Discovery）

実験で発見された問題や未解決事項を起点として研究を探索する。

起動条件の例：

- 仮説が反証された
- モデルの予測性能が不足した
- 実市場への汎化に失敗した
- パラメータが識別できなかった
- AlgoForgeで仮説を表現できなかった
- 既存研究系列が停滞した

目的は、問題を解決する新しい方法や、代替仮説を発見することである。

## 3.2 探索予算

二つの探索方式には独立した予算を設ける。

初期実験では同程度の予算を割り当て、研究成果を比較する。

評価対象：

- 新規仮説の発見数
- 実験可能な仮説の割合
- 既存研究との重複率
- 研究上の情報利得
- 有望な取引戦略への貢献
- 計算コスト

予算配分の最適化は後続実験で扱う。

---

# 4. Hypothesis Discovery Engine

## 4.1 目的

論文・OSS・既存の研究成果から、新しい研究仮説を継続的に生成する。

単なる論文要約や既知戦略の紹介ではなく、研究成果が提供する能力を抽出し、異なる研究間の組み合わせを探索する。

## 4.2 探索の4軸

| 軸 | 内容 |
|---|---|
| Phenomenon | 説明・予測したい市場現象 |
| Model | 現象を生成・説明するモデル |
| Method | 推定・学習・検証の方法 |
| Application | 研究成果の利用先 |

例：

**市場参加者モデル × Simulation-Based Inference × パラメータ推定 × 市場状態検出**

このような組み合わせを、研究上の根拠と検証可能性に基づいて探索する。

## 4.3 Capability Extraction

論文やOSSから以下の情報を抽出する。

```json
{
  "research_object": "...",
  "mechanism": "...",
  "capabilities": [],
  "required_data": [],
  "observables": [],
  "assumptions": [],
  "limitations": [],
  "transferability": [],
  "possible_experiments": [],
  "references": []
}
```

特にCapabilityを重視する。

例えば、エージェントベース市場モデルが「参加者の行動パラメータから価格系列を生成できる」なら、その能力を別の推定技術と組み合わせる。

## 4.4 Cross-Domain Composition

異なる研究のCapabilityを接続する。

接続条件：

1. 一方の出力が他方の入力として利用できる。
2. 前提条件に重大な矛盾がない。
3. 検証可能な予測を導出できる。
4. 必要なデータや実験環境を用意できる。
5. 最上位Goalとの関係を説明できる。

研究の組み合わせは、既存研究で実証されたものと新たな推測を区別する。

## 4.5 仮説生成

生成する仮説には、最低限以下を含める。

```json
{
  "title": "...",
  "research_question": "...",
  "mechanism": "...",
  "predictions": [],
  "null_hypothesis": "...",
  "falsification_conditions": [],
  "required_data": [],
  "required_capabilities": [],
  "related_research": [],
  "potential_trading_application": "...",
  "known_limitations": []
}
```

## 4.6 Hypothesis Portfolio

仮説を単一スコアだけで順位付けしない。

以下を独立して評価する。

- 科学的な新規性
- 検証可能性
- 必要な計算資源
- 既存仮説との差異
- 研究上の情報利得
- 取引戦略への応用可能性
- 失敗時に得られる知識

探索では、性質の異なる仮説を並行して保持する。

---

# 5. 研究知識の管理

## 5.1 Knowledge Base

Knowledge Baseには以下を保存する。

- 論文・プレプリント
- OSS・実装
- 市場現象
- モデル
- アルゴリズム
- Capability
- 仮定・制約
- 仮説
- 実験結果
- 研究間の関係

## 5.2 研究知識グラフ

概念的には次の関係を持つ。

```text
Paper / OSS
    |
    v
Model / Algorithm
    |
    v
Capability
    |
    +----> Observable
    |
    +----> Compatible Capability
    |
    v
Research Hypothesis
    |
    v
Experiment
    |
    v
Finding
    |
    +----> Supports / Refutes Hypothesis
    |
    +----> Generates New Questions
```

初期段階では、Study ManagerのReference・Claim・関係付け機能を調査し、利用可能な範囲で再利用する。

不足する場合のみ、独立したKnowledge Baseを実装する。

---

# 6. Study Manager連携

## 6.1 研究ツリー

登録済みの最上位Goal：

**Goal #774：FXのトレード戦略の探索**

子Goal、Hypothesis、Experiment、Findingは、研究進行に応じてAIが立案する。

最初から研究分野ごとの固定ツリーは作成しない。

## 6.2 研究サイクル

```text
Search Existing Research
          |
          v
Generate Hypothesis
          |
          v
Review Experiment Plan
          |
          v
Register Experiment
          |
          v
Execute Experiment
          |
          v
Evaluate Results
          |
          v
Register Finding
          |
          v
Generate Next Hypothesis
```

## 6.3 利用するMCP機能

- `search_nodes`
- `create_goal`
- `request_review`
- `record_review_outcome`
- `experiment_preregistration`
- `experiment_blind_holdout`
- `list_reproducibility_gaps`
- Autopilot関連機能

Hypothesis、Experiment、Findingの作成・更新にはStudy Managerの対応するMCP操作を利用する。具体的な操作名と引数は実装時に確認する。

## 6.4 実験承認

研究仮説の生成・登録と、実験の実行を分離する。

実験計画については、科学的妥当性、対照群、評価基準、計算予算をレビューする。

初期運用では、人間による計画承認を経て実行する。

自律実行の許可範囲は、後に明示的なポリシーとして設定する。

## 6.5 保留データ管理

最終評価用のデータは探索Agentから隔離する。

Study Managerの事前登録と保留データ管理を利用する。

ただし、AlgoForgeの期間指定だけでは、Agentが別の実行で保留期間を指定することまで禁止できない。

そのため、Orchestrator側でも保留期間へのアクセス権限を管理する。

---

# 7. AlgoForge連携

## 7.1 基本方針

AlgoForgeを以下の実行基盤として利用する。

- 市場データの収集・管理
- データ特性分析
- MLモデル学習
- 戦略登録・バージョン管理
- バックテスト
- 取引履歴・評価指標取得
- ログ管理

独立したバックテストエンジンは原則として開発しない。

## 7.2 Strategy Definition

戦略は宣言的JSONで定義する。

旧来の以下のPython関数インターフェースは使用しない。

- `open_signal`
- `update_signal`
- `exit_signal`

戦略定義には、通貨ペア、時間軸、インジケータ、エントリー・エグジット条件、リスク管理を含む。

例：

```json
{
  "symbol": "USDJPY",
  "timeframe": "M30",
  "indicators": [
    {
      "name": "fast_ema",
      "type": "ema",
      "period": 12,
      "source": "close"
    },
    {
      "name": "slow_ema",
      "type": "ema",
      "period": 26,
      "source": "close"
    }
  ],
  "long": {
    "entry": {
      "logic": "and",
      "conditions": [
        {
          "type": "crossover",
          "left": "fast_ema",
          "right": "slow_ema"
        }
      ]
    }
  },
  "risk": {
    "sl_pips": 20,
    "tp_pips": 40,
    "position_size": 1
  }
}
```

これは構造例であり、完全なStrategy Definitionの全必須項目を示すものではない。

## 7.3 インジケータ拡張

AlgoForgeはインジケータをディスパッチ方式で管理する。

新しいインジケータを追加する場合：

1. 計算処理を追加する。
2. 必要なウォームアップ期間を登録する。
3. 入力データと出力形式を検証する。
4. 未来情報を使用しないことを検証する。
5. 単体テストを追加する。
6. バックテストで再現性を確認する。

既存の`derived`、`rolling`も利用できる。

ただし、任意のPandas式評価については、安全な式の制限とコード実行対策が必要である。

## 7.4 条件式拡張

現在利用できる条件式には、比較、クロス、streak、rolling、regime、group_ref、ml_signal、llm_signalなどがある。

新しい条件式は`conditions.py`のディスパッチに追加できる。

研究仮説が既存機能で表現できない場合、Coding Agentが拡張案を作成する。

ただし、AlgoForgeの実装変更は通常の戦略生成とは区別し、テストとレビューを必要とする。

## 7.5 バックテスト

基本操作：

```text
create_strategy
      |
      v
start_strategy_run
      |
      v
get_run_status
      |
      v
get_run_metrics
get_run_trades
get_run_logs
```

戦略の更新には`update_strategy`を利用する。

戦略定義と実行設定のバージョンを保存し、同じ条件で再実行できるようにする。

## 7.6 時系列と約定

AlgoForgeは確定足でシグナルを評価し、次の足のOpenでエントリーを執行する。

この仕様を維持する。

追加で検証する事項：

- ウォームアップと評価期間の境界
- 同一バー内のTP/SL到達順序
- スプレッドとスリッページ
- データ欠損
- 複数時間軸の確定時刻
- MLモデルの特徴量生成時刻

評価期間をスライスする方式では、期間開始前のウォームアップデータが失われる可能性がある。必要な過去データを読み込んでも、その期間を評価対象に含めない構成を検討する。

---

# 8. MLモデル探索

## 8.1 位置づけ

MLモデルは戦略探索の手段の一つとする。

すべての仮説を機械学習で検証する必要はない。

## 8.2 学習前評価

モデル学習前に、AlgoForgeの`assess_target_difficulty`を利用する。

対象例：

- 将来リターン
- 方向
- ボラティリティ
- ジャンプ
- トレンド変化

線形モデル、HistGB、MLPなどとの比較から、学習対象に予測可能性があるかを調べる。

予測性能の改善と取引利益の改善は別々に評価する。

## 8.3 学習フロー

```text
list_datasets
      |
      v
assess_target_difficulty
      |
      v
list_preprocessed_datasets
      |
      v
create_model
      |
      v
start_training_run
      |
      v
get_training_status
      |
      v
get_model_validations
      |
      v
deploy_model
      |
      v
Strategy / ml_signal
      |
      v
Backtest
```

前処理レシピの新規作成は現時点でMCP未対応のため、必要に応じてUIまたはREST APIを利用する。

## 8.4 モデルの再現性

TrainingRunのアーティファクトは不変として管理する。

再現性を確保するため、モデルIDだけでなく、使用した学習ランIDを保存する。

`model_id`が現在デプロイされているランを参照する方式では、後のデプロイ変更で実行結果が変わり得る。

したがって、可能な限り学習ランIDを固定する。

---

# 9. Agent実行アーキテクチャ

## 9.1 Agent Manager

Agent ManagerはPythonの常駐プロセスとして実装する。

主な責務：

- タスクキュー管理
- Agentの起動・終了
- Agent間の成果物受け渡し
- MCP接続管理
- 実験予算管理
- 状態保存と再開
- 失敗時のリトライ
- Agent実行方式の切替

## 9.2 Agent Backend

以下の実行方式をサポートする。

### Coding Agent

主要な研究・実装タスクを担当する。

論文調査、仮説生成、コード変更、実験設計などに利用する。

### Python Agent

Coding Agentの利用量が不足した場合や、比較的小さなタスクを実行する場合に利用する。

### Local LLM

軽量な分類、重複判定、研究情報の抽出などで実験的に利用する。

ただし、品質を検証せずに重要な研究判断を委任しない。

## 9.3 共通インターフェース

```python
class ResearchAgent:
    async def execute(self, task, context):
        ...
```

Agentの入出力は構造化された成果物とする。

実行方式を変更しても、研究履歴や未完了タスクを引き継げるようにする。

## 9.4 Agentの役割

役割と実行プロセスは分離する。

同一Agentが複数の役割を担当してもよい。

想定する役割：

- Discovery
- Hypothesis Generator
- Experiment Designer
- Strategy Synthesizer
- Critic
- Meta-Researcher

どの役割を分離すべきかは、比較実験によって判断する。

---

# 10. Agent構成の比較実験

## 10.1 比較構成

### A. 単一Agent

仮説生成、戦略作成、結果分析、改善を同一Agentが担当する。

### B. 単一Agent＋独立レビュー

単一Agentによる探索に、Study Managerの独立レビューを追加する。

### C. Generator＋Critic

仮説・戦略生成と、評価・改善指針の作成を分離する。

### D. 複数Generator＋Meta-Researcher

複数の研究系列を並列探索し、統括Agentが研究方向と予算を管理する。

## 10.2 比較条件

- 同一LLMモデル
- 同一データ
- 同一バックテスト環境
- 同一総トークン予算
- 同一バックテスト予算
- 同一初期研究情報
- 複数独立試行
- 共通の最終評価

## 10.3 評価指標

- 有望仮説の発見数
- 実験可能な仮説の割合
- 戦略の構造的新規性
- 探索の多様性
- OOS性能
- 計算コスト
- 改善効率
- 再現性

---

# 11. Degeneracy Detector

## 11.1 目的

研究探索が既存戦略の小さな変種へ集中していないかを検出する。

また、戦略の実質的な重複や過学習の兆候を評価する。

## 11.2 検出対象

### 構造的類似性

- インジケータ構成
- 条件式ツリー
- パラメータ
- 時間軸
- リスク管理方式

AlgoForgeはJSON形式なので、正規化された条件式ツリーを直接比較できる。

### 行動的類似性

- エントリー時刻
- 売買方向
- 保有期間
- 取引イベントの重複

### 損益の類似性

- バー単位の損益系列
- ドローダウンの同期性
- 期間別利益寄与

AlgoForgeのEquity Curveを利用できる。

ただし、Equity Curveにポジション系列が含まれない場合、保有状態の比較には取引履歴からの再構成が必要となる。

### 探索停滞

- 新規仮説の減少
- 同一研究系列への集中
- OOS改善率の低下
- 同じ特徴量・条件式の反復

## 11.3 実行タイミング

1. 仮説生成後
2. 戦略JSON生成後
3. バックテスト後
4. 一定探索回数ごと

初期段階では診断結果の記録にとどめ、強制的な探索変更の効果は別途検証する。

---

# 12. 評価・統計検証

## 12.1 評価対象

収益性：

- Net PnL
- Profit Factor
- Sharpe Ratio
- Expectancy

リスク：

- Maximum Drawdown
- 損失集中
- コスト感度
- レバレッジ・取引数量

汎化：

- OOS性能
- Walk-forward安定性
- レジーム別性能
- パラメータ感度

探索品質：

- 新規性
- 多様性
- 仮説の反証可能性
- 計算効率

## 12.2 多重検定

大量の戦略候補を試すため、偶然の高成績を明示的に考慮する。

試行した候補は、失敗例も含めて保存する。

必要に応じて、時系列依存を考慮したブートストラップ、Deflated Sharpe Ratio、SPAなどを使用する。

## 12.3 データリーク対策

- 確定足のみ使用
- 次足約定
- 学習・検証・保留期間の分離
- 特徴量の因果性検証
- ウォームアップ期間の管理
- モデル学習時の時系列分割
- 保留期間のアクセス制御

バックテストエンジンの仕様による先読み防止と、研究全体における保留データ保護を区別する。

---

# 13. 非同期ジョブ管理

AlgoForgeの学習・バックテストは非同期ジョブとして管理する。

## 13.1 状態

```text
Queued
  |
Running
  |
  +---- Completed
  |
  +---- Error
  |
  +---- Stopped
  |
  +---- Stale / Timeout
```

## 13.2 実行前確認

`get_queue_status`を利用し、利用可能なワーカーを確認する。

ワーカー数が0の場合、実行を開始せず待機または代替実行方式を検討する。

## 13.3 完了通知

Webhookを優先し、必要に応じてポーリングを併用する。

WebhookはHMAC-SHA256署名を検証する。

イベントの重複受信や順序の入れ替わりに対応する。

## 13.4 障害処理

- ワーカー停止
- Heartbeat stale
- ジョブタイムアウト
- API接続失敗
- 学習・バックテスト内部エラー

これらを戦略や研究仮説の失敗と混同しない。

---

# 14. 実験成果物と再現性

## 14.1 保存する識別子

- Study Manager Goal ID
- Hypothesis ID
- Experiment ID
- AlgoForge Dataset ID
- Strategy ID
- Strategy Version
- Strategy Run ID
- Model ID
- Training Run ID
- Agent Task ID
- Code Revision

## 14.2 成果物

- ResearchTask
- CapabilityRecord
- HypothesisProposal
- ExperimentPlan
- StrategyDefinition
- BacktestReport
- RobustnessReport
- DegeneracyReport
- CritiqueReport
- ResearchDecision
- Finding

## 14.3 原則

同じ研究結果を複数システムで独立管理しない。

AlgoForgeを実行結果の正本とし、Study Managerには研究上の解釈と実行結果への参照を保存する。

Orchestratorは両者の識別子と実行状態を管理する。

---

# 15. セキュリティ

## 15.1 AlgoForge

宣言的JSONによる戦略定義を基本とする。

`derived`式、LLM条件、外部モデル連携、カスタムインジケータについては、個別に安全性を検証する。

## 15.2 Coding Agent

AlgoForgeのコード拡張を行う場合は、通常の戦略生成とは別の権限を使用する。

コード変更後はテストとレビューを実施する。

## 15.3 実取引

研究用Agentから実取引口座への発注を許可しない。

実資金運用は別の承認・リスク管理プロセスを必要とする。

---

# 16. 開発ロードマップ

## Phase 0：既存MCP連携の検証

- AlgoForge接続
- Study Manager接続
- データ取得
- 戦略登録
- バックテスト
- 結果取得
- 実験ID対応付け
- 再現性確認

## Phase 1：Hypothesis Discovery Engine

- 研究検索
- Capability抽出
- Cross-Domain Composition
- 仮説生成
- Hypothesis Portfolio
- Study Managerへの登録

## Phase 2：単一Agent研究サイクル

- 仮説立案
- 実験計画
- 戦略JSON生成
- AlgoForgeバックテスト
- 結果評価
- Finding登録
- 次の仮説生成

## Phase 3：Degeneracy Detector

- JSON構造類似性
- 取引行動類似性
- 損益系列類似性
- 探索停滞検出

## Phase 4：探索方式の比較

- 直接戦略探索
- 現象起点探索
- 研究起点探索
- Capability組み合わせ探索
- 継続的探索と課題駆動探索の比較

## Phase 5：Agent構成比較

- 単一Agent
- 単一Agent＋独立レビュー
- Generator＋Critic
- 複数Generator＋Meta-Researcher

## Phase 6：自律研究の高度化

- 探索予算の動的配分
- AlgoForgeの機能拡張提案
- 長期的な研究知識の蓄積
- 複数研究系列の統括
- 独立データによる長期検証

---

# 17. 未決定事項

以下は詳細設計時に決定する。

1. Coding Agentの起動・制御インターフェース
2. Python Agentへのフォールバック条件
3. Local LLMの適用範囲
4. 論文・OSS検索の情報源と検索頻度
5. Capabilityの正規化スキーマ
6. Knowledge Baseの保存方式
7. 仮説の重複判定方法
8. 探索予算と停止条件
9. AlgoForge拡張コードの承認方式
10. Study Manager Autopilotと独自Orchestratorの責務分担
11. 保留データのアクセス制御
12. 実験成功基準と統計的判定規則

---

# 18. 最終的な研究サイクル

本システムの中心となる研究サイクルは次のとおりである。

```text
Continuous Discovery
          |
          +---------------------+
                                |
Problem-Driven Discovery       |
          |                     |
          +----------+----------+
                     |
                     v
              Knowledge Base
                     |
                     v
             Capability Discovery
                     |
                     v
             Hypothesis Generation
                     |
                     v
             Hypothesis Portfolio
                     |
                     v
             Scientific Experiment
                     |
                     v
              Strategy Synthesis
                     |
                     v
              AlgoForge Backtest
                     |
                     v
             Evaluation & Critique
                     |
                     v
              Study Manager
                     |
                     v
                New Findings
                     |
                     +----> Continuous Discovery
                     |
                     +----> Problem-Driven Discovery
```

この構成により、AIは単に戦略パラメータを最適化するのではなく、研究成果を蓄積しながら、新しい市場仮説・モデル・実験手法・取引戦略を継続的に探索する。

**本システムの研究対象は、発見されたFX戦略だけでなく、それらを発見するAI研究プロセスそのものでもある。**

---

**Document Status:** Draft v0.3  
**Next:** Hypothesis Discovery Engineの詳細仕様とPhase 0のMCP統合契約を定義する。