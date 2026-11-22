# AlgoForge_Pipeline 評価境界分離計画 v0.1（`EvaluatorProtocol` / `LocalCausalEvaluator`）

> **ステータス**: 設計・計画のみ（実装・実験実行・AlgoForge/Study Manager 変更は未実施）

## 1. 最小 `EvaluatorProtocol` と `LocalCausalEvaluator` の設計案

### 1.1 `EvalResult` / `EvaluatorProtocol`
```python
@dataclass(frozen=True)
class EvalResult:
    status: str                  # "ok" | "error"
    metrics: Dict[str, Any]      # 既存 summary（total_trades, win_rate, profit_factor, total_return_pct 等）
    engine_id: str               # 例: "causal_fx_backtest_v1.0"
    dataset_sha256: str          # 入力データ識別子
    eval_config_hash: str        # 評価設定識別子
    run_ref: Optional[str]       # 外部実行ID（Local時は None）
    error: Optional[str] = None  # status=="error" 時のエラー詳細

class EvaluatorProtocol(Protocol):
    engine_id: str
    def evaluate(
        self,
        strategy_json: Dict[str, Any],
        dataset_ref: Any,
        eval_config: Dict[str, Any],
    ) -> EvalResult: ...
```

### 1.2 `LocalCausalEvaluator`
- `scripts/run_phase1b2_exploration.py` 内の `evaluate_algoforge_conditions_vectorized` + `run_causal_fx_backtest` 呼出しをそのままラップする。
- バックテストの計算・約定・コスト（`BacktestConfig`, `GateConfig`）・時刻契約・`eval_start_idx` は**一切変更しない**。

## 2. 変更対象ファイルと変更しないファイルの一覧

- **変更対象（予定）**:
  - `src/research_pipeline/backtest/evaluator_protocol.py`（新規：`EvalResult`, `EvaluatorProtocol`, `LocalCausalEvaluator`）
  - `scripts/run_phase1b2_exploration.py`（評価呼出しを `EvaluatorProtocol` 経由に置換）
  - `scripts/test_phase1b2_evaluator_regression.py`（新規：変更前後の完全一致回帰テスト）
- **変更しないファイル**:
  - `src/research_pipeline/backtest/causal_fx_backtest.py`（研究用 PoC・回帰基準として維持）
  - `src/research_pipeline/backtest/strategy_registry.py`（`StrategyRegistry`・キャッシュ名前空間を維持）
  - `src/research_pipeline/integration/*`（Phase 2B Unit 経路は今回対象外。ただし `EvaluatorProtocol` は将来 `algoforge_unit_v1` 経路もラップ可能な汎用シグネチャとする）
  - `algoforge/*`、`study_manager/*`、Phase 0〜4C スクリプト群

## 3. 既存動作の回帰試験計画と合否基準

- **試験方法**: 固定シード `42`、最大 `25` 試行、`usdjpy_m5_2024_oanda_snapshot.parquet` を用い、変更前の結果 JSON と変更後（`LocalCausalEvaluator` 経由）の結果 JSON を比較する。
- **合否基準（100% 一致必須）**:
  1. 候補生成順序・`strategy_hash`・`root_hash` の完全一致
  2. 重複排除件数・キャッシュヒット件数・Capability Request（3 件）の完全一致
  3. 全評価済み戦略の `total_trades`, `win_rate`, `profit_factor`, `total_return_pct` の完全一致

## 4. `StrategyRegistry` およびキャッシュ互換性の検証方法

- `StrategyRegistry` のハッシュ計算（`compute_algoforge_evaluation_hash`）は変更せず、`engine_version = evaluator.engine_id` をそのまま渡す。
- したがって既存 `algoforge_v1` キャッシュは 100% 再利用され、将来別エンジン（例：`algoforge_api_v1`）を追加した場合も `engine_id` によりキャッシュが自動分離される。

## 5. AlgoForge API 将来接続時の未解決事項（調査のみ・実装なし）

- **API スキーマ**: `POST /strategies` (`StrategyCreate`) および `POST /strategies/{id}/runs` (`StrategyRunCreate`) の必須フィールド（データセット ID、時間足、スプレッド/スリッページ指定範囲）の厳密な対応付け。
- **非同期ジョブ制御**: `202 Accepted` 後のポーリング間隔、タイムアウト、失敗時の `EvalResult(status="error")` 変換規則。
- **仕様差**: Pipeline 側 PoC と AlgoForge 本体の約定・コスト仕様差の許容範囲定義。
- **Study Manager（Goal `#774`）要件**: 次期フェーズでは実験ノード作成・`check_execution_gate` 確認・`record_result` 冪等記録（`run_id` / `eval_hash` キー）を最小クライアントで連携する。

## 6. 実装見積もりと既存資産への影響

- **見積もり**: 新規ファイル 2 件（計 ~120 行）＋ `run_phase1b2_exploration.py` の局所修正（~15 行）。
- **既存資産への影響**: ゼロ（`causal_fx_backtest.py`, `StrategyRegistry`, Phase 0〜4C, Phase 2B Unit 経路はすべて無変更で維持）。

## 7. 追加設計要件：宣言的 JSON と Capability Request

1. **宣言的 JSON 原則**: 戦略・探索・評価条件は任意の Python コードや実行可能スクリプトを含めず、許可された機能 ID・演算子・パラメータのみを宣言的 JSON で参照する。
2. **Capability Request の記録・管理境界**:
   - 未対応機能は実行前に検出し、構造化された Capability Request として記録する（今回の評価境界分離では既存の Capability Request 記録動作の維持を受入条件とする）。
   - 実装課題は GitHub Issues、研究上の必要性・仮説との関係は Study Manager（Goal `#774`）で管理し、正規化機能 ID 等で重複排除して既存 Issue 参照を優先する。
   - 未対応機能を理由に Pipeline 側へ独自のモデル・バックテスト実装を増やさない。
3. **後続フェーズの設計課題（今回対象外）**: JSON スキーマの全面再設計、および GitHub Issues の自動起票・重複照合連携は後続フェーズで扱う。
