# AlgoForge Phase 1B-2 実装計画草案（v0.2）

**日付**: 2026-11-22  
**前提**: Phase 1B-1 証跡確認済み（コミット `cfe8126`、2024年スナップショット SHA-256: `bbc7a9c24b701f78d0d4716e7f633246f05445a198f6a2cdca771861e43939a0`、評価設定 `spread=0.010, slippage=0.002`）  
**状態**: 計画承認待ち（実装・探索実行未着手）  

---

## 1. 厳格な探索予算と変更範囲

- **生成試行の厳格な上限**: **25件**（意図的未対応候補 3件、重複候補、拒否候補もすべて予算を消費し、補充生成は一切行わない）。
- **キャッシュヒット時**: `StrategyRegistry.lookup(e_hash)` でヒットした場合はバックテストを再実行しない。
- **変更対象ファイル**: **`scripts/run_phase1b2_exploration.py` のみ**（AlgoForge 本体・`causal_fx_backtest.py`・既存 `strategy_registry.db`・Phase 0〜4 履歴は一切変更しない。テスト用に一時 SQLite DB を使用）。

---

## 2. 進化的探索方式（`AlgoForgeEvoGenerator`）

- **実装場所と再利用**: `scripts/run_phase1b2_exploration.py` 内に `AlgoForgeEvoGenerator` を定義し、既存の `algoforge_adapter.py`、`strategy_registry.py`、`causal_fx_backtest.py` をそのまま再利用する。
- **構成（計 25試行、固定シード `42`）**:
  - **初期個体（10件）**: 意図的未対応 3件（非空 `exit`、未対応 `risk`、未対応 `streak` 条件）＋ 型安全ランダム生成 7件。
  - **親選択**: `Trades >= 30` かつ `root_counts[root_hash] < 3` を満たす候補を適応度（`annualized_sharpe`）降順で上位5件抽出し一様選択。
  - **突然変異（15件）**: 指標パラメータ・論理演算（`and`/`or`）・`risk`（`k_sl`, `k_tp`, `max_hold_bars`）の変異。
  - **root cap と枯渇時フォールバック**: **3件/系列** を厳格維持。適格親が枯渇した場合は制約を緩和せず、新規型安全ランダム生成へフォールバックする。

---

## 3. 型安全な条件生成（出力列マッピング）

以下の許可済み指標と出力列のみを参照し、存在しない列を参照する条件を生成しない：
- `ema` (`{id}`), `sma` (`{id}`), `rsi` (`{id}`), `roc` (`{id}`), `adx` (`{id}`)
- `bb` (`{id}_upper`, `{id}_lower`), `donchian` (`{id}_upper`, `{id}_lower`)
- `macd` (`{id}_line`, `{id}_signal`), `stochastic` (`{id}_k`, `{id}_d`)
- 比較対象: 価格列（`close`）または同型指標列・妥当範囲の定数。

---

## 4. 必須テスト・再現性・受入基準

1. **契約・因果性テスト**: NaN・等値・ウォームアップ・入れ子論理式を含むベクトル化一致（**不一致 0件**）、複数切断位置での未来改変不変性（**先読み 0件**）。
2. **root cap 厳格性**: 全系列で `count <= 3`（**違反 0件**）。
3. **保存・報告**: `data/phase1b2_exploration_results.json` に生成順序、Strategy JSON、正規化ハッシュ、評価ハッシュ、評価結果、生成数・重複数・拒否数・実評価数・キャッシュヒット数・Capability Request 件数を記録・報告する。
