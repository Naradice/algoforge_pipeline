# AlgoForge Phase 1B-2 実装計画草案（v0.3）

**日付**: 2026-11-22  
**前提**: Phase 1B-1 証跡確認済み（コミット `cfe8126`、2024年スナップショット SHA-256: `bbc7a9c24b701f78d0d4716e7f633246f05445a198f6a2cdca771861e43939a0`）  
**状態**: 計画承認待ち（実装・探索実行未着手）  

---

## 1. root cap と親選択の厳密規則

- **計数対象**: バリデーションを通過して評価対象となった「新規ユニーク採用候補」に対して `root_counts[root_hash] += 1` を計数する（バリデーション拒否・ラン内重複・キャッシュヒット時は計数しない）。
- **親選択とタイブレーク**: `Trades >= 30` かつ `root_counts[root_hash] < 3` を満たす候補から、`annualized_sharpe` 降順 $\to$ `max_drawdown_pct` 降順 $\to$ `trade_count` 降順 $\to$ `strategy_hash` 昇順で上位5件を抽出し一様選択する。
- **枯渇時フォールバック**: 適格親が枯渇した場合は制約を緩和せず、新規 `root_hash = strategy_hash` を発行して型安全ランダム生成を行う。
- **厳格上限**: 総生成試行回数は **25回を厳格上限** とし、25試行を超える再試行は禁止する。

---

## 2. ハッシュ・キャッシュ契約（事前テスト）

以下を探索実行前に契約テストとして検証する：
1. 同一戦略の `compute_algoforge_strategy_hash` 安定性。
2. 指標パラメータ・Risk 変更時の `strategy_hash` 変化。
3. データ・期間・コスト変更時の `evaluation_hash` 変化。
4. キャッシュヒット時のバックテスト実行回数 `0` 回。
5. 既存 `pipeline_v1` 経路・既存 `data/strategy_registry.db` の無変更維持（テスト用一時 SQLite DB を使用）。

---

## 3. 型安全性と比較意味論

- **価格系** (`close`, `ema`, `sma`, `bb_upper/lower`, `donchian_upper/lower`): 価格系同士のみで比較（例: `ema12 > ema48`, `close < bb20_lower`）。
- **オシレータ/モメンタム系** (`rsi`, `stochastic_k/d`, `adx` [0-100], `macd_line/signal`, `roc`): 同系列同士または妥当範囲の定数と比較（例: `rsi14 < 30`, `macd_line > macd_signal`）。
- **等価性**: NaN・等値・ウォームアップ・入れ子論理式について AlgoForge 逐次 `evaluate_conditions` との 100% 一致を検証し、浮動小数点比較の意味を事後変更しない。

---

## 4. 固定実行フローと即時停止条件

1. 生成試行消費（最大25） $\to$ 2. バリデーション $\to$ 3. 未対応時は Capability Request 保存・拒否 $\to$ 4. 正規化・`strategy_hash` $\to$ 5. ラン内重複判定 $\to$ 6. `evaluation_hash`・キャッシュ参照 $\to$ 7. 因果性/ベクトル化契約チェック $\to$ 8. バックテスト実行 $\to$ 9. 一時 Registry 保存 $\to$ 10. root cap 更新。
- **即時停止条件**: 先読み検出、条件評価不一致、root cap 違反（`> 3`）、既存 `strategy_registry.db` への意図しない書き込み。

---

## 5. 最終報告・保存項目

- `data/phase1b2_exploration_results.json` に固定シード `42`、データSHA、評価設定、コミット `cfe8126`、全25試行の生成順序・Strategy JSON・両ハッシュ・評価結果を保存。
- 生成試行数、生成成功数、ユニーク候補数、重複数、拒否数、Capability Request 数、キャッシュヒット数、新規バックテスト数、root 別生成数、実行時間、API 支出（`$0.00 USD`）を報告する。
