# HDE H0 — 収集・保存基盤 完了報告書

**Status:** Completed & Approved  
**Date:** 2026-11-15  
**Version:** 1.0 (Final)  
**Parent Specifications:**
- [architecture.md](file:///c:/Users/ksato/workspace/finance/research_pipeline/docs/architecture.md)
- [hypothesis_discovery_engine.md](file:///c:/Users/ksato/workspace/finance/research_pipeline/docs/hypothesis_discovery_engine.md)
- [H0-collection-audit.md](file:///c:/Users/ksato/workspace/finance/research_pipeline/docs/H0-collection-audit.md)
- [H0-implementation-plan-v0.4.md](file:///c:/Users/ksato/workspace/finance/research_pipeline/docs/H0-implementation-plan-v0.4.md)

---

## 1. 完了概要

StockNet Hypothesis Discovery Engine（HDE）の初期収集・保存基盤（H0）の実装および受入検証が完了し、正式に受入承認された。

- **総テスト数**: 27件（単体テスト25件、実API結合テスト2件）**全件 PASS（0失敗）**
- **実行時間**: 単体テスト約1.2秒、実API結合テスト約1.0秒
- **既存システム保護**: **AlgoForge のコード、DB、設定、Celeryキューへの変更はゼロ（無変更維持）**

---

## 2. 受入条件と検証結果の対応表

| 受入要件 | 実装仕様・対応内容 | 検証テスト | 結果 |
|---|---|---|---|
| **1. AlgoForge無侵害** | `research_pipeline` 内で完結。PostgreSQL/Redis/Celery に一切触れず独立 SQLite/Storage を使用。 | `git -C algoforge status` による変更ゼロ確認 | **PASS** |
| **2. 公式API優先** | arXiv 公式 API (Atom XML) と GitHub REST API v3 を使用。Stealth Playwright は対象外。 | `test_arxiv_collector.py`, `test_github_collector.py` | **PASS** |
| **3. 論理資料と版の分離** | `sources`（不変概念ID）と `source_versions`（原典版、格納パス、ハッシュ）を別テーブルで管理。 | `test_collection_service.py` | **PASS** |
| **4. 探索セッションと試行の分離** | `discovery_runs`（クエリ、ランキング、件数）と `collection_attempts`（成否、エラー分類）を分離。 | `test_collection_service.py` | **PASS** |
| **5. Abstract/README 中心** | 論文 PDF 全文ではなく Abstract と README を取得。README 欠落時は `metadata_only` として保存。 | `test_github_fetch_metadata_only_when_no_readme` | **PASS** |
| **6. アトミック保存と整合性** | `.tmp/` ステージング → `os.replace` アトミックリネーム → DB トランザクションコミット。失敗時は補償退避。 | `test_atomic_stage_and_commit`, `test_rollback_staged` | **PASS** |
| **7. クラッシュ復旧 (Reconciler)** | 起動時に `StorageReconciler` が孤立ディレクトリを `.orphan/` へ退避、消失ファイルを `missing_file` に更新。 | `test_reconciler_cleans_stale_tmp`, `test_reconciler_moves_orphan_directory` | **PASS** |
| **8. 排他制御 (StorageLock)** | ファイルロックによる排他制御。外部 API 通信中はロックを持たず、FS・DB クリティカルセクションのみ保護。 | `test_storage_lock_prevents_concurrent_access`, `test_runner_prevents_duplicate_execution` | **PASS** |
| **9. arXiv 版指定と検証 (R1)** | 版なしは最新要求。指定版と API 確定版が不一致の場合は `version_mismatch` で代用保存を拒絶。旧形式対応。 | `test_arxiv_fetch_version_mismatch`, `test_parse_arxiv_identifier` | **PASS** |
| **10. GitHub Commit 整合性 (R2)** | 数値 `repository_id` で論理資料管理。確定 SHA で `ref=<sha>` を指定して README 取得。複合ハッシュ算出。 | `test_github_fetch_success`, `test_github_composite_hash_deterministic` | **PASS** |
| **11. 内容変化の検知** | 同一版の再取得時に原典 API 内容が変わっていた場合、無言上書きせず `content_diverged` エラー記録。 | `test_collection_service_content_diverged` | **PASS** |
| **12. H1a 契約 (content_scope)** | `ResearchDocument` に `content_scope`（`abstract_only` 等）を付与し、要約からの主張を実証済みと誤認防止。 | `test_collection_service_arxiv_discovery` | **PASS** |
| **13. スケジューラ受入** | `--mode once` の完了、`--mode daemon` の定期実行、二重起動防止、再起動後の冪等性を実証。 | `test_runner_once_mode`, `test_runner_daemon_loop_intervals`, `test_runner_restart_idempotency` | **PASS** |
| **14. 冪等性受入** | 固定候補集合に対して連続実行し、2回目は新規収集0件・スキップ全件・ファイル/DBが完全不変。 | `test_collection_service_fixed_set_idempotency` | **PASS** |
| **15. 実API疎通** | 実 arXiv API（1706.03762）および実 GitHub API（octocat/Hello-World）との通信・パース正常性を確認。 | `test_live_arxiv_fetch_one`, `test_live_github_fetch_one` | **PASS** |

---

## 3. 実装ファイル一覧

```text
research_pipeline/
├── pyproject.toml                         # プロジェクト依存・pytest設定
├── .env.example                           # 環境設定テンプレート
├── src/research_pipeline/
│   ├── __init__.py
│   ├── config.py                          # 連絡先バリデーション・パス・定数
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── lock.py                        # OS排他ファイルロック (StorageLock)
│   │   ├── db.py                          # SQLite DDL・PRAGMA (WAL, foreign_keys)
│   │   ├── content_store.py               # アトミックファイル保存・パス検証・複合ハッシュ
│   │   ├── repository.py                  # SQLite CRUD
│   │   └── reconciler.py                  # Double-Check 型クラッシュ復旧 (StorageReconciler)
│   ├── collectors/
│   │   ├── __init__.py
│   │   ├── models.py                      # ドメインモデル・エラー例外クラス
│   │   ├── base.py                        # BaseCollector
│   │   ├── rate_limiter.py                # ヘッダー監視型レート制限・最大待機制御
│   │   ├── arxiv.py                       # arXiv 公式 API (新旧ID・版検証)
│   │   └── github.py                      # GitHub REST API (Repo ID・ref指定・複合ハッシュ)
│   ├── service/
│   │   ├── __init__.py
│   │   ├── contracts.py                   # H1a 向け不変スキーマ (ResearchDocument)
│   │   └── collection_service.py          # 収集オーケストレーション・二相コミット
│   └── scheduler/
│       ├── __init__.py
│       └── runner.py                      # CLI & 常駐デーモン実行エントリ
└── tests/
    ├── conftest.py                        # 一時環境・フィクスチャ設定
    ├── fixtures/                          # 固定候補集合モック
    │   ├── arxiv_atom_samples.xml
    │   └── github_api_samples.json
    ├── unit/                              # 単体テスト (25件)
    │   ├── test_identifiers.py
    │   ├── test_content_store.py
    │   ├── test_lock.py
    │   ├── test_reconciler.py
    │   ├── test_arxiv_collector.py
    │   ├── test_github_collector.py
    │   ├── test_collection_service.py
    │   └── test_runner.py
    └── integration/                       # 外部実API結合テスト (2件)
        ├── test_arxiv_real.py
        └── test_github_real.py
```

---

## 4. 既知の制約と運用上の留意点

1. **全文 PDF 解析の対象外**:
   - H0 では Abstract および README のみを対象としており、PDF 本文の取得・解析は行わない（H1a 以降で必要に応じて拡張）。
2. **生データ複合ハッシュの定義**:
   - GitHub の `raw_content_hash` は、受信した raw wire bytes ではなく、キーソート済み・標準セパレータで正規化された JSON バイト列の複合ハッシュである。
3. **未認証時の GitHub レート制限**:
   - `GITHUB_TOKEN` 未設定時は 60 req/h のレート制限が適用される。実運用時は `.env` へのトークン設定を推奨。

---
**Approval Recorded by:** Antigravity Agent  
**Next Phase:** Step H1a — Capability Extraction 計画策定
