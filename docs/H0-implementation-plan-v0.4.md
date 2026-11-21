# HDE H0 — 収集・保存基盤 最終実装計画書 v0.4

**Status:** Ready for Final Implementation Approval  
**Date:** 2026-11-15  
**Version:** 0.4  
**Target:** Hypothesis Discovery Engine (HDE) Phase 1 / Step H0  
**Parent Specifications:**
- [architecture.md](file:///c:/Users/ksato/workspace/finance/research_pipeline/docs/architecture.md)
- [hypothesis_discovery_engine.md](file:///c:/Users/ksato/workspace/finance/research_pipeline/docs/hypothesis_discovery_engine.md)
- [H0-collection-audit.md](file:///c:/Users/ksato/workspace/finance/research_pipeline/docs/H0-collection-audit.md)
- [H0-implementation-plan-v0.3.md](file:///c:/Users/ksato/workspace/finance/research_pipeline/docs/H0-implementation-plan-v0.3.md)

---

## 1. 概要と修正方針（R1〜R3の完全反映）

本計画書は、実装計画書 v0.3 に対する承認前レビュー（R1: arXiv版指定、R2: GitHub Commit整合性、R3: Reconciler排他制御、および各種運用安全対策）を反映した最終仕様である。

### 1.1 主要修正の要点
1. **R1. arXiv の版指定と厳密な検証**:
   - 版番号なし URL は「最新版（latest）」要求として解決する。
   - 明示された版番号（例: `v1`）は「要求版（`requested_version`）」として保持する。
   - 取得結果の版が要求版と一致しない場合は「版不一致（`version_mismatch`）」としてエラー処理し、最新版で代用して保存することは厳禁とする。
   - 過去版取得が API 上サポートされない場合は「取得不能（`unsupported_version`）」を明示して記録する。
2. **R2. GitHub の Commit 整合性と生データ複合ハッシュ**:
   - リポジトリの不変識別子（`repository_id`）を特定した後、対象の Commit SHA を最初に確定する。
   - README 取得時は必ず **`ref=<commit_sha>`** を明示的に付与してリクエストし、コミットと README の時系列ズレを完全に排除する。
   - README 欠落（404）の判定も当該 Commit SHA を対象に行う。
   - 生データハッシュ（`raw_content_hash`）は、Repository メタデータ JSON と README レスポンス（または欠落メタデータ）を結合した決定論的複合ハッシュとする。
3. **R3. Reconciler と CollectionService の共通排他制御**:
   - 手動実行、定期実行、および整合性復旧処理（Reconciler）は、共通のグローバルストレージロック（`StorageLock`）を取得して実行する。
   - `.tmp/` の削除および孤立確定ディレクトリの退避は、ロック取得後に実状態を再確認（Double-Check）して実行する。
   - 「ファイル移動直後・DBコミット前」のインフライト状態における競合シミュレーションテストを追加する。
4. **運用安全対策の追加**:
   - **連絡先バリデーション**: User-Agent に含める連絡先（`HDE_CONTACT_EMAIL`）の設定を必須化。
   - **パストラバーサル防御**: `source_id`, `version_tag`, `canonical_identifier` に対する厳格なバリデーション（`..`, `/`, `\` の禁止）。
   - **レート制限待機上限**: `MAX_RATE_LIMIT_WAIT_SECONDS`（デフォルト 120 秒）を設け、上限を超える場合は待機せず安全中断。
   - **同一版内容差異テスト**: 同一 Commit / 同一 arXiv 版の再取得時に内容差異が生じた場合の改ざん検知テストを定義。

---

## 2. ディレクトリ構成

`research_pipeline` ワークスペース内に以下を配置する（AlgoForge 側には一切触れない）。

```text
research_pipeline/
├── docs/
│   ├── architecture.md
│   ├── hypothesis_discovery_engine.md
│   ├── H0-collection-audit.md
│   ├── H0-implementation-plan-v0.2.md
│   ├── H0-implementation-plan-v0.3.md
│   └── H0-implementation-plan-v0.4.md
├── data/                                # ローカル永続化領域（.gitignore）
│   ├── hde.db                           # SQLite データベース正本
│   ├── hde.lock                         # グローバル排他ロックファイル
│   └── storage/                         # コンテンツストレージ正本
│       ├── .tmp/                        # アトミック書き込み用ステージング
│       ├── .orphan/                     # クラッシュ復旧時の退避領域
│       ├── papers/                      # 論文データ
│       │   └── {source_id}/
│       │       └── {version_tag}/
│       │           ├── raw_entry.xml    # 取得した生 Atom XML エントリ
│       │           ├── metadata.json    # 構造化メタデータ
│       │           └── abstract.txt     # 抽出済み Abstract
│       └── oss/                         # OSS データ
│           └── {source_id}/
│               └── {commit_sha}/
│                   ├── raw_repo.json    # GitHub API 生 JSON
│                   ├── raw_readme.json  # GitHub README API 生 JSON (存在時)
│                   ├── metadata.json    # 構造化メタデータ
│                   └── readme.md        # 抽出済み README (存在時)
├── src/
│   └── research_pipeline/
│       ├── __init__.py
│       ├── config.py                    # 設定値・パス・制限値・連絡先・定数
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── lock.py                  # グローバル排他ロック (StorageLock)
│       │   ├── db.py                    # SQLite 接続・DDL・マイグレーション
│       │   ├── repository.py            # SQLite CRUD 操作
│       │   ├── content_store.py         # アトミックファイル保存・パストラバーサル防御
│       │   └── reconciler.py            # ロック内 Double-Check 型整合性復旧
│       ├── collectors/
│       │   ├── __init__.py
│       │   ├── base.py                  # CollectorBase 抽象基底クラス
│       │   ├── models.py                # 内部ドメインモデル (SearchQuery, CandidateRef 等)
│       │   ├── arxiv.py                 # arXiv 公式 API (新旧ID・版指定検証)
│       │   ├── github.py                # GitHub REST API (Repo ID・ref指定・複合ハッシュ)
│       │   └── rate_limiter.py          # ヘッダー監視 & 待機上限付きレート制御
│       ├── service/
│       │   ├── __init__.py
│       │   ├── collection_service.py     # 収集オーケストレーション・排他制御・整合性担保
│       │   └── contracts.py             # H1a 引き渡しスキーマ (ResearchDocument)
│       └── scheduler/
│           ├── __init__.py
│           └── runner.py                # 定期実行ループ・CLI エントリ
├── tests/
│   ├── __init__.py
│   ├── conftest.py                      # テスト用一時DB・ストレージ初期化
│   ├── fixtures/                        # 固定候補集合・モックデータ
│   │   ├── arxiv_atom_samples.xml
│   │   └── github_api_samples.json
│   ├── unit/                            # モックによる完全自動単体テスト
│   │   ├── test_identifiers.py          # arXiv/GitHub ID正規化・パストラバーサル検証
│   │   ├── test_content_store.py        # アトミック保存・複合ハッシュ計算
│   │   ├── test_reconciler.py           # ロック競合・Double-Check・孤立退避
│   │   ├── test_arxiv_collector.py      # arXiv 版指定・版不一致・旧形式対応
│   │   ├── test_github_collector.py     # GitHub ref指定・README欠落・複合ハッシュ
│   │   ├── test_collection_service.py   # 重複スキップ・改名・部分失敗・待機上限
│   │   └── test_lock.py                 # StorageLock 排他制御検証
│   └── integration/                     # 外部実API疎通テスト (Opt-in)
│       ├── test_arxiv_real.py
│       └── test_github_real.py
├── pyproject.toml
└── .env.example
```

---

## 3. SQLite DDL と制約

データベースファイルは `data/hde.db` とする。接続初期化時に以下の PRAGMA を必ず実行する。
```sql
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;
PRAGMA busy_timeout = 5000;
```

### 3.1 スキーマ定義 DDL

```sql
-- 1. 論理資料テーブル
CREATE TABLE IF NOT EXISTS sources (
    source_id               TEXT PRIMARY KEY,                  -- 例: "arxiv:2301.12345", "github:repo:28457823"
    source_type             TEXT NOT NULL CHECK (source_type IN ('paper', 'oss')),
    canonical_identifier    TEXT NOT NULL,                     -- arXiv: "2301.12345", GitHub: "28457823" (不変ID)
    primary_name            TEXT NOT NULL,                     -- 最新の論理名（論文タイトル、または最新 "owner/repo"）
    primary_url             TEXT NOT NULL,                     -- 代表アクセスURL
    created_at              TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at              TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_sources_type_canonical ON sources(source_type, canonical_identifier);

-- 2. リポジトリ・資料別名履歴テーブル（改名追跡用）
CREATE TABLE IF NOT EXISTS source_aliases (
    alias_id                INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id               TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
    alias_name              TEXT NOT NULL,                     -- 例: "openai/gym", "farama-foundation/gymnasium"
    is_current              INTEGER NOT NULL DEFAULT 1 CHECK (is_current IN (0, 1)),
    detected_at             TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    CONSTRAINT uq_source_alias UNIQUE (source_id, alias_name)
);

CREATE INDEX IF NOT EXISTS idx_aliases_name ON source_aliases(alias_name);

-- 3. 資料版テーブル（特定の版、コミット、Abstract/README 実体）
CREATE TABLE IF NOT EXISTS source_versions (
    version_id              TEXT PRIMARY KEY,                  -- 例: "arxiv:2301.12345:v2", "github:repo:28457823:a1b2c3d..."
    source_id               TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
    upstream_version        TEXT NOT NULL,                     -- 原典版 (arXiv: "v2", GitHub: 40桁SHA)
    requested_version       TEXT,                              -- 取得要求時に指定された版 ("v1" 等。最新要求時は NULL)
    content_scope           TEXT NOT NULL CHECK (content_scope IN ('abstract_only', 'readme_only', 'metadata_only', 'full_text')),
    raw_content_hash        TEXT NOT NULL,                     -- 生レスポンス（生XMLまたはRepo+README生JSON）の複合SHA-256
    extracted_text_hash     TEXT,                              -- 抽出本文 (Abstract/README) の SHA-256 (metadata_only時はNULL可)
    extractor_version       TEXT NOT NULL,                     -- 抽出器版 (例: "arxiv:1.0.0", "github_readme:1.0.0")
    storage_rel_path        TEXT NOT NULL,                     -- 相対パス (例: "papers/arxiv_2301.12345/v2")
    metadata_json           TEXT NOT NULL,                     -- 著者、ライセンス、URL等の生メタデータJSON
    published_at            TEXT,                              -- 原典の発行日時 / コミット日時 (ISO8601)
    status                  TEXT NOT NULL CHECK (status IN ('active', 'missing_file', 'corrupted')),
    collected_at            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    CONSTRAINT uq_source_upstream UNIQUE (source_id, upstream_version)
);

CREATE INDEX IF NOT EXISTS idx_source_versions_source_id ON source_versions(source_id);
CREATE INDEX IF NOT EXISTS idx_source_versions_text_hash ON source_versions(extracted_text_hash);

-- 4. 探索実行セッションテーブル
CREATE TABLE IF NOT EXISTS discovery_runs (
    run_id                  TEXT PRIMARY KEY,                  -- UUIDv4
    mode                    TEXT NOT NULL CHECK (mode IN ('continuous', 'problem_driven')),
    collector_type          TEXT NOT NULL CHECK (collector_type IN ('arxiv', 'github')),
    collector_version       TEXT NOT NULL,                     -- コレクター実装バージョン
    search_parameters_json  TEXT NOT NULL,                     -- クエリ、ソート、フィルタ等の完全なJSON
    agent_id                TEXT,                              -- 呼び出し元識別子
    status                  TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed', 'partial')),
    search_results_json     TEXT,                              -- 返却された候補一覧 (rank, identifier, title)
    items_found             INTEGER NOT NULL DEFAULT 0,
    items_collected         INTEGER NOT NULL DEFAULT 0,
    items_skipped           INTEGER NOT NULL DEFAULT 0,
    error_message           TEXT,
    started_at              TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    completed_at            TEXT
);

-- 5. 取得試行・失敗追跡テーブル
CREATE TABLE IF NOT EXISTS collection_attempts (
    attempt_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                  TEXT NOT NULL REFERENCES discovery_runs(run_id) ON DELETE CASCADE,
    target_identifier       TEXT NOT NULL,                     -- 試行対象のURLまたは入力識別子
    resolved_source_id      TEXT,                              -- 解決された sources(source_id)
    resolved_version        TEXT,                              -- 解決された upstream_version
    attempt_number          INTEGER NOT NULL DEFAULT 1,
    status                  TEXT NOT NULL CHECK (status IN ('success', 'skipped_duplicate', 'failed')),
    error_category          TEXT CHECK (error_category IN ('transient', 'rate_limit', 'permanent', 'validation', 'version_mismatch')),
    error_message           TEXT,
    http_status             INTEGER,
    duration_ms             INTEGER,
    attempted_at            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_attempts_run_id ON collection_attempts(run_id);
CREATE INDEX IF NOT EXISTS idx_attempts_target ON collection_attempts(target_identifier);
```

---

## 4. 資料ID・版ID・エイリアス判定規則（R1, R2対応）

### 4.1 パストラバーサル防御（全識別子共通バリデーション）
ファイルシステム上のパス生成に用いられる全ての文字列（`source_id`, `version_tag`, `canonical_identifier`）は、以下の文字種のみを許可し、`..` やスラッシュ、バックスラッシュを含む場合は即座に `ValueError` を送出する。
- 許可パターン: `^[a-zA-Z0-9_\-\.:]+$`（1文字〜128文字）

### 4.2 論文（arXiv）の版指定と解決規則（R1）
1. **URL / 識別子の解析**:
   - 新形式: `\d{4}\.\d{4,5}(v\d+)?`（例: `2301.12345`, `2301.12345v1`）。
   - 旧形式: `[a-z\-]+(\.[A-Z]{2})?/\d{7}(v\d+)?`（例: `math/0309099`, `hep-th/9912012v2`）。
   - 末尾に `v\d+` が明示されている場合: `requested_version = "v1"` などの要求版として保持。
   - 末尾に `v\d+` がない場合: `requested_version = None`（最新版要求として扱う）。
   - `canonical_identifier` = `{base_id}`（例: `2301.12345`, `math/0309099`）。
   - `source_id` = `arxiv:{canonical_identifier}`。
2. **API レスポンス（Atom XML）との突合**:
   - Atom XML 内の `<id>` タグ（例: `http://arxiv.org/abs/2301.12345v3`）から実際の版 `resolved_version = "v3"` を抽出。
3. **版不一致（`version_mismatch`）の判定**:
   - もし `requested_version` が指定されており（例: `"v1"`）、`resolved_version` と一致しない場合（例: API が最新 `"v3"` のみを返し過去版が返せない場合）:
     - **最新版で代用して保存することは厳禁**。
     - `collection_attempts` に `status='failed'`, `error_category='version_mismatch'`, `error_message='Requested version v1 but API returned v3'` を記録し、保存処理を中止。
4. **最新版要求時の確定**:
   - `requested_version is None` の場合は、API から返却された `resolved_version`（例: `"v3"`）を `upstream_version` として正常採用。
   - `version_id` = `arxiv:{canonical_identifier}:{upstream_version}`。

### 4.3 OSS（GitHub）の Commit 整合性と複合ハッシュ（R2）
1. **論理資料 ID の確定（`repository_id`）**:
   - 初回取得時、GitHub REST API (`GET /repos/{owner}/{repo}`) を呼び出し、不変の数値 `id`（例: `28457823`）を取得。
   - `canonical_identifier` = `str(repo_json["id"])`。
   - `source_id` = `github:repo:{canonical_identifier}`。
2. **対象 Commit SHA の確定**:
   - リポジトリ情報レスポンスから、対象とする 40桁フルコミット SHA を最初に確定する（指定タグ/コミット、または `GET /repos/{owner}/{repo}/commits/{default_branch}` の SHA）。
   - `upstream_version` = `commit_sha`。
3. **README の `ref` 指定取得**:
   - README 取得時は、**必ず `ref=<commit_sha>` クエリパラメータを付与** してリクエストする。
     - エンドポイント: `GET /repos/{owner}/{repo}/readme?ref={commit_sha}`
   - これにより、デフォルトブランチが直後に進んだ場合でも、確定 Commit 時点と 100% 整合した README を取得する。
4. **README 欠落時の判定**:
   - 上記の `ref={commit_sha}` を指定したリクエストが **HTTP 404** を返した場合のみ、「当該 Commit において README 欠落」と正式判定。
   - `content_scope = 'metadata_only'` として処理し、`readme.md` ファイルは生成しない。
5. **生データ複合ハッシュ（`raw_content_hash`）の計算**:
   - 整合性を完全担保するため、以下の 2 つの生データを結合したバイト列の SHA-256 を計算する。
     - 生 Repo JSON バイト列（ソート済み正規化 JSON）
     - 生 README API レスポンス JSON バイト列（README 欠落時は空文字列バイト列）
     - `raw_content_hash = sha256(raw_repo_bytes + b"\n---HDE_RAW_SEP---\n" + raw_readme_bytes)`

---

## 5. コレクター仕様とレート制限

### 5.1 設定と連絡先バリデーション
- `config.py` に `HDE_CONTACT_EMAIL` を定義。
  - 起動時にメールアドレス形式（`^[\w\.-]+@[\w\.-]+\.\w+$`）を検証し、未設定または不正な場合は起動をブロック（`ConfigurationError`）。
- User-Agent 定義: `research_pipeline/0.1.0 ({HDE_CONTACT_EMAIL})`。

### 5.2 レート制限と待機上限（`MAX_RATE_LIMIT_WAIT_SECONDS`）
- `config.MAX_RATE_LIMIT_WAIT_SECONDS = 120.0`（秒）。
- **arXiv API**:
  - 連続リクエスト間の最小間隔: **3.5 秒**（Polite Delay）。
- **GitHub API**:
  - レスポンスヘッダー `X-RateLimit-Remaining`, `X-RateLimit-Reset` を毎リクエスト解析。
  - 残枠が枯渇した場合（`Remaining == 0`）または HTTP 403 / 429（Secondary Rate Limit）受信時:
    - 必要なスリープ時間 `wait_seconds = max(0, reset_timestamp - now)`（または `Retry-After`）。
    - **`wait_seconds > config.MAX_RATE_LIMIT_WAIT_SECONDS` の場合**:
      - 待機せず、ログに `Rate limit wait time (Xs) exceeds limit. Aborting run.` を記録。
      - 当該アイテムの `collection_attempts` に `error_category='rate_limit'` を記録し、探索セッション（`discovery_runs`）を `status='partial'` で安全中断。
    - **`wait_seconds <= config.MAX_RATE_LIMIT_WAIT_SECONDS` の場合**:
      - 指数ジッター付きで当該時間スリープした後に 1 回再試行。

---

## 6. 排他制御アーキテクチャ（R3対応）

### 6.1 グローバルストレージロック（`StorageLock`）
`CollectionService` による書き込みと、`StorageReconciler` による修復処理、および外部 CLI による実行が同時にファイルシステムおよび SQLite を変更しないよう、共通のロック機構を実装する。

- ロックファイルパス: `data/hde.lock`
- **ロックの実装（`research_pipeline.storage.lock.StorageLock`）**:
  - OS レベルの排他ファイルロックを使用（Windows: `msvcrt.locking`, POSIX: `fcntl.flock`）。
  - ロックファイル内には保持元のプロセス情報（PID、ホスト名、操作種別 `collection` or `reconciliation`、開始時刻）を JSON で書き込む。
  - コンテキストマネージャ `async with StorageLock(operation="collection"):` として利用。
  - ロック取得失敗時は `StorageLockBusyError` を送出し、後続処理を一切ブロックして安全終了する。

### 6.2 ロック内 Double-Check 型整合性復旧（`StorageReconciler`）
Reconciler が `.tmp/` の削除や孤立ファイルの退避を行う際は、競合を避けるため以下の手順を厳守する。

```text
[StorageReconciler.reconcile()]
      │
      ▼
1. StorageLock(operation="reconciliation") を取得 (排他ロック)
      │
      ├─ 2. .tmp/ 一時ディレクトリのクリーンアップ (Double-Check)
      │       ├─ ロック取得後に .tmp/ 配下のディレクトリ mtime を再確認
      │       └─ ロック取得時点でもなお作成後 1 時間以上経過しているもののみを削除
      │
      ├─ 3. 孤立ファイル (Orphan) の安全退避 (Double-Check)
      │       ├─ storage/ 配下の各ディレクトリを走査
      │       ├─ DB (source_versions) と照合
      │       ├─ DB に該当レコードが存在しない場合:
      │       │     └─ ロック保持下で DB を再クエリ (直前にコミット完了していないか再確認)
      │       │     └─ 依然として DB に存在しない場合のみ、data/storage/.orphan/{timestamp}_{name} に退避移動
      │       └─ 警告ログを出力
      │
      ├─ 4. 欠落ファイル (Missing File) の検知
      │       ├─ DB (source_versions.status='active') の全件について実ファイル存在確認
      │       └─ 消失しているレコードの status を 'missing_file' に更新
      │
      ├─ 5. 破損ファイル (Corrupted File) の検知
      │       ├─ 実ファイルの SHA-256 と DB の extracted_text_hash / raw_content_hash を突合
      │       └─ 不一致時は DB status を 'corrupted' に更新し、実ファイルを .orphan/ へ退避
      │
      ▼
6. StorageLock を解放
```

---

## 7. ファイル保存の状態遷移とアトミック性

```text
[State: START]
      │
      ▼ (一時ディレクトリ作成)
[State: STAGED_TMP] (data/storage/.tmp/{uuid}/)
      │  ├─ 生データ (raw_entry.xml または raw_repo.json + raw_readme.json) 書き出し
      │  ├─ 抽出テキスト (abstract.txt / readme.md) 書き出し (存在時)
      │  ├─ metadata.json 書き出し
      │  ├─ 各ファイルの fsync 実行
      │  └─ raw_content_hash (複合ハッシュ), extracted_text_hash を計算
      │
      ▼ (os.replace によるアトミック移動)
[State: MOVED_TO_FINAL] (data/storage/{type}/{source_id}/{version}/)
      │  ※ ディレクトリ移動完了。ただし DB は未コミット。
      │
      ▼ (SQLite トランザクション)
   BEGIN IMMEDIATE TRANSACTION;
      │  ├─ INSERT INTO sources ...
      │  ├─ INSERT INTO source_aliases ... (GitHub時)
      │  ├─ INSERT INTO source_versions ... (status='active')
      │  └─ INSERT INTO collection_attempts (status='success')
   COMMIT;
      │
      ├─────────────────────────────────────────┐
      │ (コミット成功)                           │ (コミット失敗・クラッシュ)
      ▼                                         ▼
[State: COMMITTED] (正常完了)             [State: DB_FAILED] (補償処理)
                                                │  ├─ 確定ディレクトリを .orphan/{uuid}/ へ退避
                                                │  ├─ collection_attempts に failed 記録
                                                │  └─ ロールバック完了
```

---

## 8. H1a への引き渡し契約 (`ResearchDocument`)

H1a（Capability Extraction）に引き渡すイミュータブルな契約モデルを定義する。

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Literal

class ResearchDocument(BaseModel):
    """H0 から H1a に引き渡される構造化研究ドキュメント"""
    source_id: str = Field(..., description="論理資料ID (例: 'arxiv:2301.12345', 'github:repo:28457823')")
    version_id: str = Field(..., description="特定版ID (例: 'arxiv:2301.12345:v3', 'github:repo:28457823:a1b2c3...')")
    source_type: Literal["paper", "oss"] = Field(..., description="資料種別")
    canonical_identifier: str = Field(..., description="不変原典識別子 ('2301.12345', '28457823')")
    primary_name: str = Field(..., description="論文タイトル、または最新リポジトリ名 (owner/repo)")
    primary_url: str = Field(..., description="原典Web URL")
    upstream_version: str = Field(..., description="確定版識別子 ('v3', 40桁コミットSHA)")
    requested_version: Optional[str] = Field(None, description="要求された版識別子")
    
    # ── H1a 認識境界 (重要) ────────────────────────────────────────────────
    content_scope: Literal["abstract_only", "readme_only", "metadata_only", "full_text"] = Field(
        ...,
        description="テキストの範囲。H1a は abstract_only / readme_only の場合、主張を 'reported' として扱い、実証済みと見なさない。"
    )
    
    # ── コンテンツ本文と整合性 ──────────────────────────────────────────
    content_text: Optional[str] = Field(None, description="AbstractまたはREADME本文 (metadata_only時はNone)")
    raw_content_hash: str = Field(..., description="取得生データ (XMLまたはRepo+README複合) の SHA-256")
    extracted_text_hash: Optional[str] = Field(None, description="content_text の SHA-256 (テキスト存在時)")
    extractor_version: str = Field(..., description="抽出器バージョン")
    
    # ── メタデータ & 追跡 ────────────────────────────────────────────────
    published_at: Optional[str] = Field(None, description="原典公開日時 (ISO8601)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="構造化メタデータ")
    local_artifact_dir: str = Field(..., description="ローカル保存先絶対パス")
    collected_at: str = Field(..., description="収集完了日時 (ISO8601)")
```

---

## 9. テスト計画と受入基準

### 9.1 単体テスト（モック環境: `pytest tests/unit`）

1. **arXiv 版指定・版不一致テスト (`test_arxiv_collector.py`)**:
   - 版番号なし要求時、API レスポンスから確定した版（例: `v3`）が正しく採用されること。
   - `v1` を明示要求したのに API が `v2` を返した場合、`version_mismatch` エラーとなり、DB に保存されないこと。
   - 旧形式 ID（`math/0309099`）が正しくパースされ、API リクエストが構築されること。
2. **GitHub ref 指定・README 欠落・複合ハッシュテスト (`test_github_collector.py`)**:
   - README API 呼び出し時に `ref=<commit_sha>` パラメータが必ず付与されていること。
   - README が 404 の場合、`content_scope='metadata_only'` として正常保存されること。
   - Repo JSON と README レスポンスから複合ハッシュが正しく算出されること。
3. **パストラバーサル防御テスト (`test_identifiers.py`)**:
   - `source_id` や `version_tag` に `../../etc/passwd` や不正文字が含まれる場合、検証で拒絶されること。
4. **Reconciler 排他制御 & インフライト競合テスト (`test_reconciler.py`)**:
   - `StorageLock` 保持中は別の収集または Reconciler が起動できないこと。
   - 「ファイル移動完了・DBコミット前」の状態をモックし、Reconciler がロック取得後に DB を Double-Check して誤って退避しないこと。
   - 作成後 1 時間未満の一時ファイルが `.tmp/` から削除されないこと。
5. **同一版の内容差異テスト (`test_content_store.py`)**:
   - 同一の `version_id` なのに保存されている実ファイルが外部から改ざんされた場合、Reconciler が `corrupted` を検知して退避すること。
6. **レート制限待機上限テスト (`test_collection_service.py`)**:
   - `Retry-After` が 300 秒（待機上限 120 秒超）の場合、スリープを行わずに `partial` で安全中断されること。

### 9.2 固定候補集合による冪等性受入テスト
固定フィクスチャ（arXiv 2件、GitHub 2件）を投入し、2回連続実行してストレージとDBの完全不変性を検証する。

- **受入基準**:
  1. 1回目実行: 全4件が `active` 保存され、ファイルハッシュが一致。
  2. 2回目実行: 全4件が `skipped_duplicate` となり、新規ファイル書き込みが 0 件、DB 行数の増加が 0 件であること。

### 9.3 結合テスト（Opt-in: `pytest tests/integration --run-external`）
- 手動フラグ指定時のみ、実 arXiv API および実 GitHub API に疎通し、1件ずつ正常取得を確認。

---

## 10. 実装順序

```text
Step 1: プロジェクト基本セットアップ
        - pyproject.toml（httpx, pydantic, aiosqlite 等）
        - src/research_pipeline/config.py（連絡先バリデーション含む）
        - tests/conftest.py & tests/fixtures/

Step 2: 排他ロック・永続化層・Reconciler 実装
        - src/research_pipeline/storage/lock.py（StorageLock）
        - src/research_pipeline/storage/db.py（DDL適用）
        - src/research_pipeline/storage/content_store.py（アトミック保存・複合ハッシュ・パス検証）
        - src/research_pipeline/storage/reconciler.py（Double-Check型整合性復旧）
        - tests/unit/test_lock.py, test_content_store.py, test_reconciler.py で検証

Step 3: Collector 基底 & レート制限実装
        - src/research_pipeline/collectors/base.py, models.py
        - src/research_pipeline/collectors/rate_limiter.py（待機上限・ヘッダー監視）
        - tests/unit/test_identifiers.py で検証

Step 4: arXiv Collector 実装
        - src/research_pipeline/collectors/arxiv.py（新旧ID・版指定検証・版不一致エラー）
        - tests/unit/test_arxiv_collector.py で検証

Step 5: GitHub Collector 実装
        - src/research_pipeline/collectors/github.py（Repo ID・ref指定・README 404・複合ハッシュ）
        - tests/unit/test_github_collector.py で検証

Step 6: CollectionService & H1a 契約実装
        - src/research_pipeline/service/contracts.py（ResearchDocument, content_scope）
        - src/research_pipeline/service/collection_service.py（排他制御・二相処理・部分失敗）
        - tests/unit/test_collection_service.py で検証

Step 7: スケジューラ Runner & 総合受入検証
        - src/research_pipeline/scheduler/runner.py（CLI & Daemon）
        - 固定候補集合による冪等性受入テストの実行
        - Opt-in 外部結合テストの実行
```

---

## 11. 未決定事項

1. **GitHub PAT の任意配備**: 未認証（60 req/h）でも待機上限とレート監視で安全動作するが、実運用時の PAT 設定手順をドキュメント化。
2. **全文 PDF 取得（L2拡張）**: Phase 1（H1a 完了後）に、Abstract のみではメカニズム特定が不十分な文献に対するオンデマンド取得インターフェースを検討。

---
**Plan Prepared by:** Antigravity Agent  
**Awaiting Review:** 最終計画 v0.4 の承認判定待ち
