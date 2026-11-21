# HDE H0 — 収集・保存基盤 最終実装計画書 v0.3

**Status:** Ready for Implementation Approval  
**Date:** 2026-11-15  
**Version:** 0.3  
**Target:** Hypothesis Discovery Engine (HDE) Phase 1 / Step H0  
**Parent Specifications:**
- [architecture.md](file:///c:/Users/ksato/workspace/finance/research_pipeline/docs/architecture.md)
- [hypothesis_discovery_engine.md](file:///c:/Users/ksato/workspace/finance/research_pipeline/docs/hypothesis_discovery_engine.md)
- [H0-collection-audit.md](file:///c:/Users/ksato/workspace/finance/research_pipeline/docs/H0-collection-audit.md)
- [H0-implementation-plan-v0.2.md](file:///c:/Users/ksato/workspace/finance/research_pipeline/docs/H0-implementation-plan-v0.2.md)

---

## 1. 概要と最終修正方針

本計画書は、実装計画書 v0.2 に対するレビュー指摘事項（11項目）を完全に反映した、HDE H0（収集・保存基盤）の最終実装仕様である。

### 1.1 最終修正の要点
1. **arXiv 版解決と旧形式対応**: 版番号なし URL を `v1` と決め打ちせず、API 応答の完全識別子から確定版を抽出。`math/0309099` などの旧形式 ID も正規化。
2. **GitHub 不変リポジトリ ID**: リポジトリの改名・移転に対応するため、GitHub の数値 `repository_id` を論理資料 ID とし、`owner/repo` はエイリアス履歴として管理。
3. **FS/DB 非原子性のクラッシュ復旧**: ファイルシステムと SQLite の不整合を防ぐため、ステージング遷移を明確化し、起動時に自動整合性検査・孤立ファイル修復を行う Reconciler を実装。
4. **4種ハッシュ・版管理の厳密化**: 原典版（`upstream_version`）、生データハッシュ（`raw_content_hash`）、抽出テキストハッシュ（`extracted_text_hash`）、抽出器版（`extractor_version`）を完全分離。
5. **再現性のための探索記録**: `discovery_runs` に検索パラメータ全文、返却候補の順位（Rank）・識別子一覧、Collector 版を記録。
6. **GitHub Secondary Rate Limit 対応**: 固定 sleep に頼らず、`X-RateLimit-*` ヘッダー監視、`Retry-After` 解析、指数バックオフを実装。
7. **README 欠落の正常処理**: README が存在しないリポジトリをエラーにせず、`metadata_only` として安全に永続化。
8. **軽量スケジューラ & 二重起動防止**: 単一プロセスのロックファイル（PID 記録・排他ロック）による多重起動防止。
9. **H1a 契約における `content_scope` の新設**: 分析対象が全文か要約（Abstract/README）かメタデータのみかを明示し、エビデンス状態の誤認を防止。
10. **固定候補集合による受入基準**: 動的検索ではなく固定フィクスチャを用いた完全決定論的テストによる冪等性検証。
11. **エッジケーステスト網羅**: クラッシュ復旧、版未確定、改名、部分失敗、同一版のコンテンツ差異に関する単体・統合テストを設計。

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
│   └── H0-implementation-plan-v0.3.md
├── data/                                # ローカル永続化領域（.gitignore）
│   ├── hde.db                           # SQLite データベース正本
│   ├── hde.lock                         # 二重起動防止ロックファイル
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
│                   ├── metadata.json    # 構造化メタデータ
│                   └── readme.md        # 抽出済み README（存在する場合のみ）
├── src/
│   └── research_pipeline/
│       ├── __init__.py
│       ├── config.py                    # 設定値・パス・制限値・バージョン定数
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── db.py                    # SQLite 接続・DDL・マイグレーション
│       │   ├── repository.py            # SQLite CRUD 操作
│       │   ├── content_store.py         # アトミックファイル保存・ハッシュ計算
│       │   └── reconciler.py            # クラッシュ後の整合性検査・復旧
│       ├── collectors/
│       │   ├── __init__.py
│       │   ├── base.py                  # CollectorBase 抽象基底クラス
│       │   ├── models.py                # 内部ドメインモデル
│       │   ├── arxiv.py                 # arXiv 公式 API（新旧ID・版解決）
│       │   ├── github.py                # GitHub REST API（Repo ID・Secondary Rate Limit）
│       │   └── rate_limiter.py          # ヘッダー監視型レート制限・Polite Delay
│       ├── service/
│       │   ├── __init__.py
│       │   ├── collection_service.py     # 収集オーケストレーション・整合性制御
│       │   └── contracts.py             # H1a 引き渡しスキーマ (ResearchDocument)
│       └── scheduler/
│           ├── __init__.py
│           ├── lock.py                  # プロセスロック（二重起動防止）
│           └── runner.py                # 定期実行ループ・CLI エントリ
├── tests/
│   ├── __init__.py
│   ├── conftest.py                      # テスト用一時DB・ストレージ初期化
│   ├── fixtures/                        # 固定候補集合・モックデータ
│   │   ├── arxiv_atom_samples.xml
│   │   └── github_api_samples.json
│   ├── unit/                            # モックによる完全自動単体テスト
│   │   ├── test_identifiers.py          # arXiv/GitHub ID正規化・版解決
│   │   ├── test_content_store.py        # アトミック保存・ハッシュ計算
│   │   ├── test_reconciler.py           # クラッシュ復旧・不整合検知
│   │   ├── test_arxiv_collector.py      # arXiv パース（新旧形式・版確定）
│   │   ├── test_github_collector.py     # GitHub レート制限・README欠落
│   │   ├── test_collection_service.py   # 重複スキップ・改名・部分失敗
│   │   └── test_lock.py                 # プロセスロック排他検証
│   └── integration/                     # 外部実API疎通テスト（Opt-in）
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
    content_scope           TEXT NOT NULL CHECK (content_scope IN ('abstract_only', 'readme_only', 'metadata_only', 'full_text')),
    raw_content_hash        TEXT NOT NULL,                     -- 生レスポンス (XML/JSON) の SHA-256
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
    error_category          TEXT CHECK (error_category IN ('transient', 'rate_limit', 'permanent', 'validation')),
    error_message           TEXT,
    http_status             INTEGER,
    duration_ms             INTEGER,
    attempted_at            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_attempts_run_id ON collection_attempts(run_id);
CREATE INDEX IF NOT EXISTS idx_attempts_target ON collection_attempts(target_identifier);
```

---

## 4. 資料ID・版ID・エイリアス判定規則

### 4.1 論文（arXiv）
arXiv の識別子は、新形式（2007年4月以降）と旧形式（1991年〜2007年3月）が存在し、URL 上に版番号が含まれない場合がある。

1. **ID パース正規化**:
   - **新形式**: `\d{4}\.\d{4,5}(v\d+)?`（例: `2301.12345`, `2301.12345v2`）。
   - **旧形式**: `[a-z\-]+(\.[A-Z]{2})?/\d{7}(v\d+)?`（例: `math/0309099`, `hep-th/9912012v1`）。
   - 抽出されたベース識別子から末尾の `v\d+` を分離。
   - `canonical_identifier` = `{base_id}`（例: `2301.12345`, `math/0309099`）。
   - `source_id` = `arxiv:{canonical_identifier}`。
2. **版（`upstream_version`）の確定**:
   - URL に `v2` などの明示がある場合でも、**API レスポンス（Atom XML の `<id>` タグ）に含まれる完全 URI を正本** とする。
     - 例: `<id>http://arxiv.org/abs/2301.12345v3</id>` → `upstream_version = "v3"`。
   - **版番号なし URL（例: `abs/2301.12345`）を `v1` と仮定することは厳禁**。API レスポンスから返却された最新確定版を `upstream_version` に登録する。
   - `version_id` = `arxiv:{canonical_identifier}:{upstream_version}`（例: `arxiv:2301.12345:v3`）。

### 4.2 OSS（GitHub）
GitHub リポジトリは所有者の変更（Organization 移管）やリポジトリ名変更（Rename）が頻繁に発生するため、可変な文字列 `owner/repo` を一意キーにしてはならない。

1. **論理資料 ID の確定（`repository_id`）**:
   - 初回取得時、GitHub REST API (`GET /repos/{owner}/{repo}`) を呼び出し、レスポンスの数値 `id`（例: `28457823`）を取得。
   - `canonical_identifier` = `str(repo_json["id"])`。
   - `source_id` = `github:repo:{canonical_identifier}`（例: `github:repo:28457823`）。
2. **改名・エイリアス追跡**:
   - `repo_json["full_name"]`（例: `"openai/gym"`）を取得し、`source_aliases` テーブルと突合。
   - 過去に存在しない別名の場合、既存別名の `is_current = 0` とし、新しい別名を `is_current = 1` で挿入。
   - `sources.primary_name` を最新の `full_name` で更新。
3. **版（`upstream_version`）の確定**:
   - デフォルトブランチ（または指定タグ）の **40桁フルコミット SHA** を用いる。
     - 例: `repo_json["default_branch"]` の HEAD コミット SHA `7f8b2c4e5d6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c`。
   - `version_id` = `github:repo:{canonical_identifier}:{commit_sha}`。

---

## 5. コレクター仕様とレート制限

### 5.1 arXiv Collector 仕様
- **API エンドポイント**: `http://export.arxiv.org/api/query`
- **リクエスト方式**: `httpx.AsyncClient` による GET。
- **レート制限 & マナー**:
  - 連続リクエストの間隔は **最低 3.5 秒**（公式推奨「1 request per 3 seconds」を厳格遵守）。
  - User-Agent にコンタクト情報を含める: `research_pipeline/0.1.0 (mailto:admin@example.com)`。
  - 1 クエリの `max_results` は最大 50 件に制限。
- **抽出内容**:
  - 生データ: Atom XML の `<entry>` タグ。
  - 抽出テキスト: `<summary>` タグ内の文字列（前後の空白・改行を正規化）。
  - メタデータ: タイトル、著者リスト、公開日（published）、更新日（updated）、主カテゴリ（primary_category）、コメント、DOI（存在する場合）。

### 5.2 GitHub Collector 仕様
- **API エンドポイント**: `https://api.github.com`
  - リポジトリ情報: `GET /repos/{owner}/{repo}`
  - README 取得: `GET /repos/{owner}/{repo}/readme`
- **Rate Limit 制御（Primary & Secondary）**:
  - **毎リクエストヘッダー解析**:
    - `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset` を取得。
    - `X-RateLimit-Remaining < 10` の場合、`X-RateLimit-Reset` 時刻まで処理を一時停止（Sleep）。
  - **Secondary Rate Limit（濫用検知）対策**:
    - 連続リクエスト間に最小 1.5 秒のジッター付き待機（未認証時は 60 秒以上）。
    - HTTP 403 / 429 受信時:
      - `Retry-After` ヘッダーが存在する場合はその秒数待機。
      - ヘッダーがない場合は指数バックオフ（初回到達で 60 秒待機、再試行時は 120 秒待機）。
      - 2 回連続で 403/429 が返った場合はジョブを `partial` として安全中断。
- **README 欠落時の `metadata_only` 処理**:
  - `GET /repos/{owner}/{repo}/readme` が **HTTP 404** を返した場合:
    - エラーとみなさず、`content_scope = 'metadata_only'` として処理を継続。
    - `readme.md` ファイルは生成せず、`raw_repo.json` と `metadata.json` のみを保存。
    - `extracted_text_hash` は `NULL` とする。

---

## 6. ファイル保存の状態遷移とアトミック性

SQLite とローカルファイルシステムは分散トランザクション（2相コミット）をネイティブにサポートしないため、**「ステージング → アトミックリネーム → DBコミット → 失敗時ロールバック」** の明示的ステートマシンを実装する。

### 6.1 4種のハッシュとバージョン管理
データの完全性と再現性を保証するため、以下の4つの属性を独立して記録する。
1. **`upstream_version`**: 原典側の版（arXiv: `"v2"`, GitHub: `40桁コミットSHA`）。
2. **`raw_content_hash`**: 取得した生レスポンス（生 XML または生 JSON）の SHA-256。
3. **`extracted_text_hash`**: 抽出・正規化された本文テキスト（Abstract または README）の SHA-256。
4. **`extractor_version`**: テキスト抽出ロジックの版（例: `"arxiv:1.0.0"`, `"github_readme:1.0.0"`）。抽出バグ修正時の再パース検知に使用。

### 6.2 保存ステートマシン

```text
[State: START]
      │
      ▼ (一時ディレクトリ作成)
[State: STAGED_TMP] (data/storage/.tmp/{uuid}/)
      │  ├─ 生データ (raw_entry.xml / raw_repo.json) 書き出し
      │  ├─ 抽出テキスト (abstract.txt / readme.md) 書き出し (存在時)
      │  ├─ metadata.json 書き出し
      │  ├─ 各ファイルの fsync 実行
      │  └─ raw_content_hash, extracted_text_hash を計算
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

## 7. クラッシュ後の整合性検査・復旧手順（Reconciler）

不意の電源断、OS 再起動、プロセス強制終了（SIGKILL）が発生した場合に備え、**起動時および定期実行前に必ず整合性チェッカー（`reconcile_storage_and_db()`）を実行** する。

### 7.1 検知する不整合と修復ルール

| 不整合パターン | 状態 | 自動修復アクション |
|---|---|---|
| **一時ファイルの残存** | `.tmp/` 配下に作成から 1 時間以上経過したディレクトリが存在する。 | 未完了セッションのゴミであるため、安全に削除（`shutil.rmtree`）。 |
| **孤立確定ファイル (Orphan File)** | ストレージ確定領域（`papers/` or `oss/`）にファイル一式が存在するが、`source_versions` にレコードが存在しない（コミット直前クラッシュ）。 | 実体ディレクトリを `data/storage/.orphan/{timestamp}_{dirname}` に退避移動し、警告ログを出力（手動監査可能にする）。 |
| **レコード欠損実体 (Missing File)** | `source_versions` に `status='active'` で登録されているが、ストレージ上の `abstract.txt` / `raw_repo.json` が消失している。 | DB レコードの `status` を `'missing_file'` に更新。次回の探索時に再取得対象として扱う。 |
| **コンテンツ破壊 (Corrupted File)** | ファイルは存在するが、ファイルの SHA-256 が DB の `extracted_text_hash` または `raw_content_hash` と一致しない。 | DB レコードの `status` を `'corrupted'` に更新し、実ファイルを `.orphan/` に退避。再取得対象とする。 |

### 7.2 整合性チェックの実行コード契約
```python
class StorageReconciler:
    async def reconcile(self) -> ReconciliationReport:
        """
        1. .tmp/ の掃除
        2. 全 source_versions レコードと実ファイルの存在・ハッシュ突合
        3. 実ファイルツリーを走査し、DB 未登録の孤立ディレクトリを検出・退避
        4. 結果レポート (修復件数、破損件数、孤立件数) を返却
        """
```

---

## 8. スケジューラと二重起動防止

### 8.1 二重起動防止（Process Lock）
複数プロセスや cron の重複実行によって SQLite ロック競合や API 多重リクエストが発生するのを完全に防止する。

- ロックファイルパス: `data/hde.lock`
- **実装方式**:
  - Windows: `msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)`
  - POSIX: `fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)`
  - ロックファイル内部に現在の PID、ホスト名、起動時刻を記録。
- **動作挙動**:
  - プロセス起動時、ロック取得を試みる。
  - ロックが既に保持されている場合、ログに `Another HDE collection process is already running (PID: ...). Exiting.` を出力し、終了コード `0`（またはスキップ用コード）で即座に安全終了する。

### 8.2 スケジュール実行方式（Runner）
H0 では外部の複雑なスケジューラ（Celery Beat など）を新設せず、以下の 2 通りのシンプルな運用を可能にする。

1. **CLI ワンショット実行**:
   ```bash
   python -m research_pipeline.scheduler.runner --mode once --query "agent-based market simulation" --limit 10
   ```
2. **軽量常駐ループ（Daemon）**:
   ```bash
   python -m research_pipeline.scheduler.runner --mode daemon --interval 3600
   ```
   - 内部で `asyncio.sleep(interval)` を回し、周回ごとに `reconcile()` → `run_discovery()` を実行。
   - SIGINT (Ctrl+C) / SIGTERM を安全にトラップし、実行中のアイテム保存完了を待ってから終了する Graceful Shutdown を実装。

---

## 9. エラー分類と再試行方針

| 分類 (`error_category`) | HTTPステータス / 例外例 | 初動アクション | 再試行・回復方針 |
|---|---|---|---|
| `transient` | `httpx.ConnectTimeout`, `httpx.ReadTimeout`, HTTP 502, 503, 504 | 指数バックオフ待機 (1s, 2s, 4s) | 同一セッション内で **最大3回** 再試行。全て失敗した場合は `collection_attempts(failed)` を記録し次へ。 |
| `rate_limit` | HTTP 429, GitHub 403 (RateLimit) | `Retry-After` または `X-RateLimit-Reset` 解析 | リセット時刻までスリープ。待機時間が 60秒を超える場合はジョブ全体を `partial` で安全終了。 |
| `permanent` | HTTP 404 (リポジトリ/論文なし), 400 (形式不正), 410 (削除) | 再試行なし | `collection_attempts(failed)` に記録し、即座に対象をスキップ。 |
| `validation` | XML パース不能、文字化け、必須フィールド欠損 | 再試行なし | 不正データを破棄し、`collection_attempts(failed)` にスキーマエラーを記録。 |

---

## 10. H1a への引き渡し契約 (`ResearchDocument`)

H1a（Capability Extraction）が「要約から得た主張」を「論文全体で実証済みの事実」と混同しないよう、**`content_scope`** を必須フィールドとして定義する。

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Literal
from datetime import datetime

class ResearchDocument(BaseModel):
    """H0 から H1a に引き渡される構造化研究ドキュメント"""
    source_id: str = Field(..., description="論理資料ID (例: 'arxiv:2301.12345', 'github:repo:28457823')")
    version_id: str = Field(..., description="特定版ID (例: 'arxiv:2301.12345:v3', 'github:repo:28457823:a1b2c3...')")
    source_type: Literal["paper", "oss"] = Field(..., description="資料種別")
    canonical_identifier: str = Field(..., description="不変原典識別子 ('2301.12345', '28457823')")
    primary_name: str = Field(..., description="論文タイトル、または最新リポジトリ名 (owner/repo)")
    primary_url: str = Field(..., description="原典Web URL")
    upstream_version: str = Field(..., description="確定版識別子 ('v3', 40桁コミットSHA)")
    
    # ── H1a 認識境界 (重要) ────────────────────────────────────────────────
    content_scope: Literal["abstract_only", "readme_only", "metadata_only", "full_text"] = Field(
        ...,
        description="テキストの範囲。H1a は abstract_only / readme_only の場合、主張を 'reported' として扱い、実証済みと見なさない。"
    )
    
    # ── コンテンツ本文と整合性 ──────────────────────────────────────────
    content_text: Optional[str] = Field(None, description="AbstractまたはREADME本文 (metadata_only時はNone)")
    raw_content_hash: str = Field(..., description="取得生データ (XML/JSON) の SHA-256")
    extracted_text_hash: Optional[str] = Field(None, description="content_text の SHA-256 (テキスト存在時)")
    extractor_version: str = Field(..., description="抽出器バージョン")
    
    # ── メタデータ & 追跡 ────────────────────────────────────────────────
    published_at: Optional[str] = Field(None, description="原典公開日時 (ISO8601)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="構造化メタデータ")
    local_artifact_dir: str = Field(..., description="ローカル保存先絶対パス")
    collected_at: str = Field(..., description="収集完了日時 (ISO8601)")
```

---

## 11. テスト計画と受入基準

### 11.1 単体テスト（モック環境: `pytest tests/unit`）
外部 API 通信を一切行わず、フィクスチャを用いてエッジケースを徹底検証する。

1. **ID 正規化・版確定テスト (`test_identifiers.py`)**:
   - arXiv 新形式（`2301.12345v2`）および版なし（`2301.12345`）が正しくパースされること。
   - arXiv 旧形式（`math/0309099`, `hep-th/9912012v1`）が正しくパースされること。
   - GitHub URL（大文字小文字、`.git`、末尾スラッシュ）から正規化された名前が抽出され、数値 ID と紐付くこと。
2. **クラッシュ復旧テスト (`test_reconciler.py`)**:
   - ステージング（`.tmp/`）に残された 1 時間超の残骸が起動時に削除されること。
   - DB コミット直前にプロセスが終了した場合の孤立実体ファイルが検知され、`.orphan/` に安全退避されること。
   - 実体ファイルが削除された DB レコードが `missing_file` に正しくステータス変更されること。
3. **改名追跡テスト (`test_github_collector.py`, `test_collection_service.py`)**:
   - 同一 `repository_id` を持つリポジトリが `ownerA/repoA` から `ownerB/repoB` に改名された際、新規 `source_id` が生成されず、同一 `source_id` のエイリアス履歴が追加され、`primary_name` が更新されること。
4. **README 欠落テスト (`test_github_collector.py`)**:
   - README API が 404 を返した際、例外にならず `content_scope='metadata_only'` として正常保存されること。
5. **部分失敗テスト (`test_collection_service.py`)**:
   - 3 件の候補中、2 件目が 404 で失敗した場合でも、1 件目と 3 件目が正常コミットされ、`discovery_runs` が `partial`、2 件目の `collection_attempts` が `failed` になること。
6. **同一版の内容差異テスト (`test_content_store.py`)**:
   - DB 上で同一の `version_id` なのに保存内容のハッシュが改ざん/破損した場合、`corrupted` として検知・退避されること。
7. **二重起動防止テスト (`test_lock.py`)**:
   - すでにロックファイルが保持されている状態で別プロセスが起動を試みた場合、即座にブロックされ終了すること。

### 11.2 固定候補集合による冪等性受入テスト
流動的な Web 検索ではなく、**あらかじめ定義した固定候補集合（Fixed Fixture）** を用いてテストする。

- **テスト手順**:
  1. テスト用 DB とストレージを初期化（空状態）。
  2. 固定候補集合（arXiv 既知論文 2 件、GitHub 既知リポジトリ 2 件のモック）を対象に 1 回目の収集を実行。
     - 検証: 全 4 件が `status='active'`、ファイルとハッシュが完全一致、`items_collected=4`。
  3. 全く同じ固定候補集合を対象に 2 回目の収集を実行。
     - **受入基準**:
       - `items_collected=0`, `items_skipped=4` であること。
       - `collection_attempts` に 4 件の `status='skipped_duplicate'` が記録されること。
       - `source_versions` のレコード総数が 4 件のまま増加しないこと。
       - ストレージ内のファイル更新日時（mtime）およびハッシュが 1 ビットも変更されないこと。

### 11.3 結合テスト（Opt-in: `pytest tests/integration --run-external`）
- 手動フラグ指定時のみ、実 arXiv API（1 件）および実 GitHub API（1 件）に疎通し、実環境でのレスポンスパースと `ResearchDocument` 生成を確認。

---

## 12. 実装順序

承認受領後の実装ステップは以下の通りとする（**本計画提出時点では実装未着手**）。

```text
Step 1: プロジェクト基本セットアップ
        - pyproject.toml（httpx, pydantic, aiosqlite 等）
        - src/research_pipeline/config.py
        - tests/conftest.py & tests/fixtures/

Step 2: 永続化層・Reconciler・プロセスロックの実装
        - src/research_pipeline/storage/db.py（DDL適用）
        - src/research_pipeline/storage/content_store.py（アトミック保存・ハッシュ）
        - src/research_pipeline/storage/reconciler.py（クラッシュ復旧）
        - src/research_pipeline/scheduler/lock.py（排他ロック）
        - tests/unit/test_content_store.py, test_reconciler.py, test_lock.py で検証

Step 3: Collector 基底 & レート制限の実装
        - src/research_pipeline/collectors/base.py, models.py
        - src/research_pipeline/collectors/rate_limiter.py（ヘッダー監視）
        - tests/unit/test_identifiers.py で検証

Step 4: arXiv Collector 実装
        - src/research_pipeline/collectors/arxiv.py（新旧ID・版確定パース）
        - tests/unit/test_arxiv_collector.py で検証

Step 5: GitHub Collector 実装
        - src/research_pipeline/collectors/github.py（Repo ID・Secondary Rate Limit・README 404）
        - tests/unit/test_github_collector.py で検証

Step 6: CollectionService & H1a 契約の実装
        - src/research_pipeline/service/contracts.py（ResearchDocument, content_scope）
        - src/research_pipeline/service/collection_service.py（二相制御・改名追跡・部分失敗）
        - tests/unit/test_collection_service.py で検証

Step 7: スケジューラ Runner & 総合受入検証
        - src/research_pipeline/scheduler/runner.py（CLI & Daemon）
        - 固定候補集合による冪等性受入テストの実行
        - Opt-in 外部結合テストの実行
```

---

## 13. 未決定事項・将来の拡張課題

1. **GitHub Personal Access Token (PAT) の設定**:
   - 未認証（60 req/h）でも動作する設計としたが、実運用では `.env` に `GITHUB_TOKEN` を設定することで 5,000 req/h まで緩和可能。トークン配備はユーザーの任意判断とする。
2. **全文 PDF 取得（L2拡張）のトリガー契機**:
   - H1a の Capability Extraction で「Abstract のみでは入出力仕様や前提条件が特定できない」と判定された場合のみ、オンデマンドで `pdf_url` から本文を取得する拡張インターフェースを Phase 1（H1a 完了後）に設計する。

---
**Plan Prepared by:** Antigravity Agent  
**Awaiting Review:** 最終計画 v0.3 の承認判定待ち
