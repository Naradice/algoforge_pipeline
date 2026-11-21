# HDE H0 — 収集・保存基盤 実装計画書 v0.2

**Status:** Proposed (Review Pending)  
**Date:** 2026-11-15  
**Version:** 0.2  
**Target:** Hypothesis Discovery Engine (HDE) Phase 1 / Step H0  
**Parent Specifications:**
- [architecture.md](file:///c:/Users/ksato/workspace/finance/research_pipeline/docs/architecture.md)
- [hypothesis_discovery_engine.md](file:///c:/Users/ksato/workspace/finance/research_pipeline/docs/hypothesis_discovery_engine.md)
- [H0-collection-audit.md](file:///c:/Users/ksato/workspace/finance/research_pipeline/docs/H0-collection-audit.md)

---

## 1. 目的とスコープ

### 1.1 目的
本計画は、Hypothesis Discovery Engine（HDE）の初期基盤（H0）として、学術論文（arXiv）およびオープンソースソフトウェア（GitHub）のメタデータ・概要（Abstract / README）を探索・収集・版管理し、次段の Capability Extraction（H1a）へ確実に受け渡すためのデータ収集・永続化パイプラインを構築することを目的とする。

### 1.2 監査結果を踏まえた基本方針（必須制約の遵守）
1. **AlgoForge 無侵害**: AlgoForge のコード、データベース（PostgreSQL/Redis）、Celery キュー、設定ファイルには一切変更を加えない。
2. **公式 API 優先**: H0 では arXiv API（Atom/XML）および GitHub REST API v3 を使用する。Stealth Playwright（Headless ブラウザ）は H0 の対象外とし、外部依存と不安定要素を排除する。
3. **論理資料と版の分離**: 論理的な研究対象（`sources`）と、その特定の版（`source_versions`）を分離管理し、arXiv の更新版（v1, v2…）や GitHub の新コミット（SHA）を追跡可能にする。
4. **探索実行と試行の分離**: 全体の探索セッション（`discovery_runs`）と、資料単位の取得試行（`collection_attempts`）を分離し、一時的な通信障害やレート制限の個別追跡・再試行を可能にする。
5. **Abstract / README 優先**: 論文 PDF 全文の取得・解析は H0 では行わず、メタデータおよび Abstract / README テキストを収集対象とする。全文取得は将来の必要に応じた拡張とする。
6. **アトミック保存と冪等性**: SQLite トランザクションとファイルシステム書き込みの二相処理により、クラッシュ時のゴミファイル発生や不整合を完全に防止する。
7. **単一プロセス & 運用制限**: 初期構成は単一非同期プロセスとし、同時実行数、API コールレート、保存容量（上限 5GB）を厳格に制御し、AlgoForge とのリソース競合を防止する。

---

## 2. ディレクトリ構成

`research_pipeline` ワークスペース配下に以下の構造を配置する。

```text
research_pipeline/
├── docs/
│   ├── architecture.md
│   ├── hypothesis_discovery_engine.md
│   ├── H0-collection-audit.md
│   └── H0-implementation-plan-v0.2.md
├── data/                                # ローカル永続化領域（.gitignore 対象）
│   ├── hde.db                           # SQLite データベース正本
│   └── storage/                         # コンテンツストレージ正本
│       ├── .tmp/                        # アトミック書き込み用一時ディレクトリ
│       ├── papers/                      # 論文データ
│       │   └── {source_id}/
│       │       └── {version_id}/
│       │           ├── metadata.json    # 生メタデータ
│       │           └── abstract.txt     # 抽出済み Abstract
│       └── oss/                         # OSS データ
│           └── {source_id}/
│               └── {version_id}/
│                   ├── metadata.json    # 生リポジトリ情報
│                   └── readme.md        # 抽出済み README
├── src/
│   └── research_pipeline/
│       ├── __init__.py
│       ├── config.py                    # 設定値・パス・制限値管理
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── db.py                    # SQLite 接続・DDL適用・トランザクション
│       │   ├── repository.py            # SQLite CRUD 操作
│       │   └── content_store.py         # アトミックファイル保存・整合性検証
│       ├── collectors/
│       │   ├── __init__.py
│       │   ├── base.py                  # CollectorBase 抽象基底クラス
│       │   ├── models.py                # 収集ドメインモデル (SearchQuery, FetchResult)
│       │   ├── arxiv.py                 # arXiv 公式 API クライアント
│       │   ├── github.py                # GitHub 公式 REST API クライアント
│       │   └── rate_limiter.py          # レート制限・Polite Delay 管理
│       └── service/
│           ├── __init__.py
│           ├── collection_service.py     # 収集オーケストレーション・整合性制御
│           └── contracts.py             # H1a 引き渡しスキーマ (ResearchDocument)
├── tests/
│   ├── __init__.py
│   ├── conftest.py                      # テスト用一時DB・ストレージ初期化
│   ├── unit/                            # モックを用いた高速単体テスト
│   │   ├── test_identifiers.py          # ID・版正規化ロジックの検証
│   │   ├── test_content_store.py        # アトミック保存・ロールバック検証
│   │   ├── test_arxiv_collector.py      # arXiv API モック検証
│   │   ├── test_github_collector.py     # GitHub API モック検証
│   │   └── test_collection_service.py   # 整合性・重複排除ロジック検証
│   └── integration/                     # 外部実API疎通テスト（手動/Opt-in）
│       ├── test_arxiv_real.py
│       └── test_github_real.py
├── pyproject.toml
└── .env.example
```

---

## 3. モジュール責務

| モジュール | 責務 | 入力 | 出力 |
|---|---|---|---|
| `config.py` | パス、API キー、レート制限、ストレージクォータ、タイムアウトの設定保持。 | 環境変数, `.env` | イミュータブルな設定オブジェクト |
| `storage/db.py` | SQLite DB 接続初期化、PRAGMA 設定（WAL, foreign_keys）、マイグレーション適用。 | DB パス | `sqlite3.Connection` (または aiosqlite) |
| `storage/content_store.py` | ファイルのアトミック保存（一時ファイル → fsync → rename）、ハッシュ計算、読み込み。 | 識別子, バイト列/テキスト | 格納パス, SHA-256 ハッシュ |
| `storage/repository.py` | `sources`, `source_versions`, `discovery_runs`, `collection_attempts` の SQL 実行。 | エンティティモデル | DB レコード, 検索結果 |
| `collectors/rate_limiter.py` | ドメインごとのレート制限（トークンバケットまたは最小インターバル）とジッター制御。 | ドメイン識別子 | 非同期スリープ・実行許可 |
| `collectors/arxiv.py` | arXiv API へのクエリ発行、Atom XML の解析、生メタデータと Abstract の抽出。 | `SearchQuery` | `list[RawCandidate]`, `RawContent` |
| `collectors/github.py` | GitHub API へのリクエスト、レート制限ヘッダー監視、リポジトリ情報と README の抽出。 | `SearchQuery` | `list[RawCandidate]`, `RawContent` |
| `service/collection_service.py`| 探索クエリの受付、重複チェック、取得、アトミック保存、DBトランザクション確定、エラー記録。 | 探索要求 | 収集サマリ, `list[ResearchDocument]` |
| `service/contracts.py` | H1a（Capability Extraction）が受領する不変データ契約 `ResearchDocument` の定義。 | - | Pydantic モデル定義 |

---

## 4. SQLite DDL と制約

データベースファイルは `data/hde.db` とする。接続時には必ず以下を設定する。
```sql
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;
```

### 4.1 テーブル定義

```sql
-- 1. 論理資料テーブル（概念としての論文・リポジトリ）
CREATE TABLE IF NOT EXISTS sources (
    source_id               TEXT PRIMARY KEY,                  -- 例: "arxiv:2301.12345", "github:owner/repo"
    source_type             TEXT NOT NULL CHECK (source_type IN ('paper', 'oss')),
    canonical_identifier    TEXT NOT NULL,                     -- 例: "2301.12345", "owner/repo"
    title                   TEXT NOT NULL,
    primary_url             TEXT NOT NULL,
    created_at              TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at              TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_sources_type_canonical ON sources(source_type, canonical_identifier);

-- 2. 資料版テーブル（特定の版、コミット、Abstract/README 実体）
CREATE TABLE IF NOT EXISTS source_versions (
    version_id              TEXT PRIMARY KEY,                  -- 例: "arxiv:2301.12345:v1", "github:owner/repo:a1b2c3d"
    source_id               TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
    version_tag             TEXT NOT NULL,                     -- 例: "v1", "v2", コミットSHA40文字
    content_hash            TEXT NOT NULL,                     -- Abstract/README テキストの SHA-256
    storage_rel_path        TEXT NOT NULL,                     -- 例: "papers/arxiv_2301.12345/v1"
    metadata_json           TEXT NOT NULL,                     -- 著者、公開日、ライセンス等の生JSON
    published_at            TEXT,                              -- 原典の発行日・コミット日時
    collected_at            TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    CONSTRAINT uq_source_version UNIQUE (source_id, version_tag)
);

CREATE INDEX IF NOT EXISTS idx_source_versions_source_id ON source_versions(source_id);
CREATE INDEX IF NOT EXISTS idx_source_versions_hash ON source_versions(content_hash);

-- 3. 探索実行セッションテーブル
CREATE TABLE IF NOT EXISTS discovery_runs (
    run_id                  TEXT PRIMARY KEY,                  -- UUIDv4
    mode                    TEXT NOT NULL CHECK (mode IN ('continuous', 'problem_driven')),
    query                   TEXT NOT NULL,
    max_items               INTEGER NOT NULL,
    agent_id                TEXT,                              -- 実行主体 (CodingAgent, Orchestrator等)
    status                  TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed', 'partial')),
    items_found             INTEGER NOT NULL DEFAULT 0,
    items_collected         INTEGER NOT NULL DEFAULT 0,
    items_skipped           INTEGER NOT NULL DEFAULT 0,
    error_message           TEXT,
    started_at              TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    completed_at            TEXT
);

-- 4. 取得試行・失敗追跡テーブル
CREATE TABLE IF NOT EXISTS collection_attempts (
    attempt_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                  TEXT NOT NULL REFERENCES discovery_runs(run_id) ON DELETE CASCADE,
    target_identifier       TEXT NOT NULL,                     -- 試行対象 (URL または canonical ID)
    source_id               TEXT,                              -- 解決された場合は sources(source_id)
    version_tag             TEXT,                              -- 解決された版
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

## 5. 資料ID・版IDの判定規則

同一資料の重複登録やバージョン混濁を防ぐため、厳格な正規化規則を適用する。

### 5.1 論文（arXiv）
- **対象形式**: `http(s)://arxiv.org/abs/{id}`, `http(s)://arxiv.org/pdf/{id}(.pdf)`, または生ID `{id}`。
- **正規化手順**:
  1. URL から `abs/` または `pdf/` の直後のID部を抽出（例: `2301.12345v2`）。
  2. 末尾のバージョン番号 `v[0-9]+` を分離。
  3. `base_id` = `2301.12345`、`version_tag` = `v2`（明示されない場合は `v1`、APIメタデータ側で特定）。
- **ID体系**:
  - `source_id`: `arxiv:2301.12345`
  - `canonical_identifier`: `2301.12345`
  - `version_id`: `arxiv:2301.12345:v2`

### 5.2 OSS（GitHub）
- **対象形式**: `https://github.com/{owner}/{repo}(.git)(/...)` または `{owner}/{repo}`。
- **正規化手順**:
  1. `owner` と `repo` を小文字化（例: `OpenAI/gym` → `openai/gym`）。
  2. 末尾の `.git` やトレイリングスラッシュを除去。
  3. バージョンは Git の **40桁フルコミット SHA** を用いる（デフォルトブランチの HEAD、または特定リリース時点の commit sha）。
- **ID体系**:
  - `source_id`: `github:openai/gym`
  - `canonical_identifier`: `openai/gym`
  - `version_id`: `github:openai/gym:7f8b2c4e5d6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c`

### 5.3 冪等性判定ルール
1. 検索結果から得られた `source_id` と `version_tag` について、`source_versions` テーブルにすでに同一の `(source_id, version_tag)` が存在する場合:
   - 取得処理をスキップ（ネットワーク転送を行わない）。
   - `collection_attempts` に `status='skipped_duplicate'` を記録。
2. 同一 `source_id` で異なる `version_tag` が検出された場合:
   - `sources` テーブルは更新（`updated_at` 更新）。
   - 新規版として取得・保存し、`source_versions` に新レコードを挿入。

---

## 6. Collector インターフェース設計

外部依存を排し、テスト容易性を最大化するための抽象クラスを定義する。

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

@dataclass(frozen=True)
class SearchQuery:
    query_text: str
    max_results: int = 10
    start_offset: int = 0

@dataclass(frozen=True)
class CandidateRef:
    source_type: str              # 'paper' | 'oss'
    canonical_identifier: str     # '2301.12345' | 'owner/repo'
    title: str
    primary_url: str
    version_tag: Optional[str]    # 判明している場合のみ

@dataclass(frozen=True)
class CollectedContent:
    candidate: CandidateRef
    version_tag: str              # 確定した版 (例: "v2", 40桁SHA)
    content_text: str             # Abstract または README 全文
    published_at: Optional[str]   # ISO8601
    metadata: Dict[str, Any]      # 著者、ライセンス、URL等の生データ辞書

class BaseCollector(ABC):
    @property
    @abstractmethod
    def source_type(self) -> str:
        """'paper' または 'oss'"""
        pass

    @abstractmethod
    async def search(self, query: SearchQuery) -> List[CandidateRef]:
        """検索を実行し、候補参照リストを返す"""
        pass

    @abstractmethod
    async def fetch(self, candidate: CandidateRef) -> CollectedContent:
        """特定の候補のメタデータとテキスト（Abstract / README）を取得する"""
        pass
```

---

## 7. 検索・取得・保存の処理フローと整合性担保

データの不整合やクラッシュ時の破損を防ぐため、**「ファイル書き込み完了後に DB トランザクションを確定する二相処理」** を適用する。

```text
[CollectionService.run_discovery]
        │
        ├─ 1. DB: discovery_runs レコード作成 (status='running')
        │
        ├─ 2. Collector.search(query) 呼び出し
        │
        └─ 3. 各 Candidate に対してループ:
                │
                ├─ 3.1 DB: 重複確認 (source_versions に存在するか?)
                │       └─ 存在する場合: attempts(skipped_duplicate) 記録し continue
                │
                ├─ 3.2 RateLimiter: 待機 (Polite Delay)
                │
                ├─ 3.3 Collector.fetch(candidate) 呼び出し
                │       └─ 失敗時: エラー分類し attempts(failed) 記録し continue
                │
                ├─ 3.4 ContentStore.stage_and_commit(...)
                │       ├─ a. data/storage/.tmp/{uuid}/ に metadata.json & 本文を書き出し
                │       ├─ b. 本文の SHA-256 (content_hash) を計算
                │       ├─ c. fsync 実行
                │       └─ d. 正本ディレクトリ (data/storage/{type}/{source}/{version}/) へアトミックリネーム (os.replace)
                │
                ├─ 3.5 DB: トランザクションコミット
                │       ├─ INSERT INTO sources (ON CONFLICT DO UPDATE)
                │       ├─ INSERT INTO source_versions
                │       └─ INSERT INTO collection_attempts (status='success')
                │       ※ DBコミット失敗時は保存したファイルを削除 (補償トランザクション)
                │
                └─ 3.6 ResearchDocument インスタンスを生成してリストに追加
        │
        └─ 4. DB: discovery_runs 完了更新 (status='completed', items_collected 等)
```

---

## 8. エラー分類と再試行方針

| 分類 (`error_category`) | 原因例 | 対応方針 |
|---|---|---|
| `transient` (一時的) | ネットワーク接続タイムアウト (ConnectTimeout)、一時的 DNS 失敗、HTTP 502/503/504 | 指数バックオフ（1s, 2s, 4s）で **最大 3 回まで再試行**。解消しない場合は当ランではスキップし、失敗ログ記録。 |
| `rate_limit` (レート制限) | HTTP 429 Too Many Requests、GitHub 403 (X-RateLimit-Remaining=0) | レスポンスヘッダー（`Retry-After` または `X-RateLimit-Reset`）の秒数だけ待機。待機時間が 60 秒を超える場合は当ランを `partial` 終了。 |
| `permanent` (恒久的) | HTTP 404 Not Found、リポジトリ削除、arXiv ID 形式不正 | **再試行しない**。`collection_attempts` に記録し、即座にスキップ。 |
| `validation` (検証エラー) | Abstract が空、文字化け、UTF-8 デコード失敗、JSON スキーマ不正 | **再試行しない**。不完全なデータを永続化しないよう破棄し、警告ログ記録。 |

---

## 9. 運用上の制限とリソース競合対策

AlgoForge（バックテスト、FastAPI、PostgreSQL、Celery）が同一ホスト上で稼働しているため、以下の制限を厳密に設ける。

1. **実行プロセス**: 単一プロセス（`asyncio` によるシングルスレッド非同期 IO）。バックグラウンド常駐デーモンや Celery ワーカーは立ち上げない。
2. **同時実行制限**: 同一時間帯における外部リクエストの並行数は **最大 2**（Semaphore 制御）。
3. **API レート制限**:
   - **arXiv API**: 最小リクエスト間隔 **3.5 秒**（arXiv 公式ガイドライン「1 request per 3 seconds」を厳格遵守）。
   - **GitHub API**:
     - 未認証時: 60 リクエスト/時（最小間隔 60 秒）。
     - トークン設定時（推奨）: 5,000 リクエスト/時（最小間隔 1.5 秒）。
     - 毎リクエスト時に `X-RateLimit-Remaining` を監視し、残数が 10% を切った場合は自動停止。
4. **ディスククォータ制限**:
   - `data/storage/` の合計容量上限を **5.0 GB** に設定。
   - 保存前に容量をチェックし、上限超過時は新規取得を拒否（`StorageQuotaExceededError`）。
5. **AlgoForge とのポート・DB競合ゼロ**:
   - ネットワークポート（8000, 3000 等）は一切リッスンしない。
   - PostgreSQL / Redis への接続は一切行わない。

---

## 10. H1a への引き渡し契約 (`ResearchDocument`)

H0 で収集・永続化されたデータは、以下の `ResearchDocument` Pydantic スキーマとして H1a（Capability Extraction）に引き渡される。H1a はこのオブジェクトのみに依存し、外部 API や通信エラーの存在を意識しない。

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

class ResearchDocument(BaseModel):
    """H0 から H1a に引き渡される構造化ドキュメント"""
    source_id: str = Field(..., description="論理資料ID (例: 'arxiv:2301.12345')")
    version_id: str = Field(..., description="特定版ID (例: 'arxiv:2301.12345:v2')")
    source_type: str = Field(..., description="'paper' または 'oss'")
    canonical_identifier: str = Field(..., description="'2301.12345' または 'owner/repo'")
    title: str = Field(..., description="論文タイトルまたはリポジトリ名")
    primary_url: str = Field(..., description="アクセス元URL")
    version_tag: str = Field(..., description="版識別子 ('v1', コミットSHA)")
    published_at: Optional[str] = Field(None, description="発行日またはコミット日時 (ISO8601)")
    
    # H1a 解析の主要インプット
    content_text: str = Field(..., description="論文AbstractまたはREADMEテキスト")
    content_hash: str = Field(..., description="content_text の SHA-256 ハッシュ")
    
    # 出典・版追跡情報
    metadata: Dict[str, Any] = Field(default_factory=dict, description="生メタデータ")
    local_artifact_dir: str = Field(..., description="ローカル保存先絶対パス")
    collected_at: str = Field(..., description="収集完了日時 (ISO8601)")
```

---

## 11. テスト計画と受入基準

テストは「モックによる自動単体テスト」と「明示的フラグでのみ実行する外部API結合テスト」に厳格に分離する。

### 11.1 単体テスト（`pytest tests/unit`）
外部ネットワークアクセスを行わず、ローカルで完結するテスト。
- `test_identifiers.py`:
  - 各種形式の arXiv URL・ID から正しい `source_id`, `version_id` が生成されること。
  - GitHub URL から小文字正規化された `owner/repo` とコミット SHA が正しく判定されること。
- `test_content_store.py`:
  - 正常時に一時ファイルから確定パスへアトミック移動されること。
  - ディスク書き込みエラー時に一時ファイルがクリーンアップされ、確定パスが汚染されないこと。
  - SHA-256 チェックサムが正しく一致すること。
- `test_arxiv_collector.py`:
  - Atom XML のモック応答をパースし、タイトル、Abstract、版情報が欠損なく抽出されること。
- `test_github_collector.py`:
  - GitHub REST API の JSON モック（Base64 エンコードされた README 含む）を正常にデコードできること。
- `test_collection_service.py`:
  - 同一資料の2回目取得時に `skipped_duplicate` となり、ファイル・DBが二重作成されないこと。
  - トランザクション失敗時に補償処理が走り、整合性が維持されること。

### 11.2 結合テスト（`pytest tests/integration --run-external`）
CI や日常テストではスキップし、開発者が明示的に指定した時のみ各 1 件ずつ実 API を叩くテスト。
- 実 arXiv API から 1 件の既知論文（例: `2301.12345`）を取得し、`ResearchDocument` が生成されること。
- 実 GitHub API から 1 件の既知パブリックリポジトリ（例: `octocat/Hello-World`）を取得し、README が保存されること。

### 11.3 受入基準
1. **無侵害の検証**: AlgoForge ディレクトリ配下の `git status` が完全にクリーンであること（変更ファイル 0 件）。
2. **全自動テストの通過**: `tests/unit/` 配下のテストが外部通信なしで 100% 成功すること。
3. **冪等性の確認**: 同一クエリを連続 2 回実行した際、2 回目は全件が `skipped_duplicate` となり、ストレージ使用量と版レコード数が増加しないこと。
4. **整合性の確認**: `source_versions` テーブルの全レコードについて、`content_hash` が実ファイルの SHA-256 と完全に一致すること。
5. **H1a 契約適合**: 取得された全ドキュメントが `ResearchDocument` のバリデーションを通過すること。

---

## 12. 実装順序

実装着手時は以下のステップ順に進める（**※現時点では計画のみであり、実装は行わない**）。

```text
Step 1: プロジェクト基盤のセットアップ
        - pyproject.toml (httpx, pydantic, aiosqlite 等の依存定義)
        - config.py
        - tests/conftest.py

Step 2: 永続化層（DB & Storage）の実装
        - storage/db.py (DDL 適用)
        - storage/content_store.py (アトミックファイル操作)
        - storage/repository.py (CRUD)
        - tests/unit/test_content_store.py で検証

Step 3: Collector 抽象基底 & レート制限の実装
        - collectors/base.py, models.py
        - collectors/rate_limiter.py
        - tests/unit/test_identifiers.py で検証

Step 4: arXiv Collector の実装
        - collectors/arxiv.py (Atom XML パース)
        - tests/unit/test_arxiv_collector.py (モックテスト)

Step 5: GitHub Collector の実装
        - collectors/github.py (REST API, README デコード)
        - tests/unit/test_github_collector.py (モックテスト)

Step 6: CollectionService & H1a 引き渡しインターフェースの実装
        - service/contracts.py (ResearchDocument)
        - service/collection_service.py (オーケストレーション & 整合性)
        - tests/unit/test_collection_service.py で検証

Step 7: 総合検証 & 受入検査
        - 結合テスト実施 (--run-external)
        - 受入基準チェックリストの確認
```

---

## 13. 未決定事項・将来の拡張課題

1. **GitHub Personal Access Token (PAT) の配備方法**:
   - 未認証だと 60 req/h に制限されるため、開発環境における `.env` への PAT 記載を推奨とするか（ユーザーの事前同意が必要）。
2. **Semantic Scholar API の追加時期**:
   - 論文の引用関係や分野タグの取得を H0 に含めるか、H1b（Entity Resolution）時に必要に応じて拡張するか（現時点では H0 は arXiv のみでミニマムスタートを推奨）。
3. **全文取得（PDF）の拡張トリガー**:
   - H1a で Abstract だけではメカニズムが不足と判断された場合に、オンデマンドで PDF を取得するインターフェースの追加検討。

---
**Plan Prepared by:** Antigravity Agent  
**Awaiting Review:** 科学的妥当性・実装リスク・受入基準のユーザー再レビュー待ち
