# H0 — AlgoForge収集基盤監査報告書

**Status:** Completed  
**Date:** 2026-11-15  
**Target:** AlgoForge 金融レポート定期収集基盤（`web_report`）  
**Scope:** 論文・OSS収集基盤（HDE H0）としての再利用可能性評価（コード監査のみ）  

---

## 1. エグゼクティブサマリ

AlgoForgeのデータ収集機構、特に金融レポート定期収集機能（`web_report`）およびジョブスケジューラ（`data/scheduler.py`）をコード監査した。

### 主な結論
1. **動作パターンの違い**:
   - AlgoForgeの `web_report` は「事前登録された特定URLを定期巡回し、PDF/HTMLをダウンロードしてローカルファイルとして保存する」**静的定期巡回型クローラー**である。
   - 一方、HDE（Hypothesis Discovery Engine）が求める収集基盤は「AI Agentが動的に検索クエリを生成し、外部API（arXiv, Semantic Scholar, GitHub等）やWeb検索を通じて文献・実装を検索・取得し、メタデータ・出典・バージョンを構造化管理する」**探索的・動的検索型リポジトリ**である。
2. **再利用できる技術的知見**:
   - Playwright による Bot 検知回避（Stealth ブラウザ起動、User-Agent 偽装、実Chrome/Edgeフォールバック）
   - ランダムジッター付きの礼儀正しいクロール間隔制御（Polite Delay）
   - ジョブ死活監視とゾンビプロセス刈り取り（Heartbeat スレッド & Reaper）
   - Redis 分散ロックによる二重実行防止パターン
3. **再利用が困難な点**:
   - `web_report` のデータモデル（`Dataset` テーブル）は「ファイル総数」のみを記録する仕様であり、DOI、arXiv ID、Git Commit SHA、著者、ライセンス等の文献・OSSメタデータの保持ができない。
   - 重複排除と更新検知がローカルファイル（`last_check_dates.json`, `checksums/*.txt`）に依存しており、HDEが求めるSQLite上の Knowledge Graph / Source Repository と統合されていない。
4. **推奨設計案**:
   - **「案C：HDE専用Collectorを `research_pipeline` 内に新規実装」を強く推奨**する。AlgoForge本体への変更をゼロにして既存金融収集への回帰リスクを完全に排除しつつ、AlgoForgeの優れた実装パターン（Stealth, Polite Delay, Heartbeat）を参考にして設計・移植するのが最も安全かつ開発効率が高い。

---

## 2. 現在の収集アーキテクチャ（AlgoForge）

### 2.1 アーキテクチャ図

```text
       [FastAPI main.py Lifespan]
                   │
                   ▼
       [data/scheduler.py: scheduler_loop] (毎分実行)
                   │
         ┌─────────┴─────────────────────┐
         ▼                               ▼
 [reap_stale_collection_jobs]    [scheduler_tick]
 (10分ハートビート断の刈取)        (next_run_at 判定)
                                         │
                                         ▼ Redis Enqueue Lock
                               [celery_app.enqueue]
                                         │
                                         ▼ Queue: "collection"
                   [celery_worker.py: run_collection_job]
                                         │
         ┌───────────────────────────────┼───────────────────────────────┐
         ▼                               ▼                               ▼
  [CollectionHeartbeat]          [_run_collector]                 [Redis Renewer]
  (60秒毎にDBへ生存報告)           │                               (30分毎にロック更新)
                                 ▼ (datasource_type == "web_report")
                   [data/collectors/web_report.py]
                                 │
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
          [Direct Download]           [Custom Steps]
          (日付展開・単一URL)          (link_parse / element_parse)
                   │                           │
                   └─────────────┬─────────────┘
                                 ▼
                     [Fetch Engine (Playwright / HTTP)]
                                 │
                     [Deduplication (_dedup_ok)]
                     - segment (パス/ファイル存在)
                     - checksum (SHA256ハッシュ)
                     - text (リンク文字列)
                                 │
                                 ▼
              [Artifact Storage: artifacts/web_reports/]
                                 │
                                 ▼
              [_on_file_saved] -> [report_forecast 抽出]
                                 │
                                 ▼
              [Dataset テーブル更新 (row_count = ファイル数)]
              [CollectionJob テーブル更新 (status = idle)]
              [Webhook: collection.completed]
```

---

## 3. 関連ファイル・クラス・関数の一覧

| レイヤ / 領域 | ファイルパス | 主要クラス / 関数 | 役割・責務 |
|---|---|---|---|
| スケジューラ & 死活監視 | `backend/data/scheduler.py` | `scheduler_loop`<br>`scheduler_tick`<br>`compute_next_run`<br>`retry_run_at`<br>`CollectionHeartbeat`<br>`reap_stale_collection_jobs` | APIプロセス内で常駐し、ジョブの定期起動、次回実行日時計算、WorkerのHeartbeat監視、Staleジョブの復旧を担当。 |
| Celery ワーカー | `backend/celery_worker.py` | `run_collection_job`<br>`_run_collection_job`<br>`_run_collector` | Redis分散ロックの維持、Heartbeatの起動、コレクターのディスパッチ、完了時のDataset/CollectionJob更新、Webhook発火。 |
| Celery 定義 | `backend/celery_app.py` | `enqueue`<br>`_acquire_lock`<br>`_release_lock` | Celeryアプリケーション設定、タスクルーティング（`collection`キュー）、二重エンキュー防止ロック。 |
| コレクター本体 | `backend/data/collectors/web_report.py` | `collect`<br>`preview_links`<br>`test_fetch`<br>`_save_file`<br>`_dedup_ok`<br>`_polite_delay`<br>`_launch_browser`<br>`_stealth_context` | Playwright/HTTPを用いたWebレポート取得、多段リンク解析、日付テンプレート展開、チェックサム重複排除、Polite Delay。 |
| 共通ユーティリティ | `backend/data/collectors/_utils.py` | `merge_into_parquet` | Parquet時系列データ向けのアトミックマージ（※レポート収集では未使用）。 |
| データモデル | `backend/data/models.py` | `Datasource`<br>`CollectionJob`<br>`Dataset`<br>`ReportForecast` | データベーステーブル定義（SQLAlchemy ORM & Pydanticスキーマ）。 |
| サービス層 | `backend/data/service.py` | `DataService.trigger_datasource_collection`<br>`create_datasource` | データソース作成、手動収集トリガーとCeleryエンキュー処理。 |
| REST API ルーター | `backend/data/router.py` | `trigger_collection`<br>`preview_web_report_links`<br>`test_web_report_fetch`<br>`list_web_report_files` | UIや外部クライアント向けのRESTエンドポイント。 |
| MCP ツール | `backend/mcp_server/tools/data.py` | `list_datasources`<br>`create_datasource`<br>`collect_data`<br>`list_datasets` | Claude/Agent向けMCPインターフェース。 |
| 自動テスト | `backend/tests/test_collection_scheduler.py`<br>`backend/tests/test_collection_backfill.py`<br>`backend/tests/test_report_forecasts.py` | 各種テスト関数 | スケジューラ、バックフィル、予想値抽出の単体・統合テスト。 |

---

## 4. 調査項目別監査結果

### 4.1 エントリーポイント、関連モジュール、主要クラス・関数
- **確認できた事実 (コード確認済)**:
  - 定期実行は `backend/data/scheduler.py:scheduler_loop` を起点とし、Celery タスク `celery_worker.run_collection_job` をエンキューする。
  - 手動実行は REST API (`POST /datasources/{id}/collect`) または MCP ツール (`collect_data`) からトリガーされる。
  - 実際のダウンロード処理は `data.collectors.web_report.collect()` にカプセル化されている。

### 4.2 スケジューラとジョブ管理
- **確認できた事実 (コード確認済)**:
  - **実行基盤**: Celery Beat 依存を排除し、FastAPI の Lifespan 内で動く非同期タスク `scheduler_loop` が担当（デフォルト60秒間隔）。
  - **スケジュール計算**: `compute_next_run` 関数が処理。`web_report` は `interval_days` と `download_time` ("HH:MM" UTC) を用いる。その他は `schedule_cron` (CronTrigger) を用いる。
  - **二重エンキュー防止**: エンキューと同時に `CollectionJob.next_run_at` を次回予定時刻に進めるため、キュー滞留中や同一周期内での多重エンキューが発生しない。

### 4.3 データソース設定・認証管理
- **確認できた事実 (コード確認済)**:
  - `Datasource.config` に JSON 形式で URL, 拡張子, 取得タイプ (`load`, `goto_load`, `goto_download`, `load_rep`), 重複ルール (`segment`, `checksum`, `text`), 多段設定 (`custom`) などを保持。
  - **認証**: ログイン認証（ID/Password, OAuth, Cookie注入, APIトークンヘッダー）の仕組みは存在しない。
  - **Bot対策**: Playwright 起動時に `--disable-blink-features=AutomationControlled` や navigator.webdriver 隠蔽スクリプト、システム Chrome/Edge の優先起動（channel="chrome"）による Stealth 化が行われている。

### 4.4 検索・取得・保存の処理フロー
- **確認できた事実 (コード確認済)**:
  - **日付展開 / バックフィル**: URLやファイル名に含まれる `{YYYYMMDD}` などの日付プレースホルダを過去 `backfill_days` 分（平日のみ）展開して巡回可能。
  - **多段探索 (`custom`)**: 親ページのリンク（`<a href>`）を正規表現マッチする `link_parse` と、CSSセレクタで要素を抽出する `element_parse` を再帰的に実行可能。
  - **保存先**: `artifacts/web_reports/{subfolder}/{filename}`。
  - **完了処理**: 収集完了後、`Dataset` テーブルの `row_count` に保存先フォルダ内の総ファイル数が記録され、`CollectionJob.status = "idle"` に復帰する。

### 4.5 重複排除と更新検知
- **確認できた事実 (コード確認済)**:
  - **重複排除 (`_dedup_ok`)**:
    - `segment`: 保存先ファイルが存在すればスキップ。
    - `checksum`: ダウンロードした生データの SHA-256 ハッシュを `artifacts/web_reports/checksums/{url_key}.txt` と突合。
    - `text`: リンクの表示テキストのハッシュで判定。
  - **更新検知**: `artifacts/web_reports/last_check_dates.json` に URL と最終アクセス日を保存し、`interval_days` 未満の再アクセスを抑止。
  - **限界**: HTTP ヘッダー（ETag, Last-Modified）の活用や、Git Commit SHA、文献のバージョン（arXiv v1/v2）を追跡する仕組みはない。

### 4.6 リトライ・レート制限・タイムアウト
- **確認できた事実 (コード確認済)**:
  - **レート制限**: `_polite_delay()` により、リクエスト間に 1.5〜2.5秒（または設定値 ±20%）のランダムジッター付きスリープを挿入。
  - **タイムアウト**: httpx は 20〜30秒、Playwright は 30〜60秒の明示的タイムアウトを設定。
  - **リトライ**: エラー時は `status = "error"` となり、`retry_run_at()` によって 30分後（環境変数 `ALGOFORGE_COLLECTION_RETRY_MINUTES`）に自動再試行がスケジュールされる。最大試行回数の制限カラムはない。

### 4.7 ログと障害時の復旧
- **確認できた事実 (コード確認済)**:
  - **分散ロック**: Redis キー `algoforge:enqueued:run_collection_job:{job_id}` で多重実行を防止。実行中はバックグラウンドスレッドが30分毎にTTLを更新。
  - **死活監視**: `CollectionHeartbeat` デーモンスレッドが60秒毎に `CollectionJob.last_heartbeat_at` を更新。
  - **ゾンビ刈り取り**: APIプロセスの `reap_stale_collection_jobs` が 10分以上ハートビートが途絶えたジョブを `status = "error"` に更新し、Webhook `collection.error` を発火して再試行をスケジュールする。

### 4.8 MCP公開機能との関係
- **確認できた事実 (コード確認済)**:
  - `list_datasources`, `create_datasource`, `collect_data`, `list_datasets` が MCP に公開されている。
  - 一方で、リンクプレビューやファイル一覧取得、生ファイルダウンロードは REST API のみであり、MCP からは利用できない。動的検索クエリを投げる MCP ツールは存在しない。

### 4.9 既存の自動テスト
- **確認できた事実 (コード確認済)**:
  - `test_collection_scheduler.py`（スケジュール計算、tick、reaper）
  - `test_collection_backfill.py`（日付パース、バックフィル展開、引数引き渡し）
  - `test_report_forecasts.py`（保存後フックによる抽出）
  - pytest 環境下では `_polite_delay` が自動スキップされる設計になっており、テストの高速実行が保証されている。

---

## 5. 論文・OSS収集への再利用性評価

### 5.1 再利用できる機能・知見（高価値）
1. **Stealth Playwright ブラウザ制御**:
   - Cloudflare / Akamai などの Bot 対策を回避して学術リポジトリや技術ブログの動的レンダリング結果を取得するノウハウ（User-Agent, `--disable-blink-features`, リアルブラウザ優先）。
2. **Polite Delay with Random Jitter**:
   - 外部サーバーへの過負荷や IP BAN を防止するジッター付き待機ロジック。
3. **Heartbeat & Reaper パターン**:
   - 長時間実行される収集ジョブが途中で死んだ場合の自動検知・復旧パターン。
4. **Redis 分散ロック & バックフィル展開**:
   - 二重エンキュー防止と安全な日次展開ロジック。

### 5.2 再利用が困難な機能・乖離点（低親和性）
1. **静的巡回 vs 動的検索**:
   - `web_report` は「登録済みの特定URL」を巡回する設計。HDE は「AIが考案したキーワードで arXiv や GitHub を検索し、上位候補を絞り込んで収集する」動的検索ワークフローであり、根本的な呼び出しモデルが異なる。
2. **データモデルの欠落**:
   - AlgoForge の `Dataset` は時系列・ファイル数を扱う構造であり、論文（DOI, 著者, アブストラクト, 引用数）や OSS（リポジトリURL, Commit SHA, スター数, ライセンス）のメタデータを格納するテーブルがない。
3. **状態管理の乖離**:
   - `web_report` は `last_check_dates.json` や `checksums/*.txt` などのローカル平文ファイルで重複排除している。HDE H0 の仕様では「SQLite の `sources` テーブルおよび `nodes` / `edges` テーブルによるリレーショナルかつグラフ構造での管理」が必須。
4. **保存先の結合度**:
   - `web_report` は AlgoForge の `artifacts/web_reports/` 配下に保存され、AlgoForge の UI やバックテストエンジンと結合している。HDEの成果物は `research_pipeline` 側の Knowledge Base に属するべきである。

---

## 6. 設計案の比較評価

| 評価軸 | 案A：既存Collector (`web_report.py`) を拡張 | 案B：共通収集基盤を抽出 (AlgoForge内リファクタリング) | 案C：HDE専用Collectorを追加 (`research_pipeline` 側) |
|---|---|---|---|
| **概要** | `web_report.py` に `paper` / `oss` モードや分岐を追加し、1つのコレクターで全て賄う。 | AlgoForge 内で共通HTTP/Browser層を切り出し、AlgoForge内に新Collectorを追加。 | AlgoForge は無変更。AlgoForgeの実装パターンを参考に、`research_pipeline` 内に専用基盤を構築。 |
| **変更範囲** | AlgoForge: `web_report.py`, `models.py`, `celery_worker.py` (広範囲) | AlgoForge: `backend/data/` 全般の再設計・抽出 (極めて広範囲) | AlgoForge: **変更なし (0行)**<br>`research_pipeline`: 新規構築 |
| **実装コスト** | 中〜高（既存コードの肥大化・複雑化） | 高（リファクタリングと移行作業） | **低〜中（制約なくクリーンに実装可能）** |
| **既存機能への影響** | **極めて高い（既存レポート収集の破壊リスク大）** | 高（既存機能全般の回帰リスク） | **皆無（リスクゼロ、完全分離）** |
| **テスト容易性** | 低（既存の回帰テストが多数必要） | 中（共通部分のテスト再設計が必要） | **極めて高い（独立して単体テスト可能）** |
| **H0要件適合度** | 低（AlgoForge DBのスキーマ変更が必要） | 中（AlgoForge DBにHDEテーブルが混入） | **最高（H0仕様のSQLiteスキーマに完全準拠）** |

---

## 7. 推奨する実装方式と根拠

### 推奨：案C（`research_pipeline` 側に HDE 専用収集基盤を新設）

#### 選定根拠:
1. **関心の分離（Separation of Concerns）**:
   - 統合設計書 v0.3 第1.3節「既存基盤を再利用する。AlgoForgeとStudy Managerの機能を重複実装しない」とあるが、これは「AlgoForgeの得意領域（バックテスト・ML学習）を重複実装しない」という意味である。
   - 文献・OSS探索と Knowledge Graph 管理は **AlgoForge の責務外（研究パイプライン固有の領域）** であり、AlgoForgeのDBに論文メタデータを持ち込むべきではない。
2. **回帰リスクの完全排除**:
   - AlgoForge はすでに安定稼働しており、日次の金融レポート収集や Celery ワーカーが実運用されている。ここに大規模な変更を加えるリスクを完全に回避できる。
3. **H0 実装の加速**:
   - `research_pipeline` 側に独立した Python パッケージ（例: `research_pipeline.collectors`）を作り、AlgoForge の `_stealth_context`, `_launch_browser`, `_polite_delay` の優れたロジックを **参照・移植（またはライブラリ化）** する方が、AlgoForge の Celery/Alembic 移行を待つより遥かに早く高品質に実装できる。

---

## 8. H0 実装計画の修正提案

統合仕様書記載の H0 フェーズについて、今回の監査結果を踏まえて以下の通り具体化を提案する。

### 修正後の H0 実装タスク
1. **H0.1: 共通ユーティリティの導入 (`research_pipeline`)**:
   - AlgoForge の知見を踏まえた `stealth_browser.py`（Playwright Stealth起動）および `rate_limiter.py`（Polite Delay）を `research_pipeline` 側に配置。
2. **H0.2: SQLite Source Repository の構築**:
   - 設計書仕様に準拠した SQLite テーブル（`sources`, `discovery_runs`）の作成と CRUD リポジトリの実装。
   - コンテンツハッシュ（SHA-256）と永続識別子（DOI, arXiv ID, Git Commit SHA）による重複判定ロジックの実装。
3. **H0.3: Paper Collector の実装**:
   - arXiv API / Web取得クライアントの実装。
   - タイトル、著者、アブストラクト、PDF URL、取得日時の構造化。
4. **H0.4: OSS Collector の実装**:
   - GitHub API / リポジトリ取得クライアントの実装。
   - README、コミットハッシュ、ライセンス、更新日時の構造化。
5. **H0.5: 単体テストと再現性検証**:
   - ネットワークアクセスをモックした単体テスト。
   - 同一資料の再取得時に正しく更新検知・重複排除されることの検証。

---

## 9. 未確認事項・今後の検討課題

1. **外部APIキーの管理方針**:
   - GitHub API（レート制限緩和用）や Semantic Scholar API のキーをどこで管理するか（`.env` の構成）。
2. **全文保存 vs メタデータ保存**:
   - 論文 PDF や Git リポジトリのワーキングツリー全体をローカルストレージに保持するか、メタデータとテキスト（Abstract / README）のみを保持し、必要時のみクローン・ダウンロードするか。
3. **AlgoForge との連携プロトコル**:
   - 将来的に AlgoForge のバックテストやインジケータ生成に OSS コードを反映させる際の受け渡しインターフェース（L2 実装ギャップ対応時）。

---
**Audit Completed by:** Antigravity Agent  
**Next Action:** ユーザーの承認後、推奨案Cに基づく H0 実装（`research_pipeline` 内の収集・保存基盤構築）の着手。
