"""
Module: algoforge_adapter.py
Location: src/research_pipeline/integration/algoforge_adapter.py

Implements non-invasive AlgoForge integration adapter, causal data contracts,
and virtual execution candidate evaluators for H3-P0.

Adheres strictly to H3-AlgoForge-integration-and-dryrun-plan-v0.4 (APPROVED).
"""

from __future__ import annotations

import hashlib
import math
import os
import sqlite3
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

from research_pipeline.discovery.econometric_modules import (
    HawkesDiscreteHazardModel,
    LocalJumpDetector,
    HamiltonLogRVRegimeModel,
)


# =============================================================================
# 1. Input Schema: MarketBarM5
# =============================================================================

class MarketBarM5(BaseModel):
    """
    Standardized M5 Market Bar schema enforcing strict temporal causality,
    explicit UTC timezones, exact 300.0-second interval, and finite numbers.
    """
    source_id: str = Field(..., description="データソース識別子 (例: 'algoforge_sqlite_m5')")
    bar_open_at: datetime = Field(..., description="バー開始時刻 (UTC)")
    bar_close_at: datetime = Field(..., description="バー終了時刻 (UTC)")
    data_available_at: datetime = Field(..., description="データ利用可能時刻 (UTC, >= bar_close_at)")
    open: float = Field(..., gt=0.0, description="始値 (正の有限値)")
    high: float = Field(..., gt=0.0, description="高値 (正の有限値)")
    low: float = Field(..., gt=0.0, description="安値 (正の有限値)")
    close: float = Field(..., gt=0.0, description="終値 (正の有限値)")
    volume: float = Field(..., ge=0.0, description="出来高 (非負の有限値)")
    is_final: bool = Field(..., description="バー確定フラグ (True必須, 未確定バー混入防止)")

    @model_validator(mode="after")
    def validate_bar_integrity(self) -> MarketBarM5:
        # 1. タイムゾーン検証 (明示的 UTC)
        for dt_name, dt_field in [
            ("bar_open_at", self.bar_open_at),
            ("bar_close_at", self.bar_close_at),
            ("data_available_at", self.data_available_at),
        ]:
            if dt_field.tzinfo is None or dt_field.tzinfo != timezone.utc:
                raise ValueError(f"Field {dt_name} must be explicitly UTC: got {dt_field}")

        # 2. M5バー幅厳密検証 (300秒)
        duration_sec = (self.bar_close_at - self.bar_open_at).total_seconds()
        if abs(duration_sec - 300.0) > 1e-6:
            raise ValueError(f"M5 bar duration must be exactly 300.0 seconds: got {duration_sec}")

        # 3. 時刻因果整合性
        if self.data_available_at < self.bar_close_at:
            raise ValueError(
                f"data_available_at ({self.data_available_at}) cannot precede bar_close_at ({self.bar_close_at})"
            )

        # 4. 有限値チェック (NaN/Inf 厳格排除)
        for val_name, val in [
            ("open", self.open),
            ("high", self.high),
            ("low", self.low),
            ("close", self.close),
            ("volume", self.volume),
        ]:
            if not math.isfinite(val):
                raise ValueError(f"Bar price field {val_name} must be finite: got {val}")

        # 5. OHLC 数値整合性
        if self.high < max(self.open, self.close):
            raise ValueError(
                f"High ({self.high}) must be >= max(Open, Close) ({max(self.open, self.close)})"
            )
        if self.low > min(self.open, self.close):
            raise ValueError(
                f"Low ({self.low}) must be <= min(Open, Close) ({min(self.open, self.close)})"
            )

        # 6. 未確定バー遮断
        if not self.is_final:
            raise ValueError("Only finalized bars (is_final=True) are accepted")

        return self


# =============================================================================
# 2. Output Schema: EconometricStateOutput
# =============================================================================

class EconometricStateOutput(BaseModel):
    """
    Standardized Econometric State and Signal Output schema enforcing the
    Fundamental Causality Inequality:
    earliest_execution >= signal_ready + delta_exec >= calc_start >= latest_input_available >= input_cutoff
    """
    model_version: str = Field(..., description="モデル実装バージョン (例: 'econometric_modules_v1.1')")
    parameter_version: str = Field(..., description="推定パラメータ・較正識別子")
    input_cutoff_at: datetime = Field(..., description="計算に使用した入力データの最新 bar_close_at (UTC)")
    latest_input_available_at: datetime = Field(..., description="計算に使用した全入力バーの中で最も遅い data_available_at (UTC)")
    calculation_started_at: datetime = Field(..., description="モデル計算開始時刻 (UTC, >= latest_input_available_at)")
    signal_ready_at: datetime = Field(..., description="計算完了時刻 (UTC, >= calculation_started_at)")
    delta_exec: float = Field(..., ge=0.0, description="執行余裕時間 (秒)")
    earliest_execution_at: datetime = Field(..., description="最早執行可能時刻 (UTC, >= signal_ready_at + delta_exec)")

    # Hawkes 状態・シグナル
    hawkes_hazard: float = Field(..., ge=0.0)
    hawkes_prob: float = Field(..., ge=0.0, le=1.0)
    hawkes_signal: Literal[-1, 0, 1] = Field(..., description="-1 (Sell), 0 (Flat), +1 (Buy)")

    # Jump 状態・シグナル
    jump_l_stat: float = Field(..., ge=0.0)
    jump_signal: Literal[-1, 0, 1] = Field(..., description="-1, 0, +1")
    bv_continuous_variance: float = Field(..., ge=0.0)

    # Hamilton レジーム状態
    regime_tranquil_prob: float = Field(..., ge=0.0, le=1.0)
    regime_crisis_prob: float = Field(..., ge=0.0, le=1.0)
    regime_fallback_mode: Literal["none", "quantile_initial", "lag1_carryover"] = Field(...)

    @model_validator(mode="after")
    def validate_causality_chain(self) -> EconometricStateOutput:
        # タイムゾーン検証
        for dt_name, dt_field in [
            ("input_cutoff_at", self.input_cutoff_at),
            ("latest_input_available_at", self.latest_input_available_at),
            ("calculation_started_at", self.calculation_started_at),
            ("signal_ready_at", self.signal_ready_at),
            ("earliest_execution_at", self.earliest_execution_at),
        ]:
            if dt_field.tzinfo is None or dt_field.tzinfo != timezone.utc:
                raise ValueError(f"Field {dt_name} must be explicitly UTC: got {dt_field}")

        # 厳格な因果不等式検証
        if self.latest_input_available_at < self.input_cutoff_at:
            raise ValueError(
                f"latest_input_available_at ({self.latest_input_available_at}) cannot precede input_cutoff_at ({self.input_cutoff_at})"
            )
        if self.calculation_started_at < self.latest_input_available_at:
            raise ValueError(
                f"calculation_started_at ({self.calculation_started_at}) cannot precede latest_input_available_at ({self.latest_input_available_at})"
            )
        if self.signal_ready_at < self.calculation_started_at:
            raise ValueError(
                f"signal_ready_at ({self.signal_ready_at}) cannot precede calculation_started_at ({self.calculation_started_at})"
            )

        min_earliest = self.signal_ready_at + timedelta(seconds=self.delta_exec)
        if self.earliest_execution_at < min_earliest:
            raise ValueError(
                f"earliest_execution_at ({self.earliest_execution_at}) cannot precede signal_ready_at + delta_exec ({min_earliest})"
            )

        # 有限値チェック
        for fld_name, fld in [
            ("hawkes_hazard", self.hawkes_hazard),
            ("hawkes_prob", self.hawkes_prob),
            ("jump_l_stat", self.jump_l_stat),
            ("bv_continuous_variance", self.bv_continuous_variance),
            ("regime_tranquil_prob", self.regime_tranquil_prob),
            ("regime_crisis_prob", self.regime_crisis_prob),
        ]:
            if not math.isfinite(fld):
                raise ValueError(f"Output state field {fld_name} must be finite: got {fld}")

        # レジーム確率正規化
        prob_sum = self.regime_tranquil_prob + self.regime_crisis_prob
        if abs(prob_sum - 1.0) > 1e-6:
            raise ValueError(f"Regime probabilities must sum to 1.0: got {prob_sum}")

        return self


# =============================================================================
# 3. Virtual Execution Candidate Event Schema: ExecutionCandidateEvent
# =============================================================================

class ExecutionCandidateEvent(BaseModel):
    """
    Virtual Execution Candidate Event schema evaluating two-way eligibility
    under the half-open interval [earliest_execution_at, signal_expiry_at).
    """
    candidate_id: str = Field(..., description="仮想執行候補の一意ID")
    signal_id: str = Field(..., description="対応する EconometricStateOutput のID")
    target_bar_open_at: datetime = Field(..., description="執行対象バーの開始時刻 (UTC)")
    signal_expiry_at: datetime = Field(..., description="シグナルの有効期限 (UTC, target_bar_close_at)")
    earliest_execution_at: datetime = Field(..., description="最早執行可能時刻 (UTC)")
    candidate_execution_at: Optional[datetime] = Field(None, description="評価対象となった価格イベントの時刻 (UTC)")
    candidate_price_event_id: Optional[str] = Field(None, description="評価対象価格イベント識別子")
    candidate_price: Optional[float] = Field(None, gt=0.0)

    # 判定フラグと棄却理由
    execution_eligible: bool = Field(..., description="因果条件を満たし約定適格であるか否か")
    rejection_reason: Optional[str] = Field(
        None,
        description="棄却理由 ('NO_CANDIDATE_PRICE_EVENT', 'CANDIDATE_PRECEDES_EARLIEST_TIME', 'EXPIRED_LATENCY_EXCEEDED', 'UNINITIALIZED_REGIME_QUANTILE_FALLBACK')",
    )

    @model_validator(mode="after")
    def validate_candidate_integrity(self) -> ExecutionCandidateEvent:
        for dt_name, dt_field in [
            ("target_bar_open_at", self.target_bar_open_at),
            ("signal_expiry_at", self.signal_expiry_at),
            ("earliest_execution_at", self.earliest_execution_at),
        ]:
            if dt_field.tzinfo is None or dt_field.tzinfo != timezone.utc:
                raise ValueError(f"Field {dt_name} must be explicitly UTC: got {dt_field}")

        if self.candidate_execution_at is not None:
            if self.candidate_execution_at.tzinfo is None or self.candidate_execution_at.tzinfo != timezone.utc:
                raise ValueError(f"candidate_execution_at must be explicitly UTC: got {self.candidate_execution_at}")

        # Two-way causality logic validation
        if self.execution_eligible:
            if self.candidate_execution_at is None:
                raise ValueError("execution_eligible=True requires candidate_execution_at")
            if self.candidate_execution_at < self.earliest_execution_at:
                raise ValueError(
                    f"Eligible candidate cannot precede earliest_execution_at: {self.candidate_execution_at} < {self.earliest_execution_at}"
                )
            if self.candidate_execution_at >= self.signal_expiry_at:
                raise ValueError(
                    f"Eligible candidate cannot be at or after signal_expiry_at: {self.candidate_execution_at} >= {self.signal_expiry_at}"
                )
            if self.rejection_reason is not None:
                raise ValueError(f"Eligible candidate must have rejection_reason=None: got {self.rejection_reason}")
        else:
            if self.rejection_reason is None:
                raise ValueError("Ineligible candidate must declare a rejection_reason")

        return self


# =============================================================================
# 4. Deterministic Virtual Clock
# =============================================================================

class DeterministicVirtualClock:
    """
    Deterministic logical clock providing monotonic, bit-for-bit reproducible
    virtual timestamps for simulation, delay injection, and replay testing.
    """

    def __init__(self, initial_time: datetime):
        if initial_time.tzinfo is None or initial_time.tzinfo != timezone.utc:
            raise ValueError("initial_time must be explicitly UTC")
        self._current_time = initial_time

    @property
    def current_time(self) -> datetime:
        return self._current_time

    def advance(self, seconds: float) -> datetime:
        if seconds < 0:
            raise ValueError("Deterministic clock cannot advance backwards")
        self._current_time += timedelta(seconds=seconds)
        return self._current_time

    def set_time(self, new_time: datetime) -> datetime:
        if new_time.tzinfo is None or new_time.tzinfo != timezone.utc:
            raise ValueError("new_time must be explicitly UTC")
        if new_time < self._current_time:
            raise ValueError(f"Deterministic clock monotonicity violated: {new_time} < {self._current_time}")
        self._current_time = new_time
        return self._current_time


# =============================================================================
# 5. Virtual Execution Candidate Evaluator
# =============================================================================

def evaluate_execution_candidate(
    signal_output: EconometricStateOutput,
    target_bar_open_at: datetime,
    candidate_price_events: Optional[List[Dict[str, Any]]] = None,
) -> ExecutionCandidateEvent:
    """
    Evaluates virtual execution candidates against a confirmed EconometricStateOutput.
    Strictly enforces:
    1. Initial regime quantile fallback -> strictly non-tradable FLAT (Signal = 0, UNINITIALIZED_REGIME_QUANTILE_FALLBACK)
    2. Zero-latency violation check: candidate before earliest_execution_at is rejected (CANDIDATE_PRECEDES_EARLIEST_TIME)
    3. Expiration check: candidate at or after target_bar_close_at (half-open [earliest, expiry)) is rejected (EXPIRED_LATENCY_EXCEEDED)
    4. Missing candidate price event check (NO_CANDIDATE_PRICE_EVENT)
    """
    signal_id = f"sig_{signal_output.input_cutoff_at.strftime('%Y%m%d_%H%M%S')}"
    candidate_id = f"cand_{target_bar_open_at.strftime('%Y%m%d_%H%M%S')}"
    signal_expiry_at = target_bar_open_at + timedelta(seconds=300.0)
    earliest_exec = signal_output.earliest_execution_at

    # Rule 1: Day 0 quantile fallback is strictly non-tradable
    if signal_output.regime_fallback_mode == "quantile_initial":
        return ExecutionCandidateEvent(
            candidate_id=candidate_id,
            signal_id=signal_id,
            target_bar_open_at=target_bar_open_at,
            signal_expiry_at=signal_expiry_at,
            earliest_execution_at=earliest_exec,
            candidate_execution_at=None,
            candidate_price_event_id=None,
            candidate_price=None,
            execution_eligible=False,
            rejection_reason="UNINITIALIZED_REGIME_QUANTILE_FALLBACK",
        )

    # Rule 4: No price events available
    if not candidate_price_events:
        return ExecutionCandidateEvent(
            candidate_id=candidate_id,
            signal_id=signal_id,
            target_bar_open_at=target_bar_open_at,
            signal_expiry_at=signal_expiry_at,
            earliest_execution_at=earliest_exec,
            candidate_execution_at=None,
            candidate_price_event_id=None,
            candidate_price=None,
            execution_eligible=False,
            rejection_reason="NO_CANDIDATE_PRICE_EVENT",
        )

    # Pick the earliest candidate price event in the stream
    # Sort events by timestamp
    sorted_events = sorted(candidate_price_events, key=lambda x: x["timestamp"])
    first_event = sorted_events[0]
    cand_time = first_event["timestamp"]
    cand_price = float(first_event["price"])
    event_id = str(first_event.get("event_id", f"tick_{cand_time.strftime('%Y%m%d_%H%M%S')}"))

    # Rule 2: Precedes earliest execution
    if cand_time < earliest_exec:
        return ExecutionCandidateEvent(
            candidate_id=candidate_id,
            signal_id=signal_id,
            target_bar_open_at=target_bar_open_at,
            signal_expiry_at=signal_expiry_at,
            earliest_execution_at=earliest_exec,
            candidate_execution_at=cand_time,
            candidate_price_event_id=event_id,
            candidate_price=cand_price,
            execution_eligible=False,
            rejection_reason="CANDIDATE_PRECEDES_EARLIEST_TIME",
        )

    # Rule 3: Expired (at or after expiry)
    if cand_time >= signal_expiry_at:
        return ExecutionCandidateEvent(
            candidate_id=candidate_id,
            signal_id=signal_id,
            target_bar_open_at=target_bar_open_at,
            signal_expiry_at=signal_expiry_at,
            earliest_execution_at=earliest_exec,
            candidate_execution_at=cand_time,
            candidate_price_event_id=event_id,
            candidate_price=cand_price,
            execution_eligible=False,
            rejection_reason="EXPIRED_LATENCY_EXCEEDED",
        )

    # Eligible: cand_time in [earliest_exec, signal_expiry_at)
    return ExecutionCandidateEvent(
        candidate_id=candidate_id,
        signal_id=signal_id,
        target_bar_open_at=target_bar_open_at,
        signal_expiry_at=signal_expiry_at,
        earliest_execution_at=earliest_exec,
        candidate_execution_at=cand_time,
        candidate_price_event_id=event_id,
        candidate_price=cand_price,
        execution_eligible=True,
        rejection_reason=None,
    )


# =============================================================================
# 6. AlgoForge Non-Invasiveness & Snapshot Handlers (3-Layer Protection)
# =============================================================================

def compute_file_sha256(filepath: str) -> Optional[str]:
    """Computes SHA-256 hash of a file if it exists, else returns None."""
    if not os.path.exists(filepath):
        return None
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def create_readonly_snapshot(source_db_path: str, snapshot_target_path: str) -> str:
    """
    Creates an isolated read-only research snapshot of an SQLite database
    using SQLite's online backup API, ensuring zero lock contention and no WAL mutations.
    """
    os.makedirs(os.path.dirname(os.path.abspath(snapshot_target_path)), exist_ok=True)
    src_uri = f"file:{os.path.abspath(source_db_path)}?mode=ro"
    src_conn = sqlite3.connect(src_uri, uri=True)
    dst_conn = sqlite3.connect(snapshot_target_path)
    try:
        src_conn.backup(dst_conn)
    finally:
        dst_conn.close()
        src_conn.close()
    return snapshot_target_path


def get_readonly_sqlite_connection(db_path: str) -> sqlite3.Connection:
    """
    Connects to SQLite with immutable read-only URI mode (?mode=ro).
    Any INSERT, UPDATE, or schema modification will raise sqlite3.OperationalError.
    """
    abs_path = os.path.abspath(db_path).replace("\\", "/")
    uri = f"file:{abs_path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def verify_algoforge_non_invasiveness(
    algoforge_repo_path: str,
    db_path: Optional[str] = None,
    pre_hashes: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, Any]:
    """
    Audits 3-layer non-invasiveness against AlgoForge:
    1. Read-only connection permission check (OperationalError on write)
    2. Git diff check via `git status --porcelain`
    3. SHA-256 checksum comparison across DB, WAL, and SHM files
    """
    results: Dict[str, Any] = {
        "git_status_clean": False,
        "git_untracked_algoforge_modified": False,
        "write_protection_active": False,
        "db_sha256_matched": True,
        "wal_sha256_matched": True,
        "shm_sha256_matched": True,
        "errors": [],
    }

    # 1. Git status check
    try:
        git_res = subprocess.run(
            ["git", "-C", algoforge_repo_path, "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        # Check if research_pipeline or new unapproved files were added to algoforge
        git_lines = git_res.stdout.strip().split("\n") if git_res.stdout.strip() else []
        has_new_research_artifacts = any("research_pipeline" in line for line in git_lines)
        results["git_status_clean"] = not has_new_research_artifacts
    except Exception as e:
        results["errors"].append(f"Git status error: {e}")

    # 2. SQLite read-only write protection verification (if DB is provided)
    if db_path and os.path.exists(db_path):
        try:
            conn = get_readonly_sqlite_connection(db_path)
            try:
                cursor = conn.cursor()
                cursor.execute("CREATE TABLE IF NOT EXISTS _test_invasive (id INT)")
                results["write_protection_active"] = False
                results["errors"].append("Write operation unexpectedly succeeded on read-only URI!")
            except sqlite3.OperationalError:
                results["write_protection_active"] = True
            finally:
                conn.close()
        except Exception as e:
            results["errors"].append(f"Connection test error: {e}")

    # 3. Hash verification
    if db_path and pre_hashes:
        post_db = compute_file_sha256(db_path)
        post_wal = compute_file_sha256(f"{db_path}-wal")
        post_shm = compute_file_sha256(f"{db_path}-shm")

        results["db_sha256_matched"] = (post_db == pre_hashes.get("db"))
        results["wal_sha256_matched"] = (post_wal == pre_hashes.get("wal"))
        results["shm_sha256_matched"] = (post_shm == pre_hashes.get("shm"))

    return results


# =============================================================================
# 7. Stream Continuity & Causal Execution Pipeline
# =============================================================================

class AlgoForgeDataStreamAdapter:
    """
    Non-invasive data adapter transforming market streams into strictly verified
    MarketBarM5 bars, feeding causal Hawkes, Jump, and Hamilton models, and
    evaluating virtual execution candidates.
    """

    def __init__(
        self,
        source_id: str = "algoforge_usdjpy_m5",
        delta_exec_m5: float = 0.001,   # 1 ms
        delta_daily: float = 1.0,        # 1.0 s
        virtual_clock: Optional[DeterministicVirtualClock] = None,
    ):
        self.source_id = source_id
        self.delta_exec_m5 = delta_exec_m5
        self.delta_daily = delta_daily
        self.clock = virtual_clock

        # Initialize core econometric modules
        self.hawkes = HawkesDiscreteHazardModel(
            mu=0.005, rho=0.80, gamma=0.30, event_std_window=20, event_threshold_mult=2.0
        )
        self.jump_detector = LocalJumpDetector(window_k=72, jump_threshold=3.5)
        self.hamilton = HamiltonLogRVRegimeModel(
            mu_0=-9.0, mu_1=-7.0, sigma_0=0.5, sigma_1=1.0, p00=0.90, p11=0.80, window_days=60
        )

        # State memory
        self.current_day_idx = 0
        self.hamilton_daily_history: List[float] = []
        self.prev_hamilton_params: Optional[Dict[str, np.ndarray]] = None
        self.current_regime_prob = np.array([0.80, 0.20], dtype=np.float64)
        self.current_fallback_mode: Literal["none", "quantile_initial", "lag1_carryover"] = "none"

    @staticmethod
    def validate_stream_continuity(bars: List[MarketBarM5]) -> None:
        """
        Validates stream-level temporal integrity:
        - Strict monotonic increasing bar_open_at
        - No duplicate timestamps
        - Continuous contiguous bars (t_{open, t+1} >= t_{close, t})
        """
        if not bars:
            return

        seen_opens = set()
        for i in range(len(bars)):
            bar = bars[i]
            if bar.bar_open_at in seen_opens:
                raise ValueError(f"Duplicate timestamp detected in stream: {bar.bar_open_at}")
            seen_opens.add(bar.bar_open_at)

            if i > 0:
                prev_bar = bars[i - 1]
                if bar.bar_open_at < prev_bar.bar_close_at:
                    raise ValueError(
                        f"Stream temporal overlap or disordered bars: current open {bar.bar_open_at} < prev close {prev_bar.bar_close_at}"
                    )

    def process_m5_bar_step(
        self,
        bar: MarketBarM5,
        historical_returns: np.ndarray,
        candidate_price_events: Optional[List[Dict[str, Any]]] = None,
        injected_calc_delay_sec: float = 0.002,
    ) -> Tuple[EconometricStateOutput, ExecutionCandidateEvent]:
        """
        Executes a single M5 bar step under strict causal contracts:
        1. Input cutoff = bar.bar_close_at
        2. Latest input available = bar.data_available_at
        3. Calculation started >= latest input available
        4. Signal ready = calc_start + calc_duration
        5. Earliest execution = signal_ready + delta_exec
        """
        cutoff_at = bar.bar_close_at
        available_at = bar.data_available_at

        # Clock synchronization
        if self.clock is not None:
            if self.clock.current_time < available_at:
                self.clock.set_time(available_at)
            calc_start_at = self.clock.current_time
            self.clock.advance(injected_calc_delay_sec)
            signal_ready_at = self.clock.current_time
        else:
            calc_start_at = max(datetime.now(timezone.utc), available_at)
            signal_ready_at = calc_start_at + timedelta(seconds=injected_calc_delay_sec)

        earliest_exec_at = signal_ready_at + timedelta(seconds=self.delta_exec_m5)

        # Module 1: Hawkes step
        if len(historical_returns) >= 20:
            window_returns = historical_returns[-50:] if len(historical_returns) >= 50 else historical_returns
            events, _ = self.hawkes.extract_events_from_returns(window_returns, 20, 2.0)
            h_traj, p_traj = self.hawkes.compute_hazard_trajectory(events, self.hawkes.mu, self.hawkes.rho, self.hawkes.gamma)
            hawkes_hazard = float(h_traj[-1])
            hawkes_prob = float(p_traj[-1])
            last_ret = historical_returns[-1]
            if hawkes_prob > 0.5:
                hawkes_sig = 1 if last_ret > 0 else (-1 if last_ret < 0 else 0)
            else:
                hawkes_sig = 0
        else:
            hawkes_hazard = float(self.hawkes.mu)
            hawkes_prob = float(1.0 - np.exp(-self.hawkes.mu))
            hawkes_sig = 0

        # Module 2: Jump step
        if len(historical_returns) >= 72:
            jump_res = self.jump_detector.detect_jumps(historical_returns)
            jump_l_stat = float(jump_res["l_statistic"][-1])
            jump_sig = int(jump_res["jump_signals"][-1])
            bv_var = float(jump_res["bv_squared"][-1])
        else:
            jump_l_stat = 0.0
            jump_sig = 0
            bv_var = 0.0

        # Output state creation
        state_output = EconometricStateOutput(
            model_version="econometric_modules_v1.1",
            parameter_version="h3_p0_causal_v0.4",
            input_cutoff_at=cutoff_at,
            latest_input_available_at=available_at,
            calculation_started_at=calc_start_at,
            signal_ready_at=signal_ready_at,
            delta_exec=self.delta_exec_m5,
            earliest_execution_at=earliest_exec_at,
            hawkes_hazard=hawkes_hazard,
            hawkes_prob=hawkes_prob,
            hawkes_signal=hawkes_sig,
            jump_l_stat=jump_l_stat,
            jump_signal=jump_sig,
            bv_continuous_variance=bv_var,
            regime_tranquil_prob=float(self.current_regime_prob[0]),
            regime_crisis_prob=float(self.current_regime_prob[1]),
            regime_fallback_mode=self.current_fallback_mode,
        )

        # Target bar open is the immediate next bar
        target_bar_open = bar.bar_close_at
        cand_event = evaluate_execution_candidate(
            signal_output=state_output,
            target_bar_open_at=target_bar_open,
            candidate_price_events=candidate_price_events,
        )

        return state_output, cand_event
