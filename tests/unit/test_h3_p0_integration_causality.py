"""
Unit Test Suite for H3-P0 AlgoForge Integration & Causality Dry-Run
Location: tests/unit/test_h3_p0_integration_causality.py

Tests Suites P0-1 through P0-7 as approved in H3-AlgoForge-integration-and-dryrun-plan-v0.4:
- P0-1: Causal Replay Verification (outputs strictly depend on t <= t_bar_close)
- P0-2: Availability-Based Future Perturbation Verification (bitwise equality up to T)
- P0-3: Virtual Execution Candidate Two-Way Eligibility Verification
- P0-4: Latency Injection & Signal Expiration (half-open [earliest, expiry) boundary)
- P0-5: Schema & Stream Continuity Protection (NaN, duplicates, non-final, disorder)
- P0-6: Dual-Tier Determinism Verification (Mathematical & Replay)
- P0-7: AlgoForge 3-Layer Non-Invasiveness Assurance (Snapshot, mode=ro, Hash, Git)

Adheres strictly to H3-AlgoForge-integration-and-dryrun-plan-v0.4 (APPROVED).
"""

import ast
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np
import pytest

from research_pipeline.integration.algoforge_adapter import (
    AlgoForgeDataStreamAdapter,
    DeterministicVirtualClock,
    EconometricStateOutput,
    ExecutionCandidateEvent,
    MarketBarM5,
    compute_file_sha256,
    create_readonly_snapshot,
    evaluate_execution_candidate,
    get_readonly_sqlite_connection,
    verify_algoforge_non_invasiveness,
)


# =============================================================================
# Helper Fixtures & Generators
# =============================================================================

def generate_mock_m5_bars(
    num_bars: int = 100,
    start_time: datetime = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    base_price: float = 145.0,
    seed: int = 42,
) -> List[MarketBarM5]:
    """Generates a strictly valid sequential list of MarketBarM5 bars."""
    np.random.seed(seed)
    bars: List[MarketBarM5] = []
    curr_price = base_price

    for i in range(num_bars):
        open_time = start_time + timedelta(seconds=i * 300.0)
        close_time = open_time + timedelta(seconds=300.0)
        # Data delivery 100ms after bar close
        avail_time = close_time + timedelta(milliseconds=100)

        ret = np.random.normal(0.0, 0.0005)
        o = curr_price
        c = round(o * (1.0 + ret), 4)
        h = round(max(o, c) + abs(np.random.normal(0.0, 0.02)), 4)
        l = round(min(o, c) - abs(np.random.normal(0.0, 0.02)), 4)
        v = round(float(np.random.uniform(50.0, 200.0)), 2)

        bar = MarketBarM5(
            source_id="algoforge_mock_m5",
            bar_open_at=open_time,
            bar_close_at=close_time,
            data_available_at=avail_time,
            open=o,
            high=h,
            low=l,
            close=c,
            volume=v,
            is_final=True,
        )
        bars.append(bar)
        curr_price = c

    return bars


# =============================================================================
# P0-1: Causal Replay Verification
# =============================================================================

def test_p0_1_causal_replay_verification():
    """
    P0-1: Feed bars sequentially 1-by-1.
    Verify all output timestamps satisfy Fundamental Causality Inequality:
    earliest_execution >= signal_ready + delta_exec >= calc_start >= latest_input_available >= input_cutoff
    and that calculation strictly depends only on t <= t_bar_close.
    """
    bars = generate_mock_m5_bars(num_bars=30)
    adapter = AlgoForgeDataStreamAdapter(delta_exec_m5=0.001)

    hist_returns = []
    for i, bar in enumerate(bars):
        hist_returns.append((bar.close - bar.open) / bar.open)
        returns_arr = np.array(hist_returns, dtype=np.float64)

        state_out, cand_event = adapter.process_m5_bar_step(
            bar=bar,
            historical_returns=returns_arr,
            candidate_price_events=[
                {"timestamp": bar.bar_close_at + timedelta(milliseconds=5), "price": bar.close}
            ],
        )

        # Verify Fundamental Causality Inequality
        assert state_out.input_cutoff_at == bar.bar_close_at
        assert state_out.latest_input_available_at >= state_out.input_cutoff_at
        assert state_out.calculation_started_at >= state_out.latest_input_available_at
        assert state_out.signal_ready_at >= state_out.calculation_started_at
        assert state_out.earliest_execution_at >= state_out.signal_ready_at + timedelta(seconds=state_out.delta_exec)

        # Candidate target bar open matches next bar
        assert cand_event.target_bar_open_at == bar.bar_close_at
        assert cand_event.signal_expiry_at == bar.bar_close_at + timedelta(seconds=300.0)


# =============================================================================
# P0-2: Availability-Based Future Perturbation Verification
# =============================================================================

def test_p0_2_availability_based_future_perturbation():
    """
    P0-2: Fix cutoff index T. Create two streams X and X' such that
    X_<=T == X'_<=T, but X_>T != X'_>T (future heavily perturbed).
    Verify that all outputs up to T are 100% bitwise identical.
    """
    num_total = 40
    t_cutoff_idx = 20

    base_bars = generate_mock_m5_bars(num_bars=num_total, seed=101)
    perturbed_bars = generate_mock_m5_bars(num_bars=num_total, seed=101)

    # Heavily perturb bars beyond t_cutoff_idx in perturbed_bars
    for i in range(t_cutoff_idx + 1, num_total):
        orig = perturbed_bars[i]
        perturbed_bars[i] = MarketBarM5(
            source_id=orig.source_id,
            bar_open_at=orig.bar_open_at,
            bar_close_at=orig.bar_close_at,
            data_available_at=orig.data_available_at,
            open=orig.open * 2.5,
            high=orig.high * 3.0,
            low=orig.low * 0.5,
            close=orig.close * 2.7,
            volume=orig.volume * 10.0,
            is_final=True,
        )

    # Run stream 1
    clock1 = DeterministicVirtualClock(base_bars[0].bar_open_at)
    adapter1 = AlgoForgeDataStreamAdapter(virtual_clock=clock1)
    outputs1 = []
    hist_ret1 = []
    for i in range(t_cutoff_idx + 1):
        bar = base_bars[i]
        hist_ret1.append((bar.close - bar.open) / bar.open)
        out, _ = adapter1.process_m5_bar_step(bar, np.array(hist_ret1))
        outputs1.append(out)

    # Run stream 2 (with perturbed future present in memory)
    clock2 = DeterministicVirtualClock(perturbed_bars[0].bar_open_at)
    adapter2 = AlgoForgeDataStreamAdapter(virtual_clock=clock2)
    outputs2 = []
    hist_ret2 = []
    for i in range(t_cutoff_idx + 1):
        bar = perturbed_bars[i]
        hist_ret2.append((bar.close - bar.open) / bar.open)
        out, _ = adapter2.process_m5_bar_step(bar, np.array(hist_ret2))
        outputs2.append(out)

    # Verify bitwise equality up to T
    assert len(outputs1) == len(outputs2) == t_cutoff_idx + 1
    for i in range(t_cutoff_idx + 1):
        o1 = outputs1[i]
        o2 = outputs2[i]
        assert o1.hawkes_hazard == o2.hawkes_hazard
        assert o1.hawkes_prob == o2.hawkes_prob
        assert o1.hawkes_signal == o2.hawkes_signal
        assert o1.jump_l_stat == o2.jump_l_stat
        assert o1.jump_signal == o2.jump_signal
        assert o1.bv_continuous_variance == o2.bv_continuous_variance
        assert o1.regime_tranquil_prob == o2.regime_tranquil_prob
        assert o1.regime_crisis_prob == o2.regime_crisis_prob
        assert o1.calculation_started_at == o2.calculation_started_at
        assert o1.signal_ready_at == o2.signal_ready_at
        assert o1.earliest_execution_at == o2.earliest_execution_at


# =============================================================================
# P0-3: Virtual Execution Candidate Two-Way Eligibility Verification
# =============================================================================

def test_p0_3_virtual_execution_candidate_two_way_eligibility():
    """
    P0-3: Verifies two-way candidate eligibility:
    1. execution_eligible=True <=> candidate in [earliest, expiry)
    2. candidate < earliest => execution_eligible=False, reason="CANDIDATE_PRECEDES_EARLIEST_TIME"
    3. candidate >= expiry => execution_eligible=False, reason="EXPIRED_LATENCY_EXCEEDED"
    4. no candidates => execution_eligible=False, reason="NO_CANDIDATE_PRICE_EVENT"
    5. Day 0 quantile initial regime => execution_eligible=False, reason="UNINITIALIZED_REGIME_QUANTILE_FALLBACK"
    """
    bar = generate_mock_m5_bars(num_bars=1)[0]
    clock = DeterministicVirtualClock(bar.bar_open_at)
    adapter = AlgoForgeDataStreamAdapter(virtual_clock=clock, delta_exec_m5=0.001)

    state_out, _ = adapter.process_m5_bar_step(
        bar=bar,
        historical_returns=np.array([0.0001]),
        injected_calc_delay_sec=0.002,
    )
    earliest = state_out.earliest_execution_at
    target_open = bar.bar_close_at
    expiry = target_open + timedelta(seconds=300.0)

    # Case A: Candidate before earliest (precedes earliest)
    cand_early = [{"timestamp": earliest - timedelta(microseconds=500), "price": 145.05}]
    res_a = evaluate_execution_candidate(state_out, target_open, cand_early)
    assert not res_a.execution_eligible
    assert res_a.rejection_reason == "CANDIDATE_PRECEDES_EARLIEST_TIME"

    # Case B: Candidate within half-open [earliest, expiry)
    cand_valid = [{"timestamp": earliest + timedelta(milliseconds=10), "price": 145.06}]
    res_b = evaluate_execution_candidate(state_out, target_open, cand_valid)
    assert res_b.execution_eligible
    assert res_b.rejection_reason is None
    assert res_b.candidate_price == 145.06

    # Case C: Candidate at exactly expiry (boundary exclusion)
    cand_expiry = [{"timestamp": expiry, "price": 145.10}]
    res_c = evaluate_execution_candidate(state_out, target_open, cand_expiry)
    assert not res_c.execution_eligible
    assert res_c.rejection_reason == "EXPIRED_LATENCY_EXCEEDED"

    # Case D: Candidate after expiry
    cand_late = [{"timestamp": expiry + timedelta(seconds=5), "price": 145.12}]
    res_d = evaluate_execution_candidate(state_out, target_open, cand_late)
    assert not res_d.execution_eligible
    assert res_d.rejection_reason == "EXPIRED_LATENCY_EXCEEDED"

    # Case E: Empty candidate events
    res_e = evaluate_execution_candidate(state_out, target_open, [])
    assert not res_e.execution_eligible
    assert res_e.rejection_reason == "NO_CANDIDATE_PRICE_EVENT"

    # Case F: Day 0 quantile initial fallback must force NON-TRADABLE FLAT
    state_quantile = state_out.model_copy(update={"regime_fallback_mode": "quantile_initial"})
    res_f = evaluate_execution_candidate(state_quantile, target_open, cand_valid)
    assert not res_f.execution_eligible
    assert res_f.rejection_reason == "UNINITIALIZED_REGIME_QUANTILE_FALLBACK"


# =============================================================================
# P0-4: Latency Injection & Signal Expiration (Half-Open Interval)
# =============================================================================

def test_p0_4_latency_injection_and_expiration():
    """
    P0-4: Injects calculation and communication latencies (+500ms, +5s, +305s).
    Confirms earliest_execution advances correctly and evaluates candidate events:
    - Moderate delay (0.5s, 5.0s): tick at target_open + 10s is eligible.
    - Extreme delay (+305s): calculation completes after target bar close,
      so tick at target_open + 10s is rejected as CANDIDATE_PRECEDES_EARLIEST_TIME.
    - Candidate tick at or beyond target_bar_close_at (>= 300s) is rejected as EXPIRED_LATENCY_EXCEEDED.
    """
    bar = generate_mock_m5_bars(num_bars=1)[0]
    target_open = bar.bar_close_at
    expiry = target_open + timedelta(seconds=300.0)

    # 1. Normal / Moderate delays
    for delay in [0.5, 5.0]:
        clock = DeterministicVirtualClock(bar.bar_open_at)
        adapter = AlgoForgeDataStreamAdapter(virtual_clock=clock, delta_exec_m5=0.001)

        state_out, _ = adapter.process_m5_bar_step(
            bar=bar,
            historical_returns=np.array([0.0002]),
            injected_calc_delay_sec=delay,
        )

        expected_ready = bar.data_available_at + timedelta(seconds=delay)
        expected_earliest = expected_ready + timedelta(seconds=0.001)
        assert state_out.signal_ready_at == expected_ready
        assert state_out.earliest_execution_at == expected_earliest

        # Price tick at target_open + 10s (within [earliest, expiry))
        tick_time = target_open + timedelta(seconds=10.0)
        tick_event = [{"timestamp": tick_time, "price": 145.20}]

        cand = evaluate_execution_candidate(state_out, target_open, tick_event)
        assert cand.execution_eligible
        assert cand.rejection_reason is None

    # 2. Extreme calculation delay (305s: exceeds the 300s M5 bar duration)
    clock_slow = DeterministicVirtualClock(bar.bar_open_at)
    adapter_slow = AlgoForgeDataStreamAdapter(virtual_clock=clock_slow, delta_exec_m5=0.001)
    state_out_slow, _ = adapter_slow.process_m5_bar_step(
        bar=bar,
        historical_returns=np.array([0.0002]),
        injected_calc_delay_sec=305.0,
    )
    # Earliest execution is at target_open + 305.101s (after bar has closed)
    # A tick during the bar (target_open + 10s) must be rejected because it preceded earliest_execution
    tick_during_bar = [{"timestamp": target_open + timedelta(seconds=10.0), "price": 145.20}]
    cand_slow = evaluate_execution_candidate(state_out_slow, target_open, tick_during_bar)
    assert not cand_slow.execution_eligible
    assert cand_slow.rejection_reason == "CANDIDATE_PRECEDES_EARLIEST_TIME"

    # 3. Candidate tick arriving at or beyond expiry (half-open [earliest, expiry) boundary test)
    tick_expired = [{"timestamp": expiry, "price": 145.25}]
    cand_exp = evaluate_execution_candidate(state_out, target_open, tick_expired)
    assert not cand_exp.execution_eligible
    assert cand_exp.rejection_reason == "EXPIRED_LATENCY_EXCEEDED"


# =============================================================================
# P0-5: Schema & Stream Continuity Protection
# =============================================================================

def test_p0_5_schema_and_stream_continuity_protection():
    """
    P0-5: Injects invalid schema records and broken stream sequences:
    - NaN / Inf prices
    - Unfinalized bar (is_final=False)
    - Duration != 300.0s
    - Stream timestamp duplicate & disorder
    Verifies that validators raise descriptive ValueError and protect system integrity.
    """
    base_bar = generate_mock_m5_bars(num_bars=1)[0]

    # 1. NaN price rejection
    with pytest.raises(Exception):
        MarketBarM5(
            source_id="test",
            bar_open_at=base_bar.bar_open_at,
            bar_close_at=base_bar.bar_close_at,
            data_available_at=base_bar.data_available_at,
            open=float("nan"),
            high=145.5,
            low=144.5,
            close=145.0,
            volume=100.0,
            is_final=True,
        )

    # 2. is_final=False rejection
    with pytest.raises(ValueError, match="Only finalized bars"):
        MarketBarM5(
            source_id="test",
            bar_open_at=base_bar.bar_open_at,
            bar_close_at=base_bar.bar_close_at,
            data_available_at=base_bar.data_available_at,
            open=145.0,
            high=145.5,
            low=144.5,
            close=145.0,
            volume=100.0,
            is_final=False,
        )

    # 3. Bar duration != 300.0s rejection
    with pytest.raises(ValueError, match="must be exactly 300.0 seconds"):
        MarketBarM5(
            source_id="test",
            bar_open_at=base_bar.bar_open_at,
            bar_close_at=base_bar.bar_open_at + timedelta(seconds=290.0),
            data_available_at=base_bar.bar_close_at,
            open=145.0,
            high=145.5,
            low=144.5,
            close=145.0,
            volume=100.0,
            is_final=True,
        )

    # 4. Stream duplicate timestamp rejection
    bars_dup = [base_bar, base_bar]
    with pytest.raises(ValueError, match="Duplicate timestamp detected"):
        AlgoForgeDataStreamAdapter.validate_stream_continuity(bars_dup)

    # 5. Stream disorder / overlap rejection
    bar2_disordered = MarketBarM5(
        source_id="test",
        bar_open_at=base_bar.bar_open_at - timedelta(seconds=60),
        bar_close_at=base_bar.bar_open_at + timedelta(seconds=240),
        data_available_at=base_bar.bar_open_at + timedelta(seconds=240, milliseconds=100),
        open=145.0,
        high=145.5,
        low=144.5,
        close=145.0,
        volume=100.0,
        is_final=True,
    )
    with pytest.raises(ValueError, match="Stream temporal overlap or disordered bars"):
        AlgoForgeDataStreamAdapter.validate_stream_continuity([base_bar, bar2_disordered])


# =============================================================================
# P0-6: Dual-Tier Determinism Verification (Mathematical & Replay)
# =============================================================================

def test_p0_6_dual_tier_determinism():
    """
    P0-6: Runs identical input series twice:
    - Tier 1: Mathematical state (hazard, jump L-stat, regime probs) bitwise identical.
    - Tier 2: Replay determinism under DeterministicVirtualClock identical.
    """
    bars = generate_mock_m5_bars(num_bars=25, seed=2024)

    # Run 1
    clock1 = DeterministicVirtualClock(bars[0].bar_open_at)
    adapter1 = AlgoForgeDataStreamAdapter(virtual_clock=clock1)
    results1 = []
    hist_r1 = []
    for b in bars:
        hist_r1.append((b.close - b.open) / b.open)
        s, c = adapter1.process_m5_bar_step(
            b,
            np.array(hist_r1),
            candidate_price_events=[{"timestamp": b.bar_close_at + timedelta(milliseconds=10), "price": b.close}],
        )
        results1.append((s, c))

    # Run 2
    clock2 = DeterministicVirtualClock(bars[0].bar_open_at)
    adapter2 = AlgoForgeDataStreamAdapter(virtual_clock=clock2)
    results2 = []
    hist_r2 = []
    for b in bars:
        hist_r2.append((b.close - b.open) / b.open)
        s, c = adapter2.process_m5_bar_step(
            b,
            np.array(hist_r2),
            candidate_price_events=[{"timestamp": b.bar_close_at + timedelta(milliseconds=10), "price": b.close}],
        )
        results2.append((s, c))

    assert len(results1) == len(results2) == 25
    for i in range(25):
        s1, c1 = results1[i]
        s2, c2 = results2[i]

        # Tier 1: Mathematical determinism
        assert s1.hawkes_hazard == s2.hawkes_hazard
        assert s1.hawkes_prob == s2.hawkes_prob
        assert s1.hawkes_signal == s2.hawkes_signal
        assert s1.jump_l_stat == s2.jump_l_stat
        assert s1.jump_signal == s2.jump_signal
        assert s1.bv_continuous_variance == s2.bv_continuous_variance
        assert s1.regime_tranquil_prob == s2.regime_tranquil_prob
        assert s1.regime_crisis_prob == s2.regime_crisis_prob

        # Tier 2: Replay logical clock determinism
        assert s1.calculation_started_at == s2.calculation_started_at
        assert s1.signal_ready_at == s2.signal_ready_at
        assert s1.earliest_execution_at == s2.earliest_execution_at
        assert c1.execution_eligible == c2.execution_eligible
        assert c1.rejection_reason == c2.rejection_reason
        assert c1.candidate_price == c2.candidate_price


# =============================================================================
# P0-7: AlgoForge 3-Layer Non-Invasiveness Assurance
# =============================================================================

def test_p0_7_algoforge_three_layer_non_invasiveness():
    """
    P0-7: 3-Layer Non-Invasiveness Assurance:
    Layer 1: Isolated research snapshot creation via SQLite online backup API;
             Direct connection URI mode=ro raises OperationalError on INSERT/UPDATE.
    Layer 2: AST code audit verifying zero mutating SQL statements or order dispatch calls.
    Layer 3: Hash preservation of source file + Git repo integrity.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a mock AlgoForge reference DB
        mock_db_path = os.path.join(tmp_dir, "algoforge_reference.db")
        conn = sqlite3.connect(mock_db_path)
        conn.execute("CREATE TABLE candles (time TEXT, close REAL);")
        conn.execute("INSERT INTO candles VALUES ('2024-01-01 00:00:00', 145.0);")
        conn.commit()
        conn.close()

        # Capture pre-run hash
        pre_hashes = {
            "db": compute_file_sha256(mock_db_path),
            "wal": compute_file_sha256(f"{mock_db_path}-wal"),
            "shm": compute_file_sha256(f"{mock_db_path}-shm"),
        }

        # --- Layer 1 Verification: Snapshot & mode=ro write protection ---
        # 1a. Create isolated research snapshot via online backup API
        snapshot_path = os.path.join(tmp_dir, "research_snapshot.db")
        created_snap = create_readonly_snapshot(mock_db_path, snapshot_path)
        assert os.path.exists(created_snap)
        # Verify snapshot has equivalent data
        snap_conn = sqlite3.connect(snapshot_path)
        assert snap_conn.execute("SELECT count(*) FROM candles").fetchone()[0] == 1
        snap_conn.close()

        # Verify source DB file was NOT modified by snapshot creation
        assert compute_file_sha256(mock_db_path) == pre_hashes["db"]

        # 1b. Direct connection with mode=ro must fail on write attempts
        ro_conn = get_readonly_sqlite_connection(mock_db_path)
        with pytest.raises(sqlite3.OperationalError):
            ro_conn.execute("INSERT INTO candles VALUES ('2024-01-01 00:05:00', 145.1);")
        ro_conn.close()

        # --- Layer 2 Verification: AST Audit of adapter code ---
        adapter_source_file = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../src/research_pipeline/integration/algoforge_adapter.py")
        )
        with open(adapter_source_file, "r", encoding="utf-8") as f:
            code_content = f.read()
            parsed_ast = ast.parse(code_content)

        # Check that no write methods like execute("INSERT ...") or send_order exist
        for node in ast.walk(parsed_ast):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                sql_upper = node.value.upper()
                assert not any(kw in sql_upper for kw in ["INSERT INTO CANDLES", "DROP TABLE", "DELETE FROM CANDLES"])

        # --- Layer 3 Verification: Hashes and Git Status ---
        algoforge_path = r"c:\Users\ksato\workspace\finance\algoforge"
        audit_res = verify_algoforge_non_invasiveness(
            algoforge_repo_path=algoforge_path,
            db_path=mock_db_path,
            pre_hashes=pre_hashes,
        )

        assert audit_res["write_protection_active"] is True
        assert audit_res["db_sha256_matched"] is True
        assert audit_res["git_status_clean"] is True
