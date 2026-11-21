"""Causal M5 FX Backtest & Statistical Evaluation Engine (H3-P3 v0.7 Canonical).

Implements:
- Bid-OHLC execution with fixed spread S and one-way slippage delta
- Method A pre-trade sizing (no post-open price cancellation)
- 5-step gap/weekend settlement priority order
- Strict Cash + UnrealizedPnL mark-to-market accounting
- statsmodels OLS-HAC one-sided t-test & Holm-Bonferroni step-down
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as student_t


@dataclass(frozen=True)
class BacktestConfig:
    """Immutable backtest execution and accounting configuration."""
    initial_capital: float = 1_000_000.0
    spread_jpy: float = 0.010       # 1.0 pip full spread
    slippage_jpy: float = 0.002     # 0.2 pips one-way slippage
    risk_fraction: float = 0.005    # 0.5% equity risk per trade
    max_leverage: float = 10.0      # 10.0x max leverage
    lot_step_usd: int = 1000        # 1,000 USD minimum / step
    k_sl: float = 2.0               # ATR multiplier for SL
    k_tp: float = 3.0               # ATR multiplier for TP
    max_hold_bars: int = 48         # 48 actual M5 bars
    gap_cooldown_bars: int = 6      # 6 M5 bars after gap reopen
    friday_close_hour: int = 21     # Friday 21:30 UTC forced close
    friday_close_minute: int = 30
    ema_fast: int = 12
    ema_slow: int = 48
    atr_window: int = 20


@dataclass(frozen=True)
class GateConfig:
    """Econometric gate configuration."""
    use_hawkes: bool = False
    hawkes_threshold: float = float("inf")
    use_jump: bool = False
    jump_cooldown_bars: int = 3
    use_hamilton: bool = False
    hamilton_threshold: float = 0.80
    hamilton_scale: float = 0.5


def compute_indicators(df: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """Compute causal EMA_12, EMA_48, ATR_20, and gap flags on Bid OHLC."""
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True)
    out = out.sort_values("time").reset_index(drop=True)

    out["diff_sec"] = out["time"].diff().dt.total_seconds().fillna(300.0)
    out["is_gap_reopen"] = out["diff_sec"] > 300.0

    # Continuous EMA across gaps (price level continuity)
    out["ema_fast"] = out["close"].ewm(span=config.ema_fast, adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=config.ema_slow, adjust=False).mean()

    # True Range & 20-bar SMA ATR
    prev_close = out["close"].shift(1)
    tr1 = out["high"] - out["low"]
    tr2 = (out["high"] - prev_close).abs()
    tr3 = (out["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = out["high"].iloc[0] - out["low"].iloc[0]
    out["atr"] = tr.rolling(window=config.atr_window, min_periods=config.atr_window).mean()

    # Crossover signals at bar close
    diff_ema = out["ema_fast"] - out["ema_slow"]
    prev_diff = diff_ema.shift(1)
    out["cross_up"] = (diff_ema > 0) & (prev_diff <= 0)
    out["cross_down"] = (diff_ema < 0) & (prev_diff >= 0)

    return out


def _calc_unrealized_pnl(pos: Dict[str, Any], bid_close: float, spread: float) -> float:
    """Mark-to-market liquidation value without slippage."""
    q = pos["qty"]
    if pos["direction"] == 1:
        return q * (bid_close - pos["entry_price"])
    else:
        ask_close = bid_close + spread
        return q * (pos["entry_price"] - ask_close)


def run_causal_fx_backtest(
    df: pd.DataFrame,
    config: BacktestConfig,
    gates: GateConfig,
    hawkes_hazard: Optional[np.ndarray] = None,
    jump_flags: Optional[np.ndarray] = None,
    hamilton_pred_prob: Optional[np.ndarray] = None,
    eval_start_idx: int = 0,
    long_signal: Optional[np.ndarray] = None,
    short_signal: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Run deterministic bar-by-bar causal backtest from eval_start_idx onwards."""
    n = len(df)
    times = df["time"].values
    opens = df["open"].values.astype(np.float64)
    highs = df["high"].values.astype(np.float64)
    lows = df["low"].values.astype(np.float64)
    closes = df["close"].values.astype(np.float64)
    atrs = df["atr"].values.astype(np.float64)
    is_gap = df["is_gap_reopen"].values
    diff_sec = df["diff_sec"].values
    cross_up = long_signal if long_signal is not None else df["cross_up"].values
    cross_down = short_signal if short_signal is not None else df["cross_down"].values

    # Pre-extract UTC date, weekday, hour, minute
    dt_index = pd.DatetimeIndex(df["time"])
    dates = dt_index.date
    weekdays = dt_index.weekday  # 4 = Friday
    hours = dt_index.hour
    minutes = dt_index.minute

    cash = config.initial_capital
    pos: Optional[Dict[str, Any]] = None
    pending_entry: Optional[Dict[str, Any]] = None
    pending_exit: Optional[Dict[str, Any]] = None

    gap_cooldown_rem = 0
    jump_cooldown_rem = 0
    post_fill_risk_exceeded_count = 0

    trades: List[Dict[str, Any]] = []
    daily_equity_records: List[Dict[str, Any]] = []

    S = config.spread_jpy
    delta = config.slippage_jpy

    def close_position(p: Dict[str, Any], exit_px: float, exit_time: Any, reason: str, carried_weekend: bool) -> float:
        nonlocal cash
        if p["direction"] == 1:
            pnl = p["qty"] * (exit_px - p["entry_price"])
        else:
            pnl = p["qty"] * (p["entry_price"] - exit_px)
        cash += pnl
        trades.append({
            "direction": "LONG" if p["direction"] == 1 else "SHORT",
            "qty": p["qty"],
            "signal_timestamp": p["signal_timestamp"],
            "information_cutoff": p["information_cutoff"],
            "order_timestamp": p["order_timestamp"],
            "fill_timestamp": p["fill_timestamp"],
            "entry_price": p["entry_price"],
            "sl_price": p["sl_price"],
            "tp_price": p["tp_price"],
            "exit_timestamp": exit_time,
            "exit_price": exit_px,
            "exit_reason": reason,
            "pnl_jpy": pnl,
            "carried_over_weekend": carried_weekend,
            "post_fill_risk_exceeded": p["post_fill_risk_exceeded"],
        })
        return pnl

    for t in range(eval_start_idx, n):
        bar_open = opens[t]
        bar_high = highs[t]
        bar_low = lows[t]
        bar_close = closes[t]
        bar_time = pd.Timestamp(times[t])

        # Track gap cooldown (6 confirmed M5 bars after gap reopen)
        if is_gap[t]:
            gap_cooldown_rem = config.gap_cooldown_bars
        elif gap_cooldown_rem > 0:
            gap_cooldown_rem -= 1

        # -----------------------------------------------------------------
        # STEP 1A: Open-of-bar execution (fills, gap SL/TP, forced close)
        # -----------------------------------------------------------------
        carried_wknd = bool(is_gap[t] and weekdays[t] == 0)

        # 1) Execute pending entry at open[t] (only if flat and continuous)
        if pending_entry is not None and pos is None:
            direction = pending_entry["direction"]
            qty = pending_entry["qty"]
            atr_val = pending_entry["atr"]
            if direction == 1:
                entry_px = bar_open + S + delta
                sl_px = entry_px - config.k_sl * atr_val
                tp_px = entry_px + config.k_tp * atr_val
            else:
                entry_px = bar_open - delta
                sl_px = entry_px + config.k_sl * atr_val
                tp_px = entry_px - config.k_tp * atr_val

            # Check post-fill risk exceedance (Method A: log only, never cancel)
            actual_notional = qty * entry_px
            actual_lev = actual_notional / max(pending_entry["equity_at_signal"], 1e-9)
            actual_sl_loss = qty * (config.k_sl * atr_val + S + 2 * delta)
            risk_exceeded = (actual_lev > config.max_leverage + 1e-9) or (
                actual_sl_loss > config.risk_fraction * pending_entry["equity_at_signal"] * 1.05
            )
            if risk_exceeded:
                post_fill_risk_exceeded_count += 1

            pos = {
                "direction": direction,
                "qty": qty,
                "entry_price": entry_px,
                "sl_price": sl_px,
                "tp_price": tp_px,
                "bars_held": 0,
                "signal_timestamp": pending_entry["signal_timestamp"],
                "information_cutoff": pending_entry["information_cutoff"],
                "order_timestamp": pending_entry["order_timestamp"],
                "fill_timestamp": bar_time,
                "post_fill_risk_exceeded": risk_exceeded,
            }
            pending_entry = None

        # 2) Gap SL/TP check at open[t]
        exited_this_bar = False
        if pos is not None:
            if pos["direction"] == 1:
                if bar_open <= pos["sl_price"]:
                    close_position(pos, bar_open - delta, bar_time, "GAP_SL", carried_wknd)
                    pos = None
                    pending_exit = None
                    exited_this_bar = True
                elif bar_open >= pos["tp_price"]:
                    close_position(pos, bar_open - delta, bar_time, "GAP_TP", carried_wknd)
                    pos = None
                    pending_exit = None
                    exited_this_bar = True
            else:
                ask_open = bar_open + S
                if ask_open >= pos["sl_price"]:
                    close_position(pos, ask_open + delta, bar_time, "GAP_SL", carried_wknd)
                    pos = None
                    pending_exit = None
                    exited_this_bar = True
                elif ask_open <= pos["tp_price"]:
                    close_position(pos, ask_open + delta, bar_time, "GAP_TP", carried_wknd)
                    pos = None
                    pending_exit = None
                    exited_this_bar = True

        # 3) Pending forced/signal exit at open[t] if not already closed
        if pos is not None and pending_exit is not None:
            exit_px = (bar_open - delta) if pos["direction"] == 1 else (bar_open + S + delta)
            close_position(pos, exit_px, bar_time, pending_exit["reason"], carried_wknd)
            pos = None
            pending_exit = None
            exited_this_bar = True

        # 4) Intra-bar high/low SL/TP check (only if not closed at open)
        if pos is not None and not exited_this_bar:
            if pos["direction"] == 1:
                hit_sl = bar_low <= pos["sl_price"]
                hit_tp = bar_high >= pos["tp_price"]
                if hit_sl:
                    close_position(pos, pos["sl_price"] - delta, bar_time, "SL", False)
                    pos = None
                    pending_exit = None
                    exited_this_bar = True
                elif hit_tp:
                    close_position(pos, pos["tp_price"] - delta, bar_time, "TP", False)
                    pos = None
                    pending_exit = None
                    exited_this_bar = True
            else:
                ask_high = bar_high + S
                ask_low = bar_low + S
                hit_sl = ask_high >= pos["sl_price"]
                hit_tp = ask_low <= pos["tp_price"]
                if hit_sl:
                    close_position(pos, pos["sl_price"] + delta, bar_time, "SL", False)
                    pos = None
                    pending_exit = None
                    exited_this_bar = True
                elif hit_tp:
                    close_position(pos, pos["tp_price"] + delta, bar_time, "TP", False)
                    pos = None
                    pending_exit = None
                    exited_this_bar = True

        # -----------------------------------------------------------------
        # STEP 1B: Bar-close event updates & exit signal generation
        # -----------------------------------------------------------------
        is_friday_cutoff = (weekdays[t] == 4) and (
            (hours[t] > config.friday_close_hour)
            or (hours[t] == config.friday_close_hour and minutes[t] >= config.friday_close_minute)
        )

        if pos is not None:
            pos["bars_held"] += 1
            if pending_exit is None:
                if is_friday_cutoff:
                    pending_exit = {"reason": "FORCED_CLOSE"}
                elif pos["bars_held"] >= config.max_hold_bars:
                    pending_exit = {"reason": "MAX_HOLD"}
                elif (pos["direction"] == 1 and cross_down[t]) or (pos["direction"] == -1 and cross_up[t]):
                    pending_exit = {"reason": "SIGNAL_CLOSE"}

        # -----------------------------------------------------------------
        # STEP 2-4: Realized Cash + UnrealizedPnL -> Equity_t
        # -----------------------------------------------------------------
        unrealized = _calc_unrealized_pnl(pos, bar_close, S) if pos is not None else 0.0
        equity_t = cash + unrealized

        # Record daily equity at last bar of each UTC date
        is_last_bar_of_day = (t == n - 1) or (dates[t + 1] != dates[t])
        if is_last_bar_of_day:
            daily_equity_records.append({
                "date": dates[t],
                "info_available_utc": bar_time + pd.Timedelta(minutes=5),
                "cash": cash,
                "unrealized_pnl": unrealized,
                "equity": equity_t,
            })

        # -----------------------------------------------------------------
        # STEP 5: New Entry Signal & Method A Sizing (Flat only, no reversal)
        # -----------------------------------------------------------------
        if gates.use_jump and jump_flags is not None and jump_flags[t] == 1:
            jump_cooldown_rem = gates.jump_cooldown_bars
        elif jump_cooldown_rem > 0:
            jump_cooldown_rem -= 1

        # Next bar must be continuous 5-min bar for new entry
        next_bar_continuous = (t < n - 1) and (diff_sec[t + 1] <= 300.0)

        if (
            pos is None
            and pending_entry is None
            and next_bar_continuous
            and not is_friday_cutoff
            and gap_cooldown_rem == 0
            and (cross_up[t] or cross_down[t])
        ):
            atr_val = atrs[t]
            if not np.isnan(atr_val) and atr_val > 0.0 and equity_t > 0.0:
                # Check econometric gates
                blocked_by_hawkes = (
                    gates.use_hawkes
                    and hawkes_hazard is not None
                    and hawkes_hazard[t] > gates.hawkes_threshold
                )
                blocked_by_jump = (
                    gates.use_jump
                    and (
                        (jump_flags is not None and jump_flags[t] == 1)
                        or jump_cooldown_rem > 0
                    )
                )
                if not blocked_by_hawkes and not blocked_by_jump:
                    m_ham = 1.0
                    if (
                        gates.use_hamilton
                        and hamilton_pred_prob is not None
                        and hamilton_pred_prob[t] >= gates.hamilton_threshold
                    ):
                        m_ham = gates.hamilton_scale

                    sl_dist = config.k_sl * atr_val
                    raw_usd = m_ham * min(
                        (config.risk_fraction * equity_t) / sl_dist,
                        (config.max_leverage * equity_t) / bar_close,
                    )
                    qty = int(math.floor(raw_usd / config.lot_step_usd)) * config.lot_step_usd
                    if qty >= config.lot_step_usd:
                        t_close = bar_time + pd.Timedelta(minutes=5)
                        pending_entry = {
                            "direction": 1 if cross_up[t] else -1,
                            "qty": qty,
                            "atr": atr_val,
                            "equity_at_signal": equity_t,
                            "signal_timestamp": t_close,
                            "information_cutoff": t_close,
                            "order_timestamp": t_close,
                        }

    trades_df = pd.DataFrame(trades)
    daily_df = pd.DataFrame(daily_equity_records)
    if len(daily_df) > 0:
        prev_eq = daily_df["equity"].shift(1).fillna(config.initial_capital)
        daily_df["net_return"] = (daily_df["equity"] - prev_eq) / prev_eq
        cum_max = daily_df["equity"].cummax()
        dd = (daily_df["equity"] - cum_max) / cum_max
        max_dd = float(dd.min())
        rets = daily_df["net_return"].values
        ann_ret = float(np.mean(rets) * 252.0)
        ann_vol = float(np.std(rets, ddof=1) * np.sqrt(252.0)) if len(rets) > 1 else 0.0
        sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0.0
    else:
        max_dd = 0.0
        sharpe = 0.0
        ann_ret = 0.0

    summary = {
        "final_equity": float(daily_df["equity"].iloc[-1]) if len(daily_df) > 0 else config.initial_capital,
        "total_return_pct": ((float(daily_df["equity"].iloc[-1]) / config.initial_capital) - 1.0) * 100.0 if len(daily_df) > 0 else 0.0,
        "annualized_sharpe": sharpe,
        "max_drawdown_pct": max_dd * 100.0,
        "trade_count": len(trades_df),
        "win_rate": float((trades_df["pnl_jpy"] > 0).mean()) if len(trades_df) > 0 else 0.0,
        "post_fill_risk_exceeded_count": post_fill_risk_exceeded_count,
    }
    return trades_df, daily_df, summary


def compute_hac_one_sided_test(
    r_target: np.ndarray,
    r_ref: np.ndarray,
    maxlags: int = 4,
) -> Dict[str, Any]:
    """Exact statsmodels OLS-HAC one-sided t-test (H1: E[r_target - r_ref] > 0)."""
    diff = np.asarray(r_target, dtype=np.float64) - np.asarray(r_ref, dtype=np.float64)
    T = len(diff)
    if T < 2:
        raise ValueError(f"Insufficient observations T={T} for HAC test")

    if np.max(np.abs(diff)) == 0.0:
        return {
            "mean_diff": 0.0,
            "se_hac": 0.0,
            "t_stat": 0.0,
            "p_val": 1.0,
            "status": "ZERO_DIFF_EXACT",
        }

    ones = np.ones((T, 1), dtype=np.float64)
    res = sm.OLS(diff, ones).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": maxlags, "kernel": "bartlett", "use_correction": True},
    )
    mean_diff = float(res.params[0])
    se_hac = float(res.bse[0])
    if not np.isfinite(se_hac) or se_hac == 0.0:
        raise RuntimeError(f"NUMERICAL_DEGENERACY_ERROR: se_hac={se_hac}")

    t_stat = mean_diff / se_hac
    p_val = float(student_t.sf(t_stat, df=T - 1))
    return {
        "mean_diff": mean_diff,
        "se_hac": se_hac,
        "t_stat": t_stat,
        "p_val": p_val,
        "status": "OK",
    }


def run_holm_bonferroni(hypotheses: List[Dict[str, Any]], alpha: float = 0.05) -> List[Dict[str, Any]]:
    """Step-down Holm-Bonferroni FWER control."""
    m = len(hypotheses)
    sorted_hyps = sorted(hypotheses, key=lambda x: x["p_val"])
    stopped = False
    for k, h in enumerate(sorted_hyps):
        alpha_k = alpha / (m - k)
        h["rank"] = k + 1
        h["holm_alpha"] = alpha_k
        if not stopped and h["p_val"] <= alpha_k:
            h["supported"] = True
            h["decision"] = "REJECT_H0 (Supported)"
        else:
            h["supported"] = False
            h["decision"] = "FAIL_TO_REJECT (Not Supported)"
            stopped = True
    return sorted_hyps
