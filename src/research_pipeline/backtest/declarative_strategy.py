"""Declarative JSON Strategy Parser & Causal Signal Generator (Phase 1).

Supports:
- Multi-indicator AND/OR condition trees for Long and Short entries
- Parameterized indicators (EMA, RSI, Bollinger Bands, Donchian)
- UTC hour entry restrictions
- Strictly causal Higher Timeframe (H1 / H4) confirmed-bar filters
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def build_causal_htf_features(df_m5: pd.DataFrame) -> pd.DataFrame:
    """Resample M5 Bid OHLC to H1 and H4, aligning ONLY confirmed HTF bars to M5."""
    m5 = df_m5[["time", "open", "high", "low", "close"]].copy()
    m5["time"] = pd.to_datetime(m5["time"], utc=True)
    m5 = m5.sort_values("time").reset_index(drop=True)
    m5["m5_close_time"] = m5["time"] + pd.Timedelta(minutes=5)

    out = m5.copy()
    for label, freq, delta in [("h1", "1h", pd.Timedelta(hours=1)), ("h4", "4h", pd.Timedelta(hours=4))]:
        htf = (
            m5.set_index("time")
            .resample(freq, label="left", closed="left")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            .dropna()
            .reset_index()
        )
        htf["htf_close_time"] = htf["time"] + delta
        htf[f"{label}_ema12"] = htf["close"].ewm(span=12, adjust=False).mean()
        htf[f"{label}_ema48"] = htf["close"].ewm(span=48, adjust=False).mean()
        htf[f"{label}_trend"] = np.where(htf[f"{label}_ema12"] > htf[f"{label}_ema48"], 1, -1)

        merged = pd.merge_asof(
            out[["m5_close_time"]],
            htf[["htf_close_time", f"{label}_trend"]],
            left_on="m5_close_time",
            right_on="htf_close_time",
            direction="backward",
        )
        out[f"{label}_trend"] = merged[f"{label}_trend"].fillna(0).astype(np.int32)

    return out


def validate_declarative_strategy(strategy: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate declarative strategy JSON schema and parameter bounds."""
    for side in ("long", "short"):
        if side not in strategy or not isinstance(strategy[side], dict):
            return False, f"REJECTED_INVALID_SCHEMA: missing {side}"
        node = strategy[side]
        if node.get("op") not in {"AND", "OR"} or not isinstance(node.get("clauses"), list) or len(node["clauses"]) == 0:
            return False, f"REJECTED_INVALID_SCHEMA: invalid {side} op/clauses"
        for c in node["clauses"]:
            ctype = c.get("type")
            if ctype == "ema_cross":
                if c.get("fast", 0) <= 0 or c.get("slow", 0) <= 0 or c["fast"] >= c["slow"]:
                    return False, "REJECTED_INVALID_SCHEMA: ema fast >= slow"
            elif ctype in {"rsi_lt", "rsi_gt"}:
                if c.get("period", 0) < 2 or not (5 <= c.get("level", -1) <= 95):
                    return False, "REJECTED_INVALID_SCHEMA: invalid rsi params"
            elif ctype in {"bb_break_upper", "bb_break_lower"}:
                if c.get("period", 0) < 5 or c.get("std_mult", 0) <= 0:
                    return False, "REJECTED_INVALID_SCHEMA: invalid bb params"
            elif ctype in {"donchian_break_high", "donchian_break_low"}:
                if c.get("period", 0) < 5:
                    return False, "REJECTED_INVALID_SCHEMA: invalid donchian params"
            else:
                return False, f"REJECTED_INVALID_SCHEMA: unknown clause {ctype}"

    risk = strategy.get("risk", {})
    if risk.get("k_sl", 0) <= 0 or risk.get("k_tp", 0) <= 0 or risk.get("max_hold_bars", 0) <= 0:
        return False, "REJECTED_INVALID_SCHEMA: invalid risk params"
    return True, "OK"


def _eval_clause(df: pd.DataFrame, clause: Dict[str, Any]) -> np.ndarray:
    """Evaluate a single causal indicator clause on M5 DataFrame."""
    ctype = clause["type"]
    close = df["close"]
    n = len(df)

    if ctype == "ema_cross":
        direction = clause.get("dir", 1)
        ef = close.ewm(span=clause["fast"], adjust=False).mean()
        es = close.ewm(span=clause["slow"], adjust=False).mean()
        diff = ef - es
        pdiff = diff.shift(1)
        if direction == 1:
            return ((diff > 0) & (pdiff <= 0)).fillna(False).values
        return ((diff < 0) & (pdiff >= 0)).fillna(False).values

    if ctype in {"rsi_lt", "rsi_gt"}:
        delta = close.diff()
        gain = delta.clip(lower=0.0).rolling(clause["period"]).mean()
        loss = (-delta.clip(upper=0.0)).rolling(clause["period"]).mean()
        rs = gain / loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        if ctype == "rsi_lt":
            return (rsi < clause["level"]).fillna(False).values
        return (rsi > clause["level"]).fillna(False).values

    if ctype in {"bb_break_upper", "bb_break_lower"}:
        ma = close.rolling(clause["period"]).mean()
        sd = close.rolling(clause["period"]).std(ddof=1)
        if ctype == "bb_break_upper":
            up = ma + clause["std_mult"] * sd
            return ((close > up) & (close.shift(1) <= up.shift(1))).fillna(False).values
        lo = ma - clause["std_mult"] * sd
        return ((close < lo) & (close.shift(1) >= lo.shift(1))).fillna(False).values

    if ctype in {"donchian_break_high", "donchian_break_low"}:
        if ctype == "donchian_break_high":
            hi = df["high"].rolling(clause["period"]).max().shift(1)
            return (close > hi).fillna(False).values
        lo = df["low"].rolling(clause["period"]).min().shift(1)
        return (close < lo).fillna(False).values

    return np.zeros(n, dtype=bool)


def evaluate_declarative_signals(df_m5_htf: pd.DataFrame, strategy: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Generate causal Long and Short boolean entry arrays."""
    n = len(df_m5_htf)
    hours = pd.DatetimeIndex(df_m5_htf["time"]).hour.values
    allowed_hours = strategy.get("hours_utc", [])
    if allowed_hours:
        hour_mask = np.isin(hours, allowed_hours)
    else:
        hour_mask = np.ones(n, dtype=bool)

    htf = strategy.get("htf_filter", {})
    htf_long_mask = np.ones(n, dtype=bool)
    htf_short_mask = np.ones(n, dtype=bool)
    if htf.get("h1_align"):
        htf_long_mask &= df_m5_htf["h1_trend"].values == 1
        htf_short_mask &= df_m5_htf["h1_trend"].values == -1
    if htf.get("h4_align"):
        htf_long_mask &= df_m5_htf["h4_trend"].values == 1
        htf_short_mask &= df_m5_htf["h4_trend"].values == -1

    def eval_tree(node: Dict[str, Any]) -> np.ndarray:
        arrays = [_eval_clause(df_m5_htf, c) for c in node["clauses"]]
        if node["op"] == "AND":
            res = np.ones(n, dtype=bool)
            for a in arrays:
                res &= a
            return res
        res = np.zeros(n, dtype=bool)
        for a in arrays:
            res |= a
        return res

    long_sig = eval_tree(strategy["long"]) & hour_mask & htf_long_mask
    short_sig = eval_tree(strategy["short"]) & hour_mask & htf_short_mask
    return long_sig, short_sig
