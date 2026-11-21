"""Phase 2B-2c: Information Value Evaluation for unit:hawkes_hazard and unit:hamilton_pred_prob."""

import hashlib
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from research_pipeline.integration.algoforge_adapter import apply_units_with_time_check
from research_pipeline.discovery.econometric_modules import HawkesDiscreteHazardModel

DATA_PATH = os.path.join(_REPO_ROOT, "data", "usdjpy_m5_2024_oanda_snapshot.parquet")
OUT_PATH = os.path.join(_REPO_ROOT, "results", "phase2b2c_unit_value_results.json")
IS_START = pd.Timestamp("2024-07-01").date()
IS_END = pd.Timestamp("2024-09-27").date()
OOS_START = pd.Timestamp("2024-09-30").date()
OOS_END = pd.Timestamp("2024-12-31").date()


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _block_bootstrap_diff(
    days_arr: np.ndarray,
    y_true: np.ndarray,
    u_pred: np.ndarray,
    b_pred: np.ndarray,
    metric_fn,
    block_len: int,
    n_iter: int = 2000,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    unique_days = np.array(sorted(set(days_arr)))
    n_days = len(unique_days)
    day_to_idx = {d: np.where(days_arr == d)[0] for d in unique_days}

    diffs = []
    invalid = 0
    n_blocks = int(np.ceil(n_days / block_len))
    for _ in range(n_iter):
        starts = rng.integers(0, n_days - block_len + 1, size=n_blocks)
        sampled_days = []
        for s in starts:
            sampled_days.extend(unique_days[s : s + block_len])
        sampled_days = sampled_days[:n_days]
        idx = np.concatenate([day_to_idx[d] for d in sampled_days])
        yt, up, bp = y_true[idx], u_pred[idx], b_pred[idx]
        mu = metric_fn(yt, up)
        mb = metric_fn(yt, bp)
        if mu is None or mb is None:
            invalid += 1
        else:
            diffs.append(mu - mb)
    if not diffs:
        return {"valid_iters": 0, "invalid_iters": invalid, "ci95": [None, None]}
    ci = np.percentile(diffs, [2.5, 97.5]).tolist()
    return {"valid_iters": len(diffs), "invalid_iters": invalid, "ci95": [round(ci[0], 6), round(ci[1], 6)]}


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray):
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _safe_spearman(y_true: np.ndarray, y_score: np.ndarray):
    if len(np.unique(y_true)) < 2 or len(np.unique(y_score)) < 2:
        return None
    r, _ = spearmanr(y_true, y_score)
    return float(r) if np.isfinite(r) else None


def main() -> None:
    t0 = time.perf_counter()
    data_sha = _file_sha256(DATA_PATH)
    df = pd.read_parquet(DATA_PATH)
    if "time" not in df.columns and "timestamp" in df.columns:
        df["time"] = df["timestamp"]
    df["time"] = pd.to_datetime(df["time"], utc=True)

    t_pred0 = time.perf_counter()
    df_u, _ = apply_units_with_time_check(
        df,
        [
            {"id": "hawkes", "type": "unit:hawkes_hazard", "params": {"mle_window": 500, "refit_interval": 50}},
            {"id": "ham", "type": "unit:hamilton_pred_prob", "params": {"window_days": 60, "min_bars_per_day": 280}},
        ],
    )
    pred_sec = time.perf_counter() - t_pred0

    # --- Hawkes evaluation ---
    close = df_u["close"].to_numpy(dtype=np.float64)
    n = len(df_u)
    ret = np.zeros(n, dtype=np.float64)
    ret[1:] = (close[1:] - close[:-1]) / np.where(close[:-1] != 0.0, close[:-1], 1.0)
    events, _ = HawkesDiscreteHazardModel.extract_events_from_returns(ret, lookback_window=20, threshold_mult=2.0)

    std20 = pd.Series(ret).rolling(20).std(ddof=1).to_numpy()
    rate20 = pd.Series(events).rolling(20).mean().to_numpy()
    ts = df_u["time"]
    gap_next = (ts.shift(-1) - ts).dt.total_seconds().to_numpy()

    # Pair (t, t+1): predict events[t+1] from hawkes[t], std20[t], rate20[t]
    valid_pair = (
        df_u["hawkes_valid"].to_numpy()[:-1]
        & np.isfinite(std20[:-1])
        & np.isfinite(rate20[:-1])
        & (gap_next[:-1] <= 300.0)
    )
    pair_dates = ts.dt.date.to_numpy()[:-1]
    y_next = events[1:]
    h_pred = df_u["hawkes"].to_numpy()[:-1]
    b_std20 = std20[:-1]
    b_rate20 = rate20[:-1]

    is_h = valid_pair & (pair_dates >= IS_START) & (pair_dates <= IS_END)
    oos_h = valid_pair & (pair_dates >= OOS_START) & (pair_dates <= OOS_END)

    hawkes_res = {
        "is_pairs": int(is_h.sum()),
        "is_pos": int(y_next[is_h].sum()),
        "is_rate": round(float(y_next[is_h].mean()), 6),
        "oos_pairs": int(oos_h.sum()),
        "oos_pos": int(y_next[oos_h].sum()),
        "oos_rate": round(float(y_next[oos_h].mean()), 6),
        "oos_auc_unit": round(_safe_auc(y_next[oos_h], h_pred[oos_h]), 6),
        "oos_auc_std20": round(_safe_auc(y_next[oos_h], b_std20[oos_h]), 6),
        "oos_auc_rate20": round(_safe_auc(y_next[oos_h], b_rate20[oos_h]), 6),
    }
    hawkes_res["delta_auc_vs_std20"] = round(hawkes_res["oos_auc_unit"] - hawkes_res["oos_auc_std20"], 6)
    hawkes_res["delta_auc_vs_rate20"] = round(hawkes_res["oos_auc_unit"] - hawkes_res["oos_auc_rate20"], 6)

    # 5-quantile event rates
    q5_h = pd.qcut(h_pred[oos_h], 5, labels=False, duplicates="drop")
    hawkes_res["oos_quintile_event_rates"] = [
        round(float(y_next[oos_h][q5_h == q].mean()), 6) for q in sorted(np.unique(q5_h))
    ]

    # 3-tercile conditional evaluation (IS boundaries)
    t_bounds = np.quantile(b_std20[is_h], [1 / 3, 2 / 3])
    t_idx = np.digitize(b_std20[oos_h], t_bounds)
    hawkes_res["tercile_cond_auc_vs_std20"] = {}
    for q in [0, 1, 2]:
        m = t_idx == q
        if m.sum() >= 500 and y_next[oos_h][m].sum() >= 10:
            hawkes_res["tercile_cond_auc_vs_std20"][f"T{q+1}"] = round(_safe_auc(y_next[oos_h][m], h_pred[oos_h][m]), 6)
        else:
            hawkes_res["tercile_cond_auc_vs_std20"][f"T{q+1}"] = "N/A"

    # Bootstrap
    hawkes_res["boot_5d_vs_std20"] = _block_bootstrap_diff(
        pair_dates[oos_h], y_next[oos_h], h_pred[oos_h], b_std20[oos_h], _safe_auc, 5, 2000, 42
    )
    hawkes_res["boot_10d_vs_std20"] = _block_bootstrap_diff(
        pair_dates[oos_h], y_next[oos_h], h_pred[oos_h], b_std20[oos_h], _safe_auc, 10, 2000, 42
    )

    # --- Hamilton evaluation ---
    df_u["bar_date"] = ts.dt.date
    daily_rows = []
    all_days = sorted(set(df_u["bar_date"]))
    ret_series = pd.Series(ret)
    for d in all_days:
        m = (df_u["bar_date"] == d).to_numpy() & np.isfinite(ret)
        rv = float(np.sum(ret[m] ** 2))
        log_rv = float(np.log(rv)) if rv > 0 and m.sum() >= 280 else np.nan
        ham_vals = df_u.loc[df_u["bar_date"] == d, "ham"]
        ham_val = float(ham_vals.iloc[0]) if df_u.loc[df_u["bar_date"] == d, "ham_valid"].all() else np.nan
        daily_rows.append({"date": d, "log_rv": log_rv, "ham": ham_val})
    ddf = pd.DataFrame(daily_rows)
    ddf["lag1"] = ddf["log_rv"].shift(1)
    ddf["ma60"] = ddf["log_rv"].shift(1).rolling(60).mean()

    valid_d = ddf["ham"].notna() & ddf["log_rv"].notna() & ddf["lag1"].notna() & ddf["ma60"].notna()
    is_d = valid_d & (ddf["date"] >= IS_START) & (ddf["date"] <= IS_END)
    oos_d = valid_d & (ddf["date"] >= OOS_START) & (ddf["date"] <= OOS_END)

    y_d = ddf.loc[oos_d, "log_rv"].to_numpy()
    u_d = ddf.loc[oos_d, "ham"].to_numpy()
    l1_d = ddf.loc[oos_d, "lag1"].to_numpy()
    m60_d = ddf.loc[oos_d, "ma60"].to_numpy()
    dates_d = ddf.loc[oos_d, "date"].to_numpy()

    ham_res = {
        "is_days": int(is_d.sum()),
        "oos_days": int(oos_d.sum()),
        "oos_spearman_unit": round(_safe_spearman(y_d, u_d), 6),
        "oos_spearman_lag1": round(_safe_spearman(y_d, l1_d), 6),
        "oos_spearman_ma60": round(_safe_spearman(y_d, m60_d), 6),
    }
    ham_res["delta_spearman_vs_lag1"] = round(ham_res["oos_spearman_unit"] - ham_res["oos_spearman_lag1"], 6)
    ham_res["delta_spearman_vs_ma60"] = round(ham_res["oos_spearman_unit"] - ham_res["oos_spearman_ma60"], 6)

    q5_d = pd.qcut(u_d, 5, labels=False, duplicates="drop")
    ham_res["oos_quintile_mean_log_rv"] = [round(float(y_d[q5_d == q].mean()), 6) for q in sorted(np.unique(q5_d))]

    t_bounds_d = np.quantile(ddf.loc[is_d, "lag1"].to_numpy(), [1 / 3, 2 / 3])
    t_idx_d = np.digitize(l1_d, t_bounds_d)
    ham_res["tercile_cond_spearman_vs_lag1"] = {}
    for q in [0, 1, 2]:
        m = t_idx_d == q
        if m.sum() >= 15:
            val = _safe_spearman(y_d[m], u_d[m])
            ham_res["tercile_cond_spearman_vs_lag1"][f"T{q+1}"] = round(val, 6) if val is not None else "N/A"
        else:
            ham_res["tercile_cond_spearman_vs_lag1"][f"T{q+1}"] = "N/A"

    ham_res["boot_5d_vs_lag1"] = _block_bootstrap_diff(dates_d, y_d, u_d, l1_d, _safe_spearman, 5, 2000, 42)
    ham_res["boot_10d_vs_lag1"] = _block_bootstrap_diff(dates_d, y_d, u_d, l1_d, _safe_spearman, 10, 2000, 42)

    out = {
        "data_sha256": data_sha,
        "seed": 42,
        "timing_sec": {"prediction": round(pred_sec, 3), "total": round(time.perf_counter() - t0, 3)},
        "hawkes": hawkes_res,
        "hamilton": ham_res,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
