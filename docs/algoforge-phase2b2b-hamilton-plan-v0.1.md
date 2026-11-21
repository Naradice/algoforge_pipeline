# AlgoForge Phase 2B-2b: Hamilton Unit Audit & Implementation Plan v0.1

## 1. Read-Only Audit of `HamiltonLogRVRegimeModel` (`econometric_modules.py`)

- **Prediction (`predict_next_day_state_prob`, lines 722-731)**:
  - Calls `_, xi_filt, _ = self.forward_filter(y_history)`.
  - Takes `xi_d_d = xi_filt[-1]` (filtered state probability at day $d$ close, 00:00 UTC).
  - Computes `xi_d_plus_1 = self.P.T @ xi_d_d`, clips to `[1e-12, 1 - 1e-12]`, and normalizes.
  - This confirms `self.P` is row-stochastic (`P[i, j] = P(S_{d+1} = j | S_d = i)`), so `P.T @ xi_d_d` produces the one-day-ahead prior $\xi_{d+1 \mid d}$.
- **Daily Update & Fallback (`step_daily_update_with_fallback`, lines 733-775)**:
  - Executes at day $d$ (00:00 UTC) on a 60-day `y_window`.
  - Runs `self.fit_em(y_window)`.
  - On failure at `day_idx == 0`: falls back to quantile initial guess (25th/75th percentiles, `P = [[0.90, 0.10], [0.20, 0.80]]`).
  - On failure at `day_idx >= 1`: carries over `prev_params` from day $d-1$.
  - Returns `pred_prob_next_day = self.predict_next_day_state_prob(y_window)`.

## 2. Phase 2B-2b Plan (Pending Approval)

1. **Target Unit**: `unit:hamilton_pred_prob` (continuous/probability output in `[0, 1]`, `warmup_bars` corresponding to 60 daily RV observations).
2. **Daily-to-M5 Alignment**: Compute daily log-RV strictly from completed UTC days, run `step_daily_update_with_fallback` at 00:00 UTC, and broadcast $\xi_{d+1 \mid d}$ to M5 bars of day $d+1$ with `available_at = day_{d+1} 00:00 UTC <= decision_time`.
3. **Contract Tests**: Future-perturbation invariance across EM refits, fallback hierarchy verification, and `algoforge_unit_v1` cache separation.
