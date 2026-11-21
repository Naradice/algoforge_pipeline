import numpy as np
import sys
sys.path.insert(0, r'c:/Users/ksato/workspace/finance/research_pipeline/src')
sys.path.insert(0, r'c:/Users/ksato/workspace/finance/research_pipeline/scripts')
from run_h3_p1_experiments import FastHamiltonRegimeModel

d_mu = 2.0
W = 60
mu_0_true = -9.0
mu_1_true = -9.0 + d_mu
sigma_true = [0.5, 0.5]
P_true = np.array([[0.95, 0.05], [0.15, 0.85]])

seed_stats = []
pool_cm = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'FLAT_0': 0, 'FLAT_1': 0}

for s_idx in range(15):
    seed = 3000 + s_idx
    np.random.seed(seed)
    S_true = np.zeros(150, dtype=np.int32)
    y = np.zeros(150, dtype=np.float64)
    curr_state = 0
    for d in range(150):
        if d > 0: curr_state = np.random.binomial(1, P_true[curr_state, 1])
        S_true[d] = curr_state
        mu_s = mu_0_true if curr_state == 0 else mu_1_true
        y[d] = np.random.normal(mu_s, sigma_true[curr_state])
        
    s_hat_op = np.full(150, -1, dtype=np.int32)
    model = FastHamiltonRegimeModel(window_days=W, max_iter=50)
    for d in range(10, 150):
        w_start = max(0, d - W)
        w_y = y[w_start:d]
        model.mu = np.array([np.percentile(w_y, 25), np.percentile(w_y, 75)])
        std_y = max(np.std(w_y, ddof=1), 1e-3)
        model.sigma = np.array([std_y / 1.414, std_y / 1.414])
        model.P = np.array([[0.90, 0.10], [0.20, 0.80]])
        if model.fit_em(w_y):
            _, xi_filt, _ = model.forward_filter(w_y)
            s_hat_op[d] = 1 if xi_filt[-1][1] > 0.50 else 0
        else:
            s_hat_op[d] = -1
            
    s_prev_true = np.zeros(150, dtype=np.int32)
    s_prev_true[1:150] = S_true[0:149]
    
    # Eval days 10..149
    eval_true = s_prev_true[10:150]
    eval_pred = s_hat_op[10:150]
    
    tp = int(np.sum((eval_true == 1) & (eval_pred == 1)))
    tn = int(np.sum((eval_true == 0) & (eval_pred == 0)))
    fp = int(np.sum((eval_true == 0) & (eval_pred == 1)))
    fn = int(np.sum((eval_true == 1) & (eval_pred == 0)))
    flat_0 = int(np.sum((eval_true == 0) & (eval_pred == -1)))
    flat_1 = int(np.sum((eval_true == 1) & (eval_pred == -1)))
    
    pool_cm['TP'] += tp
    pool_cm['TN'] += tn
    pool_cm['FP'] += fp
    pool_cm['FN'] += fn
    pool_cm['FLAT_0'] += flat_0
    pool_cm['FLAT_1'] += flat_1
    
    decided = (eval_pred >= 0)
    seed_cov = float(np.sum(decided) / 140.0)
    rec0 = tn / max(tn + fp, 1)
    rec1 = tp / max(tp + fn, 1)
    seed_cba = float((rec0 + rec1) / 2.0)
    seed_oa_140 = float((tp + tn) / 140.0)
    seed_oa_dec = float((tp + tn) / max(np.sum(decided), 1))
    seed_stats.append({
        'seed': seed, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'flat_0': flat_0, 'flat_1': flat_1,
        'cov': seed_cov, 'cba': seed_cba, 'oa_140': seed_oa_140, 'oa_dec': seed_oa_dec
    })

print('=== POOLED CONFUSION MATRIX (15 seeds x 140 days = 2100 days) ===')
for k, v in pool_cm.items():
    print(f'  {k}: {v}')
total_days = sum(pool_cm.values())
dec_days = pool_cm['TP'] + pool_cm['TN'] + pool_cm['FP'] + pool_cm['FN']
correct_days = pool_cm['TP'] + pool_cm['TN']
print(f'Total days: {total_days}, Decided days: {dec_days}, Correct days: {correct_days}')
print(f'Pooled OA (total 2100 days): {correct_days / total_days * 100:.4f}%')
print(f'Pooled OA (decided days): {correct_days / dec_days * 100:.4f}%')
print(f'Macro-average OA (140 days): {np.mean([s["oa_140"] for s in seed_stats])*100:.4f}%')
print(f'Macro-average OA (decided days): {np.mean([s["oa_dec"] for s in seed_stats])*100:.4f}%')
print(f'Macro-average CBA: {np.mean([s["cba"] for s in seed_stats])*100:.4f}%')
print(f'Macro-average Coverage: {np.mean([s["cov"] for s in seed_stats])*100:.4f}%')
