import sys
sys.path.insert(0, r'c:\Users\ksato\workspace\finance\research_pipeline')
import json
from scratch.audit_all_144_v06 import ledger

cond_sums = {
    'S0': {'M': 0, 'D': 0, 'S': 0, 'V': 0},
    'S1': {'M': 0, 'D': 0, 'S': 0, 'V': 0},
    'S2a': {'M': 0, 'D': 0, 'S': 0, 'V': 0},
    'S2b': {'M': 0, 'D': 0, 'S': 0, 'V': 0},
}

concept_cond_sums = {
    'hawkes_process': {c: {'M': 0, 'D': 0, 'S': 0, 'V': 0} for c in ['S0', 'S1', 'S2a', 'S2b']},
    'jump_diffusion': {c: {'M': 0, 'D': 0, 'S': 0, 'V': 0} for c in ['S0', 'S1', 'S2a', 'S2b']},
}

for sid, params in ledger.items():
    cond = sid.split('_')[-1]
    cid = 'hawkes_process' if 'hawkes' in sid else 'jump_diffusion'
    for pk, pv in params.items():
        for tier in ['M', 'D', 'S', 'V']:
            cond_sums[cond][tier] += pv[tier]
            concept_cond_sums[cid][cond][tier] += pv[tier]

print("Condition sums:")
for cond, s in cond_sums.items():
    print(f"  {cond}: M={s['M']}/36, D={s['D']}/36, S={s['S']}/36, V={s['V']}/36")

print("\nConcept sums:")
for cid, c_data in concept_cond_sums.items():
    print(f"=== {cid} ===")
    for cond, s in c_data.items():
        print(f"  {cond}: M={s['M']}/18, D={s['D']}/18, S={s['S']}/18, V={s['V']}/18")
