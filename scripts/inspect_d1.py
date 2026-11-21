import json

with open("c:/Users/ksato/workspace/finance/research_pipeline/data/h2_d1_diagnostic_results.json", encoding="utf-8") as f:
    d = json.load(f)

for item in d["results"]:
    print(f"=== {item.get('concept_id')} | {item.get('condition_id')} ===")
    cand = item.get('candidate', {})
    spec = cand.get('backtest_spec') or {}
    print("Claim:", cand.get('claim', ''))
    print("Mechanism:", cand.get('proposed_mechanism', ''))
    print("Rationale:", cand.get('grounding_rationale', ''))
    print("Entry:", spec.get('entry_condition', ''))
    print("Exit:", spec.get('exit_condition', ''))
    print("Feat:", spec.get('feature_engineering', ''))
    print("Flags:", item.get('evaluation', {}).get('error_flags'))
    print("Evidence:", item.get('evaluation', {}).get('evidence'))
    print("-" * 60)
