import json

with open("c:/Users/ksato/workspace/finance/research_pipeline/tests/fixtures/h2_d2_benchmark_cases.json", encoding="utf-8") as f:
    cases = json.load(f)

print(f"Total synthetic cases: {len(cases)}")
for c in cases:
    print(c["case_id"], c["concept_id"], c["perturbation_type"], c["ground_truth_errors"])
