import json
from pathlib import Path

gt_path = Path('c:/Users/ksato/workspace/finance/research_pipeline/tests/fixtures/gold_standard/h1a_ground_truth.json')
with open(gt_path, 'r', encoding='utf-8') as f:
    gt = json.load(f)

for d in gt:
    if d['source_id'] == 'github_10101':
        for e in d['entities']:
            if e['local_id'] == 'mod_01':
                e['local_id'] = 'soft_01'
                e['entity_type'] = 'software'
                e['category'] = 'simulation_environment'
        for r in d['relations']:
            if r['dst_local_id'] == 'mod_01':
                r['dst_local_id'] = 'soft_01'
    elif d['source_id'] == 'github_10103':
        for e in d['entities']:
            if e['local_id'] == 'mod_01':
                e['local_id'] = 'soft_01'
                e['entity_type'] = 'software'
                e['category'] = 'backtesting_framework'
        for r in d['relations']:
            if r['dst_local_id'] == 'mod_01':
                r['dst_local_id'] = 'soft_01'

with open(gt_path, 'w', encoding='utf-8') as f:
    json.dump(gt, f, indent=2, ensure_ascii=False)

print("Updated h1a_ground_truth.json with 7-class ontology for Development OSS!")
