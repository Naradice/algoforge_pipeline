import json
with open('c:/Users/ksato/workspace/finance/research_pipeline/tests/fixtures/gold_standard/h1a_ground_truth.json', 'r', encoding='utf-8') as f:
    gt = json.load(f)

dev_docs = [d for d in gt if d.get('split') == 'development']
for d in dev_docs:
    print('='*50)
    print(d['source_id'], 'entities:', len(d.get('entities', [])), 'relations:', len(d.get('relations', [])))
    for e in d.get('entities', []):
        print(f"  [{e['entity_type']}] {e.get('name') or e.get('statement')}")
    for r in d.get('relations', []):
        print(f"  (rel) {r.get('src_local_id')} -> {r.get('relation_type')} -> {r.get('dst_local_id')}")
