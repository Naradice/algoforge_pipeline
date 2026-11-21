"""Audit and review the 10 Development Set documents and their gold labels."""
import json
from pathlib import Path

gt_path = Path("c:/Users/ksato/workspace/finance/research_pipeline/tests/fixtures/gold_standard/h1a_ground_truth.json")
with open(gt_path, "r", encoding="utf-8") as f:
    data = json.load(f)

dev_docs = [d for d in data if d.get("split") == "development"]

print(f"Auditing {len(dev_docs)} Development Set Documents...")
all_quotes_valid = True
entity_breakdown = {}

for i, doc in enumerate(dev_docs, 1):
    sid = doc["source_id"]
    title = doc["target_fields"]["title_text"]
    stype = doc["source_type"]
    entities = doc.get("entities", [])
    relations = doc.get("relations", [])
    
    print(f"\n--- [{i}/10] {sid} ({stype}): {title} ---")
    
    for ent in entities:
        etype = ent.get("entity_type")
        entity_breakdown[etype] = entity_breakdown.get(etype, 0) + 1
        name = ent.get("name") or ent.get("statement")
        lid = ent.get("local_id")
        ev = ent.get("evidence", {})
        tf_name = ev.get("target_field_name")
        target_text = doc["target_fields"].get(tf_name, "")
        s = ev.get("start_char")
        e = ev.get("end_char")
        quote = ev.get("verbatim_quote")
        
        extracted_slice = target_text[s:e]
        is_exact = (extracted_slice == quote)
        if not is_exact:
            all_quotes_valid = False
            print(f"  [ERROR] {lid} ({etype}) quote mismatch!")
            print(f"     Expected: '{quote}'")
            print(f"     Actual:   '{extracted_slice}'")
        else:
            print(f"  [OK] {lid} ({etype}): '{name}'")
            print(f"       Quote: '{quote}' ({tf_name}[{s}:{e}])")
            
    for rel in relations:
        rid = rel.get("relation_id")
        rtype = rel.get("relation_type")
        src = rel.get("src_local_id")
        dst = rel.get("dst_local_id")
        ev = rel.get("evidence", {})
        tf_name = ev.get("target_field_name")
        target_text = doc["target_fields"].get(tf_name, "")
        s = ev.get("start_char")
        e = ev.get("end_char")
        quote = ev.get("verbatim_quote")
        extracted_slice = target_text[s:e]
        is_exact = (extracted_slice == quote)
        if not is_exact:
            all_quotes_valid = False
            print(f"  [ERROR Rel] {rid} quote mismatch!")
        else:
            print(f"  [OK Rel] {rid}: {src} --[{rtype}]--> {dst}")

print("\n" + "="*50)
print(f"All quotes strictly exact: {all_quotes_valid}")
print("Entity breakdown in Dev Set:", entity_breakdown)
print("Total entities:", sum(entity_breakdown.values()))
