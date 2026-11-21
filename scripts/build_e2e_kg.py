import json
import sqlite3
from pathlib import Path
from research_pipeline.storage.knowledge_graph import KnowledgeGraphStore
from research_pipeline.extractors.models import (
    CanonicalEntity,
    MentionProvenance,
    RelationEdge,
    RelationEvidence,
)
from research_pipeline.evaluation.matching import normalize_entity_name, are_semantically_equivalent

base_dir = Path("c:/Users/ksato/workspace/finance/research_pipeline")
e2e_db_path = base_dir / "data/knowledge_graph_e2e.db"
if e2e_db_path.exists():
    e2e_db_path.unlink()

store = KnowledgeGraphStore(e2e_db_path)

# Load C4 Trial 1 bundles
bundles_dir = base_dir / "data/experiments/phase3_holdout_expansion"
t1_files = sorted(bundles_dir.glob("bundle_phase3_exp_C4_t1_*.json"))

print(f"Building E2E KG from {len(t1_files)} C4 bundles...")

entity_map = {}
for bf in t1_files:
    with open(bf, "r", encoding="utf-8") as f:
        bundle = json.load(f)
    
    sid = bundle["source_id"]
    ver_hash = list(bundle["raw_artifact"]["target_field_hashes"].values())[0] if bundle["raw_artifact"].get("target_field_hashes") else "0"*64
    
    for cand in bundle.get("evaluation_candidates", []):
        ename = cand["name_or_statement"]
        etype = cand["entity_type"]
        cid = f"e2e_{etype}_{normalize_entity_name(ename).replace(' ', '_')}"
        entity_map[(sid, cand["candidate_local_id"])] = cid
        
        ev = cand.get("evidence") or {}
        quote = ev.get("verbatim_quote", ename)
        
        prov = MentionProvenance(
            mention_id=f"prov_{cand['candidate_local_id']}_{sid}",
            entity_id=cid,
            source_id=sid,
            version_hash=ver_hash,
            verbatim_quote=quote,
            start_char=ev.get("start_char", 0),
            end_char=ev.get("end_char") or max(1, len(quote)),
            target_field_name=ev.get("target_field_name", "abstract_text")
        )
        
        canon = CanonicalEntity(
            entity_id=cid,
            canonical_name=ename,
            entity_type=etype,
            origin_type="extracted",
            aliases=[],
            provenances=[prov]
        )
        store.insert_entity(canon)

# Load E2E relation details from phase3_holdout_results.json
results_path = bundles_dir / "phase3_holdout_results.json"
with open(results_path, "r", encoding="utf-8") as f:
    res = json.load(f)

e2e_details = res["h1b_relation_extraction"]["end_to_end"]["details"]
e2e_edges_added = 0

for det in e2e_details:
    if det["endpoints_found"] and det["hyb_pred"] != "NONE":
        sid = det["source_id"]
        # Find local IDs
        src_cid = f"e2e_{normalize_entity_name(det['gold_src']).replace(' ', '_')}"
        # Match against entity_map using semantic equivalence
        matched_src = [
            cid for (s, lid), cid in entity_map.items()
            if s == sid and are_semantically_equivalent(det["gold_src"], [det["gold_src"]], cid.split("_", 2)[-1].replace("_", " "))
        ]
        matched_dst = [
            cid for (s, lid), cid in entity_map.items()
            if s == sid and are_semantically_equivalent(det["gold_dst"], [det["gold_dst"]], cid.split("_", 2)[-1].replace("_", " "))
        ]
        
        if matched_src and matched_dst:
            s_id = matched_src[0]
            d_id = matched_dst[0]
            rtype = det["hyb_pred"]
            edge_id = f"e2e_edge_{s_id}_{rtype}_{d_id}"
            
            rel_ev = RelationEvidence(
                evidence_id=f"ev_{edge_id}_{sid}",
                edge_id=edge_id,
                source_id=sid,
                version_hash="0"*64,
                verbatim_quote=f"E2E relation extracted between {det['gold_src']} and {det['gold_dst']}",
                start_char=0,
                end_char=max(1, len(det['gold_src'])),
                reasoning="E2E relation extracted under RE-Hybrid agreement on C4 candidates."
            )
            
            edge = RelationEdge(
                edge_id=edge_id,
                src_entity_id=s_id,
                dst_entity_id=d_id,
                relation_type=rtype,
                origin_type="text_supported",
                confidence_score=0.90,
                evidences=[rel_ev]
            )
            store.insert_relation_edge(edge)
            e2e_edges_added += 1

stats = store.get_summary_statistics()
print(f"E2E Knowledge Graph created: {e2e_db_path}")
print(f"E2E Stats: {stats}")
