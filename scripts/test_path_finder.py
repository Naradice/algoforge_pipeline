from pathlib import Path
from research_pipeline.storage.knowledge_graph import KnowledgeGraphStore
from research_pipeline.discovery.path_finder import MultiHopPathFinder

db_path = Path("c:/Users/ksato/workspace/finance/research_pipeline/data/knowledge_graph_expansion.db")
store = KnowledgeGraphStore(db_path)
finder = MultiHopPathFinder(store)

print("Finding all 1-hop paths...")
p1 = finder.find_paths(min_hops=1, max_hops=1)
print(f"Total 1-hop paths: {len(p1)}")
for p in p1[:3]:
    print(" ", p.path_summary)

print("\nFinding all 2-hop paths...")
p2 = finder.find_paths(min_hops=2, max_hops=2)
print(f"Total 2-hop paths: {len(p2)}")
for p in p2[:3]:
    print(" ", p.path_summary)

print("\nFinding all 3-hop paths...")
p3 = finder.find_paths(min_hops=3, max_hops=3)
print(f"Total 3-hop paths: {len(p3)}")
for p in p3[:3]:
    print(" ", p.path_summary)

print("\nFinding cross-domain bridges (software/method -> phenomenon)...")
bridges = finder.find_cross_domain_bridges()
print(f"Total bridge paths found: {len(bridges)}")
for b in bridges[:5]:
    print("  [Bridge]:", b.path_summary)
