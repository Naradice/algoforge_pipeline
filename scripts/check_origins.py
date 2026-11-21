import sqlite3

conn = sqlite3.connect("c:/Users/ksato/workspace/finance/research_pipeline/data/knowledge_graph_expansion.db")
cur = conn.cursor()

cur.execute("SELECT origin_type, count(*) FROM canonical_entities GROUP BY origin_type")
print("Entity origins:", cur.fetchall())

cur.execute("SELECT origin_type, count(*) FROM relation_edges GROUP BY origin_type")
print("Edge origins:", cur.fetchall())

cur.execute("SELECT canonical_name, origin_type, derivation_rationale FROM canonical_entities WHERE origin_type = 'inferred'")
for row in cur.fetchall():
    print("  Inferred Node:", row)
