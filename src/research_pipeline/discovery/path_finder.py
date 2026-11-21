"""Multi-hop relational path finder over KnowledgeGraphStore for Step H2."""

from typing import List, Dict, Any, Optional, Set, Tuple
from collections import deque
import sqlite3

from research_pipeline.storage.knowledge_graph import KnowledgeGraphStore
from research_pipeline.discovery.models import GraphPath, PathHop


class MultiHopPathFinder:
    """Traverses the relational knowledge graph to discover cross-domain analogy paths."""

    def __init__(self, kg_store: KnowledgeGraphStore):
        self.kg_store = kg_store

    def find_paths(
        self,
        start_entity_type: Optional[str] = None,
        target_entity_type: Optional[str] = None,
        min_hops: int = 1,
        max_hops: int = 3,
        include_variants: bool = True,
    ) -> List[GraphPath]:
        """Find all acyclic multi-hop paths matching entity type constraints."""
        with self.kg_store._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            # Load all canonical entities
            cur.execute("SELECT entity_id, canonical_name, entity_type, origin_type FROM canonical_entities")
            entities = {row["entity_id"]: dict(row) for row in cur.fetchall()}

            # Load all edges
            edge_query = "SELECT edge_id, src_entity_id, dst_entity_id, relation_type, origin_type, confidence_score FROM relation_edges"
            if not include_variants:
                edge_query += " WHERE relation_type != 'VARIANT_OF'"
            cur.execute(edge_query)
            edges = [dict(row) for row in cur.fetchall()]

            # Load evidences for edges
            cur.execute("SELECT edge_id, verbatim_quote FROM relation_evidences")
            evidences = {}
            for row in cur.fetchall():
                evidences.setdefault(row["edge_id"], []).append(row["verbatim_quote"])

        # Build adjacency list: src -> list of (edge, dst)
        adj: Dict[str, List[Tuple[Dict[str, Any], str]]] = {}
        for edge in edges:
            adj.setdefault(edge["src_entity_id"], []).append((edge, edge["dst_entity_id"]))
            # Also allow traversing VARIANT_OF in both directions for analogy
            if edge["relation_type"] == "VARIANT_OF":
                # Inverse hop: dst has variant src
                inv_edge = dict(edge)
                inv_edge["relation_type"] = "HAS_VARIANT"
                adj.setdefault(edge["dst_entity_id"], []).append((inv_edge, edge["src_entity_id"]))

        # Identify candidate start nodes
        start_nodes = [
            eid for eid, ent in entities.items()
            if start_entity_type is None or ent["entity_type"].lower() == start_entity_type.lower()
        ]

        discovered_paths: List[GraphPath] = []
        path_counter = 1

        for start_id in start_nodes:
            # BFS queue: (current_node, visited_nodes, current_hops)
            queue = deque([(start_id, [start_id], [])])

            while queue:
                curr_node, visited, hops = queue.popleft()
                curr_len = len(hops)

                # Check if current path meets criteria
                if curr_len >= min_hops:
                    target_ent = entities.get(curr_node)
                    if target_ent:
                        matches_target = (
                            target_entity_type is None or
                            target_ent["entity_type"].lower() == target_entity_type.lower()
                        )
                        if matches_target and (start_id != curr_node):
                            src_ent = entities[start_id]
                            summary = f"[{src_ent['canonical_name']} ({src_ent['entity_type']})] " + " ".join(
                                f"--({h.relation_type})--> [{h.dst_name} ({h.dst_type})]" for h in hops
                            )
                            discovered_paths.append(
                                GraphPath(
                                    path_id=f"path_{path_counter:03d}",
                                    hops=hops,
                                    hop_count=curr_len,
                                    source_concept=src_ent["canonical_name"],
                                    target_concept=target_ent["canonical_name"],
                                    path_summary=summary,
                                )
                            )
                            path_counter += 1

                if curr_len >= max_hops:
                    continue

                for edge, nxt_node in adj.get(curr_node, []):
                    if nxt_node not in visited and nxt_node in entities:
                        s_ent = entities[curr_node]
                        d_ent = entities[nxt_node]
                        ev_list = evidences.get(edge.get("edge_id", ""), [])
                        quote = ev_list[0] if ev_list else None

                        new_hop = PathHop(
                            hop_index=curr_len + 1,
                            src_entity_id=curr_node,
                            src_name=s_ent["canonical_name"],
                            src_type=s_ent["entity_type"],
                            relation_type=edge["relation_type"],
                            dst_entity_id=nxt_node,
                            dst_name=d_ent["canonical_name"],
                            dst_type=d_ent["entity_type"],
                            origin_type=edge.get("origin_type", "text_supported"),
                            evidence_quote=quote,
                        )
                        queue.append((nxt_node, visited + [nxt_node], hops + [new_hop]))

        return discovered_paths

    def find_cross_domain_bridges(self) -> List[GraphPath]:
        """Find meaningful paths starting from tools/methods and ending at market phenomena."""
        paths = self.find_paths(
            start_entity_type="software",
            target_entity_type="phenomenon",
            min_hops=1,
            max_hops=3,
        )
        if not paths:
            # Fallback to method -> phenomenon
            paths = self.find_paths(
                start_entity_type="method",
                target_entity_type="phenomenon",
                min_hops=1,
                max_hops=3,
            )
        return paths
