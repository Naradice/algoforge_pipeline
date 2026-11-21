"""Provenance-Preserving Knowledge Graph Storage (H1b-KG).

Implements Section 6 of H1b v0.2:
- Relational SQLite schema with Foreign Key enforcement
- Separates Canonical Relation Edges from Multi-Document Evidences
- Supports SAME_AS canonical entity integration and VARIANT_OF hierarchical edges
- Cryptographic provenance tracking back to H0 SHA-256 version hashes
- Anti-orphan and anti-self-loop integrity checks
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from pydantic import BaseModel

from research_pipeline.extractors.models import (
    CanonicalEntity,
    MentionProvenance,
    RelationEdge,
    RelationEvidence,
)


class KnowledgeGraphStore:
    """Relational SQLite storage for H1b Knowledge Graph with strict integrity guarantees."""

    def __init__(self, db_path: Union[Path, str]):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self):
        """Create normalized relational tables."""
        with self._get_connection() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS canonical_entities (
                entity_id TEXT PRIMARY KEY,
                canonical_name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                origin_type TEXT NOT NULL DEFAULT 'extracted',
                derivation_rationale TEXT,
                aliases TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS mention_provenance (
                mention_id TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                version_hash TEXT NOT NULL,
                verbatim_quote TEXT NOT NULL,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL,
                target_field_name TEXT NOT NULL DEFAULT 'abstract_text',
                FOREIGN KEY (entity_id) REFERENCES canonical_entities(entity_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS relation_edges (
                edge_id TEXT PRIMARY KEY,
                src_entity_id TEXT NOT NULL,
                dst_entity_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                origin_type TEXT NOT NULL DEFAULT 'text_supported',
                confidence_score REAL NOT NULL DEFAULT 1.0,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                CHECK (src_entity_id != dst_entity_id),
                FOREIGN KEY (src_entity_id) REFERENCES canonical_entities(entity_id) ON DELETE RESTRICT,
                FOREIGN KEY (dst_entity_id) REFERENCES canonical_entities(entity_id) ON DELETE RESTRICT
            );

            CREATE TABLE IF NOT EXISTS relation_evidences (
                evidence_id TEXT PRIMARY KEY,
                edge_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                version_hash TEXT NOT NULL,
                verbatim_quote TEXT NOT NULL,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL,
                reasoning TEXT,
                FOREIGN KEY (edge_id) REFERENCES relation_edges(edge_id) ON DELETE CASCADE
            );
            """)

    def insert_entity(self, entity: CanonicalEntity):
        """Insert a canonical entity and its associated mention provenances."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO canonical_entities (entity_id, canonical_name, entity_type, origin_type, derivation_rationale, aliases)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_id) DO UPDATE SET
                    canonical_name = excluded.canonical_name,
                    origin_type = excluded.origin_type,
                    derivation_rationale = excluded.derivation_rationale,
                    aliases = excluded.aliases
                """,
                (
                    entity.entity_id,
                    entity.canonical_name,
                    entity.entity_type,
                    entity.origin_type,
                    entity.derivation_rationale,
                    json.dumps(entity.aliases, ensure_ascii=False),
                )
            )
            for m in entity.provenances:
                conn.execute(
                    """
                    INSERT INTO mention_provenance 
                    (mention_id, entity_id, source_id, version_hash, verbatim_quote, start_char, end_char, target_field_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(mention_id) DO UPDATE SET
                        verbatim_quote = excluded.verbatim_quote,
                        start_char = excluded.start_char,
                        end_char = excluded.end_char
                    """,
                    (
                        m.mention_id,
                        entity.entity_id,
                        m.source_id,
                        m.version_hash,
                        m.verbatim_quote,
                        m.start_char,
                        m.end_char,
                        m.target_field_name,
                    )
                )

    def insert_relation_edge(self, edge: RelationEdge):
        """Insert a canonical relation edge and its supporting evidences."""
        if edge.src_entity_id == edge.dst_entity_id:
            raise ValueError(f"Self-loops are forbidden in KG: {edge.src_entity_id} -> {edge.dst_entity_id}")

        with self._get_connection() as conn:
            # Check source and destination exist
            cur = conn.cursor()
            cur.execute("SELECT entity_id FROM canonical_entities WHERE entity_id IN (?, ?)", (edge.src_entity_id, edge.dst_entity_id))
            found_ids = {row[0] for row in cur.fetchall()}
            if edge.src_entity_id not in found_ids:
                raise ValueError(f"Orphan edge rejected: source entity '{edge.src_entity_id}' does not exist.")
            if edge.dst_entity_id not in found_ids:
                raise ValueError(f"Orphan edge rejected: destination entity '{edge.dst_entity_id}' does not exist.")

            conn.execute(
                """
                INSERT INTO relation_edges (edge_id, src_entity_id, dst_entity_id, relation_type, origin_type, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(edge_id) DO UPDATE SET
                    origin_type = excluded.origin_type,
                    confidence_score = excluded.confidence_score
                """,
                (
                    edge.edge_id,
                    edge.src_entity_id,
                    edge.dst_entity_id,
                    edge.relation_type,
                    edge.origin_type,
                    edge.confidence_score,
                )
            )
            for ev in edge.evidences:
                conn.execute(
                    """
                    INSERT INTO relation_evidences
                    (evidence_id, edge_id, source_id, version_hash, verbatim_quote, start_char, end_char, reasoning)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(evidence_id) DO UPDATE SET
                        verbatim_quote = excluded.verbatim_quote,
                        reasoning = excluded.reasoning
                    """,
                    (
                        ev.evidence_id,
                        edge.edge_id,
                        ev.source_id,
                        ev.version_hash,
                        ev.verbatim_quote,
                        ev.start_char,
                        ev.end_char,
                        ev.reasoning,
                    )
                )

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM canonical_entities WHERE entity_id = ?", (entity_id,))
            row = cur.fetchone()
            if not row:
                return None
            ent = dict(row)
            ent["aliases"] = json.loads(ent["aliases"])
            cur.execute("SELECT * FROM mention_provenance WHERE entity_id = ?", (entity_id,))
            ent["provenances"] = [dict(r) for r in cur.fetchall()]
            return ent

    def get_edge(self, edge_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM relation_edges WHERE edge_id = ?", (edge_id,))
            row = cur.fetchone()
            if not row:
                return None
            edge = dict(row)
            cur.execute("SELECT * FROM relation_evidences WHERE edge_id = ?", (edge_id,))
            edge["evidences"] = [dict(r) for r in cur.fetchall()]
            return edge

    def get_summary_statistics(self) -> Dict[str, int]:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM canonical_entities")
            num_entities = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM mention_provenance")
            num_mentions = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM relation_edges")
            num_edges = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM relation_evidences")
            num_evidences = cur.fetchone()[0]
            return {
                "num_canonical_entities": num_entities,
                "num_mention_provenances": num_mentions,
                "num_relation_edges": num_edges,
                "num_relation_evidences": num_evidences,
            }
