import pytest
from pathlib import Path
from research_pipeline.storage.knowledge_graph import KnowledgeGraphStore
from research_pipeline.extractors.models import (
    CanonicalEntity,
    MentionProvenance,
    RelationEdge,
    RelationEvidence,
)


@pytest.fixture
def temp_kg_store(tmp_path: Path):
    db_file = tmp_path / "test_kg.sqlite"
    return KnowledgeGraphStore(db_file)


def test_insert_canonical_entity_and_provenance(temp_kg_store: KnowledgeGraphStore):
    store = temp_kg_store
    ent = CanonicalEntity(
        entity_id="ent_msgarch",
        canonical_name="Markov-Switching GARCH",
        entity_type="model",
        aliases=["MS-GARCH", "Markov switching GARCH"],
        provenances=[
            MentionProvenance(
                mention_id="men_01",
                source_id="arxiv_2302_01001",
                version_hash="a" * 64,
                verbatim_quote="Markov-switching GARCH",
                start_char=10,
                end_char=32,
            )
        ]
    )
    store.insert_entity(ent)
    fetched = store.get_entity("ent_msgarch")
    assert fetched is not None
    assert fetched["canonical_name"] == "Markov-Switching GARCH"
    assert len(fetched["aliases"]) == 2
    assert len(fetched["provenances"]) == 1
    assert fetched["provenances"][0]["verbatim_quote"] == "Markov-switching GARCH"


def test_multi_evidence_relation_edge(temp_kg_store: KnowledgeGraphStore):
    store = temp_kg_store
    # Insert two nodes
    ent1 = CanonicalEntity(
        entity_id="mod_hawkes",
        canonical_name="Hawkes Process",
        entity_type="model",
    )
    ent2 = CanonicalEntity(
        entity_id="met_mle",
        canonical_name="Maximum Likelihood Estimation",
        entity_type="method",
    )
    store.insert_entity(ent1)
    store.insert_entity(ent2)

    # Insert edge supported by two different papers (Multi-Evidence)
    edge = RelationEdge(
        edge_id="met_mle->APPLIES_TO->mod_hawkes",
        src_entity_id="met_mle",
        dst_entity_id="mod_hawkes",
        relation_type="APPLIES_TO",
        confidence_score=0.95,
        evidences=[
            RelationEvidence(
                evidence_id="ev_doc1",
                edge_id="met_mle->APPLIES_TO->mod_hawkes",
                source_id="arxiv_2301_01002",
                version_hash="b" * 64,
                verbatim_quote="Maximum likelihood estimation of multivariate Hawkes processes",
                start_char=20,
                end_char=82,
                reasoning="MLE is used for parameter estimation of Hawkes process."
            ),
            RelationEvidence(
                evidence_id="ev_doc2",
                edge_id="met_mle->APPLIES_TO->mod_hawkes",
                source_id="arxiv_synthetic_02",
                version_hash="c" * 64,
                verbatim_quote="estimating Hawkes process parameters via MLE",
                start_char=15,
                end_char=59,
                reasoning="Second paper also applies MLE to Hawkes."
            ),
        ]
    )
    store.insert_relation_edge(edge)

    fetched_edge = store.get_edge("met_mle->APPLIES_TO->mod_hawkes")
    assert fetched_edge is not None
    assert fetched_edge["relation_type"] == "APPLIES_TO"
    assert len(fetched_edge["evidences"]) == 2
    assert fetched_edge["evidences"][0]["source_id"] == "arxiv_2301_01002"
    assert fetched_edge["evidences"][1]["source_id"] == "arxiv_synthetic_02"


def test_integrity_constraint_orphan_and_self_loop(temp_kg_store: KnowledgeGraphStore):
    store = temp_kg_store
    ent = CanonicalEntity(
        entity_id="node_a",
        canonical_name="Node A",
        entity_type="model",
    )
    store.insert_entity(ent)

    # 1. Self loop rejection
    with pytest.raises(ValueError, match="Self-loops are forbidden"):
        store.insert_relation_edge(
            RelationEdge(
                edge_id="node_a->EXPLAINS->node_a",
                src_entity_id="node_a",
                dst_entity_id="node_a",
                relation_type="EXPLAINS",
            )
        )

    # 2. Orphan edge rejection (dst nonexistent)
    with pytest.raises(ValueError, match="Orphan edge rejected"):
        store.insert_relation_edge(
            RelationEdge(
                edge_id="node_a->EXPLAINS->nonexistent",
                src_entity_id="node_a",
                dst_entity_id="nonexistent",
                relation_type="EXPLAINS",
            )
        )
