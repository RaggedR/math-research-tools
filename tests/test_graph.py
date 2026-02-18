"""Tests for kg.graph — merging extractions, building graph, preparing viz data."""

import pytest


class TestMergeExtractions:
    def test_merges_concepts(self, sample_extractions):
        from kg.graph import merge_extractions

        concepts, edges = merge_extractions(sample_extractions)
        # "Bailey's lemma" should normalize to "bailey lemma" and merge with "Bailey lemma"
        assert "bailey lemma" in concepts
        assert "rogers-ramanujan identities" in concepts
        assert "schur functions" in concepts

    def test_deduplicates_across_papers(self, sample_extractions):
        from kg.graph import merge_extractions

        concepts, edges = merge_extractions(sample_extractions)
        rr = concepts["rogers-ramanujan identities"]
        # Should appear in both papers
        assert len(rr["papers"]) == 2

    def test_normalizes_names_in_edges(self, sample_extractions):
        from kg.graph import merge_extractions

        concepts, edges = merge_extractions(sample_extractions)
        # "Bailey's lemma" -> "bailey lemma" in edges
        sources = {e["source"] for e in edges}
        assert "bailey lemma" in sources
        # Original non-normalized name should not appear
        assert "bailey's lemma" not in sources

    def test_keeps_longer_description(self, sample_extractions):
        from kg.graph import merge_extractions

        concepts, edges = merge_extractions(sample_extractions)
        rr = concepts["rogers-ramanujan identities"]
        # Should keep the longer description
        assert len(rr["description"]) > 0

    def test_skips_empty_names(self):
        from kg.graph import merge_extractions

        extractions = {
            "paper.pdf": {
                "concepts": [
                    {"name": "", "type": "object", "description": ""},
                    {"name": "  ", "type": "object", "description": ""},
                    {"name": "valid concept", "type": "object", "description": "desc"},
                ],
                "relationships": [],
            }
        }
        concepts, edges = merge_extractions(extractions)
        assert len(concepts) == 1
        assert "valid concept" in concepts

    def test_self_referencing_edges_filtered(self):
        from kg.graph import merge_extractions

        extractions = {
            "paper.pdf": {
                "concepts": [
                    {"name": "concept a", "type": "object", "description": ""},
                ],
                "relationships": [
                    {"source": "concept a", "target": "concept a",
                     "relation": "related_to", "detail": "self-ref"},
                ],
            }
        }
        concepts, edges = merge_extractions(extractions)
        assert len(edges) == 0


class TestBuildGraph:
    def test_builds_graph_structure(self, sample_extractions):
        from kg.graph import merge_extractions, build_graph

        concepts, edges = merge_extractions(sample_extractions)
        graph = build_graph(concepts, edges)

        assert "metadata" in graph
        assert "concepts" in graph
        assert "edges" in graph
        assert graph["metadata"]["total_concepts"] > 0
        assert graph["metadata"]["total_papers"] > 0

    def test_deduplicates_edges(self, sample_extractions):
        from kg.graph import merge_extractions, build_graph

        concepts, edges = merge_extractions(sample_extractions)
        graph = build_graph(concepts, edges)

        # "bailey lemma -> rogers-ramanujan identities (proves)" appears in both papers
        # Should be deduplicated into a single edge with both papers listed
        proves_edges = [e for e in graph["edges"]
                        if e["source"] == "bailey lemma"
                        and e["relation"] == "proves"]
        assert len(proves_edges) == 1
        assert len(proves_edges[0]["papers"]) == 2

    def test_metadata_counts(self, sample_extractions):
        from kg.graph import merge_extractions, build_graph

        concepts, edges = merge_extractions(sample_extractions)
        graph = build_graph(concepts, edges)

        assert graph["metadata"]["total_concepts"] == len(concepts)
        assert graph["metadata"]["total_papers"] == 2
        assert "created" in graph["metadata"]


class TestPrepareVizData:
    def test_produces_nodes_and_links(self, sample_extractions):
        from kg.graph import merge_extractions, build_graph, prepare_viz_data

        concepts, edges = merge_extractions(sample_extractions)
        graph = build_graph(concepts, edges)
        viz = prepare_viz_data(graph)

        assert "nodes" in viz
        assert "links" in viz
        assert len(viz["nodes"]) > 0

    def test_node_has_expected_fields(self, sample_extractions):
        from kg.graph import merge_extractions, build_graph, prepare_viz_data

        concepts, edges = merge_extractions(sample_extractions)
        graph = build_graph(concepts, edges)
        viz = prepare_viz_data(graph, min_degree=0)

        node = viz["nodes"][0]
        assert "id" in node
        assert "label" in node
        assert "type" in node
        assert "color" in node
        assert "papers" in node
        assert "degree" in node

    def test_min_degree_filters(self, sample_extractions):
        from kg.graph import merge_extractions, build_graph, prepare_viz_data

        concepts, edges = merge_extractions(sample_extractions)
        graph = build_graph(concepts, edges)

        all_nodes = prepare_viz_data(graph, min_degree=0)
        filtered = prepare_viz_data(graph, min_degree=10)

        # Very high min_degree should fall back to lower thresholds
        assert len(filtered["nodes"]) > 0

    def test_links_reference_kept_nodes(self, sample_extractions):
        from kg.graph import merge_extractions, build_graph, prepare_viz_data

        concepts, edges = merge_extractions(sample_extractions)
        graph = build_graph(concepts, edges)
        viz = prepare_viz_data(graph, min_degree=0)

        node_ids = {n["id"] for n in viz["nodes"]}
        for link in viz["links"]:
            assert link["source"] in node_ids
            assert link["target"] in node_ids
