"""Tests for kg.structural_holes — citation gap detection between concept clusters."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import networkx as nx
import pytest

from kg.structural_holes import (
    extract_arxiv_id,
    scan_papers,
    build_paper_concepts,
    build_citation_graph,
    jaccard,
    compute_citation_density,
    detect_structural_holes,
    find_bridge_papers,
    build_clusters,
    load_citation_cache,
    save_citation_cache,
    generate_report,
    compute_graph_stats,
)


class TestExtractArxivId:
    def test_modern_format_4digit(self):
        assert extract_arxiv_id("borodin-2007.pdf") is None
        assert extract_arxiv_id("schilling_0704.2046.pdf") == "0704.2046"

    def test_modern_format_5digit(self):
        assert extract_arxiv_id("adin-elizalde-roichman-cyclic-descents-matchings-2210.14839.pdf") == "2210.14839"

    def test_modern_format_embedded(self):
        assert extract_arxiv_id("alexandersson-sulzgruber-p-partitions-p-positivity-1807.02460.pdf") == "1807.02460"

    def test_old_format_7digit(self):
        assert extract_arxiv_id("hatayama_et_al_0102113.pdf") == "0102113"
        assert extract_arxiv_id("nakayashiki_yamada_9512027.pdf") == "9512027"

    def test_no_arxiv_id(self):
        assert extract_arxiv_id("gessel_krattenthaler_1997.pdf") is None
        assert extract_arxiv_id("2009_macdonald_polynomials.pdf") is None

    def test_prefers_modern_over_old(self):
        assert extract_arxiv_id("paper_1101.4950.pdf") == "1101.4950"

    def test_thesis_and_year_papers(self):
        assert extract_arxiv_id("2021_phd_thesis_cpp.pdf") is None
        assert extract_arxiv_id("2012_cpp_part_I.pdf") is None
        assert extract_arxiv_id("andrews-schilling-warnaar-1999.pdf") is None


class TestScanPapers:
    def test_scan_finds_pdfs(self, tmp_path):
        (tmp_path / "cluster-a").mkdir()
        (tmp_path / "cluster-a" / "paper_0704.2046.pdf").touch()
        (tmp_path / "cluster-a" / "old_paper.pdf").touch()
        (tmp_path / "cluster-b").mkdir()
        (tmp_path / "cluster-b" / "new_2210.14839.pdf").touch()

        papers = scan_papers(tmp_path)
        assert len(papers) == 3
        assert papers["cluster-a/paper_0704.2046.pdf"]["arxiv_id"] == "0704.2046"
        assert papers["cluster-a/old_paper.pdf"]["arxiv_id"] is None
        assert papers["cluster-b/new_2210.14839.pdf"]["cluster"] == "cluster-b"

    def test_skips_special_dirs(self, tmp_path):
        (tmp_path / ".hidden").mkdir()
        (tmp_path / ".hidden" / "paper.pdf").touch()
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "file.pdf").touch()
        (tmp_path / "compressed").mkdir()
        (tmp_path / "compressed" / "file.pdf").touch()
        (tmp_path / "real-dir").mkdir()
        (tmp_path / "real-dir" / "paper.pdf").touch()

        papers = scan_papers(tmp_path)
        assert len(papers) == 1
        assert "real-dir/paper.pdf" in papers


class TestBuildPaperConcepts:
    def test_maps_papers_to_concepts(self):
        kg = {
            "concepts": [
                {"name": "partitions", "papers": ["a/p1.pdf", "b/p2.pdf"]},
                {"name": "q-series", "papers": ["a/p1.pdf"]},
                {"name": "tableaux", "papers": ["b/p2.pdf"]},
            ]
        }
        pc = build_paper_concepts(kg)
        assert pc["a/p1.pdf"] == {"partitions", "q-series"}
        assert pc["b/p2.pdf"] == {"partitions", "tableaux"}


class TestCitationCache:
    def test_save_and_load(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        cache = {"paper1": {"arxiv_id": "1234.5678", "references": []}}
        save_citation_cache(cache, cache_path)
        loaded = load_citation_cache(cache_path)
        assert loaded == cache

    def test_load_nonexistent(self, tmp_path):
        assert load_citation_cache(tmp_path / "nope.json") == {}


class TestBuildCitationGraph:
    def setup_method(self):
        self.papers = {
            "a/p1.pdf": {"arxiv_id": "1111.1111", "cluster": "a"},
            "a/p2.pdf": {"arxiv_id": "2222.2222", "cluster": "a"},
            "b/p3.pdf": {"arxiv_id": "3333.3333", "cluster": "b"},
        }
        self.cache = {
            "a/p1.pdf": {
                "arxiv_id": "1111.1111",
                "references": [
                    {"paperId": "ext1", "externalIds": {"ArXiv": "3333.3333"},
                     "title": "Paper 3", "year": 2020},
                    {"paperId": "ext2", "externalIds": {},
                     "title": "External", "year": 2019},
                ]
            },
            "b/p3.pdf": {
                "arxiv_id": "3333.3333",
                "references": [
                    {"paperId": "ext2", "externalIds": {},
                     "title": "External", "year": 2019},
                ]
            },
        }

    def test_graph_has_internal_nodes(self):
        G = build_citation_graph(self.papers, self.cache)
        internal = [n for n, d in G.nodes(data=True) if d.get("internal")]
        assert len(internal) == 3

    def test_graph_has_edges(self):
        G = build_citation_graph(self.papers, self.cache)
        assert G.has_edge("a/p1.pdf", "b/p3.pdf")
        assert G.has_edge("a/p1.pdf", "ext:ext2")
        assert G.has_edge("b/p3.pdf", "ext:ext2")

    def test_graph_resolves_internal_refs(self):
        G = build_citation_graph(self.papers, self.cache)
        assert G.nodes["b/p3.pdf"].get("internal") is True

    def test_skips_not_found(self):
        cache = {"a/p1.pdf": {"arxiv_id": "1111.1111", "not_found": True}}
        G = build_citation_graph(self.papers, cache)
        assert G.number_of_edges() == 0

    def test_handles_null_paper_id(self):
        cache = {
            "a/p1.pdf": {
                "arxiv_id": "1111.1111",
                "references": [{"paperId": None, "title": "Book"}]
            }
        }
        G = build_citation_graph(self.papers, cache)
        assert G.number_of_edges() == 0


class TestJaccard:
    def test_identical_sets(self):
        assert jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets(self):
        assert jaccard({"a"}, {"b"}) == 0.0

    def test_partial_overlap(self):
        assert jaccard({"a", "b", "c"}, {"b", "c", "d"}) == 0.5

    def test_empty_sets(self):
        assert jaccard(set(), set()) == 0.0


class TestCitationDensity:
    def test_no_cross_citations(self):
        G = nx.DiGraph()
        G.add_nodes_from(["a1", "a2", "b1"], internal=True)
        G.add_edge("a1", "a2")
        density = compute_citation_density(G, ["a1", "a2"], ["b1"])
        assert density == 0.0

    def test_full_cross_citation(self):
        G = nx.DiGraph()
        G.add_nodes_from(["a1", "b1"], internal=True)
        G.add_edge("a1", "b1")
        G.add_edge("b1", "a1")
        density = compute_citation_density(G, ["a1"], ["b1"])
        assert density == 1.0

    def test_partial_cross_citation(self):
        G = nx.DiGraph()
        G.add_nodes_from(["a1", "b1"], internal=True)
        G.add_edge("a1", "b1")
        density = compute_citation_density(G, ["a1"], ["b1"])
        assert density == 0.5


class TestDetectStructuralHoles:
    def test_high_hole_score_when_concepts_overlap_but_no_citations(self):
        G = nx.DiGraph()
        G.add_nodes_from(["a1", "b1"], internal=True)
        clusters = {
            "A": {"papers": ["a1"], "concepts": {"x", "y", "z"}},
            "B": {"papers": ["b1"], "concepts": {"x", "y", "w"}},
        }
        holes = detect_structural_holes(G, clusters)
        assert len(holes) == 1
        assert holes[0]["hole_score"] > 1

    def test_low_hole_score_when_well_connected(self):
        G = nx.DiGraph()
        G.add_nodes_from(["a1", "b1"], internal=True)
        G.add_edge("a1", "b1")
        G.add_edge("b1", "a1")
        clusters = {
            "A": {"papers": ["a1"], "concepts": {"x", "y", "z"}},
            "B": {"papers": ["b1"], "concepts": {"x", "y", "w"}},
        }
        holes = detect_structural_holes(G, clusters)
        assert holes[0]["hole_score"] < holes[0]["concept_similarity"] / 0.01

    def test_sorted_by_hole_score_descending(self):
        G = nx.DiGraph()
        G.add_nodes_from(["a1", "b1", "c1"], internal=True)
        G.add_edge("a1", "b1")
        clusters = {
            "A": {"papers": ["a1"], "concepts": {"x", "y"}},
            "B": {"papers": ["b1"], "concepts": {"x", "z"}},
            "C": {"papers": ["c1"], "concepts": {"x", "y"}},
        }
        holes = detect_structural_holes(G, clusters)
        ac = next(h for h in holes if set([h["cluster_a"], h["cluster_b"]]) == {"A", "C"})
        ab = next(h for h in holes if set([h["cluster_a"], h["cluster_b"]]) == {"A", "B"})
        assert ac["hole_score"] >= ab["hole_score"]


class TestFindBridgePapers:
    def test_finds_shared_external_reference(self):
        G = nx.DiGraph()
        G.add_node("a/p1.pdf", internal=True, cluster="a")
        G.add_node("b/p2.pdf", internal=True, cluster="b")
        G.add_node("ext:bridge", internal=False, title="Bridge Paper", year=2020)
        G.add_edge("a/p1.pdf", "ext:bridge")
        G.add_edge("b/p2.pdf", "ext:bridge")

        clusters = {
            "a": {"papers": ["a/p1.pdf"], "concepts": set()},
            "b": {"papers": ["b/p2.pdf"], "concepts": set()},
        }
        holes = [{"cluster_a": "a", "cluster_b": "b"}]

        bridges = find_bridge_papers(G, clusters, holes, top_k=1)
        key = ("a", "b")
        assert len(bridges[key]["external_bridges"]) == 1
        assert bridges[key]["external_bridges"][0]["title"] == "Bridge Paper"
        assert bridges[key]["external_bridges"][0]["bridge_score"] == 1

    def test_bridge_score_multiplicative(self):
        G = nx.DiGraph()
        G.add_node("a/p1.pdf", internal=True, cluster="a")
        G.add_node("a/p2.pdf", internal=True, cluster="a")
        G.add_node("b/p3.pdf", internal=True, cluster="b")
        G.add_node("ext:bridge", internal=False, title="Bridge", year=2020)
        G.add_edge("a/p1.pdf", "ext:bridge")
        G.add_edge("a/p2.pdf", "ext:bridge")
        G.add_edge("b/p3.pdf", "ext:bridge")

        clusters = {
            "a": {"papers": ["a/p1.pdf", "a/p2.pdf"], "concepts": set()},
            "b": {"papers": ["b/p3.pdf"], "concepts": set()},
        }
        holes = [{"cluster_a": "a", "cluster_b": "b"}]

        bridges = find_bridge_papers(G, clusters, holes, top_k=1)
        assert bridges[("a", "b")]["external_bridges"][0]["bridge_score"] == 2

    def test_no_bridges_when_no_shared_refs(self):
        G = nx.DiGraph()
        G.add_node("a/p1.pdf", internal=True, cluster="a")
        G.add_node("b/p2.pdf", internal=True, cluster="b")
        G.add_node("ext:only_a", internal=False, title="Only A")
        G.add_node("ext:only_b", internal=False, title="Only B")
        G.add_edge("a/p1.pdf", "ext:only_a")
        G.add_edge("b/p2.pdf", "ext:only_b")

        clusters = {
            "a": {"papers": ["a/p1.pdf"], "concepts": set()},
            "b": {"papers": ["b/p2.pdf"], "concepts": set()},
        }
        holes = [{"cluster_a": "a", "cluster_b": "b"}]

        bridges = find_bridge_papers(G, clusters, holes, top_k=1)
        assert len(bridges[("a", "b")]["external_bridges"]) == 0


class TestReport:
    def test_generates_both_files(self, tmp_path):
        holes = [{
            "cluster_a": "A", "cluster_b": "B",
            "concept_similarity": 0.5, "citation_density": 0.01,
            "hole_score": 50.0, "shared_concepts": ["x", "y"],
            "unique_a_sample": ["a1"], "unique_b_sample": ["b1"],
            "papers_a": 3, "papers_b": 2,
        }]
        bridges = {("A", "B"): {
            "external_bridges": [{
                "title": "Bridge", "year": 2020, "bridge_score": 4,
                "citations_from_a": 2, "citations_from_b": 2,
                "arxiv_id": None, "node_id": "ext:1", "s2_id": "abc",
            }],
            "internal_cross_citations": [],
        }}
        stats = {"internal_nodes": 5, "external_nodes": 100,
                 "total_edges": 200, "int_int_edges": 10, "int_ext_edges": 190,
                 "total_nodes": 105}

        generate_report(holes, bridges, {"A": "Cluster A", "B": "Cluster B"},
                        {("A", "B"): {"queries": ["q1"], "reasoning": "test"}},
                        stats, tmp_path, top_k=1)

        assert (tmp_path / "structural_holes_report.json").exists()
        assert (tmp_path / "structural_holes.md").exists()

        md = (tmp_path / "structural_holes.md").read_text()
        assert "Bridge" in md
        assert "50.00" in md

    def test_graph_stats(self):
        G = nx.DiGraph()
        G.add_node("a", internal=True)
        G.add_node("b", internal=True)
        G.add_node("c", internal=False)
        G.add_edge("a", "b")
        G.add_edge("a", "c")

        stats = compute_graph_stats(G)
        assert stats["internal_nodes"] == 2
        assert stats["external_nodes"] == 1
        assert stats["int_int_edges"] == 1
        assert stats["int_ext_edges"] == 1
