"""Tests for kg.survey — LaTeX survey generation from INSTINCT hierarchy."""
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from kg.survey import (
    slugify,
    topological_sort_themes,
    load_data,
    find_related_l1_summaries,
    build_outline_prompt,
    build_section_prompt,
    build_intro_prompt,
    make_cite_key,
    escape_bibtex,
    generate_bib,
    generate_abstract,
    assemble,
    get_paths,
)


class TestSlugify:
    def test_basic(self):
        assert slugify("Hello World") == "hello-world"

    def test_special_chars(self):
        assert slugify("Graph Neural Network (GNN)") == "graph-neural-network-gnn"

    def test_max_length(self):
        long = "a" * 100
        assert len(slugify(long)) <= 80

    def test_strips_hyphens(self):
        assert slugify("--test--") == "test"

    def test_collapses_spaces(self):
        assert slugify("hello   world") == "hello-world"

    def test_empty_string(self):
        assert slugify("") == ""


class TestGetPaths:
    def test_returns_all_keys(self):
        paths = get_paths("/tmp/test")
        expected_keys = {
            "themes_file", "l2_graph_file", "meta_dir", "compressed_dir",
            "graph_file", "papers_file", "cache_dir", "survey_tex", "survey_bib",
        }
        assert set(paths.keys()) == expected_keys

    def test_paths_are_under_base_dir(self):
        paths = get_paths("/data/corpus")
        for key, path in paths.items():
            assert str(path).startswith("/data/corpus"), f"{key} not under base dir"

    def test_accepts_path_object(self):
        paths = get_paths(Path("/tmp/test"))
        assert isinstance(paths["themes_file"], Path)


class TestMakeCiteKey:
    def test_standard_arxiv_id(self):
        paper = {"base_id": "2210.14839"}
        assert make_cite_key(paper) == "arxiv_2210_14839"

    def test_strips_version(self):
        paper = {"base_id": "2210.14839v2"}
        assert make_cite_key(paper) == "arxiv_2210_14839"

    def test_fallback_to_arxiv_id(self):
        paper = {"arxiv_id": "1807.02460"}
        assert make_cite_key(paper) == "arxiv_1807_02460"

    def test_unknown_fallback(self):
        paper = {}
        assert make_cite_key(paper) == "arxiv_unknown"


class TestEscapeBibtex:
    def test_escapes_ampersand(self):
        assert escape_bibtex("A & B") == r"A \& B"

    def test_escapes_percent(self):
        assert escape_bibtex("100%") == r"100\%"

    def test_escapes_hash(self):
        assert escape_bibtex("C#") == r"C\#"

    def test_escapes_underscore(self):
        assert escape_bibtex("a_b") == r"a\_b"

    def test_plain_text_unchanged(self):
        assert escape_bibtex("Hello World") == "Hello World"


class TestTopologicalSort:
    def _make_themes(self, ids):
        return [{"id": tid, "name": tid.replace("-", " ").title()} for tid in ids]

    def test_linear_chain(self):
        themes = self._make_themes(["a", "b", "c"])
        edges = [
            {"source": "a", "target": "b", "relation": "r", "description": "d"},
            {"source": "b", "target": "c", "relation": "r", "description": "d"},
        ]
        order = topological_sort_themes(themes, edges)
        assert order == ["a", "b", "c"]

    def test_diamond_dag(self):
        themes = self._make_themes(["a", "b", "c", "d"])
        edges = [
            {"source": "a", "target": "b", "relation": "r", "description": "d"},
            {"source": "a", "target": "c", "relation": "r", "description": "d"},
            {"source": "b", "target": "d", "relation": "r", "description": "d"},
            {"source": "c", "target": "d", "relation": "r", "description": "d"},
        ]
        order = topological_sort_themes(themes, edges)
        assert order[0] == "a"
        assert order[-1] == "d"
        assert set(order[1:3]) == {"b", "c"}

    def test_no_edges(self):
        themes = self._make_themes(["z", "a", "m"])
        order = topological_sort_themes(themes, [])
        assert len(order) == 3
        assert order == ["a", "m", "z"]

    def test_single_theme(self):
        themes = self._make_themes(["only"])
        order = topological_sort_themes(themes, [])
        assert order == ["only"]

    def test_tie_break_by_out_degree(self):
        themes = self._make_themes(["hub", "leaf", "mid"])
        edges = [
            {"source": "hub", "target": "leaf", "relation": "r", "description": "d"},
            {"source": "hub", "target": "mid", "relation": "r", "description": "d"},
            {"source": "mid", "target": "leaf", "relation": "r", "description": "d"},
        ]
        order = topological_sort_themes(themes, edges)
        assert order[0] == "hub"

    def test_cycle_fallback(self):
        themes = self._make_themes(["a", "b", "c"])
        edges = [
            {"source": "a", "target": "b", "relation": "r", "description": "d"},
            {"source": "b", "target": "c", "relation": "r", "description": "d"},
            {"source": "c", "target": "a", "relation": "r", "description": "d"},
        ]
        order = topological_sort_themes(themes, edges)
        assert set(order) == {"a", "b", "c"}

    def test_ignores_edges_with_unknown_nodes(self):
        themes = self._make_themes(["a", "b"])
        edges = [
            {"source": "a", "target": "b", "relation": "r", "description": "d"},
            {"source": "a", "target": "ghost", "relation": "r", "description": "d"},
        ]
        order = topological_sort_themes(themes, edges)
        assert order == ["a", "b"]


class TestLoadData:
    def _setup_corpus(self, tmp_path):
        themes = {"themes": [
            {"id": "theme-a", "name": "Theme A", "description": "Desc A",
             "key_concepts": ["Concept X"]},
        ]}
        (tmp_path / "level2_themes.json").write_text(json.dumps(themes))

        l2_graph = {
            "themes": [{"name": "theme-a", "display_name": "Theme A"}],
            "edges": [{"source": "theme-a", "target": "theme-b",
                       "relation": "r", "description": "d"}],
        }
        (tmp_path / "level2_knowledge_graph.json").write_text(json.dumps(l2_graph))

        l0 = {
            "concepts": [{"name": "Concept X", "papers": ["p1.pdf"]}],
            "edges": [{"source": "Concept X", "target": "Concept Y",
                       "relation": "related", "description": "desc"}],
        }
        (tmp_path / "knowledge_graph.json").write_text(json.dumps(l0))

        papers = [{"base_id": "2210.14839", "title": "Test Paper",
                   "authors": ["Author A"], "published": "2022-10-01",
                   "pdf_url": "https://arxiv.org/pdf/2210.14839"}]
        (tmp_path / "selected_papers.json").write_text(json.dumps(papers))

        meta_dir = tmp_path / "meta-summaries"
        meta_dir.mkdir()
        (meta_dir / "theme-a.md").write_text("# Theme A\n\nMeta-summary content.")

        comp_dir = tmp_path / "compressed"
        comp_dir.mkdir()
        (comp_dir / "concept-x.md").write_text("# Concept X\n\nDetailed summary.")

        return get_paths(tmp_path)

    def test_loads_all_fields(self, tmp_path):
        paths = self._setup_corpus(tmp_path)
        data = load_data(paths)

        assert len(data["themes"]) == 1
        assert data["themes"][0]["id"] == "theme-a"
        assert len(data["l2_edges"]) == 1
        assert "theme-a" in data["meta_summaries"]
        assert "concept-x" in data["l1_summaries"]
        assert len(data["l0_concepts"]) == 1
        assert len(data["papers"]) == 1

    def test_handles_empty_meta_dir(self, tmp_path):
        paths = self._setup_corpus(tmp_path)
        for f in (tmp_path / "meta-summaries").iterdir():
            f.unlink()
        data = load_data(paths)
        assert data["meta_summaries"] == {}


class TestFindRelatedL1Summaries:
    def setup_method(self):
        self.data = {
            "l1_summaries": {
                "concept-x": "# Concept X\nAbout concept X and embeddings.",
                "concept-y": "# Concept Y\nAbout concept Y and something else.",
                "unrelated": "# Unrelated\nNothing relevant here.",
            },
            "l0_concepts": [
                {"name": "Concept X", "papers": ["p1.pdf"]},
                {"name": "Concept Y", "papers": ["p2.pdf"]},
            ],
            "l0_edges": [
                {"source": "Concept X", "target": "Concept Y",
                 "relation": "related", "description": "linked"},
            ],
        }

    def test_direct_match_by_key_concept(self):
        theme = {"key_concepts": ["Concept X"]}
        results = find_related_l1_summaries(theme, self.data)
        slugs = [slug for slug, _ in results]
        assert "concept-x" in slugs

    def test_neighbor_match_via_l0_edge(self):
        theme = {"key_concepts": ["Concept X"]}
        results = find_related_l1_summaries(theme, self.data)
        slugs = [slug for slug, _ in results]
        assert "concept-y" in slugs

    def test_unrelated_excluded(self):
        theme = {"key_concepts": ["Concept X"]}
        results = find_related_l1_summaries(theme, self.data)
        slugs = [slug for slug, _ in results]
        assert "unrelated" not in slugs

    def test_respects_max_summaries(self):
        theme = {"key_concepts": ["Concept X"]}
        results = find_related_l1_summaries(theme, self.data, max_summaries=1)
        assert len(results) <= 1

    def test_empty_key_concepts(self):
        theme = {"key_concepts": []}
        results = find_related_l1_summaries(theme, self.data)
        assert results == []

    def test_direct_match_scores_higher_than_neighbor(self):
        theme = {"key_concepts": ["Concept X"]}
        results = find_related_l1_summaries(theme, self.data)
        if len(results) >= 2:
            assert results[0][0] == "concept-x"


class TestBuildOutlinePrompt:
    def test_contains_theme_info(self):
        themes = [
            {"id": "t1", "name": "Theme One", "description": "Desc 1",
             "key_concepts": ["A", "B"]},
        ]
        edges = [{"source": "t1", "target": "t2", "relation": "r",
                  "description": "connection"}]
        prompt = build_outline_prompt(["t1"], themes, edges)
        assert "Theme One" in prompt
        assert "A, B" in prompt
        assert "connection" in prompt

    def test_contains_json_schema(self):
        themes = [{"id": "t1", "name": "T1", "description": "D",
                   "key_concepts": []}]
        prompt = build_outline_prompt(["t1"], themes, [])
        assert '"sections"' in prompt
        assert '"theme_ids"' in prompt


class TestBuildSectionPrompt:
    def setup_method(self):
        self.data = {
            "meta_summaries": {"t1": "Meta summary for theme 1."},
            "l1_summaries": {"concept-a": "# Concept A\nDetails about concept A."},
            "l0_concepts": [{"name": "Concept A", "papers": ["p1.pdf"]}],
            "l0_edges": [],
            "l2_edges": [{"source": "t1", "target": "t2",
                          "description": "themes are linked"}],
        }
        self.theme_map = {
            "t1": {"id": "t1", "name": "Theme 1", "key_concepts": ["Concept A"]},
            "t2": {"id": "t2", "name": "Theme 2", "key_concepts": []},
        }
        self.outline = {
            "sections": [
                {"id": "sec-1", "title": "Section 1", "theme_ids": ["t1"],
                 "cross_refs": ["sec-2"]},
                {"id": "sec-2", "title": "Section 2", "theme_ids": ["t2"]},
            ]
        }

    def test_includes_meta_summary(self):
        section = self.outline["sections"][0]
        prompt = build_section_prompt(section, self.outline, self.data, self.theme_map)
        assert "Meta summary for theme 1" in prompt

    def test_includes_l1_content(self):
        section = self.outline["sections"][0]
        prompt = build_section_prompt(section, self.outline, self.data, self.theme_map)
        assert "Concept A" in prompt

    def test_includes_cross_refs(self):
        section = self.outline["sections"][0]
        prompt = build_section_prompt(section, self.outline, self.data, self.theme_map)
        assert "Section 2" in prompt
        assert "sec-2" in prompt

    def test_includes_theme_connections(self):
        section = self.outline["sections"][0]
        prompt = build_section_prompt(section, self.outline, self.data, self.theme_map)
        assert "themes are linked" in prompt


class TestBuildIntroPrompt:
    def test_includes_section_titles(self):
        outline = {
            "title": "Survey Title",
            "abstract_guidance": "Write about KGs.",
            "sections": [
                {"id": "s1", "title": "Introduction to KG"},
                {"id": "s2", "title": "Advanced Topics"},
            ],
        }
        section_texts = {
            "s1": "\\section{Introduction to KG}\n\nThis section covers...",
            "s2": "\\section{Advanced Topics}\n\nAdvanced material...",
        }
        prompt = build_intro_prompt(outline, section_texts)
        assert "Introduction to KG" in prompt
        assert "Advanced Topics" in prompt
        assert "Survey Title" in prompt

    def test_includes_separator_marker(self):
        outline = {"title": "T", "sections": []}
        prompt = build_intro_prompt(outline, {})
        assert "%%% CONCLUSION %%%" in prompt


class TestGenerateBib:
    def test_creates_bib_file(self, tmp_path):
        papers = [
            {"base_id": "2210.14839", "title": "Test Paper",
             "authors": ["Author A", "Author B"], "published": "2022-10-01",
             "pdf_url": "https://arxiv.org/pdf/2210.14839"},
        ]
        bib_path = tmp_path / "survey.bib"
        generate_bib(papers, bib_path)

        bib_text = bib_path.read_text()
        assert "@article{arxiv_2210_14839" in bib_text
        assert "Author A and Author B" in bib_text
        assert "2022" in bib_text

    def test_deduplicates_by_cite_key(self, tmp_path):
        papers = [
            {"base_id": "1234.5678", "title": "Paper A", "authors": ["X"],
             "published": "2023-01-01", "pdf_url": ""},
            {"base_id": "1234.5678v2", "title": "Paper A v2", "authors": ["X"],
             "published": "2023-06-01", "pdf_url": ""},
        ]
        bib_path = tmp_path / "survey.bib"
        generate_bib(papers, bib_path)

        bib_text = bib_path.read_text()
        assert bib_text.count("@article{arxiv_1234_5678") == 1

    def test_escapes_special_chars_in_title(self, tmp_path):
        papers = [
            {"base_id": "0001.0001", "title": "A & B: 100% C#",
             "authors": ["X"], "published": "2020-01-01", "pdf_url": ""},
        ]
        bib_path = tmp_path / "survey.bib"
        generate_bib(papers, bib_path)

        bib_text = bib_path.read_text()
        assert r"\&" in bib_text
        assert r"\%" in bib_text


class TestAssemble:
    def _make_minimal_data(self, tmp_path):
        paths = get_paths(tmp_path)
        paths["cache_dir"].mkdir(parents=True, exist_ok=True)

        data = {
            "papers": [
                {"base_id": "2210.14839", "title": "Test Paper",
                 "authors": ["Author"], "published": "2022-01-01",
                 "pdf_url": "https://arxiv.org/pdf/2210.14839"},
            ]
        }
        outline = {
            "title": "A Survey on Knowledge Graphs",
            "abstract_guidance": "Cover KG methods.",
            "sections": [
                {"id": "sec-1", "title": "Embeddings", "theme_ids": ["t1"]},
                {"id": "sec-2", "title": "Applications", "theme_ids": ["t2"]},
            ],
        }
        section_texts = {
            "sec-1": "\\section{Embeddings}\n\\label{sec:sec-1}\nContent about embeddings.",
            "sec-2": "\\section{Applications}\n\\label{sec:sec-2}\nContent about applications.",
        }
        intro = "\\section{Introduction}\n\\label{sec:introduction}\nIntro text."
        conclusion = "\\section{Conclusion}\n\\label{sec:conclusion}\nConclusion text."

        return paths, data, outline, section_texts, intro, conclusion

    def test_creates_tex_file(self, tmp_path):
        paths, data, outline, section_texts, intro, conclusion = self._make_minimal_data(tmp_path)
        assemble(outline, section_texts, intro, conclusion, data, paths)
        assert paths["survey_tex"].exists()

    def test_creates_bib_file(self, tmp_path):
        paths, data, outline, section_texts, intro, conclusion = self._make_minimal_data(tmp_path)
        assemble(outline, section_texts, intro, conclusion, data, paths)
        assert paths["survey_bib"].exists()

    def test_tex_has_document_structure(self, tmp_path):
        paths, data, outline, section_texts, intro, conclusion = self._make_minimal_data(tmp_path)
        assemble(outline, section_texts, intro, conclusion, data, paths)

        tex = paths["survey_tex"].read_text()
        assert "\\documentclass" in tex
        assert "\\begin{document}" in tex
        assert "\\end{document}" in tex
        assert "\\maketitle" in tex
        assert "\\tableofcontents" in tex

    def test_tex_contains_all_sections(self, tmp_path):
        paths, data, outline, section_texts, intro, conclusion = self._make_minimal_data(tmp_path)
        assemble(outline, section_texts, intro, conclusion, data, paths)

        tex = paths["survey_tex"].read_text()
        assert "Content about embeddings" in tex
        assert "Content about applications" in tex
        assert "Intro text" in tex
        assert "Conclusion text" in tex

    def test_tex_section_order(self, tmp_path):
        paths, data, outline, section_texts, intro, conclusion = self._make_minimal_data(tmp_path)
        assemble(outline, section_texts, intro, conclusion, data, paths)

        tex = paths["survey_tex"].read_text()
        intro_pos = tex.index("Intro text")
        sec1_pos = tex.index("Content about embeddings")
        sec2_pos = tex.index("Content about applications")
        concl_pos = tex.index("Conclusion text")
        assert intro_pos < sec1_pos < sec2_pos < concl_pos

    def test_missing_section_gets_todo(self, tmp_path):
        paths, data, outline, section_texts, intro, conclusion = self._make_minimal_data(tmp_path)
        del section_texts["sec-2"]
        assemble(outline, section_texts, intro, conclusion, data, paths)

        tex = paths["survey_tex"].read_text()
        assert "TODO: generate section sec-2" in tex


class TestGenerateAbstract:
    def test_returns_latex_abstract(self):
        outline = {"abstract_guidance": "Cover KG methods."}
        abstract = generate_abstract(outline)
        assert "\\begin{abstract}" in abstract
        assert "\\end{abstract}" in abstract

    def test_includes_guidance_comment(self):
        outline = {"abstract_guidance": "Focus on embeddings."}
        abstract = generate_abstract(outline)
        assert "Focus on embeddings" in abstract
