"""
survey.py — Generate a survey paper from the INSTINCT hierarchy.

Reads the 3-tier INSTINCT data (L0 concepts, L1 summaries, L2 themes) and uses
an LLM to produce a complete survey paper in LaTeX or Markdown.

Pipeline:
    1. Load all INSTINCT data (themes, graphs, summaries)
    2. Topological sort of themes (foundational -> derived)
    3. Generate outline via LLM (sections, structure, cross-refs)
    4. Generate each section via LLM (with relevant L1/L2 context)
    5. Generate introduction + conclusion via LLM
    6. Assemble final document (LaTeX + .bib, or Markdown with references)
"""

import json
import os
import re
import time
from collections import defaultdict, deque
from pathlib import Path

from .llm import AnthropicAdapter, OpenAIAdapter, with_retry
from .utils import slugify

DEFAULT_MODEL = "gpt-4o"
MAX_TOKENS_OUTLINE = 4096
MAX_TOKENS_SECTION = 8192
MAX_TOKENS_INTRO = 6144

# ── System prompts ───────────────────────────────────────────────────

OUTLINE_SYSTEM = (
    "You are an expert academic survey writer. You produce well-structured "
    "survey papers that synthesize research themes into a coherent narrative. "
    "You write in a formal but accessible academic style."
)

SECTION_SYSTEM_LATEX = (
    "You are an expert academic survey writer producing LaTeX content. "
    "Write in formal academic style with proper LaTeX formatting. "
    "Use \\label{} for sections/subsections and "
    "\\ref{} for cross-references. Use \\cite{} for citations (cite keys are "
    "arxiv_NNNN_NNNNN format, e.g., arxiv_2002_00388). Do NOT include "
    "\\begin{document} or preamble — just the section content."
)

SECTION_SYSTEM_MD = (
    "You are an expert academic survey writer producing Markdown content. "
    "Write in formal academic style with proper Markdown formatting. "
    "Use ## for sections and ### for subsections. "
    "Use [N] for inline citations where N is a number (e.g., [1], [2, 3]). "
    "I will provide the mapping from cite keys to numbers. "
    "Reference other sections by name (e.g., 'as discussed in the "
    "Diagnostic Tests section')."
)

INTRO_SYSTEM_LATEX = (
    "You are an expert academic survey writer. Write formal LaTeX content "
    "for the introduction and conclusion of a survey paper. Use \\cite{} "
    "and \\ref{} as appropriate. Cite keys use arxiv_NNNN_NNNNN format. "
    "Do NOT include \\begin{document} or preamble."
)

INTRO_SYSTEM_MD = (
    "You are an expert academic survey writer. Write formal Markdown content "
    "for the introduction and conclusion of a survey paper. "
    "Use [N] for inline citations where N is a number. "
    "Reference other sections by name. Do NOT include YAML front matter."
)

LATEX_PREAMBLE = r"""\documentclass[11pt,a4paper]{article}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{hyperref}
\usepackage[numbers,sort&compress]{natbib}
\usepackage[margin=2.5cm]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}

\hypersetup{
  colorlinks=true,
  linkcolor=blue!60!black,
  citecolor=green!50!black,
  urlcolor=blue!70!black
}

"""

# ── Utility functions ────────────────────────────────────────────────


def get_paths(base_dir, output_format="markdown"):
    """Return a dict of standard file/directory paths for the survey pipeline.

    Args:
        base_dir: Root data directory containing INSTINCT outputs.
        output_format: 'latex' or 'markdown'. Affects cache subdirectory
            and output file extension to avoid format conflicts.
    """
    d = Path(base_dir)
    fmt = output_format  # 'latex' or 'markdown'
    ext = ".tex" if fmt == "latex" else ".md"
    paths = {
        "themes_file": d / "level2_themes.json",
        "l2_graph_file": d / "level2_knowledge_graph.json",
        "meta_dir": d / "meta-summaries",
        "compressed_dir": d / "compressed",
        "graph_file": d / "knowledge_graph.json",
        "papers_file": d / "selected_papers.json",
        "cache_dir": d / f"survey_cache_{fmt}",
        "survey_out": d / f"survey{ext}",
        "survey_bib": d / "survey.bib",
        "format": fmt,
    }
    return paths


# ── Data loading ─────────────────────────────────────────────────────


def load_data(paths):
    """Load all INSTINCT data files and return a unified data dict."""
    data = {}

    with open(paths["themes_file"]) as f:
        data["themes"] = json.load(f)["themes"]
    print(f"  Loaded {len(data['themes'])} themes")

    with open(paths["l2_graph_file"]) as f:
        l2_graph = json.load(f)
    data["l2_edges"] = l2_graph["edges"]
    data["l2_themes_graph"] = l2_graph["themes"]
    print(f"  Loaded L2 graph: {len(data['l2_edges'])} edges")

    data["meta_summaries"] = {}
    meta_dir = paths["meta_dir"]
    if meta_dir.exists():
        for md_file in sorted(meta_dir.glob("*.md")):
            data["meta_summaries"][md_file.stem] = md_file.read_text()
    print(f"  Loaded {len(data['meta_summaries'])} meta-summaries")

    data["l1_summaries"] = {}
    comp_dir = paths["compressed_dir"]
    if comp_dir.exists():
        for md_file in sorted(comp_dir.glob("*.md")):
            data["l1_summaries"][md_file.stem] = md_file.read_text()
    print(f"  Loaded {len(data['l1_summaries'])} L1 summaries")

    with open(paths["graph_file"]) as f:
        l0_graph = json.load(f)
    data["l0_concepts"] = l0_graph["concepts"]
    data["l0_edges"] = l0_graph["edges"]
    print(f"  Loaded L0 graph: {len(data['l0_concepts'])} concepts, "
          f"{len(data['l0_edges'])} edges")

    with open(paths["papers_file"]) as f:
        data["papers"] = json.load(f)
    print(f"  Loaded {len(data['papers'])} papers for bibliography")

    return data


# ── Topological sort ─────────────────────────────────────────────────


def topological_sort_themes(themes, edges):
    """Sort themes in foundational-to-derived order using topological sort.

    Falls back to appending unsorted remaining themes if the graph has cycles.
    """
    theme_ids = {t["id"] for t in themes}
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    adj = defaultdict(list)

    for e in edges:
        src, tgt = e["source"], e["target"]
        if src in theme_ids and tgt in theme_ids:
            adj[src].append(tgt)
            in_degree[tgt] += 1
            out_degree[src] += 1

    for t in themes:
        in_degree.setdefault(t["id"], 0)
        out_degree.setdefault(t["id"], 0)

    queue = []
    for t in themes:
        if in_degree[t["id"]] == 0:
            queue.append(t["id"])

    queue.sort(key=lambda tid: (-out_degree[tid], tid))
    queue = deque(queue)

    ordered = []
    while queue:
        node = queue.popleft()
        ordered.append(node)
        next_ready = []
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                next_ready.append(neighbor)
        next_ready.sort(key=lambda tid: (-out_degree[tid], tid))
        queue.extend(next_ready)

    remaining = [t["id"] for t in themes if t["id"] not in set(ordered)]
    ordered.extend(sorted(remaining))

    print(f"  Theme order ({len(ordered)} themes):")
    for i, tid in enumerate(ordered):
        name = next((t["name"] for t in themes if t["id"] == tid), tid)
        print(f"    {i+1:2d}. {name}")

    return ordered


# ── Outline generation ───────────────────────────────────────────────


def build_outline_prompt(ordered_ids, themes, edges):
    """Build the LLM prompt for generating the survey outline."""
    theme_map = {t["id"]: t for t in themes}

    theme_descriptions = []
    for tid in ordered_ids:
        t = theme_map[tid]
        theme_descriptions.append(
            f"- **{t['name']}** (id: {tid}): {t['description']}\n"
            f"  Key concepts: {', '.join(t.get('key_concepts', []))}"
        )

    edge_descriptions = []
    for e in edges:
        edge_descriptions.append(
            f"- {e['source']} -> {e['target']} ({e['relation']}): {e['description']}"
        )

    return f"""I have {len(ordered_ids)} research themes from a literature review on \
knowledge graphs and related AI techniques. I need you to design the structure for a \
comprehensive survey paper.

THEMES (in foundational -> derived order):
{chr(10).join(theme_descriptions)}

CONNECTIONS BETWEEN THEMES:
{chr(10).join(edge_descriptions)}

Design a survey paper outline. You may merge closely related themes into a single section \
if it makes narrative sense, but every theme must be covered. Return your answer as JSON \
with this exact schema:

{{
  "title": "Survey paper title",
  "abstract_guidance": "2-3 sentences describing what the abstract should cover",
  "sections": [
    {{
      "id": "section-slug",
      "title": "Section Title",
      "theme_ids": ["theme-id-1", "theme-id-2"],
      "subsections": ["Subsection Title 1", "Subsection Title 2"],
      "cross_refs": ["other-section-slug"],
      "guidance": "Brief description of what this section should cover and emphasize"
    }}
  ]
}}

Guidelines:
- Aim for 12-16 sections (plus intro and conclusion, which you don't need to include)
- Each section should have 2-4 subsections
- Merged sections should combine at most 2-3 closely related themes
- Cross-references should indicate where sections naturally reference each other
- Order sections so the paper reads as a logical narrative progression"""


def resolve_theme_ids(outline, valid_ids):
    """Resolve potentially shortened theme IDs in the outline to actual IDs.

    LLMs sometimes abbreviate theme IDs (e.g., 'research-methods' instead of
    'research-methods-in-auditory-processing'). This maps each outline theme ID
    to the best-matching valid ID using substring/prefix matching.

    Args:
        outline: Outline dict with sections containing theme_ids.
        valid_ids: Set of actual theme IDs from the data.

    Returns:
        The outline dict with corrected theme_ids (modified in place).
    """
    valid_set = set(valid_ids)

    for section in outline.get("sections", []):
        resolved = []
        for tid in section.get("theme_ids", []):
            if tid in valid_set:
                resolved.append(tid)
                continue

            # Try prefix match: find valid IDs that start with the outline ID
            candidates = [v for v in valid_set if v.startswith(tid)]
            if len(candidates) == 1:
                resolved.append(candidates[0])
                continue

            # Try substring match: find valid IDs that contain the outline ID
            candidates = [v for v in valid_set if tid in v]
            if len(candidates) == 1:
                resolved.append(candidates[0])
                continue

            # Try matching the outline ID as a prefix of valid IDs (reversed)
            candidates = [v for v in valid_set if v.startswith(tid.split("-")[0])]

            # Best-effort: pick the shortest candidate or skip
            if candidates:
                best = min(candidates, key=len)
                print(f"  Warning: resolved '{tid}' -> '{best}' (fuzzy)")
                resolved.append(best)
            else:
                print(f"  Warning: dropping unknown theme ID '{tid}'")

        section["theme_ids"] = resolved

    return outline


def generate_outline(adapter, ordered_ids, themes, edges, paths,
                     model=DEFAULT_MODEL, force=False):
    """Generate the survey outline via LLM, with caching.

    Args:
        adapter: An AnthropicAdapter instance.
        ordered_ids: Theme IDs in topological order.
        themes: List of theme dicts.
        edges: List of L2 edge dicts.
        paths: Paths dict from get_paths().
        model: LLM model identifier.
        force: If True, regenerate even if cached.

    Returns:
        Parsed outline dict with title, abstract_guidance, and sections.
    """
    cache_file = paths["cache_dir"] / "outline.json"

    if cache_file.exists() and not force:
        print("  Outline cached, loading...")
        with open(cache_file) as f:
            return json.load(f)

    print(f"  Generating outline via {model}...")
    prompt = build_outline_prompt(ordered_ids, themes, edges)

    def call():
        return adapter.chat(OUTLINE_SYSTEM, prompt, model,
                            MAX_TOKENS_OUTLINE, temperature=0.2, json_mode=True)

    raw = with_retry(call)

    json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', raw, re.DOTALL)
    if json_match:
        raw = json_match.group(1)

    outline = json.loads(raw)

    with open(cache_file, 'w') as f:
        json.dump(outline, f, indent=2)
    print(f"  Outline saved: {len(outline['sections'])} sections")

    return outline


# ── Section generation ───────────────────────────────────────────────


def find_related_l1_summaries(theme, data, max_summaries=10):
    """Find L1 summaries related to a theme, scored by relevance."""
    key_concepts = set()
    for kc in theme.get("key_concepts", []):
        key_concepts.add(slugify(kc))

    neighbor_slugs = set()
    for kc in theme.get("key_concepts", []):
        kc_lower = kc.lower()
        for e in data["l0_edges"]:
            if e["source"].lower() == kc_lower:
                neighbor_slugs.add(slugify(e["target"]))
            elif e["target"].lower() == kc_lower:
                neighbor_slugs.add(slugify(e["source"]))

    scored = []
    for slug, text in data["l1_summaries"].items():
        score = 0
        if slug in key_concepts:
            score += 10
        if slug in neighbor_slugs:
            score += 3
        header = text[:500].lower()
        for kc in theme.get("key_concepts", []):
            if kc.lower() in header:
                score += 1
        if score > 0:
            scored.append((score, slug, text))

    scored.sort(key=lambda x: -x[0])
    return [(slug, text) for _, slug, text in scored[:max_summaries]]


def build_section_prompt(section, outline, data, theme_map, output_format="latex",
                         cite_map=None):
    """Build the LLM prompt for generating a single survey section.

    Args:
        section: Section dict from the outline.
        outline: Full outline dict.
        data: Loaded INSTINCT data dict.
        theme_map: Dict mapping theme ID to theme dict.
        output_format: 'latex' or 'markdown'.
        cite_map: For markdown, a dict mapping cite_key -> number.
    """
    theme_ids = section["theme_ids"]

    meta_texts = []
    for tid in theme_ids:
        if tid in data["meta_summaries"]:
            meta_texts.append(
                f"### Theme: {theme_map[tid]['name']}\n\n"
                f"{data['meta_summaries'][tid]}"
            )

    l1_texts = []
    seen_slugs = set()
    for tid in theme_ids:
        theme = theme_map[tid]
        related = find_related_l1_summaries(theme, data)
        for slug, text in related:
            if slug not in seen_slugs:
                seen_slugs.add(slug)
                if len(text) > 3000:
                    text = text[:3000] + "\n[... truncated]"
                l1_texts.append(f"### Concept: {slug}\n\n{text}")

    connections = []
    for e in data["l2_edges"]:
        if e["source"] in theme_ids or e["target"] in theme_ids:
            connections.append(
                f"- {e['source']} <-> {e['target']}: {e['description']}"
            )

    cross_refs = section.get("cross_refs", [])
    section_map = {s["id"]: s["title"] for s in outline["sections"]}
    cross_ref_notes = []
    for ref_id in cross_refs:
        if ref_id in section_map:
            if output_format == "latex":
                cross_ref_notes.append(
                    f"- Section \"{section_map[ref_id]}\" "
                    f"(\\ref{{sec:{ref_id}}})"
                )
            else:
                cross_ref_notes.append(
                    f"- Section \"{section_map[ref_id]}\""
                )

    # Format-specific instructions
    if output_format == "latex":
        format_instructions = f"""- Write a complete LaTeX section using \\section{{{section['title']}}} \
with \\label{{sec:{section['id']}}}
- Include subsections using \\subsection{{}} with \\label{{subsec:...}}
- Use \\cite{{arxiv_NNNN_NNNNN}} for citations (underscore-separated arxiv IDs)
- Use \\ref{{sec:...}} for cross-references to other sections
- Output ONLY the LaTeX content, no preamble or document wrapper"""
    else:
        cite_hint = ""
        if cite_map:
            # Show the cite keys relevant to this section's themes
            relevant_keys = _find_relevant_cite_keys(theme_ids, data, cite_map)
            if relevant_keys:
                cite_lines = [f"  {key} = [{num}]" for key, num in relevant_keys[:30]]
                cite_hint = (
                    "\n\nAVAILABLE CITATIONS (cite_key = [number]):\n"
                    + "\n".join(cite_lines)
                )
        format_instructions = f"""- Write a Markdown section using ## {section['title']}
- Include subsections using ### headings
- Use [N] for inline citations (e.g., [1], [2, 3]) where N is the reference number
- Reference other sections by name (e.g., "as discussed in the Diagnostic Tests section")
- Output ONLY the Markdown content, no front matter{cite_hint}"""

    return f"""Write the {'LaTeX' if output_format == 'latex' else 'Markdown'} content \
for the following survey section.

SECTION: {section['title']}
SECTION ID: {section['id']}
SUBSECTIONS: {', '.join(section.get('subsections', []))}
GUIDANCE: {section.get('guidance', 'Cover the themes thoroughly.')}

CROSS-REFERENCES TO OTHER SECTIONS:
{chr(10).join(cross_ref_notes) if cross_ref_notes else '(none)'}

THEME CONNECTIONS:
{chr(10).join(connections) if connections else '(none)'}

LEVEL 2 META-SUMMARIES (comprehensive theme overviews):
{chr(10).join(meta_texts) if meta_texts else '(none available)'}

LEVEL 1 CONCEPT SUMMARIES (detailed concept-level material):
{chr(10).join(l1_texts[:10]) if l1_texts else '(none available)'}

INSTRUCTIONS:
{format_instructions}
- Synthesize the material — don't just list papers; identify patterns, \
compare approaches, note evolution
- Be thorough but concise — aim for 2-4 pages of content per section"""


def _find_relevant_cite_keys(theme_ids, data, cite_map):
    """Find cite keys relevant to the given themes, for Markdown citation hints."""
    # Build a set of paper IDs associated with these themes via L0 concepts
    theme_set = set(theme_ids)
    relevant = set()

    # Match papers whose concepts overlap with theme key concepts
    theme_concepts = set()
    for t in data["themes"]:
        if t["id"] in theme_set:
            for kc in t.get("key_concepts", []):
                theme_concepts.add(kc.lower())

    for concept in data["l0_concepts"]:
        if concept.get("name", "").lower() in theme_concepts:
            for src in concept.get("sources", []):
                relevant.add(src)

    # Map to cite keys and return sorted by number
    results = []
    for paper in data["papers"]:
        key = make_cite_key(paper)
        pid = paper.get("base_id", paper.get("arxiv_id", ""))
        if key in cite_map and (pid in relevant or not relevant):
            results.append((key, cite_map[key]))

    results.sort(key=lambda x: x[1])
    return results


def generate_section(adapter, section, outline, data, theme_map, paths,
                     model=DEFAULT_MODEL, force=False, output_format="latex",
                     cite_map=None):
    """Generate a single survey section via LLM, with caching.

    Args:
        adapter: LLM adapter (AnthropicAdapter or OpenAIAdapter).
        section: Section dict from the outline.
        outline: Full outline dict.
        data: Loaded INSTINCT data dict.
        theme_map: Dict mapping theme ID to theme dict.
        paths: Paths dict from get_paths().
        model: LLM model identifier.
        force: If True, regenerate even if cached.
        output_format: 'latex' or 'markdown'.
        cite_map: For markdown, dict mapping cite_key -> reference number.

    Returns:
        Section content string (LaTeX or Markdown).
    """
    section_id = section["id"]
    ext = ".tex" if output_format == "latex" else ".md"
    cache_file = paths["cache_dir"] / f"section_{section_id}{ext}"

    if cache_file.exists() and not force:
        print(f"    [{section_id}] cached, skipping")
        return cache_file.read_text()

    print(f"    [{section_id}] generating...")
    prompt = build_section_prompt(section, outline, data, theme_map,
                                  output_format=output_format, cite_map=cite_map)

    system = SECTION_SYSTEM_LATEX if output_format == "latex" else SECTION_SYSTEM_MD

    def call():
        return adapter.chat(system, prompt, model, MAX_TOKENS_SECTION)

    text = with_retry(call)
    cache_file.write_text(text)
    return text


# ── Introduction and conclusion ──────────────────────────────────────


def build_intro_prompt(outline, section_texts, output_format="latex"):
    """Build the LLM prompt for generating the introduction and conclusion."""
    section_previews = []
    for section in outline["sections"]:
        sid = section["id"]
        text = section_texts.get(sid, "")
        lines = text.split('\n')
        preview_lines = []
        past_header = False
        for line in lines:
            # Skip section headers in both formats
            if line.strip().startswith('\\section') or line.strip().startswith('## '):
                past_header = True
                continue
            if past_header and line.strip():
                preview_lines.append(line.strip())
                if len(' '.join(preview_lines)) > 300:
                    break
        preview = ' '.join(preview_lines)[:300]
        if output_format == "latex":
            section_previews.append(
                f"- **{section['title']}** (\\ref{{sec:{sid}}}): {preview}"
            )
        else:
            section_previews.append(
                f"- **{section['title']}**: {preview}"
            )

    if output_format == "latex":
        format_block = """Write TWO pieces of LaTeX:

PART 1 — INTRODUCTION (\\section{Introduction} with \\label{sec:introduction}):
- Motivation: why this survey area matters
- Scope: what the survey covers and its boundaries
- Methodology: briefly describe the INSTINCT literature review methodology
- Roadmap: overview of the paper structure, referencing each section with \\ref{}

PART 2 — CONCLUSION (\\section{Conclusion} with \\label{sec:conclusion}):
- Summary of key findings across the surveyed themes
- Open problems and challenges
- Future research directions
- Brief outlook

Separate the two parts with the exact marker: %%% CONCLUSION %%%

Output ONLY the LaTeX content."""
    else:
        format_block = """Write TWO pieces of Markdown:

PART 1 — INTRODUCTION (## Introduction):
- Motivation: why this survey area matters
- Scope: what the survey covers and its boundaries
- Methodology: briefly describe the INSTINCT literature review methodology
- Roadmap: overview of the paper structure, referencing each section by name

PART 2 — CONCLUSION (## Conclusion):
- Summary of key findings across the surveyed themes
- Open problems and challenges
- Future research directions
- Brief outlook

Separate the two parts with the exact marker: %%% CONCLUSION %%%

Output ONLY the Markdown content. Do not include YAML front matter."""

    return f"""Write the introduction and conclusion for a survey paper.

PAPER TITLE: {outline['title']}
ABSTRACT GUIDANCE: {outline.get('abstract_guidance', '')}

SECTIONS IN THE PAPER:
{chr(10).join(section_previews)}

{format_block}"""


def generate_intro_conclusion(adapter, outline, section_texts, paths,
                              model=DEFAULT_MODEL, force=False,
                              output_format="latex"):
    """Generate the introduction and conclusion via LLM, with caching.

    Args:
        adapter: LLM adapter (AnthropicAdapter or OpenAIAdapter).
        outline: Full outline dict.
        section_texts: Dict mapping section ID to generated content.
        paths: Paths dict from get_paths().
        model: LLM model identifier.
        force: If True, regenerate even if cached.
        output_format: 'latex' or 'markdown'.

    Returns:
        Tuple of (intro_text, conclusion_text).
    """
    ext = ".tex" if output_format == "latex" else ".md"
    intro_cache = paths["cache_dir"] / f"intro{ext}"
    concl_cache = paths["cache_dir"] / f"conclusion{ext}"

    if intro_cache.exists() and concl_cache.exists() and not force:
        print("  Intro + conclusion cached, loading...")
        return intro_cache.read_text(), concl_cache.read_text()

    print(f"  Generating introduction + conclusion via {model}...")
    prompt = build_intro_prompt(outline, section_texts,
                                output_format=output_format)

    system = INTRO_SYSTEM_LATEX if output_format == "latex" else INTRO_SYSTEM_MD

    def call():
        return adapter.chat(system, prompt, model, MAX_TOKENS_INTRO)

    raw = with_retry(call)

    if '%%% CONCLUSION %%%' in raw:
        parts = raw.split('%%% CONCLUSION %%%')
        intro_text = parts[0].strip()
        concl_text = parts[1].strip()
    else:
        if output_format == "latex":
            match = re.search(r'(\\section\{Conclusion)', raw)
        else:
            match = re.search(r'(## Conclusion)', raw)
        if match:
            intro_text = raw[:match.start()].strip()
            concl_text = raw[match.start():].strip()
        else:
            intro_text = raw
            if output_format == "latex":
                concl_text = (
                    "\\section{Conclusion}\n"
                    "\\label{sec:conclusion}\n\n"
                    "% TODO: generate conclusion"
                )
            else:
                concl_text = "## Conclusion\n\n*TODO: generate conclusion*"

    intro_cache.write_text(intro_text)
    concl_cache.write_text(concl_text)
    print("  Intro + conclusion saved")

    return intro_text, concl_text


# ── Bibliography ─────────────────────────────────────────────────────


def make_cite_key(paper):
    """Generate a BibTeX cite key from a paper dict."""
    base_id = paper.get("base_id", paper.get("arxiv_id", "unknown"))
    base_id = re.sub(r'v\d+$', '', base_id)
    return "arxiv_" + base_id.replace('.', '_')


def escape_bibtex(s):
    """Escape special characters for BibTeX fields."""
    s = s.replace('&', r'\&')
    s = s.replace('%', r'\%')
    s = s.replace('#', r'\#')
    s = s.replace('_', r'\_')
    return s


def generate_bib(papers, output_path):
    """Write a .bib file from the list of paper dicts."""
    entries = []
    seen_keys = set()

    for paper in papers:
        key = make_cite_key(paper)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        published = paper.get("published", "")
        year = published[:4] if len(published) >= 4 else "2024"

        authors = " and ".join(paper.get("authors", ["Unknown"]))
        title = escape_bibtex(paper.get("title", "Untitled"))

        entry = (
            f"@article{{{key},\n"
            f"  author = {{{authors}}},\n"
            f"  title = {{{{{title}}}}},\n"
            f"  year = {{{year}}},\n"
            f"  journal = {{arXiv preprint arXiv:"
            f"{paper.get('base_id', paper.get('arxiv_id', ''))}}},\n"
            f"  url = {{{paper.get('pdf_url', '')}}}\n"
            f"}}"
        )
        entries.append(entry)

    output_path.write_text('\n\n'.join(entries) + '\n')
    print(f"  Bibliography: {len(entries)} entries -> {output_path}")


# ── Abstract and assembly ────────────────────────────────────────────


def generate_abstract(outline, config=None, output_format="latex"):
    """Generate a placeholder abstract from the outline guidance.

    If config has a survey_abstract_domain field, uses it for the abstract text.
    Otherwise falls back to a generic abstract.
    """
    guidance = outline.get("abstract_guidance", "")
    domain_text = ""
    if config is not None and getattr(config, "survey_abstract_domain", ""):
        domain_text = config.survey_abstract_domain.strip()

    if domain_text:
        abstract_body = (
            f"{domain_text} "
            "We synthesize findings from the literature, identify key research themes "
            "and their interconnections, and outline open problems and future directions. "
            "The survey is organized around research themes identified through a "
            "systematic, multi-level literature analysis."
        )
    else:
        abstract_body = (
            "This survey provides a comprehensive review of recent advances in the "
            "surveyed research area. "
            "We synthesize findings from the literature, identify key research themes "
            "and their interconnections, and outline open problems and future directions. "
            "The survey is organized around research themes identified through a "
            "systematic, multi-level literature analysis."
        )

    if output_format == "latex":
        return (
            "% Abstract generated from outline guidance\n"
            f"% Guidance: {guidance}\n"
            "\\begin{abstract}\n"
            f"{abstract_body}\n"
            "\\end{abstract}\n"
        )
    else:
        return (
            f"> **Abstract:** {abstract_body}\n"
        )


def build_cite_map(papers):
    """Build a mapping from cite keys to sequential reference numbers.

    Returns:
        Dict mapping cite_key -> int (1-based).
    """
    cite_map = {}
    for i, paper in enumerate(papers, 1):
        key = make_cite_key(paper)
        if key not in cite_map:
            cite_map[key] = i
    return cite_map


def generate_references_md(papers):
    """Generate a Markdown references section from the list of paper dicts."""
    lines = ["## References", ""]
    seen_keys = set()
    num = 0

    for paper in papers:
        key = make_cite_key(paper)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        num += 1

        authors = ", ".join(paper.get("authors", ["Unknown"]))
        title = paper.get("title", "Untitled")
        published = paper.get("published", "")
        year = published[:4] if len(published) >= 4 else "2024"
        url = paper.get("pdf_url", "")

        line = f"[{num}] {authors}. *{title}*. {year}."
        if url:
            line += f" [{url}]({url})"
        lines.append(line)
        lines.append("")

    return "\n".join(lines)


def assemble(outline, section_texts, intro_text, concl_text, data, paths,
             config=None, output_format="latex"):
    """Assemble all generated pieces into a final document.

    For LaTeX: produces a .tex file + .bib file.
    For Markdown: produces a single .md file with inline references.

    Args:
        outline: Outline dict with title and sections.
        section_texts: Dict mapping section ID to content.
        intro_text: Introduction content string.
        concl_text: Conclusion content string.
        data: Loaded INSTINCT data dict (needs 'papers' for references).
        paths: Paths dict from get_paths().
        config: Optional DomainConfig for abstract domain text.
        output_format: 'latex' or 'markdown'.
    """
    if output_format == "latex":
        generate_bib(data["papers"], paths["survey_bib"])

        parts = []
        parts.append(LATEX_PREAMBLE)
        title = escape_bibtex(outline["title"])
        parts.append(f"\\title{{{title}}}")
        parts.append("\\author{Generated by INSTINCT Pipeline}")
        parts.append("\\date{\\today}")
        parts.append("")
        parts.append("\\begin{document}")
        parts.append("\\maketitle")
        parts.append("")
        parts.append(generate_abstract(outline, config=config,
                                        output_format="latex"))
        parts.append("")
        parts.append("\\tableofcontents")
        parts.append("\\newpage")
        parts.append("")
        parts.append("% -- Introduction "
                     "------------------------------------------------------")
        parts.append(intro_text)
        parts.append("")

        for section in outline["sections"]:
            sid = section["id"]
            parts.append(f"% -- Section: {section['title']} "
                         "----------------------------------------------")
            parts.append(section_texts.get(sid,
                         f"% TODO: generate section {sid}"))
            parts.append("")

        parts.append("% -- Conclusion "
                     "--------------------------------------------------------")
        parts.append(concl_text)
        parts.append("")
        parts.append("\\bibliographystyle{plainnat}")
        parts.append("\\bibliography{survey}")
        parts.append("")
        parts.append("\\end{document}")

    else:
        # Markdown assembly
        parts = []
        parts.append(f"# {outline['title']}")
        parts.append("")
        parts.append("*Generated by INSTINCT Pipeline*")
        parts.append("")
        parts.append(generate_abstract(outline, config=config,
                                        output_format="markdown"))
        parts.append("")
        parts.append("---")
        parts.append("")
        parts.append(intro_text)
        parts.append("")

        for section in outline["sections"]:
            sid = section["id"]
            parts.append(section_texts.get(sid,
                         f"*TODO: generate section {sid}*"))
            parts.append("")

        parts.append(concl_text)
        parts.append("")
        parts.append("---")
        parts.append("")
        parts.append(generate_references_md(data["papers"]))

    content = '\n'.join(parts)
    paths["survey_out"].write_text(content)
    print(f"  Survey: {len(content)} bytes -> {paths['survey_out']}")


# ── Main entry point ─────────────────────────────────────────────────


def run_survey(base_dir, config=None, force=False, outline_only=False,
               section=None, model=None, output_format="markdown"):
    """Run the full survey generation pipeline.

    Args:
        base_dir: Path to the data directory containing INSTINCT outputs.
        config: Optional DomainConfig from kg.config. If provided and it has
                survey-specific fields (survey_section_system,
                survey_abstract_domain), those override defaults.
        force: If True, regenerate all stages even if cached.
        outline_only: If True, stop after generating the outline.
        section: If set, only (re)generate this specific section ID.
        model: LLM model identifier. Defaults to DEFAULT_MODEL.
        output_format: 'latex' or 'markdown'. Defaults to 'markdown'.

    Returns:
        The outline dict.
    """
    if model is None:
        model = DEFAULT_MODEL
    if output_format not in ("latex", "markdown"):
        raise ValueError(f"output_format must be 'latex' or 'markdown', "
                         f"got {output_format!r}")

    paths = get_paths(base_dir, output_format=output_format)
    paths["cache_dir"].mkdir(parents=True, exist_ok=True)

    # Apply domain config overrides if provided
    if output_format == "latex":
        section_system = SECTION_SYSTEM_LATEX
    else:
        section_system = SECTION_SYSTEM_MD
    if config is not None:
        if getattr(config, "survey_section_system", ""):
            section_system = config.survey_section_system

    # Create the adapter (use OpenAI if no Anthropic key available)
    try:
        if os.environ.get("ANTHROPIC_API_KEY"):
            adapter = AnthropicAdapter()
        else:
            adapter = OpenAIAdapter()
    except Exception:
        adapter = OpenAIAdapter()

    # Stage 1: Load data
    print(f"Stage 1: Loading INSTINCT data (format={output_format})...")
    data = load_data(paths)

    # Stage 2: Topological sort
    print("Stage 2: Topological sort of themes...")
    ordered_ids = topological_sort_themes(data["themes"], data["l2_edges"])

    # Stage 3: Generate outline (format-agnostic)
    print("Stage 3: Generating outline...")
    outline = generate_outline(adapter, ordered_ids, data["themes"],
                               data["l2_edges"], paths, model=model,
                               force=force)

    # Resolve any abbreviated theme IDs the LLM may have produced
    valid_theme_ids = {t["id"] for t in data["themes"]}
    resolve_theme_ids(outline, valid_theme_ids)

    if outline_only:
        print("Outline-only mode — stopping here.")
        return outline

    # Build cite map for Markdown numbered references
    cite_map = build_cite_map(data["papers"]) if output_format == "markdown" else None

    # Stage 4: Generate sections
    print("Stage 4: Generating sections...")
    theme_map = {t["id"]: t for t in data["themes"]}
    section_texts = {}
    ext = ".tex" if output_format == "latex" else ".md"

    for sec in outline["sections"]:
        if section is not None and sec["id"] != section:
            # Load from cache if available, skip generation
            cache_file = paths["cache_dir"] / f"section_{sec['id']}{ext}"
            if cache_file.exists():
                section_texts[sec["id"]] = cache_file.read_text()
            continue

        section_force = force or (section is not None and sec["id"] == section)
        section_texts[sec["id"]] = generate_section(
            adapter, sec, outline, data, theme_map, paths,
            model=model, force=section_force,
            output_format=output_format, cite_map=cite_map,
        )

    # Stage 5: Generate introduction + conclusion
    print("Stage 5: Generating introduction + conclusion...")
    intro_text, concl_text = generate_intro_conclusion(
        adapter, outline, section_texts, paths, model=model, force=force,
        output_format=output_format,
    )

    # Stage 6: Assemble
    print("Stage 6: Assembling final document...")
    assemble(outline, section_texts, intro_text, concl_text, data, paths,
             config=config, output_format=output_format)

    print("Done.")
    return outline
