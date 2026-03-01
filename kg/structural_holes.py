"""Citation-aware structural hole detection.

Detects gaps between thematic clusters by overlaying citation network
analysis on a concept knowledge graph. Finds bridge papers that could
connect intellectual silos.
"""

import json
import logging
import os
import re
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx
import requests

log = logging.getLogger(__name__)

# ── Step 1: Extract paper metadata ────────────────────────────────────

ARXIV_MODERN = re.compile(r"(\d{4}\.\d{4,5})")
ARXIV_OLD = re.compile(r"(\d{7})")


def extract_arxiv_id(filename):
    """Extract arXiv ID from a PDF filename."""
    m = ARXIV_MODERN.search(filename)
    if m:
        return m.group(1)
    m = ARXIV_OLD.search(filename)
    if m:
        return m.group(1)
    return None


def scan_papers(base_dir):
    """Scan subdirectories for PDFs and extract metadata."""
    papers = {}
    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith(".") or subdir.name in (
            "compressed", "chroma_db", "meta-summaries", "__pycache__",
        ):
            continue
        for pdf in sorted(subdir.glob("*.pdf")):
            rel = f"{subdir.name}/{pdf.name}"
            papers[rel] = {
                "arxiv_id": extract_arxiv_id(pdf.name),
                "cluster": subdir.name,
                "filename": pdf.name,
            }
    return papers


def load_knowledge_graph(base_dir):
    """Load knowledge_graph.json."""
    kg_path = base_dir / "knowledge_graph.json"
    if not kg_path.exists():
        log.error("knowledge_graph.json not found in %s", base_dir)
        raise FileNotFoundError(f"knowledge_graph.json not found in {base_dir}")
    with open(kg_path) as f:
        return json.load(f)


def build_paper_concepts(kg):
    """Map each paper -> set of concept names."""
    paper_concepts = defaultdict(set)
    for concept in kg["concepts"]:
        for paper in concept["papers"]:
            paper_concepts[paper].add(concept["name"])
    return dict(paper_concepts)


# ── Step 2: Fetch citations ──────────────────────────────────────────

S2_BASE = "https://api.semanticscholar.org/graph/v1/paper"
S2_FIELDS = "title,year,references.paperId,references.externalIds,references.title,references.year"
RATE_LIMIT_SLEEP = 3


def load_citation_cache(cache_path):
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def save_citation_cache(cache, cache_path):
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def fetch_citations(papers, cache_path):
    """Fetch citation data from Semantic Scholar."""
    cache = load_citation_cache(cache_path)
    arxiv_papers = {k: v for k, v in papers.items() if v["arxiv_id"]}
    to_fetch = [k for k in arxiv_papers if k not in cache]

    if not to_fetch:
        log.info("All %d papers already cached", len(arxiv_papers))
        return cache

    log.info("Fetching %d papers from Semantic Scholar (%d cached)...",
             len(to_fetch), len(cache))

    for i, paper_key in enumerate(to_fetch):
        arxiv_id = arxiv_papers[paper_key]["arxiv_id"]
        url = f"{S2_BASE}/arXiv:{arxiv_id}?fields={S2_FIELDS}"
        backoff = RATE_LIMIT_SLEEP

        for attempt in range(4):
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    cache[paper_key] = {
                        "arxiv_id": arxiv_id,
                        "s2_title": data.get("title"),
                        "s2_year": data.get("year"),
                        "references": data.get("references", []),
                    }
                    save_citation_cache(cache, cache_path)
                    log.info("  [%d/%d] %s — %d refs",
                             i + 1, len(to_fetch), arxiv_id,
                             len(data.get("references", [])))
                    break
                elif resp.status_code == 404:
                    cache[paper_key] = {"arxiv_id": arxiv_id, "not_found": True}
                    save_citation_cache(cache, cache_path)
                    break
                elif resp.status_code == 429:
                    log.warning("  Rate limited, backing off %ds...", backoff)
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    break
            except requests.RequestException as e:
                log.warning("  Error: %s", e)
                if attempt < 3:
                    time.sleep(backoff)
                    backoff *= 2
                break

        time.sleep(RATE_LIMIT_SLEEP)

    return cache


# ── Step 3: Build citation graph ─────────────────────────────────────

def build_citation_graph(papers, citation_cache):
    """Build a directed citation graph."""
    G = nx.DiGraph()

    for paper_key, meta in papers.items():
        G.add_node(paper_key, internal=True, cluster=meta["cluster"],
                    arxiv_id=meta.get("arxiv_id"))

    for paper_key, cached in citation_cache.items():
        if cached.get("not_found"):
            continue
        for ref in cached.get("references", []):
            ref_id = ref.get("paperId")
            if not ref_id:
                continue

            ref_arxiv = None
            ext_ids = ref.get("externalIds") or {}
            if isinstance(ext_ids, dict):
                ref_arxiv = ext_ids.get("ArXiv")

            internal_target = None
            if ref_arxiv:
                for pk, pm in papers.items():
                    if pm.get("arxiv_id") == ref_arxiv:
                        internal_target = pk
                        break

            target = internal_target or f"ext:{ref_id}"
            if not G.has_node(target):
                G.add_node(target, internal=False,
                           title=ref.get("title"),
                           year=ref.get("year"),
                           s2_id=ref_id,
                           arxiv_id=ref_arxiv)

            G.add_edge(paper_key, target)

    return G


def compute_graph_stats(G):
    internal = [n for n, d in G.nodes(data=True) if d.get("internal")]
    external = [n for n, d in G.nodes(data=True) if not d.get("internal")]
    int_int = sum(1 for u, v in G.edges()
                  if G.nodes[u].get("internal") and G.nodes[v].get("internal"))
    int_ext = sum(1 for u, v in G.edges()
                  if G.nodes[u].get("internal") and not G.nodes[v].get("internal"))
    return {
        "total_nodes": G.number_of_nodes(),
        "internal_nodes": len(internal),
        "external_nodes": len(external),
        "total_edges": G.number_of_edges(),
        "int_int_edges": int_int,
        "int_ext_edges": int_ext,
    }


# ── Step 4: Cluster papers ───────────────────────────────────────────

def build_clusters(papers, paper_concepts):
    """Group papers by subdirectory and compute concept sets."""
    clusters = defaultdict(lambda: {"papers": [], "concepts": set()})
    for paper_key, meta in papers.items():
        cluster_name = meta["cluster"]
        clusters[cluster_name]["papers"].append(paper_key)
        concepts = paper_concepts.get(paper_key, set())
        clusters[cluster_name]["concepts"].update(concepts)
    return dict(clusters)


def generate_cluster_descriptions(clusters, domain="research"):
    """Generate cluster descriptions using GPT-4o-mini."""
    descriptions = {}
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        for name, data in clusters.items():
            top_concepts = sorted(data["concepts"])[:10]
            descriptions[name] = f"Papers about: {', '.join(top_concepts)}"
        return descriptions

    from openai import OpenAI
    client = OpenAI()

    for name, data in clusters.items():
        top_concepts = sorted(data["concepts"])[:20]
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": (
                        f"Given a cluster of {len(data['papers'])} {domain} papers "
                        f"in a directory called '{name}' with these key concepts:\n"
                        f"{', '.join(top_concepts)}\n\n"
                        "Write ONE sentence (max 25 words) describing the cluster's theme."
                    )
                }],
                max_tokens=80,
                temperature=0.3,
            )
            descriptions[name] = resp.choices[0].message.content.strip()
        except Exception as e:
            descriptions[name] = f"Papers about: {', '.join(top_concepts[:5])}"

    return descriptions


# ── Step 5: Detect structural holes ──────────────────────────────────

def jaccard(set_a, set_b):
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def compute_citation_density(G, papers_a, papers_b):
    """Fraction of possible cross-cluster citation edges that exist."""
    max_edges = len(papers_a) * len(papers_b) * 2
    if max_edges == 0:
        return 0.0
    cross = 0
    set_a, set_b = set(papers_a), set(papers_b)
    for u, v in G.edges():
        if (u in set_a and v in set_b) or (u in set_b and v in set_a):
            cross += 1
    return cross / max_edges


def detect_structural_holes(G, clusters):
    """Detect structural holes between all cluster pairs."""
    holes = []
    cluster_names = sorted(clusters.keys())

    for a, b in combinations(cluster_names, 2):
        concept_sim = jaccard(clusters[a]["concepts"], clusters[b]["concepts"])
        citation_density = compute_citation_density(
            G, clusters[a]["papers"], clusters[b]["papers"])

        hole_score = concept_sim / (citation_density + 0.01)

        shared = clusters[a]["concepts"] & clusters[b]["concepts"]
        unique_a = clusters[a]["concepts"] - clusters[b]["concepts"]
        unique_b = clusters[b]["concepts"] - clusters[a]["concepts"]

        holes.append({
            "cluster_a": a,
            "cluster_b": b,
            "concept_similarity": round(concept_sim, 4),
            "citation_density": round(citation_density, 4),
            "hole_score": round(hole_score, 4),
            "shared_concepts": sorted(shared),
            "unique_a_sample": sorted(unique_a)[:5],
            "unique_b_sample": sorted(unique_b)[:5],
            "papers_a": len(clusters[a]["papers"]),
            "papers_b": len(clusters[b]["papers"]),
        })

    holes.sort(key=lambda h: h["hole_score"], reverse=True)
    return holes


# ── Step 6: Find bridge papers ───────────────────────────────────────

def find_bridge_papers(G, clusters, holes, top_k=10):
    """For each structural hole, find external papers cited by both clusters."""
    bridges = {}

    for hole in holes[:top_k]:
        a, b = hole["cluster_a"], hole["cluster_b"]
        set_a = set(clusters[a]["papers"])
        set_b = set(clusters[b]["papers"])

        ext_cites_a = defaultdict(int)
        ext_cites_b = defaultdict(int)

        for u, v in G.edges():
            if not G.nodes[v].get("internal"):
                if u in set_a:
                    ext_cites_a[v] += 1
                elif u in set_b:
                    ext_cites_b[v] += 1

        shared_refs = set(ext_cites_a.keys()) & set(ext_cites_b.keys())
        bridge_list = []
        for ref in shared_refs:
            score = ext_cites_a[ref] * ext_cites_b[ref]
            node_data = G.nodes[ref]
            bridge_list.append({
                "node_id": ref,
                "title": node_data.get("title", "Unknown"),
                "year": node_data.get("year"),
                "arxiv_id": node_data.get("arxiv_id"),
                "s2_id": node_data.get("s2_id"),
                "citations_from_a": ext_cites_a[ref],
                "citations_from_b": ext_cites_b[ref],
                "bridge_score": score,
            })

        bridge_list.sort(key=lambda b: b["bridge_score"], reverse=True)

        internal_bridges = []
        for u, v in G.edges():
            if u in set_a and v in set_b:
                internal_bridges.append({"from": u, "to": v, "direction": f"{a}->{b}"})
            elif u in set_b and v in set_a:
                internal_bridges.append({"from": u, "to": v, "direction": f"{b}->{a}"})

        bridges[(a, b)] = {
            "external_bridges": bridge_list[:20],
            "internal_cross_citations": internal_bridges,
        }

    return bridges


# ── Step 7: Generate search queries ──────────────────────────────────

def generate_search_queries(holes, bridges, cluster_descriptions,
                            domain="research", top_k=10):
    """Generate targeted search queries for each top structural hole."""
    api_key = os.environ.get("OPENAI_API_KEY")
    queries = {}

    for hole in holes[:top_k]:
        a, b = hole["cluster_a"], hole["cluster_b"]
        key = (a, b)

        bridge_data = bridges.get(key, {})
        top_bridges = bridge_data.get("external_bridges", [])[:5]
        bridge_titles = [bp["title"] for bp in top_bridges if bp.get("title")]

        if not api_key:
            shared = hole["shared_concepts"][:5]
            unique_a = hole["unique_a_sample"][:3]
            unique_b = hole["unique_b_sample"][:3]
            queries[key] = {
                "queries": [
                    f"{' '.join(shared[:3])} {' '.join(unique_a[:2])}",
                    f"{' '.join(shared[:3])} {' '.join(unique_b[:2])}",
                    f"{' '.join(unique_a[:2])} {' '.join(unique_b[:2])}",
                ],
                "reasoning": "Template-generated (no OpenAI API key)",
            }
            continue

        from openai import OpenAI
        client = OpenAI()

        prompt = (
            f"I'm studying two clusters of {domain} papers:\n\n"
            f"Cluster A ({a}): {cluster_descriptions.get(a, 'N/A')}\n"
            f"  Key concepts unique to A: {', '.join(hole['unique_a_sample'])}\n\n"
            f"Cluster B ({b}): {cluster_descriptions.get(b, 'N/A')}\n"
            f"  Key concepts unique to B: {', '.join(hole['unique_b_sample'])}\n\n"
            f"Shared concepts: {', '.join(hole['shared_concepts'][:10])}\n\n"
        )
        if bridge_titles:
            prompt += f"Known bridge papers: {'; '.join(bridge_titles[:3])}\n\n"
        prompt += (
            "Generate 3-5 targeted search queries for Semantic Scholar/arXiv "
            "that would find papers connecting these two areas. Return as a "
            "JSON object with keys 'queries' (list of strings) and 'reasoning' "
            "(1 sentence)."
        )

        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.5,
                response_format={"type": "json_object"},
            )
            result = json.loads(resp.choices[0].message.content)
            queries[key] = result
        except Exception as e:
            shared = hole["shared_concepts"][:5]
            queries[key] = {
                "queries": [f"{' '.join(shared[:3])}"],
                "reasoning": f"Fallback (API error: {e})",
            }

    return queries


# ── Step 8: Generate report ──────────────────────────────────────────

def generate_report(holes, bridges, cluster_descriptions, search_queries,
                    graph_stats, output_dir, top_k=15):
    """Generate JSON and Markdown reports."""
    json_report = {
        "graph_stats": graph_stats,
        "cluster_descriptions": cluster_descriptions,
        "structural_holes": holes[:top_k],
        "bridges": {
            f"{a}↔{b}": v for (a, b), v in bridges.items()
        },
        "search_queries": {
            f"{a}↔{b}": v for (a, b), v in search_queries.items()
        },
    }
    json_path = output_dir / "structural_holes_report.json"
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2)

    md_lines = [
        "# Structural Holes Report",
        "",
        f"*Generated: {time.strftime('%Y-%m-%d %H:%M')}*",
        "",
        "## Citation Graph Statistics",
        "",
        f"- **Internal papers**: {graph_stats['internal_nodes']}",
        f"- **External references**: {graph_stats['external_nodes']}",
        f"- **Total edges**: {graph_stats['total_edges']}",
        f"- **Internal→Internal**: {graph_stats['int_int_edges']}",
        f"- **Internal→External**: {graph_stats['int_ext_edges']}",
        "",
        "## Clusters",
        "",
    ]

    for name, desc in sorted(cluster_descriptions.items()):
        md_lines.append(f"- **{name}**: {desc}")
    md_lines.append("")

    md_lines.extend([
        "## Structural Holes (ranked by hole score)",
        "",
        "A high hole score means concepts overlap but papers don't cite each other.",
        "",
    ])

    for i, hole in enumerate(holes[:top_k], 1):
        a, b = hole["cluster_a"], hole["cluster_b"]
        key = (a, b)
        md_lines.extend([
            f"### {i}. {a} ↔ {b}",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Hole score | **{hole['hole_score']:.2f}** |",
            f"| Concept similarity (Jaccard) | {hole['concept_similarity']:.4f} |",
            f"| Citation density | {hole['citation_density']:.4f} |",
            f"| Shared concepts | {len(hole['shared_concepts'])} |",
            "",
        ])

        if hole["shared_concepts"]:
            md_lines.append(f"**Shared concepts**: {', '.join(hole['shared_concepts'][:10])}")
            md_lines.append("")

        bridge_data = bridges.get(key, {})
        ext_bridges = bridge_data.get("external_bridges", [])
        int_cross = bridge_data.get("internal_cross_citations", [])

        if ext_bridges:
            md_lines.append("**Bridge papers** (external papers cited by both clusters):")
            md_lines.append("")
            for bp in ext_bridges[:5]:
                title = bp.get("title", "Unknown")
                year = bp.get("year", "?")
                score = bp["bridge_score"]
                arxiv = bp.get("arxiv_id")
                link = f" ([arXiv:{arxiv}](https://arxiv.org/abs/{arxiv}))" if arxiv else ""
                md_lines.append(
                    f"- **{title}** ({year}) — bridge score: {score}, "
                    f"cited by {bp['citations_from_a']} in {a} + "
                    f"{bp['citations_from_b']} in {b}{link}")
            md_lines.append("")

        if int_cross:
            md_lines.append(f"**Internal cross-citations**: {len(int_cross)} edges")
            md_lines.append("")

        query_data = search_queries.get(key, {})
        if query_data.get("queries"):
            md_lines.append("**Suggested search queries**:")
            md_lines.append("")
            for q in query_data["queries"]:
                md_lines.append(f"- `{q}`")
            if query_data.get("reasoning"):
                md_lines.append(f"\n*Reasoning: {query_data['reasoning']}*")
            md_lines.append("")

        md_lines.append("---")
        md_lines.append("")

    md_path = output_dir / "structural_holes.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))


# ── Main pipeline ─────────────────────────────────────────────────────

def run_structural_holes(base_dir, config=None, skip_fetch=False, top_k=15):
    """Run the full structural holes detection pipeline."""
    import sys

    base_dir = Path(base_dir)

    if config is None:
        from .config import load_config
        config = load_config(data_dir=base_dir)

    domain = config.cluster_description_domain if config else "research"

    log.info("Step 1: Scanning papers...")
    papers = scan_papers(base_dir)
    kg = load_knowledge_graph(base_dir)
    paper_concepts = build_paper_concepts(kg)

    arxiv_count = sum(1 for p in papers.values() if p["arxiv_id"])
    log.info("  Found %d papers (%d with arXiv IDs)", len(papers), arxiv_count)

    cache_path = base_dir / "citation_cache.json"
    if skip_fetch:
        log.info("Step 2: Skipping fetch, loading cache...")
        citation_cache = load_citation_cache(cache_path)
    else:
        log.info("Step 2: Fetching citations from Semantic Scholar...")
        citation_cache = fetch_citations(papers, cache_path)

    log.info("Step 3: Building citation graph...")
    G = build_citation_graph(papers, citation_cache)
    graph_stats = compute_graph_stats(G)

    log.info("Step 4: Clustering papers by subdirectory...")
    clusters = build_clusters(papers, paper_concepts)
    cluster_descriptions = generate_cluster_descriptions(clusters, domain)

    log.info("Step 5: Detecting structural holes...")
    holes = detect_structural_holes(G, clusters)

    log.info("Step 6: Finding bridge papers...")
    bridges = find_bridge_papers(G, clusters, holes, top_k=top_k)

    log.info("Step 7: Generating search queries...")
    search_queries = generate_search_queries(
        holes, bridges, cluster_descriptions, domain=domain, top_k=top_k)

    log.info("Step 8: Generating reports...")
    generate_report(holes, bridges, cluster_descriptions, search_queries,
                    graph_stats, base_dir, top_k=top_k)

    print("\n" + "=" * 60)
    print("STRUCTURAL HOLES SUMMARY")
    print("=" * 60)
    for i, hole in enumerate(holes[:5], 1):
        a, b = hole["cluster_a"], hole["cluster_b"]
        key = (a, b)
        bridge_count = len(bridges.get(key, {}).get("external_bridges", []))
        print(f"\n{i}. {a} ↔ {b}")
        print(f"   Hole score: {hole['hole_score']:.2f}")
        print(f"   Shared concepts: {len(hole['shared_concepts'])}")
        print(f"   Bridge papers found: {bridge_count}")

    print(f"\nFull report: {base_dir / 'structural_holes.md'}")
