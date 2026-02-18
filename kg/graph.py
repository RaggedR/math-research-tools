"""Merge extractions, build graph, and prepare visualization data."""

import time
from collections import defaultdict

from .config import MIN_DEGREE_FOR_VIZ, TYPE_COLORS
from .extract import normalize_name


def merge_extractions(all_extractions):
    """Merge per-paper extractions into a unified set of concepts and edges.

    Args:
        all_extractions: dict mapping paper_name -> extraction dict

    Returns:
        (concepts, edges) where concepts is a dict keyed by normalized name,
        and edges is a list of edge dicts.
    """
    concepts = {}
    edges = []

    for paper_name, extraction in all_extractions.items():
        for c in extraction.get("concepts", []):
            name = c.get("name", "").strip()
            if not name:
                continue
            norm = normalize_name(name)

            if norm not in concepts:
                concepts[norm] = {
                    "name": norm,
                    "display_name": c.get("name", name),
                    "type": c.get("type", "object"),
                    "description": c.get("description", ""),
                    "papers": [paper_name],
                }
            else:
                if paper_name not in concepts[norm]["papers"]:
                    concepts[norm]["papers"].append(paper_name)
                desc = c.get("description", "")
                if len(desc) > len(concepts[norm]["description"]):
                    concepts[norm]["description"] = desc

        for r in extraction.get("relationships", []):
            src = normalize_name(r.get("source", ""))
            tgt = normalize_name(r.get("target", ""))
            if src in concepts and tgt in concepts and src != tgt:
                edges.append({
                    "source": src,
                    "target": tgt,
                    "relation": r.get("relation", "related_to"),
                    "detail": r.get("detail", ""),
                    "paper": paper_name,
                })

    return concepts, edges


def build_graph(concepts, edges):
    """Deduplicate edges and build final graph structure.

    Returns a dict with "metadata", "concepts", and "edges" keys.
    """
    seen_edges = {}
    for e in edges:
        key = (e["source"], e["target"], e["relation"])
        if key not in seen_edges:
            seen_edges[key] = {
                "source": e["source"],
                "target": e["target"],
                "relation": e["relation"],
                "details": [e["detail"]] if e["detail"] else [],
                "papers": [e["paper"]],
            }
        else:
            if e["detail"] and e["detail"] not in seen_edges[key]["details"]:
                seen_edges[key]["details"].append(e["detail"])
            if e["paper"] not in seen_edges[key]["papers"]:
                seen_edges[key]["papers"].append(e["paper"])

    return {
        "metadata": {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_concepts": len(concepts),
            "total_edges": len(seen_edges),
            "total_papers": len(set(
                p for c in concepts.values() for p in c["papers"]
            )),
        },
        "concepts": list(concepts.values()),
        "edges": list(seen_edges.values()),
    }


def prepare_viz_data(graph, min_degree=MIN_DEGREE_FOR_VIZ):
    """Prepare graph data for D3.js visualization.

    Filters nodes by min_degree, falling back to lower thresholds if
    too few nodes would remain. Returns dict with "nodes" and "links".
    """
    degree = defaultdict(int)
    for e in graph["edges"]:
        degree[e["source"]] += 1
        degree[e["target"]] += 1

    keep = {c["name"] for c in graph["concepts"]
            if degree.get(c["name"], 0) >= min_degree}
    if len(keep) < 5:
        keep = {c["name"] for c in graph["concepts"]
                if degree.get(c["name"], 0) >= 1}
    if len(keep) < 5:
        keep = {c["name"] for c in graph["concepts"]}

    nodes = []
    for c in graph["concepts"]:
        if c["name"] not in keep:
            continue
        nodes.append({
            "id": c["name"],
            "label": c.get("display_name", c["name"]),
            "type": c.get("type", "object"),
            "papers": len(c["papers"]),
            "degree": degree.get(c["name"], 0),
            "description": c.get("description", ""),
            "color": TYPE_COLORS.get(c.get("type", ""), "#95A5A6"),
        })

    links = []
    for e in graph["edges"]:
        if e["source"] in keep and e["target"] in keep:
            links.append({
                "source": e["source"],
                "target": e["target"],
                "relation": e["relation"],
                "detail": e.get("details", [""])[0] if e.get("details") else "",
            })

    return {"nodes": nodes, "links": links}
