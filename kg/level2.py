"""Level 2 pipeline: themes, meta-summaries, and knowledge graph.

1. Index all Level 1 summaries into ChromaDB
2. Identify research themes via LLM analysis
3. Generate meta-summaries for each theme
4. Build Level 2 knowledge graph with theme connections
"""

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from .config import load_config, EMBEDDING_MODEL


# ── Configuration ─────────────────────────────────────────────────────

SUMMARY_MODEL = "gpt-4o-mini"
THEME_MODEL = "gpt-4o"
L1_COLLECTION = "level1_summaries"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
TOP_K_META = 25
TARGET_THEMES = 20


# ── Path helpers ──────────────────────────────────────────────────────

def get_paths(base_dir):
    """Return all path constants derived from base_dir."""
    d = Path(base_dir)
    return {
        "chroma_dir": d / "chroma_db",
        "compressed_dir": d / "compressed",
        "meta_dir": d / "meta-summaries",
        "graph_file": d / "knowledge_graph.json",
        "themes_file": d / "level2_themes.json",
        "l2_graph_file": d / "level2_knowledge_graph.json",
        "l2_html_file": d / "level2_knowledge_graph.html",
    }


# ── Step 1: Index Level 1 summaries ──────────────────────────────────

def index_summaries(openai_client, chroma_client, paths):
    """Index all Level 1 markdown summaries into ChromaDB."""
    print("\n" + "=" * 60)
    print("STEP 1: Indexing Level 1 summaries into ChromaDB")
    print("=" * 60)

    compressed_dir = paths["compressed_dir"]

    try:
        chroma_client.delete_collection(L1_COLLECTION)
        print("  Deleted existing collection.")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=L1_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    md_files = sorted(compressed_dir.glob("*.md"))
    print(f"  Found {len(md_files)} summary files")

    all_chunks = []
    all_ids = []
    all_metas = []

    for md_file in md_files:
        text = md_file.read_text()
        if len(text) < 50:
            continue

        concept_name = md_file.stem
        first_line = text.split("\n")[0]
        if first_line.startswith("# "):
            concept_name = first_line[2:].strip()

        chunks = chunk_markdown(text, concept_name, md_file.name)
        for i, chunk in enumerate(chunks):
            chunk_id = f"l1_{md_file.stem}_{i}"
            all_chunks.append(chunk["text"])
            all_ids.append(chunk_id)
            all_metas.append({
                "source": md_file.name,
                "concept": concept_name,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "section": chunk.get("section", ""),
            })

    print(f"  {len(all_chunks)} chunks from {len(md_files)} summaries")
    print(f"  Generating embeddings...")

    all_embeddings = []
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch = [t[:8000] for t in all_chunks[i:i + batch_size]]
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL, input=batch
        )
        all_embeddings.extend([e.embedding for e in response.data])

    print(f"  Storing in ChromaDB...")
    store_batch = 500
    for i in range(0, len(all_chunks), store_batch):
        j = min(i + store_batch, len(all_chunks))
        collection.add(
            ids=all_ids[i:j],
            documents=all_chunks[i:j],
            embeddings=all_embeddings[i:j],
            metadatas=all_metas[i:j],
        )

    print(f"  Done: {collection.count()} chunks indexed into '{L1_COLLECTION}'")
    return collection


def chunk_markdown(text, concept_name, filename):
    """Chunk a markdown summary, preserving section boundaries."""
    chunks = []
    current_section = "intro"
    current_text = f"[Concept: {concept_name}]\n"
    lines = text.split("\n")

    for line in lines:
        if line.startswith("## "):
            if len(current_text) > 100:
                chunks.append({"text": current_text.strip(), "section": current_section})
            current_section = line[3:].strip().lower()
            current_text = f"[Concept: {concept_name} — {current_section}]\n{line}\n"
        else:
            current_text += line + "\n"

            if len(current_text) >= CHUNK_SIZE:
                break_at = current_text.rfind("\n\n", CHUNK_OVERLAP, CHUNK_SIZE)
                if break_at < CHUNK_OVERLAP:
                    break_at = current_text.rfind(". ", CHUNK_OVERLAP, CHUNK_SIZE)
                if break_at < CHUNK_OVERLAP:
                    break_at = CHUNK_SIZE

                chunk = current_text[:break_at].strip()
                if len(chunk) > 50:
                    chunks.append({"text": chunk, "section": current_section})
                current_text = f"[Concept: {concept_name} — {current_section}]\n" + current_text[break_at - CHUNK_OVERLAP:]

    if len(current_text.strip()) > 50:
        chunks.append({"text": current_text.strip(), "section": current_section})

    return chunks


# ── Step 2: Identify research themes ─────────────────────────────────

def identify_themes(openai_client, paths, config=None):
    """Use LLM + knowledge graph structure to identify research themes."""
    print("\n" + "=" * 60)
    print("STEP 2: Identifying research themes")
    print("=" * 60)

    with open(paths["graph_file"]) as f:
        graph = json.load(f)

    degree = defaultdict(int)
    for e in graph["edges"]:
        degree[e["source"]] += 1
        degree[e["target"]] += 1

    concepts_info = []
    for c in graph["concepts"]:
        d = degree.get(c["name"], 0)
        concepts_info.append({
            "name": c.get("display_name", c["name"]),
            "type": c.get("type", "?"),
            "papers": len(c["papers"]),
            "degree": d,
            "description": c.get("description", "")[:100],
        })

    concepts_info.sort(key=lambda x: -(x["degree"] + x["papers"]))
    top_concepts = concepts_info[:100]

    domain_desc = ""
    if config:
        domain_desc = config.theme_domain_description.strip()

    prompt = f"""You are analyzing a research knowledge graph about {domain_desc or 'a research domain'}.

Here are the top 100 concepts by connectivity (name, type, # papers, degree, brief description):
{json.dumps(top_concepts, indent=2)}

The knowledge graph has {graph['metadata']['total_concepts']} total concepts and {graph['metadata']['total_edges']} edges across {graph['metadata']['total_papers']} papers.

TASK: Identify {TARGET_THEMES} HIGH-LEVEL RESEARCH THEMES that organize this research landscape.
Each theme should be a coherent research area, not just a single concept.

Return JSON with this schema:
{{
  "themes": [
    {{
      "id": "short-slug",
      "name": "Human-readable theme name",
      "description": "2-3 sentence description of what this research area covers",
      "key_concepts": ["list", "of", "5-10", "key", "concept", "names"],
      "search_queries": ["3-5 natural language queries to find relevant summaries in a vector DB"]
    }}
  ]
}}

Guidelines:
- Themes should cover the ENTIRE research landscape, not just the most connected concepts
- Include both broad themes and specific ones
- The search_queries should be diverse enough to pull in all relevant Level 1 summaries
- Themes may overlap — a concept can belong to multiple themes"""

    print(f"  Querying {THEME_MODEL} for ~{TARGET_THEMES} themes...")
    response = openai_client.chat.completions.create(
        model=THEME_MODEL,
        messages=[
            {"role": "system", "content": "You are a research analyst. Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=6000,
    )

    themes = json.loads(response.choices[0].message.content)

    with open(paths["themes_file"], 'w') as f:
        json.dump(themes, f, indent=2)

    print(f"  Identified {len(themes['themes'])} themes:")
    for t in themes["themes"]:
        print(f"    - {t['name']} ({len(t['key_concepts'])} key concepts)")

    return themes


# ── Step 3: Generate meta-summaries ──────────────────────────────────

def generate_meta_summaries(openai_client, chroma_client, themes, paths, config=None):
    """For each theme, query Level 1 DB and generate a meta-summary."""
    print("\n" + "=" * 60)
    print("STEP 3: Generating meta-summaries")
    print("=" * 60)

    meta_dir = paths["meta_dir"]
    meta_dir.mkdir(exist_ok=True)

    try:
        collection = chroma_client.get_collection(L1_COLLECTION)
    except Exception:
        print("  Error: Level 1 collection not found. Run --index-only first.")
        sys.exit(1)

    meta_system = "You are a research surveyor writing comprehensive area overviews."
    if config and config.meta_summary_system_prompt:
        meta_system = config.meta_summary_system_prompt.strip()

    for i, theme in enumerate(themes["themes"]):
        theme_id = theme["id"]
        output_file = meta_dir / f"{theme_id}.md"

        if output_file.exists() and output_file.stat().st_size > 100:
            print(f"  [{i+1}] SKIP {theme['name']} (already exists)")
            continue

        print(f"  [{i+1}/{len(themes['themes'])}] {theme['name']}...")

        all_passages = []
        seen_sources = set()

        for query in theme.get("search_queries", [theme["name"]]):
            response = openai_client.embeddings.create(
                model=EMBEDDING_MODEL, input=[query[:8000]]
            )
            query_embedding = response.data[0].embedding

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=TOP_K_META,
                include=["documents", "metadatas", "distances"],
            )

            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                source = meta["source"]
                if source not in seen_sources:
                    seen_sources.add(source)
                    similarity = 1 - dist
                    concept = meta.get("concept", source)
                    all_passages.append(
                        f"[From: {concept} ({source}), similarity: {similarity:.3f}]\n{doc.strip()}"
                    )

        passages_text = "\n\n---\n\n".join(all_passages[:40])

        prompt = f"""You are writing a high-level research area summary for a technical reference system.

THEME: **{theme['name']}**
DESCRIPTION: {theme['description']}
KEY CONCEPTS: {', '.join(theme['key_concepts'])}

Below are passages retrieved from Level 1 concept summaries.
These are SUMMARIES OF SUMMARIES — use them to write a comprehensive overview.

WRITE A 2-3 PAGE META-SUMMARY that covers:

## Overview
What this research area is about, its historical development, and why it matters.

## Core Definitions and Methods
The main technical objects and algorithms studied in this area. Give precise definitions.

## Major Results
The most important findings, with attribution and dates. Include quantitative results where available.

## Open Problems and Future Directions
Active challenges and open questions in this area.

## Connections to Other Areas
How this theme connects to other research themes in the landscape.

## Key References
The most important papers in this area (cite specific authors and years).

## Concept Map
List the Level 1 concepts that fall under this theme, organized into sub-areas.

IMPORTANT:
- This is a META-summary: give a bird's-eye view, not repeat details.
- Focus on the BIG PICTURE: main threads, how they connect, state of the art.
- Include cross-references to other themes where relevant.

RETRIEVED LEVEL 1 PASSAGES:
{passages_text}"""

        response = openai_client.chat.completions.create(
            model=SUMMARY_MODEL,
            messages=[
                {"role": "system", "content": meta_system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=5000,
            temperature=0.2,
        )

        summary = response.choices[0].message.content

        with open(output_file, 'w') as f:
            f.write(summary)

        print(f"    -> {output_file.name} ({len(summary)} chars, {len(seen_sources)} sources)")


# ── Step 4: Build Level 2 knowledge graph ────────────────────────────

def build_level2_graph(themes, openai_client, paths, config=None):
    """Build a knowledge graph over the research themes."""
    print("\n" + "=" * 60)
    print("STEP 4: Building Level 2 knowledge graph")
    print("=" * 60)

    meta_dir = paths["meta_dir"]

    nodes = []
    for t in themes["themes"]:
        meta_file = meta_dir / f"{t['id']}.md"
        size = meta_file.stat().st_size if meta_file.exists() else 0
        nodes.append({
            "name": t["id"],
            "display_name": t["name"],
            "type": "research_area",
            "description": t["description"],
            "key_concepts": t["key_concepts"],
            "summary_size": size,
        })

    theme_list = []
    for t in themes["themes"]:
        theme_list.append({
            "id": t["id"],
            "name": t["name"],
            "description": t["description"],
            "key_concepts": t["key_concepts"],
        })

    domain_desc = ""
    if config:
        domain_desc = config.l2_graph_domain_description.strip()

    prompt = f"""You are analyzing connections between research areas in the field of {domain_desc or 'a research domain'}.

Here are {len(theme_list)} research themes:

{json.dumps(theme_list, indent=2)}

TASK: Identify ALL meaningful connections between these themes.

Return JSON:
{{
  "edges": [
    {{
      "source": "theme-id-1",
      "target": "theme-id-2",
      "relation": "short label",
      "description": "1-2 sentence explanation of the connection",
      "bridging_concepts": ["concepts that connect these two areas"],
      "strength": "strong | moderate | weak"
    }}
  ]
}}

Guidelines:
- Include ALL real connections, not just obvious ones
- "strong" = deeply intertwined
- "moderate" = significant shared ideas
- "weak" = occasional overlap
- Use the actual theme IDs from the list above"""

    print("  Querying GPT-4o for theme connections...")
    response = openai_client.chat.completions.create(
        model=THEME_MODEL,
        messages=[
            {"role": "system", "content": "You are a research analyst. Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=6000,
    )

    edge_data = json.loads(response.choices[0].message.content)
    edges = []
    strength_weight = {"strong": 3, "moderate": 2, "weak": 1}

    for e in edge_data.get("edges", []):
        edges.append({
            "source": e["source"],
            "target": e["target"],
            "relation": e.get("relation", "related_to"),
            "description": e.get("description", ""),
            "bridging_concepts": e.get("bridging_concepts", []),
            "strength": e.get("strength", "moderate"),
            "weight": strength_weight.get(e.get("strength", "moderate"), 2),
        })

    graph = {
        "metadata": {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "level": 2,
            "total_themes": len(nodes),
            "total_edges": len(edges),
            "description": "Level 2 knowledge graph — research themes and their connections",
        },
        "themes": nodes,
        "edges": edges,
    }

    with open(paths["l2_graph_file"], 'w') as f:
        json.dump(graph, f, indent=2)

    strong = sum(1 for e in edges if e["strength"] == "strong")
    moderate = sum(1 for e in edges if e["strength"] == "moderate")
    weak = sum(1 for e in edges if e["strength"] == "weak")
    print(f"  {len(nodes)} themes, {len(edges)} edges ({strong} strong, {moderate} moderate, {weak} weak)")
    print(f"  Saved to {paths['l2_graph_file']}")

    for e in sorted(edges, key=lambda x: -x["weight"]):
        print(f"    {e['source']:40s} ──{e['strength']:>8s}── {e['target']}")

    generate_l2_html(graph, paths, config)

    return graph


def generate_l2_html(graph, paths, config=None):
    """Generate an interactive HTML visualization for the Level 2 graph."""
    domain_name = config.name if config else "Research"

    nodes_data = []
    for t in graph["themes"]:
        nodes_data.append({
            "id": t["name"],
            "label": t["display_name"],
            "description": t["description"],
            "concepts": len(t["key_concepts"]),
            "size": t.get("summary_size", 1000),
        })

    links_data = []
    for e in graph["edges"]:
        links_data.append({
            "source": e["source"],
            "target": e["target"],
            "weight": e.get("weight", 1),
            "strength": e.get("strength", "moderate"),
            "description": e.get("description", ""),
            "bridging": ", ".join(e.get("bridging_concepts", [])[:5]),
        })

    data = json.dumps({"nodes": nodes_data, "links": links_data})

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>INSTINCT Level 2 — Research Themes ({domain_name})</title>
<style>
  body {{ margin: 0; overflow: hidden; background: #0d1117; font-family: -apple-system, sans-serif; }}
  svg {{ width: 100vw; height: 100vh; }}
  .link {{ stroke-opacity: 0.6; }}
  .link.strong {{ stroke: #f0883e; }}
  .link.moderate {{ stroke: #58a6ff; }}
  .link.weak {{ stroke: #484f58; }}
  .node circle {{ stroke: #fff; stroke-width: 2px; cursor: pointer; fill: #238636; }}
  .node text {{ fill: #c9d1d9; font-size: 12px; font-weight: bold; pointer-events: none; }}
  #tooltip {{
    position: absolute; background: rgba(13,17,23,0.95); color: #c9d1d9;
    padding: 12px 16px; border-radius: 8px; font-size: 13px;
    pointer-events: none; display: none; max-width: 400px;
    border: 1px solid #30363d;
  }}
  #tooltip .title {{ font-weight: bold; font-size: 16px; color: #58a6ff; margin-bottom: 6px; }}
  #title {{
    position: absolute; top: 16px; left: 16px; color: #58a6ff;
    font-size: 18px; font-weight: bold;
    background: rgba(13,17,23,0.8); padding: 10px 16px; border-radius: 8px;
  }}
  #title small {{ color: #8b949e; font-size: 12px; font-weight: normal; display: block; margin-top: 4px; }}
  #legend {{
    position: absolute; bottom: 16px; left: 16px; color: #8b949e;
    background: rgba(13,17,23,0.8); padding: 10px 16px; border-radius: 8px; font-size: 12px;
  }}
  #legend span {{ display: inline-block; width: 20px; height: 3px; vertical-align: middle; margin-right: 6px; }}
</style>
</head>
<body>
<div id="title">INSTINCT Level 2 — {domain_name} Research Themes<small>{len(nodes_data)} themes, {len(links_data)} connections</small></div>
<div id="tooltip"></div>
<div id="legend">
  <span style="background:#f0883e"></span>Strong &nbsp;
  <span style="background:#58a6ff"></span>Moderate &nbsp;
  <span style="background:#484f58"></span>Weak
</div>
<svg></svg>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const data = {data};
const width = window.innerWidth, height = window.innerHeight;
const svg = d3.select("svg").attr("viewBox", [0, 0, width, height]);
const g = svg.append("g");
svg.call(d3.zoom().scaleExtent([0.3, 5]).on("zoom", (e) => g.attr("transform", e.transform)));
const simulation = d3.forceSimulation(data.nodes)
  .force("link", d3.forceLink(data.links).id(d => d.id).distance(d => 200 / (d.weight || 1)))
  .force("charge", d3.forceManyBody().strength(-500))
  .force("center", d3.forceCenter(width / 2, height / 2))
  .force("collision", d3.forceCollide().radius(60));
const link = g.append("g").selectAll("line").data(data.links).join("line")
  .attr("class", d => "link " + d.strength).attr("stroke-width", d => Math.max(1.5, d.weight * 2));
link.on("mouseover", (e, d) => {{
  tooltip.style("display","block")
    .html(`<div class="title">${{d.description}}</div><div style="margin-top:6px;color:#8b949e">Strength: ${{d.strength}}<br>Bridging: ${{d.bridging}}</div>`);
}}).on("mousemove", (e) => {{
  tooltip.style("left",(e.pageX+15)+"px").style("top",(e.pageY-10)+"px");
}}).on("mouseout", () => tooltip.style("display","none"));
const node = g.append("g").selectAll("g").data(data.nodes).join("g").attr("class","node")
  .call(d3.drag().on("start",ds).on("drag",dd).on("end",de));
node.append("circle").attr("r", d => 15 + d.concepts * 1.5);
node.append("text").text(d => d.label).attr("x", d => 18 + d.concepts * 1.5).attr("y", 4);
const tooltip = d3.select("#tooltip");
node.on("mouseover", (e, d) => {{
  tooltip.style("display","block")
    .html(`<div class="title">${{d.label}}</div><div class="desc">${{d.description}}</div>
           <div style="margin-top:8px;color:#8b949e">${{d.concepts}} key concepts</div>`);
}}).on("mousemove", (e) => {{
  tooltip.style("left",(e.pageX+15)+"px").style("top",(e.pageY-10)+"px");
}}).on("mouseout", () => tooltip.style("display","none"));
simulation.on("tick", () => {{
  link.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y).attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
  node.attr("transform",d=>`translate(${{d.x}},${{d.y}})`);
}});
function ds(e){{if(!e.active)simulation.alphaTarget(0.3).restart();e.subject.fx=e.subject.x;e.subject.fy=e.subject.y;}}
function dd(e){{e.subject.fx=e.x;e.subject.fy=e.y;}}
function de(e){{if(!e.active)simulation.alphaTarget(0);e.subject.fx=null;e.subject.fy=null;}}
</script>
</body>
</html>"""

    with open(paths["l2_html_file"], 'w') as f:
        f.write(html)
    print(f"  Visualization saved to {paths['l2_html_file']}")


# ── Main pipeline ─────────────────────────────────────────────────────

def run_level2(base_dir, config=None, index_only=False, themes_only=False, meta_only=False):
    """Run the full Level 2 pipeline."""
    import chromadb
    from openai import OpenAI

    base_dir = Path(base_dir)
    paths = get_paths(base_dir)

    if config is None:
        config = load_config(data_dir=base_dir)

    openai_client = OpenAI()
    chroma_client = chromadb.PersistentClient(path=str(paths["chroma_dir"]))

    full = not (index_only or themes_only or meta_only)

    if full or index_only:
        index_summaries(openai_client, chroma_client, paths)
        if index_only:
            return

    if full or themes_only:
        themes = identify_themes(openai_client, paths, config)
        if themes_only:
            return
    else:
        if not paths["themes_file"].exists():
            print("Error: no themes file. Run without --meta-only first.")
            sys.exit(1)
        with open(paths["themes_file"]) as f:
            themes = json.load(f)

    if full or meta_only:
        generate_meta_summaries(openai_client, chroma_client, themes, paths, config)

    build_level2_graph(themes, openai_client, paths, config)

    print("\n" + "=" * 60)
    print("LEVEL 2 COMPLETE")
    print("=" * 60)
    print(f"  Meta-summaries: {paths['meta_dir']}/")
    print(f"  Knowledge graph: {paths['l2_graph_file']}")
    print(f"  Visualization: {paths['l2_html_file']}")
    print(f"  Themes: {paths['themes_file']}")
