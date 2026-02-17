#!/usr/bin/env python3
"""
build_knowledge_graph.py — Ingest PDFs and build a knowledge graph.

Given a directory with papers/*.pdf (e.g. from /lit-review), this script:
1. Ingests PDFs into ChromaDB (if not already done)
2. Reads chunks from ChromaDB
3. Extracts concepts/relationships via GPT-4o-mini
4. Generates an interactive D3.js HTML visualization

Usage:
    python3 build_knowledge_graph.py --dir <path>           # Full pipeline
    python3 build_knowledge_graph.py --dir <path> --resume  # Resume interrupted build
    python3 build_knowledge_graph.py --dir <path> --viz-only # Regenerate HTML only
"""

import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import fitz  # PyMuPDF
import chromadb
from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
COLLECTION_NAMES = ["lit_review", "math_papers"]  # try both when reading
INGEST_COLLECTION = "lit_review"
MAX_CHUNKS_PER_PAPER = 4  # first 2 + last 2 chunks per paper
MIN_DEGREE_FOR_VIZ = 2

EXTRACTION_PROMPT = """You are a mathematical knowledge extractor. Given text from a research paper, extract:

1. **Concepts**: Mathematical objects, theorems, conjectures, techniques, and structures mentioned.
2. **Relationships**: How concepts relate to each other.

Return JSON with this exact schema:
{
  "concepts": [
    {
      "name": "short canonical name (e.g., 'Rogers-Ramanujan identities')",
      "type": "one of: object, theorem, conjecture, technique, identity, formula, person, definition",
      "description": "one sentence description if clear from context"
    }
  ],
  "relationships": [
    {
      "source": "concept name (must match a concept above)",
      "target": "concept name (must match a concept above)",
      "relation": "one of: proves, generalizes, uses, implies, is_instance_of, conjectured_by, related_to, equivalent_to, specializes_to, defined_in",
      "detail": "brief explanation"
    }
  ]
}

Rules:
- Use canonical names (e.g., "Bailey lemma" not "Bailey's lemma" or "the lemma of Bailey")
- Prefer established mathematical names over ad-hoc descriptions
- Extract 3-15 concepts per passage (focus on the most important ones)
- Only include relationships that are explicitly stated or clearly implied
- If the text is mostly formulas with little conceptual content, return fewer items
- For people, only include them if they are credited with a specific result in this passage"""

NORMALIZE = {
    "rogers-ramanujan identity": "rogers-ramanujan identities",
    "rr identities": "rogers-ramanujan identities",
    "rr identity": "rogers-ramanujan identities",
    "bailey's lemma": "bailey lemma",
    "hall-littlewood polynomial": "hall-littlewood polynomials",
    "hall-littlewood function": "hall-littlewood polynomials",
    "hall-littlewood functions": "hall-littlewood polynomials",
    "macdonald polynomial": "macdonald polynomials",
    "schur function": "schur functions",
    "schur polynomial": "schur functions",
    "cylindric plane partition": "cylindric partitions",
    "cylindric plane partitions": "cylindric partitions",
    "cpp": "cylindric partitions",
    "cpps": "cylindric partitions",
    "q-binomial coefficient": "q-binomial coefficients",
    "gaussian polynomial": "q-binomial coefficients",
    "gaussian polynomials": "q-binomial coefficients",
    "quasi-symmetric function": "quasi-symmetric functions",
    "quasisymmetric function": "quasi-symmetric functions",
    "quasisymmetric functions": "quasi-symmetric functions",
    "crystal base": "crystal bases",
    "andrews-gordon identity": "andrews-gordon identities",
    "a2 andrews-gordon identities": "a2 andrews-gordon identities",
    "plane partition": "plane partitions",
}

TYPE_COLORS = {
    "object": "#4A90D9",
    "theorem": "#E74C3C",
    "conjecture": "#F39C12",
    "technique": "#2ECC71",
    "identity": "#9B59B6",
    "formula": "#1ABC9C",
    "person": "#E67E22",
    "definition": "#3498DB",
}


# ── PDF ingestion ─────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF using PyMuPDF."""
    try:
        doc = fitz.open(str(pdf_path))
        pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                pages.append((page_num + 1, text))
        doc.close()
        return pages
    except Exception as e:
        print(f"  Warning: could not extract text from {pdf_path.name}: {e}")
        return []


def chunk_text(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split extracted pages into overlapping chunks."""
    chunks = []
    current_text = ""
    current_pages = set()

    for page_num, text in pages:
        text = text.replace('\x00', '').strip()
        if not text:
            continue
        current_text += f"\n[Page {page_num}]\n{text}"
        current_pages.add(page_num)

        while len(current_text) >= chunk_size:
            break_at = chunk_size
            para_break = current_text.rfind('\n\n', overlap, chunk_size)
            if para_break > overlap:
                break_at = para_break
            else:
                for sep in ['. ', '.\n', ';\n']:
                    sent_break = current_text.rfind(sep, overlap, chunk_size)
                    if sent_break > overlap:
                        break_at = sent_break + len(sep)
                        break

            chunk = current_text[:break_at].strip()
            if len(chunk) > 50:
                chunks.append({
                    'text': chunk,
                    'pages': sorted(current_pages),
                })
            current_text = current_text[break_at - overlap:]
            current_pages = set()
            for pn, _ in pages:
                if f"[Page {pn}]" in current_text:
                    current_pages.add(pn)

    if current_text.strip() and len(current_text.strip()) > 50:
        chunks.append({
            'text': current_text.strip(),
            'pages': sorted(current_pages),
        })

    return chunks


def get_embeddings(texts, openai_client):
    """Get embeddings from OpenAI in batches."""
    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = [t[:8000] for t in texts[i:i + batch_size]]
        response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        all_embeddings.extend([e.embedding for e in response.data])
    return all_embeddings


def ingest_pdfs(rag_dir):
    """Ingest PDFs from papers/ into chroma_db/."""
    papers_dir = rag_dir / "papers"
    chroma_dir = rag_dir / "chroma_db"

    pdfs = sorted(papers_dir.glob("*.pdf"))
    if not pdfs:
        print(f"Error: no PDFs found in {papers_dir}")
        sys.exit(1)

    # Load paper metadata if available
    metadata_map = {}
    selected_path = rag_dir / "selected_papers.json"
    if selected_path.exists():
        with open(selected_path) as f:
            for paper in json.load(f):
                key = paper.get('base_id', '').replace('/', '_') + '.pdf'
                metadata_map[key] = paper

    print(f"Ingesting {len(pdfs)} PDFs into ChromaDB...")
    openai_client = OpenAI()

    all_chunks = []
    all_ids = []
    all_metadatas = []

    for pdf_path in pdfs:
        pages = extract_text_from_pdf(pdf_path)
        if not pages:
            continue

        chunks = chunk_text(pages)
        meta = metadata_map.get(pdf_path.name, {})
        arxiv_id = meta.get('arxiv_id', pdf_path.stem)
        title = meta.get('title', pdf_path.stem)

        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(
                f"{arxiv_id}:{i}".encode()
            ).hexdigest()
            all_chunks.append(chunk['text'])
            all_ids.append(chunk_id)
            all_metadatas.append({
                'source': pdf_path.name,
                'arxiv_id': arxiv_id,
                'title': title,
                'chunk_index': i,
                'total_chunks': len(chunks),
            })

    if not all_chunks:
        print("  Warning: no text extracted from any PDF!")
        return 0

    print(f"  {len(all_chunks)} chunks from {len(pdfs)} PDFs")
    print(f"  Generating embeddings...")
    embeddings = get_embeddings(all_chunks, openai_client)

    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = chroma_client.get_or_create_collection(
        name=INGEST_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 500
    for i in range(0, len(all_chunks), batch_size):
        j = min(i + batch_size, len(all_chunks))
        collection.add(
            ids=all_ids[i:j],
            documents=all_chunks[i:j],
            embeddings=embeddings[i:j],
            metadatas=all_metadatas[i:j],
        )

    print(f"  Stored {collection.count()} chunks in {chroma_dir}")
    return collection.count()


# ── Read chunks from ChromaDB ─────────────────────────────────────────

def load_chunks_from_rag(rag_dir):
    """Load all chunks from a ChromaDB index, grouped by paper."""
    chroma_dir = rag_dir / "chroma_db"
    if not chroma_dir.exists():
        print(f"Error: no ChromaDB found at {chroma_dir}")
        sys.exit(1)

    client = chromadb.PersistentClient(path=str(chroma_dir))

    # Try known collection names
    collection = None
    for name in COLLECTION_NAMES:
        try:
            collection = client.get_collection(name)
            break
        except Exception:
            continue

    if collection is None:
        cols = client.list_collections()
        if cols:
            collection = cols[0]
        else:
            print(f"Error: no collections found in {chroma_dir}")
            sys.exit(1)

    total = collection.count()
    print(f"  ChromaDB collection '{collection.name}': {total} chunks")

    papers = defaultdict(list)
    batch_size = 1000
    for offset in range(0, total, batch_size):
        limit = min(batch_size, total - offset)
        results = collection.get(
            limit=limit,
            offset=offset,
            include=["documents", "metadatas"],
        )
        for doc, meta in zip(results["documents"], results["metadatas"]):
            source = meta.get("source", "unknown")
            chunk_index = meta.get("chunk_index", 0)
            papers[source].append({
                "text": doc,
                "chunk_index": chunk_index,
                "title": meta.get("title", ""),
                "category": meta.get("category", ""),
            })

    for source in papers:
        papers[source].sort(key=lambda c: c["chunk_index"])

    print(f"  {len(papers)} papers found")
    return papers


def select_representative_chunks(chunks, max_chunks=MAX_CHUNKS_PER_PAPER):
    """Pick first + last chunks from a paper (intro + conclusion)."""
    if len(chunks) <= max_chunks:
        return chunks
    half = max_chunks // 2
    return chunks[:half] + chunks[-half:]


# ── GPT extraction ────────────────────────────────────────────────────

def extract_concepts(text, paper_name, client):
    """Use GPT-4o-mini to extract concepts and relationships."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": f"Paper: {paper_name}\n\n{text}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=2000,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"  Error extracting from {paper_name}: {e}")
        return {"concepts": [], "relationships": []}


def normalize_name(name):
    """Normalize concept names for deduplication."""
    name = name.strip().lower()
    return NORMALIZE.get(name, name)


# ── Merge and build graph ─────────────────────────────────────────────

def merge_extractions(all_extractions):
    """Merge per-paper extractions into a single knowledge graph."""
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
    """Deduplicate edges and build final graph structure."""
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


# ── HTML visualization ────────────────────────────────────────────────

def generate_html(graph, title="Knowledge Graph", min_degree=MIN_DEGREE_FOR_VIZ):
    """Generate a standalone interactive HTML visualization."""
    degree = defaultdict(int)
    for e in graph["edges"]:
        degree[e["source"]] += 1
        degree[e["target"]] += 1

    keep = {c["name"] for c in graph["concepts"] if degree.get(c["name"], 0) >= min_degree}
    if len(keep) < 5:
        keep = {c["name"] for c in graph["concepts"] if degree.get(c["name"], 0) >= 1}
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

    data = json.dumps({"nodes": nodes, "links": links})

    legend_html = "".join(
        f'<div class="legend-item"><span class="legend-dot" style="background:{color}"></span>{typ}</div>'
        for typ, color in TYPE_COLORS.items()
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ margin: 0; overflow: hidden; background: #1a1a2e; font-family: -apple-system, sans-serif; }}
  svg {{ width: 100vw; height: 100vh; }}
  .link {{ stroke-opacity: 0.7; }}
  .node circle {{ stroke: #fff; stroke-width: 1.5px; cursor: pointer; }}
  .node text {{ fill: #ccc; font-size: 10px; pointer-events: none; }}
  .node:hover text {{ fill: #fff; font-size: 12px; font-weight: bold; }}
  #tooltip {{
    position: absolute; background: rgba(0,0,0,0.85); color: #eee;
    padding: 10px 14px; border-radius: 6px; font-size: 13px;
    pointer-events: none; display: none; max-width: 350px;
    border: 1px solid #444;
  }}
  #tooltip .title {{ font-weight: bold; font-size: 15px; margin-bottom: 4px; }}
  #tooltip .type {{ color: #aaa; font-size: 11px; }}
  #tooltip .desc {{ margin-top: 6px; color: #ccc; }}
  #tooltip .stats {{ margin-top: 6px; color: #888; font-size: 11px; }}
  #controls {{
    position: absolute; top: 12px; left: 12px; color: #aaa;
    font-size: 12px; background: rgba(0,0,0,0.6); padding: 10px;
    border-radius: 6px;
  }}
  #controls input {{ width: 200px; padding: 4px; background: #333;
    border: 1px solid #555; color: #eee; border-radius: 3px; }}
  #legend {{
    position: absolute; bottom: 12px; left: 12px; color: #aaa;
    font-size: 11px; background: rgba(0,0,0,0.6); padding: 10px;
    border-radius: 6px;
  }}
  .legend-item {{ display: flex; align-items: center; margin: 3px 0; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 50%;
    margin-right: 6px; display: inline-block; }}
  #search-results {{ color: #888; margin-top: 4px; font-size: 11px; }}
</style>
</head>
<body>
<div id="tooltip"></div>
<div id="controls">
  <div><strong style="color:#eee">{title}</strong></div>
  <div style="margin-top:6px">
    <input type="text" id="search" placeholder="Search concepts..."
           oninput="searchNodes(this.value)">
    <div id="search-results"></div>
  </div>
  <div style="margin-top:6px;color:#666">
    Drag nodes · Scroll to zoom · Click to highlight · Dbl-click to reset
  </div>
</div>
<div id="legend">
  <div style="margin-bottom:4px"><strong>Types</strong></div>
  {legend_html}
</div>
<svg></svg>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const data = {data};
const width = window.innerWidth, height = window.innerHeight;
const svg = d3.select("svg").attr("viewBox", [0, 0, width, height]);
const g = svg.append("g");
svg.call(d3.zoom().scaleExtent([0.1, 8]).on("zoom", (e) => g.attr("transform", e.transform)));
const simulation = d3.forceSimulation(data.nodes)
  .force("link", d3.forceLink(data.links).id(d => d.id).distance(80))
  .force("charge", d3.forceManyBody().strength(-120))
  .force("center", d3.forceCenter(width / 2, height / 2))
  .force("collision", d3.forceCollide().radius(d => nr(d) + 2));
const link = g.append("g").selectAll("line").data(data.links).join("line")
  .attr("class","link").attr("stroke","#7799BB").attr("stroke-width",1.5);
const node = g.append("g").selectAll("g").data(data.nodes).join("g")
  .attr("class","node")
  .call(d3.drag().on("start",ds).on("drag",dd).on("end",de));
function nr(d) {{ return Math.max(4, Math.min(20, 3 + d.degree * 1.2)); }}
node.append("circle").attr("r", d => nr(d)).attr("fill", d => d.color).attr("opacity", 0.85);
node.append("text").text(d => d.degree >= 3 ? d.label : "").attr("x", d => nr(d)+3).attr("y", 3);
const tooltip = d3.select("#tooltip");
node.on("mouseover", (e, d) => {{
  tooltip.style("display","block")
    .html(`<div class="title">${{d.label}}</div><div class="type">${{d.type}}</div>
           ${{d.description ? `<div class="desc">${{d.description}}</div>` : ''}}
           <div class="stats">${{d.papers}} papers · ${{d.degree}} connections</div>`);
}}).on("mousemove", (e) => {{
  tooltip.style("left",(e.pageX+15)+"px").style("top",(e.pageY-10)+"px");
}}).on("mouseout", () => tooltip.style("display","none"));
node.on("click", (e, d) => {{
  const nb = new Set([d.id]);
  data.links.forEach(l => {{ if(l.source.id===d.id) nb.add(l.target.id); if(l.target.id===d.id) nb.add(l.source.id); }});
  node.select("circle").attr("opacity", n => nb.has(n.id)?1:0.1);
  node.select("text").attr("fill", n => nb.has(n.id)?"#fff":"#333").text(n => nb.has(n.id)?n.label:"");
  link.attr("stroke-opacity", l => l.source.id===d.id||l.target.id===d.id?1:0.08);
}});
svg.on("dblclick", () => {{
  node.select("circle").attr("opacity",0.85);
  node.select("text").attr("fill","#ccc").text(d => d.degree>=3?d.label:"");
  link.attr("stroke-opacity",0.7);
}});
function searchNodes(q) {{
  const r=document.getElementById("search-results");
  if(!q) {{ node.select("circle").attr("opacity",0.85); node.select("text").attr("fill","#ccc").text(d=>d.degree>=3?d.label:""); link.attr("stroke-opacity",0.7); r.textContent=""; return; }}
  const m=data.nodes.filter(n=>n.id.includes(q.toLowerCase())||n.label.toLowerCase().includes(q.toLowerCase()));
  const ids=new Set(m.map(x=>x.id));
  node.select("circle").attr("opacity",n=>ids.has(n.id)?1:0.1);
  node.select("text").attr("fill",n=>ids.has(n.id)?"#fff":"#333").text(n=>ids.has(n.id)?n.label:"");
  r.textContent=`${{m.length}} matches`;
}}
simulation.on("tick", () => {{
  link.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y).attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);
  node.attr("transform",d=>`translate(${{d.x}},${{d.y}})`);
}});
function ds(e) {{ if(!e.active) simulation.alphaTarget(0.3).restart(); e.subject.fx=e.subject.x; e.subject.fy=e.subject.y; }}
function dd(e) {{ e.subject.fx=e.x; e.subject.fy=e.y; }}
function de(e) {{ if(!e.active) simulation.alphaTarget(0); e.subject.fx=null; e.subject.fy=null; }}
</script>
</body>
</html>"""
    return html, len(nodes), len(links)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    # Parse arguments
    rag_dir = None
    resume = False
    viz_only = False

    i = 0
    while i < len(args):
        if args[i] == '--dir' and i + 1 < len(args):
            rag_dir = Path(args[i + 1])
            i += 2
        elif args[i] == '--resume':
            resume = True
            i += 1
        elif args[i] == '--viz-only':
            viz_only = True
            i += 1
        else:
            i += 1

    if rag_dir is None:
        print("Error: --dir <path> is required")
        print(__doc__)
        sys.exit(1)

    graph_file = rag_dir / "knowledge_graph.json"
    html_file = rag_dir / "knowledge_graph.html"
    cache_file = rag_dir / "extraction_cache.json"

    # Viz-only mode: regenerate HTML from existing graph
    if viz_only:
        if not graph_file.exists():
            print(f"Error: no knowledge_graph.json in {rag_dir}")
            sys.exit(1)
        with open(graph_file) as f:
            graph = json.load(f)
        title = f"Knowledge Graph ({graph['metadata']['total_concepts']}c, {graph['metadata']['total_edges']}e)"
        html, n_nodes, n_links = generate_html(graph, title)
        with open(html_file, 'w') as f:
            f.write(html)
        print(f"Visualization: {n_nodes} nodes, {n_links} edges -> {html_file}")
        import webbrowser
        webbrowser.open(f"file://{html_file}")
        return

    # Auto-detect: if chroma_db/ missing but papers/ has PDFs, ingest first
    chroma_dir = rag_dir / "chroma_db"
    papers_dir = rag_dir / "papers"
    if not chroma_dir.exists() and papers_dir.exists() and list(papers_dir.glob("*.pdf")):
        print(f"No ChromaDB found — ingesting PDFs from {papers_dir}...")
        ingest_pdfs(rag_dir)
        print()

    # Load chunks from RAG
    print(f"Reading RAG at {rag_dir}...")
    papers = load_chunks_from_rag(rag_dir)

    # Load cache if resuming
    cache = {}
    if resume and cache_file.exists():
        with open(cache_file) as f:
            cache = json.load(f)
        print(f"Resuming: {len(cache)} papers already processed.\n")

    # Extract from each paper
    client = OpenAI()
    for paper_name, chunks in sorted(papers.items()):
        if paper_name in cache:
            continue

        selected = select_representative_chunks(chunks)
        text = "\n\n---\n\n".join(c["text"][:3000] for c in selected)
        if len(text) > 6000:
            text = text[:6000] + "\n[...truncated...]"

        extraction = extract_concepts(text, paper_name, client)
        n_c = len(extraction.get("concepts", []))
        n_r = len(extraction.get("relationships", []))
        print(f"  {paper_name[:55]:55s} -> {n_c}c, {n_r}r")

        cache[paper_name] = extraction

        # Save cache after each paper
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2)

        time.sleep(0.5)

    # Merge
    print(f"\nMerging {len(cache)} paper extractions...")
    concepts, edges = merge_extractions(cache)
    print(f"  {len(concepts)} unique concepts, {len(edges)} relationships")

    # Build and save graph
    graph = build_graph(concepts, edges)
    with open(graph_file, 'w') as f:
        json.dump(graph, f, indent=2)

    # Generate HTML
    title = f"Knowledge Graph ({graph['metadata']['total_concepts']}c, {graph['metadata']['total_edges']}e)"
    html, n_nodes, n_links = generate_html(graph, title)
    with open(html_file, 'w') as f:
        f.write(html)

    print(f"\nKnowledge graph saved to {graph_file}")
    print(f"  {graph['metadata']['total_concepts']} concepts")
    print(f"  {graph['metadata']['total_edges']} edges")
    print(f"  {graph['metadata']['total_papers']} papers")
    print(f"\nVisualization: {n_nodes} nodes, {n_links} edges -> {html_file}")

    # Show most connected concepts
    print(f"\n{'='*60}")
    print("MOST REFERENCED CONCEPTS (by paper count)")
    print(f"{'='*60}")
    by_papers = sorted(concepts.values(),
                       key=lambda c: len(c["papers"]), reverse=True)
    for c in by_papers[:20]:
        print(f"  {c['name']:45s}  ({len(c['papers'])} papers)")

    # Open in browser
    import webbrowser
    webbrowser.open(f"file://{html_file}")


if __name__ == "__main__":
    main()
