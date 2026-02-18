"""Generate standalone HTML visualization for CLI usage."""

import json

from .config import TYPE_COLORS
from .graph import prepare_viz_data


def generate_html(graph, title="Knowledge Graph", min_degree=None):
    """Generate a standalone interactive HTML visualization.

    Args:
        graph: graph dict from build_graph()
        title: page title
        min_degree: minimum degree for node inclusion (None = use default)

    Returns:
        (html_string, node_count, link_count)
    """
    kwargs = {}
    if min_degree is not None:
        kwargs["min_degree"] = min_degree
    viz = prepare_viz_data(graph, **kwargs)
    nodes = viz["nodes"]
    links = viz["links"]
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
