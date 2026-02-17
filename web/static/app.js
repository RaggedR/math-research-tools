// Knowledge Graph Builder — Frontend Application
// Upload → Progress (WebSocket) → D3.js Graph Visualization

// ── State ──────────────────────────────────────────────────────────────

let selectedFiles = [];
let sessionId = null;
let ws = null;
let simulation = null;

// D3 references for search/highlight (set during graph render)
let graphNode = null;
let graphLink = null;
let graphData = null;

// ── View Management ────────────────────────────────────────────────────

function showView(viewId) {
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.getElementById(viewId).classList.add('active');
}

// ── Upload View ────────────────────────────────────────────────────────

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileList = document.getElementById('file-list');
const fileCount = document.getElementById('file-count');
const buildBtn = document.getElementById('build-btn');

const SUPPORTED_EXTENSIONS = new Set(['.pdf', '.txt', '.md', '.text', '.markdown']);
const MAX_FILES = 80;

function getExtension(filename) {
    const i = filename.lastIndexOf('.');
    return i >= 0 ? filename.substring(i).toLowerCase() : '';
}

function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function addFiles(newFiles) {
    for (const file of newFiles) {
        const ext = getExtension(file.name);
        if (!SUPPORTED_EXTENSIONS.has(ext)) continue;
        // Skip duplicates
        if (selectedFiles.some(f => f.name === file.name && f.size === file.size)) continue;
        if (selectedFiles.length >= MAX_FILES) break;
        selectedFiles.push(file);
    }
    renderFileList();
}

function removeFile(index) {
    selectedFiles.splice(index, 1);
    renderFileList();
}

function renderFileList() {
    fileList.innerHTML = selectedFiles.map((f, i) => `
        <div class="file-item">
            <span class="file-name">${f.name}</span>
            <span class="file-size">${formatSize(f.size)}</span>
            <span class="file-remove" onclick="removeFile(${i})">&times;</span>
        </div>
    `).join('');

    if (selectedFiles.length > 0) {
        fileCount.textContent = `${selectedFiles.length} file${selectedFiles.length !== 1 ? 's' : ''} selected`;
        buildBtn.disabled = false;
    } else {
        fileCount.textContent = '';
        buildBtn.disabled = true;
    }
}

// Drag and drop
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    addFiles(e.dataTransfer.files);
});
fileInput.addEventListener('change', () => {
    addFiles(fileInput.files);
    fileInput.value = ''; // Reset so same files can be re-added
});

// Build button
buildBtn.addEventListener('click', startUpload);

async function startUpload() {
    if (selectedFiles.length === 0) return;
    buildBtn.disabled = true;

    const formData = new FormData();
    for (const file of selectedFiles) {
        formData.append('files', file);
    }

    try {
        const response = await fetch('/api/upload', { method: 'POST', body: formData });
        if (!response.ok) {
            const err = await response.json();
            alert('Upload failed: ' + (err.detail || 'Unknown error'));
            buildBtn.disabled = false;
            return;
        }

        const data = await response.json();
        sessionId = data.session_id;

        // Switch to progress view and connect WebSocket
        showView('progress-view');
        document.getElementById('progress-subtitle').textContent =
            `Processing ${data.file_count} file${data.file_count !== 1 ? 's' : ''}...`;
        connectWebSocket(sessionId);
    } catch (err) {
        alert('Upload failed: ' + err.message);
        buildBtn.disabled = false;
    }
}

// ── Progress View (WebSocket) ──────────────────────────────────────────

function connectWebSocket(sid) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${window.location.host}/ws/${sid}`;
    ws = new WebSocket(url);

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        handleProgressMessage(msg);
    };

    ws.onerror = () => {
        addLog('WebSocket error — falling back to polling', true);
        startPolling(sid);
    };

    ws.onclose = () => {
        // If not complete, start polling as fallback
        const detail = document.getElementById('progress-detail');
        if (!detail.textContent.includes('Complete')) {
            startPolling(sid);
        }
    };
}

function handleProgressMessage(msg) {
    switch (msg.type) {
        case 'progress':
            updateProgress(msg.stage, msg.detail, msg.percent);
            break;
        case 'complete':
            updateProgress('building', 'Complete!', 100);
            addLog('Knowledge graph ready!');
            setTimeout(() => loadGraph(msg.graph_url), 500);
            break;
        case 'error':
            addLog('Error: ' + msg.message, true);
            document.getElementById('progress-detail').textContent = 'Error: ' + msg.message;
            break;
    }
}

function updateProgress(stage, detail, percent) {
    // Update stage indicators
    const stages = ['ingesting', 'extracting', 'building'];
    const currentIdx = stages.indexOf(stage);
    stages.forEach((s, i) => {
        const el = document.getElementById('stage-' + s);
        el.className = 'stage';
        if (i < currentIdx) el.classList.add('done');
        else if (i === currentIdx) el.classList.add('active');
    });

    // Update progress bar
    document.getElementById('progress-bar').style.width = percent + '%';

    // Update detail text
    document.getElementById('progress-detail').textContent = detail;

    // Add log entry
    addLog(`[${stage}] ${detail}`);
}

function addLog(message, isError) {
    const logArea = document.getElementById('log-area');
    const entry = document.createElement('div');
    entry.className = 'log-entry' + (isError ? ' error' : '');
    entry.textContent = message;
    logArea.appendChild(entry);
    logArea.scrollTop = logArea.scrollHeight;
}

// Polling fallback
let pollTimer = null;

function startPolling(sid) {
    if (pollTimer) return;
    pollTimer = setInterval(async () => {
        try {
            const resp = await fetch(`/api/sessions/${sid}`);
            const data = await resp.json();
            if (data.status === 'complete') {
                clearInterval(pollTimer);
                pollTimer = null;
                loadGraph(`/api/graph/${sid}`);
            } else if (data.status === 'error') {
                clearInterval(pollTimer);
                pollTimer = null;
                addLog('Error: ' + data.error, true);
            }
        } catch (e) {
            // Ignore polling errors
        }
    }, 2000);
}

// ── Graph View (D3.js) ─────────────────────────────────────────────────

async function loadGraph(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            addLog('Failed to load graph data', true);
            return;
        }
        const data = await response.json();
        showView('graph-view');
        renderGraph(data);
    } catch (err) {
        addLog('Failed to load graph: ' + err.message, true);
    }
}

const TYPE_COLORS = {
    object: '#4A90D9',
    theorem: '#E74C3C',
    conjecture: '#F39C12',
    technique: '#2ECC71',
    identity: '#9B59B6',
    formula: '#1ABC9C',
    person: '#E67E22',
    definition: '#3498DB',
};

function renderGraph(data) {
    graphData = data;

    // Build legend
    const legendEl = document.getElementById('graph-legend');
    legendEl.innerHTML = '<div style="margin-bottom:4px"><strong>Types</strong></div>' +
        Object.entries(TYPE_COLORS).map(([typ, color]) =>
            `<div class="legend-item"><span class="legend-dot" style="background:${color}"></span>${typ}</div>`
        ).join('');

    // Clear previous graph
    const svgEl = document.getElementById('graph-svg');
    svgEl.innerHTML = '';

    const width = window.innerWidth;
    const height = window.innerHeight;

    const svg = d3.select('#graph-svg')
        .attr('viewBox', [0, 0, width, height]);

    const g = svg.append('g');

    svg.call(d3.zoom()
        .scaleExtent([0.1, 8])
        .on('zoom', (e) => g.attr('transform', e.transform)));

    function nodeRadius(d) {
        return Math.max(4, Math.min(20, 3 + d.degree * 1.2));
    }

    simulation = d3.forceSimulation(data.nodes)
        .force('link', d3.forceLink(data.links).id(d => d.id).distance(80))
        .force('charge', d3.forceManyBody().strength(-120))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(d => nodeRadius(d) + 2));

    graphLink = g.append('g').selectAll('line')
        .data(data.links).join('line')
        .attr('class', 'link')
        .attr('stroke', '#7799BB')
        .attr('stroke-width', 1.5);

    graphNode = g.append('g').selectAll('g')
        .data(data.nodes).join('g')
        .attr('class', 'node')
        .call(d3.drag()
            .on('start', dragStart)
            .on('drag', dragging)
            .on('end', dragEnd));

    graphNode.append('circle')
        .attr('r', d => nodeRadius(d))
        .attr('fill', d => d.color)
        .attr('opacity', 0.85);

    graphNode.append('text')
        .text(d => d.degree >= 3 ? d.label : '')
        .attr('x', d => nodeRadius(d) + 3)
        .attr('y', 3);

    // Tooltip
    const tooltip = d3.select('#tooltip');
    graphNode.on('mouseover', (e, d) => {
        tooltip.style('display', 'block')
            .html(`<div class="title">${d.label}</div><div class="type">${d.type}</div>
                   ${d.description ? `<div class="desc">${d.description}</div>` : ''}
                   <div class="stats">${d.papers} papers · ${d.degree} connections</div>`);
    }).on('mousemove', (e) => {
        tooltip.style('left', (e.pageX + 15) + 'px').style('top', (e.pageY - 10) + 'px');
    }).on('mouseout', () => tooltip.style('display', 'none'));

    // Click to highlight neighbors
    graphNode.on('click', (e, d) => {
        const neighbors = new Set([d.id]);
        data.links.forEach(l => {
            if (l.source.id === d.id) neighbors.add(l.target.id);
            if (l.target.id === d.id) neighbors.add(l.source.id);
        });
        graphNode.select('circle').attr('opacity', n => neighbors.has(n.id) ? 1 : 0.1);
        graphNode.select('text')
            .attr('fill', n => neighbors.has(n.id) ? '#fff' : '#333')
            .text(n => neighbors.has(n.id) ? n.label : '');
        graphLink.attr('stroke-opacity',
            l => l.source.id === d.id || l.target.id === d.id ? 1 : 0.08);
    });

    // Double-click to reset
    svg.on('dblclick', () => {
        graphNode.select('circle').attr('opacity', 0.85);
        graphNode.select('text').attr('fill', '#ccc').text(d => d.degree >= 3 ? d.label : '');
        graphLink.attr('stroke-opacity', 0.7);
    });

    // Tick
    simulation.on('tick', () => {
        graphLink
            .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
        graphNode.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    // Drag handlers
    function dragStart(e) {
        if (!e.active) simulation.alphaTarget(0.3).restart();
        e.subject.fx = e.subject.x;
        e.subject.fy = e.subject.y;
    }
    function dragging(e) {
        e.subject.fx = e.x;
        e.subject.fy = e.y;
    }
    function dragEnd(e) {
        if (!e.active) simulation.alphaTarget(0);
        e.subject.fx = null;
        e.subject.fy = null;
    }
}

// ── Graph Search ───────────────────────────────────────────────────────

function searchNodes(query) {
    if (!graphData || !graphNode || !graphLink) return;

    const resultsEl = document.getElementById('search-results');

    if (!query) {
        graphNode.select('circle').attr('opacity', 0.85);
        graphNode.select('text').attr('fill', '#ccc')
            .text(d => d.degree >= 3 ? d.label : '');
        graphLink.attr('stroke-opacity', 0.7);
        resultsEl.textContent = '';
        return;
    }

    const q = query.toLowerCase();
    const matches = graphData.nodes.filter(n =>
        n.id.includes(q) || n.label.toLowerCase().includes(q)
    );
    const ids = new Set(matches.map(x => x.id));

    graphNode.select('circle').attr('opacity', n => ids.has(n.id) ? 1 : 0.1);
    graphNode.select('text')
        .attr('fill', n => ids.has(n.id) ? '#fff' : '#333')
        .text(n => ids.has(n.id) ? n.label : '');

    resultsEl.textContent = `${matches.length} matches`;
}
