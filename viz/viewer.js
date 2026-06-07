// LLM-MapRepair interactive graph viewer
// ---------------------------------------------------------------
// Loads viz/data/games_index.json and per-game JSON files,
// renders the chosen state (gt / noised / llm_built / repaired)
// as a cytoscape.js graph with conflict / repair highlights.

const dataRoot = 'data';
const state = {
  index: null,
  game: null,
  gameId: null,
  dataset: null,
  stateName: 'repaired',
  layoutName: 'cola',
  cy: null,
};

const STATE_LABEL = {
  gt: '① Ground truth · how the world really connects',
  noised: '② With noise — conflicts injected',
  llm_built: '② LLM-built map — edges as the agent saw them',
  repaired: '③ After MapRepair — conflicts resolved',
};

// ---------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------

async function init() {
  const r = await fetch(`${dataRoot}/games_index.json`);
  state.index = (await r.json()).games;

  document.getElementById('datasetSelect').addEventListener('change', onDatasetChange);
  document.getElementById('gameSelect').addEventListener('change', onGameChange);
  document.getElementById('stateSelect').addEventListener('change', onStateChange);
  document.getElementById('layoutSelect').addEventListener('change', onLayoutChange);
  document.getElementById('fitBtn').addEventListener('click', () => state.cy?.fit(undefined, 40));
  document.getElementById('resetBtn').addEventListener('click', () => runLayout());

  // Deep-link via URL hash: #dataset/game[/state]
  // example: #mango/cutthroat/llm_built  or  #textworld/tw_03/repaired
  const hash = location.hash.replace(/^#/, '');
  let initialGame = null, initialState = null;
  if (hash) {
    const parts = hash.split('/').filter(Boolean);
    if (parts.length >= 2) {
      initialGame = `${parts[0]}/${parts[1]}`;
      if (parts[2]) initialState = parts[2];
      // pre-set dataset filter so the game appears in the dropdown
      document.getElementById('datasetSelect').value = parts[0];
    }
  }

  rebuildGameList();

  if (initialGame) {
    document.getElementById('gameSelect').value = initialGame;
    if (initialState) {
      state.stateName = initialState;
      document.getElementById('stateSelect').value = initialState;
    }
    await loadGame(initialGame);
  } else {
    const first = document.getElementById('gameSelect').value;
    if (first) await loadGame(first);
  }

  renderLegend();
}

function rebuildGameList() {
  const datasetFilter = document.getElementById('datasetSelect').value;
  const sel = document.getElementById('gameSelect');
  sel.innerHTML = '';
  state.index
    .filter(g => datasetFilter === 'all' || g.dataset === datasetFilter)
    .forEach(g => {
      const o = document.createElement('option');
      o.value = `${g.dataset}/${g.id}`;
      o.textContent = `${g.dataset === 'mango' ? '🏰' : '🏠'} ${g.label}`;
      sel.appendChild(o);
    });
}

async function onDatasetChange() {
  rebuildGameList();
  const first = document.getElementById('gameSelect').value;
  if (first) await loadGame(first);
}

async function onGameChange(e) {
  await loadGame(e.target.value);
}

function onStateChange(e) {
  state.stateName = e.target.value;
  renderGraph();
}

function onLayoutChange(e) {
  state.layoutName = e.target.value;
  runLayout();
}

// ---------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------

async function loadGame(path) {
  const [dataset, id] = path.split('/');
  state.dataset = dataset;
  state.gameId = id;

  // For MANGO, switch the second state label/select to 'llm_built'
  const stateSel = document.getElementById('stateSelect');
  const opt2 = stateSel.options[1];
  if (dataset === 'mango') {
    opt2.textContent = '② LLM-built map';
    opt2.value = 'llm_built';
    if (state.stateName === 'noised') state.stateName = 'llm_built';
  } else {
    opt2.textContent = '② With noise / conflicts';
    opt2.value = 'noised';
    if (state.stateName === 'llm_built') state.stateName = 'noised';
  }
  stateSel.value = state.stateName;

  const r = await fetch(`${dataRoot}/${dataset}/${id}.json`);
  state.game = await r.json();

  renderGameInfo();
  renderGraph();
  renderWalkthrough();
}

// ---------------------------------------------------------------
// Side panel rendering
// ---------------------------------------------------------------

function renderGameInfo() {
  const g = state.game;
  const info = document.getElementById('gameStats');
  const dsLabel = g.dataset === 'mango' ? 'MANGO' : 'TextWorld';
  const llm = g.states.llm_built;
  const noi = g.states.noised;
  const rep = g.states.repaired;
  let html = `<div class="stats">
    <div class="k">Dataset</div><div class="v">${dsLabel}</div>
    <div class="k">Game</div><div class="v">${g.id}</div>
    <div class="k">Rooms</div><div class="v">${g.n_rooms}</div>
    <div class="k">GT edges</div><div class="v">${g.n_edges ?? g.n_gt_edges ?? '—'}</div>`;
  if (llm) {
    html += `
    <div class="k">LLM edges</div><div class="v">${g.n_llm_edges}</div>
    <div class="k">Correct</div><div class="v" style="color:var(--green)">${llm.n_correct}</div>
    <div class="k">Spurious</div><div class="v" style="color:var(--acl-red)">${llm.n_spurious}</div>
    <div class="k">Conflicts</div><div class="v">${(llm.conflicts || []).length}</div>`;
  } else if (noi) {
    html += `
    <div class="k">Noise injected</div><div class="v" style="color:var(--acl-red)">${noi.noise_records.length}</div>
    <div class="k">Conflicts</div><div class="v">${noi.conflicts.length}</div>`;
  }
  if (rep) {
    html += `
    <div class="k">Repair actions</div><div class="v">${rep.actions.length}</div>
    <div class="k">After repair</div><div class="v">${rep.n_conflicts_before} → ${rep.n_conflicts_after}</div>`;
  }
  html += `</div>`;
  info.innerHTML = html;
}

function renderLegend() {
  const list = document.getElementById('legendList');
  list.innerHTML = `
    <li><span class="swatch" style="background:#5A6B7E"></span>GT edge (also predicted)</li>
    <li><span class="swatch" style="background:#27AE60"></span>Correct (LLM matches GT)</li>
    <li><span class="swatch" style="background:#DC143C"></span>Wrong / spurious edge</li>
    <li><span class="swatch" style="background:#FF8C00"></span>Wrong direction</li>
    <li><span class="swatch" style="background:#7B2CBF"></span>Hallucinated room</li>
    <li><span class="swatch dashed"></span>Removed by MapRepair</li>
    <li><span class="swatch" style="background:#0065BD"></span>Modified by MapRepair</li>
  `;
}

function renderDetails() {
  const titleEl = document.getElementById('detailsTitle');
  const body = document.getElementById('detailsBody');
  const g = state.game;
  const s = state.stateName;
  let title = '';
  let items = '';
  if (s === 'gt') {
    title = 'Ground-truth edges';
    items = g.states.gt.edges.slice(0, 30).map(e =>
      `<div class="detail-item"><div class="head"><span class="src-dst">${e.src} --[${e.dir}]--> ${e.dst}</span></div></div>`
    ).join('');
    if (g.states.gt.edges.length > 30) {
      items += `<div class="detail-item muted">… and ${g.states.gt.edges.length - 30} more.</div>`;
    }
  } else if (s === 'noised') {
    title = `Conflicts (${g.states.noised.conflicts.length}) + noise (${g.states.noised.noise_records.length})`;
    const conflicts = g.states.noised.conflicts;
    const noise = g.states.noised.noise_records;
    items = noise.slice(0, 8).map(n =>
      `<div class="detail-item"><div class="head"><span class="tag tag-${n.type.replace(/_.*/, '')}">${n.type}</span></div>` +
      `<div class="src-dst">${n.src ?? '?'} --[${n.direction ?? '?'}]--> ${n.dst ?? '?'}</div>` +
      (n.note ? `<div class="desc">${n.note}</div>` : '') +
      `</div>`
    ).join('');
    items += '<hr style="margin:10px 0;border:0;border-top:1px solid var(--divider)">';
    items += conflicts.slice(0, 20).map(c =>
      `<div class="detail-item"><div class="head"><span class="tag tag-${c.type}">${c.type}</span></div>` +
      `<div class="desc">${escapeHtml(c.description)}</div></div>`
    ).join('');
  } else if (s === 'llm_built') {
    title = `LLM-built — ${g.states.llm_built.edges.length} edges (${g.states.llm_built.n_correct} correct)`;
    const edges = g.states.llm_built.edges;
    const interesting = edges.filter(e => e.kind !== 'correct');
    items = `<div class="muted" style="margin-bottom:8px">${interesting.length} non-correct edges shown:</div>`;
    items += interesting.slice(0, 30).map(e =>
      `<div class="detail-item"><div class="head"><span class="tag tag-${e.kind}">${e.kind}</span></div>` +
      `<div class="src-dst">${e.src} --[${e.dir}]--> ${e.dst}</div></div>`
    ).join('');
  } else if (s === 'repaired') {
    title = `Repair actions (${g.states.repaired.actions.length}) — conflicts ${g.states.repaired.n_conflicts_before} → ${g.states.repaired.n_conflicts_after}`;
    items = g.states.repaired.actions.map(a => {
      const tag = a.kind === 'modify_edge' ? 'tag-modify'
                : a.kind === 'remove_edge' ? 'tag-remove'
                : '';
      const headInfo = a.new_dir
        ? `<span class="tag ${tag}">${a.kind}</span><span style="color:var(--muted);font-size:11px">→ ${a.new_dir}</span>`
        : `<span class="tag ${tag}">${a.kind}</span>`;
      return `<div class="detail-item"><div class="head">${headInfo}</div>` +
        `<div class="src-dst">${a.target ? a.target.join(' → ') : ''}</div>` +
        (a.reason ? `<div class="desc">${escapeHtml(a.reason)}</div>` : '') +
        `</div>`;
    }).join('') || '<div class="muted">No actions — graph was already conflict-free.</div>';
  }
  titleEl.textContent = title;
  body.innerHTML = items || '<div class="muted">No items.</div>';
}

function renderWalkthrough() {
  const body = document.getElementById('walkthroughBody');
  const w = state.game.walkthrough || [];
  if (!w.length) {
    body.innerHTML = '<div class="muted">(no walkthrough exported for this dataset)</div>';
    return;
  }
  body.innerHTML = w.slice(0, 80).map(s =>
    `<div class="wt-step">` +
    `<span class="step-id">step ${s.step}</span>` +
    `<span class="action">${s.action}</span> ` +
    `<span class="muted">→ ${s.gt_dst || '?'}</span>` +
    (s.obs ? `<div class="obs">${escapeHtml(s.obs)}</div>` : '') +
    `</div>`
  ).join('');
}

// ---------------------------------------------------------------
// Graph rendering
// ---------------------------------------------------------------

const STATE_COLOR = {
  gt: '#5A6B7E',
  correct: '#27AE60',
  spurious: '#DC143C',
  wrong_direction: '#FF8C00',
  wrong_dst: '#B07D00',
  hallucinated_dst: '#7B2CBF',
  hallucinated_src: '#7B2CBF',
  noise: '#DC143C',
  modify: '#0065BD',
};

function renderGraph() {
  const g = state.game;
  const s = state.stateName;
  // Update banner
  document.getElementById('stateBanner').className = `state-banner ${s}`;
  document.getElementById('stateBanner').innerHTML =
    `${STATE_LABEL[s]} <em>· ${g.dataset === 'mango' ? 'MANGO' : 'TextWorld'} · ${g.id}</em>`;

  let stateData = g.states[s];
  if (!stateData) {
    document.getElementById('cy').innerHTML = '';
    return;
  }

  // For "noised" / "repaired", compare against GT to color edges
  const gtSet = new Set(g.states.gt.edges.map(e => `${e.src}|${e.dst}|${e.dir}`));
  const gtPair = new Set(g.states.gt.edges.map(e => `${e.src}|${e.dst}`));
  const gtNodes = new Set(g.states.gt.edges.flatMap(e => [e.src, e.dst]));

  const elements = [];
  const noisedEdgeSet = s === 'noised'
    ? new Set((stateData.noise_records || []).map(r => `${r.src}|${r.dst}|${r.direction}`))
    : new Set();

  // Build node set
  const nodeIds = new Set();
  for (const e of stateData.edges) {
    nodeIds.add(e.src); nodeIds.add(e.dst);
  }
  if (s === 'gt') for (const r of g.rooms) nodeIds.add(r.id);

  // For "repaired" we also want to show removed edges as faded ghosts:
  // collect the union of repaired edges + 'before' (noised or llm_built) so the
  // viewer can see what got dropped.
  let ghostEdges = [];
  if (s === 'repaired') {
    const beforeKey = g.dataset === 'mango' ? 'llm_built' : 'noised';
    const before = g.states[beforeKey].edges;
    const repPairSet = new Set(stateData.edges.map(e => `${e.src}|${e.dst}|${e.dir}`));
    ghostEdges = before
      .filter(e => !repPairSet.has(`${e.src}|${e.dst}|${e.dir}`))
      .map(e => ({...e, ghost: true}));
    for (const e of ghostEdges) {
      nodeIds.add(e.src); nodeIds.add(e.dst);
    }
  }

  // Node elements
  const roomMeta = Object.fromEntries((g.rooms || []).map(r => [r.id, r]));
  for (const id of nodeIds) {
    const isGt = gtNodes.has(id);
    elements.push({
      group: 'nodes',
      data: { id, label: id, gt: isGt ? 1 : 0,
              desc: (roomMeta[id]?.description || '') },
    });
  }

  // Edge elements
  for (const e of stateData.edges) {
    const key = `${e.src}|${e.dst}|${e.dir}`;
    let kind;
    if (e.kind) {
      kind = e.kind;  // pre-classified (MANGO llm_built)
    } else if (s === 'gt') {
      kind = 'gt';
    } else if (s === 'noised') {
      if (noisedEdgeSet.has(key) || !gtPair.has(`${e.src}|${e.dst}`)) {
        kind = 'spurious';
      } else if (!gtSet.has(key)) {
        kind = 'wrong_direction';
      } else {
        kind = 'gt';
      }
    } else if (s === 'repaired') {
      if (gtSet.has(key)) kind = 'gt';
      else if (gtPair.has(`${e.src}|${e.dst}`)) kind = 'wrong_direction';
      else kind = 'spurious';
    } else {
      kind = 'gt';
    }
    elements.push({
      group: 'edges',
      data: {
        id: `${e.src}__${e.dst}__${e.dir}`,
        source: e.src, target: e.dst,
        label: e.dir, kind,
      },
    });
  }

  // Ghost (removed) edges in 'repaired' view
  for (const e of ghostEdges) {
    elements.push({
      group: 'edges',
      data: {
        id: `ghost__${e.src}__${e.dst}__${e.dir}`,
        source: e.src, target: e.dst,
        label: e.dir, kind: 'ghost',
      },
    });
  }

  buildCy(elements);
  renderDetails();
  renderFoot(stateData);
}

function buildCy(elements) {
  if (state.cy) { state.cy.destroy(); }
  state.cy = cytoscape({
    container: document.getElementById('cy'),
    elements,
    wheelSensitivity: 0.25,
    style: [
      // nodes
      {
        selector: 'node',
        style: {
          'background-color': '#FFFFFF',
          'border-color': '#0065BD',
          'border-width': 2,
          'label': 'data(label)',
          'font-family': 'IBM Plex Sans, sans-serif',
          'font-size': 12,
          'font-weight': 600,
          'color': '#1A2332',
          'text-valign': 'center',
          'text-halign': 'center',
          'text-wrap': 'wrap',
          'text-max-width': 100,
          'width': 'label',
          'height': 'label',
          'padding': '12px',
          'shape': 'round-rectangle',
        },
      },
      {
        selector: 'node[gt = 0]',
        // hallucinated room — purple
        style: { 'border-color': '#7B2CBF', 'background-color': '#F8F0FF' },
      },
      {
        selector: 'node:selected',
        style: { 'border-color': '#DC143C', 'border-width': 3 },
      },
      // edges — default
      {
        selector: 'edge',
        style: {
          'curve-style': 'bezier',
          'control-point-step-size': 60,
          'target-arrow-shape': 'triangle',
          'arrow-scale': 1.2,
          'width': 2.2,
          'line-color': '#5A6B7E',
          'target-arrow-color': '#5A6B7E',
          'label': 'data(label)',
          'font-family': 'IBM Plex Mono, monospace',
          'font-size': 10,
          'color': '#5A6B7E',
          'text-background-color': '#FFFFFF',
          'text-background-opacity': 0.85,
          'text-background-shape': 'round-rectangle',
          'text-background-padding': '2px',
        },
      },
      { selector: 'edge[kind = "gt"]',        style: { 'line-color': '#5A6B7E', 'target-arrow-color': '#5A6B7E', 'color': '#5A6B7E' } },
      { selector: 'edge[kind = "correct"]',   style: { 'line-color': '#27AE60', 'target-arrow-color': '#27AE60', 'color': '#27AE60' } },
      { selector: 'edge[kind = "spurious"]',  style: { 'line-color': '#DC143C', 'target-arrow-color': '#DC143C', 'width': 3, 'color': '#DC143C' } },
      { selector: 'edge[kind = "wrong_direction"]', style: { 'line-color': '#FF8C00', 'target-arrow-color': '#FF8C00', 'width': 3, 'color': '#FF8C00' } },
      { selector: 'edge[kind = "wrong_dst"]', style: { 'line-color': '#B07D00', 'target-arrow-color': '#B07D00', 'width': 3, 'color': '#B07D00' } },
      { selector: 'edge[kind = "hallucinated_dst"]', style: { 'line-color': '#7B2CBF', 'target-arrow-color': '#7B2CBF', 'width': 2.5, 'color': '#7B2CBF' } },
      { selector: 'edge[kind = "hallucinated_src"]', style: { 'line-color': '#7B2CBF', 'target-arrow-color': '#7B2CBF', 'width': 2.5, 'color': '#7B2CBF' } },
      { selector: 'edge[kind = "modify"]',    style: { 'line-color': '#0065BD', 'target-arrow-color': '#0065BD', 'color': '#0065BD' } },
      {
        selector: 'edge[kind = "ghost"]',
        style: {
          'line-color': '#C8D6E5',
          'target-arrow-color': '#C8D6E5',
          'line-style': 'dashed',
          'opacity': 0.6,
          'color': '#C8D6E5',
        },
      },
      // hover
      { selector: 'edge:selected', style: { 'width': 4, 'opacity': 1 } },
    ],
  });

  // Click events
  state.cy.on('tap', 'node', (evt) => {
    const n = evt.target;
    const desc = n.data('desc');
    if (desc) {
      alert(`${n.id()}\n\n${desc}`);
    }
  });

  runLayout();
}

function runLayout() {
  if (!state.cy) return;
  const opts = {
    cola: {
      name: 'cola', nodeSpacing: 24, edgeLength: 140, animate: false,
      maxSimulationTime: 3500, randomize: true, fit: true, padding: 30,
    },
    dagre: { name: 'dagre', rankDir: 'TB', nodeSep: 50, edgeSep: 18, rankSep: 60, fit: true, padding: 30 },
    grid:  { name: 'grid', padding: 30, fit: true },
  };
  state.cy.layout(opts[state.layoutName] || opts.cola).run();
  setTimeout(() => state.cy.fit(undefined, 40), 200);
}

function renderFoot(stateData) {
  const foot = document.getElementById('canvasFoot');
  const g = state.game;
  let parts = [`nodes: ${state.cy.nodes().length}`, `edges: ${state.cy.edges().length}`];
  if (state.stateName === 'noised') {
    parts.push(`noise: ${stateData.noise_records.length}`);
    parts.push(`conflicts: ${stateData.conflicts.length}`);
  } else if (state.stateName === 'repaired') {
    parts.push(`actions: ${stateData.actions.length}`);
    parts.push(`conflicts: ${stateData.n_conflicts_before} → ${stateData.n_conflicts_after}`);
  } else if (state.stateName === 'llm_built') {
    parts.push(`correct: ${stateData.n_correct}`);
    parts.push(`spurious: ${stateData.n_spurious}`);
    parts.push(`conflicts: ${(stateData.conflicts || []).length}`);
  }
  foot.textContent = parts.join('   ·   ');
}

// ---------------------------------------------------------------
// Utils
// ---------------------------------------------------------------

function escapeHtml(s) {
  return String(s || '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

init();
