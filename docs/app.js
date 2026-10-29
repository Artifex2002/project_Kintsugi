const root = document.documentElement;
const $ = (id) => document.getElementById(id);
const github = 'https://github.com/Artifex2002/project_Kintsugi';
const examples = {
  original: ['“Pick up the mug.”', 'One instruction. One observation. What action does the model predict?'],
  paraphrase: ['“Lift the cup.”', 'Similar intent, different words. Does the predicted action stay consistent?'],
  opposite: ['“Put down the mug.”', 'A change in intent. How does the predicted action respond?'],
};
document.querySelectorAll('[data-example]').forEach(button => {
  button.addEventListener('click', () => {
    document.querySelectorAll('[data-example]').forEach(b => { b.classList.toggle('selected', b === button); b.setAttribute('aria-pressed', String(b === button)); });
    const [phrase, caption] = examples[button.dataset.example];
    $('example-phrase').textContent = phrase;
    $('example-caption').textContent = caption;
  });
});

const stages = [
  { label: 'Phase 1 / Semantic geometry', title: 'Give language a measurable shape.', description: '177 saved instructions span standard commands, paraphrases, semantic opposites, and unrelated sentences. Sentence embeddings place them in a shared 768-dimensional space.', source: '/tree/main/phase_1', diagram: '<div class="diagram-flow"><div class="diagram-words"><div class="diagram-word">Standard commands</div><div class="diagram-word">Paraphrases</div><div class="diagram-word">Opposites</div><div class="diagram-word">Unrelated language</div></div><span class="diagram-arrow">→</span><div class="diagram-box accent">Text embedder<span>768 dimensions</span></div><span class="diagram-arrow">→</span><div class="diagram-stack"><span></span><span></span><span></span><small>Semantic distances</small></div></div>' },
  { label: 'Phase 1.5 / Meaning and motion', title: 'Compare two kinds of distance.', description: 'With a fixed reference observation, compare cosine distance between text embeddings against Euclidean distance between predicted actions. The committed distance matrices cover 60 standard instructions.', source: '/tree/main/phase_1.5', diagram: '<div class="diagram-flow"><div class="diagram-box">Text distance<span>Cosine</span></div><span class="diagram-arrow">↔</span><div class="diagram-box accent">Correlation<span>Spearman / Mantel</span></div><span class="diagram-arrow">↔</span><div class="diagram-box">Action distance<span>Euclidean</span></div></div>' },
  { label: 'Phase 2 / Black-box search', title: 'Let the next query follow the evidence.', description: 'A Gaussian-process surrogate guides blockwise changes to instruction words and suffixes. The current run has a 60-iteration budget, 20 optimization images, and three stochastic passes per evaluation.', source: '/tree/main/phase_2', diagram: '<div class="diagram-flow"><div class="diagram-box">Change words<span>Synonyms + suffixes</span></div><span class="diagram-arrow">→</span><div class="diagram-box accent">Query SmolVLA<span>Score target proximity</span></div><span class="diagram-arrow">→</span><div class="diagram-box">Update search<span>Expected improvement</span></div></div>' },
];
const tabs = [...document.querySelectorAll('[data-stage]')];
function showStage(index) {
  const stage = stages[index];
  tabs.forEach((tab, i) => { tab.setAttribute('aria-selected', String(i === index)); tab.tabIndex = i === index ? 0 : -1; });
  $('method-panel').setAttribute('aria-labelledby', `method-tab-${index}`);
  $('stage-label').textContent = stage.label;
  $('stage-title').textContent = stage.title;
  $('stage-description').textContent = stage.description;
  $('stage-source').href = github + stage.source;
  $('method-diagram').innerHTML = stage.diagram;
}
tabs.forEach((tab, i) => {
  tab.addEventListener('click', () => showStage(i));
  tab.addEventListener('keydown', event => {
    let next = i;
    if (event.key === 'ArrowRight') next = (i + 1) % tabs.length;
    else if (event.key === 'ArrowLeft') next = (i + tabs.length - 1) % tabs.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = tabs.length - 1;
    else return;
    event.preventDefault(); showStage(next); tabs[next].focus();
  });
});
showStage(0);

let snapshot = null;
let records = [];
let selectedIndex = 0;
let mode = 'placeholder';
let playback = null;
const slider = $('iteration-slider');
const chart = $('trace-chart');
const svgNS = 'http://www.w3.org/2000/svg';
function node(name, attributes = {}, text) {
  const element = document.createElementNS(svgNS, name);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  if (text !== undefined) element.textContent = text;
  chart.appendChild(element);
  return element;
}
function drawTrace() {
  chart.replaceChildren();
  const width = Math.max(240, $('trace-container').clientWidth);
  const height = $('trace-container').clientHeight;
  chart.setAttribute('viewBox', `0 0 ${width} ${height}`);
  const margin = { left: 41, top: 16, right: 13, bottom: 32 };
  const right = width - margin.right, bottom = height - margin.bottom;
  const maxIteration = mode === 'recorded' && records.length ? Math.max(1, records.at(-1).iteration) : 60;
  const pristine = snapshot?.evaluations.find(r => r.kind === 'pristine_baseline');
  const values = records.map(r => r.objective);
  if (pristine) values.push(pristine.objective);
  let min = values.length ? Math.min(...values) : -0.5;
  let max = values.length ? Math.max(...values) : 0;
  const span = Math.max(max - min, 0.025);
  min -= span * .18; max += span * .18;
  const x = value => margin.left + (value / maxIteration) * (right - margin.left);
  const y = value => bottom - ((value - min) / (max - min)) * (bottom - margin.top);
  for (let i = 0; i <= 3; i++) {
    const yy = margin.top + (bottom - margin.top) * i / 3;
    node('line', { x1: margin.left, x2: right, y1: yy, y2: yy, stroke: '#e4e5da', 'stroke-width': 1 });
    node('text', { x: margin.left - 8, y: yy + 3, 'text-anchor': 'end', fill: '#929987', 'font-family': 'IBM Plex Mono, monospace', 'font-size': 9 }, mode === 'recorded' && records.length ? (max - (max - min) * i / 3).toFixed(2) : '—');
  }
  for (let i = 0; i <= 3; i++) {
    const value = Math.round(maxIteration * i / 3);
    const xx = x(value);
    node('line', { x1: xx, x2: xx, y1: margin.top, y2: bottom, stroke: '#e9eade', 'stroke-width': 1 });
    node('text', { x: xx, y: bottom + 17, 'text-anchor': i === 0 ? 'start' : i === 3 ? 'end' : 'middle', fill: '#929987', 'font-family': 'IBM Plex Mono, monospace', 'font-size': 9 }, value);
  }
  if (mode === 'recorded' && records.length) {
    const visible = records.slice(0, selectedIndex + 1);
    if (pristine) node('line', { x1: margin.left, x2: right, y1: y(pristine.objective), y2: y(pristine.objective), stroke: '#68817d', 'stroke-width': 1, 'stroke-dasharray': '4 4' });
    node('polyline', { points: visible.map(r => `${x(r.iteration)},${y(r.objective)}`).join(' '), fill: 'none', stroke: '#bcc2b2', 'stroke-width': 1.3 });
    let best = -Infinity;
    node('polyline', { points: visible.map(r => { best = Math.max(best, r.objective); return `${x(r.iteration)},${y(best)}`; }).join(' '), fill: 'none', stroke: '#a6803c', 'stroke-width': 2 });
    visible.forEach((r, i) => node('circle', { cx: x(r.iteration), cy: y(r.objective), r: i === selectedIndex ? 4 : 2, fill: i === selectedIndex ? '#a6803c' : '#b4bdab' }));
    const selected = records[selectedIndex];
    chart.setAttribute('aria-label', `Recorded optimization through iteration ${selected.iteration}; selected score ${selected.objective.toFixed(4)}. Higher scores indicate closer target-action proximity.`);
  } else chart.setAttribute('aria-label', 'Empty optimization plot. Final results are pending.');
  node('text', { x: (margin.left + right) / 2, y: height - 1, 'text-anchor': 'middle', fill: '#929987', 'font-family': 'IBM Plex Mono, monospace', 'font-size': 8 }, 'ITERATION');
}
function drawActions(record) {
  const pristine = snapshot?.evaluations.find(r => r.kind === 'pristine_baseline');
  const values = record ? [pristine?.mean_action || Array(7).fill(0), record.mean_action, snapshot.target_action] : [];
  const max = Math.max(0.001, ...values.flat().map(Math.abs));
  $('action-bars').replaceChildren();
  for (let i = 0; i < 7; i++) {
    const component = document.createElement('div'); component.className = 'action-component';
    const group = document.createElement('div'); group.className = 'bar-group';
    ['baseline', 'selected', 'target'].forEach((type, series) => {
      const bar = document.createElement('span'); bar.className = `component-bar ${type}`;
      const value = record ? values[series][i] : 0;
      const h = record ? Math.max(1, Math.abs(value) / max * 28) : 1;
      bar.style.height = `${h}px`; bar.style.marginTop = `${value >= 0 ? 31 - h : 31}px`;
      if (!record) bar.style.background = '#dfe2d5';
      group.appendChild(bar);
    });
    const label = document.createElement('span'); label.className = 'micro'; label.textContent = i === 6 ? 'Gripper' : `a${['₁','₂','₃','₄','₅','₆'][i]}`;
    const value = document.createElement('span'); value.className = 'component-value'; value.textContent = record ? record.mean_action[i].toFixed(3) : '—';
    component.append(group, label, value);
    if (record) component.setAttribute('aria-label', `Component ${i+1}: untouched ${values[0][i].toFixed(3)}, selected ${values[1][i].toFixed(3)}, target ${values[2][i].toFixed(3)}`);
    $('action-bars').appendChild(component);
  }
}
function stopPlayback() { if (playback) clearInterval(playback); playback = null; $('play-button').innerHTML = '<span aria-hidden="true">▶</span>'; $('play-button').setAttribute('aria-label', 'Play recorded search'); }
function renderSelection(announce = false) {
  const record = mode === 'recorded' ? records[selectedIndex] : null;
  $('chart-empty').hidden = !!record;
  $('candidate-prompt').classList.toggle('awaiting', !record);
  if (record) {
    $('iteration-label').textContent = `${record.iteration} / ${snapshot.budget} iterations`;
    $('candidate-score').textContent = `Score ${record.objective.toFixed(4)}`;
    $('candidate-prompt').replaceChildren();
    const baseline = records[0].prompt.split(/\s+/);
    record.prompt.split(/\s+/).forEach((word, i) => {
      if (i) $('candidate-prompt').appendChild(document.createTextNode(' '));
      if (word !== baseline[i]) { const mark = document.createElement('mark'); mark.textContent = word; $('candidate-prompt').appendChild(mark); }
      else $('candidate-prompt').appendChild(document.createTextNode(word));
    });
    $('prompt-note').textContent = record.kind === 'search_baseline' ? 'Search baseline · reconstructed instruction with default suffixes' : 'Highlighted words differ from the reconstructed search baseline.';
    $('replay-time').textContent = `Step ${record.iteration.toString().padStart(2, '0')}`;
    $('action-note').textContent = 'Higher objective values mean closer target proximity; they do not measure task failure.';
    if (announce) $('explorer-announcement').textContent = `Iteration ${record.iteration}. Recorded objective ${record.objective.toFixed(4)}.`;
  } else {
    $('iteration-label').textContent = '— / 60 iterations'; $('candidate-score').textContent = 'Score —';
    $('candidate-prompt').textContent = 'The selected instruction will appear here.';
    $('prompt-note').textContent = 'Changes will be highlighted against the search baseline.';
    $('replay-time').textContent = 'Awaiting data';
    $('action-note').textContent = 'Action comparisons are waiting for recorded data.';
  }
  slider.value = selectedIndex;
  drawTrace(); drawActions(record);
}
function setMode(next) {
  stopPlayback(); mode = next;
  document.querySelectorAll('[data-mode]').forEach(button => { const active = button.dataset.mode === mode; button.classList.toggle('selected', active); button.setAttribute('aria-pressed', String(active)); });
  if (mode === 'recorded' && records.length) {
    selectedIndex = records.length - 1;
    $('explorer-notice').lastElementChild.textContent = snapshot.status === 'completed'
      ? `Completed run · ${snapshot.completedCandidates} candidates recorded against a ${snapshot.budget}-iteration budget. This is a saved experiment.`
      : `Recorded snapshot · ${snapshot.completedCandidates} of ${snapshot.budget} candidates captured. Partial results, saved at the time of publication.`;
    const stamp = new Intl.DateTimeFormat('en-US', { dateStyle: 'medium', timeStyle: 'short', timeZone: 'America/New_York' }).format(new Date(snapshot.capturedAt));
    $('snapshot-date').textContent = `Snapshot: ${stamp} ET`;
    slider.disabled = false; slider.max = records.length - 1; $('play-button').disabled = records.length < 2;
  } else {
    selectedIndex = 0; slider.disabled = true; $('play-button').disabled = true;
    $('explorer-notice').lastElementChild.textContent = mode === 'recorded' ? 'No completed evaluations are available in this snapshot yet.' : 'This is the interface preview. Select Recorded snapshot to inspect saved evaluations.';
    $('snapshot-date').textContent = 'Final evaluation pending';
  }
  renderSelection(true);
}
document.querySelectorAll('[data-mode]').forEach(button => button.addEventListener('click', () => setMode(button.dataset.mode)));
slider.addEventListener('input', () => { stopPlayback(); selectedIndex = Number(slider.value); renderSelection(true); });
$('play-button').addEventListener('click', () => {
  if (playback) { stopPlayback(); return; }
  if (selectedIndex >= records.length - 1) selectedIndex = 0;
  $('play-button').innerHTML = '<span aria-hidden="true">Ⅱ</span>'; $('play-button').setAttribute('aria-label', 'Pause recorded search'); renderSelection();
  playback = setInterval(() => { selectedIndex += 1; renderSelection(); if (selectedIndex >= records.length - 1) stopPlayback(); }, 1100);
});
new ResizeObserver(drawTrace).observe($('trace-container'));
renderSelection();

async function loadSnapshot() {
  try {
    const response = await fetch('data/run-snapshot.json');
    if (!response.ok) throw new Error('Snapshot unavailable');
    snapshot = await response.json();
    records = snapshot.evaluations.filter(r => r.kind === 'search_baseline' || r.kind === 'candidate');
    const original = snapshot.evaluations.find(r => r.kind === 'pristine_baseline');
    if (original) $('pristine-prompt').textContent = original.prompt;
    if (snapshot.status === 'completed') {
      document.querySelector('.status-pill').lastChild.textContent = ' Recorded run complete';
      document.querySelector('.experiment-heading-right p').textContent = 'Explore the recorded search and compare its predicted action outputs.';
    }
    const images = snapshot.images.filter(image => image.split === 'optimization');
    const selections = [0, 5, 10, 15].map(i => images[i]).filter(Boolean);
    if (selections.length) $('observation-image').src = selections[0].asset;
    selections.forEach((image, i) => {
      const button = document.createElement('button'); button.type = 'button'; button.setAttribute('aria-label', `View ${image.id}`); button.setAttribute('aria-pressed', String(i === 0));
      const img = document.createElement('img'); img.src = image.asset; img.alt = ''; img.loading = 'lazy'; button.appendChild(img);
      button.addEventListener('click', () => { $('observation-image').src = image.asset; $('observation-image').alt = `Saved LIBERO observation ${image.id}`; document.querySelectorAll('.observation-thumbnails button').forEach(b => b.setAttribute('aria-pressed', String(b === button))); });
      $('observation-thumbnails').appendChild(button);
    });
    if (records.length) {
      setMode('recorded');
      if (snapshot.status !== 'completed') {
        document.querySelector('.status-pill').lastChild.textContent = ' Partial results · study in progress';
        document.querySelector('.experiment-heading-right p').textContent = 'Replay the completed evaluations, inspect instruction changes, and compare predicted actions. Final validation is pending.';
      }
    }
  } catch {
    document.querySelector('[data-mode="recorded"]').disabled = true;
    $('download-data').hidden = true;
  }
}
loadSnapshot();
