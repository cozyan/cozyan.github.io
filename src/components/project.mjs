import { escapeHtml, renderExternalAttributes } from "../lib/html.mjs";

export function renderTraceVisual({ compact = false } = {}) {
  return `<div class="product-visual trace-visual${compact ? " visual-compact" : ""}" role="img" aria-label="A stylised view of the TRACE reflection interface">
    <div class="trace-orbit" aria-hidden="true"></div>
    <div class="trace-phone trace-phone-back" aria-hidden="true">
      <div class="trace-screen trace-night"><span>00:42</span><strong>Let today rest here.</strong><i></i><i></i></div>
    </div>
    <div class="trace-phone trace-phone-front">
      <div class="trace-screen">
        <span class="phone-label">Morning start</span>
        <strong>What would feel like progress today?</strong>
        <div class="trace-moods"><i></i><i></i><i></i></div>
        <div class="trace-input"><small>Today’s anchor</small><b>Finish the important step</b></div>
        <div class="trace-button">Begin today</div>
      </div>
    </div>
  </div>`;
}

export function renderGraphVisual({ compact = false } = {}) {
  return `<div class="product-visual graph-visual${compact ? " visual-compact" : ""}" role="img" aria-label="A visible Creative Graph path with ranking evidence">
    <svg viewBox="0 0 760 480" aria-hidden="true">
      <path class="graph-line" pathLength="1" d="M66 306 C160 280 170 122 286 160 S395 340 494 244 S594 104 696 142"/>
      <g class="graph-point graph-point-one" transform="translate(66 306)"><circle r="32"/><text y="4">video</text></g>
      <g class="graph-point graph-point-two" transform="translate(286 160)"><circle r="40"/><text y="4">memory</text></g>
      <g class="graph-point graph-point-three" transform="translate(494 244)"><circle r="36"/><text y="4">place</text></g>
      <g class="graph-point graph-point-four" transform="translate(696 142)"><circle r="42"/><text y="4">identity</text></g>
    </svg>
    <div class="graph-evidence"><span>Relevance <b>0.86</b></span><span>Coherence <b>0.79</b></span><span>Surprise <b>0.64</b></span></div>
  </div>`;
}

export function renderImageVisual(project, { compact = false } = {}) {
  return `<figure class="product-visual image-visual${compact ? " visual-compact" : ""}">
    <img src="${escapeHtml(project.cover)}" width="${project.coverWidth}" height="${project.coverHeight}" alt="${escapeHtml(project.coverAlt)}" loading="lazy">
    <figcaption><span>${escapeHtml(project.category)}</span><strong>${escapeHtml(project.stage)}</strong></figcaption>
  </figure>`;
}

const visualRenderers = {
  trace: renderTraceVisual,
  graph: renderGraphVisual,
};

export function renderProjectVisual(project, options) {
  if (project.visual === "image") return renderImageVisual(project, options);
  const renderer = visualRenderers[project.visual];
  if (!renderer) throw new Error(`Unknown project visual: ${project.visual}`);
  return renderer(options);
}

export function renderProjectStory(project) {
  return `<article class="project-story project-${escapeHtml(project.id)} reveal" id="${escapeHtml(project.id)}">
    <div class="project-copy">
      <p class="eyebrow">${escapeHtml(project.number)} · ${escapeHtml(project.category)}</p>
      <h2>${escapeHtml(project.title)}</h2>
      <p class="project-observation">${escapeHtml(project.observation)}</p>
      <p class="project-response">${escapeHtml(project.response)}</p>
      <a class="project-link" href="/work/${escapeHtml(project.id)}/">Read the product story <span aria-hidden="true">↗</span></a>
    </div>
    ${renderProjectVisual(project)}
  </article>`;
}

export function renderProjectFacts(project) {
  return `<dl class="project-facts">${project.facts.map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("")}</dl>`;
}

export function renderProjectExternalLink(project) {
  if (!project.externalUrl) return `<p class="project-status"><span>Current stage</span><strong>${escapeHtml(project.stage)}</strong></p>`;
  return `<a class="button button-dark" href="${escapeHtml(project.externalUrl)}" ${renderExternalAttributes()}>${escapeHtml(project.externalLabel)}</a>`;
}
