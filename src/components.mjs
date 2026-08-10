const externalLinkAttributes = 'target="_blank" rel="noopener noreferrer"';

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

export function renderSectionIntro({ eyebrow, title, description, id }) {
  return `
    <header class="section-intro reveal">
      <p class="eyebrow">${escapeHtml(eyebrow)}</p>
      <h2 id="${escapeHtml(id)}">${escapeHtml(title)}</h2>
      <p>${escapeHtml(description)}</p>
    </header>`;
}

function renderFacts(facts) {
  return facts
    .map(
      ([label, value]) => `
        <div>
          <dt>${escapeHtml(label)}</dt>
          <dd>${escapeHtml(value)}</dd>
        </div>`,
    )
    .join("");
}

function renderTraceVisual() {
  return `
    <div class="project-visual trace-stage" role="img" aria-label="A representation of the TRACE morning-start interface">
      <div class="trace-halo halo-one"></div>
      <div class="trace-halo halo-two"></div>
      <div class="phone phone-back" aria-hidden="true">
        <div class="phone-screen night-screen">
          <p>00:42</p>
          <h4>先让今天停在这里</h4>
          <span>把还在脑中的事轻轻放下。</span>
          <div class="night-card"></div>
          <div class="night-button"></div>
        </div>
      </div>
      <div class="phone phone-front">
        <div class="phone-screen morning-screen">
          <div class="phone-bar"><span>09:12</span><strong>TRACE</strong><span>•••</span></div>
          <p class="phone-kicker">MORNING START</p>
          <h4>今天想往前推一点的事是什么？</h4>
          <div class="state-row"><span></span><span></span><span></span></div>
          <div class="input-card"><small>今天的锚点</small><strong>完成最重要的一步</strong></div>
          <div class="trace-cta">开始今天</div>
          <div class="tabbar"><i></i><i></i><i></i><i></i><i></i></div>
        </div>
      </div>
    </div>`;
}

function renderGraphVisual() {
  return `
    <div class="project-visual graph-stage" role="img" aria-label="A Creative Graph path and its transparent scoring evidence">
      <div class="graph-path">
        <span class="node node-a">video</span><span class="edge edge-a"></span>
        <span class="node node-b">memory</span><span class="edge edge-b"></span>
        <span class="node node-c">travel</span><span class="edge edge-c"></span>
        <span class="node node-d">place</span><span class="edge edge-d"></span>
        <span class="node node-e">identity</span>
      </div>
      <div class="score-card">
        <div><span>RELEVANCE</span><strong>0.86</strong></div>
        <div><span>COHERENCE</span><strong>0.79</strong></div>
        <div><span>SURPRISE</span><strong>0.64</strong></div>
        <p>Ranking stays visible.<br>No hidden model call.</p>
      </div>
    </div>`;
}

const visualRenderers = {
  trace: renderTraceVisual,
  graph: renderGraphVisual,
};

export function renderProjectCard(project) {
  const renderVisual = visualRenderers[project.visual];
  if (!renderVisual) throw new Error(`Unknown project visual: ${project.visual}`);

  return `
    <article class="project project-${escapeHtml(project.id)} reveal">
      <div class="project-copy">
        <p class="project-index">${escapeHtml(project.index)} / ${escapeHtml(project.category)}</p>
        <h3>${project.title.map(escapeHtml).join("<br>")}</h3>
        <p class="project-summary">${escapeHtml(project.summary)}</p>
        <dl class="project-facts">${renderFacts(project.facts)}</dl>
        <a class="project-link" href="${escapeHtml(project.href)}" ${externalLinkAttributes}>${escapeHtml(project.linkLabel)}</a>
      </div>
      ${renderVisual()}
    </article>`;
}

export function renderWritingCard(article) {
  const classes = ["writing-card", "reveal"];
  if (article.featured) classes.push("writing-feature");
  if (article.accent) classes.push("writing-accent");

  return `
    <a class="${classes.join(" ")}" href="${escapeHtml(article.href)}" ${externalLinkAttributes}>
      <span class="writing-number">${escapeHtml(article.index)}</span>
      <div>
        <p class="writing-topic">${escapeHtml(article.topic)}</p>
        <h3>${escapeHtml(article.title)}</h3>
        ${article.description ? `<p>${escapeHtml(article.description)}</p>` : ""}
      </div>
      <strong>Read on Medium</strong>
    </a>`;
}

export function renderTicker(items) {
  const content = items
    .map((item) => `<span>${escapeHtml(item)} <i></i></span>`)
    .join("");

  return `<div class="ticker-wrap" aria-hidden="true">
    <div class="ticker">
      <div class="ticker-track">
        <div class="ticker-group">${content}</div>
        <div class="ticker-group">${content}</div>
      </div>
    </div>
  </div>`;
}

export function renderSiteHeader({ current = "home" } = {}) {
  const links = [
    ["home", "Home", "index.html#top"],
    ["work", "Work", "index.html#work"],
    ["writing", "Writing", "index.html#words"],
    ["about", "About", "about.html"],
  ];

  const navigation = links
    .map(([id, label, href]) => {
      const currentAttribute = id === current ? ' aria-current="page"' : "";
      return `<a href="${href}"${currentAttribute}>${label}</a>`;
    })
    .join("");

  return `<header class="site-header">
      <nav class="site-nav" aria-label="Main navigation">
        <a class="brand" href="index.html#top"><span class="brand-mark" aria-hidden="true">YL</span><span>Yan Liang</span></a>
        <div class="nav-links">${navigation}</div>
      </nav>
    </header>`;
}

export function renderContactSection() {
  return `<section class="contact-section" id="contact" aria-labelledby="contact-title">
      <div class="page-shell contact-inner reveal">
        <p class="eyebrow">Say hello</p>
        <h2 id="contact-title">Have an idea?<br><em>Say hello.</em></h2>
        <p>I am happy to talk about products and stories. You can also show me what you are making.</p>
        <div class="contact-links">
          <a class="button button-dark" href="https://www.linkedin.com/in/gemma-liang/" ${externalLinkAttributes}>Find me on LinkedIn</a>
          <a class="contact-github" href="https://github.com/cozyan" ${externalLinkAttributes}>Visit my GitHub</a>
        </div>
      </div>
    </section>`;
}

export function renderFooter() {
  return `<footer class="site-footer page-shell">
      <span>© <span data-year></span> Yan Liang</span>
      <span>Built with plain HTML and CSS.</span>
    </footer>`;
}
