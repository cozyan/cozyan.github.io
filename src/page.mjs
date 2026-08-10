import {
  renderContactSection,
  renderFooter,
  renderProjectCard,
  renderSectionIntro,
  renderSiteHeader,
  renderTicker,
  renderWritingCard,
} from "./components.mjs";
import { projects, tickerItems, writing } from "./content.mjs";

export function renderPage() {
  const projectCards = projects.map(renderProjectCard).join("");
  const writingCards = writing.map(renderWritingCard).join("");

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Yan Liang makes thoughtful products and writes about technology and life.">
  <meta name="theme-color" content="#07151b">
  <meta property="og:title" content="Yan Liang. Products and stories.">
  <meta property="og:description" content="I build products and write about technology and life.">
  <meta property="og:type" content="website">
  <meta property="og:url" content="https://cozyan.github.io/">
  <meta property="og:image" content="https://cozyan.github.io/assets/avatar.png">
  <title>Yan Liang. Products and stories.</title>
  <link rel="icon" href="assets/avatar.png">
  <link rel="stylesheet" href="styles.css">
  <script src="script.js" defer></script>
</head>
<body>
  <a class="skip-link" href="#main">Skip to content</a>

  ${renderSiteHeader()}

  <main id="main">
    <section class="hero page-shell" id="top" aria-labelledby="hero-title">
      <img class="hero-avatar reveal" src="assets/avatar.png" alt="Illustrated portrait of Yan Liang" width="52" height="52" fetchpriority="high">
      <p class="hero-note reveal">Products · essays · small experiments</p>
      <h1 id="hero-title" class="reveal">I make products<br>and write about<br><em>what I notice.</em></h1>
      <p class="hero-text reveal">I like unclear problems. I care about useful details. I test ideas to see what works.</p>
      <div class="hero-footer reveal">
        <a class="button button-light" href="#work">View selected work</a>
        <a class="button button-quiet" href="#contact">Say hello</a>
      </div>
      <p class="hero-location reveal">Based in the Netherlands</p>
    </section>

    <section class="work-section page-shell" id="work" aria-labelledby="work-title">
      ${renderSectionIntro({
        eyebrow: "Curated work",
        title: "Selected projects",
        description: "Two projects I keep coming back to.",
        id: "work-title",
      })}
      <div class="project-list">${projectCards}</div>
    </section>

    ${renderTicker(tickerItems)}

    <section class="words-section page-shell" id="words" aria-labelledby="words-title">
      ${renderSectionIntro({
        eyebrow: "Thoughts and stories",
        title: "Things I’ve been writing",
        description: "Writing helps me understand things. Some pieces are about technology. Others are about life or fiction.",
        id: "words-title",
      })}
      <div class="writing-grid">${writingCards}</div>
    </section>

    <section class="about-section page-shell" id="about" aria-labelledby="about-title">
      <div class="about-copy reveal">
        <p class="eyebrow">About me</p>
        <h2 id="about-title">A little about me</h2>
        <p class="about-lede">I live in the Netherlands. My work sits between products and technology.</p>
        <p>I enjoy unclear problems and small teams. Outside work I write essays and fiction.</p>
        <a class="text-link" href="about.html">Read more about me</a>
      </div>
      <dl class="about-details reveal">
        <div><dt>Based in</dt><dd>The Netherlands</dd></div>
        <div><dt>Focus</dt><dd>Products and people</dd></div>
        <div><dt>Also</dt><dd>Essays and fiction</dd></div>
      </dl>
    </section>

    ${renderContactSection()}
  </main>

  ${renderFooter()}
</body>
</html>
`;
}
