import {
  renderProjectCard,
  renderSectionIntro,
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
  <meta name="description" content="Yan Liang builds products and writes about technology and life.">
  <meta name="theme-color" content="#07151b">
  <meta property="og:title" content="Yan Liang. Product thinker, builder and writer.">
  <meta property="og:description" content="I build products and write about technology and life.">
  <meta property="og:type" content="website">
  <meta property="og:url" content="https://cozyan.github.io/">
  <meta property="og:image" content="https://cozyan.github.io/assets/avatar.png">
  <title>Yan Liang. Product thinker, builder and writer.</title>
  <link rel="icon" href="assets/avatar.png">
  <link rel="stylesheet" href="styles.css">
  <script src="script.js" defer></script>
</head>
<body>
  <a class="skip-link" href="#main">Skip to content</a>

  <header class="site-header">
    <nav class="site-nav" aria-label="Main navigation">
      <a class="brand" href="#top" aria-label="Yan Liang, home"><span class="brand-mark" aria-hidden="true">YL</span><span>Yan Liang</span></a>
      <div class="nav-links">
        <a href="#top">Home</a>
        <a href="#work">Work</a>
        <a href="#words">Writing</a>
        <a href="#about">About</a>
      </div>
    </nav>
  </header>

  <main id="main">
    <section class="hero page-shell" id="top" aria-labelledby="hero-title">
      <img class="hero-avatar reveal" src="assets/avatar.png" alt="Illustrated portrait of Yan Liang" width="52" height="52" fetchpriority="high">
      <p class="hero-note reveal">Products, essays and small experiments</p>
      <h1 id="hero-title" class="reveal">I make products<br>and write about<br><em>what I notice.</em></h1>
      <p class="hero-text reveal">I like unclear problems, useful details and ideas that are worth testing.</p>
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
        description: "I write to understand something better. Sometimes it is technology. Sometimes it is life or fiction.",
        id: "words-title",
      })}
      <div class="writing-grid">${writingCards}</div>
    </section>

    <section class="about-section page-shell" id="about" aria-labelledby="about-title">
      <div class="about-copy reveal">
        <p class="eyebrow">About me</p>
        <h2 id="about-title">A little about me</h2>
        <p class="about-lede">I live in the Netherlands and work across product, data and technology. I like unclear problems and small teams.</p>
        <p>Outside work I write essays and fiction. I also build things to see if an idea works.</p>
      </div>
      <dl class="about-details reveal">
        <div><dt>Based in</dt><dd>The Netherlands</dd></div>
        <div><dt>Focus</dt><dd>Products and people</dd></div>
        <div><dt>Also</dt><dd>Essays and fiction</dd></div>
      </dl>
    </section>

    <section class="contact-section" id="contact" aria-labelledby="contact-title">
      <div class="page-shell contact-inner reveal">
        <p class="eyebrow">Say hello</p>
        <h2 id="contact-title">If you have an idea,<br><em>say hello.</em></h2>
        <p>I am happy to talk about products and stories. You can also show me what you are making.</p>
        <div class="contact-links">
          <a class="button button-dark" href="https://www.linkedin.com/in/gemma-liang/" target="_blank" rel="noopener noreferrer">Find me on LinkedIn</a>
          <a class="contact-github" href="https://github.com/cozyan" target="_blank" rel="noopener noreferrer">Visit my GitHub</a>
        </div>
      </div>
    </section>
  </main>

  <footer class="site-footer page-shell">
    <span>© <span data-year></span> Yan Liang</span>
    <span>Built with plain HTML and CSS.</span>
  </footer>
</body>
</html>
`;
}
