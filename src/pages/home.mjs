import { site } from "../content/site.mjs";
import { renderContactSection, renderDocument } from "../components/layout.mjs";
import { renderProjectStory } from "../components/project.mjs";
import { renderWritingList } from "../components/writing.mjs";

export function renderHomePage({ articles, projects }) {
  const featuredProjects = projects.filter((project) => project.featured);
  const projectStories = featuredProjects.map((project) => renderProjectStory(project)).join("");
  const projectCountCopy = featuredProjects.length === 1 ? "One project ready to share." : `${featuredProjects.length} projects ready to share.`;
  const body = `<main id="main">
    <section class="home-hero page-shell" aria-labelledby="home-title" data-dot-grid-scene>
      <div class="hero-intro reveal">
        <img src="/assets/avatar.png" width="56" height="56" alt="Illustrated portrait of Yan Liang" fetchpriority="high">
        <p>Products · essays · small experiments</p>
      </div>
      <h1 id="home-title" class="hero-title"><span class="reveal-line">I make products</span><span class="reveal-line">and write about</span><em class="reveal-line">what I notice.</em></h1>
      <p class="hero-copy reveal">I like unclear problems. I care about useful details. I test ideas to see what works.</p>
      <div class="hero-foot reveal"><a class="button button-light" href="#selected-work">View selected work</a><a class="button button-quiet" href="#writing">Read my writing</a></div>
      <p class="hero-location reveal">Based in the Netherlands</p>
      <canvas class="hero-dot-grid" data-dot-grid width="1200" height="600" aria-hidden="true"></canvas>
    </section>

    <section class="work-home page-shell" id="selected-work" aria-labelledby="selected-title">
      <header class="selected-intro">
        <p class="eyebrow">Curated work</p>
        <h2 id="selected-title">Selected projects</h2>
        <p>${projectCountCopy}</p>
      </header>
      <div class="project-list">${projectStories}</div>
    </section>

    <section class="writing-home page-shell" id="writing" aria-label="Latest writing">
      ${renderWritingList(articles.slice(0, 3))}
      <a class="archive-link" href="/writing/">Visit the writing archive <span aria-hidden="true">↗</span></a>
    </section>

    <section class="about-home page-shell" aria-labelledby="about-home-title">
      <p class="eyebrow reveal">A little about me</p>
      <div class="about-home-grid reveal"><h2 id="about-home-title">I like unclear problems and useful details.</h2><div><p>I work where product thinking, technology, and human behaviour meet. Outside work I write essays and fiction.</p><a class="text-link" href="/about/">Read the short version</a></div></div>
    </section>

    ${renderContactSection()}
  </main>`;

  return renderDocument({
    title: site.title,
    description: site.description,
    path: "/",
    current: "home",
    body,
    bodyClass: "home-page",
  });
}
