import { escapeHtml } from "../lib/html.mjs";
import { renderContactSection, renderDocument } from "../components/layout.mjs";
import { renderProjectExternalLink, renderProjectFacts, renderProjectVisual } from "../components/project.mjs";
import { renderRelatedArticle } from "../components/writing.mjs";

export function renderProjectPage({ project, articles }) {
  const relatedArticle = articles.find((article) => article.slug === project.relatedWriting);
  const body = `<main id="main">
    <article class="project-page">
      <header class="project-page-header page-shell">
        <a class="back-link" href="/work/">All work</a>
        <p class="eyebrow reveal">${escapeHtml(project.number)} · ${escapeHtml(project.category)}</p>
        <h1 class="reveal">${escapeHtml(project.title)}</h1>
        <p class="project-page-lede reveal">${escapeHtml(project.summary)}</p>
        <div class="reveal">${renderProjectFacts(project)}</div>
      </header>
      <section class="project-page-visual page-shell reveal">${renderProjectVisual(project)}</section>
      <section class="project-prose article-prose reveal">${project.html}</section>
      <div class="project-page-actions page-shell reveal">${renderProjectExternalLink(project)}${renderRelatedArticle(relatedArticle)}</div>
    </article>
    ${renderContactSection()}
  </main>`;

  return renderDocument({ title: project.title, description: project.summary, path: `/work/${project.id}/`, current: "work", body, bodyClass: `project-document project-document-${project.id}` });
}
