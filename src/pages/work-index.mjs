import { escapeHtml } from "../lib/html.mjs";
import { renderContactSection, renderDocument, renderPageHeading } from "../components/layout.mjs";
import { renderProjectVisual } from "../components/project.mjs";

export function renderWorkIndexPage({ projects }) {
  const projectRows = projects.map((project) => `<a class="work-index-row reveal" href="/work/${project.id}/">
    <div class="work-index-copy"><p class="eyebrow">${escapeHtml(project.number)} · ${escapeHtml(project.category)}</p><h2>${escapeHtml(project.title)}</h2><p>${escapeHtml(project.summary)}</p><span>Read the story ↗</span></div>
    ${renderProjectVisual(project, { compact: true })}
  </a>`).join("");

  const body = `<main id="main">
    <section class="page-shell work-index-hero">
      ${renderPageHeading({ eyebrow: "Selected work", title: "Products built around<br><em>human questions.</em>", description: "I start with something that feels unresolved. Then I make the smallest useful way to examine it." })}
    </section>
    <section class="page-shell work-index-list">${projectRows}</section>
    ${renderContactSection()}
  </main>`;

  return renderDocument({ title: "Work", description: "Selected product work by Yan Liang.", path: "/work/", current: "work", body, bodyClass: "work-index-page" });
}
