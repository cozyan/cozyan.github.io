import { renderDocument } from "../components/layout.mjs";
import { escapeHtml } from "../lib/html.mjs";

export function renderProjectPrivacyPage({ privacy }) {
  const body = `<main id="main">
    <article class="article-page">
      <header class="article-header page-shell">
        <a class="back-link" href="/work/${escapeHtml(privacy.projectSlug)}/">The Tiny Exhibit</a>
        <p class="eyebrow reveal">Chrome extension</p>
        <h1 class="reveal">${escapeHtml(privacy.title)}</h1>
        <p class="article-summary reveal">${escapeHtml(privacy.summary)}</p>
        <p class="article-byline reveal"><span>Last updated</span><span>${escapeHtml(privacy.updated)}</span></p>
      </header>
      <section class="article-prose reveal">${privacy.html}</section>
    </article>
  </main>`;

  return renderDocument({
    title: `${privacy.title} · The Tiny Exhibit`,
    description: privacy.summary,
    path: privacy.href,
    current: "work",
    body,
    bodyClass: "article-document privacy-document",
  });
}
