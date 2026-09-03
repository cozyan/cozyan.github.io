import { escapeHtml, formatDate } from "../lib/html.mjs";

export function renderArticleInterlude(article) {
  return `<aside class="article-interlude reveal" data-preview-theme="${escapeHtml(article.topic.toLowerCase())}">
    <p class="eyebrow">A pause for writing · ${escapeHtml(article.topic)}</p>
    <blockquote>${escapeHtml(article.preview)}</blockquote>
    <a href="${article.href}">${escapeHtml(article.title)} <span aria-hidden="true">↗</span></a>
  </aside>`;
}

export function renderWritingList(articles, { heading = true } = {}) {
  const rows = articles.map((article, index) => `<a class="writing-row${index === 0 ? " is-active" : ""}" href="${article.href}" data-writing-preview="${escapeHtml(article.preview)}" data-writing-topic="${escapeHtml(article.topic)}">
    <time datetime="${article.published}">${formatDate(article.published, { short: true })}</time>
    <span class="writing-row-main"><strong style="view-transition-name: article-${escapeHtml(article.slug)}">${escapeHtml(article.title)}</strong><small>${escapeHtml(article.summary)}</small></span>
    <span class="writing-row-meta">${escapeHtml(article.topic)} · ${article.readingMinutes} min</span>
  </a>`).join("");

  return `<div class="writing-list-layout">
    <div class="writing-list-column">
      ${heading ? `<header class="writing-list-heading"><p class="eyebrow">Latest writing</p><h2>I write to understand things.</h2><p>Sometimes that means technology. Sometimes life. Sometimes a story.</p></header>` : ""}
      <div class="writing-list">${rows}</div>
    </div>
    <aside class="writing-preview" aria-live="polite" aria-label="Selected article preview">
      <div class="writing-preview-top"><span data-preview-topic>${escapeHtml(articles[0]?.topic ?? "Writing")}</span><span>Preview</span></div>
      <p data-preview-copy>${escapeHtml(articles[0]?.preview ?? "")}</p>
      <span class="writing-preview-hint" data-preview-hint>Choose a title to read another passage</span>
    </aside>
  </div>`;
}

export function renderRelatedArticle(article) {
  if (!article) return "";
  return `<aside class="related-content"><p class="eyebrow">Continue reading</p><a href="${article.href}"><strong>${escapeHtml(article.title)}</strong><span>${escapeHtml(article.summary)}</span></a></aside>`;
}

export function renderArticleBody(article) {
  return `<article class="article-page" data-article-slug="${escapeHtml(article.slug)}">
    <header class="article-header page-shell">
      <a class="back-link" href="/writing/">All writing</a>
      <p class="eyebrow reveal">${escapeHtml(article.topic)}</p>
      <h1 class="reveal" style="view-transition-name: article-${escapeHtml(article.slug)}">${escapeHtml(article.title)}</h1>
      <p class="article-summary reveal">${escapeHtml(article.summary)}</p>
      <div class="article-byline reveal"><time datetime="${article.published}">${formatDate(article.published)}</time><span>${article.readingMinutes} minute read</span>${article.editorialPreview ? "<span>Editorial preview</span>" : ""}</div>
    </header>
    <div class="article-prose reveal">${article.html}</div>
  </article>`;
}
