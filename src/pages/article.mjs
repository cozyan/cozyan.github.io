import { escapeHtml } from "../lib/html.mjs";
import { renderDocument } from "../components/layout.mjs";
import { renderProjectVisual } from "../components/project.mjs";
import { renderArticleBody } from "../components/writing.mjs";

function renderArticleEnd(article, articles, projects) {
  const project = article.relatedProject ? projects.find((entry) => entry.id === article.relatedProject) : null;
  const currentIndex = articles.findIndex((entry) => entry.slug === article.slug);
  const nextArticle = articles[(currentIndex + 1) % articles.length];

  return `<section class="article-end page-shell">
    ${project ? `<a class="article-project" href="/work/${project.id}/"><div><p class="eyebrow">Related product</p><h2>${escapeHtml(project.title)}</h2><p>${escapeHtml(project.summary)}</p><span>Read the product story ↗</span></div>${renderProjectVisual(project, { compact: true })}</a>` : ""}
    ${nextArticle && nextArticle.slug !== article.slug ? `<a class="next-article" href="${nextArticle.href}"><p class="eyebrow">Read next</p><strong>${escapeHtml(nextArticle.title)}</strong><span>${escapeHtml(nextArticle.topic)} · ${nextArticle.readingMinutes} min</span></a>` : ""}
  </section>`;
}

export function renderArticlePage({ article, articles, projects }) {
  const body = `<main id="main">${renderArticleBody(article)}${renderArticleEnd(article, articles, projects)}</main>`;
  return renderDocument({
    title: article.title,
    description: article.summary,
    path: article.href,
    current: "writing",
    body,
    bodyClass: "article-document",
  });
}
