import { renderDocument, renderPageHeading } from "../components/layout.mjs";
import { renderWritingList } from "../components/writing.mjs";

export function renderWritingIndexPage({ articles }) {
  const body = `<main id="main">
    <section class="page-shell archive-hero">
      ${renderPageHeading({
        eyebrow: `${articles.length} pieces in the archive`,
        title: "Notes on technology,<br>life, and <em>stories.</em>",
        description: "I write when a question stays with me. The subjects change. The need to understand them does not.",
      })}
    </section>
    <section class="page-shell writing-archive" aria-label="Writing archive">
      ${renderWritingList(articles, { heading: false })}
    </section>
  </main>`;

  return renderDocument({
    title: "Writing",
    description: "Essays and stories by Yan Liang about technology, life, and the questions between them.",
    path: "/writing/",
    current: "writing",
    body,
    bodyClass: "writing-index-page",
  });
}
