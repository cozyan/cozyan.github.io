import { navigation, site } from "../content/site.mjs";
import { escapeHtml, renderExternalAttributes } from "../lib/html.mjs";

export function renderDocument({ title, description, path, current, body, bodyClass = "" }) {
  const pageTitle = title === site.title ? title : `${title} | ${site.name}`;
  const canonicalUrl = `${site.url}${path}`;

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="${escapeHtml(description)}">
  <meta name="theme-color" content="#07151b">
  <meta name="color-scheme" content="dark">
  <meta property="og:title" content="${escapeHtml(pageTitle)}">
  <meta property="og:description" content="${escapeHtml(description)}">
  <meta property="og:type" content="website">
  <meta property="og:url" content="${escapeHtml(canonicalUrl)}">
  <meta property="og:image" content="${site.url}/assets/avatar.png">
  <link rel="canonical" href="${escapeHtml(canonicalUrl)}">
  <link rel="alternate" type="application/rss+xml" title="${site.name} writing" href="/feed.xml">
  <link rel="preload" href="/assets/fonts/newsreader-latin.woff2" as="font" type="font/woff2" crossorigin>
  <link rel="preload" href="/assets/fonts/instrument-sans-latin.woff2" as="font" type="font/woff2" crossorigin>
  <title>${escapeHtml(pageTitle)}</title>
  <link rel="icon" href="/assets/avatar.png">
  <style>html,body{background-color:#07151b}</style>
  <link rel="stylesheet" href="/styles.css?v=${site.assetVersion}">
  <script src="/script.js?v=${site.assetVersion}" defer></script>
</head>
<body class="${escapeHtml(bodyClass)}">
  <a class="skip-link" href="#main">Skip to content</a>
  ${renderSiteHeader(current)}
  ${body}
  ${renderFooter()}
</body>
</html>`;
}

export function renderSiteHeader(current = "home") {
  const links = navigation.map((item) => {
    const currentAttribute = item.id === current ? ' aria-current="page"' : "";
    return `<a href="${item.href}"${currentAttribute}>${escapeHtml(item.label)}</a>`;
  }).join("");

  return `<header class="site-header">
    <nav class="site-nav" aria-label="Main navigation">
      <a class="brand" href="/" aria-label="Yan Liang home"><span class="brand-mark" aria-hidden="true">YL</span><span class="brand-name">Yan Liang</span></a>
      <div class="nav-links">${links}</div>
    </nav>
  </header>`;
}

export function renderFooter() {
  return `<footer class="site-footer page-shell">
    <p>© <span data-year></span> ${site.name}</p>
    <p>Designed and written in ${site.location}</p>
  </footer>`;
}

export function renderContactSection({ centered = false } = {}) {
  const eyebrow = centered ? "Say hello" : "A final note";
  const heading = centered ? "Have an idea?<br><em>Say hello.</em>" : "Have something<br>worth <em>making?</em>";
  const copy = centered
    ? "I am happy to talk about products and stories. You can also show me what you are making."
    : "I am always happy to hear about a thoughtful product, a difficult question, or a story still taking shape.";

  return `<section class="contact${centered ? " contact-centered" : ""}" id="contact" aria-labelledby="contact-title">
    <div class="page-shell contact-inner reveal">
      <p class="eyebrow">${eyebrow}</p>
      <h2 id="contact-title">${heading}</h2>
      <p>${copy}</p>
      <div class="contact-links">
        <a class="button button-dark" href="${site.social.linkedin}" ${renderExternalAttributes()}>Find me on LinkedIn</a>
        <a class="text-link" href="${site.social.github}" ${renderExternalAttributes()}>Visit my GitHub</a>
      </div>
    </div>
  </section>`;
}

export function renderPageHeading({ eyebrow, title, description }) {
  return `<header class="page-heading reveal">
    <p class="eyebrow">${escapeHtml(eyebrow)}</p>
    <h1>${title}</h1>
    <p>${escapeHtml(description)}</p>
  </header>`;
}
