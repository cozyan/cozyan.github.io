# Yan Liang portfolio

A static product portfolio and writing archive. Projects and articles are written in Markdown. The published site has no frontend framework or runtime dependency.

## Project structure

- `src/content/projects/` stores product stories as Markdown
- `src/content/writing/` stores local articles as Markdown
- `src/components/` contains reusable HTML renderers
- `src/pages/` contains page templates
- `styles/` contains the editorial design system
- `js/main.js` handles previews, reveals, and small progressive enhancements
- `scripts/build.mjs` generates the complete site in `_site/`

Add a Markdown file to the matching content folder to publish a new project or article. The build validates required metadata, clean slugs, asset paths, and relationships between content.

Each article starts with this frontmatter:

```md
---
title: "A specific article title"
slug: "a-specific-article-title"
summary: "One clear sentence for previews and search results."
published: "2026-08-25"
topic: "Technology"
preview: "The line shown in the writing preview panel."
relatedProject: "creative-graph"
featured: false
draft: false
---
```

Set `draft: true` to keep an article out of the generated site.

Projects use the same workflow. Create a file in `src/content/projects/` and write the project story below its frontmatter:

```md
---
title: "The Tiny Exhibit"
slug: "the-tiny-exhibit"
category: "Chrome extension"
summary: "One clear sentence for previews and search results."
observation: "The problem that started the project."
response: "What you made in response."
stage: "Preparing for Chrome Web Store"
order: 1
featured: true
draft: false
visual: "image"
cover: "/assets/projects/example.png"
coverAlt: "A useful description of the project image"
coverWidth: 1280
coverHeight: 800
facts:
  - label: "Role"
    value: "Product design and development"
---

## The first part of the story

Write the full case study here.
```

Published projects are sorted by `order` and numbered automatically. Projects with `featured: true` appear on the homepage. Set `draft: true` to keep a project out of public pages, related content, and the sitemap.

## Preview locally

```sh
npm run build
python3 -m http.server 8000 --directory _site
```

Then open `http://localhost:8000`.

To preview drafts locally, use `npm run build:drafts`. This creates draft pages in `_site` but does not add them to the feed or public sitemap.

The `main` branch builds and deploys through `.github/workflows/deploy.yml`.
