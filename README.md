# Yan Liang portfolio

A single-page portfolio built with HTML, CSS and a small amount of JavaScript. There is no frontend framework or runtime dependency.

## Project structure

- `src/content.mjs` stores project and writing content
- `src/components.mjs` contains reusable static HTML renderers
- `src/page.mjs` composes the page
- `scripts/build.mjs` generates `index.html`
- `styles.css` contains the shared visual system
- `script.js` handles reveal effects and the footer year

Edit files in `src/` instead of editing the generated `index.html` directly.

## Preview locally

```sh
npm run build
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

The `main` branch builds and deploys through `.github/workflows/deploy.yml`.
