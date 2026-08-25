import MarkdownIt from "markdown-it";

const markdown = new MarkdownIt({
  html: false,
  linkify: true,
  typographer: true,
});

const defaultLinkOpen = markdown.renderer.rules.link_open
  ?? ((tokens, index, options, environment, renderer) => renderer.renderToken(tokens, index, options));

markdown.renderer.rules.link_open = (tokens, index, options, environment, renderer) => {
  const href = tokens[index].attrGet("href") ?? "";
  if (/^https?:\/\//.test(href)) {
    tokens[index].attrSet("target", "_blank");
    tokens[index].attrSet("rel", "noopener noreferrer");
  }
  return defaultLinkOpen(tokens, index, options, environment, renderer);
};

export function renderMarkdown(source) {
  return markdown.render(source);
}

export function countWords(source) {
  return source
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/[#>*_`\[\]()!-]/g, " ")
    .trim()
    .split(/\s+/)
    .filter(Boolean).length;
}
