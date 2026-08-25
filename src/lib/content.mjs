import { access, readdir, readFile } from "node:fs/promises";
import { dirname, extname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import matter from "gray-matter";
import { countWords, renderMarkdown } from "./markdown.mjs";

const libraryDirectory = dirname(fileURLToPath(import.meta.url));
const sourceRoot = resolve(libraryDirectory, "..");
const projectRoot = resolve(sourceRoot, "..");
const writingDirectory = resolve(sourceRoot, "content/writing");
const projectsDirectory = resolve(sourceRoot, "content/projects");

function requireText(value, field, filename) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new Error(`${filename}: ${field} must be a non-empty string`);
  }
}

function validateDate(value, filename) {
  requireText(value, "published", filename);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value) || Number.isNaN(Date.parse(`${value}T00:00:00Z`))) {
    throw new Error(`${filename}: published must use YYYY-MM-DD`);
  }
}

function validateSlug(value, filename) {
  requireText(value, "slug", filename);
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(value)) {
    throw new Error(`${filename}: slug must contain lowercase words separated by hyphens`);
  }
}

async function validateAsset(assetPath, filename) {
  if (!assetPath) return;
  const localPath = resolve(projectRoot, assetPath.replace(/^\//, ""));
  try {
    await access(localPath);
  } catch {
    throw new Error(`${filename}: missing asset ${assetPath}`);
  }
}

async function parseArticle(filename) {
  const source = await readFile(resolve(writingDirectory, filename), "utf8");
  const { data, content } = matter(source);

  for (const field of ["title", "slug", "summary", "published", "topic", "preview"]) {
    requireText(data[field], field, filename);
  }
  validateDate(data.published, filename);

  validateSlug(data.slug, filename);
  if (data.cover && !data.coverAlt) {
    throw new Error(`${filename}: coverAlt is required when cover is set`);
  }
  if (data.cover && (!Number.isInteger(data.coverWidth) || !Number.isInteger(data.coverHeight) || data.coverWidth < 1 || data.coverHeight < 1)) {
    throw new Error(`${filename}: coverWidth and coverHeight must be positive integers when cover is set`);
  }
  await validateAsset(data.cover, filename);

  const wordCount = countWords(content);
  return {
    ...data,
    draft: data.draft === true,
    featured: data.featured === true,
    editorialPreview: data.editorialPreview === true,
    body: content,
    html: renderMarkdown(content),
    wordCount,
    readingMinutes: Math.max(1, Math.ceil(wordCount / 220)),
    outputPath: `writing/${data.slug}/index.html`,
    href: `/writing/${data.slug}/`,
  };
}

export async function loadArticles({ includeDrafts = false } = {}) {
  const filenames = (await readdir(writingDirectory))
    .filter((filename) => extname(filename) === ".md")
    .sort();
  const articles = await Promise.all(filenames.map(parseArticle));

  const seenSlugs = new Set();
  for (const article of articles) {
    if (seenSlugs.has(article.slug)) throw new Error(`Duplicate article slug: ${article.slug}`);
    seenSlugs.add(article.slug);

  }

  return articles
    .filter((article) => includeDrafts || !article.draft)
    .sort((first, second) => second.published.localeCompare(first.published));
}

async function parseProject(filename) {
  const source = await readFile(resolve(projectsDirectory, filename), "utf8");
  const { data, content } = matter(source);

  for (const field of ["title", "slug", "category", "summary", "observation", "response", "stage", "visual"]) {
    requireText(data[field], field, filename);
  }
  validateSlug(data.slug, filename);

  if (!Number.isInteger(data.order) || data.order < 1) {
    throw new Error(`${filename}: order must be a positive integer`);
  }
  if (!Array.isArray(data.facts) || data.facts.length === 0) {
    throw new Error(`${filename}: facts must contain at least one label and value`);
  }
  for (const [index, fact] of data.facts.entries()) {
    requireText(fact?.label, `facts[${index}].label`, filename);
    requireText(fact?.value, `facts[${index}].value`, filename);
  }
  if (data.cover && !data.coverAlt) {
    throw new Error(`${filename}: coverAlt is required when cover is set`);
  }
  if (data.visual === "image" && !data.cover) {
    throw new Error(`${filename}: cover is required when visual is image`);
  }
  if (data.externalUrl && !data.externalLabel) {
    throw new Error(`${filename}: externalLabel is required when externalUrl is set`);
  }
  await validateAsset(data.cover, filename);

  return {
    ...data,
    id: data.slug,
    draft: data.draft === true,
    featured: data.featured === true,
    facts: data.facts.map(({ label, value }) => [label, value]),
    body: content,
    html: renderMarkdown(content),
    outputPath: `work/${data.slug}/index.html`,
    href: `/work/${data.slug}/`,
  };
}

export async function loadProjects({ includeDrafts = false } = {}) {
  const filenames = (await readdir(projectsDirectory))
    .filter((filename) => extname(filename) === ".md")
    .sort();
  const projects = await Promise.all(filenames.map(parseProject));
  const projectIds = new Set();

  for (const project of projects) {
    if (projectIds.has(project.id)) throw new Error(`Duplicate project id: ${project.id}`);
    projectIds.add(project.id);
  }

  return projects
    .filter((project) => includeDrafts || !project.draft)
    .sort((first, second) => first.order - second.order)
    .map((project, index) => ({ ...project, number: String(index + 1).padStart(2, "0") }));
}

export async function loadProjectPrivacy(projectSlug) {
  validateSlug(projectSlug, "project privacy path");
  const filename = resolve(projectsDirectory, projectSlug, "privacy.md");
  const source = await readFile(filename, "utf8");
  const { data, content } = matter(source);

  for (const field of ["title", "summary", "updated"]) {
    requireText(data[field], field, filename);
  }

  return {
    ...data,
    projectSlug,
    html: renderMarkdown(content),
    outputPath: `work/${projectSlug}/privacy/index.html`,
    href: `/work/${projectSlug}/privacy/`,
  };
}

export function validateContentRelationships(projects, articles) {
  const articleIds = new Set(articles.map((article) => article.slug));
  const projectIds = new Set(projects.map((project) => project.id));

  for (const project of projects) {
    if (project.relatedWriting && !articleIds.has(project.relatedWriting)) {
      throw new Error(`${project.id}: unknown related article ${project.relatedWriting}`);
    }
  }
  for (const article of articles) {
    if (article.relatedProject && !projectIds.has(article.relatedProject)) {
      throw new Error(`${article.slug}: unknown related project ${article.relatedProject}`);
    }
  }
}
