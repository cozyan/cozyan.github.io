export function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

export function escapeXml(value) {
  return escapeHtml(value);
}

export function formatDate(dateString, { short = false } = {}) {
  const date = new Date(`${dateString}T12:00:00Z`);
  return new Intl.DateTimeFormat("en", short
    ? { year: "numeric" }
    : { day: "numeric", month: "long", year: "numeric" }).format(date);
}

export function renderExternalAttributes() {
  return 'target="_blank" rel="noopener noreferrer"';
}
