/**
 * Citation linkification for RPG source references.
 * Converts [Book Name, p.XX] patterns into styled, highlighted spans.
 */

const CITATION_RE = /\[([^\]]+?,\s*p\.\s*(\d+))\]/g;

function linkifyCitations(html) {
  return html.replace(CITATION_RE, (match, citationKey, page) => {
    return `<span class="inline-block bg-amber-900/40 text-amber-300 px-1.5 py-0.5 rounded text-sm cursor-help border border-amber-700/50" title="${citationKey}">[${citationKey}]</span>`;
  });
}
