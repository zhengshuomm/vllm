'use strict';

function decodeEntities(str) {
  return String(str || '')
    .replace(/&nbsp;/gi, ' ')
    .replace(/&amp;/gi, '&')
    .replace(/&lt;/gi, '<')
    .replace(/&gt;/gi, '>')
    .replace(/&quot;/gi, '"')
    .replace(/&#39;|&apos;/gi, "'")
    .replace(/&#(\d+);/g, (_, n) => String.fromCodePoint(Number(n)))
    .replace(/&#x([0-9a-f]+);/gi, (_, h) => String.fromCodePoint(parseInt(h, 16)));
}

function parseAttrs(tag) {
  const attrs = {};
  const re = /([a-zA-Z0-9_:-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))/g;
  let m;
  while ((m = re.exec(tag)) !== null) {
    attrs[m[1].toLowerCase()] = decodeEntities(m[2] ?? m[3] ?? m[4] ?? '');
  }
  return attrs;
}

function stripTags(html) {
  return decodeEntities(
    String(html || '')
      .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, ' ')
      .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, ' ')
      .replace(/<noscript\b[^>]*>[\s\S]*?<\/noscript>/gi, ' ')
      .replace(/<svg\b[^>]*>[\s\S]*?<\/svg>/gi, ' ')
      .replace(/<!--[\s\S]*?-->/g, ' ')
      .replace(/<[^>]+>/g, ' ')
      .replace(/\s+/g, ' ')
      .trim()
  );
}

function extractSchemaTypes(node, types, orgData) {
  if (!node || typeof node !== 'object') return;
  if (Array.isArray(node)) {
    for (const item of node) extractSchemaTypes(item, types, orgData);
    return;
  }
  if (node['@type']) {
    const list = Array.isArray(node['@type']) ? node['@type'] : [node['@type']];
    for (const t of list) {
      if (typeof t === 'string') {
        types.add(t);
        if (/^(Organization|Corporation|LocalBusiness|OnlineBusiness|SoftwareApplication)$/i.test(t)) {
          if (!orgData.name && node.name) orgData.name = String(node.name);
          if (!orgData.logo && node.logo) {
            orgData.logo = typeof node.logo === 'string' ? node.logo : node.logo.url || '';
          }
          if (node.sameAs) {
            const arr = Array.isArray(node.sameAs) ? node.sameAs : [node.sameAs];
            for (const s of arr) if (typeof s === 'string') orgData.sameAs.add(s);
          }
        }
      }
    }
  }
  if (node['@graph']) extractSchemaTypes(node['@graph'], types, orgData);
  for (const [k, v] of Object.entries(node)) {
    if (k !== '@graph' && v && typeof v === 'object') {
      extractSchemaTypes(v, types, orgData);
    }
  }
}

const QUESTION_RE = /^(what|how|why|when|where|who|which|can|does|is|are|should|do)\b|\?$/i;
const SOCIAL_RE = /https?:\/\/(?:www\.)?(linkedin\.com|x\.com|twitter\.com|github\.com|youtube\.com|instagram\.com|facebook\.com|crunchbase\.com|wikipedia\.org|wikidata\.org)\/[^\s"'<>]+/i;

function analyzeHtml(html, baseUrl) {
  const raw = String(html || '');
  const origin = new URL(baseUrl).origin;
  const host = new URL(baseUrl).hostname.replace(/^www\./i, '');

  const titleMatch = raw.match(/<title\b[^>]*>([\s\S]*?)<\/title>/i);
  const title = titleMatch ? decodeEntities(titleMatch[1].replace(/\s+/g, ' ').trim()) : '';

  const metas = {};
  const metaRobots = [];
  const metaTagRe = /<meta\b[^>]*>/gi;
  let m;
  while ((m = metaTagRe.exec(raw)) !== null) {
    const a = parseAttrs(m[0]);
    const key = (a.name || a.property || '').toLowerCase();
    const content = a.content || '';
    if (!key) continue;
    metas[key] = content;
    if (/^(robots|googlebot|bingbot|gptbot|perplexitybot)$/i.test(key)) {
      metaRobots.push({ name: key, content });
    }
  }

  let canonical = '';
  const linkTagRe = /<link\b[^>]*>/gi;
  while ((m = linkTagRe.exec(raw)) !== null) {
    const a = parseAttrs(m[0]);
    if ((a.rel || '').toLowerCase().split(/\s+/).includes('canonical') && a.href) {
      canonical = a.href;
      break;
    }
  }

  // Restrictive AI/snippet directives
  const restrictiveDirectives = [];
  for (const mr of metaRobots) {
    const lower = mr.content.toLowerCase();
    for (const bad of ['noindex', 'nosnippet', 'noai', 'noimageai', 'max-snippet:0']) {
      if (lower.includes(bad)) {
        restrictiveDirectives.push(`${mr.name}: ${mr.content}`);
        break;
      }
    }
  }

  // JSON-LD structured data
  const jsonLdRe = /<script\b[^>]*type\s*=\s*["']application\/ld\+json["'][^>]*>([\s\S]*?)<\/script>/gi;
  const schemaTypes = new Set();
  const orgData = { name: '', logo: '', sameAs: new Set() };
  let jsonLdBlocks = 0;
  let jsonLdInvalid = 0;
  while ((m = jsonLdRe.exec(raw)) !== null) {
    jsonLdBlocks++;
    try {
      const parsed = JSON.parse(m[1].trim());
      extractSchemaTypes(parsed, schemaTypes, orgData);
    } catch {
      jsonLdInvalid++;
    }
  }

  // Headings
  const headings = { h1: [], h2: [], h3: [] };
  const hRe = /<(h[1-3])\b[^>]*>([\s\S]*?)<\/\1>/gi;
  while ((m = hRe.exec(raw)) !== null) {
    const level = m[1].toLowerCase();
    const text = stripTags(m[2]);
    if (text) headings[level].push(text);
  }
  const allSubheadings = [...headings.h2, ...headings.h3];
  const questionHeadings = allSubheadings.filter(h => QUESTION_RE.test(h));

  // Paragraphs & first substantial block
  const paragraphs = [];
  const pRe = /<p\b[^>]*>([\s\S]*?)<\/p>/gi;
  while ((m = pRe.exec(raw)) !== null) {
    const text = stripTags(m[1]);
    if (text.length >= 25) paragraphs.push(text);
  }
  const hasStatsOrNumbers = paragraphs.some(p => /\b\d+(?:[.,]\d+)?\s*(?:%|x|times|million|billion|hours|days|users|customers|companies|\+|k\b)/i.test(p));
  const hasTablesOrLists = /<(?:table|ul|ol|dl)\b/i.test(raw);
  const hasFaqSignals =
    schemaTypes.has('FAQPage') ||
    questionHeadings.length >= 2 ||
    /\bfaq\b|frequently asked questions/i.test(raw);

  // Visible text & SSR/SPA check
  const bodyMatch = raw.match(/<body\b[^>]*>([\s\S]*?)<\/body>/i);
  const visibleText = stripTags(bodyMatch ? bodyMatch[1] : raw);
  const words = visibleText ? visibleText.split(/\s+/).filter(Boolean) : [];
  const wordCount = words.length;
  const htmlSizeBytes = Buffer.byteLength(raw, 'utf8');
  const textToHtmlRatio = htmlSizeBytes > 0 ? +(visibleText.length / htmlSizeBytes).toFixed(3) : 0;

  const hasSpaRoot = /<div\b[^>]*id\s*=\s*["'](?:root|app|__next|__nuxt)["'][^>]*>\s*<\/div>/i.test(raw);
  const likelyCsrShell = wordCount < 80 && (hasSpaRoot || /enable javascript|you need to enable javascript/i.test(raw));

  // Links (internal key pages + social/sameAs + external citations)
  const internalLinks = new Map();
  const socialLinks = new Set([...orgData.sameAs]);
  const externalDomains = new Set();
  const aRe = /<a\b[^>]*href\s*=\s*(?:"([^"]*)"|'([^']*)')[^>]*>([\s\S]*?)<\/a>/gi;
  while ((m = aRe.exec(raw)) !== null) {
    const href = decodeEntities((m[1] ?? m[2] ?? '').trim());
    const anchor = stripTags(m[3]).slice(0, 80);
    if (!href || href.startsWith('#') || href.startsWith('mailto:') || href.startsWith('tel:') || href.startsWith('javascript:')) continue;
    let resolved;
    try {
      resolved = new URL(href, origin);
    } catch {
      continue;
    }
    if (resolved.origin === origin) {
      const cleanPath = resolved.pathname.replace(/\/+$/, '') || '/';
      if (cleanPath !== '/' && !/\.(png|jpg|jpeg|gif|svg|webp|ico|css|js|pdf|zip)$/i.test(cleanPath)) {
        const full = `${origin}${cleanPath}`;
        if (!internalLinks.has(full) && anchor && anchor.length >= 2) {
          internalLinks.set(full, anchor);
        }
      }
    } else {
      const extHost = resolved.hostname.replace(/^www\./i, '');
      if (extHost !== host) externalDomains.add(extHost);
      const sm = resolved.href.match(SOCIAL_RE);
      if (sm) socialLinks.add(resolved.href.split('?')[0]);
    }
  }

  // HTML lang & timestamps
  const langMatch = raw.match(/<html\b[^>]*\blang\s*=\s*["']([^"']+)["']/i);
  const lang = langMatch ? langMatch[1] : '';
  const hasDateSignal = Boolean(
    metas['article:modified_time'] ||
    metas['article:published_time'] ||
    metas['last-modified'] ||
    /"dateModified"\s*:/.test(raw) ||
    /"datePublished"\s*:/.test(raw)
  );
  const hasAuthorSignal = Boolean(metas.author || /"author"\s*:/.test(raw));

  return {
    title,
    description: metas.description || metas['og:description'] || '',
    ogTitle: metas['og:title'] || '',
    ogDescription: metas['og:description'] || '',
    ogImage: metas['og:image'] || '',
    twitterCard: metas['twitter:card'] || '',
    canonical,
    lang,
    metaRobots,
    restrictiveDirectives,
    jsonLdBlocks,
    jsonLdInvalid,
    schemaTypes: [...schemaTypes],
    orgData: { name: orgData.name, logo: orgData.logo, sameAs: [...orgData.sameAs] },
    headings,
    questionHeadings,
    paragraphsCount: paragraphs.length,
    leadParagraph: paragraphs[0] || '',
    wordCount,
    htmlSizeBytes,
    textToHtmlRatio,
    likelyCsrShell,
    hasStatsOrNumbers,
    hasTablesOrLists,
    hasFaqSignals,
    hasDateSignal,
    hasAuthorSignal,
    internalLinks: [...internalLinks.entries()].slice(0, 25).map(([url, label]) => ({ url, label })),
    socialLinks: [...socialLinks].slice(0, 10),
    externalDomainsCount: externalDomains.size,
  };
}

module.exports = { analyzeHtml, stripTags, decodeEntities };
