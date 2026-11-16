'use strict';

const { auditRobots } = require('./robots');
const { analyzeHtml } = require('./html');

const DEFAULT_UA = 'Mozilla/5.0 (compatible; GEOAuditBot/1.0; +https://schema.org)';
const FETCH_TIMEOUT_MS = 12000;

function normalizeUrl(input) {
  let raw = String(input || '').trim();
  if (!raw) throw new Error('Website URL is required.');
  if (!/^https?:\/\//i.test(raw)) raw = `https://${raw}`;
  const u = new URL(raw);
  if (u.protocol !== 'http:' && u.protocol !== 'https:') {
    throw new Error('Only http:// and https:// URLs are supported.');
  }
  return u;
}

async function safeFetch(url, { timeoutMs = FETCH_TIMEOUT_MS, maxBytes = 1_500_000, fetchImpl = fetch } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const started = Date.now();
  try {
    const res = await fetchImpl(url, {
      redirect: 'follow',
      signal: controller.signal,
      headers: {
        'User-Agent': DEFAULT_UA,
        Accept: 'text/html,application/xhtml+xml,text/plain,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
      },
    });
    const contentType = res.headers.get('content-type') || '';
    const xRobotsTag = res.headers.get('x-robots-tag') || '';
    const buf = Buffer.from(await res.arrayBuffer());
    const text = buf.subarray(0, maxBytes).toString('utf8');
    return {
      ok: res.ok,
      status: res.status,
      finalUrl: res.url || url,
      contentType,
      xRobotsTag,
      elapsedMs: Date.now() - started,
      text,
      error: null,
    };
  } catch (err) {
    return {
      ok: false,
      status: 0,
      finalUrl: url,
      contentType: '',
      xRobotsTag: '',
      elapsedMs: Date.now() - started,
      text: '',
      error: err.name === 'AbortError' ? `Timeout after ${timeoutMs}ms` : err.message,
    };
  } finally {
    clearTimeout(timer);
  }
}

function isRealTextFile(res) {
  if (!res.ok) return false;
  const trimmed = (res.text || '').trim();
  if (!trimmed) return false;
  // Many SPAs return index.html with HTTP 200 for unknown paths like /llms.txt
  if (/^<!doctype\s+html|^<html\b/i.test(trimmed)) return false;
  if (/text\/html/i.test(res.contentType) && /<head\b|<body\b/i.test(trimmed)) return false;
  return true;
}

function parseLlmsTxt(res) {
  if (!isRealTextFile(res)) {
    return { present: false, status: res.status, hasH1: false, hasSummary: false, linkCount: 0, sections: 0, length: 0 };
  }
  const text = res.text;
  const hasH1 = /^#\s+\S+/m.test(text);
  const hasSummary = /^>\s+\S+/m.test(text);
  const links = text.match(/\[[^\]]+\]\(https?:\/\/[^)]+\)/g) || [];
  const sections = (text.match(/^##\s+\S+/gm) || []).length;
  return {
    present: true,
    status: res.status,
    hasH1,
    hasSummary,
    linkCount: links.length,
    sections,
    length: text.length,
    preview: text.slice(0, 600),
  };
}

function parseSitemap(res) {
  if (!res.ok) return { present: false, status: res.status, urlCount: 0, sampleUrls: [] };
  const text = res.text || '';
  if (!/<urlset\b|<sitemapindex\b/i.test(text)) {
    return { present: false, status: res.status, urlCount: 0, sampleUrls: [] };
  }
  const locs = [...text.matchAll(/<loc>\s*([^<\s]+)\s*<\/loc>/gi)].map(m => m[1]);
  return {
    present: true,
    status: res.status,
    isIndex: /<sitemapindex\b/i.test(text),
    urlCount: locs.length,
    sampleUrls: locs.slice(0, 15),
  };
}

function inferBrand(htmlAnalysis, originUrl, explicitBrand) {
  if (explicitBrand && explicitBrand.trim()) return explicitBrand.trim();
  if (htmlAnalysis.orgData.name) return htmlAnalysis.orgData.name;
  if (htmlAnalysis.title) {
    const split = htmlAnalysis.title.split(/\s+[|\-–—:]\s+/);
    const candidate = split[0].trim();
    if (candidate.length >= 2 && candidate.length <= 40) return candidate;
  }
  const host = originUrl.hostname.replace(/^www\./i, '').split('.')[0];
  return host.charAt(0).toUpperCase() + host.slice(1);
}

function scoreAndBuildFindings({ html, robots, llms, llmsFull, sitemap, pageRes }) {
  const findings = [];
  const addFinding = (priority, pillar, title, impact, action) => {
    findings.push({ priority, pillar, title, impact, action });
  };

  // ----- Pillar 1: AI Crawler Access (25 pts) -----
  let p1 = 25;
  const criticalBlocked = robots.blockedBots.filter(b =>
    ['GPTBot', 'OAI-SearchBot', 'PerplexityBot', 'ClaudeBot', 'Googlebot'].includes(b)
  );
  if (!robots.present) {
    p1 -= 4;
    addFinding(
      'P1',
      'AI Crawler Access',
      'Missing robots.txt file',
      'AI crawlers have no explicit crawl guidance or sitemap pointer.',
      'Publish /robots.txt with explicit Allow rules for AI retrieval bots and a Sitemap directive.'
    );
  }
  if (criticalBlocked.length > 0) {
    p1 -= Math.min(18, criticalBlocked.length * 6);
    addFinding(
      'P0',
      'AI Crawler Access',
      `Blocking ${criticalBlocked.length} major AI crawler(s): ${criticalBlocked.join(', ')}`,
      'Blocked AI bots cannot crawl or cite your website in ChatGPT Search, Perplexity, Claude, or Google AI Overviews.',
      'Deploy the included robots-ai-snippet.txt patch to explicitly allow AI retrieval bots on public marketing and documentation pages.'
    );
  } else if (robots.present && robots.explicitAiRulesCount === 0) {
    p1 -= 3;
    addFinding(
      'P2',
      'AI Crawler Access',
      'No explicit AI User-Agent directives in robots.txt',
      'Relying only on wildcard rules leaves AI crawler permissions ambiguous when security teams or CDNs update rules.',
      'Add explicit Allow blocks for GPTBot, OAI-SearchBot, PerplexityBot, ClaudeBot, and Google-Extended.'
    );
  }
  const xRobotsRestrictive = /noindex|nosnippet|noai|max-snippet:0/i.test(pageRes.xRobotsTag || '');
  if (html.restrictiveDirectives.length > 0 || xRobotsRestrictive) {
    p1 -= 10;
    addFinding(
      'P0',
      'AI Crawler Access',
      'Restrictive meta/header directives prevent AI quotation',
      `Detected directives (${[...html.restrictiveDirectives, xRobotsRestrictive ? `X-Robots-Tag: ${pageRes.xRobotsTag}` : ''].filter(Boolean).join('; ')}) tell search engines not to index or quote snippets.`,
      'Remove noindex, nosnippet, noai, or max-snippet:0 from public pages that should appear in AI answers.'
    );
  }
  p1 = Math.max(0, p1);

  // ----- Pillar 2: AI-Native Discovery (20 pts) -----
  let p2 = 0;
  if (llms.present) {
    p2 += 8;
    if (llms.hasH1 && llms.hasSummary) p2 += 3;
    else {
      addFinding(
        'P2',
        'AI Discovery',
        'llms.txt exists but is missing standard H1 or blockquote summary',
        'LLMs parse the H1 + blockquote summary first to understand brand positioning.',
        'Format /llms.txt with `# Brand`, a 2-sentence `> summary`, and categorized markdown links.'
      );
    }
    if (llms.linkCount >= 3) p2 += 2;
  } else {
    addFinding(
      'P0',
      'AI Discovery',
      'Missing /llms.txt AI manifest file',
      'LLMs have no curated markdown index of your core value proposition, product pages, pricing, and documentation.',
      'Upload the auto-generated llms.txt file from this delivery pack to your website root (/llms.txt).'
    );
  }
  if (llmsFull.present) {
    p2 += 2;
  } else {
    addFinding(
      'P2',
      'AI Discovery',
      'No /llms-full.txt deep context file',
      'Technical or B2B buyers benefit when AI assistants can read consolidated product docs in one fetch.',
      'Optionally publish /llms-full.txt consolidating key product specs, pricing, and FAQ answers in clean markdown.'
    );
  }
  if (sitemap.present) {
    p2 += 5;
  } else {
    addFinding(
      'P1',
      'AI Discovery',
      'XML sitemap not detected at /sitemap.xml or in robots.txt',
      'Search and AI retrieval engines discover newly published or updated pages much slower without an XML sitemap.',
      'Generate /sitemap.xml and declare `Sitemap: https://yourdomain.com/sitemap.xml` in robots.txt.'
    );
  }

  // ----- Pillar 3: Structured Data & Entity Graph (30 pts) -----
  let p3 = 0;
  const hasOrg = html.schemaTypes.some(t => /^(Organization|Corporation|LocalBusiness|OnlineBusiness|Brand)$/i.test(t));
  const hasWebSite = html.schemaTypes.includes('WebSite');
  const hasFaq = html.schemaTypes.includes('FAQPage');
  const hasCommercialOrContent = html.schemaTypes.some(t =>
    /^(Product|SoftwareApplication|Service|Article|BlogPosting|HowTo|WebPage)$/i.test(t)
  );

  if (html.jsonLdBlocks > 0 && html.jsonLdInvalid === 0) {
    p3 += 6;
  } else if (html.jsonLdInvalid > 0) {
    p3 += 2;
    addFinding(
      'P0',
      'Structured Data',
      `Invalid JSON-LD syntax (${html.jsonLdInvalid} broken block(s))`,
      'Malformed JSON-LD is silently ignored by Google, OpenAI, and Perplexity entity parsers.',
      'Fix JSON syntax errors and verify using validator.schema.org.'
    );
  } else {
    addFinding(
      'P0',
      'Structured Data',
      'No Schema.org JSON-LD structured data detected',
      'Without JSON-LD, AI engines must guess your entity name, category, pricing, and official profiles from unstructured HTML.',
      'Paste the ready-to-deploy schema-jsonld.html snippet into your homepage <head>.'
    );
  }

  if (hasOrg) p3 += 6;
  else if (html.jsonLdBlocks > 0) {
    addFinding(
      'P1',
      'Structured Data',
      'Missing Organization / Brand Schema on homepage',
      'Your brand lacks an explicit entity anchor linking name, logo, URL, and verified profiles.',
      'Add Organization JSON-LD with `sameAs` links to LinkedIn, GitHub, Crunchbase, or Wikidata.'
    );
  }

  if (hasWebSite || hasCommercialOrContent) p3 += 4;
  if (hasFaq) {
    p3 += 5;
  } else {
    addFinding(
      'P1',
      'Structured Data',
      'Missing FAQPage Schema markup',
      'Conversational queries in ChatGPT and Perplexity map directly to structured Question/Answer pairs.',
      'Add a 5–8 question FAQ section backed by the included FAQPage JSON-LD block.'
    );
  }

  if (html.orgData.sameAs.length >= 2 || html.socialLinks.length >= 2) p3 += 3;
  else {
    addFinding(
      'P1',
      'Structured Data',
      'Weak external entity corroboration (sameAs / official profiles)',
      'AI engines cross-check brand claims across third-party authority profiles before citing a brand.',
      'Link verified LinkedIn, Crunchbase, G2, GitHub, and Wikipedia/Wikidata URLs in Organization `sameAs`.'
    );
  }

  if (html.title && html.title.length >= 15 && html.description && html.description.length >= 50) p3 += 3;
  else {
    addFinding(
      'P1',
      'Structured Data',
      'Missing or under-descriptive <title> / meta description',
      'Weak meta summaries reduce snippet clarity when AI search engines preview pages.',
      'Write a clear entity-first title (50–60 chars) and factual meta description (120–155 chars).'
    );
  }

  if (html.canonical && html.ogTitle && html.ogDescription) p3 += 3;
  else {
    addFinding(
      'P2',
      'Structured Data',
      'Incomplete canonical or OpenGraph tags',
      'Missing canonical URLs can split citation authority across duplicate URL variants.',
      'Add `<link rel="canonical">` alongside `og:title`, `og:description`, and `og:image`.'
    );
  }

  // ----- Pillar 4: Content Extractability & SSR Readiness (25 pts) -----
  let p4 = 0;
  if (!html.likelyCsrShell && html.wordCount >= 180) {
    p4 += 9;
  } else if (html.likelyCsrShell) {
    addFinding(
      'P0',
      'Content Extractability',
      `Likely client-side rendered (CSR) shell — only ${html.wordCount} words in raw HTML`,
      'Most AI retrieval bots (GPTBot, PerplexityBot, ClaudeBot) do not execute heavy JavaScript; your page appears nearly blank to them.',
      'Enable Server-Side Rendering (SSR) or Static Site Generation (SSG) for all public marketing, pricing, and docs pages.'
    );
  } else {
    p4 += 4;
    addFinding(
      'P1',
      'Content Extractability',
      `Thin extractable text (${html.wordCount} words in initial HTML)`,
      'Pages with very little plain text give LLMs almost no factual sentences to quote.',
      'Add concise answer-first summaries (40–60 words) explaining what you do, who it is for, and key differentiators.'
    );
  }

  if (html.headings.h1.length === 1 && html.headings.h2.length >= 2) p4 += 5;
  else {
    p4 += 2;
    addFinding(
      'P1',
      'Content Extractability',
      `Suboptimal heading hierarchy (H1: ${html.headings.h1.length}, H2: ${html.headings.h2.length})`,
      'Clean H1 -> H2 -> H3 hierarchy helps retrieval models chunk your page accurately.',
      'Use exactly one clear H1 and structure H2/H3 sections around buyer questions.'
    );
  }

  if (html.questionHeadings.length >= 2 || html.hasFaqSignals) p4 += 4;
  else {
    addFinding(
      'P1',
      'Content Extractability',
      'Few or no conversational question-style headings',
      'AI search queries are phrased as natural questions ("How does X compare to Y?", "What does X cost?").',
      'Rewrite key H2/H3 headings as direct buyer questions with a 40–60 word direct answer immediately underneath.'
    );
  }

  if (html.hasStatsOrNumbers) p4 += 3;
  else {
    addFinding(
      'P2',
      'Content Extractability',
      'Limited concrete numbers or verifiable statistics in copy',
      'GEO research shows LLMs strongly prefer citing pages that contain concrete figures, benchmarks, and transparent pricing.',
      'Include concrete metrics (customer count, latency/speed savings, SLA numbers, or transparent pricing tiers).'
    );
  }

  if (html.hasTablesOrLists) p4 += 2;
  if (html.hasDateSignal || html.hasAuthorSignal) p4 += 2;
  else {
    addFinding(
      'P2',
      'Content Extractability',
      'Missing freshness / authorship signals (dateModified / author)',
      'AI engines prioritize recently updated and attributed content when choosing citations.',
      'Expose `datePublished`, `dateModified`, and named expert authors on articles and comparison guides.'
    );
  }

  const pillars = [
    { id: 'crawler', name: 'AI Crawler Access', score: p1, max: 25 },
    { id: 'discovery', name: 'AI-Native Discovery', score: p2, max: 20 },
    { id: 'structured', name: 'Structured Data & Entity', score: p3, max: 30 },
    { id: 'content', name: 'Content Extractability & SSR', score: p4, max: 25 },
  ];
  const totalScore = pillars.reduce((acc, p) => acc + p.score, 0);
  const grade =
    totalScore >= 85 ? 'A' : totalScore >= 70 ? 'B' : totalScore >= 55 ? 'C' : totalScore >= 40 ? 'D' : 'F';

  const priorityOrder = { P0: 0, P1: 1, P2: 2 };
  findings.sort((a, b) => priorityOrder[a.priority] - priorityOrder[b.priority]);

  return { totalScore, grade, pillars, findings };
}

function generateFixes({ brand, industry, competitors, originUrl, html, sitemap }) {
  const origin = originUrl.origin;
  const summary =
    html.description ||
    html.leadParagraph.slice(0, 220) ||
    `${brand} provides ${industry || 'specialized digital'} solutions. Update this summary with a concise 2-sentence factual overview.`;

  // 1. Ready-to-deploy llms.txt
  const linkLines = [];
  const seen = new Set();
  for (const item of html.internalLinks) {
    if (seen.has(item.url) || linkLines.length >= 12) continue;
    seen.add(item.url);
    linkLines.push(`- [${item.label}](${item.url}): Key information on ${item.label.toLowerCase()}.`);
  }
  if (linkLines.length < 4 && sitemap.sampleUrls.length > 0) {
    for (const u of sitemap.sampleUrls) {
      if (seen.has(u) || linkLines.length >= 10) continue;
      seen.add(u);
      const slug = new URL(u).pathname.replace(/^\/|\/$/g, '') || 'Home';
      linkLines.push(`- [${slug}](${u}): Official ${brand} page.`);
    }
  }
  if (linkLines.length === 0) {
    linkLines.push(
      `- [Product Overview](${origin}/): Core platform capabilities and use cases.`,
      `- [Pricing](${origin}/pricing): Transparent pricing tiers and plans.`,
      `- [Documentation](${origin}/docs): Setup guides and technical references.`,
      `- [About & Contact](${origin}/about): Company background and verified contact details.`
    );
  }

  const llmsTxt = [
    `# ${brand}`,
    '',
    `> ${summary}`,
    '',
    '## Core Pages',
    ...linkLines,
    '',
    '## Key Facts for AI Assistants',
    `- Official Website: ${origin}`,
    `- Primary Category: ${industry || '[Insert exact industry/category]'}`,
    html.socialLinks.length ? `- Official Profiles: ${html.socialLinks.slice(0, 4).join(', ')}` : '- Official Profiles: [Add LinkedIn, Crunchbase, GitHub, G2 URLs]',
    '',
    '## Contact & Citation',
    `- Preferred citation name: ${brand}`,
    `- Canonical URL: ${origin}`,
    '',
  ].join('\n');

  // 2. robots.txt AI patch
  const sitemapLine = sitemap.present
    ? `Sitemap: ${origin}/sitemap.xml`
    : `Sitemap: ${origin}/sitemap.xml  # Update to your actual XML sitemap URL`;

  const robotsSnippet = [
    `# === AI Search & Retrieval Crawler Access (${brand}) ===`,
    '# Explicitly allow major AI search & citation bots on public content.',
    '',
    'User-agent: GPTBot',
    'Allow: /',
    '',
    'User-agent: OAI-SearchBot',
    'Allow: /',
    '',
    'User-agent: ChatGPT-User',
    'Allow: /',
    '',
    'User-agent: PerplexityBot',
    'Allow: /',
    '',
    'User-agent: ClaudeBot',
    'Allow: /',
    '',
    'User-agent: Google-Extended',
    'Allow: /',
    '',
    'User-agent: Googlebot',
    'Allow: /',
    '',
    sitemapLine,
    '',
  ].join('\n');

  // 3. Schema.org JSON-LD snippet
  const orgSchema = {
    '@context': 'https://schema.org',
    '@graph': [
      {
        '@type': 'Organization',
        '@id': `${origin}/#organization`,
        name: brand,
        url: origin,
        description: summary,
        ...(html.orgData.logo || html.ogImage ? { logo: html.orgData.logo || html.ogImage } : {}),
        ...(html.socialLinks.length ? { sameAs: html.socialLinks.slice(0, 6) } : {}),
      },
      {
        '@type': 'WebSite',
        '@id': `${origin}/#website`,
        url: origin,
        name: brand,
        publisher: { '@id': `${origin}/#organization` },
      },
      {
        '@type': 'FAQPage',
        mainEntity: [
          {
            '@type': 'Question',
            name: `What is ${brand}?`,
            acceptedAnswer: {
              '@type': 'Answer',
              text: summary,
            },
          },
          {
            '@type': 'Question',
            name: `Who is ${brand} best suited for?`,
            acceptedAnswer: {
              '@type': 'Answer',
              text: `[Replace with a concrete 2-sentence description of ${brand}'s ideal customer profile and primary use cases in ${industry || 'your market'}.]`,
            },
          },
          {
            '@type': 'Question',
            name: `How does ${brand} pricing work?`,
            acceptedAnswer: {
              '@type': 'Answer',
              text: `[Replace with transparent starting price, billing model, or free trial details for ${brand}.]`,
            },
          },
        ],
      },
    ],
  };

  const schemaHtml = [
    `<!-- Paste inside <head> on ${origin} -->`,
    '<script type="application/ld+json">',
    JSON.stringify(orgSchema, null, 2),
    '</script>',
    '',
  ].join('\n');

  // 4. 20-Prompt AI Search Visibility Benchmark Sheet
  const cat = industry || `${brand}'s product category`;
  const compList = competitors
    .split(',')
    .map(s => s.trim())
    .filter(Boolean);
  const comp1 = compList[0] || '[Top Competitor A]';
  const comp2 = compList[1] || '[Top Competitor B]';

  const prompts = [
    // Category discovery (non-branded)
    { group: 'Category Discovery', prompt: `What are the best ${cat} tools or providers in 2026?` },
    { group: 'Category Discovery', prompt: `Top 5 ${cat} solutions for growing teams and their pros and cons.` },
    { group: 'Category Discovery', prompt: `Which ${cat} platform offers the best ROI for small to mid-sized businesses?` },
    { group: 'Category Discovery', prompt: `Recommend a reliable ${cat} solution with transparent pricing.` },
    { group: 'Category Discovery', prompt: `What should buyers look for when choosing a ${cat} provider?` },
    // Problem / Use-case intent
    { group: 'Problem / Intent', prompt: `How do I solve [primary customer pain point] using ${cat}?` },
    { group: 'Problem / Intent', prompt: `Fastest way to set up ${cat} without a large engineering team?` },
    { group: 'Problem / Intent', prompt: `Best ${cat} options for startups vs enterprise companies.` },
    { group: 'Problem / Intent', prompt: `Which ${cat} vendors have the strongest integrations and API support?` },
    { group: 'Problem / Intent', prompt: `What are the hidden costs or common pitfalls when adopting ${cat}?` },
    // Competitor comparison
    { group: 'Competitor Comparison', prompt: `What are the best alternatives to ${comp1}?` },
    { group: 'Competitor Comparison', prompt: `Compare ${brand} vs ${comp1}: features, pricing, and who each is best for.` },
    { group: 'Competitor Comparison', prompt: `Compare ${brand} vs ${comp2} for a 2026 buyer.` },
    { group: 'Competitor Comparison', prompt: `Why do teams switch from ${comp1} to alternatives?` },
    { group: 'Competitor Comparison', prompt: `Best ${comp1} and ${comp2} competitors with better support.` },
    // Direct brand trust & accuracy
    { group: 'Brand Accuracy & Trust', prompt: `What is ${brand} and what does it do?` },
    { group: 'Brand Accuracy & Trust', prompt: `Is ${brand} legit and trustworthy? Summarize real user reviews.` },
    { group: 'Brand Accuracy & Trust', prompt: `How much does ${brand} cost? Break down its pricing tiers.` },
    { group: 'Brand Accuracy & Trust', prompt: `What are the main limitations or drawbacks of ${brand}?` },
    { group: 'Brand Accuracy & Trust', prompt: `Where can I read official documentation or contact ${brand}?` },
  ];

  const promptBenchmarkMd = [
    `# AI Search Visibility Benchmark Sheet — ${brand}`,
    '',
    `Run these 20 prompts across **ChatGPT Search**, **Perplexity**, **Google AI Overviews / Gemini**, and **Claude** (fresh logged-out or neutral session) to track citation share over time.`,
    '',
    'Score each cell: `0` = Not mentioned | `1` = Mentioned without link | `2` = Mentioned with direct citation link | `3` = Top-3 recommendation with link.',
    '',
    '| # | Intent Group | Prompt | ChatGPT | Perplexity | Gemini | Claude | Notes / Cited Competitor |',
    '| :-: | :--- | :--- | :-: | :-: | :-: | :-: | :--- |',
    ...prompts.map((p, i) => `| ${i + 1} | ${p.group} | ${p.prompt} | | | | | |`),
    '',
  ].join('\n');

  return {
    llmsTxt,
    robotsSnippet,
    schemaHtml,
    promptBenchmarkMd,
    prompts,
  };
}

async function runGeoAudit(rawUrl, options = {}) {
  const { brand = '', industry = '', competitors = '', fetchImpl = fetch } = options;
  const targetUrl = normalizeUrl(rawUrl);
  const origin = targetUrl.origin;

  const [pageRes, robotsRes, llmsRes, llmsFullRes, sitemapRes] = await Promise.all([
    safeFetch(targetUrl.href, { fetchImpl }),
    safeFetch(`${origin}/robots.txt`, { fetchImpl, maxBytes: 200_000 }),
    safeFetch(`${origin}/llms.txt`, { fetchImpl, maxBytes: 250_000 }),
    safeFetch(`${origin}/llms-full.txt`, { fetchImpl, maxBytes: 250_000 }),
    safeFetch(`${origin}/sitemap.xml`, { fetchImpl, maxBytes: 500_000 }),
  ]);

  if (!pageRes.ok && pageRes.status === 0) {
    throw new Error(`Could not reach ${targetUrl.href}: ${pageRes.error}`);
  }

  const finalOrigin = new URL(pageRes.finalUrl || targetUrl.href);
  const html = analyzeHtml(pageRes.text, finalOrigin.href);
  const robots = auditRobots(robotsRes.text, isRealTextFile(robotsRes));
  const llms = parseLlmsTxt(llmsRes);
  const llmsFull = parseLlmsTxt(llmsFullRes);

  let sitemap = parseSitemap(sitemapRes);
  if (!sitemap.present && robots.sitemaps.length > 0) {
    const altSitemapRes = await safeFetch(robots.sitemaps[0], { fetchImpl, maxBytes: 500_000 });
    sitemap = parseSitemap(altSitemapRes);
    sitemap.sourceUrl = robots.sitemaps[0];
  } else if (sitemap.present) {
    sitemap.sourceUrl = `${origin}/sitemap.xml`;
  }

  const resolvedBrand = inferBrand(html, finalOrigin, brand);
  const scoring = scoreAndBuildFindings({ html, robots, llms, llmsFull, sitemap, pageRes });
  const fixes = generateFixes({
    brand: resolvedBrand,
    industry,
    competitors,
    originUrl: finalOrigin,
    html,
    sitemap,
  });

  return {
    auditedAt: new Date().toISOString(),
    targetUrl: targetUrl.href,
    finalUrl: pageRes.finalUrl,
    httpStatus: pageRes.status,
    responseTimeMs: pageRes.elapsedMs,
    brand: resolvedBrand,
    industry: industry || '',
    competitors: competitors || '',
    ...scoring,
    checks: {
      robots,
      llms,
      llmsFull,
      sitemap,
      html,
    },
    fixes,
  };
}

module.exports = { runGeoAudit, normalizeUrl };
