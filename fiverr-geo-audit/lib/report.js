'use strict';

function esc(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function scoreColor(score) {
  if (score >= 80) return '#15803d';
  if (score >= 60) return '#b45309';
  return '#b91c1c';
}

function pillarPct(p) {
  return p.max > 0 ? Math.round((p.score / p.max) * 100) : 0;
}

function botBadge(status) {
  if (status === 'allowed_explicit') return '<span class="pill pill-green">Allowed (Explicit)</span>';
  if (status === 'allowed_default') return '<span class="pill pill-blue">Allowed (Default)</span>';
  return '<span class="pill pill-red">Blocked</span>';
}

function priorityBadge(p) {
  if (p === 'P0') return '<span class="pill pill-red">P0 · Critical</span>';
  if (p === 'P1') return '<span class="pill pill-amber">P1 · High</span>';
  return '<span class="pill pill-blue">P2 · Medium</span>';
}

function renderReportHtml(audit) {
  const color = scoreColor(audit.totalScore);
  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const dashOffset = circumference * (1 - audit.totalScore / 100);
  const dateLabel = audit.auditedAt.slice(0, 10);

  const p0Count = audit.findings.filter(f => f.priority === 'P0').length;
  const p1Count = audit.findings.filter(f => f.priority === 'P1').length;
  const p2Count = audit.findings.filter(f => f.priority === 'P2').length;

  const { robots, llms, llmsFull, sitemap, html } = audit.checks;

  const summaryBullets = [
    `<strong>Overall GEO Readiness:</strong> ${audit.brand} scored <strong>${audit.totalScore}/100 (Grade ${audit.grade})</strong> across 4 technical AI-search pillars.`,
    robots.blockedBots.length
      ? `<strong>Critical Crawler Block:</strong> <code>robots.txt</code> currently blocks <strong>${esc(robots.blockedBots.join(', '))}</strong> from crawling public pages.`
      : `<strong>AI Crawler Access:</strong> Major AI retrieval bots are currently not blocked in <code>robots.txt</code>.`,
    llms.present
      ? `<strong>AI Manifest:</strong> <code>/llms.txt</code> is live (${llms.linkCount} structured links detected).`
      : `<strong>AI Manifest Missing:</strong> No <code>/llms.txt</code> file was detected; a tailored file is included in Section 5.`,
    html.schemaTypes.length
      ? `<strong>Structured Data:</strong> Detected ${html.jsonLdBlocks} JSON-LD block(s) covering <code>${esc(html.schemaTypes.slice(0, 6).join(', '))}</code>.`
      : `<strong>Structured Data Gap:</strong> No valid Schema.org JSON-LD blocks detected; a ready-to-paste snippet is included in Section 5.`,
    html.likelyCsrShell
      ? `<strong>Rendering Risk:</strong> Only ${html.wordCount} extractable words appeared in raw HTML (likely client-side rendered SPA shell).`
      : `<strong>SSR Extractability:</strong> ${html.wordCount.toLocaleString('en-US')} extractable words parsed cleanly from initial HTML (${audit.responseTimeMs}ms).`,
  ];

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GEO &amp; AI Search Visibility Audit — ${esc(audit.brand)}</title>
  <style>
    :root {
      --ink: #0f172a;
      --muted: #475569;
      --border: #e2e8f0;
      --surface: #ffffff;
      --bg: #f8fafc;
      --accent: #2563eb;
      --accent-soft: #eff6ff;
      --green: #15803d;
      --green-bg: #dcfce7;
      --amber: #b45309;
      --amber-bg: #fef3c7;
      --red: #b91c1c;
      --red-bg: #fee2e2;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
      color: var(--ink);
      background: var(--bg);
      line-height: 1.55;
      font-size: 14px;
    }
    .page {
      max-width: 980px;
      margin: 28px auto;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 10px 30px rgba(15, 23, 42, .06);
    }
    .hero {
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 65%, #1d4ed8 100%);
      color: #fff;
      padding: 36px 40px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 28px;
      flex-wrap: wrap;
    }
    .hero > div:first-child { flex: 1 1 360px; min-width: 0; }
    .hero .eyebrow {
      text-transform: uppercase;
      letter-spacing: .1em;
      font-size: 11px;
      font-weight: 700;
      color: #93c5fd;
      margin-bottom: 8px;
    }
    .hero h1 { margin: 0 0 6px; font-size: 28px; line-height: 1.2; }
    .hero .meta { color: #cbd5e1; font-size: 13px; }
    .hero .meta a { color: #bfdbfe; text-decoration: none; }
    .gauge {
      display: flex;
      align-items: center;
      gap: 18px;
      background: rgba(255,255,255,.08);
      border: 1px solid rgba(255,255,255,.16);
      padding: 14px 20px;
      border-radius: 12px;
    }
    .gauge-text .grade { font-size: 13px; color: #cbd5e1; }
    .gauge-text .big { font-size: 24px; font-weight: 700; }
    .content { padding: 32px 40px 44px; }
    h2 {
      font-size: 18px;
      margin: 30px 0 14px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    h2:first-child { margin-top: 0; }
    .section-num {
      display: inline-grid;
      place-items: center;
      width: 24px;
      height: 24px;
      border-radius: 6px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
    }
    .summary-list {
      margin: 0;
      padding: 16px 20px 16px 36px;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 10px;
    }
    .summary-list li { margin-bottom: 6px; }
    .summary-list li:last-child { margin-bottom: 0; }
    .pillars {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 14px;
      margin-top: 16px;
    }
    .pillar {
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 16px;
      background: #fff;
    }
    .pillar .name { font-size: 12px; color: var(--muted); font-weight: 600; }
    .pillar .score { font-size: 22px; font-weight: 700; margin: 4px 0 8px; }
    .pillar .score span { font-size: 13px; color: var(--muted); font-weight: 500; }
    .bar { height: 7px; border-radius: 99px; background: #e2e8f0; overflow: hidden; }
    .bar > div { height: 100%; border-radius: 99px; }
    .stats-strip {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin: 14px 0;
    }
    .stat-box {
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 12px;
      background: var(--bg);
    }
    .stat-box .k { font-size: 11px; text-transform: uppercase; color: var(--muted); letter-spacing: .04em; }
    .stat-box .v { font-size: 16px; font-weight: 700; margin-top: 2px; }
    table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }
    th, td { padding: 9px 12px; border-bottom: 1px solid var(--border); text-align: left; vertical-align: top; }
    th { background: #f8fafc; font-size: 11.5px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
    code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      background: #f1f5f9;
      padding: 1px 5px;
      border-radius: 4px;
    }
    td:first-child code { white-space: nowrap; }
    pre {
      margin: 8px 0 18px;
      padding: 14px 16px;
      background: #0f172a;
      color: #e2e8f0;
      border-radius: 8px;
      overflow-x: auto;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.5;
    }
    .pill {
      display: inline-block;
      padding: 2px 9px;
      border-radius: 99px;
      font-size: 11.5px;
      font-weight: 600;
      white-space: nowrap;
    }
    .pill-green { background: var(--green-bg); color: var(--green); }
    .pill-blue { background: var(--accent-soft); color: var(--accent); }
    .pill-amber { background: var(--amber-bg); color: var(--amber); }
    .pill-red { background: var(--red-bg); color: var(--red); }
    .finding {
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 16px;
      margin-bottom: 10px;
      background: #fff;
    }
    .finding-head {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 6px;
      flex-wrap: wrap;
    }
    .finding-title { font-weight: 700; font-size: 14.5px; }
    .finding-pillar { font-size: 12px; color: var(--muted); margin-left: auto; }
    .finding p { margin: 4px 0; color: var(--muted); font-size: 13px; }
    .finding p strong { color: var(--ink); }
    .fix-label {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-weight: 650;
      margin-top: 14px;
    }
    .fix-label span { font-size: 12px; color: var(--muted); font-weight: 400; }
    .footer {
      border-top: 1px solid var(--border);
      padding: 18px 40px;
      background: #f8fafc;
      color: var(--muted);
      font-size: 12px;
      display: flex;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 8px;
    }
    .print-bar {
      max-width: 980px;
      margin: 16px auto 0;
      display: flex;
      justify-content: flex-end;
      gap: 8px;
    }
    .print-btn {
      padding: 8px 14px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #fff;
      color: var(--ink);
      font: inherit;
      font-weight: 600;
      cursor: pointer;
    }
    .print-btn.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
    @media (max-width: 760px) {
      .pillars, .stats-strip { grid-template-columns: repeat(2, 1fr); }
      .hero, .content { padding: 24px 20px; }
    }
    @media print {
      body { background: #fff; }
      .print-bar { display: none; }
      .page { margin: 0; border: 0; box-shadow: none; border-radius: 0; max-width: 100%; }
      .finding, pre, table { break-inside: avoid; }
    }
  </style>
</head>
<body>
  <div class="print-bar">
    <button class="print-btn primary" onclick="window.print()">Print / Save as PDF</button>
  </div>
  <div class="page">
    <header class="hero">
      <div>
        <div class="eyebrow">Generative Engine Optimization (GEO / AEO) Audit</div>
        <h1>${esc(audit.brand)}</h1>
        <div class="meta">
          Audited URL: <a href="${esc(audit.finalUrl)}">${esc(audit.finalUrl)}</a>
          ${audit.industry ? ` &middot; Industry: ${esc(audit.industry)}` : ''}
          &middot; Date: ${esc(dateLabel)}
        </div>
      </div>
      <div class="gauge">
        <svg width="128" height="128" viewBox="0 0 128 128" aria-label="GEO Score ${audit.totalScore} out of 100">
          <circle cx="64" cy="64" r="${radius}" fill="none" stroke="rgba(255,255,255,.16)" stroke-width="11"/>
          <circle cx="64" cy="64" r="${radius}" fill="none" stroke="${color}" stroke-width="11"
            stroke-linecap="round" stroke-dasharray="${circumference.toFixed(2)}" stroke-dashoffset="${dashOffset.toFixed(2)}"
            transform="rotate(-90 64 64)"/>
          <text x="64" y="61" text-anchor="middle" fill="#fff" font-size="30" font-weight="700">${audit.totalScore}</text>
          <text x="64" y="80" text-anchor="middle" fill="#cbd5e1" font-size="12">/ 100</text>
        </svg>
        <div class="gauge-text">
          <div class="grade">Readiness Grade</div>
          <div class="big">Grade ${esc(audit.grade)}</div>
          <div class="grade">${p0Count} Critical &middot; ${p1Count} High &middot; ${p2Count} Medium</div>
        </div>
      </div>
    </header>

    <main class="content">
      <h2><span class="section-num">1</span> Executive Summary &amp; Pillar Scores</h2>
      <ul class="summary-list">
        ${summaryBullets.map(b => `<li>${b}</li>`).join('\n        ')}
      </ul>

      <div class="pillars">
        ${audit.pillars.map(p => {
          const pct = pillarPct(p);
          const barColor = scoreColor(pct);
          return `<div class="pillar">
            <div class="name">${esc(p.name)}</div>
            <div class="score">${p.score} <span>/ ${p.max} (${pct}%)</span></div>
            <div class="bar"><div style="width:${pct}%;background:${barColor}"></div></div>
          </div>`;
        }).join('\n        ')}
      </div>

      <h2><span class="section-num">2</span> AI Crawler Access Matrix (robots.txt)</h2>
      <p style="margin:0 0 8px;color:var(--muted)">
        Status of major AI retrieval and indexing crawlers against <code>${esc(new URL(audit.finalUrl).origin)}/robots.txt</code>
        (${robots.present ? 'file detected' : 'not found'}).
      </p>
      <table>
        <thead>
          <tr><th>Crawler</th><th>Platform</th><th>Purpose</th><th>Access</th><th>Matched Rule</th></tr>
        </thead>
        <tbody>
          ${robots.bots.map(b => `<tr>
            <td><code>${esc(b.id)}</code></td>
            <td>${esc(b.owner)}</td>
            <td>${esc(b.role)}</td>
            <td>${botBadge(b.status)}</td>
            <td><code>${esc(b.matchedRule)}</code></td>
          </tr>`).join('\n          ')}
        </tbody>
      </table>

      <h2><span class="section-num">3</span> Technical Snapshot (Discovery, Schema &amp; SSR)</h2>
      <div class="stats-strip">
        <div class="stat-box"><div class="k">/llms.txt</div><div class="v">${llms.present ? `Present (${llms.linkCount} links)` : 'Missing'}</div></div>
        <div class="stat-box"><div class="k">/llms-full.txt</div><div class="v">${llmsFull.present ? 'Present' : 'Missing'}</div></div>
        <div class="stat-box"><div class="k">XML Sitemap</div><div class="v">${sitemap.present ? `Present (${sitemap.urlCount} URLs)` : 'Missing'}</div></div>
        <div class="stat-box"><div class="k">Extractable Text</div><div class="v">${html.wordCount.toLocaleString('en-US')} words</div></div>
      </div>
      <table>
        <thead><tr><th>Check</th><th>Observed Value</th><th>GEO Implication</th></tr></thead>
        <tbody>
          <tr><td>Page Title</td><td>${esc(html.title || '—')}</td><td>${html.title ? `${html.title.length} chars` : 'Missing entity title'}</td></tr>
          <tr><td>Meta Description</td><td>${esc((html.description || '—').slice(0, 140))}</td><td>${html.description ? `${html.description.length} chars` : 'Missing summary'}</td></tr>
          <tr><td>Canonical URL</td><td><code>${esc(html.canonical || 'Missing')}</code></td><td>${html.canonical ? 'Consolidates citation authority' : 'Add canonical link tag'}</td></tr>
          <tr><td>JSON-LD Schemas</td><td><code>${esc(html.schemaTypes.join(', ') || 'None')}</code></td><td>${html.jsonLdBlocks} block(s), ${html.jsonLdInvalid} invalid</td></tr>
          <tr><td>Heading Structure</td><td>H1: ${html.headings.h1.length} &middot; H2: ${html.headings.h2.length} &middot; H3: ${html.headings.h3.length}</td><td>${html.questionHeadings.length} question-style heading(s)</td></tr>
          <tr><td>Rendering Mode</td><td>${html.likelyCsrShell ? 'Likely CSR Shell' : 'Server-Readable HTML'}</td><td>Text/HTML ratio: ${(html.textToHtmlRatio * 100).toFixed(1)}% &middot; HTTP ${audit.httpStatus} (${audit.responseTimeMs}ms)</td></tr>
          <tr><td>Entity Profiles</td><td>${html.socialLinks.length} official profile link(s)</td><td>${html.hasStatsOrNumbers ? 'Includes concrete figures' : 'Add concrete metrics / benchmarks'}</td></tr>
        </tbody>
      </table>

      <h2><span class="section-num">4</span> Prioritized Action Roadmap</h2>
      ${audit.findings.length
        ? audit.findings.map(f => `<div class="finding">
            <div class="finding-head">
              ${priorityBadge(f.priority)}
              <span class="finding-title">${esc(f.title)}</span>
              <span class="finding-pillar">${esc(f.pillar)}</span>
            </div>
            <p><strong>Impact:</strong> ${esc(f.impact)}</p>
            <p><strong>Recommended Fix:</strong> ${esc(f.action)}</p>
          </div>`).join('\n      ')
        : '<p>No critical technical gaps detected. Focus on off-site citations, third-party reviews, and ongoing prompt benchmarking.</p>'}

      <h2><span class="section-num">5</span> Ready-to-Deploy Code Fixes</h2>
      <div class="fix-label"><div>A. Tailored <code>/llms.txt</code> Manifest</div><span>Upload to site root: ${esc(new URL(audit.finalUrl).origin)}/llms.txt</span></div>
      <pre>${esc(audit.fixes.llmsTxt)}</pre>

      <div class="fix-label"><div>B. <code>robots.txt</code> AI Crawler Patch</div><span>Append or merge into /robots.txt</span></div>
      <pre>${esc(audit.fixes.robotsSnippet)}</pre>

      <div class="fix-label"><div>C. Schema.org JSON-LD (Organization + WebSite + FAQPage)</div><span>Paste inside homepage &lt;head&gt;</span></div>
      <pre>${esc(audit.fixes.schemaHtml)}</pre>

      <h2><span class="section-num">6</span> 20-Prompt AI Search Visibility Benchmark</h2>
      <p style="margin:0 0 8px;color:var(--muted)">
        Track how often <strong>${esc(audit.brand)}</strong> is cited vs competitors across ChatGPT Search, Perplexity, Gemini, and Claude.
        Score 0 (absent) to 3 (top-3 linked recommendation).
      </p>
      <table>
        <thead>
          <tr><th>#</th><th>Intent</th><th>Buyer Prompt</th><th>ChatGPT</th><th>Perplexity</th><th>Gemini</th></tr>
        </thead>
        <tbody>
          ${audit.fixes.prompts.map((p, i) => `<tr>
            <td>${i + 1}</td>
            <td><span class="pill pill-blue">${esc(p.group)}</span></td>
            <td>${esc(p.prompt)}</td>
            <td style="width:74px">&nbsp;</td>
            <td style="width:74px">&nbsp;</td>
            <td style="width:74px">&nbsp;</td>
          </tr>`).join('\n          ')}
        </tbody>
      </table>
    </main>

    <footer class="footer">
      <span>Prepared for ${esc(audit.brand)} &middot; GEO / AI Search Visibility Audit</span>
      <span>Note: LLM responses are probabilistic; this report covers verifiable technical, structural, and entity signals rather than guaranteed rankings.</span>
    </footer>
  </div>
</body>
</html>`;
}

module.exports = { renderReportHtml };
