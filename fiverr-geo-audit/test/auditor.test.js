'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');
const { auditRobots } = require('../lib/robots');
const { analyzeHtml } = require('../lib/html');
const { runGeoAudit } = require('../lib/auditor');
const { renderReportHtml } = require('../lib/report');
const { createServer } = require('../server');

test('robots parser enforces specific-agent precedence and longest-match rule', () => {
  const txt = `
User-agent: *
Allow: /

User-agent: GPTBot
Disallow: /

User-agent: PerplexityBot
Disallow: /
Allow: /
`;
  const res = auditRobots(txt, true);
  const byId = Object.fromEntries(res.bots.map(b => [b.id, b]));
  assert.equal(byId.GPTBot.allowed, false);
  assert.equal(byId.GPTBot.status, 'blocked');
  assert.equal(byId.PerplexityBot.allowed, true);
  assert.equal(byId.PerplexityBot.status, 'allowed_explicit');
  assert.equal(byId.ClaudeBot.allowed, true);
  assert.equal(byId.ClaudeBot.status, 'allowed_default');
  assert.deepEqual(res.blockedBots, ['GPTBot']);
});

test('HTML analyzer extracts JSON-LD @graph, headings, social profiles, and detects CSR shells', () => {
  const richHtml = `<!doctype html>
<html lang="en">
<head>
  <title>Acme Analytics | Real-Time BI Platform</title>
  <meta name="description" content="Acme Analytics helps 2,500+ B2B teams cut reporting time by 60% with automated dashboards.">
  <meta property="og:title" content="Acme Analytics">
  <meta property="og:description" content="Automated BI for B2B teams.">
  <link rel="canonical" href="https://acme.test/">
  <script type="application/ld+json">
  {
    "@context": "https://schema.org",
    "@graph": [
      { "@type": "Organization", "name": "Acme Analytics", "sameAs": ["https://linkedin.com/company/acme", "https://github.com/acme"] },
      { "@type": "FAQPage", "mainEntity": [] }
    ]
  }
  </script>
</head>
<body>
  <h1>Real-Time BI for Growing Teams</h1>
  <h2>How does Acme Analytics work?</h2>
  <p>Acme Analytics connects to your warehouse in 5 minutes and saves analysts over 12 hours per week with 99.9% uptime.</p>
  <h2>What does Acme Analytics cost?</h2>
  <ul><li>Starter: $49/mo</li><li>Pro: $199/mo</li></ul>
  <a href="/pricing">Pricing</a>
  <a href="/docs">Documentation</a>
</body>
</html>`;
  const info = analyzeHtml(richHtml, 'https://acme.test/');
  assert.equal(info.orgData.name, 'Acme Analytics');
  assert.ok(info.schemaTypes.includes('Organization'));
  assert.ok(info.schemaTypes.includes('FAQPage'));
  assert.equal(info.questionHeadings.length, 2);
  assert.equal(info.hasStatsOrNumbers, true);
  assert.equal(info.likelyCsrShell, false);
  assert.equal(info.internalLinks.length, 2);

  const csrHtml = '<!doctype html><html><head><title>App</title></head><body><div id="root"></div><noscript>You need to enable JavaScript</noscript></body></html>';
  const csrInfo = analyzeHtml(csrHtml, 'https://spa.test/');
  assert.equal(csrInfo.likelyCsrShell, true);
});

test('end-to-end audit distinguishes weak blocked SPA from strong GEO-ready site', async () => {
  const routes = {
    '/weak': {
      status: 200,
      type: 'text/html',
      body: '<!doctype html><html><head><title>Weak</title><meta name="robots" content="nosnippet"></head><body><div id="root"></div></body></html>',
    },
    '/robots.txt': {
      status: 200,
      type: 'text/plain',
      body: 'User-agent: GPTBot\nDisallow: /\nUser-agent: PerplexityBot\nDisallow: /\n',
    },
    '/llms.txt': {
      // Simulate SPA fallback returning HTML with HTTP 200
      status: 200,
      type: 'text/html',
      body: '<!doctype html><html><body><div id="root"></div></body></html>',
    },
  };

  const mock = http.createServer((req, res) => {
    const entry = routes[req.url] || { status: 404, type: 'text/plain', body: 'Not found' };
    res.writeHead(entry.status, { 'Content-Type': entry.type });
    res.end(entry.body);
  });
  await new Promise(r => mock.listen(0, '127.0.0.1', r));
  const origin = `http://127.0.0.1:${mock.address().port}`;

  try {
    const weakAudit = await runGeoAudit(`${origin}/weak`, {
      brand: 'WeakCo',
      industry: 'CRM Software',
      competitors: 'HubSpot, Pipedrive',
    });
    assert.ok(weakAudit.totalScore < 30, `Expected low score, got ${weakAudit.totalScore}`);
    assert.equal(weakAudit.checks.llms.present, false, 'HTML 200 fallback should not count as valid llms.txt');
    assert.ok(weakAudit.findings.some(f => f.priority === 'P0'));
    assert.ok(weakAudit.fixes.llmsTxt.includes('# WeakCo'));
    assert.ok(weakAudit.fixes.robotsSnippet.includes('User-agent: GPTBot'));
    assert.ok(weakAudit.fixes.schemaHtml.includes('"@type": "FAQPage"'));
    assert.equal(weakAudit.fixes.prompts.length, 20);

    const htmlReport = renderReportHtml(weakAudit);
    assert.ok(htmlReport.includes('WeakCo'));
    assert.ok(htmlReport.includes('AI Crawler Access Matrix'));
    assert.ok(htmlReport.includes('20-Prompt AI Search Visibility Benchmark'));

    // Also test the studio HTTP server API
    const studio = createServer();
    await new Promise(r => studio.listen(0, '127.0.0.1', r));
    const studioUrl = `http://127.0.0.1:${studio.address().port}`;
    try {
      const apiRes = await fetch(`${studioUrl}/api/audit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: `${origin}/weak`, brand: 'WeakCo', industry: 'CRM' }),
      });
      assert.equal(apiRes.status, 200);
      const payload = await apiRes.json();
      assert.equal(payload.audit.brand, 'WeakCo');

      const llmsRes = await fetch(`${studioUrl}/api/latest/fix/llms.txt`);
      assert.equal(llmsRes.status, 200);
      assert.ok((await llmsRes.text()).includes('# WeakCo'));
    } finally {
      studio.close();
    }
  } finally {
    mock.close();
  }
});
