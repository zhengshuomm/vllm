#!/usr/bin/env node
'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { runGeoAudit } = require('./lib/auditor');
const { renderReportHtml } = require('./lib/report');

function parseArgs(argv) {
  const opts = { url: '', brand: '', industry: '', competitors: '', out: '' };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--brand') opts.brand = argv[++i] || '';
    else if (a === '--industry') opts.industry = argv[++i] || '';
    else if (a === '--competitors') opts.competitors = argv[++i] || '';
    else if (a === '--out') opts.out = argv[++i] || '';
    else if (!a.startsWith('-') && !opts.url) opts.url = a;
  }
  return opts;
}

function writeDeliveryBundle(audit, outDir) {
  fs.mkdirSync(outDir, { recursive: true });
  const files = {
    'geo-audit-report.html': renderReportHtml(audit),
    'llms.txt': audit.fixes.llmsTxt,
    'robots-ai-snippet.txt': audit.fixes.robotsSnippet,
    'schema-jsonld.html': audit.fixes.schemaHtml,
    'ai-prompt-benchmark.md': audit.fixes.promptBenchmarkMd,
    'audit-data.json': JSON.stringify(audit, null, 2),
  };
  for (const [name, content] of Object.entries(files)) {
    fs.writeFileSync(path.join(outDir, name), content, 'utf8');
  }
  return Object.keys(files).map(f => path.join(outDir, f));
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  if (!opts.url) {
    console.error('Usage: node cli.js <url> [--brand "Acme"] [--industry "B2B SaaS"] [--competitors "CompA, CompB"] [--out ./deliverables/acme]');
    process.exit(1);
  }

  console.log(`Auditing ${opts.url}...`);
  const audit = await runGeoAudit(opts.url, opts);
  const slug = new URL(audit.finalUrl).hostname.replace(/^www\./, '').replace(/[^a-z0-9.-]/gi, '_');
  const outDir = opts.out || path.join(__dirname, 'deliverables', slug);
  const written = writeDeliveryBundle(audit, outDir);

  console.log(`\nBrand: ${audit.brand} (${audit.finalUrl})`);
  console.log(`GEO Score: ${audit.totalScore}/100 (Grade ${audit.grade})`);
  for (const p of audit.pillars) {
    console.log(`  - ${p.name}: ${p.score}/${p.max}`);
  }
  console.log(`Findings: ${audit.findings.length} (${audit.findings.filter(f => f.priority === 'P0').length} P0 critical)`);
  console.log(`\nSaved Fiverr delivery bundle (${written.length} files) to:\n  ${outDir}`);
}

if (require.main === module) {
  main().catch(err => {
    console.error('Error:', err.message);
    process.exit(1);
  });
}

module.exports = { writeDeliveryBundle };
