'use strict';

const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const { runGeoAudit } = require('./lib/auditor');
const { renderReportHtml } = require('./lib/report');

const PORT = Number(process.env.PORT) || 3100;
const PUBLIC_DIR = path.join(__dirname, 'public');
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
};

function sendJson(res, status, data) {
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
  });
  res.end(JSON.stringify(data));
}

function readJson(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on('data', c => {
      size += c.length;
      if (size > 200_000) {
        reject(new Error('Request body too large'));
        req.destroy();
      } else {
        chunks.push(c);
      }
    });
    req.on('end', () => {
      try {
        resolve(chunks.length ? JSON.parse(Buffer.concat(chunks).toString('utf8')) : {});
      } catch {
        reject(new Error('Invalid JSON body'));
      }
    });
    req.on('error', reject);
  });
}

function createServer({ fetchImpl = fetch } = {}) {
  let latestAudit = null;

  return http.createServer(async (req, res) => {
    const url = new URL(req.url, 'http://localhost');
    const { pathname } = url;

    try {
      if (pathname === '/api/audit' && req.method === 'POST') {
        const body = await readJson(req);
        const audit = await runGeoAudit(body.url, {
          brand: body.brand,
          industry: body.industry,
          competitors: body.competitors,
          fetchImpl,
        });
        latestAudit = audit;
        return sendJson(res, 200, { audit, reportHtml: renderReportHtml(audit) });
      }

      if (pathname === '/api/latest/report.html' && req.method === 'GET') {
        if (!latestAudit) return sendJson(res, 404, { error: 'No audit run yet.' });
        res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store' });
        return res.end(renderReportHtml(latestAudit));
      }

      if (pathname.startsWith('/api/latest/fix/') && req.method === 'GET') {
        if (!latestAudit) return sendJson(res, 404, { error: 'No audit run yet.' });
        const key = pathname.slice('/api/latest/fix/'.length);
        const map = {
          'llms.txt': { content: latestAudit.fixes.llmsTxt, type: 'text/plain; charset=utf-8' },
          'robots-ai-snippet.txt': { content: latestAudit.fixes.robotsSnippet, type: 'text/plain; charset=utf-8' },
          'schema-jsonld.html': { content: latestAudit.fixes.schemaHtml, type: 'text/html; charset=utf-8' },
          'ai-prompt-benchmark.md': { content: latestAudit.fixes.promptBenchmarkMd, type: 'text/markdown; charset=utf-8' },
        };
        const file = map[key];
        if (!file) return sendJson(res, 404, { error: 'Unknown fix file.' });
        res.writeHead(200, {
          'Content-Type': file.type,
          'Content-Disposition': `attachment; filename="${key}"`,
          'Cache-Control': 'no-store',
        });
        return res.end(file.content);
      }

      if (pathname === '/sample-report' && req.method === 'GET') {
        const samplePath = path.join(__dirname, 'samples', 'geo-audit-report.html');
        if (fs.existsSync(samplePath)) {
          res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
          return res.end(fs.readFileSync(samplePath));
        }
        res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
        return res.end('No sample report generated yet. Run cli.js first.');
      }

      let rel = decodeURIComponent(pathname);
      if (rel === '/') rel = '/index.html';
      const target = path.normalize(path.join(PUBLIC_DIR, rel));
      if (!target.startsWith(PUBLIC_DIR + path.sep) || !fs.existsSync(target) || fs.statSync(target).isDirectory()) {
        res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
        return res.end('Not found');
      }
      res.writeHead(200, {
        'Content-Type': MIME[path.extname(target)] || 'application/octet-stream',
        'Cache-Control': 'no-cache',
      });
      res.end(fs.readFileSync(target));
    } catch (err) {
      sendJson(res, 400, { error: err.message || 'Audit failed.' });
    }
  });
}

if (require.main === module) {
  const server = createServer();
  server.listen(PORT, () => {
    console.log(`Fiverr GEO Audit Studio running at http://localhost:${PORT}`);
  });
}

module.exports = { createServer };
