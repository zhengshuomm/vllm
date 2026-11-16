# Fiverr GEO Audit Studio

Zero-dependency Node.js tool for running technical **Generative Engine Optimization (GEO / AEO)** audits and generating ready-to-deliver client packages for Fiverr orders.

## What It Checks & Generates

1. **AI Crawler Access (`robots.txt` & headers)** — Parses rules with RFC 9309 precedence for `GPTBot`, `OAI-SearchBot`, `ChatGPT-User`, `PerplexityBot`, `ClaudeBot`, `Google-Extended`, `Googlebot`, and `Applebot-Extended`, plus restrictive `noindex` / `nosnippet` / `noai` directives.
2. **AI-Native Discovery** — Validates `/llms.txt`, `/llms-full.txt` (rejecting SPA HTML 200 fallbacks), and `/sitemap.xml`.
3. **Structured Data & Entity Graph** — Parses and validates Schema.org JSON-LD (`@graph`, `Organization`, `WebSite`, `FAQPage`, `Product`), canonical URLs, OpenGraph tags, and `sameAs` entity links.
4. **Content Extractability & SSR Readiness** — Detects client-side-only SPA shells, heading hierarchy, conversational Q&A headings, concrete figures, and freshness/authorship signals.
5. **Auto-Generated Client Delivery Pack**:
   - `geo-audit-report.html` (print-to-PDF executive report)
   - `llms.txt` (tailored from the target site)
   - `robots-ai-snippet.txt`
   - `schema-jsonld.html`
   - `ai-prompt-benchmark.md` (20 tailored buyer-intent prompts)

## Quickstart

Requires **Node.js 20+**. No `npm install` needed.

```bash
# Run automated tests
npm test

# Start the local web workbench (http://localhost:3100)
npm start

# Or run directly via CLI to generate a client delivery folder
node cli.js https://example.com --brand "Example" --industry "B2B SaaS" --competitors "CompA, CompB"
```

See [FIVERR_GIG_KIT.md](FIVERR_GIG_KIT.md) for the complete copy-paste Fiverr listing, pricing table, FAQ, and 15-minute fulfillment SOP.
