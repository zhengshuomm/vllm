'use strict';

const AI_BOTS = [
  { id: 'GPTBot', owner: 'OpenAI', role: 'ChatGPT training & crawl' },
  { id: 'OAI-SearchBot', owner: 'OpenAI', role: 'ChatGPT Search indexing' },
  { id: 'ChatGPT-User', owner: 'OpenAI', role: 'ChatGPT live user browsing' },
  { id: 'PerplexityBot', owner: 'Perplexity', role: 'Perplexity AI Search indexing' },
  { id: 'ClaudeBot', owner: 'Anthropic', role: 'Claude web retrieval' },
  { id: 'Google-Extended', owner: 'Google', role: 'Gemini grounding & training' },
  { id: 'Googlebot', owner: 'Google', role: 'Google Search & AI Overviews' },
  { id: 'Applebot-Extended', owner: 'Apple', role: 'Apple Intelligence' },
];

function parseRobotsTxt(content) {
  const groups = [];
  const sitemaps = [];
  let currentAgents = [];
  let currentRules = [];
  let seenRule = false;

  const flush = () => {
    if (currentAgents.length) {
      groups.push({ agents: currentAgents, rules: currentRules });
    }
    currentAgents = [];
    currentRules = [];
    seenRule = false;
  };

  for (const rawLine of String(content || '').split(/\r?\n/)) {
    const line = rawLine.replace(/#.*$/, '').trim();
    if (!line) continue;
    const colon = line.indexOf(':');
    if (colon === -1) continue;
    const field = line.slice(0, colon).trim().toLowerCase();
    const value = line.slice(colon + 1).trim();

    if (field === 'user-agent') {
      if (seenRule) flush();
      if (value) currentAgents.push(value.toLowerCase());
    } else if (field === 'allow' || field === 'disallow') {
      seenRule = true;
      currentRules.push({ type: field, path: value });
    } else if (field === 'sitemap' && value) {
      sitemaps.push(value);
    }
  }
  flush();
  return { groups, sitemaps };
}

function ruleMatches(rulePath, targetPath) {
  if (rulePath === '') return false; // empty Disallow means allow all (no match)
  const anchoredEnd = rulePath.endsWith('$');
  const pattern = anchoredEnd ? rulePath.slice(0, -1) : rulePath;
  const escaped = pattern
    .replace(/[.+?^${}()|[\]\\]/g, '\\$&')
    .replace(/\*/g, '.*');
  const re = new RegExp(`^${escaped}${anchoredEnd ? '$' : ''}`);
  return re.test(targetPath);
}

function evaluateBot(parsed, botName, targetPath = '/') {
  const botLower = botName.toLowerCase();
  const specificRules = [];
  const wildcardRules = [];

  for (const g of parsed.groups) {
    if (g.agents.includes(botLower)) specificRules.push(...g.rules);
    if (g.agents.includes('*')) wildcardRules.push(...g.rules);
  }

  const matchedSpecific = specificRules.length > 0;
  const activeRules = matchedSpecific ? specificRules : wildcardRules;

  let best = null;
  for (const r of activeRules) {
    if (!ruleMatches(r.path, targetPath)) continue;
    const len = r.path.length;
    if (
      !best ||
      len > best.path.length ||
      (len === best.path.length && r.type === 'allow' && best.type === 'disallow')
    ) {
      best = r;
    }
  }

  const allowed = !best || best.type === 'allow';
  let status = 'allowed_default';
  if (!allowed) status = 'blocked';
  else if (matchedSpecific) status = 'allowed_explicit';

  return {
    bot: botName,
    allowed,
    status,
    matchedRule: best ? `${best.type === 'allow' ? 'Allow' : 'Disallow'}: ${best.path}` : 'None (implicit allow)',
    hasExplicitSection: matchedSpecific,
  };
}

function auditRobots(robotsText, fetchedOk) {
  if (!fetchedOk || !robotsText || !robotsText.trim()) {
    return {
      present: false,
      sitemaps: [],
      bots: AI_BOTS.map(b => ({
        ...b,
        allowed: true,
        status: 'allowed_default',
        matchedRule: 'No robots.txt found (implicit allow)',
        hasExplicitSection: false,
      })),
      blockedBots: [],
      explicitAiRulesCount: 0,
    };
  }

  const parsed = parseRobotsTxt(robotsText);
  const bots = AI_BOTS.map(b => ({ ...b, ...evaluateBot(parsed, b.id, '/') }));
  const blockedBots = bots.filter(b => !b.allowed).map(b => b.id);
  const explicitAiRulesCount = bots.filter(b => b.id !== 'Googlebot' && b.hasExplicitSection && b.allowed).length;

  return {
    present: true,
    sitemaps: [...new Set(parsed.sitemaps)],
    bots,
    blockedBots,
    explicitAiRulesCount,
  };
}

module.exports = { AI_BOTS, parseRobotsTxt, evaluateBot, auditRobots };
