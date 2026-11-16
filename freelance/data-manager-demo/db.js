'use strict';

const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { DatabaseSync } = require('node:sqlite');
const config = require('./config');

const KEY_RE = /^[a-z_][a-z0-9_]*$/i;
for (const f of config.fields) {
  if (!KEY_RE.test(f.key)) throw new Error(`Invalid field key in config: ${f.key}`);
}

function hashPassword(password) {
  const salt = crypto.randomBytes(16);
  const hash = crypto.scryptSync(password, salt, 64);
  return `scrypt$${salt.toString('hex')}$${hash.toString('hex')}`;
}

function verifyPassword(password, stored) {
  const [scheme, saltHex, hashHex] = String(stored).split('$');
  if (scheme !== 'scrypt' || !saltHex || !hashHex) return false;
  const expected = Buffer.from(hashHex, 'hex');
  const actual = crypto.scryptSync(password, Buffer.from(saltHex, 'hex'), expected.length);
  return crypto.timingSafeEqual(expected, actual);
}

function makeRandom(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function openDb(file = process.env.DB_FILE || path.join(__dirname, 'data', 'app.db')) {
  if (file !== ':memory:') fs.mkdirSync(path.dirname(file), { recursive: true });
  const db = new DatabaseSync(file);
  db.exec(`
    PRAGMA journal_mode = WAL;
    PRAGMA foreign_keys = ON;
    CREATE TABLE IF NOT EXISTS users (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      username TEXT NOT NULL UNIQUE COLLATE NOCASE,
      name TEXT NOT NULL,
      role TEXT NOT NULL CHECK (role IN ('admin', 'editor', 'viewer')),
      password_hash TEXT NOT NULL,
      active INTEGER NOT NULL DEFAULT 1,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS sessions (
      token_hash TEXT PRIMARY KEY,
      user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      expires_at INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS records (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      code TEXT UNIQUE,
      data TEXT NOT NULL,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      created_by TEXT,
      updated_at TEXT NOT NULL DEFAULT (datetime('now')),
      updated_by TEXT
    );
    CREATE TABLE IF NOT EXISTS audit_log (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts TEXT NOT NULL DEFAULT (datetime('now')),
      username TEXT,
      action TEXT NOT NULL,
      target TEXT,
      detail TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts DESC);
  `);
  return db;
}

function seed(db) {
  const userCount = db.prepare('SELECT COUNT(*) AS n FROM users').get().n;
  if (userCount === 0) {
    const insert = db.prepare('INSERT INTO users (username, name, role, password_hash) VALUES (?, ?, ?, ?)');
    for (const u of config.demoUsers) insert.run(u.username, u.name, u.role, hashPassword(u.password));
  }

  const recordCount = db.prepare('SELECT COUNT(*) AS n FROM records').get().n;
  if (recordCount > 0 || process.env.SEED_DEMO === '0') return;

  const rand = makeRandom(20261115);
  const pick = arr => arr[Math.floor(rand() * arr.length)];
  const samples = config.demoSamples || {};
  const today = new Date();
  const rows = [];
  for (let i = 0; i < config.demoRecordCount; i++) {
    const row = {};
    for (const f of config.fields) {
      const sample = samples[f.key];
      if (typeof sample === 'function') continue;
      if (Array.isArray(sample)) row[f.key] = pick(sample);
      else if (f.type === 'select') row[f.key] = pick(f.options);
      else if (f.type === 'date') {
        const d = new Date(today);
        d.setDate(d.getDate() - Math.floor(rand() * 365));
        row[f.key] = d.toISOString().slice(0, 10);
      } else if (f.type === 'number') row[f.key] = Math.floor(rand() * 100);
      else row[f.key] = '';
    }
    for (const f of config.fields) {
      const sample = samples[f.key];
      if (typeof sample === 'function') row[f.key] = sample(rand, row);
    }
    rows.push(row);
  }
  const dateKey = config.dashboard.dateField;
  rows.sort((a, b) => String(a[dateKey]).localeCompare(String(b[dateKey])));

  const insert = db.prepare(
    'INSERT INTO records (data, created_by, updated_by, created_at, updated_at) VALUES (?, ?, ?, ?, ?)');
  const setCode = db.prepare('UPDATE records SET code = ? WHERE id = ?');
  db.exec('BEGIN');
  try {
    for (const row of rows) {
      const ts = `${row[dateKey]} 02:00:00`; // 09:00 WIB
      const { lastInsertRowid } = insert.run(JSON.stringify(row), 'admin', 'admin', ts, ts);
      setCode.run(formatCode(lastInsertRowid), lastInsertRowid);
    }
    db.exec('COMMIT');
  } catch (e) {
    db.exec('ROLLBACK');
    throw e;
  }
}

function formatCode(id) {
  return `${config.codePrefix}-${String(id).padStart(5, '0')}`;
}

module.exports = { openDb, seed, hashPassword, verifyPassword, formatCode };
