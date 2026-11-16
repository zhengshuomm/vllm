'use strict';

const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const config = require('./config');
const { openDb, seed, hashPassword, verifyPassword, formatCode } = require('./db');
const { toCsv, parseCsv } = require('./lib/csv');
const { buildXlsx } = require('./lib/xlsx');

const SESSION_TTL_MS = 7 * 24 * 3600 * 1000;
const MAX_BODY = 5 * 1024 * 1024;
const ROLE_LEVEL = { viewer: 1, editor: 2, admin: 3 };
const FIELDS = config.fields;
const FIELD_BY_KEY = Object.fromEntries(FIELDS.map(f => [f.key, f]));
const PUBLIC_DIR = path.join(__dirname, 'public');

class HttpError extends Error {
  constructor(status, message, details) {
    super(message);
    this.status = status;
    this.details = details;
  }
}

// ---------- helpers ----------

function sendJson(res, status, body) {
  const data = JSON.stringify(body);
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
  });
  res.end(data);
}

function sendFile(res, status, body, contentType, filename) {
  const headers = { 'Content-Type': contentType, 'Cache-Control': 'no-store' };
  if (filename) headers['Content-Disposition'] = `attachment; filename="${filename}"`;
  res.writeHead(status, headers);
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    req.on('data', c => {
      size += c.length;
      if (size > MAX_BODY) {
        reject(new HttpError(413, 'Ukuran data terlalu besar (maks. 5 MB).'));
        req.destroy();
      } else {
        chunks.push(c);
      }
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

async function readJson(req) {
  const raw = await readBody(req);
  if (!raw) return {};
  try {
    return JSON.parse(raw);
  } catch {
    throw new HttpError(400, 'Format JSON tidak valid.');
  }
}

function parseCookies(req) {
  const out = {};
  for (const part of (req.headers.cookie || '').split(';')) {
    const i = part.indexOf('=');
    if (i > 0) out[part.slice(0, i).trim()] = decodeURIComponent(part.slice(i + 1).trim());
  }
  return out;
}

const sha256 = s => crypto.createHash('sha256').update(s).digest('hex');
const today = () => new Date().toISOString().slice(0, 10);
const jsonPath = key => `json_extract(data, '$.${key}')`;

// ---------- app ----------

function createApp(db) {
  const sseClients = new Set();
  const loginAttempts = new Map();

  const stmt = {
    userByName: db.prepare('SELECT * FROM users WHERE username = ?'),
    userById: db.prepare('SELECT * FROM users WHERE id = ?'),
    sessionUser: db.prepare(`SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id
                             WHERE s.token_hash = ? AND s.expires_at > ? AND u.active = 1`),
    insertSession: db.prepare('INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)'),
    deleteSession: db.prepare('DELETE FROM sessions WHERE token_hash = ?'),
    deleteUserSessions: db.prepare('DELETE FROM sessions WHERE user_id = ?'),
    purgeSessions: db.prepare('DELETE FROM sessions WHERE expires_at <= ?'),
    recordById: db.prepare('SELECT * FROM records WHERE id = ?'),
    insertRecord: db.prepare('INSERT INTO records (data, created_by, updated_by) VALUES (?, ?, ?)'),
    setCode: db.prepare('UPDATE records SET code = ? WHERE id = ?'),
    updateRecord: db.prepare(`UPDATE records SET data = ?, updated_by = ?, updated_at = datetime('now')
                              WHERE id = ?`),
    deleteRecord: db.prepare('DELETE FROM records WHERE id = ?'),
    audit: db.prepare('INSERT INTO audit_log (username, action, target, detail) VALUES (?, ?, ?, ?)'),
  };

  function audit(user, action, target, detail) {
    stmt.audit.run(user ? user.username : null, action, target || null,
      detail ? JSON.stringify(detail) : null);
  }

  function broadcast(event) {
    const msg = `event: change\ndata: ${JSON.stringify(event)}\n\n`;
    for (const res of sseClients) res.write(msg);
  }

  function publicUser(u) {
    return { id: u.id, username: u.username, name: u.name, role: u.role, active: !!u.active };
  }

  function currentUser(req) {
    const token = parseCookies(req).sid;
    if (!token) return null;
    return stmt.sessionUser.get(sha256(token), Date.now()) || null;
  }

  function requireRole(req, role) {
    const user = currentUser(req);
    if (!user) throw new HttpError(401, 'Silakan login terlebih dahulu.');
    if (ROLE_LEVEL[user.role] < ROLE_LEVEL[role]) {
      throw new HttpError(403, 'Anda tidak memiliki hak akses untuk tindakan ini.');
    }
    return user;
  }

  // ---------- record helpers ----------

  function rowToRecord(row) {
    return {
      id: row.id,
      code: row.code,
      ...JSON.parse(row.data),
      created_at: row.created_at,
      created_by: row.created_by,
      updated_at: row.updated_at,
      updated_by: row.updated_by,
    };
  }

  function validateRecord(input) {
    const errors = {};
    const out = {};
    for (const f of FIELDS) {
      let v = input[f.key];
      if (typeof v === 'string') v = v.trim();
      const empty = v === undefined || v === null || v === '';
      if (empty) {
        if (f.required) errors[f.key] = `${f.label} wajib diisi.`;
        out[f.key] = f.type === 'number' ? null : '';
        continue;
      }
      if (f.type === 'number') {
        const n = typeof v === 'number' ? v : Number(String(v).replace(/\s/g, '').replace(',', '.'));
        if (!Number.isFinite(n)) errors[f.key] = `${f.label} harus berupa angka.`;
        else if (f.min !== undefined && n < f.min) errors[f.key] = `${f.label} minimal ${f.min}.`;
        else out[f.key] = n;
      } else if (f.type === 'date') {
        const s = String(v);
        if (!/^\d{4}-\d{2}-\d{2}$/.test(s) || Number.isNaN(Date.parse(s))) {
          errors[f.key] = `${f.label} harus berformat tanggal (YYYY-MM-DD).`;
        } else out[f.key] = s;
      } else if (f.type === 'select') {
        const s = String(v);
        const match = f.options.find(o => o.toLowerCase() === s.toLowerCase());
        if (!match) errors[f.key] = `${f.label} harus salah satu dari: ${f.options.join(', ')}.`;
        else out[f.key] = match;
      } else {
        const s = String(v);
        const max = f.type === 'textarea' ? 2000 : 200;
        if (s.length > max) errors[f.key] = `${f.label} maksimal ${max} karakter.`;
        else out[f.key] = s;
      }
    }
    return { data: out, errors };
  }

  function buildWhere(params) {
    const clauses = [];
    const args = [];
    const q = (params.get('q') || '').trim();
    if (q) {
      const like = `%${q.replace(/[\\%_]/g, m => `\\${m}`)}%`;
      const searchable = FIELDS.filter(f => f.search).map(f => `${jsonPath(f.key)} LIKE ? ESCAPE '\\'`);
      searchable.push(`code LIKE ? ESCAPE '\\'`);
      clauses.push(`(${searchable.join(' OR ')})`);
      for (let i = 0; i < searchable.length; i++) args.push(like);
    }
    for (const f of FIELDS) {
      if (f.filter === true) {
        const v = params.get(f.key);
        if (v) {
          clauses.push(`${jsonPath(f.key)} = ?`);
          args.push(v);
        }
      } else if (f.filter === 'range') {
        const from = params.get(`${f.key}_from`);
        const to = params.get(`${f.key}_to`);
        if (from) {
          clauses.push(`${jsonPath(f.key)} >= ?`);
          args.push(from);
        }
        if (to) {
          clauses.push(`${jsonPath(f.key)} <= ?`);
          args.push(to);
        }
      }
    }
    return { sql: clauses.length ? `WHERE ${clauses.join(' AND ')}` : '', args };
  }

  function buildOrder(params) {
    const sort = params.get('sort') || config.defaultSort.field;
    const dir = (params.get('dir') || config.defaultSort.dir) === 'asc' ? 'ASC' : 'DESC';
    let expr;
    if (sort === 'code' || sort === 'updated_at' || sort === 'created_at') expr = sort;
    else if (FIELD_BY_KEY[sort]) {
      expr = FIELD_BY_KEY[sort].type === 'number'
        ? `CAST(${jsonPath(sort)} AS REAL)`
        : `${jsonPath(sort)} COLLATE NOCASE`;
    } else expr = 'id';
    return `ORDER BY ${expr} ${dir}, id ${dir}`;
  }

  function listRecords(params, { paginate = true } = {}) {
    const where = buildWhere(params);
    const order = buildOrder(params);
    const total = db.prepare(`SELECT COUNT(*) AS n FROM records ${where.sql}`).get(...where.args).n;
    let sql = `SELECT * FROM records ${where.sql} ${order}`;
    const args = [...where.args];
    let page = 1;
    let pageSize = total;
    if (paginate) {
      pageSize = Math.min(Math.max(parseInt(params.get('pageSize'), 10) || 20, 1), 200);
      page = Math.max(parseInt(params.get('page'), 10) || 1, 1);
      sql += ' LIMIT ? OFFSET ?';
      args.push(pageSize, (page - 1) * pageSize);
    }
    const rows = db.prepare(sql).all(...args).map(rowToRecord);
    return { rows, total, page, pageSize };
  }

  function computeStats(params) {
    const where = buildWhere(params);
    const { dateField, sumField, groupBy } = config.dashboard;
    const sumExpr = `COALESCE(SUM(CAST(${jsonPath(sumField)} AS REAL)), 0)`;
    const and = where.sql ? `${where.sql} AND` : 'WHERE';

    const totals = db.prepare(`SELECT COUNT(*) AS count, ${sumExpr} AS sum FROM records ${where.sql}`)
      .get(...where.args);

    const now = new Date();
    const monthStart = new Date(Date.UTC(now.getFullYear(), now.getMonth(), 1));
    const prevStart = new Date(Date.UTC(now.getFullYear(), now.getMonth() - 1, 1));
    const iso = d => d.toISOString().slice(0, 10);
    const periodQuery = db.prepare(`SELECT COUNT(*) AS count, ${sumExpr} AS sum FROM records
      ${and} ${jsonPath(dateField)} >= ? AND ${jsonPath(dateField)} < ?`);
    const thisMonth = periodQuery.get(...where.args, iso(monthStart), '9999-12-31');
    const lastMonth = periodQuery.get(...where.args, iso(prevStart), iso(monthStart));

    const groups = {};
    for (const key of groupBy) {
      groups[key] = db.prepare(`SELECT ${jsonPath(key)} AS label, COUNT(*) AS count, ${sumExpr} AS sum
        FROM records ${where.sql} GROUP BY label ORDER BY count DESC`).all(...where.args);
    }

    const trendStart = new Date(Date.UTC(now.getFullYear(), now.getMonth() - 11, 1));
    const trendRows = db.prepare(`SELECT substr(${jsonPath(dateField)}, 1, 7) AS month,
        COUNT(*) AS count, ${sumExpr} AS sum
      FROM records ${and} ${jsonPath(dateField)} >= ? GROUP BY month ORDER BY month`)
      .all(...where.args, iso(trendStart));
    const byMonth = Object.fromEntries(trendRows.map(r => [r.month, r]));
    const trend = [];
    for (let i = 0; i < 12; i++) {
      const d = new Date(Date.UTC(trendStart.getUTCFullYear(), trendStart.getUTCMonth() + i, 1));
      const m = d.toISOString().slice(0, 7);
      trend.push({ month: m, count: byMonth[m]?.count || 0, sum: byMonth[m]?.sum || 0 });
    }

    const recent = db.prepare(`SELECT * FROM records ${where.sql} ORDER BY updated_at DESC, id DESC LIMIT 5`)
      .all(...where.args).map(rowToRecord);

    return { totals, thisMonth, lastMonth, groups, trend, recent };
  }

  function exportColumns() {
    return [
      { key: 'code', label: 'Kode', type: 'text' },
      ...FIELDS.map(f => ({ key: f.key, label: f.label, type: f.type === 'select' || f.type === 'textarea' ? 'text' : f.type })),
      { key: 'updated_at', label: 'Terakhir Diubah', type: 'text' },
      { key: 'updated_by', label: 'Diubah Oleh', type: 'text' },
    ];
  }

  function importRecords(text, user) {
    const table = parseCsv(text).filter(r => r.some(c => c.trim() !== ''));
    if (table.length < 2) throw new HttpError(400, 'File kosong atau tidak memiliki baris data.');
    const header = table[0].map(h => h.trim().toLowerCase());
    const colIndex = {};
    for (const f of FIELDS) {
      const i = header.findIndex(h => h === f.key.toLowerCase() || h === f.label.toLowerCase());
      if (i !== -1) colIndex[f.key] = i;
    }
    const missing = FIELDS.filter(f => f.required && colIndex[f.key] === undefined).map(f => f.label);
    if (missing.length) {
      throw new HttpError(400, `Kolom wajib tidak ditemukan: ${missing.join(', ')}. Gunakan template impor.`);
    }

    const valid = [];
    const errors = [];
    table.slice(1).forEach((cells, i) => {
      const input = {};
      for (const [key, idx] of Object.entries(colIndex)) input[key] = cells[idx] ?? '';
      const { data, errors: rowErrors } = validateRecord(input);
      if (Object.keys(rowErrors).length) errors.push({ row: i + 2, messages: Object.values(rowErrors) });
      else valid.push(data);
    });
    if (errors.length) {
      return { imported: 0, errors: errors.slice(0, 50), errorCount: errors.length };
    }

    db.exec('BEGIN');
    try {
      for (const data of valid) {
        const { lastInsertRowid } = stmt.insertRecord.run(JSON.stringify(data), user.username, user.username);
        stmt.setCode.run(formatCode(lastInsertRowid), lastInsertRowid);
      }
      db.exec('COMMIT');
    } catch (e) {
      db.exec('ROLLBACK');
      throw e;
    }
    audit(user, 'import', null, { count: valid.length });
    broadcast({ type: 'import', by: user.username, count: valid.length });
    return { imported: valid.length, errors: [], errorCount: 0 };
  }

  // ---------- routes ----------

  async function handleApi(req, res, url) {
    const { pathname, searchParams: params } = url;
    const method = req.method;

    if (method !== 'GET' && req.headers['x-requested-with'] !== 'fetch') {
      throw new HttpError(403, 'Permintaan ditolak.');
    }

    if (pathname === '/api/login' && method === 'POST') {
      const ip = req.socket.remoteAddress || 'unknown';
      const att = loginAttempts.get(ip);
      if (att && att.count >= 10 && Date.now() - att.first < 15 * 60 * 1000) {
        throw new HttpError(429, 'Terlalu banyak percobaan login. Coba lagi dalam 15 menit.');
      }
      const { username = '', password = '' } = await readJson(req);
      const user = stmt.userByName.get(String(username).trim());
      if (!user || !user.active || !verifyPassword(String(password), user.password_hash)) {
        const cur = att && Date.now() - att.first < 15 * 60 * 1000 ? att : { count: 0, first: Date.now() };
        cur.count++;
        loginAttempts.set(ip, cur);
        throw new HttpError(401, 'Username atau password salah.');
      }
      loginAttempts.delete(ip);
      stmt.purgeSessions.run(Date.now());
      const token = crypto.randomBytes(32).toString('hex');
      stmt.insertSession.run(sha256(token), user.id, Date.now() + SESSION_TTL_MS);
      audit(user, 'login');
      res.setHeader('Set-Cookie',
        `sid=${token}; HttpOnly; SameSite=Lax; Path=/; Max-Age=${SESSION_TTL_MS / 1000}`);
      return sendJson(res, 200, { user: publicUser(user) });
    }

    if (pathname === '/api/logout' && method === 'POST') {
      const token = parseCookies(req).sid;
      if (token) stmt.deleteSession.run(sha256(token));
      res.setHeader('Set-Cookie', 'sid=; HttpOnly; SameSite=Lax; Path=/; Max-Age=0');
      return sendJson(res, 200, { ok: true });
    }

    if (pathname === '/api/meta' && method === 'GET') {
      const user = currentUser(req);
      return sendJson(res, 200, {
        appName: config.appName,
        appTagline: config.appTagline,
        entityLabel: config.entityLabel,
        fields: FIELDS.map(({ key, label, type, required, options, filter, list, currency, colors, search, min }) =>
          ({ key, label, type, required: !!required, options, filter, list: list !== false, currency: !!currency, colors, search: !!search, min })),
        dashboard: config.dashboard,
        defaultSort: config.defaultSort,
        demoUsers: config.showDemoHint
          ? config.demoUsers.map(u => ({ username: u.username, password: u.password, role: u.role }))
          : [],
        user: user ? publicUser(user) : null,
      });
    }

    if (pathname === '/api/events' && method === 'GET') {
      requireRole(req, 'viewer');
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-store',
        Connection: 'keep-alive',
      });
      res.write('retry: 3000\n\n');
      sseClients.add(res);
      const ping = setInterval(() => res.write(': ping\n\n'), 25000);
      req.on('close', () => {
        clearInterval(ping);
        sseClients.delete(res);
      });
      return;
    }

    if (pathname === '/api/me/password' && method === 'POST') {
      const user = requireRole(req, 'viewer');
      const { current = '', next = '' } = await readJson(req);
      if (!verifyPassword(String(current), user.password_hash)) throw new HttpError(400, 'Password lama salah.');
      if (String(next).length < 8) throw new HttpError(400, 'Password baru minimal 8 karakter.');
      db.prepare('UPDATE users SET password_hash = ? WHERE id = ?').run(hashPassword(String(next)), user.id);
      audit(user, 'change_password', user.username);
      return sendJson(res, 200, { ok: true });
    }

    // ----- records -----
    if (pathname === '/api/records' && method === 'GET') {
      requireRole(req, 'viewer');
      return sendJson(res, 200, listRecords(params));
    }

    if (pathname === '/api/records' && method === 'POST') {
      const user = requireRole(req, 'editor');
      const { data, errors } = validateRecord(await readJson(req));
      if (Object.keys(errors).length) throw new HttpError(422, 'Periksa kembali isian Anda.', errors);
      const { lastInsertRowid } = stmt.insertRecord.run(JSON.stringify(data), user.username, user.username);
      const code = formatCode(lastInsertRowid);
      stmt.setCode.run(code, lastInsertRowid);
      audit(user, 'create', code);
      broadcast({ type: 'create', code, by: user.username });
      return sendJson(res, 201, rowToRecord(stmt.recordById.get(lastInsertRowid)));
    }

    const recMatch = /^\/api\/records\/(\d+)$/.exec(pathname);
    if (recMatch) {
      const id = Number(recMatch[1]);
      const row = stmt.recordById.get(id);
      if (method === 'GET') {
        requireRole(req, 'viewer');
        if (!row) throw new HttpError(404, 'Data tidak ditemukan.');
        return sendJson(res, 200, rowToRecord(row));
      }
      if (method === 'PUT') {
        const user = requireRole(req, 'editor');
        if (!row) throw new HttpError(404, 'Data tidak ditemukan.');
        const { data, errors } = validateRecord(await readJson(req));
        if (Object.keys(errors).length) throw new HttpError(422, 'Periksa kembali isian Anda.', errors);
        const before = JSON.parse(row.data);
        const changes = {};
        for (const f of FIELDS) {
          if ((before[f.key] ?? '') !== (data[f.key] ?? '')) {
            changes[f.label] = [before[f.key] ?? '', data[f.key] ?? ''];
          }
        }
        stmt.updateRecord.run(JSON.stringify(data), user.username, id);
        if (Object.keys(changes).length) audit(user, 'update', row.code, changes);
        broadcast({ type: 'update', code: row.code, by: user.username });
        return sendJson(res, 200, rowToRecord(stmt.recordById.get(id)));
      }
      if (method === 'DELETE') {
        const user = requireRole(req, 'admin');
        if (!row) throw new HttpError(404, 'Data tidak ditemukan.');
        stmt.deleteRecord.run(id);
        audit(user, 'delete', row.code, JSON.parse(row.data));
        broadcast({ type: 'delete', code: row.code, by: user.username });
        return sendJson(res, 200, { ok: true });
      }
    }

    if (pathname === '/api/stats' && method === 'GET') {
      requireRole(req, 'viewer');
      return sendJson(res, 200, computeStats(params));
    }

    if ((pathname === '/api/export.csv' || pathname === '/api/export.xlsx') && method === 'GET') {
      const user = requireRole(req, 'viewer');
      const { rows } = listRecords(params, { paginate: false });
      const cols = exportColumns();
      const matrix = rows.map(r => cols.map(c => r[c.key] ?? ''));
      const stamp = today();
      audit(user, 'export', null, { format: pathname.endsWith('xlsx') ? 'xlsx' : 'csv', rows: rows.length });
      if (pathname.endsWith('.csv')) {
        return sendFile(res, 200, toCsv(cols.map(c => c.label), matrix),
          'text/csv; charset=utf-8', `data-${stamp}.csv`);
      }
      const buf = buildXlsx({ sheetName: config.entityLabel, columns: cols, rows: matrix });
      return sendFile(res, 200, buf,
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', `data-${stamp}.xlsx`);
    }

    if (pathname === '/api/import/template.csv' && method === 'GET') {
      requireRole(req, 'editor');
      const example = FIELDS.map(f => {
        if (f.type === 'date') return today();
        if (f.type === 'select') return f.options[0];
        if (f.type === 'number') return 1;
        return '';
      });
      return sendFile(res, 200, toCsv(FIELDS.map(f => f.label), [example]),
        'text/csv; charset=utf-8', 'template-impor.csv');
    }

    if (pathname === '/api/import' && method === 'POST') {
      const user = requireRole(req, 'editor');
      const text = await readBody(req);
      return sendJson(res, 200, importRecords(text, user));
    }

    // ----- users (admin) -----
    if (pathname === '/api/users' && method === 'GET') {
      requireRole(req, 'admin');
      const users = db.prepare('SELECT * FROM users ORDER BY id').all().map(u => ({ ...publicUser(u), created_at: u.created_at }));
      return sendJson(res, 200, users);
    }

    if (pathname === '/api/users' && method === 'POST') {
      const admin = requireRole(req, 'admin');
      const { username = '', name = '', role = '', password = '' } = await readJson(req);
      const errors = {};
      if (!/^[a-z0-9._-]{3,32}$/i.test(username)) errors.username = 'Username 3–32 karakter (huruf, angka, . _ -).';
      if (!String(name).trim()) errors.name = 'Nama wajib diisi.';
      if (!ROLE_LEVEL[role]) errors.role = 'Peran tidak valid.';
      if (String(password).length < 8) errors.password = 'Password minimal 8 karakter.';
      if (!errors.username && stmt.userByName.get(username)) errors.username = 'Username sudah dipakai.';
      if (Object.keys(errors).length) throw new HttpError(422, 'Periksa kembali isian Anda.', errors);
      const { lastInsertRowid } = db.prepare(
        'INSERT INTO users (username, name, role, password_hash) VALUES (?, ?, ?, ?)')
        .run(username, String(name).trim(), role, hashPassword(String(password)));
      audit(admin, 'user_create', username, { role });
      return sendJson(res, 201, publicUser(stmt.userById.get(lastInsertRowid)));
    }

    const userMatch = /^\/api\/users\/(\d+)$/.exec(pathname);
    if (userMatch && (method === 'PUT' || method === 'DELETE')) {
      const admin = requireRole(req, 'admin');
      const target = stmt.userById.get(Number(userMatch[1]));
      if (!target) throw new HttpError(404, 'Pengguna tidak ditemukan.');
      const activeAdmins = db.prepare("SELECT COUNT(*) AS n FROM users WHERE role = 'admin' AND active = 1").get().n;
      const isLastAdmin = target.role === 'admin' && target.active && activeAdmins <= 1;

      if (method === 'DELETE') {
        if (target.id === admin.id) throw new HttpError(400, 'Anda tidak dapat menghapus akun sendiri.');
        if (isLastAdmin) throw new HttpError(400, 'Minimal harus ada satu admin aktif.');
        db.prepare('DELETE FROM users WHERE id = ?').run(target.id);
        audit(admin, 'user_delete', target.username);
        return sendJson(res, 200, { ok: true });
      }

      const body = await readJson(req);
      const name = body.name !== undefined ? String(body.name).trim() : target.name;
      const role = body.role !== undefined ? body.role : target.role;
      const active = body.active !== undefined ? (body.active ? 1 : 0) : target.active;
      if (!name) throw new HttpError(422, 'Nama wajib diisi.', { name: 'Nama wajib diisi.' });
      if (!ROLE_LEVEL[role]) throw new HttpError(422, 'Peran tidak valid.', { role: 'Peran tidak valid.' });
      if (isLastAdmin && (role !== 'admin' || !active)) {
        throw new HttpError(400, 'Minimal harus ada satu admin aktif.');
      }
      if (body.password) {
        if (String(body.password).length < 8) {
          throw new HttpError(422, 'Password minimal 8 karakter.', { password: 'Password minimal 8 karakter.' });
        }
        db.prepare('UPDATE users SET password_hash = ? WHERE id = ?').run(hashPassword(String(body.password)), target.id);
      }
      db.prepare('UPDATE users SET name = ?, role = ?, active = ? WHERE id = ?').run(name, role, active, target.id);
      if (!active || body.password || role !== target.role) stmt.deleteUserSessions.run(target.id);
      audit(admin, 'user_update', target.username,
        { role: role !== target.role ? [target.role, role] : undefined, active: !!active, password_reset: !!body.password });
      return sendJson(res, 200, publicUser(stmt.userById.get(target.id)));
    }

    if (pathname === '/api/audit' && method === 'GET') {
      requireRole(req, 'admin');
      const limit = Math.min(Math.max(parseInt(params.get('limit'), 10) || 100, 1), 500);
      const rows = db.prepare('SELECT * FROM audit_log ORDER BY id DESC LIMIT ?').all(limit)
        .map(r => ({ ...r, detail: r.detail ? JSON.parse(r.detail) : null }));
      return sendJson(res, 200, rows);
    }

    throw new HttpError(404, 'Endpoint tidak ditemukan.');
  }

  const MIME = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon',
  };

  function serveStatic(req, res, url) {
    let rel = decodeURIComponent(url.pathname);
    if (rel === '/') rel = '/index.html';
    const file = path.normalize(path.join(PUBLIC_DIR, rel));
    if (!file.startsWith(PUBLIC_DIR + path.sep)) throw new HttpError(404, 'Not found');
    fs.readFile(file, (err, data) => {
      if (err) {
        res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
        return res.end('Not found');
      }
      res.writeHead(200, {
        'Content-Type': MIME[path.extname(file)] || 'application/octet-stream',
        'Cache-Control': 'no-cache',
        'X-Content-Type-Options': 'nosniff',
      });
      res.end(data);
    });
  }

  return async function handler(req, res) {
    res.setHeader('X-Frame-Options', 'DENY');
    res.setHeader('Referrer-Policy', 'same-origin');
    const url = new URL(req.url, 'http://localhost');
    try {
      if (url.pathname.startsWith('/api/')) await handleApi(req, res, url);
      else serveStatic(req, res, url);
    } catch (err) {
      if (res.headersSent) return res.end();
      if (err instanceof HttpError) {
        return sendJson(res, err.status, { error: err.message, details: err.details });
      }
      console.error(err);
      sendJson(res, 500, { error: 'Terjadi kesalahan pada server.' });
    }
  };
}

function start() {
  const db = openDb();
  seed(db);
  const server = http.createServer(createApp(db));
  server.listen(config.port, () => {
    console.log(`${config.appName} berjalan di http://localhost:${config.port}`);
  });
  return server;
}

if (require.main === module) start();

module.exports = { createApp, openDb, seed };
