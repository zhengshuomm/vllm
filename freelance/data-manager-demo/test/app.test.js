'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');
const { createApp, openDb, seed } = require('../server');

let server;
let base;

test.before(async () => {
  process.env.SEED_DEMO = '1';
  const db = openDb(':memory:');
  seed(db);
  server = http.createServer(createApp(db));
  await new Promise(r => server.listen(0, r));
  base = `http://127.0.0.1:${server.address().port}`;
});

test.after(() => server.close());

async function login(username, password) {
  const res = await fetch(`${base}/api/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'fetch' },
    body: JSON.stringify({ username, password }),
  });
  assert.equal(res.status, 200);
  const cookie = res.headers.get('set-cookie').split(';')[0];
  return (path, { method = 'GET', body, raw } = {}) => fetch(`${base}${path}`, {
    method,
    headers: {
      Cookie: cookie,
      'X-Requested-With': 'fetch',
      'Content-Type': raw ? 'text/plain' : 'application/json',
    },
    body: body === undefined ? undefined : raw ? body : JSON.stringify(body),
  });
}

const validRecord = {
  tanggal: '2026-11-01', pelanggan: 'PT Uji Coba', kategori: 'Fashion', wilayah: 'Bandung',
  jumlah: 3, nilai: 450000, status: 'Baru', pic: 'Tester', catatan: '',
};

test('rejects unauthenticated access and wrong password', async () => {
  assert.equal((await fetch(`${base}/api/records`)).status, 401);
  const res = await fetch(`${base}/api/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'fetch' },
    body: JSON.stringify({ username: 'admin', password: 'wrong' }),
  });
  assert.equal(res.status, 401);
});

test('blocks state-changing requests without the fetch header', async () => {
  const res = await fetch(`${base}/api/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username: 'admin', password: 'admin123' }),
  });
  assert.equal(res.status, 403);
});

test('viewer can read and export but cannot modify', async () => {
  const viewer = await login('viewer', 'viewer123');
  const list = await (await viewer('/api/records?pageSize=5')).json();
  assert.equal(list.rows.length, 5);
  assert.ok(list.total >= 150);
  assert.equal((await viewer('/api/records', { method: 'POST', body: validRecord })).status, 403);
  assert.equal((await viewer(`/api/records/${list.rows[0].id}`, { method: 'DELETE' })).status, 403);
  assert.equal((await viewer('/api/users')).status, 403);
  assert.equal((await viewer('/api/export.csv')).status, 200);
});

test('editor can create and update with field validation, but not delete', async () => {
  const editor = await login('editor', 'editor123');
  const bad = await editor('/api/records', { method: 'POST', body: { ...validRecord, nilai: 'abc', kategori: 'X' } });
  assert.equal(bad.status, 422);
  const badBody = await bad.json();
  assert.ok(badBody.details.nilai && badBody.details.kategori);

  const created = await (await editor('/api/records', { method: 'POST', body: validRecord })).json();
  assert.match(created.code, /^ORD-\d{5}$/);
  assert.equal(created.created_by, 'editor');

  const upd = await editor(`/api/records/${created.id}`, { method: 'PUT', body: { ...validRecord, status: 'Selesai' } });
  assert.equal((await upd.json()).status, 'Selesai');
  assert.equal((await editor(`/api/records/${created.id}`, { method: 'DELETE' })).status, 403);
});

test('search, filter, and sort narrow results', async () => {
  const admin = await login('admin', 'admin123');
  const all = await (await admin('/api/records?pageSize=200')).json();
  const filtered = await (await admin('/api/records?wilayah=Jakarta&status=Selesai&pageSize=200')).json();
  assert.ok(filtered.total > 0 && filtered.total < all.total);
  assert.ok(filtered.rows.every(r => r.wilayah === 'Jakarta' && r.status === 'Selesai'));

  const search = await (await admin('/api/records?q=maju')).json();
  assert.ok(search.total > 0);
  assert.ok(search.rows.every(r => /maju/i.test(`${r.pelanggan} ${r.pic} ${r.catatan} ${r.code}`)));

  const sorted = await (await admin('/api/records?sort=nilai&dir=desc&pageSize=50')).json();
  for (let i = 1; i < sorted.rows.length; i++) assert.ok(sorted.rows[i - 1].nilai >= sorted.rows[i].nilai);
});

test('admin delete is recorded in the audit log', async () => {
  const admin = await login('admin', 'admin123');
  const { rows } = await (await admin('/api/records?pageSize=1')).json();
  assert.equal((await admin(`/api/records/${rows[0].id}`, { method: 'DELETE' })).status, 200);
  assert.equal((await admin(`/api/records/${rows[0].id}`)).status, 404);
  const log = await (await admin('/api/audit')).json();
  assert.ok(log.some(a => a.action === 'delete' && a.target === rows[0].code));
});

test('import is all-or-nothing and accepts semicolon CSV from Excel', async () => {
  const editor = await login('editor', 'editor123');
  const before = (await (await editor('/api/records?pageSize=1')).json()).total;

  const bad = 'Tanggal;Nama Pelanggan;Kategori;Wilayah;Jumlah;Nilai (Rp);Status\n' +
    '2026-11-02;PT A;Fashion;Bandung;1;1000;Baru\n' +
    'bukan-tanggal;PT B;Fashion;Bandung;1;1000;Baru\n';
  const badRes = await (await editor('/api/import', { method: 'POST', body: bad, raw: true })).json();
  assert.equal(badRes.imported, 0);
  assert.equal(badRes.errors[0].row, 3);
  assert.equal((await (await editor('/api/records?pageSize=1')).json()).total, before);

  const good = bad.replace('bukan-tanggal', '2026-11-03');
  const goodRes = await (await editor('/api/import', { method: 'POST', body: good, raw: true })).json();
  assert.equal(goodRes.imported, 2);
  assert.equal((await (await editor('/api/records?pageSize=1')).json()).total, before + 2);
});

test('exports neutralize spreadsheet formulas and produce a valid xlsx', async () => {
  const admin = await login('admin', 'admin123');
  await admin('/api/records', { method: 'POST', body: { ...validRecord, pelanggan: '=HYPERLINK("http://x")' } });
  const csv = await (await admin('/api/export.csv?q=HYPERLINK')).text();
  assert.ok(csv.includes(`"'=HYPERLINK(""http://x"")"`));

  const xlsx = Buffer.from(await (await admin('/api/export.xlsx')).arrayBuffer());
  assert.equal(xlsx.readUInt32LE(0), 0x04034b50);
  assert.ok(xlsx.includes(Buffer.from('xl/worksheets/sheet1.xml')));
});

test('dashboard stats respect filters', async () => {
  const viewer = await login('viewer', 'viewer123');
  const all = await (await viewer('/api/stats')).json();
  const sby = await (await viewer('/api/stats?wilayah=Surabaya')).json();
  assert.equal(all.trend.length, 12);
  assert.ok(sby.totals.count < all.totals.count);
  assert.deepEqual(sby.groups.wilayah.map(g => g.label), ['Surabaya']);
});

test('cannot remove the last active admin', async () => {
  const admin = await login('admin', 'admin123');
  const users = await (await admin('/api/users')).json();
  const me = users.find(u => u.username === 'admin');
  const res = await admin(`/api/users/${me.id}`, { method: 'PUT', body: { role: 'viewer' } });
  assert.equal(res.status, 400);
});
