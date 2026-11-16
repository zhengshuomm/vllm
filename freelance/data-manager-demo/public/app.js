'use strict';

const state = {
  meta: null,
  user: null,
  data: { q: '', filters: {}, sort: null, dir: null, page: 1, pageSize: 20 },
  dash: { filters: {} },
  flashCode: null,
  events: null,
  live: false,
};

const ROLE_LABEL = { admin: 'Admin', editor: 'Editor', viewer: 'Viewer' };
const ROLE_COLOR = { admin: 'purple', editor: 'blue', viewer: 'gray' };
const ROLE_DESC = {
  admin: 'Akses penuh: kelola data, hapus data, kelola pengguna, lihat log aktivitas.',
  editor: 'Tambah, ubah, dan impor data. Tidak dapat menghapus atau mengelola pengguna.',
  viewer: 'Hanya melihat, mencari, dan mengekspor data.',
};
const ACTION_LABEL = {
  login: 'Login', create: 'Menambah data', update: 'Mengubah data', delete: 'Menghapus data',
  import: 'Impor data', export: 'Ekspor data', user_create: 'Menambah pengguna',
  user_update: 'Mengubah pengguna', user_delete: 'Menghapus pengguna', change_password: 'Ganti password',
};

// ---------- utils ----------

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const can = role => ({ viewer: 1, editor: 2, admin: 3 }[state.user?.role] || 0) >= { viewer: 1, editor: 2, admin: 3 }[role];

const fmtNum = n => new Intl.NumberFormat('id-ID').format(n ?? 0);
const fmtIdr = n => new Intl.NumberFormat('id-ID', { style: 'currency', currency: 'IDR', maximumFractionDigits: 0 }).format(n ?? 0);
const fmtIdrShort = n => new Intl.NumberFormat('id-ID', { style: 'currency', currency: 'IDR', notation: 'compact', maximumFractionDigits: 1 }).format(n ?? 0);
const fmtDate = s => {
  if (!s) return '';
  const d = new Date(/^\d{4}-\d{2}-\d{2}$/.test(s) ? `${s}T00:00:00` : s.replace(' ', 'T') + 'Z');
  return Number.isNaN(d.getTime()) ? s : d.toLocaleDateString('id-ID', { day: '2-digit', month: 'short', year: 'numeric' });
};
const fmtDateTime = s => {
  if (!s) return '';
  const d = new Date(s.replace(' ', 'T') + 'Z');
  return Number.isNaN(d.getTime()) ? s : d.toLocaleString('id-ID', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });
};
const monthLabel = m => new Date(`${m}-01T00:00:00`).toLocaleDateString('id-ID', { month: 'short' });

function fmtValue(field, value) {
  if (value === null || value === undefined || value === '') return '<span class="muted">—</span>';
  if (field.type === 'number') return field.currency ? esc(fmtIdr(value)) : esc(fmtNum(value));
  if (field.type === 'date') return esc(fmtDate(value));
  if (field.type === 'select' && field.colors) return `<span class="badge ${esc(field.colors[value] || '')}">${esc(value)}</span>`;
  return esc(value);
}

async function api(path, { method = 'GET', body, raw } = {}) {
  const opts = { method, headers: {} };
  if (method !== 'GET') opts.headers['X-Requested-With'] = 'fetch';
  if (body !== undefined) {
    if (raw) {
      opts.body = body;
      opts.headers['Content-Type'] = 'text/plain; charset=utf-8';
    } else {
      opts.body = JSON.stringify(body);
      opts.headers['Content-Type'] = 'application/json';
    }
  }
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({}));
  if (res.status === 401 && path !== '/api/login') {
    state.user = null;
    disconnectEvents();
    renderLogin();
  }
  if (!res.ok) {
    const err = new Error(data.error || `Error ${res.status}`);
    err.details = data.details;
    err.status = res.status;
    throw err;
  }
  return data;
}

function toast(message, type = '') {
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.textContent = message;
  $('#toast-root').appendChild(el);
  setTimeout(() => el.remove(), 3500);
}

function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

function queryString(obj) {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(obj)) if (v !== '' && v !== null && v !== undefined) p.set(k, v);
  return p.toString();
}

// ---------- modal ----------

function openModal({ title, body, footer, size = '', onMount }) {
  const root = $('#modal-root');
  root.innerHTML = `
    <div class="modal-backdrop">
      <div class="modal card ${size}" role="dialog" aria-modal="true">
        <div class="modal-head"><h2>${esc(title)}</h2><button class="close-x" data-close aria-label="Tutup">×</button></div>
        <div class="modal-body">${body}</div>
        ${footer ? `<div class="modal-foot">${footer}</div>` : ''}
      </div>
    </div>`;
  const backdrop = $('.modal-backdrop', root);
  backdrop.addEventListener('mousedown', e => { if (e.target === backdrop) closeModal(); });
  $$('[data-close]', root).forEach(b => b.addEventListener('click', closeModal));
  document.addEventListener('keydown', escListener);
  if (onMount) onMount($('.modal', root));
  const first = $('input:not([type=hidden]), select, textarea', root);
  if (first) first.focus();
}

function escListener(e) { if (e.key === 'Escape') closeModal(); }
function closeModal() {
  $('#modal-root').innerHTML = '';
  document.removeEventListener('keydown', escListener);
}

function confirmDialog({ title, message, confirmText = 'Ya', danger = false }) {
  return new Promise(resolve => {
    openModal({
      title,
      size: 'sm',
      body: `<p style="margin:0">${message}</p>`,
      footer: `<button class="btn" data-close>Batal</button>
               <button class="btn ${danger ? 'btn-danger-solid' : 'btn-primary'}" id="confirm-yes">${esc(confirmText)}</button>`,
      onMount: m => {
        $('#confirm-yes', m).addEventListener('click', () => { closeModal(); resolve(true); });
        $$('[data-close]', m).forEach(b => b.addEventListener('click', () => resolve(false)));
      },
    });
  });
}

function showFieldErrors(form, details) {
  $$('.field', form).forEach(f => {
    f.classList.remove('has-error');
    $('.field-error', f)?.remove();
  });
  for (const [key, msg] of Object.entries(details || {})) {
    const wrap = $(`[data-field="${key}"]`, form);
    if (!wrap) continue;
    wrap.classList.add('has-error');
    wrap.insertAdjacentHTML('beforeend', `<div class="field-error">${esc(msg)}</div>`);
  }
}

// ---------- live updates ----------

function connectEvents() {
  if (state.events) return;
  const es = new EventSource('/api/events');
  state.events = es;
  es.onopen = () => setLive(true);
  es.onerror = () => setLive(false);
  es.addEventListener('change', e => {
    const ev = JSON.parse(e.data);
    if (ev.by !== state.user?.username) {
      const verb = { create: 'menambah', update: 'mengubah', delete: 'menghapus', import: `mengimpor ${ev.count}` }[ev.type];
      toast(`${ev.by} ${verb} data${ev.code ? ` ${ev.code}` : ''}`);
      if (ev.type !== 'delete') state.flashCode = ev.code;
    }
    refreshCurrent();
  });
}

function disconnectEvents() {
  state.events?.close();
  state.events = null;
}

function setLive(on) {
  state.live = on;
  const dot = $('#live-dot');
  if (dot) {
    dot.classList.toggle('off', !on);
    dot.parentElement.title = on ? 'Terhubung — data diperbarui otomatis' : 'Koneksi terputus, mencoba lagi…';
  }
}

const refreshCurrent = debounce(() => {
  const route = currentRoute();
  if (route === 'data') loadRecords();
  if (route === 'dashboard') loadDashboard();
}, 300);

// ---------- login ----------

function renderLogin(error = '') {
  const { meta } = state;
  const demo = meta.demoUsers.length
    ? `<div class="demo-box">
        <div style="margin-bottom:4px"><b>Akun demo</b><span class="muted">klik untuk mengisi</span></div>
        ${meta.demoUsers.map(u => `<div><span>${esc(ROLE_LABEL[u.role])}</span>
          <button type="button" data-user="${esc(u.username)}" data-pass="${esc(u.password)}">${esc(u.username)} / ${esc(u.password)}</button></div>`).join('')}
       </div>`
    : '';
  $('#app').innerHTML = `
    <div class="login-wrap">
      <form class="card login-card" id="login-form" autocomplete="on">
        <div class="brand"><div class="brand-logo" style="color:#fff">${esc(meta.appName[0])}</div>
          <div><h1>${esc(meta.appName)}</h1><small>${esc(meta.appTagline)}</small></div></div>
        ${error ? `<div class="alert">${esc(error)}</div>` : ''}
        <div class="field"><label for="username">Username</label><input id="username" name="username" autocomplete="username" required></div>
        <div class="field"><label for="password">Password</label><input id="password" name="password" type="password" autocomplete="current-password" required></div>
        <button class="btn btn-primary" type="submit">Masuk</button>
        ${demo}
      </form>
    </div>`;
  const form = $('#login-form');
  $$('[data-user]', form).forEach(b => b.addEventListener('click', () => {
    form.username.value = b.dataset.user;
    form.password.value = b.dataset.pass;
  }));
  form.addEventListener('submit', async e => {
    e.preventDefault();
    const btn = $('button[type=submit]', form);
    btn.disabled = true;
    try {
      const { user } = await api('/api/login', {
        method: 'POST',
        body: { username: form.username.value, password: form.password.value },
      });
      state.user = user;
      if (!location.hash || location.hash === '#/') location.hash = '#/dashboard';
      renderShell();
    } catch (err) {
      renderLogin(err.message);
    }
  });
  form.username.focus();
}

// ---------- shell ----------

function currentRoute() {
  const r = (location.hash.replace(/^#\/?/, '') || 'dashboard').split('?')[0];
  if ((r === 'users' || r === 'activity') && !can('admin')) return 'dashboard';
  return ['dashboard', 'data', 'users', 'activity', 'account'].includes(r) ? r : 'dashboard';
}

function renderShell() {
  const { meta, user } = state;
  const link = (route, icon, label) => `<a href="#/${route}" data-route="${route}">${icon}<span>${label}</span></a>`;
  $('#app').innerHTML = `
    <div class="layout">
      <aside class="sidebar">
        <div class="brand"><div class="brand-logo">${esc(meta.appName[0])}</div>
          <div><b>${esc(meta.appName)}</b><small>${esc(meta.appTagline)}</small></div></div>
        <nav class="nav">
          ${link('dashboard', '▦', 'Dasbor')}
          ${link('data', '☰', esc(meta.entityLabel))}
          ${can('admin') ? `<div class="section">Admin</div>${link('users', '👥', 'Pengguna')}${link('activity', '🕑', 'Log Aktivitas')}` : ''}
          <div class="section">Akun</div>
          ${link('account', '⚙', 'Akun Saya')}
        </nav>
        <div class="sidebar-footer">
          <div class="user-chip">
            <div class="avatar">${esc(user.name[0].toUpperCase())}</div>
            <div><div class="name">${esc(user.name)}</div><span class="badge ${ROLE_COLOR[user.role]}">${ROLE_LABEL[user.role]}</span></div>
          </div>
          <button class="btn btn-ghost btn-sm" id="logout">Keluar</button>
        </div>
      </aside>
      <main class="main" id="view"></main>
    </div>`;
  $('#logout').addEventListener('click', async () => {
    await api('/api/logout', { method: 'POST' }).catch(() => {});
    state.user = null;
    disconnectEvents();
    renderLogin();
  });
  connectEvents();
  renderRoute();
}

function renderRoute() {
  if (!state.user) return;
  const route = currentRoute();
  $$('.nav a').forEach(a => a.classList.toggle('active', a.dataset.route === route));
  closeModal();
  ({ dashboard: viewDashboard, data: viewData, users: viewUsers, activity: viewActivity, account: viewAccount })[route]();
}

window.addEventListener('hashchange', renderRoute);

// ---------- filters (shared) ----------

function filterControls(filters, prefix) {
  return state.meta.fields.map(f => {
    if (f.filter === true) {
      return `<select data-filter="${f.key}" id="${prefix}-${f.key}" aria-label="${esc(f.label)}">
        <option value="">Semua ${esc(f.label)}</option>
        ${f.options.map(o => `<option ${filters[f.key] === o ? 'selected' : ''}>${esc(o)}</option>`).join('')}
      </select>`;
    }
    if (f.filter === 'range') {
      return `<input type="date" data-filter="${f.key}_from" value="${esc(filters[`${f.key}_from`] || '')}" title="${esc(f.label)} dari" aria-label="${esc(f.label)} dari">
              <span class="muted">s/d</span>
              <input type="date" data-filter="${f.key}_to" value="${esc(filters[`${f.key}_to`] || '')}" title="${esc(f.label)} sampai" aria-label="${esc(f.label)} sampai">`;
    }
    return '';
  }).join('');
}

function bindFilters(root, filters, onChange) {
  $$('[data-filter]', root).forEach(el => el.addEventListener('change', () => {
    filters[el.dataset.filter] = el.value;
    onChange();
  }));
}

// ---------- dashboard ----------

function viewDashboard() {
  const f = state.dash.filters;
  $('#view').innerHTML = `
    <div class="page-header">
      <div><h1>Dasbor</h1><p class="muted"><span title=""><span class="live-dot ${state.live ? '' : 'off'}" id="live-dot"></span></span>Ringkasan data terkini</p></div>
      <div class="filter-bar-dash" id="dash-filters">
        ${filterControls(f, 'dash')}
        <button class="btn btn-sm" id="dash-reset">Reset</button>
      </div>
    </div>
    <div id="dash-body"><div class="empty">Memuat…</div></div>`;
  bindFilters($('#dash-filters'), f, loadDashboard);
  $('#dash-reset').addEventListener('click', () => { state.dash.filters = {}; viewDashboard(); });
  loadDashboard();
}

function delta(cur, prev) {
  if (!prev) return cur ? '<div class="delta up">Baru bulan ini</div>' : '<div class="delta muted">—</div>';
  const pct = ((cur - prev) / prev) * 100;
  const cls = pct >= 0 ? 'up' : 'down';
  return `<div class="delta ${cls}">${pct >= 0 ? '▲' : '▼'} ${Math.abs(pct).toFixed(1)}% vs bulan lalu</div>`;
}

function trendChart(trend) {
  const W = 640, H = 220, pad = { l: 8, r: 8, t: 18, b: 26 };
  const max = Math.max(...trend.map(t => t.sum), 1);
  const bw = (W - pad.l - pad.r) / trend.length;
  const bars = trend.map((t, i) => {
    const h = (t.sum / max) * (H - pad.t - pad.b);
    const x = pad.l + i * bw + bw * 0.18;
    const y = H - pad.b - h;
    return `<g><rect class="bar" x="${x}" y="${y}" width="${bw * 0.64}" height="${Math.max(h, 1)}" rx="4">
        <title>${esc(monthLabel(t.month))} ${t.month.slice(0, 4)}: ${esc(fmtIdr(t.sum))} (${t.count} data)</title></rect>
      ${t.sum ? `<text x="${x + bw * 0.32}" y="${y - 5}" text-anchor="middle">${esc(fmtIdrShort(t.sum).replace('Rp', '').trim())}</text>` : ''}
      <text x="${x + bw * 0.32}" y="${H - 8}" text-anchor="middle">${esc(monthLabel(t.month))}</text></g>`;
  }).join('');
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Grafik tren bulanan">${bars}</svg>`;
}

function groupCard(key, rows, sumField) {
  const field = state.meta.fields.find(f => f.key === key);
  const max = Math.max(...rows.map(r => r.count), 1);
  return `<div class="card"><div class="card-header"><h2>Per ${esc(field?.label || key)}</h2></div>
    <div class="card-body">
      ${rows.length ? rows.map(r => `
        <div class="hbar">
          <div class="hbar-top"><span>${esc(r.label ?? '—')}</span><span class="muted">${fmtNum(r.count)}${sumField ? ` · ${esc(fmtIdrShort(r.sum))}` : ''}</span></div>
          <div class="hbar-track"><div class="hbar-fill" style="width:${(r.count / max) * 100}%"></div></div>
        </div>`).join('') : '<div class="muted">Belum ada data</div>'}
    </div></div>`;
}

async function loadDashboard() {
  let s;
  try {
    s = await api(`/api/stats?${queryString(state.dash.filters)}`);
  } catch (err) {
    if (err.status !== 401) toast(err.message, 'error');
    return;
  }
  const body = $('#dash-body');
  if (!body) return;
  const { meta } = state;
  const sumField = meta.fields.find(f => f.key === meta.dashboard.sumField);
  const listFields = meta.fields.filter(f => f.list).slice(0, 3);
  body.innerHTML = `
    <div class="kpis">
      <div class="card card-body kpi"><div class="label">Total ${esc(meta.entityLabel)}</div><div class="value">${fmtNum(s.totals.count)}</div><div class="delta muted">Sesuai filter</div></div>
      <div class="card card-body kpi"><div class="label">Total ${esc(sumField?.label || 'Nilai')}</div><div class="value" title="${esc(fmtIdr(s.totals.sum))}">${esc(fmtIdrShort(s.totals.sum))}</div><div class="delta muted">${esc(fmtIdr(s.totals.sum))}</div></div>
      <div class="card card-body kpi"><div class="label">Data Bulan Ini</div><div class="value">${fmtNum(s.thisMonth.count)}</div>${delta(s.thisMonth.count, s.lastMonth.count)}</div>
      <div class="card card-body kpi"><div class="label">${esc(sumField?.label || 'Nilai')} Bulan Ini</div><div class="value" title="${esc(fmtIdr(s.thisMonth.sum))}">${esc(fmtIdrShort(s.thisMonth.sum))}</div>${delta(s.thisMonth.sum, s.lastMonth.sum)}</div>
    </div>
    <div class="grid-2">
      <div class="card"><div class="card-header"><h2>Tren 12 Bulan Terakhir</h2><span class="muted">${esc(sumField?.label || '')}</span></div>
        <div class="card-body chart">${trendChart(s.trend)}</div></div>
      <div class="card"><div class="card-header"><h2>Perubahan Terbaru</h2><a href="#/data">Lihat semua</a></div>
        <div class="card-body">
          ${s.recent.length ? s.recent.map(r => `
            <div style="padding:8px 0;border-bottom:1px solid var(--border)">
              <div style="display:flex;justify-content:space-between;gap:8px"><b>${esc(r[listFields[1]?.key] ?? r.code)}</b><span class="code">${esc(r.code)}</span></div>
              <div class="muted" style="font-size:12.5px">${esc(fmtDateTime(r.updated_at))} · oleh ${esc(r.updated_by || '-')}</div>
            </div>`).join('') : '<div class="muted">Belum ada data</div>'}
        </div></div>
    </div>
    <div class="grid-3">${meta.dashboard.groupBy.map(k => groupCard(k, s.groups[k] || [], sumField)).join('')}</div>`;
}

// ---------- data list ----------

function viewData() {
  const { meta } = state;
  const d = state.data;
  $('#view').innerHTML = `
    <div class="page-header">
      <div><h1>${esc(meta.entityLabel)}</h1><p class="muted"><span><span class="live-dot ${state.live ? '' : 'off'}" id="live-dot"></span></span><span id="data-count">Memuat…</span></p></div>
      <div class="actions">
        <button class="btn" id="export-xlsx">⬇ Excel</button>
        <button class="btn" id="export-csv">⬇ CSV</button>
        ${can('editor') ? '<button class="btn" id="import-btn">⬆ Impor</button><button class="btn btn-primary" id="add-btn">+ Tambah Data</button>' : ''}
      </div>
    </div>
    <div class="card">
      <div class="toolbar" id="data-filters">
        <input class="search" type="search" id="search" placeholder="Cari kode, nama, PIC, catatan…" value="${esc(d.q)}">
        ${filterControls(d.filters, 'data')}
        <button class="btn btn-sm" id="data-reset">Reset</button>
      </div>
      <div class="table-wrap"><table><thead id="thead"></thead><tbody id="tbody"></tbody></table></div>
      <div class="pagination" id="pagination"></div>
    </div>`;

  const onFilter = () => { d.page = 1; loadRecords(); };
  $('#search').addEventListener('input', debounce(e => { d.q = e.target.value; onFilter(); }, 250));
  bindFilters($('#data-filters'), d.filters, onFilter);
  $('#data-reset').addEventListener('click', () => {
    Object.assign(d, { q: '', filters: {}, page: 1, sort: null, dir: null });
    viewData();
  });
  $('#export-xlsx').addEventListener('click', () => download('xlsx'));
  $('#export-csv').addEventListener('click', () => download('csv'));
  $('#add-btn')?.addEventListener('click', () => openRecordForm());
  $('#import-btn')?.addEventListener('click', openImport);
  loadRecords();
}

function dataQuery(extra = {}) {
  const d = state.data;
  return queryString({
    q: d.q, ...d.filters,
    sort: d.sort || state.meta.defaultSort.field,
    dir: d.dir || state.meta.defaultSort.dir,
    ...extra,
  });
}

function download(format) {
  window.location.href = `/api/export.${format}?${dataQuery()}`;
  toast(`Mengekspor ${format.toUpperCase()} sesuai filter aktif…`);
}

async function loadRecords() {
  const d = state.data;
  let result;
  try {
    result = await api(`/api/records?${dataQuery({ page: d.page, pageSize: d.pageSize })}`);
  } catch (err) {
    if (err.status !== 401) toast(err.message, 'error');
    return;
  }
  if (!$('#tbody')) return;
  const cols = state.meta.fields.filter(f => f.list);
  const sort = d.sort || state.meta.defaultSort.field;
  const dir = d.dir || state.meta.defaultSort.dir;
  const th = (key, label, cls = '') =>
    `<th class="sortable ${cls}" data-sort="${key}">${esc(label)}${sort === key ? `<span class="arrow">${dir === 'asc' ? '▲' : '▼'}</span>` : ''}</th>`;
  $('#thead').innerHTML = `<tr>${th('code', 'Kode')}${cols.map(f => th(f.key, f.label, f.type === 'number' ? 'num' : '')).join('')}</tr>`;
  $$('#thead th[data-sort]').forEach(el => el.addEventListener('click', () => {
    const key = el.dataset.sort;
    if (sort === key) d.dir = dir === 'asc' ? 'desc' : 'asc';
    else { d.sort = key; d.dir = 'asc'; }
    loadRecords();
  }));

  $('#data-count').textContent = `${fmtNum(result.total)} data ditemukan`;
  $('#tbody').innerHTML = result.rows.length
    ? result.rows.map(r => `<tr class="clickable ${state.flashCode === r.code ? 'flash' : ''}" data-id="${r.id}">
        <td class="code">${esc(r.code)}</td>
        ${cols.map(f => `<td class="${f.type === 'number' ? 'num' : ''}">${fmtValue(f, r[f.key])}</td>`).join('')}
      </tr>`).join('')
    : `<tr><td colspan="${cols.length + 1}" class="empty">Tidak ada data yang cocok dengan pencarian/filter.</td></tr>`;
  state.flashCode = null;
  $$('#tbody tr[data-id]').forEach(tr => tr.addEventListener('click', () => openRecordDetail(Number(tr.dataset.id))));

  const pages = Math.max(1, Math.ceil(result.total / result.pageSize));
  const from = result.total ? (result.page - 1) * result.pageSize + 1 : 0;
  const to = Math.min(result.page * result.pageSize, result.total);
  $('#pagination').innerHTML = `
    <span class="muted">Menampilkan ${fmtNum(from)}–${fmtNum(to)} dari ${fmtNum(result.total)}</span>
    <div class="pages">
      <select id="page-size" aria-label="Baris per halaman">${[10, 20, 50, 100].map(n => `<option ${n === d.pageSize ? 'selected' : ''}>${n}</option>`).join('')}</select>
      <button class="btn btn-sm" id="prev" ${result.page <= 1 ? 'disabled' : ''}>‹ Sebelumnya</button>
      <span>Hal. ${result.page} / ${pages}</span>
      <button class="btn btn-sm" id="next" ${result.page >= pages ? 'disabled' : ''}>Berikutnya ›</button>
    </div>`;
  $('#prev').addEventListener('click', () => { d.page--; loadRecords(); });
  $('#next').addEventListener('click', () => { d.page++; loadRecords(); });
  $('#page-size').addEventListener('change', e => { d.pageSize = Number(e.target.value); d.page = 1; loadRecords(); });
}

// ---------- record detail & form ----------

async function openRecordDetail(id) {
  let r;
  try {
    r = await api(`/api/records/${id}`);
  } catch (err) {
    return toast(err.message, 'error');
  }
  const rows = state.meta.fields.map(f => `<dt>${esc(f.label)}</dt><dd>${fmtValue(f, r[f.key])}</dd>`).join('');
  openModal({
    title: `Detail ${r.code}`,
    body: `<dl class="detail-grid">${rows}
      <dt>Dibuat</dt><dd>${esc(fmtDateTime(r.created_at))} · ${esc(r.created_by || '-')}</dd>
      <dt>Terakhir diubah</dt><dd>${esc(fmtDateTime(r.updated_at))} · ${esc(r.updated_by || '-')}</dd></dl>`,
    footer: `${can('admin') ? '<button class="btn btn-danger" id="del-btn" style="margin-right:auto">Hapus</button>' : ''}
             <button class="btn" data-close>Tutup</button>
             ${can('editor') ? '<button class="btn btn-primary" id="edit-btn">Ubah</button>' : ''}`,
    onMount: m => {
      $('#edit-btn', m)?.addEventListener('click', () => openRecordForm(r));
      $('#del-btn', m)?.addEventListener('click', async () => {
        const ok = await confirmDialog({
          title: 'Hapus data?',
          message: `Data <b>${esc(r.code)}</b> akan dihapus permanen. Tindakan ini tercatat di log aktivitas.`,
          confirmText: 'Hapus', danger: true,
        });
        if (!ok) return;
        try {
          await api(`/api/records/${r.id}`, { method: 'DELETE' });
          toast(`${r.code} dihapus`, 'success');
          loadRecords();
        } catch (err) {
          toast(err.message, 'error');
        }
      });
    },
  });
}

function inputFor(f, value) {
  const v = value ?? '';
  const common = `id="f-${f.key}" name="${f.key}" ${f.required ? 'required' : ''}`;
  if (f.type === 'select') {
    return `<select ${common}><option value="">— Pilih —</option>
      ${f.options.map(o => `<option ${o === v ? 'selected' : ''}>${esc(o)}</option>`).join('')}</select>`;
  }
  if (f.type === 'textarea') return `<textarea ${common}>${esc(v)}</textarea>`;
  if (f.type === 'number') return `<input type="number" step="any" ${f.min !== undefined ? `min="${f.min}"` : ''} ${common} value="${esc(v)}">`;
  if (f.type === 'date') return `<input type="date" ${common} value="${esc(v)}">`;
  return `<input type="text" ${common} value="${esc(v)}">`;
}

function openRecordForm(record) {
  const isEdit = !!record;
  const defaults = {};
  if (!isEdit) for (const f of state.meta.fields) if (f.type === 'date') defaults[f.key] = new Date().toISOString().slice(0, 10);
  const src = record || defaults;
  const fields = state.meta.fields.map(f => `
    <div class="field ${f.type === 'textarea' ? 'full' : ''}" data-field="${f.key}">
      <label for="f-${f.key}">${esc(f.label)} ${f.required ? '<span class="req">*</span>' : ''}</label>
      ${inputFor(f, src[f.key])}
    </div>`).join('');
  openModal({
    title: isEdit ? `Ubah ${record.code}` : `Tambah ${state.meta.entityLabel}`,
    body: `<form id="record-form" class="form-grid" novalidate>${fields}</form>`,
    footer: `<button class="btn" data-close>Batal</button><button class="btn btn-primary" id="save-btn">Simpan</button>`,
    onMount: m => {
      const form = $('#record-form', m);
      const save = async () => {
        const body = {};
        for (const f of state.meta.fields) body[f.key] = form.elements[f.key].value;
        $('#save-btn', m).disabled = true;
        try {
          const saved = await api(isEdit ? `/api/records/${record.id}` : '/api/records', {
            method: isEdit ? 'PUT' : 'POST', body,
          });
          closeModal();
          state.flashCode = saved.code;
          toast(isEdit ? `${saved.code} diperbarui` : `${saved.code} ditambahkan`, 'success');
          if (currentRoute() === 'data') loadRecords();
        } catch (err) {
          $('#save-btn', m).disabled = false;
          showFieldErrors(form, err.details);
          toast(err.message, 'error');
        }
      };
      $('#save-btn', m).addEventListener('click', save);
      form.addEventListener('submit', e => { e.preventDefault(); save(); });
    },
  });
}

// ---------- import ----------

function openImport() {
  openModal({
    title: 'Impor Data dari Excel/CSV',
    body: `
      <ol style="margin:0 0 14px;padding-left:18px;line-height:1.7">
        <li>Unduh <a href="/api/import/template.csv">template impor (CSV)</a>.</li>
        <li>Isi di Excel, lalu simpan sebagai <b>CSV</b> (File → Save As → CSV).</li>
        <li>Unggah file di bawah. Semua baris divalidasi dulu; jika ada error, tidak ada data yang disimpan.</li>
      </ol>
      <input type="file" id="import-file" accept=".csv,text/csv">
      <div id="import-result" style="margin-top:14px"></div>`,
    footer: `<button class="btn" data-close>Tutup</button><button class="btn btn-primary" id="import-go">Impor</button>`,
    onMount: m => {
      $('#import-go', m).addEventListener('click', async () => {
        const file = $('#import-file', m).files[0];
        const out = $('#import-result', m);
        if (!file) return toast('Pilih file CSV terlebih dahulu.', 'error');
        $('#import-go', m).disabled = true;
        try {
          const res = await api('/api/import', { method: 'POST', body: await file.text(), raw: true });
          if (res.errorCount) {
            out.innerHTML = `<div class="alert"><b>${res.errorCount} baris bermasalah</b> — tidak ada data yang disimpan.
              <ul style="margin:6px 0 0;padding-left:18px">${res.errors.map(e => `<li>Baris ${e.row}: ${esc(e.messages.join(' '))}</li>`).join('')}</ul></div>`;
          } else {
            closeModal();
            toast(`${res.imported} data berhasil diimpor`, 'success');
            loadRecords();
          }
        } catch (err) {
          out.innerHTML = `<div class="alert">${esc(err.message)}</div>`;
        } finally {
          const btn = $('#import-go', m);
          if (btn) btn.disabled = false;
        }
      });
    },
  });
}

// ---------- users ----------

async function viewUsers() {
  $('#view').innerHTML = `
    <div class="page-header">
      <div><h1>Pengguna & Hak Akses</h1><p class="muted">Atur siapa yang bisa melihat, mengubah, atau mengelola data.</p></div>
      <button class="btn btn-primary" id="add-user">+ Tambah Pengguna</button>
    </div>
    <div class="grid-3">${['admin', 'editor', 'viewer'].map(r => `
      <div class="card card-body"><span class="badge ${ROLE_COLOR[r]}">${ROLE_LABEL[r]}</span><p class="muted" style="margin:8px 0 0">${ROLE_DESC[r]}</p></div>`).join('')}
    </div>
    <div class="card"><div class="table-wrap"><table>
      <thead><tr><th>Nama</th><th>Username</th><th>Peran</th><th>Status</th><th>Dibuat</th><th></th></tr></thead>
      <tbody id="users-body"><tr><td colspan="6" class="empty">Memuat…</td></tr></tbody>
    </table></div></div>`;
  $('#add-user').addEventListener('click', () => openUserForm());
  let users;
  try {
    users = await api('/api/users');
  } catch (err) {
    return toast(err.message, 'error');
  }
  $('#users-body').innerHTML = users.map(u => `
    <tr>
      <td><b>${esc(u.name)}</b>${u.id === state.user.id ? ' <span class="muted">(Anda)</span>' : ''}</td>
      <td class="code">${esc(u.username)}</td>
      <td><span class="badge ${ROLE_COLOR[u.role]}">${ROLE_LABEL[u.role]}</span></td>
      <td>${u.active ? '<span class="badge green">Aktif</span>' : '<span class="badge red">Nonaktif</span>'}</td>
      <td>${esc(fmtDate(u.created_at))}</td>
      <td style="text-align:right">
        <button class="btn btn-sm" data-edit="${u.id}">Ubah</button>
        ${u.id !== state.user.id ? `<button class="btn btn-sm btn-danger" data-del="${u.id}">Hapus</button>` : ''}
      </td>
    </tr>`).join('');
  $$('[data-edit]').forEach(b => b.addEventListener('click', () => openUserForm(users.find(u => u.id === Number(b.dataset.edit)))));
  $$('[data-del]').forEach(b => b.addEventListener('click', async () => {
    const u = users.find(x => x.id === Number(b.dataset.del));
    const ok = await confirmDialog({ title: 'Hapus pengguna?', message: `Akun <b>${esc(u.username)}</b> akan dihapus.`, confirmText: 'Hapus', danger: true });
    if (!ok) return;
    try {
      await api(`/api/users/${u.id}`, { method: 'DELETE' });
      toast('Pengguna dihapus', 'success');
      viewUsers();
    } catch (err) {
      toast(err.message, 'error');
    }
  }));
}

function openUserForm(user) {
  const isEdit = !!user;
  openModal({
    title: isEdit ? `Ubah Pengguna: ${user.username}` : 'Tambah Pengguna',
    size: 'sm',
    body: `<form id="user-form" novalidate>
      ${isEdit ? '' : `<div class="field" data-field="username"><label for="u-username">Username <span class="req">*</span></label><input id="u-username" name="username" autocomplete="off"></div>`}
      <div class="field" data-field="name"><label for="u-name">Nama <span class="req">*</span></label><input id="u-name" name="name" value="${esc(user?.name || '')}"></div>
      <div class="field" data-field="role"><label for="u-role">Peran</label>
        <select id="u-role" name="role">${['admin', 'editor', 'viewer'].map(r => `<option value="${r}" ${(user?.role || 'viewer') === r ? 'selected' : ''}>${ROLE_LABEL[r]}</option>`).join('')}</select>
        <div class="muted" id="role-desc" style="font-size:12px;margin-top:4px"></div></div>
      <div class="field" data-field="password"><label for="u-password">${isEdit ? 'Reset password (opsional)' : 'Password <span class="req">*</span>'}</label>
        <input id="u-password" name="password" type="password" autocomplete="new-password" placeholder="Minimal 8 karakter"></div>
      ${isEdit ? `<div class="field"><label><input type="checkbox" name="active" ${user.active ? 'checked' : ''} style="width:auto;margin-right:6px">Akun aktif</label></div>` : ''}
    </form>`,
    footer: `<button class="btn" data-close>Batal</button><button class="btn btn-primary" id="user-save">Simpan</button>`,
    onMount: m => {
      const form = $('#user-form', m);
      const desc = () => { $('#role-desc', m).textContent = ROLE_DESC[form.role.value]; };
      form.role.addEventListener('change', desc);
      desc();
      $('#user-save', m).addEventListener('click', async () => {
        const body = { name: form.name.value, role: form.role.value };
        if (form.password.value) body.password = form.password.value;
        if (isEdit) body.active = form.active.checked;
        else body.username = form.username.value;
        try {
          await api(isEdit ? `/api/users/${user.id}` : '/api/users', { method: isEdit ? 'PUT' : 'POST', body });
          closeModal();
          toast('Pengguna disimpan', 'success');
          viewUsers();
        } catch (err) {
          showFieldErrors(form, err.details);
          toast(err.message, 'error');
        }
      });
    },
  });
}

// ---------- activity log ----------

function describeAudit(a) {
  const d = a.detail;
  if (!d) return '';
  if (a.action === 'update') {
    return Object.entries(d).map(([k, [from, to]]) => `<b>${esc(k)}</b>: ${esc(from || '—')} → ${esc(to || '—')}`).join('<br>');
  }
  if (a.action === 'import') return `${d.count} baris`;
  if (a.action === 'export') return `${d.format.toUpperCase()}, ${d.rows} baris`;
  if (a.action === 'user_create') return `Peran: ${ROLE_LABEL[d.role]}`;
  if (a.action === 'user_update') {
    const parts = [];
    if (d.role) parts.push(`Peran: ${ROLE_LABEL[d.role[0]]} → ${ROLE_LABEL[d.role[1]]}`);
    if (d.active === false) parts.push('Dinonaktifkan');
    if (d.password_reset) parts.push('Password direset');
    return esc(parts.join(', '));
  }
  if (a.action === 'delete') {
    const f = state.meta.fields.filter(x => x.list).slice(0, 3);
    return f.map(x => `${esc(x.label)}: ${esc(d[x.key] ?? '')}`).join(' · ');
  }
  return '';
}

async function viewActivity() {
  $('#view').innerHTML = `
    <div class="page-header"><div><h1>Log Aktivitas</h1><p class="muted">Siapa mengubah apa dan kapan — 200 aktivitas terakhir.</p></div></div>
    <div class="card"><div class="table-wrap"><table>
      <thead><tr><th>Waktu</th><th>Pengguna</th><th>Aktivitas</th><th>Target</th><th>Detail</th></tr></thead>
      <tbody id="audit-body"><tr><td colspan="5" class="empty">Memuat…</td></tr></tbody>
    </table></div></div>`;
  let rows;
  try {
    rows = await api('/api/audit?limit=200');
  } catch (err) {
    return toast(err.message, 'error');
  }
  $('#audit-body').innerHTML = rows.length ? rows.map(a => `
    <tr>
      <td>${esc(fmtDateTime(a.ts))}</td>
      <td>${esc(a.username || '-')}</td>
      <td>${esc(ACTION_LABEL[a.action] || a.action)}</td>
      <td class="code">${esc(a.target || '')}</td>
      <td class="audit-change">${describeAudit(a)}</td>
    </tr>`).join('') : '<tr><td colspan="5" class="empty">Belum ada aktivitas.</td></tr>';
}

// ---------- account ----------

function viewAccount() {
  const u = state.user;
  $('#view').innerHTML = `
    <div class="page-header"><div><h1>Akun Saya</h1></div></div>
    <div class="grid-2" style="grid-template-columns:1fr 1fr">
      <div class="card card-body">
        <h2 style="margin-bottom:14px">Profil</h2>
        <dl class="detail-grid">
          <dt>Nama</dt><dd>${esc(u.name)}</dd>
          <dt>Username</dt><dd>${esc(u.username)}</dd>
          <dt>Peran</dt><dd><span class="badge ${ROLE_COLOR[u.role]}">${ROLE_LABEL[u.role]}</span></dd>
          <dt>Hak akses</dt><dd class="muted">${ROLE_DESC[u.role]}</dd>
        </dl>
      </div>
      <form class="card card-body" id="pw-form" novalidate>
        <h2 style="margin-bottom:14px">Ganti Password</h2>
        <div class="field"><label for="pw-cur">Password lama</label><input id="pw-cur" type="password" autocomplete="current-password"></div>
        <div class="field"><label for="pw-new">Password baru</label><input id="pw-new" type="password" autocomplete="new-password" placeholder="Minimal 8 karakter"></div>
        <div class="field"><label for="pw-new2">Ulangi password baru</label><input id="pw-new2" type="password" autocomplete="new-password"></div>
        <button class="btn btn-primary" type="submit">Simpan Password</button>
      </form>
    </div>`;
  $('#pw-form').addEventListener('submit', async e => {
    e.preventDefault();
    const next = $('#pw-new').value;
    if (next !== $('#pw-new2').value) return toast('Konfirmasi password tidak sama.', 'error');
    try {
      await api('/api/me/password', { method: 'POST', body: { current: $('#pw-cur').value, next } });
      toast('Password berhasil diganti', 'success');
      e.target.reset();
    } catch (err) {
      toast(err.message, 'error');
    }
  });
}

// ---------- boot ----------

(async function boot() {
  try {
    state.meta = await api('/api/meta');
  } catch {
    $('#app').innerHTML = '<div class="empty">Tidak dapat terhubung ke server.</div>';
    return;
  }
  document.title = `${state.meta.appName} — ${state.meta.appTagline}`;
  state.user = state.meta.user;
  if (state.user) renderShell();
  else renderLogin();
})();
