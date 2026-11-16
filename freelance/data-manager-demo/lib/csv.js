'use strict';

// Prevents spreadsheet formula injection when exported text is opened in Excel.
const FORMULA_PREFIX = /^[=+\-@\t\r]/;

function csvCell(value) {
  if (value === null || value === undefined) return '';
  let s = String(value);
  if (typeof value === 'string' && FORMULA_PREFIX.test(s)) s = `'${s}`;
  return /[",;\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function toCsv(headers, rows) {
  const lines = [headers, ...rows].map(r => r.map(csvCell).join(','));
  return '\uFEFF' + lines.join('\r\n') + '\r\n';
}

// Excel in Indonesian locale saves CSV with ';', so detect the delimiter.
function detectDelimiter(text) {
  const end = text.search(/\r?\n/);
  const firstLine = end === -1 ? text : text.slice(0, end);
  let best = ',';
  let bestCount = 0;
  for (const d of [',', ';', '\t']) {
    const count = firstLine.split(d).length - 1;
    if (count > bestCount) {
      best = d;
      bestCount = count;
    }
  }
  return best;
}

function parseCsv(input) {
  const text = input.replace(/^\uFEFF/, '');
  const delim = detectDelimiter(text);
  const rows = [];
  let row = [];
  let field = '';
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        field += c;
      }
    } else if (c === '"' && field === '') {
      inQuotes = true;
    } else if (c === delim) {
      row.push(field);
      field = '';
    } else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(field);
      rows.push(row);
      row = [];
      field = '';
    } else {
      field += c;
    }
  }
  if (field !== '' || row.length) {
    row.push(field);
    rows.push(row);
  }
  return rows;
}

module.exports = { toCsv, parseCsv };
