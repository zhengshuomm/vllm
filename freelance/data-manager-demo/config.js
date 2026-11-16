'use strict';

// Semua kolom data diatur di sini. Ubah daftar `fields` agar sesuai dengan data
// klien (misalnya inventaris, pelanggan, aset) tanpa mengubah kode lain.
//
// type:     'text' | 'textarea' | 'number' | 'date' | 'select'
// required: wajib diisi
// search:   ikut dicari oleh kotak pencarian
// filter:   true = filter dropdown (select), 'range' = filter rentang tanggal
// list:     false = tidak ditampilkan di tabel (tetap ada di detail & ekspor)
// currency: format Rupiah
// colors:   warna label untuk nilai select (blue|amber|green|red|gray)

module.exports = {
  appName: 'DataKu',
  appTagline: 'Sistem Pencatatan & Manajemen Data',
  entityLabel: 'Data Order',
  codePrefix: 'ORD',
  port: Number(process.env.PORT) || 3000,
  defaultSort: { field: 'tanggal', dir: 'desc' },
  showDemoHint: process.env.SHOW_DEMO_HINT !== '0',

  fields: [
    { key: 'tanggal', label: 'Tanggal', type: 'date', required: true, filter: 'range' },
    { key: 'pelanggan', label: 'Nama Pelanggan', type: 'text', required: true, search: true },
    {
      key: 'kategori', label: 'Kategori', type: 'select', required: true, filter: true,
      options: ['Elektronik', 'Fashion', 'Makanan & Minuman', 'Kesehatan', 'Rumah Tangga'],
    },
    {
      key: 'wilayah', label: 'Wilayah', type: 'select', required: true, filter: true,
      options: ['Jakarta', 'Bandung', 'Surabaya', 'Medan', 'Makassar', 'Denpasar'],
    },
    { key: 'jumlah', label: 'Jumlah', type: 'number', required: true, min: 0 },
    { key: 'nilai', label: 'Nilai (Rp)', type: 'number', required: true, min: 0, currency: true },
    {
      key: 'status', label: 'Status', type: 'select', required: true, filter: true,
      options: ['Baru', 'Diproses', 'Selesai', 'Dibatalkan'],
      colors: { Baru: 'blue', Diproses: 'amber', Selesai: 'green', Dibatalkan: 'red' },
    },
    { key: 'pic', label: 'PIC', type: 'text', search: true },
    { key: 'catatan', label: 'Catatan', type: 'textarea', search: true, list: false },
  ],

  dashboard: {
    dateField: 'tanggal',
    sumField: 'nilai',
    groupBy: ['status', 'kategori', 'wilayah'],
  },

  // Akun awal, hanya dibuat saat database masih kosong. Ganti password setelah login.
  demoUsers: [
    { username: 'admin', name: 'Admin Utama', role: 'admin', password: 'admin123' },
    { username: 'editor', name: 'Staf Editor', role: 'editor', password: 'editor123' },
    { username: 'viewer', name: 'Manajer (Lihat Saja)', role: 'viewer', password: 'viewer123' },
  ],

  // Data contoh untuk demo. Set SEED_DEMO=0 agar database mulai kosong.
  demoRecordCount: 150,
  demoSamples: {
    pelanggan: [
      'PT Maju Jaya', 'CV Sinar Abadi', 'Toko Berkah Sentosa', 'UD Sumber Rejeki',
      'PT Nusantara Makmur', 'Toko Cahaya Baru', 'CV Mitra Sejahtera', 'PT Indo Prima',
      'Warung Bu Tini', 'Apotek Sehat Selalu', 'PT Karya Mandiri', 'Toko Elektronik Jaya',
    ],
    jumlah: rand => 1 + Math.floor(rand() * 40),
    nilai: (rand, row) => {
      const base = {
        Elektronik: 850000, Fashion: 175000, 'Makanan & Minuman': 45000,
        Kesehatan: 120000, 'Rumah Tangga': 95000,
      }[row.kategori] || 100000;
      return Math.round((row.jumlah * base * (0.8 + rand() * 0.4)) / 1000) * 1000;
    },
    status: rand => {
      const r = rand();
      if (r < 0.15) return 'Baru';
      if (r < 0.35) return 'Diproses';
      return r < 0.92 ? 'Selesai' : 'Dibatalkan';
    },
    pic: ['Andi', 'Siti', 'Budi', 'Dewi', 'Rina', 'Agus'],
    catatan: rand => (rand() < 0.3 ? 'Pengiriman via ekspedisi, konfirmasi H-1' : ''),
  },
};
