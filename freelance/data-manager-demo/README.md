# DataKu — Sistem Pencatatan & Manajemen Data

Aplikasi web ringan untuk memasukkan, memperbarui, mencari, dan menganalisis data tim secara terpusat — menggantikan pencatatan manual di Excel/kertas.

## Fitur

| Kebutuhan | Fitur |
| :--- | :--- |
| Sistem terpusat, real-time | Satu database untuk seluruh tim. Perubahan dari pengguna lain langsung muncul tanpa refresh (notifikasi + baris disorot). |
| Input & update data | Formulir dengan validasi per kolom (wajib isi, angka, tanggal, pilihan). Kode data otomatis (`ORD-00001`). |
| Pencarian & filter | Pencarian teks (kode, nama, PIC, catatan), filter dropdown (Kategori, Wilayah, Status), rentang tanggal, urutkan kolom, paginasi. |
| Dasbor ringkas | Total data & nilai, perbandingan bulan ini vs bulan lalu, grafik tren 12 bulan, ringkasan per Status/Kategori/Wilayah, perubahan terbaru. Dasbor ikut filter. |
| Hak akses bertingkat | **Admin**: akses penuh + kelola pengguna + log aktivitas. **Editor**: tambah, ubah, impor. **Viewer**: lihat, cari, ekspor. Dicek di server, bukan hanya disembunyikan di tampilan. |
| Ekspor Excel/CSV | Ekspor `.xlsx` (header, format angka & tanggal, filter otomatis) dan `.csv`, sesuai filter yang sedang aktif. |
| Impor dari Excel | Unggah CSV dari Excel (mendukung pemisah `,` dan `;`). Semua baris divalidasi dulu; jika ada yang salah, tidak ada data tersimpan dan baris bermasalah ditampilkan. |
| Keamanan & jejak audit | Password di-hash (scrypt), sesi HttpOnly, batas percobaan login, perlindungan formula-injection di ekspor, log aktivitas lengkap (siapa mengubah apa, nilai lama → baru). |

## Menjalankan

Butuh **Node.js 22.5+**. Tidak ada dependensi yang perlu di-install (database SQLite sudah bawaan Node).

```bash
npm start          # http://localhost:3000
npm test           # 10 tes otomatis
```

Akun demo (dibuat otomatis saat pertama kali dijalankan):

| Peran | Username | Password |
| :--- | :--- | :--- |
| Admin | `admin` | `admin123` |
| Editor | `editor` | `editor123` |
| Viewer | `viewer` | `viewer123` |

Variabel lingkungan:

| Variabel | Default | Keterangan |
| :--- | :--- | :--- |
| `PORT` | `3000` | Port server |
| `DB_FILE` | `data/app.db` | Lokasi file database |
| `SEED_DEMO` | `1` | `0` = mulai tanpa data contoh |
| `SHOW_DEMO_HINT` | `1` | `0` = sembunyikan daftar akun demo di halaman login (wajib untuk produksi) |

## Menyesuaikan dengan data Anda

Semua kolom diatur di [`config.js`](config.js) → `fields`. Contoh menambah kolom "No. Invoice":

```js
{ key: 'invoice', label: 'No. Invoice', type: 'text', search: true },
```

Tipe yang tersedia: `text`, `textarea`, `number`, `date`, `select`. Formulir, tabel, filter, impor, ekspor, dan dasbor otomatis mengikuti konfigurasi ini.

## Panduan singkat pengguna

1. **Login** dengan akun dari admin.
2. **Dasbor** — lihat ringkasan. Pilih filter di kanan atas untuk melihat angka per wilayah/kategori/periode.
3. **Data Order** — cari dengan kotak pencarian, saring dengan dropdown, klik judul kolom untuk mengurutkan. Klik baris untuk melihat detail.
4. **Tambah/Ubah** (Editor/Admin) — tombol **+ Tambah Data**, atau buka detail lalu **Ubah**. Kolom bertanda * wajib diisi.
5. **Impor** (Editor/Admin) — unduh template, isi di Excel, simpan sebagai CSV, unggah.
6. **Ekspor** — tombol **Excel** atau **CSV** mengunduh data sesuai filter aktif.
7. **Pengguna** (Admin) — tambah akun, atur peran, nonaktifkan, atau reset password.
8. **Log Aktivitas** (Admin) — riwayat semua perubahan.
9. **Akun Saya** — ganti password.

## Deploy ke server

Cocok untuk VPS kecil (1 vCPU / 1 GB) atau layanan seperti Railway/Render dengan disk persisten:

```bash
SHOW_DEMO_HINT=0 SEED_DEMO=0 PORT=3000 npm start
```

Pasang di belakang Nginx/Caddy dengan HTTPS, jalankan dengan `pm2` atau `systemd`, dan backup file `data/app.db` secara berkala (cukup salin file).

## Struktur

```
server.js        API (auth, CRUD, filter, statistik, impor/ekspor, pengguna, audit)
db.js            Skema SQLite, seed, hash password
config.js        Definisi kolom, opsi, dasbor, akun awal
lib/csv.js       Baca/tulis CSV
lib/xlsx.js      Penulis .xlsx tanpa dependensi
public/          Antarmuka (HTML/CSS/JS murni)
test/            Tes otomatis (node:test)
```
