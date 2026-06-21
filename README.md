# Sovereign Dialect-Bridge

Sistem NLP untuk meringkas pengaduan warga yang disampaikan dalam bahasa/dialek daerah Indonesia menjadi ringkasan Bahasa Indonesia baku.

**BINUS School of Computer Science — NLP Final Project, Semester 4**
Kelompok 2:
- Fadhlan Nur Rachman (2802491690)
- Dian Rakhmawati Lestari (2802539085)
- Bintang Nur Fadhlillah (2802536083)

---

## Latar Belakang

Indonesia memiliki lebih dari 700 bahasa daerah. Laporan pengaduan warga sering kali disampaikan dalam dialek lokal (Jawa, Sunda, Minangkabau, dll.), sehingga menciptakan hambatan komunikasi bagi sistem birokrasi formal yang beroperasi secara eksklusif menggunakan Bahasa Indonesia baku. Di sisi lain, memproses data pengaduan masyarakat yang bersifat sensitif melalui layanan cloud pihak ketiga rentan terhadap risiko pelanggaran privasi, ditambah dengan tantangan infrastruktur internet di daerah pelosok.

Untuk mengatasi hal tersebut, proyek ini membangun Sovereign Dialect-Bridge, sebuah pipeline Two-Stage berbasis offline Small Language Models (SLMs). Sistem ini bekerja secara lokal untuk menerjemahkan teks pengaduan dialek dan merangkumnya menjadi laporan Bahasa Indonesia baku secara otomatis, memastikan aksesibilitas birokrasi sekaligus menjaga kedaulatan dan privasi data warga.

---

## Link Deploy Apps
NLP : https://oinovenv-sovereign-dialect-bridge-api.hf.space/dashboard/

jika ingin melihat dari sisi mata kuliah software engineering
Software Engineer : https://sovereign-dialect-bridge.vercel.app/

---

## Pipeline

```
INPUT teks (dialek daerah Indonesia)
    |
    v  detect_dialect
       Model    : TF-IDF char n-gram + Logistic Regression
       Output   : kode dialek (jv / su / min / ace / ...) + confidence
       Fallback : langdetect -> "xx" (tidak terdeteksi)
    |
    v  translate_to_indonesian
       Model    : NLLB-200-distilled-600M (facebook/nllb-200-distilled-600M)
       Bahasa   : 8 dialek via NLLB; mad/nij/bbc via deep_translator
       Fallback : teks asli dikembalikan tanpa translasi
    |
    v  summarize  [pilih salah satu metode]
       Abstractive :
         - mT5-base   (OinoVenv/sovereign-mt5-nusasum)
         - IndoT5     (OinoVenv/sovereign-indot5-nusasum)
       Extractive  :
         - TextRank   (TF-IDF + PageRank, tanpa model tambahan)
         - NER-based  (cahya/bert-base-indonesian-NER)
       Fallback : 2 kalimat pertama
    |
    v
OUTPUT ringkasan Bahasa Indonesia baku
```

Pipeline tidak pernah crash — setiap stage punya fallback chain.

---

## Model yang Digunakan

| Stage | Model / Metode | Sumber | Kelebihan | Kekurangan |
|-------|---------------|--------|-----------|------------|
| Detect Dialect | TF-IDF char n-gram + Logistic Regression | Lokal (`notebook/dialect_detector/`) | Sangat cepat (<5 ms), ukuran kecil (5 MB), `predict_proba` untuk fallback | Kurang akurat pada teks campuran (code-switching) |
| Translate | facebook/nllb-200-distilled-600M | HuggingFace Hub (auto-download) | Mendukung 200 bahasa termasuk dialek Indonesia, kualitas terjemahan baik | Berat (~1.2 GB), lambat di CPU |
| Summarize — mT5-base | OinoVenv/sovereign-mt5-nusasum | HuggingFace Hub | Multilingual, fine-tuned IndoSum, ringkasan fluent | Paling berat (~2.2 GB), lambat di CPU (~10-30 detik) |
| Summarize — IndoT5 | OinoVenv/sovereign-indot5-nusasum | HuggingFace Hub | Didesain khusus untuk Bahasa Indonesia, lebih ringan dari mT5 | Hanya untuk BI (tidak multilingual) |
| Summarize — TextRank | TF-IDF + PageRank (networkx) | Tidak ada (algoritma) | Sangat cepat, tidak butuh model, tidak butuh GPU | Hanya ambil kalimat asli, tidak bisa parafrase; kurang informatif untuk teks panjang |
| Summarize — NER | cahya/bert-base-indonesian-NER | HuggingFace Hub (auto-download) | Seleksi kalimat berdasarkan entitas nyata (orang, lokasi, org) | Butuh download BERT (~440 MB), lebih lambat dari TextRank |

Model abstractive di-fine-tune pada dataset **IndoSum** (10K artikel berita Indonesia).
Dialect detector di-train pada **NusaX-MT** (12 bahasa daerah, ~6.000 sampel).

---

## Struktur Folder

```
Project_Summarize_NLP/
├── notebook/              Jupyter notebooks (training + inference)
│   ├── dialect_detector/  Training dialect classifier + model .joblib
│   ├── training_sum.ipynb Training abstractive summarizer
│   └── inference.ipynb    Demo pipeline + evaluasi ROUGE
├── dataset/
│   ├── indosum/           Fold 1 IndoSum JSONL (~14K artikel)
│   ├── indosum_parque/    IndoSum processed (parquet)
│   └── nusax/             NusaX-MT parallel corpus (12 bahasa)
├── model/                 Model weights lokal (opsional, ~5 GB)
├── application_demo/      Django backend (NLP + Software Engineering)
└── CLAUDE.md              Panduan untuk AI assistant
```

---

## Cara Menjalankan

### Akses langsung (sudah di-deploy)

Dashboard dan API sudah tersedia di:

```
https://oinovenv-sovereign-dialect-bridge-api.hf.space/dashboard/
```

Halaman dashboard menampilkan status model, arsitektur pipeline, dan NLP testing interaktif.

### Jalankan NLP Pipeline (Notebook)

Cara paling mudah untuk mencoba pipeline NLP — tidak perlu setup database atau server.

```bash
pip install transformers torch joblib scikit-learn networkx rouge-score pandas pyarrow
jupyter notebook notebook/inference.ipynb
```

Ubah variabel `INPUT_TEXT` dan `SUMMARIZE_WITH` di cell pertama, lalu jalankan semua sel.

### Jalankan application_demo (Backend)

> Untuk keperluan NLP saja, bagian ini **tidak wajib**. Lihat [catatan di bawah](#catatan-application_demo-dan-mata-kuliah-software-engineering).

**1. Masuk ke folder backend**

```bash
cd application_demo
```

**2. Buat virtual environment**

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate.bat

# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Konfigurasi `.env`**

```bash
# macOS / Linux
cp .env.example .env

# Windows
copy .env.example .env
```

Untuk testing lokal (NLP pipeline saja, tanpa fitur akun/database production), isi `.env` dengan nilai berikut — tidak perlu Supabase:

```
SECRET_KEY=dev-local-key-not-for-production
DATABASE_URL=sqlite:///db.sqlite3
NLP_ENABLED=false
```

Untuk deployment atau fitur lengkap (akun pengguna, dashboard), lihat `application_demo/README.md`.

**5. Migrate & jalankan server**

```bash
python manage.py migrate
python manage.py runserver
```

Server berjalan di `http://localhost:8000`. Dengan `NLP_ENABLED=false`, server boot <1 detik tanpa load model neural.

---

## Dataset

| Dataset | Keterangan | Lokasi |
|---------|-----------|--------|
| IndoSum | ~14K artikel berita Indonesia + ringkasan, fold 1 | `dataset/indosum/` |
| NusaX-MT | 526 kalimat × 12 bahasa paralel | `dataset/nusax/` |

---

## Catatan: application_demo dan Mata Kuliah Software Engineering

`application_demo/` adalah Django backend yang dikerjakan **bersamaan** dengan proyek mata kuliah Software Engineering (sistem pengaduan publik) dan NLP (pipeline summarisasi). Karena dua tujuan ini digabung, setup-nya terlihat kompleks — ada database, autentikasi, dan dashboard admin yang tidak relevan untuk keperluan NLP saja.

**Untuk keperluan NLP, tidak perlu menjalankan application_demo.** Notebook di `notebook/` sudah cukup dan berdiri sendiri:

```
notebook/inference.ipynb   → demo pipeline end-to-end
```

Jika ingin tetap menjalankan application_demo (misalnya untuk uji API atau dashboard), lihat `application_demo/README.md`.
