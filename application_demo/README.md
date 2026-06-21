# Application Demo — Sovereign Dialect-Bridge

Django REST API yang berfungsi sebagai backend untuk **dua mata kuliah sekaligus**:
- **NLP**: pipeline summarisasi pengaduan berbahasa daerah Indonesia
- **Software Engineering**: sistem manajemen pengaduan publik multidialek

Aplikasi sudah di-deploy dan dapat diakses di:
```
https://oinovenv-sovereign-dialect-bridge-api.hf.space/dashboard/
```
atau jika ingin melihat dari sisi frontend pada matkul Software Engineer:
```
https://sovereign-dialect-bridge.vercel.app/
```

Halaman `/dashboard/` menampilkan status model real-time, arsitektur pipeline, dan tool NLP testing interaktif.

---

## Pipeline NLP

```
INPUT pengaduan (teks dialek daerah)
    |
    v  detect_dialect
       Model    : TF-IDF + Logistic Regression (joblib, 5 MB)
       Fallback : langdetect -> "xx"
    |
    v  translate_to_indonesian
       Model    : NLLB-200-distilled-600M (facebook)
       Fallback : deep_translator -> teks asli
    |
    v  summarize
       Primary  : mT5-base (OinoVenv/sovereign-mt5-nusasum)
                  IndoT5   (OinoVenv/sovereign-indot5-nusasum)
       Fallback : NER extractive (cahya/bert-base-indonesian-NER)
                  TextRank extractive
                  2 kalimat pertama
    |
    v  extract_entities    <- cahya/bert-base-indonesian-NER -> regex
    v  classify_category   <- keyword matching (8 kategori)
    v  score_urgency       <- weighted keyword scoring
    v
OUTPUT: {dialect, translated, summary, entities, category, urgency, keywords}
```

Setiap stage punya fallback chain — pipeline tidak pernah crash.

---

## Setup Lokal

### Prasyarat

| Tool | Versi |
|------|-------|
| Python | 3.11+ |
| RAM (NLP_ENABLED=false) | 1 GB |
| RAM (NLP_ENABLED=true) | 8 GB minimum |
| Disk (model cache) | ~5 GB |

Database di-host di Supabase — tidak perlu install Postgres lokal.

### Instalasi

```bash
cd application_demo
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Konfigurasi `.env`

```bash
cp .env.example .env
```

Variabel kritis yang wajib diisi:

| Variable | Keterangan |
|----------|-----------|
| `SECRET_KEY` | `python -c "import secrets; print(secrets.token_urlsafe(50))"` |
| `DATABASE_URL` | Supabase: Project Settings -> Database -> Session Pooler URI |
| `NLP_ENABLED` | `false` untuk dev lokal (boot <1 detik), `true` untuk full pipeline |
| `MT5_MODEL_PATH` | HF Hub ID atau path lokal mT5. Default: `OinoVenv/sovereign-mt5-nusasum` |
| `INDOT5_MODEL_PATH` | HF Hub ID atau path lokal IndoT5. Default: `OinoVenv/sovereign-indot5-nusasum` |
| `SUMMARIZER_MODEL` | Model utama: `mt5`, `indot5`, `textrank`, `ner`, `first_sentences` |

### Jalankan server

```bash
python manage.py migrate
python manage.py runserver   # -> http://localhost:8000
```

**Boot time:**
- `NLP_ENABLED=false`: **<1 detik** — semua stage pakai fallback, tanpa load torch
- `NLP_ENABLED=true`: **<1 detik** untuk listen port, model load di background thread

---

## Arsitektur Lazy Loading

Model neural di-load di background setelah server listen:

1. `complaints/apps.py.ready()` memanggil `loader.start_warmup()` saat Django boot
2. Background thread load model satu per satu (tidak blocking)
3. Pipeline memakai `loader.get_if_loaded(...)` — jika model belum siap, langsung fallback tanpa error

Dengan `NLP_ENABLED=false`, seluruh `get_if_loaded()` return `None` dan pipeline otomatis pakai fallback chain.

---

## Fallback Chain per Stage

| Stage | Primary | Fallback 1 | Fallback 2 | Final |
|-------|---------|------------|------------|-------|
| Dialect | joblib LogReg (12 dialek) | langdetect | — | `"xx"` |
| Translate | NLLB-200 lokal | deep_translator | — | teks asli |
| Summarize | `SUMMARIZER_MODEL` (mt5/indot5) | NER extractive | TextRank | 2 kalimat pertama |
| NER | cahya BERT | regex (Jalan/Desa/Dinas) | — | `[]` |
| Category | keyword matching | — | — | `"Umum"` |
| Urgency | weighted keyword | — | — | `"low"` |

---

## Mapping Dialek -> NLLB Flores-200

| Dialek | NLLB Code | Catatan |
|--------|-----------|---------|
| id | ind_Latn | Bahasa Indonesia (target) |
| jv | jav_Latn | Jawa |
| su | sun_Latn | Sunda |
| min | min_Latn | Minangkabau |
| ace | ace_Latn | Aceh |
| ban | ban_Latn | Bali |
| bjn | bjn_Latn | Banjar |
| bug | bug_Latn | Bugis |
| mad | — | Fallback deep_translator |
| nij | — | Fallback deep_translator |
| bbc | — | Fallback deep_translator |

---

## Perintah Umum

```bash
python manage.py migrate
python manage.py createsuperuser       # akun admin /admin/
pytest                                  # semua test
pytest accounts/tests.py -v             # test spesifik
```

---

## Smoke Test

```bash
# Dev mode — fallback only, tanpa load model neural
NLP_ENABLED=false python manage.py shell -c "
from nlp.pipeline import run_pipeline
r = run_pipeline('Dalane rusak banget wis suwe ora dibenahi, tolong segera diperbaiki')
print('dialect:', r.dialect, r.dialect_confidence)
print('summary:', r.summary[:120])
print('urgency:', r.urgency_level, '| category:', r.category_name)
"
```

---

## Troubleshooting

| Gejala | Fix |
|--------|-----|
| Boot lama walau `NLP_ENABLED=false` | Pastikan `import torch` di-import lazy di `nlp/pipeline.py`, bukan di top-level |
| OOM saat load mT5 di Mac 8 GB | Set `NLP_ENABLED=false` — 8 GB tidak cukup untuk full pipeline + dev tools |
| `psycopg2` gagal install di macOS | `brew install libpq && pip install psycopg2-binary --no-cache` |
| `migrate` gagal | Cek apakah Supabase project sedang pause (restore dari dashboard) |
| HF Hub download lambat | Set `HF_TOKEN` env var (dari huggingface.co/settings/tokens) untuk rate limit lebih tinggi |
| Model status `failed` | Pipeline tetap jalan via fallback — cek log untuk root cause (OOM / network / disk) |

---

## Deploy ke HF Spaces

1. Upload model ke HF Hub (via `scripts/upload_mt5_to_hf.py`)
2. Buat HF Space dengan Docker SDK
3. Push folder `application_demo/` ke repo Space
4. Set Secrets: `SECRET_KEY`, `DATABASE_URL`, `NLP_ENABLED=true`, `MT5_MODEL_PATH`, `INDOT5_MODEL_PATH`
5. Aktifkan **persistent storage** agar model cache tidak hilang saat cold start
