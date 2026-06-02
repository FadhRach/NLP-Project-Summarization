# Sovereign Dialect-Bridge

**Abstractive Summarization of Indonesian Regional Dialects for Local Governance**

BINUS School of Computer Science — NLP Final Project, Semester 4, Group 2  
Bintang · Dian · Fadhlan

---

## Latar Belakang

Indonesia memiliki lebih dari 700 bahasa daerah. Laporan pengaduan warga sering disampaikan dalam dialek lokal (Jawa, Sunda, Minangkabau, dll.) yang sulit diproses oleh sistem birokrasi formal yang hanya memahami Bahasa Indonesia baku. Proyek ini membangun pipeline **Two-Stage Dialect Bridge** yang mengonversi teks pengaduan dialek menjadi ringkasan Bahasa Indonesia baku secara otomatis.

---

## Arsitektur Pipeline

```
INPUT (teks dialek atau BI)
        |
        v
 ┌─────────────────┐
 │  Stage 1        │  mT5-small fine-tuned pada NusaX-MT + IndonesianNMT + IndoNLG MT
 │  Normalizer     │  Dialek --> Bahasa Indonesia baku
 │  (opsional)     │  QC gate: output > 30% panjang input
 └────────┬────────┘
          |
          v
 ┌─────────────────────────────────────────┐
 │  Stage 2 — Abstractive Summarization   │
 │                                         │
 │  mT5-base  (google/mt5-base, ~1.2 GB)  │
 │  IndoT5    (cahya/t5-base-..., ~900 MB) │
 └────────┬────────────────────────────────┘
          |
          | (paralel)
          v
 ┌───────────────────────────────┐
 │  Extractive (selalu aktif)    │
 │  NER     : BERT indonesian    │
 │  TextRank: TF-IDF + PageRank  │
 └───────────────────────────────┘
          |
          v
   OUTPUT: ringkasan BI baku
```

---

## Struktur Repository

```
sovereign-dialect-bridge/
│
├── notebook/                          Notebook training & evaluasi (jalankan berurutan)
│   ├── text_conversion.ipynb          Step 1: IndoSum JSONL --> Parquet bersih + EDA
│   ├── training_normalizer.ipynb      Step 2: Fine-tune mT5-small normalizer
│   ├── training_sum.ipynb             Step 3: Fine-tune mT5-base + IndoT5
│   ├── training_sum_NER.ipynb         Eksperimen NER extractive (standalone)
│   ├── inference.ipynb                Step 4: Evaluasi end-to-end + demo
│   └── legacy/                        Notebook lama (referensi saja)
│
├── dataset/                           Dataset mentah (tidak di-commit, download manual)
│   ├── indosum/                       14.262 artikel berita BI + summary (fold 1)
│   │   └── README.txt                 Lisensi dan info IndoSum
│   └── nusax/                         1.000 kalimat x 12 bahasa daerah (NusaX-MT)
│
├── data/                              Data hasil preprocessing (sebagian di-commit)
│   ├── dialect_dict.json              Kamus pasangan dialek-BI auto-expanded dari NusaX
│   └── train_sample.csv              100 sampel train untuk cek cepat
│   (*.parquet di-generate notebook 1, tidak di-commit)
│
├── model/                             Model weights (tidak di-commit, download Google Drive)
│   ├── normalizer/                    Stage 1: mT5-small normalizer (~2.1 GB)
│   ├── mt5base/                       Stage 2: mT5-base summarizer (~2.2 GB)
│   ├── indot5/                        Stage 2: IndoT5 summarizer (~853 MB)
│   └── readme.md                      Instruksi download model
│
├── streamlit/                         Demo app interaktif
│   ├── app.py                         Aplikasi Streamlit utama
│   ├── requirements.txt               Dependensi Python untuk Streamlit
│   └── README.md                      Panduan menjalankan demo
│
├── outputs/                           Hasil evaluasi (di-generate notebook 4, tidak di-commit)
│   └── final_results.json
│
├── summarize.py                       Script inference CLI sederhana
└── README.md                          (file ini)
```

---

## Setup & Cara Menjalankan

### 1. Clone repository

```bash
git clone <repo-url>
cd sovereign-dialect-bridge
```

### 2. Download dataset mentah

**IndoSum** — letakkan di `dataset/indosum/`:
- Download dari: [https://github.com/kata-ai/indosum](https://github.com/kata-ai/indosum)
- File yang dibutuhkan: `train.01.jsonl`, `dev.01.jsonl`, `test.01.jsonl`

**NusaX-MT** — letakkan di `dataset/nusax/datasets/mt/`:
- Download dari: [https://github.com/IndoNLP/nusax](https://github.com/IndoNLP/nusax)
- File yang dibutuhkan: `train.csv`, `valid.csv`, `test.csv`

### 3. Download model weights

Unduh dari Google Drive dan letakkan di `model/`:

[https://drive.google.com/drive/folders/1dOkmJI__dfwsAJXDyqMSqTqlS94KnaBw?usp=sharing](https://drive.google.com/drive/folders/1dOkmJI__dfwsAJXDyqMSqTqlS94KnaBw?usp=sharing)

Lihat `model/readme.md` untuk instruksi lengkap dan struktur folder yang diharapkan.

### 4. Install dependensi

Untuk Streamlit demo (lokal):

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r streamlit/requirements.txt
```

Untuk training di vast.ai / Colab:

```bash
pip install transformers==4.40.0 datasets accelerate sentencepiece
pip install bert-score rouge-score sacrebleu sacremoses
pip install bitsandbytes PySastrawi sumy
pip install pandas pyarrow fastparquet matplotlib seaborn
```

### 5. Jalankan notebook (urutan wajib)

| Urutan | Notebook | Input | Output | Estimasi |
|--------|----------|-------|--------|----------|
| 1 | `text_conversion.ipynb` | `dataset/indosum/*.jsonl` | `data/*.parquet` | ~10 menit (CPU) |
| 2 | `training_normalizer.ipynb` | `dataset/nusax/` + online HF | `model/normalizer/` | ~45 menit (GPU) |
| 3 | `training_sum.ipynb` | `data/train.parquet` | `model/mt5base/`, `model/indot5/` | ~2.5 jam (GPU) |
| 4 | `inference.ipynb` | semua model + `data/test.parquet` | `outputs/final_results.json` | ~45 menit (GPU) |

> Hardware target: **vast.ai RTX 3090/4090**, 24 GB VRAM, bf16, Python 3.10+

### 6. Jalankan Streamlit demo

Pastikan model weights sudah diunduh ke `model/`, lalu dari root project:

```bash
streamlit run streamlit/app.py
```

Buka browser di `http://localhost:8501`. Semua model di-load otomatis dengan progress bar saat pertama kali dibuka.

---

## Metode yang Diimplementasikan

| Metode | Tipe | Model | Keterangan |
|--------|------|-------|------------|
| TextRank | Ekstraktif | — | TF-IDF + PageRank, selalu tersedia |
| NER | Ekstraktif | `cahya/bert-base-indonesian-NER` | Prioritas kalimat berisi entitas penting |
| mT5-base | Abstraktif | `google/mt5-base` fine-tuned | Baseline multilingual |
| IndoT5 | Abstraktif | `cahya/t5-base-indonesian-summarization-cased` fine-tuned | Model khusus BI |

---

## Metrik Evaluasi

- **ROUGE-1 / ROUGE-2 / ROUGE-L** — overlap n-gram antara output dan referensi
- **BERTScore-F1** — kemiripan semantik berbasis BERT
- **Compression Ratio** — rasio panjang ringkasan terhadap teks asli (target ~0.22)
- **BLEU-4 + chrF++** — evaluasi normalizer pada NusaX validation set (target BLEU > 20)

---

## Dataset

| Dataset | Sumber | Ukuran | Kegunaan |
|---------|--------|--------|----------|
| IndoSum | [Koto et al. 2018](https://arxiv.org/pdf/1810.05334) | ~14K artikel | Training & evaluasi summarizer |
| NusaX-MT | [Winata et al. 2022](https://arxiv.org/pdf/2205.15960) | 1K x 12 bahasa | Training normalizer + dialect test set |
| IndonesianNMT | [Exqrch/IndonesianNMT](https://huggingface.co/datasets/Exqrch/IndonesianNMT) | 10K–100K/dialek | Augmentasi data normalizer |
| IndoNLG MT | [GEM/indonlg](https://huggingface.co/datasets/GEM/indonlg) | ~5–10K/dialek | Data kualitas tinggi normalizer |

---

## Referensi

- Koto et al. (2018) — IndoSum: [arxiv.org/pdf/1810.05334](https://arxiv.org/pdf/1810.05334)
- Cahyawijaya et al. (2021) — IndoBART/IndoNLG: [aclanthology.org/2021.emnlp-main.699](https://aclanthology.org/2021.emnlp-main.699)
- Winata et al. (2022) — NusaX: [arxiv.org/pdf/2205.15960](https://arxiv.org/pdf/2205.15960)
- Mihalcea & Tarau (2004) — TextRank: [aclanthology.org/W04-3252.pdf](https://aclanthology.org/W04-3252.pdf)
- Lin (2004) — ROUGE: [aclanthology.org/W04-1013.pdf](https://aclanthology.org/W04-1013.pdf)
- Zhang et al. (2020) — BERTScore: [arxiv.org/abs/1904.09675](https://arxiv.org/abs/1904.09675)
- Xue et al. (2021) — mT5: [arxiv.org/abs/2010.11934](https://arxiv.org/abs/2010.11934)
