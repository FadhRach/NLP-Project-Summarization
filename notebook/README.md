# Notebook — Sovereign Dialect-Bridge

Direktori ini berisi seluruh Jupyter notebook untuk training, evaluasi, dan demo pipeline NLP.

---

## Urutan Eksekusi

```
1. dialect_detector/train_dialect_detector.ipynb  -> dialect_detector.joblib
2. training_sum.ipynb                             -> model/{mt5base, indot5, indobart}/
3. training_sum_NER.ipynb                         -> evaluasi NER BERT (opsional, standalone)
4. inference.ipynb                                -> demo pipeline + tabel ROUGE & CR
5. text_conversion.ipynb                          -> EDA + konversi indosum (.json -> .parque)
```

---

## Daftar Notebook

### `dialect_detector/train_dialect_detector.ipynb`

Melatih classifier ringan untuk mendeteksi 12 bahasa/dialek dari teks input.

- **Input** : `dataset/nusax/datasets/mt/{train,valid,test}.csv`
- **Output** : `dialect_detector/dialect_detector.joblib` + `metadata.json`
- **Isi** : EDA distribusi kelas dan panjang teks, analisis top char n-gram per dialek, perbandingan tiga model (Logistic Regression vs Linear SVC vs Naive Bayes), 5-fold cross-validation, kalibrasi confidence threshold, confusion matrix, simpan model
- **Catatan** : Logistic Regression dipilih karena menyediakan `predict_proba` untuk confidence-based fallback di backend (`"xx"` jika confidence < 0.35)

---

### `training_sum.ipynb`

Melatih tiga model abstractive summarizer dan mengevaluasi dua metode extractive baseline.

- **Input** : `dataset/indosum_parque/{train,val,test}.parquet`
- **Output** : `model/{mt5base, indot5, indobart}/` (checkpoint fine-tuned)
- **Isi** :
  - Fine-tune **IndoT5** (`cahya/t5-base-indonesian-summarization-cased`)
  - Fine-tune **IndoBART** (`indobenchmark/indoBART`)
  - Fine-tune **mT5-base** (`google/mt5-base`)
  - Evaluasi extractive **TextRank** (TF-IDF + PageRank)
  - Evaluasi extractive **NER heuristik** (lihat catatan di bawah)
  - Evaluasi ROUGE-1/2/L + Compression Ratio untuk semua metode

- **NER heuristik** (di notebook ini): kalimat diberi skor berdasarkan **density kata berkapital** — kata ke-2 dan seterusnya yang huruf pertamanya uppercase dianggap sebagai proxy named entity. Pendekatan ini cepat dan tidak membutuhkan model tambahan, tapi kurang akurat dibanding NER berbasis BERT.

---

### `training_sum_NER.ipynb`

Evaluasi metode extractive NER menggunakan model BERT untuk deteksi entitas yang lebih akurat.

- **Input** : `dataset/indosum_parque/test.parquet` (700 sampel)
- **Output** : Skor ROUGE-1/2/L + Compression Ratio, inspeksi best/worst case
- **Isi** : Load `cahya/bert-base-indonesian-NER`, batched inference per kalimat, skor tiap kalimat berdasarkan jumlah entitas nyata yang ditemukan, evaluasi dan perbandingan dengan baseline TextRank

- **NER berbasis BERT** (di notebook ini): menggunakan model checkpoint `cahya/bert-base-indonesian-NER` (~440 MB) yang mendeteksi entitas nyata — PER (orang), LOC (lokasi), ORG (organisasi), dan lainnya. Jauh lebih akurat dari pendekatan heuristik kata berkapital, tapi membutuhkan download model dan lebih lambat.

**Perbandingan dua pendekatan NER:**

| Aspek | NER Heuristik (`training_sum.ipynb`) | NER BERT (`training_sum_NER.ipynb`) |
|-------|--------------------------------------|--------------------------------------|
| Cara kerja | Hitung kata berkapital sebagai proxy entitas | Deteksi entitas nyata via BERT |
| Model tambahan | Tidak ada | `cahya/bert-base-indonesian-NER` (~440 MB) |
| Kecepatan | Sangat cepat (< 1 ms/teks) | Lebih lambat (BERT inference) |
| Akurasi | Lebih rendah (heuristik) | Lebih tinggi (model NLP) |
| Cocok untuk | Baseline cepat, resource terbatas | Evaluasi komprehensif |

---

### `inference.ipynb`

Demo end-to-end pipeline dari teks dialek hingga ringkasan, disertai evaluasi ROUGE dan Compression Ratio.

- **Input** : Teks dialek (Jawa, Sunda, Minang, BI) + `dataset/indosum_parque/test.parquet`
- **Output** : Tabel perbandingan ROUGE-1/2/L + CR untuk 4 metode, visualisasi bar chart
- **Pipeline** :
  ```
  teks dialek
    -> detect_dialect  (joblib lokal)
    -> NLLB translate  (facebook/nllb-200-distilled-600M)
    -> summarize       (TextRank | NER BERT | mT5 | IndoT5)
    -> ROUGE & CR
  ```
- **Model abstractive** diambil langsung dari HuggingFace Hub (`OinoVenv/`), tidak perlu download manual

---

## Konfigurasi Global

Nilai berikut konsisten di seluruh notebook. Lihat detail di `CLAUDE.md`.

```python
SUMM_LR         = 5e-5     # KRITIS: NaN crash jika lebih tinggi
SUMM_MAX_TARGET = 150
USE_BF16        = True     # RTX 30xx/40xx — JANGAN fp16
no_repeat_ngram_size = 3   # di semua .generate()
```
