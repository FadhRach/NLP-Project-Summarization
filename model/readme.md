# Model Weights

Direktori ini menyimpan model weights lokal (opsional). Model juga tersedia langsung di HuggingFace Hub dan dapat di-load tanpa download manual.

---

## Opsi Load Model

### Opsi 1 — HuggingFace Hub (direkomendasikan)

Model fine-tuned sudah di-upload ke HuggingFace Hub. Cukup gunakan ID berikut di kode:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# mT5-base (fine-tuned IndoSum)
tokenizer = AutoTokenizer.from_pretrained("OinoVenv/sovereign-mt5-nusasum")
model     = AutoModelForSeq2SeqLM.from_pretrained("OinoVenv/sovereign-mt5-nusasum")

# IndoT5 (fine-tuned IndoSum)
tokenizer = AutoTokenizer.from_pretrained("OinoVenv/sovereign-indot5-nusasum")
model     = AutoModelForSeq2SeqLM.from_pretrained("OinoVenv/sovereign-indot5-nusasum")
```

Model akan di-cache otomatis di `~/.cache/huggingface/` setelah pertama kali diunduh.

### Opsi 2 — Download Lokal via Google Drive

Download folder model dari Google Drive, lalu letakkan di dalam direktori `model/` ini:

**Google Drive:**
[https://drive.google.com/drive/folders/1dOkmJI__dfwsAJXDyqMSqTqlS94KnaBw?usp=sharing](https://drive.google.com/drive/folders/1dOkmJI__dfwsAJXDyqMSqTqlS94KnaBw?usp=sharing)

Setelah download, load dengan path lokal:

```python
model = AutoModelForSeq2SeqLM.from_pretrained("model/mt5base")
```

---

## Daftar Model

| Model | HF Hub ID | Folder Lokal | Ukuran | Kegunaan |
|-------|-----------|--------------|--------|----------|
| mT5-base | `OinoVenv/sovereign-mt5-nusasum` | `model/mt5base/` | ~2.2 GB | Abstractive summarizer (multilingual) |
| IndoT5 | `OinoVenv/sovereign-indot5-nusasum` | `model/indot5/` | ~853 MB | Abstractive summarizer (khusus BI) |
| NLLB-200 | `facebook/nllb-200-distilled-600M` | — (auto HF cache) | ~1.2 GB | Translasi dialek -> Bahasa Indonesia |
| NER BERT | `cahya/bert-base-indonesian-NER` | — (auto HF cache) | ~440 MB | Named entity extraction (extractive) |
| Dialect Detector | — | `notebook/dialect_detector/dialect_detector.joblib` | ~5 MB | Deteksi dialek (TF-IDF + LogReg) |

Model dengan tanda `—` pada kolom "Folder Lokal" tidak perlu diunduh manual — diambil otomatis dari HuggingFace Hub saat pertama kali dipakai.

---

## Struktur Folder Lokal (jika download manual)

```
model/
├── mt5base/             mT5-base fine-tuned IndoSum (~2.2 GB)
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── spiece.model
│   ├── tokenizer_config.json
│   └── tokenizer.json
│
└── indot5/              IndoT5 fine-tuned IndoSum (~853 MB)
    ├── config.json
    ├── generation_config.json
    ├── model.safetensors
    ├── special_tokens_map.json
    ├── spiece.model
    ├── tokenizer_config.json
    └── tokenizer.json
```

---

## Catatan

- Model IndoBART (`indobenchmark/indoBART`) dilatih di `training_sum.ipynb` tapi tidak di-upload ke HF Hub — download dari Google Drive jika diperlukan.
- Dialect detector (`.joblib`) disimpan di `notebook/dialect_detector/`, bukan di folder ini.
- Backend Django (`application_demo/`) membaca model dari path yang dikonfigurasi via environment variable `MT5_MODEL_PATH` dan `INDOT5_MODEL_PATH` — bisa diisi HF Hub ID atau path lokal.
