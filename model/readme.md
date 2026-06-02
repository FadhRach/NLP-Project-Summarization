# Model Weights

Semua file bobot model **tidak disertakan di repository** karena ukurannya yang besar (total ~5 GB).
Download melalui Google Drive di bawah, lalu letakkan setiap folder di path yang sesuai.

**Google Drive:**
[https://drive.google.com/drive/folders/1dOkmJI__dfwsAJXDyqMSqTqlS94KnaBw?usp=sharing](https://drive.google.com/drive/folders/1dOkmJI__dfwsAJXDyqMSqTqlS94KnaBw?usp=sharing)

---

## Struktur folder setelah download

```
model/
├── normalizer/          Stage 1 — dialect normalizer (~2.1 GB)
│   ├── model.safetensors
│   ├── config.json
│   ├── generation_config.json
│   ├── tokenizer_config.json
│   └── tokenizer.json
│
├── mt5base/             Stage 2 — mT5-base summarizer (~2.2 GB)
│   ├── model.safetensors
│   ├── config.json
│   ├── generation_config.json
│   ├── special_tokens_map.json
│   ├── spiece.model
│   ├── tokenizer_config.json
│   └── tokenizer.json
│
├── indot5/              Stage 2 — IndoT5 summarizer (~853 MB)
│   ├── model.safetensors
│   ├── config.json
│   ├── generation_config.json
│   ├── special_tokens_map.json
│   ├── spiece.model
│   ├── tokenizer_config.json
│   └── tokenizer.json
│
└── readme.md            (file ini)
```

---

## Keterangan model

| Folder | Base model | Dilatih pada | Ukuran | Kegunaan |
|---|---|---|---|---|
| `normalizer/` | `google/mt5-small` | NusaX-MT + IndonesianNMT + IndoNLG MT | ~2.1 GB | Stage 1: teks dialek → Bahasa Indonesia baku |
| `mt5base/` | `google/mt5-base` | IndoSum 10K (train fold 1) | ~2.2 GB | Stage 2: ringkasan abstraktif (baseline multilingual) |
| `indot5/` | `cahya/t5-base-indonesian-summarization-cased` | IndoSum 10K (train fold 1) | ~853 MB | Stage 2: ringkasan abstraktif (model khusus BI) |

> Model NER (`cahya/bert-base-indonesian-NER`) **tidak perlu diunduh manual** — diambil otomatis dari HuggingFace cache saat pertama kali dijalankan.

---

## Cara menempatkan file

1. Download folder `normalizer`, `mt5base`, dan `indot5` dari Google Drive.
2. Letakkan langsung di dalam folder `model/` ini sehingga strukturnya sesuai di atas.
3. Pastikan setiap subfolder berisi `config.json` dan file bobot (`model.safetensors` atau `pytorch_model.bin`).
4. Jalankan Streamlit: `streamlit run streamlit/app.py` dari root project.

Saat app pertama kali dibuka, semua model di-load otomatis dengan progress bar.
