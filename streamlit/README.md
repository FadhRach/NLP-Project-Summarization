# Sovereign Dialect-Bridge — Streamlit App

Demo interaktif pipeline dua tahap: teks dialek atau Bahasa Indonesia baku → normalisasi → ringkasan BI baku.

## Requirements

- Python 3.10 atau lebih baru
- Model files di folder `model/` (hasil training dari notebook 2 & 3)
- Internet pertama kali untuk download model NER dari HuggingFace (`cahya/bert-base-indonesian-NER`)

## Setup

### 1. Buat virtual environment

**Conda (disarankan):**
```bash
conda create -n dialect-bridge python=3.10
conda activate dialect-bridge
```

**venv:**
```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 2. Install dependencies

Dari folder root project:
```bash
pip install -r streamlit/requirements.txt
```

> **Apple Silicon (M1/M2/M3):** torch dari PyPI sudah mendukung MPS, tidak perlu flag tambahan.
>
> **CUDA (RTX / A100):** Ganti `torch>=2.0.0` di requirements.txt dengan versi CUDA yang sesuai:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu118
> ```

### 3. Jalankan app

Dari **root folder project** (bukan dari dalam folder `streamlit/`):
```bash
streamlit run streamlit/app.py
```

App terbuka di browser: `http://localhost:8501`

Ganti port jika sudah terpakai:
```bash
streamlit run streamlit/app.py --server.port 8502
```

## Model files

App memerlukan model yang sudah ditraining di:

| Folder | Sumber | Diperlukan untuk |
|--------|--------|-----------------|
| `model/normalizer/` | `training_normalizer.ipynb` | Stage 1 — normalisasi dialek |
| `model/mt5base/` | `training_sum.ipynb` | Stage 2 — ringkasan (prioritas utama) |
| `model/indot5/` | `training_sum.ipynb` | Stage 2 — ringkasan (fallback) |

Jika folder model tidak ada, pipeline otomatis fallback ke **TextRank** (tidak perlu model).

NER (`cahya/bert-base-indonesian-NER`) di-download otomatis dari HuggingFace saat pertama kali dijalankan dan di-cache di `~/.cache/huggingface/`.

## Troubleshooting

### `Failed building wheel for tokenizers` (Python 3.13)

`tokenizers` versi lama tidak support Python 3.13. Versi requirements.txt saat ini sudah memakai `transformers>=4.46.0` yang kompatibel dengan Python 3.13. Jika masih error, coba upgrade pip dulu:
```bash
pip install --upgrade pip
pip install -r streamlit/requirements.txt
```

Atau gunakan Python 3.12 yang lebih stabil:
```bash
conda create -n dialect-bridge python=3.12
conda activate dialect-bridge
pip install -r streamlit/requirements.txt
```

### `ModuleNotFoundError: No module named 'torch'`

Package belum terinstall di environment aktif:
```bash
pip install -r streamlit/requirements.txt
```

Pastikan virtual environment sudah diaktifkan sebelum menjalankan pip install dan streamlit.

### `ModuleNotFoundError: No module named 'PySastrawi'`

```bash
pip install PySastrawi
```

### `streamlit: command not found`

```bash
pip install streamlit
# atau
python3 -m streamlit run streamlit/app.py
```

### App sangat lambat saat pertama kali dijalankan

Normal — model neural (mT5-base ~1.2 GB, IndoT5 ~900 MB) dimuat ke memori saat pertama kali tombol "Ringkas" ditekan. Setelah itu lebih cepat.

### `CUDA out of memory`

Kurangi beban dengan menutup proses GPU lain, atau jalankan tanpa CUDA:
```bash
CUDA_VISIBLE_DEVICES="" streamlit run streamlit/app.py
```
