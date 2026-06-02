#!/usr/bin/env python3
"""
Sovereign Dialect-Bridge — ringkas teks artikel Indonesia.

Penggunaan:
    python summarize.py                         # contoh bawaan
    python summarize.py "teks artikel Anda..."  # teks langsung
    echo "isi artikel" | python summarize.py    # dari stdin / pipe

Model dipakai secara prioritas:
    mT5-base (ROUGE tertinggi) -> IndoT5 -> IndoBART -> TextRank (fallback CPU)
"""
import os, gc, re, sys, json
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── Device ─────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"    # Apple Silicon
else:
    DEVICE = "cpu"

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"

MODEL_PRIORITY = [
    ("mt5base",  "mt5"),
    ("indot5",   "t5"),
    ("indobart", "bart"),
]

GEN_KWARGS = dict(
    max_new_tokens       = 150,
    num_beams            = 4,
    no_repeat_ngram_size = 3,
    early_stopping       = True,
    length_penalty       = 1.0,
)

# ── Preprocessing — identik dengan training_sum.ipynb ─────────────────────
def clean_noise(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize_case_punct(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[!?]{2,}", "!", text)
    return re.sub(r"\s([.,!?;:])", r"\1", text).strip()

def preprocess(text: str) -> str:
    """Preprocessing abstractive: clean + normalize, tanpa stemming."""
    return normalize_case_punct(clean_noise(text))

# ── TextRank fallback (tidak butuh GPU, berjalan di CPU) ──────────────────
def textrank(text: str, n: int = 3, max_words: int = 80) -> str:
    """Pilih n kalimat paling penting via PageRank + TF-IDF cosine similarity."""
    raw_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if len(raw_sents) <= n:
        return " ".join(raw_sents[:max_words])

    # stem hanya untuk similarity, output tetap pakai raw_sents
    stem_sents = [preprocess(s) for s in raw_sents]
    try:
        import numpy as np
        import networkx as nx
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vec   = TfidfVectorizer().fit_transform(stem_sents)
        sim   = cosine_similarity(vec)
        np.fill_diagonal(sim, 0)
        ranks = nx.pagerank(nx.from_numpy_array(sim))
        top   = sorted(ranks, key=ranks.get, reverse=True)[:n]
        result = " ".join(raw_sents[i] for i in sorted(top))
    except ImportError:
        result = " ".join(raw_sents[:n])
    except Exception:
        result = " ".join(raw_sents[:n])

    return " ".join(result.split()[:max_words])

# ── Load tokenizer — handle IndoBART tokenizer_info.json fallback ──────────
def load_tokenizer(model_dir: Path, model_type: str):
    """
    IndoBART menyimpan tokenizer_info.json karena IndoNLGTokenizer.save_pretrained()
    raise NotImplementedError. Cek file tersebut dan reload dari HF jika perlu.
    """
    info_path = model_dir / "tokenizer_info.json"
    source    = str(model_dir)
    if info_path.exists():
        source = json.loads(info_path.read_text()).get("original_model", source)
        print(f"    tokenizer fallback: {source}")

    if model_type == "bart":
        try:
            from indobenchmark import IndoNLGTokenizer
            return IndoNLGTokenizer.from_pretrained(source)
        except Exception:
            pass
    return AutoTokenizer.from_pretrained(source)

# ── Safe decode — IndoNLGTokenizer tidak support clean_up_tokenization_spaces
def safe_decode(tok, ids) -> str:
    seq = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    try:
        return tok.decode(seq, skip_special_tokens=True)
    except TypeError:
        return tok.decode(seq)

# ── Summarize ──────────────────────────────────────────────────────────────
def summarize(text: str) -> tuple[str, str]:
    """
    Kembalikan (ringkasan, nama_model).
    Coba model fine-tuned secara prioritas; fallback ke TextRank jika tidak ada.
    """
    for name, model_type in MODEL_PRIORITY:
        model_dir = MODELS_DIR / name
        if not model_dir.exists():
            print(f"  [{name}] tidak ditemukan, lewati...")
            continue

        print(f"  Menggunakan model  : {name} ({model_type})")
        tok   = load_tokenizer(model_dir, model_type)
        model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir)).to(DEVICE)
        model.eval()

        inp = preprocess(text)
        if model_type == "mt5":
            inp = "summarize: " + inp

        enc = tok(inp, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            out = model.generate(**enc, **GEN_KWARGS)
        result = safe_decode(tok, out[0])

        del model, tok
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        return result, name

    print("  Tidak ada model fine-tuned ditemukan, pakai TextRank (CPU).")
    return textrank(text), "TextRank"


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Ambil teks dari: argumen CLI > stdin (pipe) > contoh bawaan
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    elif not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    else:
        text = (
            "Jakarta, CNN Indonesia -- Kementerian Perhubungan (Kemenhub) memastikan "
            "bahwa mudik Lebaran 2026 akan berlangsung lebih teratur dibandingkan tahun lalu. "
            "Menteri Perhubungan Budi Karya Sumadi mengatakan pihaknya telah menyiapkan "
            "sejumlah kebijakan untuk mengurai kemacetan di jalur mudik, termasuk rekayasa "
            "lalu lintas, penambahan armada transportasi umum, dan perluasan rest area di "
            "jalan tol. Puncak arus mudik diperkirakan terjadi pada H-2 sebelum Hari Raya. "
            "Sebanyak 12 juta kendaraan diprediksi memadati jalan tol trans-Jawa. "
            "Kemenhub mengimbau masyarakat untuk berangkat lebih awal dan memanfaatkan "
            "program mudik gratis yang disediakan pemerintah. "
            "Selain itu, pihak kepolisian juga akan melakukan pengamanan ketat di seluruh "
            "titik rawan kemacetan dan kecelakaan sepanjang jalur mudik nasional."
        )

    n_in = len(text.split())
    print(f"\nInput ({n_in} kata):")
    print(f"  {text[:280]}{'...' if len(text) > 280 else ''}")
    print(f"\nDevice : {DEVICE}  |  Models : {MODELS_DIR}")
    print()

    summary, method = summarize(text)

    n_out = len(summary.split())
    cr    = round(n_out / max(n_in, 1), 2)
    print(f"\nRingkasan [{method}]  ({n_out} kata, CR={cr}):")
    print(f"  {summary}")
    print()
