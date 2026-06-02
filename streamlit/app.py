#!/usr/bin/env python3
"""
Sovereign Dialect-Bridge — Streamlit demo app.
Run: streamlit run streamlit/app.py  (dari root folder project)
"""
import os, re, json, time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st

# ── Dependency guard ────────────────────────────────────────────────────────
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline as hf_pipeline
except ModuleNotFoundError as _err:
    st.error(f"Dependency tidak ditemukan: `{_err}`")
    st.code("pip install -r streamlit/requirements.txt", language="bash")
    st.stop()

# ── Device ─────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
MODELS_DIR  = (ROOT / "model") if (ROOT / "model").exists() else (ROOT / "models")

NORM_DIR    = MODELS_DIR / "normalizer"
INDOT5_DIR  = MODELS_DIR / "indot5"
MT5BASE_DIR = MODELS_DIR / "mt5base"

GEN_KWARGS = dict(
    max_new_tokens       = 150,
    num_beams            = 2,
    no_repeat_ngram_size = 3,
    early_stopping       = True,
    length_penalty       = 1.0,
)

_ENTITY_LABELS = {
    "PER": "Orang",     "ORG": "Organisasi", "LOC": "Lokasi",
    "QTY": "Kuantitas", "TIM": "Waktu",       "EVT": "Kejadian",
    "LAW": "Regulasi",  "CRD": "Angka",       "NOR": "Lembaga",
}

_MODEL_HELP = {
    "mT5-base": (
        "Model multilingual T5 dari Google (~1.2 GB, google/mt5-base), di-fine-tune pada 10K artikel "
        "IndoSum sebagai baseline. Mendukung banyak bahasa termasuk Bahasa Indonesia. "
        "Menghasilkan ringkasan konservatif dan faktual. "
        "Prefix 'summarize:' ditambahkan otomatis agar model mengenali tugasnya."
    ),
    "IndoT5": (
        "Model T5 khusus Bahasa Indonesia (~900 MB, cahya/t5-base-indonesian-summarization-cased), "
        "di-fine-tune pada 10K artikel IndoSum. Menghasilkan ringkasan abstraktif — "
        "dapat menyusun ulang kalimat dan memparafrase konten sumber. "
        "Umumnya lebih ekspresif dibanding mT5-base untuk teks berbahasa Indonesia."
    ),
}

# MODEL_CONFIGS hanya berisi model yang ada di disk
MODEL_CONFIGS: dict[str, tuple[Path, str]] = {}
if MT5BASE_DIR.exists():
    MODEL_CONFIGS["mT5-base"] = (MT5BASE_DIR, "mt5")
if INDOT5_DIR.exists():
    MODEL_CONFIGS["IndoT5"]   = (INDOT5_DIR,  "t5")

MIN_WORDS_NEURAL = 40

# ── Preprocessing ──────────────────────────────────────────────────────────
def clean_noise(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize_case_punct(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[!?]{2,}", "!", text)
    return re.sub(r"\s([.,!?;:])", r"\1", text).strip()

def preprocess_abstractive(text: str) -> str:
    return normalize_case_punct(clean_noise(text))

def split_sentences(text: str) -> list:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]

# ── TextRank ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _get_stemmer():
    try:
        from PySastrawi.Stemmer.StemmerFactory import StemmerFactory
    except ImportError:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    return StemmerFactory().create_stemmer()

def summarize_textrank(text: str, n_sentences: int = 3, max_words: int = 80) -> str:
    import numpy as np
    import networkx as nx
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    stemmer    = _get_stemmer()
    raw_sents  = split_sentences(text)
    stem_sents = split_sentences(stemmer.stem(normalize_case_punct(clean_noise(text))))
    if len(raw_sents) <= n_sentences:
        return " ".join(" ".join(raw_sents).split()[:max_words])
    try:
        vec    = TfidfVectorizer().fit_transform(stem_sents)
        sim    = cosine_similarity(vec, vec)
        np.fill_diagonal(sim, 0)
        scores = nx.pagerank(nx.from_numpy_array(sim))
        ranked = sorted(scores, key=scores.get, reverse=True)[:n_sentences]
        result = " ".join(raw_sents[i] for i in sorted(ranked))
    except Exception:
        result = " ".join(raw_sents[:n_sentences])
    return " ".join(result.split()[:max_words])

# ── NER ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_ner_pipeline():
    _dev = 0 if DEVICE == "cuda" else ("mps" if DEVICE == "mps" else -1)
    try:
        return hf_pipeline(
            "ner",
            model="cahya/bert-base-indonesian-NER",
            aggregation_strategy="simple",
            device=_dev,
        )
    except Exception:
        return None

def extract_entities(text: str) -> list:
    pipe = _load_ner_pipeline()
    if pipe is None:
        return []
    try:
        raw = pipe(text[:512])
    except Exception:
        return []
    seen, result = set(), []
    for e in raw:
        word = e["word"].strip()
        key  = (word.lower(), e["entity_group"])
        if key not in seen and float(e["score"]) > 0.70:
            seen.add(key)
            result.append({
                "word" : word,
                "type" : e["entity_group"],
                "score": round(float(e["score"]), 3),
            })
    return sorted(result, key=lambda x: x["score"], reverse=True)

def summarize_ner(text: str, n_sentences: int = 3, max_words: int = 80) -> tuple:
    entities  = extract_entities(text)
    ent_words = {e["word"].lower() for e in entities}
    raw_sents = split_sentences(text)
    if len(raw_sents) <= n_sentences:
        joined = " ".join(raw_sents)
        return " ".join(joined.split()[:max_words]), entities
    scores  = [sum(1 for w in ent_words if w in s.lower()) for s in raw_sents]
    ranked  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_sentences]
    summary = " ".join(raw_sents[i] for i in sorted(ranked))
    return " ".join(summary.split()[:max_words]), entities

# ── Model helpers ──────────────────────────────────────────────────────────
def _safe_decode(tok, ids) -> str:
    seq = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    try:
        return tok.decode(seq, skip_special_tokens=True)
    except TypeError:
        return tok.decode(seq)

def _postprocess_output(text: str) -> str:
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r' +', ' ', text).strip()
    text = text.lower()
    if text:
        text = text[0].upper() + text[1:]
    text = re.sub(r'([.!?] )([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
    return text

def _resolve_tokenizer_source(model_dir: Path) -> str:
    info_path = model_dir / "tokenizer_info.json"
    if info_path.exists():
        return json.loads(info_path.read_text()).get("original_model", str(model_dir))
    return str(model_dir)

@st.cache_resource(show_spinner=False)
def _load_cached_model(model_dir_str: str):
    model_dir = Path(model_dir_str)
    source    = _resolve_tokenizer_source(model_dir)
    tok = AutoTokenizer.from_pretrained(source)
    mod = AutoModelForSeq2SeqLM.from_pretrained(model_dir_str).to(DEVICE).eval()
    return tok, mod

def _run_model(tok, model, text: str, model_type: str, max_input: int = 512) -> str:
    inp = preprocess_abstractive(text)
    if model_type == "mt5":
        inp = "summarize: " + inp
    enc = tok(inp, return_tensors="pt", truncation=True, max_length=max_input).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**enc, **GEN_KWARGS)
    return _safe_decode(tok, out[0])

def _warmup_mt5() -> None:
    tok, mod = _load_cached_model(str(MT5BASE_DIR))
    _run_model(tok, mod, "Pemerintah meningkatkan anggaran pendidikan.", "mt5")

# ── Pipeline functions ──────────────────────────────────────────────────────
def _do_normalize(text: str) -> dict:
    t0   = time.time()
    n_in = len(text.split())
    tok, mod = _load_cached_model(str(NORM_DIR))
    raw  = _run_model(tok, mod, text, "mt5", max_input=256)
    qc   = (len(raw.split()) / max(n_in, 1)) >= 0.30
    return {
        "text"           : raw if qc else text,
        "raw_output"     : raw,
        "qc_passed"      : qc,
        "elapsed_ms_norm": round((time.time() - t0) * 1000),
    }

def _do_summarize(text: str, model_choice: str) -> dict:
    t0   = time.time()
    n_in = len(text.split())
    model_dir, model_type = MODEL_CONFIGS[model_choice]
    tok, mod   = _load_cached_model(str(model_dir))
    summary    = _postprocess_output(_run_model(tok, mod, text, model_type))
    elapsed_ms = round((time.time() - t0) * 1000)
    model_used = model_choice if n_in >= MIN_WORDS_NEURAL else f"{model_choice} (input pendek)"
    return {
        "summary"          : summary,
        "model_used"       : model_used,
        "compression_ratio": round(len(summary.split()) / max(n_in, 1), 3),
        "elapsed_ms"       : elapsed_ms,
        "words_per_sec"    : round(n_in / max(elapsed_ms / 1000, 0.001)),
    }

def _render_pipeline_info(use_normalizer: bool, abstractive_choices: list) -> None:
    parts = []
    if use_normalizer and NORM_DIR.exists():
        parts.append("Normalizer aktif")
    else:
        parts.append("Normalizer nonaktif")
    if abstractive_choices:
        parts.append(" + ".join(abstractive_choices))
    parts.append("NER")
    parts.append("TextRank")
    st.info("Pipeline: " + " → ".join(parts))

# ── Streamlit UI ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sovereign Dialect-Bridge",
    page_icon=None,
    layout="centered",
)

# Preload semua model saat app pertama kali dibuka dengan progress bar bertahap.
if not st.session_state.get("_models_ready"):
    _load_steps = []
    if NORM_DIR.exists():
        _load_steps.append(("Memuat Normalizer (mT5-small)...", lambda: _load_cached_model(str(NORM_DIR))))
    if MT5BASE_DIR.exists():
        _load_steps.append(("Memuat mT5-base...", lambda: _load_cached_model(str(MT5BASE_DIR))))
    if INDOT5_DIR.exists():
        _load_steps.append(("Memuat IndoT5...", lambda: _load_cached_model(str(INDOT5_DIR))))
    _load_steps.append(("Memuat NER pipeline...", _load_ner_pipeline))
    _load_steps.append(("Memuat stemmer (Sastrawi)...", _get_stemmer))
    if MT5BASE_DIR.exists():
        _load_steps.append(("Warmup mT5-base...", _warmup_mt5))

    _total = len(_load_steps)
    _status = st.empty()
    _bar    = st.progress(0)
    for _i, (_label, _fn) in enumerate(_load_steps):
        _status.caption(f"Memuat model... ({_i + 1}/{_total}) {_label}")
        _bar.progress(_i / _total)
        _fn()
    _bar.progress(1.0)
    _status.empty()
    _bar.empty()
    st.session_state["_models_ready"] = True

st.title("Sovereign Dialect-Bridge")
st.caption(
    "Pipeline dua tahap: teks dialek atau Bahasa Indonesia "
    "→ normalisasi → ringkasan Bahasa Indonesia baku."
)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Pengaturan")

    use_normalizer = st.toggle(
        "Normalizer Dialek",
        value=True,
        help="Aktifkan agar teks dialek dinormalisasi ke BI baku sebelum diringkas.",
    )

    st.divider()
    st.markdown("**Metode Ekstraktif**")
    st.caption("TextRank dan NER selalu dijalankan secara paralel.")
    with st.expander("Lihat deskripsi"):
        st.caption(
            "**TextRank** — algoritma berbasis graf TF-IDF + PageRank. "
            "Memilih kalimat paling sentral tanpa mengubah kata-kata. "
            "Selalu tersedia, paling cepat, cocok untuk berita panjang."
        )
        st.caption(
            "**NER** — berbasis entitas menggunakan BERT "
            "(cahya/bert-base-indonesian-NER, ~110M param). "
            "Memprioritaskan kalimat yang mengandung nama orang, organisasi, lokasi, "
            "dan waktu. Cocok untuk teks pengaduan warga."
        )

    st.divider()
    st.markdown("**Model Abstraktif**")
    for _name in MODEL_CONFIGS:
        st.checkbox(
            _name,
            value=True,
            key=f"chk_{_name}",
            help=_MODEL_HELP.get(_name, ""),
        )

    selected_abstractive = [
        n for n in MODEL_CONFIGS
        if st.session_state.get(f"chk_{n}", True)
    ]

    st.divider()
    st.caption(f"Device: **{DEVICE}**")
    st.caption(f"Models: `{MODELS_DIR.name}/`")
    st.caption(f"[{'x' if NORM_DIR.exists() else ' '}] normalizer")
    for _name in MODEL_CONFIGS:
        st.caption(f"[x] {_name}")

# ── Input ────────────────────────────────────────────────────────────────────
st.subheader("Input Teks")
input_text = st.text_area(
    label="Teks",
    placeholder=(
        "Masukkan teks di sini. Bisa berupa berita, laporan pengaduan, "
        "atau teks dalam dialek daerah (Jawa, Sunda, Minang, dll.)"
    ),
    height=180,
    label_visibility="collapsed",
)

if input_text:
    n_words = len(input_text.split())
    warn = f" — terlalu pendek untuk model neural, TextRank/NER tetap berjalan" if n_words < MIN_WORDS_NEURAL else ""
    st.caption(f"{n_words} kata{warn}")

# Terapkan contoh teks dari session state sebelum pipeline info dirender
if "_example_text" in st.session_state and not input_text:
    input_text     = st.session_state.pop("_example_text")
    use_normalizer = st.session_state.pop("_example_norm", use_normalizer)

# ── Pipeline info ─────────────────────────────────────────────────────────────
_render_pipeline_info(use_normalizer, selected_abstractive)

# ── Contoh teks ─────────────────────────────────────────────────────────────
EXAMPLES = {
    "Berita BI": (
        "Jakarta, CNN Indonesia -- Kementerian Perhubungan memastikan mudik Lebaran 2026 "
        "akan berlangsung lebih teratur. Menteri Perhubungan Budi Karya menyiapkan sejumlah "
        "kebijakan untuk mengurai kemacetan, termasuk rekayasa lalu lintas, penambahan armada "
        "transportasi umum, dan perluasan rest area di jalan tol. Puncak arus mudik diperkirakan "
        "terjadi pada H-2 Hari Raya. Sebanyak 12 juta kendaraan diprediksi memadati tol trans-Jawa. "
        "Kemenhub mengimbau masyarakat berangkat lebih awal dan memanfaatkan program mudik gratis.",
        False,
    ),
    "Dialek Jawa": (
        "Kulo badhe lapor bilih dalan teng ngajeng griyo kulo sampun rusak sanget. "
        "Sampun dangu mboten wonten ingkang ndandosi. Menawi jawah, banyu saking got "
        "malimpah dumugi griyo-griyo warga. Kulo nyuwun supados Dinas PU enggal ndandosi "
        "supados warga mboten kesusahan malih.",
        True,
    ),
    "Dialek Minangkabau": (
        "Ambo malaporan jalan di muko rumah ambo rusak bana, alah lamo indak dibeton. "
        "Urang nan lewat di jalan ko sering kacilakaan motonyo dek jalan nan balubuang gadang. "
        "Ambo harok supayo pemerintah daerah capek mambuek jalan tu elok baliak.",
        True,
    ),
}

with st.expander("Contoh teks"):
    for label, (sample_text, sample_norm) in EXAMPLES.items():
        if st.button(label, key=f"ex_{label}"):
            st.session_state["_example_text"] = sample_text
            st.session_state["_example_norm"] = sample_norm
            st.rerun()

# ── Tombol Ringkas ────────────────────────────────────────────────────────────
run_btn = st.button("Ringkas", type="primary", use_container_width=False)

# ── Aksi: Ringkas ─────────────────────────────────────────────────────────────
if run_btn and input_text.strip():
    text_clean = input_text.strip()

    _total_steps = (
        2  # NER + TextRank selalu jalan
        + len(selected_abstractive)
        + (1 if use_normalizer and NORM_DIR.exists() else 0)
    )
    _step   = [0]
    _status = st.empty()
    _bar    = st.progress(0)

    def _tick(label: str) -> None:
        _status.caption(label)
        _bar.progress(_step[0] / _total_steps)
        _step[0] += 1

    # Normalisasi (timing terpisah, tidak digabung ke elapsed model)
    norm_result = None
    if use_normalizer and NORM_DIR.exists():
        _tick("Menormalkan teks dialek...")
        _cached = st.session_state.get("_last_norm")
        if _cached and _cached.get("source") == text_clean and _cached.get("qc_passed"):
            norm_result = {**_cached, "elapsed_ms_norm": _cached.get("elapsed_ms_norm", 0)}
        else:
            norm_result = _do_normalize(text_clean)
            st.session_state["_last_norm"] = {"source": text_clean, **norm_result}
    text_to_sum = norm_result["text"] if norm_result else text_clean

    _tick("Menganalisis entitas (NER)...")
    ner_sum, entities = summarize_ner(text_to_sum)

    _tick("Menghitung TextRank...")
    tr_sum = summarize_textrank(text_to_sum)

    abstractive_results: dict = {}
    for _name in selected_abstractive:
        _tick(f"Meringkas dengan {_name}...")
        abstractive_results[_name] = _do_summarize(text_to_sum, _name)

    _bar.progress(1.0)
    _status.empty()
    _bar.empty()

    st.session_state["_abstractive_results"] = abstractive_results
    st.session_state["_ner_sum"]             = ner_sum
    st.session_state["_tr_sum"]              = tr_sum
    st.session_state["_entities"]            = entities
    st.session_state["_result_src"]          = text_clean
    st.session_state["_norm_result"]         = norm_result
    st.session_state["_text_to_sum"]         = text_to_sum

elif run_btn:
    st.warning("Masukkan teks terlebih dahulu.")

# ── Tampilkan hasil ──────────────────────────────────────────────────────────
if (
    "_abstractive_results" in st.session_state
    and st.session_state.get("_result_src") == input_text.strip()
):
    _norm_result         = st.session_state["_norm_result"]
    _abstractive_results = st.session_state["_abstractive_results"]
    _ner_sum             = st.session_state["_ner_sum"]
    _tr_sum              = st.session_state["_tr_sum"]
    _entities            = st.session_state["_entities"]
    _src_text            = st.session_state["_result_src"]

    # ── Block A: Normalisasi ──────────────────────────────────────────────
    if _norm_result is not None:
        _qc_label = "lolos QC" if _norm_result["qc_passed"] else "QC gagal"
        with st.expander(f"Normalisasi Dialek [{_qc_label}]", expanded=True):
            if not _norm_result["qc_passed"]:
                st.warning(
                    f"Output normalizer terlalu pendek dibanding input "
                    f"(raw: \"{_norm_result['raw_output'][:120]}...\"). "
                    "Teks asli digunakan untuk summarisasi."
                )
            _c_orig, _c_norm = st.columns(2)
            with _c_orig:
                st.markdown("**Teks asli:**")
                st.write(_src_text)
            with _c_norm:
                st.markdown("**Teks ternormalisasi:**")
                st.write(_norm_result["text"])
            st.caption(f"Waktu normalisasi: {_norm_result.get('elapsed_ms_norm', 0)} ms")

    # ── Block B: Ringkasan Abstraktif ─────────────────────────────────────
    if _abstractive_results:
        st.divider()
        st.subheader("Ringkasan Abstraktif")
        for _name, _res in _abstractive_results.items():
            st.markdown(f"**{_res['model_used']}**")
            st.success(_res["summary"])
            _c1, _c2, _c3, _c4 = st.columns(4)
            _c1.metric("Kata input",     len(_src_text.split()))
            _c2.metric("Kata ringkasan", len(_res["summary"].split()))
            _c3.metric("Compression",    _res["compression_ratio"])
            _c4.metric("Kecepatan",      f"{_res['words_per_sec']} kata/s")
            st.caption(
                f"Waktu inferensi: {_res['elapsed_ms']} ms  |  Device: **{DEVICE}**"
            )

    # ── Block C: Entitas ──────────────────────────────────────────────────
    st.divider()
    st.subheader("Entitas yang Ditemukan")
    if _entities:
        _grouped: dict = {}
        for _e in _entities:
            _grouped.setdefault(_e["type"], []).append(_e["word"])
        for _ent_type, _words in _grouped.items():
            _label = _ENTITY_LABELS.get(_ent_type, _ent_type)
            st.markdown(f"**{_label}**: {', '.join(w.title() for w in _words)}")
    else:
        st.caption("Tidak ada entitas yang terdeteksi.")

    # ── Block D: Ringkasan Ekstraktif ─────────────────────────────────────
    st.divider()
    st.subheader("Ringkasan Ekstraktif")
    _col_ner, _col_tr = st.columns(2)
    with _col_ner:
        st.markdown(f"**NER** — {len(_ner_sum.split())} kata")
        st.info(_ner_sum)
    with _col_tr:
        st.markdown(f"**TextRank** — {len(_tr_sum.split())} kata")
        st.info(_tr_sum)
