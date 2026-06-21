FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    WEB_CONCURRENCY=1 \
    GUNICORN_THREADS=4 \
    GUNICORN_TIMEOUT=180 \
    HF_HOME=/data/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.4.1+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# ── Pre-download model kecil saat build ──────────────────────────────────────
# NER (440 MB) di-bake sebagai folder lokal /app/models/ner_cahya.
# Jangan simpan ke HF_HOME build cache, karena runtime /data persistent storage
# akan menutupi cache image. Folder lokal tetap terbaca oleh transformers.
# NLLB (1.2 GB) dan mT5 (2.2 GB) terlalu besar → download ke /data saat runtime.
# Kalau network tidak tersedia saat build, ini di-skip (|| true).
RUN python -c "from huggingface_hub import snapshot_download; print('Pre-downloading cahya/bert-base-indonesian-NER to models/ner_cahya...'); snapshot_download(repo_id='cahya/bert-base-indonesian-NER', local_dir='models/ner_cahya', local_dir_use_symlinks=False); print('Done.')" \
    || echo "[WARN] Pre-download NER gagal — akan download saat runtime."

# Collectstatic
RUN SECRET_KEY=build-only DEBUG=True DATABASE_URL=sqlite:///tmp.db NLP_ENABLED=false \
    python manage.py collectstatic --noinput

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
    CMD curl -fsS http://127.0.0.1:${PORT}/health/ || exit 1

# ── Boot ──────────────────────────────────────────────────────────────────────
# TANPA --preload: setiap worker load Django sendiri → apps.ready() dipanggil
# di worker → warmup thread jalan di dalam worker itu sendiri.
# Ini fix race condition dimana --preload fork worker sebelum thread selesai load.
# WEB_CONCURRENCY=1 jadi overhead minimal, satu worker cukup untuk free tier.
CMD HF_HOME=/data/huggingface \
    python manage.py migrate --noinput && \
    gunicorn config.wsgi:application \
    --bind 0.0.0.0:${PORT} \
    --workers ${WEB_CONCURRENCY} \
    --threads ${GUNICORN_THREADS} \
    --timeout ${GUNICORN_TIMEOUT} \
    --worker-tmp-dir /dev/shm \
    --access-logfile -
