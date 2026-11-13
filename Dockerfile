# Pin by digest in your registry for a verifiable build: `docker inspect python:3.12-slim` -> "-sha256:…"
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system app && useradd --system --no-create-home --gid app app
WORKDIR /app
COPY --chown=app:app pyproject.toml README.md ./
COPY --chown=app:app autoclip ./autoclip
RUN pip install --no-cache-dir ".[server]" && rm -rf /root/.cache

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s \
    CMD python -c "import urllib.request, sys; urllib.request.urlopen('http://127.0.0.1:' + __import__('os').environ.get('PORT', '8000') + '/view/', timeout=5); sys.exit(0)" || exit 1
CMD ["python", "-m", "autoclip.server"]