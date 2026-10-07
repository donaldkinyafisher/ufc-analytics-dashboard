# Read-only snapshot of the app for Render: the API (private) and Streamlit (public) in one container.
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.25 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_CACHE=1 UV_PYTHON_DOWNLOADS=never

# Dependencies first, so code and data pushes reuse this layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --no-install-project

# The committed database and src/ml/artifacts land at the paths the app already uses.
COPY ufc_analytics.db ./
COPY src ./src
COPY scripts/start.sh ./scripts/start.sh

# No Chromium: scraping, syncing and training stay on the local copy.
ENV READ_ONLY=true PATH=/app/.venv/bin:$PATH

# Public Streamlit endpoint; publish with `-p 8501:8501`.
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)" || exit 1

CMD ["sh", "scripts/start.sh"]
