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

CMD ["sh", "scripts/start.sh"]
