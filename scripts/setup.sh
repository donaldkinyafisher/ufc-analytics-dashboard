#!/usr/bin/env bash
# One-time local setup: install dependencies, the scraping browser and a default .env.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. Install it, then run this script again:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "See https://docs.astral.sh/uv/ for other install options."
    exit 1
fi

echo "==> Installing Python dependencies"
uv sync

echo "==> Installing Chromium for scraping ufcstats.com"
uv run playwright install chromium

if [ ! -f .env ]; then
    echo "==> Creating .env from .env.example"
    cp .env.example .env
else
    echo "==> Keeping existing .env"
fi

cat <<'EOF'

Setup complete. Start the API and the app in separate terminals, API first:

  uv run fastapi dev --entrypoint src.api.v1.main:app
  uv run streamlit run src/streamlit/main.py

On first run the app opens on the Setup page to build the database.
EOF
