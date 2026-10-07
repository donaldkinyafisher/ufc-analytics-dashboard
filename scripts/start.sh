#!/bin/sh
# Container entrypoint: start the private API, wait for it, then serve Streamlit on Render's public port.
set -e

uvicorn src.api.v1.main:app --host 127.0.0.1 --port 8000 &

python - <<'EOF'
import time
import urllib.request

for _ in range(60):
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
        break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit("The API did not start within 60 seconds.")
EOF

exec streamlit run src/streamlit/main.py \
    --server.port "${PORT:-8501}" \
    --server.address 127.0.0.1 \
    --server.headless true \
    --browser.gatherUsageStats false
