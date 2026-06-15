#!/usr/bin/env bash
set -e

echo "Starting FastAPI on $API_HOST:$API_PORT..."
uvicorn api.main:app --host "$API_HOST" --port "$API_PORT" &
API_PID=$!

for i in $(seq 1 60); do
    if curl -sf "http://$API_HOST:$API_PORT/health" > /dev/null; then
        echo "FastAPI is up."
        break
    fi
    sleep 2
done

echo "Starting Streamlit on 0.0.0.0:7860..."
# XSRF/CORS disabled because Streamlit runs inside the HF Spaces iframe and the
# default protections block file uploads (causing 403 on the predict endpoint).
streamlit run dashboard/app.py \
    --server.address 0.0.0.0 \
    --server.port 7860 \
    --server.enableXsrfProtection false \
    --server.enableCORS false \
    --server.maxUploadSize 10 \
    --browser.gatherUsageStats false

kill $API_PID
