FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENV API_HOST=127.0.0.1 \
    API_PORT=8000 \
    LOG_LEVEL=INFO

EXPOSE 7860

RUN chmod +x start.sh
CMD ["./start.sh"]
