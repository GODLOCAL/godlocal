# GodLocal Backend — Picobot VPS ($5/mo, no Mac needed)
# docker build -t godlocal . && docker run -p 8000:8000 godlocal

FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App source
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Entrypoint
CMD ["python", "godlocal_v6.py", "--host", "0.0.0.0", "--port", "8000"]
