FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl git build-essential libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt* ./
RUN pip install --no-cache-dir fastapi uvicorn chromadb pydantic apscheduler 2>/dev/null || true

COPY godlocal_v5.py godlocal_v5_modules.py sleep_scheduler.py ./
COPY god_soul.example.md ./

RUN mkdir -p godlocal_data/souls godlocal_data/memory godlocal_data/logs godlocal_data/outputs

EXPOSE 8000

CMD ["python", "godlocal_v5.py"]
