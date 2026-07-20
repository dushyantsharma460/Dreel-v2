FROM python:3.11-slim

# ffmpeg: video rendering pipeline. libsndfile1: required by soundfile/librosa.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 4600

# Single worker process (the app relies on exactly one in-process reel
# worker thread, started at import time in app.py) with several threads
# for concurrent request handling.
CMD ["sh", "-c", "gunicorn --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT:-4600} app:app"]
