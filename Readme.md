# DReel AI v2

**An intelligent short-form video pipeline using audio signal processing, computer vision, and machine learning for automated reel generation.**

Data Science Degree Project — transforms uploaded photos into vertical/landscape video reels with beat-synced timing, image quality ranking, analytics dashboards, and engagement prediction.

---

## Features

| Feature | Technology | Description |
|---------|------------|-------------|
| Beat-sync timing | librosa | Slide durations aligned to detected audio beats |
| Image quality scoring | Pillow + NumPy | Blur, brightness, contrast analysis; auto-ranking |
| AI narration | edge-tts (Indian voices) / ElevenLabs fallback | Text-to-speech for reel narration |
| Video output | FFmpeg | Mobile (9:16) or desktop (16:9) formats |
| Analytics dashboard | pandas + matplotlib | KPIs and charts from collected datasets |
| Engagement prediction | scikit-learn | Random Forest model predicts 1–5 engagement score |
| User ratings | Flask | Gallery star ratings retrain the ML model |

---

## Tech Stack

- **Backend:** Python 3, Flask, Jinja2
- **Audio DS:** librosa, numpy, soundfile
- **Vision:** Pillow, NumPy (Laplacian blur detection)
- **ML:** scikit-learn (Random Forest Regressor)
- **Analytics:** pandas, matplotlib, seaborn
- **Video:** FFmpeg (external)
- **Frontend:** Bootstrap 5, Font Awesome

---

## Prerequisites

1. **Python 3.10+**
2. **FFmpeg** — must be installed and available on `PATH`

### Install FFmpeg

**Windows (winget):**
```powershell
winget install FFmpeg
```

**Ubuntu:**
```bash
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

Verify:
```bash
ffmpeg -version
```

---

## Installation

```bash
git clone <your-repo-url>
cd dreel-ai-v2

python -m venv venv

# Windows
.\venv\Scripts\activate
.\venv\Scripts\pip install -r requirements.txt

# macOS / Linux
source venv/bin/activate
pip install -r requirements.txt
```

### Environment variables (optional)

By default, narration uses **edge-tts** (free) with a random Indian voice per reel — English (`en-IN-NeerjaNeural`, `en-IN-NeerjaExpressiveNeural`, `en-IN-PrabhatNeural`) or Hindi (`hi-IN-SwaraNeural`, `hi-IN-MadhurNeural`). Copy `.env.example` to `.env` and add an ElevenLabs API key only if you want it as a fallback — its available voices depend on your account and are not guaranteed to have an Indian accent.

```bash
cp .env.example .env
```

---

## Run the Application

```bash
python app.py
```

Open in browser: **http://localhost:4600**

| Page | URL | Purpose |
|------|-----|---------|
| Home | `/` | Landing page |
| Create | `/create` | Upload images + audio |
| Gallery | `/gallery` | View reels, rate, see ML predictions |
| Analytics | `/analytics` | Data science dashboard |

---

## How It Works

```
Upload images + audio/text
        ↓
Vision analysis → rank images by quality
        ↓
TTS or uploaded audio → audio.mp3
        ↓
Beat detection (librosa) → per-slide durations
        ↓
FFmpeg → final .mp4 reel
        ↓
Log features → data/audio_features.csv, data/image_features.csv
        ↓
ML model → predict engagement score
```

---

## Project Structure

```
dreel-ai-v2/
├── app.py                      # Flask routes
├── config.py                   # Paths and settings
├── requirements.txt
├── services/
│   ├── audio_service.py        # TTS (edge-tts Indian voices + ElevenLabs fallback)
│   ├── beat_service.py         # Audio beat-sync (librosa)
│   ├── vision_service.py       # Image quality (CV)
│   ├── reel_service.py         # FFmpeg pipeline + worker
│   ├── analytics_service.py    # Dashboard charts
│   ├── image_service.py        # Upload handling
│   └── ml_service.py           # Engagement prediction
├── templates/                  # Jinja2 HTML
├── static/
│   ├── css/
│   ├── reels/                  # Generated videos
│   └── reports/charts/         # Analytics chart images
├── data/
│   ├── audio_features.csv      # Audio DS dataset
│   ├── image_features.csv      # Vision DS dataset
│   └── engagement_ratings.csv  # User ratings
├── models/
│   └── engagement_model.pkl    # Trained Random Forest
├── reports/
│   ├── model_metrics.json      # ML evaluation metrics
│   └── methodology.md          # Degree project report
├── notebooks/
│   ├── 01_audio_eda.ipynb
│   ├── 02_image_eda.ipynb
│   ├── 03_pipeline_evaluation.ipynb
│   └── 04_engagement_model.ipynb
└── uploads/                    # Per-job upload folders
```

---

## Jupyter Notebooks

Run from project root:

```bash
jupyter notebook notebooks/
```

| Notebook | Topic |
|----------|-------|
| `01_audio_eda.ipynb` | Tempo, energy, beat-sync EDA |
| `02_image_eda.ipynb` | Blur, brightness, quality analysis |
| `03_pipeline_evaluation.ipynb` | Beat-sync vs fixed timing comparison |
| `04_engagement_model.ipynb` | Random Forest training + feature importance |

---

## ML Model

- **Algorithm:** Random Forest Regressor
- **Target:** Engagement score (1–5)
- **Features:** 11 (audio + vision metrics per reel)
- **Metrics:** See `reports/model_metrics.json`
- **Retrain:** Rate any reel in the Gallery — model retrains automatically

---

## Deploy (Docker Compose + Cloudflare Tunnel)

The app needs FFmpeg, a persistent background worker thread, and real memory
headroom for librosa/matplotlib/pandas/scikit-learn — a 512MB serverless-style
free host (e.g. Render's free tier) reliably OOM-kills the process during
video encoding, even for small reels. `docker-compose.yml` instead runs the
app alongside a real Postgres container on any machine with normal desktop
RAM (a spare PC, a home server, or a cloud VM), with named volumes so
uploads/reels/datasets/model and the database survive restarts.

**Run it:**

```bash
cp .env.example .env
# fill in SECRET_KEY and POSTGRES_PASSWORD (see the generator commands in .env.example)
docker compose up -d --build
```

The app is now served on `http://localhost:80`.

**Expose it publicly (free, no account needed):**

```bash
cloudflared tunnel --url http://localhost:80
```

This prints a public `https://<random-words>.trycloudflare.com` URL that
proxies to your machine. It's a *quick tunnel* — free, no signup, but the URL
changes every time you restart it, and the site is only reachable while the
host machine, `docker compose`, and the tunnel are all running. For a stable
custom domain instead, use a [named Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps)
(free Cloudflare account + a domain).

To stop everything: `docker compose down` (add `-v` to also wipe the
volumes/data).

---

## Degree Submission Checklist

See `reports/SUBMISSION_CHECKLIST.md` for the full list.

- [ ] App runs: `python app.py`
- [ ] Create a reel end-to-end
- [ ] Analytics page shows charts
- [ ] Gallery rating + prediction works
- [ ] All 4 notebooks execute
- [ ] Report: `reports/methodology.md` → export to PDF
- [ ] Collect 15+ user ratings for stronger ML evaluation
- [ ] Record 2–3 min demo video

---

## License

MIT License
