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

## Deploy to Render (Free Tier)

The app needs FFmpeg + a persistent process (not serverless), so it's deployed via Docker:

1. Push this repo to GitHub.
2. In the Render dashboard: **New > Blueprint**, point it at the repo — it reads `render.yaml` and provisions the web service on the **Free** plan automatically.
3. `SECRET_KEY` is auto-generated by the blueprint; add `ELEVENLABS_API_KEY` manually in the service's Environment tab only if you want the ElevenLabs TTS fallback.
4. Deploy. First build takes a few minutes (installs ffmpeg + Python deps).

Free tier trade-offs (accepted by design here):

- **No persistent disk.** Uploads, generated reels, `data/*.csv`, and the trained model all live on the container's local disk — every restart or redeploy wipes them, so the gallery goes back to empty. Fine for a demo/degree submission; not fine for real long-term users.
- **No Postgres.** `render.yaml` doesn't provision one — `services/db_service.py` already treats it as an optional, best-effort mirror and logs a warning instead of failing when it's unreachable.
- **Cold starts.** Free web services spin down after ~15 minutes idle; the next visit takes 30–60s to wake back up.
- To run the same image locally: `docker build -t dreel-ai . && docker run -p 4600:4600 --env-file .env dreel-ai`.
- If you later want reels/model to survive restarts, upgrade the web service to the **Starter** plan and attach a persistent disk mounted at, say, `/var/data`, then set the `PERSIST_DIR` env var to that path (`config.py` already reads it — see `.env.example`).

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
