# DReel AI v2 — Degree Project Report

**Title:** An Intelligent Short-Form Video Pipeline Using Audio Signal Processing, Computer Vision, and Machine Learning

**Author:** _[Your Name]_  
**Institution:** _[Your University]_  
**Program:** Data Science  
**Date:** _[Submission Date]_

---

## 1. Abstract

DReel AI v2 is a web-based system that automates short-form video (reel) creation from user-uploaded images and audio. Unlike simple slideshow tools that use fixed per-image timing, this project applies **audio signal processing** (beat detection via librosa), **computer vision** (image quality scoring), and **machine learning** (Random Forest engagement prediction) to produce smarter, data-driven video output. The system logs structured datasets from every reel, provides an analytics dashboard, and supports user ratings to improve predictive models.

---

## 2. Introduction

### 2.1 Problem Statement

Content creators need fast ways to turn photo collections into engaging vertical videos. Manual editing is time-consuming. Existing automated tools often use fixed slide durations (e.g. 1 second per image), which do not align with music rhythm and ignore image quality differences.

### 2.2 Objectives

1. Detect audio tempo and beats to assign **variable slide durations** (beat-sync).
2. Score images on **blur, brightness, and contrast** and auto-rank them by quality.
3. Build a **dataset pipeline** logging audio and vision features per reel.
4. Train an **engagement prediction model** from combined features and user ratings.
5. Provide a **web interface** and **analytics dashboard** for demonstration and EDA.

### 2.3 Scope

- Input: Multiple images (JPG/PNG) + text (TTS) or uploaded audio (MP3/WAV)
- Output: MP4 reel (mobile 9:16 or desktop 16:9)
- Platform: Flask web app, local deployment

---

## 3. Literature & Background

| Area | Method Used |
|------|-------------|
| Beat tracking | librosa `beat_track` + onset strength envelope |
| Blur detection | Laplacian variance on grayscale images |
| Quality scoring | Weighted combination of blur, brightness, contrast |
| Engagement prediction | Random Forest Regressor (scikit-learn) |

---

## 4. System Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Upload    │────▶│  Vision Service  │────▶│  Rank images    │
│   Images    │     │  (CV scoring)    │     │  by quality     │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                        │
┌─────────────┐     ┌──────────────────┐               ▼
│ Audio/TTS   │────▶│  Beat Service    │────▶ FFmpeg reel (.mp4)
└─────────────┘     │  (librosa)       │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  CSV datasets    │
                    │  + ML model      │
                    └──────────────────┘
```

### 4.1 Modules

| Module | File | Role |
|--------|------|------|
| Beat sync | `services/beat_service.py` | Tempo, beats, slide durations |
| Vision | `services/vision_service.py` | Blur, brightness, contrast, quality score |
| Reel worker | `services/reel_service.py` | Background processing + FFmpeg |
| Analytics | `services/analytics_service.py` | Dashboard charts |
| ML | `services/ml_service.py` | Feature matrix, training, prediction |

---

## 5. Methodology

### 5.1 Audio Signal Processing

1. Load audio with librosa.
2. Compute onset strength envelope.
3. Detect tempo (BPM) and beat frames.
4. Allocate slide durations across detected beat boundaries.
5. Log features to `data/audio_features.csv`.

**Features logged:** duration_sec, tempo_bpm, beat_count, avg_energy, avg_slide_duration, slide_durations

### 5.2 Computer Vision

For each uploaded image:

1. Convert to grayscale (downscaled for speed).
2. **Blur:** Laplacian variance (higher = sharper).
3. **Brightness:** Mean pixel intensity.
4. **Contrast:** Standard deviation of pixels.
5. **Quality score:** Weighted combination (0–1).
6. Auto-sort images: highest quality first.

**Features logged:** blur_variance, brightness, contrast, quality_score, quality_label

### 5.3 Machine Learning

**Algorithm:** Random Forest Regressor  
**Target variable:** Engagement rating (1–5)  
**Feature vector (11 dimensions):**

- image_count, duration_sec, tempo_bpm, beat_count
- avg_energy, avg_slide_duration, slide_std
- avg_quality, min_quality, blur_mean, low_quality_ratio

**Training data:**
- Real user ratings from Gallery (`data/engagement_ratings.csv`)
- Proxy labels for unrated reels (bootstrap until more ratings collected)

**Evaluation metrics:** R², MAE, feature importance (see `reports/model_metrics.json`)

---

## 6. Datasets

| File | Rows | Description |
|------|------|-------------|
| `data/audio_features.csv` | Per reel | Audio/beat-sync features |
| `data/image_features.csv` | Per image | Vision quality features |
| `data/engagement_ratings.csv` | Per reel | User ratings 1–5 |

---

## 7. Results

### 7.1 Model Performance

_Update after collecting more ratings. Re-run `services/ml_service.train_engagement_model()`
(or rate a reel in the Gallery) and copy the latest `reports/model_metrics.json` here._

| Metric | Value |
|--------|-------|
| Training samples | 4 |
| Real user ratings | 3 |
| Train R² (in-sample) | 0.89 |
| Train MAE (in-sample) | 0.23 |
| CV R² (2-fold, mean ± std) | **-2.88 ± 1.11** |

**Important — read this before quoting R² in the viva:** Train R² (0.89) is
computed on the same rows the model was fit on, so it is optimistic by
construction and is *not* a measure of generalization. The cross-validated R²
(-2.88) is the honest estimate, and it is strongly negative — with only 4
samples the model is clearly overfitting the training set and does not yet
generalize. This is expected and worth stating directly: it is evidence *for*
the "Limitations" and "Future Work" sections below (collect more ratings
before trusting the model), not a result to hide. A professor is more likely
to be impressed by recognizing and explaining this than by a single
cherry-picked train R².

### 7.2 Top Feature Importance

| Feature | Importance |
|---------|------------|
| slide_std | 0.35 |
| beat_count | 0.13 |
| duration_sec | 0.13 |
| image_count | 0.11 |

### 7.3 Beat-Sync vs Fixed Timing

Notebook `03_pipeline_evaluation.ipynb` compares:
- **Fixed:** 1 second per image
- **Beat-sync:** Variable durations aligned to beats

Beat-sync reduces the gap between total slide time and audio duration.

---

## 8. Web Application

| Route | Function |
|-------|----------|
| `/` | Landing page |
| `/create` | Upload workflow |
| `/gallery` | View reels, rate, see predictions |
| `/analytics` | DS dashboard with KPIs and charts |

---

## 9. Limitations

1. Small training set — more user ratings needed for robust ML.
2. Beat detection accuracy depends on audio genre and quality.
3. Vision scoring is rule-based, not deep learning.
4. FFmpeg and TTS require external dependencies / network (edge-tts).

---

## 10. Future Work

- Collect 50+ user ratings for proper train/test split
- Replace proxy labels with only real ratings
- CNN-based scene classification (ResNet/MobileNet)
- Deep learning engagement model
- Cloud deployment (Docker + Gunicorn)

---

## 11. Conclusion

DReel AI v2 demonstrates a complete data science pipeline: **signal processing** for audio beat-sync, **computer vision** for image quality, **structured data logging**, **exploratory analysis** via Jupyter notebooks, and **machine learning** for engagement prediction — all integrated into a functional Flask web application suitable for short-form video automation.

---

## 12. References

1. McFee, B. et al. librosa: Audio and Music Signal Analysis in Python.
2. Pedregosa, F. et al. scikit-learn: Machine Learning in Python.
3. FFmpeg Project — https://ffmpeg.org
4. Flask Documentation — https://flask.palletsprojects.com

---

## Appendix A — How to Reproduce

```bash
pip install -r requirements.txt
python app.py
# Create reels at http://localhost:4600/create
# View analytics at http://localhost:4600/analytics
jupyter notebook notebooks/
```

## Appendix B — Demo Script (Viva)

1. Show home page and explain project goals.
2. Upload 5 images + audio on `/create`.
3. Show gallery reel with predicted engagement score.
4. Rate a reel (star click) — explain model retraining.
5. Open `/analytics` — walk through charts.
6. Open Notebook 04 — show feature importance plot.
