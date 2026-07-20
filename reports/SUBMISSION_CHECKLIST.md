# Degree Submission Checklist — DReel AI v2

## Code & Application

- [ ] `python app.py` runs without errors
- [ ] FFmpeg installed (`ffmpeg -version`)
- [ ] Virtual env + `pip install -r requirements.txt`
- [ ] Create reel end-to-end (images + audio)
- [ ] Gallery shows video + predicted engagement
- [ ] Star rating saves and retrains model
- [ ] Analytics page shows KPIs + 6 charts
- [ ] Table on analytics is readable

## Data Science Deliverables

- [ ] `data/audio_features.csv` has rows after creating reels
- [ ] `data/image_features.csv` has rows after creating reels
- [ ] `data/engagement_ratings.csv` has user ratings
- [ ] `models/engagement_model.pkl` exists
- [ ] `reports/model_metrics.json` exists

## Notebooks (run all 4)

- [ ] `notebooks/01_audio_eda.ipynb`
- [ ] `notebooks/02_image_eda.ipynb`
- [ ] `notebooks/03_pipeline_evaluation.ipynb`
- [ ] `notebooks/04_engagement_model.ipynb`

## Written Report

- [ ] Fill in name, university, date in `reports/methodology.md`
- [ ] Update Results section with latest metrics
- [ ] Export `methodology.md` to PDF for submission
- [ ] Add screenshots (gallery, analytics, notebook outputs)

## Presentation

- [ ] 2–3 minute demo video recorded
- [ ] Architecture diagram included in report/slides
- [ ] Prepare viva answers: librosa, Laplacian blur, Random Forest

## Optional Improvements

- [ ] Collect 15–20 ratings from classmates
- [ ] Add `.env` with ElevenLabs key (optional)
- [ ] GitHub repo pushed with clean README

---

**Project title for cover page:**

*DReel AI: An Intelligent Short-Form Video Pipeline Using Audio Signal Processing, Computer Vision, and Machine Learning*
