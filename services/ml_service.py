"""
ML Service — Engagement Prediction

Builds a per-reel feature matrix from audio + vision CSVs,
collects user ratings, and trains a Random Forest regressor
to predict engagement scores (1–5).
"""

import csv
import json
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

from config import DATA_FOLDER, MODELS_FOLDER, REPORTS_FOLDER
from services.analytics_service import load_audio_features, load_image_features
from services.db_service import mirror_rating

RATINGS_CSV = os.path.join(DATA_FOLDER, "engagement_ratings.csv")
MODEL_PATH = os.path.join(MODELS_FOLDER, "engagement_model.pkl")
METRICS_PATH = os.path.join(REPORTS_FOLDER, "model_metrics.json")

MIN_SAMPLES_FOR_CV = 4

FEATURE_COLUMNS = [
    "image_count",
    "duration_sec",
    "tempo_bpm",
    "beat_count",
    "avg_energy",
    "avg_slide_duration",
    "slide_std",
    "avg_quality",
    "min_quality",
    "blur_mean",
    "low_quality_ratio",
]


def _parse_slide_durations(value: str) -> list[float]:
    return [float(part) for part in str(value).split("|") if part]


def build_feature_matrix() -> pd.DataFrame:
    """Merge audio and vision features into one row per reel."""

    audio_df = load_audio_features()
    image_df = load_image_features()

    if audio_df.empty:
        return pd.DataFrame(columns=["reel_id", *FEATURE_COLUMNS])

    rows = []

    for _, audio_row in audio_df.iterrows():
        reel_id = audio_row["reel_id"]
        reel_images = image_df[image_df["reel_id"] == reel_id]

        slide_list = _parse_slide_durations(audio_row.get("slide_durations", ""))
        slide_std = float(np.std(slide_list)) if slide_list else 0.0

        if reel_images.empty:
            avg_quality = 0.5
            min_quality = 0.5
            blur_mean = 0.0
            low_quality_ratio = 0.0
        else:
            avg_quality = float(reel_images["quality_score"].mean())
            min_quality = float(reel_images["quality_score"].min())
            blur_mean = float(reel_images["blur_variance"].mean())
            low_quality_ratio = float((reel_images["quality_label"] == "poor").mean())

        rows.append({
            "reel_id": reel_id,
            "image_count": int(audio_row["image_count"]),
            "duration_sec": float(audio_row["duration_sec"]),
            "tempo_bpm": float(audio_row["tempo_bpm"]),
            "beat_count": int(audio_row["beat_count"]),
            "avg_energy": float(audio_row["avg_energy"]),
            "avg_slide_duration": float(audio_row["avg_slide_duration"]),
            "slide_std": slide_std,
            "avg_quality": avg_quality,
            "min_quality": min_quality,
            "blur_mean": blur_mean,
            "low_quality_ratio": low_quality_ratio,
        })

    return pd.DataFrame(rows)


def load_ratings() -> pd.DataFrame:
    """Load user engagement ratings (1–5)."""

    if not os.path.isfile(RATINGS_CSV):
        return pd.DataFrame(columns=["timestamp", "reel_id", "rating", "source"])

    ratings = pd.read_csv(RATINGS_CSV)
    if ratings.empty:
        return ratings

    return ratings.drop_duplicates(subset=["reel_id"], keep="last")


def get_rating_map() -> dict[str, int]:
    return {
        row["reel_id"]: int(row["rating"])
        for _, row in load_ratings().iterrows()
    }


def save_rating(reel_id: str, rating: int, source: str = "user") -> None:
    """Append or update a user rating for a reel."""

    if not 1 <= rating <= 5:
        raise ValueError("Rating must be between 1 and 5.")

    os.makedirs(DATA_FOLDER, exist_ok=True)

    ratings = load_ratings()
    ratings = ratings[ratings["reel_id"] != reel_id]

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reel_id": reel_id,
        "rating": rating,
        "source": source,
    }

    updated = pd.concat([ratings, pd.DataFrame([row])], ignore_index=True)
    updated.to_csv(RATINGS_CSV, index=False)

    mirror_rating(row)


def remove_rating(reel_id: str) -> None:
    """Drop this reel's row from the ratings CSV, if present."""

    if not os.path.isfile(RATINGS_CSV):
        return

    ratings = load_ratings()
    ratings = ratings[ratings["reel_id"] != reel_id]
    ratings.to_csv(RATINGS_CSV, index=False)


def _proxy_engagement_score(features: pd.Series) -> float:
    """
    Bootstrap label for reels without user ratings.
    Used only until enough real ratings are collected.
    """

    quality = features["avg_quality"]
    energy = min(features["avg_energy"] / 0.2, 1.0)
    dynamics = min(features["slide_std"] / 2.0, 1.0)
    sharpness = min(features["blur_mean"] / 1500.0, 1.0)

    score = 1.0 + 4.0 * (0.45 * quality + 0.2 * energy + 0.2 * dynamics + 0.15 * sharpness)
    return round(float(np.clip(score, 1.0, 5.0)), 2)


def build_training_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Returns features X, labels y, and a boolean mask for real user ratings.
    """

    features_df = build_feature_matrix()
    if features_df.empty:
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=bool)

    ratings = load_ratings()
    rating_map = get_rating_map()
    is_real = []

    labels = []
    for _, row in features_df.iterrows():
        reel_id = row["reel_id"]
        if reel_id in rating_map:
            labels.append(float(rating_map[reel_id]))
            is_real.append(True)
        else:
            labels.append(_proxy_engagement_score(row))
            is_real.append(False)

    x_data = features_df[FEATURE_COLUMNS]
    y_data = pd.Series(labels, index=features_df.index)
    real_mask = pd.Series(is_real, index=features_df.index)

    return x_data, y_data, real_mask


def train_engagement_model() -> dict:
    """Train Random Forest and persist model + metrics."""

    os.makedirs(MODELS_FOLDER, exist_ok=True)
    os.makedirs(REPORTS_FOLDER, exist_ok=True)

    features_df = build_feature_matrix()
    x_data, y_data, real_mask = build_training_data()

    if features_df.empty or len(x_data) < 2:
        return {"trained": False, "reason": "Not enough reel data to train."}

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=5,
        random_state=42,
        min_samples_leaf=1,
    )

    metrics = {
        "trained": True,
        "samples": int(len(x_data)),
        "real_ratings": int(real_mask.sum()),
        "proxy_ratings": int((~real_mask).sum()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if len(x_data) >= MIN_SAMPLES_FOR_CV:
        # Keep >=2 samples per test fold — with 1 sample per fold, R^2 is
        # mathematically undefined (no variance to compare against) and
        # sklearn silently returns NaN instead of a usable score.
        cv_folds = min(5, len(x_data) // 2)
        cv_scores = cross_val_score(model, x_data, y_data, cv=cv_folds, scoring="r2")
        metrics["cv_folds"] = cv_folds
        metrics["cv_r2_mean"] = round(float(cv_scores.mean()), 4)
        metrics["cv_r2_std"] = round(float(cv_scores.std()), 4)
    else:
        metrics["cv_folds"] = None
        metrics["cv_r2_mean"] = None
        metrics["cv_r2_std"] = None

    model.fit(x_data, y_data)
    predictions = model.predict(x_data)

    metrics["train_r2"] = round(float(r2_score(y_data, predictions)), 4)
    metrics["train_mae"] = round(float(mean_absolute_error(y_data, predictions)), 4)
    metrics["metrics_note"] = (
        "train_r2/train_mae are in-sample fit on the training data itself, not a "
        "held-out score, and will look overly optimistic. cv_r2_mean/cv_r2_std "
        "(k-fold cross-validation, when available) are the more honest estimate "
        "of generalization. Both are high-variance with this few samples — "
        "collect more real user ratings before quoting either as final performance."
    )
    metrics["feature_importance"] = {
        feature: round(float(importance), 4)
        for feature, importance in zip(FEATURE_COLUMNS, model.feature_importances_)
    }

    bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
    }

    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(bundle, model_file)

    with open(METRICS_PATH, "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    print(
        f"[ML] Engagement model trained: R²={metrics['train_r2']}, "
        f"MAE={metrics['train_mae']}, real_ratings={metrics['real_ratings']}"
    )

    return metrics


def load_model_bundle() -> dict | None:
    if not os.path.isfile(MODEL_PATH):
        return None

    with open(MODEL_PATH, "rb") as model_file:
        return pickle.load(model_file)


def ensure_model_trained() -> dict:
    bundle = load_model_bundle()
    if bundle is None:
        return train_engagement_model()
    return bundle.get("metrics", {})


def predict_engagement(reel_id: str) -> dict | None:
    """Predict engagement score for a single reel."""

    bundle = load_model_bundle()
    if bundle is None:
        metrics = train_engagement_model()
        if not metrics.get("trained"):
            return None
        bundle = load_model_bundle()

    features_df = build_feature_matrix()
    reel_features = features_df[features_df["reel_id"] == reel_id]

    if reel_features.empty:
        return None

    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    x_row = reel_features[feature_columns]
    prediction = float(model.predict(x_row)[0])
    prediction = round(float(np.clip(prediction, 1.0, 5.0)), 2)

    return {
        "predicted_score": prediction,
        "predicted_percent": round(prediction / 5.0 * 100),
    }


def enrich_reels_with_ml(reels: list[dict]) -> list[dict]:
    """Add predicted engagement and user ratings to reel metadata."""

    ensure_model_trained()
    rating_map = get_rating_map()

    for reel in reels:
        reel_id = reel["reel_id"]
        prediction = predict_engagement(reel_id)

        reel["user_rating"] = rating_map.get(reel_id)
        reel["predicted_score"] = prediction["predicted_score"] if prediction else None
        reel["predicted_percent"] = prediction["predicted_percent"] if prediction else None

    return reels


def get_model_summary() -> dict:
    """Return metrics for analytics dashboard."""

    bundle = load_model_bundle()
    if bundle is None:
        return ensure_model_trained()

    return bundle.get("metrics", {})
