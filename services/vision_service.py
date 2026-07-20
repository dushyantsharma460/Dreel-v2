"""
Vision Service

Computer vision analysis for uploaded reel images.
Scores blur, brightness, and contrast, then ranks images
by quality for smarter slideshow ordering.
"""

import csv
import os
from datetime import datetime, timezone

import numpy as np
from PIL import Image

from config import DATA_FOLDER
from services.image_service import create_input_file, get_images_from_input_file
from services.db_service import mirror_image_features

IMAGE_FEATURES_CSV = os.path.join(DATA_FOLDER, "image_features.csv")

LOW_QUALITY_THRESHOLD = 0.4
BLUR_VARIANCE_POOR = 80.0
ANALYSIS_MAX_SIDE = 800


def _load_grayscale_array(image_path: str) -> np.ndarray:
    """Load a grayscale numpy array, downscaled for faster analysis."""

    with Image.open(image_path) as image:
        image = image.convert("L")
        width, height = image.size

        if max(width, height) > ANALYSIS_MAX_SIDE:
            scale = ANALYSIS_MAX_SIDE / max(width, height)
            image = image.resize(
                (int(width * scale), int(height * scale)),
                Image.Resampling.LANCZOS,
            )

        return np.array(image, dtype=np.float64)


def compute_blur_variance(grayscale: np.ndarray) -> float:
    """
    Laplacian variance — higher values mean a sharper image.
    Common OpenCV-style blur metric implemented with NumPy.
    """

    if grayscale.shape[0] < 3 or grayscale.shape[1] < 3:
        return 0.0

    laplacian = (
        -4 * grayscale[1:-1, 1:-1]
        + grayscale[:-2, 1:-1]
        + grayscale[2:, 1:-1]
        + grayscale[1:-1, :-2]
        + grayscale[1:-1, 2:]
    )
    return float(laplacian.var())


def analyze_image(image_path: str) -> dict:
    """Extract blur, brightness, contrast, and a combined quality score."""

    grayscale = _load_grayscale_array(image_path)

    blur_variance = compute_blur_variance(grayscale)
    brightness = float(np.mean(grayscale))
    contrast = float(np.std(grayscale))
    quality_score = compute_quality_score(blur_variance, brightness, contrast)

    return {
        "blur_variance": round(blur_variance, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "quality_score": round(quality_score, 4),
        "quality_label": label_quality(quality_score),
    }


def compute_quality_score(
    blur_variance: float,
    brightness: float,
    contrast: float,
) -> float:
    """Weighted score from 0 (poor) to 1 (excellent)."""

    blur_norm = min(blur_variance / 500.0, 1.0)
    brightness_norm = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)
    contrast_norm = min(contrast / 80.0, 1.0)

    return (
        0.5 * blur_norm
        + 0.25 * brightness_norm
        + 0.25 * contrast_norm
    )


def label_quality(quality_score: float) -> str:
    if quality_score < LOW_QUALITY_THRESHOLD:
        return "poor"
    if quality_score < 0.65:
        return "fair"
    return "good"


def is_low_quality(features: dict) -> bool:
    return (
        features["quality_score"] < LOW_QUALITY_THRESHOLD
        or features["blur_variance"] < BLUR_VARIANCE_POOR
    )


def analyze_images(upload_dir: str, image_files: list[str]) -> list[dict]:
    """Analyze each image and return metadata sorted best-quality first."""

    analyzed = []

    for filename in image_files:
        image_path = os.path.join(upload_dir, filename)
        features = analyze_image(image_path)
        analyzed.append({
            "filename": filename,
            **features,
        })

    analyzed.sort(key=lambda item: item["quality_score"], reverse=True)
    return analyzed


def get_quality_warnings(upload_dir: str, image_files: list[str]) -> list[str]:
    """Human-readable warnings for images that may look bad in the reel."""

    warnings = []

    for filename in image_files:
        image_path = os.path.join(upload_dir, filename)
        features = analyze_image(image_path)

        if not is_low_quality(features):
            continue

        if features["blur_variance"] < BLUR_VARIANCE_POOR:
            warnings.append(
                f"'{filename}' looks blurry — consider replacing it for a sharper reel."
            )
        elif features["brightness"] < 60:
            warnings.append(
                f"'{filename}' is very dark — brightness may hurt video quality."
            )
        elif features["brightness"] > 210:
            warnings.append(
                f"'{filename}' is overexposed — details may be lost in the reel."
            )
        else:
            warnings.append(
                f"'{filename}' has low overall quality (score {features['quality_score']:.2f})."
            )

    return warnings


def log_image_features(reel_id: str, analyzed_images: list[dict]) -> None:
    """Append per-image vision features to the project dataset CSV."""

    os.makedirs(DATA_FOLDER, exist_ok=True)

    file_exists = os.path.isfile(IMAGE_FEATURES_CSV)
    fieldnames = [
        "timestamp",
        "reel_id",
        "filename",
        "sort_rank",
        "blur_variance",
        "brightness",
        "contrast",
        "quality_score",
        "quality_label",
    ]

    timestamp = datetime.now(timezone.utc).isoformat()
    rows = [
        {
            "timestamp": timestamp,
            "reel_id": reel_id,
            "filename": image_data["filename"],
            "sort_rank": rank,
            "blur_variance": image_data["blur_variance"],
            "brightness": image_data["brightness"],
            "contrast": image_data["contrast"],
            "quality_score": image_data["quality_score"],
            "quality_label": image_data["quality_label"],
        }
        for rank, image_data in enumerate(analyzed_images, start=1)
    ]

    with open(IMAGE_FEATURES_CSV, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)

    mirror_image_features(rows)


def remove_image_features(reel_id: str) -> None:
    """Drop this reel's rows from the image-features CSV, if present."""

    if not os.path.isfile(IMAGE_FEATURES_CSV):
        return

    with open(IMAGE_FEATURES_CSV, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames
        rows = [row for row in reader if row.get("reel_id") != reel_id]

    with open(IMAGE_FEATURES_CSV, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def apply_vision_pipeline(upload_dir: str, reel_id: str) -> dict:
    """
    Analyze images, reorder input.txt by quality, and log features.
    Best-quality images appear first in the final reel.
    """

    image_files = get_images_from_input_file(upload_dir)

    if not image_files:
        raise ValueError(f"No images found in input.txt for reel {reel_id}")

    analyzed_images = analyze_images(upload_dir, image_files)
    ranked_files = [item["filename"] for item in analyzed_images]

    create_input_file(upload_dir, ranked_files)
    log_image_features(reel_id, analyzed_images)

    low_quality_count = sum(
        1 for item in analyzed_images if item["quality_label"] == "poor"
    )
    avg_quality = round(
        sum(item["quality_score"] for item in analyzed_images) / len(analyzed_images),
        4,
    )

    print(
        f"[VISION] {reel_id}: ranked {len(ranked_files)} images, "
        f"avg_quality={avg_quality}, low_quality={low_quality_count}"
    )

    return {
        "image_count": len(analyzed_images),
        "avg_quality_score": avg_quality,
        "low_quality_count": low_quality_count,
        "ranked_files": ranked_files,
        "details": analyzed_images,
    }
