"""
Tests for services/vision_service.py — computer vision quality scoring
(Laplacian blur, brightness, contrast, and the combined quality score).
"""

import numpy as np
import pytest
from PIL import Image

from services import vision_service


def test_compute_blur_variance_flat_image_is_zero():
    flat = np.full((50, 50), 128.0)
    assert vision_service.compute_blur_variance(flat) == 0.0


def test_compute_blur_variance_sharp_edges_score_higher_than_flat():
    flat = np.full((50, 50), 128.0)
    checkerboard = (np.indices((50, 50)).sum(axis=0) % 2 * 255.0)

    assert vision_service.compute_blur_variance(checkerboard) > vision_service.compute_blur_variance(flat)


def test_compute_blur_variance_too_small_returns_zero():
    tiny = np.zeros((2, 2))
    assert vision_service.compute_blur_variance(tiny) == 0.0


def test_compute_quality_score_best_case_is_capped_at_one():
    score = vision_service.compute_quality_score(blur_variance=10_000, brightness=128, contrast=1_000)
    assert score == pytest.approx(1.0)


def test_compute_quality_score_worst_case_is_zero():
    score = vision_service.compute_quality_score(blur_variance=0, brightness=0, contrast=0)
    assert score == pytest.approx(0.0)


@pytest.mark.parametrize(
    "score, expected_label",
    [
        (0.0, "poor"),
        (0.39, "poor"),
        (0.4, "fair"),
        (0.64, "fair"),
        (0.65, "good"),
        (1.0, "good"),
    ],
)
def test_label_quality_boundaries(score, expected_label):
    assert vision_service.label_quality(score) == expected_label


def test_is_low_quality_flags_blurry_images_even_with_decent_score():
    features = {"quality_score": 0.5, "blur_variance": 10.0}
    assert vision_service.is_low_quality(features) is True


def test_analyze_image_end_to_end(tmp_path):
    image_path = tmp_path / "sharp.png"
    array = (np.indices((100, 100)).sum(axis=0) % 2 * 255).astype("uint8")
    Image.fromarray(array, mode="L").save(image_path)

    result = vision_service.analyze_image(str(image_path))

    assert 0.0 <= result["quality_score"] <= 1.0
    assert result["quality_label"] in {"good", "fair", "poor"}
    assert result["blur_variance"] > 0


def test_analyze_images_ranks_sharpest_first(tmp_path):
    sharp_path = tmp_path / "sharp.png"
    blurry_path = tmp_path / "blurry.png"

    sharp = (np.indices((100, 100)).sum(axis=0) % 2 * 255).astype("uint8")
    blurry = np.full((100, 100), 128, dtype="uint8")

    Image.fromarray(sharp, mode="L").save(sharp_path)
    Image.fromarray(blurry, mode="L").save(blurry_path)

    ranked = vision_service.analyze_images(str(tmp_path), ["blurry.png", "sharp.png"])

    assert [item["filename"] for item in ranked] == ["sharp.png", "blurry.png"]


def test_log_image_features_appends_without_duplicating_header(tmp_path, monkeypatch):
    csv_path = tmp_path / "image_features.csv"
    monkeypatch.setattr(vision_service, "IMAGE_FEATURES_CSV", str(csv_path))

    analyzed = [{
        "filename": "a.jpg",
        "blur_variance": 120.0,
        "brightness": 100.0,
        "contrast": 40.0,
        "quality_score": 0.6,
        "quality_label": "fair",
    }]

    vision_service.log_image_features("reel-1", analyzed)
    vision_service.log_image_features("reel-1", analyzed)

    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("timestamp,reel_id,filename")
    assert len(lines) == 3
