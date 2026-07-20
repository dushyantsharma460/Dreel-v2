"""
Tests for services/image_service.py — upload handling and the
FFmpeg input.txt read/write round trip.
"""

import pytest
from PIL import Image

from services import image_service


def test_create_input_file_and_read_back_roundtrip(tmp_path):
    image_service.create_input_file(str(tmp_path), ["a.jpg", "b.jpg"], durations=[1.5, 2.0])

    content = (tmp_path / "input.txt").read_text(encoding="utf-8")
    assert "file 'a.jpg'" in content
    assert "duration 1.5" in content

    images = image_service.get_images_from_input_file(str(tmp_path))
    assert images == ["a.jpg", "b.jpg"]


def test_create_input_file_rejects_mismatched_durations(tmp_path):
    with pytest.raises(ValueError):
        image_service.create_input_file(str(tmp_path), ["a.jpg", "b.jpg"], durations=[1.0])


def test_create_input_file_defaults_to_one_second_per_image(tmp_path):
    image_service.create_input_file(str(tmp_path), ["a.jpg"])

    content = (tmp_path / "input.txt").read_text(encoding="utf-8")
    assert "duration 1.0" in content


def test_get_images_from_input_file_missing_returns_empty(tmp_path):
    assert image_service.get_images_from_input_file(str(tmp_path)) == []


def test_detect_dominant_format_picks_majority_orientation(tmp_path):
    landscape = Image.new("RGB", (200, 100))
    landscape.save(tmp_path / "landscape.jpg")

    portrait = Image.new("RGB", (100, 200))
    portrait.save(tmp_path / "portrait.jpg")
    portrait.save(tmp_path / "portrait2.jpg")

    result = image_service.detect_dominant_format(
        str(tmp_path), ["landscape.jpg", "portrait.jpg", "portrait2.jpg"]
    )

    assert result == "mobile"


def test_save_and_read_video_format_roundtrip(tmp_path):
    image_service.save_video_format(str(tmp_path), "desktop")
    assert image_service.read_video_format(str(tmp_path)) == "desktop"


def test_read_video_format_defaults_to_mobile_when_missing(tmp_path):
    assert image_service.read_video_format(str(tmp_path)) == "mobile"


def test_save_video_format_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError):
        image_service.save_video_format(str(tmp_path), "square")


def test_save_and_read_voice_gender_roundtrip(tmp_path):
    image_service.save_voice_gender(str(tmp_path), "child")
    assert image_service.read_voice_gender(str(tmp_path)) == "child"


def test_read_voice_gender_defaults_to_any_when_missing(tmp_path):
    assert image_service.read_voice_gender(str(tmp_path)) == "any"


def test_save_voice_gender_falls_back_to_any_for_unknown_value(tmp_path):
    image_service.save_voice_gender(str(tmp_path), "robot")
    assert image_service.read_voice_gender(str(tmp_path)) == "any"
