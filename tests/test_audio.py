"""
Tests for services/beat_service.py — audio signal processing
(tempo/beat detection and beat-synced slide duration allocation).

Uses a synthetically generated click track instead of a real
audio file so the test suite has no external fixtures.
"""

import numpy as np
import soundfile as sf

from services import beat_service


def _write_click_track(path, duration_sec: float = 6.0, bpm: int = 120, sr: int = 22050) -> None:
    """Write a WAV file with short clicks spaced at a fixed BPM."""

    beat_interval = 60.0 / bpm
    samples = np.zeros(int(duration_sec * sr))
    click = np.sin(2 * np.pi * 1000 * np.linspace(0, 0.05, int(0.05 * sr)))

    t = 0.0
    while t < duration_sec:
        start = int(t * sr)
        end = min(start + len(click), len(samples))
        samples[start:end] += click[: end - start]
        t += beat_interval

    sf.write(path, samples, sr)


def test_extract_audio_features_returns_expected_keys(tmp_path):
    audio_path = tmp_path / "click.wav"
    _write_click_track(audio_path)

    features = beat_service.extract_audio_features(str(audio_path))

    assert set(features) == {"duration_sec", "tempo_bpm", "beat_count", "avg_energy"}
    assert features["duration_sec"] > 0
    assert features["tempo_bpm"] > 0
    assert features["beat_count"] > 0


def test_compute_beat_sync_durations_matches_image_count(tmp_path):
    audio_path = tmp_path / "click.wav"
    _write_click_track(audio_path)

    durations = beat_service.compute_beat_sync_durations(str(audio_path), num_images=5)

    assert len(durations) == 5
    assert all(
        beat_service.MIN_SLIDE_DURATION <= d <= beat_service.MAX_SLIDE_DURATION
        for d in durations
    )


def test_compute_beat_sync_durations_zero_images_returns_empty(tmp_path):
    audio_path = tmp_path / "click.wav"
    _write_click_track(audio_path)

    assert beat_service.compute_beat_sync_durations(str(audio_path), num_images=0) == []


def test_log_audio_features_appends_rows(tmp_path, monkeypatch):
    csv_path = tmp_path / "audio_features.csv"
    monkeypatch.setattr(beat_service, "AUDIO_FEATURES_CSV", str(csv_path))

    features = {
        "duration_sec": 10.0,
        "tempo_bpm": 120.0,
        "beat_count": 20,
        "avg_energy": 0.15,
    }

    beat_service.log_audio_features(
        reel_id="reel-1",
        features=features,
        image_count=4,
        sync_mode="beat_sync",
        slide_durations=[2.5, 2.5, 2.5, 2.5],
    )

    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("timestamp,reel_id,image_count")
    assert "reel-1" in lines[1]
    assert "2.5" in lines[1]
