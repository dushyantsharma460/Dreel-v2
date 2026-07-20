"""
Beat Service

Audio signal processing for beat-synced slide timing.
Uses librosa to detect tempo and beats, then allocates
image durations aligned to the audio rhythm.
"""

import csv
import os
from datetime import datetime, timezone

import librosa
import numpy as np

from config import DATA_FOLDER
from services.image_service import create_input_file, get_images_from_input_file
from services.db_service import mirror_audio_features

AUDIO_FEATURES_CSV = os.path.join(DATA_FOLDER, "audio_features.csv")

MIN_SLIDE_DURATION = 0.5
MAX_SLIDE_DURATION = 6.0


def extract_audio_features(audio_path: str) -> dict:
    """Extract tempo, beat count, duration, and energy from an audio file."""

    y, sr = librosa.load(audio_path, sr=None)
    duration = float(librosa.get_duration(y=y, sr=sr))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

    if np.ndim(tempo):
        tempo = float(tempo[0])
    else:
        tempo = float(tempo)

    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    rms = librosa.feature.rms(y=y)

    return {
        "duration_sec": round(duration, 3),
        "tempo_bpm": round(tempo, 2),
        "beat_count": int(len(beat_times)),
        "avg_energy": round(float(np.mean(rms)), 4),
    }


def compute_beat_sync_durations(audio_path: str, num_images: int) -> list[float]:
    """
    Allocate per-image durations aligned to detected beats.
    Falls back to equal splits when beat detection is weak.
    """

    if num_images <= 0:
        return []

    y, sr = librosa.load(audio_path, sr=None)
    duration = float(librosa.get_duration(y=y, sr=sr))

    if duration <= 0:
        return [1.0] * num_images

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    if len(beat_times) >= 2:
        candidates = np.unique(np.concatenate(([0.0], beat_times, [duration])))
        if len(candidates) > num_images + 1:
            indices = np.linspace(0, len(candidates) - 1, num_images + 1).astype(int)
            boundaries = candidates[indices]
        else:
            boundaries = np.linspace(0.0, duration, num_images + 1)
    else:
        boundaries = np.linspace(0.0, duration, num_images + 1)

    durations = []
    for index in range(num_images):
        slide_duration = float(boundaries[index + 1] - boundaries[index])
        slide_duration = max(MIN_SLIDE_DURATION, min(MAX_SLIDE_DURATION, slide_duration))
        durations.append(round(slide_duration, 3))

    total = sum(durations)
    if total > 0 and abs(total - duration) > 0.05:
        scale = duration / total
        durations = [round(value * scale, 3) for value in durations]

    return durations


def log_audio_features(
    reel_id: str,
    features: dict,
    image_count: int,
    sync_mode: str,
    slide_durations: list[float],
) -> None:
    """Append extracted audio features to the project dataset CSV."""

    os.makedirs(DATA_FOLDER, exist_ok=True)

    file_exists = os.path.isfile(AUDIO_FEATURES_CSV)
    fieldnames = [
        "timestamp",
        "reel_id",
        "image_count",
        "sync_mode",
        "duration_sec",
        "tempo_bpm",
        "beat_count",
        "avg_energy",
        "avg_slide_duration",
        "slide_durations",
    ]

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reel_id": reel_id,
        "image_count": image_count,
        "sync_mode": sync_mode,
        "duration_sec": features["duration_sec"],
        "tempo_bpm": features["tempo_bpm"],
        "beat_count": features["beat_count"],
        "avg_energy": features["avg_energy"],
        "avg_slide_duration": round(
            sum(slide_durations) / len(slide_durations),
            3,
        ) if slide_durations else 0,
        "slide_durations": "|".join(str(value) for value in slide_durations),
    }

    with open(AUDIO_FEATURES_CSV, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    mirror_audio_features(row)


def apply_beat_sync(upload_dir: str, reel_id: str) -> dict:
    """
    Rebuild FFmpeg input.txt using beat-synced slide durations.
    Returns extracted audio features for logging and debugging.
    """

    audio_path = os.path.join(upload_dir, "audio.mp3")
    image_files = get_images_from_input_file(upload_dir)

    if not image_files:
        raise ValueError(f"No images found in input.txt for reel {reel_id}")

    features = extract_audio_features(audio_path)
    durations = compute_beat_sync_durations(audio_path, len(image_files))

    create_input_file(upload_dir, image_files, durations=durations)
    log_audio_features(
        reel_id=reel_id,
        features=features,
        image_count=len(image_files),
        sync_mode="beat_sync",
        slide_durations=durations,
    )

    print(
        f"[BEAT-SYNC] {reel_id}: tempo={features['tempo_bpm']} BPM, "
        f"beats={features['beat_count']}, durations={durations}"
    )

    return {
        **features,
        "sync_mode": "beat_sync",
        "slide_durations": durations,
    }
