"""
Reel Service

Responsible for:
1. Generate audio from description
2. Create video reel using FFmpeg
3. Process new upload folders
"""

import os
import time
import subprocess

from config import (
    UPLOAD_FOLDER,
    REELS_FOLDER,
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    DESKTOP_VIDEO_WIDTH,
    DESKTOP_VIDEO_HEIGHT,
    VIDEO_FORMAT_DESKTOP,
)

from services.audio_service import text_to_speech_file
from services.beat_service import apply_beat_sync
from services.image_service import read_video_format, read_voice_gender
from services.vision_service import apply_vision_pipeline
from services.db_service import store_reel_file


DONE_FILE = os.path.join(UPLOAD_FOLDER, "done.txt")
POLL_INTERVAL_SECONDS = 4


def _audio_has_content(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def generate_audio(folder: str) -> bool:
    """
    Generate audio from desc.txt if available.
    Returns True when audio.mp3 exists or was created successfully.
    """

    folder_path = os.path.join(UPLOAD_FOLDER, folder)
    desc_path = os.path.join(folder_path, "desc.txt")
    audio_path = os.path.join(folder_path, "audio.mp3")

    if not os.path.exists(desc_path):
        if _audio_has_content(audio_path):
            print(f"[INFO] Using uploaded audio for {folder}")
            store_reel_file(folder, "audio", "audio.mp3", audio_path)
            return True

        print(f"[ERROR] No audio source found for {folder}")
        return False

    if _audio_has_content(audio_path):
        print(f"[INFO] Audio already exists for {folder}")
        store_reel_file(folder, "audio", "audio.mp3", audio_path)
        return True

    with open(desc_path, "r", encoding="utf-8") as file:
        text = file.read()

    voice_gender = read_voice_gender(folder_path)

    try:
        text_to_speech_file(text, folder, voice_gender)
        if _audio_has_content(audio_path):
            store_reel_file(folder, "audio", "audio.mp3", audio_path)
            return True

        print(f"[ERROR] Audio file is empty for {folder}")
        return False
    except Exception as error:
        print(f"[ERROR] Audio generation failed for {folder}: {error}")
        return False


def get_video_dimensions(folder: str) -> tuple[int, int]:
    """Return width and height based on saved mobile/desktop format."""

    folder_path = os.path.join(UPLOAD_FOLDER, folder)
    video_format = read_video_format(folder_path)

    if video_format == VIDEO_FORMAT_DESKTOP:
        return DESKTOP_VIDEO_WIDTH, DESKTOP_VIDEO_HEIGHT

    return VIDEO_WIDTH, VIDEO_HEIGHT


def build_video_filter(width: int, height: int) -> str:
    return (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black"
    )


def create_reel(folder: str) -> bool:
    """
    Generate final reel using FFmpeg.
    Returns True when the output .mp4 file was created.
    """

    input_file = os.path.join(
        UPLOAD_FOLDER,
        folder,
        "input.txt",
    )

    audio_file = os.path.join(
        UPLOAD_FOLDER,
        folder,
        "audio.mp3",
    )

    output_file = os.path.join(
        REELS_FOLDER,
        f"{folder}.mp4",
    )

    if not os.path.exists(input_file):
        print(f"[ERROR] input.txt not found: {folder}")
        return False

    if not _audio_has_content(audio_file):
        print(f"[ERROR] Audio not found or empty: {folder}")
        return False

    width, height = get_video_dimensions(folder)

    command = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", input_file,
        "-i", audio_file,
        "-vf",
        build_video_filter(width, height),
        "-c:v", "libx264",
        "-threads", "1",
        "-c:a", "aac",
        "-shortest",
        "-r", "30",
        "-pix_fmt", "yuv420p",
        output_file,
    ]

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"[SUCCESS] Reel created: {folder}")

        if os.path.exists(output_file):
            store_reel_file(folder, "video", f"{folder}.mp4", output_file)
            return True

        return False

    except subprocess.CalledProcessError as error:
        print(f"[ERROR] FFmpeg failed for {folder}: {error.stderr}")
        return False


def get_done_folders() -> list[str]:
    if not os.path.exists(DONE_FILE):
        open(DONE_FILE, "w").close()
        return []

    with open(DONE_FILE, "r", encoding="utf-8") as file:
        return [
            folder.strip()
            for folder in file.readlines()
            if folder.strip()
        ]


def mark_folder_done(folder: str) -> None:
    with open(DONE_FILE, "a", encoding="utf-8") as file:
        file.write(folder + "\n")


def list_reels() -> list[dict]:
    """
    Return reel metadata sorted newest first.
    """

    os.makedirs(REELS_FOLDER, exist_ok=True)

    reels = []

    for filename in os.listdir(REELS_FOLDER):
        if not filename.lower().endswith(".mp4"):
            continue

        file_path = os.path.join(REELS_FOLDER, filename)
        reel_id = os.path.splitext(filename)[0]
        upload_dir = os.path.join(UPLOAD_FOLDER, reel_id)

        reels.append({
            "filename": filename,
            "reel_id": reel_id,
            "format": read_video_format(upload_dir),
            "created_at": os.path.getmtime(file_path),
        })

    reels.sort(key=lambda reel: reel["created_at"], reverse=True)
    return reels


def process_new_folders() -> None:
    """
    Process all new upload folders.
    """

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(REELS_FOLDER, exist_ok=True)

    done_folders = get_done_folders()

    folders = [
        folder
        for folder in os.listdir(UPLOAD_FOLDER)
        if os.path.isdir(os.path.join(UPLOAD_FOLDER, folder))
        and folder not in done_folders
    ]

    for folder in folders:
        try:
            print(f"[PROCESSING] {folder}")

            if not generate_audio(folder):
                continue

            upload_dir = os.path.join(UPLOAD_FOLDER, folder)

            try:
                apply_vision_pipeline(upload_dir, folder)
            except Exception as error:
                print(f"[WARN] Vision analysis failed for {folder}: {error}")

            try:
                apply_beat_sync(upload_dir, folder)
            except Exception as error:
                print(f"[WARN] Beat-sync failed for {folder}, using default timing: {error}")

            if create_reel(folder):
                mark_folder_done(folder)

        except Exception as error:
            print(f"[ERROR] Failed to process {folder}: {error}")


def start_worker() -> None:
    """
    Background loop that turns uploads into gallery reels.
    """

    print("DReel AI v2 reel worker started...")

    while True:
        process_new_folders()
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    start_worker()
