"""
Image Service

Responsible for:

1. Creating upload folders
2. Saving uploaded images
3. Creating FFmpeg input.txt
"""

import os
from werkzeug.utils import secure_filename

from config import VIDEO_FORMAT_DESKTOP, VIDEO_FORMAT_MOBILE, VOICE_GENDER_CHOICES, VOICE_GENDER_DEFAULT
from PIL import Image
from services.db_service import store_reel_files


FORMAT_FILE_NAME = "format.txt"
VOICE_GENDER_FILE_NAME = "voice.txt"


def create_upload_directory(upload_folder: str, request_id: str) -> str:
    """
    Create upload directory for every request.

    Example:

    uploads/
        12345/
    """

    upload_dir = os.path.join(upload_folder, request_id)

    os.makedirs(upload_dir, exist_ok=True)

    return upload_dir


def save_uploaded_images(files, upload_dir: str):
    """
    Save all uploaded images.

    Parameters
    ----------
    files : request.files
        May contain several images under the same field name (the
        multi-select "files" input), so we must iterate multi-valued
        pairs rather than just the first value per key.

    upload_dir : str

    Returns
    -------
    list
        List of saved image names.
    """

    image_files = []
    used_names = set()

    for key, file in files.items(multi=True):

        # Ignore audio upload
        if not key.startswith("file"):
            continue

        if file.filename == "":
            continue

        filename = secure_filename(file.filename)
        filename = _unique_filename(filename, used_names)
        used_names.add(filename)

        save_path = os.path.join(upload_dir, filename)

        file.save(save_path)

        image_files.append(filename)

    store_reel_files(os.path.basename(upload_dir), "image", upload_dir, image_files)

    return image_files


def _unique_filename(filename: str, used_names: set) -> str:
    """
    Avoid collisions when multiple selected images share a name
    (e.g. IMG_0001.jpg from different folders), which would otherwise
    overwrite an earlier upload and duplicate an entry in input.txt.
    """

    if filename not in used_names:
        return filename

    stem, ext = os.path.splitext(filename)
    counter = 1
    candidate = f"{stem}_{counter}{ext}"

    while candidate in used_names:
        counter += 1
        candidate = f"{stem}_{counter}{ext}"

    return candidate


def create_input_file(
    upload_dir: str,
    image_files: list,
    durations: list[float] | None = None,
):
    """
    Create FFmpeg input.txt

    Example

    file '1.jpg'
    duration 1.5

    file '2.jpg'
    duration 2.0
    """

    if durations is None:
        durations = [1.0] * len(image_files)

    if len(durations) != len(image_files):
        raise ValueError("Image count and duration count must match.")

    input_path = os.path.join(upload_dir, "input.txt")

    with open(input_path, "w", encoding="utf-8") as file:
        for image, duration in zip(image_files, durations):
            file.write(f"file '{image}'\n")
            file.write(f"duration {duration}\n")


def get_images_from_input_file(upload_dir: str) -> list[str]:
    """Read image filenames in order from an existing FFmpeg input.txt."""

    input_path = os.path.join(upload_dir, "input.txt")

    if not os.path.isfile(input_path):
        return []

    images = []

    with open(input_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("file "):
                images.append(line.split("'")[1])

    return images


def save_description(upload_dir: str, description: str):
    """
    Save AI description.
    """

    if not description:
        return

    description_path = os.path.join(upload_dir, "desc.txt")

    with open(description_path, "w", encoding="utf-8") as file:

        file.write(description)


def save_uploaded_audio(audio_file, upload_dir: str):
    """
    Save uploaded audio as audio.mp3
    """

    if audio_file is None:
        return

    if audio_file.filename == "":
        return

    audio_path = os.path.join(upload_dir, "audio.mp3")

    audio_file.save(audio_path)


def detect_dominant_format(upload_dir: str, image_files: list[str]) -> str:
    """
    Pick mobile (9:16) or desktop (16:9) from uploaded image orientations.
    """

    landscape_count = 0
    portrait_count = 0

    for image_name in image_files:
        image_path = os.path.join(upload_dir, image_name)

        with Image.open(image_path) as image:
            width, height = image.size
            if width >= height:
                landscape_count += 1
            else:
                portrait_count += 1

    if landscape_count >= portrait_count:
        return VIDEO_FORMAT_DESKTOP

    return VIDEO_FORMAT_MOBILE


def save_video_format(upload_dir: str, video_format: str) -> None:
    """Persist the output format for FFmpeg and gallery display."""

    if video_format not in {VIDEO_FORMAT_MOBILE, VIDEO_FORMAT_DESKTOP}:
        raise ValueError(f"Unsupported video format: {video_format}")

    format_path = os.path.join(upload_dir, FORMAT_FILE_NAME)

    with open(format_path, "w", encoding="utf-8") as file:
        file.write(video_format)


def read_video_format(upload_dir: str) -> str:
    """Read saved format, defaulting to mobile for older uploads."""

    format_path = os.path.join(upload_dir, FORMAT_FILE_NAME)

    if not os.path.isfile(format_path):
        return VIDEO_FORMAT_MOBILE

    with open(format_path, "r", encoding="utf-8") as file:
        video_format = file.read().strip()

    if video_format in {VIDEO_FORMAT_MOBILE, VIDEO_FORMAT_DESKTOP}:
        return video_format

    return VIDEO_FORMAT_MOBILE


def save_voice_gender(upload_dir: str, voice_gender: str) -> None:
    """Persist the narrator voice choice ("female"/"male"/"child"/"any")."""

    if voice_gender not in {*VOICE_GENDER_CHOICES, VOICE_GENDER_DEFAULT}:
        voice_gender = VOICE_GENDER_DEFAULT

    voice_path = os.path.join(upload_dir, VOICE_GENDER_FILE_NAME)

    with open(voice_path, "w", encoding="utf-8") as file:
        file.write(voice_gender)


def read_voice_gender(upload_dir: str) -> str:
    """Read the saved voice choice, defaulting to "any" (random) when missing."""

    voice_path = os.path.join(upload_dir, VOICE_GENDER_FILE_NAME)

    if not os.path.isfile(voice_path):
        return VOICE_GENDER_DEFAULT

    with open(voice_path, "r", encoding="utf-8") as file:
        voice_gender = file.read().strip()

    if voice_gender in {*VOICE_GENDER_CHOICES, VOICE_GENDER_DEFAULT}:
        return voice_gender

    return VOICE_GENDER_DEFAULT