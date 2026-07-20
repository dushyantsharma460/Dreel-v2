"""
DReel AI v2 Configuration File
All project constants and paths are stored here.
"""

import os

from dotenv import load_dotenv

# --------------------------------------------------
# Base Directory
# --------------------------------------------------

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

load_dotenv(os.path.join(BASE_DIR, ".env"))

# --------------------------------------------------
# Flask Configuration
# --------------------------------------------------

SECRET_KEY = os.getenv("SECRET_KEY", "dreel-ai-v2-secret-key")

# --------------------------------------------------
# Project Folders
# --------------------------------------------------

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

STATIC_FOLDER = os.path.join(BASE_DIR, "static")

REELS_FOLDER = os.path.join(STATIC_FOLDER, "reels")

SONGS_FOLDER = os.path.join(STATIC_FOLDER, "songs")

REPORTS_FOLDER = os.path.join(BASE_DIR, "reports")

DATA_FOLDER = os.path.join(BASE_DIR, "data")

MODELS_FOLDER = os.path.join(BASE_DIR, "models")

# --------------------------------------------------
# Allowed Extensions
# --------------------------------------------------

ALLOWED_IMAGE_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg"
}

ALLOWED_AUDIO_EXTENSIONS = {
    "mp3",
    "wav"
}

# --------------------------------------------------
# Video Configuration
# --------------------------------------------------

VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
FPS = 30

DESKTOP_VIDEO_WIDTH = 1920
DESKTOP_VIDEO_HEIGHT = 1080

VIDEO_FORMAT_MOBILE = "mobile"
VIDEO_FORMAT_DESKTOP = "desktop"

# --------------------------------------------------
# PostgreSQL (optional analytics mirror)
# --------------------------------------------------

# Additive only — data/*.csv stays the source of truth for the Jupyter
# notebooks. This connection is just a parallel mirror of the same rows
# into Postgres tables (audio_features, image_features,
# engagement_ratings) so they're browsable in pgAdmin4 / any SQL tool.
#
# To point the app at a DIFFERENT existing database (local or remote),
# either edit the PG_* values below or set DATABASE_URL directly in
# .env — no code changes needed either way.
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DATABASE = os.getenv("PG_DATABASE", "dreel_ai")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}",
)

# --------------------------------------------------
# ElevenLabs
# --------------------------------------------------

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

# Leave ELEVENLABS_VOICE_ID / EDGE_TTS_VOICE unset (empty) in .env to get a
# random voice per reel from the lists below. Set them to a specific voice
# id/name to always use that one voice instead.
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")

ELEVENLABS_MODEL_ID = os.getenv(
    "ELEVENLABS_MODEL_ID",
    "eleven_multilingual_v2",
)

# Standard ElevenLabs premade voices (mix of genders/accents) used when
# ELEVENLABS_VOICE_ID is not pinned to a specific voice.
ELEVENLABS_VOICE_IDS = [
    "21m00Tcm4TlvDq8ikWAM",  # Rachel
    "AZnzlk1XvdvUeBnXmlld",  # Domi
    "EXAVITQu4vr4xnSDxMaL",  # Bella
    "ErXwobaYiN019PkySvjV",  # Antoni
    "TxGEqnHWrfWFTfGW9XjX",  # Josh
    "pNInz6obpgDQGcFmaJgB",  # Adam
]

EDGE_TTS_VOICE = os.getenv("EDGE_TTS_VOICE", "")

# Indian edge-tts neural voices used when EDGE_TTS_VOICE is not pinned to a
# specific voice. en-IN-* read English narration text with an Indian accent;
# hi-IN-* are for Hindi narration text.
EDGE_TTS_VOICES = [
    "en-IN-NeerjaNeural",
    "en-IN-NeerjaExpressiveNeural",
    "en-IN-PrabhatNeural",
    "hi-IN-SwaraNeural",
    "hi-IN-MadhurNeural",
]

# --------------------------------------------------
# Voice Gender Selection (narration)
# --------------------------------------------------

# Lets the user pick "male" / "female" / "child" on the Create page instead
# of always getting a fully random voice. "any" (the default when no
# selection is saved) picks from the full EDGE_TTS_VOICES /
# ELEVENLABS_VOICE_IDS lists above, preserving the original behavior.
VOICE_GENDER_CHOICES = ["female", "male", "child"]
VOICE_GENDER_DEFAULT = "any"

# Split by gender AND narration language, since a Hindi-script description
# read by an English-only voice (or vice versa) sounds wrong. audio_service
# detects "hi" vs "en" from the text itself and picks within the matching
# sub-list. edge-tts has no Indian-accent child voice, so "child" falls back
# to en-US-AnaNeural (Microsoft's only neural voice tagged as a child
# persona) for both languages.
EDGE_TTS_VOICES_BY_GENDER_LANG = {
    "female": {
        "en": ["en-IN-NeerjaNeural", "en-IN-NeerjaExpressiveNeural"],
        "hi": ["hi-IN-SwaraNeural"],
    },
    "male": {
        "en": ["en-IN-PrabhatNeural"],
        "hi": ["hi-IN-MadhurNeural"],
    },
    "child": {
        "en": ["en-US-AnaNeural"],
        "hi": ["en-US-AnaNeural"],
    },
}

# ElevenLabs' premade set has no dedicated child voice either; "child" uses
# Bella, its lightest/youngest-sounding preset, as the closest match.
ELEVENLABS_VOICE_IDS_BY_GENDER = {
    "female": ["21m00Tcm4TlvDq8ikWAM", "AZnzlk1XvdvUeBnXmlld"],  # Rachel, Domi
    "male": ["ErXwobaYiN019PkySvjV", "TxGEqnHWrfWFTfGW9XjX", "pNInz6obpgDQGcFmaJgB"],  # Antoni, Josh, Adam
    "child": ["EXAVITQu4vr4xnSDxMaL"],  # Bella
}