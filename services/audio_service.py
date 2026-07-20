"""
Audio Service

Responsible for:
1. Convert text to speech via edge-tts, using Indian voices
   (with ElevenLabs as a fallback if edge-tts is unavailable)
2. Save generated audio
"""

import asyncio
import os
import random
import re

import edge_tts
from elevenlabs.client import ElevenLabs

from config import (
    UPLOAD_FOLDER,
    ELEVENLABS_API_KEY,
    ELEVENLABS_VOICE_ID,
    ELEVENLABS_VOICE_IDS,
    ELEVENLABS_VOICE_IDS_BY_GENDER,
    ELEVENLABS_MODEL_ID,
    EDGE_TTS_VOICE,
    EDGE_TTS_VOICES,
    EDGE_TTS_VOICES_BY_GENDER_LANG,
    VOICE_GENDER_DEFAULT,
)


_DEVANAGARI_PATTERN = re.compile(r"[ऀ-ॿ]")


def _audio_has_content(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0


def detect_narration_language(text: str) -> str:
    """Return "hi" for text containing Devanagari script, otherwise "en"."""

    if _DEVANAGARI_PATTERN.search(text or ""):
        return "hi"
    return "en"


def pick_elevenlabs_voice(voice_gender: str = VOICE_GENDER_DEFAULT) -> str:
    """Use the pinned voice if set, otherwise a random one per reel.

    voice_gender narrows the random pool to ELEVENLABS_VOICE_IDS_BY_GENDER
    ("male"/"female"/"child"); "any" (or an unrecognized value) uses the
    full ELEVENLABS_VOICE_IDS list.
    """

    if ELEVENLABS_VOICE_ID:
        return ELEVENLABS_VOICE_ID

    candidates = ELEVENLABS_VOICE_IDS_BY_GENDER.get(voice_gender, ELEVENLABS_VOICE_IDS)
    return random.choice(candidates)


def pick_edge_tts_voice(voice_gender: str = VOICE_GENDER_DEFAULT, text: str = "") -> str:
    """Use the pinned voice if set, otherwise a random one per reel.

    When voice_gender is "male"/"female"/"child", the pool is narrowed to
    EDGE_TTS_VOICES_BY_GENDER_LANG[voice_gender], further filtered to the
    language detected in text (Hindi vs English) so the accent matches
    the script — e.g. a Hindi description won't get read by an
    English-only voice. "any" (or an unrecognized value) ignores
    language and picks from the full EDGE_TTS_VOICES list, same as
    before gender selection existed.
    """

    if EDGE_TTS_VOICE:
        return EDGE_TTS_VOICE

    gender_map = EDGE_TTS_VOICES_BY_GENDER_LANG.get(voice_gender)
    if gender_map is None:
        return random.choice(EDGE_TTS_VOICES)

    language = detect_narration_language(text)
    candidates = gender_map.get(language) or [
        voice for voices in gender_map.values() for voice in voices
    ]
    return random.choice(candidates)


def _elevenlabs_tts(text: str, save_file_path: str, voice_gender: str = VOICE_GENDER_DEFAULT) -> None:
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    voice_id = pick_elevenlabs_voice(voice_gender)

    audio_stream = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text.strip(),
        model_id=ELEVENLABS_MODEL_ID,
        output_format="mp3_44100_128",
    )

    with open(save_file_path, "wb") as audio_file:
        for chunk in audio_stream:
            if chunk:
                audio_file.write(chunk)

    print(f"[TTS] ElevenLabs voice used: {voice_id}")


async def _edge_tts(text: str, save_file_path: str, voice_gender: str = VOICE_GENDER_DEFAULT) -> None:
    voice = pick_edge_tts_voice(voice_gender, text)
    communicate = edge_tts.Communicate(text.strip(), voice)
    await communicate.save(save_file_path)
    print(f"[TTS] edge-tts voice used: {voice}")


def text_to_speech_file(text: str, folder: str, voice_gender: str = VOICE_GENDER_DEFAULT) -> str:
    """
    Convert text into speech and save it as audio.mp3.

    Tries edge-tts first (free, guaranteed Indian voices — see
    config.EDGE_TTS_VOICES), then falls back to ElevenLabs when
    edge-tts is unavailable. ElevenLabs' available voices depend on
    the account/API key and are not guaranteed to include an Indian
    accent, so it is kept as the fallback rather than the default.

    voice_gender is "male", "female", "child", or "any" (default) and
    narrows which voice list each provider picks a random voice from.
    """

    if not text or not text.strip():
        raise ValueError("Text is required for AI audio generation")

    save_file_path = os.path.join(
        UPLOAD_FOLDER,
        folder,
        "audio.mp3",
    )

    os.makedirs(
        os.path.dirname(save_file_path),
        exist_ok=True,
    )

    if os.path.exists(save_file_path) and not _audio_has_content(save_file_path):
        os.remove(save_file_path)

    try:
        asyncio.run(_edge_tts(text, save_file_path, voice_gender))
        if _audio_has_content(save_file_path):
            print(f"edge-tts audio generated: {save_file_path}")
            return save_file_path
        print("[WARN] edge-tts returned empty audio, trying ElevenLabs fallback")
    except Exception as error:
        print(f"[WARN] edge-tts failed ({error}), trying ElevenLabs fallback")

    if os.path.exists(save_file_path):
        os.remove(save_file_path)

    if ELEVENLABS_API_KEY:
        try:
            _elevenlabs_tts(text, save_file_path, voice_gender)
            if _audio_has_content(save_file_path):
                print(f"ElevenLabs audio generated: {save_file_path}")
                return save_file_path
        except Exception as error:
            print(f"[WARN] ElevenLabs failed ({error})")

    if os.path.exists(save_file_path):
        os.remove(save_file_path)
    raise RuntimeError("Audio generation failed with all providers")
