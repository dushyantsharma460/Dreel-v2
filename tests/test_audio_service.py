"""
Tests for services/audio_service.py — TTS voice selection and provider order.

Each reel should get a randomly chosen voice unless a specific voice
is pinned via config, so narration doesn't sound identical every time.
edge-tts (with confirmed Indian voices) must be tried before ElevenLabs,
since ElevenLabs' available voices depend on the account and are not
guaranteed to include an Indian accent.
"""

import os

from services import audio_service


def test_pick_edge_tts_voice_uses_pinned_value_when_set(monkeypatch):
    monkeypatch.setattr(audio_service, "EDGE_TTS_VOICE", "en-US-GuyNeural")

    assert audio_service.pick_edge_tts_voice() == "en-US-GuyNeural"


def test_pick_edge_tts_voice_is_random_from_list_when_unset(monkeypatch):
    monkeypatch.setattr(audio_service, "EDGE_TTS_VOICE", "")
    monkeypatch.setattr(audio_service, "EDGE_TTS_VOICES", ["voice-a", "voice-b"])

    picks = {audio_service.pick_edge_tts_voice() for _ in range(30)}

    assert picks <= {"voice-a", "voice-b"}
    assert len(picks) > 1  # should see more than one voice across 30 draws


def test_pick_edge_tts_voice_narrows_to_gender_and_detected_language(monkeypatch):
    monkeypatch.setattr(audio_service, "EDGE_TTS_VOICE", "")
    monkeypatch.setattr(
        audio_service,
        "EDGE_TTS_VOICES_BY_GENDER_LANG",
        {
            "male": {"en": ["male-en-a", "male-en-b"], "hi": ["male-hi-a"]},
            "female": {"en": ["female-en-a"], "hi": ["female-hi-a"]},
        },
    )

    english_picks = {audio_service.pick_edge_tts_voice("male", "Hello there") for _ in range(30)}
    assert english_picks <= {"male-en-a", "male-en-b"}

    hindi_picks = {audio_service.pick_edge_tts_voice("male", "नमस्ते दोस्तों") for _ in range(10)}
    assert hindi_picks == {"male-hi-a"}


def test_pick_edge_tts_voice_falls_back_to_other_language_when_gender_lacks_match(monkeypatch):
    monkeypatch.setattr(audio_service, "EDGE_TTS_VOICE", "")
    monkeypatch.setattr(
        audio_service,
        "EDGE_TTS_VOICES_BY_GENDER_LANG",
        {"child": {"en": ["child-en-a"]}},
    )

    picks = {audio_service.pick_edge_tts_voice("child", "नमस्ते दोस्तों") for _ in range(10)}

    assert picks == {"child-en-a"}


def test_pick_edge_tts_voice_pinned_value_wins_over_gender(monkeypatch):
    monkeypatch.setattr(audio_service, "EDGE_TTS_VOICE", "en-US-GuyNeural")
    monkeypatch.setattr(
        audio_service,
        "EDGE_TTS_VOICES_BY_GENDER_LANG",
        {"male": {"en": ["male-a"]}},
    )

    assert audio_service.pick_edge_tts_voice("male") == "en-US-GuyNeural"


def test_detect_narration_language_identifies_hindi_script():
    assert audio_service.detect_narration_language("नमस्ते दोस्तों") == "hi"


def test_detect_narration_language_defaults_to_english():
    assert audio_service.detect_narration_language("Hello there") == "en"
    assert audio_service.detect_narration_language("") == "en"


def test_pick_elevenlabs_voice_uses_pinned_value_when_set(monkeypatch):
    monkeypatch.setattr(audio_service, "ELEVENLABS_VOICE_ID", "pinned-id")

    assert audio_service.pick_elevenlabs_voice() == "pinned-id"


def test_pick_elevenlabs_voice_is_random_from_list_when_unset(monkeypatch):
    monkeypatch.setattr(audio_service, "ELEVENLABS_VOICE_ID", "")
    monkeypatch.setattr(audio_service, "ELEVENLABS_VOICE_IDS", ["id-a", "id-b"])

    picks = {audio_service.pick_elevenlabs_voice() for _ in range(30)}

    assert picks <= {"id-a", "id-b"}
    assert len(picks) > 1


def test_text_to_speech_file_prefers_edge_tts_over_elevenlabs(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_service, "UPLOAD_FOLDER", str(tmp_path))
    monkeypatch.setattr(audio_service, "ELEVENLABS_API_KEY", "fake-key")

    calls = []

    async def fake_edge_tts(text, save_file_path, voice_gender="any"):
        calls.append("edge")
        with open(save_file_path, "wb") as audio_file:
            audio_file.write(b"fake-audio-bytes")

    def fake_elevenlabs_tts(text, save_file_path, voice_gender="any"):
        calls.append("elevenlabs")
        raise AssertionError("ElevenLabs should not run when edge-tts succeeds")

    monkeypatch.setattr(audio_service, "_edge_tts", fake_edge_tts)
    monkeypatch.setattr(audio_service, "_elevenlabs_tts", fake_elevenlabs_tts)

    result_path = audio_service.text_to_speech_file("hello world", "reel-1")

    assert calls == ["edge"]
    assert os.path.isfile(result_path)


def test_text_to_speech_file_falls_back_to_elevenlabs_when_edge_tts_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_service, "UPLOAD_FOLDER", str(tmp_path))
    monkeypatch.setattr(audio_service, "ELEVENLABS_API_KEY", "fake-key")

    calls = []

    async def failing_edge_tts(text, save_file_path, voice_gender="any"):
        calls.append("edge")
        raise RuntimeError("network down")

    def fake_elevenlabs_tts(text, save_file_path, voice_gender="any"):
        calls.append("elevenlabs")
        with open(save_file_path, "wb") as audio_file:
            audio_file.write(b"fake-audio-bytes")

    monkeypatch.setattr(audio_service, "_edge_tts", failing_edge_tts)
    monkeypatch.setattr(audio_service, "_elevenlabs_tts", fake_elevenlabs_tts)

    result_path = audio_service.text_to_speech_file("hello world", "reel-1")

    assert calls == ["edge", "elevenlabs"]
    assert os.path.isfile(result_path)


def test_text_to_speech_file_raises_when_both_providers_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(audio_service, "UPLOAD_FOLDER", str(tmp_path))
    monkeypatch.setattr(audio_service, "ELEVENLABS_API_KEY", "")

    async def failing_edge_tts(text, save_file_path, voice_gender="any"):
        raise RuntimeError("network down")

    monkeypatch.setattr(audio_service, "_edge_tts", failing_edge_tts)

    try:
        audio_service.text_to_speech_file("hello world", "reel-1")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
