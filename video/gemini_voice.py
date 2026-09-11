"""Gemini TTS speech service for manim-voiceover, following the tome playbook:

- model gemini-3.1-flash-tts-preview via raw REST generateContent
- responseModalities AUDIO + prebuiltVoiceConfig; response is base64 raw PCM
  s16le/24kHz/mono, wrapped in a WAV header locally
- keys: GEMINI_API_KEY env first, then ~/.config/tome/config.json
  geminiApiKeys[] / geminiApiKey, deduped; rotate to the next key on
  429/RESOURCE_EXHAUSTED and retry the same segment
- anti-summarization: the documented "robust prompt" shape (AUDIO PROFILE /
  DIRECTOR'S NOTES / ## TRANSCRIPT last) plus a duration coverage check
  (seconds vs chars*0.0431); low coverage retries once with alternate phrasing
"""
import json
import os
import struct
import time
import urllib.request
import urllib.error
from pathlib import Path

from manim import logger
from manim_voiceover.helper import remove_bookmarks
from manim_voiceover.services.base import (
    SpeechService, initialize_speech_service, path_to_string,
)

MODEL = "gemini-3.1-flash-tts-preview"
SECS_PER_CHAR = 0.0431  # probed by tome: 7500 chars -> 323.4s
COVERAGE_MIN = 0.45     # short segments vary more than audiobook chunks

INSTRUCTION = (
    "Read every word of the transcript aloud, verbatim, as the narrator of an "
    "educational animated video. Do not summarize, paraphrase, abridge, skip "
    "ahead, or comment on it. Begin immediately with the first word and "
    "continue to the last."
)
INSTRUCTION_ALT = (
    "This is a production voice-over recording session for an educational "
    "video. Perform the transcript exactly as written, word for word, from "
    "the first word to the last. No preamble, no summary, no commentary - "
    "only the text itself."
)


def _resolve_keys():
    keys = []

    def push(v):
        if isinstance(v, str) and v.strip() and v.strip() not in keys:
            keys.append(v.strip())

    push(os.environ.get("GEMINI_API_KEY"))
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
        os.environ.get("HOME", "."), ".config")
    try:
        cfg = json.loads(Path(base, "tome", "config.json").read_text())
        for k in cfg.get("geminiApiKeys") or []:
            push(k)
        push(cfg.get("geminiApiKey"))
    except (OSError, ValueError):
        pass
    return keys


def _build_prompt(text, instruction, style):
    return "\n".join([
        "# AUDIO PROFILE: Explainer-video narrator",
        "",
        "## DIRECTOR'S NOTES",
        f"Style: {style}",
        "Pace: steady and unhurried, natural sentence rhythm.",
        f"Task: {instruction}",
        "",
        "## TRANSCRIPT",
        text,
    ])


def _wav_from_pcm(pcm: bytes) -> bytes:
    return b"".join([
        b"RIFF", struct.pack("<I", 36 + len(pcm)), b"WAVE",
        b"fmt ", struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16),
        b"data", struct.pack("<I", len(pcm)),
        pcm,
    ])


class QuotaError(Exception):
    pass


def _call_gemini(prompt, voice, key, timeout=300):
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{MODEL}:generateContent")
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {
                "prebuiltVoiceConfig": {"voiceName": voice}}},
        },
    }).encode()
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "x-goog-api-key": key, "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            payload = json.loads(res.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        if e.code == 429 or "RESOURCE_EXHAUSTED" in detail or "quota" in detail.lower():
            raise QuotaError(detail)
        raise RuntimeError(f"Gemini TTS HTTP {e.code}: {detail}")
    import base64
    b64 = (payload.get("candidates") or [{}])[0].get(
        "content", {}).get("parts", [{}])[0].get("inlineData", {}).get("data")
    if not b64:
        raise RuntimeError(
            f"Gemini TTS returned no audio: {json.dumps(payload)[:200]}")
    return base64.b64decode(b64)


class GeminiTTSService(SpeechService):
    """manim-voiceover SpeechService backed by Gemini TTS with key rotation."""

    def __init__(self, voice="Charon",
                 style=("measured, warm, and clear - an engaging maths/"
                        "computer-science explainer in the 3Blue1Brown vein"),
                 **kwargs):
        self.voice = voice
        self.style = style
        self.keys = _resolve_keys()
        self.key_index = 0
        if not self.keys:
            raise RuntimeError(
                "no Gemini API key - set GEMINI_API_KEY or put geminiApiKeys "
                "in ~/.config/tome/config.json")
        initialize_speech_service(self, kwargs)

    def _synth(self, text, instruction):
        while True:
            key = self.keys[self.key_index]
            try:
                return _call_gemini(
                    _build_prompt(text, instruction, self.style),
                    self.voice, key)
            except QuotaError:
                if self.key_index + 1 < len(self.keys):
                    self.key_index += 1
                    logger.info(
                        "gemini key %d exhausted - rotating to key %d of %d",
                        self.key_index - 1, self.key_index + 1, len(self.keys))
                    continue
                raise

    def generate_from_text(self, text, cache_dir=None, path=None, **kwargs):
        if cache_dir is None:
            cache_dir = self.cache_dir

        input_text = remove_bookmarks(text)
        input_data = {"input_text": input_text, "service": "gemini-tts",
                      "voice": self.voice, "model": MODEL}
        cached = self.get_cached_result(input_data, cache_dir)
        if cached is not None:
            return cached

        audio_path = (path_to_string(path) if path is not None
                      else self.get_audio_basename(input_data) + ".wav")

        expected = max(len(input_text) * SECS_PER_CHAR, 1.0)
        pcm = self._synth(input_text, INSTRUCTION)
        seconds = len(pcm) / 2 / 24000
        if seconds / expected < COVERAGE_MIN:
            logger.warning(
                "coverage %.2f (%.1fs vs ~%.1fs expected) - retrying with "
                "alternate instruction", seconds / expected, seconds, expected)
            time.sleep(2)
            pcm = self._synth(input_text, INSTRUCTION_ALT)
            seconds = len(pcm) / 2 / 24000
            if seconds / expected < COVERAGE_MIN:
                raise RuntimeError(
                    f"TTS summarized instead of reading (coverage "
                    f"{seconds / expected:.2f}) for: {input_text[:80]}...")
        logger.info("tts ok: %.1fs (coverage %.2f) for %d chars",
                    seconds, seconds / expected, len(input_text))

        Path(cache_dir, audio_path).write_bytes(_wav_from_pcm(pcm))
        return {"input_text": text, "input_data": input_data,
                "original_audio": audio_path}
