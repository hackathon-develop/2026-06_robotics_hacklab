#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD
"""Offline voice trigger helper for starting the first pick sequence."""

from __future__ import annotations

import json
import queue
import re
import time
from pathlib import Path
from typing import Any

DEFAULT_TRIGGER_PHRASES = (
    "folly start tidying up",
    "tidy up",
    "clean up",
)
DEFAULT_SAMPLE_RATE = 16_000


def parse_phrases(raw: str) -> list[str]:
    phrases = [phrase.strip() for phrase in raw.split(",") if phrase.strip()]
    if not phrases:
        raise ValueError("at least one voice trigger phrase is required")
    return phrases


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def _matches(text: str, phrases: list[str]) -> str | None:
    normalized = _normalize(text)
    if not normalized:
        return None
    for phrase in phrases:
        target = _normalize(phrase)
        if normalized == target or target in normalized:
            return phrase
    return None


def _parse_device(raw: str) -> int | str | None:
    value = raw.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _dependency_error() -> RuntimeError:
    return RuntimeError(
        "voice trigger requires optional dependencies. Install them with: "
        "pip install vosk sounddevice"
    )


def wait_for_voice_trigger(
    *,
    model_path: str | Path,
    phrases: list[str] | tuple[str, ...] = DEFAULT_TRIGGER_PHRASES,
    device: str = "",
    timeout_s: float = 0.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> str:
    """Block until one configured phrase is recognized, returning the matched phrase.

    ``timeout_s <= 0`` waits forever. Imports are intentionally lazy so regular
    perception runs do not require audio dependencies.
    """
    try:
        import sounddevice as sd
        import vosk
    except ModuleNotFoundError as exc:
        raise _dependency_error() from exc

    model_dir = Path(model_path).expanduser()
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Vosk model not found: {model_dir}. Download a model such as "
            "vosk-model-small-en-us-0.15 and pass --voice-model."
        )

    phrase_list = list(phrases)
    grammar = json.dumps(phrase_list + ["[unk]"])
    recognizer = vosk.KaldiRecognizer(vosk.Model(str(model_dir)), sample_rate, grammar)
    audio_queue: queue.Queue[bytes] = queue.Queue()
    deadline = time.monotonic() + timeout_s if timeout_s > 0 else None

    def callback(indata: Any, frames: int, callback_time: Any, status: Any) -> None:
        if status:
            print(f"[voice] audio status: {status}")
        audio_queue.put(bytes(indata))

    device_id = _parse_device(device)
    print("Listening for voice trigger: " + " | ".join(phrase_list), flush=True)
    with sd.RawInputStream(
        samplerate=sample_rate,
        blocksize=8000,
        device=device_id,
        dtype="int16",
        channels=1,
        callback=callback,
    ):
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for voice trigger")
            try:
                data = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if recognizer.AcceptWaveform(data):
                text = json.loads(recognizer.Result()).get("text", "")
            else:
                text = json.loads(recognizer.PartialResult()).get("partial", "")
            matched = _matches(text, phrase_list)
            if matched is not None:
                print(f"Voice trigger matched: {matched}", flush=True)
                return matched
