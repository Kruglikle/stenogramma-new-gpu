import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from audio_transcribator.config import settings


DEFAULT_TRANSCRIPTION_MODEL_ID = "local:whisperx-large"


class TranscriptionModelError(ValueError):
    pass


@lru_cache
def load_transcription_models() -> list[dict[str, Any]]:
    models_file = settings.transcription_models_file
    try:
        models = json.loads(models_file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TranscriptionModelError(f"Transcription models file not found: {models_file}") from exc
    except json.JSONDecodeError as exc:
        raise TranscriptionModelError(f"Invalid transcription models JSON: {models_file}") from exc

    if not isinstance(models, list):
        raise TranscriptionModelError("Transcription models JSON must contain a list")

    normalized = []
    seen_ids = set()
    for item in models:
        if not isinstance(item, dict):
            raise TranscriptionModelError("Each transcription model must be an object")

        model_id = item.get("id")
        provider = item.get("provider")
        if not model_id or not provider:
            raise TranscriptionModelError("Each transcription model requires id and provider")
        if model_id in seen_ids:
            raise TranscriptionModelError(f"Duplicate transcription model id: {model_id}")
        if provider != "whisperx":
            raise TranscriptionModelError(f"Unsupported transcription model provider: {provider}")

        seen_ids.add(model_id)
        normalized.append(dict(item))

    return normalized


def list_transcription_models() -> list[dict[str, Any]]:
    return [dict(item) for item in load_transcription_models()]


def resolve_transcription_model(model_id: str | None) -> dict[str, Any]:
    requested_id = model_id or DEFAULT_TRANSCRIPTION_MODEL_ID
    for model in load_transcription_models():
        if model["id"] == requested_id:
            return dict(model)
    raise TranscriptionModelError(f"Unknown transcription model: {requested_id}")
