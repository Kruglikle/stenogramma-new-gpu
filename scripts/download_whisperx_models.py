import argparse
import os
from pathlib import Path

from faster_whisper.utils import download_model

from audio_transcribator.config import settings
from audio_transcribator.services.transcription import resolve_whisperx_vad_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Download WhisperX models for offline runtime.")
    parser.add_argument("--model", default=settings.whisperx_model)
    args = parser.parse_args()

    # Settings enables offline mode for normal workers. This command is the
    # explicit one-time exception that populates the persistent model cache.
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    settings.whisper_local_files_only = False

    vad_path = resolve_whisperx_vad_model()
    cache_dir = settings.model_cache_dir / "whisperx"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(
        download_model(
            args.model,
            cache_dir=str(cache_dir),
            local_files_only=False,
        )
    )

    print(f"WhisperX VAD model ready: {vad_path}")
    print(f"WhisperX ASR model ready: {model_path}")


if __name__ == "__main__":
    main()
