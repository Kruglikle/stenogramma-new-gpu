import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

from audio_transcribator.config import settings
from audio_transcribator.services.model_loading import allow_legacy_torch_checkpoint_loading
from audio_transcribator.services.transcription_models import resolve_transcription_model
from audio_transcribator.utils.files import write_text_atomic


WHISPERX_VAD_MODEL_URL = "https://raw.githubusercontent.com/m-bain/whisperX/main/whisperx/assets/pytorch_model.bin"
WHISPERX_VAD_MODEL_SHA256 = "0b5b3216d60a2d32fc086b47ea8c67589aaeb26b7e07fcbe620d6d0b83e209ea"


def resolve_auto_device(device: str) -> str:
    if device != "auto":
        return device

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def format_timestamp(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    minutes, seconds_part = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"
    return f"{minutes:02d}:{seconds_part:02d}"


def normalize_transcript_text(text: str) -> str:
    return " ".join(text.split())


def resolve_whisperx_vad_model() -> Path:
    model_path = settings.whisperx_vad_model
    if model_path.exists():
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if digest != WHISPERX_VAD_MODEL_SHA256:
            raise RuntimeError(
                f"Local WhisperX VAD model checksum mismatch: {model_path}. "
                "Delete the file and download it again."
            )
        return model_path

    if settings.whisper_local_files_only:
        raise RuntimeError(
            "Local WhisperX VAD model was not found. Download it once into "
            f"{model_path} or set WHISPER_LOCAL_FILES_ONLY=false for the one-time cache step."
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading WhisperX VAD model: {WHISPERX_VAD_MODEL_URL}", flush=True)
    request = Request(WHISPERX_VAD_MODEL_URL, headers={"User-Agent": "audio-transcribator"})
    with urlopen(request, timeout=300) as response:
        model_path.write_bytes(response.read())

    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != WHISPERX_VAD_MODEL_SHA256:
        model_path.unlink(missing_ok=True)
        raise RuntimeError("Downloaded WhisperX VAD model checksum mismatch. Please retry.")

    return model_path


def transcribe(
    audio_file: Path,
    job_dir: Path,
    transcription_model_id: str | None = None,
    progress_callback=None,
) -> str:
    model_config = resolve_transcription_model(transcription_model_id)
    print(f"Transcribing with {model_config['id']}...")

    if model_config["provider"] != "whisperx":
        raise RuntimeError(f"Unsupported transcription provider: {model_config['provider']}")

    return transcribe_whisperx(
        audio_file,
        job_dir,
        model_config.get("model") or settings.whisperx_model,
        progress_callback=progress_callback,
    )


def _save_transcript_file(job_dir: Path, transcript: str) -> None:
    write_text_atomic(job_dir / "stenogramma.txt", transcript)


def _save_transcript_segments(job_dir: Path, segments: list[dict]) -> None:
    (job_dir / "transcript_segments.json").write_text(
        json.dumps(segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def transcribe_whisperx(audio_file: Path, job_dir: Path, model_name: str, progress_callback=None) -> str:
    try:
        import whisperx
    except ImportError as exc:
        raise RuntimeError(
            "WhisperX is not installed. Install whisperx==3.1.1 in the Docker image or server environment "
            "before selecting the WhisperX large-v3 transcription model."
        ) from exc

    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_auto_device(settings.whisperx_device)
    print(f"Transcribing with WhisperX {model_name} on {device}...", flush=True)
    vad_options = {"model_fp": str(resolve_whisperx_vad_model())}
    asr_options = {
        "multilingual": True,
        "max_new_tokens": None,
        "clip_timestamps": "0",
        "hallucination_silence_threshold": None,
        "hotwords": None,
    }

    with allow_legacy_torch_checkpoint_loading():
        try:
            model = whisperx.load_model(
                model_name,
                device,
                compute_type=settings.whisper_compute_type,
                language=settings.transcription_language,
                download_root=str(settings.model_cache_dir / "whisperx"),
                asr_options=asr_options,
                vad_options=vad_options,
            )
        except TypeError:
            model = whisperx.load_model(
                model_name,
                device,
                compute_type=settings.whisper_compute_type,
                download_root=str(settings.model_cache_dir / "whisperx"),
                asr_options=asr_options,
                vad_options=vad_options,
            )

    result = model.transcribe(
        str(audio_file),
        batch_size=settings.whisperx_batch_size,
        language=settings.transcription_language,
    )
    raw_segments = result.get("segments") or []
    segment_texts = []
    transcript_segments = []
    duration = max((float(segment.get("end") or 0) for segment in raw_segments), default=0)
    for segment in raw_segments:
        text = normalize_transcript_text(str(segment.get("text") or ""))
        if not text:
            continue

        start = float(segment.get("start") or 0)
        end = float(segment.get("end") or start)
        print(text, flush=True)
        segment_texts.append(text)
        transcript_segments.append({"start": start, "end": end, "text": text})
        _save_transcript_file(job_dir, normalize_transcript_text(" ".join(segment_texts)))
        _save_transcript_segments(job_dir, transcript_segments)
        if progress_callback and duration > 0:
            progress_callback(
                min(end / duration * 100, 99),
                f"Обработано {format_timestamp(end)} из {format_timestamp(duration)}",
            )

    transcript = normalize_transcript_text(" ".join(segment_texts))
    _save_transcript_file(job_dir, transcript)
    _save_transcript_segments(job_dir, transcript_segments)
    if progress_callback:
        progress_callback(100)

    return transcript
