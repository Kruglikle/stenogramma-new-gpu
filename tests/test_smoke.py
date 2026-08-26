import io
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi import UploadFile
from starlette.requests import Request


def test_entrypoint_files_exist() -> None:
    """Совместимые entrypoint-файлы должны оставаться на месте для Docker и старых команд."""
    root = Path(__file__).resolve().parents[1]

    assert (root / "app.py").is_file()
    assert (root / "process_audio_fast.py").is_file()
    assert (root / "Dockerfile").is_file()
    assert (root / "docker-compose.yml").is_file()
    assert (root / "Caddyfile").is_file()


def test_settings_load_without_model_initialization() -> None:
    """Импорт настроек не должен скачивать модели или обращаться к внешним сервисам."""
    from audio_transcribator.config import settings

    assert settings.base_dir.exists()
    assert settings.upload_dir.name == "uploads"
    assert settings.results_dir.name == "api_results"
    assert settings.model_cache_dir.name == "model_cache"
    assert settings.vllm_base_url
    assert settings.summary_model == "Qwen/Qwen3-8B-AWQ"


def test_fastapi_app_import_does_not_start_server() -> None:
    """Импорт app.py должен создать ASGI-объект без запуска uvicorn."""
    from app import app

    assert isinstance(app, FastAPI)


def test_worker_entrypoint_importable() -> None:
    """process_audio_fast.py остается совместимой оберткой над worker-ом."""
    from process_audio_fast import main

    assert callable(main)


def test_core_modules_importable() -> None:
    """Основные service-модули должны импортироваться без загрузки больших ML-моделей."""
    import audio_transcribator.services.audio
    import audio_transcribator.services.diarization
    import audio_transcribator.services.job_formatting
    import audio_transcribator.services.job_launch
    import audio_transcribator.services.job_metadata
    import audio_transcribator.services.job_results
    import audio_transcribator.services.job_storage
    import audio_transcribator.services.jobs
    import audio_transcribator.services.pipeline_steps
    import audio_transcribator.services.pipeline_progress
    import audio_transcribator.services.progress
    import audio_transcribator.services.summary
    import audio_transcribator.services.transcription

    assert audio_transcribator.services.jobs.STATUS_LABELS


def test_unsupported_upload_does_not_create_job_directory(tmp_path, monkeypatch) -> None:
    from audio_transcribator.config import settings
    from audio_transcribator.services import job_launch

    results_dir = tmp_path / "results"
    uploads_dir = tmp_path / "uploads"
    results_dir.mkdir()
    uploads_dir.mkdir()

    monkeypatch.setattr(settings, "results_dir", results_dir)
    monkeypatch.setattr(settings, "upload_dir", uploads_dir)
    monkeypatch.setattr(job_launch, "resolve_transcription_model", lambda _model_id: {"id": "local:test"})
    monkeypatch.setattr(job_launch, "ensure_user_storage_quota", lambda _user, _size: None)

    upload = UploadFile(filename="notes.txt", file=io.BytesIO(b"not media"))
    with pytest.raises(ValueError, match="Unsupported media file extension"):
        job_launch.start_uploaded_file(upload, enable_summary=False)

    assert list(results_dir.iterdir()) == []
    assert list(uploads_dir.iterdir()) == []


def test_legacy_checkpoint_loader_is_scoped_and_restores_torch_load(monkeypatch) -> None:
    import torch

    from audio_transcribator.services.model_loading import allow_legacy_torch_checkpoint_loading

    calls = []

    def fake_torch_load(*args, **kwargs):
        calls.append((args, kwargs))
        return "loaded"

    monkeypatch.setattr(torch, "load", fake_torch_load)
    with allow_legacy_torch_checkpoint_loading():
        assert torch.load("checkpoint.bin") == "loaded"
        assert torch.load("default.bin", weights_only=None) == "loaded"
        assert torch.load("explicit.bin", weights_only=True) == "loaded"

    assert torch.load is fake_torch_load
    assert calls[0][1]["weights_only"] is False
    assert calls[1][1]["weights_only"] is False
    assert calls[2][1]["weights_only"] is True


def test_diarization_failure_keeps_pipeline_result_available(tmp_path, monkeypatch) -> None:
    from audio_transcribator.config import settings
    from audio_transcribator.services import pipeline_steps

    monkeypatch.setattr(settings, "enable_diarization", True)
    monkeypatch.setattr(pipeline_steps, "timed_step", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("model unavailable")))

    pipeline_steps.maybe_diarize(tmp_path / "audio.wav", tmp_path, True)

    assert (tmp_path / "diarization_error.txt").read_text(encoding="utf-8") == "model unavailable"


def test_successful_diarization_clears_previous_error(tmp_path, monkeypatch) -> None:
    from audio_transcribator.config import settings
    from audio_transcribator.services import pipeline_steps

    error_path = tmp_path / "diarization_error.txt"
    error_path.write_text("old error", encoding="utf-8")
    monkeypatch.setattr(settings, "enable_diarization", True)
    monkeypatch.setattr(pipeline_steps, "timed_step", lambda *_args, **_kwargs: None)

    pipeline_steps.maybe_diarize(tmp_path / "audio.wav", tmp_path, True)

    assert not error_path.exists()


def test_wespeaker_cuda_session_uses_heuristic_without_cpu_fallback(monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    from audio_transcribator.services.diarization import configure_wespeaker_cuda_session

    class ONNXWeSpeakerPretrainedSpeakerEmbedding:
        embedding = "speaker-embedding.onnx"
        session_ = None

    session_options = SimpleNamespace()
    captured = {}

    class FakeSession:
        def __init__(self, model, sess_options, providers):
            captured.update(model=model, sess_options=sess_options, providers=providers)

        def get_providers(self):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    fake_ort = SimpleNamespace(
        SessionOptions=lambda: session_options,
        InferenceSession=FakeSession,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    pipeline = SimpleNamespace(_embedding=ONNXWeSpeakerPretrainedSpeakerEmbedding())

    configure_wespeaker_cuda_session(pipeline)

    assert captured["model"] == "speaker-embedding.onnx"
    assert captured["providers"] == [
        ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"})
    ]
    assert captured["sess_options"].log_severity_level == 3
    assert pipeline._embedding.session_ is not None


def test_auto_diarization_filters_tiny_speaker_clusters(monkeypatch) -> None:
    from audio_transcribator.config import settings
    from audio_transcribator.services.diarization import filter_noisy_turns

    monkeypatch.setattr(settings, "diarization_min_turn_seconds", 0.25)
    monkeypatch.setattr(settings, "diarization_min_speaker_ratio", 0.02)
    turns = [
        {"speaker": "A", "start": 0.0, "end": 100.0},
        {"speaker": "B", "start": 100.0, "end": 101.0},
    ]

    assert {turn["speaker"] for turn in filter_noisy_turns(turns)} == {"A"}


def test_manual_speaker_count_preserves_tiny_speaker_clusters(monkeypatch) -> None:
    from audio_transcribator.config import settings
    from audio_transcribator.services.diarization import filter_noisy_turns

    monkeypatch.setattr(settings, "diarization_min_turn_seconds", 0.25)
    monkeypatch.setattr(settings, "diarization_min_speaker_ratio", 0.02)
    turns = [
        {"speaker": "A", "start": 0.0, "end": 100.0},
        {"speaker": "B", "start": 100.0, "end": 101.0},
    ]

    assert {turn["speaker"] for turn in filter_noisy_turns(turns, requested_speakers=2)} == {"A", "B"}


def test_summary_chunking_uses_token_overlap(monkeypatch) -> None:
    from audio_transcribator.services import summary

    monkeypatch.setattr(summary, "get_tokenizer", lambda: None)
    monkeypatch.setattr(summary, "MIN_SUMMARY_CHUNK_TOKENS", 1)
    text = " ".join(f"Предложение {index}." for index in range(40))

    chunks = summary.chunk_text_with_overlap(text, max_tokens=12, overlap_tokens=4)

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
    assert "Предложение" in chunks[0]


def test_summary_tokenizer_resolves_huggingface_snapshot(tmp_path, monkeypatch) -> None:
    from audio_transcribator.services.summary import resolve_local_tokenizer_model

    repository = tmp_path / "models--Qwen--Qwen3-8B-AWQ"
    snapshot = repository / "snapshots" / "revision-1"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (repository / "refs").mkdir()
    (repository / "refs" / "main").write_text("revision-1", encoding="utf-8")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))

    resolved = resolve_local_tokenizer_model("Qwen/Qwen3-8B-AWQ")

    assert resolved == str(snapshot.resolve())


def test_summary_disables_qwen_thinking(monkeypatch) -> None:
    from types import SimpleNamespace

    from audio_transcribator.config import settings
    from audio_transcribator.services.summary import call_summary_model

    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Ready"))],
            usage=None,
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(settings, "summary_request_retries", 0)
    monkeypatch.setattr(settings, "summary_enable_thinking", False)

    result, _usage = call_summary_model(client, [{"role": "user", "content": "test"}], "system", 32)

    assert result == "Ready"
    assert captured["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_long_partial_summaries_are_reduced_hierarchically(monkeypatch) -> None:
    from audio_transcribator.config import settings
    from audio_transcribator.services import summary

    calls = []
    monkeypatch.setattr(settings, "summary_max_chunk_tokens", 50)
    monkeypatch.setattr(settings, "summary_chunk_max_tokens", 10)
    monkeypatch.setattr(summary, "count_tokens", lambda text: len(text.split()))
    monkeypatch.setattr(
        summary,
        "chunk_text_with_overlap",
        lambda text, _max_tokens, _overlap: [
            " ".join(text.split()[start : start + 100])
            for start in range(0, len(text.split()), 100)
        ],
    )

    def fake_call(_client, messages, system_prompt, max_tokens):
        calls.append(messages[0]["content"])
        return "short result", {"completion_tokens": 2}

    monkeypatch.setattr(summary, "call_summary_model", fake_call)
    usage_items = []

    combined = summary.reduce_partial_summaries(object(), ["word " * 300], usage_items)

    assert combined.count("short result") == 3
    assert len(calls) == 3
    assert [item["stage"] for item in usage_items] == ["reduce", "reduce", "reduce"]


def test_private_media_urls_are_blocked() -> None:
    from audio_transcribator.services.security import validate_public_media_url

    try:
        validate_public_media_url("http://127.0.0.1:8000/private.mp4")
    except ValueError as exc:
        assert "Private" in str(exc)
    else:
        raise AssertionError("Private media URL was not blocked")


def test_upload_filename_is_sanitized() -> None:
    from audio_transcribator.services.security import sanitize_upload_filename

    assert sanitize_upload_filename("../../meeting video.mp4") == "meeting_video.mp4"


def test_production_host_rejects_default_secrets(monkeypatch) -> None:
    from audio_transcribator.config import Settings

    monkeypatch.setenv("APP_HOST", "stenogramma.example.com")
    monkeypatch.setenv("API_TOKEN", "test-token")
    monkeypatch.setenv("API_PASSWORD", "admin123")
    monkeypatch.setenv("CREATOR_PASSWORD", "creator123")
    monkeypatch.setenv("ADD_USER_ADMIN_TOKEN", "change-me-add-user-token")
    monkeypatch.setenv("POSTGRES_PASSWORD", "audio_transcribator_password")

    try:
        Settings()
    except RuntimeError as exc:
        assert "Unsafe production defaults" in str(exc)
    else:
        raise AssertionError("Production defaults were not rejected")


def test_transcription_models_default_to_whisperx() -> None:
    """Пользовательский выбор распознавания сведен к локальному WhisperX."""
    from audio_transcribator.services.transcription_models import (
        DEFAULT_TRANSCRIPTION_MODEL_ID,
        list_transcription_models,
    )

    models = list_transcription_models()
    assert DEFAULT_TRANSCRIPTION_MODEL_ID == "local:whisperx-large"
    assert [model["id"] for model in models] == ["local:whisperx-large"]
    assert {model["provider"] for model in models} == {"whisperx"}


def test_transcription_models_reject_remote_provider(tmp_path, monkeypatch) -> None:
    from audio_transcribator.config import settings
    from audio_transcribator.services import transcription_models

    models_file = tmp_path / "models.json"
    models_file.write_text(
        '[{"id": "remote", "label": "Remote", "provider": "remote-ai", "model": "whisper"}]',
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "transcription_models_file", models_file)
    transcription_models.load_transcription_models.cache_clear()

    try:
        transcription_models.load_transcription_models()
    except transcription_models.TranscriptionModelError as exc:
        assert "Unsupported" in str(exc)
    else:
        raise AssertionError("Remote transcription provider was not rejected")
    finally:
        transcription_models.load_transcription_models.cache_clear()


def test_job_queue_dispatches_fifo_with_concurrency_limit(tmp_path, monkeypatch) -> None:
    from audio_transcribator.config import settings
    from audio_transcribator.services import job_queue

    def write_job(name: str, status: str, started_at: str) -> Path:
        job_dir = tmp_path / name
        job_dir.mkdir()
        (job_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "job_id": name,
                    "status": status,
                    "input_file": str(tmp_path / f"{name}.wav"),
                    "started_at": started_at,
                    "transcription_model": "local:whisperx-large",
                    "enable_transcription": True,
                    "enable_summary": False,
                    "enable_diarization": False,
                }
            ),
            encoding="utf-8",
        )
        return job_dir

    write_job("active", "running", "2026-01-01T00:00:00+00:00")
    write_job("newer", "queued", "2026-01-01T00:02:00+00:00")
    write_job("older", "queued", "2026-01-01T00:01:00+00:00")
    launched = []

    def fake_launch(job_dir: Path, metadata: dict) -> None:
        launched.append(job_dir.name)

    monkeypatch.setattr(settings, "results_dir", tmp_path)
    monkeypatch.setattr(settings, "max_concurrent_jobs", 2)
    monkeypatch.setattr(job_queue, "launch_worker_for_job", fake_launch)

    assert job_queue.dispatch_queued_jobs() == ["older"]
    assert launched == ["older"]


def test_ui_get_pages_do_not_require_origin_header(monkeypatch) -> None:
    from audio_transcribator.ui import routes

    request = Request({"type": "http", "method": "GET", "path": "/ui/upload", "headers": [(b"host", b"testserver")]})
    rendered = []

    class FakeTemplates:
        def TemplateResponse(self, request, template_name, context, status_code=200):
            rendered.append(template_name)
            return {"template": template_name, "context": context, "status_code": status_code}

    monkeypatch.setattr(routes, "templates", FakeTemplates())
    monkeypatch.setattr(routes, "require_ui_auth", lambda *args, **kwargs: "user")
    monkeypatch.setattr(routes, "build_cabinet_context", lambda username: {})
    monkeypatch.setattr(routes, "build_benchmark_context", lambda selected_run_id=None: {})

    assert routes.upload_page(request, ui_token="token", ui_user="user", ui_user_sig="sig")["template"] == "upload.html"
    assert routes.benchmark_page(request, ui_token="token", ui_user="user", ui_user_sig="sig")["template"] == "benchmark.html"
    assert rendered == ["upload.html", "benchmark.html"]


def test_worker_can_skip_transcription_and_record_timing(tmp_path, monkeypatch) -> None:
    from audio_transcribator import worker
    from audio_transcribator.config import settings

    input_file = tmp_path / "input.wav"
    input_file.write_bytes(b"fake")
    job_dir = tmp_path / "job"

    monkeypatch.setattr(worker, "prepare_audio", lambda input_path, output_dir: input_file)
    monkeypatch.setattr(settings, "enable_diarization", False)

    worker.process_file(
        input_file,
        job_dir,
        enable_transcription=False,
        enable_summary=False,
        enable_diarization=True,
    )

    metadata = json.loads((job_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert metadata["timings"]["transcription"]["status"] == "skipped"
