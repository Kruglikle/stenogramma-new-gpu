import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

from audio_transcribator.config import settings
from audio_transcribator.services.job_metadata import (
    STATUS_LABELS,
    build_timing_result,
    clean_job_title,
    load_job_metadata,
    parse_iso_datetime,
)
from audio_transcribator.services.job_storage import is_relative_to
from audio_transcribator.services.progress import load_job_progress
from audio_transcribator.utils.files import tail


def resolve_status(metadata: dict, files: list[str], log_tail: str) -> str:
    """Определить статус задачи по metadata, файлам и хвосту лога."""
    status = metadata.get("status")
    if status == "completed" and metadata.get("enable_summary", True) and "summary.txt" not in files:
        return "completed_without_summary"
    if status in STATUS_LABELS:
        return status
    if "failed" in log_tail.lower() or "traceback" in log_tail.lower():
        return "failed"
    if "stenogramma.txt" in files or "transcript.txt" in files:
        return "completed_without_summary"
    return "running"


def build_job_result(job_id: str) -> dict:
    """Собрать публичный результат задачи для API и UI."""
    job_dir = settings.results_dir / job_id
    if not job_dir.exists():
        raise FileNotFoundError("Job not found")

    files = [p.name for p in job_dir.iterdir() if p.is_file()]
    metadata = load_job_metadata(job_dir)
    log_tail = tail(job_dir / "run.log")
    status = resolve_status(metadata, files, log_tail)

    result = {
        "job_id": job_id,
        "title": metadata.get("title") or job_id,
        "user_login": metadata.get("user_login"),
        "files": files,
        "status": status,
        "status_label": STATUS_LABELS.get(status, status.title()),
        "enable_transcription": metadata.get("enable_transcription", True),
        "enable_summary": metadata.get("enable_summary", True),
        "enable_diarization": metadata.get("enable_diarization", False),
        "timings": build_timing_result(metadata),
        "progress": load_job_progress(job_dir, status),
    }

    _add_text_file(result, "transcript", job_dir / "stenogramma.txt", fallback=job_dir / "transcript.txt")
    _add_text_file(result, "edited_transcript", job_dir / "edited_transcript.txt")
    _add_text_file(result, "summary", job_dir / "summary.txt")
    _add_text_file(result, "diarization", job_dir / "diarization.txt")
    _add_text_file(result, "diarized_transcript", job_dir / "diarized_transcript.txt")
    _add_text_file(result, "diarization_error", job_dir / "diarization_error.txt")

    if "edited_transcript" in result:
        result["editing_timing"] = result["timings"].get("by_step", {}).get("editing")
    if (job_dir / "editing.lock").exists():
        result["editing_in_progress"] = True
    _add_text_file(result, "editor_error", job_dir / "editing_error.txt")
    if log_tail:
        result["log_tail"] = log_tail

    return result


def _add_text_file(result: dict, key: str, path: Path, fallback: Path | None = None) -> None:
    """Добавить текстовый файл в result, сохраняя старое поведение чтения."""
    selected_path = path if path.exists() else fallback
    if selected_path and selected_path.exists():
        result[key] = selected_path.read_text(encoding="utf-8", errors="replace")


def choose_history_download(files: list[str]) -> str | None:
    """Выбрать главный файл для быстрой загрузки из истории кабинета."""
    for filename in ("edited_transcript.txt", "diarized_transcript.txt", "stenogramma.txt", "summary.txt"):
        if filename in files:
            return filename
    return None


def list_user_jobs(user_login: str, limit: int = 40) -> list[dict]:
    """Вернуть последние задачи пользователя для боковой истории."""
    if not settings.results_dir.exists():
        return []

    jobs = []
    for job_dir in settings.results_dir.iterdir():
        if not job_dir.is_dir():
            continue

        metadata = load_job_metadata(job_dir)
        if metadata.get("user_login") != user_login:
            continue

        files = metadata.get("files")
        if not isinstance(files, list):
            files = [p.name for p in job_dir.iterdir() if p.is_file()]

        status = resolve_status(metadata, files, tail(job_dir / "run.log", lines=8))
        jobs.append(
            {
                "job_id": job_dir.name,
                "title": metadata.get("title") or job_dir.name,
                "status": status,
                "status_label": STATUS_LABELS.get(status, status.title()),
                "started_at": metadata.get("started_at") or "",
                "finished_at": metadata.get("finished_at") or "",
                "download_file": choose_history_download(files),
            }
        )

    jobs.sort(key=lambda item: item.get("started_at") or "", reverse=True)
    return jobs[:limit]


def update_job_title(job_id: str, title: str) -> dict:
    """Переименовать задачу без изменения остальных полей metadata."""
    job_dir = settings.results_dir / job_id
    if not job_dir.exists():
        raise FileNotFoundError("Job not found")

    metadata = load_job_metadata(job_dir)
    metadata["title"] = clean_job_title(title, metadata.get("title") or job_id)
    metadata["files"] = sorted(p.name for p in job_dir.iterdir() if p.is_file())
    _write_metadata(job_dir, metadata)
    return metadata


def delete_job(job_id: str) -> None:
    """Удалить задачу и связанный исходный upload-файл, если он лежит в uploads."""
    job_dir = settings.results_dir / job_id
    if not job_dir.exists():
        raise FileNotFoundError("Job not found")

    metadata = load_job_metadata(job_dir)
    input_file = metadata.get("input_file")
    if input_file and not str(input_file).startswith(("http://", "https://")):
        input_path = Path(input_file)
        if not input_path.is_absolute():
            input_path = (settings.base_dir / input_path).resolve()
        else:
            input_path = input_path.resolve()

        if is_relative_to(input_path, settings.upload_dir):
            input_path.unlink(missing_ok=True)

    shutil.rmtree(job_dir)


def cleanup_expired_jobs() -> list[str]:
    """Удалить задачи старше DATA_RETENTION_DAYS."""
    if settings.data_retention_days <= 0 or not settings.results_dir.exists():
        return []

    cutoff = datetime.now(UTC) - timedelta(days=settings.data_retention_days)
    deleted_job_ids = []
    for job_dir in settings.results_dir.iterdir():
        if not job_dir.is_dir():
            continue

        metadata = load_job_metadata(job_dir)
        created_at = parse_iso_datetime(metadata.get("started_at") or metadata.get("finished_at"))
        if created_at is None:
            created_at = datetime.fromtimestamp(job_dir.stat().st_mtime, tz=UTC)

        if created_at > cutoff:
            continue

        delete_job(job_dir.name)
        deleted_job_ids.append(job_dir.name)

    if deleted_job_ids:
        print(f"Deleted expired jobs: {', '.join(deleted_job_ids)}", flush=True)
    return deleted_job_ids


def get_job_file(job_id: str, filename: str) -> Path:
    """Проверить существование файла задачи и вернуть путь к нему."""
    job_dir = settings.results_dir / job_id
    file_path = job_dir / filename

    if not job_dir.exists():
        raise FileNotFoundError("Job not found")
    if not file_path.exists():
        raise FileNotFoundError("File not found")

    return file_path


def _write_metadata(job_dir: Path, metadata: dict) -> None:
    import json

    with open(job_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
