from pathlib import Path
from threading import Lock

from audio_transcribator.config import settings
from audio_transcribator.services.job_launch import launch_worker_for_job
from audio_transcribator.services.job_metadata import load_job_metadata, parse_iso_datetime, save_job_metadata


ACTIVE_STATUSES = {"started", "running"}
TERMINAL_STATUSES = {"completed", "completed_without_summary", "failed"}

_dispatch_lock = Lock()


def list_job_dirs() -> list[Path]:
    if not settings.results_dir.exists():
        return []
    return [path for path in settings.results_dir.iterdir() if path.is_dir()]


def job_sort_key(job_dir: Path, metadata: dict) -> tuple[str, str]:
    created_at = parse_iso_datetime(metadata.get("started_at"))
    if created_at:
        return (created_at.isoformat(), job_dir.name)
    return ("", job_dir.name)


def dispatch_queued_jobs() -> list[str]:
    if settings.max_concurrent_jobs <= 0:
        return []

    with _dispatch_lock:
        active_count = 0
        queued_jobs: list[tuple[Path, dict]] = []

        for job_dir in list_job_dirs():
            metadata = load_job_metadata(job_dir)
            status = metadata.get("status")
            if status in ACTIVE_STATUSES:
                active_count += 1
            elif status == "queued":
                queued_jobs.append((job_dir, metadata))

        available_slots = max(settings.max_concurrent_jobs - active_count, 0)
        if available_slots <= 0:
            return []

        queued_jobs.sort(key=lambda item: job_sort_key(item[0], item[1]))
        started_job_ids = []
        for job_dir, metadata in queued_jobs[:available_slots]:
            try:
                launch_worker_for_job(job_dir, metadata)
            except Exception as exc:
                save_job_metadata(
                    job_dir,
                    metadata.get("input_file") or "",
                    status="failed",
                    transcription_model_id=metadata.get("transcription_model"),
                    user_login=metadata.get("user_login"),
                    title=metadata.get("title"),
                    enable_transcription=metadata.get("enable_transcription", True),
                    enable_summary=metadata.get("enable_summary", True),
                    enable_diarization=metadata.get("enable_diarization", False),
                    diarization_speakers=int(metadata.get("diarization_speakers") or 0),
                )
                (job_dir / "queue_error.txt").write_text(str(exc), encoding="utf-8")
                continue
            started_job_ids.append(job_dir.name)

        return started_job_ids
