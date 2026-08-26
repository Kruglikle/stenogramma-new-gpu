import json
from datetime import datetime
from pathlib import Path

from audio_transcribator.services.job_formatting import format_duration
from audio_transcribator.services.time_utils import utc_now_iso
from audio_transcribator.services.transcription_models import DEFAULT_TRANSCRIPTION_MODEL_ID


STATUS_LABELS = {
    "queued": "В очереди",
    "started": "Запущено",
    "running": "В обработке",
    "completed": "Готово",
    "completed_without_summary": "Готово без резюме",
    "failed": "Ошибка",
}

TIMING_LABELS = {
    "upload": "Загрузка файла",
    "download": "Скачивание медиа",
    "prepare_audio": "Подготовка аудио",
    "transcription": "Транскрибация",
    "diarization": "Диаризация",
    "summary": "Резюме",
    "editing": "ИИ-редактура",
}

TIMING_STATUS_LABELS = {
    "completed": "готово",
    "failed": "ошибка",
    "skipped": "пропущено",
}


def clean_job_title(value: str | None, fallback: str) -> str:
    """Нормализовать пользовательское название задачи и ограничить его длину."""
    title = " ".join((value or "").strip().split())
    if not title:
        title = fallback
    return title[:160]


def parse_iso_datetime(value: str | None) -> datetime | None:
    """Разобрать ISO-дату из metadata.json, поддерживая старый суффикс Z."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def load_job_metadata(job_dir: Path) -> dict:
    """Прочитать metadata.json; битый или отсутствующий файл считается пустым."""
    metadata_file = job_dir / "metadata.json"
    if not metadata_file.exists():
        return {}

    try:
        return json.loads(metadata_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_job_metadata(
    job_dir: Path,
    input_file: Path | str,
    status: str,
    transcription_model_id: str | None = None,
    user_login: str | None = None,
    title: str | None = None,
    enable_transcription: bool | None = None,
    enable_summary: bool | None = None,
    enable_diarization: bool | None = None,
    diarization_speakers: int | None = None,
) -> None:
    """Сохранить metadata.json в прежнем публичном формате."""
    existing_metadata = load_job_metadata(job_dir)
    started_at = existing_metadata.get("started_at") or utc_now_iso()
    input_value = str(input_file)
    fallback_title = input_value if input_value.startswith(("http://", "https://")) else Path(input_value).name
    metadata = {
        "job_id": job_dir.name,
        "status": status,
        "input_file": input_value,
        "title": clean_job_title(title or existing_metadata.get("title"), fallback_title),
        "user_login": user_login or existing_metadata.get("user_login"),
        "started_at": started_at,
        "transcription_model": transcription_model_id
        or existing_metadata.get("transcription_model")
        or DEFAULT_TRANSCRIPTION_MODEL_ID,
        "enable_transcription": enable_transcription
        if enable_transcription is not None
        else existing_metadata.get("enable_transcription", True),
        "enable_summary": enable_summary if enable_summary is not None else existing_metadata.get("enable_summary", True),
        "enable_diarization": enable_diarization
        if enable_diarization is not None
        else existing_metadata.get("enable_diarization", False),
        "diarization_speakers": diarization_speakers
        if diarization_speakers is not None
        else existing_metadata.get("diarization_speakers", 0),
        "timings": existing_metadata.get("timings", {}),
        "files": sorted(p.name for p in job_dir.iterdir() if p.is_file()),
    }
    if status in {"completed", "completed_without_summary", "failed"}:
        metadata["finished_at"] = existing_metadata.get("finished_at") or utc_now_iso()
        started = parse_iso_datetime(started_at)
        finished = parse_iso_datetime(metadata["finished_at"])
        if started and finished:
            metadata["total_seconds"] = max((finished - started).total_seconds(), 0)
    with open(job_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def save_job_timing(job_dir: Path, step: str, elapsed_seconds: float, status: str = "completed") -> None:
    """Обновить секцию timings в metadata.json после выполнения этапа."""
    metadata = load_job_metadata(job_dir)
    timings = metadata.setdefault("timings", {})
    timings[step] = {
        "label": TIMING_LABELS.get(step, step),
        "seconds": round(elapsed_seconds, 3),
        "duration": format_duration(elapsed_seconds),
        "status": status,
        "finished_at": utc_now_iso(),
    }
    metadata["files"] = sorted(p.name for p in job_dir.iterdir() if p.is_file())
    with open(job_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def build_timing_result(metadata: dict) -> dict:
    """Подготовить timings к показу в результате задачи."""
    ordered_steps = [
        "upload",
        "download",
        "prepare_audio",
        "transcription",
        "diarization",
        "summary",
        "editing",
    ]
    items = []
    for step in ordered_steps:
        timing = (metadata.get("timings") or {}).get(step)
        if not timing:
            continue
        seconds = timing.get("seconds")
        items.append(
            {
                "step": step,
                "label": timing.get("label") or TIMING_LABELS.get(step, step),
                "seconds": seconds,
                "duration": timing.get("duration") or format_duration(seconds),
                "status": timing.get("status", "completed"),
                "status_label": TIMING_STATUS_LABELS.get(timing.get("status", "completed"), timing.get("status")),
            }
        )

    total_seconds = metadata.get("total_seconds")
    item_total_seconds = sum(float(item.get("seconds") or 0) for item in items)
    if total_seconds is None and items:
        total_seconds = item_total_seconds
    elif total_seconds is not None:
        total_seconds = max(float(total_seconds), item_total_seconds)

    return {
        "items": items,
        "by_step": {item["step"]: item for item in items},
        "total_seconds": total_seconds,
        "total_duration": format_duration(total_seconds),
    }
