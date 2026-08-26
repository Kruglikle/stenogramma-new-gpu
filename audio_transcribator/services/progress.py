import json
from pathlib import Path

from audio_transcribator.services.time_utils import utc_now_iso
from audio_transcribator.utils.files import write_text_atomic


PROGRESS_LABELS = {
    "queued": "Задача поставлена в очередь",
    "upload": "Загрузка файла",
    "download": "Скачивание медиа",
    "prepare_audio": "Подготовка аудио",
    "transcription": "Транскрибация",
    "diarization": "Диаризация",
    "summary": "Резюме",
    "editing": "ИИ-редактура",
    "completed": "Готово",
    "failed": "Ошибка",
}
"""Стабильные UI-лейблы для этапов progress.json."""


def save_job_progress(
    job_dir: Path,
    percent: float,
    step: str,
    label: str | None = None,
    detail: str | None = None,
    status: str = "running",
) -> None:
    """Сохранить пользовательский прогресс обработки для автообновления страницы.

    Форма JSON является частью контракта UI. Ключи нельзя менять без одновременной
    миграции страницы результата и уже существующих задач.
    """
    normalized = max(0, min(100, round(float(percent), 1)))
    payload = {
        "percent": normalized,
        "step": step,
        "label": label or PROGRESS_LABELS.get(step, step),
        "detail": detail or "",
        "status": status,
        "updated_at": utc_now_iso(),
    }
    write_text_atomic(job_dir / "progress.json", json.dumps(payload, ensure_ascii=False, indent=2))


def load_job_progress(job_dir: Path, status: str) -> dict:
    """Загрузить progress.json и нормализовать финальные статусы для UI."""
    progress_file = job_dir / "progress.json"
    if progress_file.exists():
        try:
            progress = json.loads(progress_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            progress = {}
    else:
        progress = {}

    if status in {"completed", "completed_without_summary"}:
        progress.update(
            {
                "percent": 100,
                "step": "completed",
                "label": PROGRESS_LABELS["completed"],
                "detail": "",
                "status": "completed",
            }
        )
    elif status == "failed":
        progress.update(
            {
                "percent": progress.get("percent", 0),
                "step": progress.get("step", "failed"),
                "label": PROGRESS_LABELS["failed"],
                "detail": progress.get("detail", ""),
                "status": "failed",
            }
        )

    return {
        "percent": progress.get("percent", 0),
        "step": progress.get("step", "queued"),
        "label": progress.get("label", PROGRESS_LABELS["queued"]),
        "detail": progress.get("detail", ""),
        "status": progress.get("status", status),
    }
