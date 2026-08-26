from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from audio_transcribator.services.progress import load_job_progress, save_job_progress


T = TypeVar("T")

STEP_WEIGHTS = {
    "download": 10,
    "prepare_audio": 10,
    "transcription": 45,
    "diarization": 25,
    "summary": 10,
}
"""Относительные веса для пересчета прогресса этапов в общий процент задачи."""

STEP_LABELS = {
    "download": "Скачивание медиа",
    "prepare_audio": "Подготовка аудио",
    "transcription": "Транскрибация",
    "diarization": "Диаризация",
    "summary": "Резюме",
}

ProgressPlan = dict[str, tuple[float, float]]
StageProgress = Callable[[float, str | None], None]


def build_progress_plan(
    source_is_url: bool,
    enable_transcription: bool,
    enable_diarization: bool,
    enable_summary: bool,
) -> ProgressPlan:
    """Построить диапазоны общего прогресса для включенных этапов обработки.

    Веса намеренно приблизительные: они показывают движение пайплайна в UI,
    не меняя поведение аудиообработки, транскрибации и диаризации.
    """
    steps = []
    if source_is_url:
        steps.append("download")
    steps.append("prepare_audio")
    if enable_transcription:
        steps.append("transcription")
    if enable_diarization:
        steps.append("diarization")
    if enable_summary:
        steps.append("summary")

    total_weight = sum(STEP_WEIGHTS[step] for step in steps) or 1
    current = 0.0
    plan = {}
    for step in steps:
        width = STEP_WEIGHTS[step] / total_weight * 100
        plan[step] = (current, width)
        current += width
    return plan


def update_progress(
    job_dir: Path,
    plan: ProgressPlan,
    step: str,
    stage_percent: float,
    detail: str | None = None,
) -> None:
    """Пересчитать процент внутри этапа в общий прогресс задачи."""
    start, width = plan.get(step, (0, 0))
    overall = start + width * max(0, min(100, stage_percent)) / 100
    save_job_progress(job_dir, overall, step, label=STEP_LABELS.get(step), detail=detail)


def run_progress_step(
    job_dir: Path,
    plan: ProgressPlan,
    step: str,
    action: Callable[[StageProgress], T],
    timed_step: Callable[[Path, str, Callable[[], T], Callable[[T], bool] | None], T],
    skipped: Callable[[T], bool] | None = None,
) -> T:
    """Выполнить этап с замером времени и обновлением progress.json.

    `timed_step` передается из worker.py, чтобы этот модуль не знал деталей
    хранения timing-метаданных. Так прогресс отделен от persistence-логики задач.
    """
    update_progress(job_dir, plan, step, 0)

    def stage_progress(percent: float, detail: str | None = None) -> None:
        update_progress(job_dir, plan, step, percent, detail=detail)

    result = timed_step(job_dir, step, lambda: action(stage_progress), skipped=skipped)
    update_progress(job_dir, plan, step, 100)
    return result


def mark_progress_failed(job_dir: Path, detail: str | None = None) -> None:
    """Пометить текущий прогресс как ошибочный, не сбрасывая последний процент."""
    current = load_job_progress(job_dir, "failed")
    save_job_progress(
        job_dir,
        current.get("percent", 0),
        current.get("step", "failed"),
        detail=detail or current.get("detail", ""),
        status="failed",
    )
