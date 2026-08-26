import argparse
from pathlib import Path

from audio_transcribator.services.audio import download_media, prepare_audio
from audio_transcribator.services.pipeline_progress import (
    build_progress_plan,
    mark_progress_failed,
    run_progress_step as run_pipeline_progress_step,
)
from audio_transcribator.services.pipeline_steps import (
    enforce_download_quota,
    maybe_diarize,
    maybe_summarize,
    process_edit,
    save_metadata,
    save_job_timing,
    timed_step,
)
from audio_transcribator.services.progress import save_job_progress
from audio_transcribator.services.transcription import transcribe
from audio_transcribator.services.transcription_models import DEFAULT_TRANSCRIPTION_MODEL_ID


def process_file(
    input_file: Path,
    job_dir: Path,
    transcription_model_id: str = DEFAULT_TRANSCRIPTION_MODEL_ID,
    enable_transcription: bool = True,
    enable_summary: bool = True,
    enable_diarization: bool = False,
    diarization_speakers: int = 0,
) -> None:
    """Обработать загруженный файл через те же этапы, которые ожидают API и UI."""
    if not any((enable_transcription, enable_diarization, enable_summary)):
        raise ValueError("At least one processing step must be enabled")
    if enable_summary and not enable_transcription:
        raise ValueError("Summary requires transcription to be enabled")
    job_dir.mkdir(parents=True, exist_ok=True)
    save_metadata(
        job_dir,
        input_file,
        status="running",
        transcription_model_id=transcription_model_id,
        enable_transcription=enable_transcription,
        enable_summary=enable_summary,
        enable_diarization=enable_diarization,
        diarization_speakers=diarization_speakers,
    )
    progress_plan = build_progress_plan(False, enable_transcription, enable_diarization, enable_summary)
    save_job_progress(job_dir, 0, "queued")

    try:
        audio_file = run_pipeline_progress_step(
            job_dir,
            progress_plan,
            "prepare_audio",
            lambda progress: prepare_audio(input_file, job_dir),
            timed_step,
        )
        transcript = ""
        if enable_transcription:
            transcript = run_pipeline_progress_step(
                job_dir,
                progress_plan,
                "transcription",
                lambda progress: transcribe(
                    audio_file,
                    job_dir,
                    transcription_model_id=transcription_model_id,
                    progress_callback=progress,
                ),
                timed_step,
            )
        else:
            save_job_timing(job_dir, "transcription", 0, status="skipped")
        maybe_diarize(audio_file, job_dir, enable_diarization, diarization_speakers, progress_plan)
        maybe_summarize(transcript, job_dir, enable_summary, progress_plan)
        save_metadata(job_dir, input_file, status="completed", transcription_model_id=transcription_model_id)
        save_job_progress(job_dir, 100, "completed", status="completed")
        print("Processing completed.")
    except Exception as exc:
        mark_progress_failed(job_dir, str(exc))
        save_metadata(job_dir, input_file, status="failed", transcription_model_id=transcription_model_id)
        raise


def process_url(
    source_url: str,
    job_dir: Path,
    transcription_model_id: str = DEFAULT_TRANSCRIPTION_MODEL_ID,
    enable_transcription: bool = True,
    enable_summary: bool = True,
    enable_diarization: bool = False,
    diarization_speakers: int = 0,
) -> None:
    """Обработать ссылку на медиа, сохранив публичный формат результата задачи."""
    if not any((enable_transcription, enable_diarization, enable_summary)):
        raise ValueError("At least one processing step must be enabled")
    if enable_summary and not enable_transcription:
        raise ValueError("Summary requires transcription to be enabled")
    job_dir.mkdir(parents=True, exist_ok=True)
    save_metadata(
        job_dir,
        source_url,
        status="running",
        transcription_model_id=transcription_model_id,
        enable_transcription=enable_transcription,
        enable_summary=enable_summary,
        enable_diarization=enable_diarization,
        diarization_speakers=diarization_speakers,
    )
    progress_plan = build_progress_plan(True, enable_transcription, enable_diarization, enable_summary)
    save_job_progress(job_dir, 0, "queued")

    try:
        input_file = run_pipeline_progress_step(
            job_dir,
            progress_plan,
            "download",
            lambda progress: download_media(source_url, job_dir),
            timed_step,
        )
        enforce_download_quota(job_dir, input_file)
        save_metadata(
            job_dir,
            input_file,
            status="running",
            transcription_model_id=transcription_model_id,
            enable_transcription=enable_transcription,
            enable_summary=enable_summary,
            enable_diarization=enable_diarization,
            diarization_speakers=diarization_speakers,
        )
        audio_file = run_pipeline_progress_step(
            job_dir,
            progress_plan,
            "prepare_audio",
            lambda progress: prepare_audio(input_file, job_dir),
            timed_step,
        )
        transcript = ""
        if enable_transcription:
            transcript = run_pipeline_progress_step(
                job_dir,
                progress_plan,
                "transcription",
                lambda progress: transcribe(
                    audio_file,
                    job_dir,
                    transcription_model_id=transcription_model_id,
                    progress_callback=progress,
                ),
                timed_step,
            )
        else:
            save_job_timing(job_dir, "transcription", 0, status="skipped")
        maybe_diarize(audio_file, job_dir, enable_diarization, diarization_speakers, progress_plan)
        maybe_summarize(transcript, job_dir, enable_summary, progress_plan)
        save_metadata(job_dir, input_file, status="completed", transcription_model_id=transcription_model_id)
        save_job_progress(job_dir, 100, "completed", status="completed")
        print("Processing completed.")
    except Exception as exc:
        mark_progress_failed(job_dir, str(exc))
        save_metadata(job_dir, source_url, status="failed", transcription_model_id=transcription_model_id)
        raise


def main() -> None:
    """CLI-точка входа, сохраненная для совместимости с process_audio_fast.py."""
    parser = argparse.ArgumentParser(description="Process uploaded audio/video file.")
    parser.add_argument("input_file", type=Path)
    parser.add_argument("job_dir", type=Path)
    parser.add_argument("--source-url")
    parser.add_argument("--transcription-model", default=DEFAULT_TRANSCRIPTION_MODEL_ID)
    parser.add_argument("--no-transcription", action="store_true")
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument("--diarization", action="store_true")
    parser.add_argument("--diarization-speakers", type=int, default=0)
    parser.add_argument("--edit-model")
    parser.add_argument("--edit-source", default="transcript")
    args = parser.parse_args()

    if str(args.input_file) == "edit-transcript":
        process_edit(args.job_dir, editor_model=args.edit_model, transcript_source=args.edit_source)
    elif args.source_url:
        process_url(
            args.source_url,
            args.job_dir,
            transcription_model_id=args.transcription_model,
            enable_transcription=not args.no_transcription,
            enable_summary=not args.no_summary,
            enable_diarization=args.diarization,
            diarization_speakers=max(args.diarization_speakers, 0),
        )
    else:
        process_file(
            args.input_file,
            args.job_dir,
            transcription_model_id=args.transcription_model,
            enable_transcription=not args.no_transcription,
            enable_summary=not args.no_summary,
            enable_diarization=args.diarization,
            diarization_speakers=max(args.diarization_speakers, 0),
        )


if __name__ == "__main__":
    main()
