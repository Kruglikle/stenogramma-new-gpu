from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import tempfile
import time
import traceback
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from audio_transcribator.benchmarks.diarization import (
    DATASET_ID,
    configure_dataset_cache,
    iter_dataset,
    write_wav,
)


PER_FILE_FIELDS = [
    "index",
    "item_id",
    "wer",
    "cer",
    "normalized_wer",
    "normalized_cer",
    "reference_words",
    "predicted_words",
    "reference_chars",
    "predicted_chars",
    "audio_duration_seconds",
    "processing_time_seconds",
    "rtf",
    "device",
    "status",
    "error",
]


def normalize_text_for_metric(text: str) -> str:
    normalized = text.lower().replace("ё", "е")
    normalized = re.sub(r"[^\w\s]+", " ", normalized, flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized, flags=re.UNICODE).strip()
    return normalized


def calculate_text_metrics(reference_text: str, predicted_text: str) -> dict[str, float | int]:
    from jiwer import cer, wer

    clean_reference = " ".join(reference_text.split())
    clean_prediction = " ".join(predicted_text.split())
    normalized_reference = normalize_text_for_metric(reference_text)
    normalized_prediction = normalize_text_for_metric(predicted_text)

    return {
        "wer": float(wer(clean_reference, clean_prediction)) if clean_reference else 0.0,
        "cer": float(cer(clean_reference, clean_prediction)) if clean_reference else 0.0,
        "normalized_wer": float(wer(normalized_reference, normalized_prediction)) if normalized_reference else 0.0,
        "normalized_cer": float(cer(normalized_reference, normalized_prediction)) if normalized_reference else 0.0,
        "reference_words": len(normalized_reference.split()),
        "predicted_words": len(normalized_prediction.split()),
        "reference_chars": len(normalized_reference),
        "predicted_chars": len(normalized_prediction),
    }


def extract_reference_text(sample: dict[str, Any]) -> str:
    speakers = sample.get("speakers") or []
    texts = []
    for segment in speakers:
        if isinstance(segment, dict):
            text = str(segment.get("text") or "").strip()
            if text:
                texts.append(text)
    return " ".join(texts)


def item_id(sample: dict[str, Any], index: int) -> str:
    for key in ("id", "utt_id", "audio_id", "path", "file", "filename"):
        value = sample.get(key)
        if value:
            return str(value)
    audio = sample.get("audio")
    if isinstance(audio, dict) and audio.get("path"):
        return str(audio["path"])
    return str(index)


def set_transcription_device(device: str) -> None:
    from audio_transcribator.config import settings

    settings.whisperx_device = device


def run_service_transcription(audio_file: Path, job_dir: Path, transcription_model: str | None = None) -> str:
    from audio_transcribator.services.transcription import transcribe

    return transcribe(audio_file, job_dir, transcription_model_id=transcription_model)


def empty_row(index: int, sample_id: str, device: str) -> dict[str, Any]:
    return {field: "" for field in PER_FILE_FIELDS} | {
        "index": index,
        "item_id": sample_id,
        "device": device,
    }


def process_sample(
    index: int,
    sample: dict[str, Any],
    output_dir: Path,
    device: str,
    save_artifacts: bool,
    transcription_model: str | None = None,
) -> dict[str, Any]:
    sample_id = item_id(sample, index)
    row = empty_row(index, sample_id, device)
    artifact_root = output_dir / "artifacts" / f"{index:06d}"
    temp_root = None
    started: float | None = None

    try:
        reference_text = extract_reference_text(sample)
        audio = sample.get("audio")
        if not isinstance(audio, dict) or "array" not in audio or "sampling_rate" not in audio:
            raise ValueError("Sample does not contain decoded audio with array and sampling_rate")

        if save_artifacts:
            job_dir = artifact_root
            job_dir.mkdir(parents=True, exist_ok=True)
        else:
            temp_root = tempfile.TemporaryDirectory(prefix=f"asr_bench_{index:06d}_")
            job_dir = Path(temp_root.name)

        audio_path = job_dir / "audio.wav"
        duration = write_wav(audio, audio_path)
        row["audio_duration_seconds"] = duration or float(sample.get("duration") or 0.0)

        started = time.perf_counter()
        predicted_text = run_service_transcription(audio_path, job_dir, transcription_model=transcription_model)
        processing_time = time.perf_counter() - started

        row.update(calculate_text_metrics(reference_text, predicted_text))
        row["processing_time_seconds"] = processing_time
        row["rtf"] = processing_time / row["audio_duration_seconds"] if row["audio_duration_seconds"] else ""
        row["status"] = "ok"
        row["error"] = ""

        if save_artifacts:
            (job_dir / "reference_text.txt").write_text(reference_text, encoding="utf-8")
            (job_dir / "predicted_text.txt").write_text(predicted_text, encoding="utf-8")
            (job_dir / "reference_metadata.json").write_text(
                json.dumps({key: value for key, value in sample.items() if key != "audio"}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return row

    except Exception as exc:
        if started is not None:
            processing_time = time.perf_counter() - started
            row["processing_time_seconds"] = processing_time
            try:
                duration = float(row["audio_duration_seconds"])
            except (TypeError, ValueError):
                duration = 0.0
            row["rtf"] = processing_time / duration if duration else ""
        row["status"] = "failed"
        row["error"] = str(exc)
        row["traceback"] = traceback.format_exc()
        return row
    finally:
        if temp_root is not None:
            temp_root.cleanup()
        if not save_artifacts and artifact_root.exists():
            shutil.rmtree(artifact_root, ignore_errors=True)


def float_values(rows: Sequence[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def weighted_average(rows: Sequence[dict[str, Any]], value_key: str, weight_key: str) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        try:
            value = float(row[value_key])
            weight = float(row[weight_key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value) and math.isfinite(weight) and weight > 0:
            numerator += value * weight
            denominator += weight
    return numerator / denominator if denominator else None


def aggregate(rows: Sequence[dict[str, Any]], device: str) -> dict[str, Any]:
    successful = [row for row in rows if row.get("status") == "ok"]
    processing_time = sum(float_values(rows, "processing_time_seconds"))
    audio_duration = sum(float_values(rows, "audio_duration_seconds"))

    return {
        "dataset": DATASET_ID,
        "device": device,
        "total_files": len(rows),
        "successful_files": len(successful),
        "failed_files": len(rows) - len(successful),
        "mean_wer": mean(float_values(successful, "wer")),
        "mean_cer": mean(float_values(successful, "cer")),
        "mean_normalized_wer": mean(float_values(successful, "normalized_wer")),
        "mean_normalized_cer": mean(float_values(successful, "normalized_cer")),
        "word_weighted_wer": weighted_average(successful, "wer", "reference_words"),
        "char_weighted_cer": weighted_average(successful, "cer", "reference_chars"),
        "reference_words": sum(float_values(successful, "reference_words")),
        "predicted_words": sum(float_values(successful, "predicted_words")),
        "reference_chars": sum(float_values(successful, "reference_chars")),
        "predicted_chars": sum(float_values(successful, "predicted_chars")),
        "audio_duration_seconds": audio_duration,
        "processing_time_seconds": processing_time,
        "rtf": processing_time / audio_duration if audio_duration else None,
    }


def write_outputs(output_dir: Path, rows: Sequence[dict[str, Any]], aggregate_metrics: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "asr_per_file_metrics.csv", "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PER_FILE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    (output_dir / "asr_aggregate_metrics.json").write_text(
        json.dumps(aggregate_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with open(output_dir / "asr_errors.jsonl", "w", encoding="utf-8") as errors:
        for row in rows:
            if row.get("status") == "failed":
                errors.write(
                    json.dumps(
                        {
                            "index": row.get("index"),
                            "item_id": row.get("item_id"),
                            "error": row.get("error"),
                            "traceback": row.get("traceback"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    (output_dir / "asr_summary.md").write_text(render_summary(aggregate_metrics), encoding="utf-8")


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def render_summary(metrics: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Transcription Benchmark Summary",
            "",
            f"- Dataset: `{metrics['dataset']}`",
            f"- Device: `{metrics['device']}`",
            f"- Files: {metrics['successful_files']} ok / {metrics['total_files']} total",
            f"- Mean WER: {fmt(metrics['mean_wer'])}",
            f"- Mean CER: {fmt(metrics['mean_cer'])}",
            f"- Mean normalized WER: {fmt(metrics['mean_normalized_wer'])}",
            f"- Mean normalized CER: {fmt(metrics['mean_normalized_cer'])}",
            f"- Total audio seconds: {fmt(metrics['audio_duration_seconds'], 2)}",
            f"- Total processing seconds: {fmt(metrics['processing_time_seconds'], 2)}",
            f"- RTF: {fmt(metrics['rtf'])}",
            "",
            "Reports:",
            "- `asr_per_file_metrics.csv`",
            "- `asr_aggregate_metrics.json`",
            "- `asr_errors.jsonl`",
        ]
    )


def default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("data") / "transcription_benchmarks" / timestamp


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate current transcription service on a Hugging Face dataset.")
    parser.add_argument("--limit", type=int, default=10, help="Number of dataset items to process.")
    parser.add_argument("--offset", type=int, default=0, help="Number of dataset items to skip before processing.")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu", help="WhisperX device.")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir(), help="Report output directory.")
    parser.add_argument("--save-artifacts", action="store_true", help="Keep audio and text artifacts per file.")
    parser.add_argument("--transcription-model", default=None, help="Transcription model id from transcription_models.json.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.limit < 0:
        raise SystemExit("--limit must be >= 0")
    if args.offset < 0:
        raise SystemExit("--offset must be >= 0")

    configure_dataset_cache()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = list(iter_dataset(limit=args.limit, offset=args.offset))
    set_transcription_device(args.device)

    rows: list[dict[str, Any]] = []
    for index, sample in samples:
        row = process_sample(
            index,
            sample,
            output_dir,
            args.device,
            args.save_artifacts,
            transcription_model=args.transcription_model,
        )
        rows.append(row)
        write_outputs(output_dir, rows, aggregate(rows, args.device))
        print(f"[{row['status']}] index={index} item_id={row['item_id']} error={row.get('error') or ''}", flush=True)

    write_outputs(output_dir, rows, aggregate(rows, args.device))
    print(f"Transcription benchmark reports saved to: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
