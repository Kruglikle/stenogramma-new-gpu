from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import shutil
import tempfile
import time
import traceback
import wave
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DATASET_ID = "ivkond/synthetic-speech-diarization-ru"

PER_FILE_FIELDS = [
    "index",
    "item_id",
    "der",
    "jer",
    "miss",
    "false_alarm",
    "speaker_confusion",
    "miss_seconds",
    "false_alarm_seconds",
    "speaker_confusion_seconds",
    "reference_speech_seconds",
    "reference_speakers",
    "predicted_speakers",
    "speaker_count_delta",
    "audio_duration_seconds",
    "processing_time_seconds",
    "rtf",
    "device",
    "status",
    "error",
]


@dataclass(frozen=True)
class Turn:
    speaker: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 0.0)


def turns_to_annotation(turns: Sequence[Turn | dict[str, Any]], uri: str = "audio"):
    from pyannote.core import Annotation, Segment

    annotation = Annotation(uri=uri)
    for index, turn in enumerate(turns):
        if isinstance(turn, Turn):
            speaker = turn.speaker
            start = turn.start
            end = turn.end
        else:
            speaker = str(turn["speaker"])
            start = float(turn["start"])
            end = float(turn["end"])
        if end <= start:
            continue
        annotation[Segment(start, end), f"track_{index}"] = speaker
    return annotation


def predicted_turns_to_annotation(turns: Sequence[dict[str, Any]], uri: str = "audio"):
    normalized = [
        Turn(
            speaker=str(turn.get("speaker", f"speaker_{index}")),
            start=float(turn["start"]),
            end=float(turn["end"]),
        )
        for index, turn in enumerate(turns)
        if "start" in turn and "end" in turn
    ]
    return turns_to_annotation(normalized, uri=uri)


def calculate_diarization_metrics(reference, hypothesis) -> dict[str, float]:
    from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

    der_metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    jer_metric = JaccardErrorRate(collar=0.0, skip_overlap=False)

    details = der_metric(reference, hypothesis, detailed=True)
    total = float(details.get("total", 0.0) or 0.0)
    missed = float(details.get("missed detection", 0.0) or 0.0)
    false_alarm = float(details.get("false alarm", 0.0) or 0.0)
    confusion = float(details.get("confusion", 0.0) or 0.0)

    return {
        "der": float(details.get("diarization error rate", 0.0) or 0.0),
        "jer": float(jer_metric(reference, hypothesis)),
        "miss": missed / total if total else 0.0,
        "false_alarm": false_alarm / total if total else 0.0,
        "speaker_confusion": confusion / total if total else 0.0,
        "miss_seconds": missed,
        "false_alarm_seconds": false_alarm,
        "speaker_confusion_seconds": confusion,
        "reference_speech_seconds": total,
    }


def speaker_count(annotation) -> int:
    return len(annotation.labels())


def _as_records(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, dict):
        if value and all(isinstance(item, list) for item in value.values()):
            length = max((len(item) for item in value.values()), default=0)
            return [
                {key: items[index] for key, items in value.items() if index < len(items)}
                for index in range(length)
            ]
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _first_present(record: dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def extract_reference_turns(sample: dict[str, Any]) -> list[Turn]:
    raw_segments = (
        sample.get("speakers")
        or sample.get("segments")
        or sample.get("diarization")
        or sample.get("annotations")
    )
    turns: list[Turn] = []

    for record_index, record in enumerate(_as_records(raw_segments)):
        if not isinstance(record, dict):
            continue

        speaker = _first_present(
            record,
            ("speaker", "speaker_id", "speaker_label", "label", "name", "id"),
        )
        if speaker is None:
            speaker = f"speaker_{record_index}"

        nested_segments = record.get("segments") or record.get("turns")
        if nested_segments:
            for nested_index, nested in enumerate(_as_records(nested_segments)):
                if not isinstance(nested, dict):
                    continue
                nested_speaker = _first_present(
                    nested,
                    ("speaker", "speaker_id", "speaker_label", "label", "name", "id"),
                )
                turns.extend(
                    _turn_from_record(nested, nested_speaker or speaker, f"{record_index}_{nested_index}")
                )
            continue

        turns.extend(_turn_from_record(record, speaker, str(record_index)))

    turns.sort(key=lambda item: (item.start, item.end, item.speaker))
    return turns


def _turn_from_record(record: dict[str, Any], speaker: Any, fallback_id: str) -> list[Turn]:
    start = _float_or_none(_first_present(record, ("start", "start_time", "begin", "from")))
    end = _float_or_none(_first_present(record, ("end", "end_time", "stop", "to")))
    duration = _float_or_none(_first_present(record, ("duration", "dur")))

    if start is None:
        start = 0.0
    if end is None and duration is not None:
        end = start + duration
    if end is None or end <= start:
        return []

    return [Turn(speaker=str(speaker or f"speaker_{fallback_id}"), start=start, end=end)]


def write_wav(audio: dict[str, Any], target: Path) -> float:
    array = np.asarray(audio["array"])
    sampling_rate = int(audio["sampling_rate"])
    if array.ndim == 1:
        channels = 1
        samples = array
    elif array.ndim == 2:
        if array.shape[0] <= 8 and array.shape[0] < array.shape[1]:
            array = array.T
        channels = int(array.shape[1])
        samples = array.reshape(-1)
    else:
        raise ValueError(f"Unsupported audio shape: {array.shape}")

    samples = np.nan_to_num(samples.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if samples.size and np.max(np.abs(samples)) > 1.0:
        samples = samples / max(np.max(np.abs(samples)), 1.0)
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")

    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sampling_rate)
        wav.writeframes(pcm.tobytes())

    frames = array.shape[0] if channels > 1 else array.size
    return float(frames) / sampling_rate if sampling_rate else 0.0


def iter_dataset(limit: int, offset: int) -> Iterator[tuple[int, dict[str, Any]]]:
    configure_dataset_cache()

    from datasets import DatasetDict, IterableDatasetDict, load_dataset

    dataset = load_dataset(DATASET_ID, split="train", streaming=True)
    if isinstance(dataset, (DatasetDict, IterableDatasetDict)):
        dataset = dataset["train"] if "train" in dataset else next(iter(dataset.values()))

    selected = itertools.islice(enumerate(dataset), offset, offset + limit if limit >= 0 else None)
    for index, sample in selected:
        yield index, sample


def set_diarization_device(device: str) -> None:
    from audio_transcribator.config import settings

    settings.pyannote_device = device


def run_service_diarization(audio_file: Path, job_dir: Path) -> list[dict[str, Any]]:
    from audio_transcribator.services.diarization import diarize

    return diarize(audio_file, job_dir)


def _item_id(sample: dict[str, Any], index: int) -> str:
    for key in ("id", "utt_id", "audio_id", "path", "file", "filename"):
        value = sample.get(key)
        if value:
            return str(value)
    audio = sample.get("audio")
    if isinstance(audio, dict) and audio.get("path"):
        return str(audio["path"])
    return str(index)


def _empty_row(index: int, item_id: str, device: str) -> dict[str, Any]:
    return {field: "" for field in PER_FILE_FIELDS} | {
        "index": index,
        "item_id": item_id,
        "device": device,
    }


def process_sample(
    index: int,
    sample: dict[str, Any],
    output_dir: Path,
    device: str,
    save_artifacts: bool,
) -> dict[str, Any]:
    item_id = _item_id(sample, index)
    row = _empty_row(index, item_id, device)
    artifact_root = output_dir / "artifacts" / f"{index:06d}"
    temp_root = None
    started: float | None = None
    job_dir: Path

    try:
        reference_turns = extract_reference_turns(sample)
        reference = turns_to_annotation(reference_turns, uri=item_id)
        row["reference_speech_seconds"] = sum(turn.duration for turn in reference_turns)
        row["reference_speakers"] = speaker_count(reference)
        if sample.get("num_speakers") is not None:
            row["reference_speakers"] = int(sample["num_speakers"])

        audio = sample.get("audio")
        if not isinstance(audio, dict) or "array" not in audio or "sampling_rate" not in audio:
            raise ValueError("Sample does not contain decoded audio with array and sampling_rate")

        if save_artifacts:
            job_dir = artifact_root
            job_dir.mkdir(parents=True, exist_ok=True)
        else:
            temp_root = tempfile.TemporaryDirectory(prefix=f"diar_bench_{index:06d}_")
            job_dir = Path(temp_root.name)

        audio_path = job_dir / "audio.wav"
        duration = write_wav(audio, audio_path)
        row["audio_duration_seconds"] = duration or float(sample.get("duration") or 0.0)

        started = time.perf_counter()
        predicted_turns = run_service_diarization(audio_path, job_dir)
        processing_time = time.perf_counter() - started

        hypothesis = predicted_turns_to_annotation(predicted_turns, uri=item_id)
        metrics = calculate_diarization_metrics(reference, hypothesis)
        predicted_speakers = speaker_count(hypothesis)

        row.update(metrics)
        row["predicted_speakers"] = predicted_speakers
        row["speaker_count_delta"] = predicted_speakers - int(row["reference_speakers"] or 0)
        row["processing_time_seconds"] = processing_time
        row["rtf"] = processing_time / row["audio_duration_seconds"] if row["audio_duration_seconds"] else ""
        row["status"] = "ok"
        row["error"] = ""

        if save_artifacts:
            with open(job_dir / "reference.rttm", "w", encoding="utf-8") as rttm:
                reference.write_rttm(rttm)
            (job_dir / "reference_turns.json").write_text(
                json.dumps([turn.__dict__ for turn in reference_turns], ensure_ascii=False, indent=2),
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


def _float_values(rows: Sequence[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def aggregate(rows: Sequence[dict[str, Any]], device: str) -> dict[str, Any]:
    successful = [row for row in rows if row.get("status") == "ok"]
    total_ref = sum(_float_values(successful, "reference_speech_seconds"))
    miss_seconds = sum(_float_values(successful, "miss_seconds"))
    fa_seconds = sum(_float_values(successful, "false_alarm_seconds"))
    confusion_seconds = sum(_float_values(successful, "speaker_confusion_seconds"))
    processing_time = sum(_float_values(rows, "processing_time_seconds"))
    audio_duration = sum(_float_values(rows, "audio_duration_seconds"))

    return {
        "dataset": DATASET_ID,
        "device": device,
        "total_files": len(rows),
        "successful_files": len(successful),
        "failed_files": len(rows) - len(successful),
        "aggregate_der": (miss_seconds + fa_seconds + confusion_seconds) / total_ref if total_ref else None,
        "mean_der": _mean(_float_values(successful, "der")),
        "mean_jer": _mean(_float_values(successful, "jer")),
        "miss": miss_seconds / total_ref if total_ref else None,
        "false_alarm": fa_seconds / total_ref if total_ref else None,
        "speaker_confusion": confusion_seconds / total_ref if total_ref else None,
        "miss_seconds": miss_seconds,
        "false_alarm_seconds": fa_seconds,
        "speaker_confusion_seconds": confusion_seconds,
        "reference_speech_seconds": total_ref,
        "audio_duration_seconds": audio_duration,
        "processing_time_seconds": processing_time,
        "rtf": processing_time / audio_duration if audio_duration else None,
        "speaker_count_exact_match_rate": _speaker_count_match_rate(successful),
    }


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _speaker_count_match_rate(rows: Sequence[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    matches = 0
    for row in rows:
        try:
            matches += int(row["reference_speakers"]) == int(row["predicted_speakers"])
        except (KeyError, TypeError, ValueError):
            pass
    return matches / len(rows)


def write_outputs(output_dir: Path, rows: Sequence[dict[str, Any]], aggregate_metrics: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "per_file_metrics.csv", "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PER_FILE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(aggregate_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with open(output_dir / "errors.jsonl", "w", encoding="utf-8") as errors:
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

    (output_dir / "summary.md").write_text(render_summary(aggregate_metrics), encoding="utf-8")


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def render_summary(metrics: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Diarization Benchmark Summary",
            "",
            f"- Dataset: `{metrics['dataset']}`",
            f"- Device: `{metrics['device']}`",
            f"- Files: {metrics['successful_files']} ok / {metrics['total_files']} total",
            f"- Aggregate DER: {_fmt(metrics['aggregate_der'])}",
            f"- Mean DER: {_fmt(metrics['mean_der'])}",
            f"- Mean JER: {_fmt(metrics['mean_jer'])}",
            f"- Miss: {_fmt(metrics['miss'])}",
            f"- False Alarm: {_fmt(metrics['false_alarm'])}",
            f"- Speaker Confusion: {_fmt(metrics['speaker_confusion'])}",
            f"- Total audio seconds: {_fmt(metrics['audio_duration_seconds'], 2)}",
            f"- Total processing seconds: {_fmt(metrics['processing_time_seconds'], 2)}",
            f"- RTF: {_fmt(metrics['rtf'])}",
            f"- Speaker count exact match rate: {_fmt(metrics['speaker_count_exact_match_rate'])}",
            "",
            "Reports:",
            "- `per_file_metrics.csv`",
            "- `aggregate_metrics.json`",
            "- `errors.jsonl`",
        ]
    )


def default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("data") / "diarization_benchmarks" / timestamp


def configure_dataset_cache() -> None:
    cache_root = Path("data") / "model_cache" / "third_party" / "huggingface"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    os.environ.setdefault("HF_HOME", str(cache_root.resolve()))
    os.environ.setdefault("HF_HUB_CACHE", str((cache_root / "hub").resolve()))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate current diarization service on a Hugging Face dataset.")
    parser.add_argument("--limit", type=int, default=10, help="Number of dataset items to process.")
    parser.add_argument("--offset", type=int, default=0, help="Number of dataset items to skip before processing.")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu", help="pyannote device.")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir(), help="Report output directory.")
    parser.add_argument("--save-artifacts", action="store_true", help="Keep audio, RTTM, and service artifacts per file.")
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
    set_diarization_device(args.device)

    rows: list[dict[str, Any]] = []
    for index, sample in samples:
        row = process_sample(index, sample, output_dir, args.device, args.save_artifacts)
        rows.append(row)
        write_outputs(output_dir, rows, aggregate(rows, args.device))
        print(f"[{row['status']}] index={index} item_id={row['item_id']} error={row.get('error') or ''}", flush=True)

    write_outputs(output_dir, rows, aggregate(rows, args.device))
    print(f"Benchmark reports saved to: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
