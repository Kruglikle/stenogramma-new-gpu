import json
import os
import shutil
from pathlib import Path

from audio_transcribator.config import settings
from audio_transcribator.services.model_loading import allow_legacy_torch_checkpoint_loading
from audio_transcribator.utils.files import write_text_atomic


LOCAL_PIPELINE_TEMPLATE = """version: 3.1.0
pipeline:
  name: pyannote.audio.pipelines.SpeakerDiarization
  params:
    clustering: AgglomerativeClustering
    embedding: {embedding_model_dir}
    embedding_batch_size: {embedding_batch_size}
    embedding_exclude_overlap: true
    segmentation:
      checkpoint: {segmentation_checkpoint}
    segmentation_batch_size: 32
params:
  clustering:
    method: centroid
    min_cluster_size: 12
    threshold: 0.7045654963945799
  segmentation:
    min_duration_off: 0.0
"""


def format_timestamp(seconds: float) -> str:
    total_ms = max(int(round(seconds * 1000)), 0)
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    seconds_part = total_seconds % 60
    total_minutes = total_seconds // 60
    minutes = total_minutes % 60
    hours = total_minutes // 60
    return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}.{ms:03d}"


def yaml_path(path: Path) -> str:
    return json.dumps(str(path).replace("\\", "/"), ensure_ascii=False)


def model_checkpoint(model_dir: Path) -> Path:
    checkpoint = model_dir / "pytorch_model.bin"
    if checkpoint.exists():
        return checkpoint

    candidates = sorted(model_dir.rglob("pytorch_model.bin"))
    if candidates:
        return candidates[0]

    raise RuntimeError(f"Local pyannote model checkpoint was not found in {model_dir}")


def embedding_source_checkpoint(model_dir: Path) -> Path:
    checkpoint = model_dir / "speaker-embedding.onnx"
    if checkpoint.exists():
        return checkpoint

    candidates = sorted(model_dir.rglob("*.onnx"))
    if candidates:
        return candidates[0]

    raise RuntimeError(
        f"Local WeSpeaker ONNX checkpoint was not found in {model_dir}. "
        "Download hbredin/wespeaker-voxceleb-resnet34-LM for pyannote 3.1."
    )


def embedding_checkpoint(model_dir: Path) -> Path:
    source = embedding_source_checkpoint(model_dir)
    if "pyannote" not in str(source).lower() and "wespeaker" in str(source).lower():
        return source

    alias = settings.pyannote_model_dir.parent / "wespeaker-voxceleb-resnet34-LM.onnx"
    alias.parent.mkdir(parents=True, exist_ok=True)
    if not alias.exists() or alias.stat().st_size != source.stat().st_size:
        shutil.copy2(source, alias)
    return alias


def ensure_local_pipeline_config() -> Path:
    config_path = settings.pyannote_pipeline_config

    if not settings.pyannote_segmentation_model.exists():
        raise RuntimeError(
            "Local pyannote segmentation model was not found. "
            f"Expected: {settings.pyannote_segmentation_model}. "
            "Download models once before enabling diarization."
        )
    if not settings.pyannote_embedding_model.exists():
        raise RuntimeError(
            "Local pyannote embedding model was not found. "
            f"Expected: {settings.pyannote_embedding_model}. "
            "Download models once before enabling diarization."
        )

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        LOCAL_PIPELINE_TEMPLATE.format(
            segmentation_checkpoint=yaml_path(model_checkpoint(settings.pyannote_segmentation_model)),
            embedding_model_dir=yaml_path(embedding_checkpoint(settings.pyannote_embedding_model)),
            embedding_batch_size=settings.pyannote_embedding_batch_size,
        ),
        encoding="utf-8",
    )
    return config_path


def load_pipeline():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError("Install pyannote.audio==3.1.1 to enable pyannote diarization") from exc

    config_path = ensure_local_pipeline_config()
    print(f"Loading local pyannote pipeline: {config_path}", flush=True)
    with allow_legacy_torch_checkpoint_loading():
        pipeline = Pipeline.from_pretrained(str(config_path))

    device_name = settings.pyannote_device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name:
        pipeline.to(torch.device(device_name))
        if device_name == "cuda":
            configure_wespeaker_cuda_session(pipeline)
        print(f"Pyannote diarization device: {device_name}", flush=True)
    return pipeline


def configure_wespeaker_cuda_session(pipeline) -> None:
    embedding = getattr(pipeline, "_embedding", None)
    if embedding is None or embedding.__class__.__name__ != "ONNXWeSpeakerPretrainedSpeakerEmbedding":
        return

    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.inter_op_num_threads = 1
    session_options.intra_op_num_threads = 1
    session_options.log_severity_level = 3
    session = ort.InferenceSession(
        embedding.embedding,
        sess_options=session_options,
        providers=[
            (
                "CUDAExecutionProvider",
                {"cudnn_conv_algo_search": "HEURISTIC"},
            )
        ],
    )
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError(
            "Pyannote WeSpeaker CUDA provider could not be initialized. "
            "CPU fallback is disabled for GPU diarization."
        )
    embedding.session_ = session
    print("Pyannote WeSpeaker embeddings provider: CUDAExecutionProvider", flush=True)


def build_pipeline_kwargs(diarization_speakers: int | None = None) -> dict:
    kwargs = {}
    requested_speakers = diarization_speakers or settings.diarization_speakers
    if requested_speakers > 0:
        kwargs["num_speakers"] = requested_speakers
        return kwargs
    if settings.diarization_min_speakers > 0:
        kwargs["min_speakers"] = settings.diarization_min_speakers
    if settings.diarization_max_speakers > 0:
        kwargs["max_speakers"] = settings.diarization_max_speakers
    return kwargs


class DiarizationProgressHook:
    def __init__(self, progress_callback):
        self.progress_callback = progress_callback
        self.steps = []

    def __call__(self, step_name, step_artifact, file=None, total=None, completed=None):
        if not self.progress_callback:
            return
        if step_name not in self.steps:
            self.steps.append(step_name)
        step_index = self.steps.index(step_name)
        if completed is None or total in {None, 0}:
            fraction = 0.0
        else:
            fraction = max(0.0, min(float(completed) / float(total), 1.0))

        expected_steps = max(4, len(self.steps))
        percent = min(((step_index + fraction) / expected_steps) * 100, 99)
        self.progress_callback(percent, f"Этап pyannote: {step_name}")


def turn_duration(turn: dict) -> float:
    return max(float(turn["end"]) - float(turn["start"]), 0.0)


def speaker_durations(turns: list[dict]) -> dict[str, float]:
    durations = {}
    for turn in turns:
        durations[turn["speaker"]] = durations.get(turn["speaker"], 0.0) + turn_duration(turn)
    return durations


def filter_noisy_turns(turns: list[dict], requested_speakers: int | None = None) -> list[dict]:
    if not turns:
        return turns

    min_turn_seconds = max(settings.diarization_min_turn_seconds, 0)
    filtered = [turn for turn in turns if turn_duration(turn) >= min_turn_seconds]
    if not filtered:
        return turns
    if requested_speakers and requested_speakers > 0:
        return filtered

    durations = speaker_durations(filtered)
    total_duration = sum(durations.values())
    min_speaker_ratio = max(settings.diarization_min_speaker_ratio, 0)
    if total_duration <= 0 or min_speaker_ratio <= 0:
        return filtered

    keep_speakers = {
        speaker
        for speaker, duration in durations.items()
        if duration / total_duration >= min_speaker_ratio
    }
    if not keep_speakers:
        return filtered
    return [turn for turn in filtered if turn["speaker"] in keep_speakers]


def normalize_speaker_labels(turns: list[dict]) -> list[dict]:
    ordered_speakers = []
    for turn in turns:
        speaker = turn["speaker"]
        if speaker not in ordered_speakers:
            ordered_speakers.append(speaker)
    speaker_map = {
        speaker: f"Спикер {index}"
        for index, speaker in enumerate(ordered_speakers, start=1)
    }
    return [
        {
            "speaker": speaker_map[turn["speaker"]],
            "raw_speaker": turn["speaker"],
            "start": turn["start"],
            "end": turn["end"],
        }
        for turn in turns
    ]


def annotation_to_turns(annotation, requested_speakers: int | None = None) -> list[dict]:
    turns = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turns.append(
            {
                "speaker": str(speaker),
                "start": float(turn.start),
                "end": float(turn.end),
            }
        )
    turns.sort(key=lambda item: (item["start"], item["end"]))
    return normalize_speaker_labels(filter_noisy_turns(turns, requested_speakers=requested_speakers))


def build_diarization_stats(turns: list[dict], requested_speakers: int | None = None) -> dict:
    requested = requested_speakers or settings.diarization_speakers or None
    durations = speaker_durations(turns)
    total_duration = sum(durations.values())
    speakers = []
    for speaker, duration in sorted(durations.items(), key=lambda item: item[0]):
        ratio = duration / total_duration if total_duration else 0
        speakers.append(
            {
                "speaker": speaker,
                "seconds": round(duration, 3),
                "ratio": round(ratio, 4),
                "percent": round(ratio * 100, 2),
            }
        )

    lowest_ratio = min((item["ratio"] for item in speakers), default=0)
    low_confidence = (
        (len(speakers) > 1 and lowest_ratio < settings.diarization_low_confidence_ratio)
        or (requested is not None and len(speakers) < requested)
    )
    return {
        "requested_speakers": requested,
        "detected_speakers": len(speakers),
        "total_speech_seconds": round(total_duration, 3),
        "speakers": speakers,
        "low_confidence": low_confidence,
        "min_turn_seconds": settings.diarization_min_turn_seconds,
        "min_speaker_ratio": settings.diarization_min_speaker_ratio,
        "speaker_ratio_filter_applied": requested is None,
    }


def write_diarization_outputs(job_dir: Path, annotation, turns: list[dict], stats: dict) -> None:
    lines = [
        f"{format_timestamp(item['start'])} - {format_timestamp(item['end'])}: {item['speaker']}"
        for item in turns
    ]
    write_text_atomic(job_dir / "diarization.txt", "\n".join(lines))
    (job_dir / "diarization.json").write_text(
        json.dumps(turns, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (job_dir / "diarization_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with open(job_dir / "diarization.rttm", "w", encoding="utf-8") as rttm:
        annotation.write_rttm(rttm)


def load_transcript_segments(job_dir: Path) -> list[dict]:
    segments_path = job_dir / "transcript_segments.json"
    if not segments_path.exists():
        return []
    try:
        data = json.loads(segments_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [
        {
            "start": float(item.get("start", 0)),
            "end": float(item.get("end", 0)),
            "text": str(item.get("text", "")).strip(),
        }
        for item in data
        if isinstance(item, dict) and str(item.get("text", "")).strip()
    ]


def overlap_seconds(left: dict, right: dict) -> float:
    return max(min(left["end"], right["end"]) - max(left["start"], right["start"]), 0.0)


def find_segment_speaker(segment: dict, turns: list[dict]) -> str:
    best_turn = None
    best_overlap = 0.0
    for turn in turns:
        overlap = overlap_seconds(segment, turn)
        if overlap > best_overlap:
            best_overlap = overlap
            best_turn = turn
    if best_turn and best_overlap > 0:
        return best_turn["speaker"]
    midpoint = (segment["start"] + segment["end"]) / 2
    for turn in turns:
        if turn["start"] <= midpoint <= turn["end"]:
            return turn["speaker"]
    return "Спикер ?"


def write_diarized_transcript(job_dir: Path, turns: list[dict]) -> None:
    segments = load_transcript_segments(job_dir)
    if not segments:
        return

    lines = []
    for segment in segments:
        speaker = find_segment_speaker(segment, turns)
        lines.append(
            f"{format_timestamp(segment['start'])} - {format_timestamp(segment['end'])} | "
            f"{speaker}: {segment['text']}"
        )
    write_text_atomic(job_dir / "diarized_transcript.txt", "\n".join(lines))


def diarize(
    audio_file: Path,
    job_dir: Path,
    diarization_speakers: int | None = None,
    progress_callback=None,
) -> list[dict]:
    print("Running local pyannote speaker diarization 3.1...", flush=True)
    if progress_callback:
        progress_callback(1, "Загрузка локального pipeline pyannote")
    pipeline = load_pipeline()
    if progress_callback:
        progress_callback(5, "Запуск pyannote")
    kwargs = build_pipeline_kwargs(diarization_speakers)
    requested_speakers = diarization_speakers or settings.diarization_speakers or None
    print(f"Pyannote diarization kwargs: {kwargs or '{}'}", flush=True)
    if progress_callback:
        diarization = pipeline(str(audio_file), hook=DiarizationProgressHook(progress_callback), **kwargs)
    else:
        diarization = pipeline(str(audio_file), **kwargs)
    if progress_callback:
        progress_callback(99, "Сборка стенограммы со спикерами")
    turns = annotation_to_turns(diarization, requested_speakers=requested_speakers)
    stats = build_diarization_stats(turns, requested_speakers=requested_speakers)
    print(
        "Diarization speaker distribution: "
        + ", ".join(f"{item['speaker']}={item['percent']}%" for item in stats["speakers"]),
        flush=True,
    )
    if stats["low_confidence"]:
        print(
            "Diarization warning: one speaker cluster is too small; "
            "audio may contain one dominant voice or poor speaker separation.",
            flush=True,
        )
    write_diarization_outputs(job_dir, diarization, turns, stats)
    write_diarized_transcript(job_dir, turns)
    print(f"Diarization completed: {len(turns)} speaker turns.", flush=True)
    if progress_callback:
        progress_callback(100)
    return turns
