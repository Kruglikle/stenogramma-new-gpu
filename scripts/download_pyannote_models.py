import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


PIPELINE_TEMPLATE = """version: 3.1.0
pipeline:
  name: pyannote.audio.pipelines.SpeakerDiarization
  params:
    clustering: AgglomerativeClustering
    embedding: {embedding_model_dir}
    embedding_batch_size: 32
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


def yaml_path(path: Path) -> str:
    return json.dumps(str(path.resolve()).replace("\\", "/"), ensure_ascii=False)


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


def embedding_checkpoint(model_dir: Path, target_dir: Path) -> Path:
    source = embedding_source_checkpoint(model_dir)
    if "pyannote" not in str(source).lower() and "wespeaker" in str(source).lower():
        return source

    alias = target_dir.parent / "wespeaker-voxceleb-resnet34-LM.onnx"
    alias.parent.mkdir(parents=True, exist_ok=True)
    if not alias.exists() or alias.stat().st_size != source.stat().st_size:
        shutil.copy2(source, alias)
    return alias


def download_repo(repo_id: str, target_dir: Path, token: str | None) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} -> {target_dir}")
    snapshot_download(
        repo_id,
        local_dir=str(target_dir),
        token=token,
        local_dir_use_symlinks=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pyannote diarization models for offline runtime.")
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path(os.getenv("PYANNOTE_MODEL_DIR", "data/model_cache/pyannote")),
    )
    parser.add_argument("--token", default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"))
    args = parser.parse_args()

    if not args.token:
        raise SystemExit("Set HF_TOKEN/HUGGINGFACE_TOKEN or pass --token for the one-time model download.")

    target_dir = args.target_dir.resolve()
    pipeline_dir = target_dir / "speaker-diarization-3.1"
    segmentation_dir = target_dir / "segmentation-3.0"
    embedding_dir = target_dir / "hbredin-wespeaker-voxceleb-resnet34-LM"

    download_repo("pyannote/speaker-diarization-3.1", pipeline_dir, args.token)
    download_repo("pyannote/segmentation-3.0", segmentation_dir, args.token)
    download_repo("hbredin/wespeaker-voxceleb-resnet34-LM", embedding_dir, args.token)

    config_path = pipeline_dir / "config.yaml"
    config_path.write_text(
        PIPELINE_TEMPLATE.format(
            segmentation_checkpoint=yaml_path(model_checkpoint(segmentation_dir)),
            embedding_model_dir=yaml_path(embedding_checkpoint(embedding_dir, target_dir)),
        ),
        encoding="utf-8",
    )
    print(f"Wrote offline pipeline config: {config_path}")


if __name__ == "__main__":
    main()
