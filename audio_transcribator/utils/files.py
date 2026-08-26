from pathlib import Path


ALLOWED_DOWNLOADS = {
    "stenogramma.txt",
    "edited_transcript.txt",
    "summary.txt",
    "run.log",
    "work_audio.wav",
    "diarization.txt",
    "diarization.json",
    "diarization.rttm",
    "diarized_transcript.txt",
    "transcript_segments.json",
}


def write_text_atomic(path: Path, content: str) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def tail(path: Path, lines: int = 30) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
