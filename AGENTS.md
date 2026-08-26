# Project Instructions

## Stack

- FastAPI application with Jinja templates and static CSS/JS.
- PostgreSQL stores users and job metadata.
- Background jobs run audio/video processing.
- WhisperX `large-v3` performs transcription.
- pyannote `speaker-diarization-3.1` performs diarization.
- vLLM provides the OpenAI-compatible summary endpoint, default model `Qwen/Qwen3-8B-AWQ`.

## Runtime

- Docker image is GPU-oriented: PyTorch 2.7.1, CUDA 12.8, cuDNN 9, CTranslate2 4.6.3.
- Keep `MAX_CONCURRENT_JOBS=1` by default for 16 GB VRAM cards such as RTX 5070 Ti.
- Do not silently fall back to CPU for production GPU runs.
- Keep secrets in `.env`; do not commit real tokens.

## Commands

- Tests: `python -m pytest`
- Syntax check: `python -m compileall audio_transcribator tests`
- Docker run: `docker compose up -d --build`

## Quality Gates

- Keep heavy ML model loading out of module import paths.
- Summary code must keep partial results in `summary.txt` and failures should not discard transcript results.
- Run pytest after service changes.
