FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        libavcodec-dev \
        libavdevice-dev \
        libavfilter-dev \
        libavformat-dev \
        libavutil-dev \
        libswresample-dev \
        libswscale-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.7.1 \
        torchaudio==2.7.1 \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip uninstall -y onnxruntime \
    && python -m pip install --no-cache-dir onnxruntime-gpu==1.26.0

COPY . .

ENV BASE_DIR=/app
ENV UPLOAD_DIR=/app/data/uploads
ENV RESULTS_DIR=/app/data/api_results
ENV WHISPERX_DEVICE=cuda
ENV WHISPER_COMPUTE_TYPE=float16
ENV PYANNOTE_DEVICE=cuda

CMD ["sh", "-c", "python scripts/verify_gpu.py && exec uvicorn app:app --host 0.0.0.0 --port 8001"]
