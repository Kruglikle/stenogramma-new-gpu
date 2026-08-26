# Стенограмма

Сервис для обработки аудио и видео: извлекает аудио, делает транскрибацию, диаризацию по спикерам и итоговое резюме.

Проект собран на основе прежнего интерфейса `Kruglikle/Stenogramma`: FastAPI, Jinja-шаблоны, фоновые задачи, история в кабинете, скачивание результатов и benchmark-экраны.

## Пайплайн

- подготовка аудио через `ffmpeg`;
- транскрибация через локальный `WhisperX large-v3`;
- диаризация через `pyannote.audio` и pipeline `speaker-diarization-3.1`;
- суммаризация через внешний vLLM endpoint с OpenAI-compatible API, по умолчанию `Qwen/Qwen3-8B-AWQ`;
- сохранение результатов в `data/api_results`.

## Быстрый запуск

Docker-запуск рассчитан на Linux-хост с NVIDIA GPU, NVIDIA-драйвером, Docker Engine, Docker Compose v2 и NVIDIA Container Toolkit.

Образ использует PyTorch 2.7.1, CUDA 12.8, cuDNN 9 и CTranslate2 4.6.3. Такая связка рассчитана на Blackwell-видеокарты, включая GeForce RTX 5070 Ti. Для CUDA 12.8 нужен драйвер NVIDIA не ниже 570.26.

Проверьте GPU:

```bash
nvidia-smi
```

Создайте `.env`:

```bash
cp .env.example .env
```

Для локального HTTPS оставьте:

```env
APP_HOST=localhost
```

Для сервера укажите домен, который смотрит на этот хост:

```env
APP_HOST=stenogramma.example.com
```

Caddy автоматически выпустит публичный HTTPS-сертификат для настоящего домена. Для `localhost` будет локальный сертификат; браузер может попросить подтвердить исключение безопасности.

Перед запуском на реальном домене обязательно замените дефолтные секреты в `.env`:

```env
API_TOKEN=<long-random-token>
API_PASSWORD=<strong-password>
CREATOR_PASSWORD=<strong-password>
ADD_USER_ADMIN_TOKEN=<long-random-token>
POSTGRES_PASSWORD=<strong-password>
```

Если `APP_HOST` не равен `localhost`, приложение не стартует с демонстрационными значениями этих переменных. URL-загрузка по умолчанию не принимает приватные и локальные адреса (`ALLOW_PRIVATE_MEDIA_URLS=false`), чтобы пользовательская ссылка не могла обращаться к внутренним сервисам.

Запустите vLLM отдельно, например на хосте:

```bash
docker compose up -d vllm
```

Затем запустите приложение:

```bash
docker compose up -d --build
```

Веб-интерфейс:

```text
https://localhost/ui
```

Benchmark-интерфейс:

```text
https://localhost/ui/benchmark
```

Остановить сервис:

```bash
docker compose down
```

Посмотреть логи:

```bash
docker compose logs -f api
```

## Основные настройки

```env
MAX_CONCURRENT_JOBS=1
APP_HOST=localhost

WHISPERX_MODEL=large-v3
WHISPERX_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
WHISPER_LOCAL_FILES_ONLY=true

ENABLE_DIARIZATION=true
PYANNOTE_DEVICE=cuda
PYANNOTE_MODEL_DIR=/app/data/model_cache/pyannote
PYANNOTE_PIPELINE_CONFIG=/app/data/model_cache/pyannote/speaker-diarization-3.1/config.yaml
PYANNOTE_SEGMENTATION_MODEL=/app/data/model_cache/pyannote/segmentation-3.0
PYANNOTE_EMBEDDING_MODEL=/app/data/model_cache/pyannote/hbredin-wespeaker-voxceleb-resnet34-LM

VLLM_BASE_URL=http://vllm:8000/v1
VLLM_API_KEY=EMPTY
VLLM_HF_CACHE_DIR=./data/model_cache/vllm-huggingface
VLLM_MAX_MODEL_LEN=3500
VLLM_GPU_MEMORY_UTILIZATION=0.44
SUMMARY_MODEL=Qwen/Qwen3-8B-AWQ
SUMMARY_TOKENIZER_MODEL=Qwen/Qwen3-8B-AWQ
SUMMARY_LOCAL_FILES_ONLY=true
SUMMARY_ENABLE_THINKING=false
SUMMARY_MAX_MODEL_TOKENS=3200
SUMMARY_MAX_CHUNK_TOKENS=2200
SUMMARY_OVERLAP_TOKENS=100
SUMMARY_CHUNK_MAX_TOKENS=450
SUMMARY_FINAL_MAX_TOKENS=700
SUMMARY_REQUEST_TIMEOUT_SECONDS=3600
```

По умолчанию одновременно обрабатывается одна задача, чтобы несколько копий `large-v3` не переполнили 16 ГБ видеопамяти RTX 5070 Ti. Для другой видеокарты меняйте `MAX_CONCURRENT_JOBS`, `WHISPERX_BATCH_SIZE`, `WHISPER_COMPUTE_TYPE` и параметры vLLM.

## URL

- `/ui` - загрузка файлов и просмотр результатов;
- `/ui/benchmark` - benchmark диаризации и транскрибации;
- `/login` - получение токена API;
- `/process` - загрузка файла через API;
- `/result/{job_id}` - результат обработки;
- `/download/{job_id}/{filename}` - скачивание файла результата.

## Результаты обработки

Для каждой задачи создается папка:

```text
data/api_results/<job_id>
```

Основные файлы:

- `metadata.json` - статус, настройки задачи и время этапов;
- `run.log` - технический лог обработки;
- `stenogramma.txt` - распознанный текст;
- `transcript_segments.json` - сегменты транскрибации;
- `summary.txt` - итоговое резюме;
- `partial_summary.txt` - частичное резюме, если суммаризация упала;
- `summary_usage.json` - сведения о вызовах summary-модели;
- `diarization.txt` - человекочитаемая разметка спикеров;
- `diarization.json` - сегменты диаризации;
- `diarization.rttm` - RTTM-файл;
- `diarized_transcript.txt` - стенограмма с привязкой к спикерам.

## Модели pyannote

Если `WHISPER_LOCAL_FILES_ONLY=true`, модели должны лежать в `data/model_cache`. Для pyannote используйте `scripts/download_pyannote_models.py` и задайте `HF_TOKEN` в `.env`, если скачивание идет из Hugging Face.

Для однократной загрузки WhisperX VAD и ASR-модели в постоянный Docker-кэш выполните:

```bash
docker compose exec api python -m scripts.download_whisperx_models
```

## Тесты

```bash
pytest
```
