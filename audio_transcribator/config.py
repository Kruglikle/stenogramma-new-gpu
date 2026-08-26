import os
from pathlib import Path


def load_env_file(env_file: Path) -> None:
    if not env_file.exists():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


load_env_file(Path.cwd() / ".env")


LOCAL_APP_HOSTS = {"localhost", "127.0.0.1", "::1"}
DEFAULT_UI_SESSION_MAX_AGE_SECONDS = 12 * 60 * 60
DEFAULT_MAX_UPLOAD_FILE_BYTES = 5 * 1024 * 1024 * 1024
INSECURE_PRODUCTION_DEFAULTS = {
    "API_TOKEN": "test-token",
    "API_PASSWORD": "admin123",
    "CREATOR_PASSWORD": "creator123",
    "ADD_USER_ADMIN_TOKEN": "change-me-add-user-token",
    "POSTGRES_PASSWORD": "audio_transcribator_password",
}


class Settings:
    def __init__(self) -> None:
        self.base_dir = Path(os.getenv("BASE_DIR", Path.cwd())).resolve()
        self.upload_dir = Path(os.getenv("UPLOAD_DIR", self.base_dir / "data" / "uploads")).resolve()
        self.results_dir = Path(os.getenv("RESULTS_DIR", self.base_dir / "data" / "api_results")).resolve()
        self.model_cache_dir = Path(os.getenv("MODEL_CACHE_DIR", self.base_dir / "data" / "model_cache")).resolve()
        self.third_party_cache_dir = Path(
            os.getenv("THIRD_PARTY_CACHE_DIR", self.model_cache_dir / "third_party")
        ).resolve()
        self.user_storage_quota_bytes = int(os.getenv("USER_STORAGE_QUOTA_BYTES", str(10 * 1024 * 1024 * 1024)))
        self.data_retention_days = int(os.getenv("DATA_RETENTION_DAYS", "7"))
        self.data_cleanup_interval_seconds = int(os.getenv("DATA_CLEANUP_INTERVAL_SECONDS", str(24 * 60 * 60)))
        self.max_concurrent_jobs = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))
        self.job_dispatch_interval_seconds = float(os.getenv("JOB_DISPATCH_INTERVAL_SECONDS", "3"))
        self.api_token = os.getenv("API_TOKEN", "test-token")
        self.api_username = os.getenv("API_USERNAME", "admin")
        self.api_password = os.getenv("API_PASSWORD", "admin123")
        self.creator_username = os.getenv("CREATOR_USERNAME", "creator")
        self.creator_password = os.getenv("CREATOR_PASSWORD", "creator123")
        self.add_user_admin_token = os.getenv("ADD_USER_ADMIN_TOKEN", "change-me-add-user-token")
        self.app_host = os.getenv("APP_HOST", "localhost").split(":", 1)[0].strip().lower() or "localhost"
        self.cookie_secure = os.getenv("COOKIE_SECURE", "true").lower() in {"1", "true", "yes"}
        self.ui_session_max_age_seconds = int(
            os.getenv("UI_SESSION_MAX_AGE_SECONDS", str(DEFAULT_UI_SESSION_MAX_AGE_SECONDS))
        )
        self.allow_private_media_urls = os.getenv("ALLOW_PRIVATE_MEDIA_URLS", "false").lower() in {
            "1",
            "true",
            "yes",
        }
        self.max_upload_file_bytes = int(os.getenv("MAX_UPLOAD_FILE_BYTES", str(DEFAULT_MAX_UPLOAD_FILE_BYTES)))

        self.database_url = os.getenv(
            "DATABASE_URL",
            "postgresql://audio_transcribator:audio_transcribator_password@localhost:5432/audio_transcribator",
        )
        self.database_connect_retries = int(os.getenv("DATABASE_CONNECT_RETRIES", "30"))
        self.database_connect_delay_seconds = float(os.getenv("DATABASE_CONNECT_DELAY_SECONDS", "1"))

        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_api_key = os.getenv("OLLAMA_API_KEY", "ollama")
        self.vllm_base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        self.vllm_api_key = os.getenv("VLLM_API_KEY", "EMPTY")
        self.summary_model = os.getenv("SUMMARY_MODEL", "Qwen/Qwen3-8B-AWQ")
        self.summary_tokenizer_model = os.getenv("SUMMARY_TOKENIZER_MODEL", self.summary_model)
        self.summary_local_files_only = os.getenv("SUMMARY_LOCAL_FILES_ONLY", "true").lower() in {
            "1",
            "true",
            "yes",
        }
        self.summary_enable_thinking = os.getenv("SUMMARY_ENABLE_THINKING", "false").lower() in {
            "1",
            "true",
            "yes",
        }
        self.summary_max_model_tokens = int(os.getenv("SUMMARY_MAX_MODEL_TOKENS", "3200"))
        self.summary_max_chunk_tokens = int(os.getenv("SUMMARY_MAX_CHUNK_TOKENS", "2200"))
        self.summary_overlap_tokens = int(os.getenv("SUMMARY_OVERLAP_TOKENS", "100"))
        self.summary_chunk_max_tokens = int(os.getenv("SUMMARY_CHUNK_MAX_TOKENS", "450"))
        self.summary_final_max_tokens = int(os.getenv("SUMMARY_FINAL_MAX_TOKENS", "700"))
        if self.summary_max_chunk_tokens + self.summary_chunk_max_tokens > self.summary_max_model_tokens:
            raise ValueError("SUMMARY_MAX_CHUNK_TOKENS + SUMMARY_CHUNK_MAX_TOKENS exceeds SUMMARY_MAX_MODEL_TOKENS")
        self.summary_chunk_chars = int(os.getenv("SUMMARY_CHUNK_CHARS", "2500"))
        self.editor_model = os.getenv("EDITOR_MODEL", "qwen3:8b")
        self.editor_chunk_chars = int(os.getenv("EDITOR_CHUNK_CHARS", "2500"))
        self.editor_temperature = float(os.getenv("EDITOR_TEMPERATURE", "0.1"))
        self.ollama_request_timeout_seconds = float(os.getenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "900"))
        self.summary_request_timeout_seconds = float(os.getenv("SUMMARY_REQUEST_TIMEOUT_SECONDS", "3600"))
        self.summary_request_retries = int(os.getenv("SUMMARY_REQUEST_RETRIES", "2"))
        self.transcription_models_file = Path(
            os.getenv("TRANSCRIPTION_MODELS_FILE", self.base_dir / "audio_transcribator" / "transcription_models.json")
        ).resolve()

        self.whisper_compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        self.whisperx_model = os.getenv("WHISPERX_MODEL", "large-v3")
        self.whisperx_device = os.getenv("WHISPERX_DEVICE", "auto").strip().lower()
        self.whisperx_batch_size = int(os.getenv("WHISPERX_BATCH_SIZE", "16"))
        self.whisperx_vad_model = Path(
            os.getenv("WHISPERX_VAD_MODEL", self.model_cache_dir / "whisperx" / "whisperx-vad-segmentation.bin")
        ).resolve()
        self.whisper_local_files_only = os.getenv("WHISPER_LOCAL_FILES_ONLY", "true").lower() in {
            "1",
            "true",
            "yes",
        }
        self.transcription_language = os.getenv("TRANSCRIPTION_LANGUAGE", "ru").strip() or None
        self.configure_third_party_cache()

        self.enable_diarization = os.getenv("ENABLE_DIARIZATION", "false").lower() in {"1", "true", "yes"}
        self.diarization_speakers = int(os.getenv("DIARIZATION_SPEAKERS", "0"))
        self.diarization_min_speakers = int(os.getenv("DIARIZATION_MIN_SPEAKERS", "0"))
        self.diarization_max_speakers = int(os.getenv("DIARIZATION_MAX_SPEAKERS", "0"))
        self.diarization_min_turn_seconds = float(os.getenv("DIARIZATION_MIN_TURN_SECONDS", "0.25"))
        self.diarization_min_speaker_ratio = float(os.getenv("DIARIZATION_MIN_SPEAKER_RATIO", "0.02"))
        self.diarization_low_confidence_ratio = float(os.getenv("DIARIZATION_LOW_CONFIDENCE_RATIO", "0.05"))
        self.pyannote_embedding_batch_size = max(
            int(os.getenv("PYANNOTE_EMBEDDING_BATCH_SIZE", "64")),
            1,
        )
        self.pyannote_model_dir = Path(
            os.getenv("PYANNOTE_MODEL_DIR", self.model_cache_dir / "pyannote")
        ).resolve()
        self.pyannote_pipeline_config = Path(
            os.getenv(
                "PYANNOTE_PIPELINE_CONFIG",
                self.pyannote_model_dir / "speaker-diarization-3.1" / "config.yaml",
            )
        ).resolve()
        self.pyannote_segmentation_model = Path(
            os.getenv("PYANNOTE_SEGMENTATION_MODEL", self.pyannote_model_dir / "segmentation-3.0")
        ).resolve()
        self.pyannote_embedding_model = Path(
            os.getenv("PYANNOTE_EMBEDDING_MODEL", self.pyannote_model_dir / "hbredin-wespeaker-voxceleb-resnet34-LM")
        ).resolve()
        self.pyannote_device = os.getenv("PYANNOTE_DEVICE", "auto").strip().lower()
        self.validate_security_settings()

    def configure_third_party_cache(self) -> None:
        os.environ.setdefault("HF_HOME", str(self.third_party_cache_dir / "huggingface"))
        os.environ.setdefault("HF_HUB_CACHE", str(self.third_party_cache_dir / "huggingface" / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(self.third_party_cache_dir / "transformers"))
        os.environ.setdefault("MPLCONFIGDIR", str(self.third_party_cache_dir / "matplotlib"))
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        if self.whisper_local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

    def ensure_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.third_party_cache_dir.mkdir(parents=True, exist_ok=True)
        self.pyannote_model_dir.mkdir(parents=True, exist_ok=True)

    def validate_security_settings(self) -> None:
        if self.app_host in LOCAL_APP_HOSTS:
            return

        insecure_keys = [
            key
            for key, insecure_value in INSECURE_PRODUCTION_DEFAULTS.items()
            if os.getenv(key, insecure_value) == insecure_value
        ]
        if insecure_keys:
            joined = ", ".join(insecure_keys)
            raise RuntimeError(f"Unsafe production defaults for APP_HOST={self.app_host}: replace {joined}")


settings = Settings()
