"""Совместимый фасад для операций с задачами.

Исторически API и UI импортировали все функции из этого файла. Реальная логика
разнесена по меньшим модулям, но публичные имена оставлены здесь, чтобы не
ломать старые импорты, тесты, API и фоновые worker-команды.
"""

from audio_transcribator.services.job_formatting import format_bytes, format_duration
from audio_transcribator.services.job_launch import start_uploaded_file, start_url
from audio_transcribator.services.job_metadata import (
    STATUS_LABELS,
    TIMING_LABELS,
    TIMING_STATUS_LABELS,
    build_timing_result,
    clean_job_title,
    load_job_metadata,
    parse_iso_datetime,
    save_job_metadata,
    save_job_timing,
)
from audio_transcribator.services.job_results import (
    build_job_result,
    choose_history_download,
    cleanup_expired_jobs,
    delete_job,
    get_job_file,
    list_user_jobs,
    resolve_status,
    update_job_title,
)
from audio_transcribator.services.job_storage import (
    StorageQuotaExceeded,
    ensure_user_storage_quota,
    get_upload_size,
    is_relative_to,
    list_user_job_dirs,
    path_size,
    user_storage_quota,
    user_storage_usage_bytes,
)


__all__ = [
    "STATUS_LABELS",
    "TIMING_LABELS",
    "TIMING_STATUS_LABELS",
    "StorageQuotaExceeded",
    "build_job_result",
    "build_timing_result",
    "choose_history_download",
    "clean_job_title",
    "cleanup_expired_jobs",
    "delete_job",
    "ensure_user_storage_quota",
    "format_bytes",
    "format_duration",
    "get_job_file",
    "get_upload_size",
    "is_relative_to",
    "list_user_job_dirs",
    "list_user_jobs",
    "load_job_metadata",
    "parse_iso_datetime",
    "path_size",
    "resolve_status",
    "save_job_metadata",
    "save_job_timing",
    "start_uploaded_file",
    "start_url",
    "update_job_title",
    "user_storage_quota",
    "user_storage_usage_bytes",
]
