from pathlib import Path

from fastapi import UploadFile

from audio_transcribator.config import settings
from audio_transcribator.services.job_formatting import format_bytes
from audio_transcribator.services.job_metadata import load_job_metadata


class StorageQuotaExceeded(ValueError):
    """Пользователь превысил доступное место для своих задач."""


def path_size(path: Path) -> int:
    """Посчитать размер файла или каталога."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def is_relative_to(path: Path, parent: Path) -> bool:
    """Совместимая проверка, что path находится внутри parent."""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def get_upload_size(file: UploadFile) -> int:
    """Получить размер UploadFile, не меняя текущую позицию чтения."""
    size = getattr(file, "size", None)
    if isinstance(size, int) and size >= 0:
        return size

    current_position = file.file.tell()
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(current_position)
    return size


def list_user_job_dirs(user_login: str) -> list[Path]:
    """Найти каталоги задач, принадлежащие пользователю."""
    if not settings.results_dir.exists():
        return []

    job_dirs = []
    for job_dir in settings.results_dir.iterdir():
        if not job_dir.is_dir():
            continue
        metadata = load_job_metadata(job_dir)
        if metadata.get("user_login") == user_login:
            job_dirs.append(job_dir)
    return job_dirs


def user_storage_usage_bytes(user_login: str) -> int:
    """Посчитать занятое место пользователя с учетом загруженных исходников."""
    total = 0
    counted_job_dirs = []
    for job_dir in list_user_job_dirs(user_login):
        counted_job_dirs.append(job_dir.resolve())
        total += path_size(job_dir)

        metadata = load_job_metadata(job_dir)
        input_file = metadata.get("input_file")
        if not input_file or str(input_file).startswith(("http://", "https://")):
            continue

        input_path = Path(input_file)
        if not input_path.is_absolute():
            input_path = (settings.base_dir / input_path).resolve()
        else:
            input_path = input_path.resolve()

        if any(is_relative_to(input_path, counted_dir) for counted_dir in counted_job_dirs):
            continue
        if is_relative_to(input_path, settings.upload_dir):
            total += path_size(input_path)

    return total


def user_storage_quota(user_login: str) -> dict:
    """Вернуть данные квоты в байтах и строках для кабинета."""
    used = user_storage_usage_bytes(user_login)
    limit = settings.user_storage_quota_bytes
    remaining = max(limit - used, 0)
    return {
        "used_bytes": used,
        "limit_bytes": limit,
        "remaining_bytes": remaining,
        "used": format_bytes(used),
        "limit": format_bytes(limit),
        "remaining": format_bytes(remaining),
        "percent": round((used / limit) * 100, 1) if limit else 0,
    }


def ensure_user_storage_quota(user_login: str | None, incoming_bytes: int = 0) -> None:
    """Проверить, что новая задача помещается в пользовательскую квоту."""
    if not user_login or settings.user_storage_quota_bytes <= 0:
        return

    quota = user_storage_quota(user_login)
    if quota["used_bytes"] + incoming_bytes <= quota["limit_bytes"]:
        return

    raise StorageQuotaExceeded(
        "Недостаточно места в кабинете: занято "
        f"{quota['used']} из {quota['limit']}, новый файл {format_bytes(incoming_bytes)}."
    )
