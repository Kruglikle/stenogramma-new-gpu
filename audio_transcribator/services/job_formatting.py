def format_duration(seconds: float | int | None) -> str:
    """Преобразовать длительность в короткую русскую строку для UI и API."""
    if seconds is None:
        return ""

    total_seconds = int(round(float(seconds)))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{hours} ч {minutes} мин {seconds} сек"
    if minutes:
        return f"{minutes} мин {seconds} сек"
    return f"{seconds} сек"


def format_bytes(size_bytes: int) -> str:
    """Показать размер файла в человекочитаемом виде."""
    value = float(size_bytes)
    for unit in ("Б", "КБ", "МБ", "ГБ", "ТБ"):
        if value < 1024 or unit == "ТБ":
            return f"{value:.1f} {unit}" if unit != "Б" else f"{int(value)} {unit}"
        value /= 1024
    return f"{size_bytes} Б"
