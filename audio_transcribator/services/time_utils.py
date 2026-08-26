from datetime import UTC, datetime


def utc_now_iso() -> str:
    """Вернуть текущее UTC-время в существующем формате timestamp для metadata."""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
