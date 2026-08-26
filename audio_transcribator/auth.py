from fastapi import Header, HTTPException

from audio_transcribator.config import settings
from audio_transcribator.db import verify_user_credentials


def check_auth(authorization: str | None = Header(default=None)) -> None:
    if authorization != f"Bearer {settings.api_token}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def check_add_user_auth(authorization: str | None = Header(default=None)) -> None:
    if authorization != f"Bearer {settings.add_user_admin_token}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def verify_credentials(username: str, password: str) -> bool:
    return verify_user_credentials(username, password)
